"""TorchScript U-Net adapter with lazy PyTorch loading."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.contracts.analyses import ROIBox
from app.contracts.enums import RoiMode
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.inference.adapters._utils import (
    iter_box_crops,
    open_rgb,
    output_dir,
    paste_box_probability,
    remove_small_components,
    save_binary_mask,
)
from app.inference.adapters.base import BaseSegmentationAdapter


class UNetAdapter(BaseSegmentationAdapter):
    """Run an immutable TorchScript semantic-segmentation artifact.

    The model-specific architecture stays outside the service: a handoff supplies a scripted
    module plus normalization settings in the registered YAML config.
    """

    _torch: Any = None
    _model: Any = None

    def load(self, device: str) -> None:
        try:
            torch = importlib.import_module("torch")
            resolved_device = self._resolve_device(torch, device)
            if self.config.get("loader", "torchscript") != "torchscript":
                raise ValueError("UNetAdapter supports only loader=torchscript")
            model = torch.jit.load(BytesIO(self.weight_bytes), map_location=resolved_device)
            model.eval()
            self._torch = torch
            self._model = model
            self._mark_loaded(resolved_device)
        except Exception as exc:
            self._mark_load_failed(exc)
            raise

    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        self._require_loaded()
        started = time.perf_counter()
        image = open_rgb(request.image_bytes or request.image_path)
        height, width = image.shape[:2]
        inference_height = self._inference_height(height)
        inference_image = self._prepare_inference_image(image[:inference_height])
        probability = np.zeros((height, width), dtype=np.float32)
        if request.roi_mode == RoiMode.BOXES:
            top_probability = np.zeros((inference_height, width), dtype=np.float32)
            for crop in iter_box_crops(
                inference_image,
                self._clip_boxes_to_inference_height(request.boxes, inference_height),
                context_px=request.roi_context_px,
            ):
                paste_box_probability(
                    top_probability,
                    self._predict_probability(crop.image),
                    crop,
                )
        else:
            top_probability = self._predict_probability(inference_image)
        probability[:inference_height] = top_probability
        threshold = request.threshold
        if threshold is None:
            threshold = self.metadata.default_threshold or 0.5
        binary = remove_small_components(
            self._threshold_probability(probability, threshold), request.min_area_px
        )

        destination = output_dir(request.run_dir, self.metadata.model_id)
        probability_path = destination / "probability.npy"
        binary_path = destination / "binary_mask.png"
        np.save(probability_path, probability.astype(np.float32), allow_pickle=False)
        save_binary_mask(binary_path, binary)
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return SegmentationOutput(
            width=width,
            height=height,
            probability_path=probability_path,
            binary_mask_path=binary_path,
            model_scores={"foreground_probability_mean": float(probability.mean())},
            warnings=(
                ["model_bottom_information_bar_excluded"]
                if inference_height != height
                else []
            ),
            runtime_ms=elapsed_ms,
        )

    def _predict_probability(self, image: np.ndarray) -> np.ndarray:
        tiling = self._tiling_configuration()
        if tiling is None:
            return self._predict_tile_probability(image)
        patch_height, patch_width, stride_height, stride_width = tiling
        padding_mode = str(self.config.get("tiling_padding", "none"))
        original_height, original_width = image.shape[:2]
        if padding_mode == "reflect":
            image = self._reflect_pad_to_tiling(
                image,
                patch_height=patch_height,
                patch_width=patch_width,
                stride_height=stride_height,
                stride_width=stride_width,
                pad_to_grid=self._pad_to_tile_grid(),
            )
        elif padding_mode != "none":
            raise ValueError("tiling_padding must be 'none' or 'reflect'")
        height, width = image.shape[:2]
        y_starts = self._tile_starts(height, patch_height, stride_height)
        x_starts = self._tile_starts(width, patch_width, stride_width)
        probability_sum = np.zeros((height, width), dtype=np.float64)
        weight_sum = np.zeros((height, width), dtype=np.float64)
        for y1 in y_starts:
            for x1 in x_starts:
                y2 = min(y1 + patch_height, height)
                x2 = min(x1 + patch_width, width)
                tile_probability = self._predict_tile_probability(image[y1:y2, x1:x2])
                weight = self._fusion_weight(tile_probability.shape)
                probability_sum[y1:y2, x1:x2] += tile_probability * weight
                weight_sum[y1:y2, x1:x2] += weight
        if np.any(weight_sum <= 0):  # pragma: no cover - guarded by positive blend weights
            raise RuntimeError("sliding-window fusion left uncovered pixels")
        probability = probability_sum / weight_sum
        if not np.isfinite(probability).all():
            raise ValueError("sliding-window inference produced non-finite probabilities")
        return np.asarray(probability[:original_height, :original_width], dtype=np.float32)

    @staticmethod
    def _reflect_pad_to_tiling(
        image: np.ndarray,
        *,
        patch_height: int,
        patch_width: int,
        stride_height: int,
        stride_width: int,
        pad_to_grid: bool = True,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        target_height = max(height, patch_height)
        target_width = max(width, patch_width)
        padding_height = target_height - height
        padding_width = target_width - width
        if pad_to_grid:
            padding_height += (
                stride_height - (target_height - patch_height) % stride_height
            ) % stride_height
            padding_width += (
                stride_width - (target_width - patch_width) % stride_width
            ) % stride_width
        if padding_height == 0 and padding_width == 0:
            return image
        if image.ndim not in {2, 3}:
            raise ValueError("tiled image must be two- or three-dimensional")
        if (padding_height and height < 2) or (padding_width and width < 2):
            raise ValueError("reflect padding requires at least two pixels on a padded axis")
        pad_width = [(0, padding_height), (0, padding_width)]
        if image.ndim == 3:
            pad_width.append((0, 0))
        return np.pad(
            image,
            pad_width,
            mode="reflect",
        )

    def _fusion_weight(self, shape: tuple[int, ...]) -> np.ndarray:
        fusion = str(self.config.get("overlap_fusion", "hann"))
        if fusion == "uniform":
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError("tile probability must be a non-empty two-dimensional array")
            return np.ones(shape, dtype=np.float64)
        if fusion == "hann":
            return self._blend_weight(shape, minimum=self._fusion_weight_floor())
        raise ValueError("overlap_fusion must be 'hann' or 'uniform'")

    def _pad_to_tile_grid(self) -> bool:
        value: object = self.config.get("pad_to_tile_grid", True)
        if not isinstance(value, bool):
            raise ValueError("pad_to_tile_grid must be a boolean")
        return value

    def _fusion_weight_floor(self) -> float:
        value: object = self.config.get("fusion_weight_floor", np.finfo(np.float64).eps)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("fusion_weight_floor must be a number in (0, 1]")
        minimum = float(value)
        if not 0 < minimum <= 1:
            raise ValueError("fusion_weight_floor must be a number in (0, 1]")
        return minimum

    def _inference_height(self, image_height: int) -> int:
        bottom_crop_px: object = self.config.get("bottom_crop_px", 0)
        if isinstance(bottom_crop_px, bool) or not isinstance(bottom_crop_px, int):
            raise ValueError("bottom_crop_px must be a non-negative integer")
        if bottom_crop_px < 0:
            raise ValueError("bottom_crop_px must be a non-negative integer")
        inference_height = image_height - bottom_crop_px
        if inference_height <= 0:
            raise ValueError("bottom_crop_px must leave at least one image row for inference")
        return inference_height

    @staticmethod
    def _clip_boxes_to_inference_height(
        boxes: list[ROIBox], inference_height: int
    ) -> list[ROIBox]:
        return [
            box.model_copy(update={"y2": min(box.y2, inference_height)})
            for box in boxes
            if box.active and box.y1 < inference_height
        ]

    def _threshold_probability(self, probability: np.ndarray, threshold: float) -> np.ndarray:
        comparison = str(self.config.get("threshold_comparison", "gte"))
        if comparison == "gt":
            return probability > threshold
        if comparison == "gte":
            return probability >= threshold
        raise ValueError("threshold_comparison must be 'gt' or 'gte'")

    def _predict_tile_probability(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        tensor_array = self._preprocess(image)
        tensor = self._torch.from_numpy(tensor_array).to(self._device)
        with self._torch.inference_mode():
            raw = self._model(tensor)
        probability = self._to_probability(raw)
        if probability.shape != (height, width):
            probability = np.asarray(
                Image.fromarray(probability.astype(np.float32), mode="F").resize(
                    (width, height), resample=Image.Resampling.BILINEAR
                )
            )
        return np.asarray(probability, dtype=np.float32)

    def _tiling_configuration(self) -> tuple[int, int, int, int] | None:
        patch = self.config.get("patch_size", self.config.get("input_size"))
        if patch is None:
            return None
        patch_height, patch_width = self._positive_pair(patch, name="patch_size")
        stride = self.config.get("stride", [patch_height, patch_width])
        stride_height, stride_width = self._positive_pair(stride, name="stride")
        if stride_height > patch_height or stride_width > patch_width:
            raise ValueError("stride must not exceed patch_size")
        return patch_height, patch_width, stride_height, stride_width

    @staticmethod
    def _positive_pair(value: object, *, name: str) -> tuple[int, int]:
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f"{name} must contain two positive integers")
        height, width = (int(item) for item in value)
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} must contain two positive integers")
        return height, width

    @staticmethod
    def _tile_starts(length: int, patch: int, stride: int) -> list[int]:
        if length <= patch:
            return [0]
        starts = list(range(0, length - patch + 1, stride))
        final = length - patch
        if starts[-1] != final:
            starts.append(final)
        return starts

    @staticmethod
    def _blend_weight(shape: tuple[int, ...], *, minimum: float) -> np.ndarray:
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("tile probability must be a non-empty two-dimensional array")
        height, width = shape
        y = np.hanning(height + 2)[1:-1] if height > 1 else np.ones(1)
        x = np.hanning(width + 2)[1:-1] if width > 1 else np.ones(1)
        weight = np.outer(y, x)
        return np.maximum(weight, minimum)

    def _prepare_inference_image(self, image: np.ndarray) -> np.ndarray:
        normalization = str(self.config.get("normalization", "fixed"))
        if normalization == "fixed":
            return image
        if normalization != "percentile":
            raise ValueError("normalization must be 'fixed' or 'percentile'")
        if int(self.config.get("input_channels", 3)) != 1:
            raise ValueError("percentile normalization requires input_channels=1")
        lower: object = self.config.get("lower_percentile", 1.0)
        upper: object = self.config.get("upper_percentile", 99.0)
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, int | float)
            or not isinstance(upper, int | float)
            or not 0 <= float(lower) < float(upper) <= 100
        ):
            raise ValueError("percentile bounds must satisfy 0 <= lower < upper <= 100")
        gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
        low, high = np.percentile(gray, [float(lower), float(upper)])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("percentile normalization produced non-finite bounds")
        if high <= low:
            return np.zeros(gray.shape, dtype=np.float32)
        normalized = np.clip((gray - low) / (high - low), 0.0, 1.0)
        return np.asarray(normalized, dtype=np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        channels = int(self.config.get("input_channels", 3))
        prepared: np.ndarray
        if channels == 1:
            if image.ndim == 2:
                prepared = np.asarray(image, dtype=np.float32)[..., None]
            else:
                prepared = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)[
                    ..., None
                ]
        elif channels == 3:
            prepared = image.astype(np.float32)
        else:
            raise ValueError("input_channels must be 1 or 3")
        input_size = self.config.get("input_size")
        if isinstance(input_size, list) and len(input_size) == 2:
            target_height, target_width = (int(input_size[0]), int(input_size[1]))
            image_array = prepared.squeeze() if channels == 1 else prepared.astype(np.uint8)
            resized = Image.fromarray(image_array)
            prepared = np.asarray(
                resized.resize((target_width, target_height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            if channels == 1:
                prepared = prepared[..., None]
        normalization = str(self.config.get("normalization", "fixed"))
        if normalization == "percentile":
            if not np.isfinite(prepared).all() or np.any((prepared < 0) | (prepared > 1)):
                raise ValueError("percentile-normalized input must be finite and in [0, 1]")
        elif normalization == "fixed":
            pixel_scale = np.float32(self.config.get("pixel_scale", 255.0))
            prepared = np.asarray(prepared / pixel_scale, dtype=np.float32)
            mean = np.asarray(self.config.get("mean", [0.0] * channels), dtype=np.float32)
            std = np.asarray(self.config.get("std", [1.0] * channels), dtype=np.float32)
            prepared = (prepared - mean) / std
        else:
            raise ValueError("normalization must be 'fixed' or 'percentile'")
        return np.ascontiguousarray(prepared.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    def _to_probability(self, raw: Any) -> np.ndarray:
        if isinstance(raw, Mapping):
            raw = raw.get("out", next(iter(raw.values())))
        if isinstance(raw, list | tuple):
            raw = raw[0]
        output = raw.detach().float()
        while output.ndim > 3:
            output = output[0]
        activation = str(self.config.get("output_activation", "logits"))
        if output.ndim == 3 and output.shape[0] > 1:
            class_index = int(self.config.get("foreground_class_index", 1))
            if activation == "probabilities":
                probability = output[class_index]
            else:
                probability = self._torch.softmax(output, dim=0)[class_index]
        else:
            output = output.squeeze()
            probability = output if activation == "probabilities" else self._torch.sigmoid(output)
        return np.asarray(probability.detach().cpu().numpy(), dtype=np.float32)

    def _release(self) -> None:
        self._model = None
        self._torch = None

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
