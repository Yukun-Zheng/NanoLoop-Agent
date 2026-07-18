"""TorchScript U-Net adapter with lazy PyTorch loading."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

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
        if request.roi_mode == RoiMode.BOXES:
            probability = np.zeros((height, width), dtype=np.float32)
            for crop in iter_box_crops(
                image,
                request.boxes,
                context_px=request.roi_context_px,
            ):
                paste_box_probability(
                    probability,
                    self._predict_probability(crop.image),
                    crop,
                )
        else:
            probability = self._predict_probability(image)
        threshold = request.threshold
        if threshold is None:
            threshold = self.metadata.default_threshold or 0.5
        binary = remove_small_components(probability >= threshold, request.min_area_px)

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
            runtime_ms=elapsed_ms,
        )

    def _predict_probability(self, image: np.ndarray) -> np.ndarray:
        tiling = self._tiling_configuration()
        if tiling is None:
            return self._predict_tile_probability(image)
        patch_height, patch_width, stride_height, stride_width = tiling
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
                weight = self._blend_weight(tile_probability.shape)
                probability_sum[y1:y2, x1:x2] += tile_probability * weight
                weight_sum[y1:y2, x1:x2] += weight
        if np.any(weight_sum <= 0):  # pragma: no cover - guarded by positive blend weights
            raise RuntimeError("sliding-window fusion left uncovered pixels")
        probability = probability_sum / weight_sum
        if not np.isfinite(probability).all():
            raise ValueError("sliding-window inference produced non-finite probabilities")
        return np.asarray(probability, dtype=np.float32)

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
    def _blend_weight(shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("tile probability must be a non-empty two-dimensional array")
        height, width = shape
        y = np.hanning(height + 2)[1:-1] if height > 1 else np.ones(1)
        x = np.hanning(width + 2)[1:-1] if width > 1 else np.ones(1)
        weight = np.outer(y, x)
        return np.maximum(weight, np.finfo(np.float64).eps)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        channels = int(self.config.get("input_channels", 3))
        prepared: np.ndarray
        if channels == 1:
            prepared = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)[..., None]
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
        prepared /= float(self.config.get("pixel_scale", 255.0))
        mean = np.asarray(self.config.get("mean", [0.0] * channels), dtype=np.float32)
        std = np.asarray(self.config.get("std", [1.0] * channels), dtype=np.float32)
        prepared = (prepared - mean) / std
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
