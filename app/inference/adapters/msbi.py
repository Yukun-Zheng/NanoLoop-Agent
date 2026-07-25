"""TorchScript adapter for NanoLoop-MSBI multi-head instance segmentation."""

from __future__ import annotations

import importlib
import json
import math
import time
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app.contracts.enums import RoiMode
from app.contracts.inference import InstancePrediction, SegmentationOutput, SegmentationRequest
from app.inference.adapters._utils import open_rgb, output_dir, save_binary_mask
from app.inference.adapters.base import BaseSegmentationAdapter
from app.msbi.decoding import DecodeConfig, decode_instances


class MSBIAdapter(BaseSegmentationAdapter):
    """Fuse every MSBI head before deterministic center/boundary watershed."""

    _torch: Any = None
    _model: Any = None

    def load(self, device: str) -> None:
        try:
            torch = importlib.import_module("torch")
            resolved = self._resolve_device(torch, device)
            if self.config.get("loader") != "torchscript":
                raise ValueError("MSBIAdapter supports only loader=torchscript")
            model = torch.jit.load(BytesIO(self.weight_bytes), map_location=resolved)
            model.eval()
            self._torch = torch
            self._model = model
            self._mark_loaded(resolved)
        except Exception as exc:
            self._mark_load_failed(exc)
            raise

    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        self._require_loaded()
        started = time.perf_counter()
        rgb = open_rgb(request.image_bytes or request.image_path)
        height, width = rgb.shape[:2]
        self._validate_expected_shape(height, width)
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )
        invalid_bottom = int(self.config.get("bottom_crop_px", 0))
        inference_height = height - invalid_bottom
        if inference_height <= 0:
            raise ValueError("MSBI bottom crop exhausts the image")
        normalized = self._normalize(gray, inference_height)
        preprocessed_at = time.perf_counter()
        compact_gate_means: np.ndarray | None = None
        if bool(self.config.get("runtime_compact_forward", False)):
            if bool(self.config.get("save_raw_heads", True)):
                raise ValueError(
                    "MSBI runtime_compact_forward requires save_raw_heads=false"
                )
            heads, compact_gate_means = self._predict_runtime_compact(
                normalized[:inference_height]
            )
        else:
            heads = self._predict_heads(normalized[:inference_height])
        primary_heads_at = time.perf_counter()
        blend_raw = self.config.get("anchor_multiscale_blend")
        blended_foreground: np.ndarray | None = None
        if blend_raw is not None:
            if not isinstance(blend_raw, dict):
                raise ValueError("MSBI anchor_multiscale_blend must be a mapping")
            if bool(blend_raw.get("enabled", False)):
                tiled_weight = float(blend_raw.get("tiled_weight", 0.8))
                full_frame_weight = float(
                    blend_raw.get("full_frame_weight", 1.0 - tiled_weight)
                )
                if (
                    not np.isfinite(tiled_weight)
                    or not np.isfinite(full_frame_weight)
                    or tiled_weight < 0
                    or full_frame_weight < 0
                    or not np.isclose(tiled_weight + full_frame_weight, 1.0)
                ):
                    raise ValueError(
                        "MSBI anchor blend weights must be non-negative and sum to one"
                    )
                full_probability = self._sigmoid(heads["foreground_logits"][0])
                tiled_probability = self._predict_anchor_probability(
                    normalized[:inference_height],
                    blend_raw,
                )
                blended_foreground = np.asarray(
                    tiled_weight * tiled_probability
                    + full_frame_weight * full_probability,
                    dtype=np.float32,
                )
        anchor_blend_at = time.perf_counter()
        valid = np.zeros((height, width), dtype=bool)
        valid[:inference_height] = True
        if request.roi_mode == RoiMode.BOXES:
            allowed = np.zeros_like(valid)
            for box in request.boxes:
                if not box.active:
                    continue
                x1, x2 = sorted((max(0, box.x1), min(width, box.x2)))
                y1, y2 = sorted((max(0, box.y1), min(inference_height, box.y2)))
                if x1 < x2 and y1 < y2:
                    allowed[y1:y2, x1:x2] = True
            valid &= allowed
        compact_runtime = compact_gate_means is not None
        if compact_runtime:
            foreground = np.zeros((height, width), dtype=np.float32)
            foreground[:inference_height] = (
                blended_foreground
                if blended_foreground is not None
                else self._sigmoid(heads["foreground_logits"][0])
            )
            foreground[~valid] = 0.0
            auxiliary_zero = np.zeros_like(foreground)
            center = auxiliary_zero
            boundary = auxiliary_zero
            distance = auxiliary_zero
            small = foreground
            large = foreground
            gate = None
            uncertainty_stride = int(
                self.config.get("compact_uncertainty_stride", 1)
            )
            if uncertainty_stride <= 0:
                raise ValueError("MSBI compact_uncertainty_stride must be positive")
            sampled_valid = valid[::uncertainty_stride, ::uncertainty_stride]
            valid_probability = np.clip(
                foreground[::uncertainty_stride, ::uncertainty_stride][
                    sampled_valid
                ],
                1e-6,
                1.0 - 1e-6,
            )
            mean_uncertainty = (
                float(
                    0.55
                    * np.mean(
                        -(
                            valid_probability * np.log(valid_probability)
                            + (1.0 - valid_probability)
                            * np.log(1.0 - valid_probability)
                        )
                        / math.log(2.0)
                    )
                )
                if valid_probability.size
                else 0.0
            )
            uncertainty = auxiliary_zero
        else:
            full_heads: dict[str, np.ndarray] = {}
            for name, values in heads.items():
                channels = values.shape[0]
                full = np.zeros((channels, height, width), dtype=np.float32)
                full[:, :inference_height] = values
                full_heads[name] = full
            foreground = self._sigmoid(full_heads["foreground_logits"][0])
            if blended_foreground is not None:
                foreground[:inference_height] = blended_foreground
            center = self._sigmoid(full_heads["center_logits"][0])
            boundary = self._sigmoid(full_heads["boundary_logits"][0])
            distance = np.clip(full_heads["distance_field"][0], 0.0, 1.0)
            small = self._sigmoid(full_heads["small_logits"][0])
            large = self._sigmoid(full_heads["large_logits"][0])
            gate = self._softmax(full_heads["gate_logits"], axis=0)
            for values in (foreground, center, boundary, distance, small, large):
                values[~valid] = 0.0
            gate[:, ~valid] = 0.0
            uncertainty = self._uncertainty(
                foreground=foreground,
                center=center,
                small=small,
                large=large,
                valid=valid,
            )
            mean_uncertainty = (
                float(uncertainty[valid].mean()) if valid.any() else 0.0
            )
        maps_at = time.perf_counter()
        decoder_raw = self.config.get("decoder", {})
        if not isinstance(decoder_raw, dict):
            raise ValueError("MSBI decoder config must be a mapping")
        foreground_threshold = (
            request.threshold
            if request.threshold is not None
            else float(decoder_raw.get("foreground_threshold", 0.5))
        )
        minimum_area = (
            request.min_area_px
            if request.min_area_px > 0
            else int(decoder_raw.get("min_area_px", 0))
        )
        maximum_area_raw = decoder_raw.get("max_area_px")
        decoder = DecodeConfig(
            mode=str(decoder_raw.get("mode", "watershed")),
            foreground_threshold=foreground_threshold,
            center_threshold=float(decoder_raw.get("center_threshold", 0.35)),
            center_nms_radius=int(decoder_raw.get("center_nms_radius", 5)),
            boundary_threshold=float(decoder_raw.get("boundary_threshold", 0.5)),
            boundary_penalty=float(decoder_raw.get("boundary_penalty", 2.0)),
            min_area_px=minimum_area,
            max_area_px=int(maximum_area_raw) if maximum_area_raw is not None else None,
            watershed_compactness=float(decoder_raw.get("watershed_compactness", 0.0)),
            exclude_border=bool(decoder_raw.get("exclude_border", False)),
            connectivity=int(decoder_raw.get("connectivity", 2)),
            fallback_peak_source=str(
                decoder_raw.get("fallback_peak_source", "distance")
            ),
        )
        decoded = decode_instances(
            foreground,
            center,
            boundary,
            distance_field=distance,
            valid_mask=valid,
            config=decoder,
        )
        decoded_at = time.perf_counter()
        labels = decoded.labels
        destination = output_dir(request.run_dir, self.metadata.model_id)
        artifact_arrays: dict[str, np.ndarray] = {
            "foreground_probability.npy": foreground,
        }
        gate_small = gate[0] if gate is not None else np.zeros_like(foreground)
        gate_large = gate[1] if gate is not None else np.zeros_like(foreground)
        if bool(self.config.get("save_raw_heads", True)):
            artifact_arrays.update(
                {
                    "center_probability.npy": center,
                    "boundary_probability.npy": boundary,
                    "distance_field.npy": distance,
                    "small_expert_probability.npy": small,
                    "large_expert_probability.npy": large,
                    "gate_small.npy": gate_small,
                    "gate_large.npy": gate_large,
                    "uncertainty.npy": uncertainty,
                    "instance_labels.npy": labels,
                }
            )
        for filename, values in artifact_arrays.items():
            np.save(destination / filename, values, allow_pickle=False)
        auxiliary_paths = (
            self._write_preview_layers(
                destination=destination,
                center=center,
                boundary=boundary,
                distance=distance,
                labels=labels,
                gate_small=gate_small,
                gate_large=gate_large,
                uncertainty=uncertainty,
            )
            if bool(self.config.get("save_auxiliary_previews", True))
            else {}
        )
        binary_path = destination / "binary_mask.png"
        save_binary_mask(binary_path, labels > 0)
        instances_path = destination / "instances.npz"
        instance_count = int(labels.max())
        instance_predictions = self._instance_predictions_from_labels(
            labels,
            decoded.confidences,
        )
        if bool(self.config.get("compact_instance_artifact", False)):
            compact_dtype = np.uint16 if instance_count <= np.iinfo(np.uint16).max else np.uint32
            save_archive = (
                np.savez_compressed
                if bool(self.config.get("compress_instance_artifact", True))
                else np.savez
            )
            save_archive(
                instances_path,
                label_map=labels.astype(compact_dtype, copy=False),
                instance_ids=np.arange(1, instance_count + 1, dtype=np.int32),
                confidences=np.asarray(decoded.confidences, dtype=np.float32),
            )
        else:
            masks = [labels == index for index in range(1, instance_count + 1)]
            np.savez_compressed(
                instances_path,
                masks=(
                    np.asarray(masks, dtype=bool)
                    if masks
                    else np.zeros((0, height, width), dtype=bool)
                ),
                confidences=np.asarray(decoded.confidences, dtype=np.float32),
            )
        if bool(self.config.get("save_adapter_instance_json", True)):
            (destination / "instances.json").write_text(
                json.dumps(
                    {
                        "instance_count": len(instance_predictions),
                        "instances": [
                            item.model_dump(mode="json") for item in instance_predictions
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        artifacts_at = time.perf_counter()
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        profile_scores = (
            {
                "profile_preprocess_ms": (preprocessed_at - started) * 1000.0,
                "profile_primary_heads_ms": (
                    primary_heads_at - preprocessed_at
                )
                * 1000.0,
                "profile_anchor_blend_ms": (
                    anchor_blend_at - primary_heads_at
                )
                * 1000.0,
                "profile_map_postprocess_ms": (
                    maps_at - anchor_blend_at
                )
                * 1000.0,
                "profile_decode_ms": (decoded_at - maps_at) * 1000.0,
                "profile_artifacts_ms": (artifacts_at - decoded_at) * 1000.0,
            }
            if bool(self.config.get("profile_timings", False))
            else {}
        )
        if compact_gate_means is not None:
            mean_small_gate = float(compact_gate_means[0])
            mean_large_gate = float(compact_gate_means[1])
        elif gate is not None and valid.any():
            mean_small_gate = float(gate[0, valid].mean())
            mean_large_gate = float(gate[1, valid].mean())
        else:
            mean_small_gate = 0.0
            mean_large_gate = 0.0
        return SegmentationOutput(
            width=width,
            height=height,
            probability_path=destination / "foreground_probability.npy",
            binary_mask_path=binary_path,
            instances_path=instances_path,
            instances=instance_predictions,
            auxiliary_paths=auxiliary_paths,
            model_scores={
                "instance_count": float(len(instance_predictions)),
                "marker_count": float(decoded.marker_count),
                "mean_uncertainty": mean_uncertainty,
                "mean_small_gate": mean_small_gate,
                "mean_large_gate": mean_large_gate,
                **profile_scores,
            },
            runtime_ms=elapsed_ms,
        )

    def _predict_heads(self, image: np.ndarray) -> dict[str, np.ndarray]:
        patch_size = self._pair(self.config.get("patch_size"), "patch_size")
        stride = self._pair(self.config.get("stride"), "stride")
        patch_height, patch_width = patch_size
        stride_height, stride_width = stride
        original_height, original_width = image.shape
        target_height = max(original_height, patch_height)
        target_width = max(original_width, patch_width)
        target_height += (
            stride_height - (target_height - patch_height) % stride_height
        ) % stride_height
        target_width += (
            stride_width - (target_width - patch_width) % stride_width
        ) % stride_width
        padded = np.pad(
            image,
            ((0, target_height - original_height), (0, target_width - original_width)),
            mode="reflect",
        )
        y_starts = list(range(0, target_height - patch_height + 1, stride_height))
        x_starts = list(range(0, target_width - patch_width + 1, stride_width))
        output_names = self.config.get("output_names")
        if not isinstance(output_names, list) or not all(
            isinstance(name, str) for name in output_names
        ):
            raise ValueError("MSBI output_names must be a string list")
        accumulator_dtype = (
            np.float32
            if str(self.config.get("fusion_accumulator_dtype", "float64")) == "float32"
            else np.float64
        )
        sums: dict[str, np.ndarray] = {}
        weight_sum = np.zeros((target_height, target_width), dtype=accumulator_dtype)
        fusion = str(self.config.get("overlap_fusion", "hann"))
        if fusion == "hann":
            weight = self._hann_weight(
                patch_height,
                patch_width,
                dtype=accumulator_dtype,
            )
        elif fusion == "uniform":
            weight = np.ones(
                (patch_height, patch_width),
                dtype=accumulator_dtype,
            )
        else:
            raise ValueError("MSBI overlap_fusion must be hann or uniform")
        positions = [(y, x) for y in y_starts for x in x_starts]
        tile_batch_size = int(self.config.get("tile_batch_size", 1))
        if tile_batch_size <= 0:
            raise ValueError("MSBI tile_batch_size must be positive")
        for start in range(0, len(positions), tile_batch_size):
            batch_positions = positions[start : start + tile_batch_size]
            tiles = np.stack(
                [
                    padded[y : y + patch_height, x : x + patch_width]
                    for y, x in batch_positions
                ],
            )
            predicted_batch = self._predict_tile_batch(tiles, output_names)
            for batch_index, (y, x) in enumerate(batch_positions):
                predicted = {
                    name: values[batch_index]
                    for name, values in predicted_batch.items()
                }
                for name, values in predicted.items():
                    if name not in sums:
                        sums[name] = np.zeros(
                            (values.shape[0], target_height, target_width),
                            dtype=accumulator_dtype,
                        )
                    sums[name][:, y : y + patch_height, x : x + patch_width] += (
                        values * weight[None]
                    )
                weight_sum[y : y + patch_height, x : x + patch_width] += weight
        if np.any(weight_sum <= 0):
            raise RuntimeError("MSBI tiled fusion left uncovered pixels")
        return {
            name: np.asarray(
                (values / weight_sum[None])[:, :original_height, :original_width],
                dtype=np.float32,
            )
            for name, values in sums.items()
        }

    def _predict_tile_batch(
        self,
        tiles: np.ndarray,
        output_names: list[str],
    ) -> dict[str, np.ndarray]:
        tensor = self._torch.from_numpy(
            np.asarray(tiles[:, None], dtype=np.float32),
        ).to(self._device)
        inference_amp = bool(self.config.get("inference_amp", False))
        amp_dtype_name = str(self.config.get("inference_amp_dtype", "bfloat16"))
        amp_dtype = {
            "bfloat16": self._torch.bfloat16,
            "float16": self._torch.float16,
        }.get(amp_dtype_name)
        if amp_dtype is None:
            raise ValueError("MSBI inference_amp_dtype must be bfloat16 or float16")
        device_type = str(self._device).split(":", maxsplit=1)[0]
        with self._torch.inference_mode(), self._torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=inference_amp and device_type == "cuda",
        ):
            raw = self._model(tensor)
        if isinstance(raw, dict):
            parsed = {str(name): value for name, value in raw.items()}
        elif isinstance(raw, (tuple, list)) and len(raw) == len(output_names):
            parsed = dict(zip(output_names, raw, strict=True))
        else:
            raise ValueError("MSBI TorchScript output does not match the frozen head contract")
        result: dict[str, np.ndarray] = {}
        for name in output_names:
            value = parsed.get(name)
            if value is None:
                raise ValueError(f"MSBI TorchScript output is missing {name}")
            array = np.asarray(value.detach().float().cpu().numpy(), dtype=np.float32)
            if (
                array.ndim != 4
                or array.shape[0] != len(tiles)
                or array.shape[-2:] != tiles.shape[-2:]
            ):
                raise ValueError(f"MSBI head {name} has invalid shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"MSBI head {name} contains non-finite values")
            result[name] = array
        return result

    def _predict_runtime_compact(
        self,
        image: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        runtime_method = getattr(self._model, "forward_runtime", None)
        if not callable(runtime_method):
            raise ValueError(
                "MSBI runtime_compact_forward requires exported forward_runtime"
            )
        tensor = self._torch.from_numpy(
            np.asarray(image[None, None], dtype=np.float32),
        ).to(self._device)
        inference_amp = bool(self.config.get("inference_amp", False))
        amp_dtype_name = str(self.config.get("inference_amp_dtype", "bfloat16"))
        amp_dtype = {
            "bfloat16": self._torch.bfloat16,
            "float16": self._torch.float16,
        }.get(amp_dtype_name)
        if amp_dtype is None:
            raise ValueError("MSBI inference_amp_dtype must be bfloat16 or float16")
        device_type = str(self._device).split(":", maxsplit=1)[0]
        with self._torch.inference_mode(), self._torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=inference_amp and device_type == "cuda",
        ):
            logits, gate_means = runtime_method(tensor)
        logits_array = np.asarray(
            logits.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        gate_array = np.asarray(
            gate_means.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        if logits_array.shape != (1, 1, *image.shape):
            raise ValueError(
                f"MSBI compact foreground has invalid shape: {logits_array.shape}"
            )
        if gate_array.shape != (1, 2) or not (
            np.isfinite(logits_array).all() and np.isfinite(gate_array).all()
        ):
            raise ValueError("MSBI compact runtime outputs are invalid")
        return {"foreground_logits": logits_array[0]}, gate_array[0]

    def _predict_anchor_probability(
        self,
        image: np.ndarray,
        settings: dict[str, Any],
    ) -> np.ndarray:
        patch_height, patch_width = self._pair(
            settings.get("patch_size", [256, 256]),
            "anchor_multiscale_blend.patch_size",
        )
        stride_height, stride_width = self._pair(
            settings.get("stride", [128, 128]),
            "anchor_multiscale_blend.stride",
        )
        if stride_height > patch_height or stride_width > patch_width:
            raise ValueError("MSBI anchor blend stride must not exceed patch size")
        original_height, original_width = image.shape
        target_height = max(original_height, patch_height)
        target_width = max(original_width, patch_width)
        target_height += (
            stride_height - (target_height - patch_height) % stride_height
        ) % stride_height
        target_width += (
            stride_width - (target_width - patch_width) % stride_width
        ) % stride_width
        padded = np.pad(
            image,
            ((0, target_height - original_height), (0, target_width - original_width)),
            mode="reflect",
        )
        positions = [
            (y, x)
            for y in range(0, target_height - patch_height + 1, stride_height)
            for x in range(0, target_width - patch_width + 1, stride_width)
        ]
        accumulator_dtype = (
            np.float32
            if str(settings.get("fusion_accumulator_dtype", "float32")) == "float32"
            else np.float64
        )
        fusion = str(settings.get("overlap_fusion", "uniform"))
        if fusion == "uniform":
            weight = np.ones(
                (patch_height, patch_width),
                dtype=accumulator_dtype,
            )
        elif fusion == "hann":
            weight = self._hann_weight(
                patch_height,
                patch_width,
                dtype=accumulator_dtype,
            )
        else:
            raise ValueError("MSBI anchor blend fusion must be uniform or hann")
        probability_sum = np.zeros(
            (target_height, target_width),
            dtype=accumulator_dtype,
        )
        weight_sum = np.zeros_like(probability_sum)
        batch_size = int(settings.get("tile_batch_size", 16))
        if batch_size <= 0:
            raise ValueError("MSBI anchor blend tile_batch_size must be positive")
        device_type = str(self._device).split(":", maxsplit=1)[0]
        if device_type == "cuda" and bool(settings.get("gpu_fusion", True)):
            return self._predict_anchor_probability_cuda(
                padded,
                positions=positions,
                patch_size=(patch_height, patch_width),
                original_size=(original_height, original_width),
                batch_size=batch_size,
                weight=weight,
            )
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            tiles = np.stack(
                [
                    padded[y : y + patch_height, x : x + patch_width]
                    for y, x in batch_positions
                ],
            )
            probabilities = self._predict_anchor_tile_batch(tiles)
            for index, (y, x) in enumerate(batch_positions):
                probability_sum[
                    y : y + patch_height,
                    x : x + patch_width,
                ] += probabilities[index] * weight
                weight_sum[y : y + patch_height, x : x + patch_width] += weight
        if np.any(weight_sum <= 0):
            raise RuntimeError("MSBI anchor blend left uncovered pixels")
        return np.asarray(
            (probability_sum / weight_sum)[:original_height, :original_width],
            dtype=np.float32,
        )

    def _predict_anchor_probability_cuda(
        self,
        padded: np.ndarray,
        *,
        positions: list[tuple[int, int]],
        patch_size: tuple[int, int],
        original_size: tuple[int, int],
        batch_size: int,
        weight: np.ndarray,
    ) -> np.ndarray:
        anchor_method = getattr(self._model, "forward_anchor", None)
        if not callable(anchor_method):
            raise ValueError(
                "MSBI anchor_multiscale_blend requires exported forward_anchor"
            )
        patch_height, patch_width = patch_size
        original_height, original_width = original_size
        padded_tensor = self._torch.from_numpy(
            np.asarray(padded[None, None], dtype=np.float32),
        ).to(self._device)
        probability_sum = self._torch.zeros(
            padded.shape,
            dtype=self._torch.float32,
            device=self._device,
        )
        weight_sum = self._torch.zeros_like(probability_sum)
        weight_tensor = self._torch.from_numpy(
            np.asarray(weight, dtype=np.float32),
        ).to(self._device)
        inference_amp = bool(self.config.get("inference_amp", False))
        amp_dtype_name = str(self.config.get("inference_amp_dtype", "bfloat16"))
        amp_dtype = {
            "bfloat16": self._torch.bfloat16,
            "float16": self._torch.float16,
        }.get(amp_dtype_name)
        if amp_dtype is None:
            raise ValueError("MSBI inference_amp_dtype must be bfloat16 or float16")
        with self._torch.inference_mode(), self._torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=inference_amp,
        ):
            for start in range(0, len(positions), batch_size):
                batch_positions = positions[start : start + batch_size]
                tiles = self._torch.cat(
                    [
                        padded_tensor[
                            :,
                            :,
                            y : y + patch_height,
                            x : x + patch_width,
                        ]
                        for y, x in batch_positions
                    ],
                    dim=0,
                )
                probabilities = self._torch.sigmoid(anchor_method(tiles))[:, 0].float()
                for index, (y, x) in enumerate(batch_positions):
                    probability_sum[
                        y : y + patch_height,
                        x : x + patch_width,
                    ].add_(probabilities[index] * weight_tensor)
                    weight_sum[
                        y : y + patch_height,
                        x : x + patch_width,
                    ].add_(weight_tensor)
        if bool(self._torch.any(weight_sum <= 0).item()):
            raise RuntimeError("MSBI CUDA anchor blend left uncovered pixels")
        result = (
            probability_sum[:original_height, :original_width]
            / weight_sum[:original_height, :original_width]
        )
        return np.asarray(result.cpu().numpy(), dtype=np.float32)

    def _predict_anchor_tile_batch(self, tiles: np.ndarray) -> np.ndarray:
        anchor_method = getattr(self._model, "forward_anchor", None)
        if not callable(anchor_method):
            raise ValueError(
                "MSBI anchor_multiscale_blend requires exported forward_anchor"
            )
        tensor = self._torch.from_numpy(
            np.asarray(tiles[:, None], dtype=np.float32),
        ).to(self._device)
        inference_amp = bool(self.config.get("inference_amp", False))
        amp_dtype_name = str(self.config.get("inference_amp_dtype", "bfloat16"))
        amp_dtype = {
            "bfloat16": self._torch.bfloat16,
            "float16": self._torch.float16,
        }.get(amp_dtype_name)
        if amp_dtype is None:
            raise ValueError("MSBI inference_amp_dtype must be bfloat16 or float16")
        device_type = str(self._device).split(":", maxsplit=1)[0]
        with self._torch.inference_mode(), self._torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=inference_amp and device_type == "cuda",
        ):
            logits = anchor_method(tensor)
            probability = self._torch.sigmoid(logits)
        array = np.asarray(
            probability.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        expected = (len(tiles), 1, tiles.shape[-2], tiles.shape[-1])
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"MSBI anchor output has invalid shape/content: {array.shape}")
        return array[:, 0]

    def _validate_expected_shape(self, height: int, width: int) -> None:
        expected_height = self.metadata.expected_input_height
        expected_width = self.metadata.expected_input_width
        if (
            expected_height is not None
            and expected_width is not None
            and (height, width) != (expected_height, expected_width)
        ):
            raise ValueError(
                f"MSBI expected image {(expected_width, expected_height)}, "
                f"observed {(width, height)}"
            )

    def _normalize(self, image: np.ndarray, valid_height: int) -> np.ndarray:
        normalization = str(self.config.get("normalization", "percentile"))
        if normalization == "fixed":
            pixel_scale = float(self.config.get("pixel_scale", 255.0))
            if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                raise ValueError("MSBI pixel_scale must be finite and positive")
            return np.asarray(
                np.clip(image / pixel_scale, 0.0, 1.0),
                dtype=np.float32,
            )
        if normalization != "percentile":
            raise ValueError("MSBI normalization must be percentile or fixed")
        lower = float(self.config.get("lower_percentile", 1.0))
        upper = float(self.config.get("upper_percentile", 99.0))
        low, high = np.percentile(image[:valid_height], (lower, upper))
        if high <= low:
            return np.zeros(image.shape, dtype=np.float32)
        return np.asarray(
            np.clip((image - low) / (high - low), 0.0, 1.0),
            dtype=np.float32,
        )

    @staticmethod
    def _pair(value: Any, name: str) -> tuple[int, int]:
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in value
            )
        ):
            raise ValueError(f"MSBI {name} must be two positive integers")
        return int(value[0]), int(value[1])

    @staticmethod
    def _hann_weight(
        height: int,
        width: int,
        *,
        dtype: type[np.float32] | type[np.float64] = np.float64,
    ) -> np.ndarray:
        vertical = np.asarray(np.hanning(height), dtype=dtype)
        horizontal = np.asarray(np.hanning(width), dtype=dtype)
        vertical_values = cast(Any, vertical)
        horizontal_values = cast(Any, horizontal)
        window = vertical_values[:, np.newaxis] * horizontal_values[np.newaxis, :]
        return np.asarray(
            np.maximum(window, 1e-3),
            dtype=dtype,
        )

    @staticmethod
    def _instance_predictions_from_labels(
        labels: np.ndarray,
        confidences: tuple[float, ...],
    ) -> list[InstancePrediction]:
        instance_count = int(labels.max())
        if instance_count != len(confidences):
            raise ValueError("MSBI instance labels and confidences differ in length")
        sizes = np.bincount(labels.ravel(), minlength=instance_count + 1)
        regions = ndi.find_objects(labels, max_label=instance_count)
        predictions: list[InstancePrediction] = []
        for index, region in enumerate(regions, start=1):
            if region is None:
                raise ValueError(f"MSBI instance label {index} has no pixels")
            predictions.append(
                InstancePrediction(
                    instance_index=index,
                    bbox=(
                        int(region[1].start or 0),
                        int(region[0].start or 0),
                        int(region[1].stop or 0),
                        int(region[0].stop or 0),
                    ),
                    area_px=int(sizes[index]),
                    confidence=confidences[index - 1],
                )
            )
        return predictions

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float32)

    @staticmethod
    def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exponent = np.exp(np.clip(shifted, -40.0, 40.0))
        return np.asarray(
            exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1e-12),
            dtype=np.float32,
        )

    @staticmethod
    def _uncertainty(
        *,
        foreground: np.ndarray,
        center: np.ndarray,
        small: np.ndarray,
        large: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        probability = np.clip(foreground, 1e-6, 1.0 - 1e-6)
        entropy = -(
            probability * np.log(probability)
            + (1.0 - probability) * np.log(1.0 - probability)
        ) / math.log(2.0)
        disagreement = np.abs(small - large)
        center_conflict = center * (1.0 - foreground)
        uncertainty = np.clip(
            0.55 * entropy + 0.30 * disagreement + 0.15 * center_conflict,
            0.0,
            1.0,
        ).astype(np.float32)
        uncertainty[~valid] = 0.0
        return np.asarray(uncertainty, dtype=np.float32)

    @staticmethod
    def _write_preview_layers(
        *,
        destination: Path,
        center: np.ndarray,
        boundary: np.ndarray,
        distance: np.ndarray,
        labels: np.ndarray,
        gate_small: np.ndarray,
        gate_large: np.ndarray,
        uncertainty: np.ndarray,
    ) -> dict[str, Path]:
        previews = {
            "center_probability": center,
            "boundary_probability": boundary,
            "distance_field": distance,
            "gate_small": gate_small,
            "gate_large": gate_large,
            "uncertainty": uncertainty,
        }
        paths: dict[str, Path] = {}
        for name, values in previews.items():
            path = destination / f"{name}.png"
            Image.fromarray(
                np.clip(values * 255.0, 0, 255).astype(np.uint8),
                mode="L",
            ).save(path)
            paths[name] = path
        label_path = destination / "instance_labels.png"
        red = (labels.astype(np.uint32) * 67) % 255
        green = (labels.astype(np.uint32) * 131) % 255
        blue = (labels.astype(np.uint32) * 197) % 255
        color = np.stack((red, green, blue), axis=-1).astype(np.uint8)
        color[labels == 0] = 0
        Image.fromarray(color, mode="RGB").save(label_path)
        paths["instance_labels"] = label_path
        return paths

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported MSBI device: {requested}")
        return requested

    def _release(self) -> None:
        self._model = None
        self._torch = None
