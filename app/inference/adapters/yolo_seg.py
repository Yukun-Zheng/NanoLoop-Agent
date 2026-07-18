"""Ultralytics YOLO segmentation adapter with no import-time ML dependency."""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
from PIL import Image

from app.contracts.enums import RoiMode
from app.contracts.inference import InstancePrediction, SegmentationOutput, SegmentationRequest
from app.inference.adapters._utils import (
    embed_box_mask,
    iter_box_crops,
    mask_bbox,
    open_rgb,
    output_dir,
    remove_small_components,
    save_binary_mask,
)
from app.inference.adapters.base import BaseSegmentationAdapter


class YOLOSegAdapter(BaseSegmentationAdapter):
    """Expose Ultralytics instance masks through the shared output contract."""

    _model: Any = None

    def load(self, device: str) -> None:
        try:
            ultralytics = importlib.import_module("ultralytics")
            runtime_weight = self._materialize_runtime_weight()
            self._model = ultralytics.YOLO(str(runtime_weight), task="segment")
            self._mark_loaded(device)
        except Exception as exc:
            self._mark_load_failed(exc)
            raise

    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        self._require_loaded()
        started = time.perf_counter()
        image = open_rgb(request.image_bytes or request.image_path)
        height, width = image.shape[:2]
        threshold = request.threshold
        if threshold is None:
            threshold = self.metadata.default_threshold or 0.25
        kwargs: dict[str, Any] = {
            "conf": threshold,
            "seed": request.seed,
            "deterministic": True,
            "verbose": False,
        }
        if self._device and self._device != "auto":
            kwargs["device"] = self._device
        masks: list[np.ndarray] = []
        confidences: list[float | None] = []
        if request.roi_mode == RoiMode.BOXES:
            for crop in iter_box_crops(
                image,
                request.boxes,
                context_px=request.roi_context_px,
            ):
                local_masks, local_scores = self._predict_masks(
                    crop.image,
                    target_shape=crop.image.shape[:2],
                    kwargs=kwargs,
                )
                for local_mask, confidence in zip(
                    local_masks, local_scores, strict=True
                ):
                    instance = embed_box_mask(local_mask, crop, (height, width))
                    instance = remove_small_components(instance, request.min_area_px)
                    if instance.any():
                        masks.append(instance)
                        confidences.append(confidence)
        else:
            masks, confidences = self._predict_masks(
                image,
                target_shape=(height, width),
                kwargs=kwargs,
            )
            masks = [remove_small_components(mask, request.min_area_px) for mask in masks]
            nonempty = [index for index, mask in enumerate(masks) if mask.any()]
            masks = [masks[index] for index in nonempty]
            confidences = [confidences[index] for index in nonempty]

        union = np.logical_or.reduce(masks) if masks else np.zeros((height, width), dtype=bool)
        destination = output_dir(request.run_dir, self.metadata.model_id)
        binary_path = destination / "binary_mask.png"
        instances_path = destination / "instances.npz"
        save_binary_mask(binary_path, union)
        np.savez_compressed(
            instances_path,
            masks=np.asarray(masks, dtype=bool),
            confidences=np.asarray(
                [value if value is not None else np.nan for value in confidences]
            ),
        )
        instances = [
            InstancePrediction(
                instance_index=index,
                bbox=mask_bbox(mask),
                area_px=int(mask.sum()),
                confidence=confidence,
            )
            for index, (mask, confidence) in enumerate(
                zip(masks, confidences, strict=True), start=1
            )
        ]
        scores = {}
        present_confidences = [item for item in confidences if item is not None]
        if present_confidences:
            scores["mean_confidence"] = float(np.mean(present_confidences))
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return SegmentationOutput(
            width=width,
            height=height,
            binary_mask_path=binary_path,
            instances_path=instances_path,
            instances=instances,
            model_scores=scores,
            runtime_ms=elapsed_ms,
        )

    def _predict_masks(
        self,
        source: str | np.ndarray,
        *,
        target_shape: tuple[int, int],
        kwargs: dict[str, Any],
    ) -> tuple[list[np.ndarray], list[float | None]]:
        result = self._model.predict(source=source, **kwargs)[0]
        if result.masks is None:
            return [], []
        raw_masks = np.asarray(result.masks.data.detach().cpu().numpy())
        raw_confidences = (
            np.asarray(result.boxes.conf.detach().cpu().numpy())
            if result.boxes is not None
            else np.full(len(raw_masks), np.nan)
        )
        height, width = target_shape
        masks: list[np.ndarray] = []
        confidences: list[float | None] = []
        for raw_mask, confidence in zip(raw_masks, raw_confidences, strict=True):
            resized = np.asarray(
                Image.fromarray(raw_mask.astype(np.float32), mode="F").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
            masks.append(np.asarray(resized >= 0.5, dtype=bool))
            confidences.append(None if np.isnan(confidence) else float(confidence))
        return masks, confidences

    def _release(self) -> None:
        self._model = None
