"""SAM2 adapter reserved for a separately supplied checkpoint and runtime."""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from app.contracts.enums import RoiMode
from app.contracts.inference import InstancePrediction, SegmentationOutput, SegmentationRequest
from app.inference.adapters._utils import (
    apply_box_roi,
    mask_bbox,
    open_rgb,
    output_dir,
    remove_small_components,
    save_binary_mask,
)
from app.inference.adapters.base import BaseSegmentationAdapter


class SAM2Adapter(BaseSegmentationAdapter):
    """Use the official SAM2 predictor for boxes and automatic generator otherwise."""

    _predictor: Any = None
    _generator: Any = None

    def load(self, device: str) -> None:
        try:
            model_config = self.config.get("model_config")
            if not isinstance(model_config, str) or not model_config:
                raise ValueError("SAM2 config requires model_config")
            build_module = importlib.import_module("sam2.build_sam")
            predictor_module = importlib.import_module("sam2.sam2_image_predictor")
            generator_module = importlib.import_module("sam2.automatic_mask_generator")
            model = build_module.build_sam2(
                model_config,
                str(self._materialize_runtime_weight()),
                device=device,
            )
            self._predictor = predictor_module.SAM2ImagePredictor(model)
            self._generator = generator_module.SAM2AutomaticMaskGenerator(model)
            self._mark_loaded(device)
        except Exception as exc:
            self._mark_load_failed(exc)
            raise

    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        self._require_loaded()
        started = time.perf_counter()
        image = open_rgb(request.image_bytes or request.image_path)
        height, width = image.shape[:2]
        masks: list[np.ndarray] = []
        scores: list[float | None] = []
        if request.roi_mode == RoiMode.BOXES:
            self._predictor.set_image(image)
            for box in request.boxes:
                if not box.active:
                    continue
                predicted, predicted_scores, _ = self._predictor.predict(
                    box=np.asarray([box.x1, box.y1, box.x2, box.y2], dtype=np.float32),
                    multimask_output=False,
                )
                if len(predicted):
                    mask = apply_box_roi(predicted[0], [box]).astype(bool)
                    mask = remove_small_components(mask, request.min_area_px)
                    if mask.any():
                        masks.append(mask)
                        scores.append(float(predicted_scores[0]))
        else:
            for item in self._generator.generate(image):
                mask = remove_small_components(item["segmentation"], request.min_area_px)
                if mask.any():
                    masks.append(mask)
                    value = item.get("predicted_iou")
                    scores.append(float(value) if value is not None else None)

        union = np.logical_or.reduce(masks) if masks else np.zeros((height, width), dtype=bool)
        destination = output_dir(request.run_dir, self.metadata.model_id)
        binary_path = destination / "binary_mask.png"
        instances_path = destination / "instances.npz"
        save_binary_mask(binary_path, union)
        np.savez_compressed(
            instances_path,
            masks=np.asarray(masks, dtype=bool),
            scores=np.asarray([value if value is not None else np.nan for value in scores]),
        )
        instances = [
            InstancePrediction(
                instance_index=index,
                bbox=mask_bbox(mask),
                area_px=int(mask.sum()),
                mask_score=score,
            )
            for index, (mask, score) in enumerate(zip(masks, scores, strict=True), start=1)
        ]
        present_scores = [score for score in scores if score is not None]
        model_scores = (
            {"mean_mask_score": float(np.mean(present_scores))} if present_scores else {}
        )
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return SegmentationOutput(
            width=width,
            height=height,
            probability_path=None,
            binary_mask_path=binary_path,
            instances_path=instances_path,
            instances=instances,
            model_scores=model_scores,
            runtime_ms=elapsed_ms,
        )

    def _release(self) -> None:
        self._predictor = None
        self._generator = None
