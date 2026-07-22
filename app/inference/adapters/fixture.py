"""Deterministic segmentation fixture for engineering integration tests.

This adapter intentionally does not implement a scientific model.  It turns a
versioned YAML description into a repeatable probability mask so API, database,
storage, scheduling, provenance, and export code can be exercised before a real
checkpoint is available.  It is only reachable when an operator explicitly
selects the separate ``demo_data/model_artifacts/registry.yaml`` registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from app.contracts.enums import RoiMode
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.inference.adapters._utils import (
    apply_box_roi,
    open_rgb,
    output_dir,
    remove_small_components,
    save_binary_mask,
)
from app.inference.adapters.base import BaseSegmentationAdapter

_FIXTURE_WARNING = "simulated_fixture_output_not_scientific"


class DeterministicFixtureAdapter(BaseSegmentationAdapter):
    """Render configured normalized ellipses through the real adapter contract."""

    _objects: tuple[tuple[float, float, float, float, float], ...] = ()

    def load(self, device: str) -> None:
        try:
            if device != "cpu":
                raise ValueError("the deterministic fixture supports only the CPU device")
            if self.config.get("fixture_schema_version") != 1:
                raise ValueError("fixture_schema_version must be 1")
            if self.weight_bytes != b"NanoLoop deterministic fixture model v1\n":
                raise ValueError("unexpected deterministic fixture weight marker")
            objects = self.config.get("objects")
            if not isinstance(objects, list) or not objects:
                raise ValueError("fixture objects must be a non-empty list")
            self._objects = tuple(self._parse_object(item) for item in objects)
            self._mark_loaded(device)
        except Exception as error:
            self._mark_load_failed(error)
            raise

    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        self._require_loaded()
        image = open_rgb(request.image_bytes or request.image_path)
        height, width = image.shape[:2]
        probability = np.zeros((height, width), dtype=np.float32)
        yy, xx = np.ogrid[:height, :width]
        for center_x, center_y, radius_x, radius_y, score in self._objects:
            x = center_x * width
            y = center_y * height
            rx = max(radius_x * width, 1.0)
            ry = max(radius_y * height, 1.0)
            ellipse = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2 <= 1.0
            np.maximum(probability, ellipse.astype(np.float32) * score, out=probability)

        if request.roi_mode == RoiMode.BOXES:
            probability = np.asarray(
                apply_box_roi(probability, request.boxes),
                dtype=np.float32,
            )
        threshold = request.threshold
        if threshold is None:
            threshold = self.metadata.default_threshold or 0.5
        binary = remove_small_components(probability >= threshold, request.min_area_px)

        destination = output_dir(request.run_dir, self.metadata.model_id)
        probability_path = destination / "probability.npy"
        binary_path = destination / "binary_mask.png"
        np.save(probability_path, probability, allow_pickle=False)
        save_binary_mask(binary_path, binary)
        return SegmentationOutput(
            width=width,
            height=height,
            probability_path=probability_path,
            binary_mask_path=binary_path,
            model_scores={
                "fixture_object_count": float(len(self._objects)),
                "foreground_probability_mean": float(probability.mean()),
            },
            warnings=[_FIXTURE_WARNING],
            runtime_ms=0,
        )

    def _release(self) -> None:
        self._objects = ()

    @staticmethod
    def _parse_object(value: object) -> tuple[float, float, float, float, float]:
        if not isinstance(value, Mapping):
            raise ValueError("each fixture object must be a mapping")
        center = DeterministicFixtureAdapter._pair(value.get("center"), "center")
        radius = DeterministicFixtureAdapter._pair(value.get("radius"), "radius")
        score = value.get("score", 0.9)
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError("fixture object score must be numeric")
        values = (*center, *radius, float(score))
        if not all(np.isfinite(item) for item in values):
            raise ValueError("fixture object values must be finite")
        if not all(0 < item < 1 for item in (*center, *radius)):
            raise ValueError("fixture center and radius values must be between zero and one")
        if not 0 <= values[-1] <= 1:
            raise ValueError("fixture object score must be between zero and one")
        return values

    @staticmethod
    def _pair(value: object, name: str) -> tuple[float, float]:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
            raise ValueError(f"fixture object {name} must contain two numbers")
        first, second = value
        if (
            not isinstance(first, int | float)
            or isinstance(first, bool)
            or not isinstance(second, int | float)
            or isinstance(second, bool)
        ):
            raise ValueError(f"fixture object {name} must contain two numbers")
        return float(first), float(second)
