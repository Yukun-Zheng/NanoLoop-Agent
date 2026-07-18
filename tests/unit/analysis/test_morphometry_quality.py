import numpy as np
import pytest

from app.analysis.config import MorphometryConfig, QualityGateConfig
from app.analysis.morphometry import measure
from app.analysis.postprocessing import NormalizedInstance
from app.analysis.quality import QualityInputs, evaluate
from app.contracts.enums import QualityStatus


def _instance(index: int, x: int, y: int, *, confidence: float = 0.9):
    mask = np.zeros((100, 100), dtype=bool)
    mask[y : y + 10, x : x + 10] = True
    return NormalizedInstance(
        instance_index=index,
        mask=mask,
        bbox=(x, y, x + 10, y + 10),
        area_px=100,
        confidence=confidence,
        touches_roi_boundary=False,
    )


def test_morphometry_uses_roi_area_and_physical_scale() -> None:
    roi = np.ones((100, 100), dtype=bool)
    result = measure(
        run_id="run_1",
        instances=[_instance(1, 10, 10), _instance(2, 40, 40)],
        roi_mask=roi,
        scale_nm_per_pixel=2.0,
        config=MorphometryConfig(),
    )
    summary = result.image_summary
    assert summary.particle_count == 2
    assert summary.coverage_ratio == pytest.approx(0.02)
    assert summary.number_density_px2 == pytest.approx(0.0002)
    assert summary.number_density_um2 == pytest.approx(50.0)
    assert summary.mean_equivalent_diameter_nm == pytest.approx(22.5676, rel=1e-3)
    assert not result.warnings


def test_missing_scale_keeps_pixel_metrics_and_marks_warning() -> None:
    result = measure(
        run_id="run_1",
        instances=[],
        roi_mask=np.ones((10, 10), dtype=bool),
        scale_nm_per_pixel=None,
        config=MorphometryConfig(),
    )
    assert result.image_summary.particle_count == 0
    assert result.image_summary.number_density_um2 is None
    assert result.image_summary.mean_equivalent_diameter_px is None
    assert result.warnings == ["physical_scale_missing_pixel_metrics_only"]


def test_quality_gate_exposes_review_and_warning_reasons() -> None:
    config = QualityGateConfig()
    review = evaluate(
        QualityInputs(
            roi_area_px=10_000,
            foreground_area_px=0,
            instances=[],
            minimum_area_px=8,
            validation_warnings=[],
        ),
        config,
    )
    assert review.status == QualityStatus.REVIEW_REQUIRED
    assert "foreground_ratio_too_low" in review.reasons

    warning = evaluate(
        QualityInputs(
            roi_area_px=10_000,
            foreground_area_px=1_000,
            instances=[_instance(1, 10, 10, confidence=0.2)],
            minimum_area_px=8,
            validation_warnings=[],
        ),
        config,
    )
    assert warning.status == QualityStatus.WARN
    assert "model_confidence_low" in warning.reasons
