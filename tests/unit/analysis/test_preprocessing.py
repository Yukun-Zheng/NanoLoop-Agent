import numpy as np
import pytest

from app.analysis.preprocessing import build_analysis_roi
from app.analysis.transforms import TransformRecord
from app.contracts.analyses import AnalysisROI, InvalidPixelRegion, PixelRect, ROIBox
from app.contracts.enums import RoiMode
from app.core.errors import InvalidBoxError


def test_full_and_overlapping_box_roi_use_pixel_union() -> None:
    analysis_roi = AnalysisROI(
        valid_rect=PixelRect(x1=0, y1=0, x2=100, y2=100),
        invalid_rects=[InvalidPixelRegion(x1=0, y1=90, x2=100, y2=100)],
    )
    full = build_analysis_roi(
        width=100,
        height=100,
        analysis_roi=analysis_roi,
        roi_mode=RoiMode.FULL_IMAGE,
        boxes=[],
    )
    assert int(full.sum()) == 9_000

    boxed = build_analysis_roi(
        width=100,
        height=100,
        analysis_roi=analysis_roi,
        roi_mode=RoiMode.BOXES,
        boxes=[
            ROIBox(x1=10, y1=10, x2=60, y2=60),
            ROIBox(x1=40, y1=10, x2=90, y2=60),
        ],
    )
    assert int(boxed.sum()) == 4_000
    assert not boxed[0, 0]
    assert boxed[20, 50]


def test_boxes_mode_requires_active_effective_box() -> None:
    roi = AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=100, y2=100))
    with pytest.raises(InvalidBoxError):
        build_analysis_roi(
            width=100,
            height=100,
            analysis_roi=roi,
            roi_mode=RoiMode.BOXES,
            boxes=[ROIBox(x1=10, y1=10, x2=50, y2=50, active=False)],
        )


def test_transform_round_trip_preserves_original_coordinates() -> None:
    transform = TransformRecord(
        original_width=200,
        original_height=150,
        crop=PixelRect(x1=20, y1=10, x2=180, y2=140),
    )
    original = ROIBox(x1=30, y1=20, x2=100, y2=80)
    analysis = transform.original_to_analysis(original)
    assert (analysis.x1, analysis.y1, analysis.x2, analysis.y2) == (10, 10, 80, 70)
    assert transform.analysis_to_original(analysis) == original


def test_roi_mask_dtype_is_boolean() -> None:
    roi = AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=4, y2=3))
    mask = build_analysis_roi(
        width=4,
        height=3,
        analysis_roi=roi,
        roi_mode=RoiMode.FULL_IMAGE,
        boxes=[],
    )
    assert mask.dtype == np.bool_
