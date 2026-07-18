"""Shared image loading, normalization, and effective ROI construction."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.analysis.config import PreprocessProfile
from app.analysis.transforms import TransformRecord
from app.contracts.analyses import AnalysisROI, ROIBox
from app.contracts.enums import RoiMode
from app.core.errors import InvalidBoxError, InvalidImageError

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]


def load_grayscale(path: str, profile: PreprocessProfile) -> FloatArray:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim == 3:
        rgb = raw[..., :3].astype(np.float32)
        raw = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif raw.ndim != 2:
        raise InvalidImageError(details={"reason": "unsupported_array_shape", "shape": raw.shape})
    values = raw.astype(np.float32, copy=False)
    if profile.normalization == "none":
        return values
    if profile.normalization == "percentile":
        low, high = np.percentile(
            values, [profile.lower_percentile, profile.upper_percentile]
        )
    else:
        low, high = float(values.min()), float(values.max())
    if not np.isfinite(low) or not np.isfinite(high):
        raise InvalidImageError(details={"reason": "non_finite_pixels"})
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return cast(
        FloatArray,
        np.clip((values - low) / (high - low), 0, 1).astype(np.float32),
    )


def build_analysis_roi(
    *,
    width: int,
    height: int,
    analysis_roi: AnalysisROI,
    roi_mode: RoiMode,
    boxes: list[ROIBox],
) -> BoolArray:
    """Build the valid-pixel mask; overlapping boxes contribute their pixel union once."""

    valid = analysis_roi.valid_rect
    if valid.x2 > width or valid.y2 > height:
        raise InvalidImageError(details={"reason": "analysis_roi_out_of_bounds"})
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[valid.y1 : valid.y2, valid.x1 : valid.x2] = True
    for region in analysis_roi.invalid_rects:
        if region.x2 > width or region.y2 > height:
            raise InvalidImageError(details={"reason": "invalid_region_out_of_bounds"})
        mask[region.y1 : region.y2, region.x1 : region.x2] = False

    if roi_mode == RoiMode.FULL_IMAGE:
        return mask
    active = [box for box in boxes if box.active]
    if not active:
        raise InvalidBoxError(details={"reason": "boxes_mode_requires_active_box"})
    box_union = np.zeros_like(mask)
    for box in active:
        if box.x2 > width or box.y2 > height:
            raise InvalidBoxError(details={"box_id": box.box_id, "reason": "out_of_bounds"})
        box_union[box.y1 : box.y2, box.x1 : box.x2] = True
    effective = mask & box_union
    if not effective.any():
        raise InvalidBoxError(details={"reason": "empty_effective_roi"})
    return effective


def create_transform(width: int, height: int, analysis_roi: AnalysisROI) -> TransformRecord:
    return TransformRecord(
        original_width=width,
        original_height=height,
        crop=analysis_roi.valid_rect,
    )
