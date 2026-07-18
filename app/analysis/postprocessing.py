"""Normalize semantic and native-instance model outputs into deterministic instances."""

import hashlib
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import watershed

from app.analysis.config import PostprocessProfile
from app.core.errors import InvalidImageError

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class NormalizedInstance:
    instance_index: int
    mask: BoolArray
    bbox: tuple[int, int, int, int]
    area_px: int
    confidence: float | None
    touches_roi_boundary: bool


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    """Final instances plus diagnostics observed before border exclusion."""

    instances: list[NormalizedInstance]
    candidate_count: int
    boundary_candidate_count: int
    excluded_border_count: int


def _validate_shapes(mask: NDArray[np.generic], roi_mask: BoolArray) -> None:
    if mask.shape != roi_mask.shape:
        raise InvalidImageError(
            details={
                "reason": "model_output_shape_mismatch",
                "mask_shape": mask.shape,
                "expected_shape": roi_mask.shape,
            }
        )


def _touches_roi_boundary(mask: BoolArray, roi_mask: BoolArray) -> bool:
    touches_image_edge = bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )
    dilated = ndi.binary_dilation(mask)
    return touches_image_edge or bool(np.any(dilated & ~roi_mask))


def _bbox(mask: BoolArray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _instance_from_mask(
    mask: BoolArray,
    *,
    confidence: float | None,
    roi_mask: BoolArray,
) -> NormalizedInstance:
    return NormalizedInstance(
        instance_index=0,
        mask=mask,
        bbox=_bbox(mask),
        area_px=int(mask.sum()),
        confidence=confidence,
        touches_roi_boundary=_touches_roi_boundary(mask, roi_mask),
    )


def _mask_digest(mask: BoolArray) -> bytes:
    """Return a stable compact tie-breaker for geometrically identical candidates."""

    packed = np.packbits(mask, bitorder="little")
    return hashlib.sha256(packed.tobytes()).digest()


def _spatial_key(instance: NormalizedInstance) -> tuple[object, ...]:
    confidence = instance.confidence
    return (
        instance.bbox[1],
        instance.bbox[0],
        instance.bbox[3],
        instance.bbox[2],
        instance.area_px,
        confidence is None,
        -(confidence or 0.0),
        _mask_digest(instance.mask),
    )


def _finalize(instances: list[NormalizedInstance]) -> list[NormalizedInstance]:
    ordered = sorted(instances, key=_spatial_key)
    return [replace(instance, instance_index=index) for index, instance in enumerate(ordered, 1)]


def _labels(binary: BoolArray, profile: PostprocessProfile) -> NDArray[np.int32]:
    if not profile.watershed_enabled or not binary.any():
        return cast(
            NDArray[np.int32], label(binary, connectivity=profile.connectivity).astype(np.int32)
        )
    distance = ndi.distance_transform_edt(binary)
    local_max = distance == ndi.maximum_filter(distance, size=3)
    markers = label(local_max & binary, connectivity=profile.connectivity)
    return cast(NDArray[np.int32], watershed(-distance, markers, mask=binary).astype(np.int32))


def _remove_small(binary: BoolArray, minimum_area: int, connectivity: int) -> BoolArray:
    labels = label(binary, connectivity=connectivity)
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    return cast(BoolArray, keep[labels])


def normalize_semantic_mask(
    mask: NDArray[np.generic],
    *,
    roi_mask: BoolArray,
    profile: PostprocessProfile,
    probability: NDArray[np.floating] | None = None,
) -> list[NormalizedInstance]:
    return normalize_semantic_mask_detailed(
        mask,
        roi_mask=roi_mask,
        profile=profile,
        probability=probability,
    ).instances


def normalize_semantic_mask_detailed(
    mask: NDArray[np.generic],
    *,
    roi_mask: BoolArray,
    profile: PostprocessProfile,
    probability: NDArray[np.floating] | None = None,
) -> PostprocessResult:
    _validate_shapes(mask, roi_mask)
    if probability is not None:
        _validate_shapes(probability, roi_mask)
    binary = np.asarray(mask, dtype=np.bool_) & roi_mask
    if profile.fill_holes:
        # Internal invalid ROI regions look like holes to morphology. Re-apply
        # the immutable analysis boundary after every operation that can grow
        # foreground so excluded pixels can never re-enter scientific output.
        binary = np.asarray(ndi.binary_fill_holes(binary), dtype=np.bool_) & roi_mask
    if profile.min_area_px > 0:
        binary = _remove_small(binary, profile.min_area_px, profile.connectivity)

    instances: list[NormalizedInstance] = []
    labels = _labels(binary, profile)
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if int(region.sum()) < profile.min_area_px:
            continue
        confidence = float(np.mean(probability[region])) if probability is not None else None
        instance = _instance_from_mask(region, confidence=confidence, roi_mask=roi_mask)
        instances.append(instance)
    return _apply_border_policy(_finalize(instances), profile=profile)


def _iou(a: BoolArray, b: BoolArray) -> float:
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else 0.0


def normalize_native_instances(
    masks: list[NDArray[np.generic]],
    *,
    roi_mask: BoolArray,
    profile: PostprocessProfile,
    confidences: list[float | None] | None = None,
) -> list[NormalizedInstance]:
    return normalize_native_instances_detailed(
        masks,
        roi_mask=roi_mask,
        profile=profile,
        confidences=confidences,
    ).instances


def normalize_native_instances_detailed(
    masks: list[NDArray[np.generic]],
    *,
    roi_mask: BoolArray,
    profile: PostprocessProfile,
    confidences: list[float | None] | None = None,
) -> PostprocessResult:
    if confidences is None:
        confidences = [None] * len(masks)
    if len(confidences) != len(masks):
        raise ValueError("confidences and masks must have equal length")
    candidates: list[NormalizedInstance] = []
    for raw, confidence in zip(masks, confidences, strict=True):
        _validate_shapes(raw, roi_mask)
        binary = np.asarray(raw, dtype=np.bool_) & roi_mask
        if profile.fill_holes:
            binary = np.asarray(ndi.binary_fill_holes(binary), dtype=np.bool_) & roi_mask
        area = int(binary.sum())
        if area == 0 or area < profile.min_area_px:
            continue
        instance = _instance_from_mask(binary, confidence=confidence, roi_mask=roi_mask)
        candidates.append(instance)

    # Highest confidence wins for duplicates. Geometric and content tie-breakers make
    # the winner independent of adapter output order when confidences are equal.
    candidates.sort(
        key=lambda item: (
            -(item.confidence if item.confidence is not None else -1.0),
            *_spatial_key(item),
        )
    )
    kept: list[NormalizedInstance] = []
    for candidate in candidates:
        if any(
            _iou(candidate.mask, existing.mask) >= profile.instance_iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return _apply_border_policy(_finalize(kept), profile=profile)


def _apply_border_policy(
    candidates: list[NormalizedInstance],
    *,
    profile: PostprocessProfile,
) -> PostprocessResult:
    boundary_count = sum(item.touches_roi_boundary for item in candidates)
    if profile.exclude_border:
        final = _finalize(
            [item for item in candidates if not item.touches_roi_boundary]
        )
    else:
        final = candidates
    return PostprocessResult(
        instances=final,
        candidate_count=len(candidates),
        boundary_candidate_count=boundary_count,
        excluded_border_count=len(candidates) - len(final),
    )
