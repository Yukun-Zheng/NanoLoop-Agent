"""Deterministic center/boundary-guided watershed instance decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.segmentation import watershed

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    mode: str = "watershed"
    foreground_threshold: float = 0.5
    center_threshold: float = 0.35
    center_nms_radius: int = 5
    boundary_threshold: float = 0.5
    boundary_penalty: float = 2.0
    min_area_px: int = 8
    max_area_px: int | None = None
    watershed_compactness: float = 0.0
    exclude_border: bool = False
    connectivity: int = 2
    fallback_peak_source: str = "distance"

    def __post_init__(self) -> None:
        if self.mode not in {"watershed", "connected_components"}:
            raise ValueError("mode must be watershed or connected_components")
        for name in ("foreground_threshold", "center_threshold", "boundary_threshold"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.center_nms_radius < 1:
            raise ValueError("center_nms_radius must be positive")
        if self.min_area_px < 0:
            raise ValueError("min_area_px cannot be negative")
        if self.max_area_px is not None and self.max_area_px < self.min_area_px:
            raise ValueError("max_area_px must be at least min_area_px")
        if self.connectivity not in {1, 2}:
            raise ValueError("connectivity must be 1 or 2")
        if self.fallback_peak_source not in {"distance", "sdf", "center"}:
            raise ValueError("fallback_peak_source must be distance, sdf, or center")


@dataclass(frozen=True, slots=True)
class DecodedInstances:
    labels: IntArray
    confidences: tuple[float, ...]
    marker_count: int


def _fallback_markers(
    foreground: NDArray[np.bool_],
    markers: IntArray,
    *,
    connectivity: int,
    fallback_score: FloatArray | None = None,
) -> IntArray:
    structure = ndi.generate_binary_structure(2, connectivity)
    components, count = ndi.label(foreground, structure=structure)
    result = markers.copy()
    next_marker = int(result.max()) + 1
    missing: list[tuple[int, tuple[slice, slice]]] = []
    for component_id, region in enumerate(
        ndi.find_objects(components, max_label=count),
        start=1,
    ):
        if region is None:
            continue
        component = components[region] == component_id
        if np.any(result[region][component] > 0):
            continue
        missing.append((component_id, region))
    if not missing:
        return result
    score = (
        ndi.distance_transform_edt(foreground).astype(np.float32)
        if fallback_score is None
        else fallback_score
    )
    for component_id, region in missing:
        component = components[region] == component_id
        values = np.where(component, score[region], -1.0)
        local_y, local_x = np.unravel_index(int(np.argmax(values)), values.shape)
        y = int(region[0].start or 0) + int(local_y)
        x = int(region[1].start or 0) + int(local_x)
        result[y, x] = next_marker
        next_marker += 1
    return result


def _stable_relabel(
    raw_labels: NDArray[np.generic],
    foreground_probability: FloatArray,
    config: DecodeConfig,
) -> tuple[IntArray, tuple[float, ...]]:
    raw = np.asarray(raw_labels, dtype=np.int32)
    label_count = int(raw.max())
    sizes = np.bincount(raw.ravel(), minlength=label_count + 1)
    probability_sums = np.bincount(
        raw.ravel(),
        weights=foreground_probability.ravel(),
        minlength=label_count + 1,
    )
    border_ids = set(
        np.unique(
            np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1])),
        ).tolist()
    )
    records: list[tuple[tuple[int, int, int, int, int], int, float]] = []
    for label_id, region in enumerate(
        ndi.find_objects(raw, max_label=label_count),
        start=1,
    ):
        if region is None:
            continue
        area = int(sizes[label_id])
        if area < config.min_area_px:
            continue
        if config.max_area_px is not None and area > config.max_area_px:
            continue
        if config.exclude_border and label_id in border_ids:
            continue
        y_start = int(region[0].start or 0)
        x_start = int(region[1].start or 0)
        y_stop = int(region[0].stop or 0)
        x_stop = int(region[1].stop or 0)
        key = (y_start, x_start, y_stop - 1, x_stop - 1, area)
        confidence = float(probability_sums[label_id] / max(area, 1))
        records.append((key, label_id, confidence))
    records.sort(key=lambda item: item[0])
    lookup = np.zeros(label_count + 1, dtype=np.int32)
    for index, (_key, label_id, _confidence) in enumerate(records, start=1):
        lookup[label_id] = index
    labels = lookup[raw]
    return np.asarray(labels, dtype=np.int32), tuple(
        confidence for _key, _label_id, confidence in records
    )


def decode_instances(
    foreground_probability: NDArray[np.generic],
    center_probability: NDArray[np.generic],
    boundary_probability: NDArray[np.generic],
    *,
    distance_field: NDArray[np.generic] | None = None,
    valid_mask: NDArray[np.generic] | None = None,
    config: DecodeConfig | None = None,
) -> DecodedInstances:
    """Decode all fused heads together so tile edges never create instance IDs."""

    settings = config or DecodeConfig()
    foreground_prob = np.asarray(foreground_probability, dtype=np.float32)
    center_prob = np.asarray(center_probability, dtype=np.float32)
    boundary_prob = np.asarray(boundary_probability, dtype=np.float32)
    if (
        foreground_prob.ndim != 2
        or center_prob.shape != foreground_prob.shape
        or boundary_prob.shape != foreground_prob.shape
    ):
        raise ValueError("foreground, center, and boundary maps must share a 2-D shape")
    if not np.isfinite(foreground_prob).all():
        raise ValueError("foreground decoder input must be finite")
    if settings.mode != "connected_components" and not all(
        np.isfinite(values).all() for values in (center_prob, boundary_prob)
    ):
        raise ValueError("center and boundary decoder inputs must be finite")
    valid = (
        np.ones(foreground_prob.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != foreground_prob.shape:
        raise ValueError("valid_mask shape must match decoder maps")
    foreground = (foreground_prob >= settings.foreground_threshold) & valid
    if not foreground.any():
        return DecodedInstances(
            labels=np.zeros(foreground.shape, dtype=np.int32),
            confidences=(),
            marker_count=0,
        )
    structure = ndi.generate_binary_structure(2, settings.connectivity)
    if settings.mode == "connected_components":
        raw_labels, component_count = ndi.label(foreground, structure=structure)
        labels, confidences = _stable_relabel(
            raw_labels,
            foreground_prob,
            settings,
        )
        return DecodedInstances(
            labels=labels,
            confidences=confidences,
            marker_count=int(component_count),
        )
    size = 2 * settings.center_nms_radius + 1
    local_maximum = center_prob == ndi.maximum_filter(
        center_prob,
        size=size,
        mode="constant",
        cval=0.0,
    )
    center_candidates = (
        local_maximum
        & (center_prob >= settings.center_threshold)
        & foreground
        & (boundary_prob < settings.boundary_threshold)
    )
    markers, _ = ndi.label(center_candidates, structure=structure)
    sdf: FloatArray | None = None
    if distance_field is not None:
        supplied_sdf = np.asarray(distance_field, dtype=np.float32)
        if supplied_sdf.shape != foreground.shape or not np.isfinite(supplied_sdf).all():
            raise ValueError("distance_field must be finite and match decoder maps")
        sdf = supplied_sdf
    fallback_score = (
        sdf
        if settings.fallback_peak_source == "sdf" and sdf is not None
        else center_prob
        if settings.fallback_peak_source == "center"
        else None
    )
    markers = _fallback_markers(
        foreground,
        np.asarray(markers, dtype=np.int32),
        connectivity=settings.connectivity,
        fallback_score=fallback_score,
    )
    if sdf is not None:
        base_elevation = -sdf
    else:
        distance = ndi.distance_transform_edt(foreground).astype(np.float32)
        maximum = float(distance.max())
        base_elevation = -(distance / maximum if maximum > 0 else distance)
    elevation = (
        np.asarray(base_elevation, dtype=np.float32)
        + settings.boundary_penalty * boundary_prob
    )
    raw = watershed(
        elevation,
        markers=markers,
        mask=foreground,
        connectivity=structure,
        compactness=settings.watershed_compactness,
    )
    labels, confidences = _stable_relabel(raw, foreground_prob, settings)
    return DecodedInstances(
        labels=labels,
        confidences=confidences,
        marker_count=int(markers.max()),
    )
