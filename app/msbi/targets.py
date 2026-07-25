"""Generate multi-task supervision from instance IDs or semantic pseudo-instances."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Frozen rules used to derive center, boundary, distance, and scale targets."""

    center_sigma_fraction: float = 0.35
    center_sigma_min: float = 1.0
    center_sigma_max: float = 8.0
    boundary_width_px: int = 2
    small_area_max_px: int = 256
    large_area_min_px: int = 2048
    connectivity: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.center_sigma_fraction <= 1:
            raise ValueError("center_sigma_fraction must be in (0, 1]")
        if self.center_sigma_min <= 0 or self.center_sigma_max < self.center_sigma_min:
            raise ValueError("center sigma bounds are invalid")
        if self.boundary_width_px < 1:
            raise ValueError("boundary_width_px must be positive")
        if self.small_area_max_px >= self.large_area_min_px:
            raise ValueError("small_area_max_px must be below large_area_min_px")
        if self.connectivity not in {1, 2}:
            raise ValueError("connectivity must be 1 or 2")


def semantic_to_pseudo_instances(
    semantic_mask: NDArray[np.generic],
    *,
    connectivity: int = 2,
) -> IntArray:
    """Convert a binary semantic mask to explicitly marked pseudo-instance IDs."""

    if connectivity not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2")
    structure = ndi.generate_binary_structure(2, connectivity)
    labels, _ = ndi.label(np.asarray(semantic_mask, dtype=bool), structure=structure)
    return np.asarray(labels, dtype=np.int32)


def _instance_boundary(mask: NDArray[np.bool_], width: int) -> NDArray[np.bool_]:
    eroded = ndi.binary_erosion(mask, iterations=width, border_value=0)
    return np.asarray(mask & ~eroded, dtype=bool)


def _gaussian_patch(
    shape: tuple[int, int],
    *,
    center_y: float,
    center_x: float,
    sigma: float,
) -> tuple[slice, slice, FloatArray]:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    y1 = max(0, int(np.floor(center_y)) - radius)
    y2 = min(shape[0], int(np.floor(center_y)) + radius + 1)
    x1 = max(0, int(np.floor(center_x)) - radius)
    x2 = min(shape[1], int(np.floor(center_x)) + radius + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    values = np.exp(
        -((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * sigma**2)
    )
    return slice(y1, y2), slice(x1, x2), np.asarray(values, dtype=np.float32)


def generate_instance_targets(
    instance_labels: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.generic] | None = None,
    config: TargetConfig | None = None,
) -> dict[str, NDArray[np.generic]]:
    """Create all MSBI targets without allowing invalid pixels into any loss."""

    settings = config or TargetConfig()
    labels = np.asarray(instance_labels, dtype=np.int32)
    if labels.ndim != 2 or np.any(labels < 0):
        raise ValueError("instance_labels must be a non-negative 2-D array")
    valid = (
        np.ones(labels.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != labels.shape:
        raise ValueError("valid_mask shape must match instance_labels")
    labels = np.where(valid, labels, 0).astype(np.int32, copy=False)
    foreground = labels > 0
    center = np.zeros(labels.shape, dtype=np.float32)
    boundary = np.zeros(labels.shape, dtype=np.float32)
    distance = np.zeros(labels.shape, dtype=np.float32)
    scale = np.full(labels.shape, -1, dtype=np.int64)
    area_by_instance: dict[int, int] = {}

    instance_slices = ndi.find_objects(labels)
    for instance_id, region_slices in enumerate(instance_slices, start=1):
        if region_slices is None:
            continue
        instance = labels[region_slices] == instance_id
        area = int(instance.sum())
        if area == 0:
            continue
        area_by_instance[int(instance_id)] = area
        coordinates = np.argwhere(instance)
        local_center_y, local_center_x = coordinates.mean(axis=0)
        center_y = float(local_center_y + region_slices[0].start)
        center_x = float(local_center_x + region_slices[1].start)
        equivalent_radius = sqrt(area / np.pi)
        sigma = float(
            np.clip(
                equivalent_radius * settings.center_sigma_fraction,
                settings.center_sigma_min,
                settings.center_sigma_max,
            )
        )
        ys, xs, values = _gaussian_patch(
            labels.shape,
            center_y=center_y,
            center_x=center_x,
            sigma=sigma,
        )
        center[ys, xs] = np.maximum(center[ys, xs], values)
        local_boundary = boundary[region_slices]
        local_boundary[_instance_boundary(instance, settings.boundary_width_px)] = 1.0
        local_distance = ndi.distance_transform_edt(instance).astype(np.float32)
        maximum = float(local_distance.max())
        if maximum > 0:
            local_target = distance[region_slices]
            local_target[instance] = local_distance[instance] / maximum
        local_scale = scale[region_slices]
        if area <= settings.small_area_max_px:
            local_scale[instance] = 0
        elif area >= settings.large_area_min_px:
            local_scale[instance] = 1
        else:
            # Medium instances provide a soft, spatial gate target through ignore.
            local_scale[instance] = -1

    valid_float = valid.astype(np.float32)
    return {
        "foreground": foreground.astype(np.float32) * valid_float,
        "center": center * valid_float,
        "boundary": boundary * valid_float,
        "distance": distance * valid_float,
        "scale": np.where(valid, scale, -1).astype(np.int64),
        "valid": valid_float,
        "instance_labels": labels,
        "area_by_instance": np.asarray(
            sorted(area_by_instance.items()), dtype=np.int64
        ).reshape((-1, 2)),
    }
