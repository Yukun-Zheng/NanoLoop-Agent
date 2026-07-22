"""Lightweight image/mask helpers shared by optional model adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from app.contracts.analyses import ROIBox


@dataclass(frozen=True, slots=True)
class BoxCrop:
    """One context-expanded crop with original-pixel mapping metadata."""

    box: ROIBox
    x1: int
    y1: int
    x2: int
    y2: int
    image: np.ndarray

    @property
    def local_box(self) -> tuple[int, int, int, int]:
        return (
            self.box.x1 - self.x1,
            self.box.y1 - self.y1,
            self.box.x2 - self.x1,
            self.box.y2 - self.y1,
        )


def open_rgb(source: Path | bytes) -> np.ndarray:
    image_source = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(image_source) as image:
        return np.asarray(image.convert("RGB"))


def output_dir(run_dir: Path, model_id: str) -> Path:
    """Return the run-local adapter workspace.

    A run is already bound to exactly one model, so another model-id directory
    adds no isolation. Keeping adapter intermediates at the run root also avoids
    exceeding the legacy 260-character Windows path limit for UUID-based jobs.
    """

    del model_id
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def apply_box_roi(mask: np.ndarray, boxes: Iterable[ROIBox]) -> np.ndarray:
    """Zero pixels outside active, clipped half-open boxes."""

    height, width = mask.shape
    allowed = np.zeros((height, width), dtype=bool)
    for box in boxes:
        if not box.active:
            continue
        x1 = min(max(box.x1, 0), width)
        x2 = min(max(box.x2, 0), width)
        y1 = min(max(box.y1, 0), height)
        y2 = min(max(box.y2, 0), height)
        if x1 < x2 and y1 < y2:
            allowed[y1:y2, x1:x2] = True
    return cast(np.ndarray, np.asarray(mask) * allowed)


def iter_box_crops(
    image: np.ndarray,
    boxes: Iterable[ROIBox],
    *,
    context_px: int,
) -> list[BoxCrop]:
    """Return active box crops expanded by a clipped, reproducible context margin."""

    if context_px < 0:
        raise ValueError("context_px cannot be negative")
    height, width = image.shape[:2]
    crops: list[BoxCrop] = []
    for box in boxes:
        if not box.active:
            continue
        x1 = max(0, box.x1 - context_px)
        y1 = max(0, box.y1 - context_px)
        x2 = min(width, box.x2 + context_px)
        y2 = min(height, box.y2 + context_px)
        if x1 >= x2 or y1 >= y2:
            continue
        crops.append(
            BoxCrop(
                box=box,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image=np.asarray(image[y1:y2, x1:x2]),
            )
        )
    return crops


def paste_box_probability(
    destination: np.ndarray,
    crop_probability: np.ndarray,
    crop: BoxCrop,
) -> None:
    """Max-fuse one crop probability into only the user's exact box."""

    if crop_probability.shape != crop.image.shape[:2]:
        raise ValueError("crop probability shape does not match its source crop")
    local_x1, local_y1, local_x2, local_y2 = crop.local_box
    source = crop_probability[local_y1:local_y2, local_x1:local_x2]
    target = destination[crop.box.y1 : crop.box.y2, crop.box.x1 : crop.box.x2]
    if source.shape != target.shape:
        raise ValueError("box mapping produced incompatible source and target shapes")
    np.maximum(target, source, out=target)


def embed_box_mask(mask: np.ndarray, crop: BoxCrop, shape: tuple[int, int]) -> np.ndarray:
    """Map a crop-local mask to original pixels and force all pixels outside the box to zero."""

    if mask.shape != crop.image.shape[:2]:
        raise ValueError("crop mask shape does not match its source crop")
    local_x1, local_y1, local_x2, local_y2 = crop.local_box
    clipped = np.asarray(mask[local_y1:local_y2, local_x1:local_x2], dtype=bool)
    result = np.zeros(shape, dtype=bool)
    target = result[crop.box.y1 : crop.box.y2, crop.box.x1 : crop.box.x2]
    if clipped.shape != target.shape:
        raise ValueError("box mapping produced incompatible source and target shapes")
    target[:] = clipped
    return result


def deduplicate_instance_masks(
    masks: list[np.ndarray],
    scores: list[float | None],
    *,
    iou_threshold: float = 0.7,
) -> tuple[list[np.ndarray], list[float | None]]:
    """Drop near-duplicate masks with content-based deterministic tie-breaking.

    Scientific adapters defer de-duplication to the frozen analysis postprocess
    profile. This helper remains for integrations that need it, but its result must
    never depend on backend output order.
    """

    if len(masks) != len(scores):
        raise ValueError("masks and scores must have the same length")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    ranked = sorted(
        zip(masks, scores, strict=True),
        key=lambda item: _instance_mask_order_key(item[0], item[1]),
    )
    kept: list[tuple[np.ndarray, float | None]] = []
    for raw_mask, score in ranked:
        mask = np.asarray(raw_mask, dtype=bool)
        if not mask.any():
            continue
        duplicate = False
        for existing, _existing_score in kept:
            intersection = int(np.logical_and(mask, existing).sum())
            if not intersection:
                continue
            union = int(np.logical_or(mask, existing).sum())
            if union and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((mask, score))
    kept.sort(key=lambda item: _instance_mask_spatial_key(item[0], item[1]))
    return [item[0] for item in kept], [item[1] for item in kept]


def _instance_mask_order_key(
    raw_mask: np.ndarray,
    score: float | None,
) -> tuple[object, ...]:
    mask = np.asarray(raw_mask, dtype=bool)
    return (
        -(score if score is not None else -1.0),
        *_instance_mask_spatial_key(mask, score),
    )


def _instance_mask_spatial_key(
    raw_mask: np.ndarray,
    score: float | None,
) -> tuple[object, ...]:
    mask = np.asarray(raw_mask, dtype=bool)
    bbox = mask_bbox(mask)
    digest = hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).digest()
    return (
        bbox[1],
        bbox[0],
        bbox[3],
        bbox[2],
        int(mask.sum()),
        score is None,
        -(score or 0.0),
        digest,
    )


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Apply deterministic 8-connected area filtering without optional SciPy."""

    binary = np.asarray(mask, dtype=bool)
    if min_area_px <= 1 or not binary.any():
        return binary
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    kept = np.zeros_like(binary, dtype=bool)
    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for y, x in zip(*np.nonzero(binary), strict=True):
        if visited[y, x]:
            continue
        visited[y, x] = True
        component = [(int(y), int(x))]
        cursor = 0
        while cursor < len(component):
            current_y, current_x = component[cursor]
            cursor += 1
            for dy, dx in neighbours:
                next_y, next_x = current_y + dy, current_x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and binary[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    component.append((next_y, next_x))
        if len(component) >= min_area_px:
            ys, xs = zip(*component, strict=True)
            kept[np.asarray(ys), np.asarray(xs)] = True
    return kept


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)
