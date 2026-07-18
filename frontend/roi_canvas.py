"""Offline, REST-fed ROI canvas helpers for the Streamlit workbench."""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class RoiCanvasPreview:
    """Browser-safe preview with an explicit original-to-display transform."""

    png_bytes: bytes
    display_width: int
    display_height: int
    original_width: int
    original_height: int

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True, slots=True)
class RoiCanvasChange:
    """A validated component change envelope, before domain box validation."""

    event_id: str
    boxes: tuple[dict[str, Any], ...]


def preview_dimensions(
    width: int,
    height: int,
    *,
    max_width: int = 1200,
    max_height: int = 900,
) -> tuple[int, int]:
    """Fit an image into the canvas budget without changing its aspect ratio."""

    if min(width, height, max_width, max_height) <= 0:
        raise ValueError("Image and preview dimensions must be positive")
    scale = min(1.0, max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_roi_preview(
    content: bytes,
    *,
    original_width: int,
    original_height: int,
    max_width: int = 1200,
    max_height: int = 900,
) -> RoiCanvasPreview:
    """Decode REST response bytes and produce an in-memory PNG canvas preview."""

    if not content:
        raise ValueError("Original image response is empty")
    expected_size = (original_width, original_height)
    display_size = preview_dimensions(
        original_width,
        original_height,
        max_width=max_width,
        max_height=max_height,
    )
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            if source.size != expected_size:
                raise ValueError(
                    "Original image dimensions do not match API metadata: "
                    f"bytes={source.width}x{source.height}, metadata="
                    f"{original_width}x{original_height}"
                )
            preview = _display_image(source)
            if preview.size != display_size:
                preview = preview.resize(display_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            preview.save(output, format="PNG", optimize=True)
    except ValueError:
        raise
    except (OSError, SyntaxError) as error:
        raise ValueError("Original image response cannot be decoded") from error
    return RoiCanvasPreview(
        png_bytes=output.getvalue(),
        display_width=display_size[0],
        display_height=display_size[1],
        original_width=original_width,
        original_height=original_height,
    )


def display_box_to_original(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    display_width: int,
    display_height: int,
    original_width: int,
    original_height: int,
) -> tuple[int, int, int, int]:
    """Map a dragged display rectangle to a covering half-open original box."""

    if min(display_width, display_height, original_width, original_height) <= 0:
        raise ValueError("Image and display dimensions must be positive")
    left = min(max(min(start[0], end[0]), 0.0), float(display_width))
    top = min(max(min(start[1], end[1]), 0.0), float(display_height))
    right = min(max(max(start[0], end[0]), 0.0), float(display_width))
    bottom = min(max(max(start[1], end[1]), 0.0), float(display_height))
    x1 = math.floor(left * original_width / display_width)
    y1 = math.floor(top * original_height / display_height)
    x2 = math.ceil(right * original_width / display_width)
    y2 = math.ceil(bottom * original_height / display_height)
    return (
        min(max(x1, 0), original_width),
        min(max(y1, 0), original_height),
        min(max(x2, 0), original_width),
        min(max(y2, 0), original_height),
    )


def original_box_to_display(
    box: Mapping[str, Any],
    *,
    display_width: int,
    display_height: int,
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    """Map an original half-open pixel box to display-space edges."""

    if min(display_width, display_height, original_width, original_height) <= 0:
        raise ValueError("Image and display dimensions must be positive")
    coordinates = tuple(_finite_number(box.get(name), name=name) for name in _COORDINATES)
    x1, y1, x2, y2 = coordinates
    return (
        x1 * display_width / original_width,
        y1 * display_height / original_height,
        x2 * display_width / original_width,
        y2 * display_height / original_height,
    )


def parse_canvas_change(value: object, *, max_boxes: int = 20) -> RoiCanvasChange | None:
    """Accept only the small JSON shape emitted by the bundled component."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("ROI canvas returned an invalid change payload")
    event_id = value.get("event_id")
    raw_boxes = value.get("boxes")
    if not isinstance(event_id, str) or not event_id or len(event_id) > 160:
        raise ValueError("ROI canvas returned an invalid event identifier")
    if not isinstance(raw_boxes, list):
        raise ValueError("ROI canvas returned an invalid box list")
    if len(raw_boxes) > max_boxes:
        raise ValueError(f"At most {max_boxes} ROI rows are allowed.")
    boxes: list[dict[str, Any]] = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, Mapping):
            raise ValueError("ROI canvas returned an invalid box record")
        box = {name: raw_box.get(name) for name in _COORDINATES}
        box["label"] = str(raw_box.get("label") or "").strip()
        box["active"] = bool(raw_box.get("active", True))
        box_id = raw_box.get("box_id")
        if isinstance(box_id, str) and box_id.strip():
            box["box_id"] = box_id.strip()
        boxes.append(box)
    return RoiCanvasChange(event_id=event_id, boxes=tuple(boxes))


def render_roi_canvas(
    *,
    preview: RoiCanvasPreview,
    boxes: Sequence[Mapping[str, Any]],
    valid_rect: Mapping[str, Any] | None,
    invalid_rects: Sequence[Mapping[str, Any]],
    read_only: bool,
    key: str,
    max_boxes: int = 20,
    minimum_size_px: int = 32,
) -> object:
    """Render the bundled no-network Streamlit component."""

    component = _roi_component()
    return component(
        image_data_url=preview.data_url,
        display_width=preview.display_width,
        display_height=preview.display_height,
        original_width=preview.original_width,
        original_height=preview.original_height,
        boxes=[dict(box) for box in boxes],
        valid_rect=dict(valid_rect) if valid_rect is not None else None,
        invalid_rects=[dict(rect) for rect in invalid_rects],
        read_only=read_only,
        max_boxes=max_boxes,
        minimum_size_px=minimum_size_px,
        default=None,
        key=key,
    )


@lru_cache(maxsize=1)
def _roi_component() -> Any:
    from streamlit.components import v1 as components

    component_path = Path(__file__).with_name("roi_component").resolve()
    return components.declare_component("nanoloop_roi_canvas", path=component_path)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ROI coordinate {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"ROI coordinate {name} must be finite")
    return number


def _display_image(source: Image.Image) -> Image.Image:
    """Create an 8-bit browser preview without flattening 16/32-bit SEM contrast."""

    if source.mode not in {"I", "I;16", "I;16B", "I;16L", "F"}:
        return source.convert("RGB")
    values = np.asarray(source, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite_values, (1.0, 99.0))
        if high <= low:
            scaled = np.full(values.shape, 128, dtype=np.uint8)
        else:
            normalized = np.nan_to_num((values - low) / (high - low), nan=0.0)
            scaled = np.asarray(np.clip(normalized, 0.0, 1.0) * 255.0, dtype=np.uint8)
    return Image.fromarray(scaled).convert("RGB")


_COORDINATES = ("x1", "y1", "x2", "y2")


__all__ = [
    "RoiCanvasChange",
    "RoiCanvasPreview",
    "display_box_to_original",
    "original_box_to_display",
    "parse_canvas_change",
    "prepare_roi_preview",
    "preview_dimensions",
    "render_roi_canvas",
]
