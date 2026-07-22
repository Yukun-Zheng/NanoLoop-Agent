"""Pure tests for ROI canvas transforms, payloads, and image preparation."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from frontend.roi_canvas import (
    display_box_to_original,
    original_box_to_display,
    parse_canvas_change,
    prepare_roi_preview,
    preview_dimensions,
)


def _png(width: int, height: int) -> bytes:
    output = io.BytesIO()
    Image.new("L", (width, height), color=128).save(output, format="PNG")
    return output.getvalue()


def test_preview_dimensions_preserve_aspect_ratio_and_never_upscale() -> None:
    assert preview_dimensions(2400, 1200, max_width=1200, max_height=900) == (1200, 600)
    assert preview_dimensions(600, 1200, max_width=1200, max_height=900) == (450, 900)
    assert preview_dimensions(320, 200) == (320, 200)
    with pytest.raises(ValueError, match="positive"):
        preview_dimensions(0, 200)


def test_prepare_preview_is_memory_only_png_and_checks_api_dimensions() -> None:
    preview = prepare_roi_preview(
        _png(200, 100),
        original_width=200,
        original_height=100,
        max_width=100,
        max_height=100,
    )

    assert (preview.display_width, preview.display_height) == (100, 50)
    assert (preview.original_width, preview.original_height) == (200, 100)
    assert preview.png_bytes.startswith(b"\x89PNG")
    assert preview.data_url.startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="do not match API metadata"):
        prepare_roi_preview(_png(20, 10), original_width=21, original_height=10)


def test_prepare_preview_preserves_visible_contrast_for_16_bit_sem_data() -> None:
    source = Image.fromarray(np.linspace(1000, 50000, 64, dtype=np.uint16).reshape(8, 8))
    encoded = io.BytesIO()
    source.save(encoded, format="TIFF")

    preview = prepare_roi_preview(
        encoded.getvalue(),
        original_width=8,
        original_height=8,
    )
    with Image.open(io.BytesIO(preview.png_bytes)) as rendered:
        minimum, maximum = rendered.convert("L").getextrema()

    assert minimum < 10
    assert maximum > 245


def test_display_drag_maps_to_covering_half_open_original_coordinates() -> None:
    mapped = display_box_to_original(
        (10.2, 5.1),
        (50.1, 25.2),
        display_width=100,
        display_height=50,
        original_width=1000,
        original_height=500,
    )
    reversed_and_clamped = display_box_to_original(
        (120.0, 60.0),
        (-2.0, -1.0),
        display_width=100,
        display_height=50,
        original_width=1000,
        original_height=500,
    )

    assert mapped == (102, 51, 501, 252)
    assert reversed_and_clamped == (0, 0, 1000, 500)


def test_original_box_round_trip_preserves_exact_edges_at_integer_scale() -> None:
    original = {"x1": 100, "y1": 50, "x2": 500, "y2": 250}
    display = original_box_to_display(
        original,
        display_width=100,
        display_height=50,
        original_width=1000,
        original_height=500,
    )

    assert display == (10.0, 5.0, 50.0, 25.0)
    assert display_box_to_original(
        display[:2],
        display[2:],
        display_width=100,
        display_height=50,
        original_width=1000,
        original_height=500,
    ) == (100, 50, 500, 250)


def test_canvas_change_is_strictly_shaped_and_preserves_box_identity() -> None:
    change = parse_canvas_change(
        {
            "event_id": "event-1",
            "boxes": [
                {
                    "box_id": "box_1",
                    "label": " field ",
                    "x1": 1,
                    "y1": 2,
                    "x2": 65,
                    "y2": 66,
                    "active": True,
                    "ignored": "not forwarded",
                }
            ],
        }
    )

    assert change is not None
    assert change.event_id == "event-1"
    assert change.boxes == (
        {
            "x1": 1,
            "y1": 2,
            "x2": 65,
            "y2": 66,
            "label": "field",
            "active": True,
            "box_id": "box_1",
        },
    )
    assert parse_canvas_change(None) is None
    with pytest.raises(ValueError, match="At most 1"):
        parse_canvas_change(
            {"event_id": "event-2", "boxes": [{}, {}]},
            max_boxes=1,
        )


def test_component_height_reporting_cannot_grow_from_its_own_viewport() -> None:
    source = (Path(__file__).parents[3] / "frontend" / "roi_component" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "document.body.scrollHeight" in source
    assert "height === lastPostedHeight" in source
    assert "document.documentElement.scrollHeight +" not in source
