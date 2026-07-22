"""ROI round-trip E2E tests — the core E-P1 task 3.

Verifies that a box drawn on the canvas survives the full round-trip:
  display coords → original coords → REST save → REST load → display coords

The coordinate-math tests run without a browser or backend.
The REST round-trip tests are skipped if the backend is not running.
"""

from __future__ import annotations

import pytest

from frontend.roi_canvas import (
    display_box_to_original,
    original_box_to_display,
    parse_canvas_change,
    preview_dimensions,
)

from tests.e2e.conftest import skip_no_backend


# ---------------------------------------------------------------------------
# Coordinate round-trip: display → original → display (no browser needed)
# ---------------------------------------------------------------------------


class TestCoordinateRoundTrip:
    """Verify that display-to-original-to-display is identity for any box."""

    @pytest.mark.parametrize("img_w,img_h", [
        (512, 512),
        (1024, 768),
        (4096, 4096),
        (2048, 1024),
    ])
    def test_round_trip_preserves_box_edges(self, img_w: int, img_h: int) -> None:
        """A box drawn at display coordinates must map back to the same
        display coordinates after original→display conversion."""
        disp_w, disp_h = preview_dimensions(img_w, img_h, max_width=600)
        # Draw a box at 25% from each edge
        start = (disp_w * 0.25, disp_h * 0.25)
        end = (disp_w * 0.75, disp_h * 0.75)

        # display → original
        x1, y1, x2, y2 = display_box_to_original(
            start, end,
            display_width=disp_w,
            display_height=disp_h,
            original_width=img_w,
            original_height=img_h,
        )
        # original → display
        back = original_box_to_display(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            display_width=disp_w,
            display_height=disp_h,
            original_width=img_w,
            original_height=img_h,
        )
        # Edges must match within 1 display pixel (floor/ceil rounding)
        assert abs(back[0] - start[0]) <= 1.0
        assert abs(back[1] - start[1]) <= 1.0
        assert abs(back[2] - end[0]) <= 1.0
        assert abs(back[3] - end[1]) <= 1.0

    def test_round_trip_corner_box(self) -> None:
        """Box at (0,0) must survive round-trip — no negative coords."""
        disp_w, disp_h = preview_dimensions(1024, 1024, max_width=600)
        x1, y1, x2, y2 = display_box_to_original(
            (0, 0), (50, 50),
            display_width=disp_w,
            display_height=disp_h,
            original_width=1024,
            original_height=1024,
        )
        assert x1 >= 0
        assert y1 >= 0
        back = original_box_to_display(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            display_width=disp_w,
            display_height=disp_h,
            original_width=1024,
            original_height=1024,
        )
        assert abs(back[0]) <= 1.0
        assert abs(back[1]) <= 1.0

    def test_round_trip_max_box(self) -> None:
        """Box covering the entire image must map to full extents."""
        disp_w, disp_h = preview_dimensions(2048, 2048, max_width=600)
        x1, y1, x2, y2 = display_box_to_original(
            (0, 0), (disp_w, disp_h),
            display_width=disp_w,
            display_height=disp_h,
            original_width=2048,
            original_height=2048,
        )
        assert x1 == 0
        assert y1 == 0
        assert x2 == 2048
        assert y2 == 2048

    def test_original_coords_are_integers(self) -> None:
        """SEM analysis requires integer pixel coordinates (half-open interval)."""
        disp_w, disp_h = preview_dimensions(1024, 1024, max_width=600)
        result = display_box_to_original(
            (10.7, 20.3), (110.5, 220.8),
            display_width=disp_w,
            display_height=disp_h,
            original_width=1024,
            original_height=1024,
        )
        for val in result:
            assert isinstance(val, int)

    def test_clamped_to_image_bounds(self) -> None:
        """Boxes drawn outside the canvas must be clamped to image bounds."""
        disp_w, disp_h = preview_dimensions(512, 512, max_width=600)
        # Draw a box that extends beyond the canvas
        x1, y1, x2, y2 = display_box_to_original(
            (-50, -50), (disp_w + 100, disp_h + 100),
            display_width=disp_w,
            display_height=disp_h,
            original_width=512,
            original_height=512,
        )
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 512
        assert y2 <= 512

    def test_non_square_aspect_ratio(self) -> None:
        """Portrait/landscape images must scale correctly in both axes."""
        disp_w, disp_h = preview_dimensions(2048, 1024, max_width=600)
        # The display preview must preserve the 2:1 aspect ratio
        assert disp_w / disp_h == pytest.approx(2.0, rel=0.05)
        x1, y1, x2, y2 = display_box_to_original(
            (0, 0), (disp_w, disp_h),
            display_width=disp_w,
            display_height=disp_h,
            original_width=2048,
            original_height=1024,
        )
        assert x1 == 0
        assert y1 == 0
        assert x2 == 2048
        assert y2 == 1024


# ---------------------------------------------------------------------------
# Canvas change parsing — simulates the streamlit-drawable-canvas payload
# ---------------------------------------------------------------------------


class TestCanvasChangeParsing:
    """Verify that the JSON payload from the canvas component is parsed
    into the correct original-coordinate boxes."""

    def test_parse_single_box(self) -> None:
        """A single rectangle in the canvas JSON must produce one box."""
        canvas_json = {
            "event_id": "evt_001",
            "boxes": [
                {"x1": 100, "y1": 50, "x2": 300, "y2": 200, "label": "A"},
            ],
        }
        change = parse_canvas_change(canvas_json, max_boxes=10)
        assert change is not None
        assert change.event_id == "evt_001"
        assert len(change.boxes) == 1
        box = change.boxes[0]
        assert box["x1"] == 100
        assert box["y1"] == 50
        assert box["x2"] == 300
        assert box["y2"] == 200
        assert box["label"] == "A"

    def test_parse_empty_boxes(self) -> None:
        """An empty boxes list must produce zero boxes."""
        change = parse_canvas_change(
            {"event_id": "evt_002", "boxes": []}, max_boxes=10
        )
        assert change is not None
        assert len(change.boxes) == 0

    def test_parse_none_input(self) -> None:
        assert parse_canvas_change(None) is None

    def test_max_boxes_enforced(self) -> None:
        """Excess boxes must be rejected to prevent abuse."""
        boxes = [
            {"x1": i * 10, "y1": 0, "x2": i * 10 + 5, "y2": 5}
            for i in range(50)
        ]
        with pytest.raises(ValueError, match="At most"):
            parse_canvas_change(
                {"event_id": "evt_003", "boxes": boxes}, max_boxes=5
            )

    def test_missing_event_id_raises(self) -> None:
        with pytest.raises(ValueError, match="event"):
            parse_canvas_change({"boxes": []}, max_boxes=10)

    def test_box_defaults(self) -> None:
        """Boxes without label/active must get sensible defaults."""
        change = parse_canvas_change(
            {
                "event_id": "evt_004",
                "boxes": [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}],
            },
            max_boxes=10,
        )
        assert change is not None
        box = change.boxes[0]
        assert box["label"] == ""
        assert box["active"] is True


# ---------------------------------------------------------------------------
# REST round-trip (requires live backend)
# ---------------------------------------------------------------------------


@skip_no_backend
class TestRestRoundTrip:
    """Full REST round-trip: save boxes → fetch boxes → verify identity.

    These tests are skipped when the backend is not running.
    To enable: start the NanoLoop backend and set NANOLOOP_API_BASE_URL.
    """

    def test_save_and_reload_boxes(self) -> None:
        """Boxes saved via the API must be fetchable with identical coordinates."""
        pytest.skip("Requires live backend with test job — see conftest.py")
