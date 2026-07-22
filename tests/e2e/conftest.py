"""E2E test fixtures for ROI round-trip and cross-browser matrix tests.

These tests verify the full ROI round-trip pipeline:
  1. User draws a box on the canvas (display coordinates)
  2. Box is converted to original image coordinates
  3. Box is saved to the backend (REST API)
  4. Page reloads and the box is fetched back
  5. Box is converted back to display coordinates and re-rendered
  6. The re-rendered box matches the original draw position

The coordinate-conversion tests run without a browser or backend.
The full REST round-trip tests require a live backend (skipped if unavailable).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from frontend.roi_canvas import (
    display_box_to_original,
    original_box_to_display,
    parse_canvas_change,
    preview_dimensions,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic image dimensions simulating SEM images
# ---------------------------------------------------------------------------


@pytest.fixture
def small_image_dims() -> dict[str, int]:
    """Small image (512×512) — typical for quick previews."""
    return {"width": 512, "height": 512}


@pytest.fixture
def large_image_dims() -> dict[str, int]:
    """Large SEM image (4096×4096) — tests tiling/downscaling behaviour."""
    return {"width": 4096, "height": 4096}


@pytest.fixture
def portrait_image_dims() -> dict[str, int]:
    """Portrait image (2048×1024) — tests non-square aspect ratio."""
    return {"width": 2048, "height": 1024}


@pytest.fixture
def sample_boxes() -> list[dict[str, float]]:
    """A set of ROI boxes covering edge cases."""
    return [
        {"left": 0, "top": 0, "width": 100, "height": 100},  # top-left corner
        {"left": 400, "top": 300, "width": 50, "height": 50},  # center
        {"left": 0.5, "top": 0.5, "width": 1, "height": 1},  # sub-pixel
        {"left": 900, "top": 900, "width": 100, "height": 100},  # bottom-right
    ]


# ---------------------------------------------------------------------------
# Backend availability check
# ---------------------------------------------------------------------------


def _backend_available() -> bool:
    """Check if the NanoLoop backend is reachable for REST round-trip tests."""
    import urllib.request
    import urllib.error

    base = os.getenv("NANOLOOP_API_BASE_URL", "http://127.0.0.1:8000")
    try:
        req = urllib.request.Request(f"{base}/api/v1/health", method="GET")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


skip_no_backend = pytest.mark.skipif(
    not _backend_available(),
    reason="Backend not available — start NanoLoop backend to run REST round-trip tests",
)
