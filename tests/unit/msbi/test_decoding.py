from __future__ import annotations

import numpy as np
import pytest

from app.msbi.decoding import DecodeConfig, decode_instances


def test_center_boundary_watershed_splits_touching_particles_deterministically() -> None:
    yy, xx = np.mgrid[:64, :64]
    first = (yy - 32) ** 2 + (xx - 24) ** 2 <= 13**2
    second = (yy - 32) ** 2 + (xx - 40) ** 2 <= 13**2
    foreground = np.where(first | second, 0.95, 0.02).astype(np.float32)
    center = np.zeros((64, 64), dtype=np.float32)
    center[32, 24] = 0.99
    center[32, 40] = 0.99
    boundary = np.zeros((64, 64), dtype=np.float32)
    boundary[20:45, 32] = 0.99
    config = DecodeConfig(
        foreground_threshold=0.5,
        center_threshold=0.5,
        center_nms_radius=4,
        boundary_threshold=0.5,
        min_area_px=20,
    )

    first_run = decode_instances(foreground, center, boundary, config=config)
    second_run = decode_instances(foreground, center, boundary, config=config)

    assert first_run.marker_count == 2
    assert int(first_run.labels.max()) == 2
    assert np.array_equal(first_run.labels, second_run.labels)
    assert first_run.confidences == second_run.confidences


def test_decoder_respects_invalid_bottom_region() -> None:
    foreground = np.full((32, 32), 0.9, dtype=np.float32)
    center = np.zeros_like(foreground)
    center[8, 8] = 1.0
    center[28, 28] = 1.0
    boundary = np.zeros_like(foreground)
    valid = np.ones_like(foreground, dtype=bool)
    valid[-8:] = False

    decoded = decode_instances(
        foreground,
        center,
        boundary,
        valid_mask=valid,
        config=DecodeConfig(min_area_px=1),
    )

    assert np.all(decoded.labels[-8:] == 0)


def test_connected_component_mode_matches_binary_supervision_contract() -> None:
    foreground = np.zeros((32, 32), dtype=np.float32)
    foreground[2:8, 3:9] = 0.9
    foreground[20:29, 18:30] = 0.8
    center = np.zeros_like(foreground)
    boundary = np.ones_like(foreground)

    decoded = decode_instances(
        foreground,
        center,
        boundary,
        config=DecodeConfig(
            mode="connected_components",
            foreground_threshold=0.5,
            min_area_px=1,
        ),
    )

    assert decoded.marker_count == 2
    assert int(decoded.labels.max()) == 2
    assert decoded.confidences == pytest.approx((0.9, 0.8))
