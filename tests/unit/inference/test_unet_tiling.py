from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.inference.adapters.unet import UNetAdapter


def _adapter(config: dict[str, object]) -> Any:
    adapter = object.__new__(UNetAdapter)
    adapter.config = config
    return adapter


def test_sliding_window_covers_edges_and_fuses_overlaps_without_nan() -> None:
    adapter = _adapter(
        {
            "input_size": [4, 4],
            "patch_size": [4, 4],
            "stride": [2, 2],
        }
    )
    calls: list[tuple[int, int]] = []

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        calls.append(tile.shape[:2])
        return np.asarray(tile[..., 0], dtype=np.float32) / 255.0

    adapter._predict_tile_probability = predict_tile
    image = np.zeros((7, 6, 3), dtype=np.uint8)
    image[..., 0] = np.arange(42, dtype=np.uint8).reshape(7, 6)

    probability = adapter._predict_probability(image)

    assert probability.shape == (7, 6)
    assert np.isfinite(probability).all()
    assert probability == pytest.approx(image[..., 0] / 255.0, abs=1e-7)
    assert calls == [(4, 4)] * 6


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"patch_size": [0, 4]}, "positive integers"),
        ({"patch_size": [4, 4], "stride": [5, 2]}, "must not exceed"),
        ({"patch_size": [4]}, "positive integers"),
    ],
)
def test_invalid_tiling_configuration_fails_closed(
    config: dict[str, object],
    message: str,
) -> None:
    adapter = _adapter(config)

    with pytest.raises(ValueError, match=message):
        adapter._tiling_configuration()
