from pathlib import Path

import numpy as np
from PIL import Image

from app.contracts.analyses import PixelRect
from app.contracts.enums import ModelVariant
from app.inference.profiling import profile_image_for_model


def _write(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8)).save(path)


def test_profile_image_detects_low_contrast_and_ignores_outside_valid_rect(
    tmp_path: Path,
) -> None:
    values = np.zeros((128, 128), dtype=np.uint8)
    values[:96] = 120
    values[96:] = np.tile([0, 255], (32, 64))
    path = tmp_path / "low-contrast.png"
    _write(path, values)

    profile = profile_image_for_model(
        path,
        PixelRect(x1=0, y1=0, x2=128, y2=96),
    )

    assert profile.variant == ModelVariant.LOW_CONTRAST


def test_profile_image_detects_dense_high_frequency_structure(tmp_path: Path) -> None:
    yy, xx = np.indices((128, 128))
    values = ((xx + yy) % 4 < 2).astype(np.uint8) * 255
    path = tmp_path / "dense.png"
    _write(path, values)

    profile = profile_image_for_model(
        path,
        PixelRect(x1=0, y1=0, x2=128, y2=128),
    )

    assert profile.variant == ModelVariant.DENSE_PARTICLE
    assert profile.edge_density >= 0.20
