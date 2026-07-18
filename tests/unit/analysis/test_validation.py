from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.analysis.validation import infer_analysis_roi, validate_image
from app.core.errors import InvalidImageError


def test_validate_image_sniffs_content_and_reports_depth(tmp_path: Path) -> None:
    path = tmp_path / "image.tif"
    Image.new("I;16", (20, 10), color=12).save(path)
    result = validate_image(path)
    assert result.format == "TIFF"
    assert result.width == 20
    assert result.height == 10
    assert result.bit_depth == 16


def test_validate_image_rejects_non_image_with_image_suffix(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        validate_image(path)


def test_validate_image_rejects_extension_content_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "mislabelled.tif"
    Image.new("L", (20, 10), color=12).save(path, format="PNG")

    with pytest.raises(InvalidImageError) as exc_info:
        validate_image(path)

    assert exc_info.value.details["reason"] == "extension_content_mismatch"


def test_validate_image_rejects_unsupported_extension_even_for_valid_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.bmp"
    Image.new("L", (20, 10), color=12).save(path)

    with pytest.raises(InvalidImageError) as exc_info:
        validate_image(path)

    assert exc_info.value.details["reason"] == "unsupported_extension"


def test_validate_image_rejects_oversized_dimensions_before_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.png"
    path.write_bytes(b"header-only-test-double")

    class OversizedImage:
        format = "PNG"
        size = (50_001, 2)
        mode = "L"
        verify_called = False

        def __enter__(self) -> "OversizedImage":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def verify(self) -> None:
            self.verify_called = True

    oversized = OversizedImage()
    monkeypatch.setattr(Image, "open", lambda _path: oversized)

    with pytest.raises(InvalidImageError) as exc_info:
        validate_image(path)

    assert exc_info.value.details["reason"] == "invalid_dimensions"
    assert oversized.verify_called is False


def test_infer_analysis_roi_detects_only_high_confidence_dark_instrument_footer(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    pixels = rng.integers(55, 225, size=(200, 300), dtype=np.uint8)
    pixels[160:, :] = 0
    pixels[170:174, 18:92] = 255
    pixels[182:186, 120:260] = 255
    path = tmp_path / "sem-with-footer.png"
    Image.fromarray(pixels, mode="L").save(path)

    roi = infer_analysis_roi(validate_image(path))

    assert roi.source == "detected"
    assert 158 <= roi.valid_rect.y2 <= 162
    assert len(roi.invalid_rects) == 1
    assert roi.invalid_rects[0].y1 == roi.valid_rect.y2
    assert roi.invalid_rects[0].reason == "instrument_bar_detected"


def test_infer_analysis_roi_keeps_full_frame_without_confident_footer(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    pixels = rng.integers(20, 235, size=(160, 240), dtype=np.uint8)
    path = tmp_path / "sem-full.png"
    Image.fromarray(pixels, mode="L").save(path)

    roi = infer_analysis_roi(validate_image(path))

    assert roi.source == "none"
    assert roi.valid_rect.model_dump() == {"x1": 0, "y1": 0, "x2": 240, "y2": 160}
    assert roi.invalid_rects == []
