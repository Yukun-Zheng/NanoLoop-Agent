"""Image and ROI validation independent of HTTP and persistence layers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.contracts.analyses import AnalysisROI, InvalidPixelRegion, PixelRect
from app.core.errors import InvalidImageError

ALLOWED_IMAGE_FORMATS = frozenset({"TIFF", "PNG", "JPEG"})
_SUFFIX_FORMATS = {
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
}
_MAX_IMAGE_DIMENSION = 50_000
_MAX_IMAGE_PIXELS = 80_000_000


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    path: Path
    format: str
    width: int
    height: int
    bit_depth: int
    mode: str


def _bit_depth(mode: str) -> int:
    if mode == "1":
        return 1
    if mode in {"L", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        return 8
    if mode.startswith("I;16"):
        return 16
    if mode in {"I", "F"}:
        return 32
    return 8


def validate_image(path: Path) -> ValidatedImage:
    """Sniff and decode an image rather than trusting the filename extension."""

    expected_format = _SUFFIX_FORMATS.get(path.suffix.casefold())
    if expected_format is None:
        raise InvalidImageError(
            details={
                "path": path.name,
                "reason": "unsupported_extension",
                "supported_extensions": sorted(_SUFFIX_FORMATS),
            }
        )
    try:
        with Image.open(path) as image:
            detected_format = image.format or ""
            width, height = image.size
            mode = image.mode
            if (
                width <= 0
                or height <= 0
                or width > _MAX_IMAGE_DIMENSION
                or height > _MAX_IMAGE_DIMENSION
                or width * height > _MAX_IMAGE_PIXELS
            ):
                raise InvalidImageError(
                    details={"path": path.name, "reason": "invalid_dimensions"}
                )
            image.verify()
    except (
        FileNotFoundError,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImageError(details={"path": path.name, "reason": "decode_failed"}) from exc

    if detected_format not in ALLOWED_IMAGE_FORMATS:
        raise InvalidImageError(
            details={"path": path.name, "format": detected_format, "reason": "unsupported_format"}
        )
    if detected_format != expected_format:
        raise InvalidImageError(
            details={
                "path": path.name,
                "reason": "extension_content_mismatch",
                "expected_format": expected_format,
                "detected_format": detected_format,
            }
        )
    bit_depth = _bit_depth(mode)
    if bit_depth not in {8, 16, 32}:
        raise InvalidImageError(
            details={
                "path": path.name,
                "reason": "unsupported_bit_depth",
                "bit_depth": bit_depth,
            }
        )
    return ValidatedImage(
        path=path,
        format=detected_format,
        width=width,
        height=height,
        bit_depth=bit_depth,
        mode=mode,
    )


def infer_analysis_roi(image: ValidatedImage) -> AnalysisROI:
    """Conservatively exclude a high-confidence dark SEM instrument footer.

    The detector intentionally returns the full image unless a broad, nearly black
    bottom band has both a strong horizontal boundary and sparse bright content
    consistent with instrument text. False negatives are safer than discarding
    scientific pixels.
    """

    boundary = _instrument_footer_boundary(image.path)
    if boundary is None or boundary <= 0 or boundary >= image.height:
        return AnalysisROI(
            valid_rect=PixelRect(x1=0, y1=0, x2=image.width, y2=image.height)
        )
    return AnalysisROI(
        valid_rect=PixelRect(x1=0, y1=0, x2=image.width, y2=boundary),
        invalid_rects=[
            InvalidPixelRegion(
                x1=0,
                y1=boundary,
                x2=image.width,
                y2=image.height,
                reason="instrument_bar_detected",
            )
        ],
        source="detected",
    )


def _instrument_footer_boundary(path: Path) -> int | None:
    try:
        with Image.open(path) as source:
            gray = source.convert("L")
            width, height = gray.size
            target_width = min(width, 1024)
            target_height = max(1, round(height * target_width / width))
            if target_height > 2048:
                target_height = 2048
                target_width = max(1, round(width * target_height / height))
            sampled = np.asarray(
                gray.resize((target_width, target_height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
    except (OSError, UnidentifiedImageError):
        return None

    low, high = np.percentile(sampled, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 20:
        return None
    normalized = np.clip((sampled - low) / (high - low), 0, 1)
    sample_height = normalized.shape[0]
    window = max(4, round(sample_height * 0.015))
    first = max(window, round(sample_height * 0.65))
    last = min(sample_height - window, round(sample_height * 0.92))
    candidates: list[tuple[float, int]] = []
    for y in range(first, last + 1):
        above = normalized[y - window : y]
        below = normalized[y : y + window]
        footer = normalized[y:]
        footer_dark = float(np.mean(footer <= 0.18))
        below_dark = float(np.mean(below <= 0.18))
        above_dark = float(np.mean(above <= 0.18))
        contrast = float(np.median(above) - np.median(below))
        bright_text = float(np.mean(footer >= 0.72))
        if (
            footer_dark >= 0.78
            and below_dark >= 0.72
            and above_dark <= 0.55
            and contrast >= 0.24
            and 0.002 <= bright_text <= 0.18
        ):
            score = contrast + footer_dark + (below_dark - above_dark)
            candidates.append((score, y))
    if not candidates:
        return None
    _score, sampled_boundary = max(candidates, key=lambda item: (item[0], -item[1]))
    boundary = round(sampled_boundary * height / sample_height)
    minimum_footer = max(16, round(height * 0.08))
    return boundary if height - boundary >= minimum_footer else None
