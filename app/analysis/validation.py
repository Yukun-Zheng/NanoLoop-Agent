"""Image and ROI validation independent of HTTP and persistence layers."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.analysis.sem_metadata import inspect_sem_image
from app.contracts.analyses import AnalysisROI
from app.core.errors import InvalidImageError

ALLOWED_IMAGE_FORMATS = frozenset({"TIFF", "PNG", "JPEG"})
_SUFFIX_ALLOWED_FORMATS = {
    ".tif": frozenset({"TIFF"}),
    ".tiff": frozenset({"TIFF"}),
    ".png": frozenset({"PNG"}),
    ".jpg": frozenset({"JPEG"}),
    ".jpeg": frozenset({"JPEG"}),
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

    allowed_formats = _SUFFIX_ALLOWED_FORMATS.get(path.suffix.casefold())
    if allowed_formats is None:
        raise InvalidImageError(
            details={
                "path": path.name,
                "reason": "unsupported_extension",
                "supported_extensions": sorted(_SUFFIX_ALLOWED_FORMATS),
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
                raise InvalidImageError(details={"path": path.name, "reason": "invalid_dimensions"})
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
    if detected_format not in allowed_formats:
        raise InvalidImageError(
            details={
                "path": path.name,
                "reason": "extension_content_mismatch",
                "allowed_formats": sorted(allowed_formats),
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
    """Conservatively exclude an optional light or dark SEM instrument footer."""

    return inspect_sem_image(
        image.path,
        width=image.width,
        height=image.height,
    ).analysis_roi
