"""Optional SEM footer detection and instrument metadata extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.contracts.analyses import (
    AnalysisROI,
    InvalidPixelRegion,
    PixelRect,
    SemInstrumentMetadata,
)

_ZEISS_PRIVATE_TAGS = (34118, 34119)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True, slots=True)
class SemImageInspection:
    analysis_roi: AnalysisROI
    metadata: SemInstrumentMetadata | None
    detected_scale_nm_per_pixel: float | None


@dataclass(frozen=True, slots=True)
class _Footer:
    boundary: int
    style: str


def inspect_sem_image(path: Path, *, width: int, height: int) -> SemImageInspection:
    """Inspect optional raster footer and vendor tags without guessing missing values."""

    footer = _instrument_footer(path, width=width, height=height)
    fields = _zeiss_fields(path)
    metadata = _metadata_from_zeiss(fields, footer=footer, width=width, height=height)
    if metadata is None and footer is not None:
        metadata = SemInstrumentMetadata(
            source="raster_footer",
            confidence="medium",
            footer_detected=True,
            footer_style=footer.style,
            footer_rect=PixelRect(
                x1=0,
                y1=footer.boundary,
                x2=width,
                y2=height,
            ),
            warnings=["instrument_footer_detected_metadata_unparsed"],
        )

    if footer is None:
        roi = AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=width, y2=height))
    else:
        roi = AnalysisROI(
            valid_rect=PixelRect(x1=0, y1=0, x2=width, y2=footer.boundary),
            invalid_rects=[
                InvalidPixelRegion(
                    x1=0,
                    y1=footer.boundary,
                    x2=width,
                    y2=height,
                    reason="instrument_bar_detected",
                )
            ],
            source="detected",
        )
    scale = metadata.pixel_size_nm if metadata is not None else None
    return SemImageInspection(
        analysis_roi=roi,
        metadata=metadata,
        detected_scale_nm_per_pixel=scale,
    )


def _metadata_from_zeiss(
    fields: dict[str, str],
    *,
    footer: _Footer | None,
    width: int,
    height: int,
) -> SemInstrumentMetadata | None:
    if not fields:
        return None
    pixel_size_nm = _length_to_nm(fields.get("AP_IMAGE_PIXEL_SIZE"))
    detector = _value(fields.get("DP_DETECTOR_CHANNEL")) or _value(
        fields.get("DP_DETECTOR_TYPE")
    )
    serial_value = _value(fields.get("SV_SERIAL_NUMBER"))
    instrument_model: str | None = None
    instrument_serial: str | None = None
    if serial_value:
        instrument_serial = serial_value
        instrument_model = serial_value.split("-", 1)[0].strip() or None
    acquired_at = _acquired_at(fields.get("AP_DATE"), fields.get("AP_TIME"))
    warnings: list[str] = []
    if pixel_size_nm is None:
        warnings.append("sem_pixel_size_missing")
    metadata = SemInstrumentMetadata(
        source=(
            "zeiss_tiff_private_tag+raster_footer"
            if footer is not None
            else "zeiss_tiff_private_tag"
        ),
        confidence="high",
        vendor="ZEISS",
        instrument_model=instrument_model,
        instrument_serial=instrument_serial,
        detector=detector,
        accelerating_voltage_kv=_number_in_unit(fields.get("AP_ACTUALKV"), "kv"),
        working_distance_mm=_number_in_unit(fields.get("AP_WD"), "mm"),
        magnification_x=_magnification(fields.get("AP_MAG")),
        aperture_size_um=_number_in_unit(fields.get("AP_APERTURESIZE"), "um"),
        pixel_size_nm=pixel_size_nm,
        acquired_at=acquired_at,
        footer_detected=footer is not None,
        footer_style=footer.style if footer is not None else None,
        footer_rect=(
            PixelRect(x1=0, y1=footer.boundary, x2=width, y2=height)
            if footer is not None
            else None
        ),
        warnings=warnings,
    )
    core_values = (
        metadata.pixel_size_nm,
        metadata.accelerating_voltage_kv,
        metadata.working_distance_mm,
        metadata.magnification_x,
        metadata.detector,
    )
    return metadata if sum(value is not None for value in core_values) >= 2 else None


def _zeiss_fields(path: Path) -> dict[str, str]:
    try:
        with Image.open(path) as image:
            tags = getattr(image, "tag_v2", {})
            candidates = [_tag_text(tags.get(tag)) for tag in _ZEISS_PRIVATE_TAGS]
    except (OSError, UnidentifiedImageError):
        return {}
    text = max(candidates, key=lambda item: item.count("AP_") + item.count("DP_"), default="")
    if "AP_" not in text or "DP_" not in text:
        return {}
    lines = [line.strip(" \x00") for line in text.replace("\r", "").split("\n")]
    fields: dict[str, str] = {}
    for index, line in enumerate(lines[:-1]):
        if re.fullmatch(r"(?:AP|DP|SV)_[A-Z0-9_]+", line):
            fields[line] = lines[index + 1].strip()
    return fields


def _tag_text(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        decoded: list[str] = []
        for encoding in ("utf-16le", "utf-8", "latin1"):
            try:
                decoded.append(raw.decode(encoding))
            except UnicodeDecodeError:
                continue
        return max(decoded, key=lambda item: item.count("AP_") + item.count("DP_"), default="")
    return str(raw).replace("\x00", "")


def _value(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.split("=", 1)[1].strip() if "=" in raw else raw.strip()
    return value or None


def _number(raw: str | None) -> float | None:
    value = _value(raw)
    if value is None:
        return None
    match = _NUMBER.search(value)
    if match is None:
        return None
    parsed = float(match.group())
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def _number_in_unit(raw: str | None, unit: str) -> float | None:
    value = _value(raw)
    if value is None:
        return None
    normalized = value.casefold().replace("µ", "u").replace("μ", "u")
    if unit not in normalized:
        return None
    return _number(value)


def _length_to_nm(raw: str | None) -> float | None:
    value = _value(raw)
    parsed = _number(value)
    if value is None or parsed is None:
        return None
    unit = value.casefold().replace("µ", "u").replace("μ", "u")
    if " pm" in unit:
        return parsed / 1000.0
    if " nm" in unit:
        return parsed
    if " um" in unit:
        return parsed * 1000.0
    return None


def _magnification(raw: str | None) -> float | None:
    value = _value(raw)
    parsed = _number(value)
    if value is None or parsed is None:
        return None
    normalized = value.casefold()
    if " k" in normalized:
        return parsed * 1000.0
    if " m" in normalized:
        return parsed * 1_000_000.0
    return parsed


def _acquired_at(date_raw: str | None, time_raw: str | None) -> str | None:
    date_value = _labeled_value(date_raw)
    time_value = _labeled_value(time_raw)
    if date_value is None:
        return None
    candidate = f"{date_value} {time_value}" if time_value else date_value
    for pattern in ("%d %b %Y %H:%M:%S", "%d %b %Y"):
        try:
            return datetime.strptime(candidate, pattern).isoformat()
        except ValueError:
            continue
    return date_value[:64]


def _labeled_value(raw: str | None) -> str | None:
    value = _value(raw)
    if value is None:
        return None
    if ":" in value:
        _label, candidate = value.split(":", 1)
        return candidate.strip() or None
    return value


def _instrument_footer(path: Path, *, width: int, height: int) -> _Footer | None:
    try:
        with Image.open(path) as source:
            gray = source.convert("L")
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
    candidates: list[tuple[float, int, str]] = []
    for y in range(first, last + 1):
        above = normalized[y - window : y]
        below = normalized[y : y + window]
        footer = normalized[y:]
        footer_dark = float(np.mean(footer <= 0.18))
        below_dark = float(np.mean(below <= 0.18))
        above_dark = float(np.mean(above <= 0.18))
        footer_bright = float(np.mean(footer >= 0.82))
        below_bright = float(np.mean(below >= 0.82))
        above_bright = float(np.mean(above >= 0.82))
        bright_text = float(np.mean(footer >= 0.72))
        dark_text = float(np.mean(footer <= 0.28))
        dark_contrast = float(np.median(above) - np.median(below))
        light_contrast = float(np.median(below) - np.median(above))
        if (
            footer_dark >= 0.78
            and below_dark >= 0.72
            and above_dark <= 0.55
            and dark_contrast >= 0.24
            and 0.002 <= bright_text <= 0.18
        ):
            score = dark_contrast + footer_dark + (below_dark - above_dark)
            candidates.append((score, y, "dark"))
        if (
            footer_bright >= 0.72
            and below_bright >= 0.70
            and above_bright <= 0.55
            and light_contrast >= 0.24
            and 0.004 <= dark_text <= 0.22
        ):
            score = light_contrast + footer_bright + (below_bright - above_bright)
            candidates.append((score, y, "light"))
    if not candidates:
        return None
    _score, sampled_boundary, style = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    boundary = round(sampled_boundary * height / sample_height)
    if style == "light":
        boundary = _light_footer_separator(path, boundary=boundary, height=height)
    minimum_footer = max(16, round(height * 0.08))
    if height - boundary < minimum_footer:
        return None
    return _Footer(boundary=boundary, style=style)


def _light_footer_separator(path: Path, *, boundary: int, height: int) -> int:
    """Move a light-footer boundary to the first nearby dark separator row."""

    try:
        with Image.open(path) as source:
            gray = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError):
        return boundary
    radius = max(4, round(height * 0.012))
    first = max(0, boundary - radius)
    last = min(height, boundary + radius + 1)
    dark_rows = [
        y
        for y in range(first, last)
        if float(np.mean(gray[y] <= 0.18)) >= 0.85
    ]
    return min(dark_rows) if dark_rows else boundary
