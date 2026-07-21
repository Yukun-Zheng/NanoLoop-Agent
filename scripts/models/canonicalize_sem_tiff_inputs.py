"""Normalize mislabeled SEM JPEG/TIFF inputs into audited, real TIFF files.

Some acquisition exports carry JPEG bytes while retaining a ``.tif`` filename. NanoLoop keeps
upload validation strict so stored format, filename, MIME type, and custody evidence agree. This
tool preserves the source files, writes decoded pixels to a new external TIFF directory, and emits
a portable hash manifest. The normalized directory, never the mislabeled source, is used by later
calibration, smoke, and independent-test commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL import __version__ as PILLOW_VERSION

ALLOWED_SOURCE_FORMATS = frozenset({"JPEG", "TIFF"})
ALLOWED_MODES = frozenset({"1", "L", "I", "I;16", "I;16B", "I;16L", "F", "RGB", "RGBA"})
MANIFEST_NAME = "sem-tiff-normalization-manifest.json"
INCOMPLETE_MARKER = ".normalization-incomplete"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(image.width).encode("ascii"))
    digest.update(b"x")
    digest.update(str(image.height).encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _safe_filename(value: str) -> str:
    candidate = value.strip()
    if (
        not candidate
        or Path(candidate).name != candidate
        or any(ord(character) < 32 for character in candidate)
        or Path(candidate).suffix.casefold() not in {".tif", ".tiff"}
    ):
        raise ValueError(f"filename must be a plain .tif/.tiff basename: {value!r}")
    return candidate


def _validate_roots(
    source_dir: Path,
    output_dir: Path,
    *,
    repository: Path | None = None,
) -> tuple[Path, Path]:
    source_candidate = source_dir.expanduser()
    output_candidate = output_dir.expanduser()
    if source_candidate.is_symlink():
        raise ValueError("source-dir must not be a symlink")
    if output_candidate.parent.is_symlink():
        raise ValueError("output-dir parent must not be a symlink")
    source = source_candidate.resolve(strict=True)
    output = output_candidate.resolve(strict=False)
    repo = (repository or _repository_root()).resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source-dir must be a real directory")
    if output.exists():
        raise ValueError("output-dir already exists; refusing to overwrite")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output-dir parent must be an existing real directory")
    if output == repo or output.is_relative_to(repo):
        raise ValueError("output-dir must be outside the NanoLoop-Agent repository")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source-dir and output-dir must not contain one another")
    return source, output


def _source_filenames(source: Path, requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested:
        names = tuple(_safe_filename(item) for item in requested)
    else:
        names = tuple(
            path.name
            for path in sorted(source.iterdir(), key=lambda item: item.name.casefold())
            if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}
        )
    if not names:
        raise ValueError("no .tif/.tiff inputs were selected")
    if len(names) != len(set(names)):
        raise ValueError("input filenames must be unique")
    return names


def _decode_source(path: Path) -> tuple[Image.Image, str, str, str, int]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"input must be a real file: {path.name}")
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        with Image.open(BytesIO(source_bytes)) as source:
            detected_format = source.format or ""
            source.load()
            decoded = source.copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"input cannot be decoded: {path.name}") from error
    if detected_format not in ALLOWED_SOURCE_FORMATS:
        raise ValueError(
            f"input must contain JPEG or TIFF bytes, observed {detected_format!r}: {path.name}"
        )
    if decoded.mode not in ALLOWED_MODES:
        raise ValueError(
            f"input mode requires an explicit scientific conversion policy: "
            f"{decoded.mode} ({path.name})"
        )
    return (
        decoded,
        detected_format,
        _decoded_pixel_sha256(decoded),
        source_sha256,
        len(source_bytes),
    )


def _publish_tiff(image: Image.Image, destination: Path, *, expected_pixel_sha256: str) -> None:
    pending = destination.parent / f".{destination.name}.pending"
    if pending.exists() or destination.exists():
        raise ValueError(f"refusing to overwrite normalized output: {destination.name}")
    image.save(pending, format="TIFF", compression="tiff_deflate")
    try:
        with Image.open(pending) as normalized:
            normalized.load()
            if normalized.format != "TIFF":
                raise RuntimeError("normalized output is not TIFF")
            if _decoded_pixel_sha256(normalized) != expected_pixel_sha256:
                raise RuntimeError("normalized TIFF changed decoded pixels")
        os.link(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def canonicalize_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    filenames: Sequence[str] | None = None,
    repository: Path | None = None,
) -> dict[str, object]:
    """Create one no-overwrite external TIFF directory and its portable identity manifest."""

    source, output = _validate_roots(source_dir, output_dir, repository=repository)
    selected = _source_filenames(source, filenames)
    output.mkdir(mode=0o755)
    marker = output / INCOMPLETE_MARKER
    marker.write_text(
        "Normalization did not complete; do not use this directory as scientific input.\n",
        encoding="utf-8",
    )
    records: list[dict[str, object]] = []
    for filename in selected:
        source_path = source / filename
        image, detected_format, pixel_sha256, source_sha256, source_size = _decode_source(
            source_path
        )
        destination = output / filename
        _publish_tiff(image, destination, expected_pixel_sha256=pixel_sha256)
        records.append(
            {
                "filename": filename,
                "source_detected_format": detected_format,
                "source_sha256": source_sha256,
                "source_size_bytes": source_size,
                "decoded_mode": image.mode,
                "decoded_size": [image.width, image.height],
                "decoded_pixel_sha256": pixel_sha256,
                "normalized_format": "TIFF",
                "normalized_compression": "tiff_deflate",
                "normalized_sha256": _sha256(destination),
                "normalized_size_bytes": destination.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "1",
        "status": "complete",
        "pillow_version": PILLOW_VERSION,
        "source_policy": "JPEG-or-TIFF bytes with .tif/.tiff basename",
        "normalized_policy": "real lossless TIFF with identical decoded pixels",
        "files": records,
    }
    manifest = output / MANIFEST_NAME
    pending_manifest = output / f".{MANIFEST_NAME}.pending"
    pending_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(pending_manifest, manifest)
    finally:
        pending_manifest.unlink(missing_ok=True)
    marker.unlink()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help=(
            "Plain .tif/.tiff basename to normalize; repeat as needed. "
            "Defaults to all TIFF names."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        namespace = build_parser().parse_args(argv)
        payload = canonicalize_directory(
            namespace.source_dir,
            namespace.output_dir,
            filenames=namespace.filenames,
        )
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
