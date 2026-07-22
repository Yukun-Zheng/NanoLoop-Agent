from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from scripts.models.canonicalize_sem_tiff_inputs import (
    INCOMPLETE_MARKER,
    MANIFEST_NAME,
    canonicalize_directory,
)


def _jpeg_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (12, 8), color=(40, 80, 120)).save(
        stream, format="JPEG", quality=92
    )
    return stream.getvalue()


def test_normalizes_mislabeled_jpeg_and_real_tiff_without_pixel_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "normalized"
    repository = tmp_path / "repository"
    source.mkdir()
    repository.mkdir()
    (source / "YCu-1.tif").write_bytes(_jpeg_bytes())
    Image.new("L", (12, 8), color=27).save(source / "YCu-2.tif", format="TIFF")

    payload = canonicalize_directory(
        source,
        output,
        filenames=["YCu-1.tif", "YCu-2.tif"],
        repository=repository,
    )

    assert payload["status"] == "complete"
    assert [item["source_detected_format"] for item in payload["files"]] == [
        "JPEG",
        "TIFF",
    ]
    for filename in ("YCu-1.tif", "YCu-2.tif"):
        with Image.open(source / filename) as original, Image.open(output / filename) as normalized:
            original.load()
            normalized.load()
            assert normalized.format == "TIFF"
            assert normalized.mode == original.mode
            assert normalized.size == original.size
            assert normalized.tobytes() == original.tobytes()
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == payload
    assert str(tmp_path) not in json.dumps(manifest, ensure_ascii=False)
    assert not (output / INCOMPLETE_MARKER).exists()


def test_refuses_overwrite_and_repository_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.tif").write_bytes(_jpeg_bytes())
    existing = tmp_path / "existing"
    repository = tmp_path / "repository"
    unrelated = tmp_path / "unrelated"
    existing.mkdir()
    repository.mkdir()
    unrelated.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        canonicalize_directory(source, existing, repository=unrelated)
    with pytest.raises(ValueError, match="outside"):
        canonicalize_directory(
            source,
            repository / "normalized",
            repository=repository,
        )


def test_rejects_path_like_filename(tmp_path: Path) -> None:
    source = tmp_path / "source"
    repository = tmp_path / "repository"
    source.mkdir()
    repository.mkdir()

    with pytest.raises(ValueError, match="plain"):
        canonicalize_directory(
            source,
            tmp_path / "normalized",
            filenames=["../YCu-1.tif"],
            repository=repository,
        )


def test_rejects_symlinked_source_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_link = tmp_path / "source-link"
    repository = tmp_path / "repository"
    source.mkdir()
    repository.mkdir()
    source_link.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="source-dir must not be a symlink"):
        canonicalize_directory(
            source_link,
            tmp_path / "normalized",
            repository=repository,
        )
