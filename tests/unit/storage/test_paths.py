"""Tests for canonical storage layout and traversal defenses."""

import os
from pathlib import Path

import pytest

from app.storage import StoragePathError, StoragePaths


def test_builds_frozen_job_layout(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path / "outputs")

    assert paths.job_manifest("job_001") == tmp_path / "outputs/job_001/manifest.json"
    assert paths.job_config("job_001") == tmp_path / "outputs/job_001/job_config.json"
    assert paths.upload_file("job_001", "样品.TIF") == (
        tmp_path / "outputs/job_001/input/样品.TIF"
    )
    assert paths.upload_file("job_001", "样品.TIF", image_id="img_001") == (
        tmp_path / "outputs/job_001/input/img_001/original.tif"
    )
    assert paths.boxes_revision("job_001", "img_001", 3) == (
        tmp_path / "outputs/job_001/images/img_001/boxes_revision_003.json"
    )
    assert paths.run_artifact("job_001", "img_001", "run_001", "pred_mask.png") == (
        tmp_path
        / "outputs/job_001/images/img_001/runs/run_001/pred_mask.png"
    )
    assert paths.export_zip("job_001") == (
        tmp_path / "outputs/job_001/exports/nanoloop-export.zip"
    )


@pytest.mark.parametrize(
    "unsafe_id",
    ["", ".", "..", "../escape", "nested/job", r"nested\job", "/absolute", "job id"],
)
def test_rejects_unsafe_identifiers(tmp_path: Path, unsafe_id: str) -> None:
    paths = StoragePaths(tmp_path / "outputs")

    with pytest.raises(StoragePathError):
        paths.job_dir(unsafe_id)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "",
        ".",
        "..",
        "../escape.tif",
        "nested/file.tif",
        r"nested\file.tif",
        "/tmp/x",
        r"C:\x.tif",
        "bad\x00name",
    ],
)
def test_rejects_unsafe_upload_filenames(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    paths = StoragePaths(tmp_path / "outputs")

    with pytest.raises(StoragePathError):
        paths.upload_file("job_001", unsafe_name)


def test_repo_relative_constructor_cannot_escape_repo(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    paths = StoragePaths.from_repo_root(repository, "var/outputs")
    assert paths.root == repository / "var/outputs"

    with pytest.raises(StoragePathError):
        StoragePaths.from_repo_root(repository, "../outside")
    with pytest.raises(StoragePathError):
        StoragePaths.from_repo_root(repository, tmp_path / "absolute")


def test_require_managed_rejects_outside_and_symlink_escape(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path / "outputs")
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(StoragePathError):
        paths.require_managed(outside)

    link = paths.root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    with pytest.raises(StoragePathError):
        paths.require_managed(link / "payload.bin")


def test_rejects_invalid_revision_and_export_suffix(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path / "outputs")

    with pytest.raises(StoragePathError):
        paths.boxes_revision("job_001", "img_001", -1)
    with pytest.raises(StoragePathError):
        paths.export_zip("job_001", "export.tar")
