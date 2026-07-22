"""Unit tests for streaming, atomic local storage and secure downloads."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.core.errors import StorageError
from app.storage import (
    FileTokenError,
    LocalFileStore,
    StoragePathError,
    StoragePaths,
    UploadSizeExceededError,
)

_TOKEN_SECRET = b"test-only-file-token-secret-material"


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise


@pytest.fixture
def paths(tmp_path: Path) -> StoragePaths:
    return StoragePaths(tmp_path / "outputs")


@pytest.fixture
def store(paths: StoragePaths) -> LocalFileStore:
    return LocalFileStore(
        paths,
        max_upload_bytes=3 * 1024 * 1024,
        token_secret=_TOKEN_SECRET,
        default_token_ttl_seconds=60,
    )


class TrackingUpload(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.requested_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requested_sizes.append(size)
        return super().read(size)


def test_save_upload_streams_hashes_and_uses_canonical_path(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    content = b"x" * (1024 * 1024 + 17)
    upload = TrackingUpload(content)

    stored = store.save_upload(
        "job_001",
        upload,
        "Sample.TIF",
        image_id="img_001",
    )

    assert stored.path == paths.upload_file("job_001", "Sample.TIF", image_id="img_001")
    assert stored.relative_path == "job_001/input/img_001/original.tif"
    assert stored.size_bytes == len(content)
    assert stored.sha256 == hashlib.sha256(content).hexdigest()
    assert stored.path.read_bytes() == content
    assert upload.requested_sizes == [1024 * 1024, 1024 * 1024, 1024 * 1024]
    assert store.resolve_file_token(stored.file_token) == stored.path


def test_save_upload_enforces_limit_and_removes_temporary_file(paths: StoragePaths) -> None:
    store = LocalFileStore(
        paths,
        max_upload_bytes=5,
        token_secret=_TOKEN_SECRET,
    )

    with pytest.raises(UploadSizeExceededError) as error:
        store.save_upload("job_001", io.BytesIO(b"123456"), "too-large.tif")

    assert error.value.limit_bytes == 5
    assert not paths.upload_file("job_001", "too-large.tif").exists()
    assert not list(paths.input_dir("job_001").glob("*.tmp"))


def test_create_run_dir_uses_frozen_layout(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    directory = store.create_run_dir("job_001", "img_001", "run_001")

    assert directory == paths.run_dir("job_001", "img_001", "run_001")
    assert directory.is_dir()


def test_atomic_json_adds_schema_version_and_preserves_unicode(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    target = paths.job_config("job_001")

    store.atomic_write_json(target, {"材料": "锶镍", "threshold": 0.3})

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {"schema_version": "1.0", "材料": "锶镍", "threshold": 0.3}
    assert not list(target.parent.glob("*.tmp"))


def test_json_serialization_failure_leaves_existing_file_unchanged(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    target = paths.job_config("job_001")
    store.atomic_write_bytes(target, b"old")

    with pytest.raises(ValueError):
        store.atomic_write_json(target, {"not_finite": float("nan")})

    assert target.read_bytes() == b"old"


def test_atomic_write_and_hash_reject_unmanaged_paths(
    tmp_path: Path,
    store: LocalFileStore,
) -> None:
    outside = tmp_path / "outside.bin"

    with pytest.raises(StoragePathError):
        store.atomic_write_bytes(outside, b"secret")
    with pytest.raises(StoragePathError):
        store.calculate_sha256(outside)


def test_build_zip_hashes_exact_members_and_writes_matching_manifest(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    mask = paths.run_artifact("job_001", "img_001", "run_001", "pred_mask.png")
    summary = paths.run_artifact("job_001", "img_001", "run_001", "image_summary.json")
    store.atomic_write_bytes(mask, b"mask-bytes")
    store.atomic_write_json(summary, {"particle_count": 7})

    exported = store.build_zip("job_001", [summary, mask], filename="job_001.zip")

    assert exported.path == paths.export_zip("job_001", "job_001.zip")
    assert exported.sha256 == store.calculate_sha256(exported.path)
    assert store.resolve_file_token(exported.file_token) == exported.path
    with zipfile.ZipFile(exported.path) as archive:
        names = set(archive.namelist())
        assert names == {
            "images/img_001/runs/run_001/image_summary.json",
            "images/img_001/runs/run_001/pred_mask.png",
            "export_manifest.json",
        }
        manifest_bytes = archive.read("export_manifest.json")
        assert manifest_bytes == paths.export_manifest("job_001").read_bytes()
        manifest = json.loads(manifest_bytes)
        records = {record["path"]: record for record in manifest["files"]}
        assert records["images/img_001/runs/run_001/pred_mask.png"] == {
            "path": "images/img_001/runs/run_001/pred_mask.png",
            "sha256": hashlib.sha256(b"mask-bytes").hexdigest(),
            "size_bytes": len(b"mask-bytes"),
        }
        summary_bytes = summary.read_bytes()
        assert records["images/img_001/runs/run_001/image_summary.json"]["sha256"] == (
            hashlib.sha256(summary_bytes).hexdigest()
        )


def test_content_addressed_zip_reuses_exact_selection_and_preserves_old_token_bytes(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    summary = paths.run_artifact("job_001", "img_001", "run_001", "summary.json")
    store.atomic_write_json(summary, {"particle_count": 7})

    first = store.build_zip("job_001", [summary], filename=None)
    first_bytes = first.path.read_bytes()
    first_manifest = paths.content_addressed_export_manifest(
        "job_001", first.path.stem.removeprefix("nanoloop-export-")
    )
    with zipfile.ZipFile(first.path) as archive:
        manifest_bytes = archive.read("export_manifest.json")
        manifest = json.loads(manifest_bytes)

    assert first.path.name == f"nanoloop-export-{manifest['selection_sha256']}.zip"
    assert "generated_at" not in manifest
    assert first_manifest.read_bytes() == manifest_bytes

    repeated = store.build_zip("job_001", [summary], filename=None)
    assert repeated.path == first.path
    assert repeated.sha256 == first.sha256
    assert repeated.path.read_bytes() == first_bytes

    store.atomic_write_json(summary, {"particle_count": 8})
    changed = store.build_zip("job_001", [summary], filename=None)
    assert changed.path != first.path
    assert first.path.read_bytes() == first_bytes
    assert store.resolve_file_token(first.file_token) == first.path


def test_content_addressed_zip_publish_is_concurrent_and_never_replaces_mismatch(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    summary = paths.run_artifact("job_001", "img_001", "run_001", "summary.json")
    store.atomic_write_json(summary, {"particle_count": 7})
    barrier = threading.Barrier(8)

    def build() -> tuple[Path, str]:
        barrier.wait()
        exported = store.build_zip("job_001", [summary], filename=None)
        return exported.path, exported.sha256

    with ThreadPoolExecutor(max_workers=8) as executor:
        exports = list(executor.map(lambda _: build(), range(8)))

    assert len({path for path, _digest in exports}) == 1
    assert len({digest for _path, digest in exports}) == 1
    assert not list(paths.export_dir("job_001").glob("*.tmp"))

    immutable_path = exports[0][0]
    store.atomic_write_bytes(immutable_path, b"corrupt-existing-archive")
    with pytest.raises(StorageError, match="校验失败"):
        store.build_zip("job_001", [summary], filename=None)
    assert immutable_path.read_bytes() == b"corrupt-existing-archive"


def test_build_zip_accepts_job_relative_and_root_relative_whitelist_entries(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    first = paths.job_config("job_001")
    second = paths.job_manifest("job_001")
    store.atomic_write_json(first, {"kind": "config"})
    store.atomic_write_json(second, {"kind": "manifest"})

    exported = store.build_zip(
        "job_001",
        [Path("job_config.json"), Path("job_001/manifest.json")],
    )

    with zipfile.ZipFile(exported.path) as archive:
        assert {"job_config.json", "manifest.json"} < set(archive.namelist())


def test_build_zip_rejects_files_outside_job_and_generated_collisions(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    first_job_file = paths.job_config("job_001")
    other_job_file = paths.job_config("job_002")
    store.atomic_write_json(first_job_file, {})
    store.atomic_write_json(other_job_file, {})

    with pytest.raises(StoragePathError, match="outside the requested job"):
        store.build_zip("job_001", [other_job_file])
    with pytest.raises(StoragePathError, match="duplicate export member"):
        store.build_zip("job_001", [first_job_file, first_job_file])

    store.atomic_write_json(paths.export_manifest("job_001"), {"old": True})
    with pytest.raises(StoragePathError, match="collides"):
        store.build_zip("job_001", [paths.export_manifest("job_001")])


def test_build_zip_rejects_symlink_even_when_target_is_managed(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    target = paths.job_config("job_001")
    store.atomic_write_json(target, {})
    link = paths.job_dir("job_001") / "linked.json"
    _symlink_or_skip(link, target)

    with pytest.raises(StoragePathError, match="symbolic links"):
        store.build_zip("job_001", [link])


def test_file_tokens_detect_tampering_expiry_and_missing_files(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    target = paths.job_config("job_001")
    store.atomic_write_json(target, {})
    token = store.create_file_token(target, ttl_seconds=10, now=100)

    assert store.resolve_file_token(token, now=109) == target
    with pytest.raises(FileTokenError, match="expired"):
        store.resolve_file_token(token, now=110)

    token_parts = token.split(".")
    tampered = ".".join([token_parts[0], token_parts[1][:-1] + "A", token_parts[2]])
    with pytest.raises(FileTokenError, match="invalid"):
        store.resolve_file_token(tampered, now=101)

    target.unlink()
    with pytest.raises(FileTokenError, match="available"):
        store.resolve_file_token(token, now=101)


def test_file_token_cannot_be_redirected_through_a_symlink(
    paths: StoragePaths,
    store: LocalFileStore,
) -> None:
    first = paths.job_config("job_001")
    second = paths.job_manifest("job_001")
    store.atomic_write_json(first, {"value": "first"})
    store.atomic_write_json(second, {"value": "second"})
    token = store.create_file_token(first)

    first.unlink()
    _symlink_or_skip(first, second)

    with pytest.raises(FileTokenError, match="available"):
        store.resolve_file_token(token)


def test_malformed_unicode_token_is_a_domain_error(store: LocalFileStore) -> None:
    with pytest.raises(FileTokenError, match="invalid"):
        store.resolve_file_token("v1.\N{SNOWMAN}.signature")


def test_tokens_are_bound_to_store_secret(paths: StoragePaths) -> None:
    target = paths.job_config("job_001")
    first = LocalFileStore(paths, max_upload_bytes=10, token_secret=b"a" * 32)
    second = LocalFileStore(paths, max_upload_bytes=10, token_secret=b"b" * 32)
    first.atomic_write_json(target, {})

    token = first.create_file_token(target)

    with pytest.raises(FileTokenError, match="invalid"):
        second.resolve_file_token(token)


def test_constructor_requires_secure_limits(paths: StoragePaths) -> None:
    with pytest.raises(ValueError, match="max_upload_bytes"):
        LocalFileStore(paths, max_upload_bytes=0, token_secret=_TOKEN_SECRET)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        LocalFileStore(paths, max_upload_bytes=1, token_secret=b"short")
