from __future__ import annotations

import hashlib
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from app.inference.snapshots import ModelArtifactSnapshotError, ModelArtifactSnapshotStore


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_copy_digest_mismatch_never_publishes_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"actual")
    expected = _digest(b"different")
    store = ModelArtifactSnapshotStore(tmp_path / "snapshots")

    with pytest.raises(ModelArtifactSnapshotError, match="weight sha256 mismatch"):
        store.publish(source, expected)

    digest_dir = tmp_path / "snapshots" / expected
    assert not (digest_dir / "weights.pt").exists()
    assert list(digest_dir.glob(".weights-*.tmp")) == []


def test_concurrent_publishers_share_one_complete_read_only_snapshot(tmp_path: Path) -> None:
    payload = os.urandom(2 * 1024 * 1024 + 17)
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    expected = _digest(payload)
    store = ModelArtifactSnapshotStore(tmp_path / "snapshots")
    barrier = Barrier(8)

    def publish() -> Path:
        barrier.wait()
        return store.publish(source, expected)

    with ThreadPoolExecutor(max_workers=8) as executor:
        published = list(executor.map(lambda _: publish(), range(8)))

    assert len(set(published)) == 1
    destination = published[0]
    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    assert list(destination.parent.glob(".weights-*.tmp")) == []


def test_existing_tampered_snapshot_fails_closed_without_overwrite(tmp_path: Path) -> None:
    payload = b"trusted model weights"
    source = tmp_path / "source.safetensors"
    source.write_bytes(payload)
    expected = _digest(payload)
    store = ModelArtifactSnapshotStore(tmp_path / "snapshots")
    destination = store.publish(source, expected)

    destination.chmod(0o644)
    destination.write_bytes(b"tampered snapshot")
    destination.chmod(0o444)

    with pytest.raises(ModelArtifactSnapshotError, match="snapshot sha256 mismatch"):
        store.publish(source, expected)

    assert destination.read_bytes() == b"tampered snapshot"
    assert list(destination.parent.glob(".weights-*.tmp")) == []


def test_existing_snapshot_symlink_is_never_followed_or_replaced(tmp_path: Path) -> None:
    payload = b"trusted model weights"
    source = tmp_path / "source.pt"
    source.write_bytes(payload)
    expected = _digest(payload)
    digest_dir = tmp_path / "snapshots" / expected
    digest_dir.mkdir(parents=True)
    target = tmp_path / "external.pt"
    target.write_bytes(payload)
    destination = digest_dir / "weights.pt"
    try:
        destination.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    store = ModelArtifactSnapshotStore(tmp_path / "snapshots")

    with pytest.raises(ModelArtifactSnapshotError, match="cannot be opened safely"):
        store.publish(source, expected)

    assert destination.is_symlink()
    assert target.read_bytes() == payload
