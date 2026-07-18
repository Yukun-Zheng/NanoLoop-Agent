"""Tests for race-resistant descriptor-pinned managed file reads."""

from __future__ import annotations

import errno
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import app.storage.pinned_file as pinned_file_module
from app.storage import (
    PinnedFileChangedError,
    PinnedFileIntegrityError,
    StoragePathError,
    open_pinned_managed_file,
)


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    root.mkdir()
    return root


def _write(root: Path, relative_path: str, content: bytes) -> Path:
    target = root.joinpath(*relative_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _assert_descriptor_closed(file_descriptor: int) -> None:
    with pytest.raises(OSError) as error:
        os.fstat(file_descriptor)
    assert error.value.errno == errno.EBADF


def test_open_hashes_rewinds_and_streams_zero_length_file(output_root: Path) -> None:
    _write(output_root, "job/input/empty.bin", b"")
    expected_sha256 = hashlib.sha256(b"").hexdigest()

    pinned = open_pinned_managed_file(
        output_root,
        "job/input/empty.bin",
        expected_size_bytes=0,
        expected_sha256=expected_sha256,
    )
    file_descriptor = pinned.fileno()

    assert pinned.filename == "empty.bin"
    assert pinned.relative_path == "job/input/empty.bin"
    assert pinned.size_bytes == 0
    assert pinned.sha256 == expected_sha256
    assert list(pinned.iter_chunks()) == []
    assert pinned.closed
    _assert_descriptor_closed(file_descriptor)


def test_walk_uses_required_flags_and_stream_never_reopens_path(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"pinned-content"
    _write(output_root, "job/nested/artifact.bin", content)
    calls: list[tuple[object, int, int | None]] = []
    original_open = pinned_file_module.os.open

    def tracked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        calls.append((path, flags, dir_fd))
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(pinned_file_module.os, "open", tracked_open)

    pinned = open_pinned_managed_file(output_root, "job/nested/artifact.bin")
    calls_after_open = list(calls)
    assert b"".join(pinned.iter_chunks()) == content

    assert calls == calls_after_open
    assert len(calls) == 4
    for _path, flags, _dir_fd in calls[:-1]:
        assert flags & os.O_DIRECTORY
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_CLOEXEC
        assert flags & os.O_ACCMODE == os.O_RDONLY
    final_flags = calls[-1][1]
    assert not final_flags & os.O_DIRECTORY
    assert final_flags & os.O_NOFOLLOW
    assert final_flags & os.O_CLOEXEC
    assert final_flags & os.O_ACCMODE == os.O_RDONLY


def test_large_file_hash_and_stream_use_bounded_chunks(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = (bytes(range(251)) * 25_067) + b"tail"
    _write(output_root, "job/large.bin", content)
    requested_sizes: list[int] = []
    original_read = pinned_file_module.os.read

    def tracked_read(file_descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        return original_read(file_descriptor, size)

    monkeypatch.setattr(pinned_file_module.os, "read", tracked_read)
    pinned = open_pinned_managed_file(output_root, "job/large.bin", chunk_size=64 * 1024)

    assert b"".join(pinned.iter_chunks()) == content
    assert pinned.sha256 == hashlib.sha256(content).hexdigest()
    assert pinned.size_bytes == len(content)
    assert requested_sizes
    assert max(requested_sizes) == 64 * 1024


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/absolute.bin",
        ".",
        "..",
        "job/./file.bin",
        "job/../file.bin",
        "job//file.bin",
        "job/file.bin/",
        "job\\file.bin",
        "job/control\n.bin",
        "job/delete\x7f.bin",
    ],
)
def test_path_must_be_strict_canonical_relative_posix(
    output_root: Path,
    relative_path: str,
) -> None:
    with pytest.raises(StoragePathError):
        open_pinned_managed_file(output_root, relative_path)


def test_path_length_and_depth_are_bounded(output_root: Path) -> None:
    too_long_component = "x" * 256
    too_deep = "/".join("x" for _ in range(129))

    with pytest.raises(StoragePathError, match="component is too long"):
        open_pinned_managed_file(output_root, f"job/{too_long_component}")
    with pytest.raises(StoragePathError, match="too many components"):
        open_pinned_managed_file(output_root, too_deep)


def test_rejects_final_and_parent_symlinks(output_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.bin"
    outside_file.write_bytes(b"secret")
    job_dir = output_root / "job"
    job_dir.mkdir()
    (job_dir / "file-link.bin").symlink_to(outside_file)
    (output_root / "parent-link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StoragePathError):
        open_pinned_managed_file(output_root, "job/file-link.bin")
    with pytest.raises(StoragePathError):
        open_pinned_managed_file(output_root, "parent-link/secret.bin")


def test_rejects_non_regular_final_target(output_root: Path) -> None:
    (output_root / "job" / "directory.bin").mkdir(parents=True)

    with pytest.raises(StoragePathError, match="regular file"):
        open_pinned_managed_file(output_root, "job/directory.bin")


def test_rejects_fifo_without_blocking(output_root: Path) -> None:
    job_dir = output_root / "job"
    job_dir.mkdir()
    fifo = job_dir / "pipe.bin"
    os.mkfifo(fifo)

    with pytest.raises(StoragePathError, match="regular file"):
        open_pinned_managed_file(output_root, "job/pipe.bin")


def test_rejects_symlink_output_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-outputs"
    real_root.mkdir()
    _write(real_root, "job/file.bin", b"secret")
    linked_root = tmp_path / "linked-outputs"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(StoragePathError):
        open_pinned_managed_file(linked_root, "job/file.bin")


def test_open_descriptor_survives_file_and_parent_path_replacement(output_root: Path) -> None:
    original = b"the-original-inode"
    target = _write(output_root, "job/nested/artifact.bin", original)
    pinned = open_pinned_managed_file(output_root, "job/nested/artifact.bin", chunk_size=4)

    old_parent = output_root / "job/nested-old"
    target.parent.rename(old_parent)
    _write(output_root, "job/nested/artifact.bin", b"replacement-path-bytes")

    assert b"".join(pinned.iter_chunks()) == original
    assert (output_root / "job/nested/artifact.bin").read_bytes() == b"replacement-path-bytes"


def test_parent_swap_during_walk_uses_already_open_directory(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"opened-parent-inode"
    _write(output_root, "job/nested/artifact.bin", original)
    original_open = pinned_file_module.os.open
    swapped = False

    def swap_after_job_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        file_descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "job" and not swapped:
            swapped = True
            (output_root / "job").rename(output_root / "job-old")
            _write(output_root, "job/nested/artifact.bin", b"replacement-tree")
        return file_descriptor

    monkeypatch.setattr(pinned_file_module.os, "open", swap_after_job_open)

    pinned = open_pinned_managed_file(output_root, "job/nested/artifact.bin")

    assert b"".join(pinned.iter_chunks()) == original
    assert (output_root / "job/nested/artifact.bin").read_bytes() == b"replacement-tree"


def test_detects_same_inode_mutation_during_hash(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _write(output_root, "job/artifact.bin", b"a" * (2 * 1024 * 1024))
    original_read = pinned_file_module.os.read
    mutated = False

    def mutate_after_first_read(file_descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(file_descriptor, size)
        if chunk and not mutated:
            mutated = True
            target.write_bytes(b"b" * (2 * 1024 * 1024))
        return chunk

    monkeypatch.setattr(pinned_file_module.os, "read", mutate_after_first_read)

    with pytest.raises(PinnedFileChangedError, match="完整性校验期间"):
        open_pinned_managed_file(output_root, "job/artifact.bin")


@pytest.mark.parametrize(
    ("expected_size_bytes", "expected_sha256", "detail_key"),
    [
        (4, None, "expected_size_bytes"),
        (None, "0" * 64, "expected_sha256"),
    ],
)
def test_integrity_mismatch_closes_every_opened_descriptor(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
    detail_key: str,
) -> None:
    _write(output_root, "job/nested/artifact.bin", b"trusted-content")
    opened_descriptors: list[int] = []
    original_open = pinned_file_module.os.open

    def tracked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        file_descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(file_descriptor)
        return file_descriptor

    monkeypatch.setattr(pinned_file_module.os, "open", tracked_open)

    with pytest.raises(PinnedFileIntegrityError) as error:
        open_pinned_managed_file(
            output_root,
            "job/nested/artifact.bin",
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )

    assert detail_key in error.value.details
    assert opened_descriptors
    for file_descriptor in set(opened_descriptors):
        _assert_descriptor_closed(file_descriptor)


def test_successful_exhaustion_and_double_close_release_fd_once(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(output_root, "artifact.bin", b"content")
    pinned = open_pinned_managed_file(output_root, "artifact.bin")
    file_descriptor = pinned.fileno()
    original_close = pinned_file_module.os.close
    closes = 0

    def tracked_close(candidate: int) -> None:
        nonlocal closes
        if candidate == file_descriptor:
            closes += 1
        original_close(candidate)

    monkeypatch.setattr(pinned_file_module.os, "close", tracked_close)

    assert b"".join(pinned.iter_chunks()) == b"content"
    pinned.close()
    pinned.close()
    assert closes == 1
    _assert_descriptor_closed(file_descriptor)


def test_iterator_close_before_iteration_releases_descriptor(output_root: Path) -> None:
    _write(output_root, "artifact.bin", b"content")
    pinned = open_pinned_managed_file(output_root, "artifact.bin")
    file_descriptor = pinned.fileno()
    chunks = pinned.iter_chunks()

    chunks.close()
    chunks.close()

    assert pinned.closed
    _assert_descriptor_closed(file_descriptor)
    with pytest.raises(ValueError, match="closed"):
        pinned.iter_chunks()


def test_read_error_closes_descriptor(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(output_root, "artifact.bin", b"content")
    pinned = open_pinned_managed_file(output_root, "artifact.bin")
    file_descriptor = pinned.fileno()

    def fail_read(_file_descriptor: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "simulated read failure")

    monkeypatch.setattr(pinned_file_module.os, "read", fail_read)
    chunks = pinned.iter_chunks()

    with pytest.raises(OSError) as error:
        next(chunks)
    assert error.value.errno == errno.EIO
    assert pinned.closed
    _assert_descriptor_closed(file_descriptor)


def test_stream_can_only_be_claimed_once(output_root: Path) -> None:
    _write(output_root, "artifact.bin", b"content")
    pinned = open_pinned_managed_file(output_root, "artifact.bin")
    chunks = pinned.iter_chunks()

    with pytest.raises(RuntimeError, match="already been claimed"):
        pinned.iter_chunks()

    chunks.close()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("expected_size_bytes", -1),
        ("expected_size_bytes", True),
        ("expected_sha256", "A" * 64),
        ("expected_sha256", "not-a-digest"),
        ("chunk_size", 0),
        ("chunk_size", 16 * 1024 * 1024 + 1),
    ],
)
def test_invalid_integrity_and_chunk_arguments_fail_before_file_io(
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: object,
) -> None:
    called = False
    original_open = pinned_file_module.os.open

    def tracked_open(*args: object, **kwargs: object) -> int:
        nonlocal called
        called = True
        return original_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pinned_file_module.os, "open", tracked_open)
    arguments: dict[str, object] = {keyword: value}

    with pytest.raises(ValueError):
        open_pinned_managed_file(output_root, "missing.bin", **arguments)  # type: ignore[arg-type]

    assert not called


def test_context_manager_closes_after_consumer_error(output_root: Path) -> None:
    _write(output_root, "artifact.bin", b"content")
    pinned = open_pinned_managed_file(output_root, "artifact.bin", chunk_size=2)
    file_descriptor = pinned.fileno()

    def consume_and_fail(consumer: Callable[[], bytes]) -> None:
        assert consumer() == b"co"
        raise RuntimeError("consumer cancelled")

    with (
        pytest.raises(RuntimeError, match="consumer cancelled"),
        pinned.iter_chunks() as chunks,
    ):
        consume_and_fail(lambda: next(chunks))

    assert pinned.closed
    _assert_descriptor_closed(file_descriptor)
