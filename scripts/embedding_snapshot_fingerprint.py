"""Stable fingerprints for immutable embedding model files."""

from __future__ import annotations

import hashlib
from pathlib import Path

_HASH_CHUNK_SIZE = 1024 * 1024


def canonical_snapshot_files(root: Path) -> list[Path]:
    """Return model files while excluding Hugging Face's mutable download cache."""

    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def directory_tree_sha256(root: Path) -> str:
    """Hash canonical relative paths, sizes, and contents deterministically."""

    files = canonical_snapshot_files(root)
    if not files:
        raise ValueError("embedding snapshot contains no canonical model files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        observed_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                observed_size += len(chunk)
                digest.update(chunk)
        if observed_size != size:
            raise RuntimeError(f"embedding snapshot file changed while hashing: {path}")
    return digest.hexdigest()


def total_size(root: Path) -> int:
    """Return the byte size of the canonical model files."""

    return sum(path.stat().st_size for path in canonical_snapshot_files(root))
