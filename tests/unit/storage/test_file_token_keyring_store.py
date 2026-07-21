from __future__ import annotations

import base64
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from app.storage.file_token_keyring_store import (
    FILE_TOKEN_V2_KEYRING_SCHEMA_VERSION,
    FileTokenV2KeyRingStore,
    FileTokenV2KeyRingStoreError,
)
from app.storage.file_tokens_v2 import (
    FileTokenV2Error,
    FileTokenV2KeyRing,
    FileTokenV2Purpose,
)

_TENANT_ID = f"tnt_{'1' * 32}"
_PRINCIPAL_ID = f"prn_{'2' * 32}"
_JOB_ID = f"job_{'3' * 32}"
_ARTIFACT_ID = f"art_{'4' * 32}"
_SHA256 = "5" * 64
_FIRST_KEY = b"a" * 32
_SECOND_KEY = b"b" * 32


def _token(ring: FileTokenV2KeyRing, *, now: int = 100) -> str:
    claims = ring.create_claims(
        tenant_id=_TENANT_ID,
        principal_id=_PRINCIPAL_ID,
        job_id=_JOB_ID,
        artifact_id=_ARTIFACT_ID,
        purpose=FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT,
        sha256=_SHA256,
        ttl_seconds=30,
        now=now,
    )
    return ring.issue(claims)


def _canonical_document(*, keys: dict[str, bytes], active_kid: str) -> bytes:
    document = {
        "schema_version": FILE_TOKEN_V2_KEYRING_SCHEMA_VERSION,
        "active_kid": active_kid,
        "keys": {
            kid: base64.urlsafe_b64encode(key).rstrip(b"=").decode("ascii")
            for kid, key in keys.items()
        },
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")


def _write_protected(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    path.chmod(0o600)


def test_initialize_publishes_exact_canonical_0600_document_and_loads_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "file-token-v2-keyring.json"
    store = FileTokenV2KeyRingStore(path)

    initialized = store.initialize(active_kid="key-initial", key=_FIRST_KEY)

    assert initialized.active_kid == "key-initial"
    assert initialized.retained_kids == ("key-initial",)
    assert path.read_bytes() == _canonical_document(
        keys={"key-initial": _FIRST_KEY}, active_kid="key-initial"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = store.load()
    token = _token(initialized)
    assert loaded.verify(token, now=101).aid == _ARTIFACT_ID
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_load_never_generates_a_missing_keyring(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(FileTokenV2KeyRingStoreError) as error:
        FileTokenV2KeyRingStore(path).load()

    assert error.value.code == "missing"
    assert not path.exists()


def test_concurrent_initializers_all_load_the_single_complete_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "concurrent.json"
    worker_count = 12
    barrier = Barrier(worker_count)
    candidate_keys = [bytes([index + 1]) * 32 for index in range(worker_count)]

    def initialize(index: int) -> FileTokenV2KeyRing:
        barrier.wait()
        return FileTokenV2KeyRingStore(path).initialize(
            active_kid="initial", key=candidate_keys[index]
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        rings = list(executor.map(initialize, range(worker_count)))

    winner_token = _token(rings[0])
    assert all(ring.verify(winner_token, now=101).aid == _ARTIFACT_ID for ring in rings)
    document: dict[str, Any] = json.loads(path.read_bytes())
    persisted = base64.urlsafe_b64decode(f"{document['keys']['initial']}==")
    assert persisted in candidate_keys
    assert path.read_bytes() == _canonical_document(
        keys={"initial": persisted}, active_kid="initial"
    )
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_initialize_does_not_accept_a_different_active_id_for_existing_material(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keyring.json"
    store = FileTokenV2KeyRingStore(path)
    store.initialize(active_kid="first", key=_FIRST_KEY)

    with pytest.raises(FileTokenV2KeyRingStoreError) as error:
        store.initialize(active_kid="unexpected", key=_SECOND_KEY)

    assert error.value.code == "initialization_conflict"
    assert store.load().active_kid == "first"


def test_load_rejects_a_symlink_and_a_non_regular_entry(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    FileTokenV2KeyRingStore(target).initialize(key=_FIRST_KEY)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)

    with pytest.raises(FileTokenV2KeyRingStoreError) as symlink_error:
        FileTokenV2KeyRingStore(symlink).load()
    assert symlink_error.value.code == "unsafe_type"

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(FileTokenV2KeyRingStoreError) as directory_error:
        FileTokenV2KeyRingStore(directory).load()
    assert directory_error.value.code == "unsafe_type"


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644, 0o660])
def test_load_requires_exact_0600_permissions(tmp_path: Path, mode: int) -> None:
    path = tmp_path / f"mode-{mode:o}.json"
    _write_protected(path, _canonical_document(keys={"initial": _FIRST_KEY}, active_kid="initial"))
    path.chmod(mode)

    with pytest.raises(FileTokenV2KeyRingStoreError) as error:
        FileTokenV2KeyRingStore(path).load()

    assert error.value.code == "unsafe_permissions"


@pytest.mark.parametrize(
    "content",
    [
        b'{"active_kid":"initial","keys":{},"schema_version":1,"unknown":true}',
        b'{"active_kid":"initial","keys":{"initial":"YWFh","initial":"YmJi"},"schema_version":1}',
        b'{"active_kid":"initial","active_kid":"second","keys":{},"schema_version":1}',
        b'{ "active_kid":"initial","keys":{},"schema_version":1}',
    ],
)
def test_load_rejects_unknown_duplicate_and_noncanonical_json(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "invalid.json"
    _write_protected(path, content)

    with pytest.raises(FileTokenV2KeyRingStoreError) as error:
        FileTokenV2KeyRingStore(path).load()

    assert error.value.code in {"invalid_payload", "invalid_key"}


@pytest.mark.parametrize(
    "encoded_key",
    [
        "YQ",  # too short
        base64.urlsafe_b64encode(b"x" * 65).rstrip(b"=").decode("ascii"),
        base64.urlsafe_b64encode(_FIRST_KEY).decode("ascii"),  # padded
        "*" * 43,
    ],
)
def test_load_rejects_noncanonical_or_out_of_bounds_key_material(
    tmp_path: Path, encoded_key: str
) -> None:
    path = tmp_path / "bad-key.json"
    document = {
        "active_kid": "initial",
        "keys": {"initial": encoded_key},
        "schema_version": 1,
    }
    _write_protected(
        path, json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
    )

    with pytest.raises(FileTokenV2KeyRingStoreError) as error:
        FileTokenV2KeyRingStore(path).load()

    assert error.value.code == "invalid_key"


def test_rotate_retains_old_key_activates_new_key_and_is_canonical(tmp_path: Path) -> None:
    path = tmp_path / "rotate.json"
    store = FileTokenV2KeyRingStore(path)
    old_ring = store.initialize(active_kid="key-old", key=_FIRST_KEY)
    old_token = _token(old_ring)

    rotated = store.rotate(new_kid="key-new", key=_SECOND_KEY)

    assert rotated.active_kid == "key-new"
    assert set(rotated.retained_kids) == {"key-old", "key-new"}
    assert rotated.verify(old_token, now=101).aid == _ARTIFACT_ID
    new_token = _token(rotated)
    assert new_token.startswith("v2.key-new.")
    loaded = store.load()
    assert loaded.verify(old_token, now=101).aid == _ARTIFACT_ID
    assert loaded.verify(new_token, now=101).aid == _ARTIFACT_ID
    with pytest.raises(FileTokenV2Error):
        old_ring.verify(new_token, now=101)
    assert path.read_bytes() == _canonical_document(
        keys={"key-new": _SECOND_KEY, "key-old": _FIRST_KEY},
        active_kid="key-new",
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_rotation_requires_a_fresh_id_and_respects_the_retained_key_limit(
    tmp_path: Path,
) -> None:
    store = FileTokenV2KeyRingStore(tmp_path / "bounded.json", max_keys=2)
    store.initialize(active_kid="key-one", key=_FIRST_KEY)
    store.rotate(new_kid="key-two", key=_SECOND_KEY)

    with pytest.raises(FileTokenV2KeyRingStoreError) as duplicate:
        store.rotate(new_kid="key-two", key=b"c" * 32)
    assert duplicate.value.code == "duplicate_key_id"

    with pytest.raises(FileTokenV2KeyRingStoreError) as full:
        store.rotate(new_kid="key-three", key=b"c" * 32)
    assert full.value.code == "key_limit"


def test_load_rejects_truncated_and_oversized_files(tmp_path: Path) -> None:
    truncated_path = tmp_path / "truncated.json"
    _write_protected(truncated_path, b"")
    with pytest.raises(FileTokenV2KeyRingStoreError) as truncated:
        FileTokenV2KeyRingStore(truncated_path).load()
    assert truncated.value.code == "truncated"

    oversized_path = tmp_path / "oversized.json"
    _write_protected(oversized_path, b"x" * 129)
    with pytest.raises(FileTokenV2KeyRingStoreError) as oversized:
        FileTokenV2KeyRingStore(oversized_path, max_file_bytes=128).load()
    assert oversized.value.code == "oversized"


def test_repr_and_errors_do_not_disclose_path_or_key_material(tmp_path: Path) -> None:
    secret = b"super-secret-key-material-32-byte!"
    path = tmp_path / "private-keyring-name.json"
    store = FileTokenV2KeyRingStore(path)

    assert str(path) not in repr(store)
    ring = store.initialize(key=secret)
    assert secret.decode("ascii") not in repr(ring)
    path.chmod(0o644)
    with pytest.raises(FileTokenV2KeyRingStoreError) as error:
        store.load()
    rendered = f"{error.value!s} {error.value!r}"
    assert str(path) not in rendered
    assert secret.decode("ascii") not in rendered
