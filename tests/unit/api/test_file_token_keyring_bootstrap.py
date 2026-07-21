"""Application bootstrap guarantees for stable file-token v2 signing material."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from app.core.config import Settings
from app.main import _load_file_token_v2_keyring
from app.storage import (
    FileTokenV2KeyRingStore,
    FileTokenV2KeyRingStoreError,
    FileTokenV2Purpose,
)


def _settings(tmp_path: Path, **updates: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "test",
        "database_url": f"sqlite:///{tmp_path / 'nanoloop.db'}",
        "output_root": tmp_path / "outputs",
    }
    values.update(updates)
    return Settings.model_validate(values)


def test_test_bootstrap_initializes_once_and_reloads_stable_material(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    first = _load_file_token_v2_keyring(settings)
    claims = first.create_claims(
        tenant_id=f"tnt_{'a' * 32}",
        principal_id=f"prn_{'b' * 32}",
        job_id="job_1",
        artifact_id=f"art_{'c' * 32}",
        purpose=FileTokenV2Purpose.DOWNLOAD_ORIGINAL_IMAGE,
        sha256="d" * 64,
        ttl_seconds=900,
        now=100,
    )
    token = first.issue(claims)

    second = _load_file_token_v2_keyring(settings)
    assert second.verify(token, now=101) == claims
    keyring_path = tmp_path / ".file_token_v2_keyring.json"
    assert stat.S_IMODE(keyring_path.stat().st_mode) == 0o600


def test_production_refuses_missing_or_implicit_non_sqlite_keyring(tmp_path: Path) -> None:
    explicit = tmp_path / "data" / "keys.json"
    explicit.parent.mkdir()
    with pytest.raises(FileTokenV2KeyRingStoreError) as missing:
        _load_file_token_v2_keyring(
            _settings(
                tmp_path,
                app_env="production",
                file_token_v2_keyring_path=explicit,
            )
        )
    assert missing.value.code == "missing"

    with pytest.raises(FileTokenV2KeyRingStoreError) as path_required:
        _load_file_token_v2_keyring(
            _settings(
                tmp_path,
                app_env="production",
                database_url="postgresql://db.example/nanoloop",
            )
        )
    assert path_required.value.code == "path_required"


def test_production_loads_existing_protected_keyring_and_rejects_unsafe_mode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data" / "keys.json"
    path.parent.mkdir()
    initialized = FileTokenV2KeyRingStore(path, max_ttl_seconds=3_600).initialize(
        active_kid="initial",
        key=b"k" * 32,
    )
    settings = _settings(
        tmp_path,
        app_env="production",
        file_token_v2_keyring_path=path,
    )
    loaded = _load_file_token_v2_keyring(settings)
    assert loaded.active_kid == initialized.active_kid

    path.chmod(0o644)
    with pytest.raises(FileTokenV2KeyRingStoreError) as unsafe:
        _load_file_token_v2_keyring(settings)
    assert unsafe.value.code == "unsafe_permissions"
