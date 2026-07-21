from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import scripts.manage_file_token_keyring as cli
from app.storage.file_token_keyring_store import FileTokenV2KeyRingStore
from app.storage.file_tokens_v2 import FileTokenV2Purpose


def _initialize(path: Path, *, kid: str = "key-old") -> None:
    FileTokenV2KeyRingStore(path).initialize(active_kid=kid, key=b"k" * 32)


def test_status_outputs_only_non_secret_metadata_for_explicit_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "private-keyring.json"
    _initialize(path)
    raw_document = path.read_text(encoding="ascii")
    encoded_key = json.loads(raw_document)["keys"]["key-old"]

    assert cli.main(["status", "--path", str(path)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "active_kid": "key-old",
        "retained_kids": ["key-old"],
        "count": 1,
    }
    assert str(path) not in captured.out
    assert encoded_key not in captured.out
    assert raw_document not in captured.out


@pytest.mark.parametrize(
    "environment_name",
    ["FILE_TOKEN_V2_KEYRING_PATH", "NANOLOOP_FILE_TOKEN_V2_KEYRING_PATH"],
)
def test_status_uses_environment_then_local_default_without_printing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment_name: str,
) -> None:
    environment_path = tmp_path / "environment-keyring.json"
    _initialize(environment_path, kid="environment")
    monkeypatch.delenv("FILE_TOKEN_V2_KEYRING_PATH", raising=False)
    monkeypatch.delenv("NANOLOOP_FILE_TOKEN_V2_KEYRING_PATH", raising=False)
    monkeypatch.setenv(environment_name, str(environment_path))

    assert cli.main(["status"]) == 0
    environment_output = capsys.readouterr().out
    assert json.loads(environment_output)["active_kid"] == "environment"
    assert str(environment_path) not in environment_output

    monkeypatch.delenv(environment_name)
    monkeypatch.chdir(tmp_path)
    default_path = tmp_path / "data" / ".file_token_v2_keyring.json"
    default_path.parent.mkdir()
    _initialize(default_path, kid="local-default")

    assert cli.main(["status"]) == 0
    default_output = capsys.readouterr().out
    assert json.loads(default_output)["active_kid"] == "local-default"
    assert str(default_path) not in default_output


def test_rotate_retains_old_key_and_activates_fresh_id_non_interactively(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "rotate-keyring.json"
    _initialize(path)
    old_ring = FileTokenV2KeyRingStore(path).load()
    old_claims = old_ring.create_claims(
        tenant_id=f"tnt_{'1' * 32}",
        principal_id=f"prn_{'2' * 32}",
        job_id=f"job_{'3' * 32}",
        artifact_id=f"art_{'4' * 32}",
        purpose=FileTokenV2Purpose.DOWNLOAD_RUN_ARTIFACT,
        sha256="5" * 64,
        ttl_seconds=60,
        now=100,
    )
    old_token = old_ring.issue(old_claims)

    assert cli.main(["rotate", "--path", str(path), "--new-kid", "key-new"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "active_kid": "key-new",
        "retained_kids": ["key-new", "key-old"],
        "count": 2,
    }
    loaded = FileTokenV2KeyRingStore(path).load()
    assert loaded.active_kid == "key-new"
    assert set(loaded.retained_kids) == {"key-old", "key-new"}
    assert loaded.verify(old_token, now=101).aid == f"art_{'4' * 32}"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_rotate_rejects_duplicate_id_with_safe_code_and_no_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "private-duplicate-keyring.json"
    _initialize(path)

    assert cli.main(["rotate", "--path", str(path), "--new-kid", "key-old"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == {
        "code": "duplicate_key_id",
        "message": "file-token v2 key-ring operation failed",
    }
    assert str(path) not in captured.err


@pytest.mark.parametrize("unsafe_kind", ["missing", "corrupt", "symlink"])
def test_failures_emit_only_safe_code_and_never_initialize_or_leak(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    unsafe_kind: str,
) -> None:
    path = tmp_path / f"private-{unsafe_kind}-keyring.json"
    secret = "secret-key-material-that-must-not-appear"
    expected_code = "missing"
    if unsafe_kind == "corrupt":
        path.write_text(secret, encoding="utf-8")
        path.chmod(0o600)
        expected_code = "invalid_payload"
    elif unsafe_kind == "symlink":
        target = tmp_path / "target-keyring.json"
        _initialize(target)
        path.symlink_to(target)
        expected_code = "unsafe_type"

    assert cli.main(["status", "--path", str(path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {
            "code": expected_code,
            "message": "file-token v2 key-ring operation failed",
        },
        "status": "error",
    }
    assert str(path) not in captured.err
    assert secret not in captured.err
    if unsafe_kind == "missing":
        assert not path.exists()
