from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.backup_restore as cli


def _settings(tmp_path: Path, *, database_url: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        database_url=database_url or "sqlite:///data/nanoloop.db",
        output_root=tmp_path / "outputs",
        model_snapshot_root=tmp_path / "data" / "model-snapshots",
        knowledge_source_dir=tmp_path / "knowledge" / "sources",
        faiss_index_path=tmp_path / "knowledge" / "index" / "faiss.index",
    )


def _capture_layout(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def build_layout(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(cli, "BackupLayout", build_layout)
    return captured


def test_create_maps_settings_and_default_secret_without_leaking_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    monkeypatch.delenv("NANOLOOP_FILE_TOKEN_SECRET_FILE", raising=False)
    layout = _capture_layout(monkeypatch)
    captured_call: dict[str, object] = {}

    def fake_create(
        built_layout: object,
        archive_path: Path,
        *,
        offline_confirmed: bool,
    ) -> SimpleNamespace:
        captured_call.update(
            layout=built_layout,
            archive_path=archive_path,
            offline_confirmed=offline_confirmed,
        )
        return SimpleNamespace(
            archive_sha256="a" * 64,
            file_count=7,
            total_bytes=1234,
            checksum_path=archive_path.with_suffix(".zip.sha256"),
            secret="must-not-appear",
            source_layout={"database": "/private/source.db"},
        )

    monkeypatch.setattr(cli, "create_backup", fake_create)
    archive = tmp_path / "backups" / "state.zip"

    assert cli.main(["create", str(archive), "--offline-confirmed"]) == 0

    assert layout == {
        "database_path": (tmp_path / "data" / "nanoloop.db").resolve(),
        "data_root": (tmp_path / "data").resolve(),
        "output_root": (tmp_path / "outputs").resolve(),
        "model_snapshot_root": (tmp_path / "data" / "model-snapshots").resolve(),
        "knowledge_source_root": (tmp_path / "knowledge" / "sources").resolve(),
        "knowledge_index_root": (tmp_path / "knowledge" / "index").resolve(),
        "file_token_secret_file": None,
    }
    assert captured_call["offline_confirmed"] is True
    assert captured_call["archive_path"] == archive.resolve()
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["archive_sha256"] == "a" * 64
    assert payload["file_count"] == 7
    assert payload["total_bytes"] == 1234
    assert "must-not-appear" not in output
    assert "/private/source.db" not in output
    assert ".file_token_secret" not in output


def test_create_honors_all_path_overrides_and_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    environment_secret = tmp_path / "environment" / "token-secret"
    monkeypatch.setenv("NANOLOOP_FILE_TOKEN_SECRET_FILE", str(environment_secret))
    layout = _capture_layout(monkeypatch)
    monkeypatch.setattr(
        cli,
        "create_backup",
        lambda *_args, **_kwargs: {"sha256": "b" * 64, "member_count": 2},
    )
    overrides = {
        "database_path": tmp_path / "custom" / "state.sqlite3",
        "data_root": tmp_path / "custom-data",
        "output_root": tmp_path / "custom-outputs",
        "model_snapshot_root": tmp_path / "custom-models",
        "knowledge_source_root": tmp_path / "custom-sources",
        "knowledge_index_root": tmp_path / "custom-index",
        "file_token_secret_file": tmp_path / "explicit-secret",
    }
    arguments = ["create", str(tmp_path / "state.zip"), "--offline-confirmed"]
    for name, path in overrides.items():
        arguments.extend((f"--{name.replace('_', '-')}", str(path)))

    assert cli.main(arguments) == 0

    assert layout == {name: path.resolve() for name, path in overrides.items()}
    output = capsys.readouterr().out
    assert str(environment_secret) not in output
    assert str(overrides["file_token_secret_file"]) not in output


def test_create_uses_environment_secret_file_when_not_explicitly_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    environment_secret = tmp_path / "operator-secrets" / "file-token"
    monkeypatch.setenv("NANOLOOP_FILE_TOKEN_SECRET_FILE", str(environment_secret))
    layout = _capture_layout(monkeypatch)
    monkeypatch.setattr(cli, "create_backup", lambda *_args, **_kwargs: None)

    assert cli.main(["create", str(tmp_path / "backups" / "state.zip"), "--offline-confirmed"]) == 0

    assert layout["file_token_secret_file"] == environment_secret.resolve()
    assert str(environment_secret) not in capsys.readouterr().out


def test_create_uses_existing_default_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    monkeypatch.delenv("NANOLOOP_FILE_TOKEN_SECRET_FILE", raising=False)
    secret_file = tmp_path / "data" / ".file_token_secret"
    secret_file.parent.mkdir(parents=True)
    secret_file.write_text("persisted-secret-value", encoding="utf-8")
    layout = _capture_layout(monkeypatch)
    monkeypatch.setattr(cli, "create_backup", lambda *_args, **_kwargs: None)

    assert cli.main(["create", str(tmp_path / "state.zip"), "--offline-confirmed"]) == 0

    assert layout["file_token_secret_file"] == secret_file.resolve()
    output = capsys.readouterr().out
    assert "persisted-secret-value" not in output
    assert ".file_token_secret" not in output


def test_injected_secret_does_not_require_a_default_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path))
    monkeypatch.delenv("NANOLOOP_FILE_TOKEN_SECRET_FILE", raising=False)
    injected_secret = "operator-injected-secret-that-must-not-leak"
    monkeypatch.setenv("NANOLOOP_FILE_TOKEN_SECRET", injected_secret)
    layout = _capture_layout(monkeypatch)
    monkeypatch.setattr(cli, "create_backup", lambda *_args, **_kwargs: None)

    assert cli.main(["create", str(tmp_path / "state.zip"), "--offline-confirmed"]) == 0

    assert layout["file_token_secret_file"] is None
    assert injected_secret not in capsys.readouterr().out


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://user:do-not-print@example.invalid/nanoloop",
        "sqlite:///:memory:",
        "sqlite:///file:memorydb?mode=memory&uri=true",
        "sqlite://",
    ],
)
def test_create_rejects_non_file_sqlite_without_echoing_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    database_url: str,
) -> None:
    monkeypatch.setattr(cli, "Settings", lambda: _settings(tmp_path, database_url=database_url))
    monkeypatch.setattr(
        cli,
        "create_backup",
        lambda *_args, **_kwargs: pytest.fail("core must not be called"),
    )

    assert cli.main(["create", str(tmp_path / "state.zip"), "--offline-confirmed"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert database_url not in captured.err
    assert "do-not-print" not in captured.err


def test_verify_maps_explicit_checksum_without_offline_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_verify(archive_path: Path, *, checksum_path: Path | None) -> dict[str, object]:
        captured.update(archive_path=archive_path, checksum_path=checksum_path)
        return {"archive_sha256": "c" * 64, "verified_file_count": 9}

    monkeypatch.setattr(cli, "verify_backup", fake_verify)
    archive = tmp_path / "state.zip"
    checksum = tmp_path / "checksums" / "state.sha256"

    assert cli.main(["verify", str(archive), "--checksum-path", str(checksum)]) == 0

    assert captured == {
        "archive_path": archive.resolve(),
        "checksum_path": checksum.resolve(),
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "verify"
    assert payload["archive_sha256"] == "c" * 64
    assert payload["file_count"] == 9


def test_metrics_are_derived_from_the_strict_manifest_without_printing_members() -> None:
    report = SimpleNamespace(
        archive_sha256="d" * 64,
        manifest=SimpleNamespace(
            files=(
                SimpleNamespace(path="data/nanoloop.db", size=10),
                SimpleNamespace(path="data/.file_token_secret", size=20),
            )
        ),
    )

    assert cli._safe_result_metrics(report) == {
        "archive_sha256": "d" * 64,
        "file_count": 2,
        "total_bytes": 30,
    }


def test_restore_maps_destination_checksum_and_offline_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_restore(
        archive_path: Path,
        destination_root: Path,
        *,
        offline_confirmed: bool,
        checksum_path: Path | None,
    ) -> SimpleNamespace:
        captured.update(
            archive_path=archive_path,
            destination_root=destination_root,
            offline_confirmed=offline_confirmed,
            checksum_path=checksum_path,
        )
        return SimpleNamespace(restored_file_count=4, verified_bytes=456)

    monkeypatch.setattr(cli, "restore_backup", fake_restore)
    archive = tmp_path / "state.zip"
    destination = tmp_path / "fresh-restore"
    checksum = tmp_path / "state.sha256"

    assert (
        cli.main(
            [
                "restore",
                str(archive),
                str(destination),
                "--checksum-path",
                str(checksum),
                "--offline-confirmed",
            ]
        )
        == 0
    )

    assert captured == {
        "archive_path": archive.resolve(),
        "destination_root": destination.resolve(),
        "offline_confirmed": True,
        "checksum_path": checksum.resolve(),
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["destination_root"] == str(destination.resolve())
    assert payload["file_count"] == 4
    assert payload["total_bytes"] == 456


@pytest.mark.parametrize(
    "arguments",
    [
        ["create", "state.zip"],
        ["restore", "state.zip", "fresh"],
    ],
)
def test_mutating_commands_require_explicit_offline_confirmation(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2


def test_rejects_non_zip_archive_before_calling_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "verify_backup",
        lambda *_args, **_kwargs: pytest.fail("core must not be called"),
    )

    assert cli.main(["verify", str(tmp_path / "state.tar")]) == 2
    assert capsys.readouterr().out == ""


def test_path_normalization_preserves_symlink_for_core_rejection(tmp_path: Path) -> None:
    target = tmp_path / "target.zip"
    target.write_bytes(b"not-a-backup")
    link = tmp_path / "linked.zip"
    link.symlink_to(target)

    assert cli._path(link) == link.absolute()
    assert cli._path(link).is_symlink()


def test_core_failure_has_stable_exit_code_and_redacted_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "persisted-super-secret-value"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(f"{secret} at /private/managed/source")

    monkeypatch.setattr(cli, "verify_backup", fail)

    assert cli.main(["verify", str(tmp_path / "state.zip")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert secret not in captured.err
    assert "/private/managed/source" not in captured.err
    payload = json.loads(captured.err)
    assert payload == {
        "error": {"message": "backup operation failed", "type": "RuntimeError"},
        "status": "error",
    }
