from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from pydantic import SecretStr

import scripts.manage_identity as cli
from app.contracts.identity import PrincipalKind, PrincipalRole
from app.core.identity import IssuedCredential
from app.db.identity import CredentialMetadata, IdentityActorKind

_TENANT_ID = "tnt_11111111111111111111111111111111"
_PRINCIPAL_ID = "prn_22222222222222222222222222222222"
_CREDENTIAL_ID = "crd_33333333333333333333333333333333"
_TOKEN = f"nlk_v1_{_CREDENTIAL_ID}_{'A' * 43}"
_PEPPER = SecretStr("operator-test-pepper-is-at-least-thirty-two-bytes")
_NOW = datetime(2026, 7, 18, 6, 0, tzinfo=UTC)


class _SessionContext:
    def __init__(self, *, commit_error: BaseException | None = None) -> None:
        self.commit_error = commit_error

    def __enter__(self) -> object:
        return object()

    def __exit__(
        self,
        error_type: object,
        _error: object,
        _traceback: object,
    ) -> Literal[False]:
        if error_type is None and self.commit_error is not None:
            raise self.commit_error
        return False


class _Database:
    def __init__(self, *, commit_error: BaseException | None = None) -> None:
        self.context = _SessionContext(commit_error=commit_error)
        self.disposed = False

    def session(self) -> _SessionContext:
        return self.context

    def dispose(self) -> None:
        self.disposed = True


class _Service:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.changed = True
        self.credentials: list[CredentialMetadata] = []

    def create_tenant(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("create_tenant", kwargs))
        return SimpleNamespace(
            tenant_id=kwargs["tenant_id"],
            enabled=True,
        )

    def set_tenant_enabled(
        self,
        tenant_id: str,
        **kwargs: object,
    ) -> bool:
        self.calls.append(("set_tenant_enabled", {"tenant_id": tenant_id, **kwargs}))
        return self.changed

    def create_principal(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("create_principal", kwargs))
        return SimpleNamespace(
            principal_id=kwargs["principal_id"],
            tenant_id=kwargs["tenant_id"],
            enabled=True,
        )

    def set_principal_enabled(
        self,
        principal_id: str,
        **kwargs: object,
    ) -> bool:
        self.calls.append(("set_principal_enabled", {"principal_id": principal_id, **kwargs}))
        return self.changed

    def issue_credential(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("issue_credential", kwargs))
        return SimpleNamespace(
            credential_id=kwargs["credential_id"],
            principal_id=kwargs["principal_id"],
            enabled=True,
        )

    def set_credential_enabled(
        self,
        credential_id: str,
        **kwargs: object,
    ) -> bool:
        self.calls.append(("set_credential_enabled", {"credential_id": credential_id, **kwargs}))
        return self.changed

    def revoke_credential(
        self,
        credential_id: str,
        **kwargs: object,
    ) -> bool:
        self.calls.append(("revoke_credential", {"credential_id": credential_id, **kwargs}))
        return self.changed

    def list_credentials(self, principal_id: str | None = None) -> list[CredentialMetadata]:
        self.calls.append(("list_credentials", {"principal_id": principal_id}))
        return self.credentials


def _settings() -> Any:
    return SimpleNamespace(
        credential_pepper=_PEPPER,
        database_url="sqlite:///do-not-print.db",
    )


def _issued() -> IssuedCredential:
    return IssuedCredential(
        credential_id=_CREDENTIAL_ID,
        token=SecretStr(_TOKEN),
        digest=b"d" * 32,
    )


def _metadata(**overrides: object) -> CredentialMetadata:
    values: dict[str, object] = {
        "credential_id": _CREDENTIAL_ID,
        "principal_id": _PRINCIPAL_ID,
        "label": "automation",
        "enabled": True,
        "expires_at": datetime(2027, 1, 1, tzinfo=UTC),
        "revoked_at": None,
        "version": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    values.update(overrides)
    return CredentialMetadata(**cast(Any, values))


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    service: _Service,
    *,
    commit_error: BaseException | None = None,
) -> _Database:
    database = _Database(commit_error=commit_error)
    monkeypatch.setattr(cli, "_load_settings", _settings)
    monkeypatch.setattr(cli, "_utc_now", lambda: _NOW)
    monkeypatch.setattr(cli, "Database", lambda _settings_value: database)
    monkeypatch.setattr(
        cli,
        "IdentityService",
        SimpleNamespace(from_session=lambda _session: service),
    )
    return database


def _payload(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return cast(dict[str, object], json.loads(capsys.readouterr().out))


def test_tenant_create_generates_id_and_maps_operator_audit_transaction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    database = _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "generate_tenant_id", lambda: _TENANT_ID)

    assert cli.main(["tenant", "create", "--slug", "nano-team", "--display-name", "Nano"]) == 0

    name, call = service.calls[0]
    assert name == "create_tenant"
    assert call == {
        "tenant_id": _TENANT_ID,
        "slug": "nano-team",
        "display_name": "Nano",
        "actor_kind": IdentityActorKind.OPERATOR_CLI,
        "actor_principal_id": None,
        "now": _NOW,
    }
    assert _payload(capsys) == {
        "command": "tenant.create",
        "enabled": True,
        "status": "ok",
        "tenant_id": _TENANT_ID,
    }
    assert database.disposed is True


def test_principal_create_maps_strict_enums_handle_and_generated_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "generate_principal_id", lambda: _PRINCIPAL_ID)

    assert (
        cli.main(
            [
                "principal",
                "create",
                "--tenant-id",
                _TENANT_ID,
                "--handle",
                "lab.service",
                "--display-name",
                "Lab Service",
                "--kind",
                "service",
                "--role",
                "analyst",
            ]
        )
        == 0
    )

    name, call = service.calls[0]
    assert name == "create_principal"
    assert call["kind"] is PrincipalKind.SERVICE
    assert call["role"] is PrincipalRole.ANALYST
    assert call["handle"] == "lab.service"
    assert call["principal_id"] == _PRINCIPAL_ID
    assert call["actor_kind"] is IdentityActorKind.OPERATOR_CLI
    assert _payload(capsys)["principal_id"] == _PRINCIPAL_ID


@pytest.mark.parametrize(
    ("arguments", "method", "identifier_key", "enabled"),
    [
        (
            ["tenant", "disable", "--tenant-id", _TENANT_ID],
            "set_tenant_enabled",
            "tenant_id",
            False,
        ),
        (["tenant", "enable", "--tenant-id", _TENANT_ID], "set_tenant_enabled", "tenant_id", True),
        (
            ["principal", "disable", "--principal-id", _PRINCIPAL_ID],
            "set_principal_enabled",
            "principal_id",
            False,
        ),
        (
            ["principal", "enable", "--principal-id", _PRINCIPAL_ID],
            "set_principal_enabled",
            "principal_id",
            True,
        ),
        (
            ["credential", "disable", "--credential-id", _CREDENTIAL_ID],
            "set_credential_enabled",
            "credential_id",
            False,
        ),
        (
            ["credential", "enable", "--credential-id", _CREDENTIAL_ID],
            "set_credential_enabled",
            "credential_id",
            True,
        ),
    ],
)
def test_enable_disable_commands_map_to_compare_and_set_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    method: str,
    identifier_key: str,
    enabled: bool,
) -> None:
    service = _Service()
    _wire(monkeypatch, service)

    assert cli.main(arguments) == 0

    name, call = service.calls[0]
    assert name == method
    assert call[identifier_key] == arguments[-1]
    assert call["enabled"] is enabled
    assert call["actor_kind"] is IdentityActorKind.OPERATOR_CLI
    assert call["actor_principal_id"] is None
    assert call["now"] == _NOW
    assert _payload(capsys)["status"] == "ok"


def test_revoke_maps_to_permanent_service_transition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    _wire(monkeypatch, service)

    assert cli.main(["credential", "revoke", "--credential-id", _CREDENTIAL_ID]) == 0

    assert service.calls == [
        (
            "revoke_credential",
            {
                "credential_id": _CREDENTIAL_ID,
                "actor_kind": IdentityActorKind.OPERATOR_CLI,
                "actor_principal_id": None,
                "now": _NOW,
            },
        )
    ]
    assert _payload(capsys)["revoked"] is True


def test_compare_and_set_false_has_stable_nonzero_exit_and_no_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    service.changed = False
    _wire(monkeypatch, service)

    assert cli.main(["credential", "enable", "--credential-id", _CREDENTIAL_ID]) == 3

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "identity state change was not applied",
        "error_type": "CliStateUnchanged",
        "status": "error",
    }


def test_issue_writes_private_token_then_commits_only_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    database = _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())
    token_path = tmp_path / "operator.token"

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(token_path),
            ]
        )
        == 0
    )

    assert token_path.read_text(encoding="ascii") == f"{_TOKEN}\n"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    name, call = service.calls[0]
    assert name == "issue_credential"
    assert call["token_digest"] == b"d" * 32
    assert call["actor_kind"] is IdentityActorKind.OPERATOR_CLI
    assert "token" not in call
    captured = capsys.readouterr()
    assert _TOKEN not in captured.out
    assert (b"d" * 32).hex() not in captured.out
    assert str(token_path) not in captured.out
    assert json.loads(captured.out)["token_written"] is True
    assert database.disposed is True


@pytest.mark.parametrize("existing_kind", ["file", "symlink"])
def test_issue_never_overwrites_existing_file_or_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing_kind: str,
) -> None:
    service = _Service()
    _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())
    token_path = tmp_path / "operator.token"
    target = tmp_path / "target"
    if existing_kind == "file":
        token_path.write_text("keep-me", encoding="utf-8")
    else:
        target.write_text("keep-target", encoding="utf-8")
        token_path.symlink_to(target)

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(token_path),
            ]
        )
        == 2
    )

    assert service.calls == []
    assert (target if existing_kind == "symlink" else token_path).read_text() in {
        "keep-me",
        "keep-target",
    }
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(token_path) not in captured.err
    assert _TOKEN not in captured.err


def test_issue_rejects_symlink_parent_without_creating_target_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(linked_parent / "credential"),
            ]
        )
        == 2
    )

    assert not (real_parent / "credential").exists()
    assert service.calls == []
    assert str(linked_parent) not in capsys.readouterr().err


def test_commit_failure_destroys_token_and_leaves_private_empty_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_reason = "db-password=never-print"
    service = _Service()
    _wire(monkeypatch, service, commit_error=RuntimeError(secret_reason))
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())
    token_path = tmp_path / "operator.token"

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(token_path),
            ]
        )
        == 4
    )

    assert token_path.exists()
    assert token_path.stat().st_size == 0
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    captured = capsys.readouterr()
    assert captured.out == ""
    assert secret_reason not in captured.err
    assert _TOKEN not in captured.err
    error_payload = json.loads(captured.err)
    assert error_payload["credential_id"] == _CREDENTIAL_ID
    assert error_payload["recovery_action"] == "list_then_revoke_if_present"


def test_commit_failure_does_not_delete_replacement_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token_path = tmp_path / "operator.token"

    class ReplacingService(_Service):
        def issue_credential(self, **kwargs: object) -> SimpleNamespace:
            result = super().issue_credential(**kwargs)
            token_path.unlink()
            token_path.write_text("replacement-owned-by-someone-else", encoding="utf-8")
            return result

    service = ReplacingService()
    _wire(monkeypatch, service, commit_error=RuntimeError("commit failed"))
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(token_path),
            ]
        )
        == 4
    )

    assert token_path.read_text() == "replacement-owned-by-someone-else"
    assert _TOKEN not in capsys.readouterr().err


def test_cleanup_uses_original_fd_and_never_unlinks_racing_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / "operator.token"
    replacement_path = tmp_path / "replacement"
    replacement_path.write_text("replacement-owned-by-someone-else", encoding="utf-8")
    handle = cli._ExclusiveTokenFile.create(token_path, SecretStr(_TOKEN))
    real_ftruncate = os.ftruncate
    calls = 0

    def racing_ftruncate(descriptor: int, length: int) -> None:
        nonlocal calls
        calls += 1
        os.replace(replacement_path, token_path)
        real_ftruncate(descriptor, length)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        pytest.fail("rollback cleanup must never unlink a path name")

    monkeypatch.setattr(os, "ftruncate", racing_ftruncate)
    monkeypatch.setattr(os, "unlink", forbidden_unlink)
    try:
        assert handle.destroy_token_contents() is True
        assert calls == 1
        assert os.fstat(handle.file_fd).st_size == 0
        assert token_path.read_text() == "replacement-owned-by-someone-else"
    finally:
        handle.close()


def test_keyboard_interrupt_destroys_uncommitted_token_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class InterruptingService(_Service):
        def issue_credential(self, **_kwargs: object) -> SimpleNamespace:
            raise KeyboardInterrupt

    service = InterruptingService()
    _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())
    token_path = tmp_path / "operator.token"

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(token_path),
            ]
        )
        == 130
    )

    assert token_path.exists()
    assert token_path.stat().st_size == 0
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _TOKEN not in captured.err
    assert "KeyboardInterrupt" in captured.err


def test_issue_normalizes_timezone_expiry_to_utc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    _wire(monkeypatch, service)
    monkeypatch.setattr(cli, "issue_credential", lambda _pepper: _issued())

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "temporary",
                "--expires-at",
                "2026-07-19T14:30:00+08:00",
                "--token-output",
                str(tmp_path / "credential"),
            ]
        )
        == 0
    )

    assert service.calls[0][1]["expires_at"] == datetime(2026, 7, 19, 6, 30, tzinfo=UTC)
    capsys.readouterr()


def test_issue_rejects_naive_expiry_before_generating_or_writing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    _wire(monkeypatch, service)
    generated = False

    def unexpected_issue(_pepper: object) -> IssuedCredential:
        nonlocal generated
        generated = True
        return _issued()

    monkeypatch.setattr(cli, "issue_credential", unexpected_issue)
    token_path = tmp_path / "credential"

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "temporary",
                "--expires-at",
                "2026-07-19T14:30:00",
                "--token-output",
                str(token_path),
            ]
        )
        == 2
    )

    assert generated is False
    assert not token_path.exists()
    assert service.calls == []
    assert "2026-07-19" not in capsys.readouterr().err


@pytest.mark.parametrize("expires_at", ["2026-07-18T06:00:00Z", "2026-07-18T05:59:59Z"])
def test_issue_rejects_nonfuture_expiry_before_generating_or_writing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    expires_at: str,
) -> None:
    service = _Service()
    _wire(monkeypatch, service)
    generated = False

    def unexpected_issue(_pepper: object) -> IssuedCredential:
        nonlocal generated
        generated = True
        return _issued()

    monkeypatch.setattr(cli, "issue_credential", unexpected_issue)
    token_path = tmp_path / "credential"

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "expired",
                "--expires-at",
                expires_at,
                "--token-output",
                str(token_path),
            ]
        )
        == 2
    )
    assert generated is False
    assert service.calls == []
    assert not token_path.exists()
    assert expires_at not in capsys.readouterr().err


@pytest.mark.parametrize("pepper", [None, SecretStr("short")])
def test_missing_or_short_pepper_fails_without_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pepper: SecretStr | None,
) -> None:
    settings = SimpleNamespace(credential_pepper=pepper)
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)
    token_path = tmp_path / "credential"

    assert (
        cli.main(
            [
                "credential",
                "issue",
                "--principal-id",
                _PRINCIPAL_ID,
                "--label",
                "automation",
                "--token-output",
                str(token_path),
            ]
        )
        == 2
    )

    assert not token_path.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "short" not in captured.err
    assert str(token_path) not in captured.err


def test_credential_list_returns_only_secret_free_stable_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _Service()
    record = _metadata(label=_TOKEN)
    record_with_secret = SimpleNamespace(
        **{
            field: getattr(record, field)
            for field in (
                "credential_id",
                "principal_id",
                "label",
                "enabled",
                "expires_at",
                "revoked_at",
                "version",
                "created_at",
                "updated_at",
            )
        },
        token_digest=b"sensitive-digest",
    )
    service.credentials = cast(list[CredentialMetadata], [record_with_secret])
    _wire(monkeypatch, service)

    assert cli.main(["credential", "list", "--principal-id", _PRINCIPAL_ID]) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert service.calls == [("list_credentials", {"principal_id": _PRINCIPAL_ID})]
    assert payload["credentials"] == [
        {
            "credential_id": _CREDENTIAL_ID,
            "created_at": "2026-07-18T06:00:00Z",
            "enabled": True,
            "expires_at": "2027-01-01T00:00:00Z",
            "principal_id": _PRINCIPAL_ID,
            "revoked_at": None,
            "updated_at": "2026-07-18T06:00:00Z",
            "version": 1,
        }
    ]
    assert "digest" not in output
    assert b"sensitive-digest".hex() not in output
    assert _TOKEN not in output


def test_parser_and_runtime_errors_never_echo_values_or_exception_reasons(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forbidden = "raw-token-or-database-url-or-private-path"
    monkeypatch.setattr(cli, "_load_settings", _settings)

    assert cli.main(["credential", "issue", "--token", forbidden]) == 2

    first = capsys.readouterr()
    assert first.out == ""
    assert forbidden not in first.err

    class FailingService(_Service):
        def create_tenant(self, **_kwargs: object) -> SimpleNamespace:
            raise RuntimeError(forbidden)

    _wire(monkeypatch, FailingService())
    monkeypatch.setattr(cli, "generate_tenant_id", lambda: _TENANT_ID)
    assert cli.main(["tenant", "create", "--slug", "team", "--display-name", "Team"]) == 2

    second = capsys.readouterr()
    assert forbidden not in second.err
    assert "RuntimeError" in second.err


def test_token_file_repr_redacts_secret_and_physical_path(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "private-token"
    handle = cli._ExclusiveTokenFile.create(token_path, SecretStr(_TOKEN))
    try:
        representation = repr(handle)
        assert _TOKEN not in representation
        assert str(token_path) not in representation
        assert "<redacted>" in representation
    finally:
        handle.destroy_token_contents()
        handle.close()
