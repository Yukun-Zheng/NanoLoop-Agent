#!/usr/bin/env python3
"""Secret-safe operator CLI for tenant, principal, and credential lifecycle management."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, TextIO, TypeVar, cast

from pydantic import SecretStr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.contracts.identity import (  # noqa: E402
    PrincipalKind,
    PrincipalRole,
    validate_credential_id,
    validate_principal_handle,
    validate_principal_id,
    validate_tenant_id,
    validate_tenant_slug,
)
from app.core.config import Settings  # noqa: E402
from app.core.identity import (  # noqa: E402
    generate_principal_id,
    generate_tenant_id,
    issue_credential,
)
from app.db.identity import (  # noqa: E402
    CredentialMetadata,
    IdentityActorKind,
    IdentityService,
)
from app.db.session import Database  # noqa: E402

_ERROR_TYPE_PATTERN = re.compile(r"\A[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_T = TypeVar("_T")


class CliUsageError(ValueError):
    """An invalid invocation whose original value must not be reflected."""


class CliStateUnchanged(RuntimeError):
    """A compare-and-set lifecycle operation did not change persistent state."""


class CliCredentialCommitUncertain(RuntimeError):
    """A credential row may have committed after the token had to be destroyed."""

    def __init__(self, credential_id: str) -> None:
        super().__init__("credential commit outcome is uncertain")
        self.credential_id = validate_credential_id(credential_id)


class CliTokenDestructionError(RuntimeError):
    """The CLI could not confirm destruction of uncommitted token bytes."""

    def __init__(self, credential_id: str) -> None:
        super().__init__("credential token destruction could not be confirmed")
        self.credential_id = validate_credential_id(credential_id)


class SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that never echoes an invalid command-line value."""

    def error(self, _message: str) -> NoReturn:
        raise CliUsageError("invalid identity command")


@dataclass(slots=True, repr=False)
class _ExclusiveTokenFile:
    """Open token file retained across commit so failure cleanup targets only its descriptor."""

    path: Path
    parent_fd: int
    file_fd: int
    device: int
    inode: int

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path=<redacted>, token=<redacted>)"

    @classmethod
    def create(cls, path_value: str | Path, token: SecretStr) -> _ExclusiveTokenFile:
        path = Path(path_value).expanduser().absolute()
        if path.name in {"", ".", ".."}:
            raise CliUsageError("invalid token output")
        parent = path.parent
        try:
            parent_before = parent.lstat()
        except OSError as error:
            raise CliUsageError("token output parent is unavailable") from error
        if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
            raise CliUsageError("token output parent must be a real directory")

        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            parent_fd = os.open(parent, parent_flags)
        except OSError as error:
            raise CliUsageError("token output parent is unavailable") from error
        file_fd = -1
        created: _ExclusiveTokenFile | None = None
        try:
            parent_after = os.fstat(parent_fd)
            if (
                parent_after.st_dev != parent_before.st_dev
                or parent_after.st_ino != parent_before.st_ino
            ):
                raise CliUsageError("token output parent changed")
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            file_fd = os.open(path.name, file_flags, 0o600, dir_fd=parent_fd)
            os.fchmod(file_fd, 0o600)
            identity = os.fstat(file_fd)
            if not stat.S_ISREG(identity.st_mode) or stat.S_IMODE(identity.st_mode) != 0o600:
                raise CliUsageError("token output is not a private regular file")
            created = cls(
                path=path,
                parent_fd=parent_fd,
                file_fd=file_fd,
                device=identity.st_dev,
                inode=identity.st_ino,
            )
            payload = f"{token.get_secret_value()}\n".encode("ascii")
            _write_all(file_fd, payload)
            os.fsync(file_fd)
            os.fsync(parent_fd)
            return created
        except BaseException:
            if created is not None:
                created.destroy_token_contents()
            elif file_fd >= 0:
                try:
                    identity = os.fstat(file_fd)
                    candidate = cls(path, parent_fd, file_fd, identity.st_dev, identity.st_ino)
                    candidate.destroy_token_contents()
                except OSError:
                    pass
            if file_fd >= 0:
                with suppress(OSError):
                    os.close(file_fd)
            with suppress(OSError):
                os.close(parent_fd)
            raise

    def destroy_token_contents(self) -> bool:
        """Destroy token bytes through the original descriptor without touching a path name."""

        try:
            opened = os.fstat(self.file_fd)
        except OSError:
            return False
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != self.device
            or opened.st_ino != self.inode
        ):
            return False
        try:
            os.ftruncate(self.file_fd, 0)
            os.fsync(self.file_fd)
        except OSError:
            return False
        return True

    def close(self) -> None:
        for descriptor in (self.file_fd, self.parent_fd):
            with suppress(OSError):
                os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("token output write failed")
        written += count


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_expires_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        offset = parsed.utcoffset()
    except (TypeError, ValueError, OverflowError) as error:
        raise CliUsageError("invalid credential expiry") from error
    if parsed.tzinfo is None or offset is None:
        raise CliUsageError("credential expiry requires a timezone")
    return parsed.astimezone(UTC)


def _load_settings() -> Settings:
    try:
        return Settings()
    except Exception as error:
        raise CliUsageError("NanoLoop configuration is invalid") from error


def _credential_pepper(settings: Settings) -> SecretStr | str | bytes:
    pepper = settings.credential_pepper
    if pepper is None:
        raise CliUsageError("credential pepper is not configured")
    return pepper


def _with_service(settings: Settings, operation: Callable[[IdentityService], _T]) -> _T:
    database: Database | None = None
    try:
        database = Database(settings)
        with database.session() as session:
            return operation(IdentityService.from_session(session))
    finally:
        if database is not None:
            with suppress(Exception):
                database.dispose()


def _safe_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.isoformat().replace("+00:00", "Z")


def _credential_payload(record: CredentialMetadata) -> dict[str, object]:
    return {
        "credential_id": record.credential_id,
        "created_at": _safe_datetime(record.created_at),
        "enabled": record.enabled,
        "expires_at": _safe_datetime(record.expires_at),
        "principal_id": record.principal_id,
        "revoked_at": _safe_datetime(record.revoked_at),
        "updated_at": _safe_datetime(record.updated_at),
        "version": record.version,
    }


def _run_tenant(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.action == "create":
        slug = validate_tenant_slug(args.slug)
        tenant_id = generate_tenant_id()
        record = _with_service(
            settings,
            lambda service: service.create_tenant(
                tenant_id=tenant_id,
                slug=slug,
                display_name=args.display_name,
                actor_kind=IdentityActorKind.OPERATOR_CLI,
                actor_principal_id=None,
                now=_utc_now(),
            ),
        )
        return {
            "command": "tenant.create",
            "enabled": record.enabled,
            "status": "ok",
            "tenant_id": record.tenant_id,
        }

    tenant_id = validate_tenant_id(args.tenant_id)
    enabled = args.action == "enable"
    changed = _with_service(
        settings,
        lambda service: service.set_tenant_enabled(
            tenant_id,
            enabled=enabled,
            actor_kind=IdentityActorKind.OPERATOR_CLI,
            actor_principal_id=None,
            now=_utc_now(),
        ),
    )
    if not changed:
        raise CliStateUnchanged("tenant state was unchanged")
    return {
        "command": f"tenant.{args.action}",
        "enabled": enabled,
        "status": "ok",
        "tenant_id": tenant_id,
    }


def _run_principal(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.action == "create":
        tenant_id = validate_tenant_id(args.tenant_id)
        handle = validate_principal_handle(args.handle)
        principal_id = generate_principal_id()
        record = _with_service(
            settings,
            lambda service: service.create_principal(
                principal_id=principal_id,
                tenant_id=tenant_id,
                handle=handle,
                display_name=args.display_name,
                kind=PrincipalKind(args.kind),
                role=PrincipalRole(args.role),
                actor_kind=IdentityActorKind.OPERATOR_CLI,
                actor_principal_id=None,
                now=_utc_now(),
            ),
        )
        return {
            "command": "principal.create",
            "enabled": record.enabled,
            "principal_id": record.principal_id,
            "status": "ok",
            "tenant_id": record.tenant_id,
        }

    principal_id = validate_principal_id(args.principal_id)
    enabled = args.action == "enable"
    changed = _with_service(
        settings,
        lambda service: service.set_principal_enabled(
            principal_id,
            enabled=enabled,
            actor_kind=IdentityActorKind.OPERATOR_CLI,
            actor_principal_id=None,
            now=_utc_now(),
        ),
    )
    if not changed:
        raise CliStateUnchanged("principal state was unchanged")
    return {
        "command": f"principal.{args.action}",
        "enabled": enabled,
        "principal_id": principal_id,
        "status": "ok",
    }


def _run_credential(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.action == "issue":
        return _issue_credential(args, settings)
    if args.action == "list":
        principal_id = (
            validate_principal_id(args.principal_id) if args.principal_id is not None else None
        )
        records = _with_service(
            settings,
            lambda service: service.list_credentials(principal_id),
        )
        return {
            "command": "credential.list",
            "credentials": [_credential_payload(record) for record in records],
            "status": "ok",
        }

    credential_id = validate_credential_id(args.credential_id)
    if args.action == "revoke":
        changed = _with_service(
            settings,
            lambda service: service.revoke_credential(
                credential_id,
                actor_kind=IdentityActorKind.OPERATOR_CLI,
                actor_principal_id=None,
                now=_utc_now(),
            ),
        )
        enabled: bool | None = None
    else:
        enabled = args.action == "enable"
        changed = _with_service(
            settings,
            lambda service: service.set_credential_enabled(
                credential_id,
                enabled=cast(bool, enabled),
                actor_kind=IdentityActorKind.OPERATOR_CLI,
                actor_principal_id=None,
                now=_utc_now(),
            ),
        )
    if not changed:
        raise CliStateUnchanged("credential state was unchanged")
    payload: dict[str, object] = {
        "command": f"credential.{args.action}",
        "credential_id": credential_id,
        "status": "ok",
    }
    if enabled is not None:
        payload["enabled"] = enabled
    else:
        payload["revoked"] = True
    return payload


def _issue_credential(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    principal_id = validate_principal_id(args.principal_id)
    expires_at = _parse_expires_at(args.expires_at)
    issued_at = _utc_now()
    if expires_at is not None and expires_at <= issued_at:
        raise CliUsageError("credential expiry must be in the future")
    issued = issue_credential(_credential_pepper(settings))
    token_file = _ExclusiveTokenFile.create(args.token_output, issued.token)
    committed = False
    record_staged = False
    database: Database | None = None
    try:
        database = Database(settings)
        with database.session() as session:
            record = IdentityService.from_session(session).issue_credential(
                credential_id=issued.credential_id,
                principal_id=principal_id,
                token_digest=issued.digest,
                label=args.label,
                expires_at=expires_at,
                actor_kind=IdentityActorKind.OPERATOR_CLI,
                actor_principal_id=None,
                now=issued_at,
            )
            record_staged = True
        committed = True
    except BaseException as error:
        if not token_file.destroy_token_contents():
            raise CliTokenDestructionError(issued.credential_id) from error
        if record_staged:
            raise CliCredentialCommitUncertain(issued.credential_id) from error
        raise
    finally:
        token_file.close()
        if database is not None:
            with suppress(Exception):
                database.dispose()
    if not committed:
        raise RuntimeError("credential transaction did not commit")
    return {
        "command": "credential.issue",
        "credential_id": record.credential_id,
        "enabled": record.enabled,
        "principal_id": record.principal_id,
        "status": "ok",
        "token_written": True,
    }


def _add_state_actions(
    subparsers: Any,
    *,
    entity_name: str,
    identifier_option: str,
) -> None:
    for action in ("disable", "enable"):
        parser = subparsers.add_parser(action, help=f"{action} one {entity_name}")
        parser.add_argument(identifier_option, required=True)


def build_parser() -> SafeArgumentParser:
    parser = SafeArgumentParser(description="Manage NanoLoop operator identities and credentials.")
    entities = parser.add_subparsers(dest="entity", required=True, parser_class=SafeArgumentParser)

    tenant = entities.add_parser("tenant", help="manage tenants")
    tenant_actions = tenant.add_subparsers(
        dest="action", required=True, parser_class=SafeArgumentParser
    )
    tenant_create = tenant_actions.add_parser("create", help="create one tenant")
    tenant_create.add_argument("--slug", required=True)
    tenant_create.add_argument("--display-name", required=True)
    _add_state_actions(tenant_actions, entity_name="tenant", identifier_option="--tenant-id")

    principal = entities.add_parser("principal", help="manage principals")
    principal_actions = principal.add_subparsers(
        dest="action", required=True, parser_class=SafeArgumentParser
    )
    principal_create = principal_actions.add_parser("create", help="create one principal")
    principal_create.add_argument("--tenant-id", required=True)
    principal_create.add_argument("--handle", required=True)
    principal_create.add_argument("--display-name", required=True)
    principal_create.add_argument(
        "--kind", choices=[kind.value for kind in PrincipalKind], required=True
    )
    principal_create.add_argument(
        "--role", choices=[role.value for role in PrincipalRole], required=True
    )
    _add_state_actions(
        principal_actions,
        entity_name="principal",
        identifier_option="--principal-id",
    )

    credential = entities.add_parser("credential", help="manage API credentials")
    credential_actions = credential.add_subparsers(
        dest="action", required=True, parser_class=SafeArgumentParser
    )
    issue = credential_actions.add_parser("issue", help="issue one credential")
    issue.add_argument("--principal-id", required=True)
    issue.add_argument("--label", required=True)
    issue.add_argument("--expires-at")
    issue.add_argument("--token-output", required=True)
    for action in ("disable", "enable", "revoke"):
        action_parser = credential_actions.add_parser(action, help=f"{action} one credential")
        action_parser.add_argument("--credential-id", required=True)
    list_parser = credential_actions.add_parser("list", help="list secret-free metadata")
    list_parser.add_argument("--principal-id")
    return parser


def _emit(payload: Mapping[str, object], *, stream: TextIO) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=stream)


def _safe_error_type(error: Exception) -> str:
    name = type(error).__name__
    return name if _ERROR_TYPE_PATTERN.fullmatch(name) else "Error"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        settings = _load_settings()
        if args.entity == "tenant":
            payload = _run_tenant(args, settings)
        elif args.entity == "principal":
            payload = _run_principal(args, settings)
        else:
            payload = _run_credential(args, settings)
        _emit(payload, stream=sys.stdout)
        return 0
    except CliStateUnchanged as error:
        _emit(
            {
                "error": "identity state change was not applied",
                "error_type": _safe_error_type(error),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 3
    except CliCredentialCommitUncertain as error:
        _emit(
            {
                "credential_id": error.credential_id,
                "error": "credential commit outcome is uncertain; token contents were destroyed",
                "error_type": _safe_error_type(error),
                "recovery_action": "list_then_revoke_if_present",
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 4
    except CliTokenDestructionError as error:
        _emit(
            {
                "credential_id": error.credential_id,
                "error": "credential token destruction could not be confirmed",
                "error_type": _safe_error_type(error),
                "recovery_action": "secure_token_output_then_list_and_revoke",
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 5
    except KeyboardInterrupt:
        _emit(
            {
                "error": "identity operation interrupted",
                "error_type": "KeyboardInterrupt",
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 130
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    except Exception as error:
        _emit(
            {
                "error": "identity operation failed",
                "error_type": _safe_error_type(error),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
