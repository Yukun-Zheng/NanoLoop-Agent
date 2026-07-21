from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import event, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.identity import AuthMode, PrincipalKind, PrincipalRole
from app.core.config import Settings
from app.core.identity import IssuedCredential, issue_credential
from app.db.base import Base
from app.db.identity import (
    CredentialAuthenticationStatus,
    IdentityActorKind,
    IdentityService,
)
from app.db.models import ApiCredential, IdentityAuditEvent, Principal, Tenant
from app.db.session import Database

_PEPPER = b"identity-test-pepper-material-32bytes-minimum"
_NOW = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
_TENANT_ID = f"tnt_{'1' * 32}"
_PRINCIPAL_ID = f"prn_{'2' * 32}"


@pytest.fixture
def database(tmp_path: Path) -> Database:
    instance = Database(Settings(database_url=f"sqlite:///{tmp_path / 'identity.db'}"))
    Base.metadata.create_all(instance.engine)
    try:
        yield instance
    finally:
        instance.dispose()


@pytest.fixture
def session(database: Database) -> Session:
    instance = database.session_factory()
    try:
        yield instance
    finally:
        instance.close()


def _seed_identity(
    session: Session,
    *,
    expires_at: datetime | None = None,
) -> tuple[IdentityService, IssuedCredential]:
    service = IdentityService.from_session(session)
    service.create_tenant(
        tenant_id=_TENANT_ID,
        slug="research-team",
        display_name="Research team",
        now=_NOW,
    )
    service.create_principal(
        principal_id=_PRINCIPAL_ID,
        tenant_id=_TENANT_ID,
        handle="operator",
        display_name="Operator",
        kind=PrincipalKind.USER,
        role=PrincipalRole.ANALYST,
        now=_NOW,
    )
    issued = issue_credential(_PEPPER)
    service.issue_credential(
        credential_id=issued.credential_id,
        principal_id=_PRINCIPAL_ID,
        token_digest=issued.digest,
        label="test workstation",
        expires_at=expires_at,
        now=_NOW,
    )
    session.commit()
    return service, issued


def test_active_credential_returns_safe_context_with_one_read_query(
    session: Session,
) -> None:
    service, issued = _seed_identity(session)
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(session.bind, "before_cursor_execute", capture_statement)
    try:
        result = service.authenticate(issued.credential_id, issued.digest, now=_NOW)
    finally:
        event.remove(session.bind, "before_cursor_execute", capture_statement)

    assert result.status is CredentialAuthenticationStatus.AUTHENTICATED
    assert result.authenticated is True
    assert result.principal is not None
    assert result.principal.tenant_id == _TENANT_ID
    assert result.principal.principal_id == _PRINCIPAL_ID
    assert result.principal.credential_id == issued.credential_id
    assert result.principal.kind is PrincipalKind.USER
    assert result.principal.role is PrincipalRole.ANALYST
    assert result.principal.auth_mode is AuthMode.PRINCIPAL
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("SELECT")


def test_unknown_and_bad_digest_are_read_only_failures(session: Session) -> None:
    service, issued = _seed_identity(session)
    unknown = service.authenticate(f"crd_{'f' * 32}", issued.digest, now=_NOW)
    bad_digest = service.authenticate(issued.credential_id, b"x" * 32, now=_NOW)
    wrong_length = service.authenticate(issued.credential_id, b"short", now=_NOW)

    assert unknown.status is CredentialAuthenticationStatus.UNKNOWN
    assert bad_digest.status is CredentialAuthenticationStatus.BAD_DIGEST
    assert wrong_length.status is CredentialAuthenticationStatus.BAD_DIGEST
    assert unknown.principal is bad_digest.principal is wrong_length.principal is None
    assert session.scalar(select(func.count()).select_from(IdentityAuditEvent)) == 3
    assert [column.name for column in inspect(ApiCredential).columns].count("last_used_at") == 0


def test_expiry_boundary_is_inclusive(session: Session) -> None:
    expiry = _NOW + timedelta(hours=1)
    service, issued = _seed_identity(session, expires_at=expiry)

    before = service.authenticate(
        issued.credential_id,
        issued.digest,
        now=expiry - timedelta(microseconds=1),
    )
    boundary = service.authenticate(issued.credential_id, issued.digest, now=expiry)

    assert before.status is CredentialAuthenticationStatus.AUTHENTICATED
    assert boundary.status is CredentialAuthenticationStatus.EXPIRED


@pytest.mark.parametrize(
    ("credential_enabled", "revoked", "expired", "principal_enabled", "tenant_enabled", "status"),
    [
        (False, True, True, False, False, CredentialAuthenticationStatus.REVOKED),
        (False, False, True, False, False, CredentialAuthenticationStatus.DISABLED),
        (True, False, True, False, False, CredentialAuthenticationStatus.EXPIRED),
        (
            True,
            False,
            False,
            False,
            False,
            CredentialAuthenticationStatus.PRINCIPAL_DISABLED,
        ),
        (
            True,
            False,
            False,
            True,
            False,
            CredentialAuthenticationStatus.TENANT_DISABLED,
        ),
    ],
)
def test_authentication_state_precedence_is_stable(
    session: Session,
    credential_enabled: bool,
    revoked: bool,
    expired: bool,
    principal_enabled: bool,
    tenant_enabled: bool,
    status: CredentialAuthenticationStatus,
) -> None:
    expiry = _NOW - timedelta(seconds=1) if expired else _NOW + timedelta(days=1)
    service, issued = _seed_identity(session, expires_at=expiry)
    credential = session.get(ApiCredential, issued.credential_id)
    principal = session.get(Principal, _PRINCIPAL_ID)
    tenant = session.get(Tenant, _TENANT_ID)
    assert credential is not None and principal is not None and tenant is not None
    credential.enabled = credential_enabled
    credential.revoked_at = _NOW if revoked else None
    principal.enabled = principal_enabled
    tenant.enabled = tenant_enabled
    session.commit()

    result = service.authenticate(issued.credential_id, issued.digest, now=_NOW)
    assert result.status is status


def test_plaintext_token_never_enters_schema_rows_or_repr(
    database: Database,
    session: Session,
) -> None:
    service, issued = _seed_identity(session)
    raw_token = issued.token.get_secret_value()
    credential = session.get(ApiCredential, issued.credential_id)
    assert credential is not None
    result = service.authenticate(issued.credential_id, issued.digest, now=_NOW)

    assert raw_token not in repr(credential)
    assert raw_token not in repr(result)
    assert raw_token not in repr(service)
    assert credential.token_digest == issued.digest
    columns = {column["name"] for column in inspect(database.engine).get_columns("api_credentials")}
    assert columns == {
        "credential_id",
        "principal_id",
        "label",
        "token_digest",
        "enabled",
        "expires_at",
        "revoked_at",
        "version",
        "created_at",
        "updated_at",
    }
    database_path = Path(database.engine.url.database or "")
    assert raw_token.encode("utf-8") not in database_path.read_bytes()


def test_identity_uniqueness_and_digest_length_constraints(session: Session) -> None:
    _, issued = _seed_identity(session)
    duplicate_tenant = Tenant(
        tenant_id=f"tnt_{'3' * 32}",
        slug="research-team",
        display_name="Duplicate slug",
        enabled=True,
        version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    session.add(duplicate_tenant)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    duplicate_handle = Principal(
        principal_id=f"prn_{'4' * 32}",
        tenant_id=_TENANT_ID,
        handle="operator",
        display_name="Duplicate handle",
        kind=PrincipalKind.USER.value,
        role=PrincipalRole.VIEWER.value,
        enabled=True,
        version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    session.add(duplicate_handle)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    other = issue_credential(_PEPPER)
    duplicate_digest = ApiCredential(
        credential_id=other.credential_id,
        principal_id=_PRINCIPAL_ID,
        label="duplicate digest",
        token_digest=issued.digest,
        enabled=True,
        version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    session.add(duplicate_digest)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    invalid_length = ApiCredential(
        credential_id=f"crd_{'5' * 32}",
        principal_id=_PRINCIPAL_ID,
        label="bad digest",
        token_digest=b"x" * 31,
        enabled=True,
        version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    session.add(invalid_length)
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("tenant_id", "invalid"),
        ("tenant_id", f"tnt_{'A' * 32}"),
        ("slug", "UPPERCASE"),
        ("slug", "---"),
        ("display_name", "   "),
        ("enabled", 2),
    ],
)
def test_tenant_database_constraints_reject_raw_invalid_values(
    session: Session,
    column: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "tenant_id": f"tnt_{'8' * 32}",
        "slug": "raw-tenant",
        "display_name": "Raw tenant",
        "enabled": 1,
        "version": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    values[column] = value
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                """
                INSERT INTO tenants
                    (tenant_id, slug, display_name, enabled, version, created_at, updated_at)
                VALUES
                    (:tenant_id, :slug, :display_name, :enabled, :version, :created_at, :updated_at)
                """
            ),
            values,
        )
        session.flush()
    session.rollback()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("principal_id", f"prn_{'A' * 32}"),
        ("handle", "..."),
    ],
)
def test_principal_database_constraints_reject_raw_noncanonical_values(
    session: Session,
    column: str,
    value: object,
) -> None:
    _seed_identity(session)
    values: dict[str, object] = {
        "principal_id": f"prn_{'9' * 32}",
        "tenant_id": _TENANT_ID,
        "handle": "raw-principal",
        "display_name": "Raw principal",
        "kind": PrincipalKind.USER.value,
        "role": PrincipalRole.VIEWER.value,
        "enabled": 1,
        "version": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    values[column] = value
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                """
                INSERT INTO principals
                    (principal_id, tenant_id, handle, display_name, kind, role,
                     enabled, version, created_at, updated_at)
                VALUES
                    (:principal_id, :tenant_id, :handle, :display_name, :kind, :role,
                     :enabled, :version, :created_at, :updated_at)
                """
            ),
            values,
        )
        session.flush()
    session.rollback()


def test_credential_database_constraint_rejects_raw_uppercase_id(
    session: Session,
) -> None:
    _seed_identity(session)
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                """
                INSERT INTO api_credentials
                    (credential_id, principal_id, label, token_digest, enabled, version,
                     created_at, updated_at)
                VALUES
                    (:credential_id, :principal_id, :label, :token_digest, :enabled, :version,
                     :created_at, :updated_at)
                """
            ),
            {
                "credential_id": f"crd_{'A' * 32}",
                "principal_id": _PRINCIPAL_ID,
                "label": "raw credential",
                "token_digest": b"u" * 32,
                "enabled": 1,
                "version": 1,
                "created_at": _NOW,
                "updated_at": _NOW,
            },
        )
        session.flush()
    session.rollback()


def test_persisted_identity_audit_events_reject_orm_mutation_and_deletion(
    session: Session,
) -> None:
    _seed_identity(session)
    session.commit()
    event_record = session.scalars(
        select(IdentityAuditEvent).order_by(IdentityAuditEvent.event_id)
    ).first()
    assert event_record is not None
    original_type = event_record.event_type

    event_record.event_type = "tenant.disabled"
    with pytest.raises(RuntimeError, match="append-only"):
        session.flush()
    session.rollback()
    reloaded = session.get(IdentityAuditEvent, event_record.event_id)
    assert reloaded is not None and reloaded.event_type == original_type

    session.delete(reloaded)
    with pytest.raises(RuntimeError, match="append-only"):
        session.flush()
    session.rollback()


def test_create_all_sqlite_audit_triggers_reject_direct_update_and_delete(
    session: Session,
) -> None:
    _seed_identity(session)
    event_id = session.scalar(
        select(IdentityAuditEvent.event_id).order_by(IdentityAuditEvent.event_id)
    )
    assert event_id is not None

    with pytest.raises(IntegrityError, match="append-only"):
        session.execute(
            text("UPDATE identity_audit_events SET event_type = :event_type WHERE event_id = :id"),
            {"event_type": "tenant.disabled", "id": event_id},
        )
    session.rollback()

    with pytest.raises(IntegrityError, match="append-only"):
        session.execute(
            text("DELETE FROM identity_audit_events WHERE event_id = :id"),
            {"id": event_id},
        )
    session.rollback()


def test_lifecycle_compare_and_swap_writes_only_successful_audits(session: Session) -> None:
    service, issued = _seed_identity(session)

    assert service.set_tenant_enabled(_TENANT_ID, enabled=False, now=_NOW) is True
    assert service.set_tenant_enabled(_TENANT_ID, enabled=False, now=_NOW) is False
    assert service.set_tenant_enabled(_TENANT_ID, enabled=True, now=_NOW) is True
    assert service.set_principal_enabled(_PRINCIPAL_ID, enabled=False, now=_NOW) is True
    assert service.set_principal_enabled(_PRINCIPAL_ID, enabled=False, now=_NOW) is False
    assert service.set_principal_enabled(_PRINCIPAL_ID, enabled=True, now=_NOW) is True
    assert service.set_credential_enabled(issued.credential_id, enabled=False, now=_NOW) is True
    assert service.set_credential_enabled(issued.credential_id, enabled=False, now=_NOW) is False
    assert service.set_credential_enabled(issued.credential_id, enabled=True, now=_NOW) is True
    assert service.revoke_credential(issued.credential_id, now=_NOW) is True
    assert service.revoke_credential(issued.credential_id, now=_NOW) is False
    assert service.set_credential_enabled(issued.credential_id, enabled=False, now=_NOW) is False
    session.commit()

    tenant = session.get(Tenant, _TENANT_ID)
    principal = session.get(Principal, _PRINCIPAL_ID)
    credential = session.get(ApiCredential, issued.credential_id)
    assert tenant is not None and principal is not None and credential is not None
    assert tenant.version == 3
    assert principal.version == 3
    assert credential.version == 4
    assert credential.revoked_at is not None
    event_types = session.scalars(
        select(IdentityAuditEvent.event_type).order_by(IdentityAuditEvent.event_id)
    ).all()
    assert event_types == [
        "tenant.created",
        "principal.created",
        "credential.issued",
        "tenant.disabled",
        "tenant.enabled",
        "principal.disabled",
        "principal.enabled",
        "credential.disabled",
        "credential.enabled",
        "credential.revoked",
    ]
    actor_kinds = session.scalars(
        select(IdentityAuditEvent.actor_kind).order_by(IdentityAuditEvent.event_id)
    ).all()
    assert actor_kinds == [IdentityActorKind.OPERATOR_CLI.value] * len(event_types)


def test_principal_actor_is_explicit_and_invalid_actor_shape_does_not_mutate(
    session: Session,
) -> None:
    service, issued = _seed_identity(session)

    with pytest.raises(ValueError, match="requires a principal ID"):
        service.set_credential_enabled(
            issued.credential_id,
            enabled=False,
            actor_kind=IdentityActorKind.PRINCIPAL,
            now=_NOW,
        )
    credential = session.get(ApiCredential, issued.credential_id)
    assert credential is not None and credential.enabled is True

    assert service.set_credential_enabled(
        issued.credential_id,
        enabled=False,
        actor_kind=IdentityActorKind.PRINCIPAL,
        actor_principal_id=_PRINCIPAL_ID,
        now=_NOW,
    )
    session.commit()
    event = session.scalars(
        select(IdentityAuditEvent)
        .where(IdentityAuditEvent.event_type == "credential.disabled")
        .order_by(IdentityAuditEvent.event_id.desc())
    ).first()
    assert event is not None
    assert event.actor_kind == IdentityActorKind.PRINCIPAL.value
    assert event.actor_principal_id == _PRINCIPAL_ID

    session.add(
        IdentityAuditEvent(
            tenant_id=_TENANT_ID,
            principal_id=_PRINCIPAL_ID,
            credential_id=issued.credential_id,
            actor_principal_id=_PRINCIPAL_ID,
            actor_kind=IdentityActorKind.SYSTEM.value,
            event_type="credential.disabled",
            metadata_json={},
            occurred_at=_NOW,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_list_credentials_is_stable_filterable_and_secret_free(session: Session) -> None:
    service, first = _seed_identity(session)
    second = issue_credential(_PEPPER)
    service.issue_credential(
        credential_id=second.credential_id,
        principal_id=_PRINCIPAL_ID,
        token_digest=second.digest,
        label="older credential",
        now=_NOW - timedelta(days=1),
    )
    other_principal_id = f"prn_{'6' * 32}"
    service.create_principal(
        principal_id=other_principal_id,
        tenant_id=_TENANT_ID,
        handle="automation",
        display_name="Automation",
        kind=PrincipalKind.SERVICE,
        role=PrincipalRole.VIEWER,
        now=_NOW,
    )
    third = issue_credential(_PEPPER)
    service.issue_credential(
        credential_id=third.credential_id,
        principal_id=other_principal_id,
        token_digest=third.digest,
        label="other principal",
        now=_NOW + timedelta(days=1),
    )
    session.commit()

    all_metadata = service.list_credentials()
    filtered = service.list_credentials(_PRINCIPAL_ID)

    assert [item.credential_id for item in all_metadata] == [
        second.credential_id,
        first.credential_id,
        third.credential_id,
    ]
    assert [item.credential_id for item in filtered] == [
        second.credential_id,
        first.credential_id,
    ]
    assert all(not hasattr(item, "token_digest") for item in all_metadata)
    tokens = [
        first.token.get_secret_value(),
        second.token.get_secret_value(),
        third.token.get_secret_value(),
    ]
    assert all(token not in repr(all_metadata) for token in tokens)


def test_revocation_cas_allows_only_one_session_to_win(database: Database) -> None:
    seed_session = database.session_factory()
    try:
        _, issued = _seed_identity(seed_session)
    finally:
        seed_session.close()

    first_session = database.session_factory()
    second_session = database.session_factory()
    try:
        first = IdentityService.from_session(first_session)
        second = IdentityService.from_session(second_session)
        assert first.revoke_credential(issued.credential_id, now=_NOW) is True
        first_session.commit()
        assert second.revoke_credential(issued.credential_id, now=_NOW) is False
        second_session.commit()
    finally:
        first_session.close()
        second_session.close()

    verify_session = database.session_factory()
    try:
        revoke_events = verify_session.scalars(
            select(IdentityAuditEvent).where(
                IdentityAuditEvent.credential_id == issued.credential_id,
                IdentityAuditEvent.event_type == "credential.revoked",
            )
        ).all()
        assert len(revoke_events) == 1
    finally:
        verify_session.close()


def test_identity_lifecycle_rolls_back_state_and_audit_together(
    database: Database,
) -> None:
    seed_session = database.session_factory()
    try:
        _, issued = _seed_identity(seed_session)
        original_events = len(seed_session.scalars(select(IdentityAuditEvent)).all())
        service = IdentityService.from_session(seed_session)
        assert service.set_credential_enabled(issued.credential_id, enabled=False, now=_NOW)
        seed_session.rollback()
    finally:
        seed_session.close()

    verify_session = database.session_factory()
    try:
        credential = verify_session.get(ApiCredential, issued.credential_id)
        assert credential is not None and credential.enabled is True
        assert len(verify_session.scalars(select(IdentityAuditEvent)).all()) == original_events
    finally:
        verify_session.close()
