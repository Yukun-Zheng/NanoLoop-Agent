"""Transactional persistence for tenant principals and revocable API credentials."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hmac import compare_digest
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from app.contracts.identity import (
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
    validate_credential_id,
    validate_principal_handle,
    validate_principal_id,
    validate_tenant_id,
    validate_tenant_slug,
)
from app.db.models import (
    ApiCredential,
    IdentityAuditEvent,
    Principal,
    Tenant,
)

_DIGEST_BYTES = 32
_DUMMY_DIGEST = bytes(_DIGEST_BYTES)


class CredentialAuthenticationStatus(StrEnum):
    """Internal authentication outcome; HTTP callers must keep failures indistinguishable."""

    AUTHENTICATED = "authenticated"
    UNKNOWN = "unknown"
    BAD_DIGEST = "bad_digest"
    REVOKED = "revoked"
    DISABLED = "disabled"
    EXPIRED = "expired"
    PRINCIPAL_DISABLED = "principal_disabled"
    TENANT_DISABLED = "tenant_disabled"


class IdentityActorKind(StrEnum):
    """Bounded provenance for an identity lifecycle audit event."""

    OPERATOR_CLI = "operator_cli"
    PRINCIPAL = "principal"
    MIGRATION = "migration"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class AuthenticationResult:
    status: CredentialAuthenticationStatus
    principal: PrincipalContext | None = None

    @property
    def authenticated(self) -> bool:
        return self.status is CredentialAuthenticationStatus.AUTHENTICATED


@dataclass(frozen=True, slots=True)
class TenantMetadata:
    tenant_id: str
    slug: str
    display_name: str
    enabled: bool
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PrincipalMetadata:
    principal_id: str
    tenant_id: str
    handle: str
    display_name: str
    kind: PrincipalKind
    role: PrincipalRole
    enabled: bool
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CredentialMetadata:
    """Secret-free credential metadata safe for operator responses and logs."""

    credential_id: str
    principal_id: str
    label: str
    enabled: bool
    expires_at: datetime | None
    revoked_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class _AuthenticationRecord:
    credential_id: str
    principal_id: str
    tenant_id: str
    token_digest: bytes
    credential_enabled: bool
    expires_at: datetime | None
    revoked_at: datetime | None
    principal_enabled: bool
    tenant_enabled: bool
    kind: PrincipalKind
    role: PrincipalRole

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(credential_id={self.credential_id!r}, "
            "token_digest=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class _AuditActor:
    kind: IdentityActorKind
    principal_id: str | None


class IdentityRepository:
    """SQLAlchemy identity operations without committing the caller-owned transaction."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_authentication(self, credential_id: str) -> _AuthenticationRecord | None:
        """Resolve one credential and its principal/tenant state with one PK lookup."""

        row = self.session.execute(
            select(ApiCredential, Principal, Tenant)
            .join(Principal, Principal.principal_id == ApiCredential.principal_id)
            .join(Tenant, Tenant.tenant_id == Principal.tenant_id)
            .where(ApiCredential.credential_id == credential_id)
        ).one_or_none()
        if row is None:
            return None
        credential, principal, tenant = row
        return _AuthenticationRecord(
            credential_id=credential.credential_id,
            principal_id=principal.principal_id,
            tenant_id=tenant.tenant_id,
            token_digest=bytes(credential.token_digest),
            credential_enabled=credential.enabled,
            expires_at=credential.expires_at,
            revoked_at=credential.revoked_at,
            principal_enabled=principal.enabled,
            tenant_enabled=tenant.enabled,
            kind=PrincipalKind(principal.kind),
            role=PrincipalRole(principal.role),
        )

    def add_tenant(self, record: Tenant) -> None:
        self.session.add(record)

    def add_principal(self, record: Principal) -> None:
        self.session.add(record)

    def add_credential(self, record: ApiCredential) -> None:
        self.session.add(record)

    def add_audit_event(self, record: IdentityAuditEvent) -> None:
        self.session.add(record)

    def list_credentials(self, principal_id: str | None = None) -> list[ApiCredential]:
        statement = select(ApiCredential)
        if principal_id is not None:
            statement = statement.where(ApiCredential.principal_id == principal_id)
        return list(
            self.session.scalars(
                statement.order_by(ApiCredential.created_at, ApiCredential.credential_id)
            ).all()
        )

    def tenant_state_cas(self, tenant_id: str, *, enabled: bool, now: datetime) -> bool:
        result = _cursor_result(
            self.session.execute(
                update(Tenant)
                .where(Tenant.tenant_id == tenant_id, Tenant.enabled.is_(not enabled))
                .values(
                    enabled=enabled,
                    version=Tenant.version + 1,
                    updated_at=now,
                )
            )
        )
        return result.rowcount == 1

    def principal_tenant_id(self, principal_id: str) -> str | None:
        return self.session.scalar(
            select(Principal.tenant_id).where(Principal.principal_id == principal_id)
        )

    def principal_state_cas(self, principal_id: str, *, enabled: bool, now: datetime) -> bool:
        result = _cursor_result(
            self.session.execute(
                update(Principal)
                .where(
                    Principal.principal_id == principal_id,
                    Principal.enabled.is_(not enabled),
                )
                .values(
                    enabled=enabled,
                    version=Principal.version + 1,
                    updated_at=now,
                )
            )
        )
        return result.rowcount == 1

    def credential_scope(self, credential_id: str) -> tuple[str, str] | None:
        row = self.session.execute(
            select(Principal.tenant_id, ApiCredential.principal_id)
            .join(ApiCredential, ApiCredential.principal_id == Principal.principal_id)
            .where(ApiCredential.credential_id == credential_id)
        ).one_or_none()
        return None if row is None else (row.tenant_id, row.principal_id)

    def credential_state_cas(
        self,
        credential_id: str,
        *,
        enabled: bool,
        now: datetime,
    ) -> bool:
        result = _cursor_result(
            self.session.execute(
                update(ApiCredential)
                .where(
                    ApiCredential.credential_id == credential_id,
                    ApiCredential.enabled.is_(not enabled),
                    ApiCredential.revoked_at.is_(None),
                )
                .values(
                    enabled=enabled,
                    version=ApiCredential.version + 1,
                    updated_at=now,
                )
            )
        )
        return result.rowcount == 1

    def credential_revoke_cas(self, credential_id: str, *, now: datetime) -> bool:
        result = _cursor_result(
            self.session.execute(
                update(ApiCredential)
                .where(
                    ApiCredential.credential_id == credential_id,
                    ApiCredential.revoked_at.is_(None),
                )
                .values(
                    revoked_at=now,
                    version=ApiCredential.version + 1,
                    updated_at=now,
                )
            )
        )
        return result.rowcount == 1

    def flush(self) -> None:
        self.session.flush()


class IdentityService:
    """Identity lifecycle service whose caller owns commit or rollback."""

    def __init__(self, repository: IdentityRepository) -> None:
        self.repository = repository

    @classmethod
    def from_session(cls, session: Session) -> IdentityService:
        return cls(IdentityRepository(session))

    def authenticate(
        self,
        credential_id: str,
        candidate_digest: bytes,
        *,
        now: datetime,
    ) -> AuthenticationResult:
        """Authenticate without writes; every credential lookup is one indexed query."""

        record = self.repository.get_for_authentication(credential_id)
        digest_is_valid = (
            isinstance(candidate_digest, bytes) and len(candidate_digest) == _DIGEST_BYTES
        )
        candidate = candidate_digest if digest_is_valid else _DUMMY_DIGEST
        expected = record.token_digest if record is not None else _DUMMY_DIGEST
        digest_matches = compare_digest(candidate, expected)
        if record is None:
            return AuthenticationResult(CredentialAuthenticationStatus.UNKNOWN)
        if not digest_is_valid or not digest_matches:
            return AuthenticationResult(CredentialAuthenticationStatus.BAD_DIGEST)
        if record.revoked_at is not None:
            return AuthenticationResult(CredentialAuthenticationStatus.REVOKED)
        if not record.credential_enabled:
            return AuthenticationResult(CredentialAuthenticationStatus.DISABLED)
        checked_at = _utc(now)
        if record.expires_at is not None and checked_at >= _utc(record.expires_at):
            return AuthenticationResult(CredentialAuthenticationStatus.EXPIRED)
        if not record.principal_enabled:
            return AuthenticationResult(CredentialAuthenticationStatus.PRINCIPAL_DISABLED)
        if not record.tenant_enabled:
            return AuthenticationResult(CredentialAuthenticationStatus.TENANT_DISABLED)
        return AuthenticationResult(
            CredentialAuthenticationStatus.AUTHENTICATED,
            principal=PrincipalContext(
                tenant_id=record.tenant_id,
                principal_id=record.principal_id,
                credential_id=record.credential_id,
                kind=record.kind,
                role=record.role,
                auth_mode=AuthMode.PRINCIPAL,
            ),
        )

    def create_tenant(
        self,
        *,
        tenant_id: str,
        slug: str,
        display_name: str,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> TenantMetadata:
        created_at = _utc(now)
        record = Tenant(
            tenant_id=validate_tenant_id(tenant_id),
            slug=validate_tenant_slug(slug),
            display_name=_required_text(display_name, field="tenant display name", limit=255),
            enabled=True,
            version=1,
            created_at=created_at,
            updated_at=created_at,
        )
        actor = _audit_actor(actor_kind, actor_principal_id)
        self.repository.add_tenant(record)
        self.repository.flush()
        self._audit(
            tenant_id=record.tenant_id,
            principal_id=None,
            credential_id=None,
            actor=actor,
            event_type="tenant.created",
            metadata={},
            now=created_at,
        )
        self.repository.flush()
        return _tenant_metadata(record)

    def create_principal(
        self,
        *,
        principal_id: str,
        tenant_id: str,
        handle: str,
        display_name: str,
        kind: PrincipalKind,
        role: PrincipalRole,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> PrincipalMetadata:
        created_at = _utc(now)
        record = Principal(
            principal_id=validate_principal_id(principal_id),
            tenant_id=validate_tenant_id(tenant_id),
            handle=validate_principal_handle(handle),
            display_name=_required_text(display_name, field="principal display name", limit=255),
            kind=PrincipalKind(kind).value,
            role=PrincipalRole(role).value,
            enabled=True,
            version=1,
            created_at=created_at,
            updated_at=created_at,
        )
        actor = _audit_actor(actor_kind, actor_principal_id)
        self.repository.add_principal(record)
        self.repository.flush()
        self._audit(
            tenant_id=record.tenant_id,
            principal_id=record.principal_id,
            credential_id=None,
            actor=actor,
            event_type="principal.created",
            metadata={},
            now=created_at,
        )
        self.repository.flush()
        return _principal_metadata(record)

    def issue_credential(
        self,
        *,
        credential_id: str,
        principal_id: str,
        token_digest: bytes,
        label: str,
        expires_at: datetime | None = None,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> CredentialMetadata:
        if not isinstance(token_digest, bytes) or len(token_digest) != _DIGEST_BYTES:
            raise ValueError("credential digest must contain exactly 32 bytes")
        canonical_principal_id = validate_principal_id(principal_id)
        tenant_id = self.repository.principal_tenant_id(canonical_principal_id)
        if tenant_id is None:
            raise ValueError("credential principal does not exist")
        created_at = _utc(now)
        normalized_expiry = _utc(expires_at) if expires_at is not None else None
        record = ApiCredential(
            credential_id=validate_credential_id(credential_id),
            principal_id=canonical_principal_id,
            label=_required_text(label, field="credential label", limit=120),
            token_digest=bytes(token_digest),
            enabled=True,
            expires_at=normalized_expiry,
            revoked_at=None,
            version=1,
            created_at=created_at,
            updated_at=created_at,
        )
        actor = _audit_actor(actor_kind, actor_principal_id)
        self.repository.add_credential(record)
        self.repository.flush()
        self._audit(
            tenant_id=tenant_id,
            principal_id=canonical_principal_id,
            credential_id=record.credential_id,
            actor=actor,
            event_type="credential.issued",
            metadata={"has_expiry": normalized_expiry is not None},
            now=created_at,
        )
        self.repository.flush()
        return _credential_metadata(record)

    def set_tenant_enabled(
        self,
        tenant_id: str,
        *,
        enabled: bool,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> bool:
        canonical_tenant_id = validate_tenant_id(tenant_id)
        actor = _audit_actor(actor_kind, actor_principal_id)
        changed_at = _utc(now)
        changed = self.repository.tenant_state_cas(
            canonical_tenant_id, enabled=enabled, now=changed_at
        )
        if changed:
            self._audit(
                tenant_id=canonical_tenant_id,
                principal_id=None,
                credential_id=None,
                actor=actor,
                event_type=f"tenant.{'enabled' if enabled else 'disabled'}",
                metadata={},
                now=changed_at,
            )
            self.repository.flush()
        return changed

    def set_principal_enabled(
        self,
        principal_id: str,
        *,
        enabled: bool,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> bool:
        canonical_principal_id = validate_principal_id(principal_id)
        actor = _audit_actor(actor_kind, actor_principal_id)
        tenant_id = self.repository.principal_tenant_id(canonical_principal_id)
        if tenant_id is None:
            return False
        changed_at = _utc(now)
        changed = self.repository.principal_state_cas(
            canonical_principal_id, enabled=enabled, now=changed_at
        )
        if changed:
            self._audit(
                tenant_id=tenant_id,
                principal_id=canonical_principal_id,
                credential_id=None,
                actor=actor,
                event_type=f"principal.{'enabled' if enabled else 'disabled'}",
                metadata={},
                now=changed_at,
            )
            self.repository.flush()
        return changed

    def set_credential_enabled(
        self,
        credential_id: str,
        *,
        enabled: bool,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> bool:
        canonical_credential_id = validate_credential_id(credential_id)
        actor = _audit_actor(actor_kind, actor_principal_id)
        scope = self.repository.credential_scope(canonical_credential_id)
        if scope is None:
            return False
        tenant_id, principal_id = scope
        changed_at = _utc(now)
        changed = self.repository.credential_state_cas(
            canonical_credential_id, enabled=enabled, now=changed_at
        )
        if changed:
            self._audit(
                tenant_id=tenant_id,
                principal_id=principal_id,
                credential_id=canonical_credential_id,
                actor=actor,
                event_type=f"credential.{'enabled' if enabled else 'disabled'}",
                metadata={},
                now=changed_at,
            )
            self.repository.flush()
        return changed

    def revoke_credential(
        self,
        credential_id: str,
        *,
        actor_kind: IdentityActorKind = IdentityActorKind.OPERATOR_CLI,
        actor_principal_id: str | None = None,
        now: datetime,
    ) -> bool:
        canonical_credential_id = validate_credential_id(credential_id)
        actor = _audit_actor(actor_kind, actor_principal_id)
        scope = self.repository.credential_scope(canonical_credential_id)
        if scope is None:
            return False
        tenant_id, principal_id = scope
        revoked_at = _utc(now)
        changed = self.repository.credential_revoke_cas(
            canonical_credential_id, now=revoked_at
        )
        if changed:
            self._audit(
                tenant_id=tenant_id,
                principal_id=principal_id,
                credential_id=canonical_credential_id,
                actor=actor,
                event_type="credential.revoked",
                metadata={},
                now=revoked_at,
            )
            self.repository.flush()
        return changed

    def list_credentials(
        self,
        principal_id: str | None = None,
    ) -> list[CredentialMetadata]:
        """Return stable, secret-free credential metadata for operator tooling."""

        canonical_principal_id = (
            validate_principal_id(principal_id) if principal_id is not None else None
        )
        return [
            _credential_metadata(record)
            for record in self.repository.list_credentials(canonical_principal_id)
        ]

    def _audit(
        self,
        *,
        tenant_id: str,
        principal_id: str | None,
        credential_id: str | None,
        actor: _AuditActor,
        event_type: str,
        metadata: dict[str, Any],
        now: datetime,
    ) -> None:
        self.repository.add_audit_event(
            IdentityAuditEvent(
                tenant_id=tenant_id,
                principal_id=principal_id,
                credential_id=credential_id,
                actor_principal_id=actor.principal_id,
                actor_kind=actor.kind.value,
                event_type=event_type,
                metadata_json=metadata,
                occurred_at=now,
            )
        )


def _cursor_result(result: object) -> CursorResult[Any]:
    return cast(CursorResult[Any], result)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _required_text(value: str, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > limit:
        raise ValueError(f"invalid {field}")
    return normalized


def _audit_actor(
    kind: IdentityActorKind,
    principal_id: str | None,
) -> _AuditActor:
    canonical_kind = IdentityActorKind(kind)
    canonical_principal_id = (
        validate_principal_id(principal_id) if principal_id is not None else None
    )
    if canonical_kind is IdentityActorKind.PRINCIPAL and canonical_principal_id is None:
        raise ValueError("principal audit actor requires a principal ID")
    if canonical_kind is not IdentityActorKind.PRINCIPAL and canonical_principal_id is not None:
        raise ValueError("only a principal audit actor may carry a principal ID")
    return _AuditActor(kind=canonical_kind, principal_id=canonical_principal_id)


def _tenant_metadata(record: Tenant) -> TenantMetadata:
    return TenantMetadata(
        tenant_id=record.tenant_id,
        slug=record.slug,
        display_name=record.display_name,
        enabled=record.enabled,
        version=record.version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


def _principal_metadata(record: Principal) -> PrincipalMetadata:
    return PrincipalMetadata(
        principal_id=record.principal_id,
        tenant_id=record.tenant_id,
        handle=record.handle,
        display_name=record.display_name,
        kind=PrincipalKind(record.kind),
        role=PrincipalRole(record.role),
        enabled=record.enabled,
        version=record.version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )


def _credential_metadata(record: ApiCredential) -> CredentialMetadata:
    return CredentialMetadata(
        credential_id=record.credential_id,
        principal_id=record.principal_id,
        label=record.label,
        enabled=record.enabled,
        expires_at=_utc(record.expires_at) if record.expires_at is not None else None,
        revoked_at=_utc(record.revoked_at) if record.revoked_at is not None else None,
        version=record.version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
    )
