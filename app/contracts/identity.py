"""Stable identity values shared by authentication and persistence layers."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from app.contracts.common import ContractModel

ENTITY_ID_HEX_LENGTH = 32
TENANT_ID_PREFIX = "tnt"
PRINCIPAL_ID_PREFIX = "prn"
CREDENTIAL_ID_PREFIX = "crd"

LEGACY_TENANT_ID = f"{TENANT_ID_PREFIX}_{'0' * ENTITY_ID_HEX_LENGTH}"
LEGACY_PRINCIPAL_ID = f"{PRINCIPAL_ID_PREFIX}_{'0' * ENTITY_ID_HEX_LENGTH}"

_TENANT_ID_PATTERN = re.compile(rf"\A{TENANT_ID_PREFIX}_[0-9a-f]{{32}}\Z")
_PRINCIPAL_ID_PATTERN = re.compile(rf"\A{PRINCIPAL_ID_PREFIX}_[0-9a-f]{{32}}\Z")
_CREDENTIAL_ID_PATTERN = re.compile(rf"\A{CREDENTIAL_ID_PREFIX}_[0-9a-f]{{32}}\Z")
_SLUG_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_HANDLE_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?\Z")


class AuthMode(StrEnum):
    """Authentication boundary active for one request."""

    DISABLED = "disabled"
    SHARED_KEY = "shared_key"
    PRINCIPAL = "principal"


class PrincipalKind(StrEnum):
    """Whether a principal represents a person or an automation identity."""

    USER = "user"
    SERVICE = "service"


class PrincipalRole(StrEnum):
    """Tenant-scoped authorization role carried by an authenticated principal."""

    TENANT_ADMIN = "tenant_admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


def validate_tenant_id(value: str) -> str:
    """Return a canonical tenant ID or raise a value-redacting error."""

    return _validate_pattern(value, _TENANT_ID_PATTERN, "tenant ID")


def validate_principal_id(value: str) -> str:
    """Return a canonical principal ID or raise a value-redacting error."""

    return _validate_pattern(value, _PRINCIPAL_ID_PATTERN, "principal ID")


def validate_credential_id(value: str) -> str:
    """Return a canonical credential ID or raise a value-redacting error."""

    return _validate_pattern(value, _CREDENTIAL_ID_PATTERN, "credential ID")


def validate_tenant_slug(value: str) -> str:
    """Validate a lowercase, URL-safe tenant slug of at most 63 characters."""

    return _validate_pattern(value, _SLUG_PATTERN, "tenant slug")


def validate_principal_handle(value: str) -> str:
    """Validate a lowercase, log-safe principal handle of at most 64 characters."""

    return _validate_pattern(value, _HANDLE_PATTERN, "principal handle")


class PrincipalContext(ContractModel):
    """Immutable caller identity propagated through one request.

    Principal authentication always carries all three persisted IDs. Compatibility modes may
    omit IDs because they do not represent a revocable persisted credential.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        use_enum_values=False,
        frozen=True,
    )

    tenant_id: str | None = Field(default=None, max_length=36)
    principal_id: str | None = Field(default=None, max_length=36)
    credential_id: str | None = Field(default=None, max_length=36)
    kind: PrincipalKind
    role: PrincipalRole
    auth_mode: AuthMode

    @model_validator(mode="after")
    def validate_identity_shape(self) -> Self:
        if self.tenant_id is not None:
            validate_tenant_id(self.tenant_id)
        if self.principal_id is not None:
            validate_principal_id(self.principal_id)
        if self.credential_id is not None:
            validate_credential_id(self.credential_id)
        if self.auth_mode is AuthMode.PRINCIPAL and (
            self.tenant_id is None or self.principal_id is None or self.credential_id is None
        ):
            raise ValueError(
                "principal authentication requires tenant, principal, and credential IDs"
            )
        if self.auth_mode is not AuthMode.PRINCIPAL and (
            self.tenant_id != LEGACY_TENANT_ID
            or self.principal_id != LEGACY_PRINCIPAL_ID
            or self.credential_id is not None
            or self.kind is not PrincipalKind.SERVICE
            or self.role is not PrincipalRole.TENANT_ADMIN
        ):
            raise ValueError(
                "compatibility authentication requires the fixed legacy principal "
                "without a credential"
            )
        return self


def _validate_pattern(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value
