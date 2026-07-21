from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, Table, UniqueConstraint
from sqlalchemy.orm import configure_mappers

from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
)
from app.contracts.queries import QueryActorAuthMode, QueryActorDTO
from app.db.models import AnalysisJob, ApiCredential, QueryLog

_TENANT_ID = f"tnt_{'1' * 32}"
_PRINCIPAL_ID = f"prn_{'2' * 32}"
_CREDENTIAL_ID = f"crd_{'3' * 32}"


def test_query_actor_freezes_verified_principal_without_legacy_unknown_issuance() -> None:
    principal = PrincipalContext(
        tenant_id=_TENANT_ID,
        principal_id=_PRINCIPAL_ID,
        credential_id=_CREDENTIAL_ID,
        kind=PrincipalKind.USER,
        role=PrincipalRole.ANALYST,
        auth_mode=AuthMode.PRINCIPAL,
    )

    actor = QueryActorDTO.from_principal(principal)

    assert actor == QueryActorDTO(
        tenant_id=_TENANT_ID,
        principal_id=_PRINCIPAL_ID,
        credential_id=_CREDENTIAL_ID,
        role=PrincipalRole.ANALYST,
        auth_mode=QueryActorAuthMode.PRINCIPAL,
    )
    assert QueryActorAuthMode.LEGACY_UNKNOWN.value not in {
        mode.value for mode in AuthMode
    }


@pytest.mark.parametrize("auth_mode", [AuthMode.DISABLED, AuthMode.SHARED_KEY])
def test_query_actor_freezes_compatibility_principal(auth_mode: AuthMode) -> None:
    principal = PrincipalContext(
        tenant_id=LEGACY_TENANT_ID,
        principal_id=LEGACY_PRINCIPAL_ID,
        credential_id=None,
        kind=PrincipalKind.SERVICE,
        role=PrincipalRole.TENANT_ADMIN,
        auth_mode=auth_mode,
    )

    actor = QueryActorDTO.from_principal(principal)

    assert actor.auth_mode.value == auth_mode.value
    assert actor.credential_id is None


@pytest.mark.parametrize(
    "payload",
    [
        {
            "tenant_id": _TENANT_ID,
            "principal_id": _PRINCIPAL_ID,
            "credential_id": None,
            "role": PrincipalRole.ANALYST,
            "auth_mode": QueryActorAuthMode.PRINCIPAL,
        },
        {
            "tenant_id": LEGACY_TENANT_ID,
            "principal_id": LEGACY_PRINCIPAL_ID,
            "credential_id": _CREDENTIAL_ID,
            "role": PrincipalRole.TENANT_ADMIN,
            "auth_mode": QueryActorAuthMode.SHARED_KEY,
        },
        {
            "tenant_id": _TENANT_ID,
            "principal_id": _PRINCIPAL_ID,
            "credential_id": None,
            "role": PrincipalRole.ANALYST,
            "auth_mode": QueryActorAuthMode.LEGACY_UNKNOWN,
        },
    ],
    ids=["principal-without-credential", "compatibility-with-credential", "forged-legacy"],
)
def test_query_actor_rejects_ambiguous_identity_shapes(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QueryActorDTO.model_validate(payload)


def test_legacy_unknown_actor_has_one_exact_contract_shape() -> None:
    actor = QueryActorDTO(
        tenant_id=LEGACY_TENANT_ID,
        principal_id=LEGACY_PRINCIPAL_ID,
        credential_id=None,
        role=PrincipalRole.TENANT_ADMIN,
        auth_mode=QueryActorAuthMode.LEGACY_UNKNOWN,
    )

    assert actor.auth_mode is QueryActorAuthMode.LEGACY_UNKNOWN


def test_query_actor_models_declare_relational_audit_proofs() -> None:
    query_table = cast(Table, QueryLog.__table__)
    job_table = cast(Table, AnalysisJob.__table__)
    credential_table = cast(Table, ApiCredential.__table__)

    assert _unique_columns(job_table, "uq_analysis_jobs_job_tenant") == (
        "job_id",
        "tenant_id",
    )
    assert _unique_columns(
        credential_table,
        "uq_api_credentials_credential_principal",
    ) == ("credential_id", "principal_id")
    assert _foreign_key_shape(query_table, "fk_query_logs_job_actor_tenant") == (
        ("job_id", "actor_tenant_id"),
        ("analysis_jobs.job_id", "analysis_jobs.tenant_id"),
        "CASCADE",
    )
    assert _foreign_key_shape(query_table, "fk_query_logs_actor_principal_tenant") == (
        ("actor_principal_id", "actor_tenant_id"),
        ("principals.principal_id", "principals.tenant_id"),
        "RESTRICT",
    )
    assert _foreign_key_shape(query_table, "fk_query_logs_actor_credential_principal") == (
        ("actor_credential_id", "actor_principal_id"),
        ("api_credentials.credential_id", "api_credentials.principal_id"),
        "RESTRICT",
    )
    assert {
        constraint.name
        for constraint in query_table.constraints
        if isinstance(constraint, CheckConstraint)
    } >= {
        "ck_query_logs_actor_role_known",
        "ck_query_logs_actor_auth_mode_known",
        "ck_query_logs_actor_credential_shape",
        "ck_query_logs_compatibility_actor_shape",
    }
    actor_index = next(
        index for index in query_table.indexes if index.name == "ix_query_logs_actor_created"
    )
    assert isinstance(actor_index, Index)
    assert tuple(actor_index.columns.keys()) == (
        "actor_tenant_id",
        "actor_principal_id",
        "created_at",
    )


def test_query_job_mapper_uses_the_stable_job_identifier() -> None:
    configure_mappers()

    assert {column.key for column in QueryLog.job.property.local_columns} == {"job_id"}


def _unique_columns(table: Table, name: str) -> tuple[str, ...]:
    constraint = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint) and constraint.name == name
    )
    return tuple(constraint.columns.keys())


def _foreign_key_shape(
    table: Table,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    constraint = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, ForeignKeyConstraint) and constraint.name == name
    )
    return (
        tuple(constraint.columns.keys()),
        tuple(element.target_fullname for element in constraint.elements),
        constraint.ondelete,
    )
