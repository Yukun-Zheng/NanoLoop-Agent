from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.analysis.authorization import require_create, require_mutation, require_read
from app.contracts.analyses import AnalysisJobDTO
from app.contracts.enums import JobStatus
from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
)
from app.contracts.repositories import AnalysisResourceScope
from app.core.errors import ForbiddenError, ResourceNotFoundError
from app.core.identity import legacy_principal_context

_TENANT_A = f"tnt_{'a' * 32}"
_TENANT_B = f"tnt_{'b' * 32}"
_OWNER = f"prn_{'c' * 32}"
_PEER = f"prn_{'d' * 32}"


def _principal(
    role: PrincipalRole,
    *,
    tenant_id: str = _TENANT_A,
    principal_id: str = _OWNER,
) -> PrincipalContext:
    return PrincipalContext(
        tenant_id=tenant_id,
        principal_id=principal_id,
        credential_id=f"crd_{principal_id.removeprefix('prn_')}",
        kind=PrincipalKind.USER,
        role=role,
        auth_mode=AuthMode.PRINCIPAL,
    )


def _scope(
    *,
    tenant_id: str = _TENANT_A,
    owner_principal_id: str = _OWNER,
) -> AnalysisResourceScope:
    now = datetime(2026, 7, 18, tzinfo=UTC)
    return AnalysisResourceScope(
        job=AnalysisJobDTO(
            job_id="job_authorized",
            name="authorization fixture",
            status=JobStatus.READY_FOR_CONFIGURATION,
            created_at=now,
            updated_at=now,
        ),
        tenant_id=tenant_id,
        owner_principal_id=owner_principal_id,
    )


@pytest.mark.parametrize("role", [PrincipalRole.TENANT_ADMIN, PrincipalRole.ANALYST])
def test_create_allows_tenant_admin_and_analyst(role: PrincipalRole) -> None:
    require_create(_principal(role))


def test_create_rejects_viewer_with_fixed_forbidden_error() -> None:
    with pytest.raises(ForbiddenError) as error:
        require_create(_principal(PrincipalRole.VIEWER))

    assert error.value.code == "FORBIDDEN"
    assert error.value.status_code == 403
    assert error.value.details == {}


@pytest.mark.parametrize(
    "role",
    [PrincipalRole.TENANT_ADMIN, PrincipalRole.ANALYST, PrincipalRole.VIEWER],
)
def test_read_allows_every_role_inside_the_tenant(role: PrincipalRole) -> None:
    require_read(_principal(role), _scope())


@pytest.mark.parametrize(
    "role",
    [PrincipalRole.TENANT_ADMIN, PrincipalRole.ANALYST, PrincipalRole.VIEWER],
)
def test_tenant_mismatch_is_not_found_before_role_checks(role: PrincipalRole) -> None:
    principal = _principal(role, tenant_id=_TENANT_B)

    with pytest.raises(ResourceNotFoundError) as error:
        require_mutation(principal, _scope())

    assert error.value.status_code == 404
    assert error.value.details == {"resource": "job", "job_id": "job_authorized"}


def test_mutation_allows_tenant_admin_and_only_the_owning_analyst() -> None:
    scope = _scope()
    require_mutation(_principal(PrincipalRole.TENANT_ADMIN, principal_id=_PEER), scope)
    require_mutation(_principal(PrincipalRole.ANALYST), scope)

    with pytest.raises(ForbiddenError):
        require_mutation(
            _principal(PrincipalRole.ANALYST, principal_id=_PEER),
            scope,
        )
    with pytest.raises(ForbiddenError):
        require_mutation(_principal(PrincipalRole.VIEWER), scope)


def test_legacy_admin_uses_the_same_policy_without_a_bypass() -> None:
    now = datetime(2026, 7, 18, tzinfo=UTC)
    legacy_scope = AnalysisResourceScope(
        job=AnalysisJobDTO(
            job_id="job_legacy",
            name="legacy",
            status=JobStatus.READY_FOR_CONFIGURATION,
            created_at=now,
            updated_at=now,
        ),
        tenant_id=LEGACY_TENANT_ID,
        owner_principal_id=LEGACY_PRINCIPAL_ID,
    )

    require_mutation(legacy_principal_context(AuthMode.DISABLED), legacy_scope)
    require_mutation(legacy_principal_context(AuthMode.SHARED_KEY), legacy_scope)
