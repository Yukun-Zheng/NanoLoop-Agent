"""Pure tenant and role policy for analysis aggregates."""

from __future__ import annotations

from app.contracts.identity import PrincipalContext, PrincipalRole
from app.contracts.repositories import AnalysisResourceScope
from app.core.errors import ForbiddenError, ResourceNotFoundError

_CREATOR_ROLES = frozenset({PrincipalRole.TENANT_ADMIN, PrincipalRole.ANALYST})
_READER_ROLES = frozenset(
    {PrincipalRole.TENANT_ADMIN, PrincipalRole.ANALYST, PrincipalRole.VIEWER}
)


def require_create(principal: PrincipalContext) -> None:
    """Allow tenant administrators and analysts to create owned analyses."""

    if principal.role not in _CREATOR_ROLES:
        raise ForbiddenError()


def require_read(principal: PrincipalContext, scope: AnalysisResourceScope) -> None:
    """Allow every role to read an analysis only inside its authenticated tenant."""

    if principal.tenant_id != scope.tenant_id:
        raise _not_found(scope)
    if principal.role not in _READER_ROLES:
        raise ForbiddenError()


def require_mutation(principal: PrincipalContext, scope: AnalysisResourceScope) -> None:
    """Allow tenant admins, or the owning analyst, to mutate one analysis."""

    require_read(principal, scope)
    if principal.role is PrincipalRole.TENANT_ADMIN:
        return
    if (
        principal.role is PrincipalRole.ANALYST
        and principal.principal_id == scope.owner_principal_id
    ):
        return
    raise ForbiddenError()


def _not_found(scope: AnalysisResourceScope) -> ResourceNotFoundError:
    return ResourceNotFoundError(
        details={"resource": "job", "job_id": scope.job.job_id}
    )
