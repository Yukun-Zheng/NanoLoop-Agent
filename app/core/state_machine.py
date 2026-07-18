"""Explicit state transitions for jobs and immutable analysis runs."""

from app.contracts.enums import JobStatus
from app.core.errors import JobStateConflictError

TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS, JobStatus.FAILED}
)

ACTIVE_RUN_ORDER = (
    JobStatus.QUEUED,
    JobStatus.PREPROCESSING,
    JobStatus.SEGMENTING,
    JobStatus.POSTPROCESSING,
    JobStatus.QUALITY_CHECKING,
    JobStatus.ANALYZING,
    JobStatus.AGGREGATING,
)

ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.CREATED: frozenset({JobStatus.VALIDATING, JobStatus.FAILED}),
    JobStatus.VALIDATING: frozenset({JobStatus.READY_FOR_CONFIGURATION, JobStatus.FAILED}),
    JobStatus.READY_FOR_CONFIGURATION: frozenset({JobStatus.QUEUED, JobStatus.FAILED}),
    JobStatus.QUEUED: frozenset({JobStatus.PREPROCESSING, JobStatus.FAILED}),
    JobStatus.PREPROCESSING: frozenset({JobStatus.SEGMENTING, JobStatus.FAILED}),
    JobStatus.SEGMENTING: frozenset({JobStatus.POSTPROCESSING, JobStatus.FAILED}),
    JobStatus.POSTPROCESSING: frozenset({JobStatus.QUALITY_CHECKING, JobStatus.FAILED}),
    JobStatus.QUALITY_CHECKING: frozenset({JobStatus.ANALYZING, JobStatus.FAILED}),
    JobStatus.ANALYZING: frozenset({JobStatus.AGGREGATING, JobStatus.FAILED}),
    JobStatus.AGGREGATING: frozenset(
        {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS, JobStatus.FAILED}
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.COMPLETED_WITH_WARNINGS: frozenset(),
    JobStatus.FAILED: frozenset(),
}


def ensure_transition(current: JobStatus, target: JobStatus) -> None:
    """Raise a stable domain error when a state transition is not allowed."""

    if target not in ALLOWED_TRANSITIONS[current]:
        raise JobStateConflictError(
            details={"current_status": current.value, "target_status": target.value}
        )


def ensure_job_transition(current: JobStatus, target: JobStatus, *, starts_new_runs: bool) -> None:
    """Validate a job transition, including explicit requeue after immutable prior runs."""

    if starts_new_runs and current in TERMINAL_STATUSES and target == JobStatus.QUEUED:
        return
    ensure_transition(current, target)


def aggregate_job_status(run_statuses: list[JobStatus]) -> JobStatus:
    """Derive a job status from its runs using the ADR-0002 partial-failure policy."""

    if not run_statuses:
        return JobStatus.READY_FOR_CONFIGURATION

    active = [status for status in run_statuses if status not in TERMINAL_STATUSES]
    if active:
        rank = {status: index for index, status in enumerate(ACTIVE_RUN_ORDER)}
        # CREATED/VALIDATING/READY are earlier than a queued execution and can occur briefly.
        return min(active, key=lambda status: rank.get(status, -1))

    successful = sum(status == JobStatus.COMPLETED for status in run_statuses)
    warned = sum(status == JobStatus.COMPLETED_WITH_WARNINGS for status in run_statuses)
    failed = sum(status == JobStatus.FAILED for status in run_statuses)
    if failed == len(run_statuses):
        return JobStatus.FAILED
    if warned or failed:
        return JobStatus.COMPLETED_WITH_WARNINGS
    if successful == len(run_statuses):
        return JobStatus.COMPLETED
    raise JobStateConflictError(details={"run_statuses": [status.value for status in run_statuses]})
