import pytest

from app.contracts.enums import JobStatus
from app.core.errors import JobStateConflictError
from app.core.state_machine import aggregate_job_status, ensure_transition


def test_run_transition_is_forward_only() -> None:
    ensure_transition(JobStatus.CREATED, JobStatus.VALIDATING)
    with pytest.raises(JobStateConflictError):
        ensure_transition(JobStatus.CREATED, JobStatus.SEGMENTING)


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], JobStatus.READY_FOR_CONFIGURATION),
        ([JobStatus.SEGMENTING, JobStatus.QUEUED], JobStatus.QUEUED),
        ([JobStatus.COMPLETED, JobStatus.COMPLETED], JobStatus.COMPLETED),
        (
            [JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS],
            JobStatus.COMPLETED_WITH_WARNINGS,
        ),
        ([JobStatus.COMPLETED, JobStatus.FAILED], JobStatus.COMPLETED_WITH_WARNINGS),
        ([JobStatus.FAILED, JobStatus.FAILED], JobStatus.FAILED),
    ],
)
def test_job_status_aggregation(
    statuses: list[JobStatus], expected: JobStatus
) -> None:
    assert aggregate_job_status(statuses) == expected
