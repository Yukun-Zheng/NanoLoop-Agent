"""Helpers that make the persist-before-dispatch ordering explicit."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from app.orchestration.dispatcher import (
    DispatcherNotRunningError,
    DispatcherQueueFullError,
    TaskDispatcher,
)


@dataclass(frozen=True, slots=True)
class SubmissionBatch:
    submitted_run_ids: tuple[str, ...]
    duplicate_run_ids: tuple[str, ...]
    deferred_run_ids: tuple[str, ...]

    @property
    def all_dispatched(self) -> bool:
        return not self.deferred_run_ids


def submit_committed_run_ids(
    dispatcher: TaskDispatcher,
    run_ids: Iterable[str],
) -> SubmissionBatch:
    """Submit IDs that are already durable without blocking for queue space.

    A full/stopped dispatcher leaves the durable run in ``QUEUED`` for startup recovery or an
    explicit retry. This is preferable to rolling back scientific inputs after commit.
    """

    unique, repeated = _unique_run_ids(run_ids)
    submitted: list[str] = []
    duplicates = list(repeated)
    deferred: list[str] = []
    for run_id in unique:
        try:
            accepted = dispatcher.submit(run_id)
        except (DispatcherQueueFullError, DispatcherNotRunningError):
            deferred.append(run_id)
        else:
            if accepted:
                submitted.append(run_id)
            else:
                duplicates.append(run_id)
    return SubmissionBatch(
        submitted_run_ids=tuple(submitted),
        duplicate_run_ids=tuple(duplicates),
        deferred_run_ids=tuple(deferred),
    )


def commit_then_submit(
    commit: Callable[[], None],
    dispatcher: TaskDispatcher,
    run_ids: Iterable[str],
) -> SubmissionBatch:
    """Commit a transaction, then enqueue only its durable run IDs."""

    materialized = tuple(run_ids)
    # Validate before committing, while preserving the required commit-before-submit boundary.
    _unique_run_ids(materialized)
    commit()
    return submit_committed_run_ids(dispatcher, materialized)


def _unique_run_ids(run_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    unique: list[str] = []
    repeated: list[str] = []
    seen: set[str] = set()
    for raw_run_id in run_ids:
        if not isinstance(raw_run_id, str) or not raw_run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        run_id = raw_run_id.strip()
        if run_id in seen:
            repeated.append(run_id)
        else:
            seen.add(run_id)
            unique.append(run_id)
    return unique, repeated
