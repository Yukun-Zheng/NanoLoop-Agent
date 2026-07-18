"""Startup recovery policy for durable segmentation runs."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.contracts.enums import JobStatus
from app.orchestration.dispatcher import (
    DispatcherNotRunningError,
    DispatcherQueueFullError,
    TaskDispatcher,
)

STALE_ACTIVE_STATUSES = (
    JobStatus.PREPROCESSING,
    JobStatus.SEGMENTING,
    JobStatus.POSTPROCESSING,
    JobStatus.QUALITY_CHECKING,
    JobStatus.ANALYZING,
    JobStatus.AGGREGATING,
)
RECOVERY_STATUSES = (JobStatus.QUEUED, *STALE_ACTIVE_STATUSES)
RESTART_ERROR_CODE = "INTERRUPTED_BY_RESTART"


class StaleRunPolicy(StrEnum):
    FAIL = "fail"
    REQUEUE = "requeue"


class RecoveryRequeueBlockedError(RuntimeError):
    """A stale run was failed because its inputs cannot be cloned safely."""

    def __init__(self, run_id: str, reason: str) -> None:
        super().__init__(reason)
        self.run_id = run_id
        self.reason = reason


@dataclass(frozen=True, slots=True)
class RecoverableRun:
    run_id: str
    status: JobStatus


class RunRecoveryStore(Protocol):
    """Transactional persistence seam used only during process startup.

    Implementations must commit each mutation before returning. ``requeue`` may return a replacement
    run ID when immutable-run policy requires failing the stale row and cloning its configuration,
    or raise :class:`RecoveryRequeueBlockedError` *after* committing a safe terminal failure when
    an external input cannot be cloned reproducibly.
    """

    def list_by_status(self, statuses: tuple[JobStatus, ...]) -> Iterable[RecoverableRun]: ...

    def mark_failed(
        self,
        run_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> None: ...

    def requeue(self, run: RecoverableRun) -> str: ...


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    queued_run_ids: tuple[str, ...]
    stale_run_ids: tuple[str, ...]
    submitted_run_ids: tuple[str, ...]
    duplicate_run_ids: tuple[str, ...]
    failed_stale_run_ids: tuple[str, ...]
    requeued_run_ids: tuple[str, ...]
    deferred_run_ids: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]

    @property
    def requires_attention(self) -> bool:
        return bool(self.deferred_run_ids or self.errors)


class StartupRecovery:
    """Reconcile durable queued/in-flight rows after an unclean process exit."""

    def __init__(
        self,
        store: RunRecoveryStore,
        dispatcher: TaskDispatcher,
        *,
        stale_policy: StaleRunPolicy = StaleRunPolicy.FAIL,
        start_dispatcher: bool = True,
        restart_error_code: str = RESTART_ERROR_CODE,
    ) -> None:
        self.store = store
        self.dispatcher = dispatcher
        self.stale_policy = stale_policy
        self.start_dispatcher = start_dispatcher
        self.restart_error_code = restart_error_code

    def recover(self) -> RecoveryReport:
        if self.start_dispatcher:
            self.dispatcher.start()
        rows = self._deduplicate(self.store.list_by_status(RECOVERY_STATUSES))
        queued = [row.run_id for row in rows if row.status == JobStatus.QUEUED]
        stale = [row for row in rows if row.status in STALE_ACTIVE_STATUSES]
        submitted: list[str] = []
        duplicates: list[str] = []
        failed_stale: list[str] = []
        requeued: list[str] = []
        deferred: list[str] = []
        errors: list[tuple[str, str]] = []

        for run_id in queued:
            self._try_submit(run_id, submitted, duplicates, deferred, errors)

        for run in stale:
            if self.stale_policy == StaleRunPolicy.FAIL:
                try:
                    self.store.mark_failed(
                        run.run_id,
                        error_code=self.restart_error_code,
                        error_message=(
                            f"Run was left in {run.status.value} when the process restarted"
                        ),
                    )
                except Exception as exc:
                    errors.append((run.run_id, f"mark_failed: {type(exc).__name__}: {exc}"))
                else:
                    failed_stale.append(run.run_id)
                continue

            try:
                queued_run_id = self.store.requeue(run).strip()
                if not queued_run_id:
                    raise ValueError("requeue returned an empty run_id")
            except RecoveryRequeueBlockedError as exc:
                # The persistence adapter commits the terminal failure before raising. Keep the
                # run visible in both failure and attention reporting: an operator must recreate
                # the run from its external input artifact instead of accepting a lossy clone.
                failed_stale.append(run.run_id)
                errors.append((run.run_id, f"requeue_blocked: {exc.reason}"))
                continue
            except Exception as exc:
                errors.append((run.run_id, f"requeue: {type(exc).__name__}: {exc}"))
                continue
            requeued.append(queued_run_id)
            self._try_submit(queued_run_id, submitted, duplicates, deferred, errors)

        return RecoveryReport(
            queued_run_ids=tuple(queued),
            stale_run_ids=tuple(run.run_id for run in stale),
            submitted_run_ids=tuple(submitted),
            duplicate_run_ids=tuple(duplicates),
            failed_stale_run_ids=tuple(failed_stale),
            requeued_run_ids=tuple(requeued),
            deferred_run_ids=tuple(deferred),
            errors=tuple(errors),
        )

    async def recover_async(self) -> RecoveryReport:
        """Run potentially blocking persistence recovery outside the event loop."""

        return await asyncio.to_thread(self.recover)

    def _try_submit(
        self,
        run_id: str,
        submitted: list[str],
        duplicates: list[str],
        deferred: list[str],
        errors: list[tuple[str, str]],
    ) -> None:
        try:
            accepted = self.dispatcher.submit(run_id)
        except (DispatcherQueueFullError, DispatcherNotRunningError):
            deferred.append(run_id)
        except Exception as exc:
            errors.append((run_id, f"submit: {type(exc).__name__}: {exc}"))
        else:
            if accepted:
                submitted.append(run_id)
            else:
                duplicates.append(run_id)

    @staticmethod
    def _deduplicate(rows: Iterable[RecoverableRun]) -> list[RecoverableRun]:
        result: list[RecoverableRun] = []
        seen: set[str] = set()
        for row in rows:
            if not row.run_id.strip():
                continue
            if row.status not in RECOVERY_STATUSES or row.run_id in seen:
                continue
            seen.add(row.run_id)
            result.append(row)
        return result
