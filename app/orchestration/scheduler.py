"""Periodic reconciliation of durable queued runs with the bounded worker queue."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.contracts.enums import JobStatus
from app.orchestration.dispatcher import TaskDispatcher
from app.orchestration.post_commit import SubmissionBatch, submit_committed_run_ids
from app.orchestration.recovery import RunRecoveryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    """Small operational snapshot suitable for health diagnostics."""

    running: bool
    iterations: int
    last_queued_count: int
    last_deferred_count: int
    error_count: int


class QueuedRunScheduler:
    """Retry durable ``QUEUED`` rows until the bounded dispatcher accepts them.

    A single create-runs request may contain more IDs than the in-memory queue can
    accept. The database remains the source of truth; this scheduler periodically
    reconciles queued rows instead of leaving overflow work stranded until restart.
    Duplicate suppression remains the dispatcher's responsibility.
    """

    def __init__(
        self,
        store: RunRecoveryStore,
        dispatcher: TaskDispatcher,
        *,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._store = store
        self._dispatcher = dispatcher
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._iterations = 0
        self._last_queued_count = 0
        self._last_deferred_count = 0
        self._error_count = 0

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start one idempotent reconciliation task on the current event loop."""

        if self.is_running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="nanoloop-queued-run-scheduler",
        )

    async def run_once(self) -> SubmissionBatch:
        """Submit one database snapshot without blocking the event loop."""

        rows = await asyncio.to_thread(
            lambda: list(self._store.list_by_status((JobStatus.QUEUED,)))
        )
        run_ids = [row.run_id for row in rows]
        result = submit_committed_run_ids(self._dispatcher, run_ids)
        self._iterations += 1
        self._last_queued_count = len(run_ids)
        self._last_deferred_count = len(result.deferred_run_ids)
        return result

    async def astop(self, *, timeout: float | None = None) -> bool:
        """Stop future submissions before the dispatcher begins draining."""

        task = self._task
        if task is None:
            return True
        self._stop_event.set()
        try:
            if timeout is None:
                await task
            else:
                await asyncio.wait_for(task, timeout=timeout)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return False
        finally:
            if task.done():
                self._task = None
        return True

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            running=self.is_running,
            iterations=self._iterations,
            last_queued_count=self._last_queued_count,
            last_deferred_count=self._last_deferred_count,
            error_count=self._error_count,
        )

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = await self.run_once()
                if result.deferred_run_ids:
                    logger.debug(
                        "queued_run_reconciliation_deferred",
                        extra={
                            "event": "queued_run_reconciliation_deferred",
                            "detail": str(len(result.deferred_run_ids)),
                        },
                    )
            except Exception as error:
                self._error_count += 1
                logger.exception(
                    "queued_run_reconciliation_failed",
                    exc_info=error,
                    extra={"event": "queued_run_reconciliation_failed"},
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except TimeoutError:
                continue
