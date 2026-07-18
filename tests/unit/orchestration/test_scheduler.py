from __future__ import annotations

import asyncio
import threading

import pytest

from app.contracts.enums import JobStatus
from app.orchestration import (
    InlineDispatcher,
    QueuedRunScheduler,
    RecoverableRun,
)


class _QueuedStore:
    def __init__(self, *run_ids: str) -> None:
        self.run_ids = list(run_ids)
        self.called = threading.Event()

    def list_by_status(self, statuses: tuple[JobStatus, ...]) -> list[RecoverableRun]:
        assert statuses == (JobStatus.QUEUED,)
        self.called.set()
        return [RecoverableRun(run_id=run_id, status=JobStatus.QUEUED) for run_id in self.run_ids]


@pytest.mark.asyncio
async def test_durable_queued_rows_are_retried_after_dispatcher_recovers() -> None:
    executed: list[str] = []
    store = _QueuedStore("run-1", "run-2")
    dispatcher = InlineDispatcher(executed.append, autostart=False)
    scheduler = QueuedRunScheduler(store, dispatcher, poll_interval_seconds=0.01)

    deferred = await scheduler.run_once()
    assert deferred.deferred_run_ids == ("run-1", "run-2")
    assert executed == []

    dispatcher.start()
    submitted = await scheduler.run_once()

    assert submitted.submitted_run_ids == ("run-1", "run-2")
    assert executed == ["run-1", "run-2"]
    assert scheduler.snapshot().last_deferred_count == 0


@pytest.mark.asyncio
async def test_scheduler_lifecycle_is_idempotent_and_stops_cleanly() -> None:
    store = _QueuedStore()
    dispatcher = InlineDispatcher(lambda _: None)
    scheduler = QueuedRunScheduler(store, dispatcher, poll_interval_seconds=0.01)

    scheduler.start()
    scheduler.start()
    assert await asyncio.to_thread(store.called.wait, 1)
    assert scheduler.is_running

    assert await scheduler.astop(timeout=1)
    assert not scheduler.is_running
    assert scheduler.snapshot().iterations >= 1
    assert await scheduler.astop(timeout=1)


def test_scheduler_rejects_nonpositive_poll_interval() -> None:
    store = _QueuedStore()
    dispatcher = InlineDispatcher(lambda _: None)

    with pytest.raises(ValueError, match="poll_interval_seconds"):
        QueuedRunScheduler(store, dispatcher, poll_interval_seconds=0)
