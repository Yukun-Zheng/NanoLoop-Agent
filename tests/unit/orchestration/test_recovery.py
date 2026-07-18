from __future__ import annotations

from collections.abc import Iterable

import pytest

from app.contracts.enums import JobStatus
from app.orchestration import (
    RECOVERY_STATUSES,
    RESTART_ERROR_CODE,
    DispatcherQueueFullError,
    InlineDispatcher,
    RecoverableRun,
    StaleRunPolicy,
    StartupRecovery,
)


class FakeRecoveryStore:
    def __init__(self, rows: list[RecoverableRun]) -> None:
        self.rows = rows
        self.queries: list[tuple[JobStatus, ...]] = []
        self.failed: list[tuple[str, str, str]] = []
        self.requeued: list[RecoverableRun] = []
        self.fail_mark_for: set[str] = set()

    def list_by_status(self, statuses: tuple[JobStatus, ...]) -> Iterable[RecoverableRun]:
        self.queries.append(statuses)
        return [row for row in self.rows if row.status in statuses]

    def mark_failed(
        self,
        run_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        if run_id in self.fail_mark_for:
            raise RuntimeError("database unavailable")
        self.failed.append((run_id, error_code, error_message))

    def requeue(self, run: RecoverableRun) -> str:
        self.requeued.append(run)
        return f"retry-{run.run_id}"


def active_rows() -> list[RecoverableRun]:
    return [
        RecoverableRun(run_id=f"stale-{status.value.lower()}", status=status)
        for status in RECOVERY_STATUSES
        if status != JobStatus.QUEUED
    ]


def test_default_recovery_submits_queued_and_fails_every_stale_active_stage() -> None:
    rows = [
        RecoverableRun("queued", JobStatus.QUEUED),
        *active_rows(),
        RecoverableRun("completed", JobStatus.COMPLETED),
    ]
    store = FakeRecoveryStore(rows)
    executed: list[str] = []
    recovery = StartupRecovery(store, InlineDispatcher(executed.append))

    report = recovery.recover()

    assert store.queries == [RECOVERY_STATUSES]
    assert executed == ["queued"]
    assert report.submitted_run_ids == ("queued",)
    assert set(report.stale_run_ids) == {row.run_id for row in active_rows()}
    assert set(report.failed_stale_run_ids) == {row.run_id for row in active_rows()}
    assert all(error_code == RESTART_ERROR_CODE for _, error_code, _ in store.failed)
    assert all("process restarted" in message for _, _, message in store.failed)
    assert not report.requires_attention


def test_requeue_policy_persists_replacement_before_dispatch() -> None:
    stale = RecoverableRun("stale", JobStatus.SEGMENTING)
    store = FakeRecoveryStore([stale])
    persistence_was_visible: list[bool] = []

    def task(run_id: str) -> None:
        persistence_was_visible.append(
            run_id == "retry-stale" and store.requeued == [stale]
        )

    recovery = StartupRecovery(
        store,
        InlineDispatcher(task),
        stale_policy=StaleRunPolicy.REQUEUE,
    )

    report = recovery.recover()

    assert persistence_was_visible == [True]
    assert report.requeued_run_ids == ("retry-stale",)
    assert report.submitted_run_ids == ("retry-stale",)
    assert report.failed_stale_run_ids == ()


def test_recovery_deduplicates_duplicate_store_rows() -> None:
    duplicate = RecoverableRun("queued", JobStatus.QUEUED)
    store = FakeRecoveryStore([duplicate, duplicate])
    executed: list[str] = []

    report = StartupRecovery(store, InlineDispatcher(executed.append)).recover()

    assert executed == ["queued"]
    assert report.queued_run_ids == ("queued",)


def test_recovery_reports_store_errors_and_continues() -> None:
    first = RecoverableRun("first", JobStatus.PREPROCESSING)
    second = RecoverableRun("second", JobStatus.ANALYZING)
    store = FakeRecoveryStore([first, second])
    store.fail_mark_for.add("first")

    report = StartupRecovery(store, InlineDispatcher(lambda _: None)).recover()

    assert report.failed_stale_run_ids == ("second",)
    assert report.errors[0][0] == "first"
    assert report.requires_attention


class CapacityOneDispatcher(InlineDispatcher):
    def __init__(self) -> None:
        super().__init__(lambda _: None)
        self.calls = 0

    def submit(self, run_id: str) -> bool:
        self.calls += 1
        if self.calls > 1:
            raise DispatcherQueueFullError(details={"run_id": run_id})
        return super().submit(run_id)


def test_recovery_reports_queue_capacity_without_blocking() -> None:
    store = FakeRecoveryStore(
        [
            RecoverableRun("first", JobStatus.QUEUED),
            RecoverableRun("second", JobStatus.QUEUED),
        ]
    )

    report = StartupRecovery(store, CapacityOneDispatcher()).recover()

    assert report.submitted_run_ids == ("first",)
    assert report.deferred_run_ids == ("second",)
    assert report.requires_attention


@pytest.mark.asyncio
async def test_async_recovery_keeps_store_scan_off_event_loop() -> None:
    store = FakeRecoveryStore([RecoverableRun("queued", JobStatus.QUEUED)])
    executed: list[str] = []
    recovery = StartupRecovery(store, InlineDispatcher(executed.append))

    report = await recovery.recover_async()

    assert report.submitted_run_ids == ("queued",)
    assert executed == ["queued"]
