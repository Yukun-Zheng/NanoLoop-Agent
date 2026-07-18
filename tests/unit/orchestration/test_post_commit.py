from __future__ import annotations

import pytest

from app.orchestration import (
    InlineDispatcher,
    commit_then_submit,
    submit_committed_run_ids,
)


def test_commit_happens_before_any_task_execution() -> None:
    committed = False
    observed: list[tuple[str, bool]] = []

    def commit() -> None:
        nonlocal committed
        committed = True

    dispatcher = InlineDispatcher(lambda run_id: observed.append((run_id, committed)))

    result = commit_then_submit(commit, dispatcher, ["run-1", "run-2"])

    assert observed == [("run-1", True), ("run-2", True)]
    assert result.submitted_run_ids == ("run-1", "run-2")
    assert result.all_dispatched


def test_commit_failure_prevents_dispatch() -> None:
    executed: list[str] = []
    dispatcher = InlineDispatcher(executed.append)

    def fail_commit() -> None:
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        commit_then_submit(fail_commit, dispatcher, ["run-1"])

    assert executed == []


def test_batch_deduplicates_input_even_for_synchronous_dispatcher() -> None:
    executed: list[str] = []
    dispatcher = InlineDispatcher(executed.append)

    result = submit_committed_run_ids(dispatcher, ["same", "same"])

    assert executed == ["same"]
    assert result.duplicate_run_ids == ("same",)


def test_stopped_dispatcher_leaves_committed_ids_deferred() -> None:
    dispatcher = InlineDispatcher(lambda _: None, autostart=False)

    result = submit_committed_run_ids(dispatcher, ["queued"])

    assert result.submitted_run_ids == ()
    assert result.deferred_run_ids == ("queued",)
    assert not result.all_dispatched
