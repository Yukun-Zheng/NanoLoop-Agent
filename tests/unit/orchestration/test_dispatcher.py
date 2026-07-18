from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest

from app.orchestration import (
    DispatcherNotRunningError,
    DispatcherQueueFullError,
    InlineDispatcher,
    InProcessTaskDispatcher,
    TaskDispatcher,
)


def _stop_safely(dispatcher: InProcessTaskDispatcher, release: threading.Event) -> None:
    release.set()
    dispatcher.stop(drain=False, timeout=2)


def test_protocol_is_structural() -> None:
    threaded = InProcessTaskDispatcher(lambda _: None)
    inline = InlineDispatcher(lambda _: None)

    assert isinstance(threaded, TaskDispatcher)
    assert isinstance(inline, TaskDispatcher)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: InProcessTaskDispatcher(lambda _: None, worker_count=0), "worker_count"),
        (lambda: InProcessTaskDispatcher(lambda _: None, queue_capacity=0), "queue_capacity"),
    ],
)
def test_invalid_bounds_are_rejected(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_queue_is_bounded_and_pending_run_ids_are_deduplicated() -> None:
    entered = threading.Event()
    release = threading.Event()
    completed: list[str] = []

    def task(run_id: str) -> None:
        if run_id == "run-1":
            entered.set()
            assert release.wait(2)
        completed.append(run_id)

    dispatcher = InProcessTaskDispatcher(task, worker_count=1, queue_capacity=1)
    dispatcher.start()
    try:
        assert dispatcher.submit("run-1") is True
        assert entered.wait(1)
        assert dispatcher.submit("run-1") is False
        assert dispatcher.submit("run-2") is True
        assert dispatcher.submit("run-2") is False
        with pytest.raises(DispatcherQueueFullError):
            dispatcher.submit("run-3")

        snapshot = dispatcher.snapshot()
        assert snapshot.in_flight_run_ids == ("run-1",)
        assert snapshot.queued_run_ids == ("run-2",)

        release.set()
        assert dispatcher.drain(timeout=2)
        assert completed == ["run-1", "run-2"]
    finally:
        _stop_safely(dispatcher, release)


def test_task_failure_is_recorded_and_does_not_kill_worker() -> None:
    executed: list[str] = []
    callback_failures: list[tuple[str, BaseException]] = []

    def task(run_id: str) -> None:
        executed.append(run_id)
        if run_id == "bad":
            raise RuntimeError("deliberate failure")

    dispatcher = InProcessTaskDispatcher(
        task,
        worker_count=1,
        queue_capacity=2,
        on_error=lambda run_id, exc: callback_failures.append((run_id, exc)),
    )
    dispatcher.start()
    try:
        dispatcher.submit("bad")
        dispatcher.submit("good")
        assert dispatcher.drain(timeout=2)

        assert executed == ["bad", "good"]
        assert callback_failures[0][0] == "bad"
        assert isinstance(callback_failures[0][1], RuntimeError)
        failure = dispatcher.failure_history()[0]
        assert failure.run_id == "bad"
        assert failure.error_type == "RuntimeError"
        assert failure.error_message == "deliberate failure"
    finally:
        dispatcher.stop(timeout=2)


def test_callback_failure_is_isolated_from_following_work() -> None:
    completed: list[str] = []

    def task(run_id: str) -> None:
        if run_id == "bad":
            raise ValueError("bad")
        completed.append(run_id)

    def broken_callback(_run_id: str, _exc: BaseException) -> None:
        raise RuntimeError("callback failed")

    dispatcher = InProcessTaskDispatcher(task, on_error=broken_callback, worker_count=1)
    dispatcher.start()
    try:
        dispatcher.submit("bad")
        dispatcher.submit("good")
        assert dispatcher.drain(timeout=2)
        assert completed == ["good"]
    finally:
        dispatcher.stop(timeout=2)


def test_stop_and_restart_have_explicit_lifecycle() -> None:
    completed: list[str] = []
    dispatcher = InProcessTaskDispatcher(completed.append, worker_count=1)
    dispatcher.start()
    dispatcher.start()  # idempotent while accepting
    dispatcher.submit("first")
    assert dispatcher.stop(drain=True, timeout=2)

    with pytest.raises(DispatcherNotRunningError):
        dispatcher.submit("rejected")

    dispatcher.start()
    dispatcher.submit("second")
    assert dispatcher.stop(drain=True, timeout=2)
    assert completed == ["first", "second"]


def test_timed_out_graceful_stop_still_finishes_shutdown_in_background() -> None:
    entered = threading.Event()
    release = threading.Event()

    def task(_run_id: str) -> None:
        entered.set()
        assert release.wait(2)

    dispatcher = InProcessTaskDispatcher(task, worker_count=1)
    dispatcher.start()
    dispatcher.submit("slow")
    assert entered.wait(1)

    assert dispatcher.stop(drain=True, timeout=0) is False
    assert not dispatcher.is_accepting
    release.set()

    assert dispatcher.drain(timeout=2)
    for _ in range(100):
        if not dispatcher.is_running:
            break
        threading.Event().wait(0.01)
    assert not dispatcher.is_running


@pytest.mark.asyncio
async def test_async_stop_does_not_block_event_loop() -> None:
    entered = threading.Event()
    release = threading.Event()

    def task(_run_id: str) -> None:
        entered.set()
        assert release.wait(2)

    dispatcher = InProcessTaskDispatcher(task, worker_count=1)
    dispatcher.start()
    dispatcher.submit("slow")
    assert await asyncio.to_thread(entered.wait, 1)

    stop_task = asyncio.create_task(dispatcher.astop(timeout=2))
    await asyncio.sleep(0)
    assert not stop_task.done()
    release.set()
    assert await stop_task


def test_inline_dispatcher_runs_synchronously_and_suppresses_nested_duplicate() -> None:
    events: list[str] = []
    nested_results: list[bool] = []
    holder: list[InlineDispatcher] = []

    def task(run_id: str) -> None:
        events.append(run_id)
        nested_results.append(holder[0].submit(run_id))

    dispatcher = InlineDispatcher(task)
    holder.append(dispatcher)

    assert dispatcher.submit("inline") is True
    assert events == ["inline"]
    assert nested_results == [False]


def test_inline_failure_callback_matches_threaded_semantics() -> None:
    failures: list[str] = []

    def fail(_run_id: str) -> None:
        raise RuntimeError("inline failure")

    dispatcher = InlineDispatcher(fail, on_error=lambda run_id, _exc: failures.append(run_id))

    assert dispatcher.submit("bad") is True
    assert failures == ["bad"]
    assert dispatcher.failure_history()[0].error_message == "inline failure"

    propagating = InlineDispatcher(fail, propagate_exceptions=True)
    with pytest.raises(RuntimeError, match="inline failure"):
        propagating.submit("bad")
