"""Bounded in-process dispatchers for durable run IDs.

Only the ``run_id`` crosses the queue boundary. Workers must open their own transaction and read the
committed run configuration; request-scoped sessions and ORM objects must never be captured here.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from app.core.errors import ServiceUnavailableError

logger = logging.getLogger(__name__)

RunTask = Callable[[str], None]
TaskErrorCallback = Callable[[str, BaseException], None]
TaskCompleteCallback = Callable[[str], None]


class DispatcherUnavailableError(ServiceUnavailableError):
    """Raised when a dispatcher cannot currently accept work."""

    default_message = "任务调度器当前不可用"


class DispatcherNotRunningError(DispatcherUnavailableError):
    default_message = "任务调度器尚未启动或正在关闭"


class DispatcherQueueFullError(DispatcherUnavailableError):
    default_message = "任务队列已满，请稍后重试"


class DispatcherLifecycleError(RuntimeError):
    """Raised for contradictory lifecycle operations."""


@dataclass(frozen=True, slots=True)
class TaskFailure:
    run_id: str
    error_type: str
    error_message: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class DispatcherSnapshot:
    running: bool
    accepting: bool
    worker_count: int
    queue_capacity: int
    queued_run_ids: tuple[str, ...]
    in_flight_run_ids: tuple[str, ...]
    failure_count: int


@runtime_checkable
class TaskDispatcher(Protocol):
    """Minimal lifecycle and submission contract used by application services."""

    def start(self) -> None: ...

    def submit(self, run_id: str) -> bool:
        """Submit committed work; return false while the same run is already pending."""

    def drain(self, timeout: float | None = None) -> bool: ...

    def stop(self, *, drain: bool = True, timeout: float | None = None) -> bool: ...

    async def asubmit(self, run_id: str) -> bool: ...

    async def adrain(self, timeout: float | None = None) -> bool: ...

    async def astop(self, *, drain: bool = True, timeout: float | None = None) -> bool: ...


class InProcessTaskDispatcher:
    """Fixed-size worker pool with a bounded, non-blocking submission queue.

    ``submit`` uses ``put_nowait`` and therefore never waits for model work or queue capacity. The
    synchronous task runs only on dedicated worker threads. Slow shutdown helpers have async
    counterparts backed by ``asyncio.to_thread`` so a FastAPI lifespan need not block its loop.
    """

    def __init__(
        self,
        task: RunTask,
        *,
        worker_count: int = 2,
        queue_capacity: int = 32,
        on_error: TaskErrorCallback | None = None,
        on_complete: TaskCompleteCallback | None = None,
        thread_name_prefix: str = "nanoloop-run",
        daemon: bool = True,
        failure_history_size: int = 100,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if failure_history_size < 1:
            raise ValueError("failure_history_size must be at least 1")
        self._task = task
        self._worker_count = worker_count
        self._queue_capacity = queue_capacity
        self._on_error = on_error
        self._on_complete = on_complete
        self._thread_name_prefix = thread_name_prefix
        self._daemon = daemon
        self._queue: queue.Queue[str] = queue.Queue(maxsize=queue_capacity)
        self._condition = threading.Condition(threading.RLock())
        self._shutdown = threading.Event()
        self._threads: list[threading.Thread] = []
        self._known_run_ids: set[str] = set()
        self._in_flight_run_ids: set[str] = set()
        self._failures: deque[TaskFailure] = deque(maxlen=failure_history_size)
        self._running = False
        self._accepting = False
        self._live_workers = 0

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._running

    @property
    def is_accepting(self) -> bool:
        with self._condition:
            return self._accepting

    def start(self) -> None:
        with self._condition:
            if self._running:
                if self._accepting:
                    return
                raise DispatcherLifecycleError("dispatcher shutdown is still in progress")
            if self._known_run_ids or not self._queue.empty():
                raise DispatcherLifecycleError("dispatcher cannot start with abandoned queue state")
            self._shutdown.clear()
            self._running = True
            self._accepting = True
            self._live_workers = self._worker_count
            self._threads = [
                threading.Thread(
                    target=self._worker_loop,
                    name=f"{self._thread_name_prefix}-{index + 1}",
                    daemon=self._daemon,
                )
                for index in range(self._worker_count)
            ]
            threads = list(self._threads)
        try:
            for thread in threads:
                thread.start()
        except BaseException:
            self._shutdown.set()
            with self._condition:
                self._accepting = False
            raise

    def submit(self, run_id: str) -> bool:
        normalized = self._validate_run_id(run_id)
        with self._condition:
            if not self._running or not self._accepting:
                raise DispatcherNotRunningError(details={"run_id": normalized})
            if normalized in self._known_run_ids:
                return False
            try:
                self._queue.put_nowait(normalized)
            except queue.Full as exc:
                raise DispatcherQueueFullError(
                    details={
                        "run_id": normalized,
                        "queue_capacity": self._queue_capacity,
                    }
                ) from exc
            self._known_run_ids.add(normalized)
            self._condition.notify_all()
        return True

    async def asubmit(self, run_id: str) -> bool:
        # Submission is already bounded and non-blocking; no executor hop is needed.
        return self.submit(run_id)

    def drain(self, timeout: float | None = None) -> bool:
        deadline = self._deadline(timeout)
        with self._condition:
            while self._known_run_ids:
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    async def adrain(self, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self.drain, timeout)

    def stop(self, *, drain: bool = True, timeout: float | None = None) -> bool:
        deadline = self._deadline(timeout)
        with self._condition:
            if not self._running:
                self._accepting = False
                return True
            self._accepting = False

        drained = not drain or self.drain(self._remaining(deadline))
        if not drain:
            self._discard_queued()

        # Even when graceful draining exceeds its caller's deadline, shutdown has been initiated:
        # queued work continues on workers, then they exit once the queue becomes empty.
        self._shutdown.set()
        with self._condition:
            self._condition.notify_all()
            threads = list(self._threads)
        current = threading.current_thread()
        for thread in threads:
            if thread is current:
                continue
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                break
            thread.join(remaining)
        with self._condition:
            return drained and not self._running

    async def astop(self, *, drain: bool = True, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self.stop, drain=drain, timeout=timeout)

    def snapshot(self) -> DispatcherSnapshot:
        with self._condition:
            queued = self._known_run_ids - self._in_flight_run_ids
            return DispatcherSnapshot(
                running=self._running,
                accepting=self._accepting,
                worker_count=self._worker_count,
                queue_capacity=self._queue_capacity,
                queued_run_ids=tuple(sorted(queued)),
                in_flight_run_ids=tuple(sorted(self._in_flight_run_ids)),
                failure_count=len(self._failures),
            )

    def failure_history(self) -> tuple[TaskFailure, ...]:
        with self._condition:
            return tuple(self._failures)

    def _worker_loop(self) -> None:
        try:
            while True:
                try:
                    run_id = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._shutdown.is_set():
                        return
                    continue
                with self._condition:
                    self._in_flight_run_ids.add(run_id)
                    self._condition.notify_all()
                try:
                    self._task(run_id)
                except BaseException as exc:
                    self._record_failure(run_id, exc)
                else:
                    self._notify_complete(run_id)
                finally:
                    with self._condition:
                        self._in_flight_run_ids.discard(run_id)
                        self._known_run_ids.discard(run_id)
                        self._condition.notify_all()
                    self._queue.task_done()
        finally:
            with self._condition:
                self._live_workers -= 1
                if self._live_workers <= 0:
                    self._live_workers = 0
                    self._running = False
                    self._accepting = False
                self._condition.notify_all()

    def _discard_queued(self) -> None:
        discarded: list[str] = []
        while True:
            try:
                run_id = self._queue.get_nowait()
            except queue.Empty:
                break
            discarded.append(run_id)
            self._queue.task_done()
        if discarded:
            with self._condition:
                self._known_run_ids.difference_update(discarded)
                self._condition.notify_all()

    def _record_failure(self, run_id: str, exc: BaseException) -> None:
        failure = TaskFailure(
            run_id=run_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            occurred_at=datetime.now(UTC),
        )
        with self._condition:
            self._failures.append(failure)
        if self._on_error is not None:
            try:
                self._on_error(run_id, exc)
            except BaseException:
                logger.exception("task error callback failed", extra={"run_id": run_id})

    def _notify_complete(self, run_id: str) -> None:
        if self._on_complete is not None:
            try:
                self._on_complete(run_id)
            except BaseException:
                logger.exception("task completion callback failed", extra={"run_id": run_id})

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        return run_id.strip()

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        return time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())


class InlineDispatcher:
    """Synchronous dispatcher for deterministic service tests.

    It mirrors duplicate suppression and failure callbacks while executing in the submitting
    thread. By default task exceptions are isolated like the threaded dispatcher; tests that need
    direct exception assertions may enable ``propagate_exceptions``.
    """

    def __init__(
        self,
        task: RunTask,
        *,
        on_error: TaskErrorCallback | None = None,
        on_complete: TaskCompleteCallback | None = None,
        propagate_exceptions: bool = False,
        autostart: bool = True,
        failure_history_size: int = 100,
    ) -> None:
        if failure_history_size < 1:
            raise ValueError("failure_history_size must be at least 1")
        self._task = task
        self._on_error = on_error
        self._on_complete = on_complete
        self._propagate_exceptions = propagate_exceptions
        self._condition = threading.Condition(threading.RLock())
        self._known_run_ids: set[str] = set()
        self._failures: deque[TaskFailure] = deque(maxlen=failure_history_size)
        self._running = autostart
        self._accepting = autostart

    def start(self) -> None:
        with self._condition:
            self._running = True
            self._accepting = True

    def submit(self, run_id: str) -> bool:
        normalized = InProcessTaskDispatcher._validate_run_id(run_id)
        with self._condition:
            if not self._running or not self._accepting:
                raise DispatcherNotRunningError(details={"run_id": normalized})
            if normalized in self._known_run_ids:
                return False
            self._known_run_ids.add(normalized)
        try:
            self._task(normalized)
        except BaseException as exc:
            self._record_failure(normalized, exc)
            if self._propagate_exceptions:
                raise
        else:
            self._notify_complete(normalized)
        finally:
            with self._condition:
                self._known_run_ids.discard(normalized)
                self._condition.notify_all()
        return True

    async def asubmit(self, run_id: str) -> bool:
        # This is a test-only synchronous implementation by design.
        return self.submit(run_id)

    def drain(self, timeout: float | None = None) -> bool:
        deadline = InProcessTaskDispatcher._deadline(timeout)
        with self._condition:
            while self._known_run_ids:
                remaining = InProcessTaskDispatcher._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    async def adrain(self, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self.drain, timeout)

    def stop(self, *, drain: bool = True, timeout: float | None = None) -> bool:
        with self._condition:
            self._accepting = False
        drained = not drain or self.drain(timeout)
        with self._condition:
            if drained:
                self._running = False
            return drained

    async def astop(self, *, drain: bool = True, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self.stop, drain=drain, timeout=timeout)

    def failure_history(self) -> tuple[TaskFailure, ...]:
        with self._condition:
            return tuple(self._failures)

    def _record_failure(self, run_id: str, exc: BaseException) -> None:
        with self._condition:
            self._failures.append(
                TaskFailure(
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    occurred_at=datetime.now(UTC),
                )
            )
        if self._on_error is not None:
            try:
                self._on_error(run_id, exc)
            except BaseException:
                logger.exception("inline task error callback failed", extra={"run_id": run_id})

    def _notify_complete(self, run_id: str) -> None:
        if self._on_complete is not None:
            try:
                self._on_complete(run_id)
            except BaseException:
                logger.exception(
                    "inline task completion callback failed", extra={"run_id": run_id}
                )
