"""Server-owned reconciliation for durable Agent continuations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.agent.control import AgentControlService, AgentResumeBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentSchedulerSnapshot:
    running: bool
    iterations: int
    last_selected_count: int
    last_resumed_count: int
    last_failed_count: int
    error_count: int


class AgentTaskScheduler:
    """Resume due Agent work from database state, independent of browser lifetime."""

    def __init__(
        self,
        service: AgentControlService,
        *,
        poll_interval_seconds: float = 1.0,
        batch_size: int = 20,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if batch_size < 1 or batch_size > 100:
            raise ValueError("batch_size must be between 1 and 100")
        self._service = service
        self._poll_interval_seconds = poll_interval_seconds
        self._batch_size = batch_size
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._iterations = 0
        self._last_selected_count = 0
        self._last_resumed_count = 0
        self._last_failed_count = 0
        self._error_count = 0

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="nanoloop-agent-task-scheduler",
        )

    async def run_once(self) -> AgentResumeBatch:
        result = await asyncio.to_thread(
            self._service.resume_ready_tasks,
            limit=self._batch_size,
        )
        self._iterations += 1
        self._last_selected_count = len(result.selected_task_ids)
        self._last_resumed_count = len(result.resumed_task_ids)
        self._last_failed_count = len(result.failed_task_ids)
        return result

    async def astop(self, *, timeout: float | None = None) -> bool:
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

    def snapshot(self) -> AgentSchedulerSnapshot:
        return AgentSchedulerSnapshot(
            running=self.is_running,
            iterations=self._iterations,
            last_selected_count=self._last_selected_count,
            last_resumed_count=self._last_resumed_count,
            last_failed_count=self._last_failed_count,
            error_count=self._error_count,
        )

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = await self.run_once()
                if result.failed_task_ids:
                    logger.warning(
                        "agent_task_reconciliation_incomplete",
                        extra={
                            "event": "agent_task_reconciliation_incomplete",
                            "detail": str(len(result.failed_task_ids)),
                        },
                    )
            except Exception as error:
                self._error_count += 1
                logger.exception(
                    "agent_task_reconciliation_failed",
                    exc_info=error,
                    extra={"event": "agent_task_reconciliation_failed"},
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except TimeoutError:
                continue
