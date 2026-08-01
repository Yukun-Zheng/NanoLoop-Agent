from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.agent.control import AgentResumeBatch
from app.agent.scheduler import AgentTaskScheduler


@dataclass
class _ResumeService:
    calls: list[int]

    def resume_ready_tasks(self, *, limit: int) -> AgentResumeBatch:
        self.calls.append(limit)
        return AgentResumeBatch(
            selected_task_ids=("agt_1",),
            resumed_task_ids=("agt_1",),
            failed_task_ids=(),
        )


def test_agent_scheduler_runs_one_bounded_reconciliation() -> None:
    service = _ResumeService([])
    scheduler = AgentTaskScheduler(  # type: ignore[arg-type]
        service,
        poll_interval_seconds=0.01,
        batch_size=7,
    )

    result = asyncio.run(scheduler.run_once())
    snapshot = scheduler.snapshot()

    assert result.resumed_task_ids == ("agt_1",)
    assert service.calls == [7]
    assert snapshot.iterations == 1
    assert snapshot.last_selected_count == 1
    assert snapshot.last_resumed_count == 1
    assert snapshot.error_count == 0


@pytest.mark.parametrize(
    ("poll_interval_seconds", "batch_size"),
    [(0, 20), (1, 0), (1, 101)],
)
def test_agent_scheduler_rejects_unsafe_bounds(
    poll_interval_seconds: float,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        AgentTaskScheduler(  # type: ignore[arg-type]
            _ResumeService([]),
            poll_interval_seconds=poll_interval_seconds,
            batch_size=batch_size,
        )
