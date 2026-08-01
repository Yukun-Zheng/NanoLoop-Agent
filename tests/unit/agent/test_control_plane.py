from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.agent.context_budget import bounded_newest_mappings
from app.agent.control import AgentControlService
from app.agent.model_provider import AgentModelProviderError
from app.agent.protocols import AgentToolContext
from app.agent.tool_registry import AgentToolRegistry, RegisteredAgentTool
from app.contracts.agent_runtime import (
    AgentDecision,
    AgentDecisionKind,
    AgentDecisionRequest,
    AgentModelIdentity,
    AgentTaskStatus,
    AgentToolObservation,
    AgentToolOutcome,
    AgentToolRisk,
    AgentToolSpec,
    CreateAgentTaskRequest,
    ResolveAgentApprovalRequest,
)
from app.contracts.analyses import AnalysisJobDTO
from app.contracts.common import ContractModel, HealthComponent
from app.contracts.enums import JobStatus
from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
)
from app.core.config import Settings
from app.db.base import Base
from app.db.models import AgentTask, ApiCredential, Principal, Tenant
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database


class _ToolArguments(ContractModel):
    value: str


@dataclass
class _RecordingTool:
    name: str
    requires_approval: bool
    calls: list[dict[str, Any]]

    @property
    def spec(self) -> AgentToolSpec:
        return AgentToolSpec(
            name=self.name,
            description=f"test tool {self.name}",
            input_schema=_ToolArguments.model_json_schema(),
            risk=(
                AgentToolRisk.CONTROLLED_WRITE
                if self.requires_approval
                else AgentToolRisk.READ_ONLY
            ),
            requires_approval=self.requires_approval,
            idempotent=not self.requires_approval,
        )

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        self.calls.append(
            {
                "task_id": context.task_id,
                "job_id": context.job_id,
                "arguments": arguments,
            }
        )
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary=f"{self.name} completed",
            data={"value": arguments["value"]},
            evidence_refs=["evidence-1"],
        )


class _ScriptedModel:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self._decisions = decisions
        self.requests: list[AgentDecisionRequest] = []

    @property
    def identity(self) -> AgentModelIdentity:
        return AgentModelIdentity(provider="scripted", model="test-controller")

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy")

    def decide(self, request: AgentDecisionRequest) -> AgentDecision:
        self.requests.append(request)
        if not self._decisions:
            raise AssertionError("unexpected model call")
        return self._decisions.pop(0)


class _FailingModel(_ScriptedModel):
    def decide(self, request: AgentDecisionRequest) -> AgentDecision:
        self.requests.append(request)
        raise AgentModelProviderError("local model returned invalid JSON")


@dataclass
class _PollingTool(_RecordingTool):
    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        self.calls.append({"task_id": context.task_id, "arguments": arguments})
        if len(self.calls) == 1:
            return AgentToolObservation(
                outcome=AgentToolOutcome.OK,
                summary="external work is still running",
                suggested_poll_after_seconds=2,
                continuation_tool=self.name,
                continuation_arguments=arguments,
            )
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary="external work completed",
            data={"value": arguments["value"]},
            evidence_refs=["evidence-complete"],
        )


@dataclass
class _CrashOnceTool(_RecordingTool):
    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        self.calls.append({"task_id": context.task_id, "arguments": arguments})
        if len(self.calls) == 1:
            raise SystemExit("simulated process crash")
        return AgentToolObservation(
            outcome=AgentToolOutcome.OK,
            summary="recovered idempotent tool",
            data={"value": arguments["value"]},
            evidence_refs=["evidence-recovered"],
        )


@pytest.fixture
def control_database(tmp_path: Path) -> Iterator[Database]:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        output_root=tmp_path / "outputs",
        model_registry_path=tmp_path / "registry.yaml",
        faiss_index_path=tmp_path / "faiss.index",
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with database.session() as session:
        SqlAlchemyRepositorySet(session).jobs.create(
            AnalysisJobDTO(
                job_id="job-agent",
                name="agent test",
                status=JobStatus.READY_FOR_CONFIGURATION,
                created_at=now,
                updated_at=now,
            ),
            tenant_id=LEGACY_TENANT_ID,
            owner_principal_id=LEGACY_PRINCIPAL_ID,
        )
    try:
        yield database
    finally:
        database.dispose()


def test_read_tool_observation_drives_follow_up_and_completion(
    control_database: Database,
) -> None:
    tool = _RecordingTool("inspect_fixture", False, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "inspect"},
            ),
            _decision(
                kind=AgentDecisionKind.FINISH,
                final_answer="检查完成，证据见工具事件。",
            ),
        ]
    )
    service = _service(control_database, model, tool)

    task = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="检查当前科研任务"),
        principal=_principal(),
    )

    assert task.status is AgentTaskStatus.COMPLETED
    assert task.step_count == 2
    assert task.failure_count == 0
    assert task.final_answer == "检查完成，证据见工具事件。"
    assert [event.sequence for event in task.events] == list(
        range(1, len(task.events) + 1)
    )
    assert len(tool.calls) == 1
    assert model.requests[1].latest_observations[-1]["tool_name"] == tool.name
    model_events = [
        event for event in task.events if event.event_type.value == "model.decision"
    ]
    assert model_events[-1].payload["model"] == {
        "provider": "scripted",
        "model": "test-controller",
    }


def test_write_tool_waits_for_human_approval_before_execution(
    control_database: Database,
) -> None:
    tool = _RecordingTool("mutate_fixture", True, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "approved-write"},
            ),
            _decision(
                kind=AgentDecisionKind.FINISH,
                final_answer="受控写操作已经完成。",
            ),
        ]
    )
    service = _service(control_database, model, tool)

    waiting = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="执行一个受控写操作"),
        principal=_principal(),
    )

    assert waiting.status is AgentTaskStatus.WAITING_FOR_APPROVAL
    assert tool.calls == []
    approval = waiting.approvals[0]
    assert waiting.pending_action is not None
    after_approval = service.resolve_approval(
        waiting.task_id,
        approval.approval_id,
        ResolveAgentApprovalRequest(decision="approve", comment="确认执行"),
        principal=_principal(),
    )
    assert after_approval.status is AgentTaskStatus.RUNNING
    assert len(tool.calls) == 1

    completed = service.run(after_approval.task_id, principal=_principal())
    assert completed.status is AgentTaskStatus.COMPLETED
    assert completed.approvals[0].status.value == "approved"


def test_rejected_write_returns_to_planning_without_calling_tool(
    control_database: Database,
) -> None:
    tool = _RecordingTool("mutate_fixture", True, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "do-not-run"},
            )
        ]
    )
    service = _service(control_database, model, tool)
    waiting = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="提出一个待拒绝动作"),
        principal=_principal(),
    )

    resumed = service.resolve_approval(
        waiting.task_id,
        waiting.approvals[0].approval_id,
        ResolveAgentApprovalRequest(decision="reject", comment="参数不合适"),
        principal=_principal(),
    )

    assert resumed.status is AgentTaskStatus.RUNNING
    assert resumed.pending_action is None
    assert resumed.latest_observations[-1]["data"]["rejected"] is True
    assert tool.calls == []


def test_model_failure_is_persisted_and_stops_only_the_current_run(
    control_database: Database,
) -> None:
    tool = _RecordingTool("inspect_fixture", False, [])
    model = _FailingModel([])
    service = _service(control_database, model, tool)

    task = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="测试本地模型异常"),
        principal=_principal(),
    )

    assert task.status is AgentTaskStatus.RUNNING
    assert task.failure_count == 1
    assert "invalid JSON" in (task.error or "")
    assert tool.calls == []


def test_async_polling_does_not_consume_additional_model_steps(
    control_database: Database,
) -> None:
    tool = _PollingTool("inspect_fixture", False, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "poll"},
            ),
            _decision(
                kind=AgentDecisionKind.FINISH,
                final_answer="异步工作已经完成。",
                final_evidence_refs=["evidence-complete"],
            ),
        ]
    )
    service = _service(control_database, model, tool)

    waiting = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="等待外部运行完成"),
        principal=_principal(),
    )
    assert waiting.status is AgentTaskStatus.WAITING_FOR_EXTERNAL
    assert waiting.step_count == 1
    assert waiting.next_wakeup_at is not None

    premature = service.run(waiting.task_id, principal=_principal())
    assert premature.status is AgentTaskStatus.WAITING_FOR_EXTERNAL
    assert len(tool.calls) == 1

    with control_database.session() as session:
        persisted = session.get(AgentTask, waiting.task_id)
        assert persisted is not None
        persisted.next_wakeup_at = datetime.now(UTC) - timedelta(seconds=1)

    batch = service.resume_ready_tasks()
    completed = service.get(waiting.task_id, principal=_principal())
    assert completed.status is AgentTaskStatus.COMPLETED
    assert completed.step_count == 2
    assert len(tool.calls) == 2
    assert len(model.requests) == 2
    assert batch.resumed_task_ids == (waiting.task_id,)


def test_crashed_non_idempotent_write_is_never_replayed(
    control_database: Database,
) -> None:
    tool = _CrashOnceTool("mutate_fixture", True, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "write-once"},
            )
        ]
    )
    service = _service(control_database, model, tool)
    waiting = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="执行不可重复的受控写操作"),
        principal=_principal(),
    )

    with pytest.raises(SystemExit, match="simulated process crash"):
        service.resolve_approval(
            waiting.task_id,
            waiting.approvals[0].approval_id,
            ResolveAgentApprovalRequest(decision="approve"),
            principal=_principal(),
        )

    recovered = service.run(waiting.task_id, principal=_principal())

    assert recovered.status is AgentTaskStatus.FAILED
    assert recovered.pending_action is None
    assert len(tool.calls) == 1
    assert "避免重复写入" in (recovered.error or "")
    assert recovered.latest_observations[-1]["data"]["reason"] == (
        "ambiguous_non_idempotent_outcome"
    )


def test_crashed_idempotent_read_is_safely_replayed(
    control_database: Database,
) -> None:
    tool = _CrashOnceTool("inspect_fixture", False, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "read"},
            ),
            _decision(
                kind=AgentDecisionKind.FINISH,
                final_answer="只读检查已恢复并完成。",
                final_evidence_refs=["evidence-recovered"],
            ),
        ]
    )
    service = _service(control_database, model, tool)

    with pytest.raises(SystemExit, match="simulated process crash"):
        service.create(
            "job-agent",
            CreateAgentTaskRequest(goal="恢复可重放的只读检查"),
            principal=_principal(),
        )

    with control_database.session() as session:
        task_id = session.scalar(
            select(AgentTask.task_id).order_by(AgentTask.created_at.desc())
        )
    assert task_id is not None

    recovered = service.run(task_id, principal=_principal())

    assert recovered.status is AgentTaskStatus.COMPLETED
    assert len(tool.calls) == 2


def test_server_resume_stops_when_originating_credential_is_revoked(
    control_database: Database,
) -> None:
    principal = _seed_principal_job(control_database)
    tool = _PollingTool("inspect_fixture", False, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "principal-poll"},
            )
        ]
    )
    service = _service(control_database, model, tool)
    waiting = service.create(
        "job-principal",
        CreateAgentTaskRequest(goal="使用可撤销身份等待后台任务"),
        principal=principal,
    )
    assert waiting.status is AgentTaskStatus.WAITING_FOR_EXTERNAL

    with control_database.session() as session:
        credential = session.get(ApiCredential, principal.credential_id)
        task = session.get(AgentTask, waiting.task_id)
        assert credential is not None
        assert task is not None
        credential.revoked_at = datetime.now(UTC)
        task.next_wakeup_at = datetime.now(UTC) - timedelta(seconds=1)

    batch = service.resume_ready_tasks()
    stopped = service.get(waiting.task_id, principal=principal)

    assert batch.failed_task_ids == (waiting.task_id,)
    assert stopped.status is AgentTaskStatus.FAILED
    assert stopped.pending_action is None
    assert "授权已失效" in (stopped.error or "")
    assert len(tool.calls) == 1


def test_completion_rejects_evidence_not_returned_by_tools(
    control_database: Database,
) -> None:
    tool = _RecordingTool("inspect_fixture", False, [])
    model = _ScriptedModel(
        [
            _decision(
                kind=AgentDecisionKind.CALL_TOOL,
                tool_name=tool.name,
                tool_arguments={"value": "inspect"},
            ),
            _decision(
                kind=AgentDecisionKind.FINISH,
                final_answer="这个结论引用了不存在的证据。",
                final_evidence_refs=["invented-evidence"],
            ),
        ]
    )
    service = _service(control_database, model, tool)

    task = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="拒绝伪造的完成证据"),
        principal=_principal(),
    )

    assert task.status is AgentTaskStatus.RUNNING
    assert task.final_answer is None
    assert task.final_evidence_refs == []
    assert task.failure_count == 1
    assert "不存在的证据" in (task.error or "")


def test_agent_event_rows_are_append_only_at_database_boundary(
    control_database: Database,
) -> None:
    tool = _RecordingTool("inspect_fixture", False, [])
    service = _service(control_database, _FailingModel([]), tool)
    task = service.create(
        "job-agent",
        CreateAgentTaskRequest(goal="创建一条不可变事件"),
        principal=_principal(),
    )

    with (
        pytest.raises(IntegrityError, match="append-only"),
        control_database.engine.begin() as connection,
    ):
        connection.exec_driver_sql(
            "UPDATE agent_task_events SET summary = 'tampered' WHERE task_id = ?",
            (task.task_id,),
        )


def test_observation_history_uses_one_total_budget_and_keeps_newest() -> None:
    observations = [
        {"sequence": index, "payload": str(index) * 900}
        for index in range(20)
    ]

    bounded = bounded_newest_mappings(
        observations,
        max_items=12,
        max_chars=2_000,
    )

    encoded = json.dumps(
        bounded,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(encoded) <= 2_000
    assert bounded[-1]["sequence"] == 19
    assert len(bounded) < 12


def test_single_large_observation_is_truncated_inside_total_budget() -> None:
    bounded = bounded_newest_mappings(
        [{"sequence": 1, "payload": "测" * 20_000}],
        max_items=12,
        max_chars=1_000,
    )

    encoded = json.dumps(
        bounded,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(encoded) <= 1_000
    assert bounded == [
        {
            "truncated": True,
            "available_keys": ["payload", "sequence"],
            "preview": bounded[0]["preview"],
        }
    ]


def _service(
    database: Database,
    model: _ScriptedModel,
    tool: _RecordingTool,
) -> AgentControlService:
    registry = AgentToolRegistry(
        [RegisteredAgentTool(tool=tool, arguments_model=_ToolArguments)]
    )
    return AgentControlService(
        session_factory=database.session_factory,
        model=model,
        tools=registry,
    )


def _decision(
    *,
    kind: AgentDecisionKind,
    tool_name: str | None = None,
    tool_arguments: dict[str, Any] | None = None,
    final_answer: str | None = None,
    final_evidence_refs: list[str] | None = None,
) -> AgentDecision:
    return AgentDecision(
        kind=kind,
        plan=["检查", "完成"],
        current_step="执行测试步骤",
        rationale_summary="根据当前公开状态选择下一步。",
        tool_name=tool_name,
        tool_arguments=tool_arguments or {},
        final_answer=final_answer,
        final_evidence_refs=(
            final_evidence_refs
            if final_evidence_refs is not None
            else (["evidence-1"] if kind is AgentDecisionKind.FINISH else [])
        ),
    )


def _principal() -> PrincipalContext:
    return PrincipalContext(
        tenant_id=LEGACY_TENANT_ID,
        principal_id=LEGACY_PRINCIPAL_ID,
        kind=PrincipalKind.SERVICE,
        role=PrincipalRole.TENANT_ADMIN,
        auth_mode=AuthMode.DISABLED,
    )


def _seed_principal_job(database: Database) -> PrincipalContext:
    tenant_id = f"tnt_{'1' * 32}"
    principal_id = f"prn_{'2' * 32}"
    credential_id = f"crd_{'3' * 32}"
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            Tenant(
                tenant_id=tenant_id,
                slug="agent-tenant",
                display_name="Agent tenant",
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            Principal(
                principal_id=principal_id,
                tenant_id=tenant_id,
                handle="agent-user",
                display_name="Agent user",
                kind=PrincipalKind.USER.value,
                role=PrincipalRole.ANALYST.value,
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            ApiCredential(
                credential_id=credential_id,
                principal_id=principal_id,
                label="agent scheduler test",
                token_digest=b"x" * 32,
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        SqlAlchemyRepositorySet(session).jobs.create(
            AnalysisJobDTO(
                job_id="job-principal",
                name="principal agent test",
                status=JobStatus.READY_FOR_CONFIGURATION,
                created_at=now,
                updated_at=now,
            ),
            tenant_id=tenant_id,
            owner_principal_id=principal_id,
        )
    return PrincipalContext(
        tenant_id=tenant_id,
        principal_id=principal_id,
        credential_id=credential_id,
        kind=PrincipalKind.USER,
        role=PrincipalRole.ANALYST,
        auth_mode=AuthMode.PRINCIPAL,
    )
