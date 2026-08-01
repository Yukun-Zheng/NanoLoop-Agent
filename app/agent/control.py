"""Durable bounded planner-executor loop for scientific tasks."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, selectinload

from app.agent.context_budget import bounded_mapping, bounded_newest_mappings
from app.agent.model_provider import AgentModelProviderError
from app.agent.protocols import AgentDecisionModel
from app.agent.tool_registry import (
    AgentToolRegistry,
    InvalidAgentToolArgumentsError,
    UnknownAgentToolError,
)
from app.analysis.authorization import require_mutation, require_read
from app.contracts.agent_runtime import (
    AgentApprovalDTO,
    AgentApprovalStatus,
    AgentBudget,
    AgentDecision,
    AgentDecisionKind,
    AgentDecisionRequest,
    AgentEventDTO,
    AgentEventType,
    AgentModelIdentity,
    AgentTaskDTO,
    AgentTaskListData,
    AgentTaskStatus,
    AgentToolObservation,
    AgentToolOutcome,
    AgentToolRisk,
    CreateAgentTaskRequest,
    ResolveAgentApprovalRequest,
    SubmitAgentInputRequest,
)
from app.contracts.common import HealthComponent, utc_now
from app.contracts.identity import (
    LEGACY_PRINCIPAL_ID,
    LEGACY_TENANT_ID,
    AuthMode,
    PrincipalContext,
    PrincipalKind,
    PrincipalRole,
)
from app.core.errors import JobStateConflictError, NanoLoopError, ResourceNotFoundError
from app.core.identity import legacy_principal_context
from app.db.models import (
    AgentApproval,
    AgentTask,
    AgentTaskEvent,
    ApiCredential,
    Principal,
    Tenant,
)
from app.db.repositories import SqlAlchemyRepositorySet

SessionFactory = Callable[[], Session]
_TERMINAL_TASK_STATUSES = {
    AgentTaskStatus.COMPLETED.value,
    AgentTaskStatus.FAILED.value,
    AgentTaskStatus.CANCELLED.value,
}


@dataclass(frozen=True, slots=True)
class AgentExecutionContext:
    task_id: str
    job_id: str
    action_id: str
    principal: PrincipalContext


@dataclass(frozen=True, slots=True)
class AgentResumeBatch:
    selected_task_ids: tuple[str, ...]
    resumed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]


class AgentResumeAuthorizationError(RuntimeError):
    """A persisted task creator can no longer authorize autonomous work."""


@dataclass(slots=True)
class _TaskLockEntry:
    lock: threading.RLock
    users: int = 0


class AgentControlService:
    """Coordinate one model decision at a time under deterministic policy."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        model: AgentDecisionModel,
        tools: AgentToolRegistry,
        server_budget: AgentBudget | None = None,
        max_observations: int = 12,
        max_observation_chars: int = 12_000,
    ) -> None:
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        if max_observation_chars < 1_000:
            raise ValueError("max_observation_chars must be at least 1000")
        self._session_factory = session_factory
        self._model = model
        self._tools = tools
        self._server_budget = server_budget or AgentBudget()
        self._max_observations = max_observations
        self._max_observation_chars = max_observation_chars
        self._locks_guard = threading.Lock()
        self._task_locks: dict[str, _TaskLockEntry] = {}

    @property
    def model_identity(self) -> AgentModelIdentity:
        return self._model.identity

    def health(self) -> HealthComponent:
        model_health = self._model.health()
        if model_health.status == "unavailable":
            return model_health
        return HealthComponent(
            status=model_health.status,
            detail=(
                f"bounded agent runtime ready; provider={self._model.identity.provider}; "
                f"model={self._model.identity.model}; tools={len(self._tools.specs())}"
            ),
        )

    def resume_ready_tasks(self, *, limit: int = 20) -> AgentResumeBatch:
        """Resume durable continuations and approved pending actions without a browser."""

        if limit < 1 or limit > 100:
            raise ValueError("resume limit must be between 1 and 100")
        now = utc_now()
        session = self._session_factory()
        try:
            task_ids = tuple(
                session.scalars(
                    select(AgentTask.task_id)
                    .where(
                        or_(
                            and_(
                                AgentTask.status
                                == AgentTaskStatus.WAITING_FOR_EXTERNAL.value,
                                or_(
                                    AgentTask.next_wakeup_at.is_(None),
                                    AgentTask.next_wakeup_at <= now,
                                ),
                            ),
                            and_(
                                AgentTask.status == AgentTaskStatus.RUNNING.value,
                                AgentTask.pending_action_json.is_not(None),
                            ),
                        )
                    )
                    .order_by(AgentTask.updated_at.asc())
                    .limit(limit)
                )
            )
        finally:
            session.close()

        resumed: list[str] = []
        failed: list[str] = []
        for task_id in task_ids:
            try:
                principal = self._principal_for_resume(task_id)
                self.run(task_id, principal=principal)
                resumed.append(task_id)
            except AgentResumeAuthorizationError:
                self._fail_resume_authorization(task_id)
                failed.append(task_id)
            except (JobStateConflictError, ResourceNotFoundError):
                continue
            except Exception:
                failed.append(task_id)
        return AgentResumeBatch(
            selected_task_ids=task_ids,
            resumed_task_ids=tuple(resumed),
            failed_task_ids=tuple(failed),
        )

    def create(
        self,
        job_id: str,
        request: CreateAgentTaskRequest,
        *,
        principal: PrincipalContext,
    ) -> AgentTaskDTO:
        tenant_id, principal_id = _identity(principal)
        now = utc_now()
        effective_budget = AgentBudget(
            max_steps=min(request.budget.max_steps, self._server_budget.max_steps),
            max_failures=min(
                request.budget.max_failures,
                self._server_budget.max_failures,
            ),
            max_auto_steps_per_run=min(
                request.budget.max_auto_steps_per_run,
                self._server_budget.max_auto_steps_per_run,
            ),
        )
        task = AgentTask(
            task_id=f"agt_{uuid4().hex}",
            tenant_id=tenant_id,
            job_id=job_id,
            created_by=principal_id,
            created_credential_id=principal.credential_id,
            auth_mode=principal.auth_mode.value,
            goal=request.goal.strip(),
            status=AgentTaskStatus.CREATED.value,
            plan_json=[],
            budget_json=effective_budget.model_dump(mode="json"),
            context_json=request.context,
            observations_json=[],
            user_inputs_json=[],
            model_provider=self._model.identity.provider,
            model_name=self._model.identity.model,
            step_count=0,
            failure_count=0,
            version=1,
            created_at=now,
            updated_at=now,
        )
        session = self._session_factory()
        try:
            scope = SqlAlchemyRepositorySet(session).jobs.get_scope(
                job_id,
                tenant_id=tenant_id,
            )
            require_mutation(principal, scope)
            session.add(task)
            session.flush()
            self._append_event(
                task,
                AgentEventType.TASK_CREATED,
                "已创建科研 Agent 任务。",
                {
                    "goal": task.goal,
                    "budget": task.budget_json,
                    "model": {
                        "provider": task.model_provider,
                        "model": task.model_name,
                    },
                },
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        if request.auto_start:
            return self.run(task.task_id, principal=principal)
        return self.get(task.task_id, principal=principal)

    def list(
        self,
        job_id: str,
        *,
        principal: PrincipalContext,
    ) -> AgentTaskListData:
        tenant_id, _ = _identity(principal)
        session = self._session_factory()
        try:
            scope = SqlAlchemyRepositorySet(session).jobs.get_scope(
                job_id,
                tenant_id=tenant_id,
            )
            require_read(principal, scope)
            tasks = list(
                session.scalars(
                    select(AgentTask)
                    .where(
                        AgentTask.job_id == job_id,
                        AgentTask.tenant_id == tenant_id,
                    )
                    .options(
                        selectinload(AgentTask.events),
                        selectinload(AgentTask.approvals),
                    )
                    .order_by(AgentTask.updated_at.desc())
                    .limit(50)
                )
            )
            return AgentTaskListData(tasks=[_task_dto(task) for task in tasks])
        finally:
            session.close()

    def get(
        self,
        task_id: str,
        *,
        principal: PrincipalContext,
    ) -> AgentTaskDTO:
        session = self._session_factory()
        try:
            task = self._load_task(session, task_id, principal=principal, mutation=False)
            return _task_dto(task)
        finally:
            session.close()

    def run(
        self,
        task_id: str,
        *,
        principal: PrincipalContext,
        max_steps: int | None = None,
    ) -> AgentTaskDTO:
        """Advance until a boundary: approval, input, async work, completion, or call budget."""

        with self._task_lock(task_id):
            current = self.get(task_id, principal=principal)
            if current.status in {
                AgentTaskStatus.WAITING_FOR_APPROVAL,
                AgentTaskStatus.WAITING_FOR_INPUT,
            }:
                return current
            if current.status in {
                AgentTaskStatus.COMPLETED,
                AgentTaskStatus.FAILED,
                AgentTaskStatus.CANCELLED,
            }:
                raise JobStateConflictError(
                    "Agent 任务已进入终态",
                    details={"task_id": task_id, "status": current.status.value},
                )
            if current.status is AgentTaskStatus.WAITING_FOR_EXTERNAL:
                if (
                    current.next_wakeup_at is not None
                    and _utc_datetime(current.next_wakeup_at) > utc_now()
                ):
                    return current
                observation = self._execute_pending_action(task_id, principal=principal)
                current = self.get(task_id, principal=principal)
                if (
                    observation.continuation_tool is not None
                    or current.status is not AgentTaskStatus.RUNNING
                ):
                    return current
            elif current.pending_action is not None:
                observation = self._execute_pending_action(task_id, principal=principal)
                current = self.get(task_id, principal=principal)
                if (
                    observation.continuation_tool is not None
                    or current.status is not AgentTaskStatus.RUNNING
                ):
                    return current
            limit = max_steps or current.budget.max_auto_steps_per_run
            limit = min(limit, current.budget.max_auto_steps_per_run)
            for _ in range(limit):
                stop = self._advance_once(task_id, principal=principal)
                current = self.get(task_id, principal=principal)
                if stop or current.status is not AgentTaskStatus.RUNNING:
                    break
            return current

    def submit_input(
        self,
        task_id: str,
        request: SubmitAgentInputRequest,
        *,
        principal: PrincipalContext,
    ) -> AgentTaskDTO:
        with self._task_lock(task_id):
            session = self._session_factory()
            try:
                task = self._load_task(session, task_id, principal=principal, mutation=True)
                if task.status != AgentTaskStatus.WAITING_FOR_INPUT.value:
                    raise JobStateConflictError(
                        "Agent 当前未等待用户输入",
                        details={"task_id": task_id, "status": task.status},
                    )
                inputs = [*task.user_inputs_json, request.content.strip()][-8:]
                task.user_inputs_json = inputs
                task.waiting_question = None
                task.next_wakeup_at = None
                task.status = AgentTaskStatus.RUNNING.value
                task.error = None
                self._touch(task)
                self._append_event(
                    task,
                    AgentEventType.USER_INPUT_RECEIVED,
                    "已收到继续任务所需的用户输入。",
                    {"content": request.content.strip()},
                )
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
            return self.get(task_id, principal=principal)

    def resolve_approval(
        self,
        task_id: str,
        approval_id: str,
        request: ResolveAgentApprovalRequest,
        *,
        principal: PrincipalContext,
    ) -> AgentTaskDTO:
        with self._task_lock(task_id):
            session = self._session_factory()
            execute_action = False
            try:
                task = self._load_task(session, task_id, principal=principal, mutation=True)
                approval = next(
                    (
                        item
                        for item in task.approvals
                        if item.approval_id == approval_id
                    ),
                    None,
                )
                if approval is None:
                    raise ResourceNotFoundError(
                        details={"resource": "agent_approval", "approval_id": approval_id}
                    )
                if (
                    task.status != AgentTaskStatus.WAITING_FOR_APPROVAL.value
                    or approval.status != AgentApprovalStatus.PENDING.value
                    or task.pending_action_json is None
                    or task.pending_action_json.get("action_id") != approval.action_id
                ):
                    raise JobStateConflictError(
                        "Agent 审批已处理或不再是当前动作",
                        details={"task_id": task_id, "approval_id": approval_id},
                    )
                now = utc_now()
                approval.status = (
                    AgentApprovalStatus.APPROVED.value
                    if request.decision == "approve"
                    else AgentApprovalStatus.REJECTED.value
                )
                approval.resolved_at = now
                approval.resolved_by = _identity(principal)[1]
                approval.comment = request.comment
                self._append_event(
                    task,
                    AgentEventType.APPROVAL_RESOLVED,
                    "用户已批准待执行动作。"
                    if request.decision == "approve"
                    else "用户已拒绝待执行动作。",
                    {
                        "approval_id": approval.approval_id,
                        "action_id": approval.action_id,
                        "decision": request.decision,
                        "comment": request.comment,
                    },
                )
                if request.decision == "approve":
                    task.next_wakeup_at = None
                    task.status = AgentTaskStatus.RUNNING.value
                    execute_action = True
                else:
                    self._record_observation(
                        task,
                        {
                            "action_id": approval.action_id,
                            "tool_name": approval.tool_name,
                            "outcome": AgentToolOutcome.ERROR.value,
                            "summary": "用户拒绝了该动作；需要调整计划或结束任务。",
                            "data": {"rejected": True, "comment": request.comment},
                            "retryable": True,
                            "evidence_refs": [],
                        },
                    )
                    task.pending_action_json = None
                    task.next_wakeup_at = None
                    task.status = AgentTaskStatus.RUNNING.value
                task.error = None
                self._touch(task)
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
            if execute_action:
                self._execute_pending_action(task_id, principal=principal)
            return self.get(task_id, principal=principal)

    def cancel(
        self,
        task_id: str,
        *,
        principal: PrincipalContext,
    ) -> AgentTaskDTO:
        with self._task_lock(task_id):
            session = self._session_factory()
            try:
                task = self._load_task(session, task_id, principal=principal, mutation=True)
                if task.status in _TERMINAL_TASK_STATUSES:
                    raise JobStateConflictError(
                        "Agent 任务已进入终态",
                        details={"task_id": task_id, "status": task.status},
                    )
                task.status = AgentTaskStatus.CANCELLED.value
                task.pending_action_json = None
                task.next_wakeup_at = None
                task.waiting_question = None
                self._touch(task)
                self._append_event(
                    task,
                    AgentEventType.TASK_CANCELLED,
                    "用户已取消 Agent 任务。",
                    {},
                )
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
            return self.get(task_id, principal=principal)

    def _advance_once(
        self,
        task_id: str,
        *,
        principal: PrincipalContext,
    ) -> bool:
        snapshot = self.get(task_id, principal=principal)
        if snapshot.step_count >= snapshot.budget.max_steps:
            self._fail_for_budget(task_id, principal=principal)
            return True
        request = AgentDecisionRequest(
            task_id=snapshot.task_id,
            job_id=snapshot.job_id,
            goal=snapshot.goal,
            task_context=bounded_mapping(
                snapshot.context,
                self._max_observation_chars,
            ),
            plan=snapshot.plan,
            current_step=snapshot.current_step,
            step_count=snapshot.step_count,
            remaining_steps=snapshot.budget.max_steps - snapshot.step_count,
            failure_count=snapshot.failure_count,
            latest_observations=bounded_newest_mappings(
                snapshot.latest_observations,
                max_items=self._max_observations,
                max_chars=self._max_observation_chars,
            ),
            user_inputs=snapshot.user_inputs,
            tools=self._tools.specs(),
        )
        try:
            decision = self._model.decide(request)
        except AgentModelProviderError as error:
            self._record_model_failure(task_id, str(error), principal=principal)
            return True
        return self._apply_decision(task_id, decision, principal=principal)

    def _apply_decision(
        self,
        task_id: str,
        decision: AgentDecision,
        *,
        principal: PrincipalContext,
    ) -> bool:
        session = self._session_factory()
        execute_action = False
        stop = False
        try:
            task = self._load_task(session, task_id, principal=principal, mutation=True)
            if task.status not in {
                AgentTaskStatus.CREATED.value,
                AgentTaskStatus.RUNNING.value,
            }:
                raise JobStateConflictError(
                    "Agent 当前状态不允许模型继续决策",
                    details={"task_id": task_id, "status": task.status},
                )
            task.status = AgentTaskStatus.RUNNING.value
            task.model_provider = self._model.identity.provider
            task.model_name = self._model.identity.model
            task.plan_json = decision.plan
            task.current_step = decision.current_step
            task.step_count += 1
            task.error = None
            self._touch(task)
            self._append_event(
                task,
                AgentEventType.MODEL_DECISION,
                decision.rationale_summary,
                {
                    "kind": decision.kind.value,
                    "plan": decision.plan,
                    "current_step": decision.current_step,
                    "tool_name": decision.tool_name,
                    "model": self._model.identity.model_dump(mode="json"),
                },
            )
            if decision.kind is AgentDecisionKind.CALL_TOOL:
                execute_action, stop = self._prepare_tool_action(task, decision)
            elif decision.kind is AgentDecisionKind.ASK_USER:
                task.status = AgentTaskStatus.WAITING_FOR_INPUT.value
                task.next_wakeup_at = None
                task.waiting_question = decision.user_question
                stop = True
            elif decision.kind is AgentDecisionKind.FINISH:
                invalid_evidence = _invalid_completion_evidence(task, decision)
                if invalid_evidence is not None:
                    self._record_failure(
                        task,
                        invalid_evidence,
                        retryable=True,
                    )
                else:
                    task.status = AgentTaskStatus.COMPLETED.value
                    task.next_wakeup_at = None
                    task.final_answer = decision.final_answer
                    task.final_evidence_refs_json = decision.final_evidence_refs
                    self._append_event(
                        task,
                        AgentEventType.TASK_COMPLETED,
                        "Agent 已根据可见工具证据完成任务。",
                        {
                            "final_answer": decision.final_answer,
                            "evidence_refs": decision.final_evidence_refs,
                        },
                    )
                stop = True
            else:
                task.status = AgentTaskStatus.FAILED.value
                task.next_wakeup_at = None
                task.error = decision.failure_reason
                self._append_event(
                    task,
                    AgentEventType.TASK_FAILED,
                    "Agent 判断任务无法在当前约束下安全完成。",
                    {"reason": decision.failure_reason},
                )
                stop = True
            session.commit()
        except (UnknownAgentToolError, InvalidAgentToolArgumentsError) as error:
            session.rollback()
            self._record_model_failure(task_id, str(error), principal=principal)
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        if execute_action:
            observation = self._execute_pending_action(task_id, principal=principal)
            return observation.suggested_poll_after_seconds is not None
        return stop

    def _prepare_tool_action(
        self,
        task: AgentTask,
        decision: AgentDecision,
    ) -> tuple[bool, bool]:
        tool_name = decision.tool_name
        if tool_name is None:
            raise InvalidAgentToolArgumentsError("tool name is required")
        registered, arguments = self._tools.validate_arguments(
            tool_name,
            decision.tool_arguments,
        )
        action_id = f"act_{uuid4().hex}"
        pending = {
            "action_id": action_id,
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "rationale_summary": decision.rationale_summary,
        }
        task.pending_action_json = pending
        task.next_wakeup_at = None
        if registered.tool.spec.requires_approval:
            approval = AgentApproval(
                approval_id=f"apr_{uuid4().hex}",
                task_id=task.task_id,
                action_id=action_id,
                tool_name=tool_name,
                tool_arguments_json=arguments,
                status=AgentApprovalStatus.PENDING.value,
                reason=decision.rationale_summary,
                requested_at=utc_now(),
            )
            task.approvals.append(approval)
            task.status = AgentTaskStatus.WAITING_FOR_APPROVAL.value
            self._append_event(
                task,
                AgentEventType.APPROVAL_REQUESTED,
                f"动作 {tool_name} 需要人工批准。",
                {
                    "approval_id": approval.approval_id,
                    "action_id": action_id,
                    "tool_name": tool_name,
                    "tool_arguments": arguments,
                },
            )
            return False, True
        return True, False

    def _execute_pending_action(
        self,
        task_id: str,
        *,
        principal: PrincipalContext,
    ) -> AgentToolObservation:
        session = self._session_factory()
        try:
            task = self._load_task(session, task_id, principal=principal, mutation=True)
            pending = dict(task.pending_action_json or {})
            action_id = str(pending.get("action_id") or "")
            tool_name = str(pending.get("tool_name") or "")
            arguments = pending.get("tool_arguments")
            if not action_id or not tool_name or not isinstance(arguments, dict):
                raise JobStateConflictError(
                    "Agent 没有可执行的待处理动作",
                    details={"task_id": task_id},
                )
            execution_started = pending.get("execution_started") is True
            try:
                idempotent = self._tools.get(tool_name).tool.spec.idempotent
            except UnknownAgentToolError:
                idempotent = False
            if execution_started and not idempotent:
                observation = AgentToolObservation(
                    outcome=AgentToolOutcome.ERROR,
                    summary=(
                        "该非幂等动作曾开始执行但没有留下完成记录；"
                        "为避免重复写入，任务已停止并等待人工核对现有制品或运行。"
                    ),
                    data={
                        "action_id": action_id,
                        "tool_name": tool_name,
                        "reason": "ambiguous_non_idempotent_outcome",
                    },
                    retryable=False,
                )
                self._record_observation(
                    task,
                    {
                        "action_id": action_id,
                        "tool_name": tool_name,
                        **observation.model_dump(mode="json"),
                    },
                )
                task.pending_action_json = None
                task.next_wakeup_at = None
                self._record_failure(task, observation.summary, retryable=False)
                self._touch(task)
                self._append_event(
                    task,
                    AgentEventType.TOOL_FAILED,
                    observation.summary,
                    {
                        "action_id": action_id,
                        "tool_name": tool_name,
                        "reason": "ambiguous_non_idempotent_outcome",
                    },
                )
                self._append_event(
                    task,
                    AgentEventType.TASK_FAILED,
                    "Agent 因非幂等动作执行结果不明确而安全停止。",
                    {"action_id": action_id, "tool_name": tool_name},
                )
                session.commit()
                return observation
            execution_attempt = int(pending.get("execution_attempts") or 0) + 1
            pending["execution_started"] = True
            pending["execution_attempts"] = execution_attempt
            task.pending_action_json = pending
            task.next_wakeup_at = None
            self._append_event(
                task,
                AgentEventType.TOOL_STARTED,
                f"开始执行工具 {tool_name}。",
                {
                    "action_id": action_id,
                    "tool_name": tool_name,
                    "execution_attempt": execution_attempt,
                    "idempotent": idempotent,
                },
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        context = AgentExecutionContext(
            task_id=task_id,
            job_id=task.job_id,
            action_id=action_id,
            principal=principal,
        )
        try:
            observation = self._tools.execute(
                tool_name,
                context=context,
                arguments=arguments,
            )
        except NanoLoopError as error:
            observation = AgentToolObservation(
                outcome=AgentToolOutcome.ERROR,
                summary=error.message,
                data={"error_code": error.code, "details": error.details},
                retryable=error.retryable,
            )
        except (UnknownAgentToolError, InvalidAgentToolArgumentsError) as error:
            observation = AgentToolObservation(
                outcome=AgentToolOutcome.ERROR,
                summary="工具名称或参数没有通过运行时校验。",
                data={"error_type": type(error).__name__},
                retryable=True,
            )
        except Exception as error:
            observation = AgentToolObservation(
                outcome=AgentToolOutcome.ERROR,
                summary="工具执行发生未预期错误，未继续扩大操作范围。",
                data={"error_type": type(error).__name__},
                retryable=False,
            )

        session = self._session_factory()
        try:
            task = self._load_task(session, task_id, principal=principal, mutation=True)
            current = dict(task.pending_action_json or {})
            if current.get("action_id") != action_id:
                raise JobStateConflictError(
                    "Agent 待处理动作已改变",
                    details={"task_id": task_id, "action_id": action_id},
                )
            self._record_observation(
                task,
                {
                    "action_id": action_id,
                    "tool_name": tool_name,
                    **observation.model_dump(mode="json"),
                },
            )
            task.pending_action_json = None
            task.next_wakeup_at = None
            if observation.outcome is AgentToolOutcome.ERROR:
                self._record_failure(task, observation.summary, retryable=observation.retryable)
                event_type = AgentEventType.TOOL_FAILED
            else:
                self._prepare_continuation(task, observation)
                task.error = None
                event_type = AgentEventType.TOOL_COMPLETED
            self._touch(task)
            self._append_event(
                task,
                event_type,
                observation.summary,
                {
                    "action_id": action_id,
                    "tool_name": tool_name,
                    "observation": observation.model_dump(mode="json"),
                },
            )
            if task.status == AgentTaskStatus.FAILED.value:
                self._append_event(
                    task,
                    AgentEventType.TASK_FAILED,
                    "Agent 因工具失败且无法安全重试而停止。",
                    {"action_id": action_id, "tool_name": tool_name},
                )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return observation

    def _prepare_continuation(
        self,
        task: AgentTask,
        observation: AgentToolObservation,
    ) -> None:
        if observation.continuation_tool is None:
            task.status = AgentTaskStatus.RUNNING.value
            task.next_wakeup_at = None
            return
        registered, arguments = self._tools.validate_arguments(
            observation.continuation_tool,
            observation.continuation_arguments,
        )
        if (
            registered.tool.spec.risk is not AgentToolRisk.READ_ONLY
            or registered.tool.spec.requires_approval
        ):
            raise RuntimeError("automatic continuation must use a read-only tool")
        task.pending_action_json = {
            "action_id": f"act_{uuid4().hex}",
            "tool_name": observation.continuation_tool,
            "tool_arguments": arguments,
            "rationale_summary": "等待异步科研工具完成后自动复查状态。",
            "automatic_continuation": True,
            "poll_after_seconds": observation.suggested_poll_after_seconds,
        }
        poll_after = observation.suggested_poll_after_seconds
        if poll_after is None:
            raise RuntimeError("automatic continuation requires a poll delay")
        task.next_wakeup_at = utc_now() + timedelta(seconds=poll_after)
        task.status = AgentTaskStatus.WAITING_FOR_EXTERNAL.value

    def _record_model_failure(
        self,
        task_id: str,
        reason: str,
        *,
        principal: PrincipalContext,
    ) -> None:
        session = self._session_factory()
        try:
            task = self._load_task(session, task_id, principal=principal, mutation=True)
            self._record_failure(task, reason, retryable=True)
            self._append_event(
                task,
                AgentEventType.TASK_FAILED
                if task.status == AgentTaskStatus.FAILED.value
                else AgentEventType.MODEL_DECISION,
                "决策模型本轮返回不可用结果，任务已安全停止本次推进。",
                {
                    "error_type": "agent_model_error",
                    "model": self._model.identity.model_dump(mode="json"),
                },
            )
            self._touch(task)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _record_failure(
        self,
        task: AgentTask,
        reason: str,
        *,
        retryable: bool,
    ) -> None:
        task.failure_count += 1
        budget = AgentBudget.model_validate(task.budget_json)
        terminal = not retryable or task.failure_count > budget.max_failures
        task.status = (
            AgentTaskStatus.FAILED.value if terminal else AgentTaskStatus.RUNNING.value
        )
        if terminal:
            task.pending_action_json = None
            task.next_wakeup_at = None
        task.error = reason

    def _fail_for_budget(
        self,
        task_id: str,
        *,
        principal: PrincipalContext,
    ) -> None:
        session = self._session_factory()
        try:
            task = self._load_task(session, task_id, principal=principal, mutation=True)
            task.status = AgentTaskStatus.FAILED.value
            task.pending_action_json = None
            task.next_wakeup_at = None
            task.error = "已达到 Agent 最大步骤数，任务停止。"
            self._touch(task)
            self._append_event(
                task,
                AgentEventType.TASK_FAILED,
                "已达到 Agent 最大步骤数，未继续执行。",
                {"max_steps": AgentBudget.model_validate(task.budget_json).max_steps},
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _record_observation(self, task: AgentTask, observation: dict[str, Any]) -> None:
        task.observations_json = [
            *task.observations_json,
            observation,
        ][-self._max_observations :]

    def _principal_for_resume(self, task_id: str) -> PrincipalContext:
        session = self._session_factory()
        try:
            task = session.get(AgentTask, task_id)
            if task is None:
                raise ResourceNotFoundError(
                    details={"resource": "agent_task", "task_id": task_id}
                )
            try:
                auth_mode = AuthMode(task.auth_mode)
            except ValueError as error:
                raise AgentResumeAuthorizationError(
                    "persisted task authentication mode is invalid"
                ) from error
            if auth_mode is not AuthMode.PRINCIPAL:
                if (
                    task.tenant_id != LEGACY_TENANT_ID
                    or task.created_by != LEGACY_PRINCIPAL_ID
                    or task.created_credential_id is not None
                ):
                    raise AgentResumeAuthorizationError(
                        "compatibility task identity is invalid"
                    )
                return legacy_principal_context(auth_mode)

            credential_id = task.created_credential_id
            if credential_id is None:
                raise AgentResumeAuthorizationError(
                    "principal task has no originating credential"
                )
            credential = session.get(ApiCredential, credential_id)
            creator = session.get(Principal, task.created_by)
            tenant = session.get(Tenant, task.tenant_id)
            if (
                credential is None
                or creator is None
                or tenant is None
                or credential.principal_id != creator.principal_id
                or creator.tenant_id != tenant.tenant_id
            ):
                raise AgentResumeAuthorizationError(
                    "persisted task identity is no longer resolvable"
                )
            now = utc_now()
            if (
                not credential.enabled
                or credential.revoked_at is not None
                or (
                    credential.expires_at is not None
                    and _utc_datetime(credential.expires_at) <= now
                )
                or not creator.enabled
                or not tenant.enabled
            ):
                raise AgentResumeAuthorizationError(
                    "persisted task identity is no longer active"
                )
            return PrincipalContext(
                tenant_id=tenant.tenant_id,
                principal_id=creator.principal_id,
                credential_id=credential.credential_id,
                kind=PrincipalKind(creator.kind),
                role=PrincipalRole(creator.role),
                auth_mode=AuthMode.PRINCIPAL,
            )
        finally:
            session.close()

    def _fail_resume_authorization(self, task_id: str) -> None:
        with self._task_lock(task_id):
            session = self._session_factory()
            try:
                task = session.scalar(
                    select(AgentTask)
                    .where(AgentTask.task_id == task_id)
                    .options(selectinload(AgentTask.events))
                )
                if task is None or task.status in _TERMINAL_TASK_STATUSES:
                    return
                task.status = AgentTaskStatus.FAILED.value
                task.pending_action_json = None
                task.next_wakeup_at = None
                task.error = (
                    "任务创建者的授权已失效，Agent 未继续执行任何自动动作。"
                )
                self._touch(task)
                self._append_event(
                    task,
                    AgentEventType.TASK_FAILED,
                    "Agent 自动续跑因创建者授权失效而停止。",
                    {"reason": "resume_authorization_inactive"},
                )
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def _load_task(
        self,
        session: Session,
        task_id: str,
        *,
        principal: PrincipalContext,
        mutation: bool,
    ) -> AgentTask:
        tenant_id, _ = _identity(principal)
        task = session.scalar(
            select(AgentTask)
            .where(
                AgentTask.task_id == task_id,
                AgentTask.tenant_id == tenant_id,
            )
            .options(
                selectinload(AgentTask.events),
                selectinload(AgentTask.approvals),
            )
        )
        if task is None:
            raise ResourceNotFoundError(
                details={"resource": "agent_task", "task_id": task_id}
            )
        scope = SqlAlchemyRepositorySet(session).jobs.get_scope(
            task.job_id,
            tenant_id=tenant_id,
        )
        if mutation:
            require_mutation(principal, scope)
        else:
            require_read(principal, scope)
        return task

    @staticmethod
    def _append_event(
        task: AgentTask,
        event_type: AgentEventType,
        summary: str,
        payload: dict[str, Any],
    ) -> None:
        sequence = max((event.sequence for event in task.events), default=0) + 1
        task.events.append(
            AgentTaskEvent(
                task_id=task.task_id,
                sequence=sequence,
                event_type=event_type.value,
                summary=summary,
                payload_json=payload,
                created_at=utc_now(),
            )
        )

    @staticmethod
    def _touch(task: AgentTask) -> None:
        task.version += 1
        task.updated_at = utc_now()

    @contextmanager
    def _task_lock(self, task_id: str) -> Iterator[None]:
        with self._locks_guard:
            entry = self._task_locks.setdefault(
                task_id,
                _TaskLockEntry(lock=threading.RLock()),
            )
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._locks_guard:
                entry.users -= 1
                if entry.users == 0 and self._task_locks.get(task_id) is entry:
                    del self._task_locks[task_id]


def _identity(principal: PrincipalContext) -> tuple[str, str]:
    if principal.tenant_id is None or principal.principal_id is None:
        raise ValueError("agent control requires tenant and principal IDs")
    return principal.tenant_id, principal.principal_id


def _task_dto(task: AgentTask) -> AgentTaskDTO:
    model = (
        AgentModelIdentity(provider=task.model_provider, model=task.model_name)
        if task.model_provider and task.model_name
        else None
    )
    return AgentTaskDTO(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        job_id=task.job_id,
        created_by=task.created_by,
        goal=task.goal,
        status=AgentTaskStatus(task.status),
        plan=list(task.plan_json),
        current_step=task.current_step,
        step_count=task.step_count,
        failure_count=task.failure_count,
        budget=AgentBudget.model_validate(task.budget_json),
        context=dict(task.context_json),
        latest_observations=list(task.observations_json),
        user_inputs=list(task.user_inputs_json),
        pending_action=(
            dict(task.pending_action_json) if task.pending_action_json is not None else None
        ),
        next_wakeup_at=task.next_wakeup_at,
        waiting_question=task.waiting_question,
        final_answer=task.final_answer,
        final_evidence_refs=list(task.final_evidence_refs_json),
        error=task.error,
        model=model,
        version=task.version,
        created_at=task.created_at,
        updated_at=task.updated_at,
        events=[
            AgentEventDTO(
                event_id=event.event_id,
                sequence=event.sequence,
                event_type=AgentEventType(event.event_type),
                summary=event.summary,
                payload=dict(event.payload_json),
                created_at=event.created_at,
            )
            for event in task.events
        ],
        approvals=[
            AgentApprovalDTO(
                approval_id=approval.approval_id,
                task_id=approval.task_id,
                action_id=approval.action_id,
                tool_name=approval.tool_name,
                tool_arguments=dict(approval.tool_arguments_json),
                status=AgentApprovalStatus(approval.status),
                reason=approval.reason,
                requested_at=approval.requested_at,
                resolved_at=approval.resolved_at,
                resolved_by=approval.resolved_by,
                comment=approval.comment,
            )
            for approval in task.approvals
        ],
    )


def _invalid_completion_evidence(
    task: AgentTask,
    decision: AgentDecision,
) -> str | None:
    available = {
        str(reference)
        for observation in task.observations_json
        for reference in observation.get("evidence_refs", [])
        if reference
    }
    if not available:
        return "模型试图在没有任何工具证据时结束任务。"
    unknown = sorted(set(decision.final_evidence_refs) - available)
    if unknown:
        return "模型引用了当前工具观察中不存在的证据，任务未完成。"
    return None


def _utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
