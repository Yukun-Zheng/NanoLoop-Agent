"""Stable contracts for the bounded scientific-agent control plane."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from app.contracts.common import ContractModel


class AgentTaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_EXTERNAL = "waiting_for_external"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentDecisionKind(StrEnum):
    CALL_TOOL = "call_tool"
    ASK_USER = "ask_user"
    FINISH = "finish"
    FAIL = "fail"


class AgentToolRisk(StrEnum):
    READ_ONLY = "read_only"
    CONTROLLED_WRITE = "controlled_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class AgentToolTransport(StrEnum):
    """Execution boundary advertised by a tool adapter."""

    PYTHON = "python"
    HTTP = "http"
    MCP = "mcp"


class AgentToolOutcome(StrEnum):
    OK = "ok"
    ERROR = "error"


class AgentApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AgentEventType(StrEnum):
    TASK_CREATED = "task.created"
    MODEL_DECISION = "model.decision"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    USER_INPUT_RECEIVED = "user_input.received"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"


class AgentBudget(ContractModel):
    """Hard runtime limits; the model cannot increase them."""

    max_steps: int = Field(default=12, ge=1, le=64)
    max_failures: int = Field(default=3, ge=0, le=16)
    max_auto_steps_per_run: int = Field(default=4, ge=1, le=16)


class AgentToolSpec(ContractModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str = Field(min_length=1, max_length=1000)
    input_schema: dict[str, Any]
    transport: AgentToolTransport = AgentToolTransport.PYTHON
    risk: AgentToolRisk
    requires_approval: bool
    idempotent: bool

    @model_validator(mode="after")
    def write_tools_require_approval(self) -> Self:
        if self.risk is not AgentToolRisk.READ_ONLY and not self.requires_approval:
            raise ValueError("write and external-side-effect tools must require approval")
        return self


class AgentToolObservation(ContractModel):
    outcome: AgentToolOutcome
    summary: str = Field(min_length=1, max_length=4000)
    data: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    suggested_poll_after_seconds: int | None = Field(default=None, ge=1, le=300)
    continuation_tool: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{1,63}$",
    )
    continuation_arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_continuation_shape(self) -> Self:
        if self.continuation_tool is None and self.continuation_arguments:
            raise ValueError("continuation arguments require continuation_tool")
        if self.continuation_tool is not None and self.suggested_poll_after_seconds is None:
            raise ValueError("continuation_tool requires suggested_poll_after_seconds")
        return self


class AgentDecision(ContractModel):
    """One model-selected next action without hidden chain-of-thought."""

    kind: AgentDecisionKind
    plan: list[str] = Field(min_length=1, max_length=20)
    current_step: str = Field(min_length=1, max_length=500)
    rationale_summary: str = Field(min_length=1, max_length=800)
    tool_name: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{1,63}$",
    )
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    user_question: str | None = Field(default=None, min_length=1, max_length=2000)
    final_answer: str | None = Field(default=None, min_length=1, max_length=8000)
    final_evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    failure_reason: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> Self:
        if any(not item.strip() or len(item) > 500 for item in self.plan):
            raise ValueError("plan items must contain 1 to 500 visible characters")
        if self.kind is AgentDecisionKind.CALL_TOOL:
            if self.tool_name is None:
                raise ValueError("call_tool requires tool_name")
            if (
                self.user_question
                or self.final_answer
                or self.final_evidence_refs
                or self.failure_reason
            ):
                raise ValueError("call_tool cannot carry terminal or user-input fields")
        elif self.kind is AgentDecisionKind.ASK_USER:
            if not self.user_question:
                raise ValueError("ask_user requires user_question")
            if (
                self.tool_name
                or self.tool_arguments
                or self.final_answer
                or self.final_evidence_refs
                or self.failure_reason
            ):
                raise ValueError("ask_user must contain only its user question")
        elif self.kind is AgentDecisionKind.FINISH:
            if not self.final_answer:
                raise ValueError("finish requires final_answer")
            if not self.final_evidence_refs:
                raise ValueError("finish requires at least one evidence reference")
            if self.tool_name or self.tool_arguments or self.user_question or self.failure_reason:
                raise ValueError("finish must contain only its final answer")
        else:
            if not self.failure_reason:
                raise ValueError("fail requires failure_reason")
            if (
                self.tool_name
                or self.tool_arguments
                or self.user_question
                or self.final_answer
                or self.final_evidence_refs
            ):
                raise ValueError("fail must contain only its failure reason")
        return self


class AgentModelIdentity(ContractModel):
    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=255)


class AgentDecisionRequest(ContractModel):
    """Bounded state visible to a decision model."""

    task_id: str
    job_id: str
    goal: str
    task_context: dict[str, Any] = Field(default_factory=dict)
    plan: list[str] = Field(default_factory=list, max_length=20)
    current_step: str | None = None
    step_count: int = Field(ge=0)
    remaining_steps: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    latest_observations: list[dict[str, Any]] = Field(default_factory=list, max_length=12)
    user_inputs: list[str] = Field(default_factory=list, max_length=8)
    tools: list[AgentToolSpec] = Field(min_length=1, max_length=64)


class CreateAgentTaskRequest(ContractModel):
    goal: str = Field(min_length=3, max_length=4000)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    context: dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True

    @model_validator(mode="after")
    def bound_persisted_context(self) -> Self:
        try:
            encoded = json.dumps(
                self.context,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except TypeError as error:
            raise ValueError("agent task context must be JSON serializable") from error
        if len(encoded) > 20_000:
            raise ValueError("agent task context exceeds 20000 characters")
        return self


class ResolveAgentApprovalRequest(ContractModel):
    decision: Literal["approve", "reject"]
    comment: str | None = Field(default=None, max_length=1000)


class SubmitAgentInputRequest(ContractModel):
    content: str = Field(min_length=1, max_length=4000)


class AgentRunRequest(ContractModel):
    max_steps: int | None = Field(default=None, ge=1, le=16)


class AgentEventDTO(ContractModel):
    event_id: int = Field(ge=1)
    sequence: int = Field(ge=1)
    event_type: AgentEventType
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AgentApprovalDTO(ContractModel):
    approval_id: str
    task_id: str
    action_id: str
    tool_name: str
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    status: AgentApprovalStatus
    reason: str
    requested_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    comment: str | None = None


class AgentTaskDTO(ContractModel):
    task_id: str
    tenant_id: str
    job_id: str
    created_by: str
    goal: str
    status: AgentTaskStatus
    plan: list[str] = Field(default_factory=list)
    current_step: str | None = None
    step_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    budget: AgentBudget
    context: dict[str, Any] = Field(default_factory=dict)
    latest_observations: list[dict[str, Any]] = Field(default_factory=list)
    user_inputs: list[str] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    next_wakeup_at: datetime | None = None
    waiting_question: str | None = None
    final_answer: str | None = None
    final_evidence_refs: list[str] = Field(default_factory=list)
    error: str | None = None
    model: AgentModelIdentity | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    events: list[AgentEventDTO] = Field(default_factory=list)
    approvals: list[AgentApprovalDTO] = Field(default_factory=list)


class AgentTaskListData(ContractModel):
    tasks: list[AgentTaskDTO] = Field(default_factory=list)
