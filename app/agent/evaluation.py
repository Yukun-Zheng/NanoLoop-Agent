"""Deterministic evaluation harness for small local Agent controllers."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any

from pydantic import Field

from app.agent.model_provider import (
    AGENT_CONTROLLER_PROMPT_ID,
    AGENT_CONTROLLER_SYSTEM_PROMPT,
    AgentModelProviderError,
)
from app.agent.protocols import AgentDecisionModel
from app.agent.scientific_tools import (
    scientific_tool_arguments_are_valid,
    scientific_tool_specs,
)
from app.contracts.agent_runtime import (
    AgentDecision,
    AgentDecisionKind,
    AgentDecisionRequest,
    AgentModelIdentity,
    AgentToolRisk,
)
from app.contracts.common import ContractModel, utc_now


class AgentEvaluationExpectation(ContractModel):
    allowed_kinds: list[AgentDecisionKind] = Field(min_length=1)
    allowed_tool_names: list[str] = Field(default_factory=list)
    forbidden_tool_names: list[str] = Field(default_factory=list)
    required_evidence_refs: list[str] = Field(default_factory=list)


class AgentEvaluationCase(ContractModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    description: str = Field(min_length=1, max_length=500)
    goal: str = Field(min_length=3, max_length=4000)
    task_context: dict[str, Any] = Field(default_factory=dict)
    plan: list[str] = Field(default_factory=list)
    current_step: str | None = None
    step_count: int = Field(default=0, ge=0)
    remaining_steps: int = Field(default=8, ge=0)
    failure_count: int = Field(default=0, ge=0)
    latest_observations: list[dict[str, Any]] = Field(default_factory=list)
    user_inputs: list[str] = Field(default_factory=list)
    expected: AgentEvaluationExpectation

    def request(self) -> AgentDecisionRequest:
        return AgentDecisionRequest(
            task_id=f"agt_eval_{self.case_id}",
            job_id=f"job_eval_{self.case_id}",
            goal=self.goal,
            task_context=self.task_context,
            plan=self.plan,
            current_step=self.current_step,
            step_count=self.step_count,
            remaining_steps=self.remaining_steps,
            failure_count=self.failure_count,
            latest_observations=self.latest_observations,
            user_inputs=self.user_inputs,
            tools=scientific_tool_specs(),
        )


class AgentEvaluationSuite(ContractModel):
    suite_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=1000)
    cases: list[AgentEvaluationCase] = Field(min_length=1, max_length=100)


class AgentEvaluationCaseResult(ContractModel):
    case_id: str
    passed: bool
    schema_valid: bool
    tool_arguments_valid: bool
    unexpected_write: bool
    latency_ms: int = Field(ge=0)
    decision: dict[str, Any] | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    error_type: str | None = None


class AgentEvaluationReport(ContractModel):
    suite_id: str
    suite_version: str
    prompt_id: str
    system_prompt_chars: int = Field(ge=1)
    tool_contract_chars: int = Field(ge=1)
    max_raw_state_chars: int = Field(ge=1)
    mean_raw_state_chars: int = Field(ge=1)
    model: AgentModelIdentity
    generated_at: str
    case_count: int = Field(ge=1)
    schema_valid_count: int = Field(ge=0)
    tool_arguments_valid_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    unexpected_write_count: int = Field(ge=0)
    schema_valid_rate: float = Field(ge=0, le=1)
    tool_arguments_valid_rate: float = Field(ge=0, le=1)
    pass_rate: float = Field(ge=0, le=1)
    results: list[AgentEvaluationCaseResult]


def load_agent_evaluation_suite(path: Path) -> AgentEvaluationSuite:
    return AgentEvaluationSuite.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_agent_controller(
    model: AgentDecisionModel,
    suite: AgentEvaluationSuite,
) -> AgentEvaluationReport:
    tools = {spec.name: spec for spec in scientific_tool_specs()}
    results: list[AgentEvaluationCaseResult] = []
    raw_state_chars: list[int] = []
    for case in suite.cases:
        request = case.request()
        raw_state_chars.append(len(request.model_dump_json(exclude_none=True)))
        started = monotonic()
        try:
            decision = model.decide(request)
        except AgentModelProviderError as error:
            results.append(
                AgentEvaluationCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    schema_valid=False,
                    tool_arguments_valid=False,
                    unexpected_write=False,
                    latency_ms=_elapsed_ms(started),
                    failure_reasons=["model response did not satisfy the action contract"],
                    error_type=type(error).__name__,
                )
            )
            continue
        except Exception as error:
            results.append(
                AgentEvaluationCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    schema_valid=False,
                    tool_arguments_valid=False,
                    unexpected_write=False,
                    latency_ms=_elapsed_ms(started),
                    failure_reasons=["evaluation call failed before a valid decision"],
                    error_type=type(error).__name__,
                )
            )
            continue
        failure_reasons, unexpected_write, arguments_valid = _score_decision(
            decision,
            case.expected,
            tools=tools,
            available_evidence_refs={
                str(reference)
                for observation in case.latest_observations
                for reference in observation.get("evidence_refs", [])
                if reference
            },
        )
        results.append(
            AgentEvaluationCaseResult(
                case_id=case.case_id,
                passed=not failure_reasons,
                schema_valid=True,
                tool_arguments_valid=arguments_valid,
                unexpected_write=unexpected_write,
                latency_ms=_elapsed_ms(started),
                decision=decision.model_dump(mode="json"),
                failure_reasons=failure_reasons,
            )
        )

    valid_count = sum(result.schema_valid for result in results)
    valid_arguments_count = sum(result.tool_arguments_valid for result in results)
    passed_count = sum(result.passed for result in results)
    unexpected_write_count = sum(result.unexpected_write for result in results)
    count = len(results)
    return AgentEvaluationReport(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        prompt_id=AGENT_CONTROLLER_PROMPT_ID,
        system_prompt_chars=len(AGENT_CONTROLLER_SYSTEM_PROMPT),
        tool_contract_chars=len(
            AgentDecisionRequest(
                task_id="agt_contract_measurement",
                job_id="job_contract_measurement",
                goal="measure tool contracts",
                step_count=0,
                remaining_steps=1,
                failure_count=0,
                tools=list(tools.values()),
            ).model_dump_json(include={"tools"})
        ),
        max_raw_state_chars=max(raw_state_chars),
        mean_raw_state_chars=round(sum(raw_state_chars) / count),
        model=model.identity,
        generated_at=utc_now().isoformat(),
        case_count=count,
        schema_valid_count=valid_count,
        tool_arguments_valid_count=valid_arguments_count,
        passed_count=passed_count,
        unexpected_write_count=unexpected_write_count,
        schema_valid_rate=valid_count / count,
        tool_arguments_valid_rate=valid_arguments_count / count,
        pass_rate=passed_count / count,
        results=results,
    )


def write_agent_evaluation_report(
    report: AgentEvaluationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _score_decision(
    decision: AgentDecision,
    expected: AgentEvaluationExpectation,
    *,
    tools: dict[str, Any],
    available_evidence_refs: set[str],
) -> tuple[list[str], bool, bool]:
    failures: list[str] = []
    if decision.kind not in expected.allowed_kinds:
        failures.append(f"unexpected decision kind: {decision.kind.value}")
    if len(decision.plan) > 8:
        failures.append("plan exceeds the eight-step controller limit")

    tool_name = decision.tool_name
    arguments_valid = True
    if decision.kind is AgentDecisionKind.CALL_TOOL:
        arguments_valid = bool(
            tool_name
            and scientific_tool_arguments_are_valid(
                tool_name,
                decision.tool_arguments,
            )
        )
        if not arguments_valid:
            failures.append("tool arguments do not satisfy the production contract")
        if expected.allowed_tool_names and tool_name not in expected.allowed_tool_names:
            failures.append(f"unexpected tool: {tool_name or 'missing'}")
        if tool_name in expected.forbidden_tool_names:
            failures.append(f"forbidden tool selected: {tool_name}")
    elif expected.allowed_tool_names:
        failures.append("expected a tool call")

    if decision.kind is AgentDecisionKind.FINISH:
        unknown_evidence = sorted(
            set(decision.final_evidence_refs) - available_evidence_refs
        )
        if unknown_evidence:
            failures.append("final answer cites evidence absent from observations")
        missing_evidence = sorted(
            set(expected.required_evidence_refs) - set(decision.final_evidence_refs)
        )
        if missing_evidence:
            failures.append("required evidence references were omitted")

    selected_spec = tools.get(tool_name) if tool_name else None
    unexpected_write = bool(
        selected_spec is not None
        and selected_spec.risk is not AgentToolRisk.READ_ONLY
        and tool_name not in expected.allowed_tool_names
    )
    if unexpected_write:
        failures.append("unexpected write-capable tool selected")
    return failures, unexpected_write, arguments_valid


def _elapsed_ms(started: float) -> int:
    return max(0, round((monotonic() - started) * 1000))
