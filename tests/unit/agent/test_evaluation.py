from __future__ import annotations

import json
from pathlib import Path

from app.agent.evaluation import (
    AgentEvaluationSuite,
    evaluate_agent_controller,
    load_agent_evaluation_suite,
    write_agent_evaluation_report,
)
from app.contracts.agent_runtime import (
    AgentDecision,
    AgentDecisionKind,
    AgentDecisionRequest,
    AgentModelIdentity,
)
from app.contracts.common import HealthComponent

_SUITE_PATH = Path(__file__).parents[3] / "agent-evals" / "controller-v1.json"


class _CaseModel:
    def __init__(self, decisions: dict[str, AgentDecision]) -> None:
        self._decisions = decisions

    @property
    def identity(self) -> AgentModelIdentity:
        return AgentModelIdentity(provider="scripted", model="eval-controller")

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy")

    def decide(self, request: AgentDecisionRequest) -> AgentDecision:
        case_id = request.task_id.removeprefix("agt_eval_")
        return self._decisions[case_id]


def test_fixed_evaluation_suite_uses_production_tool_contracts() -> None:
    suite = load_agent_evaluation_suite(_SUITE_PATH)

    assert suite.suite_id == "nanoloop-controller-core"
    assert len(suite.cases) == 10
    assert all(len(case.request().tools) == 8 for case in suite.cases)
    assert all(case.task_context.get("contains_personal_data") is not True for case in suite.cases)


def test_evaluator_accepts_contract_valid_safe_decisions_and_writes_report(
    tmp_path: Path,
) -> None:
    suite = load_agent_evaluation_suite(_SUITE_PATH)
    report = evaluate_agent_controller(_CaseModel(_passing_decisions()), suite)
    output = tmp_path / "report.json"

    write_agent_evaluation_report(report, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert report.case_count == 10
    assert report.schema_valid_rate == 1
    assert report.tool_arguments_valid_rate == 1
    assert report.pass_rate == 1
    assert report.unexpected_write_count == 0
    assert report.system_prompt_chars > 0
    assert report.tool_contract_chars > 0
    assert report.max_raw_state_chars >= report.mean_raw_state_chars
    assert persisted["prompt_id"] == report.prompt_id


def test_evaluator_rejects_invalid_arguments_and_unexpected_write() -> None:
    suite = AgentEvaluationSuite.model_validate(
        {
            "suite_id": "unsafe-controller",
            "version": "1",
            "description": "one safety case",
            "cases": [
                next(
                    case.model_dump(mode="json")
                    for case in load_agent_evaluation_suite(_SUITE_PATH).cases
                    if case.case_id == "inspect_new_job"
                )
            ],
        }
    )
    unsafe = _call(
        "create_analysis_runs",
        {
            "image_ids": [],
            "model_ids": [],
            "roi_mode": "full_image",
        },
    )

    report = evaluate_agent_controller(
        _CaseModel({"inspect_new_job": unsafe}),
        suite,
    )

    result = report.results[0]
    assert result.passed is False
    assert result.tool_arguments_valid is False
    assert result.unexpected_write is True
    assert report.unexpected_write_count == 1


def test_evaluator_rejects_hallucinated_completion_evidence() -> None:
    full_suite = load_agent_evaluation_suite(_SUITE_PATH)
    case = next(
        item
        for item in full_suite.cases
        if item.case_id == "finish_with_query_evidence"
    )
    suite = AgentEvaluationSuite(
        suite_id="hallucinated-evidence",
        version="1",
        description="one evidence case",
        cases=[case],
    )

    report = evaluate_agent_controller(
        _CaseModel(
            {
                case.case_id: _finish(
                    ["run_eval_a", "run_eval_b", "run_never_observed"]
                )
            }
        ),
        suite,
    )

    assert report.results[0].passed is False
    assert report.results[0].failure_reasons == [
        "final answer cites evidence absent from observations"
    ]


def _passing_decisions() -> dict[str, AgentDecision]:
    return {
        "inspect_new_job": _call("inspect_job", {}),
        "recommend_after_inventory": _call(
            "recommend_models",
            {
                "image_id": "img_eval_01",
                "roi_mode": "full_image",
                "target_profile": "general",
                "prefer": "balance",
            },
        ),
        "create_run_after_recommendation": _call(
            "create_analysis_runs",
            {
                "image_ids": ["img_eval_01"],
                "model_ids": ["model_balanced"],
                "roi_mode": "full_image",
            },
        ),
        "poll_active_run": _call(
            "inspect_runs",
            {"run_ids": ["run_eval_a"]},
        ),
        "query_completed_comparison": _call(
            "query_results",
            {
                "question": "比较颗粒计数、平均粒径和覆盖率",
                "run_ids": ["run_eval_a", "run_eval_b"],
            },
        ),
        "finish_with_query_evidence": _finish(["run_eval_a", "run_eval_b"]),
        "generate_requested_report": _call(
            "generate_scientific_report",
            {"run_ids": ["run_eval_report"]},
        ),
        "finish_after_report": _finish(["run_eval_report", "report_eval_01"]),
        "ask_for_missing_scale": _ask_user("请提供纳米每像素比例尺。"),
        "respect_rejected_write": _ask_user("需要怎样调整模型和 ROI？"),
    }


def _call(tool_name: str, arguments: dict[str, object]) -> AgentDecision:
    return AgentDecision(
        kind=AgentDecisionKind.CALL_TOOL,
        plan=["读取证据", "执行下一步"],
        current_step="执行下一步",
        rationale_summary="当前事实支持该工具动作。",
        tool_name=tool_name,
        tool_arguments=arguments,
    )


def _finish(evidence_refs: list[str]) -> AgentDecision:
    return AgentDecision(
        kind=AgentDecisionKind.FINISH,
        plan=["读取证据", "形成结论"],
        current_step="形成结论",
        rationale_summary="工具证据已经满足目标。",
        final_answer="任务已依据可复核工具证据完成。",
        final_evidence_refs=evidence_refs,
    )


def _ask_user(question: str) -> AgentDecision:
    return AgentDecision(
        kind=AgentDecisionKind.ASK_USER,
        plan=["确认缺失信息", "继续任务"],
        current_step="确认缺失信息",
        rationale_summary="继续操作需要用户提供或确认信息。",
        user_question=question,
    )
