from __future__ import annotations

import json
from typing import Any

import pytest

from app.agent.model_provider import (
    AGENT_CONTROLLER_SYSTEM_PROMPT,
    AgentModelProviderError,
    OpenAICompatibleDecisionModel,
)
from app.agent.scientific_tools import scientific_tool_specs
from app.contracts.agent_runtime import (
    AgentDecisionKind,
    AgentDecisionRequest,
    AgentToolRisk,
    AgentToolSpec,
)


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _Client:
    def __init__(self, completion: str | list[str]) -> None:
        self.completions = (
            [completion] if isinstance(completion, str) else list(completion)
        )
        self.payloads: list[dict[str, Any]] = []

    def get(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        assert headers in ({"Authorization": "Bearer local"}, {})
        assert timeout > 0
        return _Response({"data": [{"id": "local-4b"}]})

    def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> _Response:
        assert headers in ({"Authorization": "Bearer local"}, {})
        assert timeout > 0
        self.payloads.append(json)
        completion = self.completions.pop(0)
        return _Response(
            {"choices": [{"message": {"content": completion}}]}
        )


def test_openai_compatible_adapter_returns_one_validated_action() -> None:
    client = _Client(
        """```json
{"kind":"call_tool","plan":["检查任务"],"current_step":"读取状态",
"rationale_summary":"需要先读取确定性状态。",
"tool_name":"inspect_job","tool_arguments":{},
"user_question":null,"final_answer":null,"failure_reason":null}
```"""
    )
    model = OpenAICompatibleDecisionModel(
        base_url="http://local/v1",
        api_key="local",
        model="local-4b",
        client=client,
    )

    decision = model.decide(_request())

    assert decision.kind is AgentDecisionKind.CALL_TOOL
    assert decision.tool_name == "inspect_job"
    payload = client.payloads[0]
    assert payload["messages"][0]["content"] == AGENT_CONTROLLER_SYSTEM_PROMPT
    assert payload["response_format"] == {"type": "json_object"}
    assert "think" not in payload


def test_local_adapter_allows_servers_without_authorization_header_or_json_mode() -> None:
    client = _Client(
        """{"kind":"call_tool","plan":["检查"],"current_step":"读取状态",
"rationale_summary":"需要工具证据。","tool_name":"inspect_job","tool_arguments":{}}"""
    )
    model = OpenAICompatibleDecisionModel(
        base_url="http://local/v1",
        api_key=None,
        model="local-4b",
        client=client,
        json_mode=False,
    )

    assert model.health().status == "healthy"
    decision = model.decide(_request())

    assert decision.tool_name == "inspect_job"
    assert "response_format" not in client.payloads[0]


def test_adapter_rejects_free_text_or_wrong_action_shape() -> None:
    invalid = '{"kind":"call_tool","plan":["检查"],"current_step":"检查"}'
    client = _Client([invalid, invalid])
    model = OpenAICompatibleDecisionModel(
        base_url="http://local/v1",
        api_key="local",
        model="local-4b",
        client=client,
    )

    with pytest.raises(AgentModelProviderError, match="invalid response"):
        model.decide(_request())
    assert len(client.payloads) == 2


def test_adapter_repairs_one_malformed_small_model_action() -> None:
    client = _Client(
        [
            "我认为应该先检查任务。",
            """{"kind":"call_tool","plan":["检查"],"current_step":"读取状态",
"rationale_summary":"需要工具证据。","tool_name":"inspect_job","tool_arguments":{}}""",
        ]
    )
    model = OpenAICompatibleDecisionModel(
        base_url="http://local/v1",
        api_key=None,
        model="local-4b",
        client=client,
    )

    decision = model.decide(_request())

    assert decision.tool_name == "inspect_job"
    assert len(client.payloads) == 2
    assert client.payloads[1]["messages"][-1]["role"] == "user"


def test_adapter_enforces_one_total_input_budget_and_keeps_newest_evidence() -> None:
    client = _Client(
        """{"kind":"call_tool","plan":["检查"],"current_step":"读取状态",
"rationale_summary":"需要工具证据。","tool_name":"inspect_job","tool_arguments":{}}"""
    )
    model = OpenAICompatibleDecisionModel(
        base_url="http://local/v1",
        api_key=None,
        model="local-4b",
        client=client,
        max_input_chars=16_000,
    )
    request = AgentDecisionRequest(
        task_id="agt_budget",
        job_id="job_budget",
        goal="比较所有已完成运行并给出有证据的结论" + "目标" * 2_000,
        task_context={"notes": "上下文" * 8_000},
        plan=[f"计划步骤 {index} " + "说明" * 300 for index in range(20)],
        current_step="读取最新结果",
        step_count=11,
        remaining_steps=1,
        failure_count=0,
        latest_observations=[
            {"sequence": index, "payload": str(index) * 4_000}
            for index in range(12)
        ],
        user_inputs=["补充信息" * 1_000 for _ in range(8)],
        tools=scientific_tool_specs(),
    )

    decision = model.decide(request)

    assert decision.tool_name == "inspect_job"
    messages = client.payloads[0]["messages"]
    assert sum(len(message["content"]) for message in messages) <= 16_000
    state = messages[1]["content"]
    state_payload = json.loads(
        state.removeprefix("BEGIN_AGENT_STATE_JSON\n").removesuffix(
            "\nEND_AGENT_STATE_JSON"
        )
    )
    newest = state_payload["latest_observations"][-1]
    assert newest.get("sequence") == 11 or '"sequence":11' in newest["preview"]
    assert all(item.get("sequence") != 0 for item in state_payload["latest_observations"])


def test_format_repair_does_not_grow_past_total_input_budget() -> None:
    client = _Client(
        [
            "错误输出" * 20_000,
            """{"kind":"call_tool","plan":["检查"],"current_step":"读取状态",
"rationale_summary":"需要工具证据。","tool_name":"inspect_job","tool_arguments":{}}""",
        ]
    )
    model = OpenAICompatibleDecisionModel(
        base_url="http://local/v1",
        api_key=None,
        model="local-4b",
        client=client,
        max_input_chars=12_000,
    )

    decision = model.decide(_request())

    assert decision.tool_name == "inspect_job"
    assert all(
        sum(len(message["content"]) for message in payload["messages"]) <= 12_000
        for payload in client.payloads
    )


def _request() -> AgentDecisionRequest:
    return AgentDecisionRequest(
        task_id="agt_test",
        job_id="job_test",
        goal="检查任务",
        step_count=0,
        remaining_steps=4,
        failure_count=0,
        tools=[
            AgentToolSpec(
                name="inspect_job",
                description="inspect",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                risk=AgentToolRisk.READ_ONLY,
                requires_approval=False,
                idempotent=True,
            )
        ],
    )
