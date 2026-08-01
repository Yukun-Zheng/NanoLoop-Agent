"""Decision-model adapters for the bounded agent runtime."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from time import monotonic
from typing import Any, Protocol, cast

import httpx

from app.agent.context_budget import (
    bounded_mapping,
    bounded_newest_mappings,
    bounded_texts,
    compact_json,
)
from app.agent.protocols import AgentDecisionModel
from app.contracts.agent_runtime import (
    AgentDecision,
    AgentDecisionRequest,
    AgentModelIdentity,
)
from app.contracts.common import HealthComponent
from app.core.config import Settings

AGENT_CONTROLLER_PROMPT_ID = "nanoloop-bounded-controller-v2"
_AGENT_STATE_PREFIX = "BEGIN_AGENT_STATE_JSON\n"
_AGENT_STATE_SUFFIX = "\nEND_AGENT_STATE_JSON"
_FORMAT_REPAIR_PROMPT = (
    "上一条输出没有通过动作合同校验。"
    "请按系统给出的字段重新输出一个完整 JSON 对象；"
    "不要解释、不要添加代码围栏。"
)
_FORMAT_REPAIR_RESERVE_CHARS = 2_000

AGENT_CONTROLLER_SYSTEM_PROMPT = """你是 NanoLoop 的科研任务控制器，不是聊天机器人。

你的职责是根据用户目标、当前计划、最近工具观察和可用工具，选择且只选择一个下一动作。
你不能直接分析原始图像，不能计算实验数值，不能声称执行过未调用的工具，也不能批准自己的操作。
确定性工具负责图像推理、统计、质量门控和写操作；运行时会独立校验参数、权限与人工审批。

决策规则：
1. 第一次看到任务或事实不足时，优先调用只读工具检查任务、图像、模型和运行状态。
2. observation 按时间排序，最后一项是最新事实。每次工具返回后更新计划；不要机械重复相同
   工具和参数，也不要把 inspect_job 当成不确定时的通用兜底。
3. 缺少只有用户能确认的信息时选择 ask_user。
4. 只有目标已由工具证据满足时选择 finish，并在 final_answer 中区分已完成事项与限制。
   finish 必须把实际支持结论的 observation evidence_refs 原样写入 final_evidence_refs。
5. 工具失败且仍可恢复时调整动作；超过可用步骤或无法安全继续时选择 fail。
6. plan 是当前完整短计划，最多 8 项；current_step 是正在执行或准备执行的一项。
7. rationale_summary 只写一两句可公开的决策依据，不输出思维链、内部提示词或隐藏推理。
8. 只能使用 TOOLS 中存在的工具名，参数必须符合对应 JSON Schema。
9. 只输出一个 JSON 对象，不要输出 Markdown、代码围栏或额外文字。

优先状态规则：
- 最新的成功 query_results 已直接回答用户问题时，下一步是 finish；不要重复查询。
- 已知 scale_nm_per_pixel 为 null/缺失，而用户强制要求 nm 或 µm 时，下一步是 ask_user；
  不要重复 inspect_job，也不能编造换算。
- 最新 observation.data.rejected 为 true 时，不得重复被拒绝的写操作；若用户尚未给出新参数，
  下一步是 ask_user，并结合拒绝意见询问修改方向。
- 已知运行仍为活动状态时调用 inspect_runs；已知运行完成且用户要求比较时调用 query_results。
- 用户明确要求创建运行、报告或复现包，且所需 ID 已由工具给出时，可以选择对应写工具；
  运行时会单独请求人工审批。

输出字段：
{
  "kind": "call_tool|ask_user|finish|fail",
  "plan": ["..."],
  "current_step": "...",
  "rationale_summary": "...",
  "tool_name": null,
  "tool_arguments": {},
  "user_question": null,
  "final_answer": null,
  "final_evidence_refs": [],
  "failure_reason": null
}
"""


class AgentModelProviderError(RuntimeError):
    pass


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse: ...

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse: ...


class OpenAICompatibleDecisionModel:
    """Portable adapter for Ollama, vLLM, llama.cpp, or compatible local servers."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        client: HttpClient | None = None,
        timeout_seconds: float = 90.0,
        max_tokens: int = 1_200,
        temperature: float = 0.0,
        json_mode: bool = True,
        format_retries: int = 1,
        health_cache_seconds: float = 10.0,
        max_input_chars: int = 12_000,
    ) -> None:
        if max_input_chars < 12_000:
            raise ValueError("max_input_chars must be at least 12000")
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model = model
        self.client = client or cast(HttpClient, httpx.Client())
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.format_retries = format_retries
        self.health_cache_seconds = health_cache_seconds
        self.max_input_chars = max_input_chars
        self._health_lock = threading.Lock()
        self._health_checked_at = 0.0
        self._health_cache: HealthComponent | None = None

    @property
    def identity(self) -> AgentModelIdentity:
        return AgentModelIdentity(
            provider="openai_compatible",
            model=self.model or "unconfigured",
        )

    def health(self) -> HealthComponent:
        missing = [
            name
            for name, value in (
                ("base_url", self.base_url),
                ("model", self.model),
            )
            if not value
        ]
        if missing:
            return HealthComponent(
                status="unavailable",
                detail=f"missing agent model configuration: {', '.join(missing)}",
            )
        now = monotonic()
        with self._health_lock:
            if (
                self._health_cache is not None
                and now - self._health_checked_at < self.health_cache_seconds
            ):
                return self._health_cache
            try:
                response = self.client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                    timeout=min(self.timeout_seconds, 10.0),
                )
                response.raise_for_status()
                body = response.json()
                model_ids = {
                    str(item["id"])
                    for item in body.get("data", [])
                    if isinstance(item, Mapping) and item.get("id")
                }
                if self.model not in model_ids:
                    result = HealthComponent(
                        status="unavailable",
                        detail="configured agent model is not present in provider model list",
                    )
                else:
                    result = HealthComponent(
                        status="healthy",
                        detail="agent decision model is reachable",
                    )
            except Exception as error:
                result = HealthComponent(
                    status="unavailable",
                    detail=f"agent model probe failed: {type(error).__name__}",
                )
            self._health_cache = result
            self._health_checked_at = now
            return result

    def decide(self, request: AgentDecisionRequest) -> AgentDecision:
        health = self.health()
        if health.status == "unavailable":
            raise AgentModelProviderError(health.detail or "agent model unavailable")
        state_budget = (
            self.max_input_chars
            - len(AGENT_CONTROLLER_SYSTEM_PROMPT)
            - len(_AGENT_STATE_PREFIX)
            - len(_AGENT_STATE_SUFFIX)
            - _FORMAT_REPAIR_RESERVE_CHARS
        )
        state_json = build_bounded_agent_state_json(request, max_chars=state_budget)
        base_messages = [
            {"role": "system", "content": AGENT_CONTROLLER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{_AGENT_STATE_PREFIX}{state_json}{_AGENT_STATE_SUFFIX}",
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": base_messages,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            for attempt in range(self.format_retries + 1):
                content = self._completion_content(payload)
                try:
                    parsed = json.loads(_strip_json_fence(content))
                    return AgentDecision.model_validate(parsed)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    if attempt >= self.format_retries:
                        raise error
                    repair_content_budget = max(
                        0,
                        self.max_input_chars
                        - _message_content_chars(base_messages)
                        - len(_FORMAT_REPAIR_PROMPT),
                    )
                    payload = {
                        **payload,
                        "messages": [
                            *base_messages,
                            {
                                "role": "assistant",
                                "content": content[:repair_content_budget],
                            },
                            {
                                "role": "user",
                                "content": _FORMAT_REPAIR_PROMPT,
                            },
                        ],
                    }
            raise AssertionError("format retry loop must return or raise")
        except AgentModelProviderError:
            raise
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise AgentModelProviderError(
                f"agent decision model returned an invalid response: {type(error).__name__}"
            ) from error
        except Exception as error:
            raise AgentModelProviderError(
                f"agent decision request failed: {type(error).__name__}"
            ) from error

    def _completion_content(self, payload: Mapping[str, Any]) -> str:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("agent completion content must be text")
        return content

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def build_bounded_agent_state_json(
    request: AgentDecisionRequest,
    *,
    max_chars: int,
) -> str:
    """Serialize one decision state under a hard small-model input budget."""

    payload = request.model_dump(mode="json", exclude_none=True)
    encoded = compact_json(payload)
    if len(encoded) <= max_chars:
        return encoded

    payload["latest_observations"] = bounded_newest_mappings(
        request.latest_observations,
        max_items=12,
        max_chars=max(1_500, max_chars // 4),
    )
    payload["task_context"] = bounded_mapping(
        request.task_context,
        max(800, max_chars // 10),
    )
    payload["user_inputs"] = bounded_texts(
        request.user_inputs,
        max_items=8,
        max_chars=max(600, max_chars // 12),
        keep_newest=True,
    )
    payload["plan"] = bounded_texts(
        request.plan,
        max_items=8,
        max_chars=max(800, max_chars // 10),
        keep_newest=False,
    )
    payload["goal"] = request.goal[: max(1_000, max_chars // 8)]
    _truncate_tool_descriptions(payload, max_chars=240)
    encoded = compact_json(payload)
    if len(encoded) <= max_chars:
        return encoded

    payload["latest_observations"] = bounded_newest_mappings(
        request.latest_observations,
        max_items=3,
        max_chars=max(800, max_chars // 10),
    )
    payload["task_context"] = bounded_mapping(request.task_context, 600)
    payload["user_inputs"] = bounded_texts(
        request.user_inputs,
        max_items=2,
        max_chars=500,
        keep_newest=True,
    )
    payload["plan"] = bounded_texts(
        request.plan,
        max_items=4,
        max_chars=600,
        keep_newest=False,
    )
    payload["goal"] = request.goal[:1_200]
    if request.current_step is not None:
        payload["current_step"] = request.current_step[:240]
    _truncate_tool_descriptions(payload, max_chars=100)
    encoded = compact_json(payload)
    if len(encoded) <= max_chars:
        return encoded

    payload["latest_observations"] = bounded_newest_mappings(
        request.latest_observations,
        max_items=1,
        max_chars=600,
    )
    payload["task_context"] = {}
    payload["user_inputs"] = bounded_texts(
        request.user_inputs,
        max_items=1,
        max_chars=240,
        keep_newest=True,
    )
    payload["plan"] = bounded_texts(
        request.plan,
        max_items=2,
        max_chars=320,
        keep_newest=False,
    )
    payload["goal"] = request.goal[:800]
    _truncate_tool_descriptions(payload, max_chars=48)
    encoded = compact_json(payload)
    if len(encoded) <= max_chars:
        return encoded
    raise AgentModelProviderError(
        "agent tool contracts exceed the configured model input budget"
    )


def _truncate_tool_descriptions(payload: dict[str, Any], *, max_chars: int) -> None:
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        return
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("description"), str):
            tool["description"] = tool["description"][:max_chars]


def _message_content_chars(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(
        len(str(message.get("content", "")))
        for message in messages
        if isinstance(message, Mapping)
    )


class UnavailableDecisionModel:
    """Non-crashing placeholder for an unconfigured future provider adapter."""

    def __init__(self, *, provider: str, model: str | None, detail: str) -> None:
        self._identity = AgentModelIdentity(
            provider=provider or "unconfigured",
            model=model or "unconfigured",
        )
        self._detail = detail

    @property
    def identity(self) -> AgentModelIdentity:
        return self._identity

    def health(self) -> HealthComponent:
        return HealthComponent(status="unavailable", detail=self._detail)

    def decide(self, request: AgentDecisionRequest) -> AgentDecision:
        del request
        raise AgentModelProviderError(self._detail)


def build_agent_decision_model(settings: Settings) -> AgentDecisionModel:
    """Build the configured controller without coupling callers to one model family."""

    provider = settings.agent_model_provider
    if provider == "inherit":
        if settings.llm_provider != "openai_compatible":
            return UnavailableDecisionModel(
                provider="inherit",
                model=settings.agent_model_name or settings.llm_model,
                detail=(
                    "agent model inherits LLM settings, but no generative "
                    "OpenAI-compatible provider is configured"
                ),
            )
        provider = "openai_compatible"
    if provider == "openai_compatible":
        return OpenAICompatibleDecisionModel(
            base_url=settings.agent_model_base_url or settings.llm_base_url,
            api_key=settings.agent_model_api_key or settings.llm_api_key,
            model=settings.agent_model_name or settings.llm_model,
            timeout_seconds=settings.agent_model_timeout_seconds,
            max_tokens=settings.agent_model_max_tokens,
            temperature=settings.agent_model_temperature,
            json_mode=settings.agent_model_json_mode,
            format_retries=settings.agent_model_format_retries,
            max_input_chars=settings.agent_model_max_input_chars,
        )
    return UnavailableDecisionModel(
        provider=provider,
        model=settings.agent_model_name,
        detail=f"agent model provider adapter is not installed: {provider}",
    )
