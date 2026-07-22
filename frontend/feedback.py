"""Structured feedback components for actionable error/recovery UX.

Implements three E-P1 priorities in one coherent module:

- 401 ``AUTHENTICATION_REQUIRED`` → operator guidance instead of a raw error.
- 429 ``RATE_LIMITED`` → Retry-After-aware read-retry messaging, never
  auto-replaying mutation requests.
- Long-running / partially-completed work → a status panel that preserves
  recovery actions instead of forcing the user to re-create a run.

These renderers are pure HTML builders (plus a thin Streamlit wrapper), so
they can be unit-tested without booting Streamlit.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from frontend.state import status_tone

__all__ = [
    "LongTaskStage",
    "RenderOptions",
    "build_actionable_error_html",
    "build_auth_guidance_html",
    "build_long_task_html",
    "build_rate_limit_html",
    "build_retry_section_html",
    "build_service_unavailable_html",
    "parse_retry_after_seconds",
    "render_actionable_error",
    "render_auth_guidance",
    "render_long_task",
    "render_rate_limit",
    "render_retry_section",
    "render_service_unavailable",
]


@dataclass(frozen=True)
class RenderOptions:
    """Optional knobs shared by the feedback renderers."""

    show_request_id: bool = True
    show_error_code: bool = True
    compact: bool = False


@dataclass(frozen=True)
class LongTaskStage:
    """One stage of a long-running operation."""

    label: str
    status: str  # "pending" | "active" | "completed" | "failed"
    detail: str | None = None


_RETRY_AFTER_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def parse_retry_after_seconds(value: object) -> float | None:
    """Best-effort conversion of a ``Retry-After`` header value to seconds.

    Returns ``None`` when the value is missing, malformed, or a HTTP-date
    (which we intentionally do not parse — the server should send seconds).
    """

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    if isinstance(value, str):
        match = _RETRY_AFTER_PATTERN.match(value)
        if match:
            return max(0.0, float(match.group(1)))
    return None


def build_actionable_error_html(
    *,
    title: str,
    code: str,
    message: str,
    tone: str = "bad",
    request_id: str | None = None,
    hint: str | None = None,
    recovery_label: str | None = None,
    recovery_anchor: str | None = None,
    options: RenderOptions = RenderOptions(),
) -> str:
    """Render an actionable error panel.

    Unlike ``st.error``, this panel always carries a *next step* so the user
    never has to guess whether they should refresh, contact an operator, or
    simply wait.
    """

    code_html = (
        f'<span class="nl-error-code">{html.escape(code)}</span>' if options.show_error_code else ""
    )
    request_html = (
        f'<span class="nl-error-request-id">request_id <code>{html.escape(request_id)}</code></span>'
        if options.show_request_id and request_id
        else ""
    )
    hint_html = (
        f'<div class="nl-error-hint">{html.escape(hint)}</div>' if hint else ""
    )
    recovery_html = ""
    if recovery_label and recovery_anchor:
        recovery_html = (
            f'<a class="nl-error-recovery" href="#{html.escape(recovery_anchor, quote=True)}">'
            f"{html.escape(recovery_label)}</a>"
        )
    meta_html = ""
    if code_html or request_html:
        meta_html = f'<div class="nl-error-meta">{code_html}{request_html}</div>'
    return (
        f'<div role="alert" class="nl-error-panel nl-error-panel-{html.escape(tone, quote=True)}">'
        f'<div class="nl-error-title">{html.escape(title)}</div>'
        f'<div class="nl-error-message">{html.escape(message)}</div>'
        f"{hint_html}{meta_html}{recovery_html}"
        "</div>"
    )


def build_auth_guidance_html(
    *,
    operator_env_var: str = "NANOLOOP_API_KEY",
    base_url_env_var: str = "NANOLOOP_API_BASE_URL",
    contact_hint: str | None = None,
    options: RenderOptions = RenderOptions(),
) -> str:
    """Render 401 ``AUTHENTICATION_REQUIRED`` guidance for operators.

    The frontend must NOT print the key or the locked URL — only point the
    operator at the env vars they need to update.
    """

    contact_html = (
        f'<div class="nl-error-hint">{html.escape(contact_hint)}</div>' if contact_hint else ""
    )
    return (
        '<div role="alert" class="nl-error-panel nl-error-panel-bad">'
        '<div class="nl-error-title">后端拒绝了当前 API Key</div>'
        '<div class="nl-error-message">'
        "本次部署使用共享 API Key 进行认证。当前进程持有的 Key 已失效、缺失或与后端不一致。"
        "</div>"
        '<div class="nl-error-hint">'
        "请运维同学在<b>运行 Streamlit 的环境</b>中更新以下环境变量后重启前端："
        "</div>"
        '<ul class="nl-error-steps">'
        f"<li><code>{html.escape(operator_env_var)}</code> — 后端要求的共享 Key 值</li>"
        f"<li><code>{html.escape(base_url_env_var)}</code> — 与 Key 绑定的后端地址</li>"
        "</ul>"
        '<div class="nl-error-hint">'
        "更新后请重启 Streamlit 进程；session 中保存的旧地址会被自动丢弃。"
        "</div>"
        f"{contact_html}"
        "</div>"
    )


def build_rate_limit_html(
    *,
    retry_after_seconds: float | None,
    is_read_request: bool,
    request_id: str | None = None,
    options: RenderOptions = RenderOptions(),
) -> str:
    """Render 429 ``RATE_LIMITED`` messaging.

    Only *safe read* requests are eligible for retry — mutation requests
    (create run / save boxes / export) must never be silently replayed.
    """

    wait_html = (
        f'<div class="nl-error-message">请等待约 <b>{retry_after_seconds:.0f} 秒</b> 后重试。</div>'
        if retry_after_seconds is not None
        else '<div class="nl-error-message">请稍后片刻再重试。</div>'
    )
    if is_read_request:
        action_html = (
            '<div class="nl-error-hint">'
            "这是一次<b>读取</b>操作，等待结束后可安全重试；界面不会自动重放，"
            "请在倒计时结束后点击重试按钮。"
            "</div>"
        )
    else:
        action_html = (
            '<div class="nl-error-hint">'
            "这是一次<b>写入</b>操作。为避免产生重复运行或重复保存，"
            "本界面<b>不会</b>自动重试——请确认后端状态后再决定是否手动重试。"
            "</div>"
        )
    request_html = (
        f'<div class="nl-error-meta"><span class="nl-error-request-id">'
        f"request_id <code>{html.escape(request_id)}</code></span></div>"
        if options.show_request_id and request_id
        else ""
    )
    return (
        '<div role="alert" class="nl-error-panel nl-error-panel-warn">'
        '<div class="nl-error-title">请求被限流（429 RATE_LIMITED）</div>'
        f"{wait_html}{action_html}{request_html}"
        "</div>"
    )


def build_service_unavailable_html(
    *,
    code: str,
    base_url: str | None = None,
    hint: str | None = None,
    options: RenderOptions = RenderOptions(),
) -> str:
    """Render a "service unreachable" panel for network/gateway failures.

    This is for HTTP 502 / 503 / connection-refused / non-JSON responses —
    situations where the *backend process itself* is not answering, as
    opposed to a business-level error. We must NOT imply the user should
    blindly retry; instead, point them at the actual cause.
    """

    base_html = (
        f'<div class="nl-error-hint">目标地址：<code>{html.escape(base_url)}</code></div>'
        if base_url
        else ""
    )
    hint_html = (
        f'<div class="nl-error-hint">{html.escape(hint)}</div>'
        if hint
        else (
            '<div class="nl-error-hint">'
            "最常见的三种原因：后端进程未启动 / 反向代理或网关宕机 / 网络中间设备返回了非 JSON 错误页。"
            "请先确认后端服务是否在运行、地址是否填写正确，再回到此页面点击「检查连接」。"
            "</div>"
        )
    )
    code_html = (
        f'<span class="nl-error-code">{html.escape(code)}</span>' if options.show_error_code else ""
    )
    meta_html = (
        f'<div class="nl-error-meta">{code_html}</div>' if code_html else ""
    )
    return (
        '<div role="alert" class="nl-error-panel nl-error-panel-bad">'
        '<div class="nl-error-title">无法连接到后端服务</div>'
        '<div class="nl-error-message">'
        "后端没有返回有效的 JSON 响应。这通常不是业务逻辑错误，"
        "而是服务本身不可达。"
        "</div>"
        f"{base_html}{hint_html}{meta_html}"
        "</div>"
    )


def build_long_task_html(
    *,
    title: str,
    stages: Sequence[LongTaskStage],
    summary: str | None = None,
    recoverable_action_label: str | None = None,
    recoverable_action_anchor: str | None = None,
    partial_failure_count: int = 0,
) -> str:
    """Render a long-task / partial-failure panel.

    The key invariant: the user can always see *which* stage failed and how
    to recover, without re-creating the whole run.
    """

    items: list[str] = []
    for stage in stages:
        tone = status_tone(stage.status)
        detail_html = (
            f'<div class="nl-stage-detail">{html.escape(stage.detail)}</div>'
            if stage.detail
            else ""
        )
        items.append(
            f'<li class="nl-stage nl-stage-{html.escape(tone, quote=True)}">'
            f'<div class="nl-stage-label">{html.escape(stage.label)}</div>'
            f"{detail_html}"
            "</li>"
        )
    stages_html = f'<ul class="nl-stage-list">{"".join(items)}</ul>'

    summary_html = (
        f'<div class="nl-long-task-summary">{html.escape(summary)}</div>' if summary else ""
    )
    partial_html = ""
    if partial_failure_count > 0:
        partial_html = (
            f'<div class="nl-long-task-partial">'
            f"⚠️ {partial_failure_count} 个图像/运行部分失败；其余结果仍可查看，"
            "无需重新创建整个运行。"
            "</div>"
        )
    recover_html = ""
    if recoverable_action_label and recoverable_action_anchor:
        recover_html = (
            f'<a class="nl-long-task-recover" '
            f'href="#{html.escape(recoverable_action_anchor, quote=True)}">'
            f"{html.escape(recoverable_action_label)}</a>"
        )
    return (
        '<div role="status" aria-live="polite" aria-atomic="true" '
        'class="nl-long-task">'
        f'<div class="nl-long-task-title">{html.escape(title)}</div>'
        f"{summary_html}{partial_html}{stages_html}{recover_html}"
        "</div>"
    )


def build_retry_section_html(
    *,
    action_label: str,
    retry_after_seconds: float | None = None,
    hint: str | None = None,
) -> str:
    """Render a retry prompt with optional countdown text.

    Designed to appear *below* a rate-limit or retryable-error panel so the
    user has an explicit, visible "retry" affordance instead of having to
    re-navigate to the original control.
    """

    if retry_after_seconds is not None and retry_after_seconds > 0:
        wait_text = f"请等待约 {retry_after_seconds:.0f} 秒后再重试。"
    else:
        wait_text = "可以立即重试。"
    hint_html = (
        f'<div class="nl-error-hint">{html.escape(hint)}</div>' if hint else ""
    )
    return (
        '<div class="nl-announcement nl-announcement-live" '
        'role="status" aria-live="polite" aria-atomic="true">'
        f'<div class="nl-announcement-title">{html.escape(action_label)}</div>'
        f'<div class="nl-announcement-body">{wait_text}</div>'
        f"{hint_html}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Streamlit-bound renderers (thin wrappers so pages stay declarative)
# ---------------------------------------------------------------------------


def render_actionable_error(
    streamlit: Any,
    *,
    title: str,
    code: str,
    message: str,
    tone: str = "bad",
    request_id: str | None = None,
    hint: str | None = None,
    recovery_label: str | None = None,
    recovery_anchor: str | None = None,
) -> None:
    streamlit.markdown(
        build_actionable_error_html(
            title=title,
            code=code,
            message=message,
            tone=tone,
            request_id=request_id,
            hint=hint,
            recovery_label=recovery_label,
            recovery_anchor=recovery_anchor,
        ),
        unsafe_allow_html=True,
    )


def render_auth_guidance(
    streamlit: Any,
    *,
    operator_env_var: str = "NANOLOOP_API_KEY",
    base_url_env_var: str = "NANOLOOP_API_BASE_URL",
    contact_hint: str | None = None,
) -> None:
    streamlit.markdown(
        build_auth_guidance_html(
            operator_env_var=operator_env_var,
            base_url_env_var=base_url_env_var,
            contact_hint=contact_hint,
        ),
        unsafe_allow_html=True,
    )


def render_rate_limit(
    streamlit: Any,
    *,
    retry_after_seconds: float | None,
    is_read_request: bool,
    request_id: str | None = None,
) -> None:
    streamlit.markdown(
        build_rate_limit_html(
            retry_after_seconds=retry_after_seconds,
            is_read_request=is_read_request,
            request_id=request_id,
        ),
        unsafe_allow_html=True,
    )


def render_service_unavailable(
    streamlit: Any,
    *,
    code: str,
    base_url: str | None = None,
    hint: str | None = None,
) -> None:
    streamlit.markdown(
        build_service_unavailable_html(code=code, base_url=base_url, hint=hint),
        unsafe_allow_html=True,
    )


def render_long_task(
    streamlit: Any,
    *,
    title: str,
    stages: Sequence[LongTaskStage],
    summary: str | None = None,
    recoverable_action_label: str | None = None,
    recoverable_action_anchor: str | None = None,
    partial_failure_count: int = 0,
) -> None:
    streamlit.markdown(
        build_long_task_html(
            title=title,
            stages=stages,
            summary=summary,
            recoverable_action_label=recoverable_action_label,
            recoverable_action_anchor=recoverable_action_anchor,
            partial_failure_count=partial_failure_count,
        ),
        unsafe_allow_html=True,
    )


def render_retry_section(
    streamlit: Any,
    *,
    action_label: str,
    retry_after_seconds: float | None = None,
    hint: str | None = None,
) -> None:
    streamlit.markdown(
        build_retry_section_html(
            action_label=action_label,
            retry_after_seconds=retry_after_seconds,
            hint=hint,
        ),
        unsafe_allow_html=True,
    )
