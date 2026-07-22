"""Regression tests for ``_api_action`` error routing — the "Key rotation
regression" E-P1 task.

These tests verify that when the shared API key is rotated (401), rate-limited
(429), or the backend is unreachable (502/503), the correct structured feedback
component is rendered and no sensitive information (API key value) is leaked
into the UI.

The ``_api_action`` function is the central error router in ``app.py``.  It
receives an ``operation`` callable, runs it inside a spinner, and — on
failure — inspects the resulting ``ApiClientError`` to decide which
``render_*`` component to invoke.  These tests exercise every branch of that
routing logic with a lightweight fake Streamlit that captures all output.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from frontend.api_client import ApiClientError
from frontend.app import _api_action, _is_read_action

# ---------------------------------------------------------------------------
# Fake Streamlit — captures markdown / error / caption calls for assertion
# ---------------------------------------------------------------------------


class _FakeStreamlit:
    """Minimal stand-in for the ``streamlit`` module used by renderers."""

    def __init__(self) -> None:
        self.markdown_calls: list[str] = []
        self.error_calls: list[str] = []
        self.caption_calls: list[str] = []
        self.json_calls: list[Any] = []
        self.button_calls: list[str] = []
        self.warning_calls: list[str] = []

    # st.spinner is a context manager; the fake just yields.
    @contextmanager
    def spinner(self, label: str = "") -> Any:
        yield

    def markdown(self, body: str, unsafe_allow_html: bool = False, **_kw: Any) -> None:
        self.markdown_calls.append(body)

    def error(self, text: str) -> None:
        self.error_calls.append(text)

    def caption(self, text: str) -> None:
        self.caption_calls.append(text)

    def json(self, data: Any) -> None:
        self.json_calls.append(data)

    def warning(self, text: str) -> None:
        """Capture warning calls from render_exception / render_run_summary."""
        if not hasattr(self, "warning_calls"):
            self.warning_calls = []
        self.warning_calls.append(text)

    def button(self, label: str = "", **_kw: Any) -> bool:
        """Record the button label and always return False (not clicked)."""
        self.button_calls.append(label)
        return False

    @contextmanager
    def expander(self, label: str) -> Any:
        yield


def _make_state() -> dict[str, Any]:
    return {"api_base_url": "https://api.example.com"}


def _raise(error: ApiClientError) -> Any:
    """Return a callable that always raises *error*."""

    def _op() -> Any:
        raise error

    return _op


# ---------------------------------------------------------------------------
# _is_read_action heuristic
# ---------------------------------------------------------------------------


class TestIsReadAction:
    """The heuristic that decides whether a 429 retry is safe."""

    @pytest.mark.parametrize(
        "action",
        [
            "加载模型",
            "刷新运行状态",
            "检查连接",
            "查看结果",
            "获取详情",
            "列出运行",
            "推荐模型",
            "预览图像",
            "下载制品",
        ],
    )
    def test_known_read_verbs_are_detected(self, action: str) -> None:
        assert _is_read_action(action) is True

    @pytest.mark.parametrize(
        "action",
        [
            "创建分析",
            "保存ROI",
            "导出报告",
            "提交运行",
            "删除任务",
        ],
    )
    def test_write_actions_are_not_read(self, action: str) -> None:
        assert _is_read_action(action) is False

    def test_unknown_action_is_treated_as_write(self) -> None:
        """Conservative default: unknown → mutation, never auto-replayed."""

        assert _is_read_action("执行操作") is False

    def test_read_keyword_as_substring(self) -> None:
        """Keywords match as substrings, not exact equality."""

        assert _is_read_action("正在加载模型列表") is True
        assert _is_read_action("重新刷新连接状态") is True


# ---------------------------------------------------------------------------
# 401 / Key rotation routing
# ---------------------------------------------------------------------------


class TestAuthGuidanceRouting:
    """When the shared API key is rotated or invalid, the 401 path must show
    operator guidance — not a raw error — and must never leak the key value."""

    def test_401_status_code_renders_auth_guidance(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message="API key is invalid",
                    request_id="req_001",
                )
            ),
        )

        assert len(st.markdown_calls) >= 1
        html = st.markdown_calls[0]
        assert "后端拒绝了当前 API Key" in html
        assert "NANOLOOP_API_KEY" in html
        assert "NANOLOOP_API_BASE_URL" in html
        # The guidance should mention restarting the process.
        assert "重启" in html

    def test_401_by_code_alone_triggers_auth_guidance(self) -> None:
        """Even without ``status_code == 401``, the code
        ``AUTHENTICATION_REQUIRED`` must route to auth guidance."""

        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "检查连接",
            _raise(
                ApiClientError(
                    status_code=0,
                    code="AUTHENTICATION_REQUIRED",
                    message="auth required",
                    request_id="req_003",
                )
            ),
        )

        assert any("后端拒绝了当前 API Key" in h for h in st.markdown_calls)

    def test_401_does_not_leak_key_value_in_message(self) -> None:
        """The backend error message might contain the key; the UI must
        **never** echo it."""

        st = _FakeStreamlit()
        secret = "sk-secret-12345"
        _api_action(
            st,
            _make_state(),
            "检查连接",
            _raise(
                ApiClientError(
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message=f"Key '{secret}' is not valid",
                    request_id="req_002",
                )
            ),
        )

        for html in st.markdown_calls:
            assert secret not in html, "API key value must not appear in UI"

    def test_401_does_not_leak_key_value_in_error_calls(self) -> None:
        """``render_exception`` (the fallback) must also not leak the key."""

        st = _FakeStreamlit()
        secret = "sk-leaked-98765"
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message=f"Key '{secret}' rejected",
                    request_id="req_002b",
                )
            ),
        )

        for text in st.error_calls:
            assert secret not in text

    def test_401_state_records_error_details(self) -> None:
        st = _FakeStreamlit()
        state = _make_state()
        _api_action(
            st,
            state,
            "检查连接",
            _raise(
                ApiClientError(
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message="invalid key",
                    request_id="req_004",
                )
            ),
        )

        assert state["last_error"] is not None
        assert state["last_error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert state["last_error"]["request_id"] == "req_004"
        assert state["last_error"]["action"] == "检查连接"

    def test_401_does_not_render_rate_limit_or_service_unavailable(self) -> None:
        """The 401 branch must not accidentally render other panels."""

        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message="invalid",
                    request_id="req_005",
                )
            ),
        )

        combined = " ".join(st.markdown_calls)
        assert "限流" not in combined
        assert "无法连接到后端服务" not in combined


# ---------------------------------------------------------------------------
# 429 / Rate limiting routing
# ---------------------------------------------------------------------------


class TestRateLimitRouting:
    """429 errors must show differentiated retry messaging based on whether
    the failed operation was a read or a write."""

    def test_429_with_retry_after_shows_countdown(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="Too many requests",
                    request_id="req_010",
                    retryable=True,
                    retry_after="60",
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "限流" in html
        assert "60 秒" in html
        assert "读取" in html  # "加载" is a read action

    def test_429_without_retry_after_shows_generic_wait(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="Too many requests",
                    request_id="req_011",
                    retryable=True,
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "限流" in html
        assert "稍后片刻" in html
        assert "60 秒" not in html

    def test_429_write_request_forbids_auto_replay(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "创建分析",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="Too many requests",
                    request_id="req_012",
                    retryable=True,
                    retry_after="30",
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "写入" in html
        assert "不会" in html  # "不会自动重试"
        # Countdown should still appear (it tells the user how long to wait),
        # but the messaging must say "do not auto-replay".
        assert "30 秒" in html

    def test_429_read_request_allows_safe_retry(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "刷新运行",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="Too many requests",
                    request_id="req_013",
                    retryable=True,
                    retry_after="15",
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "读取" in html
        assert "可安全重试" in html

    def test_429_by_code_alone_triggers_rate_limit(self) -> None:
        """Even without ``status_code == 429``, the code ``RATE_LIMITED``
        must route to the rate-limit panel."""

        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=0,
                    code="RATE_LIMITED",
                    message="rate limited",
                    request_id="req_014",
                )
            ),
        )

        assert any("限流" in h for h in st.markdown_calls)

    def test_429_does_not_render_auth_guidance(self) -> None:
        """The 429 branch must not accidentally render the 401 panel."""

        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="limited",
                    request_id="req_015",
                )
            ),
        )

        combined = " ".join(st.markdown_calls)
        assert "后端拒绝了当前 API Key" not in combined


# ---------------------------------------------------------------------------
# 502 / 503 / Service unavailable routing
# ---------------------------------------------------------------------------


class TestServiceUnavailableRouting:
    """Gateway and server errors must render the "service unreachable" panel,
    not a generic business error."""

    def test_502_renders_service_unavailable(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=502,
                    code="HTTP_502",
                    message="Bad Gateway",
                    request_id="req_020",
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "无法连接到后端服务" in html
        assert "api.example.com" in html  # base_url is shown for diagnosis

    def test_503_renders_service_unavailable(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=503,
                    code="SERVICE_UNAVAILABLE",
                    message="Service Unavailable",
                    request_id="req_021",
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "无法连接到后端服务" in html

    def test_500_renders_service_unavailable(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=500,
                    code="HTTP_500",
                    message="Internal Server Error",
                    request_id="req_022",
                )
            ),
        )

        html = st.markdown_calls[0]
        assert "无法连接到后端服务" in html

    def test_service_unavailable_by_code(self) -> None:
        """The named error codes ``SERVICE_UNAVAILABLE`` / ``BAD_GATEWAY`` /
        ``GATEWAY_TIMEOUT`` must also route here, even without a 5xx status."""

        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=0,
                    code="BAD_GATEWAY",
                    message="gateway down",
                    request_id="req_023",
                )
            ),
        )

        assert any("无法连接到后端服务" in h for h in st.markdown_calls)


# ---------------------------------------------------------------------------
# Retryable error routing
# ---------------------------------------------------------------------------


class TestRetryableErrorRouting:
    """Errors marked ``retryable=True`` that are not 401/429/5xx must render
    the actionable-error panel with a warning tone."""

    def test_retryable_error_renders_actionable_error(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "保存ROI",
            _raise(
                ApiClientError(
                    status_code=409,
                    code="CONFLICT",
                    message="Resource version mismatch",
                    request_id="req_030",
                    retryable=True,
                )
            ),
        )

        assert len(st.markdown_calls) >= 1
        html = st.markdown_calls[0]
        assert "nl-error-panel" in html
        assert "CONFLICT" in html
        assert "nl-error-panel-warn" in html  # warning tone, not bad

    def test_retryable_error_does_not_render_service_unavailable(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=0,
                    code="REQUEST_TIMEOUT",
                    message="timeout",
                    request_id="req_031",
                    retryable=True,
                )
            ),
        )

        combined = " ".join(st.markdown_calls)
        # Should render actionable error, not service unavailable
        assert "nl-error-panel" in combined


# ---------------------------------------------------------------------------
# Non-retryable error routing
# ---------------------------------------------------------------------------


class TestNonRetryableErrorRouting:
    """Non-retryable business errors fall through to ``render_exception``."""

    def test_non_retryable_error_renders_exception(self) -> None:
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "创建分析",
            _raise(
                ApiClientError(
                    status_code=400,
                    code="VALIDATION_ERROR",
                    message="Invalid input",
                    request_id="req_040",
                )
            ),
        )

        # render_exception uses st.error, not st.markdown
        assert len(st.error_calls) >= 1
        assert "VALIDATION_ERROR" in st.error_calls[0]

    def test_generic_exception_renders_exception(self) -> None:
        st = _FakeStreamlit()

        def _op() -> Any:
            raise ValueError("unexpected")

        _api_action(st, _make_state(), "加载模型", _op)

        assert len(st.error_calls) >= 1
        assert "unexpected" in st.error_calls[0]


# ---------------------------------------------------------------------------
# Successful operation
# ---------------------------------------------------------------------------


class TestSuccessfulOperation:
    """On success, ``_api_action`` must return the result dict and clear
    any previously stored error."""

    def test_success_returns_result_and_clears_error(self) -> None:
        st = _FakeStreamlit()
        state = _make_state()
        state["last_error"] = {
            "action": "old",
            "code": "OLD",
            "message": "old",
            "request_id": "old",
        }

        def _op() -> Any:
            return {"id": "test_001", "status": "success"}

        result = _api_action(st, state, "加载模型", _op)

        assert result is not None
        assert result["id"] == "test_001"
        assert state["last_error"] is None

    def test_success_does_not_render_anything(self) -> None:
        st = _FakeStreamlit()

        def _op() -> Any:
            return {"id": "ok"}

        _api_action(st, _make_state(), "加载模型", _op)

        assert len(st.markdown_calls) == 0


# ---------------------------------------------------------------------------
# Retry button rendering (E-P1 task 4)
# ---------------------------------------------------------------------------


class TestRetryButtonRendering:
    """Verify that explicit retry buttons appear after recoverable errors."""

    def test_429_read_action_shows_retry_button(self) -> None:
        """Read actions that hit 429 must show a retry button."""
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="limited",
                    request_id="req_retry_1",
                    retry_after="10",
                )
            ),
        )
        assert any("重试" in label for label in st.button_calls)

    def test_429_write_action_does_not_show_retry_button(self) -> None:
        """Write actions that hit 429 must NOT show an auto-retry button."""
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "创建分析",
            _raise(
                ApiClientError(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="limited",
                    request_id="req_retry_2",
                    retry_after="10",
                )
            ),
        )
        assert not any("重试" in label for label in st.button_calls)

    def test_service_unavailable_shows_retry_button(self) -> None:
        """502/503 errors must show a retry-connection button."""
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=502,
                    code="BAD_GATEWAY",
                    message="bad gateway",
                    request_id="req_retry_3",
                )
            ),
        )
        assert any("重试" in label for label in st.button_calls)

    def test_retryable_error_shows_retry_button(self) -> None:
        """Retryable errors must show a retry button."""
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=0,
                    code="REQUEST_TIMEOUT",
                    message="timeout",
                    request_id="req_retry_4",
                    retryable=True,
                )
            ),
        )
        assert any("重试" in label for label in st.button_calls)

    def test_401_does_not_show_retry_button(self) -> None:
        """401 auth errors must NOT show a retry button (requires operator action)."""
        st = _FakeStreamlit()
        _api_action(
            st,
            _make_state(),
            "加载模型",
            _raise(
                ApiClientError(
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message="invalid key",
                    request_id="req_retry_5",
                )
            ),
        )
        assert not any("重试" in label for label in st.button_calls)
        assert len(st.error_calls) == 0
