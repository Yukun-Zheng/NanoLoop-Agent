"""Unit tests for E-P1 structured feedback components.

Covers the three E-P1 sample slices: 401 operator guidance, 429 differentiated
retry messaging, and long-task / partial-failure rendering.
"""

from __future__ import annotations

from frontend.feedback import (
    LongTaskStage,
    build_actionable_error_html,
    build_auth_guidance_html,
    build_long_task_html,
    build_rate_limit_html,
    build_service_unavailable_html,
    parse_retry_after_seconds,
)


class TestParseRetryAfter:
    def test_none_returns_none(self) -> None:
        assert parse_retry_after_seconds(None) is None

    def test_integer_returns_float(self) -> None:
        assert parse_retry_after_seconds(7) == 7.0

    def test_float_returns_float(self) -> None:
        assert parse_retry_after_seconds(3.5) == 3.5

    def test_string_integer_returns_float(self) -> None:
        assert parse_retry_after_seconds("12") == 12.0

    def test_string_with_whitespace_returns_float(self) -> None:
        assert parse_retry_after_seconds("  8  ") == 8.0

    def test_http_date_string_returns_none(self) -> None:
        # We intentionally do not parse HTTP-dates; the server should send seconds.
        assert parse_retry_after_seconds("Wed, 21 Oct 2026 07:28:00 GMT") is None

    def test_negative_is_clamped_to_zero(self) -> None:
        assert parse_retry_after_seconds(-5) == 0.0

    def test_malformed_string_returns_none(self) -> None:
        assert parse_retry_after_seconds("abc") is None

    def test_bool_is_treated_as_missing(self) -> None:
        # bool is a subclass of int but semantically not a number here
        assert parse_retry_after_seconds(True) is None


class TestBuildAuthGuidance:
    def test_renders_operator_env_vars(self) -> None:
        html = build_auth_guidance_html()
        assert "NANOLOOP_API_KEY" in html
        assert "NANOLOOP_API_BASE_URL" in html
        assert "运维" in html
        assert 'role="alert"' in html

    def test_does_not_leak_key_value(self) -> None:
        # The panel must never render the actual key content
        html = build_auth_guidance_html()
        # Should reference env vars by name, not contain any actual secret-like value
        assert "sk-" not in html
        assert "bearer" not in html.lower()

    def test_contact_hint_rendered_when_provided(self) -> None:
        html = build_auth_guidance_html(contact_hint="联系 ops@example.com")
        assert "联系 ops@example.com" in html


class TestBuildRateLimit:
    def test_read_request_shows_countdown(self) -> None:
        html = build_rate_limit_html(retry_after_seconds=7.0, is_read_request=True)
        assert "7" in html
        assert "秒" in html
        assert "读取" in html
        assert "可安全重试" in html
        assert 'role="alert"' in html

    def test_write_request_forbids_auto_replay(self) -> None:
        html = build_rate_limit_html(retry_after_seconds=3.0, is_read_request=False)
        assert "写入" in html
        assert "不会" in html
        assert "自动重试" in html

    def test_missing_retry_after_shows_generic_wait(self) -> None:
        html = build_rate_limit_html(retry_after_seconds=None, is_read_request=True)
        assert "稍后" in html

    def test_request_id_rendered_when_provided(self) -> None:
        html = build_rate_limit_html(
            retry_after_seconds=1.0,
            is_read_request=True,
            request_id="req_abc123",
        )
        assert "req_abc123" in html


class TestBuildActionableError:
    def test_includes_title_code_message(self) -> None:
        html = build_actionable_error_html(
            title="创建运行失败",
            code="SCHEMA_VALIDATION_FAILED",
            message="ROI 框超出图像边界",
        )
        assert "创建运行失败" in html
        assert "SCHEMA_VALIDATION_FAILED" in html
        assert "ROI 框超出图像边界" in html
        assert 'role="alert"' in html

    def test_recovery_link_rendered_when_provided(self) -> None:
        html = build_actionable_error_html(
            title="失败",
            code="E",
            message="msg",
            recovery_label="重试",
            recovery_anchor="retry-anchor",
        )
        assert 'href="#retry-anchor"' in html
        assert "重试" in html

    def test_request_id_optional(self) -> None:
        html = build_actionable_error_html(title="t", code="c", message="m", request_id="req_x")
        assert "req_x" in html


class TestBuildServiceUnavailable:
    def test_renders_unreachable_title(self) -> None:
        html = build_service_unavailable_html(code="HTTP_502")
        assert "无法连接到后端服务" in html
        assert "HTTP_502" in html
        assert 'role="alert"' in html

    def test_renders_base_url_when_provided(self) -> None:
        html = build_service_unavailable_html(code="HTTP_502", base_url="http://localhost:8000")
        assert "http://localhost:8000" in html

    def test_default_hint_mentions_common_causes(self) -> None:
        html = build_service_unavailable_html(code="HTTP_503")
        assert "后端进程未启动" in html
        assert "反向代理" in html

    def test_custom_hint_overrides_default(self) -> None:
        html = build_service_unavailable_html(code="HTTP_502", hint="自定义提示")
        assert "自定义提示" in html


class TestBuildLongTask:
    def test_renders_all_stages(self) -> None:
        html = build_long_task_html(
            title="Run X",
            stages=[
                LongTaskStage(label="上传", status="completed"),
                LongTaskStage(label="推理", status="active"),
                LongTaskStage(label="报告", status="pending"),
            ],
        )
        assert "上传" in html
        assert "推理" in html
        assert "报告" in html
        assert 'role="status"' in html
        assert 'aria-live="polite"' in html

    def test_partial_failure_banner(self) -> None:
        html = build_long_task_html(
            title="Run X",
            stages=[LongTaskStage(label="a", status="completed")],
            partial_failure_count=2,
        )
        assert "2 个图像/运行部分失败" in html

    def test_recovery_action_rendered(self) -> None:
        html = build_long_task_html(
            title="Run X",
            stages=[LongTaskStage(label="a", status="failed")],
            recoverable_action_label="仅重试失败图像",
            recoverable_action_anchor="retry",
        )
        assert 'href="#retry"' in html
        assert "仅重试失败图像" in html

    def test_stage_detail_rendered(self) -> None:
        html = build_long_task_html(
            title="Run X",
            stages=[
                LongTaskStage(label="推理", status="failed", detail="img_003 OOM"),
            ],
        )
        assert "img_003 OOM" in html

    def test_summary_rendered(self) -> None:
        html = build_long_task_html(
            title="Run X",
            stages=[],
            summary="2/3 完成",
        )
        assert "2/3 完成" in html
