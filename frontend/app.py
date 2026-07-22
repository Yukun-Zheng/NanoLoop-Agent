"""NanoLoop Streamlit workbench backed exclusively by the versioned REST API."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Any, TypeVar

from frontend.components import (
    display_enum,
    render_artifact_links,
    render_connection_status,
    render_empty,
    render_exception,
    render_health_matrix,
    render_job_overview,
    render_query_response,
    render_run_summary,
    render_run_table,
    section_header,
    status_badge,
)
from frontend.feedback import (
    LongTaskStage,
    render_actionable_error,
    render_auth_guidance,
    render_long_task,
    render_rate_limit,
    render_service_unavailable,
)
from frontend.a11y import (
    render_skip_link,
    render_status_announcement,
)
from frontend.model_catalog import (
    model_availability,
    model_filter_query,
    model_is_runnable,
)
from frontend.result_layers import (
    RESULT_LAYER_LABELS,
    RESULT_LAYER_ORDER,
    prepare_result_layer_display,
    result_layer_sources,
)
from frontend.roi_canvas import (
    RoiCanvasPreview,
    parse_canvas_change,
    prepare_roi_preview,
    render_roi_canvas,
)
from frontend.state import (
    DEFAULT_API_BASE_URL,
    append_history,
    as_dict,
    build_analysis_metadata,
    build_run_payload,
    comparable_runs_by_image,
    ensure_session_state,
    exportable_run_ids,
    health_rollup,
    knowledge_document_toggle,
    normalize_base_url,
    parse_aliases,
    parse_json_object,
    pollable_run_ids,
    preferred_preview_artifact,
    rows_from_editor,
    select_comparison_runs,
    validate_box_rows,
)
from frontend.styles import PAGE_CONFIG, apply_styles

T = TypeVar("T")
State = MutableMapping[str, Any]
PAGES = (
    "Connection",
    "Project",
    "ROI & models",
    "Runs & results",
    "Ask NanoLoop",
    "Knowledge base",
)
PAGE_LABELS = {
    "Connection": "连接与能力",
    "Project": "分析项目",
    "ROI & models": "ROI 与模型",
    "Runs & results": "运行与结果",
    "Ask NanoLoop": "证据问答",
    "Knowledge base": "知识库",
}
ARTIFACT_LABELS = {
    "mask_url": "分割掩膜",
    "overlay_url": "叠加预览",
    "probability_url": "概率图",
    "instances_url": "实例数据",
    "labeled_particles_url": "颗粒标注图",
    "particles_csv_url": "颗粒 CSV",
    "quality_report_url": "质量报告",
    "execution_provenance_url": "执行溯源记录",
}


def _inject_menu_i18n(st: Any) -> None:
    """Translate the Streamlit hamburger menu into Chinese.

    The menu labels are hard-coded by Streamlit, so we inject a tiny script
    via st.components.v1.html (an iframe) that reaches into the parent
    document to replace text nodes once the menu portal is rendered.
    Theme picker buttons are already hidden via CSS so users cannot switch
    away from the configured Light theme.
    """
    try:
        components = importlib.import_module("streamlit.components.v1")
    except Exception:
        return
    components.html(
        """
        <script>
        (function () {
          var parentDoc = window.parent.document;
          /* Set the document language to Chinese for screen readers (WCAG 3.1.1).
             Streamlit defaults to lang="en" which causes Chinese text to be
             mispronounced by assistive technology. */
          if (parentDoc.documentElement) {
            parentDoc.documentElement.lang = "zh-CN";
          }
          var i18n = {
            "System": "跟随系统",
            "Light": "浅色",
            "Dark": "深色",
            "Rerun": "重新运行",
            "Auto rerun": "自动重新运行",
            "Clear cache": "清除缓存",
            "Print": "打印",
            "Record screen": "录制屏幕",
            "About": "关于",
            "Made with Streamlit": "由 Streamlit 驱动"
          };

          function translateNode(node) {
            if (node.nodeType !== Node.TEXT_NODE) return;
            var text = node.textContent.trim();
            if (i18n[text]) {
              node.textContent = node.textContent.replace(text, i18n[text]);
            }
          }

          function translateMenu(root) {
            var walker = parentDoc.createTreeWalker(
              root, NodeFilter.SHOW_TEXT, null, false
            );
            var nodes = [];
            var n;
            while ((n = walker.nextNode())) nodes.push(n);
            nodes.forEach(translateNode);
          }

          /* Hide Streamlit's "Press Enter to submit" help caption that
             appears under text inputs when focused.  CSS selectors are
             fragile across Streamlit versions, so we match by text content. */
          function hideHelpCaptions(root) {
            var walker = parentDoc.createTreeWalker(
              root, NodeFilter.SHOW_TEXT, null, false
            );
            var n;
            while ((n = walker.nextNode())) {
              var t = n.textContent.trim();
              if (t.indexOf("Press Enter to submit") !== -1 ||
                  t.indexOf("press enter to submit") !== -1) {
                var el = n.parentElement;
                if (el) el.style.display = "none";
              }
            }
          }

          /* Sidebar input borders are now handled entirely by CSS animation
             override (see styles.py).  CSS active-animation priority outranks
             inline styles, so BaseWeb's focus JS can no longer re-add its
             border/shadow.  No JS intervention needed. */

          /* NOTE: We intentionally do NOT enforce number_input min_value via
             JS.  Streamlit's number_input is a React controlled input —
             directly setting input.value is ignored by React (it resets the
             DOM to its internal state on the next render), and setting the
             HTML min attribute causes a visual "flash to min" when Streamlit
             re-renders.  Instead, min_value enforcement is done reliably on
             the Python side via max(10.0, ...) clamping + a visible
             st.warning when the user submits a value below 10. */

          var observer = new parentDoc.defaultView.MutationObserver(function (mutations) {
            mutations.forEach(function (m) {
              m.addedNodes.forEach(function (node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                  translateMenu(node);
                  hideHelpCaptions(node);
                }
              });
            });
          });

          observer.observe(parentDoc.body, { childList: true, subtree: true });
          translateMenu(parentDoc.body);
          hideHelpCaptions(parentDoc.body);
          /* Re-run help-caption hiding after a short delay — Streamlit
             re-renders widgets asynchronously, so the first pass may miss. */
          setTimeout(function () { hideHelpCaptions(parentDoc.body); }, 500);
          setTimeout(function () { hideHelpCaptions(parentDoc.body); }, 1500);
        })();
        </script>
        """,
        width=0,
        height=0,
    )



def main() -> None:
    st = importlib.import_module("streamlit")
    st.set_page_config(**PAGE_CONFIG)
    apply_styles(st)
    _inject_menu_i18n(st)
    state = ensure_session_state(st.session_state)
    client = _sidebar(st, state)

    # 全局跳转链接 (WCAG 2.4.1) — 每个页面都出现，不止 Connection 页。
    render_skip_link(st, target_id="nl-main-content", label="跳转到主内容")
    # 主地标 (WCAG 1.3.1) — 给屏幕阅读器一个跳转目标。
    st.markdown('<div id="nl-main-content" role="main"></div>', unsafe_allow_html=True)

    page = str(state["navigation"])
    if page == "Connection":
        _connection_page(st, state, client)
    elif page == "Project":
        _project_page(st, state, client)
    elif page == "ROI & models":
        _roi_models_page(st, state, client)
    elif page == "Runs & results":
        _runs_page(st, state, client)
    elif page == "Ask NanoLoop":
        _query_page(st, state, client)
    elif page == "Knowledge base":
        _knowledge_page(st, state, client)


def _sidebar(st: Any, state: State) -> Any | None:
    api_key_configured = bool(os.getenv("NANOLOOP_API_KEY"))
    configured_base_value = os.getenv("NANOLOOP_API_BASE_URL", DEFAULT_API_BASE_URL)
    if api_key_configured:
        try:
            locked_base_url = normalize_base_url(configured_base_value)
        except ValueError:
            locked_base_url = None
        if locked_base_url is not None and state["api_base_url"] != locked_base_url:
            state["api_base_url"] = locked_base_url
            state["health"] = None
            _drop_client(state)

    with st.sidebar:
        st.markdown(
            '<div class="nl-brand"><span class="nl-brand-mark">◌</span>'
            "<strong>NanoLoop Agent</strong>"
            "<small>可追溯 SEM 科研工作台</small></div>",
            unsafe_allow_html=True,
        )
        render_connection_status(st, _mapping_or_none(state.get("health")))

        with st.expander("API 连接", expanded=state.get("health") is None):
            with st.form("connection_settings"):
                base_url = st.text_input(
                    "后端服务地址",
                    value=str(state["api_base_url"]),
                    help=(
                        "已由 NANOLOOP_API_BASE_URL 锁定，避免共享 API Key 被发送到其他地址。"
                        if api_key_configured
                        else "请填写服务根地址；客户端会自动添加 /api/v1。"
                    ),
                    disabled=api_key_configured,
                )
                timeout = st.number_input(
                    "请求超时（秒）",
                    min_value=2.0,
                    max_value=300.0,
                    value=float(state["api_timeout_seconds"]),
                    step=1.0,
                )
                apply_connection = st.form_submit_button("保存连接设置", use_container_width=True)
            if apply_connection:
                try:
                    normalized = normalize_base_url(
                        configured_base_value if api_key_configured else base_url
                    )
                    changed = (
                        normalized != state["api_base_url"]
                        or float(timeout) != state["api_timeout_seconds"]
                    )
                    state["api_base_url"] = normalized
                    state["api_timeout_seconds"] = float(timeout)
                    if changed:
                        _drop_client(state)
                        state["health"] = None
                    st.success("连接设置已保存。")
                    st.rerun()
                except ValueError as error:
                    st.error(_localized_error(error))

        client = _get_client(st, state)
        refresh_health = st.button("检查连接", use_container_width=True)
        if refresh_health and client is not None:
            _refresh_health(st, state, client)

        st.divider()
        st.radio(
            "工作区",
            PAGES,
            key="navigation",
            label_visibility="collapsed",
            format_func=lambda value: PAGE_LABELS[value],
        )
        st.divider()
        with st.form("load_project_sidebar"):
            job_id = st.text_input(
                "按 job_id 打开项目",
                value=str(state.get("active_job_id") or ""),
                placeholder="job_…",
            )
            load = st.form_submit_button("加载项目", use_container_width=True)
        if load and client is not None:
            if job_id.strip():
                _load_job(st, state, client, job_id.strip())
            else:
                st.warning("请输入 job_id。")

        detail = _mapping_or_none(state.get("job_detail"))
        if detail:
            job = _mapping_or_none(detail.get("job")) or {}
            st.caption("当前项目")
            st.write(job.get("name") or job.get("job_id"))
            st.markdown(
                status_badge(str(job.get("status", "unknown"))),
                unsafe_allow_html=True,
            )
    return client


def _connection_page(st: Any, state: State, client: Any | None) -> None:
    section_header(
        st,
        eyebrow="系统就绪状态",
        title="连接与能力检查",
        description=(
            "开始分析前逐项确认后端能力。模型或知识组件降级时会持续明确显示，"
            "且不会阻断无关的确定性分析流程。"
        ),
    )
    # E-P1 样本展示：结构化反馈与可访问性原语，即使后端未运行也可见。
    _render_ep1_showcase(st)
    health = _mapping_or_none(state.get("health"))
    if not health:
        render_empty(
            st,
            "尚未检查系统状态",
            "请在侧边栏点击“检查连接”。工作台不会预设模型、数据库或 RAG 索引可用。",
        )
        return
    render_health_matrix(st, health)
    rollup = health_rollup(health)
    if rollup.status == "unavailable":
        st.error("核心 API 或数据库不可用，当前不能执行写入类操作。")
    elif rollup.status == "degraded":
        st.warning("API 已连接，但部分能力受限。启动模型推理或知识查询前，请先检查组件状态。")
    else:
        st.success("所有组件均报告为正常。")
    columns = st.columns(3)
    columns[0].metric("API 版本", health.get("version") or "—")
    columns[1].metric("后端地址", str(state["api_base_url"]), help=None)
    columns[2].metric("请求超时", f"{state['api_timeout_seconds']:g} 秒")
    st.markdown(
        '<div class="nl-note">本界面仅调用带版本的 REST API，'
        "不会直接读取 SQLite、模型文件、知识源或输出目录。"
        "</div>",
        unsafe_allow_html=True,
    )
    if client is None:
        st.error("无法创建 REST 客户端，请检查连接设置。")


def _render_ep1_showcase(st: Any) -> None:
    """E-P1 样本展示：结构化反馈与可访问性原语。

    刻意放在 Connection 页、health 检查之前，即使后端未运行也可见；
    不改变其他页面的行为。
    """
    st.divider()
    st.markdown("## E-P1 反馈与可访问性样本")
    st.caption(
        "以下样本展示了三类 E-P1 改进：键盘可达性 / 结构化错误恢复 / 长任务反馈。"
        "它们已经接到 _api_action 的错误分支，真实出现 401/429 时会自动启用。"
    )

    sample_tabs = st.tabs(("401 运维指引", "429 限流反馈", "长任务 / 部分失败", "键盘与屏幕阅读器"))

    with sample_tabs[0]:
        st.markdown("#### 共享 API Key 失效时的运维指引")
        st.caption(
            "替代当前仅显示一行 `401 AUTHENTICATION_REQUIRED` 的做法——"
            "明确告知运维该改哪两个环境变量、需要重启进程，并避免泄露 Key 本身。"
        )
        render_auth_guidance(
            st,
            contact_hint="若多次轮换 Key 仍失败，请联系后端运维核对限流桶与可信主机配置。",
        )

    with sample_tabs[1]:
        st.markdown("#### 限流响应的差异化处理")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**读取操作（可安全重试）**")
            render_rate_limit(
                st,
                retry_after_seconds=7.0,
                is_read_request=True,
                request_id="web_demo_read",
            )
        with col_b:
            st.markdown("**写入操作（绝不自动重放）**")
            render_rate_limit(
                st,
                retry_after_seconds=None,
                is_read_request=False,
                request_id="web_demo_write",
            )
        st.caption(
            "根据 `_is_read_action` 启发式判断：动作名包含“加载/刷新/检查/查看/获取/列出/推荐/预览/下载”"
            "的视为读取；其余视为写入。读取仅提示倒计时，不自动重放。"
        )

    with sample_tabs[2]:
        st.markdown("#### 长任务分阶段反馈 + 部分失败保留现场")
        render_long_task(
            st,
            title="运行 run_demo_7（U-Net @ v1）",
            summary="3 张图像 · 2 张已完成，1 张后处理失败",
            partial_failure_count=1,
            stages=[
                LongTaskStage(label="图像校验与上传", status="completed", detail="3/3 通过"),
                LongTaskStage(label="ROI 解析与模型加载", status="completed", detail="revision 12 · U-Net 就绪"),
                LongTaskStage(label="语义分割推理", status="completed", detail="3/3 完成"),
                LongTaskStage(
                    label="后处理与形貌统计",
                    status="failed",
                    detail="image img_003 连通域异常；其余 2 张已写入 canonical 实例。",
                ),
                LongTaskStage(label="报告生成", status="pending", detail="等待失败图像处理决策"),
            ],
            recoverable_action_label="仅重试失败图像",
            recoverable_action_anchor="nl-recover-demo",
        )
        st.caption(
            "关键设计：失败的图像被单独标记，其余结果仍可查看 / 下载 / 对比；"
            "用户不会被强迫重新创建整个 run。"
        )

    with sample_tabs[3]:
        st.markdown("#### 键盘可达性与屏幕阅读器支持")
        render_status_announcement(
            st,
            role="status",
            tone="live",
            title="aria-live 区域示例",
            body=(
                "此面板会被屏幕阅读器自动朗读，无需用户手动查找。"
                "页面顶部还有“跳转到主内容”链接，按 Tab 即可看到。"
            ),
        )
        st.markdown(
            """
- **跳转链接**：页面顶部隐藏链接，键盘 Tab 第一次聚焦时显示（WCAG 2.4.1）。
- **焦点环**：所有交互元素都有 3px 高对比度 focus-visible 环；侧栏使用亮黄色以适配深底。
- **屏幕阅读器专用文本**：`.nl-sr-only` 类隐藏视觉但保留朗读。
- **aria-live 区域**：状态变化通过 `role="status"` / `role="alert"` 自动播报。
- **减少动画**：尊重 `prefers-reduced-motion`，所有过渡均降级。
            """.strip()
        )



def _project_page(st: Any, state: State, client: Any | None) -> None:
    section_header(
        st,
        eyebrow="01 · 项目接入",
        title="创建或查看分析项目",
        description=(
            "上传 1–20 张显微图像，并在模型运行前明确记录样品、材料、实验条件和尺度信息。"
        ),
    )
    detail = _mapping_or_none(state.get("job_detail"))
    if detail:
        render_job_overview(st, detail)
        images = _list_of_mappings(detail.get("images"))
        if images:
            st.dataframe(
                [
                    {
                        "image_id": image.get("image_id"),
                        "filename": image.get("filename"),
                        "sample_id": image.get("sample_id"),
                        "材料": image.get("material_formula")
                        or image.get("material_name")
                        or "未提供",
                        "尺寸": f"{image.get('width')}×{image.get('height')}",
                        "位深": image.get("bit_depth"),
                        "尺度 (nm/pixel)": image.get("scale_nm_per_pixel"),
                    }
                    for image in images
                ],
                hide_index=True,
                use_container_width=True,
            )
        if st.button("刷新当前项目") and client is not None:
            _load_job(st, state, client, str(state["active_job_id"]))

    st.markdown("## 新建项目")
    uploads = st.file_uploader(
        "显微图像",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="请选择 1–20 个文件，原始文件名不能重复。",
        key="project_images",
    )
    if not uploads:
        render_empty(
            st,
            "等待选择图像",
            "选择文件后将显示元数据字段；系统不会猜测样品信息或尺度。",
        )
        return
    if len(uploads) > 20:
        st.error("一个项目最多接收 20 张图像。")
        return
    filenames = [str(upload.name) for upload in uploads]
    if len(set(filenames)) != len(filenames):
        st.error("所选文件中存在重名，请重命名后再上传。")
        return

    drafts: dict[str, dict[str, Any]] = {}
    with st.form("create_project"):
        job_name = st.text_input("项目名称", placeholder="必填")
        for upload in uploads:
            filename = str(upload.name)
            key = _widget_key(filename)
            with st.expander(filename, expanded=len(uploads) <= 3):
                columns = st.columns(2)
                sample_id = columns[0].text_input(
                    "样品 ID *", key=f"sample_{key}", placeholder="必填"
                )
                formula = columns[1].text_input(
                    "材料化学式", key=f"formula_{key}", placeholder="选填"
                )
                material_name = columns[0].text_input(
                    "材料名称", key=f"material_{key}", placeholder="选填"
                )
                scale_mode = columns[1].selectbox(
                    "尺度模式",
                    ("pixel_only", "nm_per_pixel"),
                    format_func=lambda value: display_enum(value),
                    key=f"scale_mode_{key}",
                )
                scale_value = columns[1].number_input(
                    "尺度（nm/pixel）",
                    min_value=0.000001,
                    value=None,
                    placeholder=(
                        "物理尺度模式下必填" if scale_mode == "nm_per_pixel" else "仅像素模式下忽略"
                    ),
                    key=f"scale_value_{key}",
                )
                conditions_text = st.text_area(
                    "实验条件（JSON 对象）",
                    value="{}",
                    key=f"conditions_{key}",
                    height=84,
                )
                drafts[filename] = {
                    "sample_id": sample_id,
                    "material_formula": formula,
                    "material_name": material_name,
                    "scale_mode": scale_mode,
                    "scale_value": scale_value,
                    "conditions_text": conditions_text,
                }
        create = st.form_submit_button("创建分析项目", type="primary", use_container_width=True)

    if not create:
        return
    if client is None:
        st.error("请先连接后端，再创建项目。")
        return
    try:
        for filename, values in drafts.items():
            values["experiment_conditions"] = parse_json_object(
                str(values.pop("conditions_text")),
                field_name=f"{filename} 实验条件",
            )
        metadata = build_analysis_metadata(job_name, filenames, drafts)
    except (TypeError, ValueError) as error:
        st.error(_localized_error(error))
        return
    upload_parts = [_upload_part(upload) for upload in uploads]
    result = _api_action(
        st,
        state,
        "创建项目",
        lambda: client.create_analysis(upload_parts, metadata),
    )
    if result is not None:
        _store_job_detail(state, result)
        st.success(f"项目 {state['active_job_id']} 已创建并完成校验。")


def _roi_models_page(st: Any, state: State, client: Any | None) -> None:
    section_header(
        st,
        eyebrow="02 · 分析配置",
        title="ROI、模型就绪状态与运行提交",
        description=(
            "在原图画布上拖拽 ROI，并用数值表格精调半开区间像素坐标；"
            "每次保存都明确保留框选版本。"
        ),
    )
    detail = _require_job(st, state)
    if detail is None:
        return
    images = _list_of_mappings(detail.get("images"))
    image_by_id = {str(image["image_id"]): image for image in images if image.get("image_id")}
    if not image_by_id:
        st.error("当前项目没有已通过校验的图像。")
        return
    valid_defaults = [
        image_id for image_id in state.get("selected_image_ids", []) if image_id in image_by_id
    ]
    selected_images = st.multiselect(
        "下一次运行使用的图像",
        list(image_by_id),
        default=valid_defaults or [next(iter(image_by_id))],
        format_func=lambda image_id: (
            f"{image_by_id[image_id].get('filename')} · {image_by_id[image_id].get('sample_id')}"
        ),
        key="run_image_selection",
        max_selections=20,
    )
    state["selected_image_ids"] = selected_images
    active_image_id = st.selectbox(
        "ROI 编辑图像",
        list(image_by_id),
        format_func=lambda image_id: str(image_by_id[image_id].get("filename")),
    )
    image = image_by_id[active_image_id]
    analysis_roi = _mapping_or_none(image.get("analysis_roi")) or {}
    valid_rect = _mapping_or_none(analysis_roi.get("valid_rect"))
    invalid_rects = _list_of_mappings(analysis_roi.get("invalid_rects"))
    load_column, context_column = st.columns([1, 3])
    if load_column.button("加载 / 刷新 ROI revision", use_container_width=True):
        if client is None:
            st.error("请先连接后端，再加载 ROI 框。")
        else:
            loaded_box_set = _api_action(
                st,
                state,
                "加载 ROI 框",
                lambda: client.get_boxes(str(state["active_job_id"]), active_image_id),
            )
            if loaded_box_set is not None:
                state["box_sets"][active_image_id] = loaded_box_set
                _reset_roi_draft(state, active_image_id, loaded_box_set)
                st.rerun()
    context_column.caption(
        f"{image.get('width')}×{image.get('height')} px · {image.get('bit_depth')} bit · "
        f"尺度 {image.get('scale_nm_per_pixel') or '仅像素'} · "
        f"analysis_roi revision {analysis_roi.get('revision', '—')}"
    )

    box_set = _mapping_or_none(state.get("box_sets", {}).get(active_image_id))
    draft = _roi_draft(state, active_image_id, box_set) if box_set is not None else None
    canvas_column, editor_column = st.columns([1.55, 1])

    with editor_column:
        st.markdown("### 数值精调 / 回退")
        st.caption(
            "坐标使用原图像素和半开区间 [x1:x2, y1:y2]。空白行会被忽略；"
            "单边至少 32 px；所有框必须位于 valid_rect 内，且不得与 invalid_rects 相交。"
        )
        if box_set is None or draft is None:
            render_empty(
                st,
                "尚未加载 ROI revision",
                "编辑前请加载服务端当前 revision，以保持乐观锁一致性。",
            )
        else:
            draft_rows = _list_of_mappings(draft.get("rows"))
            rows = draft_rows or [
                {
                    "box_id": None,
                    "label": "",
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                    "active": True,
                }
            ]
            edited = st.data_editor(
                rows,
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key=(
                    f"roi_editor_{active_image_id}_{box_set.get('revision', 0)}_"
                    f"{draft.get('editor_generation', 0)}"
                ),
                column_config={
                    "box_id": st.column_config.TextColumn("Box ID", disabled=True),
                    "label": st.column_config.TextColumn("标签"),
                    "x1": st.column_config.NumberColumn("x1", min_value=0, step=1),
                    "y1": st.column_config.NumberColumn("y1", min_value=0, step=1),
                    "x2": st.column_config.NumberColumn("x2", min_value=1, step=1),
                    "y2": st.column_config.NumberColumn("y2", min_value=1, step=1),
                    "active": st.column_config.CheckboxColumn("启用"),
                },
            )
            edited_rows = rows_from_editor(edited)
            draft["rows"] = [dict(row) for row in edited_rows]
            if st.button("保存完整 ROI revision", type="primary"):
                validation = validate_box_rows(
                    edited_rows,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    valid_rect=valid_rect,
                    invalid_rects=invalid_rects,
                )
                if validation.errors:
                    for error in validation.errors:
                        st.error(_localized_error(error))
                elif client is None:
                    st.error("请先连接后端，再保存 ROI 框。")
                else:
                    saved = _api_action(
                        st,
                        state,
                        "保存 ROI revision",
                        lambda: client.replace_boxes(
                            str(state["active_job_id"]),
                            active_image_id,
                            expected_revision=int(box_set.get("revision", 0)),
                            boxes=list(validation.boxes),
                        ),
                    )
                    if saved is not None:
                        state["box_sets"][active_image_id] = saved
                        _reset_roi_draft(state, active_image_id, saved)
                        st.success(f"ROI revision {saved.get('revision')} 已保存。")

    with canvas_column:
        st.markdown("### 图像 ROI 画布")
        st.caption(
            "画布缩放只改变显示，不改变坐标。拖拽端点按 floor/ceil 映射为原图半开区间；"
            "青色虚线为 valid_rect，红色阴影为 invalid_rect。"
        )
        canvas_preview = _roi_canvas_preview(st, state, client, image)
        if canvas_preview is None:
            render_empty(st, "预览不可用", "无法获取或解码受管原始图像。")
        else:
            canvas_rows = _drawable_roi_rows(draft.get("rows") if draft is not None else [])
            raw_change = render_roi_canvas(
                preview=canvas_preview,
                boxes=canvas_rows,
                valid_rect=valid_rect,
                invalid_rects=invalid_rects,
                read_only=box_set is None,
                key=f"roi_canvas_{active_image_id}",
            )
            try:
                change = parse_canvas_change(raw_change)
            except ValueError as error:
                st.error(_localized_error(error))
            else:
                if change is not None and draft is not None:
                    events = state.get("roi_canvas_events")
                    if not isinstance(events, dict):
                        events = {}
                        state["roi_canvas_events"] = events
                    if events.get(active_image_id) != change.event_id:
                        events[active_image_id] = change.event_id
                        draft["rows"] = [dict(box) for box in change.boxes]
                        draft["editor_generation"] = int(draft.get("editor_generation", 0)) + 1
                        st.rerun()

        st.markdown("#### 分析区域约束")
        if valid_rect:
            st.caption(
                "valid_rect："
                f"[{valid_rect.get('x1')}:{valid_rect.get('x2')}, "
                f"{valid_rect.get('y1')}:{valid_rect.get('y2')}] · "
                f"来源 {display_enum(analysis_roi.get('source', '—'))}"
            )
        else:
            st.warning("后端未返回 analysis_roi.valid_rect，将仅按整幅图边界校验。")
        if invalid_rects:
            with st.expander(f"invalid_rects（{len(invalid_rects)}）"):
                st.dataframe(
                    [
                        {
                            "排除原因": display_enum(region.get("reason") or "未说明"),
                            "x1": region.get("x1"),
                            "y1": region.get("y1"),
                            "x2": region.get("x2"),
                            "y2": region.get("y2"),
                        }
                        for region in invalid_rects
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.caption("没有 invalid_rects。")

    st.divider()
    _model_configuration(st, state, client, active_image_id, selected_images)


def _model_configuration(
    st: Any,
    state: State,
    client: Any | None,
    active_image_id: str,
    selected_images: Sequence[str],
) -> None:
    st.markdown("## 模型注册表")
    st.caption(
        "筛选条件由后端 GET /models 执行；材料名称为不区分大小写的精确匹配。"
    )
    with st.form("model_registry_filters"):
        filters = st.columns(5)
        family = filters[0].selectbox(
            "模型族",
            (None, "unet", "yolo_seg", "sam2"),
            format_func=lambda value: "全部" if value is None else display_enum(value),
            key="model_filter_family",
        )
        variant = filters[1].selectbox(
            "变体",
            (
                None,
                "general",
                "small_particle",
                "large_particle",
                "dense_particle",
                "low_contrast",
            ),
            format_func=lambda value: "全部" if value is None else display_enum(value),
            key="model_filter_variant",
        )
        quality_tier = filters[2].selectbox(
            "质量档位",
            (None, "fast", "balanced", "accurate"),
            format_func=lambda value: "全部" if value is None else display_enum(value),
            key="model_filter_quality_tier",
        )
        model_status = filters[3].selectbox(
            "状态",
            (None, "ready", "loading", "unavailable", "disabled"),
            format_func=lambda value: "全部" if value is None else display_enum(value),
            key="model_filter_status",
        )
        material = filters[4].text_input(
            "适用材料",
            placeholder="例如 TiO2",
            max_chars=255,
            key="model_filter_material",
        )
        refresh_models = st.form_submit_button("应用筛选 / 刷新模型", use_container_width=True)

    query = model_filter_query(
        family=family,
        variant=variant,
        quality_tier=quality_tier,
        status=model_status,
        material=material,
    )
    if refresh_models and client is None:
        st.error("请先连接后端，再加载模型注册表。")
    elif refresh_models and client is not None:
        result = _api_action(
            st,
            state,
            "加载模型",
            lambda: client.list_models(**query),
        )
        if result is not None:
            state["models"] = result.get("models") or []
            state["models_loaded"] = True
            state["model_catalog_filters"] = query
    models = _list_of_mappings(state.get("models"))
    if models:
        active_filters = {
            key: value
            for key, value in (_mapping_or_none(state.get("model_catalog_filters")) or {}).items()
            if value
        }
        filter_summary = " · ".join(
            f"{key}={value}" for key, value in active_filters.items()
        )
        st.caption(
            f"后端返回 {len(models)} 个模型"
            + (f" · {filter_summary}" if filter_summary else " · 未限定筛选")
        )
        st.dataframe(
            [
                {
                    "model_id": model.get("model_id"),
                    "版本": model.get("version"),
                    "模型族": display_enum(model.get("family")),
                    "变体": display_enum(model.get("variant")),
                    "质量档位": display_enum(model.get("quality_tier")),
                    "状态": display_enum(model.get("status")),
                    "可提交运行": "是" if model_is_runnable(model) else "否",
                    "适用材料": "、".join(map(str, model.get("applicable_materials") or []))
                    or "未声明",
                    "支持框提示": "是" if model.get("supports_box_prompt") else "否",
                    "默认阈值": model.get("default_threshold"),
                    "健康原因": model.get("health_error") or "—",
                }
                for model in models
            ],
            hide_index=True,
            use_container_width=True,
        )
        _render_model_detail(st, state, models)
    else:
        if state.get("models_loaded"):
            render_empty(
                st,
                "没有匹配当前筛选的模型",
                "请放宽筛选条件后重试；工作台不会伪造可用模型。",
            )
        else:
            render_empty(
                st,
                "尚未加载模型记录",
                "请刷新注册表。缺少权重的模型必须显示为“不可用”，不能作为候选项。",
            )

    with st.expander("请求模型推荐", expanded=not state.get("recommendations")):
        columns = st.columns(4)
        roi_mode_for_recommendation = columns[0].selectbox(
            "ROI 模式",
            ("full_image", "boxes"),
            key="recommend_roi_mode",
            format_func=display_enum,
        )
        target_profile = columns[1].selectbox(
            "目标特征",
            ("general", "small_particle", "large_particle", "dense_particle", "low_contrast"),
            format_func=display_enum,
        )
        prefer = columns[2].selectbox(
            "偏好", ("accuracy", "balance", "speed"), format_func=display_enum
        )
        device = columns[3].selectbox(
            "设备", ("auto", "cpu", "cuda", "mps"), format_func=display_enum
        )
        if st.button("推荐模型", use_container_width=True) and client is not None:
            recommendation = _api_action(
                st,
                state,
                "推荐模型",
                lambda: client.recommend_models(
                    {
                        "image_id": active_image_id,
                        "roi_mode": roi_mode_for_recommendation,
                        "target_profile": target_profile,
                        "prefer": prefer,
                        "device": device,
                        "max_gpu_memory_mb": None,
                    }
                ),
            )
            if recommendation is not None:
                state["recommendations"] = recommendation.get("candidates") or []

        recommendations = _list_of_mappings(state.get("recommendations"))
        if recommendations:
            st.warning("模型推荐必须经过人工确认，之后才能创建运行。")
            st.dataframe(
                [
                    {
                        "model_id": item.get("model_id"),
                        "推荐分数": item.get("score"),
                        "推荐理由": "；".join(map(str, item.get("reasons") or [])),
                    }
                    for item in recommendations
                ],
                hide_index=True,
                use_container_width=True,
            )

    ready_models = {
        str(model["model_id"]): model
        for model in models
        if model.get("model_id") and model_is_runnable(model)
    }
    st.markdown("## 提交不可变运行")
    chosen_models = st.multiselect(
        "已确认的就绪模型（1–3 个）",
        list(ready_models),
        default=[
            model_id for model_id in state.get("selected_model_ids", []) if model_id in ready_models
        ],
        max_selections=3,
        format_func=lambda model_id: (
            f"{model_id} · {display_enum(ready_models[model_id].get('quality_tier'))}"
        ),
    )
    state["selected_model_ids"] = chosen_models
    config = st.columns(3)
    roi_mode = config[0].selectbox(
        "运行 ROI 模式", ("full_image", "boxes"), format_func=display_enum
    )
    threshold = config[1].number_input(
        "阈值覆盖值",
        min_value=0.0,
        max_value=1.0,
        value=None,
        placeholder="使用注册表默认值",
        step=0.01,
    )
    min_area = int(config[2].number_input("最小面积（px）", min_value=0, value=8, step=1))
    options = st.columns(3)
    watershed = options[0].checkbox("启用分水岭")
    exclude_border = options[1].checkbox("排除触边颗粒", value=True)
    run_device = options[2].selectbox(
        "推理设备", ("auto", "cpu", "cuda", "mps"), format_func=display_enum
    )
    if st.button("创建运行", type="primary", use_container_width=True):
        if client is None:
            st.error("请先连接后端，再创建运行。")
            return
        # E-P1: 防重复提交守卫——若上一次“创建运行”请求仍在处理中
        # （例如用户双击），拒绝创建重复 run。
        if state.get("_creating_runs"):
            st.warning("上一次创建运行的请求仍在处理中，请等待完成后再试。")
            return
        try:
            payload = build_run_payload(
                image_ids=selected_images,
                model_ids=chosen_models,
                roi_mode=roi_mode,
                box_sets=state.get("box_sets", {}),
                threshold=threshold,
                min_area_px=min_area,
                watershed_enabled=watershed,
                exclude_border=exclude_border,
                device=run_device,
            )
        except ValueError as error:
            st.error(_localized_error(error))
            return
        state["_creating_runs"] = True
        try:
            created = _api_action(
                st,
                state,
                "创建运行",
                lambda: client.create_runs(str(state["active_job_id"]), payload),
            )
        finally:
            state["_creating_runs"] = False
        if created is not None:
            run_ids = [str(run_id) for run_id in created.get("run_ids") or []]
            # 去重：即使后端在连续调用中重复返回，也绝不添加同一个 run_id 两次。
            existing = set(state.get("run_ids", []))
            new_ids = [rid for rid in run_ids if rid not in existing]
            state["run_ids"] = list(dict.fromkeys([*state.get("run_ids", []), *new_ids]))
            if new_ids:
                st.success(f"已提交 {len(new_ids)} 个新运行，请前往“运行与结果”监控进度。")
            else:
                st.info("这些运行已存在，未重复创建。")


def _render_model_detail(st: Any, state: State, models: Sequence[Mapping[str, Any]]) -> None:
    """Render every scientific and operational field for one selected registry model."""

    model_by_id = {
        str(model["model_id"]): dict(model)
        for model in models
        if model.get("model_id")
    }
    if not model_by_id:
        return
    selected_id = str(state.get("model_catalog_selected_id") or "")
    if selected_id not in model_by_id:
        state["model_catalog_selected_id"] = next(iter(model_by_id))

    selected_id = st.selectbox(
        "查看模型详情",
        list(model_by_id),
        key="model_catalog_selected_id",
        format_func=lambda model_id: (
            f"{model_id} · {display_enum(model_by_id[model_id].get('status'))}"
        ),
    )
    model = model_by_id[selected_id]
    availability = model_availability(model)

    st.markdown("### 选中模型详情")
    identity, status_column = st.columns([4, 1])
    identity.subheader(selected_id)
    identity.caption(
        f"版本 {model.get('version') or '未声明'} · "
        f"{display_enum(model.get('family'))} / {display_enum(model.get('variant'))} · "
        f"{display_enum(model.get('quality_tier'))}"
    )
    status_column.markdown(
        status_badge(str(model.get("status") or "unknown")),
        unsafe_allow_html=True,
    )
    getattr(st, availability.severity)(availability.message)

    attributes = st.columns(4)
    attributes[0].metric("支持框提示", "是" if model.get("supports_box_prompt") else "否")
    threshold = model.get("default_threshold")
    attributes[1].metric("默认阈值", "未声明" if threshold is None else str(threshold))
    materials = [str(item) for item in model.get("applicable_materials") or []]
    attributes[2].metric("适用材料数", str(len(materials)))
    attributes[3].metric("可提交运行", "是" if availability.runnable else "否")

    st.markdown("**适用材料**")
    st.caption("、".join(materials) if materials else "未声明适用材料。")
    profiles = st.columns(2)
    profiles[0].markdown("**预处理配置**")
    profiles[0].code(str(model.get("preprocess_profile") or "未声明"), language=None)
    profiles[1].markdown("**后处理配置**")
    profiles[1].code(str(model.get("postprocess_profile") or "未声明"), language=None)

    metrics = model.get("metrics")
    st.markdown("**验证指标（metrics）**")
    if isinstance(metrics, Mapping) and metrics:
        st.dataframe(
            [{"指标": str(name), "值": value} for name, value in metrics.items()],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("未报告验证指标。")

    metric_context = model.get("metric_context")
    st.markdown("**指标上下文（metric_context）**")
    if isinstance(metric_context, Mapping) and metric_context:
        st.json(dict(metric_context), expanded=True)
    else:
        st.caption("未报告指标上下文。")

    st.markdown("**备注（notes）**")
    st.write(str(model.get("notes") or "未提供备注。"))


def _runs_page(st: Any, state: State, client: Any | None) -> None:
    section_header(
        st,
        eyebrow="03 · 执行与复核",
        title="运行监控、结果与导出",
        description=(
            "轮询不可变运行，查看确定性指标与质量门禁，下载受管制品，并创建可审计的复核子运行。"
        ),
    )
    detail = _require_job(st, state)
    if detail is None:
        return
    run_ids = list(
        dict.fromkeys(
            [
                *state.get("run_ids", []),
                *[
                    str(run.get("run_id"))
                    for run in _list_of_mappings(detail.get("runs"))
                    if run.get("run_id")
                ],
            ]
        )
    )
    state["run_ids"] = run_ids
    if not run_ids:
        render_empty(
            st,
            "暂无可监控运行",
            "请先在“ROI 与模型”中创建运行；界面不会合成或补造结果。",
        )
        return

    controls = st.columns([1, 1, 2])
    auto_poll = controls[0].toggle("自动轮询", value=True)
    poll_seconds = controls[1].selectbox(
        "轮询间隔", (2, 5, 10, 20), index=1, format_func=lambda value: f"{value} 秒"
    )
    manual_refresh = controls[2].button("立即刷新全部", use_container_width=True)
    if manual_refresh and client is not None:
        _poll_runs(st, state, client, run_ids)

    def poll_panel() -> None:
        if auto_poll and client is not None and _has_pending_runs(state, run_ids):
            _poll_runs(st, state, client, run_ids, quiet=True)
        runs = _ordered_runs(state, run_ids)
        render_run_table(st, runs)

    if hasattr(st, "fragment"):
        fragment = st.fragment(run_every=poll_seconds if auto_poll else None)(poll_panel)
        fragment()
    else:  # pragma: no cover - compatibility fallback for older Streamlit
        poll_panel()

    runs = _ordered_runs(state, run_ids)
    available_ids = [str(run["run_id"]) for run in runs if run.get("run_id")]
    if not available_ids:
        st.info("尚未获取运行记录，请点击“立即刷新全部”。")
        return
    view_mode = st.radio(
        "结果视图",
        ("single", "comparison"),
        horizontal=True,
        format_func=lambda value: {
            "single": "单运行详情",
            "comparison": "同图像并排对比",
        }[value],
    )
    if view_mode == "comparison":
        _run_comparison_panel(st, state, client, detail, runs)
        return

    selected_run_id = st.selectbox("查看运行", available_ids)
    run = _mapping_or_none(state.get("runs", {}).get(selected_run_id))
    if run is None:
        return
    result_tab, artifacts_tab, review_tab, export_tab = st.tabs(
        ("结果", "制品", "复核子运行", "导出")
    )
    with result_tab:
        _single_run_layer_panel(st, state, client, detail, run)
        render_run_summary(st, run)
    with artifacts_tab:
        render_artifact_links(st, run)
        _artifact_download_panel(st, state, client, run)
    with review_tab:
        _review_panel(st, state, client, run)
    with export_tab:
        _export_panel(st, state, client, runs)


def _single_run_layer_panel(
    st: Any,
    state: State,
    client: Any | None,
    detail: Mapping[str, Any],
    run: Mapping[str, Any],
) -> None:
    run_id = str(run.get("run_id") or "unknown")
    image_id = str(run.get("image_id") or "")
    image = next(
        (
            item
            for item in _list_of_mappings(detail.get("images"))
            if item.get("image_id") == image_id
        ),
        None,
    )
    sources = result_layer_sources(dict(run), image)
    st.markdown("### 结果图层")
    if not sources:
        st.info("原始图像与结果制品均未发布可用的签名下载地址。")
        return

    by_key = {source.key: source for source in sources}
    available_keys = [key for key in RESULT_LAYER_ORDER if key in by_key]
    preferred_key = "overlay_url" if "overlay_url" in by_key else available_keys[0]
    selected_key = st.radio(
        "预览图层",
        available_keys,
        index=available_keys.index(preferred_key),
        horizontal=True,
        format_func=lambda key: RESULT_LAYER_LABELS[key],
        key=f"single_result_layer_{_widget_key(run_id)}",
    )
    unavailable = [RESULT_LAYER_LABELS[key] for key in RESULT_LAYER_ORDER if key not in by_key]
    if unavailable:
        st.caption("本运行未提供：" + "、".join(unavailable))

    source = by_key[selected_key]
    refresh = st.button(
        "刷新当前图层",
        key=f"single_result_layer_refresh_{_widget_key(run_id)}",
    )
    cache = _result_layer_cache(state)
    cache_key = f"{run_id}:{selected_key}"
    if refresh:
        cache.pop(cache_key, None)
    preview = _single_run_layer_preview(st, state, client, source, cache, cache_key)
    if preview is None:
        if client is None:
            st.info("连接后端后可通过签名 REST 地址加载该图层。")
        else:
            st.warning("当前图层暂时无法预览；可刷新重试或在“制品”页下载原文件。")
        return

    st.caption(
        f"{source.label} · {preview['width']} × {preview['height']} px · "
        "所有图层共用同一固定高度视口。"
    )
    with st.container(
        height=560,
        border=True,
        key=f"single_result_viewport_{_widget_key(run_id)}",
        vertical_alignment="center",
    ):
        st.image(
            preview["display_content"],
            caption=source.label,
            use_container_width=True,
        )
    note = preview.get("note")
    if note:
        st.info(str(note))
    st.download_button(
        f"下载原始{source.label}文件",
        data=preview["raw_content"],
        file_name=str(preview["filename"]),
        mime=str(preview["raw_content_type"]),
        key=f"single_result_layer_download_{_widget_key(run_id + selected_key)}",
    )


def _result_layer_cache(state: State) -> dict[str, Any]:
    raw_cache = state.get("result_layer_previews")
    if isinstance(raw_cache, dict):
        return raw_cache
    cache: dict[str, Any] = {}
    state["result_layer_previews"] = cache
    return cache


def _single_run_layer_preview(
    st: Any,
    state: State,
    client: Any | None,
    source: Any,
    cache: dict[str, Any],
    cache_key: str,
) -> dict[str, Any] | None:
    cached = _mapping_or_none(cache.get(cache_key))
    if cached and cached.get("source") == source.download_url:
        return cached
    if client is None:
        return None
    download = _download_action(st, state, client, source.download_url)
    if download is None:
        return None
    filename = download.filename or _artifact_fallback_filename(
        cache_key.split(":", maxsplit=1)[0],
        source.key,
        download.content_type,
    )
    try:
        display = prepare_result_layer_display(
            layer_key=source.key,
            content=download.content,
            content_type=download.content_type,
            filename=filename,
        )
    except ValueError as error:
        st.error(f"无法预览{source.label}：{error}")
        return None
    record = {
        "source": source.download_url,
        "display_content": display.content,
        "display_content_type": display.content_type,
        "raw_content": download.content,
        "raw_content_type": download.content_type,
        "filename": filename,
        "width": display.width,
        "height": display.height,
        "note": display.note,
    }
    cache[cache_key] = record
    return record


def _run_comparison_panel(
    st: Any,
    state: State,
    client: Any | None,
    detail: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> None:
    groups = comparable_runs_by_image(runs)
    if not groups:
        render_empty(
            st,
            "暂无可并排对比的结果",
            "同一图像至少需要 2 个已完成或已完成（有警告）的运行。",
        )
        return

    image_names = {
        str(image["image_id"]): str(image.get("filename") or image["image_id"])
        for image in _list_of_mappings(detail.get("images"))
        if image.get("image_id")
    }
    st.markdown("## 同图像多模型 / 多运行对比")
    st.caption("仅比较同一原始图像的终态运行；预览和下载均通过后端签名 REST 制品接口获取。")
    image_id = st.selectbox(
        "对比图像",
        list(groups),
        format_func=lambda value: f"{image_names.get(value, value)} · {value}",
        key="comparison_image_id",
    )
    candidates = groups[image_id]
    by_id = {str(run["run_id"]): run for run in candidates}
    candidate_ids = list(by_id)
    selected_ids = st.multiselect(
        "选择 2–3 个运行",
        candidate_ids,
        default=candidate_ids[:3],
        max_selections=3,
        format_func=lambda run_id: _comparison_run_label(by_id[run_id]),
        key=f"comparison_run_ids_{_widget_key(image_id)}",
    )
    if len(selected_ids) < 2:
        st.info("请至少选择 2 个同图像的已完成运行。")
        return
    try:
        selected_runs = select_comparison_runs(
            runs,
            image_id=image_id,
            run_ids=selected_ids,
        )
    except ValueError as error:
        st.error(_localized_error(error))
        return

    refresh_preview = st.button(
        "刷新所选预览",
        key=f"refresh_comparison_{_widget_key('|'.join(selected_ids))}",
    )
    if refresh_preview:
        raw_cache = state.get("comparison_previews")
        if isinstance(raw_cache, dict):
            for run_id in selected_ids:
                raw_cache.pop(run_id, None)

    previews = [_comparison_preview(st, state, client, run) for run in selected_runs]
    columns = st.columns(len(selected_runs), gap="large")
    for column, run, preview in zip(columns, selected_runs, previews, strict=True):
        with column:
            _render_comparison_run(st, state, client, run, preview)


def _comparison_run_label(run: Mapping[str, Any]) -> str:
    configuration = _mapping_or_none(run.get("configuration")) or {}
    version = configuration.get("model_version")
    version_label = f" @ {version}" if version else ""
    return f"{run.get('model_id', '未知模型')}{version_label} · {run.get('run_id', '—')}"


def _comparison_preview(
    st: Any,
    state: State,
    client: Any | None,
    run: Mapping[str, Any],
) -> dict[str, Any] | None:
    run_id = str(run.get("run_id") or "")
    artifact = preferred_preview_artifact(run)
    if not run_id or artifact is None:
        return None
    raw_cache = state.get("comparison_previews")
    cache: dict[str, Any]
    if isinstance(raw_cache, dict):
        cache = raw_cache
    else:
        cache = {}
        state["comparison_previews"] = cache
    cached = _mapping_or_none(cache.get(run_id))
    if cached and cached.get("source") == artifact.download_url:
        return cached
    if client is None:
        return None
    download = _download_action(st, state, client, artifact.download_url)
    if download is None:
        return None
    record = {
        "artifact_key": artifact.key,
        "source": artifact.download_url,
        "content": download.content,
        "filename": download.filename
        or _artifact_fallback_filename(run_id, artifact.key, download.content_type),
        "content_type": download.content_type,
    }
    cache[run_id] = record
    return record


def _render_comparison_run(
    st: Any,
    state: State,
    client: Any | None,
    run: Mapping[str, Any],
    preview: Mapping[str, Any] | None,
) -> None:
    run_id = str(run.get("run_id") or "unknown")
    configuration = _mapping_or_none(run.get("configuration")) or {}
    st.markdown(f"### {run.get('model_id') or '未知模型'}")
    st.markdown(status_badge(str(run.get("status", "unknown"))), unsafe_allow_html=True)
    st.caption(
        f"run {run_id} · 版本 {configuration.get('model_version') or '—'} · "
        f"ROI {display_enum(run.get('roi_mode', '—'))}"
    )

    if preview and str(preview.get("content_type") or "").startswith("image/"):
        st.image(
            preview["content"],
            caption=ARTIFACT_LABELS.get(str(preview.get("artifact_key")), "结果预览"),
            use_container_width=True,
        )
        st.download_button(
            "下载当前预览",
            data=preview["content"],
            file_name=str(preview["filename"]),
            mime=str(preview["content_type"]),
            key=f"comparison_preview_download_{_widget_key(run_id)}",
            use_container_width=True,
        )
    elif preferred_preview_artifact(run) is None:
        st.info("该运行未发布叠加图、颗粒标注图或掩膜预览。")
    elif client is None:
        st.info("连接后端后可加载受管预览。")
    else:
        st.warning("预览制品暂时无法加载，可在下方重试或下载其他制品。")

    quality = _mapping_or_none(run.get("quality"))
    st.markdown("#### 质量判断")
    if quality is None:
        st.info("后端未返回质量门禁报告；下方数值不能替代质量判断。")
    else:
        quality_status = str(quality.get("status", "unknown"))
        st.markdown(
            status_badge(quality_status),
            unsafe_allow_html=True,
        )
        st.caption(f"耗时：{run.get('runtime_ms') or '—'} ms")
        if quality_status.upper() == "REVIEW_REQUIRED":
            st.warning(
                "该运行结果需要人工复核。请切换到单运行详情页，在“复核子运行”页签"
                "提交修正参数或校正掩膜。"
            )
        reasons = quality.get("reasons")
        if isinstance(reasons, list) and reasons:
            for reason in reasons[:3]:
                st.warning(str(reason))
        else:
            st.caption("后端未报告额外的质量风险原因。")
        recommendations = quality.get("recommendations")
        if isinstance(recommendations, list) and recommendations:
            with st.expander("质量改进建议", expanded=True):
                for recommendation in recommendations:
                    st.write(f"• {recommendation}")
        metrics = quality.get("metrics")
        if isinstance(metrics, Mapping) and metrics:
            with st.expander("质量指标详情"):
                st.json(dict(metrics))

    summary = _mapping_or_none(run.get("summary"))
    st.markdown("#### 核心统计")
    if summary is None:
        st.info("后端未返回确定性汇总指标。")
    else:
        st.metric("颗粒数", summary.get("particle_count") or 0)
        diameter_nm = summary.get("mean_equivalent_diameter_nm")
        diameter_px = summary.get("mean_equivalent_diameter_px")
        st.metric(
            "平均等效直径",
            f"{_format_metric_number(diameter_nm)} nm"
            if diameter_nm is not None
            else f"{_format_metric_number(diameter_px)} px",
        )
        density_um = summary.get("number_density_um2")
        density_px = summary.get("number_density_px2")
        st.metric(
            "数量密度",
            f"{_format_metric_number(density_um)} µm⁻²"
            if density_um is not None
            else f"{_format_metric_number(density_px)} px⁻²",
        )
        coverage = summary.get("coverage_ratio")
        st.metric(
            "覆盖率",
            "—" if coverage is None else f"{float(coverage) * 100:.2f}%",
        )

    _comparison_artifact_downloads(st, state, client, run)


def _comparison_artifact_downloads(
    st: Any,
    state: State,
    client: Any | None,
    run: Mapping[str, Any],
) -> None:
    run_id = str(run.get("run_id") or "unknown")
    artifacts = _mapping_or_none(run.get("artifacts")) or {}
    choices = {key: str(url) for key, url in artifacts.items() if key in ARTIFACT_LABELS and url}
    st.markdown("#### 对应制品")
    if not choices:
        st.info("该运行暂无可下载制品。")
        return
    selected_key = st.selectbox(
        "选择制品",
        list(choices),
        format_func=lambda key: ARTIFACT_LABELS[key],
        key=f"comparison_artifact_choice_{_widget_key(run_id)}",
    )
    source = choices[selected_key]
    if st.button(
        "获取所选制品",
        key=f"comparison_artifact_fetch_{_widget_key(run_id)}",
        use_container_width=True,
    ):
        if client is None:
            st.error("请先连接后端，再下载制品。")
        else:
            download = _download_action(st, state, client, source)
            if download is not None:
                raw_downloads = state.get("comparison_downloads")
                downloads: dict[str, Any]
                if isinstance(raw_downloads, dict):
                    downloads = raw_downloads
                else:
                    downloads = {}
                    state["comparison_downloads"] = downloads
                downloads[run_id] = {
                    "source": source,
                    "content": download.content,
                    "filename": download.filename
                    or _artifact_fallback_filename(run_id, selected_key, download.content_type),
                    "content_type": download.content_type,
                }

    raw_downloads = state.get("comparison_downloads")
    prepared = (
        _mapping_or_none(raw_downloads.get(run_id)) if isinstance(raw_downloads, Mapping) else None
    )
    if prepared and prepared.get("source") == source:
        st.download_button(
            f"下载{ARTIFACT_LABELS[selected_key]}",
            data=prepared["content"],
            file_name=str(prepared["filename"]),
            mime=str(prepared["content_type"]),
            key=f"comparison_artifact_download_{_widget_key(run_id + selected_key)}",
            type="primary",
            use_container_width=True,
        )


def _artifact_fallback_filename(run_id: str, artifact_key: str, content_type: str) -> str:
    suffixes = {
        "image/png": ".png",
        "image/tiff": ".tif",
        "text/csv": ".csv",
        "application/json": ".json",
    }
    suffix = suffixes.get(content_type.split(";", maxsplit=1)[0].strip(), "")
    stem = artifact_key.removesuffix("_url")
    return f"{run_id}-{stem}{suffix}"


def _format_metric_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4g}"
    return str(value)


def _artifact_download_panel(
    st: Any,
    state: State,
    client: Any | None,
    run: Mapping[str, Any],
) -> None:
    artifacts = _mapping_or_none(run.get("artifacts")) or {}
    choices = {
        ARTIFACT_LABELS.get(key, key.replace("_url", "")): str(url)
        for key, url in artifacts.items()
        if key.endswith("_url") and url
    }
    if not choices:
        return
    selected = st.selectbox("准备本地下载", list(choices), key="artifact_choice")
    if st.button("获取所选制品"):
        if client is None:
            st.error("请先连接后端，再下载制品。")
            return
        download = _download_action(st, state, client, choices[selected])
        if download is not None:
            state["prepared_download"] = {
                "content": download.content,
                "filename": download.filename or selected.replace(" ", "_").lower(),
                "content_type": download.content_type,
                "source": choices[selected],
            }
    prepared = _mapping_or_none(state.get("prepared_download"))
    if prepared and prepared.get("source") == choices[selected]:
        st.download_button(
            "下载到本机",
            data=prepared["content"],
            file_name=str(prepared["filename"]),
            mime=str(prepared["content_type"]),
            type="primary",
        )


def _review_panel(
    st: Any,
    state: State,
    client: Any | None,
    run: Mapping[str, Any],
) -> None:
    st.markdown("### 创建不可变复核子运行")
    st.caption("复核不会修改当前运行。校正掩膜会先安全暂存，其 token 将记录到子运行配置中。")
    threshold_enabled = st.checkbox("修改阈值")
    run_threshold = run.get("threshold")
    threshold = st.number_input(
        "复核阈值",
        min_value=0.0,
        max_value=1.0,
        value=(
            float(run_threshold)
            if isinstance(run_threshold, (int, float)) and not isinstance(run_threshold, bool)
            else None
        ),
        placeholder="仅在修改阈值时填写",
    )
    area_enabled = st.checkbox("修改最小面积")
    inference = _mapping_or_none(run.get("inference")) or {}
    min_area = int(
        st.number_input(
            "复核最小面积（px）",
            min_value=0,
            value=int(inference.get("min_area_px") or 0),
            step=1,
        )
    )
    watershed_enabled = st.checkbox("修改分水岭设置")
    watershed = st.checkbox(
        "复核运行启用分水岭", value=bool(inference.get("watershed_enabled", False))
    )
    border_enabled = st.checkbox("修改触边排除设置")
    exclude_border = st.checkbox(
        "复核运行排除触边颗粒", value=bool(inference.get("exclude_border", True))
    )
    corrected = st.file_uploader(
        "校正掩膜（选填）",
        type=["png", "tif", "tiff"],
        accept_multiple_files=False,
        key=f"corrected_mask_{run.get('run_id')}",
    )
    if st.button("创建复核运行", type="primary"):
        if client is None:
            st.error("请先连接后端，再创建复核运行。")
            return
        payload: dict[str, Any] = {}
        if threshold_enabled:
            if threshold is None:
                st.error("请输入复核阈值。")
                return
            payload["threshold"] = float(threshold)
        if area_enabled:
            payload["min_area_px"] = min_area
        if watershed_enabled:
            payload["watershed_enabled"] = watershed
        if border_enabled:
            payload["exclude_border"] = exclude_border
        if corrected is not None:
            staged = _api_action(
                st,
                state,
                "暂存校正掩膜",
                lambda: client.upload_corrected_mask(str(run["run_id"]), _upload_part(corrected)),
            )
            if staged is None:
                return
            payload["corrected_mask_token"] = staged.get("corrected_mask_token")
        if not payload:
            st.error("请至少修改一个参数，或上传校正掩膜。")
            return
        reviewed = _api_action(
            st,
            state,
            "创建复核运行",
            lambda: client.review_run(str(run["run_id"]), payload),
        )
        if reviewed is not None:
            child_id = str(reviewed.get("run_id"))
            state["run_ids"] = list(dict.fromkeys([*state["run_ids"], child_id]))
            st.success(f"复核子运行 {child_id} 已进入队列。")


def _export_panel(
    st: Any,
    state: State,
    client: Any | None,
    runs: Sequence[Mapping[str, Any]],
) -> None:
    terminal_ids = exportable_run_ids(runs)
    if not terminal_ids:
        render_empty(
            st,
            "暂无可导出运行",
            "仅“已完成”或“已完成（有警告）”的运行可以导出。",
        )
        return
    selected = st.multiselect(
        "纳入导出的终态运行", terminal_ids, default=terminal_ids, key="export_run_ids"
    )
    if st.button("生成可复现 ZIP", type="primary"):
        if client is None:
            st.error("请先连接后端，再执行导出。")
            return
        if not selected:
            st.error("请至少选择一个可导出的终态运行。")
            return
        exported = _api_action(
            st,
            state,
            "生成导出包",
            lambda: client.export_analysis(str(state["active_job_id"]), run_ids=selected),
        )
        if exported is not None:
            state["last_export"] = exported
    exported = _mapping_or_none(state.get("last_export"))
    if not exported:
        return
    st.success(f"导出包已就绪 · SHA-256 {exported.get('sha256')}")
    url = str(exported.get("download_url") or "")
    if url and st.button("获取导出 ZIP") and client is not None:
        download = _download_action(st, state, client, url)
        if download is not None:
            state["prepared_export"] = {
                "content": download.content,
                "filename": download.filename or exported.get("filename") or "nanoloop.zip",
                "content_type": download.content_type,
            }
    prepared = _mapping_or_none(state.get("prepared_export"))
    if prepared:
        st.download_button(
            "下载导出 ZIP",
            data=prepared["content"],
            file_name=str(prepared["filename"]),
            mime=str(prepared["content_type"]),
            type="primary",
        )


def _query_page(st: Any, state: State, client: Any | None) -> None:
    section_header(
        st,
        eyebrow="04 · 证据工作区",
        title="向 NanoLoop 提问",
        description=(
            "将问题路由到确定性分析工具、材料知识库或二者。"
            "每个数值结果保留来源 run，材料陈述必须附带引用。"
        ),
    )
    detail = _require_job(st, state)
    if detail is None:
        return
    images = _list_of_mappings(detail.get("images"))
    runs = _ordered_runs(state, state.get("run_ids", []))
    image_options = [str(image["image_id"]) for image in images if image.get("image_id")]
    run_options = [str(run["run_id"]) for run in runs if run.get("run_id")]
    with st.form("query_workbench"):
        question = st.text_area(
            "问题",
            placeholder="可询问测量结果、组间比较、材料背景或混合问题……",
            height=110,
        )
        controls = st.columns(3)
        query_type = controls[0].selectbox(
            "路由方式",
            ("auto", "analysis_data", "material_knowledge", "mixed"),
            format_func=display_enum,
        )
        image_id = controls[1].selectbox(
            "图像上下文",
            [None, *image_options],
            format_func=lambda value: "不指定图像" if value is None else value,
        )
        run_ids = controls[2].multiselect("run 上下文", run_options, max_selections=50)
        st.markdown("#### 材料上下文（选填）")
        material = st.columns(3)
        formula = material[0].text_input("化学式")
        name = material[1].text_input("材料名称")
        aliases = material[2].text_input("材料别名", help="可用逗号、分号或换行分隔")
        ask = st.form_submit_button("基于可追溯证据提问", type="primary")
    if ask:
        if client is None:
            st.error("请先连接后端，再发起查询。")
            return
        if not question.strip():
            st.error("请输入问题。")
            return
        material_context = None
        alias_values = parse_aliases(aliases)
        if formula.strip() or name.strip() or alias_values:
            material_context = {
                "formula": formula.strip() or None,
                "name": name.strip() or None,
                "aliases": alias_values,
                "source": "user_confirmation",
            }
        payload = {
            "question": question.strip(),
            "query_type": query_type,
            "image_id": image_id,
            "run_ids": run_ids,
            "material_context": material_context,
        }
        response = _api_action(
            st,
            state,
            "回答问题",
            lambda: client.query_analysis(str(state["active_job_id"]), payload),
        )
        if response is not None:
            state["query_history"] = append_history(
                state.get("query_history", []),
                {
                    "question": question.strip(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "response": response,
                },
            )
    history = state.get("query_history", [])
    if not history:
        render_empty(
            st,
            "尚未发起查询",
            "回答将在此展示，并保留局限性、文献引用、工具证据和调用日志。",
        )
        return
    latest = history[-1]
    st.caption(f"最近一次提问 · {latest.get('created_at', '')}")
    st.markdown(f"### {latest.get('question', '')}")
    response = _mapping_or_none(latest.get("response"))
    if response:
        render_query_response(st, response)
    if len(history) > 1:
        with st.expander(f"历史问题（{len(history) - 1}）"):
            for item in reversed(history[:-1]):
                st.write(f"• {item.get('question')}")


def _knowledge_page(st: Any, state: State, client: Any | None) -> None:
    section_header(
        st,
        eyebrow="05 · 策展证据",
        title="知识库管理",
        description=(
            "摄取具有明确授权和引用元数据的 PDF、TXT 或 Markdown 文档；"
            "查看索引状态并执行重建，不以虚假向量服务替代缺失能力。"
        ),
    )
    health = _mapping_or_none(state.get("health")) or {}
    rag = _mapping_or_none(health.get("rag_index"))
    if rag:
        st.markdown(
            f"知识检索 {status_badge(str(rag.get('status', 'unknown')))}",
            unsafe_allow_html=True,
        )
        if rag.get("detail"):
            st.caption(_localized_text(str(rag["detail"])))
        if rag.get("status") != "healthy":
            st.warning("知识检索当前降级或不可用；在迁移、提取器和索引就绪前，文档摄取仍可能失败。")
    else:
        st.info("请先检查连接状态，以确认知识检索是否就绪。")

    ingest_tab, catalogue_tab, rebuild_tab = st.tabs(("摄取文档", "文档目录", "重建索引"))
    with ingest_tab:
        document = st.file_uploader(
            "知识文档", type=["pdf", "txt", "md", "markdown"], key="knowledge_file"
        )
        with st.form("knowledge_ingest"):
            title = st.text_input("标题 *")
            columns = st.columns(2)
            source_type = columns[0].selectbox(
                "来源类型",
                ("paper", "report", "material_note", "other"),
                format_func=display_enum,
            )
            year = columns[1].number_input(
                "年份", min_value=1000, max_value=3000, value=None, step=1
            )
            citation = st.text_area("引用文本 *", height=80)
            aliases = st.text_input("材料别名")
            license_note = st.text_area("授权 / 使用说明 *", height=80)
            allowed = st.checkbox("允许用于本次演示")
            ingest = st.form_submit_button("摄取并建立索引", type="primary")
        if ingest:
            if client is None:
                st.error("请先连接后端，再摄取文档。")
            elif document is None:
                st.error("请选择一个文档。")
            elif not title.strip() or not citation.strip() or not license_note.strip():
                st.error("标题、引用文本和授权说明均为必填项。")
            else:
                metadata = {
                    "title": title.strip(),
                    "source_type": source_type,
                    "year": int(year) if year is not None else None,
                    "citation_text": citation.strip(),
                    "material_aliases": parse_aliases(aliases),
                    "license_note": license_note.strip(),
                    "allowed_for_demo": allowed,
                }
                report = _api_action(
                    st,
                    state,
                    "摄取知识文档",
                    lambda: client.ingest_knowledge_document(_upload_part(document), metadata),
                )
                if report is not None:
                    state["last_ingest_report"] = report
                    st.success(
                        f"已索引 {report.get('chunks_created', 0)} 个文本块 · "
                        f"文档 {report.get('doc_id')}"
                    )
        report = _mapping_or_none(state.get("last_ingest_report"))
        if report:
            metrics = st.columns(4)
            metrics[0].metric("总页数", report.get("pages_total", 0))
            metrics[1].metric("已提取页数", report.get("pages_extracted", 0))
            metrics[2].metric("已创建文本块", report.get("chunks_created", 0))
            metrics[3].metric("已跳过文本块", report.get("chunks_skipped", 0))
            st.caption(
                f"doc_id {report.get('doc_id')} · index_version {report.get('index_version')} · "
                f"SHA-256 {report.get('sha256')}"
            )
            for warning in report.get("warnings") or []:
                st.warning(_localized_text(str(warning)))

    with catalogue_tab:
        status_notice = _mapping_or_none(state.pop("knowledge_status_notice", None))
        if status_notice and status_notice.get("message"):
            st.success(str(status_notice["message"]))
        if st.button("刷新文档目录") and client is not None:
            listed = _api_action(
                st,
                state,
                "列出知识文档",
                client.list_knowledge_documents,
            )
            if listed is not None:
                state["knowledge_documents"] = listed.get("documents") or []
        documents = _list_of_mappings(state.get("knowledge_documents"))
        if documents:
            st.dataframe(
                [
                    {
                        "doc_id": item.get("doc_id"),
                        "标题": item.get("title"),
                        "来源类型": display_enum(item.get("source_type")),
                        "年份": item.get("year"),
                        "状态": display_enum(item.get("status")),
                        "允许演示": "是" if item.get("allowed_for_demo") else "否",
                        "材料别名": ", ".join(map(str, item.get("material_aliases") or [])),
                        "sha256": item.get("sha256"),
                        "授权说明": item.get("license_note"),
                    }
                    for item in documents
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("#### 文档启用状态")
            st.caption("禁用文档会立即从后续知识检索中排除；重新启用后恢复检索资格。")
            for item in documents:
                transition = knowledge_document_toggle(item.get("status"))
                if transition is None:
                    continue
                doc_id = str(item.get("doc_id") or "")
                title = str(item.get("title") or doc_id or "未命名文档")
                columns = st.columns((4, 1, 1))
                columns[0].write(f"**{title}**")
                columns[0].caption(doc_id)
                columns[1].markdown(
                    status_badge(str(item.get("status", "unknown"))),
                    unsafe_allow_html=True,
                )
                clicked = columns[2].button(
                    transition.button_label,
                    key=f"knowledge_toggle_{doc_id}_{transition.enabled}",
                    use_container_width=True,
                    disabled=client is None or not doc_id,
                )
                if clicked and client is not None:
                    updated = _api_action(
                        st,
                        state,
                        transition.button_label,
                        partial(
                            client.update_knowledge_document,
                            doc_id,
                            enabled=transition.enabled,
                        ),
                    )
                    if updated is not None:
                        listed = _api_action(
                            st,
                            state,
                            "刷新文档目录",
                            client.list_knowledge_documents,
                        )
                        if listed is not None:
                            state["knowledge_documents"] = listed.get("documents") or []
                            state["knowledge_status_notice"] = {
                                "message": (
                                    f"{transition.completed_label}《{title}》，文档目录已刷新。"
                                )
                            }
                            st.rerun()
                        else:
                            st.warning(
                                f"{transition.completed_label}《{title}》，但目录刷新失败；"
                                "请手动刷新以核对最新状态。"
                            )
        else:
            render_empty(
                st,
                "尚未加载文档目录",
                "请刷新目录，以区分知识库为空和连接失败。",
            )

    with rebuild_tab:
        st.warning(
            "重建索引会重新提取已存文档。单个文档失败会明确保留，且不会计入成功数；"
            "强制模式将接受源文件哈希变化。"
        )
        force = st.checkbox("强制接受已变化的源文件内容")
        if st.button("重建知识索引", type="primary"):
            if client is None:
                st.error("请先连接后端，再重建索引。")
            else:
                report = _api_action(
                    st,
                    state,
                    "重建知识索引",
                    lambda: client.reindex_knowledge(force=force),
                )
                if report is not None:
                    state["last_reindex_report"] = report
        report = _mapping_or_none(state.get("last_reindex_report"))
        if report:
            metrics = st.columns(3)
            metrics[0].metric("已索引文档", report.get("documents_indexed", 0))
            metrics[1].metric("已索引文本块", report.get("chunks_indexed", 0))
            metrics[2].metric("已跳过文本块", report.get("chunks_skipped", 0))
            for warning in report.get("warnings") or []:
                st.warning(_localized_text(str(warning)))


def _get_client(st: Any, state: State) -> Any | None:
    api_key = os.getenv("NANOLOOP_API_KEY") or None
    try:
        requested_base_url = normalize_base_url(str(state["api_base_url"]))
        if api_key is not None:
            trusted_base_url = normalize_base_url(
                os.getenv("NANOLOOP_API_BASE_URL", DEFAULT_API_BASE_URL)
            )
            if requested_base_url != trusted_base_url:
                raise ValueError(
                    "配置共享 API Key 时，后端服务地址必须使用运维锁定值"
                )
    except (KeyError, TypeError, ValueError) as error:
        _drop_client(state)
        render_exception(st, error, action="创建 REST 客户端")
        return None

    api_key_fingerprint = (
        hashlib.sha256(api_key.encode("utf-8")).hexdigest() if api_key is not None else None
    )
    key = (
        requested_base_url,
        float(state["api_timeout_seconds"]),
        api_key_fingerprint,
    )
    if state.get("_api_client_key") == key and state.get("_api_client") is not None:
        return state["_api_client"]
    _drop_client(state)
    try:
        module = importlib.import_module("frontend.api_client")
        client_class = module.NanoLoopApiClient
        parameters = inspect.signature(client_class).parameters
        timeout = float(state["api_timeout_seconds"])
        client_arguments: dict[str, Any] = {}
        if "api_key" in parameters:
            client_arguments["api_key"] = api_key
        if "timeout_seconds" in parameters:
            client_arguments["timeout_seconds"] = timeout
        else:
            client_arguments["timeout"] = timeout
            client_arguments["upload_timeout"] = max(timeout, 120.0)
        client = client_class(requested_base_url, **client_arguments)
    except Exception as error:
        render_exception(st, error, action="创建 REST 客户端")
        return None
    state["_api_client"] = client
    state["_api_client_key"] = key
    return client


def _drop_client(state: State) -> None:
    client = state.get("_api_client")
    if client is not None and hasattr(client, "close"):
        client.close()
    state["_api_client"] = None
    state["_api_client_key"] = None


def _refresh_health(st: Any, state: State, client: Any) -> None:
    result = _api_action(st, state, "检查系统状态", client.health)
    if result is not None:
        state["health"] = result
        state["health_checked_at"] = datetime.now(UTC).isoformat()
        st.rerun()


def _load_job(st: Any, state: State, client: Any, job_id: str) -> None:
    result = _api_action(st, state, "加载项目", lambda: client.get_analysis(job_id))
    if result is not None:
        _store_job_detail(state, result)
        st.success(f"项目 {job_id} 已加载。")


def _store_job_detail(state: State, detail: dict[str, Any]) -> None:
    job = _mapping_or_none(detail.get("job")) or {}
    previous_job_id = str(state.get("active_job_id") or "")
    next_job_id = str(job.get("job_id") or "")
    state["active_job_id"] = next_job_id
    state["job_detail"] = detail
    runs = _list_of_mappings(detail.get("runs"))
    state["run_ids"] = [str(run["run_id"]) for run in runs if run.get("run_id")]
    state["runs"] = {str(run["run_id"]): dict(run) for run in runs if run.get("run_id")}
    if previous_job_id != next_job_id:
        state["box_sets"] = {}
        state["image_preview"] = None
        state["roi_drafts"] = {}
        state["roi_canvas_events"] = {}
        state["comparison_previews"] = {}
        state["comparison_downloads"] = {}
        state["result_layer_previews"] = {}


def _roi_draft(
    state: State,
    image_id: str,
    box_set: Mapping[str, Any],
) -> dict[str, Any]:
    drafts = state.get("roi_drafts")
    if not isinstance(drafts, dict):
        drafts = {}
        state["roi_drafts"] = drafts
    revision = int(box_set.get("revision", 0))
    existing = drafts.get(image_id)
    if isinstance(existing, dict) and existing.get("revision") == revision:
        return existing
    return _reset_roi_draft(state, image_id, box_set)


def _reset_roi_draft(
    state: State,
    image_id: str,
    box_set: Mapping[str, Any],
) -> dict[str, Any]:
    drafts = state.get("roi_drafts")
    if not isinstance(drafts, dict):
        drafts = {}
        state["roi_drafts"] = drafts
    record: dict[str, Any] = {
        "revision": int(box_set.get("revision", 0)),
        "rows": _list_of_mappings(box_set.get("boxes")),
        "editor_generation": 0,
    }
    drafts[image_id] = record
    return record


def _drawable_roi_rows(value: object) -> list[dict[str, Any]]:
    """Filter incomplete editor rows before handing them to the browser canvas."""

    rows = value if isinstance(value, list) else []
    drawable: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        values = [raw_row.get(name) for name in ("x1", "y1", "x2", "y2")]
        numbers: list[float] = []
        for item in values:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                break
            numbers.append(float(item))
        if len(numbers) != 4:
            continue
        x1, y1, x2, y2 = numbers
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue
        row = {
            "label": str(raw_row.get("label") or "").strip(),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "active": bool(raw_row.get("active", True)),
        }
        box_id = raw_row.get("box_id")
        if isinstance(box_id, str) and box_id.strip():
            row["box_id"] = box_id.strip()
        drawable.append(row)
    return drawable


def _roi_canvas_preview(
    st: Any,
    state: State,
    client: Any | None,
    image: Mapping[str, Any],
) -> RoiCanvasPreview | None:
    image_url = str(image.get("original_download_url") or "")
    if not image_url or client is None:
        return None
    raw_record = state.get("image_preview")
    record = raw_record if isinstance(raw_record, dict) else None
    if record is None or record.get("source") != image_url:
        downloaded = _download_action(st, state, client, image_url)
        if downloaded is None:
            return None
        record = {"source": image_url, "content": downloaded.content}
        state["image_preview"] = record
    cached = record.get("roi_canvas")
    if isinstance(cached, RoiCanvasPreview):
        return cached
    content = record.get("content")
    if not isinstance(content, bytes):
        return None
    try:
        preview = prepare_roi_preview(
            content,
            original_width=int(image["width"]),
            original_height=int(image["height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"准备 ROI 画布失败：{error}")
        return None
    record["roi_canvas"] = preview
    return preview


def _poll_runs(
    st: Any,
    state: State,
    client: Any,
    run_ids: Sequence[str],
    *,
    quiet: bool = False,
) -> None:
    for run_id in run_ids:
        try:
            result = as_dict(client.get_run(run_id))
            state["runs"][run_id] = result
        except Exception as error:
            if quiet:
                continue
            code = str(getattr(error, "code", type(error).__name__))
            status_code = getattr(error, "status_code", None)
            retry_after = getattr(error, "retry_after_seconds", None)
            if status_code == 429 or code == "RATE_LIMITED":
                render_rate_limit(
                    st,
                    retry_after_seconds=retry_after,
                    is_read_request=True,
                    request_id=getattr(error, "request_id", None),
                )
            elif status_code == 401 or code == "AUTHENTICATION_REQUIRED":
                render_auth_guidance(st)
            else:
                render_exception(st, error, action=f"刷新运行 {run_id}")


def _has_pending_runs(state: Mapping[str, Any], run_ids: Sequence[str]) -> bool:
    runs = state.get("runs", {})
    for run_id in run_ids:
        run = runs.get(run_id)
        if not isinstance(run, Mapping):
            return True
        if run_id in pollable_run_ids({run_id: run}):
            return True
    return False


def _ordered_runs(state: Mapping[str, Any], run_ids: Sequence[str]) -> list[dict[str, Any]]:
    runs = state.get("runs", {})
    return [dict(runs[run_id]) for run_id in run_ids if isinstance(runs.get(run_id), Mapping)]


_READ_ACTION_KEYWORDS = (
    "加载",
    "刷新",
    "检查",
    "查看",
    "获取",
    "列出",
    "推荐",
    "预览",
    "下载",
)


def _is_read_action(action: str) -> bool:
    """Heuristic: an action is "safe to retry" iff it does not mutate state.

    Intentionally conservative — anything not matching a known read verb is
    treated as a mutation, so we never silently replay a write.
    """

    return any(keyword in action for keyword in _READ_ACTION_KEYWORDS)


def _api_action(
    st: Any,
    state: State,
    action: str,
    operation: Callable[[], Any],
) -> dict[str, Any] | None:
    try:
        with st.spinner(f"{action}…"):
            result = as_dict(operation())
        state["last_error"] = None
        return result
    except Exception as error:
        code = str(getattr(error, "code", type(error).__name__))
        message = str(getattr(error, "message", str(error)))
        request_id = getattr(error, "request_id", None)
        status_code = getattr(error, "status_code", None)
        retry_after = getattr(error, "retry_after_seconds", None)
        state["last_error"] = {
            "action": action,
            "code": code,
            "message": message,
            "request_id": request_id,
        }
        is_service_unavailable = (
            code.startswith("HTTP_5")
            or (status_code is not None and 500 <= status_code < 600)
            or code in {"SERVICE_UNAVAILABLE", "BAD_GATEWAY", "GATEWAY_TIMEOUT"}
        )
        if status_code == 401 or code == "AUTHENTICATION_REQUIRED":
            render_auth_guidance(st)
        elif status_code == 429 or code == "RATE_LIMITED":
            render_rate_limit(
                st,
                retry_after_seconds=retry_after,
                is_read_request=_is_read_action(action),
                request_id=request_id,
            )
            if _is_read_action(action):
                if st.button("重试", key=f"retry_{action}_{code}", use_container_width=True):
                    st.rerun()
        elif is_service_unavailable:
            base_url = None
            try:
                base_url = str(state.get("api_base_url") or "") or None
            except Exception:
                base_url = None
            render_service_unavailable(st, code=code, base_url=base_url)
            if st.button("重试连接", key=f"retry_{action}_{code}", use_container_width=True):
                st.rerun()
        elif getattr(error, "retryable", False):
            render_actionable_error(
                st,
                title=f"{action}失败（可重试）",
                code=code,
                message=message,
                tone="warn",
                request_id=request_id,
                hint=(
                    "后端将此错误标记为可重试。为安全起见，界面不会自动重放"
                    "——请确认后端状态后，再手动重新发起该操作。"
                ),
            )
            if _is_read_action(action):
                if st.button("重试", key=f"retry_{action}_{code}", use_container_width=True):
                    st.rerun()
        else:
            render_exception(st, error, action=action)
        return None


def _download_action(
    st: Any,
    state: State,
    client: Any,
    download_url: str,
) -> Any | None:
    try:
        with st.spinner("正在获取受管制品……"):
            result = client.download_artifact(download_url)
        state["last_error"] = None
        return result
    except Exception as error:
        render_exception(st, error, action="下载制品")
        return None


def _upload_part(upload: Any) -> Any:
    module = importlib.import_module("frontend.api_client")
    return module.UploadPart(
        filename=str(upload.name),
        content=upload.getvalue(),
        content_type=getattr(upload, "type", None),
    )


def _require_job(st: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
    detail = _mapping_or_none(state.get("job_detail"))
    if detail is None:
        render_empty(
            st,
            "没有当前项目",
            "请先创建项目，或从侧边栏按 job_id 加载项目。",
        )
    return detail


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _widget_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _localized_error(error: Exception | str) -> str:
    return _localized_text(str(error))


def _localized_text(value: str) -> str:
    exact = {
        "API base URL must be an absolute http:// or https:// URL": (
            "API 地址必须是完整的 http:// 或 https:// URL。"
        ),
        "API base URL cannot contain a query string or fragment": (
            "API 地址不能包含查询参数或 fragment。"
        ),
        "Project name is required": "项目名称为必填项。",
        "Select between 1 and 20 images": "请选择 1–20 张图像。",
        "Image filenames must be unique within a project": "项目内的图像文件名不能重复。",
        "Select at least one and at most 20 images": "请选择 1–20 张图像。",
        "Select between one and three models": "请选择 1–3 个模型。",
        "ROI mode must be full_image or boxes": "ROI 模式必须为整幅图或 ROI 框。",
        "Threshold must be between 0 and 1": "阈值必须位于 0 到 1 之间。",
        "Minimum area cannot be negative": "最小面积不能为负数。",
        "Select two or three runs for comparison": "请选择 2–3 个运行进行对比。",
        "Comparison run IDs must be unique and non-empty": ("对比运行 ID 必须唯一且不能为空。"),
        "Select an image for comparison": "请选择要对比的图像。",
        "Image dimensions are unavailable.": "图像尺寸不可用。",
        "Analysis valid_rect is invalid.": "analysis_roi.valid_rect 无效。",
        "Minimum ROI size must be positive.": "ROI 最小尺寸必须为正数。",
        "All reported components are healthy.": "所有组件均报告为正常。",
        "reachable; migrations not applied": "数据库可连接，但尚未应用迁移",
        "registry present; gateway unavailable": "模型注册表存在，但推理网关不可用",
        "registry and gateway unavailable": "模型注册表与推理网关均不可用",
        "registry contains no models": "模型注册表中没有模型",
        "no ready models": "当前没有就绪模型",
        "keyword data ready; vector index absent": "关键词索引可用，向量索引缺失",
        "knowledge index not built": "知识索引尚未构建",
        "duplicate_sha256: existing document reused": "文档内容重复，已复用现有索引。",
    }
    if value in exact:
        return exact[value]
    translated = value
    for english, chinese in {
        ": sample ID is required": "：必须填写样品 ID",
        ": unsupported scale mode": "：尺度模式不受支持",
        ": nm/pixel scale is required": "：必须填写 nm/pixel 尺度",
        ": nm/pixel scale must be positive": "：nm/pixel 尺度必须大于 0",
        ": experiment conditions must be an object": "：实验条件必须是 JSON 对象",
        " must be valid JSON": " 必须是有效 JSON",
        " must be a JSON object": " 必须是 JSON 对象",
        " must be an integer.": " 必须为整数。",
        "width and height must each be at least ": "宽和高均须至少为 ",
        " pixels.": " 像素。",
        "require 0 ≤ x1 < x2 and 0 ≤ y1 < y2.": ("必须满足 0 ≤ x1 < x2 且 0 ≤ y1 < y2。"),
        " exceeds ": " 超出图像尺寸 ",
        " falls outside analysis valid_rect ": " 超出分析有效区域 valid_rect ",
        " intersects invalid region ": " 与无效分析区域相交 ",
        " has no active saved ROI boxes": " 没有已保存且启用的 ROI 框",
        " has no valid saved box revision": " 没有有效的已保存 box revision",
        "Save ROI boxes for ": "请先为 ",
        " before submitting": " 保存 ROI 框再提交",
        "Invalid analysis region ": "无效分析区域 ",
        " has malformed coordinates.": " 的坐标格式错误。",
        "At most ": "最多允许 ",
        " ROI rows are allowed.": " 行 ROI。",
        "Comparison runs must be completed runs from the selected image: ": (
            "对比项必须是所选图像的已完成运行："
        ),
    }.items():
        translated = translated.replace(english, chinese)
    return translated


if __name__ == "__main__":
    main()
