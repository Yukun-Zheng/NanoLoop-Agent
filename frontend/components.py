"""Reusable Streamlit renderers for traceable scientific state."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from frontend.state import health_rollup, status_tone

ENUM_LABELS = {
    "unknown": "未知",
    "healthy": "正常",
    "degraded": "降级运行",
    "unavailable": "不可用",
    "ready": "就绪",
    "loading": "加载中",
    "disabled": "已禁用",
    "indexing": "索引中",
    "created": "已创建",
    "validating": "校验中",
    "ready_for_configuration": "等待配置",
    "queued": "排队中",
    "preprocessing": "预处理中",
    "segmenting": "分割中",
    "postprocessing": "后处理中",
    "quality_checking": "质量检查中",
    "analyzing": "分析中",
    "aggregating": "汇总中",
    "completed": "已完成",
    "completed_with_warnings": "已完成（有警告）",
    "failed": "失败",
    "pass": "通过",
    "warn": "警告",
    "review_required": "需要复核",
    "success": "成功",
    "accepted": "已接受",
    "error": "错误",
    "low": "低",
    "medium": "中",
    "high": "高",
    "full_image": "整幅图",
    "boxes": "ROI 框",
    "unet": "U-Net",
    "yolo_seg": "YOLO-Seg",
    "sam2": "SAM 2",
    "general": "通用",
    "small_particle": "小颗粒",
    "large_particle": "大颗粒",
    "dense_particle": "致密颗粒",
    "low_contrast": "低对比度",
    "fast": "快速",
    "balanced": "均衡",
    "accurate": "高精度",
    "accuracy": "精度优先",
    "balance": "均衡优先",
    "speed": "速度优先",
    "auto": "自动",
    "cpu": "CPU",
    "cuda": "CUDA",
    "mps": "Apple MPS",
    "pixel_only": "仅像素",
    "nm_per_pixel": "物理尺度（nm/pixel）",
    "paper": "论文",
    "report": "报告",
    "material_note": "材料说明",
    "other": "其他",
    "analysis_data": "实验数据",
    "material_knowledge": "材料知识",
    "mixed": "混合查询",
    "none": "未裁剪",
    "manual": "人工指定",
    "detected": "自动检测",
    "instrument_bar": "仪器信息栏",
    "instrument_bar_detected": "检测到仪器信息栏",
}

_DETAIL_TRANSLATIONS = {
    "All reported components are healthy.": "所有组件均报告为正常。",
    "Run a connection check before starting a workflow.": "开始工作前请先检查连接。",
    "reachable; migrations not applied": "数据库可连接，但尚未应用迁移",
    "registry present; gateway unavailable": "模型注册表存在，但推理网关不可用",
    "registry and gateway unavailable": "模型注册表与推理网关均不可用",
    "registry contains no models": "模型注册表中没有模型",
    "no ready models": "当前没有就绪模型",
    "keyword data ready; vector index absent": "关键词索引可用，向量索引缺失",
    "knowledge index not built": "知识索引尚未构建",
}


def display_enum(value: object) -> str:
    """Return a Chinese display label without changing the machine value."""

    text_value = str(value)
    return ENUM_LABELS.get(text_value.casefold(), text_value)


def section_header(
    streamlit: Any,
    *,
    eyebrow: str,
    title: str,
    description: str,
) -> None:
    streamlit.markdown(
        (
            '<section class="nl-hero">'
            f'<div class="nl-eyebrow">{html.escape(eyebrow)}</div>'
            f"<h1>{html.escape(title)}</h1>"
            f"<p>{html.escape(description)}</p>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def status_badge(status: str | None, *, label: str | None = None) -> str:
    tone = status_tone(status)
    visible = label or display_enum(status or "unknown")
    return (
        f'<span class="nl-status nl-status-{tone}">'
        f"{html.escape(str(visible).replace('_', ' '))}</span>"
    )


def render_connection_status(streamlit: Any, health: Mapping[str, Any] | None) -> None:
    rollup = health_rollup(health)
    label = {
        "Not checked": "尚未检查",
        "Core unavailable": "核心服务不可用",
        "Connected with limitations": "已连接（能力受限）",
        "Connected": "已连接",
    }.get(rollup.label, rollup.label)
    streamlit.markdown(
        f"{status_badge(rollup.status, label=label)} "
        f'<span class="nl-muted">{html.escape(_translate_detail(rollup.detail))}</span>',
        unsafe_allow_html=True,
    )


def render_health_matrix(streamlit: Any, health: Mapping[str, Any]) -> None:
    labels = {
        "service": "API 服务",
        "database": "数据库",
        "model_registry": "模型注册表",
        "rag_index": "知识检索",
    }
    columns = streamlit.columns(4)
    for column, (key, label) in zip(columns, labels.items(), strict=True):
        component = health.get(key)
        record = component if isinstance(component, Mapping) else {}
        status = str(record.get("status", "unavailable"))
        detail = _translate_detail(str(record.get("detail") or "暂无补充说明"))
        with column:
            streamlit.markdown(
                '<div class="nl-card">'
                f'<div class="nl-card-title">{html.escape(label)}</div>'
                f"{status_badge(status)}"
                f'<div class="nl-card-copy">{html.escape(detail)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def render_job_overview(streamlit: Any, detail: Mapping[str, Any]) -> None:
    raw_job = detail.get("job")
    job: Mapping[str, Any] = raw_job if isinstance(raw_job, Mapping) else {}
    raw_images = detail.get("images")
    images: list[Any] = raw_images if isinstance(raw_images, list) else []
    raw_runs = detail.get("runs")
    runs: list[Any] = raw_runs if isinstance(raw_runs, list) else []
    raw_failures = detail.get("partial_failures")
    failures: list[Any] = raw_failures if isinstance(raw_failures, list) else []
    top = streamlit.columns([2.2, 1, 1, 1])
    with top[0]:
        streamlit.markdown(f"### {job.get('name') or '未命名项目'}")
        streamlit.caption(str(job.get("job_id") or "暂无 job_id"))
        streamlit.markdown(status_badge(str(job.get("status", "unknown"))), unsafe_allow_html=True)
    top[1].metric("图像", len(images))
    top[2].metric("运行", len(runs))
    top[3].metric("失败", len(failures))
    error_code = job.get("error_code")
    if error_code:
        streamlit.error(f"项目错误：{error_code}")


def render_run_table(streamlit: Any, runs: Sequence[Mapping[str, Any]]) -> None:
    if not runs:
        render_empty(
            streamlit,
            "暂无运行记录",
            "请先选择图像、ROI 模式和就绪模型，再提交运行。",
        )
        return
    rows = [
        {
            "run_id": run.get("run_id"),
            "image_id": run.get("image_id"),
            "模型": run.get("model_id"),
            "运行状态": display_enum(run.get("status")),
            "质量状态": display_enum(_nested(run, "quality", "status") or "—"),
            "耗时 (ms)": run.get("runtime_ms"),
            "错误码": run.get("error_code"),
        }
        for run in runs
    ]
    streamlit.dataframe(rows, hide_index=True, width="stretch")


def render_run_summary(streamlit: Any, run: Mapping[str, Any]) -> None:
    streamlit.markdown(
        f"### 运行 `{run.get('run_id', 'unknown')}` &nbsp; "
        f"{status_badge(str(run.get('status', 'unknown')))}",
        unsafe_allow_html=True,
    )
    streamlit.caption(
        f"图像 {run.get('image_id', '—')} · 模型 {run.get('model_id', '—')} · "
        f"ROI {display_enum(run.get('roi_mode', '—'))}"
    )
    if run.get("error_code"):
        streamlit.error(
            f"{run.get('error_code')}：{run.get('error_message') or '后端未提供错误详情。'}"
        )
    history = run.get("status_history")
    if isinstance(history, list) and history:
        timeline = [
            {
                "时间": event.get("created_at"),
                "从": display_enum(event.get("from_status") or "—"),
                "到": display_enum(event.get("to_status") or "—"),
                "错误码": event.get("error_code"),
                "错误说明": event.get("error_message"),
            }
            for event in history
            if isinstance(event, Mapping)
        ]
        with streamlit.expander(f"状态时间线（{len(timeline)}）"):
            streamlit.dataframe(timeline, hide_index=True, width="stretch")

    quality = run.get("quality")
    streamlit.markdown("#### 质量判断")
    if isinstance(quality, Mapping):
        quality_status = str(quality.get("status", "unknown"))
        streamlit.markdown(
            f"质量门禁 {status_badge(quality_status)}",
            unsafe_allow_html=True,
        )
        if quality_status.upper() == "REVIEW_REQUIRED":
            streamlit.warning(
                "该运行结果需要人工复核。请在“复核子运行”页签提交修正参数或校正掩膜，"
                "生成不可变复核子运行。复核不会修改当前运行。"
            )
        reasons = quality.get("reasons")
        if isinstance(reasons, list) and reasons:
            for reason in reasons:
                streamlit.warning(str(reason))
        else:
            streamlit.caption("后端未报告额外的质量风险原因。")
        recommendations = quality.get("recommendations")
        if isinstance(recommendations, list) and recommendations:
            with streamlit.expander("质量改进建议", expanded=True):
                for recommendation in recommendations:
                    streamlit.write(f"• {recommendation}")
        metrics = quality.get("metrics")
        if isinstance(metrics, Mapping) and metrics:
            with streamlit.expander("质量指标详情"):
                streamlit.json(dict(metrics))
    else:
        streamlit.info("该运行尚无质量门禁报告；下方数值不能替代质量判断。")

    streamlit.markdown("#### 数值汇总")
    summary = run.get("summary")
    if not isinstance(summary, Mapping):
        streamlit.info("该运行尚无确定性汇总指标。界面不会自行推测或补造数值。")
    else:
        metric_columns = streamlit.columns(4)
        metric_columns[0].metric("颗粒数", _display(summary.get("particle_count")))
        diameter_nm = summary.get("mean_equivalent_diameter_nm")
        if diameter_nm is not None:
            metric_columns[1].metric("平均等效直径", f"{_number(diameter_nm)} nm")
        else:
            metric_columns[1].metric(
                "平均等效直径", f"{_number(summary.get('mean_equivalent_diameter_px'))} px"
            )
        density_um = summary.get("number_density_um2")
        if density_um is not None:
            metric_columns[2].metric("数量密度", f"{_number(density_um)} µm⁻²")
        else:
            metric_columns[2].metric(
                "数量密度", f"{_number(summary.get('number_density_px2'))} px⁻²"
            )
        coverage = summary.get("coverage_ratio")
        metric_columns[3].metric(
            "覆盖率", "—" if coverage is None else f"{float(coverage) * 100:.2f}%"
        )


def render_artifact_links(
    streamlit: Any,
    run: Mapping[str, Any],
) -> None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        render_empty(streamlit, "暂无制品", "后端尚未发布制品下载令牌。")
        return
    labels = {
        "mask_url": "分割掩膜",
        "overlay_url": "叠加预览",
        "probability_url": "概率图",
        "instances_url": "实例数据",
        "labeled_particles_url": "颗粒标注图",
        "particles_csv_url": "颗粒 CSV",
        "quality_report_url": "质量报告",
        "execution_provenance_url": "执行溯源记录",
    }
    available = [(key, label) for key, label in labels.items() if artifacts.get(key)]
    if not available:
        render_empty(
            streamlit,
            "暂无可下载制品",
            "后端发布受管文件令牌后，下载项才会在此出现。",
        )
        return
    streamlit.caption(
        "可用受管制品："
        + ", ".join(label for _key, label in available)
        + "。请在下方选择并通过工作台安全下载。"
    )


def render_query_response(streamlit: Any, response: Mapping[str, Any]) -> None:
    outcome = str(response.get("outcome_code", "OK"))
    confidence = str(response.get("confidence", "low"))
    query_type = str(response.get("query_type", "auto"))
    needs_clarification = bool(response.get("needs_clarification", False))

    # Routing summary — tells the user how the backend classified the question
    route_badge = status_badge(query_type, label=display_enum(query_type))
    streamlit.caption(f"后端路由方式：{route_badge}", unsafe_allow_html=True)

    if outcome == "INSUFFICIENT_EVIDENCE":
        streamlit.warning("现有实验数据或知识证据不足，无法形成有依据的回答。")
    streamlit.markdown(
        f"#### 回答 · 置信度 {status_badge(confidence)}",
        unsafe_allow_html=True,
    )
    streamlit.write(response.get("answer") or "后端未返回回答正文。")

    if needs_clarification:
        streamlit.info(
            "后端标记此回答需要进一步澄清。请补充更具体的图像、运行或材料上下文后重新提问。"
        )

    # Echo the material context the backend received/confirmed
    material_context = response.get("material_context")
    if isinstance(material_context, Mapping):
        parts = []
        formula = material_context.get("formula")
        name = material_context.get("name")
        aliases = material_context.get("aliases")
        source = material_context.get("source")
        if formula:
            parts.append(f"化学式 {formula}")
        if name:
            parts.append(f"名称 {name}")
        if isinstance(aliases, list) and aliases:
            parts.append(f"别名 {', '.join(map(str, aliases))}")
        if source:
            parts.append(f"来源 {display_enum(source)}")
        if parts:
            streamlit.caption("材料上下文：" + " · ".join(parts))

    limitations = response.get("limitations")
    if isinstance(limitations, list) and limitations:
        with streamlit.expander("局限性", expanded=outcome == "INSUFFICIENT_EVIDENCE"):
            for limitation in limitations:
                streamlit.write(f"• {limitation}")

    # Citations are top-level UnifiedQueryResponse records. A null page means
    # that the citation applies to the full document and must remain visible.
    citations = response.get("citations")
    streamlit.markdown("#### 材料知识引用")
    if isinstance(citations, list) and citations:
        for citation in citations:
            if not isinstance(citation, Mapping):
                continue
            page = citation.get("page")
            heading = (
                f"{citation.get('citation_id', 'citation')} · {citation.get('title', '未命名文献')}"
            )
            if page is not None:
                heading += f" · 第 {page} 页"
            else:
                heading += " · 全文引用"
            with streamlit.expander(heading):
                streamlit.write(citation.get("excerpt") or "未提供引用摘录。")
                streamlit.caption(
                    f"doc {citation.get('doc_id', '—')} · chunk {citation.get('chunk_id', '—')} · "
                    f"检索分数 {_number(citation.get('retrieval_score'))}"
                )
                if citation.get("citation_text"):
                    streamlit.write(citation["citation_text"])
    elif outcome == "INSUFFICIENT_EVIDENCE":
        streamlit.info("未返回材料知识引用。")
    else:
        streamlit.info("未返回材料知识引用；当前回答中涉及材料背景的陈述不应视为已有文献来源支持。")

    # --- 实验数据结论 区块 ---
    evidence = response.get("data_evidence")
    streamlit.markdown("#### 实验数据结论")
    if isinstance(evidence, list) and evidence:
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            with streamlit.expander(str(item.get("tool_name", "数据工具"))):
                streamlit.caption(
                    "来源 run：" + ", ".join(map(str, item.get("source_run_ids") or []))
                )
                if item.get("quality_warnings"):
                    for warning in item["quality_warnings"]:
                        streamlit.warning(str(warning))
                streamlit.write("已校验参数")
                streamlit.json(item.get("validated_arguments") or {})
                if item.get("aggregates"):
                    streamlit.write("聚合结果")
                    streamlit.json(item["aggregates"])
                if item.get("rows"):
                    streamlit.dataframe(item["rows"], hide_index=True, width="stretch")
                if item.get("units"):
                    streamlit.caption(f"单位：{item['units']}")
    else:
        streamlit.info("未返回实验数据结论；当前回答不包含来自确定性分析工具的数值证据。")

    calls = response.get("tool_calls")
    if isinstance(calls, list) and calls:
        with streamlit.expander("工具调用审计日志"):
            streamlit.dataframe(calls, hide_index=True, width="stretch")


def render_empty(streamlit: Any, title: str, detail: str) -> None:
    streamlit.markdown(
        '<div class="nl-card">'
        f'<div class="nl-card-title">{html.escape(title)}</div>'
        f'<div class="nl-card-copy">{html.escape(detail)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_exception(streamlit: Any, error: Exception, *, action: str) -> None:
    code = getattr(error, "code", type(error).__name__)
    message = getattr(error, "message", str(error)) or "未返回错误详情。"
    request_id = getattr(error, "request_id", None)
    retryable = bool(getattr(error, "retryable", False))
    status_code = getattr(error, "status_code", None)
    streamlit.error(f"{action}失败 · {code}：{message}")
    guidance = _error_guidance(code, status_code)
    if guidance:
        streamlit.warning(guidance)
    details = getattr(error, "details", None)
    if details:
        with streamlit.expander("错误详情"):
            streamlit.json(details)
    caption = []
    if request_id:
        caption.append(f"request_id {request_id}")
    if status_code:
        caption.append(f"HTTP {status_code}")
    caption.append("可重试" if retryable else "未标记为可重试")
    streamlit.caption(" · ".join(caption))


def _error_guidance(code: str, status_code: Any) -> str | None:
    """Return a readable, state-specific prompt for known error codes.

    The mapping covers every status the backend middleware and domain layer emit
    (401, 429, 503, transport/timeout, REVIEW_REQUIRED, RAG index unavailable).
    Unknown codes return ``None`` so the generic exception line stands alone.
    """

    normalized_code = str(code).strip().upper()
    sc = int(status_code) if isinstance(status_code, (int, float)) else 0

    # Authentication / authorization
    if normalized_code == "UNAUTHORIZED" or sc == 401:
        return "API Key 缺失或无效，请检查连接设置中的密钥配置，或联系管理员获取有效凭证。"
    if normalized_code == "FORBIDDEN" or sc == 403:
        return "当前身份无权执行此操作。请确认账户权限，或联系管理员调整角色。"

    # Rate limiting
    if normalized_code == "RATE_LIMITED" or sc == 429:
        return "请求过于频繁，已被限流。请等待数秒后重试；如持续触发，请联系管理员调整限流策略。"

    # Service unavailable / model not ready / RAG index not ready
    if (
        normalized_code in {"SERVICE_UNAVAILABLE", "MODEL_NOT_READY", "RAG_INDEX_NOT_READY"}
        or sc == 503
    ):
        if normalized_code == "MODEL_NOT_READY":
            return "模型当前不可用，可能仍在加载或权重缺失。请在模型注册表页面确认状态后重试。"
        if normalized_code == "RAG_INDEX_NOT_READY":
            return "知识检索索引尚未就绪。请先在知识库页面完成文档摄取与索引构建，再发起知识查询。"
        return "后端依赖服务暂时不可用。请稍后重试；如问题持续，请检查连接页面的组件健康状态。"

    # Transport-level failures (status_code == 0)
    if normalized_code == "TRANSPORT_ERROR":
        return "无法连接后端服务。请确认后端地址正确、服务已启动，且网络可达。"
    if normalized_code == "REQUEST_TIMEOUT":
        return "请求后端超时。可能是大文件上传或模型推理耗时较长，请增加超时设置后重试。"

    # Quality gate: review required
    if normalized_code == "REVIEW_REQUIRED":
        return (
            "质量门禁判定该结果需要人工复核。请在运行结果页面查看质量原因，"
            "并在复核子运行中提交修正。"
        )

    # Resource conflicts
    if normalized_code == "BOX_REVISION_CONFLICT":
        return "ROI 框版本已更新。请重新加载 ROI revision，基于最新版本再保存。"
    if normalized_code == "JOB_STATE_CONFLICT":
        return "当前项目状态不允许此操作。请刷新项目详情，确认状态后再试。"
    if normalized_code == "EXPORT_NOT_READY":
        return "结果尚未达到可导出状态。请等待运行完成后再执行导出。"

    # Not found
    if normalized_code in {"RESOURCE_NOT_FOUND", "MODEL_NOT_FOUND"} or sc == 404:
        return "请求的资源不存在。请确认 job_id、run_id 或 model_id 正确，且资源未被删除。"

    # Payload / media type
    if normalized_code == "PAYLOAD_TOO_LARGE" or sc == 413:
        return "上传内容超过大小限制。请缩减文件数量或大小后重试。"
    if normalized_code == "UNSUPPORTED_MEDIA_TYPE" or sc == 415:
        return "不支持的文件类型。请确认上传文件格式符合接口要求。"

    # Validation
    if normalized_code == "VALIDATION_ERROR" or sc == 422:
        return "请求参数校验失败。请检查表单输入，修正标记的字段后重新提交。"

    return None


def _nested(record: Mapping[str, Any], parent: str, child: str) -> Any:
    value = record.get(parent)
    return value.get(child) if isinstance(value, Mapping) else None


def _display(value: Any) -> str:
    return "—" if value is None else str(value)


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def _translate_detail(value: str) -> str:
    if value in _DETAIL_TRANSLATIONS:
        return _DETAIL_TRANSLATIONS[value]
    translated = value
    for english, chinese in {
        "service:": "API 服务：",
        "database:": "数据库：",
        "model_registry:": "模型注册表：",
        "rag_index:": "知识检索：",
    }.items():
        translated = translated.replace(english, chinese)
    for english, chinese in _DETAIL_TRANSLATIONS.items():
        translated = translated.replace(english, chinese)
    return translated


__all__ = [
    "display_enum",
    "render_artifact_links",
    "render_connection_status",
    "render_empty",
    "render_exception",
    "render_health_matrix",
    "render_job_overview",
    "render_query_response",
    "render_run_summary",
    "render_run_table",
    "section_header",
    "status_badge",
]
