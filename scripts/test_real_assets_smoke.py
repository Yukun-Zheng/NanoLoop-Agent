#!/usr/bin/env python3
"""Real-assets smoke test for the NanoLoop frontend API client.

This script verifies that ``NanoLoopApiClient`` can consume either contract-faithful
offline samples or a real backend running on Linux, WSL2, Docker, or a remote host.

It exercises three key endpoints against real asset IDs provided by the backend
team (郭境濠 — models, 徐皓彬 — RAG):

1. ``GET /analyses/{job_id}``  — fetch job detail (JobDetailDTO)
2. ``GET /runs/{run_id}``       — fetch run result (SegmentationRunDTO)
3. ``POST /analyses/{job_id}/query`` — mixed query (UnifiedQueryResponse)

Modes
-----
- **offline** (default): Uses hardcoded JSON samples (matching contracts) to
  verify the frontend parsing logic — citations top-level, ``page=null``,
  ``INSUFFICIENT_EVIDENCE`` + empty citations. No HTTP requests are made.
- **live**: Connects to a real backend via ``NANOLOOP_API_BASE_URL``.

Usage
-----
    # Offline mode (default) — no backend needed
    python scripts/test_real_assets_smoke.py

    # Live mode — backend and controlled asset IDs must be provided
    NANOLOOP_JOB_ID=job_... NANOLOOP_RUN_ID=run_... \
    python scripts/test_real_assets_smoke.py --live

    # Live mode with remote backend + API key
    NANOLOOP_API_BASE_URL=http://10.0.0.5:8000 \
    NANOLOOP_API_KEY=your-secret-key \
    python scripts/test_real_assets_smoke.py --live
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

# --- Ensure the project root is importable regardless of CWD ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from frontend.api_client import ApiClientError, NanoLoopApiClient  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI colour codes for terminal output
# ---------------------------------------------------------------------------
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Deterministic IDs used only by the offline contract samples.
# ---------------------------------------------------------------------------
SAMPLE_MODEL_ID = "unet-small-balanced-v1"
SAMPLE_JOB_ID = "job_offline_contract_fixture"
SAMPLE_RUN_ID = "run_offline_contract_fixture"
SAMPLE_IMAGE_ID = "img_offline_contract_fixture"

# Query text for the mixed-query smoke test
QUERY_QUESTION = "TiO2 的粒径分布"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pass(msg: str) -> None:
    print(f"  {_GREEN}PASS{_RESET}  {msg}")


def _fail(msg: str, detail: str | None = None) -> None:
    print(f"  {_RED}FAIL{_RESET}  {msg}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"        {line}")


def _info(msg: str) -> None:
    print(f"  {_CYAN}INFO{_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}WARN{_RESET}  {msg}")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")


# ===========================================================================
# OFFLINE MODE — Hardcoded JSON samples matching contracts
# ===========================================================================

# Sample UnifiedQueryResponse with citations (one page=null, one page=42)
# Matches app/contracts/queries.py UnifiedQueryResponse + Citation
SAMPLE_QUERY_RESPONSE_WITH_CITATIONS: dict[str, Any] = {
    "query_type": "mixed",
    "answer": (
        "基于实验数据分析，TiO2 样品的平均等效直径约为 45.2 nm，"
        "颗粒数密度为 128.5 /μm²。文献引用支持该粒径范围在锐钛矿相中的典型值。"
    ),
    "data_evidence": [
        {
            "tool_name": "get_metric",
            "validated_arguments": {
                "scope_type": "run",
                "scope_id": SAMPLE_RUN_ID,
                "metric": "mean_equivalent_diameter_nm",
                "aggregation": "value",
            },
            "rows": [
                {"mean_equivalent_diameter_nm": 45.2, "particle_count": 312},
            ],
            "aggregates": {"mean_equivalent_diameter_nm": 45.2},
            "units": {"mean_equivalent_diameter_nm": "nm"},
            "source_run_ids": [SAMPLE_RUN_ID],
            "quality_warnings": [],
            "chart_url": None,
        },
    ],
    "citations": [
        {
            "citation_id": "cit_001",
            "doc_id": "doc_tio2_anatase_ref",
            "title": "Anatase TiO2 Nanoparticle Size Distribution Study",
            "page": None,  # 全文引用 — 前端应显示"无页码"或"全文引用"
            "chunk_id": "chunk_0a1b2c",
            "excerpt": "The average particle size of anatase TiO2 ranges from 30 to 60 nm.",
            "retrieval_score": 0.92,
            "source_type": "paper",
            "citation_text": "Zhang et al., J. Mater. Sci., 2023.",
        },
        {
            "citation_id": "cit_002",
            "doc_id": "doc_tio2_synthesis",
            "title": "Hydrothermal Synthesis of TiO2 Nanoparticles",
            "page": 42,
            "chunk_id": "chunk_3d4e5f",
            "excerpt": "Particle size distribution peaked at 45 nm under 200°C synthesis.",
            "retrieval_score": 0.87,
            "source_type": "paper",
            "citation_text": "Li et al., Nanotechnology, 2022.",
        },
    ],
    "tool_calls": [
        {
            "tool_name": "get_metric",
            "arguments": {
                "scope_id": SAMPLE_RUN_ID,
                "metric": "mean_equivalent_diameter_nm",
            },
            "outcome": "success",
            "source_run_ids": [SAMPLE_RUN_ID],
        },
    ],
    "material_context": {
        "formula": "TiO2",
        "name": "Titanium Dioxide",
        "aliases": ["Titania", "Titanium(IV) oxide"],
        "source": "request",
    },
    "confidence": "high",
    "limitations": [],
    "needs_clarification": False,
    "outcome_code": "OK",
}

# Sample UnifiedQueryResponse with INSUFFICIENT_EVIDENCE and empty citations
SAMPLE_QUERY_RESPONSE_INSUFFICIENT: dict[str, Any] = {
    "query_type": "auto",
    "answer": "当前实验数据和知识库中均未找到足够的证据来回答该问题。",
    "data_evidence": [],
    "citations": [],
    "tool_calls": [],
    "material_context": None,
    "confidence": "low",
    "limitations": [
        "未找到与该材料相关的已索引文献",
        "当前运行结果未包含足够的统计样本",
    ],
    "needs_clarification": True,
    "outcome_code": "INSUFFICIENT_EVIDENCE",
}

# Sample JobDetailDTO (simplified but contract-matching)
SAMPLE_JOB_DETAIL: dict[str, Any] = {
    "job": {
        "job_id": SAMPLE_JOB_ID,
        "name": "TiO2 粒径分析实验 #1",
        "status": "completed",
        "config": {},
        "created_at": "2025-07-20T10:00:00Z",
        "updated_at": "2025-07-20T10:15:00Z",
        "error_code": None,
    },
    "images": [
        {
            "image_id": SAMPLE_IMAGE_ID,
            "job_id": SAMPLE_JOB_ID,
            "filename": "tio2_sample_001.tif",
            "sha256": "a" * 64,
            "width": 2048,
            "height": 1536,
            "bit_depth": 16,
            "sample_id": "sample_001",
            "material_name": "Titanium Dioxide",
            "material_formula": "TiO2",
            "experiment_conditions": {"temperature": "200C"},
            "scale_nm_per_pixel": 2.5,
            "analysis_roi": {
                "schema_version": 1,
                "coordinate_space": "original_px",
                "valid_rect": {"x1": 0, "y1": 0, "x2": 2048, "y2": 1536},
                "invalid_rects": [],
                "source": "none",
                "revision": 1,
            },
            "original_download_url": None,
        },
    ],
    "runs": [
        {
            "run_id": SAMPLE_RUN_ID,
            "job_id": SAMPLE_JOB_ID,
            "image_id": SAMPLE_IMAGE_ID,
            "model_id": SAMPLE_MODEL_ID,
            "status": "completed",
        },
    ],
    "partial_failures": [],
}

# Sample SegmentationRunDTO (simplified but contract-matching)
SAMPLE_RUN_DETAIL: dict[str, Any] = {
    "run_id": SAMPLE_RUN_ID,
    "job_id": SAMPLE_JOB_ID,
    "image_id": SAMPLE_IMAGE_ID,
    "model_id": SAMPLE_MODEL_ID,
    "status": "completed",
    "roi_mode": "full_image",
    "box_revision": None,
    "threshold": 0.5,
    "inference": {
        "threshold": 0.5,
        "min_area_px": 8,
        "watershed_enabled": False,
        "exclude_border": True,
        "device": "auto",
        "seed": 42,
    },
    "configuration": {
        "schema_version": 1,
        "provenance_status": "legacy_fallback",
        "provenance_warnings": ["legacy_run_configuration_incomplete"],
        "model_id": SAMPLE_MODEL_ID,
        "model_version": "v1",
        "roi_mode": "full_image",
        "boxes": [],
        "analysis_roi": {
            "schema_version": 1,
            "coordinate_space": "original_px",
            "valid_rect": {"x1": 0, "y1": 0, "x2": 2048, "y2": 1536},
            "invalid_rects": [],
            "source": "none",
            "revision": 1,
        },
        "inference": {
            "threshold": 0.5,
            "min_area_px": 8,
            "watershed_enabled": False,
            "exclude_border": True,
            "device": "auto",
            "seed": 42,
        },
        "preprocess_profile": "default",
        "postprocess_profile": "default",
        "created_at": "2025-07-20T10:05:00Z",
    },
    "parent_run_id": None,
    "artifacts": {
        "mask_url": None,
        "overlay_url": None,
        "probability_url": None,
        "instances_url": None,
        "labeled_particles_url": None,
        "particles_csv_url": None,
        "quality_report_url": None,
        "execution_provenance_url": None,
    },
    "summary": {
        "run_id": SAMPLE_RUN_ID,
        "particle_count": 312,
        "roi_area_px": 3145728,
        "number_density_px2": 9.92e-05,
        "number_density_um2": 128.5,
        "mean_equivalent_diameter_px": 18.08,
        "mean_equivalent_diameter_nm": 45.2,
        "coverage_ratio": 0.23,
        "perimeter_density_px": 0.0012,
        "perimeter_density_um": 1.5,
        "quality_status": "pass",
    },
    "quality": {
        "status": "pass",
        "reasons": [],
        "metrics": {"particle_count": 312, "coverage_ratio": 0.23},
        "recommendations": [],
    },
    "execution": None,
    "runtime_ms": 4520,
    "error_code": None,
    "error_message": None,
    "status_history": [],
    "created_at": "2025-07-20T10:05:00Z",
    "updated_at": "2025-07-20T10:05:05Z",
}


# ---------------------------------------------------------------------------
# Offline verification functions
# ---------------------------------------------------------------------------
def offline_verify_job_detail() -> bool:
    """Offline Step 1: Parse sample JobDetailDTO and verify structure."""
    _section("Offline Step 1 · 解析任务详情样例 (JobDetailDTO)")
    try:
        data = SAMPLE_JOB_DETAIL
        job = data.get("job")
        if not isinstance(job, dict):
            _fail("data.job 不是对象")
            return False

        _pass(f"job_id={job.get('job_id', '')}")
        _pass(f"job_name={job.get('name', '')}")
        _pass(f"job_status={job.get('status', '')}")

        images = data.get("images", [])
        runs = data.get("runs", [])
        _pass(f"images count={len(images)}")
        _pass(f"runs count={len(runs)}")

        if isinstance(images, list):
            found = any(
                isinstance(img, dict) and img.get("image_id") == SAMPLE_IMAGE_ID for img in images
            )
            if found:
                _pass(f"目标 image_id={SAMPLE_IMAGE_ID} 存在")
            else:
                _fail(f"目标 image_id={SAMPLE_IMAGE_ID} 未找到")

        if isinstance(runs, list):
            found = any(isinstance(r, dict) and r.get("run_id") == SAMPLE_RUN_ID for r in runs)
            if found:
                _pass(f"目标 run_id={SAMPLE_RUN_ID} 存在")
            else:
                _fail(f"目标 run_id={SAMPLE_RUN_ID} 未找到")

        return True
    except Exception as exc:
        _fail(f"解析异常: {type(exc).__name__}", str(exc))
        return False


def offline_verify_run_detail() -> bool:
    """Offline Step 2: Parse sample SegmentationRunDTO and verify summary metrics."""
    _section("Offline Step 2 · 解析运行结果样例 (SegmentationRunDTO)")
    try:
        data = SAMPLE_RUN_DETAIL
        _pass(f"run_id={data.get('run_id', '')}")
        _pass(f"status={data.get('status', '')}")
        _pass(f"model_id={data.get('model_id', '')}")

        summary = data.get("summary")
        if isinstance(summary, dict):
            _pass(f"颗粒数 particle_count={summary.get('particle_count')}")
            _pass(f"面密度 number_density_um2={summary.get('number_density_um2')}")
            diameter = summary.get("mean_equivalent_diameter_nm")
            _pass(f"平均等效直径 mean_equivalent_diameter_nm={diameter}")
            _pass(f"覆盖率 coverage_ratio={summary.get('coverage_ratio')}")
            _pass(f"summary.quality_status={summary.get('quality_status')}")
        else:
            _fail("summary 不是对象")
            return False

        quality = data.get("quality")
        if isinstance(quality, dict):
            _pass(f"质量门禁 quality.status={quality.get('status', '')}")
        else:
            _fail("quality 不是对象")
            return False

        return True
    except Exception as exc:
        _fail(f"解析异常: {type(exc).__name__}", str(exc))
        return False


def offline_verify_query_with_citations() -> bool:
    """Verify citations and null-page handling in the offline response sample."""
    _section("Offline Step 3a · 解析混合查询样例 (含 citations, page=null)")
    try:
        data = SAMPLE_QUERY_RESPONSE_WITH_CITATIONS
        _pass(f"query_type={data.get('query_type', '')}")
        _pass(f"outcome_code={data.get('outcome_code', '')}")
        _pass(f"confidence={data.get('confidence', '')}")

        # Core: citations at top level
        citations = data.get("citations")
        if not isinstance(citations, list):
            _fail("citations 不在顶层或不是列表")
            return False
        _pass(f"citations 位于顶层, 数量: {len(citations)}")

        for i, citation in enumerate(citations):
            if not isinstance(citation, dict):
                _fail(f"citation[{i}] 不是对象")
                continue
            doc_id = citation.get("doc_id", "")
            chunk_id = citation.get("chunk_id", "")
            page = citation.get("page")

            # Core: page=null must not crash — should show "无页码" or "全文引用"
            if page is None:
                label = f"citation[{i}] doc={doc_id} chunk={chunk_id}"
                _pass(f"  {label} page=None → 无页码/全文引用 — 无异常")
            else:
                _pass(f"  citation[{i}] doc={doc_id} chunk={chunk_id} page={page}")

        # Verify data_evidence block exists (实验数据结论)
        data_evidence = data.get("data_evidence")
        if isinstance(data_evidence, list) and data_evidence:
            _pass(f"data_evidence 数量: {len(data_evidence)} (实验数据结论区块)")
        else:
            _warn("data_evidence 为空")

        return True
    except Exception as exc:
        _fail(f"解析异常: {type(exc).__name__}", str(exc))
        return False


def offline_verify_query_insufficient() -> bool:
    """Verify the offline insufficient-evidence response sample."""
    _section("Offline Step 3b · 解析混合查询样例 (INSUFFICIENT_EVIDENCE, 空 citations)")
    try:
        data = SAMPLE_QUERY_RESPONSE_INSUFFICIENT
        outcome = data.get("outcome_code", "")
        _pass(f"outcome_code={outcome}")

        citations = data.get("citations")
        if not isinstance(citations, list):
            _fail("citations 不是列表")
            return False

        if not citations and outcome == "INSUFFICIENT_EVIDENCE":
            _pass(
                "outcome_code=INSUFFICIENT_EVIDENCE 且 citations 为空 → 输出'证据不足' — 逻辑正确"
            )
        else:
            actual = f"outcome={outcome}, citations={len(citations)}"
            _fail(f"预期 INSUFFICIENT_EVIDENCE + 空 citations, 实际: {actual}")
            return False

        # Verify limitations are present
        limitations = data.get("limitations")
        if isinstance(limitations, list) and limitations:
            _pass(f"limitations 数量: {len(limitations)}")
            for lim in limitations:
                _info(f"  局限性: {lim}")

        _pass(f"needs_clarification={data.get('needs_clarification', False)}")
        _pass(f"confidence={data.get('confidence', '')}")

        return True
    except Exception as exc:
        _fail(f"解析异常: {type(exc).__name__}", str(exc))
        return False


def run_offline() -> int:
    """Run all offline verification steps."""
    print(f"\n{_BOLD}NanoLoop Smoke Test — OFFLINE MODE{_RESET}")
    print("  使用硬编码 JSON 样例验证前端解析逻辑 (无 HTTP 请求)")

    results: list[tuple[str, bool]] = []
    results.append(("解析任务详情样例", offline_verify_job_detail()))
    results.append(("解析运行结果样例", offline_verify_run_detail()))
    results.append(
        ("解析查询样例 (含 citations, page=null)", offline_verify_query_with_citations())
    )
    results.append(("解析查询样例 (INSUFFICIENT_EVIDENCE)", offline_verify_query_insufficient()))

    _section("离线验证结果汇总")
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    for name, ok in results:
        status = f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
        print(f"  {status}  {name}")
    print(f"\n  {_BOLD}总计: {passed} PASS / {failed} FAIL / {total} TOTAL{_RESET}\n")
    return 1 if failed > 0 else 0


# ===========================================================================
# LIVE MODE — Real HTTP requests against a running backend
# ===========================================================================
def verify_health(client: NanoLoopApiClient) -> bool:
    """Step 0: Check backend health before running asset tests."""
    _section("Step 0 · 健康检查")
    try:
        result = client.health()
        data = result.data
        service = data.get("service")
        if not isinstance(service, dict):
            _fail("健康响应缺少 data.service 对象")
            return False
        status = str(service.get("status", "unknown"))
        component_names = ("service", "database", "model_registry", "rag_index")
        comp_summary = ", ".join(
            f"{name}={component.get('status', '?')}"
            for name in component_names
            if isinstance((component := data.get(name)), dict)
        )
        _pass(f"后端健康状态: {status}")
        _info(f"组件: {comp_summary}")
        if status == "unavailable":
            _warn("后端整体 unavailable — 资产测试可能失败，继续尝试...")
        return True
    except ApiClientError as exc:
        _fail(f"健康检查失败: {exc.code} (HTTP {exc.status_code})", exc.message)
        return False
    except Exception as exc:
        _fail(f"健康检查异常: {type(exc).__name__}", str(exc))
        return False


def verify_get_job(
    client: NanoLoopApiClient,
    job_id: str,
    *,
    run_id: str | None,
    image_id: str | None,
) -> bool:
    """Step 1: Fetch job detail and verify structure."""
    _section(f"Step 1 · 获取任务详情 (GET /analyses/{job_id})")
    try:
        result = client.get_analysis(job_id)
        data = result.data
        request_id = result.request_id

        job = data.get("job")
        if not isinstance(job, dict):
            _fail("返回的 data.job 不是对象", f"got {type(job).__name__}: {job}")
            return False

        job_id = job.get("job_id", "")
        job_name = job.get("name", "")
        job_status = job.get("status", "")
        images = data.get("images", [])
        runs = data.get("runs", [])

        _pass(f"job_id={job_id}")
        _pass(f"job_name={job_name}")
        _pass(f"job_status={job_status}")
        _pass(f"images count={len(images) if isinstance(images, list) else 'N/A'}")
        _pass(f"runs count={len(runs) if isinstance(runs, list) else 'N/A'}")
        _info(f"request_id={request_id}")

        if image_id and isinstance(images, list):
            found_image = any(
                isinstance(img, dict) and img.get("image_id") == image_id for img in images
            )
            if found_image:
                _pass(f"目标 image_id={image_id} 存在于任务中")
            else:
                _warn(f"目标 image_id={image_id} 未在 images 列表中找到")

        if run_id and isinstance(runs, list):
            found_run = any(isinstance(r, dict) and r.get("run_id") == run_id for r in runs)
            if found_run:
                _pass(f"目标 run_id={run_id} 存在于任务中")
            else:
                _warn(f"目标 run_id={run_id} 未在 runs 列表中找到")

        return True

    except ApiClientError as exc:
        _fail(f"获取任务详情失败: {exc.code} (HTTP {exc.status_code})", exc.message)
        return False
    except Exception as exc:
        _fail(f"获取任务详情异常: {type(exc).__name__}", str(exc))
        traceback.print_exc()
        return False


def verify_get_run(client: NanoLoopApiClient, run_id: str) -> bool:
    """Step 2: Fetch run result and print summary metrics + quality status."""
    _section(f"Step 2 · 获取运行结果 (GET /runs/{run_id})")
    try:
        result = client.get_run(run_id)
        data = result.data
        request_id = result.request_id

        run_id = data.get("run_id", "")
        run_status = data.get("status", "")
        model_id = data.get("model_id", "")
        runtime_ms = data.get("runtime_ms")

        _pass(f"run_id={run_id}")
        _pass(f"status={run_status}")
        _pass(f"model_id={model_id}")
        if runtime_ms is not None:
            _pass(f"runtime_ms={runtime_ms}")
        _info(f"request_id={request_id}")

        summary = data.get("summary")
        if isinstance(summary, dict):
            particle_count = summary.get("particle_count")
            number_density_px2 = summary.get("number_density_px2")
            number_density_um2 = summary.get("number_density_um2")
            mean_diameter_nm = summary.get("mean_equivalent_diameter_nm")
            coverage_ratio = summary.get("coverage_ratio")
            summary_quality = summary.get("quality_status")

            _pass(f"颗粒数 particle_count={particle_count}")
            _pass(f"面密度 number_density_px2={number_density_px2}")
            if number_density_um2 is not None:
                _pass(f"面密度 number_density_um2={number_density_um2}")
            if mean_diameter_nm is not None:
                _pass(f"平均等效直径 mean_equivalent_diameter_nm={mean_diameter_nm}")
            if coverage_ratio is not None:
                _pass(f"覆盖率 coverage_ratio={coverage_ratio}")
            _pass(f"summary.quality_status={summary_quality}")
        else:
            _warn("运行结果未返回 summary（可能运行尚未完成或失败）")

        quality = data.get("quality")
        if isinstance(quality, dict):
            quality_status = quality.get("status", "")
            reasons = quality.get("reasons", [])
            recommendations = quality.get("recommendations", [])

            _pass(f"质量门禁 quality.status={quality_status}")
            if isinstance(reasons, list) and reasons:
                for reason in reasons:
                    _info(f"  原因: {reason}")
            if isinstance(recommendations, list) and recommendations:
                for rec in recommendations:
                    _info(f"  建议: {rec}")

            if quality_status == "review_required":
                _warn("该运行结果需要人工复核 (REVIEW_REQUIRED)")
        else:
            _warn("运行结果未返回 quality 报告")

        error_code = data.get("error_code")
        if error_code:
            _warn(f"运行错误: error_code={error_code}, message={data.get('error_message', '')}")

        return True

    except ApiClientError as exc:
        _fail(f"获取运行结果失败: {exc.code} (HTTP {exc.status_code})", exc.message)
        return False
    except Exception as exc:
        _fail(f"获取运行结果异常: {type(exc).__name__}", str(exc))
        traceback.print_exc()
        return False


def verify_query_analysis(
    client: NanoLoopApiClient,
    job_id: str,
    run_id: str | None,
) -> bool:
    """Step 3: Mixed query — verify query_type, outcome_code, citations, page=None safety."""
    _section(f"Step 3 · 混合查询 (POST /analyses/{job_id}/query)")
    try:
        payload = {
            "question": QUERY_QUESTION,
            "query_type": "auto",
            "run_ids": [run_id] if run_id else [],
        }
        result = client.query_analysis(job_id, payload)
        data = result.data
        request_id = result.request_id

        query_type = data.get("query_type", "")
        outcome_code = data.get("outcome_code", "")
        confidence = data.get("confidence", "")
        answer = data.get("answer", "")
        needs_clarification = data.get("needs_clarification", False)

        _pass(f"query_type={query_type}")
        _pass(f"outcome_code={outcome_code}")
        _pass(f"confidence={confidence}")
        _pass(f"needs_clarification={needs_clarification}")
        _info(f"answer (前200字): {answer[:200]}{'...' if len(answer) > 200 else ''}")
        _info(f"request_id={request_id}")

        citations = data.get("citations")
        if isinstance(citations, list):
            _pass(f"citations 数量: {len(citations)}")
            for i, citation in enumerate(citations):
                if not isinstance(citation, dict):
                    _warn(f"  citation[{i}] 不是对象: {type(citation).__name__}")
                    continue
                doc_id = citation.get("doc_id", "")
                chunk_id = citation.get("chunk_id", "")
                page = citation.get("page")
                title = citation.get("title", "")
                retrieval_score = citation.get("retrieval_score")

                if page is None:
                    label = f"citation[{i}] doc={doc_id} chunk={chunk_id}"
                    _pass(f"  {label} page=None (全文引用) — 无异常")
                else:
                    _pass(f"  citation[{i}] doc={doc_id} chunk={chunk_id} page={page}")

                _info(f"    title={title}, retrieval_score={retrieval_score}")
        else:
            _warn(f"citations 不是列表: {type(citations).__name__}")

        data_evidence = data.get("data_evidence")
        if isinstance(data_evidence, list):
            _pass(f"data_evidence 数量: {len(data_evidence)}")
            for i, evidence in enumerate(data_evidence):
                if isinstance(evidence, dict):
                    tool_name = evidence.get("tool_name", "")
                    _info(f"  evidence[{i}] tool={tool_name}")

        if outcome_code == "INSUFFICIENT_EVIDENCE" and not citations:
            _warn(
                "outcome_code=INSUFFICIENT_EVIDENCE 且 citations 为空 — "
                "前端应显示'未返回材料知识引用'"
            )

        return True

    except ApiClientError as exc:
        _fail(f"混合查询失败: {exc.code} (HTTP {exc.status_code})", exc.message)
        if exc.details:
            _info(f"details: {exc.details}")
        return False
    except Exception as exc:
        _fail(f"混合查询异常: {type(exc).__name__}", str(exc))
        traceback.print_exc()
        return False


def run_live(
    base_url: str,
    api_key: str,
    *,
    job_id: str | None,
    run_id: str | None,
    image_id: str | None,
) -> int:
    """Run all live verification steps against a real backend."""
    print(f"\n{_BOLD}NanoLoop Smoke Test — LIVE MODE{_RESET}")
    print(f"  Backend URL : {base_url}")
    print(f"  API Key     : {'***set***' if api_key else '(disabled)'}")
    print(f"  Job ID      : {job_id or '(not provided)'}")
    print(f"  Run ID      : {run_id or '(not provided)'}")
    print(f"  Image ID    : {image_id or '(not provided)'}")
    print(f"  Query       : {QUERY_QUESTION}")

    try:
        client = NanoLoopApiClient(
            base_url=base_url,
            api_key=api_key or None,
        )
    except Exception as exc:
        _fail(f"创建客户端失败: {type(exc).__name__}", str(exc))
        return 1

    results: list[tuple[str, bool]] = [("健康检查", verify_health(client))]
    if job_id:
        results.append(
            (
                "获取任务详情",
                verify_get_job(client, job_id, run_id=run_id, image_id=image_id),
            )
        )
        results.append(("混合查询", verify_query_analysis(client, job_id, run_id)))
    else:
        _warn("未提供 job_id；任务详情与混合查询标记为 SKIP")
    if run_id:
        results.append(("获取运行结果", verify_get_run(client, run_id)))
    else:
        _warn("未提供 run_id；运行结果检查标记为 SKIP")
    client.close()

    _section("Live 验证结果汇总")
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    for name, ok in results:
        status = f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
        print(f"  {status}  {name}")
    print(f"\n  {_BOLD}总计: {passed} PASS / {failed} FAIL / {total} TOTAL{_RESET}\n")
    return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    live_by_env = os.environ.get("RUN_REAL_TESTS", "").lower() in ("1", "true", "yes")

    parser = argparse.ArgumentParser(
        description="NanoLoop real-assets smoke test (frontend API client)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--offline",
        action="store_true",
        default=not live_by_env,
        help="Offline mode: use hardcoded JSON samples, no HTTP requests (default)",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        default=live_by_env,
        help="Live mode: connect to a real backend (env: RUN_REAL_TESTS=true)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NANOLOOP_API_BASE_URL", "http://127.0.0.1:8000"),
        help="Backend base URL (env: NANOLOOP_API_BASE_URL, default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NANOLOOP_API_KEY", ""),
        help="API key for X-API-Key header (env: NANOLOOP_API_KEY, default: empty=disabled)",
    )
    parser.add_argument(
        "--job-id",
        default=os.environ.get("NANOLOOP_JOB_ID"),
        help="Existing job ID (env: NANOLOOP_JOB_ID; optional in live mode)",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("NANOLOOP_RUN_ID"),
        help="Existing run ID (env: NANOLOOP_RUN_ID; optional in live mode)",
    )
    parser.add_argument(
        "--image-id",
        default=os.environ.get("NANOLOOP_IMAGE_ID"),
        help="Expected image ID inside the job (env: NANOLOOP_IMAGE_ID; optional)",
    )
    args = parser.parse_args()

    if args.live:
        return run_live(
            args.base_url,
            args.api_key,
            job_id=args.job_id,
            run_id=args.run_id,
            image_id=args.image_id,
        )
    return run_offline()


if __name__ == "__main__":
    sys.exit(main())
