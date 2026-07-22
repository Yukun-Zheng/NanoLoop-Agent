"""Contract-faithful degraded backend stub for local integration testing.

This server emits REAL HTTP status codes (503, 429, 401, 200) with the EXACT
error/success envelope shapes defined in app/contracts/common.py and
app/api/errors.py. It is NOT static display data — it is a test harness that
simulates the degraded state where models are unavailable and RAG index is not
built, so the frontend's _error_guidance() function is exercised by real HTTP
responses rather than mocked exceptions.

Run:
    python scripts/degraded_stub_server.py [--port 8001] [--auth-mode key]

When --auth-mode key is set, all /api/v1/* routes except /health require an
X-API-Key header matching the --api-key value (default: test-secret-001).
Without the header, the server returns 401 AUTHENTICATION_REQUIRED.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Ensure project root is on sys.path so we could import contracts if needed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Envelope helpers — mirror app/api/responses.py exactly.
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    rid = request.headers.get("x-request-id")
    if rid and 1 <= len(rid) <= 100:
        return rid
    return f"stub_{uuid4().hex}"


def _success(data: dict, request: Request, *, accepted: bool = False) -> JSONResponse:
    body = {
        "request_id": _request_id(request),
        "status": "accepted" if accepted else "success",
        "data": data,
        "error": None,
    }
    return JSONResponse(
        status_code=202 if accepted else 200,
        content=body,
        headers={"X-Request-ID": body["request_id"]},
    )


def _error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
    retryable: bool = False,
    headers: dict | None = None,
) -> JSONResponse:
    body = {
        "request_id": _request_id(request),
        "status": "error",
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "retryable": retryable,
        },
    }
    response_headers = {"X-Request-ID": body["request_id"]}
    if headers:
        response_headers.update(headers)
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=response_headers,
    )


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_models = [
    {
        "model_id": "unet-small-balanced-v1",
        "family": "unet",
        "variant": "small_particle",
        "quality_tier": "balanced",
        "version": "1",
        "status": "unavailable",
        "supports_box_prompt": False,
        "default_threshold": 0.30,
        "default_min_area_px": None,
        "preprocess_profile": "sem-gray-unit-crop-bottom-130-v1",
        "postprocess_profile": "semantic-mask-v1",
        "inference_invalid_bottom_px": 130,
        "applicable_materials": [],
        "metrics": {},
        "metric_context": {},
        "notes": "Small-particle TorchScript U-Net; external artifact remains unavailable.",
        "health_error": "Model checkpoint is not bundled with this repository.",
    },
    {
        "model_id": "unet-large-optimized-v1",
        "family": "unet",
        "variant": "large_particle",
        "quality_tier": "balanced",
        "version": "1",
        "status": "unavailable",
        "supports_box_prompt": False,
        "default_threshold": 0.50,
        "default_min_area_px": 512,
        "preprocess_profile": "sem-gray-unit-crop-bottom-180-v1",
        "postprocess_profile": "semantic-mask-v1",
        "inference_invalid_bottom_px": 180,
        "applicable_materials": [],
        "metrics": {},
        "metric_context": {},
        "notes": "Large-particle TorchScript U-Net; external artifact remains unavailable.",
        "health_error": "Model checkpoint is not bundled with this repository.",
    },
]

# Rate limiter for 429 testing
_rate_bucket: dict[str, list[float]] = {}
_RATE_LIMIT = 3
_RATE_WINDOW = 60.0


def create_app(*, auth_mode: str = "disabled", api_key: str = "test-secret-001") -> FastAPI:
    app = FastAPI(title="NanoLoop Degraded Stub")

    # --- Auth + rate limit middleware ---
    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        path = request.url.path

        # Allow /docs, /openapi.json, and non-API paths
        if not path.startswith("/api/v1"):
            return await call_next(request)

        # Auth check (only when enabled)
        if auth_mode == "key":
            provided = request.headers.get("x-api-key", "")
            if not provided or provided != api_key:
                return _error(
                    request,
                    status_code=401,
                    code="AUTHENTICATION_REQUIRED",
                    message="需要有效的 API Key",
                    headers={"WWW-Authenticate": 'ApiKey realm="nanoloop"'},
                )

        # Rate limit check (only when enabled via query param ?rate=test)
        if request.query_params.get("rate") == "test":
            peer = request.client.host if request.client else "unknown"
            now = time.monotonic()
            bucket = _rate_bucket.setdefault(peer, [])
            _rate_bucket[peer] = [t for t in bucket if now - t < _RATE_WINDOW]
            if len(_rate_bucket[peer]) >= _RATE_LIMIT:
                return _error(
                    request,
                    status_code=429,
                    code="RATE_LIMITED",
                    message="请求过于频繁，请稍后重试",
                    details={"limit": _RATE_LIMIT, "window_seconds": _RATE_WINDOW},
                    retryable=True,
                    headers={"Retry-After": str(int(_RATE_WINDOW))},
                )
            _rate_bucket[peer].append(now)

        return await call_next(request)

    # --- Health ---
    @app.get("/api/v1/health")
    async def health(request: Request):
        data = {
            "service": {"status": "healthy", "detail": None},
            "database": {"status": "healthy", "detail": None},
            "model_registry": {
                "status": "unavailable",
                "detail": "registry and gateway unavailable",
            },
            "rag_index": {
                "status": "unavailable",
                "detail": "knowledge index not built",
            },
            "version": "0.1.0-stub",
        }
        return _success(data, request)

    # --- Models ---
    @app.get("/api/v1/models")
    async def list_models(request: Request):
        return _success({"models": _models}, request)

    @app.post("/api/v1/models/recommend")
    async def recommend_models(request: Request):
        # Even recommendation returns empty candidates when all models are unavailable
        return _success(
            {"candidates": [], "requires_user_confirmation": True}, request
        )

    # --- Analyses ---
    @app.post("/api/v1/analyses")
    async def create_analysis(request: Request):
        form = await request.form()
        metadata_json = form.get("metadata_json")
        if not metadata_json:
            return _error(
                request,
                status_code=422,
                code="VALIDATION_ERROR",
                message="请求参数校验失败",
                details={"issues": [{"location": ["metadata_json"], "message": "field required", "type": "value_error.missing"}]},
            )

        import json

        try:
            metadata = json.loads(str(metadata_json))
        except (json.JSONDecodeError, TypeError):
            return _error(
                request,
                status_code=422,
                code="VALIDATION_ERROR",
                message="请求参数校验失败",
                details={"issues": [{"location": ["metadata_json"], "message": "invalid JSON", "type": "value_error"}]},
            )

        files = form.getlist("files")
        if not files:
            return _error(
                request,
                status_code=422,
                code="VALIDATION_ERROR",
                message="请求参数校验失败",
                details={"issues": [{"location": ["files"], "message": "at least one file required", "type": "value_error"}]},
            )

        job_id = f"job_{uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()

        images = []
        for upload in files:
            # Accept both fastapi.UploadFile and starlette UploadFile
            if not hasattr(upload, "read") or not hasattr(upload, "filename"):
                continue
            content = await upload.read()
            sha256 = hashlib.sha256(content).hexdigest()
            image_id = f"img_{uuid4().hex[:12]}"
            images.append(
                {
                    "image_id": image_id,
                    "job_id": job_id,
                    "filename": upload.filename or "unknown.png",
                    "sha256": sha256,
                    "width": 1024,
                    "height": 768,
                    "bit_depth": 8,
                    "sample_id": metadata.get("images", [{}])[0].get("sample_id", "S001"),
                    "material_name": metadata.get("images", [{}])[0].get("material_name"),
                    "material_formula": metadata.get("images", [{}])[0].get("material_formula"),
                    "experiment_conditions": metadata.get("images", [{}])[0].get("experiment_conditions", {}),
                    "scale_nm_per_pixel": None,
                    "analysis_roi": {
                        "schema_version": 1,
                        "coordinate_space": "original_px",
                        "valid_rect": {"x1": 0, "y1": 0, "x2": 1024, "y2": 768},
                        "invalid_rects": [],
                        "source": "none",
                        "revision": 1,
                    },
                    "original_download_url": None,
                }
            )

        job = {
            "job_id": job_id,
            "name": metadata.get("job_name", "stub-job"),
            "status": "READY_FOR_CONFIGURATION",
            "config": {},
            "created_at": now,
            "updated_at": now,
            "error_code": None,
        }
        _jobs[job_id] = {"job": job, "images": images, "runs": []}

        return _success({"job": job, "images": images, "runs": [], "partial_failures": []}, request)

    @app.get("/api/v1/analyses/{job_id}")
    async def get_analysis(job_id: str, request: Request):
        job_data = _jobs.get(job_id)
        if not job_data:
            return _error(
                request,
                status_code=404,
                code="RESOURCE_NOT_FOUND",
                message="找不到指定资源",
                retryable=False,
            )
        return _success(
            {
                "job": job_data["job"],
                "images": job_data["images"],
                "runs": job_data["runs"],
                "partial_failures": [],
            },
            request,
        )

    # --- Boxes ---
    @app.get("/api/v1/analyses/{job_id}/images/{image_id}/boxes")
    async def get_boxes(job_id: str, image_id: str, request: Request):
        return _success(
            {"image_id": image_id, "revision": 0, "boxes": []}, request
        )

    @app.put("/api/v1/analyses/{job_id}/images/{image_id}/boxes")
    async def replace_boxes(job_id: str, image_id: str, request: Request):
        body = await request.json()
        boxes = body.get("boxes", [])
        return _success(
            {"image_id": image_id, "revision": 1, "boxes": boxes}, request
        )

    # --- Runs (THIS IS THE KEY DEGRADED PATH) ---
    @app.post("/api/v1/analyses/{job_id}/runs")
    async def create_runs(job_id: str, request: Request):
        # Simulate ModelNotReadyError — models are all unavailable
        return _error(
            request,
            status_code=503,
            code="MODEL_NOT_READY",
            message="模型当前不可用",
            details={
                "model_ids": [m["model_id"] for m in _models],
                "reason": "all registered models are unavailable; checkpoint not bundled",
            },
            retryable=False,
        )

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        return _error(
            request,
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message="找不到指定资源",
        )

    # --- Queries (KEY DEGRADED PATH: RAG index not ready) ---
    @app.post("/api/v1/analyses/{job_id}/query")
    async def query_analysis(job_id: str, request: Request):
        body = await request.json()
        query_type = body.get("query_type", "auto")

        # ========== 新增：专用于 UI 渲染测试的分支 ==========
        if query_type == "test_render":
            test_data = {
                "query_type": "material_knowledge",
                "answer": "这是测试回答，其中一条引文没有页码。",
                "citations": [
                    {
                        "citation_id": "C1",
                        "doc_id": "doc_test_001",
                        "title": "有页码的文档",
                        "page": 3,
                        "chunk_id": "chunk_001",
                        "excerpt": "这是正文...",
                        "retrieval_score": 0.9,
                        "source_type": "paper",
                        "citation_text": "Test et al."
                    },
                    {
                        "citation_id": "C2",
                        "doc_id": "doc_test_002",
                        "title": "无页码的网页或TXT",
                        "page": None,   # 这就是你要测试的 null 值
                        "chunk_id": "chunk_002",
                        "excerpt": "这条来自没有页码的文档...",
                        "retrieval_score": 0.8,
                        "source_type": "web",
                        "citation_text": "Web source"
                    }
                ],
                "material_context": {"formula": "TiO2", "name": "二氧化钛"},
                "confidence": "high",
                "outcome_code": "OK"
            }
            return _success(test_data, request)
    # ========== 新增结束 ==========

        # When RAG index is not built, knowledge queries return 503
        if query_type in ("material_knowledge", "mixed", "auto"):
            return _error(
                request,
                status_code=503,
                code="RAG_INDEX_NOT_READY",
                message="知识索引尚未就绪",
                details={
                    "index_status": "unavailable",
                    "detail": "knowledge index not built",
                },
                retryable=False,
            )

        # For analysis_data queries with no runs, return INSUFFICIENT_EVIDENCE
        # as a 200 success (special envelope per app/api/errors.py)
        data = {
            "query_type": "analysis_data",
            "answer": "当前实验数据不足，无法形成有依据的回答。",
            "data_evidence": [],
            "citations": [],
            "tool_calls": [],
            "material_context": None,
            "confidence": "low",
            "limitations": ["知识库证据不足", "尚未完成任何分析运行"],
            "needs_clarification": False,
            "outcome_code": "INSUFFICIENT_EVIDENCE",
        }
        return _success(data, request)

    # --- Knowledge ---
    @app.get("/api/v1/knowledge/documents")
    async def list_knowledge_documents(request: Request):
        return _success({"documents": [], "pagination": {"limit": 50, "offset": 0, "total": 0}}, request)

    @app.post("/api/v1/knowledge/documents")
    async def ingest_knowledge_document(request: Request):
        return _error(
            request,
            status_code=503,
            code="RAG_INDEX_NOT_READY",
            message="知识索引尚未就绪，无法摄取文档",
            retryable=False,
        )

    @app.post("/api/v1/knowledge/reindex")
    async def reindex_knowledge(request: Request):
        return _error(
            request,
            status_code=503,
            code="RAG_INDEX_NOT_READY",
            message="知识索引尚未就绪，无法重建索引",
            retryable=False,
        )

    @app.patch("/api/v1/knowledge/documents/{doc_id}")
    async def update_knowledge_document(doc_id: str, request: Request):
        return _error(
            request,
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message="找不到指定资源",
        )

    # --- Export ---
    @app.get("/api/v1/analyses/{job_id}/export")
    async def export_analysis(job_id: str, request: Request):
        return _error(
            request,
            status_code=409,
            code="EXPORT_NOT_READY",
            message="结果尚未达到可导出状态",
        )

    # --- Files (artifact download) ---
    @app.get("/api/v1/files/{token}")
    async def download_file(token: str, request: Request):
        return _error(
            request,
            status_code=404,
            code="RESOURCE_NOT_FOUND",
            message="找不到指定资源",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoLoop degraded backend stub")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument(
        "--auth-mode",
        choices=["disabled", "key"],
        default="disabled",
        help="Auth mode: disabled (no key needed) or key (X-API-Key required)",
    )
    parser.add_argument("--api-key", default="test-secret-001", help="API key for auth mode")
    args = parser.parse_args()

    app = create_app(auth_mode=args.auth_mode, api_key=args.api_key)
    print(f"[stub] Starting degraded backend on port {args.port} (auth={args.auth_mode})")
    print(f"[stub] Health:       GET http://127.0.0.1:{args.port}/api/v1/health")
    print(f"[stub] Models:       GET http://127.0.0.1:{args.port}/api/v1/models")
    print(f"[stub] Create run:   POST http://127.0.0.1:{args.port}/api/v1/analyses/{{job_id}}/runs -> 503 MODEL_NOT_READY")
    print(f"[stub] Query:        POST http://127.0.0.1:{args.port}/api/v1/analyses/{{job_id}}/query -> 503 RAG_INDEX_NOT_READY")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
