#!/usr/bin/env python
"""Backend connectivity verification script.

Uses the real NanoLoopApiClient (no mocks) to verify that the backend
at http://127.0.0.1:8000 is reachable and responding correctly.

Checks:
  1. GET /health          -> service status (healthy/degraded/unavailable)
  2. GET /models          -> model count and individual statuses
  3. GET /analyses/{job}  -> optional job detail retrieval

Usage:
    python scripts/test_backend_connectivity.py
    NANOLOOP_API_BASE_URL=http://localhost:8000 python scripts/test_backend_connectivity.py
    NANOLOOP_API_KEY=secret python scripts/test_backend_connectivity.py
    python scripts/test_backend_connectivity.py --job-id job_...

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
    2 = backend not reachable (not started)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `frontend` package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402

from frontend.api_client import ApiClientError, NanoLoopApiClient  # noqa: E402

# --- Configuration ---------------------------------------------------------

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# --- Helpers ---------------------------------------------------------------


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, stripping surrounding whitespace."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _make_client(base_url: str, api_key: str | None) -> NanoLoopApiClient:
    """Build a NanoLoopApiClient from environment configuration."""
    return NanoLoopApiClient(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(15.0, connect=5.0, pool=5.0),
    )


def _is_transport_error(exc: ApiClientError) -> bool:
    """Check if an ApiClientError is a transport-level connection failure.

    The client wraps httpx.ConnectError / TimeoutException into
    ApiClientError(status_code=0, code='TRANSPORT_ERROR').
    """
    return exc.status_code == 0 or exc.code == "TRANSPORT_ERROR"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_kv(key: str, value: object) -> None:
    print(f"  {key:<24} {value}")


# --- Individual checks -----------------------------------------------------


def check_health(client: NanoLoopApiClient) -> bool:
    """Call GET /health and print the service status."""
    _print_header("Check 1/3: GET /health")
    try:
        result = client.health()
    except httpx.ConnectError as exc:
        print(f"  FAIL: 后端未启动 — 连接失败: {exc}")
        return False
    except httpx.TimeoutException as exc:
        print(f"  FAIL: 请求超时 — 后端可能未启动: {exc}")
        return False
    except ApiClientError as exc:
        if _is_transport_error(exc):
            print(f"  FAIL: 后端未启动 — 无法连接后端服务 ({exc.details})")
        else:
            print(f"  FAIL: 后端返回错误 [{exc.status_code}] {exc.code}: {exc.message}")
            print(f"       request_id: {exc.request_id}")
        return False

    data = result.data or {}
    service = data.get("service", {})
    status = service.get("status", "unknown") if isinstance(service, dict) else "unknown"
    detail = service.get("detail", "") if isinstance(service, dict) else ""

    _print_kv("request_id", result.request_id)
    _print_kv("api_status", result.status)
    _print_kv("service.status", status)
    if detail:
        _print_kv("service.detail", detail)

    # Print all health components if present.
    for key, value in data.items():
        if key == "service":
            continue
        if isinstance(value, dict):
            comp_status = value.get("status", "unknown")
            comp_detail = value.get("detail", "")
            _print_kv(f"{key}.status", comp_status)
            if comp_detail:
                _print_kv(f"{key}.detail", comp_detail)

    if status in ("healthy", "degraded"):
        print("  -> PASS")
        return True
    print(f"  -> FAIL: service status is '{status}'")
    return False


def check_list_models(client: NanoLoopApiClient) -> bool:
    """Call GET /models and print the model count and statuses."""
    _print_header("Check 2/3: GET /models")
    try:
        result = client.list_models()
    except httpx.ConnectError as exc:
        print(f"  FAIL: 后端未启动 — 连接失败: {exc}")
        return False
    except httpx.TimeoutException as exc:
        print(f"  FAIL: 请求超时: {exc}")
        return False
    except ApiClientError as exc:
        if _is_transport_error(exc):
            print(f"  FAIL: 后端未启动 — 无法连接后端服务 ({exc.details})")
        else:
            print(f"  FAIL: 后端返回错误 [{exc.status_code}] {exc.code}: {exc.message}")
            print(f"       request_id: {exc.request_id}")
        return False

    data = result.data or {}
    models = data.get("models", [])
    if not isinstance(models, list):
        models = []

    _print_kv("request_id", result.request_id)
    _print_kv("api_status", result.status)
    _print_kv("model_count", len(models))

    # Aggregate statuses.
    status_counts: dict[str, int] = {}
    for model in models:
        if isinstance(model, dict):
            model_status = model.get("status", "unknown")
            status_counts[model_status] = status_counts.get(model_status, 0) + 1
            _print_kv(
                f"  model: {model.get('model_id', '?')}",
                f"status={model_status}  family={model.get('family', '?')}",
            )

    if status_counts:
        print()
        for s, count in sorted(status_counts.items()):
            _print_kv(f"  status '{s}'", count)

    # PASS if we got a valid response with at least one model.
    if len(models) > 0:
        print("  -> PASS")
        return True
    print("  -> FAIL: no models returned (model registry may be empty)")
    return False


def check_get_analysis(client: NanoLoopApiClient, job_id: str) -> bool:
    """Call GET /analyses/{job_id} and print the job detail."""
    _print_header(f"Check 3/3: GET /analyses/{job_id}")
    try:
        result = client.get_analysis(job_id)
    except httpx.ConnectError as exc:
        print(f"  FAIL: 后端未启动 — 连接失败: {exc}")
        return False
    except httpx.TimeoutException as exc:
        print(f"  FAIL: 请求超时: {exc}")
        return False
    except ApiClientError as exc:
        if _is_transport_error(exc):
            print(f"  FAIL: 后端未启动 — 无法连接后端服务 ({exc.details})")
        else:
            print(f"  FAIL: 后端返回错误 [{exc.status_code}] {exc.code}: {exc.message}")
            print(f"       request_id: {exc.request_id}")
            if exc.details:
                print(f"       details: {exc.details}")
        return False

    data = result.data or {}
    _print_kv("request_id", result.request_id)
    _print_kv("api_status", result.status)
    job = data.get("job")
    if not isinstance(job, dict):
        print("  -> FAIL: response data.job is not an object")
        return False
    _print_kv("job_id", job.get("job_id", "?"))
    _print_kv("status", job.get("status", "?"))
    _print_kv("name", job.get("name", "?"))

    images = data.get("images", [])
    if isinstance(images, list):
        _print_kv("image_count", len(images))
        for img in images:
            if isinstance(img, dict):
                _print_kv(
                    f"  image: {img.get('image_id', '?')}",
                    f"filename={img.get('filename', '?')}",
                )

    runs = data.get("runs", [])
    if isinstance(runs, list):
        _print_kv("run_count", len(runs))

    # PASS if we got a valid response with a matching job_id.
    returned_job_id = job.get("job_id")
    if returned_job_id == job_id:
        print("  -> PASS")
        return True
    if returned_job_id:
        print(f"  -> FAIL: returned job_id '{returned_job_id}' != requested '{job_id}'")
        return False
    print("  -> FAIL: no job_id in response")
    return False


# --- Main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the NanoLoop backend REST connection")
    parser.add_argument(
        "--base-url",
        default=_env("NANOLOOP_API_BASE_URL", DEFAULT_BASE_URL),
        help="Backend origin (env: NANOLOOP_API_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=_env("NANOLOOP_API_KEY"),
        help="Shared API key (env: NANOLOOP_API_KEY)",
    )
    parser.add_argument(
        "--job-id",
        default=_env("NANOLOOP_JOB_ID"),
        help="Optional existing job to fetch (env: NANOLOOP_JOB_ID)",
    )
    args = parser.parse_args()
    base_url = args.base_url or DEFAULT_BASE_URL
    api_key = args.api_key
    job_id = args.job_id

    _print_header("NanoLoop-Agent Backend Connectivity Test")
    _print_kv("base_url", base_url)
    _print_kv("api_key", "set" if api_key else "not set (optional)")
    _print_kv("job_id", job_id or "not set (job check skipped)")

    client = _make_client(base_url, api_key)

    # Quick connectivity probe before running full checks.
    try:
        client.health()
    except httpx.ConnectError:
        print(f"\n  [FAIL] 后端未启动 — 无法连接到 {base_url}")
        print("  请先启动后端:")
        print("    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
        return 2
    except httpx.TimeoutException:
        print(f"\n  [FAIL] 后端连接超时 — {base_url} 无响应")
        print("  请检查后端是否正在运行。")
        return 2
    except ApiClientError as exc:
        if _is_transport_error(exc):
            print(f"\n  [FAIL] 后端未启动 — 无法连接到 {base_url}")
            print("  请先启动后端:")
            print("    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
            return 2
        # Server is running but returned an error — proceed with full checks.

    results = [check_health(client), check_list_models(client)]
    if job_id:
        results.append(check_get_analysis(client, job_id))
    else:
        print("\n  [SKIP] GET /analyses/{job_id}: provide --job-id or NANOLOOP_JOB_ID")
    client.close()

    passed = sum(results)
    total = len(results)
    _print_header(f"Summary: {passed}/{total} checks passed")

    if passed == total:
        print("  ALL PASS")
        return 0
    failed = total - passed
    print(f"  {failed} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
