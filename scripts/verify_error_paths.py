"""Verify backend error contracts against real HTTP responses from the degraded stub.

This script uses the shared synchronous API client to verify that every
required backend error path is triggered by a genuine HTTP status code from
the stub server, not a mocked exception. Browser presentation of these errors
is covered by the TypeScript frontend tests.

Usage:
    # Start the stub first (in another terminal or background):
    python scripts/degraded_stub_server.py --port 8001

    # Run verification:
    python scripts/verify_error_paths.py --base-url http://127.0.0.1:8001

    # For 401 test, start stub with --auth-mode key, then:
    python scripts/verify_error_paths.py --base-url http://127.0.0.1:8001 --test-401

    # For 429 test:
    python scripts/verify_error_paths.py --base-url http://127.0.0.1:8001 --test-429
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import httpx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.nanoloop_api_client import (  # noqa: E402
    ApiClientError,
    NanoLoopApiClient,
    UploadPart,
)

# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

_passed: list[str] = []
_failed: list[str] = []
_skipped: list[str] = []


def _record(category: str, description: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {category}: {description}"
    if detail:
        line += f" — {detail}"
    print(line)
    if ok:
        _passed.append(f"{category}: {description}")
    else:
        _failed.append(f"{category}: {description} — {detail}")


def _skip(category: str, description: str, reason: str) -> None:
    print(f"  [SKIP] {category}: {description} — {reason}")
    _skipped.append(f"{category}: {description} — {reason}")


# ---------------------------------------------------------------------------
# Individual verification tests
# ---------------------------------------------------------------------------


def verify_health(client: NanoLoopApiClient) -> dict | None:
    """GET /health — verify degraded rollup with unavailable components."""
    print("\n=== 1. Health Check (degraded state) ===")
    try:
        result = client.health()
        data = result.data
        # Verify model_registry is unavailable
        mr = data.get("model_registry", {})
        mr_status = str(mr.get("status", "")).casefold()
        _record(
            "Health",
            "model_registry status is unavailable",
            mr_status == "unavailable",
            f"got status={mr_status!r}, detail={mr.get('detail')!r}",
        )

        # Verify rag_index is unavailable
        ri = data.get("rag_index", {})
        ri_status = str(ri.get("status", "")).casefold()
        _record(
            "Health",
            "rag_index status is unavailable",
            ri_status == "unavailable",
            f"got status={ri_status!r}, detail={ri.get('detail')!r}",
        )

        component_names = ("service", "database", "model_registry", "rag_index")
        statuses = {
            name: str((data.get(name) or {}).get("status", "unavailable")).casefold()
            for name in component_names
        }
        unhealthy = {name for name, status in statuses.items() if status != "healthy"}
        unavailable_core = any(statuses[name] == "unavailable" for name in ("service", "database"))
        rollup_status = (
            "unavailable" if unavailable_core else "degraded" if unhealthy else "healthy"
        )

        # Verify the backend component records imply a degraded service.
        _record(
            "Health",
            "component records imply degraded service",
            rollup_status == "degraded",
            f"got status={rollup_status!r}",
        )

        # Verify unhealthy components include model_registry and rag_index
        _record(
            "Health",
            "unhealthy components include model_registry and rag_index",
            {"model_registry", "rag_index"}.issubset(unhealthy),
            f"got unhealthy={sorted(unhealthy)}",
        )

        return data
    except Exception as exc:
        _record("Health", "health check completed", False, str(exc))
        traceback.print_exc()
        return None


def verify_models(client: NanoLoopApiClient) -> list:
    """GET /models — verify all models show unavailable, not empty list or crash."""
    print("\n=== 2. Model Catalog (all unavailable) ===")
    try:
        result = client.list_models()
        models = result.data.get("models", [])

        _record(
            "Models",
            "model list is non-empty (not blank/crash)",
            len(models) > 0,
            f"got {len(models)} models",
        )

        all_unavailable = all(str(m.get("status", "")).casefold() == "unavailable" for m in models)
        _record(
            "Models",
            "all models have status=unavailable",
            all_unavailable,
            f"statuses={[m.get('status') for m in models]}",
        )

        # The backend contract must not advertise an unavailable model as ready.
        for m in models:
            _record(
                "Models",
                f"{m['model_id']} is not advertised as ready",
                str(m.get("status", "")).casefold() != "ready",
                f"status={m.get('status')!r}",
            )

        return models
    except Exception as exc:
        _record("Models", "model list completed", False, str(exc))
        traceback.print_exc()
        return []


def verify_upload(client: NanoLoopApiClient) -> str | None:
    """POST /analyses — verify upload succeeds even in degraded state."""
    print("\n=== 3. Upload (should succeed in degraded state) ===")
    try:
        # Create a minimal valid 1x1 PNG using struct
        import struct
        import zlib

        def _make_png() -> bytes:
            width, height = 1, 1
            raw = b"\x00\x00\x00\xff"  # filter byte + RGBA pixel
            compressed = zlib.compress(raw)
            chunks = []
            # IHDR
            ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
            chunks.append(b"IHDR" + ihdr_data)
            # IDAT
            chunks.append(b"IDAT" + compressed)
            # IEND
            chunks.append(b"IEND")
            png = b"\x89PNG\r\n\x1a\n"
            for chunk_type in chunks:
                chunk_data = chunk_type[4:]
                chunk_name = chunk_type[:4]
                crc = zlib.crc32(chunk_name + chunk_data) & 0xFFFFFFFF
                png += (
                    struct.pack(">I", len(chunk_data))
                    + chunk_name
                    + chunk_data
                    + struct.pack(">I", crc)
                )
            return png

        png_bytes = _make_png()
        upload = UploadPart(filename="test_sample.png", content=png_bytes, content_type="image/png")
        metadata = {
            "job_name": "degraded-test-job",
            "images": [
                {
                    "filename": "test_sample.png",
                    "sample_id": "S001",
                    "material_name": None,
                    "material_formula": None,
                    "experiment_conditions": {},
                    "scale": {"mode": "pixel_only", "value": None},
                }
            ],
        }
        result = client.create_analysis([upload], metadata)
        data = result.data
        job_id = data.get("job", {}).get("job_id", "")

        _record(
            "Upload",
            "upload succeeds (job created)",
            bool(job_id),
            f"job_id={job_id!r}",
        )

        images = data.get("images", [])
        _record(
            "Upload",
            "job has at least one image asset",
            len(images) > 0,
            f"got {len(images)} images",
        )

        if images:
            img = images[0]
            _record(
                "Upload",
                "image has analysis_roi with valid_rect",
                isinstance(img.get("analysis_roi"), dict)
                and "valid_rect" in (img.get("analysis_roi") or {}),
                f"analysis_roi keys={list((img.get('analysis_roi') or {}).keys())}",
            )

        return job_id or None
    except ApiClientError as exc:
        _record(
            "Upload",
            "upload succeeds",
            False,
            f"ApiClientError: {exc.code} (HTTP {exc.status_code}): {exc.message}",
        )
        return None
    except Exception as exc:
        _record("Upload", "upload succeeds", False, str(exc))
        traceback.print_exc()
        return None


def verify_run_submission(client: NanoLoopApiClient, job_id: str) -> None:
    """POST /analyses/{job_id}/runs — verify 503 MODEL_NOT_READY."""
    print("\n=== 4. Run Submission (503 MODEL_NOT_READY) ===")
    if not job_id:
        _skip("Run", "create run", "no job_id from upload")
        return

    try:
        # Attempt to create a run — should get 503
        payload = {
            "image_ids": ["img_test"],
            "model_ids": ["unet-small-balanced-v1"],
            "roi_mode": "full_image",
            "box_revisions": {},
            "inference": {
                "threshold": None,
                "min_area_px": 8,
                "watershed_enabled": False,
                "exclude_border": True,
                "device": "auto",
                "seed": 42,
            },
        }
        client.create_runs(job_id, payload)
        _record("Run", "create_runs raises ApiClientError", False, "no exception raised")
    except ApiClientError as exc:
        # Verify the error details
        _record(
            "Run",
            "error code is MODEL_NOT_READY",
            exc.code == "MODEL_NOT_READY",
            f"got code={exc.code!r}",
        )
        _record(
            "Run",
            "HTTP status is 503",
            exc.status_code == 503,
            f"got status_code={exc.status_code}",
        )
        _record(
            "Run",
            "error is NOT retryable (model won't auto-fix)",
            exc.retryable is False,
            f"got retryable={exc.retryable}",
        )
        _record(
            "Run",
            "error has request_id",
            bool(exc.request_id)
            and (exc.request_id.startswith("stub_") or exc.request_id.startswith("web_")),
            f"request_id={exc.request_id!r}",
        )

        # Verify error details contain model_ids
        _record(
            "Run",
            "error details contain model_ids",
            "model_ids" in exc.details,
            f"details={exc.details}",
        )
    except Exception as exc:
        _record(
            "Run",
            "create_runs raises ApiClientError",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def verify_rag_query(client: NanoLoopApiClient, job_id: str) -> None:
    """POST /analyses/{job_id}/query — verify 503 RAG_INDEX_NOT_READY."""
    print("\n=== 5. RAG Query (503 RAG_INDEX_NOT_READY) ===")
    if not job_id:
        _skip("Query", "RAG query", "no job_id from upload")
        return

    try:
        payload = {
            "question": "这种材料的平均粒径是多少？",
            "query_type": "auto",
        }
        client.query_analysis(job_id, payload)
        _record("Query", "query raises ApiClientError", False, "no exception raised")
    except ApiClientError as exc:
        _record(
            "Query",
            "error code is RAG_INDEX_NOT_READY",
            exc.code == "RAG_INDEX_NOT_READY",
            f"got code={exc.code!r}",
        )
        _record(
            "Query",
            "HTTP status is 503",
            exc.status_code == 503,
            f"got status_code={exc.status_code}",
        )

    except Exception as exc:
        _record(
            "Query",
            "query raises ApiClientError",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def verify_insufficient_evidence(client: NanoLoopApiClient, job_id: str) -> None:
    """POST query with type=analysis_data — verify 200 INSUFFICIENT_EVIDENCE success."""
    print("\n=== 6. Insufficient Evidence (200 success envelope) ===")
    if not job_id:
        _skip("Query", "insufficient evidence", "no job_id from upload")
        return

    try:
        payload = {
            "question": "实验数据中粒径分布如何？",
            "query_type": "analysis_data",
        }
        result = client.query_analysis(job_id, payload)
        data = result.data

        _record(
            "Query",
            "insufficient evidence returns as success (not error)",
            result.status == "success",
            f"status={result.status!r}",
        )
        _record(
            "Query",
            "outcome_code is INSUFFICIENT_EVIDENCE",
            data.get("outcome_code") == "INSUFFICIENT_EVIDENCE",
            f"outcome_code={data.get('outcome_code')!r}",
        )
        _record(
            "Query",
            "confidence is low",
            data.get("confidence") == "low",
            f"confidence={data.get('confidence')!r}",
        )
        _record(
            "Query",
            "has non-empty limitations",
            isinstance(data.get("limitations"), list) and len(data["limitations"]) > 0,
            f"limitations={data.get('limitations')!r}",
        )
        _record(
            "Query",
            "data_evidence is empty (no fake data)",
            data.get("data_evidence") == [],
            f"data_evidence={data.get('data_evidence')!r}",
        )
        _record(
            "Query",
            "citations is empty (no fake references)",
            data.get("citations") == [],
            f"citations={data.get('citations')!r}",
        )
    except ApiClientError as exc:
        _record(
            "Query",
            "insufficient evidence as success",
            False,
            f"got error: {exc.code} (HTTP {exc.status_code}): {exc.message}",
        )
    except Exception as exc:
        _record(
            "Query",
            "insufficient evidence as success",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def verify_knowledge_reindex(client: NanoLoopApiClient) -> None:
    """POST /knowledge/reindex — verify 503 RAG_INDEX_NOT_READY."""
    print("\n=== 7. Knowledge Reindex (503 RAG_INDEX_NOT_READY) ===")
    try:
        client.reindex_knowledge(force=False)
        _record("Knowledge", "reindex raises ApiClientError", False, "no exception raised")
    except ApiClientError as exc:
        _record(
            "Knowledge",
            "error code is RAG_INDEX_NOT_READY",
            exc.code == "RAG_INDEX_NOT_READY",
            f"got code={exc.code!r}",
        )
        _record(
            "Knowledge",
            "HTTP status is 503",
            exc.status_code == 503,
            f"got status_code={exc.status_code}",
        )
    except Exception as exc:
        _record(
            "Knowledge",
            "reindex raises ApiClientError",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def verify_export(client: NanoLoopApiClient, job_id: str) -> None:
    """GET /analyses/{job_id}/export — verify 409 EXPORT_NOT_READY."""
    print("\n=== 8. Export (409 EXPORT_NOT_READY) ===")
    if not job_id:
        _skip("Export", "export", "no job_id from upload")
        return

    try:
        client.export_analysis(job_id)
        _record("Export", "export raises ApiClientError", False, "no exception raised")
    except ApiClientError as exc:
        _record(
            "Export",
            "error code is EXPORT_NOT_READY",
            exc.code == "EXPORT_NOT_READY",
            f"got code={exc.code!r}",
        )
        _record(
            "Export",
            "HTTP status is 409",
            exc.status_code == 409,
            f"got status_code={exc.status_code}",
        )
    except Exception as exc:
        _record(
            "Export",
            "export raises ApiClientError",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def verify_401(base_url: str) -> None:
    """Connect without API key when auth is required — verify 401."""
    print("\n=== 9. Auth 401 (AUTHENTICATION_REQUIRED) ===")
    # Connect WITHOUT an API key
    client = NanoLoopApiClient(base_url, api_key=None)
    try:
        with client:
            client.health()
            _record("Auth401", "health raises ApiClientError", False, "no exception raised")
    except ApiClientError as exc:
        _record(
            "Auth401",
            "HTTP status is 401",
            exc.status_code == 401,
            f"got status_code={exc.status_code}",
        )
        # Backend uses AUTHENTICATION_REQUIRED, not UNAUTHORIZED
        _record(
            "Auth401",
            "error code is AUTHENTICATION_REQUIRED",
            exc.code == "AUTHENTICATION_REQUIRED",
            f"got code={exc.code!r}",
        )
    except Exception as exc:
        _record(
            "Auth401",
            "health raises ApiClientError",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def verify_429(base_url: str, api_key: str | None) -> None:
    """Hit rate limit — verify 429 RATE_LIMITED."""
    print("\n=== 10. Rate Limit 429 (RATE_LIMITED) ===")
    # This verifies the raw 429 contract, so surface the first limited response
    # instead of waiting for the production client's bounded GET retry loop.
    client = NanoLoopApiClient(base_url, api_key=api_key, max_retries=0)
    try:
        with client:
            hit_429 = False
            for i in range(10):
                try:
                    # Use ?rate=test to trigger the stub's rate limiter
                    # _request_json will raise ApiClientError on non-2xx
                    client._request_json(
                        "GET",
                        "/health",
                        params=__import__("httpx").QueryParams("rate=test"),
                    )
                except ApiClientError as exc:
                    if exc.status_code == 429:
                        hit_429 = True
                        _record(
                            "Rate429",
                            "HTTP status is 429",
                            True,
                            f"triggered on request #{i + 1}",
                        )
                        _record(
                            "Rate429",
                            "error code is RATE_LIMITED",
                            exc.code == "RATE_LIMITED",
                            f"got code={exc.code!r}",
                        )
                        _record(
                            "Rate429",
                            "error is retryable",
                            exc.retryable is True,
                            f"got retryable={exc.retryable}",
                        )
                        break
            if not hit_429:
                _record("Rate429", "429 triggered within 10 requests", False, "never hit 429")
    except Exception as exc:
        _record("Rate429", "429 test completed", False, f"unexpected: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def verify_transport_error() -> None:
    """Connect to a non-existent server — verify TRANSPORT_ERROR."""
    print("\n=== 11. Transport Error (TRANSPORT_ERROR) ===")
    transport = httpx.Client(trust_env=False)
    client = NanoLoopApiClient(
        "http://127.0.0.1:59999",
        api_key=None,
        timeout=3.0,
        client=transport,
    )
    try:
        client.health()
        _record("Transport", "health raises ApiClientError", False, "no exception raised")
    except ApiClientError as exc:
        _record(
            "Transport",
            "error code is TRANSPORT_ERROR",
            exc.code == "TRANSPORT_ERROR",
            f"got code={exc.code!r}",
        )
        _record(
            "Transport",
            "status_code is 0 (no HTTP response)",
            exc.status_code == 0,
            f"got status_code={exc.status_code}",
        )
        _record(
            "Transport",
            "error is retryable",
            exc.retryable is True,
            f"got retryable={exc.retryable}",
        )
    except Exception as exc:
        _record(
            "Transport",
            "health raises ApiClientError",
            False,
            f"unexpected: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
    finally:
        transport.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify backend error contracts against real HTTP")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Stub server base URL")
    parser.add_argument("--api-key", default=None, help="API key (for auth mode)")
    parser.add_argument("--test-401", action="store_true", help="Run 401 auth test")
    parser.add_argument("--test-429", action="store_true", help="Run 429 rate limit test")
    args = parser.parse_args()

    print("=" * 70)
    print("NanoLoop Backend Error Contract Verification")
    print(f"Backend: {args.base_url}")
    print("=" * 70)

    # Always run: health, models, upload, run, query, insufficient, knowledge, export
    client = NanoLoopApiClient(args.base_url, api_key=args.api_key)
    with client:
        verify_health(client)
        verify_models(client)
        job_id = verify_upload(client)
        verify_run_submission(client, job_id or "")
        verify_rag_query(client, job_id or "")
        verify_insufficient_evidence(client, job_id or "")
        verify_knowledge_reindex(client)
        verify_export(client, job_id or "")

    # Conditional tests
    if args.test_401:
        verify_401(args.base_url)
    else:
        _skip(
            "Auth401", "401 test", "not requested (use --test-401, requires stub --auth-mode key)"
        )

    if args.test_429:
        verify_429(args.base_url, args.api_key)
    else:
        _skip("Rate429", "429 test", "not requested (use --test-429)")

    # Transport error test (always run — no server needed)
    verify_transport_error()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PASS: {len(_passed)}")
    print(f"  FAIL: {len(_failed)}")
    print(f"  SKIP: {len(_skipped)}")
    print(f"  TOTAL: {len(_passed) + len(_failed) + len(_skipped)}")

    if _failed:
        print("\nFAILED:")
        for f in _failed:
            print(f"  - {f}")

    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
