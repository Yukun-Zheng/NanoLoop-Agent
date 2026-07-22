#!/usr/bin/env python3
"""Run an offline, fixture-backed NanoLoop backend MVP through public HTTP routes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import create_app  # noqa: E402

_MODEL_ID = "unet-deterministic-fixture-v1"
_TERMINAL = {"COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"}


class FixtureSmokeError(RuntimeError):
    """A public MVP step returned an invalid or unsuccessful result."""


def run_fixture_mvp(state_root: Path, *, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Exercise migration, API, DB, scheduler, model bundle, artifacts, query, and export."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    state_root = Path(state_root).expanduser().resolve()
    data_root = state_root / "data"
    output_root = state_root / "outputs"
    knowledge_root = state_root / "knowledge"
    for directory in (data_root, output_root, knowledge_root / "sources", knowledge_root / "index"):
        directory.mkdir(parents=True, exist_ok=True)
    database_path = data_root / "nanoloop.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    settings = Settings(
        app_env="test",
        database_url=database_url,
        output_root=output_root,
        model_registry_path=_PROJECT_ROOT / "demo_data" / "model_artifacts" / "registry.yaml",
        model_snapshot_root=data_root / "model-snapshots",
        knowledge_source_dir=knowledge_root / "sources",
        faiss_index_path=knowledge_root / "index" / "faiss.index",
        analysis_worker_count=1,
        analysis_queue_capacity=1,
        analysis_scheduler_poll_seconds=0.05,
        shutdown_timeout_seconds=10,
        log_level="WARNING",
    )

    with _migration_environment(database_url):
        command.upgrade(Config(str(_PROJECT_ROOT / "alembic.ini")), "head")
        with TestClient(create_app(settings=settings), raise_server_exceptions=False) as client:
            health = _success(client.get("/api/v1/health"), "health")
            if health["database"]["status"] != "healthy":
                raise FixtureSmokeError(f"database health is {health['database']!r}")
            if health["model_registry"]["status"] != "healthy":
                raise FixtureSmokeError(
                    f"fixture model registry health is {health['model_registry']!r}"
                )

            image_bytes = _fixture_image_bytes()
            metadata = {
                "job_name": "NanoLoop engineering fixture MVP",
                "images": [
                    {
                        "filename": "engineering-fixture.png",
                        "sample_id": "fixture-sample-01",
                        "material_name": "engineering fixture",
                        "scale": {"mode": "pixel_only"},
                    }
                ],
            }
            created = _success(
                client.post(
                    "/api/v1/analyses",
                    files={"files": ("engineering-fixture.png", image_bytes, "image/png")},
                    data={"metadata_json": json.dumps(metadata, ensure_ascii=False)},
                ),
                "create_analysis",
                expected_status=201,
            )
            job_id = _required_string(created["job"], "job_id", "create_analysis")
            image_id = _required_string(created["images"][0], "image_id", "create_analysis")

            models = _success(client.get("/api/v1/models"), "list_models")["models"]
            fixture_models = [model for model in models if model.get("model_id") == _MODEL_ID]
            if len(fixture_models) != 1 or fixture_models[0].get("status") != "ready":
                raise FixtureSmokeError("the explicit engineering fixture model is not ready")

            submitted = _success(
                client.post(
                    f"/api/v1/analyses/{job_id}/runs",
                    json={
                        "image_ids": [image_id],
                        "model_ids": [_MODEL_ID],
                        "roi_mode": "full_image",
                        "inference": {
                            "threshold": 0.5,
                            "min_area_px": 8,
                            "watershed_enabled": False,
                            "exclude_border": True,
                            "device": "cpu",
                            "seed": 42,
                        },
                    },
                ),
                "create_run",
                expected_status=202,
            )
            run_ids = submitted["run_ids"]
            if not isinstance(run_ids, list) or len(run_ids) != 1:
                raise FixtureSmokeError("create_run did not return exactly one run_id")
            run_id = _required_string({"run_id": run_ids[0]}, "run_id", "create_run")
            run = _wait_for_run(client, run_id, timeout_seconds=timeout_seconds)
            if run["status"] == "FAILED":
                raise FixtureSmokeError(
                    f"fixture run failed: {run.get('error_code')} {run.get('error_message')}"
                )
            configuration = run.get("configuration")
            if not isinstance(configuration, dict):
                raise FixtureSmokeError("run response has no configuration")
            if configuration.get("schema_version") != 3 or not configuration.get("model_bundle"):
                raise FixtureSmokeError("fixture run did not freeze a schema-v3 model bundle")
            execution = run.get("execution")
            if not isinstance(execution, dict) or not str(execution.get("backend", "")).endswith(
                ".DeterministicFixtureAdapter"
            ):
                raise FixtureSmokeError("run execution did not use the fixture adapter")

            artifacts = run.get("artifacts")
            if not isinstance(artifacts, dict):
                raise FixtureSmokeError("run response has no artifact links")
            for name in ("mask_url", "overlay_url", "instances_url", "particles_csv_url"):
                url = _required_string(artifacts, name, "run_artifacts")
                response = client.get(url)
                if response.status_code != 200 or not response.content:
                    raise FixtureSmokeError(f"artifact {name} is not downloadable")

            query = _success(
                client.post(
                    f"/api/v1/analyses/{job_id}/query",
                    json={
                        "question": "What is the particle count for this run?",
                        "query_type": "analysis_data",
                        "run_ids": [run_id],
                    },
                ),
                "analysis_query",
            )
            if query.get("outcome_code") != "OK":
                raise FixtureSmokeError(f"analysis query outcome is {query.get('outcome_code')!r}")

            export = _success(
                client.get(
                    f"/api/v1/analyses/{job_id}/export",
                    params=[("run_ids", run_id)],
                ),
                "export",
            )
            archive_response = client.get(_required_string(export, "download_url", "export"))
            if archive_response.status_code != 200:
                raise FixtureSmokeError(f"export download returned {archive_response.status_code}")
            archive_bytes = archive_response.content
            observed_sha256 = hashlib.sha256(archive_bytes).hexdigest()
            if observed_sha256 != export.get("sha256"):
                raise FixtureSmokeError("export SHA-256 does not match downloaded bytes")
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("export_manifest.json"))
            required_members = {
                "export_manifest.json",
                "job_summary.json",
                "audit_summary.json",
                "run_summary.csv",
            }
            if not required_members.issubset(names):
                missing = sorted(required_members - names)
                raise FixtureSmokeError(f"export is missing required members: {missing}")

            summary = run.get("summary")
            particle_count = summary.get("particle_count") if isinstance(summary, dict) else None
            return {
                "mode": "engineering_fixture_not_scientific",
                "job_id": job_id,
                "image_id": image_id,
                "run_id": run_id,
                "run_status": run["status"],
                "particle_count": particle_count,
                "configuration_schema_version": configuration["schema_version"],
                "model_bundle_id": configuration["model_bundle"]["bundle_id"],
                "execution_backend": execution["backend"],
                "export_sha256": observed_sha256,
                "export_selection_sha256": manifest["selection_sha256"],
            }


def _wait_for_run(
    client: TestClient,
    run_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        run = _success(client.get(f"/api/v1/runs/{run_id}"), "get_run")
        if run.get("status") in _TERMINAL:
            return run
        time.sleep(0.02)
    raise FixtureSmokeError(f"run {run_id} did not reach a terminal state")


def _fixture_image_bytes() -> bytes:
    height, width = 192, 256
    horizontal = np.linspace(72, 168, width, dtype=np.float32)
    vertical = np.linspace(0, 28, height, dtype=np.float32)[:, None]
    image = np.clip(horizontal[None, :] + vertical, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _success(response: Any, step: str, *, expected_status: int = 200) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise FixtureSmokeError(
            f"{step} returned HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    if payload.get("status") not in {"success", "accepted"}:
        raise FixtureSmokeError(f"{step} returned an invalid response envelope")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise FixtureSmokeError(f"{step} returned no object data")
    return data


def _required_string(value: object, key: str, step: str) -> str:
    if not isinstance(value, dict):
        raise FixtureSmokeError(f"{step} returned an invalid object")
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise FixtureSmokeError(f"{step} returned no {key}")
    return candidate


@contextmanager
def _migration_environment(database_url: str) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in ("APP_ENV", "DATABASE_URL")}
    os.environ["APP_ENV"] = "test"
    os.environ["DATABASE_URL"] = database_url
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline engineering-fixture backend loop. Output is simulated and must "
            "not be used as scientific evidence."
        )
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Optional persistent state directory; otherwise a temporary directory is used.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.state_dir is not None:
        result = run_fixture_mvp(args.state_dir, timeout_seconds=args.timeout)
    else:
        with tempfile.TemporaryDirectory(prefix="nanoloop-fixture-mvp-") as directory:
            result = run_fixture_mvp(Path(directory), timeout_seconds=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
