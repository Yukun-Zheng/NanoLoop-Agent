#!/usr/bin/env python3
"""End-to-end NanoLoop scientific smoke test over the public REST API only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from frontend.api_client import (  # noqa: E402
    ApiClientError,
    ApiResult,
    ArtifactDownload,
    JsonObject,
    NanoLoopApiClient,
    UploadPart,
)

_TERMINAL_STATUSES = {"COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()

class FixtureError(ValueError):
    pass


class SmokeTestFailure(RuntimeError):
    def __init__(
        self,
        step: str,
        message: str,
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        suffix = f" request_id={self.request_id}" if self.request_id else ""
        return f"[{self.step}] {self.message}{suffix}"


@dataclass(frozen=True, slots=True)
class ImageFixture:
    path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class KnowledgeFixture:
    path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SmokeFixture:
    job_name: str
    images: tuple[ImageFixture, ...]
    box_image_index: int
    box: dict[str, Any]
    inference: dict[str, Any]
    knowledge: KnowledgeFixture
    analysis_question: str
    material_question: str
    material_context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SmokeReport:
    mode: str
    job_id: str | None = None
    run_ids: tuple[str, ...] = ()
    export_sha256: str | None = None
    manifest: dict[str, Any] | None = None


class SmokeRunner:
    def __init__(
        self,
        client: NanoLoopApiClient,
        fixture: SmokeFixture,
        *,
        poll_timeout: float = 300.0,
        poll_interval: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        output: Callable[[str], None] = print,
    ) -> None:
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.client = client
        self.fixture = fixture
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.sleep = sleep
        self.monotonic = monotonic
        self.output = output

    def run(self, *, allow_degraded: bool = False) -> SmokeReport:
        health = self._api_step("health", self.client.health)
        self._verify_core_health(health)
        if allow_degraded:
            return self._run_degraded(health)
        return self._run_scientific_loop(health)

    def _run_degraded(self, health: ApiResult[JsonObject]) -> SmokeReport:
        model_health = _component_status(
            health.data,
            "model_registry",
            "health",
            request_id=health.request_id,
        )
        rag_health = _component_status(
            health.data,
            "rag_index",
            "health",
            request_id=health.request_id,
        )
        if model_health not in {"degraded", "unavailable"}:
            raise SmokeTestFailure(
                "degraded_verification",
                f"model registry reported {model_health!r}, not degraded/unavailable",
                request_id=health.request_id,
            )
        if rag_health not in {"degraded", "unavailable"}:
            raise SmokeTestFailure(
                "degraded_verification",
                f"RAG index reported {rag_health!r}, not degraded/unavailable",
                request_id=health.request_id,
            )
        try:
            models = self.client.list_models()
        except ApiClientError as error:
            if error.code not in {"NOT_IMPLEMENTED", "SERVICE_UNAVAILABLE"}:
                raise SmokeTestFailure(
                    "list_models_degraded",
                    f"{error.code}: {error.message}",
                    request_id=error.request_id,
                ) from error
            self.output(
                f"[PASS] list_models_degraded unavailable={error.code} "
                f"request_id={error.request_id}"
            )
            model_records: list[dict[str, Any]] = []
            model_request_id = error.request_id
        except ValueError as error:
            raise SmokeTestFailure("list_models_degraded", str(error)) from error
        else:
            self.output(
                f"[PASS] list_models_degraded request_id={models.request_id} "
                f"status={models.status}"
            )
            model_records = _object_list(
                models.data,
                "models",
                "list_models_degraded",
                request_id=models.request_id,
            )
            model_request_id = models.request_id
        ready = [record for record in model_records if record.get("status") == "ready"]
        if ready:
            raise SmokeTestFailure(
                "degraded_verification",
                "ready models exist; --allow-degraded cannot claim model unavailability",
                request_id=model_request_id,
            )
        self.output(
            "[SKIP] DEGRADED MODE VERIFIED: no ready model and RAG is unavailable; "
            "the scientific closed loop was explicitly skipped."
        )
        return SmokeReport(mode="degraded")

    def _run_scientific_loop(self, health: ApiResult[JsonObject]) -> SmokeReport:
        model_health = _component_status(
            health.data,
            "model_registry",
            "health",
            request_id=health.request_id,
        )
        if model_health == "unavailable":
            raise SmokeTestFailure(
                "health",
                "model registry is unavailable; a scientific smoke test cannot proceed",
                request_id=health.request_id,
            )

        created = self._api_step(
            "upload_analysis",
            lambda: self.client.create_analysis(
                [
                    UploadPart(image.path.name, image.path.read_bytes())
                    for image in self.fixture.images
                ],
                {
                    "job_name": self.fixture.job_name,
                    "images": [
                        {"filename": image.path.name, **image.metadata}
                        for image in self.fixture.images
                    ],
                },
            ),
        )
        job = _object_value(
            created.data,
            "job",
            "upload_analysis",
            request_id=created.request_id,
        )
        job_id = _required_string(
            job,
            "job_id",
            "upload_analysis",
            request_id=created.request_id,
        )
        images = _object_list(
            created.data,
            "images",
            "upload_analysis",
            request_id=created.request_id,
        )
        if len(images) != len(self.fixture.images):
            raise SmokeTestFailure(
                "upload_analysis",
                "backend image count does not match the uploaded fixture",
                request_id=created.request_id,
            )
        target_filename = self.fixture.images[self.fixture.box_image_index].path.name
        target_image = next(
            (item for item in images if item.get("filename") == target_filename),
            None,
        )
        if target_image is None:
            raise SmokeTestFailure(
                "upload_analysis",
                f"uploaded image {target_filename!r} is absent from the response",
                request_id=created.request_id,
            )
        image_id = _required_string(
            target_image,
            "image_id",
            "upload_analysis",
            request_id=created.request_id,
        )

        current_boxes = self._api_step(
            "get_boxes",
            lambda: self.client.get_boxes(job_id, image_id),
        )
        revision = _required_int(
            current_boxes.data,
            "revision",
            "get_boxes",
            request_id=current_boxes.request_id,
        )
        saved_boxes = self._api_step(
            "save_box",
            lambda: self.client.replace_boxes(
                job_id,
                image_id,
                expected_revision=revision,
                boxes=[self.fixture.box],
            ),
        )
        saved_revision = _required_int(
            saved_boxes.data,
            "revision",
            "save_box",
            request_id=saved_boxes.request_id,
        )
        if saved_revision != revision + 1:
            raise SmokeTestFailure(
                "save_box",
                f"box revision did not advance exactly once ({revision} -> {saved_revision})",
                request_id=saved_boxes.request_id,
            )

        models = self._api_step(
            "select_ready_model",
            lambda: self.client.list_models(status="ready"),
        )
        model_records = _object_list(
            models.data,
            "models",
            "select_ready_model",
            request_id=models.request_id,
        )
        ready_models = sorted(
            (
                record
                for record in model_records
                if record.get("status") == "ready" and isinstance(record.get("model_id"), str)
            ),
            key=lambda record: cast(str, record["model_id"]),
        )
        if not ready_models:
            raise SmokeTestFailure(
                "select_ready_model",
                "backend reported no ready model; no model was fabricated or substituted",
                request_id=models.request_id,
            )
        model_id = cast(str, ready_models[0]["model_id"])

        created_runs = self._api_step(
            "create_runs",
            lambda: self.client.create_runs(
                job_id,
                {
                    "image_ids": [image_id],
                    "model_ids": [model_id],
                    "roi_mode": "boxes",
                    "box_revisions": {image_id: saved_revision},
                    "inference": self.fixture.inference,
                },
            ),
        )
        run_ids = _string_list(
            created_runs.data,
            "run_ids",
            "create_runs",
            request_id=created_runs.request_id,
        )
        if not run_ids:
            raise SmokeTestFailure(
                "create_runs",
                "backend accepted the request without returning run IDs",
                request_id=created_runs.request_id,
            )
        self._poll_runs(run_ids)

        ingested = self._api_step(
            "ingest_knowledge",
            lambda: self.client.ingest_knowledge_document(
                UploadPart(
                    self.fixture.knowledge.path.name,
                    self.fixture.knowledge.path.read_bytes(),
                ),
                self.fixture.knowledge.metadata,
            ),
        )
        ingested_doc_id = _required_string(
            ingested.data,
            "doc_id",
            "ingest_knowledge",
            request_id=ingested.request_id,
        )
        chunks_created = _required_int(
            ingested.data,
            "chunks_created",
            "ingest_knowledge",
            request_id=ingested.request_id,
        )
        if chunks_created < 1:
            raise SmokeTestFailure(
                "ingest_knowledge",
                "knowledge ingestion created no searchable chunks",
                request_id=ingested.request_id,
            )

        analysis_query = self._api_step(
            "analysis_data_query",
            lambda: self.client.query_analysis(
                job_id,
                {
                    "question": self.fixture.analysis_question,
                    "query_type": "analysis_data",
                    "image_id": image_id,
                    "run_ids": run_ids,
                },
            ),
        )
        self._verify_analysis_query(analysis_query, set(run_ids))

        material_query = self._api_step(
            "material_knowledge_query",
            lambda: self.client.query_analysis(
                job_id,
                {
                    "question": self.fixture.material_question,
                    "query_type": "material_knowledge",
                    "image_id": image_id,
                    "run_ids": run_ids,
                    "material_context": self.fixture.material_context,
                },
            ),
        )
        self._verify_material_query(material_query, ingested_doc_id)

        exported = self._api_step(
            "export_analysis",
            lambda: self.client.export_analysis(job_id, run_ids=run_ids),
        )
        download_url = _required_string(
            exported.data,
            "download_url",
            "export_analysis",
            request_id=exported.request_id,
        )
        expected_sha256 = _required_string(
            exported.data,
            "sha256",
            "export_analysis",
            request_id=exported.request_id,
        )
        if not _SHA256_PATTERN.fullmatch(expected_sha256):
            raise SmokeTestFailure(
                "export_analysis",
                "export response sha256 is malformed",
                request_id=exported.request_id,
            )
        artifact = self._download_step(download_url)
        observed_sha256 = hashlib.sha256(artifact.content).hexdigest()
        if observed_sha256 != expected_sha256:
            raise SmokeTestFailure(
                "validate_export",
                "downloaded ZIP sha256 does not match the export response",
                request_id=artifact.request_id,
            )
        try:
            manifest = validate_export_zip(
                artifact.content,
                expected_job_id=job_id,
                expected_run_ids=set(run_ids),
            )
        except ValueError as error:
            raise SmokeTestFailure(
                "validate_export",
                str(error),
                request_id=artifact.request_id,
            ) from error
        self.output(
            f"[PASS] validate_export request_id={artifact.request_id} "
            f"files={len(cast(list[object], manifest['files']))}"
        )
        self.output(
            f"[PASS] SCIENTIFIC CLOSED LOOP COMPLETE job_id={job_id} "
            f"run_ids={','.join(run_ids)}"
        )
        return SmokeReport(
            mode="full",
            job_id=job_id,
            run_ids=tuple(run_ids),
            export_sha256=expected_sha256,
            manifest=manifest,
        )

    def _poll_runs(self, run_ids: list[str]) -> dict[str, JsonObject]:
        deadline = self.monotonic() + self.poll_timeout
        pending = set(run_ids)
        completed: dict[str, JsonObject] = {}
        last_request_id: str | None = None
        while pending:
            for run_id in sorted(pending):
                result = self._api_step(
                    "poll_runs",
                    partial(self.client.get_run, run_id),
                    announce=False,
                )
                last_request_id = result.request_id
                response_run_id = _required_string(
                    result.data,
                    "run_id",
                    "poll_runs",
                    request_id=result.request_id,
                )
                if response_run_id != run_id:
                    raise SmokeTestFailure(
                        "poll_runs",
                        f"requested {run_id!r} but backend returned {response_run_id!r}",
                        request_id=result.request_id,
                    )
                status = _required_string(
                    result.data,
                    "status",
                    "poll_runs",
                    request_id=result.request_id,
                )
                if status not in _TERMINAL_STATUSES:
                    continue
                if status == "FAILED":
                    error_code = result.data.get("error_code")
                    raise SmokeTestFailure(
                        "poll_runs",
                        f"run {run_id} failed with error_code={error_code!r}",
                        request_id=result.request_id,
                    )
                if not isinstance(result.data.get("summary"), dict):
                    raise SmokeTestFailure(
                        "poll_runs",
                        f"completed run {run_id} has no persisted scientific summary",
                        request_id=result.request_id,
                    )
                completed[run_id] = result.data
                pending.remove(run_id)
                self.output(
                    f"[PASS] poll_runs run_id={run_id} status={status} "
                    f"request_id={result.request_id}"
                )
            if not pending:
                break
            now = self.monotonic()
            if now >= deadline:
                raise SmokeTestFailure(
                    "poll_runs",
                    f"timed out waiting for runs: {','.join(sorted(pending))}",
                    request_id=last_request_id,
                )
            self.sleep(min(self.poll_interval, max(0.0, deadline - now)))
        return completed

    @staticmethod
    def _verify_core_health(result: ApiResult[JsonObject]) -> None:
        service = _component_status(
            result.data,
            "service",
            "health",
            request_id=result.request_id,
        )
        database = _component_status(
            result.data,
            "database",
            "health",
            request_id=result.request_id,
        )
        if service != "healthy":
            raise SmokeTestFailure(
                "health",
                f"service health is {service!r}",
                request_id=result.request_id,
            )
        if database != "healthy":
            raise SmokeTestFailure(
                "health",
                f"database health is {database!r}; migrations/connectivity are required",
                request_id=result.request_id,
            )

    @staticmethod
    def _verify_analysis_query(
        result: ApiResult[JsonObject],
        expected_run_ids: set[str],
    ) -> None:
        if result.data.get("outcome_code") != "OK":
            raise SmokeTestFailure(
                "analysis_data_query",
                "analysis-data query returned insufficient evidence",
                request_id=result.request_id,
            )
        evidence = _object_list(
            result.data,
            "data_evidence",
            "analysis_data_query",
            request_id=result.request_id,
        )
        source_ids: set[str] = set()
        for item in evidence:
            item_source_ids = item.get("source_run_ids")
            if not isinstance(item_source_ids, list) or any(
                not isinstance(source_id, str) or not source_id
                for source_id in item_source_ids
            ):
                raise SmokeTestFailure(
                    "analysis_data_query",
                    "analysis evidence has malformed source_run_ids",
                    request_id=result.request_id,
                )
            source_ids.update(cast(list[str], item_source_ids))
        if not evidence or not expected_run_ids.issubset(source_ids):
            raise SmokeTestFailure(
                "analysis_data_query",
                "analysis evidence does not cite every completed run",
                request_id=result.request_id,
            )

    @staticmethod
    def _verify_material_query(
        result: ApiResult[JsonObject],
        ingested_doc_id: str,
    ) -> None:
        if result.data.get("outcome_code") != "OK":
            raise SmokeTestFailure(
                "material_knowledge_query",
                "material-knowledge query returned insufficient evidence",
                request_id=result.request_id,
            )
        citations = _object_list(
            result.data,
            "citations",
            "material_knowledge_query",
            request_id=result.request_id,
        )
        if not citations or any(
            not isinstance(citation.get("doc_id"), str)
            or not citation.get("doc_id")
            or not isinstance(citation.get("chunk_id"), str)
            or not citation.get("chunk_id")
            for citation in citations
        ):
            raise SmokeTestFailure(
                "material_knowledge_query",
                "material answer has no valid persisted citations",
                request_id=result.request_id,
            )
        cited_doc_ids = {cast(str, citation["doc_id"]) for citation in citations}
        if ingested_doc_id not in cited_doc_ids:
            raise SmokeTestFailure(
                "material_knowledge_query",
                "material answer does not cite the document ingested by this smoke test",
                request_id=result.request_id,
            )

    def _api_step(
        self,
        step: str,
        action: Callable[[], ApiResult[JsonObject]],
        *,
        announce: bool = True,
    ) -> ApiResult[JsonObject]:
        try:
            result = action()
        except ApiClientError as error:
            raise SmokeTestFailure(
                step,
                f"{error.code}: {error.message}",
                request_id=error.request_id,
            ) from error
        except (OSError, ValueError) as error:
            raise SmokeTestFailure(step, str(error)) from error
        if announce:
            self.output(
                f"[PASS] {step} request_id={result.request_id} status={result.status}"
            )
        return result

    def _download_step(self, download_url: str) -> ArtifactDownload:
        try:
            artifact = self.client.download_artifact(download_url)
        except ApiClientError as error:
            raise SmokeTestFailure(
                "download_export",
                f"{error.code}: {error.message}",
                request_id=error.request_id,
            ) from error
        except ValueError as error:
            raise SmokeTestFailure("download_export", str(error)) from error
        self.output(
            f"[PASS] download_export request_id={artifact.request_id} "
            f"bytes={len(artifact.content)}"
        )
        return artifact


def load_fixture(path: Path, *, require_files: bool = True) -> SmokeFixture:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FixtureError(f"cannot read fixture: {path}") from error
    except json.JSONDecodeError as error:
        raise FixtureError(f"fixture is not valid JSON: {error}") from error
    root = _mapping(raw, "fixture")
    if root.get("schema_version") != "1.0":
        raise FixtureError("fixture.schema_version must be '1.0'")
    job_name = _fixture_string(root, "job_name")

    image_records = _fixture_list(root, "images")
    if not 1 <= len(image_records) <= 2:
        raise FixtureError("fixture.images must contain one or two real SEM images")
    images: list[ImageFixture] = []
    filenames: set[str] = set()
    for index, value in enumerate(image_records):
        record = _mapping(value, f"fixture.images[{index}]")
        image_path = _fixture_path(path, _fixture_string(record, "path"))
        if image_path.name in filenames:
            raise FixtureError("fixture image basenames must be unique")
        filenames.add(image_path.name)
        if require_files:
            _require_regular_file(image_path, f"fixture.images[{index}].path")
        metadata = {
            key: value
            for key, value in record.items()
            if key in {
                "sample_id",
                "material_name",
                "material_formula",
                "experiment_conditions",
                "scale",
            }
        }
        if not isinstance(metadata.get("sample_id"), str):
            raise FixtureError(f"fixture.images[{index}].sample_id must be a string")
        if not isinstance(metadata.get("scale"), dict):
            raise FixtureError(f"fixture.images[{index}].scale must be an object")
        images.append(ImageFixture(path=image_path, metadata=metadata))

    box_record = _mapping(root.get("box"), "fixture.box")
    image_index = _fixture_int(box_record, "image_index")
    if not 0 <= image_index < len(images):
        raise FixtureError("fixture.box.image_index is outside fixture.images")
    coordinates = {
        key: _fixture_int(box_record, key) for key in ("x1", "y1", "x2", "y2")
    }
    if coordinates["x1"] >= coordinates["x2"] or coordinates["y1"] >= coordinates["y2"]:
        raise FixtureError("fixture.box must satisfy x1 < x2 and y1 < y2")
    box: dict[str, Any] = {
        **coordinates,
        "label": box_record.get("label", "smoke ROI"),
        "active": box_record.get("active", True),
    }

    inference_value = root.get("inference", {})
    inference = dict(_mapping(inference_value, "fixture.inference"))
    queries = _mapping(root.get("queries"), "fixture.queries")
    analysis_question = _fixture_string(queries, "analysis_data")
    material_question = _fixture_string(queries, "material_knowledge")
    material_context = dict(
        _mapping(queries.get("material_context"), "fixture.queries.material_context")
    )

    knowledge_record = _mapping(
        root.get("knowledge_document"),
        "fixture.knowledge_document",
    )
    knowledge_path = _fixture_path(
        path,
        _fixture_string(knowledge_record, "path"),
    )
    if require_files:
        _require_regular_file(knowledge_path, "fixture.knowledge_document.path")
    knowledge_metadata = dict(
        _mapping(
            knowledge_record.get("metadata"),
            "fixture.knowledge_document.metadata",
        )
    )
    for key in ("title", "source_type", "citation_text", "license_note"):
        _fixture_string(knowledge_metadata, key)
    if knowledge_metadata.get("allowed_for_demo") is not True:
        raise FixtureError(
            "fixture.knowledge_document.metadata.allowed_for_demo must be true"
        )
    return SmokeFixture(
        job_name=job_name,
        images=tuple(images),
        box_image_index=image_index,
        box=box,
        inference=inference,
        knowledge=KnowledgeFixture(path=knowledge_path, metadata=knowledge_metadata),
        analysis_question=analysis_question,
        material_question=material_question,
        material_context=material_context,
    )


def validate_export_zip(
    content: bytes,
    *,
    expected_job_id: str,
    expected_run_ids: set[str],
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("export ZIP contains duplicate member names")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("export ZIP contains an unsafe member path")
            if "export_manifest.json" not in names:
                raise ValueError("export ZIP has no export_manifest.json")
            try:
                manifest_raw = json.loads(archive.read("export_manifest.json"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("export manifest is not valid UTF-8 JSON") from error
            manifest = dict(_mapping(manifest_raw, "export_manifest"))
            if manifest.get("schema_version") != "1.0":
                raise ValueError("export manifest schema_version is not 1.0")
            if manifest.get("job_id") != expected_job_id:
                raise ValueError("export manifest job_id does not match the created job")
            records = manifest.get("files")
            if not isinstance(records, list) or not records:
                raise ValueError("export manifest has no file records")
            recorded_paths: set[str] = set()
            validated_records: list[dict[str, Any]] = []
            for index, value in enumerate(records):
                record = _mapping(value, f"export_manifest.files[{index}]")
                member = _fixture_string(record, "path")
                sha256 = _fixture_string(record, "sha256")
                size_bytes = _fixture_int(record, "size_bytes")
                if member in recorded_paths or member == "export_manifest.json":
                    raise ValueError("export manifest contains a duplicate/generated member")
                recorded_paths.add(member)
                if member not in names:
                    raise ValueError(f"manifest member is absent from ZIP: {member}")
                if not _SHA256_PATTERN.fullmatch(sha256):
                    raise ValueError(f"manifest sha256 is malformed for {member}")
                member_bytes = archive.read(member)
                if len(member_bytes) != size_bytes:
                    raise ValueError(f"manifest size does not match ZIP member: {member}")
                if hashlib.sha256(member_bytes).hexdigest() != sha256:
                    raise ValueError(f"manifest hash does not match ZIP member: {member}")
                validated_records.append(dict(record))
            selection_sha256 = manifest.get("selection_sha256")
            if selection_sha256 is None:
                if not isinstance(manifest.get("generated_at"), str):
                    raise ValueError(
                        "export manifest has neither selection_sha256 nor generated_at"
                    )
            else:
                if not isinstance(selection_sha256, str) or not _SHA256_PATTERN.fullmatch(
                    selection_sha256
                ):
                    raise ValueError("export manifest selection_sha256 is malformed")
                canonical_records = json.dumps(
                    validated_records,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                if hashlib.sha256(canonical_records).hexdigest() != selection_sha256:
                    raise ValueError("export manifest selection_sha256 does not match files")
            archive_payload_paths = set(names) - {"export_manifest.json"}
            if recorded_paths != archive_payload_paths:
                unrecorded = sorted(archive_payload_paths - recorded_paths)
                raise ValueError(
                    "export ZIP contains members absent from the manifest: "
                    + ",".join(unrecorded)
                )
            for run_id in expected_run_ids:
                marker = f"/runs/{run_id}/"
                if not any(marker in f"/{member}" for member in recorded_paths):
                    raise ValueError(f"export manifest has no artifacts for run {run_id}")
            return manifest
    except zipfile.BadZipFile as error:
        raise ValueError("downloaded export is not a valid ZIP archive") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NanoLoop REST scientific closed-loop smoke test.",
    )
    parser.add_argument("--base-url", required=True, help="Backend origin, e.g. http://localhost:8000")
    parser.add_argument("--fixture", required=True, type=Path, help="Path to a smoke fixture JSON")
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for all runs (default: 300)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between run polls (default: 2)",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "Verify truthful model/RAG unavailability and exit after explicitly "
            "skipping the scientific loop"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fixture = load_fixture(
            args.fixture,
            require_files=not args.allow_degraded,
        )
        with NanoLoopApiClient(args.base_url) as client:
            runner = SmokeRunner(
                client,
                fixture,
                poll_timeout=args.poll_timeout,
                poll_interval=args.poll_interval,
            )
            runner.run(allow_degraded=args.allow_degraded)
        return 0
    except (FixtureError, SmokeTestFailure, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[FAIL] smoke test interrupted", file=sys.stderr)
        return 130


def _component_status(
    data: JsonObject,
    name: str,
    step: str,
    *,
    request_id: str | None = None,
) -> str:
    component = _object_value(data, name, step, request_id=request_id)
    return _required_string(component, "status", step, request_id=request_id)


def _object_value(
    data: Mapping[str, object],
    key: str,
    step: str,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    value = data.get(key, _MISSING)
    if not isinstance(value, dict):
        raise SmokeTestFailure(
            step,
            f"response field {key!r} must be an object",
            request_id=request_id,
        )
    return {str(item_key): item for item_key, item in value.items()}


def _object_list(
    data: Mapping[str, object],
    key: str,
    step: str,
    *,
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    value = data.get(key, _MISSING)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SmokeTestFailure(
            step,
            f"response field {key!r} must be an object list",
            request_id=request_id,
        )
    return [dict(item) for item in value]


def _required_string(
    data: Mapping[str, object],
    key: str,
    step: str,
    *,
    request_id: str | None = None,
) -> str:
    value = data.get(key, _MISSING)
    if not isinstance(value, str) or not value:
        raise SmokeTestFailure(
            step,
            f"response field {key!r} must be a non-empty string",
            request_id=request_id,
        )
    return value


def _required_int(
    data: Mapping[str, object],
    key: str,
    step: str,
    *,
    request_id: str | None = None,
) -> int:
    value = data.get(key, _MISSING)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SmokeTestFailure(
            step,
            f"response field {key!r} must be an integer",
            request_id=request_id,
        )
    return value


def _string_list(
    data: Mapping[str, object],
    key: str,
    step: str,
    *,
    request_id: str | None = None,
) -> list[str]:
    value = data.get(key, _MISSING)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SmokeTestFailure(
            step,
            f"response field {key!r} must be a string list",
            request_id=request_id,
        )
    return list(dict.fromkeys(cast(list[str], value)))


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise FixtureError(f"{location} must be an object")
    return {str(key): item for key, item in value.items()}


def _fixture_list(data: Mapping[str, object], key: str) -> list[object]:
    value = data.get(key, _MISSING)
    if not isinstance(value, list):
        raise FixtureError(f"fixture.{key} must be an array")
    return value


def _fixture_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key, _MISSING)
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"{key} must be a non-empty string")
    return value


def _fixture_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key, _MISSING)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureError(f"{key} must be an integer")
    return value


def _fixture_path(fixture_path: Path, configured: str) -> Path:
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = fixture_path.parent / candidate
    return candidate.resolve(strict=False)


def _require_regular_file(path: Path, location: str) -> None:
    if not path.is_file():
        raise FixtureError(f"{location} does not reference a file: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
