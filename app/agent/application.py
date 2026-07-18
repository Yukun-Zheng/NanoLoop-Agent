"""Audited unified-query use case over grounded providers and persisted scopes."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.router import QueryRouter
from app.contracts.common import utc_now
from app.contracts.enums import QueryType
from app.contracts.limits import MAX_MATERIAL_ALIASES
from app.contracts.queries import MaterialContext, UnifiedQueryRequest, UnifiedQueryResponse
from app.core.errors import ResourceNotFoundError
from app.core.logging import log_context
from app.db.models import AnalysisJob, ImageAsset, QueryLog, SegmentationRun
from app.storage import LocalFileStore

_MATERIAL_SEPARATORS = re.compile(r"[\s_-]+")
logger = logging.getLogger(__name__)


class UnifiedQueryProtocol(Protocol):
    def answer(self, job_id: str, request: UnifiedQueryRequest) -> UnifiedQueryResponse: ...


class QueryApplicationService:
    """Validate ownership, answer through allowed tools, and persist the evidence trail."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        unified_query: UnifiedQueryProtocol,
        file_store: LocalFileStore,
    ) -> None:
        self.session_factory = session_factory
        self.unified_query = unified_query
        self.file_store = file_store
        self._artifact_lock = threading.Lock()

    def answer(self, job_id: str, request: UnifiedQueryRequest) -> UnifiedQueryResponse:
        resolved_request, clarification = self._validate_scope(job_id, request)
        response = clarification or self.unified_query.answer(job_id, resolved_request)
        query_id = f"query_{uuid4().hex}"
        created_at = utc_now()
        session = self.session_factory()
        try:
            session.add(
                QueryLog(
                    query_id=query_id,
                    job_id=job_id,
                    image_id=resolved_request.image_id,
                    query_type=response.query_type.value,
                    question=resolved_request.question,
                    request_json=resolved_request.model_dump(mode="json"),
                    answer_json=response.model_dump(mode="json"),
                    created_at=created_at,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        # QueryLog is the source of truth.  JSONL/JSON files are rebuildable
        # projections and must stay best-effort after the database commit.
        try:
            self._write_artifacts(
                job_id=job_id,
                query_id=query_id,
                request=resolved_request,
                response=response,
                created_at=created_at.isoformat(),
            )
        except Exception:
            with log_context(job_id=job_id, image_id=resolved_request.image_id):
                logger.exception(
                    "query_audit_projection_failed",
                    extra={
                        "component": "query_audit_projection",
                        "detail": f"query_id={query_id}",
                        "event": "projection_write_failed",
                        "outcome": "degraded",
                    },
                )
        return response

    def _validate_scope(
        self,
        job_id: str,
        request: UnifiedQueryRequest,
    ) -> tuple[UnifiedQueryRequest, UnifiedQueryResponse | None]:
        session = self.session_factory()
        try:
            if session.get(AnalysisJob, job_id) is None:
                raise ResourceNotFoundError(details={"resource": "job", "job_id": job_id})
            image: ImageAsset | None = None
            if request.image_id is not None:
                image = session.get(ImageAsset, request.image_id)
                if image is None or image.job_id != job_id:
                    raise ResourceNotFoundError(
                        details={
                            "resource": "image",
                            "job_id": job_id,
                            "image_id": request.image_id,
                        }
                    )
            run_ids = list(dict.fromkeys(request.run_ids))
            if run_ids:
                rows = session.execute(
                    select(SegmentationRun.run_id, SegmentationRun.job_id).where(
                        SegmentationRun.run_id.in_(run_ids)
                    )
                ).all()
                owned = {run_id for run_id, owner_job_id in rows if owner_job_id == job_id}
                missing = [run_id for run_id in run_ids if run_id not in owned]
                if missing:
                    raise ResourceNotFoundError(
                        details={"resource": "run", "job_id": job_id, "run_ids": missing}
                    )
            if image is not None:
                image_context = _image_material_context(image)
                if _has_material_identity(request.material_context):
                    if (
                        request.material_context is not None
                        and request.material_context.source != "user_confirmation"
                        and image_context is not None
                        and _material_contexts_conflict(
                            request.material_context,
                            image_context,
                        )
                    ):
                        return request, _material_conflict_response(
                            request,
                            image_context,
                        )
                    return request, None
                if image_context is None:
                    return request, None
                return (
                    request.model_copy(update={"material_context": image_context}),
                    None,
                )

            if _has_material_identity(request.material_context):
                return request, None
            if not _query_requires_material_context(request):
                return request, None
            job_materials = _job_material_contexts(session, job_id)
            if len(job_materials) == 1:
                return (
                    request.model_copy(update={"material_context": job_materials[0]}),
                    None,
                )
            if len(job_materials) > 1:
                return request, _multiple_materials_response(request, job_materials)
            if QueryRouter.requires_material_context(request.question):
                return request, _missing_material_response(request)
            return request, None
        finally:
            session.close()

    def _write_artifacts(
        self,
        *,
        job_id: str,
        query_id: str,
        request: UnifiedQueryRequest,
        response: UnifiedQueryResponse,
        created_at: str,
    ) -> None:
        history_path = self.file_store.paths.query_history(job_id)
        citations_path = self.file_store.paths.rag_citations(job_id)
        record = {
            "schema_version": "1.0",
            "query_id": query_id,
            "created_at": created_at,
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        with self._artifact_lock:
            previous = history_path.read_bytes() if history_path.is_file() else b""
            line = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            self.file_store.atomic_write_bytes(history_path, previous + line)

            citation_queries: list[object] = []
            if citations_path.is_file():
                try:
                    existing = json.loads(citations_path.read_text(encoding="utf-8"))
                    candidate = existing.get("queries", []) if isinstance(existing, dict) else []
                    if isinstance(candidate, list):
                        citation_queries = candidate
                except (OSError, UnicodeError, json.JSONDecodeError):
                    citation_queries = []
            citation_queries.append(
                {
                    "query_id": query_id,
                    "created_at": created_at,
                    "citations": [
                        citation.model_dump(mode="json") for citation in response.citations
                    ],
                }
            )
            self.file_store.atomic_write_json(
                citations_path,
                {"job_id": job_id, "queries": citation_queries},
            )


def _has_material_identity(context: MaterialContext | None) -> bool:
    if context is None:
        return False
    return any(
        value.strip()
        for value in (context.formula, context.name, *context.aliases)
        if value is not None
    )


def _nonblank(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _image_material_context(image: ImageAsset) -> MaterialContext | None:
    formula = _nonblank(image.material_formula)
    name = _nonblank(image.material_name)
    if formula is None and name is None:
        return None
    return MaterialContext(
        formula=formula,
        name=name,
        source="image_metadata",
    )


def _query_requires_material_context(request: UnifiedQueryRequest) -> bool:
    if request.query_type in {QueryType.MATERIAL_KNOWLEDGE, QueryType.MIXED}:
        return True
    return request.query_type == QueryType.AUTO and QueryRouter.requires_material_context(
        request.question
    )


def _job_material_contexts(session: Session, job_id: str) -> list[MaterialContext]:
    rows = session.execute(
        select(ImageAsset.material_formula, ImageAsset.material_name)
        .where(ImageAsset.job_id == job_id)
        .order_by(ImageAsset.image_id)
    ).all()
    contexts: list[MaterialContext] = []
    for raw_formula, raw_name in rows:
        formula = _nonblank(raw_formula)
        name = _nonblank(raw_name)
        if formula is None and name is None:
            continue
        candidate = MaterialContext(
            formula=formula,
            name=name,
            source="image_metadata",
        )
        overlapping = [
            index
            for index, context in enumerate(contexts)
            if _material_identity_values(context) & _material_identity_values(candidate)
        ]
        if not overlapping:
            contexts.append(candidate)
            continue
        first = overlapping[0]
        merged = contexts[first]
        for index in overlapping[1:]:
            merged = _merge_material_context(merged, contexts[index])
        merged = _merge_material_context(merged, candidate)
        contexts = [
            context
            for index, context in enumerate(contexts)
            if index not in overlapping[1:]
        ]
        contexts[first] = merged
    return contexts


def _merge_material_context(
    first: MaterialContext,
    second: MaterialContext,
) -> MaterialContext:
    formula = first.formula or second.formula
    name = first.name or second.name
    canonical = {
        normalized
        for value in (formula, name)
        if (normalized := _normalized_material_value(value)) is not None
    }
    aliases: list[str] = []
    seen = set(canonical)
    for value in (
        *first.aliases,
        *second.aliases,
        first.formula,
        first.name,
        second.formula,
        second.name,
    ):
        normalized = _normalized_material_value(value)
        cleaned = _nonblank(value)
        if normalized is None or cleaned is None or normalized in seen:
            continue
        aliases.append(cleaned)
        seen.add(normalized)
    return MaterialContext(
        formula=formula,
        name=name,
        aliases=aliases[:MAX_MATERIAL_ALIASES],
        source="image_metadata",
    )


def _material_contexts_conflict(
    requested: MaterialContext,
    image_context: MaterialContext,
) -> bool:
    requested_formula = _normalized_material_value(requested.formula)
    image_formula = _normalized_material_value(image_context.formula)
    if (
        requested_formula is not None
        and image_formula is not None
        and requested_formula != image_formula
    ):
        return True
    requested_values = _material_identity_values(requested)
    image_values = _material_identity_values(image_context)
    return bool(requested_values and image_values and requested_values.isdisjoint(image_values))


def _material_identity_values(context: MaterialContext) -> set[str]:
    values = (context.formula, context.name, *context.aliases)
    return {
        normalized
        for value in values
        if (normalized := _normalized_material_value(value)) is not None
    }


def _normalized_material_value(value: str | None) -> str | None:
    normalized = _nonblank(value)
    if normalized is None:
        return None
    return _MATERIAL_SEPARATORS.sub("", normalized.casefold())


def _multiple_materials_response(
    request: UnifiedQueryRequest,
    choices: list[MaterialContext],
) -> UnifiedQueryResponse:
    labels = sorted({_material_label(choice) for choice in choices}, key=str.casefold)
    rendered = "、".join(labels)
    return UnifiedQueryResponse(
        query_type=request.query_type,
        answer=f"该任务包含多种材料，请明确选择后重试：{rendered}。",
        confidence="low",
        limitations=(
            "检测到多个材料上下文，未执行知识检索",
            f"可选材料：{rendered}",
        ),
        needs_clarification=True,
        outcome_code="INSUFFICIENT_EVIDENCE",
    )


def _material_conflict_response(
    request: UnifiedQueryRequest,
    image_context: MaterialContext,
) -> UnifiedQueryResponse:
    requested = _material_label(request.material_context)
    image_material = _material_label(image_context)
    return UnifiedQueryResponse(
        query_type=request.query_type,
        answer=(
            "请求材料上下文与所选图像元数据冲突："
            f"请求为 {requested}，图像为 {image_material}。"
            "请确认使用图像材料，或以 source=user_confirmation 明确覆盖。"
        ),
        material_context=request.material_context,
        confidence="low",
        limitations=(
            "材料上下文冲突，未执行知识检索",
            "仅 source=user_confirmation 可覆盖图像元数据",
        ),
        needs_clarification=True,
        outcome_code="INSUFFICIENT_EVIDENCE",
    )


def _missing_material_response(request: UnifiedQueryRequest) -> UnifiedQueryResponse:
    return UnifiedQueryResponse(
        query_type=request.query_type,
        answer=(
            "该问题需要明确材料，但任务内没有可用材料元数据。"
            "请选择带材料元数据的图像，或提供明确的 material_context。"
        ),
        confidence="low",
        limitations=("缺少材料上下文，未执行知识检索",),
        needs_clarification=True,
        outcome_code="INSUFFICIENT_EVIDENCE",
    )


def _material_label(context: MaterialContext | None) -> str:
    if context is None:
        return "未指定材料"
    formula = _nonblank(context.formula)
    name = _nonblank(context.name)
    if formula and name and _normalized_material_value(formula) != _normalized_material_value(name):
        return f"{formula}（{name}）"
    return formula or name or "未指定材料"
