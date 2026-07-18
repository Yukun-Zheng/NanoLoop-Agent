from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from app.agent.application import QueryApplicationService
from app.contracts.enums import JobStatus, QueryType
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.contracts.queries import (
    Citation,
    MaterialContext,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from app.core.config import Settings
from app.core.errors import ResourceNotFoundError, StorageError
from app.db.base import Base
from app.db.models import AnalysisJob, ImageAsset, QueryLog
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.storage import LocalFileStore, StoragePaths


class FakeUnifiedQuery:
    def __init__(self) -> None:
        self.requests: list[UnifiedQueryRequest] = []

    def answer(self, job_id: str, request: UnifiedQueryRequest) -> UnifiedQueryResponse:
        assert job_id == "job_1"
        self.requests.append(request)
        return UnifiedQueryResponse(
            query_type=QueryType.MATERIAL_KNOWLEDGE,
            answer=f"grounded: {request.question} [C1]",
            citations=[
                Citation(
                    citation_id="C1",
                    doc_id="doc_1",
                    title="fixture",
                    page=1,
                    chunk_id="chunk_1",
                    excerpt="evidence",
                    retrieval_score=1.0,
                )
            ],
            material_context=request.material_context,
            confidence="medium",
        )


def _service(
    tmp_path: Path,
) -> tuple[QueryApplicationService, Database, LocalFileStore, FakeUnifiedQuery]:
    database = Database(
        Settings(
            app_env="test",
            database_url=f"sqlite:///{tmp_path / 'query.db'}",
            output_root=tmp_path / "outputs",
        )
    )
    Base.metadata.create_all(database.engine)
    with database.session() as session:
        session.add(
            AnalysisJob(
                job_id="job_1",
                tenant_id=LEGACY_TENANT_ID,
                owner_principal_id=LEGACY_PRINCIPAL_ID,
                name="query fixture",
                status=JobStatus.READY_FOR_CONFIGURATION.value,
                config_json={},
            )
        )
        session.add(
            ImageAsset(
                image_id="image_1",
                job_id="job_1",
                filename="sample.tif",
                storage_path="jobs/job_1/images/image_1/original.tif",
                sha256="a" * 64,
                width=64,
                height=64,
                bit_depth=8,
                sample_id="sample_1",
                material_name="Strontium nickelate",
                material_formula=" SrNiO3-x ",
                experiment_conditions_json={},
                analysis_roi_json={},
                scale_nm_per_pixel=1.0,
                box_revision=0,
            )
        )
    store = LocalFileStore(
        StoragePaths(tmp_path / "outputs"),
        max_upload_bytes=1024,
        token_secret=b"q" * 32,
    )
    unified_query = FakeUnifiedQuery()
    service = QueryApplicationService(
        session_factory=database.session_factory,
        unified_query=unified_query,
        file_store=store,
    )
    return service, database, store, unified_query


def _add_material_image(
    database: Database,
    *,
    image_id: str,
    material_formula: str,
    material_name: str,
    sha_character: str,
) -> None:
    with database.session() as session:
        session.add(
            ImageAsset(
                image_id=image_id,
                job_id="job_1",
                filename=f"{image_id}.tif",
                storage_path=f"jobs/job_1/images/{image_id}/original.tif",
                sha256=sha_character * 64,
                width=64,
                height=64,
                bit_depth=8,
                sample_id=image_id,
                material_name=material_name,
                material_formula=material_formula,
                experiment_conditions_json={},
                analysis_roi_json={},
                scale_nm_per_pixel=1.0,
                box_revision=0,
            )
        )


def test_query_is_persisted_and_written_to_audit_artifacts(tmp_path: Path) -> None:
    service, database, store, _unified_query = _service(tmp_path)
    try:
        for question in ("first", "second"):
            response = service.answer(
                "job_1",
                UnifiedQueryRequest(
                    question=question,
                    query_type=QueryType.MATERIAL_KNOWLEDGE,
                ),
            )
            assert response.citations[0].doc_id == "doc_1"

        with database.session() as session:
            logs = session.scalars(select(QueryLog).order_by(QueryLog.created_at)).all()
        assert [log.question for log in logs] == ["first", "second"]
        history = store.paths.query_history("job_1").read_text(encoding="utf-8").splitlines()
        assert len(history) == 2
        assert json.loads(history[0])["response"]["citations"][0]["doc_id"] == "doc_1"
        citations = json.loads(store.paths.rag_citations("job_1").read_text(encoding="utf-8"))
        assert len(citations["queries"]) == 2
        assert citations["schema_version"] == "1.0"
    finally:
        database.dispose()


@pytest.mark.parametrize("failing_method", ["atomic_write_bytes", "atomic_write_json"])
def test_committed_query_survives_audit_projection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing_method: str,
) -> None:
    service, database, store, _unified_query = _service(tmp_path)

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise StorageError("simulated projection outage")

    monkeypatch.setattr(store, failing_method, fail_projection)
    try:
        with caplog.at_level("WARNING", logger="app.agent.application"):
            response = service.answer(
                "job_1",
                UnifiedQueryRequest(
                    question="committed despite projection outage",
                    query_type=QueryType.MATERIAL_KNOWLEDGE,
                ),
            )

        assert response.answer.startswith("grounded:")
        with database.session() as session:
            persisted = session.scalar(
                select(QueryLog).where(
                    QueryLog.question == "committed despite projection outage"
                )
            )
        assert persisted is not None
        with database.session() as session:
            audit_records = SqlAlchemyRepositorySet(session).queries.list_by_job("job_1")
        assert audit_records[0].request.question == "committed despite projection outage"
        assert audit_records[0].response == response
        warning = next(
            record
            for record in caplog.records
            if record.getMessage() == "query_audit_projection_failed"
        )
        assert warning.event == "projection_write_failed"
        assert warning.component == "query_audit_projection"
        assert warning.outcome == "degraded"
        assert str(warning.detail).startswith("query_id=query_")
    finally:
        database.dispose()


def test_query_rejects_unknown_job_before_provider_call(tmp_path: Path) -> None:
    service, database, _store, _unified_query = _service(tmp_path)
    try:
        with pytest.raises(ResourceNotFoundError):
            service.answer(
                "missing",
                UnifiedQueryRequest(question="question", query_type=QueryType.ANALYSIS_DATA),
            )
    finally:
        database.dispose()


def test_query_resolves_image_material_metadata_and_preserves_explicit_context(
    tmp_path: Path,
) -> None:
    service, database, store, unified_query = _service(tmp_path)
    try:
        response = service.answer(
            "job_1",
            UnifiedQueryRequest(
                question="该材料有什么用途？",
                query_type=QueryType.MATERIAL_KNOWLEDGE,
                image_id="image_1",
            ),
        )

        resolved = unified_query.requests[-1].material_context
        assert resolved == MaterialContext(
            formula="SrNiO3-x",
            name="Strontium nickelate",
            source="image_metadata",
        )
        assert response.material_context == resolved
        history = store.paths.query_history("job_1").read_text(encoding="utf-8").splitlines()
        assert json.loads(history[-1])["request"]["material_context"] == {
            "formula": "SrNiO3-x",
            "name": "Strontium nickelate",
            "aliases": [],
            "source": "image_metadata",
        }

        call_count = len(unified_query.requests)
        conflict = service.answer(
            "job_1",
            UnifiedQueryRequest(
                question="该材料有什么性质？",
                query_type=QueryType.MATERIAL_KNOWLEDGE,
                image_id="image_1",
                material_context=MaterialContext(formula="YCu", source="request"),
            ),
        )
        assert conflict.needs_clarification
        assert conflict.outcome_code == "INSUFFICIENT_EVIDENCE"
        assert "source=user_confirmation" in conflict.answer
        assert len(unified_query.requests) == call_count

        explicit = MaterialContext(
            formula="confirmed-formula",
            source="user_confirmation",
        )
        service.answer(
            "job_1",
            UnifiedQueryRequest(
                question="该材料有什么性质？",
                query_type=QueryType.MATERIAL_KNOWLEDGE,
                image_id="image_1",
                material_context=explicit,
            ),
        )
        assert unified_query.requests[-1].material_context == explicit
    finally:
        database.dispose()


def test_query_without_image_uses_the_jobs_unique_material(tmp_path: Path) -> None:
    service, database, _store, unified_query = _service(tmp_path)
    try:
        _add_material_image(
            database,
            image_id="image_3",
            material_formula="Sr-NiO3_x",
            material_name="Perovskite nickelate",
            sha_character="c",
        )
        response = service.answer(
            "job_1",
            UnifiedQueryRequest(
                question="这种材料有什么用途？",
                query_type=QueryType.MATERIAL_KNOWLEDGE,
            ),
        )

        expected = MaterialContext(
            formula="SrNiO3-x",
            name="Strontium nickelate",
            aliases=["Perovskite nickelate"],
            source="image_metadata",
        )
        assert unified_query.requests[-1].image_id is None
        assert unified_query.requests[-1].material_context == expected
        assert response.material_context == expected
    finally:
        database.dispose()


def test_query_without_image_clarifies_multiple_job_materials_before_retrieval(
    tmp_path: Path,
) -> None:
    service, database, _store, unified_query = _service(tmp_path)
    try:
        _add_material_image(
            database,
            image_id="image_2",
            material_formula="YCu",
            material_name="Yttrium copper",
            sha_character="b",
        )

        explicit = service.answer(
            "job_1",
            UnifiedQueryRequest(
                question="材料有什么用途？",
                query_type=QueryType.MATERIAL_KNOWLEDGE,
            ),
        )
        contextual_auto = service.answer(
            "job_1",
            UnifiedQueryRequest(question="这个材料怎么样？"),
        )

        assert not unified_query.requests
        for response in (explicit, contextual_auto):
            assert response.needs_clarification
            assert response.outcome_code == "INSUFFICIENT_EVIDENCE"
            assert not response.citations
            assert "SrNiO3-x" in response.answer
            assert "YCu" in response.answer
            assert any("可选材料" in limitation for limitation in response.limitations)
    finally:
        database.dispose()
