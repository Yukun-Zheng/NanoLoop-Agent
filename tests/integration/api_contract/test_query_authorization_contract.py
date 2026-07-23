from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

from app.agent.application import QueryApplicationService
from app.agent.router import QueryRouter
from app.agent.unified_query import (
    DataQuery,
    DataQueryResult,
    DataToolService,
    UnifiedQueryService,
)
from app.contracts.queries import MaterialContext
from app.core.config import Settings
from app.core.identity import IssuedCredential
from app.db.base import Base
from app.db.models import QueryLog
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.main import create_app
from app.rag.service import KnowledgeAnswer, KnowledgeService
from app.storage import LocalFileStore, StoragePaths
from tests.integration.api_contract.test_analysis_authorization_contract import (
    _ADMIN_A,
    _OWNER_A,
    _PEER_A,
    _PEPPER,
    _TENANT_A,
    _VIEWER_A,
    _Gateway,
    _seed_analyses,
    _seed_identities,
)


class _DataToolSpy:
    def __init__(self) -> None:
        self.queries: list[DataQuery] = []

    def answer(self, query: DataQuery) -> DataQueryResult:
        self.queries.append(query)
        return DataQueryResult(
            answer="tenant-scoped data result",
            confidence="high",
        )


class _KnowledgeBoundarySpy:
    """Stand in for the whole global retrieval/provider pipeline."""

    def __init__(self) -> None:
        self.fts_calls = 0
        self.vector_calls = 0
        self.provider_calls = 0

    def answer(
        self,
        _question: str,
        *,
        material_context: MaterialContext | None = None,
    ) -> KnowledgeAnswer:
        self.fts_calls += 1
        self.vector_calls += 1
        self.provider_calls += 1
        return KnowledgeAnswer(
            answer="unexpected global knowledge result",
            citations=(),
            confidence="low",
            limitations=("knowledge boundary should not have been reached",),
            outcome_code="INSUFFICIENT_EVIDENCE",
            material_context=material_context,
        )


@dataclass(slots=True)
class QueryAuthorizationHarness:
    app: FastAPI
    client: TestClient
    database: Database
    file_store: LocalFileStore
    credentials: dict[str, IssuedCredential]
    data_tools: _DataToolSpy
    knowledge: _KnowledgeBoundarySpy


@pytest.fixture
def query_authorization_harness(tmp_path: Path) -> Iterator[QueryAuthorizationHarness]:
    settings = Settings(
        app_env="test",
        auth_mode="principal",
        credential_pepper=_PEPPER,
        database_url=f"sqlite:///{tmp_path / 'query-authorization.db'}",
        output_root=tmp_path / "outputs",
        knowledge_source_dir=tmp_path / "knowledge-sources",
        model_registry_path=tmp_path / "registry.yaml",
        faiss_index_path=tmp_path / "faiss.index",
        log_level="WARNING",
        api_rate_limit_requests=0,
        api_principal_preauth_rate_limit_requests=1000,
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    file_store = LocalFileStore(
        StoragePaths(settings.output_root),
        max_upload_bytes=1024 * 1024,
        token_secret=b"q" * 32,
    )
    credentials = _seed_identities(database)
    _seed_analyses(database, file_store)
    data_tools = _DataToolSpy()
    knowledge = _KnowledgeBoundarySpy()
    unified = UnifiedQueryService(
        router=QueryRouter(),
        knowledge_service=cast(KnowledgeService, knowledge),
        data_tools=cast(DataToolService, data_tools),
    )
    query_service = QueryApplicationService(
        session_factory=database.session_factory,
        unified_query=unified,
        file_store=file_store,
    )
    app = create_app(
        settings=settings,
        database=database,
        file_store=file_store,
        inference_gateway=_Gateway(),
        query_application_service=query_service,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield QueryAuthorizationHarness(
            app=app,
            client=client,
            database=database,
            file_store=file_store,
            credentials=credentials,
            data_tools=data_tools,
            knowledge=knowledge,
        )
    database.dispose()


def _headers(harness: QueryAuthorizationHarness, actor: str) -> dict[str, str]:
    return {"X-API-Key": harness.credentials[actor].token.get_secret_value()}


def _query_count(harness: QueryAuthorizationHarness) -> int:
    with harness.database.session() as session:
        return int(session.scalar(select(func.count()).select_from(QueryLog)) or 0)


def _projection_snapshot(
    harness: QueryAuthorizationHarness,
    *job_ids: str,
) -> dict[Path, bytes | None]:
    paths = {
        path
        for job_id in job_ids
        for path in (
            harness.file_store.paths.query_history(job_id),
            harness.file_store.paths.rag_citations(job_id),
        )
    }
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def _assert_no_provider_calls(harness: QueryAuthorizationHarness) -> None:
    assert harness.data_tools.queries == []
    assert harness.knowledge.fts_calls == 0
    assert harness.knowledge.vector_calls == 0
    assert harness.knowledge.provider_calls == 0


def test_same_tenant_roles_can_query_and_freeze_the_exact_actor(
    query_authorization_harness: QueryAuthorizationHarness,
) -> None:
    harness = query_authorization_harness
    expected = {
        "owner": (_OWNER_A, "analyst"),
        "peer": (_PEER_A, "analyst"),
        "viewer": (_VIEWER_A, "viewer"),
        "admin": (_ADMIN_A, "tenant_admin"),
    }

    for actor in expected:
        response = harness.client.post(
            "/api/v1/analyses/job_a/query",
            headers=_headers(harness, actor),
            json={
                "question": f"actor audit {actor}",
                "query_type": "analysis_data",
                "run_ids": ["run_a"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["answer"] == "tenant-scoped data result"

    assert [query.tenant_id for query in harness.data_tools.queries] == [_TENANT_A] * 4
    with harness.database.session() as session:
        records = SqlAlchemyRepositorySet(session).queries.list_by_job_scoped(
            "job_a",
            tenant_id=_TENANT_A,
        )
    by_question = {record.request.question: record.actor for record in records}
    assert set(by_question) == {f"actor audit {actor}" for actor in expected}
    for actor, (principal_id, role) in expected.items():
        frozen = by_question[f"actor audit {actor}"].model_dump(mode="json")
        assert frozen == {
            "tenant_id": _TENANT_A,
            "principal_id": principal_id,
            "credential_id": harness.credentials[actor].credential_id,
            "role": role,
            "auth_mode": "principal",
        }

    history_path = harness.file_store.paths.query_history("job_a")
    projected = [json.loads(line) for line in history_path.read_text().splitlines()]
    projected_by_question = {
        record["request"]["question"]: record["actor"] for record in projected
    }
    assert projected_by_question == {
        question: actor.model_dump(mode="json") for question, actor in by_question.items()
    }


def test_foreign_and_missing_jobs_share_404_without_query_side_effects(
    query_authorization_harness: QueryAuthorizationHarness,
) -> None:
    harness = query_authorization_harness
    before_count = _query_count(harness)
    before_projection = _projection_snapshot(harness, "job_b", "job_missing")

    responses = [
        harness.client.post(
            f"/api/v1/analyses/{job_id}/query",
            headers=_headers(harness, "owner"),
            json={"question": "任务概览", "query_type": "analysis_data"},
        )
        for job_id in ("job_b", "job_missing")
    ]

    assert {
        (response.status_code, response.json()["error"]["code"])
        for response in responses
    } == {(404, "RESOURCE_NOT_FOUND")}
    assert _query_count(harness) == before_count
    assert _projection_snapshot(harness, "job_b", "job_missing") == before_projection
    _assert_no_provider_calls(harness)


@pytest.mark.parametrize(
    "child_scope",
    [
        {"image_id": "img_b"},
        {"run_ids": ["run_b"]},
    ],
    ids=["foreign-image", "foreign-run"],
)
def test_explicit_foreign_child_ids_are_404_before_tools_or_audit(
    query_authorization_harness: QueryAuthorizationHarness,
    child_scope: dict[str, object],
) -> None:
    harness = query_authorization_harness
    before_projection = _projection_snapshot(harness, "job_a")

    response = harness.client.post(
        "/api/v1/analyses/job_a/query",
        headers=_headers(harness, "viewer"),
        json={
            "question": "任务概览",
            "query_type": "analysis_data",
            **child_scope,
        },
    )

    assert (response.status_code, response.json()["error"]["code"]) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert _query_count(harness) == 0
    assert _projection_snapshot(harness, "job_a") == before_projection
    _assert_no_provider_calls(harness)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "question": "这个材料为什么具有催化用途？",
            "query_type": "material_knowledge",
            "material_context": {"formula": "TiO2"},
        },
        {
            "question": "这个材料为什么有这种性质，我们这批覆盖率是多少？",
            "query_type": "mixed",
            "material_context": {"formula": "TiO2"},
        },
        {
            "question": "这个材料为什么具有催化用途？",
            "query_type": "auto",
            "material_context": {"formula": "TiO2"},
        },
    ],
    ids=["explicit-knowledge", "explicit-mixed", "auto-to-knowledge"],
)
def test_principal_knowledge_paths_fail_closed_before_retrieval_or_audit(
    query_authorization_harness: QueryAuthorizationHarness,
    payload: dict[str, object],
) -> None:
    harness = query_authorization_harness
    before_projection = _projection_snapshot(harness, "job_a")

    response = harness.client.post(
        "/api/v1/analyses/job_a/query",
        headers=_headers(harness, "owner"),
        json=payload,
    )

    assert (response.status_code, response.json()["error"]["code"]) == (
        503,
        "SERVICE_UNAVAILABLE",
    )
    assert response.json()["error"]["details"] == {"component": "knowledge_tenant_scope"}
    assert _query_count(harness) == 0
    assert _projection_snapshot(harness, "job_a") == before_projection
    _assert_no_provider_calls(harness)


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/api/v1/knowledge/documents", {}),
        (
            "patch",
            "/api/v1/knowledge/documents/doc_global",
            {"json": {"enabled": False}},
        ),
        ("post", "/api/v1/knowledge/reindex", {"json": {"force": False}}),
        (
            "post",
            "/api/v1/knowledge/documents",
            {
                "files": {"file": ("note.md", b"global evidence", "text/markdown")},
                "data": {
                    "metadata_json": json.dumps(
                        {
                            "title": "global note",
                            "source_type": "material_note",
                            "license_note": "test",
                            "allowed_for_demo": False,
                        }
                    )
                },
            },
        ),
    ],
)
def test_principal_knowledge_management_fails_closed(
    query_authorization_harness: QueryAuthorizationHarness,
    method: str,
    path: str,
    kwargs: dict[str, object],
) -> None:
    harness = query_authorization_harness

    response = getattr(harness.client, method)(
        path,
        headers=_headers(harness, "admin"),
        **kwargs,
    )

    assert (response.status_code, response.json()["error"]["code"]) == (
        503,
        "SERVICE_UNAVAILABLE",
    )
    assert response.json()["error"]["details"] == {"component": "knowledge_tenant_scope"}


def test_successful_query_performs_one_identity_join(
    query_authorization_harness: QueryAuthorizationHarness,
) -> None:
    harness = query_authorization_harness
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(" ".join(statement.split()))

    event.listen(harness.database.engine, "before_cursor_execute", capture_statement)
    try:
        response = harness.client.post(
            "/api/v1/analyses/job_a/query",
            headers=_headers(harness, "owner"),
            json={
                "question": "任务概览",
                "query_type": "analysis_data",
                "run_ids": ["run_a"],
            },
        )
    finally:
        event.remove(harness.database.engine, "before_cursor_execute", capture_statement)

    assert response.status_code == 200, response.text
    identity_reads = [
        statement for statement in statements if "FROM api_credentials JOIN principals" in statement
    ]
    assert len(identity_reads) == 1
