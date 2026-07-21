from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    ImageAssetDTO,
    InferenceOptions,
    PixelRect,
    ROIBox,
    RunArtifacts,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    QueryType,
    RoiMode,
)
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID, PrincipalRole
from app.contracts.queries import (
    QueryActorAuthMode,
    QueryActorDTO,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from app.contracts.repositories import StoredImageAsset
from app.core.config import Settings
from app.core.errors import ResourceNotFoundError
from app.db.base import Base
from app.db.models import ApiCredential, ModelRegistryRecord, Principal, QueryLog, Tenant
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database

_TENANT_A = f"tnt_{'a' * 32}"
_TENANT_B = f"tnt_{'b' * 32}"
_PRINCIPAL_A = f"prn_{'c' * 32}"
_PRINCIPAL_B = f"prn_{'d' * 32}"
_CREDENTIAL_A = f"crd_{'e' * 32}"
_CREDENTIAL_B = f"crd_{'f' * 32}"
_MODEL_ID = "authorization-model"


@pytest.fixture
def session(tmp_path: Path) -> Generator[Session, None, None]:
    database = Database(Settings(database_url=f"sqlite:///{tmp_path / 'authorization.db'}"))
    Base.metadata.create_all(database.engine)
    db_session = database.session_factory()
    try:
        yield db_session
    finally:
        db_session.close()
        database.dispose()


@pytest.fixture
def repositories(session: Session) -> SqlAlchemyRepositorySet:
    now = datetime(2026, 7, 18, tzinfo=UTC)
    session.add_all(
        [
            Tenant(
                tenant_id=_TENANT_A,
                slug="authorization-a",
                display_name="Authorization A",
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            ),
            Tenant(
                tenant_id=_TENANT_B,
                slug="authorization-b",
                display_name="Authorization B",
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            Principal(
                principal_id=_PRINCIPAL_A,
                tenant_id=_TENANT_A,
                handle="owner-a",
                display_name="Owner A",
                kind="user",
                role="analyst",
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            ),
            Principal(
                principal_id=_PRINCIPAL_B,
                tenant_id=_TENANT_B,
                handle="owner-b",
                display_name="Owner B",
                kind="user",
                role="analyst",
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            ),
        ]
    )
    session.add(
        ModelRegistryRecord(
            model_id=_MODEL_ID,
            family=ModelFamily.UNET.value,
            variant=ModelVariant.GENERAL.value,
            quality_tier=QualityTier.BALANCED.value,
            version="1.0.0",
            adapter="tests.fake:AuthorizationAdapter",
            status=ModelStatus.READY.value,
        )
    )
    session.flush()
    session.add_all(
        [
            ApiCredential(
                credential_id=_CREDENTIAL_A,
                principal_id=_PRINCIPAL_A,
                label="authorization a",
                token_digest=b"a" * 32,
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            ),
            ApiCredential(
                credential_id=_CREDENTIAL_B,
                principal_id=_PRINCIPAL_B,
                label="authorization b",
                token_digest=b"b" * 32,
                enabled=True,
                version=1,
                created_at=now,
                updated_at=now,
            ),
        ]
    )
    session.flush()

    result = SqlAlchemyRepositorySet(session)
    for suffix, tenant_id, principal_id in (
        ("a", _TENANT_A, _PRINCIPAL_A),
        ("b", _TENANT_B, _PRINCIPAL_B),
    ):
        job_id = f"job_{suffix}"
        image_id = f"img_{suffix}"
        run_id = f"run_{suffix}"
        result.jobs.create(
            AnalysisJobDTO(
                job_id=job_id,
                name=f"job {suffix}",
                status=JobStatus.COMPLETED,
                created_at=now,
                updated_at=now,
            ),
            tenant_id=tenant_id,
            owner_principal_id=principal_id,
        )
        result.images.add_many(
            [
                StoredImageAsset(
                    asset=ImageAssetDTO(
                        image_id=image_id,
                        job_id=job_id,
                        filename=f"{suffix}.tif",
                        sha256=suffix * 64,
                        width=128,
                        height=96,
                        bit_depth=8,
                        sample_id=f"sample_{suffix}",
                        analysis_roi=AnalysisROI(
                            valid_rect=PixelRect(x1=0, y1=0, x2=128, y2=96)
                        ),
                    ),
                    storage_path=f"{job_id}/input/{image_id}/original.tif",
                )
            ]
        )
        result.runs.create_many(
            [
                SegmentationRunDTO(
                    run_id=run_id,
                    job_id=job_id,
                    image_id=image_id,
                    model_id=_MODEL_ID,
                    status=JobStatus.COMPLETED,
                    roi_mode=RoiMode.FULL_IMAGE,
                    inference=InferenceOptions(),
                    configuration=RunConfiguration(
                        model_id=_MODEL_ID,
                        model_version="1.0.0",
                        roi_mode=RoiMode.FULL_IMAGE,
                        analysis_roi=AnalysisROI(
                            valid_rect=PixelRect(x1=0, y1=0, x2=128, y2=96)
                        ),
                        inference=InferenceOptions(),
                        preprocess_profile="fixture",
                        postprocess_profile="fixture",
                        created_at=now,
                    ),
                    artifacts=RunArtifacts(mask_url=f"{job_id}/mask.png"),
                    created_at=now,
                    updated_at=now,
                )
            ]
        )
        session.add(
            QueryLog(
                query_id=f"query_{suffix}",
                job_id=job_id,
                image_id=image_id,
                query_type="auto",
                question=f"question {suffix}",
                request_json={"question": f"question {suffix}", "query_type": "auto"},
                answer_json={
                    "query_type": "auto",
                    "answer": f"answer {suffix}",
                    "confidence": "high",
                },
                actor_tenant_id=tenant_id,
                actor_principal_id=principal_id,
                actor_credential_id=(
                    _CREDENTIAL_A if tenant_id == _TENANT_A else _CREDENTIAL_B
                ),
                actor_role="analyst",
                actor_auth_mode="principal",
                created_at=now,
            )
        )
    result.jobs.create(
        AnalysisJobDTO(
            job_id="job_empty",
            name="empty job",
            status=JobStatus.READY_FOR_CONFIGURATION,
            created_at=now,
            updated_at=now,
        ),
        tenant_id=_TENANT_A,
        owner_principal_id=_PRINCIPAL_A,
    )
    session.commit()
    return result


def _assert_not_found(callable_: object) -> None:
    assert callable(callable_)
    with pytest.raises(ResourceNotFoundError) as error:
        callable_()
    assert error.value.code == "RESOURCE_NOT_FOUND"
    assert error.value.status_code == 404


def test_job_scope_contains_owner_and_hides_cross_tenant(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    scope = repositories.jobs.get_scope("job_a", tenant_id=_TENANT_A)
    assert scope.job.job_id == "job_a"
    assert scope.tenant_id == _TENANT_A
    assert scope.owner_principal_id == _PRINCIPAL_A

    _assert_not_found(lambda: repositories.jobs.get_scope("job_a", tenant_id=_TENANT_B))
    _assert_not_found(lambda: repositories.jobs.get_scope("job_missing", tenant_id=_TENANT_A))


def test_image_scoped_reads_require_matching_job_and_tenant(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    image = repositories.images.get_scoped("job_a", "img_a", tenant_id=_TENANT_A)
    assert image.image_id == "img_a"
    assert [item.image_id for item in repositories.images.list_by_job_scoped(
        "job_a", tenant_id=_TENANT_A
    )] == ["img_a"]
    assert repositories.images.get_storage_path_scoped(
        "job_a", "img_a", tenant_id=_TENANT_A
    ).endswith("original.tif")

    _assert_not_found(
        lambda: repositories.images.get_scoped("job_b", "img_b", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.images.get_scoped("job_a", "img_b", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.images.get_scoped("job_a", "img_missing", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.images.list_by_job_scoped("job_b", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.images.get_storage_path_scoped(
            "job_a", "img_b", tenant_id=_TENANT_A
        )
    )


def test_box_scoped_reads_and_replace_cannot_cross_aggregate(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    assert repositories.boxes.get_active_scoped(
        "job_a", "img_a", tenant_id=_TENANT_A
    ).revision == 0
    assert [item.revision for item in repositories.boxes.list_by_job_scoped(
        "job_a", tenant_id=_TENANT_A
    )] == [0]

    _assert_not_found(
        lambda: repositories.boxes.replace_scoped(
            "job_a",
            "img_b",
            0,
            [ROIBox(x1=10, y1=10, x2=50, y2=50)],
            tenant_id=_TENANT_A,
        )
    )
    _assert_not_found(
        lambda: repositories.boxes.get_active_scoped(
            "job_a", "img_missing", tenant_id=_TENANT_A
        )
    )
    assert repositories.boxes.get_active_scoped(
        "job_b", "img_b", tenant_id=_TENANT_B
    ).revision == 0


def test_run_scoped_reads_return_scope_and_hide_cross_tenant(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    run, scope = repositories.runs.get_with_scope("run_a", tenant_id=_TENANT_A)
    assert run.run_id == "run_a"
    assert scope.job.job_id == "job_a"
    assert scope.owner_principal_id == _PRINCIPAL_A
    assert [item.run_id for item in repositories.runs.list_by_job_scoped(
        "job_a", tenant_id=_TENANT_A
    )] == ["run_a"]
    assert repositories.runs.get_artifact_paths_scoped(
        "run_a", tenant_id=_TENANT_A
    )["mask_url"] == "job_a/mask.png"

    _assert_not_found(lambda: repositories.runs.get_with_scope("run_b", tenant_id=_TENANT_A))
    _assert_not_found(
        lambda: repositories.runs.get_with_scope("run_missing", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.runs.list_by_job_scoped("job_b", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.runs.get_artifact_paths_scoped("run_b", tenant_id=_TENANT_A)
    )


def test_query_export_snapshot_is_tenant_scoped(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    records = repositories.queries.list_by_job_scoped("job_a", tenant_id=_TENANT_A)
    assert [record.query_id for record in records] == ["query_a"]

    _assert_not_found(
        lambda: repositories.queries.list_by_job_scoped("job_b", tenant_id=_TENANT_A)
    )
    _assert_not_found(
        lambda: repositories.queries.list_by_job_scoped("job_missing", tenant_id=_TENANT_A)
    )


def test_query_write_rechecks_embedded_run_scope_in_final_transaction(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    actor = QueryActorDTO(
        tenant_id=_TENANT_A,
        principal_id=_PRINCIPAL_A,
        credential_id=_CREDENTIAL_A,
        role=PrincipalRole.ANALYST,
        auth_mode="principal",
    )

    with pytest.raises(ResourceNotFoundError):
        repositories.queries.create_scoped(
            query_id="query_raced",
            job_id="job_a",
            image_id="img_a",
            actor=actor,
            request=UnifiedQueryRequest(
                question="任务概览",
                query_type=QueryType.ANALYSIS_DATA,
                run_ids=["run_disappeared"],
            ),
            response=UnifiedQueryResponse(
                query_type=QueryType.ANALYSIS_DATA,
                answer="stale provider result",
                confidence="high",
            ),
            created_at=datetime(2026, 7, 18, tzinfo=UTC),
            tenant_id=_TENANT_A,
        )


def test_query_write_rejects_migration_only_legacy_unknown_actor(
    repositories: SqlAlchemyRepositorySet,
    session: Session,
) -> None:
    actor = QueryActorDTO(
        tenant_id=LEGACY_TENANT_ID,
        principal_id=LEGACY_PRINCIPAL_ID,
        credential_id=None,
        role=PrincipalRole.TENANT_ADMIN,
        auth_mode=QueryActorAuthMode.LEGACY_UNKNOWN,
    )

    with pytest.raises(ValueError, match="migration-only"):
        repositories.queries.create_scoped(
            query_id="query_migration_only_actor",
            job_id="job_a",
            image_id=None,
            actor=actor,
            request=UnifiedQueryRequest(
                question="任务概览",
                query_type=QueryType.ANALYSIS_DATA,
            ),
            response=UnifiedQueryResponse(
                query_type=QueryType.ANALYSIS_DATA,
                answer="must not be persisted",
                confidence="high",
            ),
            created_at=datetime(2026, 7, 18, tzinfo=UTC),
            tenant_id=_TENANT_A,
        )

    assert session.get(QueryLog, "query_migration_only_actor") is None


def test_scoped_lists_distinguish_an_accessible_empty_job_from_hidden_jobs(
    repositories: SqlAlchemyRepositorySet,
) -> None:
    assert repositories.images.list_by_job_scoped("job_empty", tenant_id=_TENANT_A) == []
    assert repositories.boxes.list_by_job_scoped("job_empty", tenant_id=_TENANT_A) == []
    assert repositories.runs.list_by_job_scoped("job_empty", tenant_id=_TENANT_A) == []
    assert repositories.queries.list_by_job_scoped("job_empty", tenant_id=_TENANT_A) == []
