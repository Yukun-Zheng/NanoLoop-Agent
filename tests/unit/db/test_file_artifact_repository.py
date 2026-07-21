from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.contracts.file_artifacts import (
    FileArtifactKind,
    FileArtifactRegistration,
    FileArtifactState,
)
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import Settings
from app.core.errors import ResourceNotFoundError
from app.db.base import Base
from app.db.models import (
    AnalysisJob,
    FileArtifact,
    ImageAsset,
    ModelRegistryRecord,
    Principal,
    SegmentationRun,
    Tenant,
)
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database

_TENANT_B = f"tnt_{'b' * 32}"
_PRINCIPAL_B = f"prn_{'c' * 32}"


@pytest.fixture
def artifact_database(tmp_path) -> Iterator[Database]:
    database = Database(Settings(database_url=f"sqlite:///{tmp_path / 'artifacts.db'}"))
    Base.metadata.create_all(database.engine)
    try:
        yield database
    finally:
        database.dispose()


def _seed_scopes(session: Session) -> None:
    session.add(
        Tenant(
            tenant_id=_TENANT_B,
            slug="tenant-b",
            display_name="Tenant B",
            enabled=True,
            version=1,
        )
    )
    session.flush()
    session.add(
        Principal(
            principal_id=_PRINCIPAL_B,
            tenant_id=_TENANT_B,
            handle="owner-b",
            display_name="Owner B",
            kind="user",
            role="analyst",
            enabled=True,
            version=1,
        )
    )
    session.flush()
    session.add_all(
        [
            AnalysisJob(
                job_id="job_a",
                tenant_id=LEGACY_TENANT_ID,
                owner_principal_id=LEGACY_PRINCIPAL_ID,
                name="Job A",
                status="CREATED",
                config_json={},
            ),
            AnalysisJob(
                job_id="job_b",
                tenant_id=_TENANT_B,
                owner_principal_id=_PRINCIPAL_B,
                name="Job B",
                status="CREATED",
                config_json={},
            ),
            ModelRegistryRecord(
                model_id="model_a",
                family="unet",
                variant="general",
                quality_tier="balanced",
                version="1",
                adapter="tests.fake:FakeAdapter",
                status="ready",
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            _image("image_a", "job_a", "a"),
            _image("image_a2", "job_a", "b"),
            _image("image_b", "job_b", "c"),
        ]
    )
    session.flush()
    session.add_all(
        [
            _run("run_a", "job_a", "image_a"),
            _run("run_b", "job_b", "image_b"),
        ]
    )
    session.commit()


def _image(image_id: str, job_id: str, digest_character: str) -> ImageAsset:
    return ImageAsset(
        image_id=image_id,
        job_id=job_id,
        filename=f"{image_id}.tif",
        storage_path=f"jobs/{job_id}/input/{image_id}/original.tif",
        sha256=digest_character * 64,
        width=32,
        height=32,
        bit_depth=8,
        sample_id=image_id,
        experiment_conditions_json={},
        analysis_roi_json={},
        box_revision=0,
    )


def _run(run_id: str, job_id: str, image_id: str) -> SegmentationRun:
    return SegmentationRun(
        run_id=run_id,
        job_id=job_id,
        image_id=image_id,
        model_id="model_a",
        roi_mode="full_image",
        status="CREATED",
        inference_json={},
        run_config_json={},
        paths_json={},
    )


def _registration(**updates: object) -> FileArtifactRegistration:
    values: dict[str, object] = {
        "job_id": "job_a",
        "image_id": "image_a",
        "run_id": None,
        "artifact_kind": FileArtifactKind.ORIGINAL_IMAGE,
        "storage_path": "jobs/job_a/input/image_a/original.tif",
        "filename": "sample.tif",
        "media_type": "image/tiff",
        "sha256": "a" * 64,
        "size_bytes": 512,
    }
    values.update(updates)
    return FileArtifactRegistration(**values)


def test_registration_is_tenant_scoped_and_idempotent_only_for_identical_facts(
    artifact_database: Database,
) -> None:
    with artifact_database.session_factory() as session:
        _seed_scopes(session)
        artifacts = SqlAlchemyRepositorySet(session).file_artifacts
        registration = _registration()

        created = artifacts.register(registration, tenant_id=LEGACY_TENANT_ID)
        session.commit()
        repeated = artifacts.register(registration, tenant_id=LEGACY_TENANT_ID)

        assert repeated.artifact_id == created.artifact_id
        assert repeated.state is FileArtifactState.ACTIVE
        assert repeated.storage_path == registration.storage_path
        with pytest.raises(ValueError, match="different immutable facts"):
            artifacts.register(
                registration.model_copy(update={"sha256": "f" * 64}),
                tenant_id=LEGACY_TENANT_ID,
            )

        assert artifacts.get_active(created.artifact_id, tenant_id=LEGACY_TENANT_ID) == repeated
        assert (
            artifacts.get_active_by_storage_path(
                registration.storage_path,
                tenant_id=LEGACY_TENANT_ID,
            )
            == repeated
        )
        with pytest.raises(ResourceNotFoundError):
            artifacts.get_active(created.artifact_id, tenant_id=_TENANT_B)


def test_registration_rejects_orphans_exact_run_image_mismatch_and_cross_tenant_scope(
    artifact_database: Database,
) -> None:
    with artifact_database.session_factory() as session:
        _seed_scopes(session)
        artifacts = SqlAlchemyRepositorySet(session).file_artifacts

        with pytest.raises(ResourceNotFoundError):
            artifacts.register(
                _registration(image_id="missing_image"),
                tenant_id=LEGACY_TENANT_ID,
            )
        with pytest.raises(ResourceNotFoundError):
            artifacts.register(
                _registration(
                    image_id="image_a",
                    run_id="missing_run",
                    artifact_kind=FileArtifactKind.RUN_ARTIFACT,
                ),
                tenant_id=LEGACY_TENANT_ID,
            )
        with pytest.raises(ResourceNotFoundError):
            artifacts.register(
                _registration(
                    image_id="image_a2",
                    run_id="run_a",
                    artifact_kind=FileArtifactKind.RUN_ARTIFACT,
                ),
                tenant_id=LEGACY_TENANT_ID,
            )
        with pytest.raises(ResourceNotFoundError):
            artifacts.register(
                _registration(
                    job_id="job_b",
                    image_id="image_b",
                    storage_path="jobs/job_b/input/image_b/original.tif",
                    sha256="c" * 64,
                ),
                tenant_id=LEGACY_TENANT_ID,
            )

        private_path = artifacts.register(
            _registration(
                image_id=None,
                artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
                storage_path="shared/global-name.zip",
                filename="export.zip",
                media_type="application/zip",
            ),
            tenant_id=LEGACY_TENANT_ID,
        )
        session.commit()
        assert private_path.state is FileArtifactState.ACTIVE
        with pytest.raises(ResourceNotFoundError):
            artifacts.register(
                _registration(
                    job_id="job_b",
                    image_id=None,
                    artifact_kind=FileArtifactKind.ANALYSIS_EXPORT,
                    storage_path="shared/global-name.zip",
                    filename="export.zip",
                    media_type="application/zip",
                ),
                tenant_id=_TENANT_B,
            )


def test_corrected_mask_consumption_is_one_way_and_terminal_retry_never_reactivates(
    artifact_database: Database,
) -> None:
    registration = _registration(
        run_id="run_a",
        artifact_kind=FileArtifactKind.CORRECTED_MASK_INPUT,
        storage_path="jobs/job_a/review/run_a/corrected-mask.png",
        filename="corrected-mask.png",
        media_type="image/png",
    )
    with artifact_database.session_factory() as session:
        _seed_scopes(session)
        artifacts = SqlAlchemyRepositorySet(session).file_artifacts
        created = artifacts.register(registration, tenant_id=LEGACY_TENANT_ID)
        session.commit()

        consumed_at = datetime.now(UTC) + timedelta(seconds=1)
        assert artifacts.consume_corrected_mask(
            created.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
            consumed_at=consumed_at,
        )
        session.commit()
        assert not artifacts.consume_corrected_mask(
            created.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        repeated = artifacts.register(registration, tenant_id=LEGACY_TENANT_ID)
        assert repeated.artifact_id == created.artifact_id
        assert repeated.state is FileArtifactState.CONSUMED
        assert repeated.consumed_at == consumed_at
        with pytest.raises(ResourceNotFoundError):
            artifacts.get_active(created.artifact_id, tenant_id=LEGACY_TENANT_ID)


def test_corrected_mask_compare_and_swap_rejects_a_stale_competing_reader(
    artifact_database: Database,
) -> None:
    registration = _registration(
        run_id="run_a",
        artifact_kind=FileArtifactKind.CORRECTED_MASK_INPUT,
        storage_path="jobs/job_a/review/run_a/concurrent-mask.png",
        filename="concurrent-mask.png",
        media_type="image/png",
    )
    with artifact_database.session_factory() as setup_session:
        _seed_scopes(setup_session)
        artifact = SqlAlchemyRepositorySet(setup_session).file_artifacts.register(
            registration,
            tenant_id=LEGACY_TENANT_ID,
        )
        setup_session.commit()

    with (
        artifact_database.session_factory() as winner_session,
        artifact_database.session_factory() as stale_session,
    ):
        stale_record = stale_session.get(FileArtifact, artifact.artifact_id)
        assert stale_record is not None and stale_record.state == "active"

        winner = SqlAlchemyRepositorySet(winner_session).file_artifacts
        assert winner.consume_corrected_mask(
            artifact.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        winner_session.commit()

        loser = SqlAlchemyRepositorySet(stale_session).file_artifacts
        assert not loser.consume_corrected_mask(
            artifact.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        stale_session.commit()


def test_non_corrected_artifact_cannot_be_consumed(artifact_database: Database) -> None:
    with artifact_database.session_factory() as session:
        _seed_scopes(session)
        artifacts = SqlAlchemyRepositorySet(session).file_artifacts
        artifact = artifacts.register(_registration(), tenant_id=LEGACY_TENANT_ID)
        session.commit()
        with pytest.raises(ValueError, match="only corrected-mask"):
            artifacts.consume_corrected_mask(
                artifact.artifact_id,
                tenant_id=LEGACY_TENANT_ID,
            )


def test_revocation_is_tenant_scoped_and_terminal(artifact_database: Database) -> None:
    with artifact_database.session_factory() as session:
        _seed_scopes(session)
        artifacts = SqlAlchemyRepositorySet(session).file_artifacts
        artifact = artifacts.register(_registration(), tenant_id=LEGACY_TENANT_ID)
        session.commit()

        with pytest.raises(ResourceNotFoundError):
            artifacts.revoke(artifact.artifact_id, tenant_id=_TENANT_B)
        revoked_at = datetime.now(UTC) + timedelta(seconds=1)
        assert artifacts.revoke(
            artifact.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
            revoked_at=revoked_at,
        )
        session.commit()
        assert not artifacts.revoke(
            artifact.artifact_id,
            tenant_id=LEGACY_TENANT_ID,
        )
        persisted = session.get(FileArtifact, artifact.artifact_id)
        assert persisted is not None
        assert persisted.state == FileArtifactState.REVOKED.value
        assert persisted.revoked_at is not None
        assert persisted.consumed_at is None
        with pytest.raises(ResourceNotFoundError):
            artifacts.get_active(artifact.artifact_id, tenant_id=LEGACY_TENANT_ID)
