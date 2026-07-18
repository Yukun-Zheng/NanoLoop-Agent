from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.analysis.config import capture_execution_build_provenance
from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    ImageAssetDTO,
    ImageSummaryDTO,
    InferenceOptions,
    InvalidPixelRegion,
    PixelRect,
    QualityReportDTO,
    ROIBox,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityStatus,
    QualityTier,
    RoiMode,
)
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.repositories import StoredImageAsset
from app.core.config import Settings
from app.core.errors import BoxRevisionConflictError, InvalidBoxError, JobStateConflictError
from app.db.base import Base
from app.db.models import ModelRegistryRecord, SegmentationRun
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database


@pytest.fixture
def session(tmp_path) -> Session:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    db_session = database.session_factory()
    try:
        yield db_session
    finally:
        db_session.close()
        database.dispose()


def _seed_job_image(session: Session) -> SqlAlchemyRepositorySet:
    now = datetime.now(UTC)
    repositories = SqlAlchemyRepositorySet(session)
    repositories.jobs.create(
        AnalysisJobDTO(
            job_id="job_1",
            name="test",
            status=JobStatus.READY_FOR_CONFIGURATION,
            created_at=now,
            updated_at=now,
        )
    )
    repositories.images.add_many(
        [
            StoredImageAsset(
                storage_path="job_1/input/img_1/original.tif",
                asset=ImageAssetDTO(
                    image_id="img_1",
                    job_id="job_1",
                    filename="sample.tif",
                    sha256="a" * 64,
                    width=256,
                    height=200,
                    bit_depth=16,
                    sample_id="sample_1",
                    material_formula="SrNiO3-x",
                    scale_nm_per_pixel=0.5,
                    analysis_roi=AnalysisROI(
                        valid_rect=PixelRect(x1=0, y1=0, x2=256, y2=200),
                        invalid_rects=[
                            InvalidPixelRegion(x1=0, y1=180, x2=256, y2=200)
                        ],
                        source="detected",
                    ),
                ),
            )
        ]
    )
    session.add(
        ModelRegistryRecord(
            model_id="unet-small-balanced-v1",
            family=ModelFamily.UNET.value,
            variant=ModelVariant.SMALL_PARTICLE.value,
            quality_tier=QualityTier.BALANCED.value,
            version="1.0.0",
            adapter="tests.fake:FakeAdapter",
            status=ModelStatus.READY.value,
        )
    )
    session.commit()
    return repositories


def _run_configuration(now: datetime) -> RunConfiguration:
    return RunConfiguration(
        model_id="unet-small-balanced-v1",
        model_version="1.0.0",
        roi_mode=RoiMode.FULL_IMAGE,
        analysis_roi=AnalysisROI(
            valid_rect=PixelRect(x1=0, y1=0, x2=256, y2=200)
        ),
        inference=InferenceOptions(),
        preprocess_profile="sem_gray_v1",
        postprocess_profile="small_particle_v1",
        created_at=now,
    )


def _execution(now: datetime) -> ExecutionRuntimeProvenance:
    return ExecutionRuntimeProvenance(
        executor_build=capture_execution_build_provenance(),
        build_identity_matches_contract=False,
        requested_device=InferenceOptions().device,
        actual_device="not_applicable",
        seed=42,
        python_random_seeded=False,
        numpy_random_seeded=False,
        torch_deterministic_algorithms=False,
        global_inference_serialized=False,
        backend="legacy-test-adapter",
        warnings=["runtime_execution_evidence_unavailable_legacy_adapter"],
        executed_at=now,
    )


def test_box_revisions_include_empty_snapshots(session: Session) -> None:
    repositories = _seed_job_image(session)

    initial = repositories.boxes.get_active("img_1")
    assert initial.revision == 0
    assert initial.boxes == []

    saved = repositories.boxes.replace(
        "img_1", 0, [ROIBox(label="ROI", x1=20, y1=20, x2=100, y2=100)]
    )
    session.commit()
    assert saved.revision == 1
    assert saved.boxes[0].box_id is not None

    empty = repositories.boxes.replace("img_1", 1, [])
    session.commit()
    assert empty.revision == 2
    assert repositories.boxes.get_active("img_1").boxes == []
    revisions = repositories.boxes.list_by_job("job_1")
    assert [revision.revision for revision in revisions] == [0, 1, 2]
    assert [len(revision.boxes) for revision in revisions] == [0, 1, 0]

    with pytest.raises(BoxRevisionConflictError):
        repositories.boxes.replace("img_1", 1, [])


@pytest.mark.parametrize(
    "box,reason",
    [
        (ROIBox(x1=0, y1=0, x2=10, y2=40), "minimum_size"),
        (ROIBox(x1=220, y1=20, x2=260, y2=60), "out_of_bounds"),
        (ROIBox(x1=20, y1=150, x2=80, y2=190), "overlaps_invalid_region"),
    ],
)
def test_box_validation_is_strict(
    session: Session, box: ROIBox, reason: str
) -> None:
    repositories = _seed_job_image(session)
    with pytest.raises(InvalidBoxError) as exc_info:
        repositories.boxes.replace("img_1", 0, [box])
    assert exc_info.value.details["reason"] == reason


def test_run_configuration_is_immutable_but_result_fields_are_writable(
    session: Session,
) -> None:
    repositories = _seed_job_image(session)
    now = datetime.now(UTC)
    run = SegmentationRunDTO(
        run_id="run_1",
        job_id="job_1",
        image_id="img_1",
        model_id="unet-small-balanced-v1",
        status=JobStatus.CREATED,
        roi_mode=RoiMode.FULL_IMAGE,
        inference=InferenceOptions(),
        configuration=_run_configuration(now),
        created_at=now,
        updated_at=now,
    )
    repositories.runs.create_many([run])
    session.commit()

    record = session.get(SegmentationRun, "run_1")
    assert record is not None
    record.threshold = 0.8
    with pytest.raises(JobStateConflictError):
        session.flush()
    session.rollback()

    repositories.runs.update_status("run_1", JobStatus.VALIDATING)
    session.commit()
    stored = repositories.runs.get("run_1")
    assert stored.status == JobStatus.VALIDATING
    assert [event.to_status for event in stored.status_history] == [
        JobStatus.CREATED,
        JobStatus.VALIDATING,
    ]
    assert stored.status_history[0].from_status is None
    assert stored.status_history[1].from_status == JobStatus.CREATED


def test_queued_run_claim_is_compare_and_swap(session: Session) -> None:
    repositories = _seed_job_image(session)
    now = datetime.now(UTC)
    repositories.runs.create_many(
        [
            SegmentationRunDTO(
                run_id="run_claim",
                job_id="job_1",
                image_id="img_1",
                model_id="unet-small-balanced-v1",
                status=JobStatus.QUEUED,
                roi_mode=RoiMode.FULL_IMAGE,
                inference=InferenceOptions(),
                configuration=_run_configuration(now),
                created_at=now,
                updated_at=now,
            )
        ]
    )
    session.commit()

    assert repositories.runs.claim_queued("run_claim") is True
    session.commit()
    assert repositories.runs.claim_queued("run_claim") is False
    stored = repositories.runs.get("run_claim")
    assert stored.status == JobStatus.PREPROCESSING
    assert [event.to_status for event in stored.status_history] == [
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
    ]


def test_run_result_round_trip(session: Session) -> None:
    repositories = _seed_job_image(session)
    now = datetime.now(UTC)
    repositories.runs.create_many(
        [
            SegmentationRunDTO(
                run_id="run_1",
                job_id="job_1",
                image_id="img_1",
                model_id="unet-small-balanced-v1",
                status=JobStatus.ANALYZING,
                roi_mode=RoiMode.FULL_IMAGE,
                inference=InferenceOptions(),
                configuration=_run_configuration(now),
                created_at=now,
                updated_at=now,
            )
        ]
    )
    summary = ImageSummaryDTO(
        run_id="run_1",
        particle_count=0,
        roi_area_px=46080,
        number_density_px2=0,
        number_density_um2=0,
        mean_equivalent_diameter_px=None,
        mean_equivalent_diameter_nm=None,
        coverage_ratio=0,
        perimeter_density_px=0,
        perimeter_density_um=0,
        quality_status=QualityStatus.WARN,
    )
    quality = QualityReportDTO(
        status=QualityStatus.WARN,
        reasons=["empty_mask"],
        recommendations=["review model and ROI"],
    )
    repositories.runs.save_result(
        "run_1",
        particles=[],
        summary=summary,
        quality=quality,
        execution=_execution(now),
        runtime_ms=12,
        paths={"mask_url": "/api/v1/files/token", "overlay_url": None},
    )
    session.commit()

    stored = repositories.runs.get("run_1")
    assert stored.summary == summary
    assert stored.quality == quality
    assert stored.execution == _execution(now)
    assert stored.artifacts.mask_url == "/api/v1/files/token"
