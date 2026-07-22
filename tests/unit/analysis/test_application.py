import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import update

import app.analysis.application as analysis_application_module
from app.analysis.application import (
    AnalysisApplicationService,
    AnalysisCreationService,
    AnalysisUpload,
)
from app.analysis.config import MorphometryConfig, PostprocessProfile, QualityGateConfig
from app.analysis.instance_artifacts import decode_binary_mask
from app.analysis.morphometry import MorphometryResult
from app.analysis.morphometry import measure as measure_particles
from app.analysis.postprocessing import NormalizedInstance
from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    CreateAnalysisMetadata,
    CreateRunsRequest,
    ImageAssetDTO,
    ImageMetadataInput,
    InferenceOptions,
    PixelRect,
    ReviewRunRequest,
    ROIBox,
    RunConfiguration,
    ScaleInput,
)
from app.contracts.enums import (
    DevicePreference,
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
    ScaleMode,
)
from app.contracts.execution import InferenceExecutionEvidence
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelHealth, ModelMetadata
from app.contracts.repositories import StoredImageAsset, UnitOfWork
from app.core.config import Settings
from app.core.errors import (
    ExecutionBuildMismatchError,
    InputArtifactMismatchError,
    InvalidImageError,
    ModelNotReadyError,
)
from app.db.base import Base
from app.db.models import ImageAsset as ImageAssetRecord
from app.db.models import ModelRegistryRecord, SegmentationRun
from app.db.repositories import SqlAlchemyRepositorySet, SqlAlchemyUnitOfWork
from app.db.session import Database
from app.inference.gateway import InferenceGateway
from app.storage import FileTokenError, LocalFileStore, StoragePathError, StoragePaths
from tests.unit.inference.fakes import FakeAdapter
from tests.unit.inference.helpers import build_registry, model_entry

ApplicationHarness = tuple[
    Database,
    LocalFileStore,
    Callable[[], UnitOfWork],
]


def _model_bundle(model: ModelMetadata) -> ModelBundleReference:
    assert model.weight_sha256 is not None
    assert model.adapter_sha256 is not None
    prefix = f"bundles/{model.model_id}"
    return ModelBundleReference(
        bundle_id=model.weight_sha256,
        manifest_ref=f"{prefix}/manifest.json",
        weight_ref=f"{prefix}/weights.bin",
        config_ref=f"{prefix}/config.yaml",
        model_card_ref=f"{prefix}/model-card.md",
        adapter_ref=f"{prefix}/adapter.py",
        adapter_sha256=model.adapter_sha256,
    )


def _execution_evidence(backend: str) -> InferenceExecutionEvidence:
    return InferenceExecutionEvidence(
        actual_device="cpu",
        python_random_seeded=True,
        numpy_random_seeded=True,
        torch_deterministic_algorithms=False,
        global_inference_serialized=True,
        backend=backend,
    )


class FakeGateway:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.predict_calls = 0
        self.model = ModelMetadata(
            model_id="unet-general-balanced-v1",
            family=ModelFamily.UNET,
            variant=ModelVariant.GENERAL,
            quality_tier=QualityTier.BALANCED,
            version="1.0.0",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.5,
            preprocess_profile="sem_gray_v1",
            postprocess_profile="default_v1",
            applicable_materials=[],
            adapter_path="tests.fake:FakeAdapter",
            weight_sha256="a" * 64,
            config_sha256="b" * 64,
            model_card_sha256="c" * 64,
            adapter_sha256="d" * 64,
        )

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        return [self.model]

    def health(self) -> list[ModelHealth]:
        return [
            ModelHealth(
                model_id=self.model.model_id,
                status=ModelStatus.READY,
                weight_sha256=self.model.weight_sha256,
            )
        ]

    def freeze_model_bundle(
        self,
        model_id: str,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
    ) -> ModelBundleReference:
        assert model_id == self.model.model_id
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 == self.model.adapter_sha256
        return _model_bundle(self.model)

    def predict(
        self,
        _model_id: str,
        request: SegmentationRequest,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
        model_bundle: ModelBundleReference | None = None,
    ) -> SegmentationOutput:
        self.predict_calls += 1
        assert request.image_bytes is not None
        assert not request.image_path.exists()
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 == self.model.adapter_sha256
        assert model_bundle == _model_bundle(self.model)
        if self.fail:
            raise RuntimeError("fixture inference failed")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[1, 1] = 255  # raw speck removed by shared postprocessing
        mask[20:30, 20:30] = 255
        probability = np.zeros((64, 64), dtype=np.float32)
        probability[20:30, 20:30] = 0.9
        mask_path = request.run_dir / "pred_mask.png"
        probability_path = request.run_dir / "probability.npy"
        Image.fromarray(mask).save(mask_path)
        np.save(probability_path, probability, allow_pickle=False)
        return SegmentationOutput(
            width=64,
            height=64,
            binary_mask_path=mask_path,
            probability_path=probability_path,
            runtime_ms=7,
            execution=_execution_evidence("tests.fake:FakeAdapter"),
        )


class HoleGateway(FakeGateway):
    def predict(
        self,
        model_id: str,
        request: SegmentationRequest,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
        model_bundle: ModelBundleReference | None = None,
    ) -> SegmentationOutput:
        output = super().predict(
            model_id,
            request,
            expected_model_version=expected_model_version,
            expected_adapter_path=expected_adapter_path,
            expected_weight_sha256=expected_weight_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_model_card_sha256=expected_model_card_sha256,
            expected_adapter_sha256=expected_adapter_sha256,
            model_bundle=model_bundle,
        )
        with Image.open(output.binary_mask_path) as image:
            mask = np.asarray(image).copy()
        mask[25, 25] = 0
        Image.fromarray(mask).save(output.binary_mask_path)
        return output


class BoundaryGateway(FakeGateway):
    def predict(
        self,
        _model_id: str,
        request: SegmentationRequest,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
        model_bundle: ModelBundleReference | None = None,
    ) -> SegmentationOutput:
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 == self.model.adapter_sha256
        assert model_bundle == _model_bundle(self.model)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:10, 20:30] = 255
        mask_path = request.run_dir / "pred_mask.png"
        Image.fromarray(mask).save(mask_path)
        return SegmentationOutput(
            width=64,
            height=64,
            binary_mask_path=mask_path,
            runtime_ms=2,
            execution=_execution_evidence("tests.fake:BoundaryAdapter"),
        )


class CartesianGateway:
    def __init__(self) -> None:
        self.models = [
            ModelMetadata(
                model_id="unet-general-balanced-v1",
                family=ModelFamily.UNET,
                variant=ModelVariant.GENERAL,
                quality_tier=QualityTier.BALANCED,
                version="1.0.0",
                status=ModelStatus.READY,
                supports_box_prompt=False,
                default_threshold=0.31,
                preprocess_profile="semantic_general_v1",
                postprocess_profile="general_v1",
                adapter_path="tests.fake:GeneralAdapter",
                weight_sha256="1" * 64,
                config_sha256="a" * 64,
                model_card_sha256="d" * 64,
                adapter_sha256="4" * 64,
            ),
            ModelMetadata(
                model_id="yolo-dense-fast-v2",
                family=ModelFamily.YOLO_SEG,
                variant=ModelVariant.DENSE_PARTICLE,
                quality_tier=QualityTier.FAST,
                version="2.1.0",
                status=ModelStatus.READY,
                supports_box_prompt=False,
                default_threshold=0.52,
                preprocess_profile="instance_dense_v2",
                postprocess_profile="dense_v2",
                adapter_path="tests.fake:DenseAdapter",
                weight_sha256="2" * 64,
                config_sha256="b" * 64,
                model_card_sha256="e" * 64,
                adapter_sha256="5" * 64,
            ),
            ModelMetadata(
                model_id="sam2-low-accurate-v3",
                family=ModelFamily.SAM2,
                variant=ModelVariant.LOW_CONTRAST,
                quality_tier=QualityTier.ACCURATE,
                version="3.2.0",
                status=ModelStatus.READY,
                supports_box_prompt=True,
                default_threshold=0.73,
                preprocess_profile="prompt_low_contrast_v3",
                postprocess_profile="low_contrast_v3",
                adapter_path="tests.fake:LowContrastAdapter",
                weight_sha256="3" * 64,
                config_sha256="c" * 64,
                model_card_sha256="f" * 64,
                adapter_sha256="6" * 64,
            ),
        ]

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        if only_ready:
            return [model for model in self.models if model.status == ModelStatus.READY]
        return list(self.models)

    def health(self) -> list[ModelHealth]:
        return [
            ModelHealth(
                model_id=model.model_id,
                status=model.status,
                weight_sha256=model.weight_sha256,
            )
            for model in self.models
        ]

    def freeze_model_bundle(
        self,
        model_id: str,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
    ) -> ModelBundleReference:
        model = next(item for item in self.models if item.model_id == model_id)
        assert expected_model_version == model.version
        assert expected_adapter_path == model.adapter_path
        assert expected_weight_sha256 == model.weight_sha256
        assert expected_config_sha256 == model.config_sha256
        assert expected_model_card_sha256 == model.model_card_sha256
        assert expected_adapter_sha256 == model.adapter_sha256
        return _model_bundle(model)

    def predict(
        self,
        _model_id: str,
        _request: SegmentationRequest,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
        model_bundle: ModelBundleReference | None = None,
    ) -> SegmentationOutput:
        raise AssertionError("Cartesian-product creation must not execute inference")


class RecordingDispatcher:
    def __init__(self) -> None:
        self.submissions: list[str] = []

    def submit(self, run_id: str) -> bool:
        self.submissions.append(run_id)
        return True


@pytest.fixture
def application_harness(tmp_path: Path) -> Iterator[ApplicationHarness]:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'application.db'}",
        output_root=tmp_path / "outputs",
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    file_store = LocalFileStore(
        StoragePaths(settings.output_root),
        max_upload_bytes=1_000_000,
        token_secret=b"a" * 32,
    )
    image_bytes = BytesIO()
    Image.new("L", (64, 64), color=10).save(image_bytes, format="PNG")
    image_bytes.seek(0)
    stored = file_store.save_upload("job_1", image_bytes, "sample.png", image_id="img_1")
    now = datetime.now(UTC)
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        repositories.jobs.create(
            AnalysisJobDTO(
                job_id="job_1",
                name="application test",
                status=JobStatus.READY_FOR_CONFIGURATION,
                created_at=now,
                updated_at=now,
            )
        )
        repositories.images.add_many(
            [
                StoredImageAsset(
                    storage_path=stored.relative_path,
                    asset=ImageAssetDTO(
                        image_id="img_1",
                        job_id="job_1",
                        filename="sample.png",
                        sha256=stored.sha256,
                        width=64,
                        height=64,
                        bit_depth=8,
                        sample_id="sample_1",
                        material_formula="TiO2",
                        scale_nm_per_pixel=1.0,
                        analysis_roi=AnalysisROI(
                            valid_rect=PixelRect(x1=0, y1=0, x2=64, y2=64)
                        ),
                    ),
                )
            ]
        )
        session.add(
            ModelRegistryRecord(
                model_id="unet-general-balanced-v1",
                family=ModelFamily.UNET.value,
                variant=ModelVariant.GENERAL.value,
                quality_tier=QualityTier.BALANCED.value,
                version="1.0.0",
                adapter="tests.fake:FakeAdapter",
                status=ModelStatus.READY.value,
            )
        )
    def factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(database.session_factory)

    yield database, file_store, factory
    database.dispose()


def test_create_and_execute_run_closes_the_analysis_loop(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(),
    )
    run_ids = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
            inference=InferenceOptions(min_area_px=8, exclude_border=True),
        ),
    )
    assert len(run_ids) == 1

    completed = service.execute_run(run_ids[0])
    assert completed.status == JobStatus.COMPLETED
    assert completed.summary is not None
    assert completed.summary.particle_count == 1
    assert completed.summary.coverage_ratio == pytest.approx(100 / 4096)
    assert completed.configuration.weight_sha256 == "a" * 64
    assert completed.configuration.config_sha256 == "b" * 64
    assert completed.configuration.model_card_sha256 == "c" * 64
    assert completed.configuration.adapter_sha256 == "d" * 64
    assert completed.configuration.schema_version == 3
    assert completed.configuration.model_bundle is not None
    assert completed.execution is not None
    assert completed.execution.actual_device == "cpu"
    assert completed.execution.backend == "tests.fake:FakeAdapter"
    assert completed.execution.model_bundle_id == "a" * 64
    assert completed.execution.adapter_sha256 == "d" * 64
    assert "runtime_execution_evidence_unavailable_legacy_adapter" not in (
        completed.execution.warnings
    )
    assert [event.to_status for event in completed.status_history] == [
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
        JobStatus.SEGMENTING,
        JobStatus.POSTPROCESSING,
        JobStatus.QUALITY_CHECKING,
        JobStatus.ANALYZING,
        JobStatus.AGGREGATING,
        JobStatus.COMPLETED,
    ]

    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        assert repositories.jobs.get("job_1").status == JobStatus.COMPLETED
        paths = repositories.runs.get_artifact_paths(run_ids[0])
    assert paths["particles_csv_path"] is not None
    assert paths["overlay_path"] is not None
    execution_path = paths["execution_provenance_path"]
    assert execution_path is not None
    assert (file_store.paths.root / paths["particles_csv_path"]).is_file()
    assert (file_store.paths.root / paths["overlay_path"]).is_file()
    execution_payload = json.loads(
        (file_store.paths.root / execution_path).read_text(encoding="utf-8")
    )
    assert execution_payload["actual_device"] == "cpu"
    assert execution_payload["seed"] == 42
    mask_path = paths["pred_mask_path"]
    assert mask_path is not None
    with Image.open(file_store.paths.root / mask_path) as normalized:
        normalized_pixels = np.asarray(normalized)
    assert normalized_pixels[1, 1] == 0
    assert normalized_pixels[25, 25] == 255
    instances_path = paths["instances_path"]
    assert instances_path is not None
    instances_payload = json.loads(
        (file_store.paths.root / instances_path).read_text(encoding="utf-8")
    )
    assert instances_payload["instance_count"] == 1
    mask_record = instances_payload["instances"][0]["mask"]
    decoded_instance = decode_binary_mask(
        starts=mask_record["starts"],
        lengths=mask_record["lengths"],
        width=64,
        height=64,
    )
    assert np.array_equal(decoded_instance, normalized_pixels > 0)


def test_create_runs_fails_closed_without_bundle_freeze_support(
    application_harness: ApplicationHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _database, file_store, factory = application_harness
    gateway = FakeGateway()
    monkeypatch.setattr(gateway, "freeze_model_bundle", None)
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )

    with pytest.raises(ModelNotReadyError) as captured:
        service.create_runs(
            "job_1",
            CreateRunsRequest(
                image_ids=["img_1"],
                model_ids=["unet-general-balanced-v1"],
                roi_mode=RoiMode.FULL_IMAGE,
            ),
        )

    assert captured.value.details["reason"] == "model_bundle_freeze_unsupported"
    with factory() as uow:
        assert uow.repositories.runs.list_by_job("job_1") == []


def test_create_runs_fails_closed_without_adapter_digest(
    application_harness: ApplicationHarness,
) -> None:
    _database, file_store, factory = application_harness
    gateway = FakeGateway()
    gateway.model = gateway.model.model_copy(update={"adapter_sha256": None})
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )

    with pytest.raises(ModelNotReadyError) as captured:
        service.create_runs(
            "job_1",
            CreateRunsRequest(
                image_ids=["img_1"],
                model_ids=["unet-general-balanced-v1"],
                roi_mode=RoiMode.FULL_IMAGE,
            ),
        )

    assert captured.value.details["reason"] == "artifact_provenance_incomplete"
    assert captured.value.details["missing_fields"] == ["adapter_sha256"]
    with factory() as uow:
        assert uow.repositories.runs.list_by_job("job_1") == []


def test_create_runs_fails_closed_for_invalid_bundle_reference(
    application_harness: ApplicationHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _database, file_store, factory = application_harness
    gateway = FakeGateway()

    def invalid_freeze(*_args: object, **_kwargs: object) -> object:
        return None

    monkeypatch.setattr(gateway, "freeze_model_bundle", invalid_freeze)
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )

    with pytest.raises(ModelNotReadyError) as captured:
        service.create_runs(
            "job_1",
            CreateRunsRequest(
                image_ids=["img_1"],
                model_ids=["unet-general-balanced-v1"],
                roi_mode=RoiMode.FULL_IMAGE,
            ),
        )

    assert captured.value.details["reason"] == "invalid_model_bundle_reference"
    with factory() as uow:
        assert uow.repositories.runs.list_by_job("job_1") == []


def test_queued_run_executes_complete_bundle_after_config_and_card_sources_change(
    application_harness: ApplicationHarness,
    tmp_path: Path,
) -> None:
    _database, file_store, factory = application_harness
    artifact_root = tmp_path / "model"
    artifact_root.mkdir()
    entry = model_entry(
        artifact_root,
        "unet-general-balanced-v1",
        config={"marker": "queued"},
    )
    observed_configs: list[dict[str, object]] = []

    class QueuedBundleAdapter(FakeAdapter):
        def load(self, device: str) -> None:
            observed_configs.append(dict(self.config))
            super().load(device)

        def predict(self, request: SegmentationRequest) -> SegmentationOutput:
            assert request.image_bytes is not None
            assert not request.image_path.exists()
            mask_path = request.run_dir / "bundle-mask.png"
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[20:30, 20:30] = 255
            Image.fromarray(mask).save(mask_path)
            return SegmentationOutput(
                width=64,
                height=64,
                binary_mask_path=mask_path,
                runtime_ms=3,
            )

    registry = build_registry(
        artifact_root,
        [entry],
        resolver=lambda _: QueuedBundleAdapter,
    )
    gateway = InferenceGateway(registry)
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with factory() as uow:
        queued = uow.repositories.runs.get(run_id)
    assert queued.configuration.schema_version == 3
    assert queued.configuration.model_bundle is not None
    assert queued.configuration.adapter_sha256 is not None

    (artifact_root / entry["config_path"]).write_text(
        "marker: changed-after-queue\n", encoding="utf-8"
    )
    (artifact_root / entry["model_card_path"]).write_text(
        "# changed after queue\n", encoding="utf-8"
    )
    registry.refresh()

    bundle = queued.configuration.model_bundle
    assert bundle is not None
    completed = service.execute_run(run_id)

    assert observed_configs == [{"marker": "queued"}]
    assert completed.execution is not None
    assert completed.execution.actual_device == "cpu"
    assert completed.execution.seed == 42
    assert completed.execution.global_inference_serialized is True
    assert completed.execution.model_bundle_id == bundle.bundle_id
    assert completed.execution.adapter_sha256 == queued.configuration.adapter_sha256
    assert completed.execution.build_identity_matches_contract is True

    inference_review_id = service.create_review_run(
        run_id,
        ReviewRunRequest(threshold=0.61),
    )
    corrected_mask = BytesIO()
    Image.new("L", (64, 64), color=255).save(corrected_mask, format="PNG")
    corrected_mask.seek(0)
    staged = service.stage_corrected_mask(run_id, corrected_mask, "corrected.png")
    staged_path = file_store.resolve_file_token(staged.corrected_mask_token)
    corrected_review_id = service.create_review_run(
        run_id,
        ReviewRunRequest(corrected_mask_token=staged.corrected_mask_token),
    )
    assert not staged_path.exists()
    with pytest.raises(FileTokenError, match="available"):
        file_store.resolve_file_token(staged.corrected_mask_token)
    with factory() as uow:
        inference_review = uow.repositories.runs.get(inference_review_id)
        corrected_review = uow.repositories.runs.get(corrected_review_id)
    assert inference_review.configuration.schema_version == 3
    assert inference_review.configuration.model_bundle == bundle
    assert inference_review.configuration.adapter_sha256 == bundle.adapter_sha256
    assert corrected_review.configuration.schema_version == 3
    assert corrected_review.configuration.model_bundle == bundle
    assert corrected_review.configuration.review_source == "corrected_mask"


def test_execution_uses_only_the_frozen_scientific_configuration(
    application_harness: ApplicationHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, file_store, factory = application_harness
    captured_morphometry: list[tuple[MorphometryConfig, float | None]] = []

    def recording_measure(
        *,
        run_id: str,
        instances: list[NormalizedInstance],
        roi_mask: NDArray[np.bool_],
        scale_nm_per_pixel: float | None,
        config: MorphometryConfig,
    ) -> MorphometryResult:
        captured_morphometry.append((config, scale_nm_per_pixel))
        return measure_particles(
            run_id=run_id,
            instances=instances,
            roi_mask=roi_mask,
            scale_nm_per_pixel=scale_nm_per_pixel,
            config=config,
        )

    monkeypatch.setattr(analysis_application_module, "measure", recording_measure)
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=HoleGateway(),
        postprocess_config=PostprocessProfile(fill_holes=False, connectivity=1),
        morphometry_config=MorphometryConfig(perimeter_neighborhood=4),
        quality_config=QualityGateConfig(
            foreground_ratio_review_low=0.1,
            foreground_ratio_warn_high=0.6,
            foreground_ratio_review_high=0.7,
        ),
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]

    # These are deliberately changed after the immutable run was created.
    service.postprocess_config = PostprocessProfile(fill_holes=True, connectivity=2)
    service.morphometry_config = MorphometryConfig(perimeter_neighborhood=8)
    service.quality_config = QualityGateConfig(foreground_ratio_review_low=0.0)
    with database.session() as session:
        image = session.get(ImageAssetRecord, "img_1")
        assert image is not None
        image.scale_nm_per_pixel = 9.0

    completed = service.execute_run(run_id)

    assert completed.configuration.provenance_status == "complete"
    assert completed.configuration.schema_version == 3
    assert completed.configuration.model_bundle == _model_bundle(HoleGateway().model)
    assert completed.configuration.image_sha256 is not None
    assert completed.configuration.scale_nm_per_pixel == 1.0
    assert completed.configuration.resolved_postprocess == PostprocessProfile(
        profile_id="default_v1",
        fill_holes=False,
        connectivity=1,
    )
    assert completed.configuration.resolved_morphometry == MorphometryConfig(
        perimeter_neighborhood=4
    )
    assert completed.configuration.resolved_quality_gate is not None
    assert completed.configuration.resolved_quality_gate.foreground_ratio_review_low == 0.1
    assert completed.quality is not None
    assert "foreground_ratio_too_low" in completed.quality.reasons
    assert completed.summary is not None
    assert completed.summary.coverage_ratio == pytest.approx(99 / 4096)
    assert completed.summary.mean_equivalent_diameter_nm == pytest.approx(
        completed.summary.mean_equivalent_diameter_px
    )
    assert captured_morphometry == [(MorphometryConfig(perimeter_neighborhood=4), 1.0)]

    with factory() as uow:
        run_config_path = uow.repositories.runs.get_artifact_paths(run_id)["run_config_path"]
    assert run_config_path is not None
    payload = json.loads((file_store.paths.root / run_config_path).read_text(encoding="utf-8"))
    assert payload["provenance_status"] == "complete"
    assert payload["resolved_postprocess"]["fill_holes"] is False
    assert payload["resolved_morphometry"]["perimeter_neighborhood"] == 4
    assert payload["resolved_quality_gate"]["foreground_ratio_review_low"] == 0.1


def test_schema_v3_build_mismatch_fails_before_adapter_load(
    application_harness: ApplicationHarness,
    tmp_path: Path,
) -> None:
    database, file_store, factory = application_harness
    artifact_root = tmp_path / "mismatched-build-model"
    artifact_root.mkdir()
    entry = model_entry(artifact_root, "unet-general-balanced-v1")
    load_calls: list[str] = []

    class NeverLoadedAdapter(FakeAdapter):
        def load(self, device: str) -> None:
            load_calls.append(device)
            super().load(device)

    registry = build_registry(
        artifact_root,
        [entry],
        resolver=lambda _: NeverLoadedAdapter,
    )
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=InferenceGateway(registry),
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with database.session() as session:
        record = session.get(SegmentationRun, run_id)
        assert record is not None
        payload = dict(record.run_config_json)
        creator_build = dict(payload["execution_build"])
        creator_build["application_source_sha256"] = "0" * 64
        payload["execution_build"] = creator_build
    with database.engine.begin() as connection:
        connection.execute(
            update(SegmentationRun)
            .where(SegmentationRun.run_id == run_id)
            .values(run_config_json=payload)
        )

    with pytest.raises(ExecutionBuildMismatchError) as captured:
        service.execute_run(run_id)

    assert captured.value.details["mismatched_fields"] == [
        "application_source_sha256"
    ]
    assert load_calls == []
    with database.session() as session:
        failed = session.get(SegmentationRun, run_id)
        assert failed is not None
        assert failed.status == JobStatus.FAILED.value
        assert failed.error_code == "EXECUTION_BUILD_MISMATCH"


def test_changed_original_image_fails_before_gateway_call(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    gateway = FakeGateway()
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with factory() as uow:
        storage_path = uow.repositories.images.get_storage_path("img_1")
    original_path = file_store.paths.require_managed(storage_path, must_exist=True)
    replacement = BytesIO()
    Image.new("L", (64, 64), color=99).save(replacement, format="PNG")
    file_store.atomic_write_bytes(original_path, replacement.getvalue())

    with pytest.raises(InputArtifactMismatchError) as captured:
        service.execute_run(run_id)

    assert captured.value.details["image_id"] == "img_1"
    assert gateway.predict_calls == 0
    with database.session() as session:
        failed = session.get(SegmentationRun, run_id)
        assert failed is not None
        assert failed.status == JobStatus.FAILED.value
        assert failed.error_code == "INPUT_ARTIFACT_MISMATCH"


@pytest.mark.parametrize("path_failure", ["missing", "outside_managed_root"])
def test_unreadable_original_path_fails_claimed_run_and_does_not_block_next_run(
    application_harness: ApplicationHarness,
    path_failure: str,
) -> None:
    database, file_store, factory = application_harness
    gateway = FakeGateway()
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )
    failed_run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with factory() as uow:
        storage_path = uow.repositories.images.get_storage_path("img_1")
    original_path = file_store.paths.require_managed(storage_path, must_exist=True)
    original_bytes = original_path.read_bytes()

    if path_failure == "missing":
        original_path.unlink()
    else:
        with database.session() as session:
            image = session.get(ImageAssetRecord, "img_1")
            assert image is not None
            image.storage_path = "../outside-managed-root.png"

    with pytest.raises(StoragePathError):
        service.execute_run(failed_run_id)

    with database.session() as session:
        failed = session.get(SegmentationRun, failed_run_id)
        assert failed is not None
        assert failed.status == JobStatus.FAILED.value
        assert failed.error_code == "INFERENCE_FAILED"

    if path_failure == "missing":
        file_store.atomic_write_bytes(original_path, original_bytes)
    else:
        with database.session() as session:
            image = session.get(ImageAssetRecord, "img_1")
            assert image is not None
            image.storage_path = storage_path

    next_run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    assert service.execute_run(next_run_id).status == JobStatus.COMPLETED
    assert gateway.predict_calls == 1


@pytest.mark.parametrize(
    "missing_field",
    [
        "adapter_path",
        "weight_sha256",
        "config_sha256",
        "model_card_sha256",
        "adapter_sha256",
        "model_bundle",
        "image_sha256",
        "resolved_postprocess",
        "resolved_morphometry",
        "resolved_quality_gate",
        "execution_build",
    ],
)
def test_complete_configuration_rejects_missing_provenance_field(
    application_harness: ApplicationHarness,
    missing_field: str,
) -> None:
    _database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(),
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with factory() as uow:
        payload = uow.repositories.runs.get(run_id).configuration.model_dump(mode="python")
    payload.pop(missing_field)

    with pytest.raises(ValidationError, match="complete run provenance"):
        RunConfiguration.model_validate(payload)


def test_schema_v1_run_uses_explicit_legacy_fallback_and_review_upgrades_snapshot(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(),
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with database.session() as session:
        record = session.get(SegmentationRun, run_id)
        assert record is not None
        legacy = dict(record.run_config_json)
        legacy["schema_version"] = 1
        for key in (
            "provenance_status",
            "provenance_warnings",
            "image_sha256",
            "scale_nm_per_pixel",
            "resolved_postprocess",
            "resolved_morphometry",
            "resolved_quality_gate",
            "execution_build",
        ):
            legacy.pop(key, None)
    with database.engine.begin() as connection:
        connection.execute(
            update(SegmentationRun)
            .where(SegmentationRun.run_id == run_id)
            .values(run_config_json=legacy)
        )
        connection.execute(
            update(ImageAssetRecord)
            .where(ImageAssetRecord.image_id == "img_1")
            .values(scale_nm_per_pixel=9.0)
        )

    completed = service.execute_run(run_id)

    assert completed.configuration.provenance_status == "legacy_fallback"
    assert completed.configuration.provenance_warnings == [
        "legacy_run_configuration_incomplete"
    ]
    assert completed.summary is not None
    assert completed.summary.mean_equivalent_diameter_nm is None
    assert completed.quality is not None
    assert "legacy_run_configuration_fallback" in completed.quality.reasons
    assert "legacy_physical_scale_not_frozen_pixel_metrics_only" in completed.quality.reasons

    review_id = service.create_review_run(run_id, ReviewRunRequest(threshold=0.7))
    with factory() as uow:
        review = uow.repositories.runs.get(review_id)
    assert review.configuration.schema_version == 2
    assert review.configuration.provenance_status == "complete"
    assert review.configuration.image_sha256 is not None
    assert review.configuration.scale_nm_per_pixel == 9.0
    assert review.configuration.resolved_postprocess is not None
    assert review.configuration.resolved_morphometry is not None
    assert review.configuration.resolved_quality_gate is not None
    assert review.configuration.provenance_warnings == [
        "review_configuration_resolved_from_legacy_parent"
    ]


def test_create_runs_persists_two_by_three_cartesian_product_without_duplicate_dispatch(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    gateway = CartesianGateway()
    dispatcher = RecordingDispatcher()

    second_image = BytesIO()
    Image.new("L", (64, 64), color=11).save(second_image, format="PNG")
    second_image.seek(0)
    stored = file_store.save_upload(
        "job_1",
        second_image,
        "second.png",
        image_id="img_2",
    )
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        repositories.images.add_many(
            [
                StoredImageAsset(
                    storage_path=stored.relative_path,
                    asset=ImageAssetDTO(
                        image_id="img_2",
                        job_id="job_1",
                        filename="second.png",
                        sha256=stored.sha256,
                        width=64,
                        height=64,
                        bit_depth=8,
                        sample_id="sample_2",
                        material_formula="SiO2",
                        scale_nm_per_pixel=0.5,
                        analysis_roi=AnalysisROI(
                            valid_rect=PixelRect(x1=0, y1=0, x2=60, y2=62),
                            source="detected",
                        ),
                    ),
                )
            ]
        )
        session.add_all(
            [
                ModelRegistryRecord(
                    model_id=model.model_id,
                    family=model.family.value,
                    variant=model.variant.value,
                    quality_tier=model.quality_tier.value,
                    version=model.version,
                    adapter=model.adapter_path or "tests.fake:MissingAdapter",
                    status=model.status.value,
                )
                for model in gateway.models[1:]
            ]
        )
        repositories.boxes.replace(
            "img_1",
            0,
            [ROIBox(box_id="box_img_1_old", x1=0, y1=0, x2=40, y2=40)],
        )
        img_1_boxes = repositories.boxes.replace(
            "img_1",
            1,
            [
                ROIBox(
                    box_id="box_img_1_frozen",
                    label="image one frozen ROI",
                    x1=4,
                    y1=5,
                    x2=44,
                    y2=45,
                )
            ],
        )
        img_2_boxes = repositories.boxes.replace(
            "img_2",
            0,
            [
                ROIBox(
                    box_id="box_img_2_frozen",
                    label="image two frozen ROI",
                    x1=10,
                    y1=10,
                    x2=50,
                    y2=50,
                )
            ],
        )

    image_ids = ["img_2", "img_1"]
    model_ids = [model.model_id for model in reversed(gateway.models)]
    requested_inference = InferenceOptions(
        min_area_px=17,
        watershed_enabled=True,
        exclude_border=False,
        device=DevicePreference.CPU,
        seed=314,
    )
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
        dispatcher=dispatcher,
    )

    run_ids = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=image_ids,
            model_ids=model_ids,
            roi_mode=RoiMode.BOXES,
            box_revisions={"img_1": 2, "img_2": 1},
            inference=requested_inference,
        ),
    )

    assert len(run_ids) == 6
    assert len(set(run_ids)) == 6
    assert dispatcher.submissions == run_ids
    assert len(set(dispatcher.submissions)) == 6

    frozen_models = {model.model_id: model for model in gateway.models}
    frozen_boxes = {
        "img_1": img_1_boxes,
        "img_2": img_2_boxes,
    }
    frozen_analysis_rois = {
        "img_1": AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=64, y2=64)),
        "img_2": AnalysisROI(
            valid_rect=PixelRect(x1=0, y1=0, x2=60, y2=62),
            source="detected",
        ),
    }
    gateway.models = [
        model.model_copy(
            update={
                "version": "changed-after-creation",
                "adapter_path": "tests.fake:ChangedAdapter",
                "weight_sha256": "0" * 64,
            }
        )
        for model in gateway.models
    ]
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        repositories.boxes.replace(
            "img_1",
            2,
            [ROIBox(box_id="box_img_1_new", x1=12, y1=12, x2=52, y2=52)],
        )
        persisted_runs = [repositories.runs.get(run_id) for run_id in run_ids]

    expected_pairs = [
        (image_id, model_id)
        for image_id in image_ids
        for model_id in model_ids
    ]
    assert [(run.image_id, run.model_id) for run in persisted_runs] == expected_pairs
    expected_revisions = {"img_1": 2, "img_2": 1}
    for run in persisted_runs:
        model = frozen_models[run.model_id]
        box_set = frozen_boxes[run.image_id]
        assert run.status == JobStatus.QUEUED
        assert [event.to_status for event in run.status_history] == [JobStatus.QUEUED]
        assert run.box_revision == expected_revisions[run.image_id]
        assert run.configuration.box_revision == expected_revisions[run.image_id]
        assert run.configuration.boxes == box_set.boxes
        assert run.configuration.analysis_roi == frozen_analysis_rois[run.image_id]
        assert run.configuration.model_id == model.model_id
        assert run.configuration.model_version == model.version
        assert run.configuration.adapter_path == model.adapter_path
        assert run.configuration.weight_sha256 == model.weight_sha256
        assert run.configuration.config_sha256 == model.config_sha256
        assert run.configuration.model_card_sha256 == model.model_card_sha256
        assert run.configuration.adapter_sha256 == model.adapter_sha256
        assert run.configuration.schema_version == 3
        assert run.configuration.model_bundle == _model_bundle(model)
        assert run.configuration.preprocess_profile == model.preprocess_profile
        assert run.configuration.postprocess_profile == model.postprocess_profile
        assert run.threshold == model.default_threshold
        assert run.inference.threshold == model.default_threshold
        assert run.configuration.inference == run.inference
        assert run.inference.model_copy(update={"threshold": None}) == requested_inference


def test_border_exclusion_preserves_prefilter_quality_diagnostics(
    application_harness: ApplicationHarness,
) -> None:
    _database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=BoundaryGateway(),
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
            inference=InferenceOptions(min_area_px=8, exclude_border=True),
        ),
    )[0]

    completed = service.execute_run(run_id)

    assert completed.status == JobStatus.COMPLETED_WITH_WARNINGS
    assert completed.summary is not None
    assert completed.summary.particle_count == 0
    assert completed.quality is not None
    assert completed.quality.metrics["edge_touch_ratio"] == 1.0
    assert completed.quality.metrics["candidate_instance_count"] == 1
    assert completed.quality.metrics["boundary_instance_count"] == 1
    assert completed.quality.metrics["excluded_border_instance_count"] == 1
    assert "roi_edge_truncation" in completed.quality.reasons
    with factory() as uow:
        path = uow.repositories.runs.get_artifact_paths(run_id)["instances_path"]
    assert path is not None
    payload = json.loads((file_store.paths.root / path).read_text(encoding="utf-8"))
    assert payload["instance_count"] == 0


def test_execution_failure_is_isolated_and_persisted(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(fail=True),
    )
    run_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    with pytest.raises(RuntimeError, match="fixture inference failed"):
        service.execute_run(run_id)
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        failed = repositories.runs.get(run_id)
        assert failed.status == JobStatus.FAILED
        assert failed.error_code == "INFERENCE_FAILED"
        assert repositories.jobs.get("job_1").status == JobStatus.FAILED


def test_create_analysis_validates_and_persists_upload(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisCreationService(uow_factory=factory, file_store=file_store)
    content = BytesIO()
    Image.new("I;16", (17, 13), color=250).save(content, format="PNG")
    content.seek(0)

    detail = service.create_analysis(
        CreateAnalysisMetadata(
            job_name="fresh analysis",
            images=[
                ImageMetadataInput(
                    filename="fresh.png",
                    sample_id="sample_fresh",
                    material_formula="SiO2",
                    scale=ScaleInput(mode=ScaleMode.NM_PER_PIXEL, value=0.75),
                )
            ],
        ),
        [AnalysisUpload(filename="fresh.png", stream=content)],
    )

    assert detail.job.status == JobStatus.READY_FOR_CONFIGURATION
    assert len(detail.images) == 1
    image = detail.images[0]
    assert (image.width, image.height, image.bit_depth) == (17, 13, 16)
    assert image.scale_nm_per_pixel == 0.75
    assert image.analysis_roi.valid_rect == PixelRect(x1=0, y1=0, x2=17, y2=13)
    assert image.original_download_url is not None
    assert file_store.paths.job_config(detail.job.job_id).is_file()
    assert file_store.paths.job_manifest(detail.job.job_id).is_file()
    assert file_store.paths.image_metadata(detail.job.job_id, image.image_id).is_file()
    assert file_store.paths.boxes_revision(detail.job.job_id, image.image_id, 0).is_file()
    with database.session() as session:
        persisted = SqlAlchemyRepositorySet(session).images.get(image.image_id)
    assert persisted.sha256 == image.sha256


def test_create_analysis_rejects_metadata_upload_mismatch(
    application_harness: ApplicationHarness,
) -> None:
    _database, file_store, factory = application_harness
    service = AnalysisCreationService(uow_factory=factory, file_store=file_store)
    metadata = CreateAnalysisMetadata(
        job_name="mismatch",
        images=[ImageMetadataInput(filename="declared.png", sample_id="sample_1")],
    )

    with pytest.raises(InvalidImageError) as captured:
        service.create_analysis(
            metadata,
            [AnalysisUpload(filename="actual.png", stream=BytesIO(b"not-read"))],
        )

    assert captured.value.details["missing_uploads"] == ["declared.png"]
    assert captured.value.details["missing_metadata"] == ["actual.png"]


def test_create_analysis_failure_removes_only_unreferenced_uploads(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisCreationService(uow_factory=factory, file_store=file_store)
    image_bytes = BytesIO()
    Image.new("L", (19, 13), color=42).save(image_bytes, format="PNG")
    payload = image_bytes.getvalue()
    existing_jobs = set(file_store.paths.root.glob("job_*"))

    with pytest.raises(InvalidImageError, match="重复上传"):
        service.create_analysis(
            CreateAnalysisMetadata(
                job_name="duplicate upload cleanup",
                images=[
                    ImageMetadataInput(filename="first.png", sample_id="sample_1"),
                    ImageMetadataInput(filename="second.png", sample_id="sample_2"),
                ],
            ),
            [
                AnalysisUpload(filename="first.png", stream=BytesIO(payload)),
                AnalysisUpload(filename="second.png", stream=BytesIO(payload)),
            ],
        )

    failed_jobs = set(file_store.paths.root.glob("job_*")) - existing_jobs
    assert len(failed_jobs) == 1
    failed_job_dir = failed_jobs.pop()
    assert not [path for path in (failed_job_dir / "input").rglob("*") if path.is_file()]
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        assert repositories.images.list_by_job(failed_job_dir.name) == []


def test_create_analysis_late_failure_retains_database_referenced_original(
    application_harness: ApplicationHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisCreationService(uow_factory=factory, file_store=file_store)
    image_bytes = BytesIO()
    Image.new("L", (17, 13), color=42).save(image_bytes, format="PNG")
    image_bytes.seek(0)

    def fail_report_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture report write failed")

    monkeypatch.setattr(file_store, "atomic_write_json", fail_report_write)
    with pytest.raises(OSError, match="fixture report write failed"):
        service.create_analysis(
            CreateAnalysisMetadata(
                job_name="referenced original retention",
                images=[ImageMetadataInput(filename="retained.png", sample_id="sample_1")],
            ),
            [AnalysisUpload(filename="retained.png", stream=image_bytes)],
        )

    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        failed_jobs = [
            job_dir.name
            for job_dir in file_store.paths.root.glob("job_*")
            if job_dir.name != "job_1"
        ]
        assert len(failed_jobs) == 1
        images = repositories.images.list_by_job(failed_jobs[0])
        assert len(images) == 1
        original_path = file_store.paths.require_managed(
            repositories.images.get_storage_path(images[0].image_id),
            must_exist=True,
        )
    assert original_path.is_file()


def test_review_creates_immutable_child_run(application_harness: ApplicationHarness) -> None:
    database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(),
    )
    parent_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    service.execute_run(parent_id)
    with database.session() as session:
        image = session.get(ImageAssetRecord, "img_1")
        assert image is not None
        image.scale_nm_per_pixel = 7.0
    service.postprocess_config = PostprocessProfile(fill_holes=False, connectivity=1)
    service.morphometry_config = MorphometryConfig(perimeter_neighborhood=4)
    service.quality_config = QualityGateConfig(foreground_ratio_review_low=0.2)

    child_id = service.create_review_run(
        parent_id,
        ReviewRunRequest(threshold=0.8, min_area_px=12, watershed_enabled=True),
    )

    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        parent = repositories.runs.get(parent_id)
        child = repositories.runs.get(child_id)
    assert parent.parent_run_id is None
    assert parent.threshold == 0.5
    assert child.parent_run_id == parent_id
    assert child.threshold == 0.8
    assert child.inference.min_area_px == 12
    assert child.inference.watershed_enabled is True
    assert child.configuration.analysis_roi == parent.configuration.analysis_roi
    assert child.configuration.provenance_status == "complete"
    assert child.configuration.scale_nm_per_pixel == 1.0
    assert child.configuration.resolved_postprocess is not None
    assert child.configuration.resolved_postprocess.fill_holes is True
    assert child.configuration.resolved_postprocess.connectivity == 2
    assert child.configuration.resolved_postprocess.min_area_px == 12
    assert child.configuration.resolved_morphometry == parent.configuration.resolved_morphometry
    assert child.configuration.resolved_quality_gate == parent.configuration.resolved_quality_gate

    completed_child = service.execute_run(child_id)
    assert completed_child.quality is not None
    assert "foreground_ratio_too_low" not in completed_child.quality.reasons


def test_review_reuses_frozen_bundle_after_registry_provenance_changes(
    application_harness: ApplicationHarness,
) -> None:
    _database, file_store, factory = application_harness
    gateway = FakeGateway()
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )
    parent_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    service.execute_run(parent_id)
    with factory() as uow:
        parent_configuration = uow.repositories.runs.get(parent_id).configuration
    gateway.model = gateway.model.model_copy(update={"config_sha256": "d" * 64})

    child_id = service.create_review_run(parent_id, ReviewRunRequest(threshold=0.8))

    with factory() as uow:
        child_configuration = uow.repositories.runs.get(child_id).configuration
    assert child_configuration.schema_version == 3
    assert child_configuration.config_sha256 == parent_configuration.config_sha256
    assert child_configuration.adapter_sha256 == parent_configuration.adapter_sha256
    assert child_configuration.model_bundle == parent_configuration.model_bundle


def test_corrected_mask_review_bypasses_model_and_preserves_hash(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    gateway = FakeGateway()
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=gateway,
    )
    parent_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    service.execute_run(parent_id)

    corrected = np.zeros((64, 64), dtype=np.uint8)
    corrected[10:18, 11:19] = 255
    buffer = BytesIO()
    Image.fromarray(corrected).save(buffer, format="PNG")
    staged_path = file_store.paths.job_dir("job_1") / "review-mask.png"
    file_store.atomic_write_bytes(staged_path, buffer.getvalue())
    token = file_store.create_file_token(staged_path)

    gateway.fail = True
    child_id = service.create_review_run(
        parent_id,
        ReviewRunRequest(corrected_mask_token=token, min_area_px=4),
    )
    completed = service.execute_run(child_id)

    assert completed.status == JobStatus.COMPLETED_WITH_WARNINGS
    assert completed.summary is not None
    assert completed.summary.particle_count == 1
    assert completed.configuration.review_source == "corrected_mask"
    assert completed.configuration.corrected_mask_sha256 is not None
    with database.session() as session:
        persisted = SqlAlchemyRepositorySet(session).runs.get(child_id)
    assert persisted.configuration.corrected_mask_sha256 == (
        completed.configuration.corrected_mask_sha256
    )


def test_corrected_mask_review_never_deletes_a_referenced_original(
    application_harness: ApplicationHarness,
) -> None:
    database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(),
    )
    parent_id = service.create_runs(
        "job_1",
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["unet-general-balanced-v1"],
            roi_mode=RoiMode.FULL_IMAGE,
        ),
    )[0]
    service.execute_run(parent_id)
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        original_path = file_store.paths.require_managed(
            repositories.images.get_storage_path("img_1"), must_exist=True
        )
    original_bytes = original_path.read_bytes()
    original_token = file_store.create_file_token(original_path)

    service.create_review_run(
        parent_id,
        ReviewRunRequest(corrected_mask_token=original_token, min_area_px=4),
    )

    assert original_path.read_bytes() == original_bytes
    assert file_store.resolve_file_token(original_token) == original_path


def test_corrected_mask_rejects_wrong_dimensions_before_pixel_decode(
    application_harness: ApplicationHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _database, file_store, factory = application_harness
    service = AnalysisApplicationService(
        uow_factory=factory,
        file_store=file_store,
        inference_gateway=FakeGateway(),
    )
    staged_path = file_store.paths.job_dir("job_1") / "oversized-review.png"
    file_store.atomic_write_bytes(staged_path, b"header-only-test-double")
    token = file_store.create_file_token(staged_path)

    class WrongSizeImage:
        size = (50_001, 50_001)
        width = 50_001
        height = 50_001

        def __enter__(self) -> "WrongSizeImage":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            raise AssertionError("wrong-sized image must not be decoded")

    monkeypatch.setattr(Image, "open", lambda _path: WrongSizeImage())

    with pytest.raises(InvalidImageError) as exc_info:
        service._load_corrected_mask(token, expected_shape=(64, 64))

    assert exc_info.value.details["reason"] == "corrected_mask_shape_mismatch"
