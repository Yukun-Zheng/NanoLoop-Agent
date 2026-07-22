from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text

from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    ImageAssetDTO,
    InferenceOptions,
    PixelRect,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.execution import InferenceExecutionEvidence
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import (
    ModelBundleReference,
    ModelCandidate,
    ModelHealth,
    ModelMetadata,
    ModelRecommendationRequest,
)
from app.contracts.repositories import StoredImageAsset
from app.core.config import Settings
from app.db.base import Base
from app.db.migration_state import expected_alembic_heads
from app.db.models import ModelRegistryRecord
from app.db.repositories import SqlAlchemyRepositorySet
from app.db.session import Database
from app.main import create_app
from app.storage import LocalFileStore, StoragePaths


class FakeInferenceGateway:
    def __init__(self) -> None:
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
            applicable_materials=["TiO2"],
            adapter_path="tests.fake:FakeAdapter",
            weight_sha256="a" * 64,
            config_sha256="b" * 64,
            model_card_sha256="c" * 64,
            adapter_sha256="d" * 64,
        )

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        if only_ready and self.model.status != ModelStatus.READY:
            return []
        return [self.model]

    def recommend(self, _request: ModelRecommendationRequest) -> list[ModelCandidate]:
        return [
            ModelCandidate(
                model_id=self.model.model_id,
                score=0.91,
                reasons=["general-purpose fixture"],
            )
        ]

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
        prefix = f"bundles/{model_id}"
        assert self.model.weight_sha256 is not None
        assert self.model.adapter_sha256 is not None
        return ModelBundleReference(
            bundle_id=self.model.weight_sha256,
            manifest_ref=f"{prefix}/manifest.json",
            weight_ref=f"{prefix}/weights.bin",
            config_ref=f"{prefix}/config.yaml",
            model_card_ref=f"{prefix}/model-card.md",
            adapter_ref=f"{prefix}/adapter.py",
            adapter_sha256=self.model.adapter_sha256,
        )

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
        assert model_id == self.model.model_id
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 == self.model.adapter_sha256
        assert model_bundle == self.freeze_model_bundle(
            model_id,
            expected_model_version=expected_model_version,
            expected_adapter_path=expected_adapter_path,
            expected_weight_sha256=expected_weight_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_model_card_sha256=expected_model_card_sha256,
            expected_adapter_sha256=expected_adapter_sha256,
        )
        image_source = (
            BytesIO(request.image_bytes) if request.image_bytes is not None else request.image_path
        )
        with Image.open(image_source) as image:
            width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)
        y1, y2 = max(1, height // 3), min(height - 1, height // 3 + 6)
        x1, x2 = max(1, width // 3), min(width - 1, width // 3 + 6)
        mask[y1:y2, x1:x2] = 255
        probability = (mask > 0).astype(np.float32) * 0.9
        mask_path = request.run_dir / "pred_mask.png"
        probability_path = request.run_dir / "probability.npy"
        Image.fromarray(mask).save(mask_path)
        np.save(probability_path, probability, allow_pickle=False)
        return SegmentationOutput(
            width=width,
            height=height,
            binary_mask_path=mask_path,
            probability_path=probability_path,
            runtime_ms=3,
            execution=InferenceExecutionEvidence(
                actual_device="cpu",
                python_random_seeded=True,
                numpy_random_seeded=True,
                torch_deterministic_algorithms=False,
                global_inference_serialized=True,
                backend="tests.fake:FakeAdapter",
            ),
        )


@dataclass(slots=True)
class ApiHarness:
    client: TestClient
    database: Database
    file_store: LocalFileStore
    gateway: FakeInferenceGateway
    download_token: str


@pytest.fixture
def api_harness(tmp_path: Path) -> ApiHarness:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        output_root=tmp_path / "outputs",
        model_registry_path=tmp_path / "registry.yaml",
        faiss_index_path=tmp_path / "faiss.index",
        log_level="WARNING",
    )
    database = Database(settings)
    Base.metadata.create_all(database.engine)
    _stamp_current_migration_head(database)
    _install_fts5(database)
    file_store = LocalFileStore(
        StoragePaths(settings.output_root),
        max_upload_bytes=1024 * 1024,
        token_secret="test-file-token-secret-that-is-long-enough",
    )
    _seed_persisted_read_models(database)
    artifact = settings.output_root / "job_1" / "exports" / "particles.csv"
    file_store.atomic_write_bytes(artifact, b"particle_id,area_px\np_1,12.5\n")
    download_token = file_store.create_file_token(artifact, ttl_seconds=3600)

    gateway = FakeInferenceGateway()
    app = create_app(
        settings=settings,
        database=database,
        file_store=file_store,
        inference_gateway=gateway,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield ApiHarness(
            client=client,
            database=database,
            file_store=file_store,
            gateway=gateway,
            download_token=download_token,
        )
    database.dispose()


def _stamp_current_migration_head(database: Database) -> None:
    heads = expected_alembic_heads()
    if len(heads) != 1:
        raise RuntimeError(f"API contract fixture requires one Alembic head, found {heads}")
    with database.engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:head)"),
            {"head": heads[0]},
        )


def _seed_persisted_read_models(database: Database) -> None:
    now = datetime.now(UTC)
    analysis_roi = AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=256, y2=200))
    with database.session() as session:
        repositories = SqlAlchemyRepositorySet(session)
        repositories.jobs.create(
            AnalysisJobDTO(
                job_id="job_1",
                name="contract fixture",
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
                        material_name="titanium dioxide",
                        material_formula="TiO2",
                        scale_nm_per_pixel=0.5,
                        analysis_roi=analysis_roi,
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
        repositories.runs.create_many(
            [
                SegmentationRunDTO(
                    run_id="run_1",
                    job_id="job_1",
                    image_id="img_1",
                    model_id="unet-general-balanced-v1",
                    # Keep this read-model fixture out of startup recovery so
                    # unrelated request-contract tests do not race a worker.
                    status=JobStatus.COMPLETED,
                    roi_mode=RoiMode.FULL_IMAGE,
                    threshold=0.5,
                    inference=InferenceOptions(threshold=0.5),
                    configuration=RunConfiguration(
                        model_id="unet-general-balanced-v1",
                        model_version="1.0.0",
                        roi_mode=RoiMode.FULL_IMAGE,
                        analysis_roi=analysis_roi,
                        inference=InferenceOptions(threshold=0.5),
                        preprocess_profile="sem_gray_v1",
                        postprocess_profile="default_v1",
                        created_at=now,
                    ),
                    created_at=now,
                    updated_at=now,
                )
            ]
        )


def _install_fts5(database: Database) -> None:
    statements = (
        """
        CREATE VIRTUAL TABLE knowledge_chunks_fts USING fts5(
            chunk_id UNINDEXED, text, section_title, material_tags
        )
        """,
        """
        CREATE TRIGGER knowledge_chunks_fts_insert AFTER INSERT ON knowledge_chunks BEGIN
            INSERT INTO knowledge_chunks_fts(chunk_id, text, section_title, material_tags)
            VALUES (new.chunk_id, new.text, new.section_title, new.material_tags_json);
        END
        """,
        """
        CREATE TRIGGER knowledge_chunks_fts_delete AFTER DELETE ON knowledge_chunks BEGIN
            DELETE FROM knowledge_chunks_fts WHERE chunk_id = old.chunk_id;
        END
        """,
        """
        CREATE TRIGGER knowledge_chunks_fts_update AFTER UPDATE ON knowledge_chunks BEGIN
            DELETE FROM knowledge_chunks_fts WHERE chunk_id = old.chunk_id;
            INSERT INTO knowledge_chunks_fts(chunk_id, text, section_title, material_tags)
            VALUES (new.chunk_id, new.text, new.section_title, new.material_tags_json);
        END
        """,
    )
    with database.engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)
