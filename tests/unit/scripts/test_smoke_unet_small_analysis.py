from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.enums import (
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
)
from app.contracts.execution import InferenceExecutionEvidence
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelHealth, ModelMetadata
from app.core.config import Settings
from app.db.base import Base
from app.db.models import ModelRegistryRecord
from app.db.session import Database
from app.storage import LocalFileStore, StoragePaths
from scripts.models.smoke_unet_small_analysis import (
    SmokeParameters,
    _validated_parameters,
    execute_analysis,
)


class FakeGateway:
    def __init__(self) -> None:
        self.model = ModelMetadata(
            model_id="unet-small-balanced-v1",
            family=ModelFamily.UNET,
            variant=ModelVariant.SMALL_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1.0.0",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.30,
            preprocess_profile="unet_small_gray_v1",
            postprocess_profile="unet_small_components_v1",
            inference_invalid_bottom_px=130,
            expected_input_width=40,
            expected_input_height=200,
            adapter_path="tests.fake:FakeAdapter",
            weight_sha256="a" * 64,
            config_sha256="b" * 64,
            model_card_sha256="c" * 64,
            adapter_sha256="d" * 64,
        )
        self.bundle = ModelBundleReference(
            bundle_id="e" * 64,
            manifest_ref=f"bundles/{'e' * 64}/manifest.json",
            weight_ref=f"{'a' * 64}/weights.pt",
            config_ref=f"{'b' * 64}/config.yaml",
            model_card_ref=f"{'c' * 64}/model-card.md",
            adapter_ref=f"{'d' * 64}/adapter.py",
            adapter_sha256="d" * 64,
        )

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        assert not only_ready or self.model.status == ModelStatus.READY
        return [self.model]

    def health(self) -> list[ModelHealth]:
        return [
            ModelHealth(
                model_id=self.model.model_id,
                status=ModelStatus.READY,
                weight_sha256=self.model.weight_sha256,
            )
        ]

    def freeze_model_bundle(self, model_id: str, **expected: Any) -> ModelBundleReference:
        assert model_id == self.model.model_id
        assert expected["expected_model_version"] == self.model.version
        assert expected["expected_adapter_path"] == self.model.adapter_path
        assert expected["expected_weight_sha256"] == self.model.weight_sha256
        assert expected["expected_config_sha256"] == self.model.config_sha256
        assert expected["expected_model_card_sha256"] == self.model.model_card_sha256
        assert expected["expected_adapter_sha256"] == self.model.adapter_sha256
        return self.bundle

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
        assert request.threshold == pytest.approx(0.30)
        assert request.min_area_px == 64
        assert request.device == DevicePreference.CPU
        assert request.seed == 2026
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 == self.model.adapter_sha256
        assert model_bundle == self.bundle
        mask = np.zeros((200, 40), dtype=np.uint8)
        mask[10:25, 10:25] = 255
        mask_path = request.run_dir / "fake-mask.png"
        Image.fromarray(mask).save(mask_path)
        return SegmentationOutput(
            width=40,
            height=200,
            binary_mask_path=mask_path,
            runtime_ms=1,
            execution=InferenceExecutionEvidence(
                actual_device="cpu",
                python_random_seeded=True,
                numpy_random_seeded=True,
                torch_deterministic_algorithms=True,
                global_inference_serialized=True,
                backend="app.inference.adapters.unet.UNetAdapter",
            ),
        )


def _write_test_image(path: Path) -> None:
    rows = np.arange(200, dtype=np.uint8)[:, None]
    columns = np.arange(40, dtype=np.uint8)[None, :]
    gray = (rows * 7 + columns * 11) % 251
    rgb = np.stack((gray, (gray + 37) % 251, (gray + 83) % 251), axis=2)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def test_execute_analysis_freezes_calibration_and_excludes_bottom_bar(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    registry_path = tmp_path / "private-registry.yaml"
    output_root = tmp_path / "smoke-output"
    artifact_root = output_root / "artifacts"
    _write_test_image(image_path)
    registry_path.write_text("models: []\n", encoding="utf-8")
    output_root.mkdir()
    database = Database(
        Settings(
            app_env="test",
            database_url=f"sqlite:///{(output_root / 'analysis.sqlite3').as_posix()}",
            output_root=artifact_root,
        )
    )
    Base.metadata.create_all(database.engine)
    gateway = FakeGateway()
    with database.session() as session:
        session.add(
            ModelRegistryRecord(
                model_id=gateway.model.model_id,
                family=gateway.model.family.value,
                variant=gateway.model.variant.value,
                quality_tier=gateway.model.quality_tier.value,
                version=gateway.model.version,
                adapter=gateway.model.adapter_path or "tests.fake:FakeAdapter",
                status=ModelStatus.READY.value,
            )
        )
    try:
        result = execute_analysis(
            SmokeParameters(
                image=image_path,
                registry=registry_path,
                output_root=output_root,
                scale_nm_per_pixel=0.5,
                sample_id="sample-1",
            ),
            database=database,
            file_store=LocalFileStore(
                StoragePaths(artifact_root),
                max_upload_bytes=image_path.stat().st_size,
            ),
            gateway=gateway,
        )
        assert result["evidence_class"] == "engineering_acceptance"
        assert result["readiness_eligible"] is True
        assert result["scientific_acceptance_eligible"] is False
        assert result["limitations"]
        assert result["model"]["bundle"] == gateway.bundle.model_dump(mode="json")
        assert result["model"]["adapter_sha256"] == gateway.model.adapter_sha256
    finally:
        database.dispose()

    frozen = result["frozen_inference"]
    assert frozen == {
        "threshold": 0.30,
        "min_area_px": 64,
        "watershed_enabled": False,
        "exclude_border": True,
        "device": "cpu",
        "seed": 2026,
    }
    assert result["resolved_postprocess"]["min_area_px"] == 64
    assert result["scale_nm_per_pixel"] == 0.5
    assert result["roi"]["effective_roi_area_px"] == 40 * (200 - 130)
    exclusion = result["roi"]["bottom_exclusion"]
    assert exclusion["matching_model_bottom_region_present"] is True
    assert exclusion["expected_bottom_area_px"] == 40 * 130
    statistics = result["scientific_results"]
    assert statistics["mean_equivalent_diameter_nm"] is not None
    assert statistics["number_density_um2"] is not None
    assert statistics["perimeter_density_um"] is not None
    assert result["artifacts"]["mask"] is not None
    assert result["artifacts"]["execution_evidence"] is not None


@pytest.mark.parametrize("protected_kind", ["existing", "repository"])
def test_output_root_protection(tmp_path: Path, protected_kind: str) -> None:
    image = tmp_path / "sample.png"
    registry = tmp_path / "registry.yaml"
    _write_test_image(image)
    registry.write_text("models: []\n", encoding="utf-8")
    output_root = (
        tmp_path / "already-exists"
        if protected_kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-smoke-output"
    )
    if protected_kind == "existing":
        output_root.mkdir()
    namespace = argparse.Namespace(
        image=image,
        registry=registry,
        output_root=output_root,
        scale_nm_per_pixel=0.5,
        pixel_only=False,
        sample_id="sample-1",
        model_id="unet-small-balanced-v1",
        threshold=0.30,
        min_area_px=64,
        seed=2026,
        device="cpu",
    )

    with pytest.raises(ValueError, match="output-root"):
        _validated_parameters(namespace)


def test_pixel_only_parameters_do_not_invent_a_physical_scale(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    registry = tmp_path / "registry.yaml"
    output_root = tmp_path / "smoke-output"
    _write_test_image(image)
    registry.write_text("models: []\n", encoding="utf-8")
    namespace = argparse.Namespace(
        image=image,
        registry=registry,
        output_root=output_root,
        scale_nm_per_pixel=None,
        pixel_only=True,
        sample_id="sample-1",
        model_id="unet-small-balanced-v1",
        threshold=0.30,
        min_area_px=64,
        seed=2026,
        device="cpu",
    )

    parameters = _validated_parameters(namespace)

    assert parameters.scale_nm_per_pixel is None
