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
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelHealth, ModelMetadata
from app.core.config import Settings
from app.db.base import Base
from app.db.models import ModelRegistryRecord
from app.db.session import Database
from app.storage import LocalFileStore, StoragePaths
from scripts.models.smoke_unet_agglomerated_analysis import (
    EXPECTED_IMAGE_SIZE,
    MODEL_ID,
    TEST_FILENAMES,
    TORCHSCRIPT_SHA256,
    SmokeParameters,
    _mask_bottom_evidence,
    _ready_smoke_registry_entry,
    _validate_output_root,
    _validated_parameters,
    build_parser,
    execute_analyses,
    load_calibrated_analysis,
)


def _config() -> dict[str, Any]:
    return {
        "bottom_crop_px": 130,
        "threshold_comparison": "gte",
        "calibrated_analysis": {
            "threshold": 0.25,
            "threshold_comparison": "gte",
            "min_area_px": 1024,
            "min_area_nm2": 302.45746691871454,
            "min_area_equivalent_diameter_nm": 19.623985514704565,
            "watershed_enabled": False,
            "fill_holes": True,
            "exclude_border": True,
            "connectivity": 2,
            "perimeter_neighborhood": 8,
            "bottom_crop_px": 130,
            "scale_nm_per_pixel": 100 / 184,
        },
    }


class FakeGateway:
    freeze_model_bundle: Any = None

    def __init__(self) -> None:
        self.requests: list[SegmentationRequest] = []
        self.model = ModelMetadata(
            model_id=MODEL_ID,
            family=ModelFamily.UNET,
            variant=ModelVariant.DENSE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.25,
            preprocess_profile="sem-gray-p1-p99-crop-bottom-130-v1",
            postprocess_profile="semantic-agglomerate-mask-v1",
            inference_invalid_bottom_px=130,
            adapter_path="tests.fake:FakeAdapter",
            weight_sha256=TORCHSCRIPT_SHA256,
            config_sha256="b" * 64,
            model_card_sha256="c" * 64,
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
        self.requests.append(request)
        assert request.threshold == pytest.approx(0.25)
        assert request.min_area_px == 1024
        assert request.device == DevicePreference.CPU
        assert request.seed == 2026
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 is None
        assert model_bundle is None
        mask = np.zeros((240, 80), dtype=np.uint8)
        # A 40x40 component is above the frozen 1024 px minimum and remains
        # strictly inside the 80x110 effective ROI after the 130 px bottom crop.
        mask[10:50, 10:50] = 255
        mask_path = request.run_dir / "fake-mask.png"
        Image.fromarray(mask).save(mask_path)
        return SegmentationOutput(
            width=80,
            height=240,
            binary_mask_path=mask_path,
            runtime_ms=1,
        )


def _write_test_images(image_dir: Path, *, size: tuple[int, int] = (80, 240)) -> None:
    image_dir.mkdir()
    rows = np.arange(size[1], dtype=np.uint16)[:, None]
    columns = np.arange(size[0], dtype=np.uint16)[None, :]
    gray = (rows * 7 + columns * 11) % 251
    for index, filename in enumerate(TEST_FILENAMES):
        image = (gray + index) % 251
        Image.fromarray(image.astype(np.uint8)).save(image_dir / filename)


def _database(output_root: Path) -> Database:
    database = Database(
        Settings(
            app_env="test",
            database_url=f"sqlite:///{(output_root / 'analysis.sqlite3').as_posix()}",
            output_root=output_root / "artifacts",
        )
    )
    Base.metadata.create_all(database.engine)
    return database


def test_loads_all_frozen_agglomerated_scientific_parameters() -> None:
    gateway = FakeGateway()

    calibrated = load_calibrated_analysis(_config(), gateway.model)

    assert calibrated.threshold == 0.25
    assert calibrated.min_area_px == 1024
    assert calibrated.watershed_enabled is False
    assert calibrated.fill_holes is True
    assert calibrated.exclude_border is True
    assert calibrated.connectivity == 2
    assert calibrated.perimeter_neighborhood == 8
    assert calibrated.bottom_crop_px == 130
    assert calibrated.scale_nm_per_pixel == pytest.approx(100 / 184)
    assert calibrated.postprocess_profile().min_area_px == 1024
    assert calibrated.morphometry_config().perimeter_neighborhood == 8


def test_rejects_internally_inconsistent_calibrated_config() -> None:
    gateway = FakeGateway()
    config = _config()
    config["calibrated_analysis"]["min_area_nm2"] = 1.0

    with pytest.raises(ValueError, match="physical conversion is inconsistent"):
        load_calibrated_analysis(config, gateway.model)


def test_full_image_run_freezes_config_and_excludes_bottom_roi(tmp_path: Path) -> None:
    image_dir = tmp_path / "validation-images"
    output_root = tmp_path / "smoke-output"
    registry_path = tmp_path / "private-registry.yaml"
    output_root.mkdir()
    registry_path.write_text("models: []\n", encoding="utf-8")
    _write_test_images(image_dir)
    database = _database(output_root)
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
        result = execute_analyses(
            SmokeParameters(image_dir, registry_path, output_root),
            load_calibrated_analysis(_config(), gateway.model),
            database=database,
            file_store=LocalFileStore(
                StoragePaths(output_root / "artifacts"),
                max_upload_bytes=max(
                    (image_dir / filename).stat().st_size for filename in TEST_FILENAMES
                ),
            ),
            gateway=gateway,
        )
    finally:
        database.dispose()

    assert result["validation_images"] == list(TEST_FILENAMES)
    assert "no masks or YCu test data were read" in result["test_scope"]
    assert result["private_registry_ready_eligible"] is True
    assert len(result["runs"]) == 1
    assert len(gateway.requests) == 1
    for run in result["runs"]:
        assert run["frozen_inference"] == {
            "threshold": 0.25,
            "min_area_px": 1024,
            "watershed_enabled": False,
            "exclude_border": True,
            "device": "cpu",
            "seed": 2026,
        }
        assert run["resolved_postprocess"] == {
            "profile_id": "semantic-agglomerate-mask-v1",
            "min_area_px": 1024,
            "fill_holes": True,
            "watershed_enabled": False,
            "exclude_border": True,
            "connectivity": 2,
            "instance_iou_threshold": 0.7,
        }
        assert run["resolved_morphometry"] == {"perimeter_neighborhood": 8}
        assert run["roi"]["effective_roi_area_px"] == 80 * (240 - 130)
        bottom = run["roi"]["bottom_exclusion"]
        assert bottom["matching_model_bottom_region_present"] is True
        assert bottom["expected_bottom_area_px"] == 80 * 130
        assert run["roi"]["prediction_bottom_check"] == {
            "bottom_rows_checked": 130,
            "bottom_nonzero_pixels": 0,
            "bottom_prediction_is_zero": True,
        }
        density = run["roi"]["density_consistency"]
        assert density["all_particle_bboxes_within_effective_roi"] is True
        assert density["coverage_uses_effective_roi"] is True
        assert density["number_density_uses_effective_roi"] is True
        assert density["perimeter_density_uses_effective_roi"] is True
        assert run["scientific_results"]["particle_count"] == 1
        for artifact in (
            "mask",
            "instances",
            "particles_csv",
            "overlay",
            "labeled_particles",
            "report",
            "execution_evidence",
        ):
            assert run["artifacts"][artifact] is not None


@pytest.mark.parametrize("kind", ["existing", "repository"])
def test_output_root_protection(tmp_path: Path, kind: str) -> None:
    output_root = (
        tmp_path / "already-exists"
        if kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-agglomerated-smoke"
    )
    if kind == "existing":
        output_root.mkdir()

    with pytest.raises(ValueError, match="output-root"):
        _validate_output_root(output_root)


def test_cli_has_no_test_mask_input(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    registry = tmp_path / "registry.yaml"
    output_root = tmp_path / "output"
    _write_test_images(image_dir, size=EXPECTED_IMAGE_SIZE)
    registry.write_text("models: []\n", encoding="utf-8")
    parser = build_parser()
    assert all(
        "mask" not in option
        for action in parser._actions
        for option in action.option_strings
    )
    namespace = argparse.Namespace(
        image_dir=image_dir,
        registry=registry,
        output_root=output_root,
        model_id=MODEL_ID,
    )

    validated = _validated_parameters(namespace)

    assert validated.image_dir == image_dir.resolve()


def test_cli_rejects_public_registry_and_test_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "test_images"
    registry = tmp_path / "private-registry.yaml"
    _write_test_images(image_dir, size=EXPECTED_IMAGE_SIZE)
    registry.write_text("models: []\n", encoding="utf-8")
    namespace = argparse.Namespace(
        image_dir=image_dir,
        registry=registry,
        output_root=tmp_path / "output",
        model_id=MODEL_ID,
    )
    with pytest.raises(ValueError, match="independent test directory"):
        _validated_parameters(namespace)

    namespace.image_dir = tmp_path / "validation"
    _write_test_images(namespace.image_dir, size=EXPECTED_IMAGE_SIZE)
    namespace.registry = (
        Path(__file__).resolve().parents[3] / "model_artifacts" / "registry.yaml"
    )
    with pytest.raises(ValueError, match="private registry"):
        _validated_parameters(namespace)


def test_bottom_prediction_check_rejects_nonzero_pixels(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((240, 80), dtype=np.uint8)
    mask[-1, -1] = 255
    Image.fromarray(mask).save(mask_path)

    with pytest.raises(RuntimeError, match="bottom-bar pixels"):
        _mask_bottom_evidence(mask_path, width=80, height=240, bottom_crop_px=130)


def test_preflight_private_registry_must_not_be_ready(tmp_path: Path) -> None:
    registry = tmp_path / "private-preflight.yaml"
    registry.write_text(
        "models:\n"
        "  - metadata:\n"
        f"      model_id: {MODEL_ID}\n"
        "      status: ready\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remain unavailable before the smoke"):
        _ready_smoke_registry_entry(registry)
