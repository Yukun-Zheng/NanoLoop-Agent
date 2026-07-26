from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
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
from scripts.models import smoke_unet_agglomerated_analysis as smoke
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
            expected_input_width=80,
            expected_input_height=240,
            adapter_path="tests.fake:FakeAdapter",
            weight_sha256=TORCHSCRIPT_SHA256,
            config_sha256="b" * 64,
            model_card_sha256="c" * 64,
            adapter_sha256="d" * 64,
        )
        self.bundle = ModelBundleReference(
            bundle_id="e" * 64,
            manifest_ref=f"bundles/{'e' * 64}/manifest.json",
            weight_ref=f"{TORCHSCRIPT_SHA256}/weights.pt",
            config_ref=f"{'b' * 64}/config.yaml",
            model_card_ref=f"{'c' * 64}/model-card.md",
            adapter_ref=f"{'d' * 64}/adapter.py",
            adapter_sha256="d" * 64,
        )

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
        assert expected_model_version in {None, self.model.version}
        assert expected_adapter_path in {None, self.model.adapter_path}
        assert expected_weight_sha256 in {None, self.model.weight_sha256}
        assert expected_config_sha256 in {None, self.model.config_sha256}
        assert expected_model_card_sha256 in {None, self.model.model_card_sha256}
        assert expected_adapter_sha256 in {None, self.model.adapter_sha256}
        return self.bundle

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
        assert expected_adapter_sha256 == "d" * 64
        assert model_bundle == self.bundle
        mask = np.zeros((240, 80), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        mask_path = request.run_dir / "fake-mask.png"
        Image.fromarray(mask).save(mask_path)
        probability_path = request.run_dir / "probability.npy"
        np.save(probability_path, mask.astype(np.float32) / 255.0, allow_pickle=False)
        return SegmentationOutput(
            width=80,
            height=240,
            binary_mask_path=mask_path,
            probability_path=probability_path,
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


def _write_test_images(image_dir: Path, *, size: tuple[int, int] = (80, 240)) -> None:
    image_dir.mkdir()
    rows = np.arange(size[1], dtype=np.uint16)[:, None]
    columns = np.arange(size[0], dtype=np.uint16)[None, :]
    gray = (rows * 7 + columns * 11) % 251
    for filename in TEST_FILENAMES:
        Image.fromarray(gray.astype(np.uint8)).save(image_dir / filename)


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


def test_loads_frozen_runtime_settings_without_external_calibration_evidence() -> None:
    gateway = FakeGateway()

    frozen = load_calibrated_analysis(_config(), gateway.model)

    assert frozen.threshold == 0.25
    assert frozen.min_area_px == 1024
    assert frozen.bottom_crop_px == 130
    assert frozen.postprocess_profile().min_area_px == 1024
    assert frozen.morphometry_config().perimeter_neighborhood == 8


def test_rejects_runtime_config_that_differs_from_frozen_contract() -> None:
    gateway = FakeGateway()
    config = _config()
    config["calibrated_analysis"]["min_area_px"] = 2048

    with pytest.raises(ValueError, match="frozen agglomerated contract"):
        load_calibrated_analysis(config, gateway.model)


def test_full_image_a_only_run_generates_canonical_artifacts_and_report_zip(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "a-only-images"
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

    assert result["test_scope"] == (
        "Agglomerated-A runtime integration only; no GT or scientific evaluation"
    )
    assert "validation_images" not in result
    assert "scientific_results" not in result["runs"][0]
    assert len(gateway.requests) == 1
    run = result["runs"][0]
    assert run["roi"]["prediction_bottom_check"]["bottom_prediction_is_zero"] is True
    assert run["predict_summary"]["resolved_device"] == "cpu"
    assert all(run["artifact_linkage"].values())
    assert set(run["canonical_artifact_identities"]) == {
        "pred_mask_path",
        "probability_path",
        "instances_path",
        "particles_csv_path",
        "overlay_path",
        "labeled_particles_path",
        "image_summary_path",
        "quality_report_path",
        "execution_provenance_path",
        "run_config_path",
        "transform_path",
    }
    report = output_root / str(result["report_zip"]["path"])
    assert report.is_file()
    assert hashlib.sha256(report.read_bytes()).hexdigest() == result["report_zip"]["sha256"]
    assert all(result["report_zip"]["binding"].values())
    with zipfile.ZipFile(report) as archive:
        assert "export_manifest.json" in archive.namelist()
        assert archive.testzip() is None


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


def test_cli_is_a_only_and_validates_fixed_input_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "images"
    registry = tmp_path / "registry.yaml"
    output_root = tmp_path / "output"
    _write_test_images(image_dir, size=EXPECTED_IMAGE_SIZE)
    registry.write_text("models: []\n", encoding="utf-8")
    digest = hashlib.sha256((image_dir / TEST_FILENAMES[0]).read_bytes()).hexdigest()
    monkeypatch.setattr(smoke, "INPUT_SHA256", digest)
    parser = build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--threshold-evidence" not in options
    assert "--min-area-evidence" not in options
    assert all("mask" not in option for option in options)
    namespace = argparse.Namespace(
        image_dir=image_dir,
        registry=registry,
        output_root=output_root,
        model_id=MODEL_ID,
    )

    validated = _validated_parameters(namespace)

    assert validated.image_dir == image_dir.resolve()

    monkeypatch.setattr(smoke, "INPUT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        _validated_parameters(namespace)


def test_cli_rejects_public_registry_and_independent_test_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "test_images"
    registry = tmp_path / "private-registry.yaml"
    _write_test_images(image_dir, size=EXPECTED_IMAGE_SIZE)
    registry.write_text("models: []\n", encoding="utf-8")
    digest = hashlib.sha256((image_dir / TEST_FILENAMES[0]).read_bytes()).hexdigest()
    monkeypatch.setattr(smoke, "INPUT_SHA256", digest)
    namespace = argparse.Namespace(
        image_dir=image_dir,
        registry=registry,
        output_root=tmp_path / "output",
        model_id=MODEL_ID,
    )
    with pytest.raises(ValueError, match="independent test directory"):
        _validated_parameters(namespace)

    namespace.image_dir = tmp_path / "a-only-input"
    _write_test_images(namespace.image_dir, size=EXPECTED_IMAGE_SIZE)
    namespace.registry = Path(__file__).resolve().parents[3] / "model_artifacts" / "registry.yaml"
    with pytest.raises(ValueError, match="private registry"):
        _validated_parameters(namespace)


def test_bottom_prediction_check_rejects_nonzero_pixels(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((240, 80), dtype=np.uint8)
    mask[-1, -1] = 255
    Image.fromarray(mask).save(mask_path)

    with pytest.raises(RuntimeError, match="bottom-bar pixels"):
        _mask_bottom_evidence(mask_path, width=80, height=240, bottom_crop_px=130)


def test_preflight_private_registry_must_not_claim_ready(tmp_path: Path) -> None:
    registry = tmp_path / "private-preflight.yaml"
    registry.write_text(
        f"models:\n  - metadata:\n      model_id: {MODEL_ID}\n      status: ready\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remain unavailable before the smoke"):
        _ready_smoke_registry_entry(registry)


def test_ready_registry_exports_computed_adapter_source_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    config_path = repository_root / "model_artifacts/configs/unet-agglomerated-specialized-v1.yaml"
    model_card_path = (
        repository_root / "model_artifacts/model_cards/unet-agglomerated-specialized-v1.md"
    )
    adapter_path = repository_root / "app/inference/adapters/unet.py"
    for path, content in (
        (config_path, b"config"),
        (model_card_path, b"model card"),
        (adapter_path, b"adapter source"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    weight_path = tmp_path / "weights.pt"
    weight_path.write_bytes(b"torchscript")
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    registration = SimpleNamespace(
        metadata=SimpleNamespace(status=ModelStatus.UNAVAILABLE),
        weight_path=weight_path,
        weight_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        config_path=config_path,
        config={"calibrated_analysis": {}},
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        model_card_path=model_card_path,
        model_card_sha256=hashlib.sha256(model_card_path.read_bytes()).hexdigest(),
        adapter_source_path=adapter_path,
        adapter_sha256=adapter_sha256,
    )

    class FakeRegistry:
        registry_error = None

        @staticmethod
        def get_registration(_model_id: str) -> Any:
            return registration

    registry_path = tmp_path / "private-preflight.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "metadata": {"model_id": MODEL_ID, "status": "unavailable"},
                        "adapter_path": "app.inference.adapters.unet:UNetAdapter",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(smoke, "ModelRegistryService", lambda _path: FakeRegistry())
    monkeypatch.setattr(smoke, "_repository_root", lambda: repository_root)
    monkeypatch.setattr(smoke, "TORCHSCRIPT_SHA256", registration.weight_sha256)
    monkeypatch.setattr(smoke, "load_calibrated_analysis", lambda *_args: None)

    ready_entry = _ready_smoke_registry_entry(registry_path)

    assert ready_entry["adapter_sha256"] == adapter_sha256
