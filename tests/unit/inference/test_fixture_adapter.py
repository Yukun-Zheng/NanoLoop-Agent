from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.contracts.analyses import ROIBox
from app.contracts.enums import (
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelMetadata
from app.inference.adapters.fixture import DeterministicFixtureAdapter
from app.inference.gateway import InferenceGateway
from app.inference.registry import ModelRegistryService


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="unet-deterministic-fixture-v1",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="fixture-1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.5,
        preprocess_profile="fixture-none-v1",
        postprocess_profile="semantic-mask-v1",
    )


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (80, 60), color=96).save(buffer, format="PNG")
    return buffer.getvalue()


def test_fixture_adapter_is_deterministic_and_honors_box_roi(tmp_path: Path) -> None:
    adapter = DeterministicFixtureAdapter(
        metadata=_metadata(),
        weight_path=Path("fixture.weights"),
        weight_bytes=b"NanoLoop deterministic fixture model v1\n",
        weight_sha256="a" * 64,
        config={
            "fixture_schema_version": 1,
            "objects": [
                {"center": [0.25, 0.5], "radius": [0.1, 0.15], "score": 0.9},
                {"center": [0.75, 0.5], "radius": [0.1, 0.15], "score": 0.8},
            ],
        },
    )
    adapter.load("cpu")
    request = SegmentationRequest(
        image_id="image-fixture",
        image_path=tmp_path / "absent-pinned-image",
        image_bytes=_image_bytes(),
        run_dir=tmp_path / "run",
        roi_mode=RoiMode.BOXES,
        boxes=[ROIBox(x1=0, y1=0, x2=40, y2=60, active=True)],
        threshold=0.5,
        min_area_px=8,
        device=DevicePreference.CPU,
        seed=42,
    )

    first = adapter.predict(request)
    first_probability = np.load(first.probability_path, allow_pickle=False)
    first_mask = np.asarray(Image.open(first.binary_mask_path), dtype=np.uint8)
    second = adapter.predict(request)
    second_probability = np.load(second.probability_path, allow_pickle=False)

    assert first.width == 80
    assert first.height == 60
    assert first.runtime_ms == 0
    assert first.warnings == ["simulated_fixture_output_not_scientific"]
    assert np.array_equal(first_probability, second_probability)
    assert np.any(first_mask[:, :40])
    assert not np.any(first_mask[:, 40:])

    adapter.unload()
    assert adapter.health().status == ModelStatus.READY


def test_demo_registry_freezes_and_executes_through_real_gateway(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    registry = ModelRegistryService(
        project_root / "demo_data" / "model_artifacts" / "registry.yaml",
        snapshot_root=tmp_path / "snapshots",
    )
    gateway = InferenceGateway(registry)
    metadata = registry.get_metadata("unet-deterministic-fixture-v1")

    assert metadata.status == ModelStatus.READY
    bundle = gateway.freeze_model_bundle(
        metadata.model_id,
        expected_model_version=metadata.version,
        expected_adapter_path=metadata.adapter_path,
        expected_weight_sha256=metadata.weight_sha256,
        expected_config_sha256=metadata.config_sha256,
        expected_model_card_sha256=metadata.model_card_sha256,
        expected_adapter_sha256=metadata.adapter_sha256,
    )
    output = gateway.predict(
        metadata.model_id,
        SegmentationRequest(
            image_id="image-gateway-fixture",
            image_path=tmp_path / "absent-pinned-image",
            image_bytes=_image_bytes(),
            run_dir=tmp_path / "gateway-run",
            roi_mode=RoiMode.FULL_IMAGE,
            device=DevicePreference.CPU,
        ),
        expected_model_version=metadata.version,
        expected_adapter_path=metadata.adapter_path,
        expected_weight_sha256=metadata.weight_sha256,
        expected_config_sha256=metadata.config_sha256,
        expected_model_card_sha256=metadata.model_card_sha256,
        expected_adapter_sha256=metadata.adapter_sha256,
        model_bundle=bundle,
    )

    assert output.binary_mask_path.is_file()
    assert output.execution is not None
    assert output.execution.actual_device == "cpu"
    assert output.execution.backend.endswith(".DeterministicFixtureAdapter")
    assert output.warnings == ["simulated_fixture_output_not_scientific"]
