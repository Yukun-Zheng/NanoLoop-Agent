from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.contracts.analyses import ROIBox
from app.contracts.enums import DevicePreference, ModelStatus, RoiMode
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelMetadata
from app.inference.gateway import InferenceGateway
from app.inference.registry import ModelRegistryService

MODEL_ID = "unet-small-balanced-v1"
EXPECTED_WIDTH = 2048
EXPECTED_HEIGHT = 1536
BOTTOM_CROP_PX = 130
SEED = 2026

PRIVATE_REGISTRY = os.environ.get("NANOLOOP_SMALL_PRIVATE_REGISTRY")
SMOKE_IMAGE = os.environ.get("NANOLOOP_SMALL_SMOKE_IMAGE")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not PRIVATE_REGISTRY or not SMOKE_IMAGE,
        reason=(
            "set NANOLOOP_SMALL_PRIVATE_REGISTRY and NANOLOOP_SMALL_SMOKE_IMAGE "
            "for controlled cloud validation"
        ),
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request(
    *,
    image: Path,
    run_dir: Path,
    roi_mode: RoiMode,
    boxes: list[ROIBox] | None = None,
) -> SegmentationRequest:
    return SegmentationRequest(
        image_id="small-cloud-validation",
        image_path=image,
        run_dir=run_dir,
        roi_mode=roi_mode,
        boxes=boxes or [],
        threshold=0.30,
        min_area_px=0,
        roi_context_px=16,
        device=DevicePreference.CPU,
        seed=SEED,
    )


def _frozen_expectations(metadata: ModelMetadata) -> dict[str, str | None]:
    return {
        "expected_model_version": metadata.version,
        "expected_adapter_path": metadata.adapter_path,
        "expected_weight_sha256": metadata.weight_sha256,
        "expected_config_sha256": metadata.config_sha256,
        "expected_model_card_sha256": metadata.model_card_sha256,
        "expected_adapter_sha256": metadata.adapter_sha256,
    }


def test_small_private_bundle_gateway_lifecycle_roi_and_determinism(tmp_path: Path) -> None:
    assert PRIVATE_REGISTRY is not None
    assert SMOKE_IMAGE is not None
    registry_path = Path(PRIVATE_REGISTRY).expanduser().resolve(strict=True)
    image_path = Path(SMOKE_IMAGE).expanduser().resolve(strict=True)
    assert registry_path.is_file()
    assert image_path.is_file()

    with Image.open(image_path) as image:
        assert image.size == (EXPECTED_WIDTH, EXPECTED_HEIGHT)

    registry = ModelRegistryService(registry_path, snapshot_root=tmp_path / "model-snapshots")
    assert registry.registry_error is None
    metadata = registry.get_metadata(MODEL_ID)
    assert metadata.status == ModelStatus.READY, metadata.health_error
    assert metadata.expected_input_width == EXPECTED_WIDTH
    assert metadata.expected_input_height == EXPECTED_HEIGHT
    assert metadata.weight_sha256 is not None
    assert metadata.config_sha256 is not None
    assert metadata.model_card_sha256 is not None
    assert metadata.adapter_sha256 is not None

    gateway = InferenceGateway(registry)
    before = next(item for item in gateway.health() if item.model_id == MODEL_ID)
    assert before.status == ModelStatus.READY
    bundle = gateway.freeze_model_bundle(MODEL_ID, **_frozen_expectations(metadata))

    first = gateway.predict(
        MODEL_ID,
        _request(
            image=image_path,
            run_dir=tmp_path / "full-first",
            roi_mode=RoiMode.FULL_IMAGE,
        ),
        model_bundle=bundle,
        **_frozen_expectations(metadata),
    )
    second = gateway.predict(
        MODEL_ID,
        _request(
            image=image_path,
            run_dir=tmp_path / "full-second",
            roi_mode=RoiMode.FULL_IMAGE,
        ),
        model_bundle=bundle,
        **_frozen_expectations(metadata),
    )

    assert first.width == second.width == EXPECTED_WIDTH
    assert first.height == second.height == EXPECTED_HEIGHT
    assert first.probability_path is not None
    assert second.probability_path is not None
    assert _sha256(first.probability_path) == _sha256(second.probability_path)
    assert _sha256(first.binary_mask_path) == _sha256(second.binary_mask_path)
    assert first.execution is not None
    assert first.execution.torch_deterministic_algorithms is True
    assert first.execution.global_inference_serialized is True

    full_probability = np.load(first.probability_path, allow_pickle=False)
    full_mask = np.asarray(Image.open(first.binary_mask_path), dtype=np.uint8)
    assert full_probability.shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert full_mask.shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert np.all(full_probability[-BOTTOM_CROP_PX:] == 0)
    assert np.all(full_mask[-BOTTOM_CROP_PX:] == 0)

    box = ROIBox(x1=512, y1=384, x2=1536, y2=1152, active=True)
    boxed = gateway.predict(
        MODEL_ID,
        _request(
            image=image_path,
            run_dir=tmp_path / "boxes",
            roi_mode=RoiMode.BOXES,
            boxes=[box],
        ),
        model_bundle=bundle,
        **_frozen_expectations(metadata),
    )
    assert boxed.probability_path is not None
    boxed_probability = np.load(boxed.probability_path, allow_pickle=False)
    boxed_mask = np.asarray(Image.open(boxed.binary_mask_path), dtype=np.uint8)
    exterior = np.ones((EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=bool)
    exterior[box.y1 : box.y2, box.x1 : box.x2] = False
    assert np.all(boxed_probability[exterior] == 0)
    assert np.all(boxed_mask[exterior] == 0)
    assert np.all(boxed_probability[-BOTTOM_CROP_PX:] == 0)
    assert np.all(boxed_mask[-BOTTOM_CROP_PX:] == 0)

    loaded = next(item for item in gateway.health() if item.model_id == MODEL_ID)
    assert loaded.status == ModelStatus.READY
    assert loaded.device == "cpu"
    gateway.cache.unload(MODEL_ID)
    assert gateway.cache.loaded() == []
    after_unload = next(item for item in gateway.health() if item.model_id == MODEL_ID)
    assert after_unload.status == ModelStatus.READY
    assert after_unload.device is None
