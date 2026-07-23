from __future__ import annotations

import hashlib
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import app.inference.registry as registry_module
from app.contracts.enums import ModelStatus
from app.inference.registry import ModelRegistryService

MODEL_ID = "unet-small-balanced-v1"
CHECKPOINT_SHA256 = "915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008"


def _artifact_root() -> Path:
    return Path(__file__).parents[3] / "model_artifacts"


def _registry_entry() -> dict[str, object]:
    payload = yaml.safe_load((_artifact_root() / "registry.yaml").read_text(encoding="utf-8"))
    return next(
        entry for entry in payload["models"] if entry["metadata"]["model_id"] == MODEL_ID
    )


def test_small_unet_config_freezes_confirmed_engineering_contract() -> None:
    config = yaml.safe_load(
        (_artifact_root() / "configs" / f"{MODEL_ID}.yaml").read_text(encoding="utf-8")
    )

    assert config == {
        "schema_version": "1",
        "loader": "torchscript",
        "input_channels": 1,
        "input_size": [256, 256],
        "expected_image_size": [1536, 2048],
        "patch_size": [256, 256],
        "stride": [128, 128],
        "tiling_padding": "reflect",
        "overlap_fusion": "uniform",
        "bottom_crop_px": 130,
        "pixel_scale": 255.0,
        "mean": [0.0],
        "std": [1.0],
        "output_activation": "logits",
        "threshold_comparison": "gt",
        "default_threshold": 0.30,
    }


def test_small_public_registry_remains_truthfully_unavailable() -> None:
    entry = _registry_entry()
    metadata = entry["metadata"]

    assert metadata["status"] == "unavailable"
    assert metadata["default_threshold"] == 0.30
    assert metadata["inference_invalid_bottom_px"] == 130
    assert metadata["expected_input_width"] == 2048
    assert metadata["expected_input_height"] == 1536
    assert metadata["metric_context"] == {
        "engineering_default_threshold": 0.30,
        "threshold_comparison": "gt",
        "scientific_calibration_status": "pending_small_b",
        "cloud_validation_status": "pending",
    }
    assert entry["adapter_path"] == "app.inference.adapters.unet:UNetAdapter"
    assert entry["weight_path"] == f"weights/{MODEL_ID}.pt"
    assert entry["weight_sha256"] is None
    assert entry["config_path"] == f"configs/{MODEL_ID}.yaml"
    assert entry["model_card_path"] == f"model_cards/{MODEL_ID}.md"
    assert metadata["health_error"] == "Private TorchScript is not bundled with this repository."


def test_small_external_bundle_resolves_assets_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _artifact_root()
    bundle = tmp_path / "private-model-bundle"
    (bundle / "weights").mkdir(parents=True)
    (bundle / "configs").mkdir()
    (bundle / "model_cards").mkdir()
    shutil.copy2(source / "configs" / f"{MODEL_ID}.yaml", bundle / "configs" / f"{MODEL_ID}.yaml")
    shutil.copy2(
        source / "model_cards" / f"{MODEL_ID}.md",
        bundle / "model_cards" / f"{MODEL_ID}.md",
    )
    weight_path = bundle / "weights" / f"{MODEL_ID}.pt"
    weight_bytes = b"external-small-torchscript-placeholder"
    weight_path.write_bytes(weight_bytes)

    entry = deepcopy(_registry_entry())
    metadata = entry["metadata"]
    assert isinstance(metadata, dict)
    metadata["status"] = "ready"
    metadata.pop("health_error", None)
    entry["weight_sha256"] = hashlib.sha256(weight_bytes).hexdigest()
    registry_path = bundle / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"schema_version": "2.0", "models": [entry]}, sort_keys=False),
        encoding="utf-8",
    )

    original_find_spec = registry_module.importlib.util.find_spec
    monkeypatch.setattr(
        registry_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else original_find_spec(name),
    )

    registry = ModelRegistryService(registry_path, snapshot_root=bundle / "snapshots")
    registration = registry.get_registration(MODEL_ID)
    assert registration.weight_path == weight_path
    assert registration.weight_sha256 == hashlib.sha256(weight_bytes).hexdigest()
    assert registration.metadata.status == ModelStatus.READY
    assert registration.metadata.expected_input_width == 2048
    assert registration.metadata.expected_input_height == 1536

    weight_path.write_bytes(b"tampered-small-torchscript-placeholder")
    mismatched = ModelRegistryService(
        registry_path,
        snapshot_root=bundle / "tampered-snapshots",
    ).get_metadata(MODEL_ID)
    assert mismatched.status == ModelStatus.UNAVAILABLE
    assert "weight sha256 mismatch" in (mismatched.health_error or "")


def test_small_model_card_records_identity_permissions_and_pending_cloud_validation() -> None:
    card = (_artifact_root() / "model_cards" / f"{MODEL_ID}.md").read_text(encoding="utf-8")
    normalized = " ".join(card.split())

    assert CHECKPOINT_SHA256 in card
    assert "郭境濠" in card
    assert "`small_batchnorm`" in card
    assert "128 keys" in card
    assert "`expected_image_size=[1536, 2048]`" in card
    assert "strict `probability > 0.30`" in card
    assert "not** a scientifically calibrated threshold" in card
    assert "pending cloud validation" in card
    assert "must not be distributed publicly" in normalized
