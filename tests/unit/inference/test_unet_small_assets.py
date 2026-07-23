from __future__ import annotations

import hashlib
import json
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
DELIVERED_TORCHSCRIPT_SHA256 = "e31bd7100d410fe3af93041ccf6956e27d562214d9ddcb40ac76b905840d6d28"
RUNTIME_TORCHSCRIPT_SHA256 = "09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d"


def _artifact_root() -> Path:
    return Path(__file__).parents[3] / "model_artifacts"


def _registry_entry() -> dict[str, object]:
    payload = yaml.safe_load((_artifact_root() / "registry.yaml").read_text(encoding="utf-8"))
    return next(
        entry for entry in payload["models"] if entry["metadata"]["model_id"] == MODEL_ID
    )


def _delivery_audit() -> dict[str, object]:
    path = _artifact_root() / "evidence" / MODEL_ID / "delivery-audit-2026-07-23.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_small_runtime_weight_matches_compatible_export_identity() -> None:
    weight_path = _artifact_root() / "weights" / f"{MODEL_ID}.pt"

    assert weight_path.stat().st_size == 13_560_272
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == RUNTIME_TORCHSCRIPT_SHA256


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


def test_small_public_registry_is_runtime_ready_with_science_pending() -> None:
    entry = _registry_entry()
    metadata = entry["metadata"]

    assert metadata["status"] == "ready"
    assert metadata["default_threshold"] == 0.30
    assert metadata["inference_invalid_bottom_px"] == 130
    assert metadata["expected_input_width"] == 2048
    assert metadata["expected_input_height"] == 1536
    assert metadata["metric_context"] == {
        "engineering_default_threshold": 0.30,
        "threshold_comparison": "gt",
        "scientific_calibration_status": "pending_small_b",
        "runtime_asset_delivered": True,
        "runtime_asset_verified": True,
        "runtime_verification_device": "cpu",
        "compatible_torch_minimum_verified": "2.6.0",
        "current_torch_verified": "2.13.0",
        "target_linux_artifact_load_verified": True,
        "target_linux_torch_verified": "2.6.0+cpu",
        "checkpoint_strict_load_verified": True,
        "eager_torchscript_max_abs_diff": 0.0,
        "deterministic_repeat_max_abs_diff": 0.0,
        "evidence_bundle_delivered": False,
        "scientific_acceptance_status": "pending_small_b",
        "redistribution_permission_status": "project_owner_requested_repository_integration",
    }
    assert metadata["metrics"] == {}
    assert entry["adapter_path"] == "app.inference.adapters.unet:UNetAdapter"
    assert entry["weight_path"] == f"weights/{MODEL_ID}.pt"
    assert entry["weight_sha256"] == RUNTIME_TORCHSCRIPT_SHA256
    assert entry["config_path"] == f"configs/{MODEL_ID}.yaml"
    assert entry["model_card_path"] == f"model_cards/{MODEL_ID}.md"
    assert "health_error" not in metadata


def test_small_delivery_audit_records_compatibility_reexport_and_limits() -> None:
    audit = _delivery_audit()

    assert audit["schema_version"] == "1"
    assert audit["model_id"] == MODEL_ID
    assert audit["review_scope"] == "runtime_asset_and_engineering_contract_only"
    source_package = audit["source_package"]
    assert source_package == {
        "filename": "ModelAssets-small-a.zip",
        "size_bytes": 24_964_343,
        "sha256": "b88da3904b7e03d20779088df24838d794e0cb29b17d75547ed4d0479182a5fe",
        "entry_count": 14,
        "crc_verified": True,
        "archive_safety_verified": True,
        "retention": "external_private_not_committed",
    }
    checkpoint = audit["source_checkpoint"]
    assert checkpoint["sha256"] == CHECKPOINT_SHA256
    assert checkpoint["key_count"] == 128
    assert checkpoint["strict_architecture_load_verified"] is True
    assert checkpoint["missing_keys"] == []
    assert checkpoint["unexpected_keys"] == []
    assert checkpoint["shape_mismatches"] == {}

    delivered = audit["delivered_torchscript"]
    assert delivered["sha256"] == DELIVERED_TORCHSCRIPT_SHA256
    assert delivered["torch_2_13_cpu_load_verified"] is True
    assert delivered["torch_2_6_cpu_load_verified"] is False
    assert "aten::_upsample_lanczos2d_aa" in delivered["torch_2_6_failure"]
    assert delivered["repository_imported"] is False

    compatible = audit["compatible_runtime_artifact"]
    assert compatible["sha256"] == RUNTIME_TORCHSCRIPT_SHA256
    assert compatible["loads_under_torch_2_6"] is True
    assert compatible["loads_under_torch_2_13"] is True
    assert compatible["eager_torchscript_max_abs_diff"] == 0.0
    assert compatible["repeat_max_abs_diff"] == 0.0
    assert compatible["delivered_compatible_max_abs_diff_under_torch_2_13"] == 0.0

    weight_path = _artifact_root() / str(compatible["repository_path"]).removeprefix(
        "model_artifacts/"
    )
    assert weight_path.stat().st_size == compatible["size_bytes"]
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == compatible["sha256"]

    gateway = audit["gateway_smoke"]
    assert gateway["fixture_kind"] == "deterministic_synthetic_2048x1536_not_scientific"
    assert all(value is True for key, value in gateway.items() if key != "fixture_kind")

    target_linux = audit["target_linux_artifact_smoke"]
    assert target_linux == {
        "platform": "debian_12_linux_arm64",
        "python": "3.12.13",
        "torch": "2.6.0+cpu",
        "device": "cpu",
        "repository_sha256_verified": True,
        "input_shape": [1, 1, 256, 256],
        "output_shape": [1, 1, 256, 256],
        "output_dtype": "torch.float32",
        "output_finite": True,
        "repeat_max_abs_diff": 0.0,
        "scope": "artifact_load_and_forward_only_not_full_gateway_or_scientific_acceptance",
    }

    acceptance = audit["scientific_acceptance"]
    assert acceptance["status"] == "pending_small_b"
    assert acceptance["metrics"] == {}
    assert "small_b_calibration_and_evaluation_not_delivered" in acceptance["blockers"]


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


def test_small_model_card_records_runtime_identity_permissions_and_pending_science() -> None:
    card = (_artifact_root() / "model_cards" / f"{MODEL_ID}.md").read_text(encoding="utf-8")
    normalized = " ".join(card.split())

    assert CHECKPOINT_SHA256 in card
    assert "郭境濠" in card
    assert "`small_batchnorm`" in card
    assert "128 keys" in card
    assert "`expected_image_size=[1536, 2048]`" in card
    assert "strict `probability > 0.30`" in card
    assert "not** a scientifically calibrated threshold" in card
    assert RUNTIME_TORCHSCRIPT_SHA256 in card
    assert DELIVERED_TORCHSCRIPT_SHA256 in card
    assert "runtime ready; scientific acceptance pending Small-B" in card
    assert "maximum absolute error `0.0`" in normalized
    assert "does not grant third-party redistribution" in normalized
