from __future__ import annotations

import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import app.inference.registry as registry_module
from app.contracts.enums import ModelStatus
from app.inference.registry import ModelRegistryService


def _artifact_root() -> Path:
    return Path(__file__).parents[3] / "model_artifacts"


def _registry_models() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load((_artifact_root() / "registry.yaml").read_text(encoding="utf-8"))
    return {entry["metadata"]["model_id"]: entry for entry in payload["models"]}


def _large_delivery_audit() -> dict[str, object]:
    path = (
        _artifact_root() / "evidence" / "unet-large-optimized-v1" / "delivery-audit-2026-07-23.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_large_runtime_weight_matches_registered_identity() -> None:
    weight_path = _artifact_root() / "weights" / "unet-large-optimized-v1.pt"

    assert weight_path.stat().st_size == 13_505_917
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == (
        "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05"
    )


def test_large_delivery_audit_recomputes_historical_pixel_metrics() -> None:
    audit = _large_delivery_audit()

    assert audit["schema_version"] == "1"
    assert audit["model_id"] == "unet-large-optimized-v1"
    assert audit["review_scope"] == "historical_bundle_pixel_metrics_only"
    assert audit["source_packages"] == [
        {
            "filename": "ModelAssets-large-a.zip",
            "size_bytes": 173_627_966,
            "sha256": "4173d7979d444fb1e74a7f5d3894a85f1feaed62e8c9443bdb9d7216dddd4815",
            "retention": "external_private_not_committed",
            "purpose": "runtime_and_historical_analysis_bundle",
        },
        {
            "filename": "ModelAssets-large-b.zip",
            "size_bytes": 72_236_041,
            "sha256": "c23a1000c27b2290950661cff5b2d7716d6f4ffcfb5c657f13d640c656306351",
            "retention": "external_private_not_committed",
            "purpose": "historical_b_module_outputs_and_scripts",
        },
        {
            "filename": "large-unet-independent-evaluation-v1.tar.gz",
            "size_bytes": 60_753,
            "sha256": "5f2de1c2db12a87e7396b434382c75b050770da234c56906c46a0352bf05b2b8",
            "retention": "external_private_not_committed",
            "purpose": "historical_pixel_metrics_and_review_images",
        },
    ]

    runtime = audit["runtime_identity"]
    assert isinstance(runtime, dict)
    weight_path = _artifact_root() / str(runtime["repository_path"]).removeprefix(
        "model_artifacts/"
    )
    assert weight_path.stat().st_size == runtime["torchscript_size_bytes"]
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == runtime["torchscript_sha256"]
    assert runtime["large_a_torchscript_matches_repository"] is True

    historical = audit["historical_run_bundle"]
    assert isinstance(historical, dict)
    assert historical["weight_sha256"] == runtime["torchscript_sha256"]
    assert historical["matches_current_runtime_weight"] is True
    assert historical["matches_current_source_contract"] is False
    current_hashes = {
        "adapter_sha256": hashlib.sha256(
            (Path(__file__).parents[3] / "app" / "inference" / "adapters" / "unet.py").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (_artifact_root() / "configs" / "unet-large-optimized-v1.yaml").read_bytes()
        ).hexdigest(),
        "model_card_sha256": hashlib.sha256(
            (_artifact_root() / "model_cards" / "unet-large-optimized-v1.md").read_bytes()
        ).hexdigest(),
    }
    for field, current_sha256 in current_hashes.items():
        assert historical[field] != current_sha256

    evaluation = audit["independent_pixel_evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["all_reported_hashes_and_counts_matched"] is True
    samples = evaluation["samples"]
    assert isinstance(samples, list)
    assert [sample["sample_id"] for sample in samples] == ["SrZr-3", "BaCu-2", "PrCu-3"]
    assert len({sample["filename"] for sample in samples}) == 3

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    metric_names = ("dice", "iou", "precision", "recall")
    observed_metrics: dict[str, list[float]] = {name: [] for name in metric_names}
    for sample in samples:
        counts = sample["counts"]
        metrics = sample["metrics"]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        expected = {
            "dice": (2 * tp) / ((2 * tp) + fp + fn),
            "iou": tp / (tp + fp + fn),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
        }
        for name in metric_names:
            assert metrics[name] == pytest.approx(expected[name], rel=0.0, abs=1e-15)
            observed_metrics[name].append(metrics[name])
        for name in totals:
            totals[name] += counts[name]

    macro = evaluation["macro_average"]
    for name in metric_names:
        assert macro[name] == pytest.approx(
            sum(observed_metrics[name]) / len(observed_metrics[name]),
            rel=0.0,
            abs=1e-15,
        )

    micro = evaluation["micro_average"]
    assert {name: micro[name] for name in totals} == totals
    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    expected_micro = {
        "dice": (2 * tp) / ((2 * tp) + fp + fn),
        "iou": tp / (tp + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
    }
    for name in metric_names:
        assert math.isclose(micro[name], expected_micro[name], rel_tol=0.0, abs_tol=1e-15)

    acceptance = audit["scientific_acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["status"] == "pending_current_bundle_rerun_and_policy"
    assert "license_or_written_authorization_missing" in acceptance["blockers"]


def test_large_unet_config_freezes_confirmed_inference_contract() -> None:
    config = yaml.safe_load(
        (_artifact_root() / "configs" / "unet-large-optimized-v1.yaml").read_text(encoding="utf-8")
    )

    assert config == {
        "schema_version": "1",
        "loader": "torchscript",
        "input_channels": 1,
        "input_size": [512, 512],
        "expected_image_size": [1536, 2048],
        "patch_size": [512, 512],
        "stride": [256, 256],
        "tiling_padding": "reflect",
        "overlap_fusion": "uniform",
        "bottom_crop_px": 180,
        "pixel_scale": 255.0,
        "mean": [0.0],
        "std": [1.0],
        "output_activation": "logits",
        "threshold_comparison": "gt",
        "default_threshold": 0.50,
        "calibrated_analysis": {
            "threshold": 0.50,
            "threshold_comparison": "gt",
            "min_area_px": 512,
            "min_area_nm2": 151.22873345935727,
            "min_area_equivalent_diameter_nm": 13.87625323135418,
            "watershed_enabled": False,
            "fill_holes": True,
            "exclude_border": True,
            "connectivity": 2,
            "perimeter_neighborhood": 8,
            "bottom_crop_px": 180,
            "scale_nm_per_pixel": 100 / 184,
        },
    }


def test_large_public_registry_is_runtime_ready_with_180_px_invalid_bottom() -> None:
    entry = _registry_models()["unet-large-optimized-v1"]
    metadata = entry["metadata"]

    assert metadata["status"] == "ready"
    assert metadata["inference_invalid_bottom_px"] == 180
    assert metadata["default_threshold"] == 0.50
    assert metadata["default_min_area_px"] == 512
    assert metadata["expected_input_width"] == 2048
    assert metadata["expected_input_height"] == 1536
    assert metadata["metrics"] == {
        "developer_reported_threshold_macro_dice": 0.7086369503967919,
        "developer_reported_threshold_macro_iou": 0.6023708166699938,
        "developer_reported_threshold_micro_dice": 0.722377672920259,
        "developer_reported_threshold_micro_iou": 0.5654078342317297,
        "developer_reported_min_area_gt_retention": 1.0,
        "historical_bundle_verified_pixel_macro_dice": 0.8053234061081441,
        "historical_bundle_verified_pixel_macro_iou": 0.6854416626935307,
        "historical_bundle_verified_pixel_micro_dice": 0.7734422618347466,
        "historical_bundle_verified_pixel_micro_iou": 0.630579578741805,
    }
    assert metadata["metric_context"] == {
        "evidence_status": "historical_pixel_metrics_verified_current_bundle_acceptance_pending",
        "runtime_asset_delivered": True,
        "runtime_asset_verified": True,
        "runtime_verification_device": "cpu",
        "evidence_bundle_delivered": False,
        "developer_evidence_archives_delivered": True,
        "historical_pixel_metrics_independently_recomputed": True,
        "historical_run_matches_current_source_contract": False,
        "independent_test_scope": "field_of_view",
        "independent_test_image_count": 3,
        "current_bundle_scientific_rerun_status": "pending",
        "license_review_status": "missing",
        "tolerance_policy_status": "missing",
        "asset_ledger_delivered": False,
        "redistribution_permission_delivered": False,
        "training_split_verification": "unknown",
        "validation_scope": "field_of_view",
        "validation_image_count": 6,
        "threshold_comparison": "gt",
        "calibrated_threshold": 0.50,
        "calibrated_min_area_px": 512,
        "calibrated_min_area_nm2": 151.22873345935727,
        "calibrated_min_area_equivalent_diameter_nm": 13.87625323135418,
        "watershed_enabled": False,
        "fill_holes": True,
        "exclude_border": True,
        "connectivity": 2,
        "perimeter_neighborhood": 8,
        "bottom_crop_px": 180,
        "scale_nm_per_pixel": 100 / 184,
    }
    assert entry["adapter_path"] == "app.inference.adapters.unet:UNetAdapter"
    assert entry["weight_path"] == "weights/unet-large-optimized-v1.pt"
    assert entry["weight_sha256"] == (
        "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05"
    )
    assert entry["config_path"] == "configs/unet-large-optimized-v1.yaml"
    assert entry["model_card_path"] == "model_cards/unet-large-optimized-v1.md"
    assert "health_error" not in metadata


def test_large_external_bundle_resolves_relative_assets_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _artifact_root()
    bundle = tmp_path / "external-model-artifacts"
    (bundle / "weights").mkdir(parents=True)
    (bundle / "configs").mkdir()
    (bundle / "model_cards").mkdir()
    shutil.copy2(
        source / "configs" / "unet-large-optimized-v1.yaml",
        bundle / "configs" / "unet-large-optimized-v1.yaml",
    )
    shutil.copy2(
        source / "model_cards" / "unet-large-optimized-v1.md",
        bundle / "model_cards" / "unet-large-optimized-v1.md",
    )
    weight_path = bundle / "weights" / "unet-large-optimized-v1.pt"
    weight_bytes = b"external-large-torchscript-fixture"
    weight_path.write_bytes(weight_bytes)

    entry = deepcopy(_registry_models()["unet-large-optimized-v1"])
    metadata = entry["metadata"]
    assert isinstance(metadata, dict)
    metadata["status"] = "ready"
    metadata.pop("health_error", None)
    entry["weight_sha256"] = hashlib.sha256(weight_bytes).hexdigest()
    registry_path = bundle / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"schema_version": "1", "models": [entry]}, sort_keys=False),
        encoding="utf-8",
    )

    def load_registry() -> ModelRegistryService:
        return ModelRegistryService(registry_path, snapshot_root=bundle / "snapshots")

    original_find_spec = registry_module.importlib.util.find_spec
    monkeypatch.setattr(
        registry_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else original_find_spec(name),
    )
    verified = load_registry()
    registration = verified.get_registration("unet-large-optimized-v1")
    metadata = verified.get_metadata("unet-large-optimized-v1")
    assert registration.weight_path == weight_path
    assert registration.config_path == bundle / "configs" / "unet-large-optimized-v1.yaml"
    assert registration.model_card_path == bundle / "model_cards" / "unet-large-optimized-v1.md"
    assert registration.weight_sha256 == hashlib.sha256(weight_bytes).hexdigest()
    assert metadata.status == ModelStatus.READY
    assert metadata.health_error is None

    monkeypatch.setattr(
        registry_module.importlib.util,
        "find_spec",
        lambda name: None if name == "torch" else original_find_spec(name),
    )
    missing_torch = load_registry().get_metadata("unet-large-optimized-v1")
    assert missing_torch.status == ModelStatus.UNAVAILABLE
    assert "optional dependency is missing: torch" in (missing_torch.health_error or "")

    monkeypatch.setattr(
        registry_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else original_find_spec(name),
    )
    weight_path.unlink()
    missing = load_registry().get_metadata("unet-large-optimized-v1")
    assert missing.status == ModelStatus.UNAVAILABLE
    assert "weight file is missing" in (missing.health_error or "")

    weight_path.write_bytes(b"tampered-external-large-torchscript-fixture")
    mismatched = load_registry().get_metadata("unet-large-optimized-v1")
    assert mismatched.status == ModelStatus.UNAVAILABLE
    assert "weight sha256 mismatch" in (mismatched.health_error or "")


def test_small_unet_asset_contract_includes_confirmed_input_dimensions() -> None:
    config = yaml.safe_load(
        (_artifact_root() / "configs" / "unet-small-balanced-v1.yaml").read_text(encoding="utf-8")
    )
    entry = _registry_models()["unet-small-balanced-v1"]

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
    assert entry["metadata"]["status"] == "unavailable"
    assert entry["metadata"]["default_threshold"] == 0.30
    assert entry["metadata"]["inference_invalid_bottom_px"] == 130
    assert entry["metadata"]["expected_input_width"] == 2048
    assert entry["metadata"]["expected_input_height"] == 1536


def test_large_model_card_records_export_and_scientific_readiness_limits() -> None:
    card = (_artifact_root() / "model_cards" / "unet-large-optimized-v1.md").read_text(
        encoding="utf-8"
    )

    assert "5c5dbcae61f40f8eb1fef27c7b69592a727260898330abc546f7e7a6833035bd" in card
    assert "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05" in card
    assert "State-dict loading: strict" in card
    assert "`56/56`" in card
    assert "130 px" in card and "180 px" in card
    assert "`probability > 0.50`" in card
    assert all(
        filename in card
        for filename in (
            "NdZn-2.tif",
            "LaMn-3.tif",
            "LaMn-1.tif",
            "BaCo-3.tif",
            "BaCu-1.tif",
            "BaCr-3.tif",
        )
    )
    assert "0.7086369503967919" in card
    assert "0.6023708166699938" in card
    assert "0.722377672920259" in card
    assert "0.5654078342317297" in card
    assert "material-domain sensitivity" in card
    assert "substantial under-segmentation" in card
    assert "substantial false positives" in card
    assert "`min_area_px=512`" in card
    assert "`151.22873345935727 nm²`" in card
    assert "`13.87625323135418 nm`" in card
    assert "GT retention at this candidate was `100%`" in card
    assert "Macro Composite MAPE" in card
    assert "`perimeter_neighborhood=8`" in card
    assert "not sample-level independent" in card
    assert "scientific acceptance remains pending" in card
    assert "historical run's pixel metrics" in card
    assert "current-bundle rerun" in card
    assert "not independently recomputed" in card
    assert "`expected_image_size=[1536, 2048]`" in card


def test_large_model_card_records_frozen_independent_test_evidence() -> None:
    card = (_artifact_root() / "model_cards" / "unet-large-optimized-v1.md").read_text(
        encoding="utf-8"
    )

    per_image_metrics = {
        "SrZr-3": (
            "0.9392828149931417",
            "0.8855167317639607",
            "0.923919927306771",
            "0.9551652480856273",
        ),
        "BaCu-2": (
            "0.724665460199322",
            "0.5682159759529292",
            "0.8119095758655747",
            "0.654351788772072",
        ),
        "PrCu-3": (
            "0.7520219431319688",
            "0.602592280363702",
            "0.7653143163046803",
            "0.7391834247410116",
        ),
    }
    for sample_id, metrics in per_image_metrics.items():
        assert sample_id in card
        assert all(metric in card for metric in metrics)

    assert all(
        metric in card
        for metric in (
            "0.8053234061081441",
            "0.6854416626935307",
            "0.8337146064923419",
            "0.7829001538662369",
            "0.7734422618347466",
            "0.630579578741805",
            "0.8211666401761994",
            "0.7309604656803299",
        )
    )
    assert "`TP=144660`" in card
    assert "`FP=31504`" in card
    assert "`FN=53244`" in card
    assert "`TN=8101856`" in card
    assert "top `2048 x 1356 px`" in card
    assert "bottom 180 px (`y=1356..1536`)" in card
    assert "`pred_mask.png`" in card and "corresponding human mask" in card
    assert "did not repeat inference" in card
    assert "must not be used to tune" in card
    assert "under-detection limitation" in card
    assert "not three sample-level independent" in card

    metadata = _registry_models()["unet-large-optimized-v1"]["metadata"]
    assert metadata["status"] == "ready"
    assert metadata["metric_context"]["historical_pixel_metrics_independently_recomputed"] is True
    assert metadata["metric_context"]["historical_run_matches_current_source_contract"] is False
    assert metadata["metric_context"]["evidence_bundle_delivered"] is False
    assert "scientific acceptance remains pending" in card
