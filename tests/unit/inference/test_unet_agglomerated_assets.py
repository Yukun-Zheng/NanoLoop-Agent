from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

MODEL_ID = "unet-agglomerated-specialized-v1"
TORCHSCRIPT_SHA256 = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"


def _artifact_root() -> Path:
    return Path(__file__).parents[3] / "model_artifacts"


def _registry_models() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load((_artifact_root() / "registry.yaml").read_text(encoding="utf-8"))
    return {entry["metadata"]["model_id"]: entry for entry in payload["models"]}


def _delivery_audit() -> dict[str, object]:
    path = _artifact_root() / "evidence" / MODEL_ID / "delivery-audit-2026-07-24.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_agglomerated_config_freezes_validation_calibrated_analysis_contract() -> None:
    config = yaml.safe_load(
        (_artifact_root() / "configs" / f"{MODEL_ID}.yaml").read_text(encoding="utf-8")
    )

    assert config == {
        "schema_version": "1",
        "loader": "torchscript",
        "input_channels": 1,
        "input_size": [384, 384],
        "expected_image_size": [1536, 2048],
        "patch_size": [384, 384],
        "stride": [288, 288],
        "tiling_padding": "reflect",
        "pad_to_tile_grid": False,
        "overlap_fusion": "hann",
        "fusion_weight_floor": 0.05,
        "bottom_crop_px": 130,
        "normalization": "percentile",
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "output_activation": "logits",
        "threshold_comparison": "gte",
        "default_threshold": 0.25,
        "target_definition": "whole_agglomerate",
        "default_watershed_enabled": False,
        "scale_nm_per_pixel": 100 / 184,
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


def test_public_registry_keeps_agglomerated_placeholder_unavailable() -> None:
    entry = _registry_models()[MODEL_ID]
    metadata = entry["metadata"]

    assert metadata["status"] == "unavailable"
    assert metadata["default_threshold"] == 0.25
    assert metadata["default_min_area_px"] == 1024
    assert metadata["inference_invalid_bottom_px"] == 130
    assert metadata["expected_input_width"] == 2048
    assert metadata["expected_input_height"] == 1536
    assert metadata["metrics"] == {}
    assert metadata["metric_context"] == {
        "calibration_status": "developer_reported_not_independently_verified",
        "validation_scope": "field_of_view",
        "validation_image_count": 4,
        "independent_test_image_count": 3,
        "target_definition": "whole_agglomerate",
        "calibrated_threshold": 0.25,
        "calibrated_min_area_px": 1024,
        "evidence_bundle_delivered": True,
        "runtime_asset_delivered_to_operator": True,
        "private_runtime_smoke_verified": True,
        "private_registry_ready_manifest_delivered": True,
        "runtime_verification_device": "cpu",
        "runtime_verification_scope": "private_exact_asset_full_analysis_smoke",
        "public_runtime_asset_embedded": False,
        "redistribution_permission_delivered": False,
        "scientific_acceptance_status": "pending",
        "asset_ledger_delivered": False,
        "training_split_verification": "unknown",
    }
    assert entry["adapter_path"] == "app.inference.adapters.unet:UNetAdapter"
    assert entry["weight_path"] == "weights/unet-agglomerated-specialized-v1.pt"
    assert entry["weight_sha256"] is None
    assert entry["config_path"] == f"configs/{MODEL_ID}.yaml"
    assert entry["model_card_path"] == f"model_cards/{MODEL_ID}.md"


def test_delivery_audit_records_exact_private_runtime_smoke_and_public_boundary() -> None:
    audit = _delivery_audit()

    assert audit["schema_version"] == "1"
    assert audit["model_id"] == MODEL_ID
    assert audit["review_scope"] == "private_runtime_asset_and_full_analysis_smoke_only"
    assert audit["source_package"] == {
        "filename": "agglomerated-a-linux-final.zip",
        "size_bytes": 56_451_110,
        "sha256": "e86e4a0530c84f011b4bbdf86a5d2823df044170ae76faf97d904ac084a58b62",
        "entry_count": 46,
        "uncompressed_size_bytes": 61_153_466,
        "crc_verified": True,
        "archive_safety_verified": True,
        "retention": "external_private_not_committed",
    }

    runtime = audit["runtime_identity"]
    assert runtime["torchscript_sha256"] == TORCHSCRIPT_SHA256
    assert runtime["torchscript_size_bytes"] == 20_702_390
    assert runtime["model_bundle_id"] == runtime["bundle_manifest_sha256"]
    assert runtime["content_addressed_paths_verified"] is True

    equivalence = audit["source_contract_equivalence"]
    assert equivalence["delivery_line_endings"] == "crlf"
    assert equivalence["repository_line_endings"] == "lf"
    assert all(
        equivalence[field] is True
        for field in (
            "adapter_matches_after_crlf_normalization",
            "config_matches_after_crlf_normalization",
            "model_card_matches_after_crlf_normalization",
        )
    )
    project_root = Path(__file__).parents[3]
    unchanged_paths = {
        "adapter": project_root / "app" / "inference" / "adapters" / "unet.py",
        "config": _artifact_root() / "configs" / f"{MODEL_ID}.yaml",
    }
    base_sha256 = equivalence["repository_base_sha256"]
    assert {name: base_sha256[name] for name in unchanged_paths} == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in unchanged_paths.items()
    }
    assert equivalence["model_card_updated_after_review"] is True
    current_card_sha256 = hashlib.sha256(
        (_artifact_root() / "model_cards" / f"{MODEL_ID}.md").read_bytes()
    ).hexdigest()
    assert current_card_sha256 != base_sha256["model_card"]

    smoke = audit["gateway_analysis_smoke"]
    assert smoke["final_status"] == "COMPLETED_WITH_WARNINGS"
    assert smoke["quality_reasons"] == ["small_fragment_ratio_high"]
    assert smoke["private_registry_ready_eligible"] is True
    assert smoke["resolved_device"] == "cpu"
    assert smoke["canonical_artifact_count"] == 11
    assert all(
        smoke[field] is True
        for field in (
            "registry_loaded",
            "snapshot_bundle_validated",
            "load_completed_via_predict",
            "predict_completed",
            "analysis_completed",
            "unload_completed",
            "bottom_130_rows_zero",
            "effective_roi_excludes_bottom_130_rows",
            "execution_bundle_present",
            "build_identity_matches_contract",
        )
    )
    assert smoke["cache_count_after_unload"] == 0

    local = audit["local_cpu_reverification"]
    assert local["platform"] == "linux_aarch64"
    assert local["torch_version"] == "2.13.0+cpu"
    assert local["registry_status"] == "ready"
    assert local["resolved_device"] == "cpu"
    assert local["probability_shape"] == [1536, 2048]
    assert local["probability_finite"] is True
    assert local["delivered_probability_max_abs_diff"] <= 2e-6
    assert local["threshold_pixel_diff"] == 0
    assert local["binary_mask_matches_delivery"] is True
    assert local["binary_mask_sha256"] == (
        "b1daa041e479b0041695bc0942a22a07c66c8c1d82dda35cfd391fe9f9034c51"
    )
    assert local["bottom_130_probability_nonzero"] == 0
    assert local["bottom_130_mask_nonzero"] == 0
    assert local["load_health_status"] == "ready"
    assert local["unload_health_status"] == "ready"
    assert local["cache_device_after_unload"] is None

    policy = audit["repository_import_policy"]
    assert policy["public_registry_status"] == "unavailable"
    assert policy["runtime_asset_committed"] is False
    assert policy["private_input_or_outputs_committed"] is False
    assert "torchscript" in policy["not_imported"]
    assert audit["scientific_acceptance"]["status"] == "pending"


def test_model_card_records_asset_identity_science_definition_and_data_boundary() -> None:
    card = (_artifact_root() / "model_cards" / f"{MODEL_ID}.md").read_text(encoding="utf-8")

    assert "e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21" in card
    assert "raw `OrderedDict` state_dict" in card
    assert "State-dict keys: `63`" in card
    assert "`missing_keys=[]`, `unexpected_keys=[]`" in card
    assert "whole agglomerate" in card
    assert "internal single-particle statistics" in card
    assert "bottom 130 px" in card
    assert "P1 and P99" in card
    assert "Hann window" in card and "`0.05`" in card
    assert "`100/184 nm_per_pixel`" in card
    assert all(
        filename in card for filename in ("BiCu-3.tif", "BaNi-3.tif", "BaNi-1.tif", "BaNi-2.tif")
    )
    assert all(filename in card for filename in ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif"))
    assert "training metadata JSON is missing" in card
    assert "`default_threshold=0.25` is frozen" in card
    assert "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9" in card
    assert "347/357" in card
    assert "9c76289a61ab870b59cda079eb732222a1267d3ecf47636244967872c4130a02" in card
    assert "`0.7554787140413239`" in card
    assert "`0.7351176348009941`" in card
    assert "Tiny GT contained only 12" in card
    assert "especially `BaNi-2`" in card
    assert "visible false-positive regions" in card
    assert "`min_area_px=1024` is frozen" in card
    assert "0e3d4c94aa4ff532e65d2803ed5259583eabbc4dcd8ba889595329823f9399c5" in card
    assert "`0.16633235387069195`" in card
    assert "`0.7471895809869473`" in card
    assert "`307/350` (`0.8771428571428571`)" in card
    assert "removed 43 GT agglomerates" in card
    assert "does **not** mean" in card
    assert "## Full Analysis smoke acceptance gate" in card
    assert (
        "`AnalysisCreationService -> create_runs(FULL_IMAGE) -> InferenceGateway -> execute_run`"
    ) in card
    assert "`BiCu-3.tif`" in card
    assert "zero foreground pixels in the bottom 130 rows" in card
    assert "repository-external private registry only" in card
    assert "CLI private-registry input must itself remain `unavailable`" in card
    assert "`private-registry-ready.yaml`" in card
    assert "must remain `unavailable`" in card
    assert "agglomerated-a-linux-final.zip" in card
    assert "runtime-ready only through its repository-external private registry" in card
    assert "COMPLETED_WITH_WARNINGS" in card
    assert "`small_fragment_ratio_high`" in card
    assert "independent fields of view (FOV)" in card
    assert "**not** three independent materials" in card
    assert "`probability >= 0.25`, `min_area_px=1024`, bottom `130 px`" in card
    assert "whole-agglomerate semantics" in card
    assert "Macro Dice `0.8678420653`" in card
    assert "Micro Dice `0.8575525540`" in card
    assert "Macro IoU `0.7707467511`" in card
    assert "Macro MAPE was `3.7712858326%`" in card
    assert "agglomerate-count Macro MAPE was `51.2323232323%`" in card
    assert "diameter Macro MAPE was `82.3173359269%`" in card
    assert "number-density Macro MAPE was `51.2323232323%`" in card
    assert "perimeter-density Macro MAPE was `16.8649123230%`" in card
    assert "`YCu-2` has substantial missed small targets" in card
    assert "must **not** be claimed as a high-precision tool" in card
    assert "developer-reported" in card
    assert "training/test independence cannot be" in card
    assert "`expected_image_size=[1536, 2048]`" in card


def test_existing_small_and_large_configs_do_not_enable_agglomerated_behavior() -> None:
    config_root = _artifact_root() / "configs"
    small = yaml.safe_load(
        (config_root / "unet-small-balanced-v1.yaml").read_text(encoding="utf-8")
    )
    large = yaml.safe_load(
        (config_root / "unet-large-optimized-v1.yaml").read_text(encoding="utf-8")
    )

    assert (small["patch_size"], small["stride"], small["overlap_fusion"]) == (
        [256, 256],
        [128, 128],
        "uniform",
    )
    assert (large["patch_size"], large["stride"], large["overlap_fusion"]) == (
        [512, 512],
        [256, 256],
        "uniform",
    )
    for config in (small, large):
        assert "normalization" not in config
        assert "pad_to_tile_grid" not in config
        assert "fusion_weight_floor" not in config

    registry = _registry_models()
    assert registry["unet-small-balanced-v1"]["metadata"]["status"] == "ready"
    assert registry["unet-large-optimized-v1"]["metadata"]["status"] == "ready"
