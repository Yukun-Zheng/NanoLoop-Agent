from __future__ import annotations

from pathlib import Path

import yaml

MODEL_ID = "unet-agglomerated-specialized-v1"


def _artifact_root() -> Path:
    return Path(__file__).parents[3] / "model_artifacts"


def _registry_models() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load((_artifact_root() / "registry.yaml").read_text(encoding="utf-8"))
    return {entry["metadata"]["model_id"]: entry for entry in payload["models"]}


def test_agglomerated_config_freezes_validation_calibrated_analysis_contract() -> None:
    config = yaml.safe_load(
        (_artifact_root() / "configs" / f"{MODEL_ID}.yaml").read_text(encoding="utf-8")
    )

    assert config == {
        "schema_version": "1",
        "loader": "torchscript",
        "input_channels": 1,
        "input_size": [384, 384],
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
    assert metadata["default_threshold"] is None
    assert metadata["inference_invalid_bottom_px"] == 130
    assert metadata["metrics"] == {}
    assert metadata["metric_context"] == {
        "calibration_status": "pending",
        "validation_scope": "field_of_view",
        "validation_image_count": 4,
        "independent_test_image_count": 3,
        "target_definition": "whole_agglomerate",
    }
    assert entry["adapter_path"] == "app.inference.adapters.unet:UNetAdapter"
    assert entry["weight_path"] == "weights/unet-agglomerated-specialized-v1.pt"
    assert entry["weight_sha256"] is None
    assert entry["config_path"] == f"configs/{MODEL_ID}.yaml"
    assert entry["model_card_path"] == f"model_cards/{MODEL_ID}.md"


def test_model_card_records_asset_identity_science_definition_and_data_boundary() -> None:
    card = (_artifact_root() / "model_cards" / f"{MODEL_ID}.md").read_text(
        encoding="utf-8"
    )

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
        filename in card
        for filename in ("BiCu-3.tif", "BaNi-3.tif", "BaNi-1.tif", "BaNi-2.tif")
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
        "`AnalysisCreationService -> create_runs(FULL_IMAGE) -> "
        "InferenceGateway -> execute_run`"
    ) in card
    assert "`BiCu-3.tif`" in card
    assert "zero foreground pixels in the bottom 130 rows" in card
    assert "repository-external private registry only" in card
    assert "CLI private-registry input must itself remain `unavailable`" in card
    assert "`private-registry-ready.yaml`" in card
    assert "must remain `unavailable`" in card
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
    assert "public `default_threshold`" in card


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
    assert registry["unet-small-balanced-v1"]["metadata"]["status"] == "unavailable"
    assert registry["unet-large-optimized-v1"]["metadata"]["status"] == "unavailable"
