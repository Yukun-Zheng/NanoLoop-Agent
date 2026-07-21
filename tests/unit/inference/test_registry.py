from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from app.contracts.enums import ModelFamily, ModelStatus
from app.core.errors import ModelNotFoundError, ModelNotReadyError
from app.inference.registry import ModelRegistryService
from tests.unit.inference.fakes import FakeAdapter
from tests.unit.inference.helpers import build_registry, model_entry


def test_bundled_registry_declares_three_truthfully_unavailable_families() -> None:
    project_root = Path(__file__).parents[3]
    registry = ModelRegistryService(project_root / "model_artifacts" / "registry.yaml")

    models = registry.list_models()

    assert {model.family for model in models} == {
        ModelFamily.UNET,
        ModelFamily.YOLO_SEG,
        ModelFamily.SAM2,
    }
    assert {model.status for model in models} == {ModelStatus.UNAVAILABLE}
    assert all(model.health_error for model in models)
    assert registry.list_models(only_ready=True) == []


def test_valid_artifacts_become_ready_and_adapter_resolution_is_lazy(tmp_path: Path) -> None:
    calls: list[str] = []

    def resolve(adapter_path: str) -> type[FakeAdapter]:
        calls.append(adapter_path)
        return FakeAdapter

    entry = model_entry(tmp_path, "ready-model")
    registry = build_registry(tmp_path, [entry], resolver=resolve)

    model = registry.get_metadata("ready-model")
    assert model.status == ModelStatus.READY
    assert model.weight_sha256 == entry["weight_sha256"]
    assert model.config_sha256 == hashlib.sha256(
        (tmp_path / entry["config_path"]).read_bytes()
    ).hexdigest()
    assert model.model_card_sha256 == hashlib.sha256(
        (tmp_path / entry["model_card_path"]).read_bytes()
    ).hexdigest()
    assert calls == []

    adapter = registry.create_adapter("ready-model")

    assert isinstance(adapter, FakeAdapter)
    assert calls == [entry["adapter_path"]]


def _calibrated_unet_entry(tmp_path: Path) -> dict[str, object]:
    entry = model_entry(
        tmp_path,
        "calibrated-unet",
        config={
            "bottom_crop_px": 130,
            "expected_image_size": [1536, 2048],
            "default_threshold": 0.25,
            "threshold_comparison": "gte",
            "calibrated_analysis": {
                "threshold": 0.25,
                "min_area_px": 1024,
                "bottom_crop_px": 130,
                "threshold_comparison": "gte",
            },
        },
    )
    entry["adapter_path"] = "app.inference.adapters.unet:UNetAdapter"
    metadata = entry["metadata"]
    assert isinstance(metadata, dict)
    metadata.update(
        {
            "default_threshold": 0.25,
            "default_min_area_px": 1024,
            "inference_invalid_bottom_px": 130,
            "expected_input_width": 2048,
            "expected_input_height": 1536,
        }
    )
    return entry


def test_calibrated_unet_registry_contract_can_become_ready(tmp_path: Path) -> None:
    entry = _calibrated_unet_entry(tmp_path)

    model = build_registry(tmp_path, [entry]).get_metadata("calibrated-unet")

    assert model.status == ModelStatus.READY
    assert model.default_threshold == pytest.approx(0.25)
    assert model.default_min_area_px == 1024
    assert model.inference_invalid_bottom_px == 130
    assert model.expected_input_width == 2048
    assert model.expected_input_height == 1536


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("default_threshold", 0.30, "config default_threshold differs"),
        ("calibrated_min_area_px", 512, "calibrated min_area_px differs"),
        ("bottom_crop_px", 129, "bottom_crop_px differs"),
        ("expected_image_height", 1024, "expected_image_size height differs"),
    ],
)
def test_calibrated_unet_registry_contract_drift_fails_closed(
    tmp_path: Path,
    field: str,
    value: float | int,
    message: str,
) -> None:
    entry = _calibrated_unet_entry(tmp_path)
    config_path = tmp_path / str(entry["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if field == "calibrated_min_area_px":
        config["calibrated_analysis"]["min_area_px"] = value
    elif field == "expected_image_height":
        config["expected_image_size"][0] = value
    else:
        config[field] = value
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    model = build_registry(tmp_path, [entry]).get_metadata("calibrated-unet")

    assert model.status == ModelStatus.UNAVAILABLE
    assert message in (model.health_error or "")


def test_unet_without_frozen_default_threshold_fails_closed(tmp_path: Path) -> None:
    entry = _calibrated_unet_entry(tmp_path)
    metadata = entry["metadata"]
    assert isinstance(metadata, dict)
    metadata["default_threshold"] = None

    model = build_registry(tmp_path, [entry]).get_metadata("calibrated-unet")

    assert model.status == ModelStatus.UNAVAILABLE
    assert "metadata.default_threshold is required" in (model.health_error or "")


def test_unet_bottom_crop_without_frozen_expected_image_size_fails_closed(
    tmp_path: Path,
) -> None:
    entry = _calibrated_unet_entry(tmp_path)
    config_path = tmp_path / str(entry["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.pop("expected_image_size")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    model = build_registry(tmp_path, [entry]).get_metadata("calibrated-unet")

    assert model.status == ModelStatus.UNAVAILABLE
    assert "expected_image_size is required" in (model.health_error or "")


def test_non_ascii_model_version_has_stable_artifact_cache_key(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "unicode-version")
    entry["metadata"]["version"] = "版本-β"
    registry = build_registry(tmp_path, [entry])

    first = registry.validate_artifacts("unicode-version").cache_key
    second = registry.validate_artifacts("unicode-version").cache_key

    assert first == second
    assert len(first) == 64


def test_legacy_validate_then_create_reuses_bundle_without_reopening_source(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "compatibility")
    source_path = tmp_path / entry["weight_path"]
    original = source_path.read_bytes()
    registry = build_registry(tmp_path, [entry])

    provenance = registry.validate_artifacts("compatibility")
    source_path.write_bytes(b"changed after validation")
    adapter = registry.create_adapter(
        model_id="compatibility",
        expected_provenance=provenance,
    )

    assert adapter.weight_path != source_path
    assert adapter.weight_bytes == original


def test_validated_bundle_persists_complete_content_addressed_references(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "complete-bundle", config={"marker": "frozen"})
    registry = build_registry(tmp_path, [entry])

    bundle = registry.validate_bundle("complete-bundle")
    reference = bundle.reference

    assert reference.bundle_id == hashlib.sha256(
        registry.snapshot_store.read_reference(reference.manifest_ref, reference.bundle_id)
    ).hexdigest()
    assert registry.snapshot_store.read_reference(
        reference.weight_ref, bundle.provenance.weight_sha256
    ) == (tmp_path / entry["weight_path"]).read_bytes()
    assert registry.snapshot_store.read_reference(
        reference.config_ref, bundle.provenance.config_sha256
    ) == (tmp_path / entry["config_path"]).read_bytes()
    assert registry.snapshot_store.read_reference(
        reference.model_card_ref, bundle.provenance.model_card_sha256
    ) == (tmp_path / entry["model_card_path"]).read_bytes()
    assert registry.snapshot_store.read_reference(
        reference.adapter_ref, bundle.provenance.adapter_sha256
    )


def test_missing_optional_dependency_forces_unavailable(tmp_path: Path) -> None:
    entry = model_entry(
        tmp_path,
        "missing-runtime",
        required_modules=["nanoloop_dependency_that_does_not_exist"],
    )
    registry = build_registry(tmp_path, [entry])

    metadata = registry.get_metadata("missing-runtime")

    assert metadata.status == ModelStatus.UNAVAILABLE
    assert "optional dependency is missing" in (metadata.health_error or "")
    with pytest.raises(ModelNotReadyError):
        registry.create_adapter("missing-runtime")


def test_missing_weight_and_sha_mismatch_force_unavailable(tmp_path: Path) -> None:
    missing = model_entry(tmp_path, "missing-weight")
    (tmp_path / missing["weight_path"]).unlink()
    mismatched = model_entry(tmp_path, "mismatched-weight")
    mismatched["weight_sha256"] = "0" * 64
    registry = build_registry(tmp_path, [missing, mismatched])

    by_id = {model.model_id: model for model in registry.list_models()}

    assert by_id["missing-weight"].status == ModelStatus.UNAVAILABLE
    assert "weight file is missing" in (by_id["missing-weight"].health_error or "")
    assert by_id["mismatched-weight"].status == ModelStatus.UNAVAILABLE
    assert "sha256 mismatch" in (by_id["mismatched-weight"].health_error or "")


def test_invalid_config_card_and_adapter_are_health_failures(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "broken-declaration")
    (tmp_path / entry["config_path"]).write_text("- not-a-mapping\n", encoding="utf-8")
    (tmp_path / entry["model_card_path"]).write_text("", encoding="utf-8")
    entry["adapter_path"] = "not a valid path"
    registry = build_registry(tmp_path, [entry])

    metadata = registry.get_metadata("broken-declaration")

    assert metadata.status == ModelStatus.UNAVAILABLE
    assert "invalid config" in (metadata.health_error or "")
    assert "model card is empty" in (metadata.health_error or "")
    assert "invalid adapter path" in (metadata.health_error or "")


def test_malformed_registry_degrades_without_crashing(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"models": "not-a-list"}), encoding="utf-8")

    registry = ModelRegistryService(path)

    assert registry.list_models() == []
    assert registry.registry_error is not None


def test_unknown_model_uses_domain_error(tmp_path: Path) -> None:
    registry = build_registry(tmp_path, [])

    try:
        registry.get_metadata("unknown")
    except ModelNotFoundError as exc:
        assert exc.details == {"model_id": "unknown"}
    else:  # pragma: no cover - assertion aid
        raise AssertionError("ModelNotFoundError was not raised")
