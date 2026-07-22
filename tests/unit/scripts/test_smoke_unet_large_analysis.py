from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.analyses import RunConfiguration
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
from scripts.models import smoke_unet_large_analysis as smoke_module
from scripts.models.smoke_unet_large_analysis import (
    ADAPTER_PATH,
    ADAPTER_SHA256,
    CONFIG_SHA256,
    MODEL_CARD_SHA256,
    MODEL_ID,
    TEST_FILENAMES,
    TORCHSCRIPT_SHA256,
    SmokeParameters,
    _validate_output_root,
    _validated_parameters,
    build_parser,
    execute_analyses,
    load_calibrated_analysis,
)


def _config(*, width: int = 2048, height: int = 1536) -> dict[str, Any]:
    return {
        "bottom_crop_px": 180,
        "expected_image_size": [height, width],
        "threshold_comparison": "gt",
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


class FakeGateway:
    def __init__(self, *, width: int = 2048, height: int = 1536) -> None:
        self.requests: list[SegmentationRequest] = []
        self.model = ModelMetadata(
            model_id=MODEL_ID,
            family=ModelFamily.UNET,
            variant=ModelVariant.LARGE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.50,
            default_min_area_px=512,
            preprocess_profile="sem-gray-unit-crop-bottom-180-v1",
            postprocess_profile="semantic-mask-v1",
            inference_invalid_bottom_px=180,
            expected_input_width=width,
            expected_input_height=height,
            adapter_path=ADAPTER_PATH,
            weight_sha256=TORCHSCRIPT_SHA256,
            config_sha256=CONFIG_SHA256,
            model_card_sha256=MODEL_CARD_SHA256,
            adapter_sha256=ADAPTER_SHA256,
        )
        self.bundle = ModelBundleReference(
            bundle_id="e" * 64,
            manifest_ref=f"bundles/{'e' * 64}/manifest.json",
            weight_ref=f"{TORCHSCRIPT_SHA256}/weights.pt",
            config_ref=f"{CONFIG_SHA256}/config.yaml",
            model_card_ref=f"{MODEL_CARD_SHA256}/model-card.md",
            adapter_ref=f"{ADAPTER_SHA256}/adapter.py",
            adapter_sha256=ADAPTER_SHA256,
        )

    def freeze_model_bundle(self, _model_id: str, **_kwargs: Any) -> ModelBundleReference:
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
        assert request.threshold == pytest.approx(0.50)
        assert request.min_area_px == 512
        assert request.device == DevicePreference.CPU
        assert request.seed == 2026
        assert expected_model_version == self.model.version
        assert expected_adapter_path == self.model.adapter_path
        assert expected_weight_sha256 == self.model.weight_sha256
        assert expected_config_sha256 == self.model.config_sha256
        assert expected_model_card_sha256 == self.model.model_card_sha256
        assert expected_adapter_sha256 == self.model.adapter_sha256
        assert model_bundle == self.bundle
        mask = np.zeros((240, 80), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[100:230, 50:70] = 255
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


def _write_test_images(image_dir: Path) -> None:
    image_dir.mkdir()
    rows = np.arange(240, dtype=np.uint8)[:, None]
    columns = np.arange(80, dtype=np.uint8)[None, :]
    gray = (rows * 7 + columns * 11) % 251
    for index, filename in enumerate(TEST_FILENAMES):
        # The application rejects duplicate uploads by content hash.  Keep the
        # images structurally identical for this test, but make each file a
        # distinct upload payload.
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


def _execute_fake_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], LocalFileStore, Path]:
    monkeypatch.setattr(smoke_module, "IMAGE_WIDTH", 80)
    monkeypatch.setattr(smoke_module, "IMAGE_HEIGHT", 240)
    image_dir = tmp_path / "test-images"
    output_root = tmp_path / "smoke-output"
    registry_path = tmp_path / "private-registry.yaml"
    output_root.mkdir()
    registry_path.write_text("models: []\n", encoding="utf-8")
    _write_test_images(image_dir)
    database = _database(output_root)
    gateway = FakeGateway(width=80, height=240)
    file_store = LocalFileStore(
        StoragePaths(output_root / "artifacts"),
        max_upload_bytes=max((image_dir / filename).stat().st_size for filename in TEST_FILENAMES),
    )
    with database.session() as session:
        session.add(
            ModelRegistryRecord(
                model_id=gateway.model.model_id,
                family=gateway.model.family.value,
                variant=gateway.model.variant.value,
                quality_tier=gateway.model.quality_tier.value,
                version=gateway.model.version,
                adapter=gateway.model.adapter_path or ADAPTER_PATH,
                status=ModelStatus.READY.value,
            )
        )
    try:
        result = execute_analyses(
            SmokeParameters(image_dir, registry_path, output_root),
            load_calibrated_analysis(_config(width=80, height=240), gateway.model),
            database=database,
            file_store=file_store,
            gateway=gateway,
        )
    finally:
        database.dispose()
    assert len(gateway.requests) == 3
    return result, file_store, output_root


def _configuration(run: dict[str, Any]) -> RunConfiguration:
    return smoke_module._load_contract_artifact(
        Path(str(run["artifacts"]["run_config_path"])),
        RunConfiguration,
    )


def _validate_run_chain(
    run: dict[str, Any],
    *,
    file_store: LocalFileStore,
    output_root: Path,
) -> None:
    smoke_module._validate_canonical_chain(
        artifacts=run["artifacts"],
        configuration=_configuration(run),
        width=80,
        height=240,
        sample_id=str(run["sample_id"]),
        run_id=str(run["run_id"]),
        output_root=output_root,
        file_store=file_store,
    )


def _rewrite_particles_csv(path: Path, field: str, value: str) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    rows[0][field] = value
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _zip_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _replace_export_manifest(members: dict[str, bytes], manifest: dict[str, Any]) -> None:
    members["export_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _selection_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_loads_all_scientific_parameters_from_large_config() -> None:
    gateway = FakeGateway()

    calibrated = load_calibrated_analysis(_config(), gateway.model)

    assert calibrated.threshold == 0.50
    assert calibrated.min_area_px == 512
    assert calibrated.watershed_enabled is False
    assert calibrated.fill_holes is True
    assert calibrated.exclude_border is True
    assert calibrated.connectivity == 2
    assert calibrated.perimeter_neighborhood == 8
    assert calibrated.bottom_crop_px == 180
    assert calibrated.scale_nm_per_pixel == pytest.approx(100 / 184)
    assert calibrated.postprocess_profile().min_area_px == 512
    assert calibrated.morphometry_config().perimeter_neighborhood == 8


def test_rejects_internally_inconsistent_calibrated_config() -> None:
    gateway = FakeGateway()
    config = _config()
    config["calibrated_analysis"]["min_area_nm2"] = 1.0

    with pytest.raises(ValueError, match="physical conversion is inconsistent"):
        load_calibrated_analysis(config, gateway.model)


def test_report_loader_removes_only_file_schema_metadata(tmp_path: Path) -> None:
    path = tmp_path / "image_summary.json"
    payload = {
        "schema_version": "1.0",
        "run_id": "run_1",
        "particle_count": 1,
        "roi_area_px": 100,
        "number_density_px2": 0.01,
        "number_density_um2": 1.0,
        "mean_equivalent_diameter_px": 2.0,
        "mean_equivalent_diameter_nm": 3.0,
        "coverage_ratio": 0.1,
        "perimeter_density_px": 0.2,
        "perimeter_density_um": 4.0,
        "quality_status": "PASS",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = smoke_module._load_report_artifact(
        path,
        smoke_module.ImageSummaryDTO,
        artifact="image_summary.json",
        context="sample=test, run=run_1",
    )

    assert summary.run_id == "run_1"
    assert summary.particle_count == 1
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_three_full_image_runs_freeze_config_and_exclude_bottom_roi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)

    assert result["test_images"] == list(TEST_FILENAMES)
    assert "no test masks were read" in result["test_scope"]
    assert len(result["runs"]) == 3
    assert Path(result["export_path"]).is_file()
    assert result["export_sha256"] == file_store.calculate_sha256(Path(result["export_path"]))
    assert result["export_manifest"]["job_id"] == result["job_id"]
    for run in result["runs"]:
        assert run["frozen_inference"] == {
            "threshold": 0.50,
            "min_area_px": 512,
            "watershed_enabled": False,
            "exclude_border": True,
            "device": "cpu",
            "seed": 2026,
        }
        assert run["resolved_postprocess"] == {
            "profile_id": "semantic-mask-v1",
            "min_area_px": 512,
            "fill_holes": True,
            "watershed_enabled": False,
            "exclude_border": True,
            "connectivity": 2,
            "instance_iou_threshold": 0.7,
        }
        assert run["resolved_morphometry"] == {"perimeter_neighborhood": 8}
        assert run["roi"]["effective_roi_area_px"] == 80 * (240 - 180)
        bottom = run["roi"]["bottom_exclusion"]
        assert bottom["matching_model_bottom_region_present"] is True
        assert bottom["expected_bottom_area_px"] == 80 * 180
        density = run["roi"]["density_consistency"]
        assert density["all_particle_bboxes_within_effective_roi"] is True
        assert density["coverage_uses_effective_roi"] is True
        assert density["number_density_uses_effective_roi"] is True
        assert density["perimeter_density_uses_effective_roi"] is True
        assert run["scientific_results"]["particle_count"] == 1
        for artifact in (
            "pred_mask_path",
            "instances_path",
            "particles_csv_path",
            "overlay_path",
            "labeled_particles_path",
            "image_summary_path",
            "execution_provenance_path",
            "run_config_path",
            "transform_path",
            "quality_report_path",
            "probability_path",
        ):
            assert run["artifacts"][artifact] is not None
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
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


def test_rejects_instances_union_that_differs_from_prediction_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    path = Path(str(run["artifacts"]["pred_mask_path"]))
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    pixels[0, 0] = 255
    Image.fromarray(pixels).save(path)

    with pytest.raises(RuntimeError, match=r"instances\.json union differs from pred_mask\.png"):
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


def test_rejects_invalid_instances_rle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    path = Path(str(run["artifacts"]["instances_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["instances"][0]["mask"] = {
        "encoding": "flat_rle_v1",
        "order": "row_major",
        "starts": [80 * 240 - 1],
        "lengths": [2],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mask RLE is invalid"):
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("instance_index", "999", "instance_index does not match instances.json"),
        ("bbox_x2", "69", "bbox differs from instances.json"),
        ("area_px", "901", "area_px differs from instances.json"),
    ],
)
def test_rejects_particles_csv_instance_identity_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    _rewrite_particles_csv(Path(str(run["artifacts"]["particles_csv_path"])), field, value)

    with pytest.raises(RuntimeError, match=message):
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


def test_rejects_summary_that_differs_from_particles_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    path = Path(str(run["artifacts"]["image_summary_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["particle_count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"image_summary\.json\.particle_count"):
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


def test_rejects_quality_status_that_differs_from_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    path = Path(str(run["artifacts"]["quality_report_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "WARN" if payload["status"] != "WARN" else "PASS"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="quality_status differs"):
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


def test_rejects_noncanonical_quality_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, output_root = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    path = Path(str(run["artifacts"]["quality_report_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["foreground_ratio"] = 0.123
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"quality_report\.json\.foreground_ratio differs"):
        _validate_run_chain(run, file_store=file_store, output_root=output_root)


def _canonical_member_path(
    run: dict[str, Any],
    *,
    key: str,
    file_store: LocalFileStore,
    job_id: str,
) -> str:
    return Path(str(run["artifacts"][key])).relative_to(file_store.paths.job_dir(job_id)).as_posix()


def test_rejects_export_missing_canonical_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, _ = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    member = _canonical_member_path(
        run,
        key="instances_path",
        file_store=file_store,
        job_id=result["job_id"],
    )
    members = _zip_members(Path(result["export_path"]))
    manifest = json.loads(members["export_manifest.json"])
    members.pop(member)
    manifest["files"] = [record for record in manifest["files"] if record["path"] != member]
    manifest["selection_sha256"] = _selection_sha256(manifest["files"])
    _replace_export_manifest(members, manifest)
    validated = smoke_module.validate_export_zip(
        _zip_bytes(members),
        expected_job_id=result["job_id"],
        expected_run_ids={str(item["run_id"]) for item in result["runs"]},
    )

    with pytest.raises(
        RuntimeError,
        match="manifest SHA differs from canonical artifact instances_path",
    ):
        smoke_module._validate_export_canonical_artifacts(
            validated,
            runs=result["runs"],
            file_store=file_store,
            job_id=result["job_id"],
        )


def test_rejects_export_manifest_sha_that_differs_from_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, _ = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    member = _canonical_member_path(
        run,
        key="particles_csv_path",
        file_store=file_store,
        job_id=result["job_id"],
    )
    members = _zip_members(Path(result["export_path"]))
    manifest = json.loads(members["export_manifest.json"])
    next(record for record in manifest["files"] if record["path"] == member)["sha256"] = "0" * 64
    _replace_export_manifest(members, manifest)

    with pytest.raises(ValueError, match="manifest hash does not match ZIP member"):
        smoke_module.validate_export_zip(
            _zip_bytes(members),
            expected_job_id=result["job_id"],
            expected_run_ids={str(item["run_id"]) for item in result["runs"]},
        )


def test_rejects_tampered_export_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, file_store, _ = _execute_fake_smoke(tmp_path, monkeypatch)
    run = result["runs"][0]
    member = _canonical_member_path(
        run,
        key="image_summary_path",
        file_store=file_store,
        job_id=result["job_id"],
    )
    members = _zip_members(Path(result["export_path"]))
    members[member] = b"tampered"

    with pytest.raises(
        ValueError,
        match=r"manifest size does not match ZIP member|manifest hash",
    ):
        smoke_module.validate_export_zip(
            _zip_bytes(members),
            expected_job_id=result["job_id"],
            expected_run_ids={str(item["run_id"]) for item in result["runs"]},
        )


@pytest.mark.parametrize("kind", ["existing", "repository"])
def test_output_root_protection(tmp_path: Path, kind: str) -> None:
    output_root = (
        tmp_path / "already-exists"
        if kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-large-smoke"
    )
    if kind == "existing":
        output_root.mkdir()

    with pytest.raises(ValueError, match="output-root"):
        _validate_output_root(output_root)


def test_cli_has_no_test_mask_input(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    registry = tmp_path / "registry.yaml"
    output_root = tmp_path / "output"
    _write_test_images(image_dir)
    registry.write_text("models: []\n", encoding="utf-8")
    parser = build_parser()
    assert all(
        "mask" not in option for action in parser._actions for option in action.option_strings
    )
    namespace = argparse.Namespace(
        image_dir=image_dir,
        registry=registry,
        output_root=output_root,
    )

    validated = _validated_parameters(namespace)

    assert validated.image_dir == image_dir.resolve()
