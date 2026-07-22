from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.enums import (
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
)
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelHealth, ModelMetadata
from app.core.config import Settings
from app.db.base import Base
from app.db.models import ModelRegistryRecord
from app.db.session import Database
from app.storage import LocalFileStore, StoragePaths
from scripts.models.run_unet_agglomerated_independent_test_analysis import (
    MODEL_ID,
    TEST_FILENAMES,
    TORCHSCRIPT_SHA256,
    IndependentTestParameters,
    _validate_inputs,
    _validated_parameters,
    build_parser,
    execute_analyses,
)
from scripts.models.smoke_unet_agglomerated_analysis import load_calibrated_analysis


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
    freeze_model_bundle: Any = None

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
            adapter_path="tests.fake:FakeAdapter",
            weight_sha256=TORCHSCRIPT_SHA256,
            config_sha256="b" * 64,
            model_card_sha256="c" * 64,
            adapter_sha256="d" * 64,
        )

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        assert not only_ready or self.model.status == ModelStatus.READY
        return [self.model]

    def health(self) -> list[ModelHealth]:
        return [
            ModelHealth(
                model_id=MODEL_ID, status=ModelStatus.READY, weight_sha256=TORCHSCRIPT_SHA256
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
        assert expected_model_version == "1"
        assert expected_adapter_path == "tests.fake:FakeAdapter"
        assert expected_weight_sha256 == TORCHSCRIPT_SHA256
        assert expected_config_sha256 == "b" * 64
        assert expected_model_card_sha256 == "c" * 64
        assert expected_adapter_sha256 is None
        assert model_bundle is None
        mask = np.zeros((240, 80), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        path = request.run_dir / "fake-mask.png"
        Image.fromarray(mask).save(path)
        return SegmentationOutput(width=80, height=240, binary_mask_path=path, runtime_ms=1)


def _write_images(
    image_dir: Path, *, names: tuple[str, ...] = TEST_FILENAMES, size: tuple[int, int] = (80, 240)
) -> None:
    image_dir.mkdir()
    pixels = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
    for index, name in enumerate(names):
        Image.fromarray((pixels + index).astype(np.uint8)).save(image_dir / name)


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


def test_three_ycu_full_image_runs_are_created_and_mapped_without_evaluation_reads(
    tmp_path: Path,
) -> None:
    image_dir, output_root = tmp_path / "images", tmp_path / "output"
    output_root.mkdir()
    _write_images(image_dir)
    database, gateway = _database(output_root), FakeGateway()
    with database.session() as session:
        session.add(
            ModelRegistryRecord(
                model_id=MODEL_ID,
                family="unet",
                variant="dense_particle",
                quality_tier="balanced",
                version="1",
                adapter="tests.fake:FakeAdapter",
                status="ready",
            )
        )
    try:
        result = execute_analyses(
            IndependentTestParameters(image_dir, tmp_path / "private-ready.yaml", output_root),
            load_calibrated_analysis(_config(), gateway.model),
            database=database,
            file_store=LocalFileStore(
                StoragePaths(output_root / "artifacts"),
                max_upload_bytes=max(path.stat().st_size for path in image_dir.iterdir()),
            ),
            gateway=gateway,
        )
    finally:
        database.dispose()

    assert result["inputs"] == list(TEST_FILENAMES)
    assert "ground truth or evaluation metrics were read" in result["scope"]
    assert result["public_registry_status"] == "unavailable (unchanged)"
    assert len(gateway.requests) == len(TEST_FILENAMES) == len(result["runs"])
    assert {run["filename"] for run in result["runs"]} == set(TEST_FILENAMES)
    assert len({run["run_id"] for run in result["runs"]}) == 3
    assert len({run["image_id"] for run in result["runs"]}) == 3
    for run in result["runs"]:
        assert run["sample_id"] == Path(run["filename"]).stem
        assert Path(run["image_metadata"]).is_file()
        assert Path(run["pred_mask"]).is_file()
        assert Path(run["run_config"]).is_file()
        assert run["prediction_bottom_check"] == {
            "bottom_rows_checked": 130,
            "bottom_nonzero_pixels": 0,
            "bottom_prediction_is_zero": True,
        }
        assert run["frozen_inference"] == {
            "threshold": 0.25,
            "min_area_px": 1024,
            "watershed_enabled": False,
            "exclude_border": True,
            "device": "cpu",
            "seed": 2026,
        }


def test_input_set_requires_only_the_three_top_level_ycu_targets(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_images(image_dir, names=("YCu-1.tif", "YCu-2.tif"), size=(2048, 1536))
    with pytest.raises(ValueError, match="missing"):
        _validate_inputs(image_dir)

    # Extra top-level TIFF files are unrelated and must not be selected.
    Image.new("L", (2048, 1536)).save(image_dir / "Other.tif")
    Image.new("L", (2048, 1536)).save(image_dir / "YCu-3.tif")
    assert _validate_inputs(image_dir) == image_dir.resolve()

    # A target found below a child directory is not an input candidate.
    (image_dir / "nested").mkdir()
    (image_dir / "YCu-3.tif").unlink()
    Image.new("L", (2048, 1536)).save(image_dir / "nested" / "YCu-3.tif")
    with pytest.raises(ValueError, match="top-level"):
        _validate_inputs(image_dir)


def test_cli_has_no_ground_truth_or_metric_inputs_and_requires_fixed_external_output(
    tmp_path: Path,
) -> None:
    image_dir, registry, output_root = (
        tmp_path / "images",
        tmp_path / "private-ready.yaml",
        tmp_path / "result",
    )
    _write_images(image_dir, size=(2048, 1536))
    registry.write_text("models: []\n", encoding="utf-8")
    parser = build_parser()
    options = [option for action in parser._actions for option in action.option_strings]
    assert not any("mask" in option or "metric" in option for option in options)
    validated = _validated_parameters(
        argparse.Namespace(
            image_dir=image_dir, registry=registry, output_root=output_root, model_id=MODEL_ID
        ),
        expected_output_root=output_root,
    )
    assert validated.output_root == output_root.resolve()
    output_root.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _validated_parameters(
            argparse.Namespace(
                image_dir=image_dir,
                registry=registry,
                output_root=output_root,
                model_id=MODEL_ID,
            ),
            expected_output_root=output_root,
        )
    with pytest.raises(ValueError, match="fixed independent-test root"):
        _validated_parameters(
            argparse.Namespace(
                image_dir=image_dir,
                registry=registry,
                output_root=tmp_path / "wrong",
                model_id=MODEL_ID,
            ),
            expected_output_root=output_root,
        )
