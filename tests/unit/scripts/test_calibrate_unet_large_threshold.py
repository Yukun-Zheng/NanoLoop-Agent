from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.enums import (
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
)
from app.contracts.execution import InferenceExecutionEvidence
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelMetadata
from scripts.models import calibrate_unet_large_threshold as threshold_module
from scripts.models.calibrate_unet_large_threshold import (
    ADAPTER_SHA256,
    BOTTOM_CROP_PX,
    CANDIDATE_THRESHOLDS,
    CONFIG_SHA256,
    CURRENT_EXPERIMENT_THRESHOLD,
    MODEL_CARD_SHA256,
    SEED,
    TORCHSCRIPT_SHA256,
    VALIDATION_FILENAMES,
    _validate_output_root,
    cache_probabilities,
    evaluate_thresholds,
    select_threshold,
)


class FakeGateway:
    def __init__(self) -> None:
        self.requests: list[SegmentationRequest] = []

    def predict(self, _model_id: str, request: SegmentationRequest, **_kwargs: Any) -> Any:
        self.requests.append(request)
        probability_path = request.run_dir / "probability.npy"
        mask_path = request.run_dir / "binary-mask.png"
        np.save(probability_path, np.zeros((181, 2), dtype=np.float32), allow_pickle=False)
        Image.fromarray(np.zeros((181, 2), dtype=np.uint8)).save(mask_path)
        return SegmentationOutput(
            width=2,
            height=181,
            probability_path=probability_path,
            binary_mask_path=mask_path,
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


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="unet-large-optimized-v1",
        family=ModelFamily.UNET,
        variant=ModelVariant.LARGE_PARTICLE,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.50,
        preprocess_profile="sem-gray-unit-crop-bottom-180-v1",
        postprocess_profile="semantic-mask-v1",
        inference_invalid_bottom_px=180,
        expected_input_width=2048,
        expected_input_height=1536,
        adapter_path="app.inference.adapters.unet:UNetAdapter",
        weight_sha256=TORCHSCRIPT_SHA256,
        config_sha256=CONFIG_SHA256,
        model_card_sha256=MODEL_CARD_SHA256,
        adapter_sha256=ADAPTER_SHA256,
    )


def _bundle() -> ModelBundleReference:
    bundle_id = "e" * 64
    return ModelBundleReference(
        bundle_id=bundle_id,
        manifest_ref=f"bundles/{bundle_id}/manifest.json",
        weight_ref=f"{TORCHSCRIPT_SHA256}/weights.pt",
        config_ref=f"{CONFIG_SHA256}/config.yaml",
        model_card_ref=f"{MODEL_CARD_SHA256}/model-card.md",
        adapter_ref=f"{ADAPTER_SHA256}/adapter.py",
        adapter_sha256=ADAPTER_SHA256,
    )


def test_fixed_validation_and_threshold_contract() -> None:
    assert VALIDATION_FILENAMES == (
        "NdZn-2.tif",
        "LaMn-3.tif",
        "LaMn-1.tif",
        "BaCo-3.tif",
        "BaCu-1.tif",
        "BaCr-3.tif",
    )
    expected_thresholds = tuple(round(value / 100, 2) for value in range(20, 81, 5))
    assert expected_thresholds == CANDIDATE_THRESHOLDS
    assert BOTTOM_CROP_PX == 180
    assert SEED == 2026


def test_gateway_is_called_once_per_validation_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threshold_module, "IMAGE_WIDTH", 2)
    monkeypatch.setattr(threshold_module, "IMAGE_HEIGHT", 181)
    image_paths: list[Path] = []
    for filename in VALIDATION_FILENAMES:
        path = tmp_path / filename
        Image.fromarray(np.zeros((181, 2), dtype=np.uint8)).save(path)
        image_paths.append(path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    gateway = FakeGateway()

    probability_paths, evidence = cache_probabilities(
        gateway=gateway,
        metadata=_metadata(),
        model_bundle=_bundle(),
        image_paths=image_paths,
        output_root=output_root,
    )

    assert len(gateway.requests) == len(VALIDATION_FILENAMES)
    assert set(probability_paths) == set(VALIDATION_FILENAMES)
    assert set(evidence) == set(VALIDATION_FILENAMES)
    assert all(request.device.value == "cpu" for request in gateway.requests)
    assert all(request.seed == 2026 for request in gateway.requests)
    assert all(request.min_area_px == 0 for request in gateway.requests)
    assert all(request.threshold == 0.50 for request in gateway.requests)
    assert all(item["probability_cache"]["sha256"] for item in evidence.values())
    assert all(
        item["model"]["model_bundle"] == _bundle().model_dump(mode="json")
        for item in evidence.values()
    )


def test_threshold_evaluation_is_strict_and_excludes_bottom_180_pixels() -> None:
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        probability = np.ones((181, 2), dtype=np.float32)
        target = np.zeros((181, 2), dtype=bool)
        probability[0] = [CURRENT_EXPERIMENT_THRESHOLD, 0.61]
        target[0] = [True, True]
        probabilities[filename] = probability
        targets[filename] = target

    result = evaluate_thresholds(
        probabilities,
        targets,
        thresholds=[CURRENT_EXPERIMENT_THRESHOLD],
    )[0]

    for image_result in result["images"]:
        assert image_result["tp"] == 1
        assert image_result["fp"] == 0
        assert image_result["fn"] == 1
        assert image_result["tn"] == 0
    assert result["micro"]["tp"] == 6
    assert result["micro"]["fp"] == 0
    assert result["micro"]["fn"] == 6
    assert result["micro"]["tn"] == 0


def test_selection_rule_uses_dice_then_iou_then_distance_from_point_six() -> None:
    results = [
        {"threshold": 0.40, "macro": {"dice": 0.8, "iou": 0.7}},
        {"threshold": 0.55, "macro": {"dice": 0.8, "iou": 0.75}},
        {"threshold": 0.65, "macro": {"dice": 0.8, "iou": 0.75}},
        {"threshold": 0.60, "macro": {"dice": 0.79, "iou": 0.9}},
    ]

    with pytest.raises(ValueError, match="did not resolve the tie"):
        select_threshold(results)

    results[2]["macro"]["iou"] = 0.74
    assert select_threshold(results)["threshold"] == 0.55


@pytest.mark.parametrize("kind", ["existing", "repository"])
def test_output_root_protection(tmp_path: Path, kind: str) -> None:
    output_root = (
        tmp_path / "already-exists"
        if kind == "existing"
        else Path(__file__).resolve().parents[3] / "forbidden-large-calibration"
    )
    if kind == "existing":
        output_root.mkdir()

    with pytest.raises(ValueError, match="output-root"):
        _validate_output_root(output_root)
