"""Calibrate the large U-Net threshold from six fixed validation fields of view."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.contracts.enums import DevicePreference, ModelStatus, RoiMode
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelBundleReference, ModelMetadata
from app.inference.gateway import InferenceGateway
from app.inference.registry import ModelRegistryService

MODEL_ID = "unet-large-optimized-v1"
MODEL_VERSION = "1"
ADAPTER_PATH = "app.inference.adapters.unet:UNetAdapter"
TORCHSCRIPT_SHA256 = "007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05"
# Refresh these centralized source-asset digests together when the version-1 bundle changes.
CONFIG_SHA256 = "4e48c75d960faaa17868f0318da5526a6ba72211396ec106c2e57ce7eecc8856"
MODEL_CARD_SHA256 = "5dedae0a718f57805a939484493bd079cc56a78d205fe98c55feebb86581a4ec"
ADAPTER_SHA256 = "6055db452f0a78a0352732d66ea3436f16a558cf19d1a6f022a78627136dfab6"
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
BOTTOM_CROP_PX = 180
SEED = 2026
CURRENT_EXPERIMENT_THRESHOLD = 0.60
FROZEN_THRESHOLD = 0.50
VALIDATION_FILENAMES = (
    "NdZn-2.tif",
    "LaMn-3.tif",
    "LaMn-1.tif",
    "BaCo-3.tif",
    "BaCu-1.tif",
    "BaCr-3.tif",
)
CANDIDATE_THRESHOLDS = (
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)
EXPECTED_CONFIG = {
    "loader": "torchscript",
    "input_channels": 1,
    "input_size": [512, 512],
    "expected_image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
    "patch_size": [512, 512],
    "stride": [256, 256],
    "tiling_padding": "reflect",
    "overlap_fusion": "uniform",
    "bottom_crop_px": BOTTOM_CROP_PX,
    "pixel_scale": 255.0,
    "mean": [0.0],
    "std": [1.0],
    "output_activation": "logits",
    "threshold_comparison": "gt",
    "default_threshold": FROZEN_THRESHOLD,
}


@dataclass(frozen=True, slots=True)
class CalibrationPaths:
    image_dir: Path
    mask_dir: Path
    registry: Path
    output_root: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_exact_dataset_layout(directory: Path, *, label: str) -> None:
    expected = set(VALIDATION_FILENAMES)
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    observed_directories = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if observed_files != expected or observed_directories:
        raise ValueError(
            f"{label} must contain exactly the six fixed validation files at its root: "
            f"missing={sorted(expected - observed_files)}, "
            f"unexpected={sorted(observed_files - expected)}, "
            f"directories={observed_directories}"
        )


def _image_identity(path: Path, *, label: str) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(
                    f"{label} dimensions are not {IMAGE_WIDTH}x{IMAGE_HEIGHT}: {path.name}"
                )
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256(path),
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")
    return resolved


def _validated_paths(namespace: argparse.Namespace) -> CalibrationPaths:
    image_dir = namespace.image_dir.expanduser().resolve(strict=True)
    mask_dir = namespace.mask_dir.expanduser().resolve(strict=True)
    registry = namespace.registry.expanduser().resolve(strict=True)
    output_root = _validate_output_root(namespace.output_root)
    if not image_dir.is_dir():
        raise ValueError(f"image-dir is not a directory: {image_dir}")
    if not mask_dir.is_dir():
        raise ValueError(f"mask-dir is not a directory: {mask_dir}")
    if not registry.is_file():
        raise ValueError(f"registry is not a file: {registry}")
    _validate_exact_dataset_layout(image_dir, label="image-dir")
    _validate_exact_dataset_layout(mask_dir, label="mask-dir")
    for filename in VALIDATION_FILENAMES:
        if not (image_dir / filename).is_file():
            raise ValueError(f"validation image is missing: {filename}")
        if not (mask_dir / filename).is_file():
            raise ValueError(f"validation mask is missing: {filename}")
    return CalibrationPaths(
        image_dir=image_dir,
        mask_dir=mask_dir,
        registry=registry,
        output_root=output_root,
    )


def _validate_model_contract(
    metadata: ModelMetadata,
    config: Mapping[str, Any],
) -> None:
    if metadata.model_id != MODEL_ID:
        raise ValueError(f"unexpected model id: {metadata.model_id}")
    if metadata.version != MODEL_VERSION:
        raise ValueError(f"unexpected model version: {metadata.version}")
    if metadata.status != ModelStatus.READY:
        raise ValueError(f"private registry model is not ready: {metadata.status.value}")
    if metadata.weight_sha256 != TORCHSCRIPT_SHA256:
        raise ValueError("private registry TorchScript SHA-256 does not match the calibrated model")
    expected_provenance = {
        "adapter_path": ADAPTER_PATH,
        "config_sha256": CONFIG_SHA256,
        "model_card_sha256": MODEL_CARD_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "expected_input_width": IMAGE_WIDTH,
        "expected_input_height": IMAGE_HEIGHT,
    }
    provenance_mismatches = {
        name: {"expected": expected, "observed": getattr(metadata, name)}
        for name, expected in expected_provenance.items()
        if getattr(metadata, name) != expected
    }
    if provenance_mismatches:
        raise ValueError(
            "private registry model provenance differs from the frozen Large bundle: "
            + json.dumps(provenance_mismatches, ensure_ascii=False, sort_keys=True)
        )
    if metadata.inference_invalid_bottom_px != BOTTOM_CROP_PX:
        raise ValueError("private registry invalid-bottom metadata must be 180 px")
    if metadata.default_threshold != FROZEN_THRESHOLD:
        raise ValueError("private registry default threshold must be the frozen 0.50")
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "private registry config does not match the fixed large inference contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _validate_bundle(bundle: ModelBundleReference) -> None:
    expected = {
        "manifest_ref": f"bundles/{bundle.bundle_id}/manifest.json",
        "weight_ref": f"{TORCHSCRIPT_SHA256}/weights.pt",
        "config_ref": f"{CONFIG_SHA256}/config.yaml",
        "model_card_ref": f"{MODEL_CARD_SHA256}/model-card.md",
        "adapter_ref": f"{ADAPTER_SHA256}/adapter.py",
        "adapter_sha256": ADAPTER_SHA256,
    }
    mismatches = {
        name: {"expected": value, "observed": getattr(bundle, name)}
        for name, value in expected.items()
        if getattr(bundle, name) != value
    }
    if mismatches:
        raise ValueError(
            "frozen model bundle does not bind the expected Large assets: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _validate_probability(path: Path, *, filename: str) -> np.ndarray:
    probability = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if probability.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"probability dimensions are not {IMAGE_WIDTH}x{IMAGE_HEIGHT} for {filename}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"probability contains non-finite values for {filename}")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(f"probability is outside [0, 1] for {filename}")
    return probability


def _managed_probability_path(path: Path, output_root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(output_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError("gateway probability output escaped output-root") from error
    if resolved.suffix.lower() != ".npy":
        raise ValueError("gateway probability output must be a .npy file")
    return resolved


def cache_probabilities(
    *,
    gateway: Any,
    metadata: ModelMetadata,
    model_bundle: ModelBundleReference,
    image_paths: Sequence[Path],
    output_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    """Call the gateway exactly once per image and retain each raw probability array."""

    if tuple(path.name for path in image_paths) != VALIDATION_FILENAMES:
        raise ValueError("gateway inputs do not exactly match the fixed validation order")
    _validate_bundle(model_bundle)
    probability_paths: dict[str, Path] = {}
    execution_evidence: dict[str, dict[str, object]] = {}
    for image_path in image_paths:
        image_identity = _image_identity(image_path, label="validation image")
        run_dir = output_root / "probability-cache" / image_path.stem
        run_dir.mkdir(parents=True, exist_ok=False)
        output = gateway.predict(
            metadata.model_id,
            SegmentationRequest(
                image_id=image_path.stem,
                image_path=image_path,
                image_bytes=image_path.read_bytes(),
                run_dir=run_dir,
                roi_mode=RoiMode.FULL_IMAGE,
                threshold=FROZEN_THRESHOLD,
                min_area_px=0,
                device=DevicePreference.CPU,
                seed=SEED,
            ),
            expected_model_version=metadata.version,
            expected_adapter_path=metadata.adapter_path,
            expected_weight_sha256=metadata.weight_sha256,
            expected_config_sha256=metadata.config_sha256,
            expected_model_card_sha256=metadata.model_card_sha256,
            expected_adapter_sha256=metadata.adapter_sha256,
            model_bundle=model_bundle,
        )
        if output.probability_path is None:
            raise ValueError(f"gateway did not return probability output for {image_path.name}")
        if (output.width, output.height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError(f"gateway output dimensions differ for {image_path.name}")
        probability_path = _managed_probability_path(
            output.probability_path,
            output_root,
        )
        probability = _validate_probability(probability_path, filename=image_path.name)
        execution = output.execution
        if execution is None:
            raise ValueError(f"gateway returned no execution evidence for {image_path.name}")
        if execution.actual_device != "cpu":
            raise ValueError(f"gateway did not execute on CPU for {image_path.name}")
        controls = (
            execution.python_random_seeded,
            execution.numpy_random_seeded,
            execution.torch_deterministic_algorithms,
            execution.global_inference_serialized,
        )
        if not all(controls) or not execution.backend.endswith(".UNetAdapter"):
            raise ValueError(f"gateway execution controls are incomplete for {image_path.name}")
        probability_paths[image_path.name] = probability_path
        execution_evidence[image_path.name] = {
            "sample_id": image_path.stem,
            "input_image": image_identity,
            "request": {
                "roi_mode": RoiMode.FULL_IMAGE.value,
                "threshold": FROZEN_THRESHOLD,
                "min_area_px": 0,
                "device": DevicePreference.CPU.value,
                "seed": SEED,
            },
            "model": {
                "model_id": metadata.model_id,
                "model_version": metadata.version,
                "adapter_path": metadata.adapter_path,
                "weight_sha256": metadata.weight_sha256,
                "config_sha256": metadata.config_sha256,
                "model_card_sha256": metadata.model_card_sha256,
                "adapter_sha256": metadata.adapter_sha256,
                "model_bundle": model_bundle.model_dump(mode="json"),
            },
            "probability_cache": {
                "path": str(probability_path.relative_to(output_root.resolve(strict=True))),
                "sha256": _sha256(probability_path),
                "shape": list(probability.shape),
                "dtype": str(probability.dtype),
                "minimum": float(np.min(probability)),
                "maximum": float(np.max(probability)),
                "finite": True,
                "range": [0.0, 1.0],
            },
            "execution": execution.model_dump(mode="json"),
        }
    return probability_paths, execution_evidence


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _result_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, int | float]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_ratio(tp, tp + fp + fn),
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
    }


def _confusion(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    return tp, fp, fn, tn


def evaluate_thresholds(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
    bottom_crop_px: int = BOTTOM_CROP_PX,
) -> list[dict[str, object]]:
    if tuple(probabilities) != tuple(targets):
        raise ValueError("probability and target image order differs")
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        image_results: list[dict[str, object]] = []
        total_tp = total_fp = total_fn = total_tn = 0
        for filename in probabilities:
            probability = np.asarray(probabilities[filename], dtype=np.float32)
            target = np.asarray(targets[filename], dtype=bool)
            if probability.shape != target.shape:
                raise ValueError(f"probability/GT shape mismatch for {filename}")
            if probability.ndim != 2 or probability.shape[0] <= bottom_crop_px:
                raise ValueError(f"invalid full-image shape for {filename}: {probability.shape}")
            if not np.isfinite(probability).all():
                raise ValueError(f"probability contains non-finite values for {filename}")
            valid_probability = probability[:-bottom_crop_px]
            valid_target = target[:-bottom_crop_px]
            prediction = valid_probability > threshold
            tp, fp, fn, tn = _confusion(prediction, valid_target)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            image_results.append({"filename": filename, **_metrics(tp, fp, fn, tn)})
        macro = {
            metric: float(
                np.mean([_result_float(item[metric], field=metric) for item in image_results])
            )
            for metric in ("dice", "iou", "precision", "recall")
        }
        results.append(
            {
                "threshold": float(threshold),
                "images": image_results,
                "macro": macro,
                "micro": _metrics(total_tp, total_fp, total_fn, total_tn),
            }
        )
    return results


def select_threshold(results: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not results:
        raise ValueError("threshold results are empty")
    best_dice = max(_macro_metric(item, "dice") for item in results)
    dice_winners = [item for item in results if _macro_metric(item, "dice") == best_dice]
    best_iou = max(_macro_metric(item, "iou") for item in dice_winners)
    iou_winners = [item for item in dice_winners if _macro_metric(item, "iou") == best_iou]
    best_distance = min(
        round(
            abs(
                _result_float(item.get("threshold"), field="threshold")
                - CURRENT_EXPERIMENT_THRESHOLD
            ),
            12,
        )
        for item in iou_winners
    )
    winners = [
        item
        for item in iou_winners
        if round(
            abs(
                _result_float(item.get("threshold"), field="threshold")
                - CURRENT_EXPERIMENT_THRESHOLD
            ),
            12,
        )
        == best_distance
    ]
    if len(winners) != 1:
        raise ValueError("the prespecified threshold selection rule did not resolve the tie")
    return winners[0]


def _macro_metric(result: Mapping[str, object], metric: str) -> float:
    macro = result.get("macro")
    if not isinstance(macro, Mapping) or metric not in macro:
        raise ValueError(f"threshold result is missing macro {metric}")
    return float(macro[metric])


def _load_full_size_targets(mask_dir: Path) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        with Image.open(mask_dir / filename) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(f"GT dimensions are not 2048x1536 for {filename}")
            targets[filename] = np.asarray(image.convert("L")) > 0
    return targets


def _load_probabilities(paths: Mapping[str, Path]) -> dict[str, np.ndarray]:
    if tuple(paths) != VALIDATION_FILENAMES:
        raise ValueError("probability cache does not match the fixed validation order")
    return {
        filename: _validate_probability(path, filename=filename) for filename, path in paths.items()
    }


def _write_comparisons(
    output_root: Path,
    targets: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    threshold: float,
) -> dict[str, str]:
    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, str] = {}
    for filename in VALIDATION_FILENAMES:
        target = np.asarray(targets[filename][:-BOTTOM_CROP_PX], dtype=np.uint8) * 255
        prediction = (np.asarray(probabilities[filename][:-BOTTOM_CROP_PX]) > threshold).astype(
            np.uint8
        ) * 255
        comparison = np.concatenate((target, prediction), axis=1)
        destination = comparison_dir / f"{Path(filename).stem}-gt-pred.png"
        Image.fromarray(comparison, mode="L").save(destination)
        paths[filename] = str(destination)
    return paths


def _write_csv(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "threshold",
        "scope",
        "filename",
        "tp",
        "fp",
        "fn",
        "tn",
        "dice",
        "iou",
        "precision",
        "recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            threshold = result["threshold"]
            image_results = result.get("images")
            macro = result.get("macro")
            micro = result.get("micro")
            if (
                not isinstance(image_results, list)
                or not isinstance(macro, Mapping)
                or not isinstance(micro, Mapping)
            ):
                raise ValueError("threshold result has an invalid output structure")
            for image_result in image_results:
                if not isinstance(image_result, Mapping):
                    raise ValueError("threshold image result is invalid")
                writer.writerow({"threshold": threshold, "scope": "image", **image_result})
            writer.writerow(
                {
                    "threshold": threshold,
                    "scope": "macro",
                    **macro,
                }
            )
            writer.writerow(
                {
                    "threshold": threshold,
                    "scope": "micro",
                    **micro,
                }
            )


def run(paths: CalibrationPaths) -> dict[str, object]:
    _validate_exact_dataset_layout(paths.image_dir, label="image-dir")
    _validate_exact_dataset_layout(paths.mask_dir, label="mask-dir")
    output_root = _validate_output_root(paths.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    registry = ModelRegistryService(
        paths.registry,
        snapshot_root=output_root / "model-snapshots",
    )
    if registry.registry_error is not None:
        raise ValueError(f"private registry is invalid: {registry.registry_error}")
    registration = registry.get_registration(MODEL_ID)
    metadata = registration.metadata
    _validate_model_contract(metadata, registration.config)
    gateway = InferenceGateway(registry)
    bundle = gateway.freeze_model_bundle(
        MODEL_ID,
        expected_model_version=metadata.version,
        expected_adapter_path=metadata.adapter_path,
        expected_weight_sha256=metadata.weight_sha256,
        expected_config_sha256=metadata.config_sha256,
        expected_model_card_sha256=metadata.model_card_sha256,
        expected_adapter_sha256=metadata.adapter_sha256,
    )
    _validate_bundle(bundle)
    image_paths = [paths.image_dir / filename for filename in VALIDATION_FILENAMES]
    probability_paths, execution_evidence = cache_probabilities(
        gateway=gateway,
        metadata=metadata,
        model_bundle=bundle,
        image_paths=image_paths,
        output_root=output_root,
    )
    probabilities = _load_probabilities(probability_paths)
    targets = _load_full_size_targets(paths.mask_dir)
    input_evidence: dict[str, dict[str, object]] = {}
    for filename in VALIDATION_FILENAMES:
        ground_truth = _image_identity(
            paths.mask_dir / filename,
            label="validation ground truth",
        )
        sample_evidence = execution_evidence[filename]
        input_evidence[filename] = {
            "sample_id": Path(filename).stem,
            "input_image": sample_evidence["input_image"],
            "ground_truth": ground_truth,
            "probability_cache": sample_evidence["probability_cache"],
        }
    results = evaluate_thresholds(probabilities, targets)
    selected = select_threshold(results)
    selected_threshold = _result_float(selected.get("threshold"), field="selected.threshold")
    comparison_paths = _write_comparisons(
        output_root,
        targets,
        probabilities,
        selected_threshold,
    )
    csv_path = output_root / "threshold-calibration.csv"
    json_path = output_root / "threshold-calibration.json"
    _write_csv(csv_path, results)
    probability_artifacts = {
        name: {
            "path": str(path.relative_to(output_root)),
            "sha256": _sha256(path),
        }
        for name, path in probability_paths.items()
    }
    comparison_artifacts = {
        name: {
            "path": str(Path(path).relative_to(output_root)),
            "sha256": _sha256(Path(path)),
        }
        for name, path in comparison_paths.items()
    }
    payload: dict[str, object] = {
        "evidence_schema_version": 2,
        "evidence_status": (
            "future rerun contract; this file is evidence only after a successful execution"
        ),
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "model_card_sha256": MODEL_CARD_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "model_bundle": bundle.model_dump(mode="json"),
        "validation_images": list(VALIDATION_FILENAMES),
        "validation_scope": (
            "non-overlapping field-of-view validation; not sample-level independent"
        ),
        "device": DevicePreference.CPU.value,
        "seed": SEED,
        "bottom_crop_px": BOTTOM_CROP_PX,
        "comparison_rule": "probability > threshold",
        "inference_contract": {
            "input_color_mode": "grayscale",
            **EXPECTED_CONFIG,
        },
        "metric_zero_division_policy": "return 1.0 when the metric denominator is zero",
        "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
        "selection_rule": [
            "maximize macro Dice",
            "then maximize macro IoU",
            "then minimize absolute distance from 0.60",
            "fail if the prespecified rules still leave a tie",
        ],
        "selected": selected,
        "threshold_results": results,
        "input_evidence": input_evidence,
        "artifacts": {
            "json": str(json_path.relative_to(output_root)),
            "csv": {
                "path": str(csv_path.relative_to(output_root)),
                "sha256": _sha256(csv_path),
            },
            "probabilities": probability_artifacts,
            "comparisons": comparison_artifacts,
        },
        "execution_evidence": execution_evidence,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        paths = _validated_paths(build_parser().parse_args(argv))
        payload = run(paths)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
