"""Calibrate agglomerated U-Net threshold on four fixed validation fields only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from skimage.measure import label

from app.contracts.enums import DevicePreference, ModelStatus, RoiMode
from app.contracts.inference import SegmentationRequest
from app.inference.gateway import InferenceGateway
from app.inference.registry import ModelRegistryService

MODEL_ID = "unet-agglomerated-specialized-v1"
TORCHSCRIPT_SHA256 = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"
CHECKPOINT_SHA256 = "e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21"
VALIDATION_FILENAMES = ("BiCu-3.tif", "BaNi-3.tif", "BaNi-1.tif", "BaNi-2.tif")
INDEPENDENT_TEST_FILENAMES = ("YCu-1.tif", "YCu-2.tif", "YCu-3.tif")
CANDIDATE_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
BOTTOM_CROP_PX = 130
VALID_HEIGHT = IMAGE_HEIGHT - BOTTOM_CROP_PX
SCALE_NM_PER_PIXEL = 100 / 184
DETECTION_COVERAGE_THRESHOLD = 0.30
SIZE_BUCKETS = ("tiny", "small", "large")
EXPECTED_CONFIG = {
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
    "bottom_crop_px": BOTTOM_CROP_PX,
    "normalization": "percentile",
    "lower_percentile": 1.0,
    "upper_percentile": 99.0,
    "output_activation": "logits",
    "threshold_comparison": "gte",
    "default_threshold": 0.25,
    "target_definition": "whole_agglomerate",
    "default_watershed_enabled": False,
    "scale_nm_per_pixel": SCALE_NM_PER_PIXEL,
}


ProbabilityInferencer = Callable[[Path], np.ndarray]


@dataclass(frozen=True, slots=True)
class CalibrationPaths:
    image_dir: Path
    mask_dir: Path
    torchscript: Path
    output_root: Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    return _repository_root() / "model_artifacts" / "configs" / f"{MODEL_ID}.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_root(output_root: Path, *, repository: Path | None = None) -> Path:
    resolved = output_root.expanduser().resolve(strict=False)
    repository = (repository or _repository_root()).resolve(strict=True)
    if resolved.exists():
        raise ValueError(f"output-root already exists: {resolved}")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("output-root must be outside the repository")
    return resolved


def validate_validation_filenames(filenames: Sequence[str]) -> tuple[str, ...]:
    observed = tuple(filenames)
    forbidden = sorted(set(observed) & set(INDEPENDENT_TEST_FILENAMES))
    if forbidden:
        raise ValueError(f"independent test samples are forbidden during calibration: {forbidden}")
    if observed != VALIDATION_FILENAMES:
        raise ValueError("calibration input must be exactly the four frozen validation images")
    return observed


def _reject_test_directory(path: Path, *, field: str) -> None:
    forbidden_parts = {"test_images", "test_mask_human"}
    if any(part.casefold() in forbidden_parts for part in path.parts):
        raise ValueError(f"{field} must not point to an independent test directory")


def _validated_paths(namespace: argparse.Namespace) -> CalibrationPaths:
    image_dir = namespace.image_dir.expanduser().resolve(strict=True)
    mask_dir = namespace.mask_dir.expanduser().resolve(strict=True)
    torchscript = namespace.torchscript.expanduser().resolve(strict=True)
    output_root = _validate_output_root(namespace.output_root)
    if not image_dir.is_dir():
        raise ValueError(f"image-dir is not a directory: {image_dir}")
    if not mask_dir.is_dir():
        raise ValueError(f"mask-dir is not a directory: {mask_dir}")
    if not torchscript.is_file():
        raise ValueError(f"torchscript is not a file: {torchscript}")
    _reject_test_directory(image_dir, field="image-dir")
    _reject_test_directory(mask_dir, field="mask-dir")
    validate_validation_filenames(VALIDATION_FILENAMES)
    for filename in VALIDATION_FILENAMES:
        if not (image_dir / filename).is_file():
            raise ValueError(f"validation image is missing: {filename}")
        if not (mask_dir / filename).is_file():
            raise ValueError(f"validation mask is missing: {filename}")
    return CalibrationPaths(image_dir, mask_dir, torchscript, output_root)


def _load_config() -> dict[str, Any]:
    payload = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("agglomerated config must be a mapping")
    if payload != EXPECTED_CONFIG:
        raise ValueError("agglomerated config differs from the frozen A-stage inference contract")
    return payload


def _verify_torchscript(path: Path) -> str:
    observed = _sha256(path)
    if observed != TORCHSCRIPT_SHA256:
        raise ValueError(
            "TorchScript SHA-256 differs from the frozen agglomerated asset: " + observed
        )
    return observed


class GatewayProbabilityInferencer:
    """Generate calibration probabilities only through the frozen Gateway contract."""

    def __init__(self, torchscript: Path, config: Mapping[str, Any]) -> None:
        if dict(config) != EXPECTED_CONFIG:
            raise ValueError("agglomerated config differs from the frozen inference contract")
        self._temporary = tempfile.TemporaryDirectory(
            prefix="nanoloop-agglomerated-calibration-gateway-"
        )
        root = Path(self._temporary.name)
        registry_path = root / "registry.yaml"
        config_path = _config_path().resolve(strict=True)
        model_card_path = (
            _repository_root()
            / "model_artifacts"
            / "model_cards"
            / f"{MODEL_ID}.md"
        ).resolve(strict=True)
        registry_payload = {
            "schema_version": "2.0",
            "models": [
                {
                    "metadata": {
                        "model_id": MODEL_ID,
                        "family": "unet",
                        "variant": "dense_particle",
                        "quality_tier": "balanced",
                        "version": "1",
                        "status": "ready",
                        "supports_box_prompt": False,
                        "default_threshold": 0.25,
                        "default_min_area_px": 1024,
                        "preprocess_profile": "sem-gray-p1-p99-crop-bottom-130-v1",
                        "postprocess_profile": "semantic-agglomerate-mask-v1",
                        "inference_invalid_bottom_px": BOTTOM_CROP_PX,
                    },
                    "adapter_path": "app.inference.adapters.unet:UNetAdapter",
                    "weight_path": str(torchscript.resolve(strict=True)),
                    "weight_sha256": TORCHSCRIPT_SHA256,
                    "config_path": str(config_path),
                    "model_card_path": str(model_card_path),
                    "required_modules": ["torch"],
                }
            ],
        }
        registry_path.write_text(
            yaml.safe_dump(registry_payload, sort_keys=False), encoding="utf-8"
        )
        registry = ModelRegistryService(
            registry_path,
            snapshot_root=root / "model-snapshots",
        )
        if registry.registry_error is not None:
            self._temporary.cleanup()
            raise ValueError(
                f"generated calibration registry is invalid: {registry.registry_error}"
            )
        self.metadata = registry.get_metadata(MODEL_ID)
        if self.metadata.status != ModelStatus.READY:
            reason = self.metadata.health_error or "unknown readiness error"
            self._temporary.cleanup()
            raise ValueError(f"generated calibration registry is not ready: {reason}")
        self.gateway = InferenceGateway(registry)
        self.bundle = self.gateway.freeze_model_bundle(
            MODEL_ID,
            expected_model_version=self.metadata.version,
            expected_adapter_path=self.metadata.adapter_path,
            expected_weight_sha256=self.metadata.weight_sha256,
            expected_config_sha256=self.metadata.config_sha256,
            expected_model_card_sha256=self.metadata.model_card_sha256,
            expected_adapter_sha256=self.metadata.adapter_sha256,
        )
        self.run_root = root / "runs"
        self.run_root.mkdir()
        self.execution_evidence: dict[str, dict[str, object] | None] = {}

    def __call__(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(f"validation image must be 2048x1536: {image_path.name}")
        run_dir = self.run_root / image_path.stem
        run_dir.mkdir(exist_ok=False)
        output = self.gateway.predict(
            MODEL_ID,
            SegmentationRequest(
                image_id=image_path.stem,
                image_path=image_path,
                image_bytes=image_path.read_bytes(),
                run_dir=run_dir,
                roi_mode=RoiMode.FULL_IMAGE,
                threshold=0.25,
                min_area_px=0,
                device=DevicePreference.CPU,
                seed=2026,
            ),
            expected_model_version=self.metadata.version,
            expected_adapter_path=self.metadata.adapter_path,
            expected_weight_sha256=self.metadata.weight_sha256,
            expected_config_sha256=self.metadata.config_sha256,
            expected_model_card_sha256=self.metadata.model_card_sha256,
            expected_adapter_sha256=self.metadata.adapter_sha256,
            model_bundle=self.bundle,
        )
        if output.probability_path is None:
            raise ValueError(f"gateway did not return probability output: {image_path.name}")
        probability_path = output.probability_path.resolve(strict=True)
        if not probability_path.is_relative_to(run_dir.resolve(strict=True)):
            raise ValueError("gateway probability output escaped its calibration run directory")
        probability = np.asarray(np.load(probability_path, allow_pickle=False), dtype=np.float32)
        if probability.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"unexpected probability shape for {image_path.name}")
        self.execution_evidence[image_path.name] = (
            output.execution.model_dump(mode="json") if output.execution is not None else None
        )
        return probability

    def close(self) -> None:
        self.gateway.cache.clear()
        self._temporary.cleanup()


def cache_probabilities(
    *,
    image_paths: Sequence[Path],
    output_root: Path,
    infer_probability: ProbabilityInferencer,
    expected_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    bottom_crop_px: int = BOTTOM_CROP_PX,
) -> dict[str, Path]:
    """Infer exactly once per validation image and persist reusable raw arrays."""

    validate_validation_filenames(tuple(path.name for path in image_paths))
    probability_dir = output_root / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=False)
    expected_shape = (expected_size[1], expected_size[0])
    paths: dict[str, Path] = {}
    for image_path in image_paths:
        probability = np.asarray(infer_probability(image_path), dtype=np.float32)
        if probability.shape != expected_shape:
            raise ValueError(
                f"probability shape mismatch for {image_path.name}: {probability.shape}"
            )
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"probability must be finite and in [0, 1]: {image_path.name}")
        if probability.shape[0] <= bottom_crop_px:
            raise ValueError(f"bottom crop removes the complete image: {image_path.name}")
        destination = probability_dir / f"{image_path.stem}.npy"
        np.save(destination, probability, allow_pickle=False)
        paths[image_path.name] = destination
    return paths


def _load_probabilities(paths: Mapping[str, Path]) -> dict[str, np.ndarray]:
    return {name: np.load(path, allow_pickle=False) for name, path in paths.items()}


def _load_targets(
    mask_dir: Path, *, expected_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for filename in VALIDATION_FILENAMES:
        with Image.open(mask_dir / filename) as image:
            if image.size != expected_size:
                raise ValueError(f"validation mask has unexpected dimensions: {filename}")
            pixels = np.asarray(image)
        if pixels.ndim == 2:
            target = pixels != 0
        elif pixels.ndim == 3:
            target = np.any(pixels != 0, axis=2)
        else:
            raise ValueError(f"validation mask has unsupported dimensions: {filename}")
        targets[filename] = np.asarray(target, dtype=bool)
    return targets


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 1.0


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
    return (
        int(np.count_nonzero(prediction & target)),
        int(np.count_nonzero(prediction & ~target)),
        int(np.count_nonzero(~prediction & target)),
        int(np.count_nonzero(~prediction & ~target)),
    )


def _size_bucket(area_px: int) -> str:
    equivalent_diameter_px = 2.0 * math.sqrt(area_px / math.pi)
    if equivalent_diameter_px <= 8:
        return "tiny"
    if equivalent_diameter_px <= 18:
        return "small"
    return "large"


def _empty_bucket_counts() -> dict[str, dict[str, int]]:
    return {bucket: {"gt_count": 0, "detected_count": 0} for bucket in SIZE_BUCKETS}


def _component_detection(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    labels = label(np.asarray(target, dtype=bool), connectivity=2)
    bucket_counts = _empty_bucket_counts()
    total = detected = 0
    for label_id in range(1, int(labels.max()) + 1):
        component = labels == label_id
        area_px = int(component.sum())
        bucket = _size_bucket(area_px)
        covered = int(np.count_nonzero(prediction & component))
        is_detected = covered / area_px >= DETECTION_COVERAGE_THRESHOLD
        total += 1
        detected += int(is_detected)
        bucket_counts[bucket]["gt_count"] += 1
        bucket_counts[bucket]["detected_count"] += int(is_detected)
    by_bucket = {
        bucket: {
            **counts,
            "recall": (
                counts["detected_count"] / counts["gt_count"]
                if counts["gt_count"]
                else None
            ),
        }
        for bucket, counts in bucket_counts.items()
    }
    return (
        {
            "gt_count": total,
            "detected_count": detected,
            "recall": detected / total if total else None,
            "by_size_bucket": by_bucket,
        },
        bucket_counts,
    )


def _aggregate_detection(bucket_counts: Mapping[str, Mapping[str, int]]) -> dict[str, object]:
    by_bucket: dict[str, dict[str, int | float | None]] = {}
    available_recalls: list[float] = []
    total = detected = 0
    for bucket in SIZE_BUCKETS:
        counts = bucket_counts[bucket]
        gt_count = int(counts["gt_count"])
        detected_count = int(counts["detected_count"])
        recall = detected_count / gt_count if gt_count else None
        by_bucket[bucket] = {
            "gt_count": gt_count,
            "detected_count": detected_count,
            "recall": recall,
        }
        total += gt_count
        detected += detected_count
        if recall is not None:
            available_recalls.append(recall)
    if not available_recalls:
        raise ValueError("validation GT contains no agglomerate components")
    return {
        "gt_count": total,
        "detected_count": detected,
        "recall": detected / total,
        "by_size_bucket": by_bucket,
        "available_size_buckets": [
            bucket for bucket in SIZE_BUCKETS if by_bucket[bucket]["recall"] is not None
        ],
        "mean_available_size_bucket_recall": float(np.mean(available_recalls)),
    }


def evaluate_thresholds(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
    bottom_crop_px: int = BOTTOM_CROP_PX,
) -> list[dict[str, object]]:
    if tuple(probabilities) != VALIDATION_FILENAMES or tuple(targets) != VALIDATION_FILENAMES:
        raise ValueError("probability and target inputs must match the fixed validation order")
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        image_results: list[dict[str, object]] = []
        total_tp = total_fp = total_fn = total_tn = 0
        aggregate_buckets = _empty_bucket_counts()
        for filename in VALIDATION_FILENAMES:
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
            prediction = valid_probability >= threshold
            tp, fp, fn, tn = _confusion(prediction, valid_target)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            detection, bucket_counts = _component_detection(prediction, valid_target)
            for bucket in SIZE_BUCKETS:
                aggregate_buckets[bucket]["gt_count"] += bucket_counts[bucket]["gt_count"]
                aggregate_buckets[bucket]["detected_count"] += bucket_counts[bucket][
                    "detected_count"
                ]
            image_results.append(
                {
                    "filename": filename,
                    **_metrics(tp, fp, fn, tn),
                    "gt_agglomerate_detection": detection,
                }
            )
        macro = {
            metric: float(np.mean([float(item[metric]) for item in image_results]))
            for metric in ("dice", "iou", "precision", "recall")
        }
        micro = _metrics(total_tp, total_fp, total_fn, total_tn)
        detection = _aggregate_detection(aggregate_buckets)
        score = 0.50 * float(micro["dice"]) + 0.50 * float(
            detection["mean_available_size_bucket_recall"]
        )
        results.append(
            {
                "threshold": float(threshold),
                "images": image_results,
                "macro": macro,
                "micro": micro,
                "gt_agglomerate_detection": detection,
                "selection_score": score,
            }
        )
    return results


def _numeric(result: Mapping[str, object], key: str) -> float:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"threshold result is missing numeric {key}")
    return float(value)


def _micro_dice(result: Mapping[str, object]) -> float:
    micro = result.get("micro")
    if not isinstance(micro, Mapping):
        raise ValueError("threshold result is missing micro metrics")
    value = micro.get("dice")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("threshold result is missing Micro Dice")
    return float(value)


def select_threshold(results: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not results:
        raise ValueError("threshold results are empty")
    best_score = max(_numeric(item, "selection_score") for item in results)
    winners = [
        item
        for item in results
        if math.isclose(
            _numeric(item, "selection_score"), best_score, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    best_dice = max(_micro_dice(item) for item in winners)
    winners = [
        item
        for item in winners
        if math.isclose(_micro_dice(item), best_dice, rel_tol=0.0, abs_tol=1e-12)
    ]
    lowest_threshold = min(_numeric(item, "threshold") for item in winners)
    winners = [
        item
        for item in winners
        if math.isclose(
            _numeric(item, "threshold"), lowest_threshold, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if len(winners) != 1:
        raise ValueError("the prespecified threshold selection rule did not resolve the tie")
    return winners[0]


def _write_probability_images(
    output_root: Path, probabilities: Mapping[str, np.ndarray]
) -> dict[str, str]:
    directory = output_root / "probability-images"
    directory.mkdir(parents=True, exist_ok=False)
    paths: dict[str, str] = {}
    for filename in VALIDATION_FILENAMES:
        destination = directory / f"{Path(filename).stem}-probability.png"
        rendered = np.clip(probabilities[filename] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rendered).save(destination)
        paths[filename] = destination.relative_to(output_root).as_posix()
    return paths


def _write_final_reviews(
    output_root: Path,
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    threshold: float,
    *,
    bottom_crop_px: int = BOTTOM_CROP_PX,
) -> tuple[dict[str, str], dict[str, str]]:
    prediction_dir = output_root / "final-predictions"
    review_dir = output_root / "reviews"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    review_dir.mkdir(parents=True, exist_ok=False)
    prediction_paths: dict[str, str] = {}
    review_paths: dict[str, str] = {}
    for filename in VALIDATION_FILENAMES:
        probability = probabilities[filename]
        target = targets[filename]
        prediction = probability >= threshold
        prediction[-bottom_crop_px:] = False
        prediction_path = prediction_dir / f"{Path(filename).stem}-pred-mask.png"
        Image.fromarray(prediction.astype(np.uint8) * 255).save(prediction_path)

        valid_prediction = prediction[:-bottom_crop_px]
        valid_target = target[:-bottom_crop_px]
        gt_panel = np.repeat((valid_target.astype(np.uint8) * 255)[..., None], 3, axis=2)
        pred_panel = np.repeat(
            (valid_prediction.astype(np.uint8) * 255)[..., None], 3, axis=2
        )
        error_panel = np.zeros((*valid_target.shape, 3), dtype=np.uint8)
        error_panel[valid_prediction & valid_target] = (0, 180, 0)
        error_panel[valid_prediction & ~valid_target] = (255, 0, 0)
        error_panel[~valid_prediction & valid_target] = (0, 80, 255)
        review = np.concatenate((gt_panel, pred_panel, error_panel), axis=1)
        review_path = review_dir / f"{Path(filename).stem}-gt-pred-error.png"
        Image.fromarray(review, mode="RGB").save(review_path)
        prediction_paths[filename] = prediction_path.relative_to(output_root).as_posix()
        review_paths[filename] = review_path.relative_to(output_root).as_posix()
    return prediction_paths, review_paths


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
        "gt_count",
        "detected_count",
        "gt_detection_recall",
        "tiny_recall",
        "small_recall",
        "large_recall",
        "mean_available_size_bucket_recall",
        "selection_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            threshold = result["threshold"]
            images = result["images"]
            if not isinstance(images, list):
                raise ValueError("threshold result images must be a list")
            for image in images:
                if not isinstance(image, Mapping):
                    raise ValueError("threshold image result must be an object")
                detection = image["gt_agglomerate_detection"]
                if not isinstance(detection, Mapping):
                    raise ValueError("image detection result must be an object")
                buckets = detection["by_size_bucket"]
                writer.writerow(
                    {
                        "threshold": threshold,
                        "scope": "image",
                        **{key: image[key] for key in fields[2:11]},
                        "gt_count": detection["gt_count"],
                        "detected_count": detection["detected_count"],
                        "gt_detection_recall": detection["recall"],
                        **{
                            f"{bucket}_recall": buckets[bucket]["recall"]
                            for bucket in SIZE_BUCKETS
                        },
                    }
                )
            macro = result["macro"]
            micro = result["micro"]
            detection = result["gt_agglomerate_detection"]
            buckets = detection["by_size_bucket"]
            writer.writerow({"threshold": threshold, "scope": "macro", **macro})
            writer.writerow(
                {
                    "threshold": threshold,
                    "scope": "micro_and_detection",
                    **micro,
                    "gt_count": detection["gt_count"],
                    "detected_count": detection["detected_count"],
                    "gt_detection_recall": detection["recall"],
                    **{
                        f"{bucket}_recall": buckets[bucket]["recall"]
                        for bucket in SIZE_BUCKETS
                    },
                    "mean_available_size_bucket_recall": detection[
                        "mean_available_size_bucket_recall"
                    ],
                    "selection_score": result["selection_score"],
                }
            )


def run_with_inferencer(
    paths: CalibrationPaths,
    *,
    infer_probability: ProbabilityInferencer,
    torchscript_sha256: str = TORCHSCRIPT_SHA256,
    expected_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    bottom_crop_px: int = BOTTOM_CROP_PX,
    execution_evidence: Mapping[str, dict[str, object] | None] | None = None,
) -> dict[str, object]:
    output_root = _validate_output_root(paths.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    image_paths = [paths.image_dir / filename for filename in VALIDATION_FILENAMES]
    probability_paths = cache_probabilities(
        image_paths=image_paths,
        output_root=output_root,
        infer_probability=infer_probability,
        expected_size=expected_size,
        bottom_crop_px=bottom_crop_px,
    )
    probabilities = _load_probabilities(probability_paths)
    targets = _load_targets(paths.mask_dir, expected_size=expected_size)
    results = evaluate_thresholds(
        probabilities,
        targets,
        bottom_crop_px=bottom_crop_px,
    )
    selected = select_threshold(results)
    selected_threshold = float(selected["threshold"])
    probability_images = _write_probability_images(output_root, probabilities)
    predictions, reviews = _write_final_reviews(
        output_root,
        probabilities,
        targets,
        selected_threshold,
        bottom_crop_px=bottom_crop_px,
    )
    csv_path = output_root / "threshold-calibration.csv"
    json_path = output_root / "threshold-calibration.json"
    _write_csv(csv_path, results)
    payload: dict[str, object] = {
        "schema_version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        # Human-readable path labels are deliberately portable; SHA-256 is the
        # identity and the reviewed asset can live at a different mount point.
        "torchscript_path": paths.torchscript.name,
        "torchscript_sha256": torchscript_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "inference_config": {
            "path": _config_path().relative_to(_repository_root()).as_posix(),
            "sha256": _sha256(_config_path()),
        },
        "validation_images": list(VALIDATION_FILENAMES),
        "validation_inputs": {
            filename: {
                "image_path": filename,
                "image_sha256": _sha256(paths.image_dir / filename),
                "mask_path": filename,
                "mask_sha256": _sha256(paths.mask_dir / filename),
            }
            for filename in VALIDATION_FILENAMES
        },
        "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
        "comparison_rule": "probability >= threshold",
        "fixed_inference_contract": {
            **EXPECTED_CONFIG,
            "tta_enabled": False,
            "min_area_px_during_threshold_calibration": 0,
            "effective_image_size_px": [expected_size[0], expected_size[1] - bottom_crop_px],
        },
        "scale_nm_per_pixel": SCALE_NM_PER_PIXEL,
        "gt_component_rule": {
            "target": "whole agglomerate",
            "connectivity": 2,
            "detection_coverage_threshold": DETECTION_COVERAGE_THRESHOLD,
            "size_buckets_equivalent_diameter_px": {
                "tiny": "<=8",
                "small": ">8 and <=18",
                "large": ">18",
            },
        },
        "selection_rule": {
            "score": (
                "0.50 * Micro Dice + 0.50 * mean recall across available GT size buckets"
            ),
            "tie_breakers": [
                "higher Micro Dice",
                "lower threshold",
                "fail if still unresolved",
            ],
        },
        "selected": selected,
        "selected_threshold": selected_threshold,
        "threshold_results": results,
        "artifacts": {
            "json": json_path.relative_to(output_root).as_posix(),
            "csv": csv_path.relative_to(output_root).as_posix(),
            "probability_arrays": {
                name: {
                    "path": path.relative_to(output_root).as_posix(),
                    "sha256": _sha256(path),
                }
                for name, path in probability_paths.items()
            },
            "probability_images": probability_images,
            "final_prediction_masks": predictions,
            "reviews": reviews,
        },
        "execution_evidence": dict(execution_evidence or {}),
        "evidence_statement": (
            "This is new traceable validation calibration evidence, not a replacement for or "
            "reconstruction of the missing training metadata JSON."
        ),
        "limitations": [
            (
                "Validation contains only four field-of-view images and is not "
                "sample-level independent."
            ),
            "The scientific object is each whole agglomerate, not its internal primary particles.",
            (
                "YCu-1.tif, YCu-2.tif, and YCu-3.tif were not read and remain "
                "reserved for independent test."
            ),
            "This task selects only a pixel threshold; min_area_px remains uncalibrated.",
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def run(paths: CalibrationPaths) -> dict[str, object]:
    observed_sha256 = _verify_torchscript(paths.torchscript)
    config = _load_config()
    inferencer = GatewayProbabilityInferencer(paths.torchscript, config)
    try:
        return run_with_inferencer(
            paths,
            infer_probability=inferencer,
            torchscript_sha256=observed_sha256,
            execution_evidence=inferencer.execution_evidence,
        )
    finally:
        inferencer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--torchscript", required=True, type=Path)
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
