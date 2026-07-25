#!/usr/bin/env python3
"""Evaluate current U-Net teachers on the frozen MSBI validation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from app.contracts.enums import DevicePreference, RoiMode
from app.contracts.inference import SegmentationRequest
from app.inference.adapters.unet import UNetAdapter
from app.inference.registry import ModelRegistryService
from app.msbi.metrics import evaluate_binary_instance_prediction


def _manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: list[dict[str, Any]], key: str) -> float | None:
    present = [float(row[key]) for row in values if row.get(key) is not None]
    return float(np.mean(present)) if present else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional evaluation-only threshold override.",
    )
    parser.add_argument(
        "--model-ids",
        default="unet-small-balanced-v1,unet-large-optimized-v1",
        help="Comma-separated registered baseline model IDs.",
    )
    parser.add_argument(
        "--small-config",
        type=Path,
        help="Optional experimental config override for the small U-Net.",
    )
    parser.add_argument(
        "--probability-output-dir",
        type=Path,
        help="Optional validation-only directory for cached probability arrays.",
    )
    parser.add_argument(
        "--compat-weight-dir",
        type=Path,
        help=(
            "Optional directory containing numerically verified compatibility "
            "re-exports named <model_id>.pt"
        ),
    )
    args = parser.parse_args()
    records = _manifest(args.validation_manifest)
    registry = ModelRegistryService(args.registry)
    model_ids = tuple(
        value.strip() for value in args.model_ids.split(",") if value.strip()
    )
    if not model_ids:
        raise ValueError("--model-ids must include at least one model")
    adapters = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runtime: dict[str, list[int]] = {model_id: [] for model_id in model_ids}
    runtime_weight_sha256: dict[str, str] = {}
    try:
        for model_id in model_ids:
            if args.compat_weight_dir is None:
                adapter = registry.create_adapter(model_id)
            else:
                registration = registry.get_registration(model_id)
                weight_path = args.compat_weight_dir / f"{model_id}.pt"
                weight_sha256 = _sha256(weight_path)
                runtime_weight_sha256[model_id] = weight_sha256
                config = registration.config
                if model_id == "unet-small-balanced-v1" and args.small_config is not None:
                    config = yaml.safe_load(
                        args.small_config.read_text(encoding="utf-8")
                    )
                    if not isinstance(config, dict):
                        raise ValueError("--small-config must contain a YAML mapping")
                adapter = UNetAdapter(
                    metadata=registration.metadata,
                    weight_path=weight_path,
                    weight_bytes=weight_path.read_bytes(),
                    config=config,
                    weight_sha256=weight_sha256,
                )
            adapter.load(args.device)
            adapters[model_id] = adapter
        for record in records:
            with Image.open(record["mask_path"]) as image:
                truth = np.asarray(image.convert("L")) > 0
            valid = np.ones(truth.shape, dtype=bool)
            invalid_bottom = int(record["invalid_bottom_px"])
            if invalid_bottom:
                valid[-invalid_bottom:] = False
            for model_id in model_ids:
                registration = registry.get_registration(model_id)
                with tempfile.TemporaryDirectory(prefix="msbi-baseline-") as temporary:
                    output = adapters[model_id].predict(
                        SegmentationRequest(
                            image_id=record["source_image_id"],
                            image_path=Path(record["image_path"]),
                            run_dir=Path(temporary),
                            roi_mode=RoiMode.FULL_IMAGE,
                            device=DevicePreference(args.device),
                            seed=2026,
                            threshold=args.threshold,
                        )
                    )
                    prediction = np.asarray(
                        Image.open(output.binary_mask_path).convert("L")
                    ) > 0
                    if args.probability_output_dir is not None:
                        probability = np.load(output.probability_path, allow_pickle=False)
                        probability_destination = (
                            args.probability_output_dir
                            / model_id
                            / f"{record['source_image_id']}.npy"
                        )
                        probability_destination.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        np.save(
                            probability_destination,
                            probability,
                            allow_pickle=False,
                        )
                metrics = evaluate_binary_instance_prediction(
                    truth,
                    prediction,
                    valid_mask=valid,
                    prediction_min_area_px=(
                        registration.metadata.default_min_area_px or 0
                    ),
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "record_id": record["record_id"],
                        "source_image_id": record["source_image_id"],
                        "morphology_group": record["morphology_group"],
                        "runtime_ms": output.runtime_ms,
                        **metrics,
                    }
                )
                runtime[model_id].append(output.runtime_ms)
    finally:
        for adapter in adapters.values():
            adapter.unload()
    fieldnames = list(rows[0])
    with (args.output_dir / "baseline-per-sample.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [
        key
        for key in rows[0]
        if key
        not in {
            "model_id",
            "record_id",
            "source_image_id",
            "morphology_group",
        }
    ]
    summary = {
        model_id: {
            "sample_count": sum(row["model_id"] == model_id for row in rows),
            "metrics_macro": {
                name: _mean(
                    [row for row in rows if row["model_id"] == model_id],
                    name,
                )
                for name in metric_names
            },
            "runtime_ms": {
                "mean": float(np.mean(runtime[model_id])),
                "p95": float(np.percentile(runtime[model_id], 95)),
            },
            "runtime_weight_sha256": runtime_weight_sha256.get(model_id),
            "gt_limitation": (
                "instance metrics use connected components of binary semantic GT, "
                "not human instance IDs"
            ),
        }
        for model_id in model_ids
    }
    (args.output_dir / "baseline-metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
