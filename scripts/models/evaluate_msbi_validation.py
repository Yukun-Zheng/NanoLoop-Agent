#!/usr/bin/env python3
"""Evaluate an exported MSBI pilot on the frozen validation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from app.contracts.enums import (
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelMetadata
from app.inference.adapters.msbi import MSBIAdapter
from app.msbi.metrics import evaluate_binary_instance_prediction


def _manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--status", default="PILOT_NOT_ACCEPTED")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-area-px", type=int)
    args = parser.parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.min_area_px is not None and args.min_area_px < 0:
        raise ValueError("--min-area-px must be non-negative")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = _manifest(args.validation_manifest)
    if args.limit is not None:
        records = records[: args.limit]
    adapter = MSBIAdapter(
        metadata=ModelMetadata(
            model_id="msbi-instance-balanced-v1",
            family=ModelFamily.MSBI,
            variant=ModelVariant.DENSE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="pilot",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.5,
            default_min_area_px=int(config["decoder"]["min_area_px"]),
            preprocess_profile="sem-gray-p1-p99-crop-bottom-128-v1",
            postprocess_profile="msbi-center-boundary-watershed-v1",
            inference_invalid_bottom_px=int(config["bottom_crop_px"]),
            expected_input_width=2048,
            expected_input_height=1536,
        ),
        weight_path=args.weight,
        weight_bytes=args.weight.read_bytes(),
        config=config,
    )
    rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter.load(args.device)
    try:
        for record in records:
            run_dir = args.output_dir / "runs" / record["source_image_id"]
            output = adapter.predict(
                SegmentationRequest(
                    image_id=record["source_image_id"],
                    image_path=Path(record["image_path"]),
                    run_dir=run_dir,
                    roi_mode=RoiMode.FULL_IMAGE,
                    device=DevicePreference(args.device),
                    seed=2026,
                    threshold=args.threshold,
                    min_area_px=args.min_area_px or 0,
                )
            )
            with Image.open(record["mask_path"]) as image:
                truth = np.asarray(image.convert("L")) > 0
            labels_path = output.binary_mask_path.parent / "instance_labels.npy"
            if labels_path.is_file():
                labels = np.load(labels_path, allow_pickle=False)
            elif output.instances_path is not None:
                with np.load(output.instances_path, allow_pickle=False) as archive:
                    labels = np.asarray(archive["label_map"], dtype=np.int32)
            else:
                raise ValueError("MSBI validation output has no instance labels")
            valid = np.ones(truth.shape, dtype=bool)
            valid[-int(record["invalid_bottom_px"]) :] = False
            metrics = evaluate_binary_instance_prediction(
                truth,
                labels > 0,
                valid_mask=valid,
                prediction_labels=labels,
            )
            rows.append(
                {
                    "record_id": record["record_id"],
                    "source_image_id": record["source_image_id"],
                    "morphology_group": record["morphology_group"],
                    "runtime_ms": output.runtime_ms,
                    "mean_uncertainty": output.model_scores["mean_uncertainty"],
                    "mean_small_gate": output.model_scores["mean_small_gate"],
                    "mean_large_gate": output.model_scores["mean_large_gate"],
                    **{
                        name: value
                        for name, value in output.model_scores.items()
                        if name.startswith("profile_")
                    },
                    **metrics,
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    finally:
        adapter.unload()
    with (args.output_dir / "candidate-per-sample.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model_id": "msbi-instance-balanced-v1",
        "status": args.status,
        "sample_count": len(rows),
        "metrics_macro": {
            name: _mean(rows, name)
            for name in rows[0]
            if name
            not in {
                "record_id",
                "source_image_id",
                "morphology_group",
            }
        },
        "runtime_ms": {
            "mean": _mean(rows, "runtime_ms"),
            "p95": float(np.percentile([row["runtime_ms"] for row in rows], 95)),
        },
        "request_overrides": {
            "threshold": args.threshold,
            "min_area_px": args.min_area_px,
        },
        "gt_limitation": (
            "instance metrics use connected components of binary semantic GT, "
            "not human instance IDs"
        ),
    }
    summary["metrics_macro"]["runtime_p95_ms"] = summary["runtime_ms"]["p95"]
    (args.output_dir / "candidate-metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
