#!/usr/bin/env python3
"""Build stratified, ablation, runtime, and failure-analysis evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

IDENTITY_FIELDS = {
    "model_id",
    "record_id",
    "source_image_id",
    "morphology_group",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["source_image_id"]: record
        for record in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    return float(raw) if raw not in (None, "") else None


def _stratify(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metric_names = [key for key in rows[0] if key not in IDENTITY_FIELDS]
    summaries: list[dict[str, Any]] = []
    for group, values in sorted(groups.items()):
        summary: dict[str, Any] = dict(zip(keys, group, strict=True))
        summary["sample_count"] = len(values)
        for metric in metric_names:
            present = [
                value
                for row in values
                if (value := _number(row, metric)) is not None
            ]
            summary[metric] = float(np.mean(present)) if present else None
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _unit_float(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32))
    minimum = float(values.min())
    maximum = float(values.max())
    return np.clip((values - minimum) / max(maximum - minimum, 1e-8), 0.0, 1.0)


def _heat(values: np.ndarray) -> Image.Image:
    unit = _unit_float(values)
    rgb = np.stack(
        (
            np.clip(2.0 * unit, 0.0, 1.0),
            np.clip(2.0 - 2.0 * np.abs(unit - 0.5), 0.0, 1.0),
            np.clip(2.0 * (1.0 - unit), 0.0, 1.0),
        ),
        axis=-1,
    )
    return Image.fromarray(np.asarray(rgb * 255, dtype=np.uint8))


def _label_overlay(image: np.ndarray, labels: np.ndarray) -> Image.Image:
    base = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    ids = labels.astype(np.uint32)
    color = np.stack(
        (
            (ids * 47) % 255,
            (ids * 89) % 255,
            (ids * 137) % 255,
        ),
        axis=-1,
    ).astype(np.float32)
    alpha = (labels > 0)[..., None].astype(np.float32) * 0.55
    return Image.fromarray(np.asarray(base * (1.0 - alpha) + color * alpha, dtype=np.uint8))


def _caption(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    thumb = image.convert("RGB")
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size[0], size[1] + 28), "white")
    panel.paste(thumb, ((size[0] - thumb.width) // 2, 28))
    ImageDraw.Draw(panel).text((8, 7), label, fill="black")
    return panel


def _write_failure_case(
    *,
    case_key: str,
    record: dict[str, Any],
    run_root: Path,
    output_root: Path,
) -> dict[str, str]:
    with Image.open(record["image_path"]) as opened:
        original = np.asarray(opened.convert("L"))
    with Image.open(record["mask_path"]) as opened:
        truth = np.asarray(opened.convert("L"))
    inference = run_root / record["source_image_id"] / "inference" / "msbi-instance-balanced-v1"
    prediction = np.asarray(Image.open(inference / "binary_mask.png").convert("L"))
    labels = np.load(inference / "instance_labels.npy", allow_pickle=False)
    center = np.load(inference / "center_probability.npy", allow_pickle=False)
    boundary = np.load(inference / "boundary_probability.npy", allow_pickle=False)
    gate_small = np.load(inference / "gate_small.npy", allow_pickle=False)
    gate_large = np.load(inference / "gate_large.npy", allow_pickle=False)
    uncertainty = np.load(inference / "uncertainty.npy", allow_pickle=False)

    gate_rgb = np.stack(
        (
            np.clip(gate_large, 0.0, 1.0),
            np.zeros_like(gate_large),
            np.clip(gate_small, 0.0, 1.0),
        ),
        axis=-1,
    )
    layers = {
        "original": Image.fromarray(original),
        "ground_truth": Image.fromarray(truth),
        "prediction": Image.fromarray(prediction),
        "instance_overlay": _label_overlay(original, labels),
        "center_map": _heat(center),
        "boundary_map": _heat(boundary),
        "gate_map": Image.fromarray(np.asarray(gate_rgb * 255, dtype=np.uint8)),
        "uncertainty_map": _heat(uncertainty),
    }
    panels = []
    for name, image in layers.items():
        destination = output_root / name / f"{case_key}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        panels.append(_caption(image, name.replace("_", " "), (384, 288)))
    sheet = Image.new("RGB", (4 * 384, 2 * 316), "#eeeeee")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 4) * 384, (index // 4) * 316))
    sheet_path = output_root / "cases" / f"{case_key}.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    return {
        name: str(output_root / name / f"{case_key}.png")
        for name in layers
    } | {"panel": str(sheet_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-per-sample", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-per-sample", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    baseline_metrics = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    candidate_metrics = json.loads(args.candidate_metrics.read_text(encoding="utf-8"))
    baseline_rows = _csv(args.baseline_per_sample)
    candidate_rows = _csv(args.candidate_per_sample)
    _write_csv(
        args.output_root / "baselines" / "baseline-size-stratified.csv",
        _stratify(baseline_rows, ("model_id", "morphology_group")),
    )
    (args.output_root / "baselines" / "baseline-runtime.json").write_text(
        json.dumps(
            {
                model_id: values["runtime_ms"]
                for model_id, values in baseline_metrics.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        args.output_root / "candidate-validation" / "candidate-size-stratified.csv",
        _stratify(candidate_rows, ("morphology_group",)),
    )

    stages = [
        {
            "ablation_id": "A0",
            "description": "Large U-Net",
            "status": "COMPLETED",
            "training_budget": "delivered baseline",
            "parameter_count": 3_349_697,
            **baseline_metrics["unet-large-optimized-v1"]["metrics_macro"],
        },
        {
            "ablation_id": "A1",
            "description": "Small U-Net",
            "status": "COMPLETED",
            "training_budget": "delivered baseline",
            "parameter_count": 3_351_681,
            **baseline_metrics["unet-small-balanced-v1"]["metrics_macro"],
        },
    ]
    for ablation_id, description in (
        ("A2", "single-encoder foreground semantic"),
        ("A3", "A2 plus center boundary watershed"),
        ("A4", "dual experts with fixed mean"),
        ("A5", "dual experts with learned gate"),
        ("A6", "A5 plus teacher distillation"),
    ):
        stages.append(
            {
                "ablation_id": ablation_id,
                "description": description,
                "status": "NOT_RUN_NO_CUDA_BUDGET",
                "training_budget": "none",
                "parameter_count": None,
            }
        )
    stages.append(
        {
            "ablation_id": "A7",
            "description": "A6 plus SDF and uncertainty (combined real-data pilot)",
            "status": "COMPLETED_PILOT_FAILED",
            "training_budget": "3 epochs, 192 MPS training patches",
            "parameter_count": 28_623_080,
            **candidate_metrics["metrics_macro"],
        }
    )
    ablation_root = args.output_root / "ablation"
    _write_csv(ablation_root / "ablation-summary.csv", stages)
    (ablation_root / "ablation-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "INCOMPLETE",
                "shared_split": str(args.validation_manifest),
                "stages": stages,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        ablation_root / "runtime-complexity.csv",
        [
            {
                "model_id": model_id,
                "parameter_count": parameter_count,
                "runtime_ms_mean": baseline_metrics[model_id]["runtime_ms"]["mean"],
                "runtime_ms_p95": baseline_metrics[model_id]["runtime_ms"]["p95"],
                "conv_linear_macs_256": None,
            }
            for model_id, parameter_count in (
                ("unet-large-optimized-v1", 3_349_697),
                ("unet-small-balanced-v1", 3_351_681),
            )
        ]
        + [
            {
                "model_id": "msbi-instance-balanced-v1",
                "parameter_count": 28_623_080,
                "runtime_ms_mean": candidate_metrics["metrics_macro"]["runtime_ms"],
                "runtime_ms_p95": None,
                "conv_linear_macs_256": 7_653_736_448,
            }
        ],
    )

    ranked = {
        "largest-count-error": max(
            candidate_rows,
            key=lambda row: float(row["count_absolute_error"]),
        ),
        "lowest-boundary-f1": min(
            candidate_rows,
            key=lambda row: float(row["boundary_f1"]),
        ),
        "most-over-split": max(
            candidate_rows,
            key=lambda row: float(row["prediction_instance_count"])
            - float(row["truth_instance_count"]),
        ),
        "most-under-split": min(
            candidate_rows,
            key=lambda row: float(row["prediction_instance_count"])
            - float(row["truth_instance_count"]),
        ),
        "highest-uncertainty": max(
            candidate_rows,
            key=lambda row: float(row["mean_uncertainty"]),
        ),
        "largest-size-distribution-error": max(
            candidate_rows,
            key=lambda row: float(row["diameter_wasserstein_px"]),
        ),
    }
    records = _manifest(args.validation_manifest)
    cases: list[dict[str, Any]] = []
    failure_root = args.output_root / "failure-analysis"
    for category, row in ranked.items():
        source_id = row["source_image_id"]
        case_key = f"{category}-{source_id}"
        paths = _write_failure_case(
            case_key=case_key,
            record=records[source_id],
            run_root=args.candidate_run_root,
            output_root=failure_root,
        )
        cases.append(
            {
                "category": category,
                "source_image_id": source_id,
                "metrics": {
                    key: _number(row, key)
                    for key in (
                        "count_absolute_error",
                        "boundary_f1",
                        "instance_f1_50",
                        "diameter_wasserstein_px",
                        "mean_uncertainty",
                    )
                },
                "paths": paths,
            }
        )
    (failure_root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "validation",
                "independent_test_accessed": False,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ablation_status": "INCOMPLETE", "failure_cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
