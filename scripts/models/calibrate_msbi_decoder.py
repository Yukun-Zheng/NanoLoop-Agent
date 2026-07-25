#!/usr/bin/env python3
"""Bounded validation-only calibration from cached formal MSBI heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from app.msbi.decoding import DecodeConfig, decode_instances
from app.msbi.metrics import evaluate_calibration_metrics

FOREGROUND_THRESHOLDS = (0.80, 0.85, 0.90, 0.925, 0.95)
CENTER_THRESHOLDS = (0.35, 0.50, 0.65)
CENTER_NMS_RADII = (5, 9, 13)
MINIMUM_AREAS = (32, 64, 128)

_CACHE: list[dict[str, Any]] = []
_FIXED: dict[str, Any] = {}


def _float_list(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed or any(not 0.0 <= item <= 1.0 for item in parsed):
        raise argparse.ArgumentTypeError("thresholds must be in [0, 1]")
    return parsed


def _positive_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> list[dict[str, Any]]:
    if "independent" in path.name.lower() or "test" in path.name.lower():
        raise ValueError("decoder calibration accepts validation manifests only")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or any(record.get("split") != "validation" for record in records):
        raise ValueError("all decoder-calibration records must be validation")
    return records


def _load_cache(
    records: list[dict[str, Any]],
    *,
    runs_root: Path,
    model_id: str,
) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    for record in records:
        root = (
            runs_root
            / str(record["source_image_id"])
            / "inference"
            / model_id
        )
        with Image.open(record["mask_path"]) as image:
            truth = np.asarray(image.convert("L")) > 0
        valid = np.ones(truth.shape, dtype=bool)
        invalid_bottom = int(record["invalid_bottom_px"])
        if invalid_bottom:
            valid[-invalid_bottom:] = False
        arrays = {
            name: np.load(root / filename, allow_pickle=False)
            for name, filename in {
                "foreground": "foreground_probability.npy",
                "center": "center_probability.npy",
                "boundary": "boundary_probability.npy",
                "distance": "distance_field.npy",
            }.items()
        }
        if any(
            values.shape != truth.shape or not np.isfinite(values).all()
            for values in arrays.values()
        ):
            raise ValueError(f"invalid cached heads for {record['record_id']}")
        cache.append(
            {
                "record_id": record["record_id"],
                "truth": truth,
                "valid": valid,
                **arrays,
            }
        )
    return cache


def _evaluate(parameters: tuple[float, float, int, int]) -> dict[str, Any]:
    foreground_threshold, center_threshold, center_nms_radius, min_area_px = parameters
    config = DecodeConfig(
        mode=str(_FIXED["mode"]),
        foreground_threshold=foreground_threshold,
        center_threshold=center_threshold,
        center_nms_radius=center_nms_radius,
        boundary_threshold=float(_FIXED["boundary_threshold"]),
        boundary_penalty=float(_FIXED["boundary_penalty"]),
        min_area_px=min_area_px,
        max_area_px=_FIXED["max_area_px"],
        watershed_compactness=float(_FIXED["watershed_compactness"]),
        exclude_border=bool(_FIXED["exclude_border"]),
        connectivity=int(_FIXED["connectivity"]),
        fallback_peak_source=str(_FIXED["fallback_peak_source"]),
    )
    per_sample: list[dict[str, float]] = []
    for record in _CACHE:
        decoded = decode_instances(
            record["foreground"],
            record["center"],
            record["boundary"],
            distance_field=record["distance"],
            valid_mask=record["valid"],
            config=config,
        )
        per_sample.append(
            evaluate_calibration_metrics(
                record["truth"],
                decoded.labels,
                valid_mask=record["valid"],
            )
        )
    return {
        "foreground_threshold": foreground_threshold,
        "center_threshold": center_threshold,
        "center_nms_radius": center_nms_radius,
        "min_area_px": min_area_px,
        **{
            name: float(np.mean([row[name] for row in per_sample]))
            for name in per_sample[0]
        },
    }


def _rules(policy: dict[str, Any]) -> dict[str, tuple[str, float]]:
    rules: dict[str, tuple[str, float]] = {}
    for section in ("primary_rules", "effect_rules", "guardrail_rules"):
        if section not in policy:
            continue
        for rule in policy[section]:
            metric = str(rule["metric"])
            if metric in {
                "pixel_dice",
                "boundary_f1",
                "instance_f1_50",
                "count_absolute_error",
            }:
                rules[metric] = (str(rule["operator"]), float(rule["threshold"]))
    required = {
        "pixel_dice",
        "boundary_f1",
        "instance_f1_50",
        "count_absolute_error",
    }
    if set(rules) != required:
        raise ValueError("tolerance policy lacks the four calibration rules")
    return rules


def _annotate(
    row: dict[str, Any],
    rules: dict[str, tuple[str, float]],
) -> dict[str, Any]:
    margins: list[float] = []
    passes = 0
    for metric, (operator, threshold) in rules.items():
        value = float(row[metric])
        if operator in {">=", ">"}:
            passed = value >= threshold
            margin = value / threshold - 1.0
            if operator == ">":
                passed = value > threshold
        elif operator in {"<=", "<"}:
            passed = value <= threshold
            margin = (
                threshold / value - 1.0
                if value > 0
                else float("inf")
            )
            if operator == "<":
                passed = value < threshold
        else:
            raise ValueError(f"unsupported policy operator: {operator}")
        row[f"passes_{metric}"] = passed
        passes += int(passed)
        margins.append(margin)
    row["passed_scientific_rule_count"] = passes
    row["minimum_relative_margin"] = min(margins)
    row["passes_all_scientific_rules"] = passes == len(rules)
    return row


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row["passes_all_scientific_rules"]),
        float(row["minimum_relative_margin"]),
        int(row["passed_scientific_rule_count"]),
        float(row["instance_f1_50"]),
        -float(row["count_absolute_error"]),
        float(row["boundary_f1"]),
        float(row["pixel_dice"]),
        -abs(float(row["foreground_threshold"]) - 0.85),
        -abs(float(row["center_threshold"]) - 0.35),
        -abs(int(row["center_nms_radius"]) - 5),
        -abs(int(row["min_area_px"]) - 32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--candidate-runs-root", type=Path, required=True)
    parser.add_argument("--model-id", default="msbi-instance-balanced-v1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--foreground-thresholds",
        type=_float_list,
        default=FOREGROUND_THRESHOLDS,
    )
    parser.add_argument(
        "--center-thresholds",
        type=_float_list,
        default=CENTER_THRESHOLDS,
    )
    parser.add_argument(
        "--center-nms-radii",
        type=_positive_int_list,
        default=CENTER_NMS_RADII,
    )
    parser.add_argument(
        "--minimum-areas",
        type=_positive_int_list,
        default=MINIMUM_AREAS,
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    records = _manifest(args.validation_manifest)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    decoder = dict(config["decoder"])
    global _CACHE, _FIXED
    _CACHE = _load_cache(
        records,
        runs_root=args.candidate_runs_root,
        model_id=args.model_id,
    )
    _FIXED = {
        "mode": decoder.get("mode", "watershed"),
        "boundary_threshold": decoder["boundary_threshold"],
        "boundary_penalty": decoder["boundary_penalty"],
        "max_area_px": decoder.get("max_area_px"),
        "watershed_compactness": decoder.get("watershed_compactness", 0.0),
        "exclude_border": decoder.get("exclude_border", False),
        "connectivity": decoder.get("connectivity", 2),
        "fallback_peak_source": decoder.get("fallback_peak_source", "distance"),
    }
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    rules = _rules(policy)
    candidates = list(
        itertools.product(
            args.foreground_thresholds,
            args.center_thresholds,
            args.center_nms_radii,
            args.minimum_areas,
        )
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("fork"),
    ) as executor:
        rows = [
            _annotate(row, rules)
            for row in executor.map(_evaluate, candidates, chunksize=1)
        ]
    rows.sort(key=_rank, reverse=True)
    selected = rows[0]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "decoder-calibration.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected))
        writer.writeheader()
        writer.writerows(rows)
    evidence = {
        "schema_version": 1,
        "scope": "validation_only",
        "independent_test_accessed": False,
        "sample_count": len(records),
        "candidate_count": len(candidates),
        "search_space": {
            "foreground_threshold": args.foreground_thresholds,
            "center_threshold": args.center_thresholds,
            "center_nms_radius": args.center_nms_radii,
            "min_area_px": args.minimum_areas,
        },
        "fixed_decoder_parameters": _FIXED,
        "selection_rule": (
            "maximize all-rule pass, then minimum relative frozen-policy "
            "margin, passed rule count, instance F1, count MAE, boundary F1, "
            "Dice, then proximity to the prior frozen decoder"
        ),
        "validation_manifest_sha256": _sha256(args.validation_manifest),
        "candidate_config_sha256": _sha256(args.config),
        "tolerance_policy_sha256": _sha256(args.policy),
        "candidate_runs_root": str(args.candidate_runs_root.resolve()),
        "selected": selected,
        "selected_decoder": {
            name: selected[name]
            for name in (
                "foreground_threshold",
                "center_threshold",
                "center_nms_radius",
                "min_area_px",
            )
        },
        "base_decoder": decoder,
        "decode_contract": asdict(
            DecodeConfig(
                mode=str(_FIXED["mode"]),
                foreground_threshold=float(selected["foreground_threshold"]),
                center_threshold=float(selected["center_threshold"]),
                center_nms_radius=int(selected["center_nms_radius"]),
                boundary_threshold=float(_FIXED["boundary_threshold"]),
                boundary_penalty=float(_FIXED["boundary_penalty"]),
                min_area_px=int(selected["min_area_px"]),
                max_area_px=_FIXED["max_area_px"],
                watershed_compactness=float(_FIXED["watershed_compactness"]),
                exclude_border=bool(_FIXED["exclude_border"]),
                connectivity=int(_FIXED["connectivity"]),
                fallback_peak_source=str(_FIXED["fallback_peak_source"]),
            )
        ),
    }
    (args.output_dir / "decoder-calibration.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
