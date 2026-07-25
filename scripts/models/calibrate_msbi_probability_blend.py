#!/usr/bin/env python3
"""Calibrate a validation-only full-frame/tiled probability blend."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.msbi.decoding import DecodeConfig, decode_instances
from app.msbi.metrics import evaluate_calibration_metrics

_CACHE: list[dict[str, Any]] = []
_RULES: dict[str, tuple[str, float]] = {}


def _numbers(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed or any(not np.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("values must be finite")
    return parsed


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    if "test" in path.name.lower() or "independent" in path.name.lower():
        raise ValueError("probability calibration accepts validation manifests only")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or any(record.get("split") != "validation" for record in records):
        raise ValueError("all probability-calibration records must be validation")
    return records


def _rules(policy: dict[str, Any]) -> dict[str, tuple[str, float]]:
    required = {
        "pixel_dice",
        "boundary_f1",
        "instance_f1_50",
        "count_absolute_error",
    }
    result: dict[str, tuple[str, float]] = {}
    for section in ("primary_rules", "effect_rules", "guardrail_rules"):
        for rule in policy.get(section, []):
            metric = str(rule["metric"])
            if metric in required:
                result[metric] = (str(rule["operator"]), float(rule["threshold"]))
    if set(result) != required:
        raise ValueError("policy lacks the four probability-calibration rules")
    return result


def _passes(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"unsupported operator: {operator}")


def _evaluate(parameters: tuple[float, float, int]) -> dict[str, Any]:
    tiled_weight, threshold, min_area_px = parameters
    metrics: list[dict[str, float]] = []
    for record in _CACHE:
        probability = np.clip(
            tiled_weight * record["tiled"]
            + (1.0 - tiled_weight) * record["full"],
            0.0,
            1.0,
        ).astype(np.float32)
        decoded = decode_instances(
            probability,
            np.zeros_like(probability),
            np.zeros_like(probability),
            valid_mask=record["valid"],
            config=DecodeConfig(
                mode="connected_components",
                foreground_threshold=threshold,
                min_area_px=min_area_px,
                connectivity=2,
            ),
        )
        metrics.append(
            evaluate_calibration_metrics(
                record["truth"],
                decoded.labels,
                valid_mask=record["valid"],
            )
        )
    row: dict[str, Any] = {
        "tiled_weight": tiled_weight,
        "full_frame_weight": 1.0 - tiled_weight,
        "foreground_threshold": threshold,
        "min_area_px": min_area_px,
        **{
            metric: float(np.mean([sample[metric] for sample in metrics]))
            for metric in metrics[0]
        },
    }
    margins: list[float] = []
    for metric, (operator, target) in _RULES.items():
        value = float(row[metric])
        passed = _passes(value, operator, target)
        row[f"passes_{metric}"] = passed
        margins.append(
            value / target - 1.0
            if operator in {">", ">="}
            else target / value - 1.0
            if value > 0
            else float("inf")
        )
    row["passed_rule_count"] = sum(
        bool(row[f"passes_{metric}"]) for metric in _RULES
    )
    row["passes_all_rules"] = row["passed_rule_count"] == len(_RULES)
    row["minimum_relative_margin"] = min(margins)
    return row


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row["passes_all_rules"]),
        float(row["minimum_relative_margin"]),
        int(row["passed_rule_count"]),
        float(row["instance_f1_50"]),
        -float(row["count_absolute_error"]),
        float(row["boundary_f1"]),
        float(row["pixel_dice"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tiled-root", type=Path, required=True)
    parser.add_argument("--full-frame-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tiled-weights", type=_numbers, required=True)
    parser.add_argument("--thresholds", type=_numbers, required=True)
    parser.add_argument("--minimum-areas", type=_positive_ints, required=True)
    args = parser.parse_args()
    records = _records(args.validation_manifest)
    global _CACHE, _RULES
    for record in records:
        name = f"{record['source_image_id']}.npy"
        tiled_path = args.tiled_root / name
        full_path = args.full_frame_root / name
        tiled = np.load(tiled_path, allow_pickle=False)
        full = np.load(full_path, allow_pickle=False)
        with Image.open(record["mask_path"]) as image:
            truth = np.asarray(image.convert("L")) > 0
        if tiled.shape != truth.shape or full.shape != truth.shape:
            raise ValueError(f"probability shape mismatch for {record['record_id']}")
        valid = np.ones(truth.shape, dtype=bool)
        invalid_bottom = int(record["invalid_bottom_px"])
        if invalid_bottom:
            valid[-invalid_bottom:] = False
        _CACHE.append(
            {
                "record_id": record["record_id"],
                "tiled": tiled,
                "full": full,
                "truth": truth,
                "valid": valid,
                "tiled_sha256": _sha256(tiled_path),
                "full_sha256": _sha256(full_path),
            }
        )
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    _RULES = _rules(policy)
    candidates = list(
        itertools.product(
            args.tiled_weights,
            args.thresholds,
            args.minimum_areas,
        )
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("fork"),
    ) as executor:
        rows = list(executor.map(_evaluate, candidates, chunksize=1))
    rows.sort(key=_rank, reverse=True)
    result = {
        "schema_version": 1,
        "scope": "validation_only",
        "independent_test_accessed": False,
        "validation_manifest_sha256": _sha256(args.validation_manifest),
        "policy_sha256": _sha256(args.policy),
        "candidate_count": len(rows),
        "search_space": {
            "tiled_weight": args.tiled_weights,
            "foreground_threshold": args.thresholds,
            "min_area_px": args.minimum_areas,
        },
        "selected": rows[0],
        "top_candidates": rows[:20],
        "probability_evidence": [
            {
                "record_id": record["record_id"],
                "tiled_sha256": record["tiled_sha256"],
                "full_sha256": record["full_sha256"],
            }
            for record in _CACHE
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
