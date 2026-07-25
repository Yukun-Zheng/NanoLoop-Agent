#!/usr/bin/env python3
"""Assemble the machine-readable final MSBI outcome from verified evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from app.msbi.acceptance import load_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--ablation-results", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--tolerance-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline_results)
    candidate = load_json(args.candidate_results)
    ablation = load_json(args.ablation_results)
    gate = load_json(args.acceptance_gate)
    export = load_json(args.export_manifest)
    if gate["status"] != "FAILED":
        raise ValueError("this finalizer is for the current failed validation outcome")
    summary = {
        "status": "FAILED_VALIDATION_GATE",
        "model_id": "msbi-instance-balanced-v1",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "checkpoint_sha256": export["checkpoint_sha256"],
        "torchscript_sha256": export["torchscript_sha256"],
        "config_sha256": sha256_file(args.runtime_config),
        "training_config_sha256": sha256_file(args.training_config),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "tolerance_policy_sha256": sha256_file(args.tolerance_policy),
        "baseline_results": baseline,
        "candidate_results": candidate,
        "validation_acceptance_gate": gate,
        "ablation_results": ablation,
        "runtime": {
            "device": "cuda_rtx_4090",
            "full_image_mean_ms": candidate["metrics_macro"]["runtime_ms"],
            "parameters": export["parameter_count"],
            "conv_linear_macs_per_256_patch": 7_653_736_448,
            "approximate_flops_per_256_patch": 15_307_472_896,
            "torchscript_eager_max_abs_diff": export[
                "eager_torchscript_max_abs_diff"
            ],
            "torchscript_repeat_max_abs_diff": export[
                "deterministic_repeat_max_abs_diff"
            ],
        },
        "independent_test_status": "SEALED_NOT_ACCESSED",
        "independent_test_results": None,
        "limitations": [
            "binary semantic GT provides connected-component pseudo-instances only",
            "no human instance IDs for touching particles",
            "no frozen physical scale calibration",
            "three official training views are teacher-only because masks are missing",
            "formal training and validation use one seed only",
            "checkpoint redistribution permission is undocumented",
        ],
        "ready_recommendation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "status": summary["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
