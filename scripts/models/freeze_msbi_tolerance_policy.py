#!/usr/bin/env python3
"""Freeze the validation tolerance policy before any independent-test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.msbi.acceptance import build_tolerance_policy, load_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = build_tolerance_policy(
        load_json(args.baseline_results),
        baseline_results_sha256=sha256_file(args.baseline_results),
        split_manifest_sha256=sha256_file(args.split_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "reference_model_id": policy["reference_model_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
