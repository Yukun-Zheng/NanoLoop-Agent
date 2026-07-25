#!/usr/bin/env python3
"""Apply the frozen validation policy without opening independent-test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.msbi.acceptance import (
    evaluate_candidate,
    load_json,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_candidate(
        load_json(args.policy),
        load_json(args.candidate_results),
        policy_sha256=sha256_file(args.policy),
        candidate_results_sha256=sha256_file(args.candidate_results),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
