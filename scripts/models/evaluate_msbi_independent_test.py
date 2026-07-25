#!/usr/bin/env python3
"""Fail closed before delegating an authorized independent-test evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app.msbi.acceptance import load_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()

    # This section intentionally runs before the test manifest is opened or parsed.
    gate = load_json(args.acceptance_gate)
    if (
        gate.get("status") != "PASSED"
        or gate.get("policy_sha256") != sha256_file(args.policy)
        or gate.get("candidate_results_sha256")
        != sha256_file(args.candidate_results)
    ):
        print(
            "REFUSED: validation acceptance gate did not pass; "
            "independent test remains sealed.",
            file=sys.stderr,
        )
        return 3

    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_msbi_validation.py")),
        "--weight",
        str(args.weight),
        "--config",
        str(args.config),
        "--validation-manifest",
        str(args.test_manifest),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--status",
        "INDEPENDENT_TEST_EVALUATED",
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
