#!/usr/bin/env python3
"""Relocate the sealed independent-test manifest after a bound gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.msbi.acceptance import load_json, sha256_file


def _mapping(value: str) -> tuple[str, str]:
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise argparse.ArgumentTypeError("mapping must be OLD_PREFIX=NEW_PREFIX")
    return source, destination


def _relocate(value: str, mappings: list[tuple[str, str]]) -> str:
    for source, destination in mappings:
        if value.startswith(source):
            return destination + value[len(source) :]
    raise ValueError(f"path does not match an authorized mapping: {value}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--map", type=_mapping, action="append", required=True)
    args = parser.parse_args()

    # Deliberately verify the bound gate before reading the sealed manifest.
    gate = load_json(args.acceptance_gate)
    if (
        gate.get("status") != "PASSED"
        or gate.get("policy_sha256") != sha256_file(args.policy)
        or gate.get("candidate_results_sha256")
        != sha256_file(args.candidate_results)
    ):
        raise SystemExit(
            "REFUSED: validation acceptance gate did not authorize test relocation"
        )

    records: list[dict[str, Any]] = []
    for line in args.input_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") != "independent_test":
            raise ValueError("authorized manifest contains a non-test record")
        for field in ("image_path", "mask_path"):
            value = record.get(field)
            if not isinstance(value, str):
                raise ValueError(f"{record.get('record_id')} is missing {field}")
            relocated = _relocate(value, args.map)
            path = Path(relocated)
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_sha = record[f"{field.removesuffix('_path')}_sha256"]
            if _file_sha256(path) != expected_sha:
                raise ValueError(f"{field} SHA mismatch: {record['record_id']}")
            record[field] = relocated
        record["sealed"] = False
        record["gt_type"] = "human_binary_independent_test_after_strict_gate"
        records.append(record)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "AUTHORIZED_AND_RELOCATED",
        "record_count": len(records),
        "policy_sha256": sha256_file(args.policy),
        "candidate_results_sha256": sha256_file(args.candidate_results),
        "acceptance_gate_sha256": sha256_file(args.acceptance_gate),
        "source_manifest_sha256": sha256_file(args.input_manifest),
        "output_manifest_sha256": sha256_file(args.output_manifest),
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
