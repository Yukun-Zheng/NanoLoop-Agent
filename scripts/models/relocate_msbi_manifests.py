#!/usr/bin/env python3
"""Relocate only train/validation manifests for a controlled training host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map", type=_mapping, action="append", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        source = args.input_dir / f"{split}.jsonl"
        records: list[dict[str, Any]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for field in (
                "image_path",
                "mask_path",
                "instance_mask_path",
                "target_path",
            ):
                value = record.get(field)
                if isinstance(value, str):
                    relocated = _relocate(value, args.map)
                    record[field] = relocated
                    path = Path(relocated)
                    if not path.is_file():
                        raise FileNotFoundError(path)
            image_path = Path(record["image_path"])
            if _sha256(image_path) != record["image_sha256"]:
                raise ValueError(f"image SHA mismatch: {record['record_id']}")
            mask_value = record.get("mask_path")
            if isinstance(mask_value, str) and _sha256(Path(mask_value)) != record["mask_sha256"]:
                raise ValueError(f"mask SHA mismatch: {record['record_id']}")
            records.append(record)
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        counts[split] = len(records)
    split_source = args.input_dir / "split-manifest.json"
    (args.output_dir / "split-manifest.json").write_bytes(split_source.read_bytes())
    report = {
        "schema_version": 1,
        "source_split_manifest_sha256": _sha256(split_source),
        "counts": counts,
        "independent_test_transferred": False,
        "mappings": [
            {"source_prefix": source, "destination_prefix": destination}
            for source, destination in args.map
        ],
    }
    (args.output_dir / "relocation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
