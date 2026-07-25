#!/usr/bin/env python3
"""Cache immutable Small/Large U-Net probabilities for MSBI distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.enums import DevicePreference, RoiMode
from app.contracts.inference import SegmentationRequest
from app.inference.registry import ModelRegistryService


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path, *, unlabeled_only: bool) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if unlabeled_only:
        values = [record for record in values if not record["supervision_available"]]
    return values


def _replace_target_cache(
    path: Path,
    *,
    teacher_small: np.ndarray,
    teacher_large: np.ndarray,
    teacher_valid: np.ndarray,
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays.update(
        {
            "teacher_small": teacher_small.astype(np.float16),
            "teacher_large": teacher_large.astype(np.float16),
            "teacher_valid": teacher_valid.astype(np.uint8),
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--all-labeled", action="store_true")
    args = parser.parse_args()
    records = _records(args.train_manifest, unlabeled_only=not args.all_labeled)
    if not records:
        raise ValueError("no manifest records selected for teacher caching")
    registry = ModelRegistryService(args.registry)
    teacher_ids = ("unet-small-balanced-v1", "unet-large-optimized-v1")
    adapters: dict[str, Any] = {}
    registrations = {}
    for model_id in teacher_ids:
        registrations[model_id] = registry.get_registration(model_id)
        adapter = registry.create_adapter(model_id)
        adapter.load(args.device)
        adapters[model_id] = adapter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = {
        teacher_ids[0]: args.output_dir / "small",
        teacher_ids[1]: args.output_dir / "large",
    }
    for directory in cache_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    try:
        for record in records:
            probabilities: dict[str, np.ndarray] = {}
            for model_id in teacher_ids:
                run_dir = args.output_dir / "adapter-runs" / record["source_image_id"] / model_id
                output = adapters[model_id].predict(
                    SegmentationRequest(
                        image_id=record["source_image_id"],
                        image_path=Path(record["image_path"]),
                        run_dir=run_dir,
                        roi_mode=RoiMode.FULL_IMAGE,
                        threshold=None,
                        min_area_px=0,
                        device=DevicePreference(args.device),
                        seed=2026,
                    )
                )
                if output.probability_path is None:
                    raise ValueError(f"teacher {model_id} produced no probability")
                probability = np.load(output.probability_path, allow_pickle=False)
                probabilities[model_id] = np.asarray(probability, dtype=np.float32)
                cache_path = cache_dirs[model_id] / f"{record['source_image_id']}.npy"
                np.save(cache_path, probability.astype(np.float16), allow_pickle=False)
            shape = probabilities[teacher_ids[0]].shape
            if probabilities[teacher_ids[1]].shape != shape:
                raise ValueError("teacher probability shapes differ")
            valid = np.zeros(shape, dtype=bool)
            # Distill only where both teachers have scientific image content.
            invalid_bottom = max(
                registrations[model_id].metadata.inference_invalid_bottom_px
                for model_id in teacher_ids
            )
            valid[: shape[0] - invalid_bottom] = True
            target_path = Path(record["target_path"])
            _replace_target_cache(
                target_path,
                teacher_small=probabilities[teacher_ids[0]],
                teacher_large=probabilities[teacher_ids[1]],
                teacher_valid=valid,
            )
            output_records.append(
                {
                    "record_id": record["record_id"],
                    "source_image_id": record["source_image_id"],
                    "target_cache_sha256": _sha256(target_path),
                    "teacher_valid_px": int(valid.sum()),
                }
            )
    finally:
        for adapter in adapters.values():
            adapter.unload()
    manifest = {
        "schema_version": 1,
        "records": output_records,
        "teachers": [
            {
                "model_id": model_id,
                "weight_sha256": registrations[model_id].weight_sha256,
                "config_sha256": registrations[model_id].config_sha256,
                "threshold": registrations[model_id].metadata.default_threshold,
                "invalid_bottom_px": registrations[
                    model_id
                ].metadata.inference_invalid_bottom_px,
            }
            for model_id in teacher_ids
        ],
        "confidence_rule": "probability >= 0.9 or probability <= 0.1",
        "scope": "all_labeled" if args.all_labeled else "unlabeled_only",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths = [
        *(path for directory in cache_dirs.values() for path in sorted(directory.glob("*.npy"))),
        manifest_path,
    ]
    (args.output_dir / "sha256sums.txt").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(args.output_dir)}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
