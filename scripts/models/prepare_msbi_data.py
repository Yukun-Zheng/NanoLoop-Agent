#!/usr/bin/env python3
"""Freeze grouped manifests and derived pseudo-instance targets without test leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.msbi.targets import TargetConfig, generate_instance_targets, semantic_to_pseudo_instances


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stem(path: Path) -> str:
    return path.stem.removesuffix("_mask")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _save_targets(
    mask_path: Path,
    destination: Path,
    *,
    invalid_bottom_px: int,
    config: TargetConfig,
) -> dict[str, Any]:
    with Image.open(mask_path) as image:
        semantic = np.asarray(image.convert("L")) > 0
    valid = np.ones(semantic.shape, dtype=bool)
    if invalid_bottom_px:
        valid[-invalid_bottom_px:] = False
    labels = semantic_to_pseudo_instances(semantic & valid, connectivity=config.connectivity)
    targets = generate_instance_targets(labels, valid_mask=valid, config=config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        instance_labels=np.asarray(targets["instance_labels"], dtype=np.int32),
        center=np.asarray(targets["center"], dtype=np.float16),
        boundary=np.asarray(targets["boundary"], dtype=np.uint8),
        distance=np.asarray(targets["distance"], dtype=np.float16),
        scale=np.asarray(targets["scale"], dtype=np.int8),
    )
    areas = np.asarray(targets["area_by_instance"], dtype=np.int64)
    return {
        "pseudo_instance_count": len(areas),
        "foreground_px": int(np.count_nonzero(semantic & valid)),
        "minimum_instance_area_px": int(areas[:, 1].min()) if len(areas) else None,
        "maximum_instance_area_px": int(areas[:, 1].max()) if len(areas) else None,
    }


def _read_official_tables(
    archive_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str]:
    with zipfile.ZipFile(archive_path) as archive:
        train_name = next(
            name for name in archive.namelist() if name.endswith("/train_groups.csv")
        )
        test_name = next(
            name for name in archive.namelist() if name.endswith("/test_groups.csv")
        )
        train_bytes = archive.read(train_name)
        test_bytes = archive.read(test_name)
    train = list(csv.DictReader(io.StringIO(train_bytes.decode("utf-8-sig"))))
    test = list(csv.DictReader(io.StringIO(test_bytes.decode("utf-8-sig"))))
    return (
        train,
        test,
        hashlib.sha256(train_bytes).hexdigest(),
        hashlib.sha256(test_bytes).hexdigest(),
    )


def _sealed_test_records(
    *,
    archive_path: Path,
    sem_images: dict[str, Path],
    invalid_bottom_px: int,
    official_test: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {
            Path(name).stem.removesuffix("_mask"): name
            for name in archive.namelist()
            if "/test_mask_human/" in name and not name.endswith("/")
        }
        for official in official_test:
            stem = Path(official["filename"]).stem
            member = members.get(stem)
            if member is None:
                raise ValueError(f"official test record has no held-out mask: {stem}")
            image_path = sem_images.get(stem)
            if image_path is None:
                raise ValueError(f"sealed test mask has no SEM source image: {stem}")
            group = stem.split("-", maxsplit=1)[0]
            records.append(
                {
                    "record_id": f"independent-test-{stem}",
                    "sample_id": group,
                    "source_image_id": stem,
                    "image_path": str(image_path.resolve()),
                    "mask_path": f"zip://{archive_path.resolve()}::{member}",
                    "instance_mask_path": None,
                    "target_path": None,
                    "material_name": group,
                    "material_formula": None,
                    "scale_nm_per_pixel": None,
                    "invalid_bottom_px": invalid_bottom_px,
                    "split": "independent_test",
                    "group_id": group,
                    "morphology_group": official["group"],
                    "license_status": (
                        "training_use_authorized_in_current_task_redistribution_unknown"
                    ),
                    "image_sha256": _sha256(image_path),
                    # Hashing bytes freezes identity without decoding or viewing the held-out GT.
                    "mask_sha256": _member_sha256(archive, member),
                    "derived_pseudo_instance": None,
                    "gt_type": "sealed_until_tolerance_policy_is_frozen",
                    "sealed": True,
                    "supervision_available": True,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--sem-all-dir", type=Path, required=True)
    parser.add_argument("--official-delivery-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--invalid-bottom-px", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    image_paths = {
        _stem(path): path
        for path in args.images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"}
    }
    mask_paths = {
        _stem(path): path
        for path in args.masks_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png"}
    }
    sem_images = {
        _stem(path): path
        for path in args.sem_all_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"}
    }
    official_train, official_test, train_table_sha, test_table_sha = _read_official_tables(
        args.official_delivery_zip
    )
    train_stems = {Path(record["filename"]).stem for record in official_train}
    test_stems = {Path(record["filename"]).stem for record in official_test}
    if train_stems & test_stems:
        raise ValueError("official train and test tables overlap")
    if set(image_paths) != train_stems:
        raise ValueError(
            f"official train/image mismatch: table_only={sorted(train_stems-set(image_paths))}, "
            f"images_only={sorted(set(image_paths)-train_stems)}"
        )
    if set(mask_paths) - train_stems:
        undeclared = sorted(set(mask_paths) - train_stems)
        raise ValueError(f"train masks not declared by official table: {undeclared}")
    official_by_stem = {
        Path(record["filename"]).stem: record for record in official_train
    }
    groups = sorted({stem.split("-", maxsplit=1)[0] for stem in train_stems})
    labeled_groups = sorted(
        {stem.split("-", maxsplit=1)[0] for stem in train_stems if stem in mask_paths}
    )
    shuffled = groups.copy()
    shuffled = labeled_groups.copy()
    random.Random(args.seed).shuffle(shuffled)
    validation_count = max(1, round(len(groups) * args.validation_fraction))
    validation_groups = set(shuffled[:validation_count])
    config = TargetConfig()
    records: list[dict[str, Any]] = []
    missing_masks: list[str] = []
    for stem in sorted(train_stems):
        image_path = image_paths[stem]
        mask_path = mask_paths.get(stem)
        group = stem.split("-", maxsplit=1)[0]
        split = "validation" if group in validation_groups else "train"
        target_path = args.target_cache_dir / f"{stem}.npz"
        if mask_path is None:
            missing_masks.append(stem)
            with Image.open(image_path) as image:
                shape = np.asarray(image.convert("L")).shape
            valid = np.zeros(shape, dtype=np.float32)
            valid[: shape[0] - args.invalid_bottom_px] = 1.0
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target_path,
                instance_labels=np.zeros(shape, dtype=np.int32),
                center=np.zeros(shape, dtype=np.float16),
                boundary=np.zeros(shape, dtype=np.uint8),
                distance=np.zeros(shape, dtype=np.float16),
                scale=np.full(shape, -1, dtype=np.int8),
            )
            target_stats = {
                "pseudo_instance_count": None,
                "foreground_px": None,
                "minimum_instance_area_px": None,
                "maximum_instance_area_px": None,
            }
        else:
            target_stats = _save_targets(
                mask_path,
                target_path,
                invalid_bottom_px=args.invalid_bottom_px,
                config=config,
            )
        records.append(
            {
                "record_id": f"{split}-{stem}",
                "sample_id": group,
                "source_image_id": stem,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()) if mask_path is not None else None,
                "instance_mask_path": (
                    str(target_path.resolve()) if mask_path is not None else None
                ),
                "target_path": str(target_path.resolve()),
                "material_name": group,
                "material_formula": None,
                "scale_nm_per_pixel": None,
                "invalid_bottom_px": args.invalid_bottom_px,
                "split": split,
                "group_id": group,
                "morphology_group": official_by_stem[stem]["group"],
                "license_status": "training_use_authorized_in_current_task_redistribution_unknown",
                "image_sha256": _sha256(image_path),
                "mask_sha256": _sha256(mask_path) if mask_path is not None else None,
                "derived_pseudo_instance": True if mask_path is not None else None,
                "gt_type": (
                    "binary_semantic_with_connected_component_pseudo_instances"
                    if mask_path is not None
                    else "unlabeled_teacher_distillation_only"
                ),
                "sealed": False,
                "supervision_available": mask_path is not None,
                **target_stats,
            }
        )
    test_records = _sealed_test_records(
        archive_path=args.official_delivery_zip,
        sem_images=sem_images,
        invalid_bottom_px=args.invalid_bottom_px,
        official_test=official_test,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        "train": [record for record in records if record["split"] == "train"],
        "validation": [record for record in records if record["split"] == "validation"],
        "independent_test": test_records,
    }
    for split, values in split_records.items():
        _write_jsonl(args.output_dir / f"{split}.jsonl", values)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "official_split_unit": "source_sem_view",
        "validation_split_unit": "material_group_within_official_train_pool",
        "validation_fraction": args.validation_fraction,
        "invalid_bottom_px_policy": {
            "value": args.invalid_bottom_px,
            "basis": (
                "all training masks are empty in the bottom 128 rows and SEM audit "
                "shows a consistent instrument information strip"
            ),
        },
        "groups": {
            "train": sorted(set(groups) - validation_groups),
            "validation": sorted(validation_groups),
            "independent_test": sorted({record["group_id"] for record in test_records}),
        },
        "counts": {name: len(values) for name, values in split_records.items()},
        "labeled_training_pool_count": len(records) - len(missing_masks),
        "unlabeled_training_pool_count": len(missing_masks),
        "unlabeled_training_records": sorted(missing_masks),
        "official_delivery_zip_sha256": _sha256(args.official_delivery_zip),
        "official_train_table_sha256": train_table_sha,
        "official_test_table_sha256": test_table_sha,
        "test_ground_truth_sealed": True,
        "gt_limitation": (
            "training labels are binary semantic masks; instance IDs are "
            "connected-component pseudo labels"
        ),
        "scale_status": "unknown_not_inferred_from_filename_or_unfrozen_scale_bar",
        "material_tokens_shared_between_official_train_and_test": sorted(
            {stem.split("-", maxsplit=1)[0] for stem in train_stems}
            & {stem.split("-", maxsplit=1)[0] for stem in test_stems}
        ),
    }
    split_manifest_path = args.output_dir / "split-manifest.json"
    split_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths = [
        args.output_dir / "train.jsonl",
        args.output_dir / "validation.jsonl",
        args.output_dir / "independent_test.jsonl",
        split_manifest_path,
    ]
    (args.output_dir / "sha256sums.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
