"""Manifest-backed SEM patch dataset and domain-specific augmentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    record_id: str
    image_path: Path
    mask_path: Path | None
    target_path: Path
    invalid_bottom_px: int
    split: str
    group_id: str
    supervision_available: bool
    morphology_group: str = "unknown"


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            target_path = raw.get("target_path")
            if not isinstance(target_path, str) or not target_path:
                raise ValueError(f"manifest record {raw.get('record_id')} lacks target_path")
            records.append(
                ManifestRecord(
                    record_id=str(raw["record_id"]),
                    image_path=Path(raw["image_path"]),
                    mask_path=(
                        Path(raw["mask_path"])
                        if isinstance(raw.get("mask_path"), str)
                        else None
                    ),
                    target_path=Path(target_path),
                    invalid_bottom_px=int(raw["invalid_bottom_px"]),
                    split=str(raw["split"]),
                    group_id=str(raw["group_id"]),
                    supervision_available=bool(raw.get("supervision_available", True)),
                    morphology_group=str(raw.get("morphology_group", "unknown")),
                )
            )
    if not records:
        raise ValueError("manifest contains no records")
    return records


def _normalize_sem(image: np.ndarray, valid_height: int) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    reference = values[:valid_height]
    lower, upper = np.percentile(reference, (1.0, 99.0))
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.float32)
    return np.asarray(
        np.clip((values - lower) / (upper - lower), 0.0, 1.0),
        dtype=np.float32,
    )


def _normalize_fixed(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _load_record_arrays(
    record: ManifestRecord,
    *,
    normalization: str,
) -> dict[str, np.ndarray | None]:
    image = _load_gray(record.image_path)
    with np.load(record.target_path, allow_pickle=False) as archive:
        arrays: dict[str, np.ndarray | None] = {
            "instance_labels": np.asarray(archive["instance_labels"]).copy(),
            "center": np.asarray(archive["center"]).copy(),
            "boundary": np.asarray(archive["boundary"]).copy(),
            "distance": np.asarray(archive["distance"]).copy(),
            "scale": np.asarray(archive["scale"]).copy(),
            "teacher_small": (
                np.asarray(archive["teacher_small"]).copy()
                if "teacher_small" in archive
                else None
            ),
            "teacher_large": (
                np.asarray(archive["teacher_large"]).copy()
                if "teacher_large" in archive
                else None
            ),
            "teacher_valid": (
                np.asarray(archive["teacher_valid"]).copy()
                if "teacher_valid" in archive
                else None
            ),
        }
    labels = arrays["instance_labels"]
    if labels is None or image.shape != labels.shape:
        raise ValueError(f"image/target shape mismatch for {record.record_id}")
    valid_height = image.shape[0] - record.invalid_bottom_px
    if normalization == "percentile":
        arrays["image"] = _normalize_sem(image, valid_height)
    elif normalization == "fixed":
        arrays["image"] = _normalize_fixed(image)
    else:
        raise ValueError("normalization must be percentile or fixed")
    return arrays


def copy_paste_instances(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    max_instances: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Translate same-domain instances while preserving collision-free IDs."""

    output_image = np.asarray(image, dtype=np.float32).copy()
    output_labels = np.asarray(labels, dtype=np.int32).copy()
    instance_ids = np.unique(output_labels)
    instance_ids = instance_ids[instance_ids > 0]
    if len(instance_ids) == 0 or max_instances <= 0:
        return output_image, output_labels, {"pasted_ids": [], "touching": []}
    selected = rng.choice(
        instance_ids,
        size=min(max_instances, len(instance_ids)),
        replace=False,
    )
    pasted: list[int] = []
    touching: list[bool] = []
    next_id = int(output_labels.max()) + 1
    height, width = output_labels.shape
    for source_id in np.atleast_1d(selected):
        source = output_labels == int(source_id)
        ys, xs = np.nonzero(source)
        if len(ys) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        crop_mask = source[y1:y2, x1:x2]
        crop_image = output_image[y1:y2, x1:x2].copy()
        crop_height, crop_width = crop_mask.shape
        if crop_height >= height or crop_width >= width:
            continue
        target_y = int(rng.integers(0, height - crop_height + 1))
        target_x = int(rng.integers(0, width - crop_width + 1))
        destination = output_labels[
            target_y : target_y + crop_height,
            target_x : target_x + crop_width,
        ]
        collision = crop_mask & (destination > 0)
        if float(collision.mean()) > 0.15:
            continue
        before = ndi.binary_dilation(destination > 0)
        target_mask = crop_mask & ~collision
        if not target_mask.any():
            continue
        image_region = output_image[
            target_y : target_y + crop_height,
            target_x : target_x + crop_width,
        ]
        image_region[target_mask] = crop_image[target_mask]
        destination[target_mask] = next_id
        pasted.append(next_id)
        touching.append(bool(np.any(before & target_mask)))
        next_id += 1
    return output_image, output_labels, {"pasted_ids": pasted, "touching": touching}


class MSBIPatchDataset:
    """PyTorch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        records: list[ManifestRecord],
        *,
        patch_size: int,
        samples_per_epoch: int,
        seed: int,
        augment: bool,
        density_sampling_probability: float = 0.65,
        preload_records: bool = False,
        morphology_balanced_sampling: bool = False,
        normalization: str = "percentile",
    ) -> None:
        if patch_size <= 0 or samples_per_epoch <= 0:
            raise ValueError("patch_size and samples_per_epoch must be positive")
        self.records = list(records)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.augment = augment
        self.density_sampling_probability = density_sampling_probability
        self.morphology_balanced_sampling = morphology_balanced_sampling
        if normalization not in {"percentile", "fixed"}:
            raise ValueError("normalization must be percentile or fixed")
        self.normalization = normalization
        self.epoch = 0
        grouped: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            grouped.setdefault(record.morphology_group, []).append(index)
        self.morphology_record_indices = {
            name: tuple(indices)
            for name, indices in sorted(grouped.items())
        }
        self.record_cache = (
            [
                _load_record_arrays(record, normalization=self.normalization)
                for record in self.records
            ]
            if preload_records
            else None
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _crop_origin(
        self,
        labels: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        height, width = labels.shape
        patch = self.patch_size
        if height < patch or width < patch:
            raise ValueError("source image is smaller than requested patch")
        if rng.random() < self.density_sampling_probability and np.any(labels > 0):
            ys, xs = np.nonzero(labels > 0)
            chosen = int(rng.integers(0, len(ys)))
            y = int(np.clip(ys[chosen] - patch // 2, 0, height - patch))
            x = int(np.clip(xs[chosen] - patch // 2, 0, width - patch))
            return y, x
        return (
            int(rng.integers(0, height - patch + 1)),
            int(rng.integers(0, width - patch + 1)),
        )

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        if self.augment:
            if self.morphology_balanced_sampling:
                names = tuple(self.morphology_record_indices)
                group_name = names[int(rng.integers(0, len(names)))]
                indices = self.morphology_record_indices[group_name]
                record_index = indices[int(rng.integers(0, len(indices)))]
            else:
                record_index = int(rng.integers(0, len(self.records)))
        else:
            record_index = index % len(self.records)
        record = self.records[record_index]
        loaded = (
            self.record_cache[record_index]
            if self.record_cache is not None
            else _load_record_arrays(record, normalization=self.normalization)
        )
        normalized = loaded["image"]
        labels = loaded["instance_labels"]
        center = loaded["center"]
        boundary = loaded["boundary"]
        distance = loaded["distance"]
        scale = loaded["scale"]
        required = (normalized, labels, center, boundary, distance, scale)
        if any(value is None for value in required):
            raise ValueError(f"required target is missing for {record.record_id}")
        normalized = np.asarray(normalized)
        labels = np.asarray(labels)
        center = np.asarray(center)
        boundary = np.asarray(boundary)
        distance = np.asarray(distance)
        scale = np.asarray(scale)
        valid_height = normalized.shape[0] - record.invalid_bottom_px
        y, x = self._crop_origin(labels[:valid_height], rng=rng)
        patch = self.patch_size
        slices = (slice(y, y + patch), slice(x, x + patch))
        teacher_arrays = {
            name: (
                np.asarray(loaded[name])[slices].copy()
                if loaded[name] is not None
                else np.zeros((patch, patch), dtype=np.float32)
            )
            for name in ("teacher_small", "teacher_large", "teacher_valid")
        }
        arrays: dict[str, np.ndarray] = {
            "image": normalized[slices].copy(),
            "instance_labels": labels[slices].copy(),
            "center": center[slices].copy(),
            "boundary": boundary[slices].copy(),
            "distance": distance[slices].copy(),
            "scale": scale[slices].copy(),
            "valid": np.ones((patch, patch), dtype=np.float32),
            **teacher_arrays,
        }
        if self.augment:
            rotation = int(rng.integers(0, 4))
            flip_horizontal = bool(rng.random() < 0.5)
            flip_vertical = bool(rng.random() < 0.5)
            for name, values in arrays.items():
                values = np.rot90(values, rotation).copy()
                if flip_horizontal:
                    values = np.flip(values, axis=1).copy()
                if flip_vertical:
                    values = np.flip(values, axis=0).copy()
                arrays[name] = values
            intensity = arrays["image"]
            contrast = float(rng.uniform(0.85, 1.15))
            brightness = float(rng.uniform(-0.08, 0.08))
            gamma = float(rng.uniform(0.85, 1.15))
            intensity = np.clip((intensity - 0.5) * contrast + 0.5 + brightness, 0, 1)
            intensity = np.power(intensity, gamma).astype(np.float32)
            if rng.random() < 0.35:
                intensity += rng.normal(0.0, 0.015, intensity.shape).astype(np.float32)
            if rng.random() < 0.2:
                stripe = rng.normal(0.0, 0.02, (intensity.shape[0], 1)).astype(np.float32)
                intensity += stripe
            if rng.random() < 0.15:
                pil = Image.fromarray(
                    np.clip(intensity * 255, 0, 255).astype(np.uint8), mode="L"
                )
                intensity = np.asarray(
                    pil.filter(ImageFilter.GaussianBlur(radius=0.6)),
                    dtype=np.float32,
                ) / 255.0
            arrays["image"] = np.clip(intensity, 0, 1).astype(np.float32)
        foreground = (arrays["instance_labels"] > 0).astype(np.float32)
        return {
            "image": arrays["image"][None].astype(np.float32),
            "foreground": foreground[None],
            "center": arrays["center"][None].astype(np.float32),
            "boundary": arrays["boundary"][None].astype(np.float32),
            "distance": arrays["distance"][None].astype(np.float32),
            "scale": arrays["scale"][None].astype(np.int64),
            "valid": arrays["valid"][None].astype(np.float32),
            "supervised_valid": (
                arrays["valid"][None].astype(np.float32)
                if record.supervision_available
                else np.zeros((1, patch, patch), dtype=np.float32)
            ),
            "teacher_small": arrays["teacher_small"][None].astype(np.float32),
            "teacher_large": arrays["teacher_large"][None].astype(np.float32),
            "teacher_valid": arrays["teacher_valid"][None].astype(np.float32),
            "record_id": record.record_id,
        }
