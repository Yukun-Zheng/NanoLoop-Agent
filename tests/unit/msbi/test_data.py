from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.msbi.data import ManifestRecord, MSBIPatchDataset, copy_paste_instances


def test_copy_paste_preserves_existing_ids_and_assigns_new_ids() -> None:
    image = np.zeros((48, 48), dtype=np.float32)
    labels = np.zeros((48, 48), dtype=np.int32)
    labels[4:10, 4:10] = 1
    image[labels == 1] = 0.8

    augmented_image, augmented_labels, metadata = copy_paste_instances(
        image,
        labels,
        rng=np.random.default_rng(2026),
        max_instances=1,
    )

    assert np.all(augmented_labels[4:10, 4:10] == 1)
    assert int(augmented_labels.max()) in {1, 2}
    assert set(metadata) == {"pasted_ids", "touching"}
    assert augmented_image.shape == image.shape


def test_dataset_geometric_augmentation_keeps_targets_aligned(
    tmp_path: Path,
) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:12, 17:25] = 1
    image[labels > 0] = 255
    image_path = tmp_path / "image.png"
    target_path = tmp_path / "target.npz"
    Image.fromarray(image).save(image_path)
    foreground = (labels > 0).astype(np.float32)
    np.savez_compressed(
        target_path,
        instance_labels=labels,
        center=foreground,
        boundary=foreground,
        distance=foreground,
        scale=foreground.astype(np.int64),
    )
    dataset = MSBIPatchDataset(
        [
            ManifestRecord(
                record_id="aligned",
                image_path=image_path,
                mask_path=None,
                target_path=target_path,
                invalid_bottom_px=0,
                split="train",
                group_id="group",
                supervision_available=True,
            )
        ],
        patch_size=32,
        samples_per_epoch=1,
        seed=2026,
        augment=True,
        density_sampling_probability=0.0,
    )
    sample = dataset[0]
    foreground_augmented = sample["foreground"][0] > 0.5
    assert np.array_equal(foreground_augmented, sample["center"][0] > 0.5)
    assert np.array_equal(foreground_augmented, sample["boundary"][0] > 0.5)
    assert np.array_equal(foreground_augmented, sample["distance"][0] > 0.5)


def test_preloaded_dataset_matches_on_demand_loading(tmp_path: Path) -> None:
    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[6:20, 9:25] = 1
    image_path = tmp_path / "image.tif"
    target_path = tmp_path / "target.npz"
    Image.fromarray(image).save(image_path)
    np.savez_compressed(
        target_path,
        instance_labels=labels,
        center=(labels > 0).astype(np.float16),
        boundary=(labels > 0).astype(np.uint8),
        distance=(labels > 0).astype(np.float16),
        scale=(labels > 0).astype(np.int8),
    )
    record = ManifestRecord(
        record_id="cached",
        image_path=image_path,
        mask_path=None,
        target_path=target_path,
        invalid_bottom_px=0,
        split="train",
        group_id="group",
        supervision_available=True,
    )
    common = {
        "patch_size": 24,
        "samples_per_epoch": 1,
        "seed": 2026,
        "augment": False,
        "density_sampling_probability": 0.0,
    }
    on_demand = MSBIPatchDataset([record], preload_records=False, **common)[0]
    preloaded = MSBIPatchDataset([record], preload_records=True, **common)[0]

    assert set(on_demand) == set(preloaded)
    for name, value in on_demand.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, preloaded[name]), name
        else:
            assert value == preloaded[name]


def test_morphology_balanced_sampling_balances_record_groups(
    tmp_path: Path,
) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    image_path = tmp_path / "image.png"
    target_path = tmp_path / "target.npz"
    Image.fromarray(image).save(image_path)
    np.savez_compressed(
        target_path,
        instance_labels=labels,
        center=(labels > 0).astype(np.float16),
        boundary=(labels > 0).astype(np.uint8),
        distance=(labels > 0).astype(np.float16),
        scale=(labels > 0).astype(np.int8),
    )
    morphologies = ["small"] * 6 + ["large", "agglomerated"]
    records = [
        ManifestRecord(
            record_id=f"{morphology}-{index}",
            image_path=image_path,
            mask_path=None,
            target_path=target_path,
            invalid_bottom_px=0,
            split="train",
            group_id=f"group-{index}",
            supervision_available=True,
            morphology_group=morphology,
        )
        for index, morphology in enumerate(morphologies)
    ]
    dataset = MSBIPatchDataset(
        records,
        patch_size=32,
        samples_per_epoch=300,
        seed=2026,
        augment=True,
        density_sampling_probability=0.0,
        morphology_balanced_sampling=True,
    )
    counts = {name: 0 for name in {"small", "large", "agglomerated"}}
    for index in range(len(dataset)):
        record_id = str(dataset[index]["record_id"])
        counts[record_id.split("-", 1)[0]] += 1

    assert all(70 <= count <= 130 for count in counts.values())
