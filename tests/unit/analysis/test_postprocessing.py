import numpy as np
import pytest

from app.analysis.config import PostprocessProfile
from app.analysis.postprocessing import normalize_native_instances, normalize_semantic_mask


def test_semantic_mask_becomes_deterministic_instances_with_confidence() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 30:40] = 1
    mask[20:35, 5:15] = 1
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[mask.astype(bool)] = 0.8
    roi = np.ones((64, 64), dtype=bool)

    instances = normalize_semantic_mask(
        mask,
        roi_mask=roi,
        probability=probability,
        profile=PostprocessProfile(min_area_px=8, exclude_border=False),
    )
    assert [item.instance_index for item in instances] == [1, 2]
    assert instances[0].bbox == (30, 5, 40, 15)
    assert instances[1].bbox == (5, 20, 15, 35)
    assert instances[0].confidence == pytest.approx(0.8)


def test_semantic_mask_is_zero_outside_roi_and_border_instance_can_be_excluded() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:8, 0:8] = 1
    mask[10:20, 10:20] = 1
    roi = np.ones((32, 32), dtype=bool)
    instances = normalize_semantic_mask(
        mask,
        roi_mask=roi,
        profile=PostprocessProfile(min_area_px=4, exclude_border=True),
    )
    assert len(instances) == 1
    assert instances[0].bbox == (10, 10, 20, 20)


def test_native_duplicate_instances_keep_higher_confidence() -> None:
    first = np.zeros((32, 32), dtype=bool)
    second = np.zeros((32, 32), dtype=bool)
    first[5:20, 5:20] = True
    second[6:21, 6:21] = True
    roi = np.ones((32, 32), dtype=bool)
    instances = normalize_native_instances(
        [first, second],
        roi_mask=roi,
        confidences=[0.4, 0.9],
        profile=PostprocessProfile(
            min_area_px=4,
            exclude_border=False,
            instance_iou_threshold=0.7,
        ),
    )
    assert len(instances) == 1
    assert instances[0].confidence == 0.9
    assert instances[0].bbox == (6, 6, 21, 21)


def test_empty_native_instance_is_ignored_when_min_area_is_zero() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    roi = np.ones((8, 8), dtype=bool)

    instances = normalize_native_instances(
        [empty],
        roi_mask=roi,
        profile=PostprocessProfile(min_area_px=0, exclude_border=False),
    )

    assert instances == []


def test_native_instance_order_is_independent_of_adapter_output_order() -> None:
    first = np.zeros((16, 16), dtype=bool)
    second = np.zeros((16, 16), dtype=bool)
    first[2:10, 2] = True
    first[9, 2:10] = True
    second[2, 2:10] = True
    second[2:10, 9] = True
    roi = np.ones((16, 16), dtype=bool)
    profile = PostprocessProfile(
        min_area_px=1,
        exclude_border=False,
        instance_iou_threshold=0.7,
    )

    forward = normalize_native_instances(
        [first, second],
        roi_mask=roi,
        confidences=[0.5, 0.5],
        profile=profile,
    )
    reverse = normalize_native_instances(
        [second, first],
        roi_mask=roi,
        confidences=[0.5, 0.5],
        profile=profile,
    )

    assert [item.bbox for item in forward] == [item.bbox for item in reverse]
    assert [item.instance_index for item in forward] == [1, 2]
    assert all(
        np.array_equal(left.mask, right.mask)
        for left, right in zip(forward, reverse, strict=True)
    )


def test_equal_confidence_duplicate_winner_is_deterministic() -> None:
    first = np.zeros((16, 16), dtype=bool)
    second = np.zeros((16, 16), dtype=bool)
    first[2:12, 2:12] = True
    second[2:12, 2:12] = True
    second[2, 2] = False
    roi = np.ones((16, 16), dtype=bool)
    profile = PostprocessProfile(
        min_area_px=1,
        exclude_border=False,
        instance_iou_threshold=0.9,
    )

    forward = normalize_native_instances(
        [first, second],
        roi_mask=roi,
        confidences=[0.5, 0.5],
        profile=profile,
    )
    reverse = normalize_native_instances(
        [second, first],
        roi_mask=roi,
        confidences=[0.5, 0.5],
        profile=profile,
    )

    assert len(forward) == len(reverse) == 1
    assert np.array_equal(forward[0].mask, reverse[0].mask)


@pytest.mark.parametrize("mode", ["semantic", "native"])
def test_hole_filling_never_restores_invalid_roi_pixels(mode: str) -> None:
    mask = np.ones((7, 7), dtype=np.uint8)
    roi = np.ones((7, 7), dtype=bool)
    roi[3, 3] = False
    profile = PostprocessProfile(
        min_area_px=1,
        fill_holes=True,
        exclude_border=False,
    )

    if mode == "semantic":
        instances = normalize_semantic_mask(mask, roi_mask=roi, profile=profile)
    else:
        instances = normalize_native_instances([mask], roi_mask=roi, profile=profile)

    assert len(instances) == 1
    assert instances[0].area_px == int(roi.sum())
    assert not instances[0].mask[3, 3]
    assert not np.any(instances[0].mask & ~roi)
