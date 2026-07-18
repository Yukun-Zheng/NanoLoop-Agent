from __future__ import annotations

import numpy as np

from app.contracts.analyses import ROIBox
from app.inference.adapters._utils import (
    deduplicate_instance_masks,
    embed_box_mask,
    iter_box_crops,
    mask_bbox,
    paste_box_probability,
)


def test_box_crops_expand_context_and_clip_to_image_bounds() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    boxes = [
        ROIBox(box_id="left", x1=2, y1=3, x2=30, y2=40),
        ROIBox(box_id="right", x1=70, y1=60, x2=100, y2=80),
    ]

    crops = iter_box_crops(image, boxes, context_px=16)

    assert (crops[0].x1, crops[0].y1, crops[0].x2, crops[0].y2) == (0, 0, 46, 56)
    assert crops[0].local_box == (2, 3, 30, 40)
    assert (crops[1].x1, crops[1].y1, crops[1].x2, crops[1].y2) == (54, 44, 100, 80)
    assert crops[1].local_box == (16, 16, 46, 36)


def test_probability_and_instance_mapping_force_box_exterior_to_zero() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    crop = iter_box_crops(
        image,
        [ROIBox(x1=20, y1=15, x2=50, y2=45)],
        context_px=10,
    )[0]
    local_probability = np.ones(crop.image.shape[:2], dtype=np.float32)
    probability = np.zeros(image.shape[:2], dtype=np.float32)

    paste_box_probability(probability, local_probability, crop)
    mapped_mask = embed_box_mask(local_probability > 0, crop, image.shape[:2])

    assert int(np.count_nonzero(probability)) == 30 * 30
    assert int(mapped_mask.sum()) == 30 * 30
    assert probability[14, 20] == 0
    assert not mapped_mask[45, 49]
    assert probability[15, 20] == 1
    assert mapped_mask[44, 49]


def test_duplicate_instances_prefer_confidence_and_sort_by_position() -> None:
    first = np.zeros((40, 40), dtype=bool)
    first[20:30, 20:30] = True
    duplicate = first.copy()
    duplicate[20, 20] = False
    earlier = np.zeros((40, 40), dtype=bool)
    earlier[2:8, 3:9] = True

    masks, scores = deduplicate_instance_masks(
        [first, duplicate, earlier],
        [0.4, 0.9, None],
    )

    assert len(masks) == 2
    assert [mask_bbox(mask) for mask in masks] == [(3, 2, 9, 8), (20, 20, 30, 30)]
    assert scores == [None, 0.9]


def test_equal_confidence_duplicate_selection_ignores_input_order() -> None:
    first = np.zeros((20, 20), dtype=bool)
    second = np.zeros((20, 20), dtype=bool)
    first[2:12, 2:12] = True
    second[2:12, 2:12] = True
    second[2, 2] = False

    forward, forward_scores = deduplicate_instance_masks(
        [first, second],
        [0.5, 0.5],
        iou_threshold=0.9,
    )
    reverse, reverse_scores = deduplicate_instance_masks(
        [second, first],
        [0.5, 0.5],
        iou_threshold=0.9,
    )

    assert len(forward) == len(reverse) == 1
    assert np.array_equal(forward[0], reverse[0])
    assert forward_scores == reverse_scores == [0.5]
