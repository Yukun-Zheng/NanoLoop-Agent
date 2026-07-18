import numpy as np
import pytest

from app.analysis.instance_artifacts import (
    canonical_instances_payload,
    decode_binary_mask,
    encode_binary_mask,
)
from app.analysis.postprocessing import NormalizedInstance


def test_canonical_instance_rle_round_trips_exact_mask() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[0, 1:4] = True
    mask[2:4, 5:7] = True
    instance = NormalizedInstance(
        instance_index=1,
        mask=mask,
        bbox=(1, 0, 7, 4),
        area_px=7,
        confidence=0.75,
        touches_roi_boundary=True,
    )

    payload = canonical_instances_payload([instance], width=7, height=5)
    record = payload["instances"][0]
    encoded = record["mask"]
    decoded = decode_binary_mask(
        starts=encoded["starts"],
        lengths=encoded["lengths"],
        width=7,
        height=5,
    )

    assert np.array_equal(decoded, mask)
    assert payload["coordinate_space"] == "original_px"
    assert payload["instance_count"] == 1
    assert record["bbox_xyxy"] == [1, 0, 7, 4]
    assert len(record["mask"]["sha256"]) == 64


def test_rle_rejects_invalid_ranges() -> None:
    starts, lengths = encode_binary_mask(np.zeros((2, 2), dtype=bool))
    assert starts == []
    assert lengths == []

    with pytest.raises(ValueError, match="invalid or overlapping"):
        decode_binary_mask(starts=[2, 1], lengths=[1, 1], width=2, height=2)
