"""Canonical, model-agnostic serialization of final postprocessed instances."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.analysis.postprocessing import NormalizedInstance


def canonical_instances_payload(
    instances: list[NormalizedInstance],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Serialize exact full-image masks with deterministic row-major RLE."""

    if width <= 0 or height <= 0:
        raise ValueError("instance artifact dimensions must be positive")
    records: list[dict[str, Any]] = []
    for instance in instances:
        if instance.mask.shape != (height, width):
            raise ValueError("instance mask dimensions do not match artifact dimensions")
        starts, lengths = encode_binary_mask(instance.mask)
        records.append(
            {
                "instance_index": instance.instance_index,
                "bbox_xyxy": list(instance.bbox),
                "area_px": instance.area_px,
                "confidence": instance.confidence,
                "touches_roi_boundary": instance.touches_roi_boundary,
                "mask": {
                    "encoding": "flat_rle_v1",
                    "order": "row_major",
                    "starts": starts,
                    "lengths": lengths,
                    "sha256": hashlib.sha256(
                        np.packbits(instance.mask, bitorder="little").tobytes()
                    ).hexdigest(),
                },
            }
        )
    return {
        "coordinate_space": "original_px",
        "width": width,
        "height": height,
        "instance_count": len(records),
        "instances": records,
    }


def encode_binary_mask(mask: NDArray[np.bool_]) -> tuple[list[int], list[int]]:
    """Return zero-based starts and lengths for true runs in row-major order."""

    flat = np.asarray(mask, dtype=np.bool_).reshape(-1)
    padded = np.pad(flat.astype(np.int8), (1, 1))
    transitions = np.flatnonzero(np.diff(padded))
    starts = transitions[0::2]
    ends = transitions[1::2]
    return starts.astype(int).tolist(), (ends - starts).astype(int).tolist()


def decode_binary_mask(
    *,
    starts: list[int],
    lengths: list[int],
    width: int,
    height: int,
) -> NDArray[np.bool_]:
    """Decode ``flat_rle_v1`` for contract tests and downstream consumers."""

    if len(starts) != len(lengths):
        raise ValueError("RLE starts and lengths must have equal length")
    size = width * height
    flat = np.zeros(size, dtype=np.bool_)
    previous_end = 0
    for start, length in zip(starts, lengths, strict=True):
        end = start + length
        if start < previous_end or length <= 0 or end > size:
            raise ValueError("invalid or overlapping RLE run")
        flat[start:end] = True
        previous_end = end
    return flat.reshape((height, width))
