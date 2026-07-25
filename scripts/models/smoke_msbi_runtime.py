#!/usr/bin/env python3
"""Load an exported MSBI TorchScript through the production adapter twice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from app.contracts.enums import (
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelMetadata
from app.inference.adapters.msbi import MSBIAdapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture(path: Path) -> None:
    height, width = 512, 512
    yy, xx = np.mgrid[:height, :width]
    image = np.full((height, width), 64.0, dtype=np.float32)
    for center_y, center_x, radius, value in (
        (150, 160, 36, 200),
        (160, 210, 34, 190),
        (300, 310, 70, 180),
    ):
        image[(yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2] = value
    image[-128:] = 245
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("runtime config must be a mapping")
    image_path = args.image or args.output_dir / "synthetic-sem.png"
    if args.image is None:
        _fixture(image_path)
    adapter = MSBIAdapter(
        metadata=ModelMetadata(
            model_id="msbi-instance-balanced-v1",
            family=ModelFamily.MSBI,
            variant=ModelVariant.DENSE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.5,
            preprocess_profile="sem-gray-p1-p99-crop-bottom-128-v1",
            postprocess_profile="msbi-center-boundary-watershed-v1",
            inference_invalid_bottom_px=int(config.get("bottom_crop_px", 0)),
        ),
        weight_path=args.weight,
        weight_bytes=args.weight.read_bytes(),
        config=config,
        weight_sha256=_sha256(args.weight),
    )
    adapter.load(args.device)
    outputs = []
    try:
        for index in range(2):
            outputs.append(
                adapter.predict(
                    SegmentationRequest(
                        image_id="runtime-smoke",
                        image_path=image_path,
                        run_dir=args.output_dir / f"run-{index}",
                        roi_mode=RoiMode.FULL_IMAGE,
                        device=DevicePreference(args.device),
                        seed=2026,
                    )
                )
            )
    finally:
        adapter.unload()
    first_labels = np.load(
        outputs[0].binary_mask_path.parent / "instance_labels.npy",
        allow_pickle=False,
    )
    second_labels = np.load(
        outputs[1].binary_mask_path.parent / "instance_labels.npy",
        allow_pickle=False,
    )
    report = {
        "status": "PASS" if np.array_equal(first_labels, second_labels) else "FAIL",
        "device": args.device,
        "weight_sha256": _sha256(args.weight),
        "deterministic_instance_labels": bool(np.array_equal(first_labels, second_labels)),
        "instance_count": int(first_labels.max()),
        "runtime_ms": [output.runtime_ms for output in outputs],
        "bottom_invalid_zero": bool(
            np.all(first_labels[-int(config.get("bottom_crop_px", 0)) :] == 0)
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime-smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
