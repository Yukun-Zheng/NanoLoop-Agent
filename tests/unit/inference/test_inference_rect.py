from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.contracts.analyses import PixelRect
from app.contracts.enums import (
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelMetadata
from app.inference.adapters.sam2 import SAM2Adapter
from app.inference.adapters.yolo_seg import YOLOSegAdapter


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (80, 60), color=96).save(buffer, format="PNG")
    return buffer.getvalue()


def _metadata(model_id: str, family: ModelFamily) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        family=family,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.5,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
    )


def _request(tmp_path: Path, name: str) -> SegmentationRequest:
    return SegmentationRequest(
        image_id="image-footer",
        image_path=tmp_path / "absent-pinned-image",
        image_bytes=_image_bytes(),
        run_dir=tmp_path / name,
        roi_mode=RoiMode.FULL_IMAGE,
        inference_rect=PixelRect(x1=0, y1=0, x2=80, y2=45),
        threshold=0.5,
    )


def test_yolo_never_sends_detected_footer_to_model(tmp_path: Path) -> None:
    adapter = YOLOSegAdapter(
        metadata=_metadata("yolo-footer-test", ModelFamily.YOLO_SEG),
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={},
    )
    adapter._loaded = True
    adapter._device = "cpu"
    observed_shapes: list[tuple[int, int]] = []

    def predict_masks(
        image: np.ndarray,
        *,
        target_shape: tuple[int, int],
        kwargs: dict[str, Any],
    ) -> tuple[list[np.ndarray], list[float | None]]:
        del kwargs
        observed_shapes.append(image.shape[:2])
        return [np.ones(target_shape, dtype=bool)], [0.9]

    adapter._predict_masks = predict_masks

    output = adapter.predict(_request(tmp_path, "yolo-run"))

    mask = np.asarray(Image.open(output.binary_mask_path)) > 0
    assert observed_shapes == [(45, 80)]
    assert mask[:45].all()
    assert not mask[45:].any()


def test_sam_automatic_generator_never_sees_detected_footer(tmp_path: Path) -> None:
    adapter = SAM2Adapter(
        metadata=_metadata("sam-footer-test", ModelFamily.SAM2),
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={},
    )
    adapter._loaded = True

    class Generator:
        def generate(self, image: np.ndarray) -> list[dict[str, object]]:
            assert image.shape[:2] == (45, 80)
            return [
                {
                    "segmentation": np.ones((45, 80), dtype=bool),
                    "predicted_iou": 0.8,
                }
            ]

    adapter._generator = Generator()

    output = adapter.predict(_request(tmp_path, "sam-run"))

    mask = np.asarray(Image.open(output.binary_mask_path)) > 0
    assert mask[:45].all()
    assert not mask[45:].any()
