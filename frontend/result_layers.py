"""Pure helpers for honest, signed-artifact result layer previews."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

RESULT_LAYER_LABELS = {
    "original_image": "原始图像",
    "mask_url": "分割掩膜",
    "overlay_url": "叠加预览",
    "labeled_particles_url": "颗粒标注图",
    "probability_url": "概率图",
}
RESULT_LAYER_ORDER = tuple(RESULT_LAYER_LABELS)


@dataclass(frozen=True, slots=True)
class ResultLayerSource:
    """One previewable layer and its backend-issued signed download URL."""

    key: str
    label: str
    download_url: str


@dataclass(frozen=True, slots=True)
class ResultLayerDisplay:
    """Browser-safe display representation; raw source bytes remain unchanged."""

    content: bytes
    content_type: str
    width: int
    height: int
    note: str | None = None


def result_layer_sources(
    run: dict[str, Any],
    image: dict[str, Any] | None,
) -> tuple[ResultLayerSource, ...]:
    """Return available layers in a stable, scientifically meaningful order."""

    artifacts = run.get("artifacts")
    artifact_urls = artifacts if isinstance(artifacts, dict) else {}
    original_url = image.get("original_download_url") if image else None
    urls = {"original_image": original_url, **artifact_urls}
    return tuple(
        ResultLayerSource(
            key=key,
            label=RESULT_LAYER_LABELS[key],
            download_url=value.strip(),
        )
        for key in RESULT_LAYER_ORDER
        if isinstance((value := urls.get(key)), str) and value.strip()
    )


def prepare_result_layer_display(
    *,
    layer_key: str,
    content: bytes,
    content_type: str,
    filename: str | None,
) -> ResultLayerDisplay:
    """Validate image bytes or convert a probability NPY into a fixed-scale PNG."""

    if not content:
        raise ValueError("结果图层文件为空")
    normalized_type = content_type.split(";", maxsplit=1)[0].strip().casefold()
    suffix = Path(filename or "").suffix.casefold()
    if (
        layer_key == "probability_url"
        and normalized_type
        not in {
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/webp",
        }
        and suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    ):
        return _probability_npy_display(content)
    return _validated_image_display(content)


def _validated_image_display(content: bytes) -> ResultLayerDisplay:
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
    except (OSError, ValueError) as error:
        raise ValueError("制品不是可预览的图像") from error
    if width <= 0 or height <= 0:
        raise ValueError("制品图像尺寸无效")
    return ResultLayerDisplay(
        content=content,
        content_type="image/png" if content.startswith(b"\x89PNG\r\n\x1a\n") else "image/*",
        width=width,
        height=height,
    )


def _probability_npy_display(content: bytes) -> ResultLayerDisplay:
    try:
        loaded = np.load(BytesIO(content), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("概率制品既不是图像，也不是有效的 NPY 数组") from error
    if not isinstance(loaded, np.ndarray) or loaded.ndim != 2:
        raise ValueError("概率 NPY 必须是二维数组")
    if loaded.size == 0 or not np.issubdtype(loaded.dtype, np.number):
        raise ValueError("概率 NPY 必须包含数值")
    probability = np.asarray(loaded, dtype=np.float64)
    if not np.isfinite(probability).all():
        raise ValueError("概率 NPY 包含非有限值")
    minimum = float(probability.min())
    maximum = float(probability.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("概率 NPY 的数值必须位于 0–1")

    pixels = np.rint(probability * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    width, height = image.size
    return ResultLayerDisplay(
        content=output.getvalue(),
        content_type="image/png",
        width=width,
        height=height,
        note="概率预览使用固定 0–1 灰度：黑色为 0，白色为 1；原始 NPY 文件未被改写。",
    )


__all__ = [
    "RESULT_LAYER_LABELS",
    "RESULT_LAYER_ORDER",
    "ResultLayerDisplay",
    "ResultLayerSource",
    "prepare_result_layer_display",
    "result_layer_sources",
]
