"""Conservative image-context profiling for per-image model recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.contracts.analyses import PixelRect
from app.contracts.enums import ModelVariant


@dataclass(frozen=True)
class ImageModelProfile:
    variant: ModelVariant
    contrast: float
    edge_density: float

    @property
    def reason(self) -> str:
        labels = {
            ModelVariant.GENERAL: "通用形貌",
            ModelVariant.SMALL_PARTICLE: "细小/高频颗粒形貌",
            ModelVariant.LARGE_PARTICLE: "大颗粒/平滑形貌",
            ModelVariant.DENSE_PARTICLE: "高密度/团聚形貌",
            ModelVariant.LOW_CONTRAST: "低对比度形貌",
        }
        return (
            f"图像预检：{labels[self.variant]}"
            f"（对比度 {self.contrast:.3f}，边缘密度 {self.edge_density:.3f}）"
        )


def profile_image_for_model(path: Path, valid_rect: PixelRect) -> ImageModelProfile:
    """Infer a coarse model variant without running a segmentation model.

    This is deliberately conservative: it only changes away from ``general``
    when downsampled grayscale evidence crosses a clear morphology threshold.
    The selected model remains a recommendation, not a scientific conclusion.
    """

    with Image.open(path) as source:
        grayscale = source.convert("L")
        cropped = grayscale.crop(
            (valid_rect.x1, valid_rect.y1, valid_rect.x2, valid_rect.y2)
        )
        cropped.thumbnail((512, 512), Image.Resampling.BILINEAR)
        values = np.asarray(cropped, dtype=np.float32)

    if values.size < 64:
        return ImageModelProfile(ModelVariant.GENERAL, 0.0, 0.0)
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return ImageModelProfile(ModelVariant.LOW_CONTRAST, 0.0, 0.0)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    contrast = float(np.percentile(normalized, 90) - np.percentile(normalized, 10))
    horizontal = np.abs(np.diff(normalized, axis=1))
    vertical = np.abs(np.diff(normalized, axis=0))
    edge_density = float(
        ((horizontal > 0.08).mean() + (vertical > 0.08).mean()) / 2.0
    )

    if contrast < 0.20:
        variant = ModelVariant.LOW_CONTRAST
    elif edge_density >= 0.20:
        variant = ModelVariant.DENSE_PARTICLE
    elif edge_density >= 0.09:
        variant = ModelVariant.SMALL_PARTICLE
    elif edge_density <= 0.035:
        variant = ModelVariant.LARGE_PARTICLE
    else:
        variant = ModelVariant.GENERAL
    return ImageModelProfile(variant, contrast, edge_density)
