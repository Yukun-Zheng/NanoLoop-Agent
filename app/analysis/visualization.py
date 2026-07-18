"""Deterministic overlays for scientific review; never used to compute metrics."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.analysis.postprocessing import NormalizedInstance
from app.core.errors import InvalidImageError

_PALETTE = (
    (230, 57, 70),
    (29, 78, 216),
    (42, 157, 80),
    (245, 166, 35),
    (145, 70, 255),
    (0, 145, 150),
)


def write_review_visualizations(
    *,
    image_path: Path,
    image_bytes: bytes | None = None,
    binary_mask: np.ndarray,
    instances: list[NormalizedInstance],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write mask overlay and numbered-instance preview at original resolution."""

    source_value = BytesIO(image_bytes) if image_bytes is not None else image_path
    with Image.open(source_value) as source:
        original = source.convert("RGB")
    rgb = np.asarray(original, dtype=np.uint8).copy()
    if binary_mask.shape != rgb.shape[:2]:
        raise InvalidImageError(
            details={
                "reason": "visualization_shape_mismatch",
                "mask_shape": binary_mask.shape,
                "image_shape": rgb.shape[:2],
            }
        )
    mask = np.asarray(binary_mask, dtype=bool)
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    overlay[mask] = (0.55 * rgb[mask] + 0.45 * red[mask]).astype(np.uint8)

    labeled = rgb.copy()
    for instance in instances:
        blend_color = np.asarray(_PALETTE[(instance.instance_index - 1) % len(_PALETTE)])
        labeled[instance.mask] = (
            0.45 * labeled[instance.mask] + 0.55 * blend_color
        ).astype(np.uint8)
    labeled_image = Image.fromarray(labeled)
    draw = ImageDraw.Draw(labeled_image)
    for instance in instances:
        x1, y1, x2, y2 = instance.bbox
        outline_color = _PALETTE[(instance.instance_index - 1) % len(_PALETTE)]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=outline_color, width=1)
        draw.text((x1 + 2, y1 + 1), str(instance.instance_index), fill=(255, 255, 255))

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay.png"
    labeled_path = output_dir / "labeled_particles.png"
    Image.fromarray(overlay).save(overlay_path, format="PNG")
    labeled_image.save(labeled_path, format="PNG")
    return overlay_path, labeled_path
