#!/usr/bin/env python3
"""Generate the public, non-scientific UI acceptance image.

The fixture is entirely synthetic and deterministic.  It exists only to exercise
the upload, model-runtime, artifact, review, query, and export paths without
redistributing a private SEM field of view.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

WIDTH = 2048
HEIGHT = 1536
SEED = 20260723
PARTICLES = (
    (250, 300, 38),
    (410, 420, 29),
    (575, 330, 48),
    (730, 510, 34),
    (890, 390, 56),
    (1080, 520, 42),
    (1260, 350, 31),
    (1430, 470, 52),
    (1640, 330, 37),
    (1815, 545, 45),
    (310, 730, 46),
    (505, 820, 33),
    (690, 700, 58),
    (900, 830, 36),
    (1110, 730, 50),
    (1305, 865, 34),
    (1510, 720, 62),
    (1760, 850, 39),
    (410, 1080, 55),
    (650, 1160, 32),
    (920, 1040, 47),
    (1190, 1160, 36),
    (1460, 1050, 50),
    (1740, 1180, 31),
)


def build_fixture() -> Image.Image:
    rng = random.Random(SEED)

    # A low-frequency grayscale texture keeps the generated image compact while
    # resembling an engineering microscope fixture at display scale.
    texture = Image.new("L", (256, 192))
    texture.putdata([rng.randint(48, 112) for _ in range(256 * 192)])
    image = texture.resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(radius=3.2))

    draw = ImageDraw.Draw(image, "L")
    draw.polygon(
        [(0, 180), (760, 40), (1260, 380), (2048, 250), (2048, 0), (0, 0)],
        fill=62,
    )
    draw.polygon(
        [(0, 1180), (610, 980), (1130, 1260), (2048, 1060), (2048, 1536), (0, 1536)],
        fill=78,
    )

    for index, (cx, cy, radius) in enumerate(PARTICLES):
        shadow = max(6, radius // 4)
        draw.ellipse(
            (
                cx - radius + shadow,
                cy - radius + shadow,
                cx + radius + shadow,
                cy + radius + shadow,
            ),
            fill=42,
        )
        value = 170 + (index % 5) * 12
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=value)
        highlight = max(5, radius // 4)
        draw.ellipse(
            (
                cx - radius // 3 - highlight,
                cy - radius // 3 - highlight,
                cx - radius // 3 + highlight,
                cy - radius // 3 + highlight,
            ),
            fill=min(245, value + 30),
        )

    return image.filter(ImageFilter.GaussianBlur(radius=1.1))


def build_corrected_mask() -> Image.Image:
    mask = Image.new("L", (WIDTH, HEIGHT), color=0)
    draw = ImageDraw.Draw(mask, "L")
    for cx, cy, radius in PARTICLES:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return mask


def save_with_digest(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"{output} {image.width}x{image.height} sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_data/acceptance/nanoloop_ui_acceptance_fixture.png"),
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=Path(
            "demo_data/acceptance/nanoloop_ui_acceptance_corrected_mask.png"
        ),
    )
    args = parser.parse_args()

    save_with_digest(build_fixture(), args.output)
    save_with_digest(build_corrected_mask(), args.mask_output)


if __name__ == "__main__":
    main()
