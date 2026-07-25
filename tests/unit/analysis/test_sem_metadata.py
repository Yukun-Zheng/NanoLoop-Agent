from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from app.analysis.sem_metadata import inspect_sem_image


def test_inspect_sem_image_extracts_zeiss_metadata_and_automatic_scale(
    tmp_path: Path,
) -> None:
    pixels = np.full((240, 320), 96, dtype=np.uint8)
    pixels[192:194, :] = 0
    pixels[194:, :] = 255
    pixels[206:211, 20:105] = 0
    metadata = "\r\n".join(
        (
            "AP_IMAGE_PIXEL_SIZE",
            "Image Pixel Size = 558.2 pm",
            "AP_ACTUALKV",
            "EHT = 3.00 kV",
            "AP_WD",
            "WD = 5.6 mm",
            "AP_MAG",
            "Mag = 100.00 K X",
            "AP_APERTURESIZE",
            "Aperture Size = 20.00 um",
            "DP_DETECTOR_CHANNEL",
            "Signal A = InLens",
            "AP_DATE",
            "Date: 22 Jan 2026",
            "AP_TIME",
            "Time: 21:48:36",
            "SV_SERIAL_NUMBER",
            "Serial No. = Sigma 300-8211011244",
        )
    )
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[34118] = metadata
    path = tmp_path / "zeiss-sem.tif"
    Image.fromarray(pixels, mode="L").save(path, tiffinfo=info)

    inspected = inspect_sem_image(path, width=320, height=240)

    assert inspected.detected_scale_nm_per_pixel == 0.5582
    assert inspected.analysis_roi.valid_rect.y2 == 192
    assert inspected.metadata is not None
    assert inspected.metadata.vendor == "ZEISS"
    assert inspected.metadata.instrument_model == "Sigma 300"
    assert inspected.metadata.detector == "InLens"
    assert inspected.metadata.accelerating_voltage_kv == 3.0
    assert inspected.metadata.working_distance_mm == 5.6
    assert inspected.metadata.magnification_x == 100_000
    assert inspected.metadata.aperture_size_um == 20
    assert inspected.metadata.acquired_at == "2026-01-22T21:48:36"
    assert inspected.metadata.footer_detected is True
    assert inspected.metadata.footer_style == "light"


def test_inspect_sem_image_without_footer_or_vendor_tags_keeps_full_frame(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(23)
    pixels = rng.integers(25, 230, size=(180, 260), dtype=np.uint8)
    path = tmp_path / "plain-sem.png"
    Image.fromarray(pixels, mode="L").save(path)

    inspected = inspect_sem_image(path, width=260, height=180)

    assert inspected.metadata is None
    assert inspected.detected_scale_nm_per_pixel is None
    assert inspected.analysis_roi.source == "none"
    assert inspected.analysis_roi.valid_rect.model_dump() == {
        "x1": 0,
        "y1": 0,
        "x2": 260,
        "y2": 180,
    }
