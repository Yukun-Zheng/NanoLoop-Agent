from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.analyses import ROIBox
from app.contracts.enums import ModelFamily, ModelStatus, ModelVariant, QualityTier, RoiMode
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelMetadata
from app.inference.adapters.unet import UNetAdapter


def _adapter(config: dict[str, object]) -> Any:
    adapter = object.__new__(UNetAdapter)
    adapter.config = config
    return adapter


def test_sliding_window_covers_edges_and_fuses_overlaps_without_nan() -> None:
    adapter = _adapter(
        {
            "input_size": [4, 4],
            "patch_size": [4, 4],
            "stride": [2, 2],
        }
    )
    calls: list[tuple[int, int]] = []

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        calls.append(tile.shape[:2])
        return np.asarray(tile[..., 0], dtype=np.float32) / 255.0

    adapter._predict_tile_probability = predict_tile
    image = np.zeros((7, 6, 3), dtype=np.uint8)
    image[..., 0] = np.arange(42, dtype=np.uint8).reshape(7, 6)

    probability = adapter._predict_probability(image)

    assert probability.shape == (7, 6)
    assert np.isfinite(probability).all()
    assert probability == pytest.approx(image[..., 0] / 255.0, abs=1e-7)
    assert calls == [(4, 4)] * 6


def test_small_unet_grayscale_preprocess_accepts_read_only_pil_array() -> None:
    adapter = _adapter(
        {
            "input_channels": 1,
            "input_size": [256, 256],
            "pixel_scale": 255.0,
        }
    )
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, 24:] = 255

    prepared = adapter._preprocess(image)

    assert prepared.shape == (1, 1, 256, 256)
    assert prepared.dtype == np.float32
    assert float(prepared.min()) == pytest.approx(0.0)
    assert float(prepared.max()) == pytest.approx(1.0)
    assert np.all((prepared >= 0.0) & (prepared <= 1.0))


def test_small_unet_reflect_padding_restores_original_cropped_extent() -> None:
    adapter = _adapter(
        {
            "patch_size": [4, 4],
            "stride": [2, 2],
            "tiling_padding": "reflect",
            "overlap_fusion": "uniform",
        }
    )
    image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    image = np.repeat(image[..., None], 3, axis=2)
    calls: list[np.ndarray] = []

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        calls.append(tile.copy())
        return np.asarray(tile[..., 0], dtype=np.float32)

    adapter._predict_tile_probability = predict_tile

    probability = adapter._predict_probability(image)

    padded = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode="reflect")
    assert probability == pytest.approx(image[..., 0], abs=1e-7)
    assert [call.shape for call in calls] == [(4, 4, 3), (4, 4, 3)]
    assert calls[0] == pytest.approx(padded[:, :4])
    assert calls[1] == pytest.approx(padded[:, 2:6])


def test_small_unet_uniform_fusion_uses_equal_overlap_weights() -> None:
    adapter = _adapter(
        {
            "patch_size": [4, 4],
            "stride": [2, 2],
            "overlap_fusion": "uniform",
        }
    )
    calls = 0

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        nonlocal calls
        value = float(calls)
        calls += 1
        return np.full(tile.shape[:2], value, dtype=np.float32)

    adapter._predict_tile_probability = predict_tile

    probability = adapter._predict_probability(np.zeros((4, 6, 3), dtype=np.uint8))

    assert calls == 2
    assert probability[:, :2] == pytest.approx(0.0)
    assert probability[:, 2:4] == pytest.approx(0.5)
    assert probability[:, 4:] == pytest.approx(1.0)


def test_small_unet_crops_bottom_bar_and_uses_strict_threshold(tmp_path: Path) -> None:
    metadata = ModelMetadata(
        model_id="unet-small-balanced-v1",
        family=ModelFamily.UNET,
        variant=ModelVariant.SMALL_PARTICLE,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.30,
        preprocess_profile="sem-gray-unit-crop-bottom-130-v1",
        postprocess_profile="semantic-mask-v1",
        inference_invalid_bottom_px=130,
        expected_input_width=5,
        expected_input_height=132,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={
            "bottom_crop_px": 130,
            "expected_image_size": [132, 5],
            "threshold_comparison": "gt",
        },
    )
    adapter._loaded = True

    def predict_probability(image: np.ndarray) -> np.ndarray:
        assert image.shape == (2, 5, 3)
        return np.asarray([[0.30] * 5, [0.31] * 5], dtype=np.float32)

    adapter._predict_probability = predict_probability
    image_bytes = BytesIO()
    Image.new("L", (5, 132), color=1).save(image_bytes, format="PNG")
    output = adapter.predict(
        SegmentationRequest(
            image_id="image-1",
            image_path=tmp_path / "pinned-image-bytes",
            image_bytes=image_bytes.getvalue(),
            run_dir=tmp_path / "run",
            roi_mode=RoiMode.FULL_IMAGE,
            threshold=0.30,
        )
    )

    with Image.open(output.binary_mask_path) as mask:
        observed = np.asarray(mask) > 0
    assert (output.width, output.height) == (5, 132)
    assert not observed[0].any()
    assert observed[1].all()
    assert not observed[2:].any()
    assert output.warnings == []


def test_agglomerated_percentile_preprocess_uses_full_image_p1_p99() -> None:
    adapter = _adapter(
        {
            "input_channels": 1,
            "normalization": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
        }
    )
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    image = np.repeat(gray[..., None], 3, axis=2)

    normalized = adapter._prepare_inference_image(image)
    prepared = adapter._preprocess(normalized)

    low, high = np.percentile(gray.astype(np.float32), [1.0, 99.0])
    expected = np.clip((gray.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    assert normalized == pytest.approx(expected)
    assert prepared.shape == (1, 1, 10, 10)
    assert prepared[0, 0] == pytest.approx(expected)


def test_agglomerated_percentile_preprocess_maps_constant_image_to_zero() -> None:
    adapter = _adapter(
        {
            "input_channels": 1,
            "normalization": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
        }
    )

    normalized = adapter._prepare_inference_image(
        np.full((20, 30, 3), 47, dtype=np.uint8)
    )

    assert normalized.shape == (20, 30)
    assert normalized.dtype == np.float32
    assert not normalized.any()


def test_agglomerated_384_288_tiling_uses_hann_floor_without_grid_padding() -> None:
    adapter = _adapter(
        {
            "patch_size": [384, 384],
            "stride": [288, 288],
            "tiling_padding": "reflect",
            "pad_to_tile_grid": False,
            "overlap_fusion": "hann",
            "fusion_weight_floor": 0.05,
        }
    )
    calls: list[tuple[int, int]] = []

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        calls.append(tile.shape[:2])
        return np.ones(tile.shape[:2], dtype=np.float32)

    adapter._predict_tile_probability = predict_tile
    probability = adapter._predict_probability(np.zeros((600, 700), dtype=np.float32))
    weight = adapter._fusion_weight((384, 384))

    assert probability.shape == (600, 700)
    assert probability == pytest.approx(1.0)
    assert calls == [(384, 384)] * 6
    assert float(weight.min()) == pytest.approx(0.05)
    assert float(weight.max()) <= 1.0


def test_agglomerated_small_image_reflect_pads_only_to_minimum_patch() -> None:
    adapter = _adapter(
        {
            "patch_size": [384, 384],
            "stride": [288, 288],
            "tiling_padding": "reflect",
            "pad_to_tile_grid": False,
            "overlap_fusion": "hann",
            "fusion_weight_floor": 0.05,
        }
    )
    calls: list[tuple[int, int]] = []

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        calls.append(tile.shape[:2])
        return np.ones(tile.shape[:2], dtype=np.float32)

    adapter._predict_tile_probability = predict_tile

    probability = adapter._predict_probability(np.zeros((300, 320), dtype=np.float32))

    assert probability.shape == (300, 320)
    assert probability == pytest.approx(1.0)
    assert calls == [(384, 384)]


def test_agglomerated_crops_130_bottom_rows_and_uses_gte_threshold(
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-agglomerated-specialized-v1",
        family=ModelFamily.UNET,
        variant=ModelVariant.DENSE_PARTICLE,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.25,
        default_min_area_px=1024,
        preprocess_profile="sem-gray-p1-p99-crop-bottom-130-v1",
        postprocess_profile="semantic-agglomerate-mask-v1",
        inference_invalid_bottom_px=130,
        expected_input_width=5,
        expected_input_height=132,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={
            "bottom_crop_px": 130,
            "expected_image_size": [132, 5],
            "threshold_comparison": "gte",
        },
    )
    adapter._loaded = True

    def predict_probability(image: np.ndarray) -> np.ndarray:
        assert image.shape == (2, 5, 3)
        return np.asarray([[0.50] * 5, [0.49] * 5], dtype=np.float32)

    adapter._predict_probability = predict_probability
    image_bytes = BytesIO()
    Image.new("L", (5, 132), color=1).save(image_bytes, format="PNG")

    output = adapter.predict(
        SegmentationRequest(
            image_id="image-1",
            image_path=tmp_path / "pinned-image-bytes",
            image_bytes=image_bytes.getvalue(),
            run_dir=tmp_path / "run",
            roi_mode=RoiMode.FULL_IMAGE,
            threshold=0.50,
        )
    )

    with Image.open(output.binary_mask_path) as mask:
        observed = np.asarray(mask) > 0
    probability = np.load(output.probability_path, allow_pickle=False)
    assert observed[0].all()
    assert not observed[1:].any()
    assert probability[0] == pytest.approx(0.50)
    assert probability[1] == pytest.approx(0.49)
    assert not probability[2:].any()
    assert output.warnings == []


def test_gte_zero_threshold_keeps_box_exterior_and_invalid_bottom_zero(
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-agglomerated-specialized-v1",
        family=ModelFamily.UNET,
        variant=ModelVariant.DENSE_PARTICLE,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.25,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
        inference_invalid_bottom_px=2,
        expected_input_width=5,
        expected_input_height=4,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={
            "bottom_crop_px": 2,
            "expected_image_size": [4, 5],
            "threshold_comparison": "gte",
        },
    )
    adapter._loaded = True
    adapter._predict_probability = lambda image: np.zeros(
        image.shape[:2], dtype=np.float32
    )
    image_bytes = BytesIO()
    Image.new("L", (5, 4), color=1).save(image_bytes, format="PNG")

    output = adapter.predict(
        SegmentationRequest(
            image_id="image-1",
            image_path=tmp_path / "pinned-image-bytes",
            image_bytes=image_bytes.getvalue(),
            run_dir=tmp_path / "run-boxes",
            roi_mode=RoiMode.BOXES,
            boxes=[ROIBox(x1=1, y1=0, x2=3, y2=2)],
            threshold=0.0,
        )
    )

    with Image.open(output.binary_mask_path) as mask:
        observed = np.asarray(mask) > 0
    expected = np.zeros((4, 5), dtype=bool)
    expected[0:2, 1:3] = True
    assert np.array_equal(observed, expected)


def test_unet_rejects_metadata_and_config_bottom_crop_drift(tmp_path: Path) -> None:
    metadata = ModelMetadata(
        model_id="unet-drift",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.0,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
        inference_invalid_bottom_px=10,
        expected_input_width=20,
        expected_input_height=20,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={"bottom_crop_px": 9, "expected_image_size": [20, 20]},
    )
    adapter._loaded = True
    image_bytes = BytesIO()
    Image.new("L", (20, 20), color=0).save(image_bytes, format="PNG")

    with pytest.raises(ValueError, match="must match registry"):
        adapter.predict(
            SegmentationRequest(
                image_id="image-1",
                image_path=tmp_path / "pinned-image-bytes",
                image_bytes=image_bytes.getvalue(),
                run_dir=tmp_path / "run",
                roi_mode=RoiMode.FULL_IMAGE,
            )
        )


def test_unet_rejects_input_dimensions_outside_frozen_crop_contract(
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-size-contract",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.5,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
        inference_invalid_bottom_px=10,
        expected_input_width=20,
        expected_input_height=20,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={"bottom_crop_px": 10, "expected_image_size": [20, 20]},
    )
    adapter._loaded = True
    image_bytes = BytesIO()
    Image.new("L", (20, 21), color=0).save(image_bytes, format="PNG")

    with pytest.raises(ValueError, match="expected 20x20, observed 20x21"):
        adapter.predict(
            SegmentationRequest(
                image_id="image-1",
                image_path=tmp_path / "pinned-image-bytes",
                image_bytes=image_bytes.getvalue(),
                run_dir=tmp_path / "run",
                roi_mode=RoiMode.FULL_IMAGE,
            )
        )


def test_unet_requires_expected_image_size_before_fixed_bottom_crop(
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-missing-size-contract",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.5,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
        inference_invalid_bottom_px=10,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={"bottom_crop_px": 10},
    )
    adapter._loaded = True
    image_bytes = BytesIO()
    Image.new("L", (20, 20), color=0).save(image_bytes, format="PNG")

    with pytest.raises(ValueError, match="expected_image_size is required"):
        adapter.predict(
            SegmentationRequest(
                image_id="image-1",
                image_path=tmp_path / "pinned-image-bytes",
                image_bytes=image_bytes.getvalue(),
                run_dir=tmp_path / "run",
                roi_mode=RoiMode.FULL_IMAGE,
            )
        )


def test_unet_rejects_too_small_cropped_extent_for_reflect_padding(
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-reflect-contract",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.5,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
        inference_invalid_bottom_px=2,
        expected_input_width=4,
        expected_input_height=3,
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={
            "bottom_crop_px": 2,
            "expected_image_size": [3, 4],
            "patch_size": [4, 4],
            "stride": [2, 2],
            "tiling_padding": "reflect",
        },
    )
    adapter._loaded = True
    image_bytes = BytesIO()
    Image.new("L", (4, 3), color=0).save(image_bytes, format="PNG")

    with pytest.raises(ValueError, match="at least two pixels"):
        adapter.predict(
            SegmentationRequest(
                image_id="image-1",
                image_path=tmp_path / "pinned-image-bytes",
                image_bytes=image_bytes.getvalue(),
                run_dir=tmp_path / "run",
                roi_mode=RoiMode.FULL_IMAGE,
            )
        )


@pytest.mark.parametrize(
    ("default_threshold", "expected_foreground", "error"),
    [(0.0, True, None), (None, False, "requires a frozen threshold")],
)
def test_unet_default_threshold_is_exact_and_missing_default_fails_closed(
    tmp_path: Path,
    default_threshold: float | None,
    expected_foreground: bool,
    error: str | None,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-threshold-contract",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=default_threshold,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={"bottom_crop_px": 0, "threshold_comparison": "gte"},
    )
    adapter._loaded = True
    adapter._predict_probability = lambda image: np.full(
        image.shape[:2], 0.25, dtype=np.float32
    )
    image_bytes = BytesIO()
    Image.new("L", (2, 2), color=0).save(image_bytes, format="PNG")
    request = SegmentationRequest(
        image_id="image-1",
        image_path=tmp_path / "pinned-image-bytes",
        image_bytes=image_bytes.getvalue(),
        run_dir=tmp_path / "run",
        roi_mode=RoiMode.FULL_IMAGE,
    )

    if error is not None:
        with pytest.raises(ValueError, match=error):
            adapter.predict(request)
        return

    output = adapter.predict(request)
    with Image.open(output.binary_mask_path) as mask:
        assert bool((np.asarray(mask) > 0).all()) is expected_foreground


def test_unet_adapter_emits_raw_semantic_mask_without_minimum_area_filtering(
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata(
        model_id="unet-raw-semantic-mask-contract",
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="1",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        default_threshold=0.5,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
    )
    adapter = UNetAdapter(
        metadata=metadata,
        weight_path=tmp_path / "external.pt",
        weight_bytes=b"test-only",
        config={"bottom_crop_px": 0, "threshold_comparison": "gt"},
    )
    adapter._loaded = True
    adapter._predict_probability = lambda image: np.asarray(
        [[0.9, 0.1], [0.1, 0.1]], dtype=np.float32
    )
    image_bytes = BytesIO()
    Image.new("L", (2, 2), color=0).save(image_bytes, format="PNG")
    request = SegmentationRequest(
        image_id="image-1",
        image_path=tmp_path / "pinned-image-bytes",
        image_bytes=image_bytes.getvalue(),
        run_dir=tmp_path / "run",
        roi_mode=RoiMode.FULL_IMAGE,
        min_area_px=512,
    )

    output = adapter.predict(request)

    with Image.open(output.binary_mask_path) as mask:
        assert np.array_equal(
            np.asarray(mask) > 0,
            np.asarray([[True, False], [False, False]]),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"patch_size": [0, 4]}, "positive integers"),
        ({"patch_size": [4, 4], "stride": [5, 2]}, "must not exceed"),
        ({"patch_size": [4]}, "positive integers"),
    ],
)
def test_invalid_tiling_configuration_fails_closed(
    config: dict[str, object],
    message: str,
) -> None:
    adapter = _adapter(config)

    with pytest.raises(ValueError, match=message):
        adapter._tiling_configuration()
