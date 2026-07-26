from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from app.contracts.enums import (  # noqa: E402
    DevicePreference,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.inference import SegmentationRequest  # noqa: E402
from app.contracts.models import ModelMetadata  # noqa: E402
from app.inference.adapters.msbi import MSBIAdapter  # noqa: E402


class _SyntheticMSBI(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = values[:, :1].shape
        foreground = torch.full(shape, 4.0, device=values.device)
        center = torch.full(shape, -5.0, device=values.device)
        center[:, :, shape[-2] // 2, shape[-1] // 2] = 5.0
        boundary = torch.full(shape, -5.0, device=values.device)
        distance = torch.ones(shape, device=values.device) * 0.5
        gate = torch.cat((torch.ones(shape), -torch.ones(shape)), dim=1)
        return {
            "foreground_logits": foreground,
            "center_logits": center,
            "boundary_logits": boundary,
            "distance_field": distance,
            "small_logits": foreground,
            "large_logits": foreground,
            "gate_logits": gate,
        }


def _script_bytes() -> bytes:
    module = torch.jit.trace(
        _SyntheticMSBI(),
        torch.zeros(1, 1, 32, 32),
        strict=False,
    )
    buffer = BytesIO()
    torch.jit.save(module, buffer)
    return buffer.getvalue()


def test_msbi_adapter_fuses_all_heads_and_masks_invalid_bottom(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((64, 64), 120, dtype=np.uint8), mode="L").save(image_path)
    adapter = MSBIAdapter(
        metadata=ModelMetadata(
            model_id="msbi-test",
            family=ModelFamily.MSBI,
            variant=ModelVariant.DENSE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.5,
            preprocess_profile="test",
            postprocess_profile="test",
        ),
        weight_path=tmp_path / "model.pt",
        weight_bytes=_script_bytes(),
        config={
            "loader": "torchscript",
            "patch_size": [32, 32],
            "stride": [16, 16],
            "bottom_crop_px": 8,
            "normalization": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "output_names": list(_SyntheticMSBI().forward(torch.zeros(1, 1, 1, 1))),
            "decoder": {
                "foreground_threshold": 0.5,
                "center_threshold": 0.5,
                "center_nms_radius": 3,
                "boundary_threshold": 0.5,
                "min_area_px": 1,
                "connectivity": 2,
            },
        },
    )
    adapter.load("cpu")
    output = adapter.predict(
        SegmentationRequest(
            image_id="image",
            image_path=image_path,
            run_dir=tmp_path / "run",
            roi_mode=RoiMode.FULL_IMAGE,
            device=DevicePreference.CPU,
        )
    )
    labels = np.load(
        output.binary_mask_path.parent / "instance_labels.npy",
        allow_pickle=False,
    )
    gate_small = np.load(
        output.binary_mask_path.parent / "gate_small.npy",
        allow_pickle=False,
    )
    gate_large = np.load(
        output.binary_mask_path.parent / "gate_large.npy",
        allow_pickle=False,
    )

    assert output.instances_path is not None
    assert len(output.instances) >= 1
    assert np.all(labels[-8:] == 0)
    assert np.allclose(gate_small[:-8] + gate_large[:-8], 1.0, atol=1e-6)
    adapter.unload()


def test_msbi_adapter_treats_registered_dimensions_as_reference(tmp_path) -> None:
    image_path = tmp_path / "portrait.png"
    Image.fromarray(np.full((48, 40), 120, dtype=np.uint8), mode="L").save(image_path)
    adapter = MSBIAdapter(
        metadata=ModelMetadata(
            model_id="msbi-adaptive-test",
            family=ModelFamily.MSBI,
            variant=ModelVariant.DENSE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.5,
            preprocess_profile="test",
            postprocess_profile="test",
            expected_input_width=64,
            expected_input_height=64,
            inference_invalid_bottom_px=8,
        ),
        weight_path=tmp_path / "model.pt",
        weight_bytes=_script_bytes(),
        config={
            "loader": "torchscript",
            "patch_size": [32, 32],
            "stride": [16, 16],
            "bottom_crop_px": 8,
            "normalization": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "output_names": list(_SyntheticMSBI().forward(torch.zeros(1, 1, 1, 1))),
            "decoder": {
                "foreground_threshold": 0.5,
                "center_threshold": 0.5,
                "center_nms_radius": 3,
                "boundary_threshold": 0.5,
                "min_area_px": 1,
                "connectivity": 2,
            },
        },
    )
    adapter.load("cpu")

    output = adapter.predict(
        SegmentationRequest(
            image_id="image",
            image_path=image_path,
            run_dir=tmp_path / "adaptive-run",
            roi_mode=RoiMode.FULL_IMAGE,
            device=DevicePreference.CPU,
        )
    )

    assert (output.width, output.height) == (40, 48)
    assert output.warnings == [
        "input_dimensions_adapted:reference=64x64:observed=40x48"
    ]
    adapter.unload()


def test_msbi_adapter_batches_tiles_and_writes_compact_instances(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((64, 64), 120, dtype=np.uint8), mode="L").save(image_path)
    adapter = MSBIAdapter(
        metadata=ModelMetadata(
            model_id="msbi-compact-test",
            family=ModelFamily.MSBI,
            variant=ModelVariant.DENSE_PARTICLE,
            quality_tier=QualityTier.BALANCED,
            version="1",
            status=ModelStatus.READY,
            supports_box_prompt=False,
            default_threshold=0.5,
            preprocess_profile="test",
            postprocess_profile="test",
        ),
        weight_path=tmp_path / "model.pt",
        weight_bytes=_script_bytes(),
        config={
            "loader": "torchscript",
            "patch_size": [32, 32],
            "stride": [16, 16],
            "tile_batch_size": 3,
            "fusion_accumulator_dtype": "float32",
            "bottom_crop_px": 8,
            "normalization": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "save_raw_heads": False,
            "save_auxiliary_previews": False,
            "compact_instance_artifact": True,
            "save_adapter_instance_json": False,
            "output_names": list(_SyntheticMSBI().forward(torch.zeros(1, 1, 1, 1))),
            "decoder": {
                "foreground_threshold": 0.5,
                "center_threshold": 0.5,
                "center_nms_radius": 3,
                "boundary_threshold": 0.5,
                "min_area_px": 1,
                "connectivity": 2,
                "fallback_peak_source": "sdf",
            },
        },
    )
    adapter.load("cpu")

    output = adapter.predict(
        SegmentationRequest(
            image_id="image",
            image_path=image_path,
            run_dir=tmp_path / "run",
            roi_mode=RoiMode.FULL_IMAGE,
            device=DevicePreference.CPU,
        )
    )

    assert output.instances_path is not None
    with np.load(output.instances_path, allow_pickle=False) as archive:
        assert "masks" not in archive
        labels = np.asarray(archive["label_map"])
        assert list(archive["instance_ids"]) == list(range(1, int(labels.max()) + 1))
    assert np.all(labels[-8:] == 0)
    assert not (output.binary_mask_path.parent / "center_probability.npy").exists()
    assert not (output.binary_mask_path.parent / "instances.json").exists()
    assert output.auxiliary_paths == {}
    for instance in output.instances:
        mask = labels == instance.instance_index
        assert instance.area_px == int(mask.sum())
    adapter.unload()
