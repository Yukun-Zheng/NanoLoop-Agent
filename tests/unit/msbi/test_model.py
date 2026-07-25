from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.msbi.losses import MSBILoss  # noqa: E402
from app.msbi.model import (  # noqa: E402
    build_msbi_model,
    load_unet_small_anchor_state,
)
from scripts.models.export_unet_small_torchscript import UNet  # noqa: E402


def test_multi_head_shapes_gate_sum_and_loss_backward() -> None:
    model = build_msbi_model(
        {
            "encoder": "convnext_micro",
            "encoder_pretrained": False,
            "encoder_dims": [16, 32, 64, 128],
            "encoder_depths": [1, 1, 1, 1],
            "fpn_channels": 16,
            "fusion_mode": "gate",
            "enable_sdf": True,
        }
    )
    image = torch.rand(2, 1, 64, 64)
    outputs = model(image)

    assert set(outputs) == set(model.output_names)
    assert outputs["foreground_logits"].shape == (2, 1, 64, 64)
    assert outputs["gate_logits"].shape == (2, 2, 64, 64)
    gate = torch.softmax(outputs["gate_logits"], dim=1)
    assert torch.allclose(gate.sum(dim=1), torch.ones_like(gate[:, 0]), atol=1e-6)

    targets = {
        "foreground": torch.zeros(2, 1, 64, 64),
        "center": torch.zeros(2, 1, 64, 64),
        "boundary": torch.zeros(2, 1, 64, 64),
        "distance": torch.zeros(2, 1, 64, 64),
        "scale": torch.full((2, 1, 64, 64), -1, dtype=torch.long),
        "valid": torch.ones(2, 1, 64, 64),
        "supervised_valid": torch.ones(2, 1, 64, 64),
    }
    criterion = MSBILoss(
        {
            "foreground": 1.0,
            "center": 1.0,
            "boundary": 1.0,
            "sdf": 0.2,
            "gate": 0.1,
        }
    )
    loss, parts = criterion(outputs, targets)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mobilenet_student_preserves_multi_head_contract() -> None:
    model = build_msbi_model(
        {
            "encoder": "mobilenet_v3_small",
            "encoder_pretrained": False,
            "fpn_channels": 32,
            "fusion_mode": "gate",
            "enable_sdf": True,
        }
    )

    outputs = model(torch.rand(2, 1, 128, 128))

    assert set(outputs) == set(model.output_names)
    assert outputs["foreground_logits"].shape == (2, 1, 128, 128)
    assert outputs["gate_logits"].shape == (2, 2, 128, 128)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_500_000


def test_unet_anchor_starts_from_exact_verified_foreground() -> None:
    reference = UNet().eval()
    model = build_msbi_model(
        {
            "encoder": "unet_small_anchor",
            "expert_channels": 16,
            "enable_sdf": True,
            "freeze_encoder": True,
            "freeze_anchor_head": True,
        }
    ).eval()
    loaded = load_unet_small_anchor_state(model, reference.state_dict())
    image = torch.rand(1, 1, 64, 64)

    with torch.inference_mode():
        reference_logits = reference(image)
        outputs = model(image)

    assert len(loaded) == len(reference.state_dict())
    assert torch.equal(outputs["foreground_logits"], reference_logits)
    assert outputs["center_logits"].shape == reference_logits.shape
    assert outputs["gate_logits"].shape == (1, 2, 64, 64)


def test_efficient_unet_anchor_keeps_foreground_exact_and_exposes_anchor_method() -> None:
    reference = UNet().eval()
    model = build_msbi_model(
        {
            "encoder": "unet_small_efficient_anchor",
            "expert_channels": 8,
            "foreground_correction_limit": 0.0,
            "freeze_encoder": True,
            "freeze_anchor_head": True,
        }
    ).eval()
    load_unet_small_anchor_state(model, reference.state_dict())
    image = torch.rand(1, 1, 64, 64)

    with torch.inference_mode():
        reference_logits = reference(image)
        outputs = model(image)
        anchor_logits = model.forward_anchor(image)
        runtime_logits, runtime_gate = model.forward_runtime(image)

    assert torch.equal(outputs["foreground_logits"], reference_logits)
    assert torch.equal(anchor_logits, reference_logits)
    assert torch.equal(runtime_logits, reference_logits)
    assert runtime_gate.shape == (1, 2)
    assert outputs["boundary_logits"].shape == reference_logits.shape


def test_efficient_unet_runtime_uses_bounded_foreground_correction() -> None:
    reference = UNet().eval()
    model = build_msbi_model(
        {
            "encoder": "unet_small_efficient_anchor",
            "expert_channels": 8,
            "foreground_correction_limit": 0.25,
            "freeze_encoder": True,
            "freeze_anchor_head": True,
        }
    ).eval()
    load_unet_small_anchor_state(model, reference.state_dict())
    with torch.no_grad():
        assert model.foreground_correction.bias is not None
        model.foreground_correction.bias.fill_(2.0)
    image = torch.rand(1, 1, 64, 64)

    with torch.inference_mode():
        reference_logits = reference(image)
        outputs = model(image)
        runtime_logits, runtime_gate = model.forward_runtime(image)

    assert torch.allclose(runtime_logits, outputs["foreground_logits"], atol=1e-6)
    assert not torch.equal(runtime_logits, reference_logits)
    assert runtime_gate.shape == (1, 2)
    assert torch.allclose(runtime_gate.sum(dim=1), torch.ones(1), atol=1e-6)
