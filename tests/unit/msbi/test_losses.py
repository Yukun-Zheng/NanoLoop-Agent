from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.msbi.losses import MSBILoss  # noqa: E402


def test_contour_consistency_loss_is_finite_and_weighted() -> None:
    shape = (2, 1, 32, 32)
    outputs = {
        "foreground_logits": torch.randn(shape, requires_grad=True),
        "center_logits": torch.randn(shape, requires_grad=True),
        "boundary_logits": torch.randn(shape, requires_grad=True),
        "distance_field": torch.randn(shape, requires_grad=True).tanh(),
        "small_logits": torch.randn(shape, requires_grad=True),
        "large_logits": torch.randn(shape, requires_grad=True),
        "gate_logits": torch.randn((2, 2, 32, 32), requires_grad=True),
    }
    foreground = torch.zeros(shape)
    foreground[:, :, 8:24, 8:24] = 1.0
    boundary = torch.zeros(shape)
    boundary[:, :, 8, 8:24] = 1.0
    boundary[:, :, 23, 8:24] = 1.0
    boundary[:, :, 8:24, 8] = 1.0
    boundary[:, :, 8:24, 23] = 1.0
    targets = {
        "foreground": foreground,
        "center": torch.zeros(shape),
        "boundary": boundary,
        "distance": torch.zeros(shape),
        "scale": torch.zeros(shape, dtype=torch.long),
        "valid": torch.ones(shape),
        "supervised_valid": torch.ones(shape),
    }
    criterion = MSBILoss({"contour": 0.5})

    total, parts = criterion(outputs, targets)
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(parts["contour"])
    assert torch.allclose(total, 0.5 * parts["contour"])
    assert outputs["foreground_logits"].grad is not None


def test_mentor_and_teacher_foreground_distillation_are_finite() -> None:
    shape = (1, 1, 16, 16)
    outputs = {
        "foreground_logits": torch.randn(shape, requires_grad=True),
        "center_logits": torch.randn(shape, requires_grad=True),
        "boundary_logits": torch.randn(shape, requires_grad=True),
        "distance_field": torch.randn(shape, requires_grad=True).tanh(),
        "small_logits": torch.randn(shape, requires_grad=True),
        "large_logits": torch.randn(shape, requires_grad=True),
        "gate_logits": torch.randn((1, 2, 16, 16), requires_grad=True),
    }
    mentor = {name: value.detach().clone() for name, value in outputs.items()}
    targets = {
        "foreground": torch.zeros(shape),
        "center": torch.zeros(shape),
        "boundary": torch.zeros(shape),
        "distance": torch.zeros(shape),
        "scale": torch.zeros(shape, dtype=torch.long),
        "valid": torch.ones(shape),
        "supervised_valid": torch.ones(shape),
    }
    weights = {
        "teacher_foreground": 0.5,
        "mentor_foreground": 0.5,
        "mentor_center": 0.25,
        "mentor_boundary": 0.5,
        "mentor_sdf": 0.1,
        "mentor_gate": 0.05,
    }
    criterion = MSBILoss(weights)

    total, parts = criterion(
        outputs,
        targets,
        teacher_small=torch.rand(shape),
        teacher_valid=torch.ones(shape),
        mentor_outputs=mentor,
    )
    total.backward()

    assert torch.isfinite(total)
    assert all(torch.isfinite(parts[name]) for name in weights)
    assert outputs["foreground_logits"].grad is not None
