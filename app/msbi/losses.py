"""Configurable, masked multi-task objective for NanoLoop-MSBI."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weights = valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _focal_bce(logits: Tensor, target: Tensor, valid: Tensor, gamma: float = 2.0) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    probability_correct = probability * target + (1.0 - probability) * (1.0 - target)
    return _masked_mean(((1.0 - probability_correct) ** gamma) * bce, valid)


def _dice_loss(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    probability = torch.sigmoid(logits) * valid
    return _probability_dice_loss(probability, target, valid)


def _probability_dice_loss(
    probability: Tensor,
    target: Tensor,
    valid: Tensor,
) -> Tensor:
    probability = probability * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _tversky_loss(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    false_positive_weight: float = 0.7,
) -> Tensor:
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    false_negative_weight = 1.0 - false_positive_weight
    true_positive = (probability * target).sum(dim=(-2, -1))
    false_positive = (probability * (1.0 - target)).sum(dim=(-2, -1))
    false_negative = ((1.0 - probability) * target).sum(dim=(-2, -1))
    score = (true_positive + 1.0) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + 1.0
    )
    return (1.0 - score).mean()


class MSBILoss(nn.Module):
    def __init__(self, weights: Mapping[str, float]) -> None:
        super().__init__()
        self.weights = {name: float(value) for name, value in weights.items()}

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        teacher_small: Tensor | None = None,
        teacher_large: Tensor | None = None,
        teacher_valid: Tensor | None = None,
        consistency_outputs: Mapping[str, Tensor] | None = None,
        mentor_outputs: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        valid = targets["valid"].float()
        supervised_valid = targets.get("supervised_valid", valid).float()
        foreground = targets["foreground"].float()
        center = targets["center"].float()
        boundary = targets["boundary"].float()
        distance = targets["distance"].float()
        scale = targets["scale"].long()
        parts: dict[str, Tensor] = {}
        parts["foreground"] = _focal_bce(
            outputs["foreground_logits"], foreground, supervised_valid
        ) + _dice_loss(outputs["foreground_logits"], foreground, supervised_valid)
        edge_weight = torch.where(boundary > 0.05, 5.0, 1.0)
        parts["foreground_edge"] = _masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["foreground_logits"],
                foreground,
                reduction="none",
            )
            * edge_weight,
            supervised_valid,
        )
        parts["foreground_tversky"] = _tversky_loss(
            outputs["foreground_logits"],
            foreground,
            supervised_valid,
        )
        center_weight = torch.where(center > 0.05, 4.0, 1.0)
        parts["center"] = _masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["center_logits"], center, reduction="none"
            )
            * center_weight,
            supervised_valid,
        )
        parts["boundary"] = _focal_bce(
            outputs["boundary_logits"], boundary, supervised_valid
        ) + _dice_loss(outputs["boundary_logits"], boundary, supervised_valid)
        foreground_probability = torch.sigmoid(outputs["foreground_logits"])
        dilated = F.max_pool2d(
            foreground_probability,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        eroded = -F.max_pool2d(
            -foreground_probability,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        foreground_contour = (dilated - eroded).clamp(1e-5, 1.0 - 1e-5)
        contour_logits = torch.logit(foreground_contour)
        contour_weight = torch.where(boundary > 0.5, 4.0, 1.0)
        parts["contour"] = _masked_mean(
            F.binary_cross_entropy_with_logits(
                contour_logits,
                boundary,
                reduction="none",
            )
            * contour_weight,
            supervised_valid,
        ) + _probability_dice_loss(
            foreground_contour,
            boundary,
            supervised_valid,
        )
        sdf_valid = supervised_valid * foreground
        parts["sdf"] = _masked_mean(
            F.smooth_l1_loss(
                outputs["distance_field"], distance, reduction="none"
            ),
            sdf_valid,
        )
        gate_valid = (scale >= 0) & (supervised_valid[:, 0] > 0)
        if gate_valid.any():
            gate_loss = F.cross_entropy(
                outputs["gate_logits"],
                scale[:, 0],
                ignore_index=-1,
                reduction="none",
            )
            parts["gate"] = _masked_mean(gate_loss, gate_valid)
        else:
            gate_probability = torch.softmax(outputs["gate_logits"], dim=1)
            mean_weights = gate_probability.mean(dim=(0, 2, 3))
            parts["gate"] = ((mean_weights - 0.5) ** 2).sum()
        distill = outputs["foreground_logits"].new_zeros(())
        confidence_threshold = 0.9
        distill_valid = valid if teacher_valid is None else valid * teacher_valid
        if teacher_small is not None:
            confident = (teacher_small >= confidence_threshold) | (
                teacher_small <= 1.0 - confidence_threshold
            )
            distill = distill + _masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["small_logits"], teacher_small, reduction="none"
                ),
                distill_valid * confident,
            )
        if teacher_large is not None:
            confident = (teacher_large >= confidence_threshold) | (
                teacher_large <= 1.0 - confidence_threshold
            )
            distill = distill + _masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["large_logits"], teacher_large, reduction="none"
                ),
                distill_valid * confident,
            )
        parts["distill"] = distill
        if teacher_small is None:
            parts["teacher_foreground"] = outputs["foreground_logits"].new_zeros(())
        else:
            parts["teacher_foreground"] = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["foreground_logits"],
                    teacher_small,
                    reduction="none",
                ),
                distill_valid,
            )
        if mentor_outputs is None:
            for name in (
                "mentor_foreground",
                "mentor_center",
                "mentor_boundary",
                "mentor_sdf",
                "mentor_gate",
            ):
                parts[name] = outputs["foreground_logits"].new_zeros(())
        else:
            parts["mentor_foreground"] = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["foreground_logits"],
                    torch.sigmoid(mentor_outputs["foreground_logits"].detach()),
                    reduction="none",
                ),
                valid,
            )
            parts["mentor_center"] = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["center_logits"],
                    torch.sigmoid(mentor_outputs["center_logits"].detach()),
                    reduction="none",
                ),
                valid,
            )
            parts["mentor_boundary"] = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["boundary_logits"],
                    torch.sigmoid(mentor_outputs["boundary_logits"].detach()),
                    reduction="none",
                ),
                valid,
            )
            parts["mentor_sdf"] = _masked_mean(
                F.smooth_l1_loss(
                    outputs["distance_field"],
                    mentor_outputs["distance_field"].detach(),
                    reduction="none",
                ),
                valid,
            )
            mentor_gate = torch.softmax(
                mentor_outputs["gate_logits"].detach(),
                dim=1,
            )
            student_log_gate = torch.log_softmax(outputs["gate_logits"], dim=1)
            gate_divergence = -(mentor_gate * student_log_gate).sum(dim=1)
            parts["mentor_gate"] = _masked_mean(gate_divergence, valid[:, 0])
        if consistency_outputs is None:
            parts["consistency"] = outputs["foreground_logits"].new_zeros(())
        else:
            parts["consistency"] = _masked_mean(
                (
                    torch.sigmoid(outputs["foreground_logits"])
                    - torch.sigmoid(consistency_outputs["foreground_logits"])
                )
                ** 2,
                valid,
            )
        total = outputs["foreground_logits"].new_zeros(())
        for name, value in parts.items():
            total = total + self.weights.get(name, 0.0) * value
        if not torch.isfinite(total):
            raise FloatingPointError("MSBI loss became non-finite")
        return total, parts
