"""NanoLoop-MSBI multi-scale, boundary-aware instance network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class ResidualConv(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, dilation=dilation),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.activation(inputs + self.block(inputs)))


class ConvNeXtBlock(nn.Module):
    """Compact ConvNeXt block used by the no-download development encoder."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, expansion * channels)
        self.project = nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        values = self.depthwise(inputs).permute(0, 2, 3, 1)
        values = self.norm(values)
        values = self.project(F.gelu(self.expand(values)))
        values = values * self.gamma
        return cast(Tensor, residual + values.permute(0, 3, 1, 2))


class CompactConvNeXtEncoder(nn.Module):
    """Four-level ConvNeXt pyramid with strides 4, 8, 16, and 32."""

    def __init__(self, dims: Sequence[int], depths: Sequence[int]) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("ConvNeXt encoder requires four dimensions and depths")
        self.out_channels = tuple(int(value) for value in dims)
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*(ConvNeXtBlock(dims[index]) for _ in range(depths[index])))
                for index in range(4)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, dims[index]),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
                for index in range(3)
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = self.stem(inputs)
        outputs: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            values = stage(values)
            outputs.append(values)
            if index < len(self.downsamples):
                values = self.downsamples[index](values)
        return outputs[0], outputs[1], outputs[2], outputs[3]


class TorchvisionConvNeXtTinyEncoder(nn.Module):
    """ImageNet ConvNeXt-Tiny adapted to a single SEM channel by weight averaging."""

    out_channels = (96, 192, 384, 768)

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import (
            ConvNeXt_Tiny_Weights,
            convnext_tiny,
        )

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        original = backbone.features[0][0]
        replacement = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight.mean(dim=1, keepdim=True))
            if original.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(original.bias)
        backbone.features[0][0] = replacement
        self.features = backbone.features

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = inputs
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.features):
            values = layer(values)
            if index in {1, 3, 5, 7}:
                outputs.append(values)
        return outputs[0], outputs[1], outputs[2], outputs[3]


class TorchvisionMobileNetV3SmallEncoder(nn.Module):
    """ImageNet MobileNetV3-Small pyramid adapted to one-channel SEM input."""

    out_channels = (16, 24, 48, 96)
    output_indices = (1, 3, 8, 11)

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        original = backbone.features[0][0]
        replacement = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight.mean(dim=1, keepdim=True))
            if original.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(original.bias)
        backbone.features[0][0] = replacement
        self.features = backbone.features

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = inputs
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.features):
            values = layer(values)
            if index in self.output_indices:
                outputs.append(values)
        return outputs[0], outputs[1], outputs[2], outputs[3]


class SmallUNetDoubleConv(nn.Module):
    """Exact convolution block used by the verified Small U-Net baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.conv(inputs))


class SmallUNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            SmallUNetDoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.mpconv(inputs))


class SmallUNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        self.conv = SmallUNetDoubleConv(in_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        values = self.up(inputs)
        difference_y = skip.size(2) - values.size(2)
        difference_x = skip.size(3) - values.size(3)
        values = F.pad(
            values,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )
        return cast(Tensor, self.conv(torch.cat((skip, values), dim=1)))


class SmallUNetFeatureBackbone(nn.Module):
    """Verified Small U-Net encoder/decoder ending at its 32-channel feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.inc = SmallUNetDoubleConv(1, 32)
        self.down1 = SmallUNetDown(32, 64)
        self.down2 = SmallUNetDown(64, 128)
        self.down3 = SmallUNetDown(128, 256)
        self.down4 = SmallUNetDown(256, 256)
        self.up1 = SmallUNetUp(512, 128)
        self.up2 = SmallUNetUp(256, 64)
        self.up3 = SmallUNetUp(128, 32)
        self.up4 = SmallUNetUp(64, 32)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        first = self.inc(inputs)
        second = self.down1(first)
        third = self.down2(second)
        fourth = self.down3(third)
        fifth = self.down4(fourth)
        values = self.up1(fifth, fourth)
        values = self.up2(values, third)
        values = self.up3(values, second)
        values = self.up4(values, first)
        return values, fourth


class BaselineAnchoredMSBI(nn.Module):
    """Small U-Net anchor plus trainable multi-scale boundary/instance residuals."""

    output_names = (
        "foreground_logits",
        "center_logits",
        "boundary_logits",
        "distance_field",
        "small_logits",
        "large_logits",
        "gate_logits",
    )

    def __init__(
        self,
        *,
        expert_channels: int = 32,
        enable_sdf: bool = True,
        freeze_encoder: bool = True,
        freeze_anchor_head: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SmallUNetFeatureBackbone()
        self.outc = nn.Conv2d(32, 1, 1)
        self.small_expert = nn.Sequential(
            ConvNormAct(32, expert_channels),
            ResidualConv(expert_channels),
        )
        self.large_project = ConvNormAct(256, expert_channels, kernel_size=1)
        self.large_expert = nn.Sequential(
            ConvNormAct(32 + expert_channels, expert_channels),
            ResidualConv(expert_channels, dilation=2),
        )
        self.gate = nn.Sequential(
            ConvNormAct(2 * expert_channels, expert_channels),
            nn.Conv2d(expert_channels, 2, 1),
        )
        self.foreground_correction = nn.Conv2d(expert_channels, 1, 1)
        self.small_correction = nn.Conv2d(expert_channels, 1, 1)
        self.large_correction = nn.Conv2d(expert_channels, 1, 1)
        self.center_head = nn.Conv2d(expert_channels, 1, 1)
        self.boundary_head = nn.Conv2d(expert_channels, 1, 1)
        self.distance_head = nn.Conv2d(expert_channels, 1, 1)
        self.enable_sdf = enable_sdf
        self.freeze_encoder = freeze_encoder
        self.freeze_anchor_head = freeze_anchor_head
        for layer in (
            self.foreground_correction,
            self.small_correction,
            self.large_correction,
        ):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        gate_output = self.gate[-1]
        if not isinstance(gate_output, nn.Conv2d):
            raise TypeError("anchored MSBI gate output must be Conv2d")
        nn.init.zeros_(gate_output.weight)
        if gate_output.bias is not None:
            nn.init.zeros_(gate_output.bias)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        if freeze_anchor_head:
            for parameter in self.outc.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> BaselineAnchoredMSBI:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        if self.freeze_anchor_head:
            self.outc.eval()
        return self

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        features, coarse = self.encoder(inputs)
        anchor_logits = self.outc(features)
        small = self.small_expert(features)
        context = F.interpolate(
            self.large_project(coarse),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        large = self.large_expert(torch.cat((features, context), dim=1))
        gate_logits = self.gate(torch.cat((small, large), dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        fused = gate[:, 0:1] * small + gate[:, 1:2] * large
        distance = torch.tanh(self.distance_head(fused))
        if not self.enable_sdf:
            distance = torch.zeros_like(distance)
        return {
            "foreground_logits": anchor_logits + self.foreground_correction(fused),
            "center_logits": self.center_head(fused),
            "boundary_logits": self.boundary_head(fused),
            "distance_field": distance,
            "small_logits": anchor_logits + self.small_correction(small),
            "large_logits": anchor_logits + self.large_correction(large),
            "gate_logits": gate_logits,
        }


class ConvBatchNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class EfficientAnchoredMSBI(nn.Module):
    """Exact semantic anchor with low-resolution trainable instance heads."""

    output_names = BaselineAnchoredMSBI.output_names

    def __init__(
        self,
        *,
        expert_channels: int = 16,
        enable_sdf: bool = True,
        freeze_encoder: bool = True,
        freeze_anchor_head: bool = True,
        foreground_correction_limit: float = 0.0,
        expert_correction_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if foreground_correction_limit < 0 or expert_correction_limit < 0:
            raise ValueError("correction limits must be non-negative")
        self.encoder = SmallUNetFeatureBackbone()
        self.outc = nn.Conv2d(32, 1, 1)
        self.small_expert = nn.Sequential(
            ConvBatchNormAct(32, expert_channels),
            ConvBatchNormAct(expert_channels, expert_channels),
        )
        self.large_expert = nn.Sequential(
            ConvBatchNormAct(256, expert_channels, kernel_size=1),
            ConvBatchNormAct(expert_channels, expert_channels),
        )
        self.gate = nn.Sequential(
            ConvBatchNormAct(2 * expert_channels, expert_channels),
            nn.Conv2d(expert_channels, 2, 1),
        )
        self.foreground_correction = nn.Conv2d(expert_channels, 1, 1)
        self.small_correction = nn.Conv2d(expert_channels, 1, 1)
        self.large_correction = nn.Conv2d(expert_channels, 1, 1)
        self.center_head = nn.Conv2d(expert_channels, 1, 1)
        self.boundary_head = nn.Conv2d(expert_channels, 1, 1)
        self.distance_head = nn.Conv2d(expert_channels, 1, 1)
        self.enable_sdf = enable_sdf
        self.freeze_encoder = freeze_encoder
        self.freeze_anchor_head = freeze_anchor_head
        self.foreground_correction_limit = float(foreground_correction_limit)
        self.expert_correction_limit = float(expert_correction_limit)
        for layer in (
            self.foreground_correction,
            self.small_correction,
            self.large_correction,
        ):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        gate_output = self.gate[-1]
        if not isinstance(gate_output, nn.Conv2d):
            raise TypeError("efficient anchored MSBI gate output must be Conv2d")
        nn.init.zeros_(gate_output.weight)
        if gate_output.bias is not None:
            nn.init.zeros_(gate_output.bias)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        if freeze_anchor_head:
            for parameter in self.outc.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> EfficientAnchoredMSBI:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        if self.freeze_anchor_head:
            self.outc.eval()
        return self

    def forward_anchor(self, inputs: Tensor) -> Tensor:
        features, _coarse = self.encoder(inputs)
        return cast(Tensor, self.outc(features))

    def forward_runtime(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features, coarse = self.encoder(inputs)
        anchor_logits = self.outc(features)
        low_resolution = F.avg_pool2d(features, kernel_size=4, stride=4)
        small = self.small_expert(low_resolution)
        large = F.interpolate(
            self.large_expert(coarse),
            size=small.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = torch.softmax(
            self.gate(torch.cat((small, large), dim=1)),
            dim=1,
        )
        if self.foreground_correction_limit == 0.0:
            foreground_logits = anchor_logits
        else:
            fused = gate[:, 0:1] * small + gate[:, 1:2] * large
            foreground_delta = self.foreground_correction_limit * torch.tanh(
                F.interpolate(
                    self.foreground_correction(fused),
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            foreground_logits = anchor_logits + foreground_delta
        return foreground_logits, gate.mean(dim=(2, 3))

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        output_size = inputs.shape[-2:]
        features, coarse = self.encoder(inputs)
        anchor_logits = self.outc(features)
        low_resolution = F.avg_pool2d(features, kernel_size=4, stride=4)
        small = self.small_expert(low_resolution)
        large = F.interpolate(
            self.large_expert(coarse),
            size=small.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate_logits_low = self.gate(torch.cat((small, large), dim=1))
        gate = torch.softmax(gate_logits_low, dim=1)
        fused = gate[:, 0:1] * small + gate[:, 1:2] * large

        def upsample(values: Tensor) -> Tensor:
            return F.interpolate(
                values,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        foreground_delta = self.foreground_correction_limit * torch.tanh(
            upsample(self.foreground_correction(fused))
        )
        small_delta = self.expert_correction_limit * torch.tanh(
            upsample(self.small_correction(small))
        )
        large_delta = self.expert_correction_limit * torch.tanh(
            upsample(self.large_correction(large))
        )
        distance = torch.tanh(upsample(self.distance_head(fused)))
        if not self.enable_sdf:
            distance = torch.zeros_like(distance)
        return {
            "foreground_logits": anchor_logits + foreground_delta,
            "center_logits": upsample(self.center_head(fused)),
            "boundary_logits": upsample(self.boundary_head(fused)),
            "distance_field": distance,
            "small_logits": anchor_logits + small_delta,
            "large_logits": anchor_logits + large_delta,
            "gate_logits": upsample(gate_logits_low),
        }


def load_unet_small_anchor_state(
    model: nn.Module,
    state: Mapping[str, Tensor],
) -> tuple[str, ...]:
    """Strictly map every verified Small U-Net tensor into the anchored MSBI."""

    if not isinstance(model, BaselineAnchoredMSBI | EfficientAnchoredMSBI):
        raise ValueError("Small U-Net anchor state requires unet_small_anchor")
    current = model.state_dict()
    mapped: dict[str, Tensor] = {}
    for name, value in state.items():
        target = name if name.startswith("outc.") else f"encoder.{name}"
        expected = current.get(target)
        if expected is None:
            raise ValueError(f"unexpected Small U-Net anchor key: {name}")
        if expected.shape != value.shape:
            raise ValueError(
                f"Small U-Net anchor shape mismatch for {name}: "
                f"expected {tuple(expected.shape)}, observed {tuple(value.shape)}"
            )
        mapped[target] = value
    required = {
        name
        for name in current
        if name.startswith("encoder.") or name.startswith("outc.")
    }
    missing = sorted(required - set(mapped))
    if missing:
        raise ValueError(f"Small U-Net anchor is missing keys: {missing}")
    incompatible = model.load_state_dict(mapped, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Small U-Net anchor produced unexpected keys: {incompatible.unexpected_keys}"
        )
    return tuple(sorted(mapped))


class FeaturePyramid(nn.Module):
    def __init__(self, input_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, output_channels, 1) for channels in input_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvNormAct(output_channels, output_channels) for _ in input_channels]
        )

    def forward(
        self,
        features: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        laterals = [layer(value) for layer, value in zip(self.lateral, features, strict=True)]
        for index in range(2, -1, -1):
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        outputs = [
            layer(value) for layer, value in zip(self.smooth, laterals, strict=True)
        ]
        return outputs[0], outputs[1], outputs[2], outputs[3]


class SmallParticleExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(2 * channels, channels)
        self.refine = nn.Sequential(ResidualConv(channels), ResidualConv(channels))

    def forward(self, p2: Tensor, p3: Tensor) -> Tensor:
        p3_up = F.interpolate(
            p3,
            size=p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return cast(Tensor, self.refine(self.project(torch.cat((p2, p3_up), dim=1))))


class LargeAgglomerateExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(3 * channels, channels)
        self.context = nn.ModuleList(
            [
                ConvNormAct(channels, channels, dilation=1),
                ConvNormAct(channels, channels, dilation=2),
                ConvNormAct(channels, channels, dilation=4),
            ]
        )
        self.fuse = ConvNormAct(3 * channels, channels, kernel_size=1)

    def forward(self, p3: Tensor, p4: Tensor, p5: Tensor, output_size: list[int]) -> Tensor:
        p4_up = F.interpolate(
            p4,
            size=p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p5_up = F.interpolate(
            p5,
            size=p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        values = self.project(torch.cat((p3, p4_up, p5_up), dim=1))
        values = self.fuse(torch.cat([branch(values) for branch in self.context], dim=1))
        return F.interpolate(
            values,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class NanoLoopMSBI(nn.Module):
    """ConvNeXt-FPN dual-expert network with interpretable pixel-wise gating."""

    output_names = (
        "foreground_logits",
        "center_logits",
        "boundary_logits",
        "distance_field",
        "small_logits",
        "large_logits",
        "gate_logits",
    )

    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_channels: Sequence[int],
        fpn_channels: int = 96,
        enable_sdf: bool = True,
        freeze_encoder: bool = False,
        fusion_mode: str = "gate",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"single", "mean", "gate"}:
            raise ValueError("fusion_mode must be single, mean, or gate")
        self.encoder = encoder
        self.fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.small_expert = SmallParticleExpert(fpn_channels)
        self.large_expert = LargeAgglomerateExpert(fpn_channels)
        self.gate = nn.Sequential(
            ConvNormAct(3 * fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, 2, 1),
        )
        self.foreground_head = nn.Conv2d(fpn_channels, 1, 1)
        self.center_head = nn.Conv2d(fpn_channels, 1, 1)
        self.boundary_head = nn.Conv2d(fpn_channels, 1, 1)
        self.distance_head = nn.Conv2d(fpn_channels, 1, 1)
        self.small_head = nn.Conv2d(fpn_channels, 1, 1)
        self.large_head = nn.Conv2d(fpn_channels, 1, 1)
        self.enable_sdf = enable_sdf
        self.freeze_encoder = freeze_encoder
        self.fusion_mode = fusion_mode
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> NanoLoopMSBI:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        input_size = inputs.shape[-2:]
        features = self.encoder(inputs)
        p2, p3, p4, p5 = self.fpn(features)
        small = self.small_expert(p2, p3)
        large = self.large_expert(p3, p4, p5, [p2.shape[-2], p2.shape[-1]])
        gate_logits_low = self.gate(torch.cat((small, large, p2), dim=1))
        gate_weights = torch.softmax(gate_logits_low, dim=1)
        if self.fusion_mode == "single":
            fused = p2
        elif self.fusion_mode == "mean":
            fused = 0.5 * (small + large)
        else:
            fused = gate_weights[:, 0:1] * small + gate_weights[:, 1:2] * large

        def upsample(values: Tensor) -> Tensor:
            return F.interpolate(
                values,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        distance = torch.tanh(self.distance_head(fused))
        if not self.enable_sdf:
            distance = torch.zeros_like(distance)
        return {
            "foreground_logits": upsample(self.foreground_head(fused)),
            "center_logits": upsample(self.center_head(fused)),
            "boundary_logits": upsample(self.boundary_head(fused)),
            "distance_field": upsample(distance),
            "small_logits": upsample(self.small_head(small)),
            "large_logits": upsample(self.large_head(large)),
            "gate_logits": upsample(gate_logits_low),
        }


def build_msbi_model(config: Mapping[str, Any], *, for_export: bool = False) -> nn.Module:
    """Build the exact encoder declared in a frozen YAML configuration."""

    encoder_name = str(config.get("encoder", "convnext_tiny"))
    pretrained = bool(config.get("encoder_pretrained", True)) and not for_export
    encoder: nn.Module
    channels: Sequence[int]
    if encoder_name == "unet_small_anchor":
        return BaselineAnchoredMSBI(
            expert_channels=int(config.get("expert_channels", 32)),
            enable_sdf=bool(config.get("enable_sdf", True)),
            freeze_encoder=bool(config.get("freeze_encoder", True)),
            freeze_anchor_head=bool(config.get("freeze_anchor_head", True)),
        )
    if encoder_name == "unet_small_efficient_anchor":
        return EfficientAnchoredMSBI(
            expert_channels=int(config.get("expert_channels", 16)),
            enable_sdf=bool(config.get("enable_sdf", True)),
            freeze_encoder=bool(config.get("freeze_encoder", True)),
            freeze_anchor_head=bool(config.get("freeze_anchor_head", True)),
            foreground_correction_limit=float(
                config.get("foreground_correction_limit", 0.0)
            ),
            expert_correction_limit=float(
                config.get("expert_correction_limit", 0.5)
            ),
        )
    if encoder_name == "convnext_tiny":
        encoder = TorchvisionConvNeXtTinyEncoder(pretrained=pretrained)
        channels = TorchvisionConvNeXtTinyEncoder.out_channels
    elif encoder_name == "mobilenet_v3_small":
        encoder = TorchvisionMobileNetV3SmallEncoder(pretrained=pretrained)
        channels = TorchvisionMobileNetV3SmallEncoder.out_channels
    elif encoder_name == "convnext_micro":
        dimensions = tuple(int(value) for value in config.get("encoder_dims", [32, 64, 128, 256]))
        depths = tuple(int(value) for value in config.get("encoder_depths", [2, 2, 3, 2]))
        encoder = CompactConvNeXtEncoder(dimensions, depths)
        channels = dimensions
    else:
        raise ValueError(f"unsupported MSBI encoder: {encoder_name}")
    return NanoLoopMSBI(
        encoder=encoder,
        encoder_channels=channels,
        fpn_channels=int(config.get("fpn_channels", 96)),
        enable_sdf=bool(config.get("enable_sdf", True)),
        freeze_encoder=bool(config.get("freeze_encoder", False)),
        fusion_mode=str(config.get("fusion_mode", "gate")),
    )
