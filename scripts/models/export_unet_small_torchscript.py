"""Export a confirmed U-Net architecture profile as a CPU TorchScript artifact.

The checkpoint and exported artifact are intentionally external to this repository.  This script
only receives their locations through command-line arguments and refuses an output path inside the
repository.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SMALL_BATCHNORM = "small_batchnorm"
LARGE_GROUPNORM_OPTIMIZED = "large_groupnorm_optimized"


class DoubleConv(nn.Module):
    """Exact DoubleConv declaration from the small U-Net training and test notebooks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        self.up = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            if bilinear
            else nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        )
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff_y, diff_x = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """Exact one-channel, BatchNorm U-Net declaration from the notebooks."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1, self.down2 = Down(32, 64), Down(64, 128)
        self.down3, self.down4 = Down(128, 256), Down(256, 256)
        self.up1, self.up2 = Up(512, 128, bilinear), Up(256, 64, bilinear)
        self.up3, self.up4 = Up(128, 32, bilinear), Up(64, 32, bilinear)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class LargeDoubleConv(nn.Module):
    """Exact GroupNorm block from the optimized large U-Net notebooks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class LargeDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), LargeDoubleConv(in_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.mpconv(x)


class LargeUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = LargeDoubleConv(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff_y, diff_x = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class LargeOptimizedUNet(nn.Module):
    """Exact one-channel optimized large U-Net declaration from the notebooks."""

    def __init__(self) -> None:
        super().__init__()
        self.inc = LargeDoubleConv(1, 32)
        self.down1, self.down2 = LargeDown(32, 64), LargeDown(64, 128)
        self.down3, self.down4 = LargeDown(128, 256), LargeDown(256, 256)
        self.up1, self.up2 = LargeUp(512, 128), LargeUp(256, 64)
        self.up3, self.up4 = LargeUp(128, 32), LargeUp(64, 32)
        self.outc = nn.Conv2d(32, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


@dataclass(frozen=True, slots=True)
class ArchitectureProfile:
    name: str
    patch_size: int
    expected_key_count: int
    known_risks: tuple[str, ...] = ()

    def build_model(self) -> nn.Module:
        if self.name == SMALL_BATCHNORM:
            return UNet()
        if self.name == LARGE_GROUPNORM_OPTIMIZED:
            return LargeOptimizedUNet()
        raise ValueError(f"unknown architecture profile: {self.name}")


ARCHITECTURE_PROFILES = {
    SMALL_BATCHNORM: ArchitectureProfile(
        name=SMALL_BATCHNORM,
        patch_size=256,
        expected_key_count=128,
    ),
    LARGE_GROUPNORM_OPTIMIZED: ArchitectureProfile(
        name=LARGE_GROUPNORM_OPTIMIZED,
        patch_size=512,
        expected_key_count=56,
        known_risks=(
            "large training cropped 130 bottom pixels, while confirmed production inference "
            "crops 180 bottom pixels",
        ),
    ),
}


def _architecture_profile(name: str) -> ArchitectureProfile:
    try:
        return ARCHITECTURE_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown architecture profile: {name}") from error


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_paths(checkpoint_path: Path, output_path: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is not a file: {checkpoint}")
    if _is_within(output, _repository_root()):
        raise ValueError("output path must be outside the NanoLoop-Agent repository")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")
    if output == checkpoint:
        raise ValueError("output path must differ from the checkpoint path")
    return checkpoint, output


def _load_state_dict(checkpoint: Path) -> Mapping[str, Tensor]:
    payload: Any = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a plain state_dict mapping")
    if not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in payload.items()
    ):
        raise TypeError("checkpoint state_dict must map string keys to tensors")
    return payload


def _diagnose_state_dict(
    model: nn.Module,
    checkpoint_state: Mapping[str, Tensor],
    *,
    expected_key_count: int,
) -> dict[str, object]:
    """Fail before strict loading with the exact key and tensor-shape differences."""

    expected_state = model.state_dict()
    if len(expected_state) != expected_key_count:
        raise RuntimeError(
            "architecture profile state_dict key contract changed: "
            f"expected={expected_key_count}, observed={len(expected_state)}"
        )
    expected_keys = set(expected_state)
    checkpoint_keys = set(checkpoint_state)
    missing_keys = sorted(expected_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - expected_keys)
    shape_mismatches = {
        key: {
            "checkpoint": list(checkpoint_state[key].shape),
            "model": list(expected_state[key].shape),
        }
        for key in sorted(expected_keys & checkpoint_keys)
        if tuple(checkpoint_state[key].shape) != tuple(expected_state[key].shape)
    }
    diagnostic: dict[str, object] = {
        "checkpoint_key_count": len(checkpoint_keys),
        "model_key_count": len(expected_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "keys_match": not missing_keys and not unexpected_keys,
        "shapes_match": not shape_mismatches,
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "checkpoint state_dict does not match the confirmed UNet definition: "
            + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)
        )
    return diagnostic


def _tensor_report(eager: Tensor, scripted: Tensor) -> dict[str, object]:
    if tuple(eager.shape) != tuple(scripted.shape):
        raise RuntimeError(
            f"TorchScript output shape mismatch: eager={tuple(eager.shape)}, "
            f"scripted={tuple(scripted.shape)}"
        )
    eager_finite = bool(torch.isfinite(eager).all().item())
    scripted_finite = bool(torch.isfinite(scripted).all().item())
    if not eager_finite or not scripted_finite:
        raise RuntimeError("eager or TorchScript output contains non-finite values")
    return {
        "shape": list(eager.shape),
        "eager_finite": eager_finite,
        "torchscript_finite": scripted_finite,
        "max_abs_error": float(torch.max(torch.abs(eager - scripted)).item()),
    }


def export(
    checkpoint_path: Path,
    output_path: Path,
    architecture_profile: str = SMALL_BATCHNORM,
) -> dict[str, object]:
    """Export on CPU and return a JSON-serializable eager/TorchScript comparison report."""

    checkpoint, output = _validate_paths(checkpoint_path, output_path)
    profile = _architecture_profile(architecture_profile)
    model = profile.build_model().to("cpu")
    checkpoint_state = _load_state_dict(checkpoint)
    state_dict_diagnostic = _diagnose_state_dict(
        model,
        checkpoint_state,
        expected_key_count=profile.expected_key_count,
    )
    # Strict: model.load_state_dict(_load_state_dict(checkpoint), strict=True).
    model.load_state_dict(checkpoint_state, strict=True)
    model.eval()

    example = torch.linspace(
        0.0,
        1.0,
        steps=profile.patch_size * profile.patch_size,
        dtype=torch.float32,
        device="cpu",
    ).reshape(1, 1, profile.patch_size, profile.patch_size)
    with torch.inference_mode():
        eager_logits = model(example)
        eager_probability = torch.sigmoid(eager_logits)

    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu")
    reloaded.eval()
    with torch.inference_mode():
        scripted_logits = reloaded(example)
        scripted_probability = torch.sigmoid(scripted_logits)

    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "architecture_profile": profile.name,
        "device": "cpu",
        "input": {"shape": list(example.shape), "dtype": str(example.dtype)},
        "state_dict": state_dict_diagnostic,
        "logits": _tensor_report(eager_logits, scripted_logits),
        "probability": _tensor_report(eager_probability, scripted_probability),
    }
    if profile.known_risks:
        report["known_risks"] = list(profile.known_risks)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="External .pth state_dict")
    output_help = "External TorchScript output path"
    parser.add_argument("--output", required=True, type=Path, help=output_help)
    parser.add_argument(
        "--architecture-profile",
        choices=sorted(ARCHITECTURE_PROFILES),
        default=SMALL_BATCHNORM,
        help="Confirmed network definition; defaults to the original small U-Net profile",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = export(
            args.checkpoint,
            args.output,
            architecture_profile=args.architecture_profile,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"export failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
