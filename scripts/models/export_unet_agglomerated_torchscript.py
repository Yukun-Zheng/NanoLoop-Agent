"""Export the frozen agglomerated U-Net state_dict as an external TorchScript asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EXPECTED_CHECKPOINT_SHA256 = (
    "e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21"
)
EXPECTED_KEY_COUNT = 63
PATCH_SIZE = 384


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Exact residual GroupNorm block used by the specialized training code."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.main(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1, bias=False
        )
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class AgglomeratedUNet(nn.Module):
    """Exact 63-key architecture used by the frozen external checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = ResidualBlock(1, 32)
        self.encoder2 = DownBlock(32, 64)
        self.encoder3 = DownBlock(64, 128)
        self.encoder4 = DownBlock(128, 256)
        self.context = nn.Sequential(
            ResidualBlock(256, 256, dilation=2),
            ResidualBlock(256, 256, dilation=4),
        )
        self.decoder3 = UpBlock(256, 128, 128)
        self.decoder2 = UpBlock(128, 64, 64)
        self.decoder1 = UpBlock(64, 32, 32)
        # The verified source definition and 63-key checkpoint include output.bias.
        # All residual, skip, and downsampling convolutions above use bias=False.
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.context(self.encoder4(x3))
        x = self.decoder3(x4, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        return self.output(x)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_paths(checkpoint_path: Path, output_path: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is not a file: {checkpoint}")
    if _is_within(output, _repository_root()):
        raise ValueError("output path must be outside the NanoLoop-Agent repository")
    if output.suffix.lower() != ".pt":
        raise ValueError("output path must use the .pt suffix")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")
    if output == checkpoint:
        raise ValueError("output path must differ from the checkpoint path")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint SHA-256 differs from the frozen agglomerated asset identity: "
            f"{checkpoint_sha256}"
        )
    return checkpoint, output


def _load_state_dict(checkpoint: Path) -> Mapping[str, Tensor]:
    payload: Any = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a plain OrderedDict/state_dict mapping")
    if not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in payload.items()
    ):
        raise TypeError("checkpoint state_dict must map string keys to tensors")
    return payload


def _diagnose_state_dict(
    model: nn.Module, checkpoint_state: Mapping[str, Tensor]
) -> dict[str, object]:
    expected_state = model.state_dict()
    if len(expected_state) != EXPECTED_KEY_COUNT:
        raise RuntimeError(
            "AgglomeratedUNet key contract changed: "
            f"expected={EXPECTED_KEY_COUNT}, observed={len(expected_state)}"
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
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "checkpoint does not match the frozen AgglomeratedUNet definition: "
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


def export(checkpoint_path: Path, output_path: Path) -> dict[str, object]:
    checkpoint, output = _validate_paths(checkpoint_path, output_path)
    model = AgglomeratedUNet().to("cpu")
    checkpoint_state = _load_state_dict(checkpoint)
    state_dict_diagnostic = _diagnose_state_dict(model, checkpoint_state)
    model.load_state_dict(checkpoint_state, strict=True)
    model.eval()

    example = torch.linspace(
        0.0,
        1.0,
        steps=PATCH_SIZE * PATCH_SIZE,
        dtype=torch.float32,
        device="cpu",
    ).reshape(1, 1, PATCH_SIZE, PATCH_SIZE)
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

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "output": str(output),
        "torchscript_sha256": _sha256(output),
        "architecture": "AgglomeratedUNet",
        "device": "cpu",
        "input": {"shape": list(example.shape), "dtype": str(example.dtype)},
        "state_dict": state_dict_diagnostic,
        "logits": _tensor_report(eager_logits, scripted_logits),
        "probability": _tensor_report(eager_probability, scripted_probability),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = export(args.checkpoint, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"export failed: {type(error).__name__}: {error}")
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
