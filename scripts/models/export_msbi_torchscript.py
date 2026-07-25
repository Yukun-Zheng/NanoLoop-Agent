#!/usr/bin/env python3
"""Strictly export an MSBI checkpoint and verify eager/TorchScript equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from app.msbi.model import build_msbi_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "app/msbi/model.py",
        root / "app/msbi/decoding.py",
        root / "app/inference/adapters/msbi.py",
        Path(__file__).resolve(),
    ]
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(relative)
        digest.update(content)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--foreground-correction-limit", type=float)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("checkpoint lacks the resolved MSBI model config")
    if args.foreground_correction_limit is not None:
        if args.foreground_correction_limit < 0:
            raise ValueError("--foreground-correction-limit must be non-negative")
        config = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
        config["model"]["foreground_correction_limit"] = args.foreground_correction_limit
    model = build_msbi_model(config["model"], for_export=True)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    patch_size = int(config["data"]["patch_size"])
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    example = torch.rand(1, 1, patch_size, patch_size, generator=generator)
    has_anchor_method = callable(getattr(model, "forward_anchor", None))
    has_runtime_method = callable(getattr(model, "forward_runtime", None))
    with torch.inference_mode():
        eager = model(example)
        if has_anchor_method:
            method_inputs = {
                "forward": example,
                "forward_anchor": example,
            }
            if has_runtime_method:
                method_inputs["forward_runtime"] = example
            traced = torch.jit.trace_module(
                model,
                method_inputs,
                strict=False,
            )
            preserved = ["forward_anchor"]
            if has_runtime_method:
                preserved.append("forward_runtime")
            traced = torch.jit.freeze(
                traced.eval(),
                preserved_attrs=preserved,
            )
        else:
            traced = torch.jit.trace(model, example, strict=False)
            traced = torch.jit.freeze(traced.eval())
        scripted_output = traced(example)
    if set(eager) != set(scripted_output):
        raise ValueError("TorchScript output keys differ from eager output")
    max_difference = max(
        float(torch.max(torch.abs(eager[name] - scripted_output[name])))
        for name in eager
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, args.output)
    loaded = torch.jit.load(args.output, map_location="cpu")
    loaded.eval()
    with torch.inference_mode():
        first = loaded(example)
        second = loaded(example)
    repeat_difference = max(
        float(torch.max(torch.abs(first[name] - second[name]))) for name in first
    )
    finite = all(torch.isfinite(value).all().item() for value in first.values())
    equivalence_atol = 1e-4
    equivalence_rtol = 1e-5
    equivalent = all(
        torch.allclose(
            eager[name],
            scripted_output[name],
            atol=equivalence_atol,
            rtol=equivalence_rtol,
        )
        for name in eager
    )
    anchor_difference: float | None = None
    anchor_equivalent: bool | None = None
    runtime_difference: float | None = None
    runtime_equivalent: bool | None = None
    if has_anchor_method:
        with torch.inference_mode():
            eager_anchor = model.forward_anchor(example)
            scripted_anchor = traced.forward_anchor(example)
        anchor_difference = float(
            torch.max(torch.abs(eager_anchor - scripted_anchor))
        )
        anchor_equivalent = bool(
            torch.allclose(
                eager_anchor,
                scripted_anchor,
                atol=equivalence_atol,
                rtol=equivalence_rtol,
            )
        )
    if has_runtime_method:
        with torch.inference_mode():
            eager_runtime = model.forward_runtime(example)
            scripted_runtime = traced.forward_runtime(example)
        runtime_difference = max(
            float(torch.max(torch.abs(eager_value - scripted_value)))
            for eager_value, scripted_value in zip(
                eager_runtime,
                scripted_runtime,
                strict=True,
            )
        )
        runtime_equivalent = all(
            torch.allclose(
                eager_value,
                scripted_value,
                atol=equivalence_atol,
                rtol=equivalence_rtol,
            )
            for eager_value, scripted_value in zip(
                eager_runtime,
                scripted_runtime,
                strict=True,
            )
        )
    shapes = {name: list(value.shape) for name, value in first.items()}
    manifest = {
        "schema_version": 1,
        "model_id": config.get("model_id", "msbi-instance-balanced-v1"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "torchscript_path": str(args.output.resolve()),
        "torchscript_sha256": _sha256(args.output),
        "source_sha256": _source_sha256(),
        "export_overrides": {
            "foreground_correction_limit": args.foreground_correction_limit,
        },
        "config_sha256": hashlib.sha256(
            yaml.safe_dump(config, sort_keys=True).encode()
        ).hexdigest(),
        "git_commit": checkpoint.get("git_commit"),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "eager_torchscript_max_abs_diff": max_difference,
        "eager_torchscript_allclose": equivalent,
        "anchor_method_exported": has_anchor_method,
        "anchor_eager_torchscript_allclose": anchor_equivalent,
        "anchor_eager_torchscript_max_abs_diff": anchor_difference,
        "runtime_method_exported": has_runtime_method,
        "runtime_eager_torchscript_allclose": runtime_equivalent,
        "runtime_eager_torchscript_max_abs_diff": runtime_difference,
        "equivalence_atol": equivalence_atol,
        "equivalence_rtol": equivalence_rtol,
        "deterministic_repeat_max_abs_diff": repeat_difference,
        "finite_outputs": finite,
        "output_shapes": shapes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": torch.__version__,
    }
    if (
        not finite
        or not equivalent
        or (has_anchor_method and not anchor_equivalent)
        or (has_runtime_method and not runtime_equivalent)
        or repeat_difference != 0.0
    ):
        raise ValueError(f"TorchScript verification failed: {manifest}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
