#!/usr/bin/env python3
"""Train NanoLoop-MSBI with resumable, provenance-complete local runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from app.msbi.data import MSBIPatchDataset, read_manifest
from app.msbi.losses import MSBILoss
from app.msbi.model import build_msbi_model, load_unet_small_anchor_state


class ExponentialMovingAverage:
    """EMA over trainable parameters without duplicating a frozen encoder."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach(), 1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        parameters = dict(model.named_parameters())
        backup = {
            name: parameters[name].detach().clone()
            for name in self.shadow
        }
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    parameters[name].copy_(average)
            yield
        finally:
            with torch.no_grad():
                for name, values in backup.items():
                    parameters[name].copy_(values)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay differs from the resumed run")
        supplied = state["shadow"]
        if set(supplied) != set(self.shadow):
            raise ValueError("EMA parameter names differ from the resumed run")
        for name, values in supplied.items():
            self.shadow[name].copy_(values)


def _expand(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(expanded).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _command_output(*args: str) -> str:
    result = subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _pixel_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    predicted = (torch.sigmoid(logits) >= 0.5) & (valid > 0)
    truth = (target > 0.5) & (valid > 0)
    intersection = torch.count_nonzero(predicted & truth).item()
    denominator = torch.count_nonzero(predicted).item() + torch.count_nonzero(truth).item()
    return float((2 * intersection + 1) / (denominator + 1))


def _config_for_smoke(config: dict[str, Any]) -> dict[str, Any]:
    cloned = json.loads(json.dumps(config))
    cloned["model"].update(
        {
            "encoder": "convnext_micro",
            "encoder_pretrained": False,
            "encoder_dims": [16, 32, 64, 128],
            "encoder_depths": [1, 1, 1, 1],
            "fpn_channels": 16,
            "freeze_encoder": False,
        }
    )
    cloned["data"].update(
        {
            "patch_size": min(128, int(cloned["data"]["patch_size"])),
            "train_samples_per_epoch": 8,
            "validation_samples_per_epoch": 4,
        }
    )
    cloned["training"].update(
        {
            "epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "warmup_epochs": 1,
            "early_stopping_patience": 0,
        }
    )
    return cloned


def _build_loaders(config: dict[str, Any]) -> tuple[MSBIPatchDataset, DataLoader, DataLoader]:
    data = config["data"]
    manifest_root = _expand(str(data["manifest_root"]))
    train_records = read_manifest(manifest_root / "train.jsonl")
    validation_records = read_manifest(manifest_root / "validation.jsonl")
    patch_size = int(data["patch_size"])
    seed = int(config["training"]["seed"])
    train_dataset = MSBIPatchDataset(
        train_records,
        patch_size=patch_size,
        samples_per_epoch=int(data["train_samples_per_epoch"]),
        seed=seed,
        augment=True,
        density_sampling_probability=float(data.get("density_sampling_probability", 0.65)),
        preload_records=bool(data.get("preload_records", False)),
        morphology_balanced_sampling=bool(
            data.get("morphology_balanced_sampling", False)
        ),
        normalization=str(data.get("normalization", "percentile")),
    )
    validation_dataset = MSBIPatchDataset(
        validation_records,
        patch_size=patch_size,
        samples_per_epoch=int(data["validation_samples_per_epoch"]),
        seed=seed + 10_000,
        augment=False,
        density_sampling_probability=0.0,
        preload_records=bool(data.get("preload_records", False)),
        morphology_balanced_sampling=False,
        normalization=str(data.get("normalization", "percentile")),
    )
    training = config["training"]
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": False,
    }
    return (
        train_dataset,
        DataLoader(train_dataset, shuffle=False, **loader_kwargs),
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs),
    )


def _one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MSBILoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    ema: ExponentialMovingAverage | None = None,
    mentor_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    dice_values: list[float] = []
    context = nullcontext if training else torch.no_grad
    if training:
        optimizer.zero_grad(set_to_none=True)
    with context():
        for batch_index, raw_batch in enumerate(loader):
            batch = _batch_to_device(dict(raw_batch), device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                mentor_outputs = (
                    mentor_model(batch["image"].float())
                    if mentor_model is not None
                    else None
                )
                outputs = model(batch["image"].float())
                total, parts = criterion(
                    outputs,
                    batch,
                    teacher_small=batch.get("teacher_small"),
                    teacher_large=batch.get("teacher_large"),
                    teacher_valid=batch.get("teacher_valid"),
                    mentor_outputs=mentor_outputs,
                )
            if training:
                scaled_loss = total / gradient_accumulation_steps
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                final_batch = batch_index + 1 == len(loader)
                update = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or final_batch
                )
                if update:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
            batch_size = int(batch["image"].shape[0])
            sample_count += batch_size
            totals["loss"] = totals.get("loss", 0.0) + float(total.detach().cpu()) * batch_size
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
            dice_values.append(
                _pixel_dice(
                    outputs["foreground_logits"],
                    batch["foreground"],
                    batch["valid"],
                )
            )
    return {
        **{name: value / max(sample_count, 1) for name, value in totals.items()},
        "pixel_dice": float(np.mean(dice_values)) if dice_values else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--anchor-state-dict", type=Path)
    parser.add_argument("--mentor-checkpoint", type=Path)
    parser.add_argument("--mentor-config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("training config root must be a mapping")
    if args.smoke:
        config = _config_for_smoke(config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    training = config["training"]
    seed = int(training["seed"])
    _seed_everything(seed)
    device = _device(args.device)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _expand(str(training["output_root"])) / f"{timestamp}-seed-{seed}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "resolved-config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "git-diff.patch").write_text(_git("diff") + "\n", encoding="utf-8")
    manifest_root = _expand(str(config["data"]["manifest_root"]))
    split_manifest = manifest_root / "split-manifest.json"
    run_manifest = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "command": " ".join(os.sys.argv),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "split_manifest_path": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "seed": seed,
        "device": str(device),
        "python": platform.python_version(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": (
            _command_output(
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            )
            if device.type == "cuda"
            else "not_cuda"
        ),
    }
    (output_root / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_dataset, train_loader, validation_loader = _build_loaders(config)
    model = build_msbi_model(config["model"]).to(device)
    if (args.mentor_checkpoint is None) != (args.mentor_config is None):
        raise ValueError("--mentor-checkpoint and --mentor-config must be supplied together")
    mentor_model: torch.nn.Module | None = None
    if args.mentor_checkpoint is not None and args.mentor_config is not None:
        mentor_config = yaml.safe_load(args.mentor_config.read_text(encoding="utf-8"))
        if not isinstance(mentor_config, dict) or not isinstance(
            mentor_config.get("model"),
            dict,
        ):
            raise ValueError("mentor config must contain a model mapping")
        mentor_model = build_msbi_model(
            mentor_config["model"],
            for_export=True,
        ).to(device)
        mentor_checkpoint = torch.load(
            args.mentor_checkpoint,
            map_location=device,
            weights_only=False,
        )
        mentor_model.load_state_dict(mentor_checkpoint["model_state"], strict=True)
        mentor_model.eval()
        for parameter in mentor_model.parameters():
            parameter.requires_grad_(False)
        run_manifest["mentor_checkpoint_path"] = str(args.mentor_checkpoint.resolve())
        run_manifest["mentor_checkpoint_sha256"] = _sha256(args.mentor_checkpoint)
        run_manifest["mentor_config_path"] = str(args.mentor_config.resolve())
        run_manifest["mentor_config_sha256"] = _sha256(args.mentor_config)
        (output_root / "run-manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    initialization_inputs = (
        args.resume,
        args.init_checkpoint,
        args.anchor_state_dict,
    )
    if sum(value is not None for value in initialization_inputs) > 1:
        raise ValueError(
            "--resume, --init-checkpoint, and --anchor-state-dict are mutually exclusive"
        )
    if args.init_checkpoint is not None:
        initialized = torch.load(
            args.init_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(initialized["model_state"], strict=True)
        run_manifest["init_checkpoint_path"] = str(args.init_checkpoint.resolve())
        run_manifest["init_checkpoint_sha256"] = _sha256(args.init_checkpoint)
    if args.anchor_state_dict is not None:
        anchor_state = torch.load(
            args.anchor_state_dict,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(anchor_state, Mapping) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in anchor_state.items()
        ):
            raise ValueError("anchor state_dict must map string keys to tensors")
        loaded_anchor_keys = load_unet_small_anchor_state(model, anchor_state)
        run_manifest["anchor_state_dict_path"] = str(
            args.anchor_state_dict.resolve()
        )
        run_manifest["anchor_state_dict_sha256"] = _sha256(args.anchor_state_dict)
        run_manifest["anchor_loaded_key_count"] = len(loaded_anchor_keys)
        (output_root / "run-manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if (
        config["model"].get("encoder") == "convnext_tiny"
        and config["model"].get("encoder_pretrained", True)
    ):
        encoder_weight = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / "convnext_tiny-983f1562.pth"
        )
        run_manifest["encoder_pretrained_revision"] = (
            "torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
        )
        run_manifest["encoder_pretrained_path"] = str(encoder_weight)
        run_manifest["encoder_pretrained_sha256"] = (
            _sha256(encoder_weight) if encoder_weight.is_file() else None
        )
        (output_root / "run-manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif (
        config["model"].get("encoder") == "mobilenet_v3_small"
        and config["model"].get("encoder_pretrained", True)
    ):
        encoder_weight = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / "mobilenet_v3_small-047dcff4.pth"
        )
        run_manifest["encoder_pretrained_revision"] = (
            "torchvision.MobileNet_V3_Small_Weights.IMAGENET1K_V1"
        )
        run_manifest["encoder_pretrained_path"] = str(encoder_weight)
        run_manifest["encoder_pretrained_sha256"] = (
            _sha256(encoder_weight) if encoder_weight.is_file() else None
        )
        (output_root / "run-manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    criterion = MSBILoss(config["loss"]["weights"])
    encoder_module = getattr(model, "encoder", None)
    if not isinstance(encoder_module, torch.nn.Module):
        raise ValueError("MSBI model must expose an encoder module")
    encoder_parameters = [
        parameter for parameter in encoder_module.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    parameter_groups: list[dict[str, Any]] = []
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": float(training["encoder_learning_rate"]),
            }
        )
    parameter_groups.append(
        {"params": head_parameters, "lr": float(training["head_learning_rate"])}
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    warmup_epochs = int(training.get("warmup_epochs", 0))
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
    )
    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(training.get("warmup_start_factor", 0.1)),
            total_iters=warmup_epochs,
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        )
    else:
        scheduler = cosine
    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    amp_dtype_name = str(training.get("amp_dtype", "float16"))
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(amp_dtype_name)
    if amp_dtype is None:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if amp_enabled and amp_dtype is torch.float16
        else None
    )
    ema_decay = float(training.get("ema_decay", 0.0))
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay else None
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    early_stopping_patience = int(training.get("early_stopping_patience", 0))
    start_epoch = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(
            checkpoint.get("training_model_state", checkpoint["model_state"]),
            strict=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.load_state_dict(checkpoint["ema_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        previous_best = args.resume.with_name("best.pt")
        if previous_best.is_file():
            shutil.copyfile(previous_best, output_root / "best.pt")
        else:
            torch.save(checkpoint, output_root / "best.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_batch = _batch_to_device(dict(next(iter(train_loader))), device)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        first_mentor_outputs = (
            mentor_model(first_batch["image"].float())
            if mentor_model is not None
            else None
        )
        first_outputs = model(first_batch["image"].float())
        first_loss, _ = criterion(
            first_outputs,
            first_batch,
            teacher_small=first_batch.get("teacher_small"),
            teacher_large=first_batch.get("teacher_large"),
            teacher_valid=first_batch.get("teacher_valid"),
            mentor_outputs=first_mentor_outputs,
        )
    if not torch.isfinite(first_loss):
        raise FloatingPointError("dry-run loss is non-finite")
    if args.dry_run:
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            first_loss.backward()
        else:
            scaler.scale(first_loss).backward()
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training["gradient_clip_norm"]),
        )
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "device": str(device),
                    "loss": float(first_loss.detach().cpu()),
                    "backward": "ok",
                    "peak_device_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                    "output_shapes": {
                        name: list(value.shape) for name, value in first_outputs.items()
                    },
                    "output_dir": str(output_root),
                },
                indent=2,
            )
        )
        return 0
    metrics_path = output_root / "metrics.csv"
    fields = [
        "epoch",
        "train_loss",
        "train_pixel_dice",
        "validation_loss",
        "validation_pixel_dice",
        "learning_rate",
        "epoch_seconds",
        "peak_device_memory_mb",
    ]
    loss_names = (
        "foreground",
        "foreground_edge",
        "foreground_tversky",
        "center",
        "boundary",
        "contour",
        "sdf",
        "gate",
        "distill",
        "teacher_foreground",
        "mentor_foreground",
        "mentor_center",
        "mentor_boundary",
        "mentor_sdf",
        "mentor_gate",
        "consistency",
    )
    for split in ("train", "validation"):
        fields.extend(f"{split}_{name}" for name in loss_names)
    writer = SummaryWriter(log_dir=output_root / "tensorboard")
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_dataset.set_epoch(epoch)
            train_metrics = _one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                gradient_accumulation_steps=accumulation_steps,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                ema=ema,
                mentor_model=mentor_model,
            )
            validation_context = (
                ema.average_parameters(model)
                if ema is not None
                else nullcontext()
            )
            with validation_context:
                validation_metrics = _one_epoch(
                    model=model,
                    loader=validation_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    gradient_clip_norm=float(training["gradient_clip_norm"]),
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    mentor_model=mentor_model,
                )
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_pixel_dice": train_metrics["pixel_dice"],
                "validation_loss": validation_metrics["loss"],
                "validation_pixel_dice": validation_metrics["pixel_dice"],
                "learning_rate": scheduler.get_last_lr()[-1],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "peak_device_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            for name in loss_names:
                row[f"train_{name}"] = train_metrics.get(name, 0.0)
                row[f"validation_{name}"] = validation_metrics.get(name, 0.0)
            csv_writer.writerow(row)
            stream.flush()
            for name, value in row.items():
                if name != "epoch":
                    writer.add_scalar(name, value, epoch)
            improved = validation_metrics["loss"] < best_validation
            if improved:
                best_validation = validation_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint: dict[str, Any] = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "ema_state": ema.state_dict() if ema is not None else None,
                "best_validation_loss": best_validation,
                "epochs_without_improvement": epochs_without_improvement,
                "config": config,
                "git_commit": run_manifest["git_commit"],
            }
            torch.save(checkpoint, output_root / "last.pt")
            if improved:
                best_checkpoint = dict(checkpoint)
                if ema is not None:
                    with ema.average_parameters(model):
                        best_checkpoint["model_state"] = {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        }
                torch.save(best_checkpoint, output_root / "best.pt")
            with (output_root / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    json.dumps(
                        {
                            "status": "EARLY_STOPPING",
                            "epoch": epoch,
                            "patience": early_stopping_patience,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
    writer.close()
    run_manifest["completed_at"] = datetime.now(UTC).isoformat()
    run_manifest["best_validation_loss"] = best_validation
    run_manifest["best_checkpoint_sha256"] = _sha256(output_root / "best.pt")
    (output_root / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "TRAINING_COMPLETE",
                "output_dir": str(output_root),
                "best_validation_loss": best_validation,
                "best_checkpoint_sha256": _sha256(output_root / "best.pt"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
