"""Registry fixture builders."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from app.inference.adapters.base import SegmentationAdapter
from app.inference.registry import ModelRegistryService
from tests.unit.inference.fakes import FakeAdapter


def model_entry(
    root: Path,
    model_id: str,
    *,
    family: str = "unet",
    variant: str = "general",
    tier: str = "balanced",
    status: str = "ready",
    supports_box_prompt: bool = False,
    config: dict[str, Any] | None = None,
    required_modules: list[str] | None = None,
) -> dict[str, Any]:
    weights = root / f"{model_id}.weights"
    weights.write_bytes(f"weights:{model_id}".encode())
    config_path = root / f"{model_id}.yaml"
    config_path.write_text(yaml.safe_dump(config or {"fixture": True}), encoding="utf-8")
    card_path = root / f"{model_id}.md"
    card_path.write_text(f"# {model_id}\n\nFixture model card.\n", encoding="utf-8")
    return {
        "metadata": {
            "model_id": model_id,
            "family": family,
            "variant": variant,
            "quality_tier": tier,
            "version": "test-1",
            "status": status,
            "supports_box_prompt": supports_box_prompt,
            "default_threshold": 0.5,
            "preprocess_profile": "fixture-preprocess",
            "postprocess_profile": "fixture-postprocess",
        },
        "adapter_path": "app.inference.adapters.base:BaseSegmentationAdapter",
        "weight_path": weights.name,
        "weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "config_path": config_path.name,
        "model_card_path": card_path.name,
        "required_modules": required_modules or [],
    }


def build_registry(
    root: Path,
    entries: list[dict[str, Any]],
    *,
    resolver: Callable[[str], type[SegmentationAdapter]] | None = None,
) -> ModelRegistryService:
    registry_path = root / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"schema_version": "test", "models": entries}, sort_keys=False),
        encoding="utf-8",
    )
    return ModelRegistryService(
        registry_path,
        adapter_resolver=resolver or (lambda _: FakeAdapter),
        snapshot_root=root / "snapshots",
    )
