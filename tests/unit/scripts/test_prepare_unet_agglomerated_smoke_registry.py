from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.models import smoke_unet_agglomerated_analysis as smoke
from scripts.models.prepare_unet_agglomerated_smoke_registry import (
    MODEL_ID,
    OUTPUT_FILENAME,
    prepare_smoke_registry,
)


def test_generated_registry_is_accepted_by_agglomerated_smoke_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    weight_path = tmp_path / "weights.pt"
    config_path = tmp_path / "config.yaml"
    model_card_path = tmp_path / "model-card.md"
    weight_path.write_bytes(b"fixture torchscript")
    config_path.write_bytes(
        (
            repository_root / "model_artifacts/configs/unet-agglomerated-specialized-v1.yaml"
        ).read_bytes()
    )
    model_card_path.write_bytes(
        (
            repository_root / "model_artifacts/model_cards/unet-agglomerated-specialized-v1.md"
        ).read_bytes()
    )
    weight_sha256 = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    source_path = tmp_path / "private-registry-ready.yaml"
    source_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0",
                "models": [
                    {
                        "metadata": {
                            "model_id": MODEL_ID,
                            "family": "unet",
                            "variant": "dense_particle",
                            "quality_tier": "balanced",
                            "version": "1",
                            "status": "ready",
                            "supports_box_prompt": False,
                            "default_threshold": 0.25,
                            "default_min_area_px": 1024,
                            "preprocess_profile": "sem-gray-p1-p99-crop-bottom-130-v1",
                            "postprocess_profile": "semantic-agglomerate-mask-v1",
                            "inference_invalid_bottom_px": 130,
                            "expected_input_width": 2048,
                            "expected_input_height": 1536,
                            "applicable_materials": [],
                            "metrics": {},
                            "metric_context": {
                                "checkpoint_sha256": "e" * 64,
                                "architecture_profile": "unet-agglomerated-specialized-v1",
                            },
                            "adapter_sha256": "0" * 64,
                        },
                        "adapter_path": "app.inference.adapters.unet:UNetAdapter",
                        "weight_path": weight_path.name,
                        "weight_sha256": weight_sha256,
                        "config_path": config_path.name,
                        "model_card_path": model_card_path.name,
                        "required_modules": [],
                        "adapter_sha256": "0" * 64,
                        "provenance": {"generated": True},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / OUTPUT_FILENAME
    monkeypatch.setattr(smoke, "TORCHSCRIPT_SHA256", weight_sha256)

    prepare_smoke_registry(source_path, output_path)
    ready_entry = smoke._ready_smoke_registry_entry(output_path)

    generated = yaml.safe_load(output_path.read_text(encoding="utf-8"))["models"][0]
    assert generated["metadata"]["status"] == "unavailable"
    assert generated["metadata"]["default_threshold"] == pytest.approx(0.25)
    assert generated["metadata"]["metric_context"]["architecture_profile"] == (
        "unet-agglomerated-specialized-v1"
    )
    assert generated["weight_path"] == weight_path.name
    assert "adapter_sha256" not in generated
    assert "adapter_sha256" not in generated["metadata"]
    assert "provenance" not in generated
    assert ready_entry["metadata"]["status"] == "ready"
    assert (
        ready_entry["adapter_sha256"]
        == hashlib.sha256(
            (repository_root / "app/inference/adapters/unet.py").read_bytes()
        ).hexdigest()
    )
