"""Small contract fakes; production code never fabricates model output."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.contracts.enums import ModelStatus
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelHealth, ModelMetadata


class FakeAdapter:
    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        weight_path: Path,
        config: Mapping[str, Any],
        weight_bytes: bytes = b"",
        weight_sha256: str | None = None,
    ) -> None:
        self._metadata = metadata
        self.weight_path = weight_path
        self.weight_bytes = bytes(weight_bytes)
        self.config = dict(config)
        self.weight_sha256 = weight_sha256
        self.load_calls = 0
        self.predict_calls = 0
        self.unload_calls = 0
        self.device: str | None = None

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def load(self, device: str) -> None:
        self.load_calls += 1
        if self.config.get("fail_load"):
            raise RuntimeError("fake load failed")
        self.device = device

    def health(self) -> ModelHealth:
        return ModelHealth(
            model_id=self.metadata.model_id,
            status=ModelStatus.READY if self.device else self.metadata.status,
            device=self.device,
            weight_sha256=self.weight_sha256,
        )

    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        self.predict_calls += 1
        if self.config.get("fail_predict"):
            raise RuntimeError("fake predict failed")
        return SegmentationOutput(
            width=8,
            height=6,
            binary_mask_path=request.run_dir / "fake-mask.png",
            runtime_ms=1,
        )

    def unload(self) -> None:
        self.unload_calls += 1
        self.device = None
