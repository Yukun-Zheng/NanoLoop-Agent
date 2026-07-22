"""Stable adapter boundary for segmentation model integrations.

Adapter modules may depend on heavyweight, optional runtimes.  Those dependencies must be
imported by ``load`` rather than at module import time so that the API can start without model
extras installed.
"""

from __future__ import annotations

import os
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.contracts.enums import ModelStatus
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import ModelHealth, ModelMetadata
from app.core.errors import ModelNotReadyError


@runtime_checkable
class SegmentationAdapter(Protocol):
    """Structural contract implemented by every segmentation backend."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def load(self, device: str) -> None: ...

    def health(self) -> ModelHealth: ...

    def predict(self, request: SegmentationRequest) -> SegmentationOutput: ...

    def unload(self) -> None: ...


class BaseSegmentationAdapter(ABC):
    """Small lifecycle implementation shared by real model adapters.

    Subclasses remain responsible for importing their optional runtime inside ``load`` and for
    releasing backend-specific objects in ``_release``.
    """

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        weight_path: Path,
        weight_bytes: bytes,
        config: Mapping[str, Any],
        weight_sha256: str | None = None,
    ) -> None:
        self._metadata = metadata.model_copy(deep=True)
        self.weight_path = Path(weight_path)
        self.weight_bytes = bytes(weight_bytes)
        self.config = dict(config)
        self.weight_sha256 = weight_sha256
        self._loaded = False
        self._device: str | None = None
        self._load_error: str | None = None
        self._runtime_weight_path: Path | None = None

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata.model_copy(deep=True)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, device: str) -> None:
        """Load the immutable model artifact onto ``device``."""

    @abstractmethod
    def predict(self, request: SegmentationRequest) -> SegmentationOutput:
        """Run inference and persist output artifacts below ``request.run_dir``."""

    def health(self) -> ModelHealth:
        if self._load_error is not None:
            status = ModelStatus.UNAVAILABLE
        elif self._loaded:
            status = ModelStatus.READY
        else:
            status = self._metadata.status
        return ModelHealth(
            model_id=self._metadata.model_id,
            status=status,
            error_summary=self._load_error or self._metadata.health_error,
            device=self._device,
            weight_sha256=self.weight_sha256,
        )

    def unload(self) -> None:
        try:
            self._release()
        finally:
            if self._runtime_weight_path is not None:
                self._unlink_runtime_weight(self._runtime_weight_path)
                self._runtime_weight_path = None
            self._loaded = False
            self._device = None

    @abstractmethod
    def _release(self) -> None:
        """Release backend objects."""

    def _mark_loaded(self, device: str) -> None:
        self._loaded = True
        self._device = device
        self._load_error = None

    def _mark_load_failed(self, exc: BaseException) -> None:
        self._loaded = False
        self._device = None
        self._load_error = f"{type(exc).__name__}: {exc}"

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotReadyError(
                details={"model_id": self._metadata.model_id, "reason": "adapter_not_loaded"}
            )

    def _materialize_runtime_weight(self) -> Path:
        """Materialize pinned bytes only for third-party loaders that require a filename."""

        if self._runtime_weight_path is not None:
            return self._runtime_weight_path
        descriptor, name = tempfile.mkstemp(suffix=self.weight_path.suffix or ".weights")
        path = Path(name)
        try:
            view = memoryview(self.weight_bytes)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS contract guard
                    raise OSError("short write while materializing pinned model bytes")
                view = view[written:]
            os.fsync(descriptor)
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o400)
            else:
                os.chmod(path, stat.S_IREAD)
        except Exception:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        self._runtime_weight_path = path
        return path

    @staticmethod
    def _unlink_runtime_weight(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            if os.name != "nt" or not path.exists():
                raise
            os.chmod(path, stat.S_IWRITE)
            path.unlink(missing_ok=True)
