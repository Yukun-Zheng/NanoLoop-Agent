"""Pluggable embedding contracts with explicit offline-only behavior."""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

from app.contracts.common import HealthComponent


class EmbeddingUnavailableError(RuntimeError):
    """Raised when vector embedding is requested without a configured model."""


class InvalidEmbeddingError(ValueError):
    """Raised when an embedding backend returns unsafe or inconsistent vectors."""


class EmbeddingProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def health(self) -> HealthComponent: ...

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Sequence[float]]: ...


class UnavailableEmbeddingProvider:
    """Honest placeholder used until an embedding model is installed and configured."""

    def __init__(self, detail: str = "embedding model is not configured") -> None:
        self.detail = detail

    @property
    def fingerprint(self) -> str:
        raise EmbeddingUnavailableError(self.detail)

    def health(self) -> HealthComponent:
        return HealthComponent(status="unavailable", detail=self.detail)

    def embed_query(self, text: str) -> Sequence[float]:
        del text
        raise EmbeddingUnavailableError(self.detail)

    def embed_documents(self, texts: Sequence[str]) -> list[Sequence[float]]:
        del texts
        raise EmbeddingUnavailableError(self.detail)


class CallableEmbeddingProvider:
    """Small adapter for a verified embedding callable supplied by integration code."""

    def __init__(
        self,
        embed: Callable[[Sequence[str]], list[Sequence[float]]],
        *,
        detail: str = "configured embedding provider",
        model_fingerprint: str | None = None,
    ) -> None:
        self._embed = embed
        self.detail = detail
        self._fingerprint = model_fingerprint

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            raise EmbeddingUnavailableError(
                "callable embedding provider has no immutable model fingerprint"
            )
        return self._fingerprint

    def health(self) -> HealthComponent:
        return HealthComponent(status="healthy", detail=self.detail)

    def embed_query(self, text: str) -> Sequence[float]:
        vectors = self._embed([text])
        if len(vectors) != 1:
            raise ValueError("embedding provider returned an unexpected query batch size")
        return vectors[0]

    def embed_documents(self, texts: Sequence[str]) -> list[Sequence[float]]:
        vectors = self._embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedding provider returned an unexpected document batch size")
        return vectors


class SentenceTransformerEmbeddingProvider:
    """Lazy, local-files-only SentenceTransformers adapter.

    The optional dependency and model are loaded only on the first health/embedding
    request. ``local_files_only=True`` is unconditional: a configured Hugging Face ID
    therefore means "use this model if it is already in the local cache", never
    "download it at runtime".
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        revision: str | None = None,
        model_fingerprint: str | None = None,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        normalized = model_id.strip()
        if not normalized:
            raise ValueError("embedding model_id must be non-empty")
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        normalized_revision = revision.strip() if revision else None
        if normalized_revision is not None and re.fullmatch(
            r"[0-9a-f]{40,64}", normalized_revision
        ) is None:
            raise ValueError("embedding revision must be an immutable commit digest")
        if model_fingerprint is not None and re.fullmatch(
            r"[0-9a-f]{64}", model_fingerprint
        ) is None:
            raise ValueError("embedding model_fingerprint must be a sha256 digest")
        if model_factory is not None and model_fingerprint is None:
            raise ValueError(
                "an injected embedding model_factory requires model_fingerprint"
            )
        self.model_id = normalized
        self.device = device.strip() if device and device.strip() != "auto" else None
        self.batch_size = batch_size
        self.revision = normalized_revision
        self._uses_default_factory = model_factory is None
        self._model_factory = model_factory or self._default_model_factory
        self._model_reference = normalized
        self._model_source_path: Path | None = None
        self._fingerprint = model_fingerprint
        self._model: object | None = None
        self._load_error: EmbeddingUnavailableError | None = None
        self._dimension: int | None = None
        self._lock = RLock()

    @property
    def dimension(self) -> int | None:
        """Return the verified dimension, without forcing a model load."""

        with self._lock:
            return self._dimension

    @property
    def fingerprint(self) -> str:
        """Resolve and hash the exact local model snapshot used for embeddings."""

        with self._lock:
            self._ensure_model_source_locked()
            if self._fingerprint is None:
                raise EmbeddingUnavailableError(
                    "embedding model fingerprint could not be resolved"
                )
            return self._fingerprint

    def health(self) -> HealthComponent:
        with self._lock:
            if self._load_error is not None:
                return HealthComponent(status="unavailable", detail=str(self._load_error))
            if self._model is not None:
                dimension = self._dimension
                suffix = f", dimension={dimension}" if dimension is not None else ""
                return HealthComponent(
                    status="healthy",
                    detail=f"local-only SentenceTransformers model ready{suffix}",
                )
            if self._uses_default_factory and not self._dependency_available():
                return HealthComponent(
                    status="unavailable",
                    detail="optional sentence-transformers dependency is not installed",
                )
            try:
                self._ensure_model_source_locked()
            except EmbeddingUnavailableError as error:
                return HealthComponent(status="unavailable", detail=str(error))
            fingerprint = self._fingerprint or "unknown"
            return HealthComponent(
                status="degraded",
                detail=(
                    "local-only embedding model snapshot verified; lazy load not yet "
                    f"verified, fingerprint={fingerprint}"
                ),
            )

    @staticmethod
    def _dependency_available() -> bool:
        try:
            return find_spec("sentence_transformers") is not None
        except (ImportError, ValueError):
            return False

    def embed_query(self, text: str) -> Sequence[float]:
        vectors = self._embed([text])
        return vectors[0]

    def embed_documents(self, texts: Sequence[str]) -> list[Sequence[float]]:
        if not texts:
            return []
        return list(self._embed(texts))

    def _embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        normalized_texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in normalized_texts):
            raise ValueError("embedding texts must be non-empty strings")
        with self._lock:
            model = self._ensure_model_locked()
            encode = getattr(model, "encode", None)
            if not callable(encode):
                raise EmbeddingUnavailableError(
                    "local embedding model does not expose a callable encode method"
                )
            try:
                raw_vectors = encode(
                    normalized_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as error:
                raise EmbeddingUnavailableError(
                    f"local embedding inference failed: {type(error).__name__}: {error}"
                ) from error
            matrix = self._validate_matrix(raw_vectors, expected_rows=len(normalized_texts))
            self._record_dimension(int(matrix.shape[1]))
            return [tuple(float(value) for value in row) for row in matrix]

    def _ensure_model_locked(self) -> object:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise self._load_error
        self._ensure_model_source_locked()
        kwargs: dict[str, object] = {"local_files_only": True}
        if self.device is not None:
            kwargs["device"] = self.device
        try:
            model = self._model_factory(self._model_reference, **kwargs)
            if self._model_source_path is not None:
                observed = self._model_tree_sha256(self._model_source_path)
                if observed != self._fingerprint:
                    raise EmbeddingUnavailableError(
                        "embedding model snapshot changed while it was loading"
                    )
            reported_dimension = self._reported_dimension(model)
            if reported_dimension is not None:
                self._record_dimension(reported_dimension)
        except Exception as error:
            unavailable = (
                error
                if isinstance(error, EmbeddingUnavailableError)
                else EmbeddingUnavailableError(
                    "local embedding model unavailable: "
                    f"{type(error).__name__}: {error}"
                )
            )
            self._load_error = unavailable
            raise unavailable from error
        self._model = model
        return model

    def _ensure_model_source_locked(self) -> None:
        if self._fingerprint is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        try:
            candidate = Path(self.model_id).expanduser()
            if candidate.exists():
                if not candidate.is_dir():
                    raise EmbeddingUnavailableError(
                        "local embedding model path must be a directory"
                    )
                source = candidate.resolve(strict=True)
            else:
                if self.revision is None:
                    raise EmbeddingUnavailableError(
                        "a Hugging Face embedding model requires an immutable revision digest"
                    )
                try:
                    hub = importlib.import_module("huggingface_hub")
                    snapshot = hub.snapshot_download(
                        repo_id=self.model_id,
                        revision=self.revision,
                        local_files_only=True,
                    )
                except Exception as error:
                    raise EmbeddingUnavailableError(
                        "local embedding snapshot cannot be resolved: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                source = Path(snapshot).resolve(strict=True)
                if not source.is_dir():
                    raise EmbeddingUnavailableError(
                        "resolved embedding snapshot is not a directory"
                    )
            self._model_source_path = source
            self._model_reference = str(source)
            self._fingerprint = self._model_tree_sha256(source)
        except Exception as error:
            unavailable = (
                error
                if isinstance(error, EmbeddingUnavailableError)
                else EmbeddingUnavailableError(
                    "embedding model identity cannot be verified: "
                    f"{type(error).__name__}: {error}"
                )
            )
            self._load_error = unavailable
            raise unavailable from error

    @staticmethod
    def _model_tree_sha256(root: Path) -> str:
        files = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        if not files:
            raise EmbeddingUnavailableError("embedding model snapshot contains no files")
        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()

    def _record_dimension(self, dimension: int) -> None:
        if dimension <= 0:
            raise InvalidEmbeddingError("embedding dimension must be positive")
        if self._dimension is not None and self._dimension != dimension:
            raise InvalidEmbeddingError(
                "embedding dimension changed within one provider instance: "
                f"expected {self._dimension}, received {dimension}"
            )
        self._dimension = dimension

    @staticmethod
    def _reported_dimension(model: object) -> int | None:
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        if not callable(getter):
            return None
        value = getter()
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidEmbeddingError("model reported an invalid embedding dimension")
        return int(value)

    @staticmethod
    def _validate_matrix(
        value: object,
        *,
        expected_rows: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            matrix = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidEmbeddingError("embedding output is not a numeric matrix") from error
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] <= 0:
            raise InvalidEmbeddingError(
                "embedding output has an invalid shape: "
                f"expected ({expected_rows}, d>0), received {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise InvalidEmbeddingError("embedding output contains non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if any(not math.isfinite(float(norm)) or float(norm) <= 0 for norm in norms):
            raise InvalidEmbeddingError("embedding output contains an empty vector")
        return np.asarray(matrix / norms[:, np.newaxis], dtype=np.float32)

    @staticmethod
    def _default_model_factory(model_id: str, **kwargs: object) -> object:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as error:
            raise EmbeddingUnavailableError(
                "optional sentence-transformers dependency is not installed"
            ) from error
        model_class: Any = getattr(module, "SentenceTransformer", None)
        if model_class is None:
            raise EmbeddingUnavailableError(
                "sentence-transformers does not expose SentenceTransformer"
            )
        return model_class(model_id, **kwargs)
