"""Vector-index contracts and an atomic, optional FAISS implementation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, Protocol, cast
from uuid import uuid4

import numpy as np

from app.contracts.common import HealthComponent

_MANIFEST_SCHEMA_VERSION = 2
_METRIC = "cosine_ip_normalized"


class VectorStoreUnavailableError(RuntimeError):
    """Raised when vector search is requested without a usable index."""


class VectorIndexValidationError(VectorStoreUnavailableError):
    """Raised when an index, manifest, or database mapping is inconsistent."""


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class VectorIndexRecord:
    """One chunk vector plus the digest of the exact text used to embed it."""

    chunk_id: str
    vector: Sequence[float]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class VectorPublishResult:
    generation: str
    vector_count: int
    dimension: int


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    chunk_id: str
    vector_id: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _Manifest:
    generation: str
    model_id: str
    model_fingerprint: str
    dimension: int
    index_file: str | None
    index_sha256: str | None
    entries: tuple[_ManifestEntry, ...]


class VectorStore(Protocol):
    def health(self) -> HealthComponent: ...

    def search(self, vector: Sequence[float], *, limit: int) -> list[VectorSearchHit]: ...


class UnavailableVectorStore:
    def __init__(self, detail: str = "vector index is not configured") -> None:
        self.detail = detail

    def health(self) -> HealthComponent:
        return HealthComponent(status="unavailable", detail=self.detail)

    def search(self, vector: Sequence[float], *, limit: int) -> list[VectorSearchHit]:
        del vector, limit
        raise VectorStoreUnavailableError(self.detail)


class InMemoryVectorStore:
    """Deterministic cosine index for tests and tiny explicitly supplied corpora."""

    def __init__(self, vectors: dict[str, Sequence[float]] | None = None) -> None:
        self._vectors: dict[str, tuple[float, ...]] = {
            chunk_id: tuple(float(value) for value in vector)
            for chunk_id, vector in (vectors or {}).items()
        }

    def health(self) -> HealthComponent:
        if not self._vectors:
            return HealthComponent(status="degraded", detail="vector index is empty")
        return HealthComponent(status="healthy", detail=f"{len(self._vectors)} vectors")

    def search(self, vector: Sequence[float], *, limit: int) -> list[VectorSearchHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query = tuple(float(value) for value in vector)
        if not query:
            raise ValueError("query vector cannot be empty")
        hits: list[VectorSearchHit] = []
        for chunk_id, candidate in self._vectors.items():
            if len(candidate) != len(query):
                continue
            similarity = self._cosine(query, candidate)
            if similarity > 0:
                hits.append(VectorSearchHit(chunk_id=chunk_id, score=similarity))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))[:limit]

    @staticmethod
    def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
        numerator = sum(left * right for left, right in zip(first, second, strict=True))
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if not first_norm or not second_norm:
            return 0.0
        return numerator / (first_norm * second_norm)


class PersistentFaissVectorStore:
    """Persistent cosine FAISS index published through an atomic manifest pointer.

    Each successful build creates an immutable generation-specific index file. The
    small JSON manifest is replaced atomically only after that file has been read back
    and validated. A failed build therefore cannot overwrite the currently published
    generation. Optional database validation prevents stale vectors from being used
    after chunk text or ready/disabled membership changes.
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        model_id: str,
        model_fingerprint: str | Callable[[], str],
        database_path: str | Path | None = None,
        faiss_loader: Callable[[], object] | None = None,
    ) -> None:
        normalized_model = model_id.strip()
        if not normalized_model:
            raise ValueError("vector model_id must be non-empty")
        self.index_path = Path(index_path).expanduser()
        self.manifest_path = self.index_path.with_name(
            f"{self.index_path.name}.manifest.json"
        )
        self.model_id = normalized_model
        self._model_fingerprint_source = model_fingerprint
        if not callable(model_fingerprint) and not self._is_sha256(model_fingerprint):
            raise ValueError("vector model_fingerprint must be a sha256 digest")
        self.database_path = (
            Path(database_path).expanduser() if database_path is not None else None
        )
        self._faiss_loader = faiss_loader or self._default_faiss_loader
        self._backend: object | None = None
        self._index: object | None = None
        self._manifest: _Manifest | None = None
        self._manifest_signature: tuple[int, int] | None = None
        self._lock = RLock()

    def health(self) -> HealthComponent:
        with self._lock:
            try:
                manifest = self._ensure_loaded_locked()
            except VectorStoreUnavailableError as error:
                return HealthComponent(status="unavailable", detail=str(error))
            if not manifest.entries:
                return HealthComponent(status="degraded", detail="vector index is empty")
            return HealthComponent(
                status="healthy",
                detail=(
                    f"{len(manifest.entries)} vectors, dimension={manifest.dimension}, "
                    f"generation={manifest.generation}"
                ),
            )

    def search(self, vector: Sequence[float], *, limit: int) -> list[VectorSearchHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            manifest = self._ensure_loaded_locked()
            if not manifest.entries:
                return []
            query = self._normalized_matrix([vector], expected_dimension=manifest.dimension)
            if self._index is None:
                raise VectorIndexValidationError("published FAISS index is not loaded")
            search = getattr(self._index, "search", None)
            if not callable(search):
                raise VectorIndexValidationError("FAISS index does not expose search")
            try:
                raw_scores, raw_ids = search(query, min(limit, len(manifest.entries)))
                scores = np.asarray(raw_scores, dtype=np.float32)
                vector_ids = np.asarray(raw_ids, dtype=np.int64)
            except Exception as error:
                raise VectorStoreUnavailableError(
                    f"FAISS search failed: {type(error).__name__}: {error}"
                ) from error
            if scores.shape != vector_ids.shape or scores.ndim != 2 or scores.shape[0] != 1:
                raise VectorIndexValidationError("FAISS search returned an invalid result shape")
            chunk_by_vector_id = {entry.vector_id: entry.chunk_id for entry in manifest.entries}
            hits: list[VectorSearchHit] = []
            for raw_score, raw_vector_id in zip(scores[0], vector_ids[0], strict=True):
                vector_id = int(raw_vector_id)
                if vector_id < 0:
                    continue
                chunk_id = chunk_by_vector_id.get(vector_id)
                score = float(raw_score)
                if chunk_id is None:
                    raise VectorIndexValidationError(
                        f"FAISS returned unknown vector id {vector_id}"
                    )
                if not math.isfinite(score):
                    raise VectorIndexValidationError("FAISS returned a non-finite score")
                if score > 0:
                    hits.append(VectorSearchHit(chunk_id=chunk_id, score=min(score, 1.0)))
            return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))[:limit]

    def publish(self, records: Sequence[VectorIndexRecord]) -> VectorPublishResult:
        """Build and atomically publish a complete replacement generation."""

        if not records:
            return self.publish_empty()
        with self._lock:
            entries, matrix = self._prepare_records(records)
            generation = uuid4().hex
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            backend = self._backend_locked()
            index = self._new_index(backend, dimension=int(matrix.shape[1]))
            vector_ids = np.asarray(
                [entry.vector_id for entry in entries],
                dtype=np.int64,
            )
            add_with_ids = getattr(index, "add_with_ids", None)
            if not callable(add_with_ids):
                raise VectorStoreUnavailableError("FAISS ID map does not expose add_with_ids")
            try:
                add_with_ids(matrix, vector_ids)
            except Exception as error:
                raise VectorStoreUnavailableError(
                    f"FAISS index build failed: {type(error).__name__}: {error}"
                ) from error

            index_file = self._generation_filename(generation)
            final_index_path = self.index_path.parent / index_file
            temp_index_path = final_index_path.with_name(
                f".{final_index_path.name}.{uuid4().hex}.tmp"
            )
            try:
                self._write_index(backend, index, temp_index_path)
                self._fsync_file(temp_index_path)
                os.replace(temp_index_path, final_index_path)
                self._fsync_directory(final_index_path.parent)
                index_sha256 = self._sha256(final_index_path)
                manifest = _Manifest(
                    generation=generation,
                    model_id=self.model_id,
                    model_fingerprint=self._expected_model_fingerprint(),
                    dimension=int(matrix.shape[1]),
                    index_file=index_file,
                    index_sha256=index_sha256,
                    entries=entries,
                )
                self._validate_index_file(backend, manifest, final_index_path)
                self._atomic_write_manifest(manifest)
            except Exception:
                temp_index_path.unlink(missing_ok=True)
                if not self._manifest_references(final_index_path.name):
                    final_index_path.unlink(missing_ok=True)
                raise

            self._index = index
            self._manifest = manifest
            self._manifest_signature = self._current_manifest_signature()
            self._cleanup_old_generations(keep=index_file)
            return VectorPublishResult(
                generation=generation,
                vector_count=len(entries),
                dimension=manifest.dimension,
            )

    def publish_empty(self) -> VectorPublishResult:
        """Atomically publish an empty/tombstone generation without importing FAISS."""

        with self._lock:
            generation = uuid4().hex
            manifest = _Manifest(
                generation=generation,
                model_id=self.model_id,
                model_fingerprint=self._expected_model_fingerprint(),
                dimension=0,
                index_file=None,
                index_sha256=None,
                entries=(),
            )
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_manifest(manifest)
            self._manifest = manifest
            self._index = None
            self._manifest_signature = self._current_manifest_signature()
            self._cleanup_old_generations(keep=None)
            return VectorPublishResult(generation=generation, vector_count=0, dimension=0)

    def _ensure_loaded_locked(self) -> _Manifest:
        signature = self._current_manifest_signature()
        if signature is None:
            self._index = None
            self._manifest = None
            self._manifest_signature = None
            raise VectorStoreUnavailableError(
                f"vector manifest is not published: {self.manifest_path}"
            )
        if self._manifest is not None and signature == self._manifest_signature:
            self._validate_model_identity(self._manifest)
            self._validate_database_mapping(self._manifest)
            return self._manifest

        manifest = self._read_manifest()
        self._validate_database_mapping(manifest)
        if not manifest.entries:
            self._index = None
            self._manifest = manifest
            self._manifest_signature = signature
            return manifest
        if manifest.index_file is None:
            raise VectorIndexValidationError("non-empty manifest has no index file")
        backend = self._backend_locked()
        active_path = self.index_path.parent / manifest.index_file
        index = self._validate_index_file(backend, manifest, active_path)
        self._index = index
        self._manifest = manifest
        self._manifest_signature = signature
        return manifest

    def _read_manifest(self) -> _Manifest:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VectorIndexValidationError(
                f"vector manifest cannot be read: {type(error).__name__}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise VectorIndexValidationError("vector manifest root must be an object")
        if payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise VectorIndexValidationError("unsupported vector manifest schema version")
        if payload.get("metric") != _METRIC:
            raise VectorIndexValidationError("unsupported vector index metric")
        model_id = payload.get("model_id")
        if model_id != self.model_id:
            raise VectorIndexValidationError(
                f"vector model mismatch: expected {self.model_id!r}, found {model_id!r}"
            )
        model_fingerprint = payload.get("model_fingerprint")
        if not self._is_sha256(model_fingerprint):
            raise VectorIndexValidationError(
                "vector manifest model_fingerprint is invalid"
            )
        expected_fingerprint = self._expected_model_fingerprint()
        if model_fingerprint != expected_fingerprint:
            raise VectorIndexValidationError(
                "vector embedding fingerprint mismatch: "
                f"expected {expected_fingerprint}, found {model_fingerprint}"
            )
        generation = self._required_string(payload, "generation")
        dimension = payload.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise VectorIndexValidationError("vector manifest dimension is invalid")
        raw_index_file = payload.get("index_file")
        raw_index_sha256 = payload.get("index_sha256")
        if raw_index_file is not None and (
            not isinstance(raw_index_file, str)
            or Path(raw_index_file).name != raw_index_file
        ):
            raise VectorIndexValidationError("vector manifest index_file is unsafe")
        if raw_index_sha256 is not None and not self._is_sha256(raw_index_sha256):
            raise VectorIndexValidationError("vector manifest index_sha256 is invalid")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise VectorIndexValidationError("vector manifest entries must be a list")
        entries: list[_ManifestEntry] = []
        seen_chunks: set[str] = set()
        seen_ids: set[int] = set()
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise VectorIndexValidationError("vector manifest entry must be an object")
            chunk_id = self._required_string(raw_entry, "chunk_id")
            vector_id = raw_entry.get("vector_id")
            content_sha256 = raw_entry.get("content_sha256")
            if (
                isinstance(vector_id, bool)
                or not isinstance(vector_id, int)
                or vector_id < 0
            ):
                raise VectorIndexValidationError("vector manifest vector_id is invalid")
            if not self._is_sha256(content_sha256):
                raise VectorIndexValidationError("vector manifest content_sha256 is invalid")
            if chunk_id in seen_chunks or vector_id in seen_ids:
                raise VectorIndexValidationError("vector manifest contains duplicate mappings")
            if vector_id != self.stable_vector_id(chunk_id):
                raise VectorIndexValidationError(
                    f"unstable vector mapping for chunk {chunk_id}"
                )
            seen_chunks.add(chunk_id)
            seen_ids.add(vector_id)
            entries.append(
                _ManifestEntry(
                    chunk_id=chunk_id,
                    vector_id=vector_id,
                    content_sha256=cast(str, content_sha256),
                )
            )
        if entries and dimension <= 0:
            raise VectorIndexValidationError("non-empty vector manifest has no dimension")
        if not entries and (raw_index_file is not None or raw_index_sha256 is not None):
            raise VectorIndexValidationError("empty vector manifest must not reference an index")
        if entries and (raw_index_file is None or raw_index_sha256 is None):
            raise VectorIndexValidationError("non-empty vector manifest index metadata is missing")
        return _Manifest(
            generation=generation,
            model_id=cast(str, model_id),
            model_fingerprint=cast(str, model_fingerprint),
            dimension=dimension,
            index_file=raw_index_file,
            index_sha256=raw_index_sha256,
            entries=tuple(entries),
        )

    def _prepare_records(
        self,
        records: Sequence[VectorIndexRecord],
    ) -> tuple[tuple[_ManifestEntry, ...], np.ndarray[Any, np.dtype[np.float32]]]:
        ordered = sorted(records, key=lambda record: record.chunk_id)
        if any(not record.chunk_id for record in ordered):
            raise ValueError("vector chunk_id must be non-empty")
        if len({record.chunk_id for record in ordered}) != len(ordered):
            raise ValueError("vector records contain duplicate chunk_id values")
        if any(not self._is_sha256(record.content_sha256) for record in ordered):
            raise ValueError("vector records require lowercase SHA-256 content digests")
        first_dimension = len(ordered[0].vector)
        matrix = self._normalized_matrix(
            [record.vector for record in ordered],
            expected_dimension=first_dimension,
        )
        entries = tuple(
            _ManifestEntry(
                chunk_id=record.chunk_id,
                vector_id=self.stable_vector_id(record.chunk_id),
                content_sha256=record.content_sha256,
            )
            for record in ordered
        )
        if len({entry.vector_id for entry in entries}) != len(entries):
            raise VectorIndexValidationError("stable vector-id collision detected")
        return entries, matrix

    def _validate_database_mapping(self, manifest: _Manifest) -> None:
        if self.database_path is None:
            return
        path = self.database_path.resolve()
        if not path.is_file():
            raise VectorIndexValidationError(f"vector database is missing: {path}")
        try:
            connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
            try:
                rows = connection.execute(
                    """
                    SELECT c.chunk_id, c.text, c.vector_id
                    FROM knowledge_chunks AS c
                    JOIN knowledge_documents AS d ON d.doc_id = c.doc_id
                    WHERE d.status = 'ready'
                    ORDER BY c.chunk_id
                    """
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise VectorIndexValidationError(
                f"vector database mapping cannot be read: {error}"
            ) from error
        expected = {entry.chunk_id: entry for entry in manifest.entries}
        if {str(row[0]) for row in rows} != set(expected):
            raise VectorIndexValidationError(
                "vector/database chunk membership mismatch; rebuild the vector index"
            )
        for chunk_id_value, text_value, database_vector_id in rows:
            chunk_id = str(chunk_id_value)
            entry = expected[chunk_id]
            if entry.content_sha256 != self.content_sha256(str(text_value)):
                raise VectorIndexValidationError(
                    f"vector/database content mismatch for chunk {chunk_id}"
                )
            if database_vector_id is not None and int(database_vector_id) != entry.vector_id:
                raise VectorIndexValidationError(
                    f"vector/database id mismatch for chunk {chunk_id}"
                )

    def _validate_index_file(
        self,
        backend: object,
        manifest: _Manifest,
        path: Path,
    ) -> object:
        if not path.is_file() or manifest.index_sha256 is None:
            raise VectorIndexValidationError("published FAISS index file is missing")
        if self._sha256(path) != manifest.index_sha256:
            raise VectorIndexValidationError("published FAISS index digest mismatch")
        index = self._read_index(backend, path)
        dimension = getattr(index, "d", None)
        total = getattr(index, "ntotal", None)
        if dimension != manifest.dimension:
            raise VectorIndexValidationError(
                f"FAISS dimension mismatch: expected {manifest.dimension}, found {dimension}"
            )
        if total != len(manifest.entries):
            raise VectorIndexValidationError(
                f"FAISS count mismatch: expected {len(manifest.entries)}, found {total}"
            )
        vector_to_array = getattr(backend, "vector_to_array", None)
        id_map = getattr(index, "id_map", None)
        if callable(vector_to_array) and id_map is not None:
            try:
                stored_ids = {int(value) for value in vector_to_array(id_map)}
            except Exception as error:
                raise VectorIndexValidationError("FAISS id map cannot be inspected") from error
            expected_ids = {entry.vector_id for entry in manifest.entries}
            if stored_ids != expected_ids:
                raise VectorIndexValidationError("FAISS id map does not match the manifest")
        return index

    def _atomic_write_manifest(self, manifest: _Manifest) -> None:
        payload = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "generation": manifest.generation,
            "model_id": manifest.model_id,
            "model_fingerprint": manifest.model_fingerprint,
            "dimension": manifest.dimension,
            "metric": _METRIC,
            "index_file": manifest.index_file,
            "index_sha256": manifest.index_sha256,
            "entries": [
                {
                    "chunk_id": entry.chunk_id,
                    "vector_id": entry.vector_id,
                    "content_sha256": entry.content_sha256,
                }
                for entry in manifest.entries
            ],
        }
        data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
        temp_path = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{uuid4().hex}.tmp"
        )
        try:
            with temp_path.open("xb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_path, self.manifest_path)
            self._fsync_directory(self.manifest_path.parent)
        finally:
            temp_path.unlink(missing_ok=True)

    def _backend_locked(self) -> object:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._faiss_loader()
        except Exception as error:
            if isinstance(error, VectorStoreUnavailableError):
                raise
            raise VectorStoreUnavailableError(
                f"optional FAISS dependency unavailable: {type(error).__name__}: {error}"
            ) from error
        self._backend = backend
        return backend

    @staticmethod
    def _new_index(backend: object, *, dimension: int) -> object:
        flat_factory = getattr(backend, "IndexFlatIP", None)
        id_map_factory = getattr(backend, "IndexIDMap2", None)
        if not callable(flat_factory) or not callable(id_map_factory):
            raise VectorStoreUnavailableError("FAISS backend lacks required index classes")
        return id_map_factory(flat_factory(dimension))

    @staticmethod
    def _write_index(backend: object, index: object, path: Path) -> None:
        writer = getattr(backend, "write_index", None)
        if not callable(writer):
            raise VectorStoreUnavailableError("FAISS backend does not expose write_index")
        try:
            writer(index, str(path))
        except Exception as error:
            raise VectorStoreUnavailableError(
                f"FAISS index write failed: {type(error).__name__}: {error}"
            ) from error

    @staticmethod
    def _read_index(backend: object, path: Path) -> object:
        reader = getattr(backend, "read_index", None)
        if not callable(reader):
            raise VectorStoreUnavailableError("FAISS backend does not expose read_index")
        try:
            return reader(str(path))
        except Exception as error:
            raise VectorIndexValidationError(
                f"FAISS index cannot be read: {type(error).__name__}: {error}"
            ) from error

    @staticmethod
    def _normalized_matrix(
        vectors: Sequence[Sequence[float]],
        *,
        expected_dimension: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if expected_dimension <= 0:
            raise ValueError("vector dimension must be positive")
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("vectors must form a numeric matrix") from error
        if matrix.ndim != 2 or matrix.shape[0] != len(vectors):
            raise ValueError("vectors must form a two-dimensional matrix")
        if matrix.shape[1] != expected_dimension:
            raise ValueError(
                f"vector dimension mismatch: expected {expected_dimension}, "
                f"found {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("vectors contain non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if any(not math.isfinite(float(norm)) or float(norm) <= 0 for norm in norms):
            raise ValueError("vectors contain an empty vector")
        return np.asarray(matrix / norms[:, np.newaxis], dtype=np.float32)

    def _generation_filename(self, generation: str) -> str:
        suffix = self.index_path.suffix or ".index"
        stem = (
            self.index_path.name[: -len(suffix)]
            if self.index_path.suffix
            else self.index_path.name
        )
        return f"{stem}.{generation}{suffix}"

    def _cleanup_old_generations(self, *, keep: str | None) -> None:
        suffix = self.index_path.suffix or ".index"
        stem = (
            self.index_path.name[: -len(suffix)]
            if self.index_path.suffix
            else self.index_path.name
        )
        pattern = f"{stem}.*{suffix}"
        for candidate in self.index_path.parent.glob(pattern):
            if candidate.name != keep and candidate.is_file():
                try:
                    candidate.unlink()
                except OSError:
                    continue

    def _manifest_references(self, filename: str) -> bool:
        try:
            return self._read_manifest().index_file == filename
        except VectorStoreUnavailableError:
            return False

    def _current_manifest_signature(self) -> tuple[int, int] | None:
        try:
            stat = self.manifest_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _expected_model_fingerprint(self) -> str:
        source = self._model_fingerprint_source
        try:
            value = source() if callable(source) else source
        except Exception as error:
            raise VectorStoreUnavailableError(
                "embedding model fingerprint is unavailable: "
                f"{type(error).__name__}: {error}"
            ) from error
        if not self._is_sha256(value):
            raise VectorStoreUnavailableError(
                "embedding model fingerprint is not a sha256 digest"
            )
        return value

    def _validate_model_identity(self, manifest: _Manifest) -> None:
        expected = self._expected_model_fingerprint()
        if manifest.model_fingerprint != expected:
            raise VectorIndexValidationError(
                "vector embedding fingerprint mismatch: "
                f"expected {expected}, found {manifest.model_fingerprint}"
            )

    @staticmethod
    def stable_vector_id(chunk_id: str) -> int:
        if not chunk_id:
            raise ValueError("chunk_id must be non-empty")
        digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) & 0x7FFF_FFFF_FFFF_FFFF

    @staticmethod
    def content_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _required_string(payload: Mapping[str, object], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise VectorIndexValidationError(f"vector manifest {key} is invalid")
        return value

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and value == value.casefold()
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as source:
            os.fsync(source.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(descriptor)
            except OSError:
                return
        finally:
            os.close(descriptor)

    @staticmethod
    def _default_faiss_loader() -> ModuleType:
        try:
            return importlib.import_module("faiss")
        except ImportError as error:
            raise VectorStoreUnavailableError(
                "optional faiss-cpu dependency is not installed"
            ) from error
