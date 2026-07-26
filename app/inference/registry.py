"""YAML-backed model declarations and runtime readiness validation."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
import types
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from app.contracts.enums import ModelStatus
from app.contracts.models import ModelBundleReference, ModelHealth, ModelMetadata
from app.core.config import get_settings
from app.core.errors import ModelNotFoundError, ModelNotReadyError
from app.inference.adapters.base import SegmentationAdapter
from app.inference.snapshots import ModelArtifactSnapshotError, ModelArtifactSnapshotStore

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_RE = re.compile(r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):(?P<name>[A-Za-z_]\w*)$")


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Validated declaration plus resolved local artifact paths."""

    metadata: ModelMetadata
    adapter_path: str
    weight_path: Path
    config_path: Path
    model_card_path: Path
    adapter_source_path: Path | None
    weight_sha256: str | None
    config_sha256: str | None
    model_card_sha256: str | None
    adapter_sha256: str | None
    required_modules: tuple[str, ...]
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelArtifactProvenance:
    model_version: str
    adapter_path: str
    weight_sha256: str
    config_sha256: str
    model_card_sha256: str
    adapter_sha256: str

    @property
    def cache_key(self) -> str:
        joined = ":".join(
            (
                self.model_version,
                self.adapter_path,
                self.weight_sha256,
                self.config_sha256,
                self.model_card_sha256,
                self.adapter_sha256,
            )
        )
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidatedModelBundle:
    """One internally consistent set of immutable inputs for an adapter load."""

    provenance: ModelArtifactProvenance
    reference: ModelBundleReference
    snapshot_weight_path: Path
    weight_bytes: bytes
    config_bytes: bytes
    model_card_bytes: bytes
    adapter_source_bytes: bytes
    parsed_config: Mapping[str, Any]
    metadata: ModelMetadata


AdapterResolver = Callable[[str], type[SegmentationAdapter]]


class ModelRegistryService:
    """Load model declarations and expose only evidence-backed readiness.

    The YAML file is the declaration source of truth.  A declaration can become ``ready`` only
    when its adapter reference, config, model card, weights, SHA-256, and optional imports all
    validate.  Validation failures are represented as health state and never prevent API startup.
    """

    def __init__(
        self,
        registry_path: str | Path | None = None,
        *,
        adapter_resolver: AdapterResolver | None = None,
        snapshot_store: ModelArtifactSnapshotStore | None = None,
        snapshot_root: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        configured_path = registry_path or settings.model_registry_path
        self.registry_path = Path(configured_path).expanduser().resolve()
        self._adapter_resolver = adapter_resolver or self._resolve_adapter_class
        self._uses_snapshot_adapter = adapter_resolver is None
        self.snapshot_store = snapshot_store or ModelArtifactSnapshotStore(
            snapshot_root or settings.model_snapshot_root
        )
        self._lock = RLock()
        self._registrations: dict[str, ModelRegistration] = {}
        self._validated_bundles: dict[tuple[str, str], ValidatedModelBundle] = {}
        self._snapshot_modules: dict[str, types.ModuleType] = {}
        self._registry_error: str | None = None
        self.refresh()

    @property
    def registry_error(self) -> str | None:
        return self._registry_error

    def refresh(self) -> None:
        """Reload declarations atomically.

        A malformed top-level registry is recorded as a degraded health state. Individual malformed
        declarations are retained where a valid ``model_id`` is available and marked unavailable.
        """

        try:
            raw = yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("registry root must be a mapping")
            entries = raw.get("models")
            if not isinstance(entries, list):
                raise ValueError("registry 'models' must be a list")
            parsed = self._parse_entries(entries)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            with self._lock:
                self._registrations = {}
                self._validated_bundles = {}
                self._registry_error = f"{type(exc).__name__}: {exc}"
            return

        with self._lock:
            self._registrations = parsed
            self._validated_bundles = {}
            self._registry_error = None

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        with self._lock:
            values = [item.metadata.model_copy(deep=True) for item in self._registrations.values()]
        if only_ready:
            values = [item for item in values if item.status == ModelStatus.READY]
        return sorted(values, key=lambda item: item.model_id)

    def get_metadata(self, model_id: str) -> ModelMetadata:
        return self.get_registration(model_id).metadata.model_copy(deep=True)

    def get(self, model_id: str) -> ModelMetadata:
        """Compatibility shorthand for callers that treat the service as a registry mapping."""

        return self.get_metadata(model_id)

    def get_registration(self, model_id: str) -> ModelRegistration:
        with self._lock:
            registration = self._registrations.get(model_id)
        if registration is None:
            raise ModelNotFoundError(details={"model_id": model_id})
        return replace(
            registration,
            metadata=registration.metadata.model_copy(deep=True),
            config=deepcopy(registration.config),
        )

    def health(self) -> list[ModelHealth]:
        with self._lock:
            registrations = list(self._registrations.values())
        return [
            ModelHealth(
                model_id=item.metadata.model_id,
                status=item.metadata.status,
                error_summary=item.metadata.health_error,
                weight_sha256=(
                    item.weight_sha256 if item.metadata.status == ModelStatus.READY else None
                ),
            )
            for item in sorted(registrations, key=lambda value: value.metadata.model_id)
        ]

    def create_adapter(
        self,
        model_id: ValidatedModelBundle | str,
        *,
        expected_provenance: ModelArtifactProvenance | None = None,
    ) -> SegmentationAdapter:
        """Create from a validated bundle; retain the historical model-id entry point.

        The compatibility entry point validates and snapshots first. If the caller already used
        :meth:`validate_artifacts`, its cached bundle is reused rather than reading mutable source
        files a second time.
        """

        if isinstance(model_id, ValidatedModelBundle):
            bundle = model_id
            resolved_model_id = bundle.metadata.model_id
            self.verify_bundle(bundle)
        else:
            resolved_model_id = model_id
            bundle = self._compatibility_bundle(resolved_model_id, expected_provenance)
            registration = self.get_registration(resolved_model_id)
            if registration.metadata.status != ModelStatus.READY:
                raise ModelNotReadyError(
                    details={
                        "model_id": resolved_model_id,
                        "status": registration.metadata.status.value,
                        "reason": registration.metadata.health_error,
                    }
                )
            current_provenance = self._registration_provenance(registration)
            if current_provenance != bundle.provenance or (
                expected_provenance is not None and bundle.provenance != expected_provenance
            ):
                raise ModelNotReadyError(
                    details={
                        "model_id": resolved_model_id,
                        "reason": "registry_changed_during_adapter_load",
                    }
                )
            self.verify_bundle(bundle)

        if self._uses_snapshot_adapter:
            resolved_class = self._resolve_bundle_adapter_class(bundle)
        else:
            resolved_class = self._adapter_resolver(bundle.provenance.adapter_path)
        adapter_class = cast(Callable[..., SegmentationAdapter], resolved_class)
        adapter = adapter_class(
            metadata=bundle.metadata.model_copy(deep=True),
            # A handoff adapter receives bytes plus a non-existent label retaining only the
            # suffix. It cannot accidentally reopen the replaceable content-store path.
            weight_path=(
                Path("/__nanoloop_pinned_bundle__")
                / bundle.reference.bundle_id
                / f"weights{bundle.snapshot_weight_path.suffix}"
            ),
            weight_bytes=bundle.weight_bytes,
            config=deepcopy(bundle.parsed_config),
            weight_sha256=bundle.provenance.weight_sha256,
        )
        if not isinstance(adapter, SegmentationAdapter):
            raise TypeError(
                f"{bundle.provenance.adapter_path} does not implement SegmentationAdapter"
            )
        return adapter

    def _compatibility_bundle(
        self,
        model_id: str,
        expected_provenance: ModelArtifactProvenance | None,
    ) -> ValidatedModelBundle:
        if expected_provenance is not None:
            with self._lock:
                cached = self._validated_bundles.get((model_id, expected_provenance.cache_key))
            if cached is not None:
                return cached
            return self.validate_bundle(
                model_id,
                expected_model_version=expected_provenance.model_version,
                expected_adapter_path=expected_provenance.adapter_path,
                expected_weight_sha256=expected_provenance.weight_sha256,
                expected_config_sha256=expected_provenance.config_sha256,
                expected_model_card_sha256=expected_provenance.model_card_sha256,
                expected_adapter_sha256=expected_provenance.adapter_sha256,
            )
        return self.validate_bundle(model_id)

    def validate_artifacts(
        self,
        model_id: str,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
    ) -> ModelArtifactProvenance:
        """Compatibility wrapper returning provenance for a fully validated bundle."""

        return self.validate_bundle(
            model_id,
            expected_model_version=expected_model_version,
            expected_adapter_path=expected_adapter_path,
            expected_weight_sha256=expected_weight_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_model_card_sha256=expected_model_card_sha256,
            expected_adapter_sha256=expected_adapter_sha256,
        ).provenance

    def validate_bundle(
        self,
        model_id: str,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
    ) -> ValidatedModelBundle:
        """Read each mutable declaration input once and create an immutable load bundle.

        Registry refresh and adapter caching are not allowed to turn an in-place
        artifact mutation into a scientifically different result under the same
        run configuration.
        """

        registration = self.get_registration(model_id)
        if registration.metadata.status != ModelStatus.READY:
            raise ModelNotReadyError(
                details={
                    "model_id": model_id,
                    "status": registration.metadata.status.value,
                    "reason": registration.metadata.health_error,
                }
            )
        if (
            registration.weight_sha256 is None
            or registration.config_sha256 is None
            or registration.model_card_sha256 is None
            or registration.adapter_sha256 is None
            or registration.adapter_source_path is None
        ):
            raise ModelNotReadyError(
                details={"model_id": model_id, "reason": "artifact_provenance_incomplete"}
            )
        try:
            config_bytes = registration.config_path.read_bytes()
            model_card_bytes = registration.model_card_path.read_bytes()
            adapter_source_bytes = registration.adapter_source_path.read_bytes()
            observed_config_sha = hashlib.sha256(config_bytes).hexdigest()
            observed_card_sha = hashlib.sha256(model_card_bytes).hexdigest()
            observed_adapter_sha = hashlib.sha256(adapter_source_bytes).hexdigest()
            snapshot_weight_path = self.snapshot_store.publish(
                registration.weight_path, registration.weight_sha256
            )
        except (OSError, ModelArtifactSnapshotError) as error:
            reason = f"model artifact cannot be read: {type(error).__name__}: {error}"
            self.mark_unavailable(model_id, reason)
            raise ModelNotReadyError(
                details={
                    "model_id": model_id,
                    "reason": (
                        "artifact_integrity_mismatch"
                        if isinstance(error, ModelArtifactSnapshotError)
                        else "artifact_unreadable"
                    ),
                    "artifacts": ["weight_sha256"]
                    if isinstance(error, ModelArtifactSnapshotError)
                    else None,
                }
            ) from error

        observed = {
            "weight_sha256": registration.weight_sha256,
            "config_sha256": observed_config_sha,
            "model_card_sha256": observed_card_sha,
            "adapter_sha256": observed_adapter_sha,
        }
        registered = {
            "weight_sha256": registration.weight_sha256,
            "config_sha256": registration.config_sha256,
            "model_card_sha256": registration.model_card_sha256,
            "adapter_sha256": registration.adapter_sha256,
        }
        mutated = [name for name in registered if observed[name] != registered[name]]
        if mutated:
            reason = "immutable model artifact changed after registry validation: " + ", ".join(
                sorted(mutated)
            )
            self.mark_unavailable(model_id, reason)
            raise ModelNotReadyError(
                details={
                    "model_id": model_id,
                    "reason": "artifact_integrity_mismatch",
                    "artifacts": sorted(mutated),
                }
            )

        try:
            self._parse_config_bytes(config_bytes)
            if not model_card_bytes.decode("utf-8").strip():
                raise ValueError("model card is empty")
        except (UnicodeError, ValueError, yaml.YAMLError) as error:
            reason = f"model artifact cannot be parsed: {type(error).__name__}: {error}"
            self.mark_unavailable(model_id, reason)
            raise ModelNotReadyError(
                details={"model_id": model_id, "reason": "artifact_unreadable"}
            ) from error

        run_expected = {
            "model_version": expected_model_version,
            "adapter_path": expected_adapter_path,
            "weight_sha256": expected_weight_sha256,
            "config_sha256": expected_config_sha256,
            "model_card_sha256": expected_model_card_sha256,
            "adapter_sha256": expected_adapter_sha256,
        }
        current_values = {
            "model_version": registration.metadata.version,
            "adapter_path": registration.adapter_path,
            **observed,
        }
        run_mismatches = [
            name
            for name, expected_hash in run_expected.items()
            if expected_hash is not None and current_values[name] != expected_hash
        ]
        if run_mismatches:
            raise ModelNotReadyError(
                "模型工件与已冻结运行配置不一致",
                details={
                    "model_id": model_id,
                    "reason": "run_artifact_mismatch",
                    "artifacts": sorted(run_mismatches),
                },
            )
        provenance = ModelArtifactProvenance(
            model_version=registration.metadata.version,
            adapter_path=registration.adapter_path,
            weight_sha256=observed["weight_sha256"],
            config_sha256=observed["config_sha256"],
            model_card_sha256=observed["model_card_sha256"],
            adapter_sha256=observed["adapter_sha256"],
        )
        config_snapshot = self.snapshot_store.publish_bytes(
            f"config{registration.config_path.suffix or '.yaml'}",
            config_bytes,
            provenance.config_sha256,
        )
        model_card_snapshot = self.snapshot_store.publish_bytes(
            f"model-card{registration.model_card_path.suffix or '.md'}",
            model_card_bytes,
            provenance.model_card_sha256,
        )
        adapter_snapshot = self.snapshot_store.publish_bytes(
            "adapter.py",
            adapter_source_bytes,
            provenance.adapter_sha256,
        )
        artifact_references = {
            "weight_ref": self.snapshot_store.reference(snapshot_weight_path),
            "config_ref": self.snapshot_store.reference(config_snapshot),
            "model_card_ref": self.snapshot_store.reference(model_card_snapshot),
            "adapter_ref": self.snapshot_store.reference(adapter_snapshot),
        }
        manifest = {
            "schema_version": 1,
            "metadata": registration.metadata.model_dump(mode="json"),
            "provenance": {
                "model_version": provenance.model_version,
                "adapter_path": provenance.adapter_path,
                "weight_sha256": provenance.weight_sha256,
                "config_sha256": provenance.config_sha256,
                "model_card_sha256": provenance.model_card_sha256,
                "adapter_sha256": provenance.adapter_sha256,
            },
            "artifacts": artifact_references,
        }
        manifest_bytes = self._canonical_json(manifest)
        bundle_id = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = self.snapshot_store.publish_bundle_manifest(bundle_id, manifest_bytes)
        reference = ModelBundleReference(
            bundle_id=bundle_id,
            manifest_ref=self.snapshot_store.reference(manifest_path),
            adapter_sha256=provenance.adapter_sha256,
            **artifact_references,
        )
        bundle = self.open_bundle(reference)
        with self._lock:
            self._validated_bundles[(model_id, provenance.cache_key)] = bundle
        return bundle

    def verify_bundle(self, bundle: ValidatedModelBundle) -> None:
        """Verify bytes already pinned by descriptor; no filesystem path is reopened."""

        observed = {
            "weight_sha256": hashlib.sha256(bundle.weight_bytes).hexdigest(),
            "config_sha256": hashlib.sha256(bundle.config_bytes).hexdigest(),
            "model_card_sha256": hashlib.sha256(bundle.model_card_bytes).hexdigest(),
            "adapter_sha256": hashlib.sha256(bundle.adapter_source_bytes).hexdigest(),
        }
        expected = {
            "weight_sha256": bundle.provenance.weight_sha256,
            "config_sha256": bundle.provenance.config_sha256,
            "model_card_sha256": bundle.provenance.model_card_sha256,
            "adapter_sha256": bundle.provenance.adapter_sha256,
        }
        mismatches = sorted(name for name in expected if observed[name] != expected[name])
        if mismatches:
            raise ModelNotReadyError(
                details={
                    "model_id": bundle.metadata.model_id,
                    "reason": "pinned_bundle_integrity_mismatch",
                    "artifacts": mismatches,
                }
            )

    def open_bundle(self, reference: ModelBundleReference) -> ValidatedModelBundle:
        """Open a persisted bundle exclusively through hash-checked pinned descriptors."""

        try:
            manifest_bytes = self.snapshot_store.read_reference(
                reference.manifest_ref, reference.bundle_id
            )
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict) or self._canonical_json(manifest) != manifest_bytes:
                raise ValueError("bundle manifest is not canonical JSON")
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported model bundle manifest schema")
            artifacts = manifest.get("artifacts")
            provenance_raw = manifest.get("provenance")
            metadata_raw = manifest.get("metadata")
            if not isinstance(artifacts, dict) or not isinstance(provenance_raw, dict):
                raise ValueError("bundle manifest sections are invalid")
            expected_refs = {
                "weight_ref": reference.weight_ref,
                "config_ref": reference.config_ref,
                "model_card_ref": reference.model_card_ref,
                "adapter_ref": reference.adapter_ref,
            }
            if artifacts != expected_refs:
                raise ValueError("bundle reference does not match its manifest")
            provenance = ModelArtifactProvenance(
                model_version=str(provenance_raw["model_version"]),
                adapter_path=str(provenance_raw["adapter_path"]),
                weight_sha256=str(provenance_raw["weight_sha256"]),
                config_sha256=str(provenance_raw["config_sha256"]),
                model_card_sha256=str(provenance_raw["model_card_sha256"]),
                adapter_sha256=str(provenance_raw["adapter_sha256"]),
            )
            if provenance.adapter_sha256 != reference.adapter_sha256:
                raise ValueError("adapter digest does not match bundle reference")
            metadata = ModelMetadata.model_validate(metadata_raw)
            if (
                metadata.model_id == ""
                or metadata.version != provenance.model_version
                or metadata.adapter_path != provenance.adapter_path
                or metadata.weight_sha256 != provenance.weight_sha256
                or metadata.config_sha256 != provenance.config_sha256
                or metadata.model_card_sha256 != provenance.model_card_sha256
                or metadata.adapter_sha256 != provenance.adapter_sha256
            ):
                raise ValueError("bundle metadata does not match provenance")
            weight_bytes = self.snapshot_store.read_reference(
                reference.weight_ref, provenance.weight_sha256
            )
            config_bytes = self.snapshot_store.read_reference(
                reference.config_ref, provenance.config_sha256
            )
            model_card_bytes = self.snapshot_store.read_reference(
                reference.model_card_ref, provenance.model_card_sha256
            )
            adapter_source_bytes = self.snapshot_store.read_reference(
                reference.adapter_ref, provenance.adapter_sha256
            )
            parsed_config = self._parse_config_bytes(config_bytes)
            if not model_card_bytes.decode("utf-8").strip():
                raise ValueError("model card is empty")
        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
            yaml.YAMLError,
            ModelArtifactSnapshotError,
            ValidationError,
        ) as error:
            raise ModelNotReadyError(
                details={
                    "bundle_id": reference.bundle_id,
                    "reason": "model_bundle_integrity_mismatch",
                }
            ) from error

        bundle = ValidatedModelBundle(
            provenance=provenance,
            reference=reference.model_copy(deep=True),
            snapshot_weight_path=self.snapshot_store.root.joinpath(
                *PurePosixPath(reference.weight_ref).parts
            ),
            weight_bytes=weight_bytes,
            config_bytes=config_bytes,
            model_card_bytes=model_card_bytes,
            adapter_source_bytes=adapter_source_bytes,
            parsed_config=deepcopy(parsed_config),
            metadata=metadata.model_copy(deep=True),
        )
        self.verify_bundle(bundle)
        return bundle

    def _resolve_bundle_adapter_class(
        self, bundle: ValidatedModelBundle
    ) -> type[SegmentationAdapter]:
        cached = self._snapshot_modules.get(bundle.reference.bundle_id)
        match = _ADAPTER_RE.fullmatch(bundle.provenance.adapter_path)
        if match is None:
            raise ImportError(f"invalid adapter path: {bundle.provenance.adapter_path}")
        class_name = match.group("name")
        if cached is None:
            original_module = match.group("module")
            package = original_module.rpartition(".")[0]
            suffix = bundle.reference.bundle_id[:20]
            module_name = (
                f"{package}._nanoloop_bundle_{suffix}"
                if package
                else f"_nanoloop_bundle_{suffix}"
            )
            module = types.ModuleType(module_name)
            module.__file__ = bundle.reference.adapter_ref
            module.__package__ = package
            sys.modules[module_name] = module
            try:
                source = bundle.adapter_source_bytes.decode("utf-8")
                exec(compile(source, bundle.reference.adapter_ref, "exec"), module.__dict__)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            self._snapshot_modules[bundle.reference.bundle_id] = module
            cached = module
        adapter_class = getattr(cached, class_name, None)
        if not isinstance(adapter_class, type):
            raise ImportError(
                f"adapter class does not exist in frozen bundle: {bundle.provenance.adapter_path}"
            )
        return cast(type[SegmentationAdapter], adapter_class)

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _registration_provenance(
        registration: ModelRegistration,
    ) -> ModelArtifactProvenance:
        if (
            registration.weight_sha256 is None
            or registration.config_sha256 is None
            or registration.model_card_sha256 is None
            or registration.adapter_sha256 is None
        ):
            raise ModelNotReadyError(
                details={
                    "model_id": registration.metadata.model_id,
                    "reason": "artifact_provenance_incomplete",
                }
            )
        return ModelArtifactProvenance(
            model_version=registration.metadata.version,
            adapter_path=registration.adapter_path,
            weight_sha256=registration.weight_sha256,
            config_sha256=registration.config_sha256,
            model_card_sha256=registration.model_card_sha256,
            adapter_sha256=registration.adapter_sha256,
        )

    def mark_unavailable(self, model_id: str, reason: str) -> None:
        """Record a runtime load failure until the next explicit registry refresh."""

        with self._lock:
            current = self._registrations.get(model_id)
            if current is None:
                raise ModelNotFoundError(details={"model_id": model_id})
            metadata = current.metadata.model_copy(
                update={"status": ModelStatus.UNAVAILABLE, "health_error": reason}, deep=True
            )
            self._registrations[model_id] = replace(current, metadata=metadata)

    def _parse_entries(self, entries: list[object]) -> dict[str, ModelRegistration]:
        registrations: dict[str, ModelRegistration] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            registration = self._parse_entry(cast(dict[str, Any], entry), index=index)
            model_id = registration.metadata.model_id
            if model_id in registrations:
                duplicate = registration.metadata.model_copy(
                    update={
                        "status": ModelStatus.UNAVAILABLE,
                        "health_error": f"duplicate model_id: {model_id}",
                    },
                    deep=True,
                )
                registrations[model_id] = replace(registration, metadata=duplicate)
            else:
                registrations[model_id] = registration
        return registrations

    def _parse_entry(self, entry: dict[str, Any], *, index: int) -> ModelRegistration:
        metadata_source = entry.get("metadata")
        if isinstance(metadata_source, dict):
            metadata_raw = dict(metadata_source)
        else:
            metadata_raw = {
                name: entry[name] for name in ModelMetadata.model_fields if name in entry
            }

        model_id = str(metadata_raw.get("model_id") or f"invalid-model-{index}")
        errors: list[str] = []
        try:
            metadata = ModelMetadata.model_validate(metadata_raw)
        except ValidationError as exc:
            errors.append(f"invalid metadata: {exc.errors(include_url=False)}")
            metadata = self._fallback_metadata(model_id, str(exc))

        adapter_path = self._read_string(entry, "adapter", "adapter_path", "adapter_class")
        adapter_source_path: Path | None = None
        adapter_sha256: str | None = None
        if not adapter_path:
            errors.append("adapter path is missing")
            adapter_path = "invalid:Adapter"
        else:
            adapter_error = self._validate_adapter_path(adapter_path)
            if adapter_error:
                errors.append(adapter_error)
            else:
                try:
                    adapter_source_path = self._adapter_source_path(adapter_path)
                    adapter_source_bytes = adapter_source_path.read_bytes()
                    adapter_source_bytes.decode("utf-8")
                    adapter_sha256 = hashlib.sha256(adapter_source_bytes).hexdigest()
                except (ImportError, OSError, UnicodeError, ValueError) as exc:
                    errors.append(
                        f"adapter implementation cannot be frozen: {type(exc).__name__}: {exc}"
                    )
        declared_adapter_sha_raw = entry.get("adapter_sha256")
        if declared_adapter_sha_raw is not None:
            declared_adapter_sha = str(declared_adapter_sha_raw).lower()
            if not _SHA256_RE.fullmatch(declared_adapter_sha):
                errors.append("adapter_sha256 must be 64 lowercase hexadecimal characters")
            elif adapter_sha256 != declared_adapter_sha:
                errors.append(
                    "adapter sha256 mismatch: "
                    f"expected {declared_adapter_sha}, observed {adapter_sha256}"
                )

        weight_path = self._resolve_path(
            self._read_string(entry, "weight_path", "weights", "weights_path") or ""
        )
        config_path = self._resolve_path(self._read_string(entry, "config_path", "config") or "")
        model_card_path = self._resolve_path(
            self._read_string(entry, "model_card_path", "model_card", "card_path") or ""
        )

        config: Mapping[str, Any] = {}
        config_sha256: str | None = None
        if not config_path.is_file():
            errors.append(f"config file is missing: {config_path}")
        else:
            try:
                config_bytes = config_path.read_bytes()
                config = self._parse_config_bytes(config_bytes)
                config_sha256 = hashlib.sha256(config_bytes).hexdigest()
                if adapter_path == "app.inference.adapters.unet:UNetAdapter":
                    errors.extend(self._unet_contract_errors(metadata, config))
            except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                errors.append(f"invalid config: {type(exc).__name__}: {exc}")

        model_card_sha256: str | None = None
        if not model_card_path.is_file():
            errors.append(f"model card is missing: {model_card_path}")
        else:
            try:
                model_card_bytes = model_card_path.read_bytes()
                if not model_card_bytes.decode("utf-8").strip():
                    errors.append(f"model card is empty: {model_card_path}")
                else:
                    model_card_sha256 = hashlib.sha256(model_card_bytes).hexdigest()
            except (OSError, UnicodeError) as exc:
                errors.append(f"invalid model card: {type(exc).__name__}: {exc}")

        expected_sha = entry.get("weight_sha256")
        weight_sha256 = str(expected_sha).lower() if expected_sha is not None else None
        if not weight_path.is_file():
            errors.append(f"weight file is missing: {weight_path}")
        elif weight_sha256 is None:
            errors.append("weight_sha256 is required")
        elif not _SHA256_RE.fullmatch(weight_sha256):
            errors.append("weight_sha256 must be 64 lowercase hexadecimal characters")
        else:
            observed_sha = self._sha256(weight_path)
            if observed_sha != weight_sha256:
                errors.append(
                    f"weight sha256 mismatch: expected {weight_sha256}, observed {observed_sha}"
                )

        required_raw = entry.get("required_modules", entry.get("optional_dependencies", []))
        required_modules = self._normalize_modules(required_raw)
        for module_name in required_modules:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            if not available:
                errors.append(f"optional dependency is missing: {module_name}")

        if errors:
            status = ModelStatus.UNAVAILABLE
            health_error = "; ".join(errors)
        elif metadata.status == ModelStatus.DISABLED:
            status = ModelStatus.DISABLED
            health_error = metadata.health_error or "model is disabled by registry declaration"
        elif metadata.status == ModelStatus.UNAVAILABLE:
            status = ModelStatus.UNAVAILABLE
            health_error = metadata.health_error or "model is unavailable by registry declaration"
        else:
            status = ModelStatus.READY
            health_error = None
        metadata = metadata.model_copy(
            update={
                "status": status,
                "health_error": health_error,
                "adapter_path": adapter_path,
                "weight_sha256": (
                    weight_sha256 if _SHA256_RE.fullmatch(weight_sha256 or "") else None
                ),
                "config_sha256": config_sha256,
                "model_card_sha256": model_card_sha256,
                "adapter_sha256": adapter_sha256,
                **self._unet_expected_metadata(adapter_path, config),
            },
            deep=True,
        )

        return ModelRegistration(
            metadata=metadata,
            adapter_path=adapter_path,
            weight_path=weight_path,
            config_path=config_path,
            model_card_path=model_card_path,
            adapter_source_path=adapter_source_path,
            weight_sha256=weight_sha256 if _SHA256_RE.fullmatch(weight_sha256 or "") else None,
            config_sha256=config_sha256,
            model_card_sha256=model_card_sha256,
            adapter_sha256=adapter_sha256,
            required_modules=required_modules,
            config=config,
        )

    def _resolve_path(self, value: str) -> Path:
        if not value:
            return self.registry_path.parent / "__missing__"
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        registry_relative = (self.registry_path.parent / path).resolve()
        if registry_relative.exists():
            return registry_relative
        project_root = self._find_project_root()
        project_relative = (project_root / path).resolve()
        if project_relative.exists() or path.parts[:1] == ("model_artifacts",):
            return project_relative
        return registry_relative

    def _find_project_root(self) -> Path:
        for candidate in (self.registry_path.parent, *self.registry_path.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
        return self.registry_path.parent

    @staticmethod
    def _read_string(entry: Mapping[str, Any], *names: str) -> str | None:
        for name in names:
            value = entry.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _normalize_modules(value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())

    @staticmethod
    def _parse_config_bytes(value: bytes) -> Mapping[str, Any]:
        config_raw = yaml.safe_load(value.decode("utf-8"))
        if not isinstance(config_raw, dict):
            raise ValueError("config root must be a mapping")
        return cast(dict[str, Any], config_raw)

    @staticmethod
    def _unet_contract_errors(
        metadata: ModelMetadata,
        config: Mapping[str, Any],
    ) -> list[str]:
        """Keep Analysis defaults and Adapter preprocessing on one scientific contract."""

        errors: list[str] = []
        bottom_crop = config.get("bottom_crop_px", 0)
        if (
            isinstance(bottom_crop, bool)
            or not isinstance(bottom_crop, int)
            or bottom_crop < 0
        ):
            errors.append("U-Net bottom_crop_px must be a non-negative integer")
        elif bottom_crop != metadata.inference_invalid_bottom_px:
            errors.append(
                "U-Net bottom_crop_px differs from metadata.inference_invalid_bottom_px"
            )

        expected_size = ModelRegistryService._unet_expected_image_size(config)
        if expected_size is None:
            if "expected_image_size" in config:
                errors.append(
                    "U-Net expected_image_size must be [height, width] positive integers"
                )
            if bottom_crop:
                errors.append(
                    "U-Net expected_image_size is required when bottom_crop_px is non-zero"
                )
            if (
                metadata.expected_input_height is not None
                or metadata.expected_input_width is not None
            ):
                errors.append(
                    "U-Net metadata expected input dimensions require config expected_image_size"
                )
        else:
            expected_height, expected_width = expected_size
            if (
                metadata.expected_input_height is not None
                and metadata.expected_input_height != expected_height
            ):
                errors.append(
                    "U-Net expected_image_size height differs from metadata expected input height"
                )
            if (
                metadata.expected_input_width is not None
                and metadata.expected_input_width != expected_width
            ):
                errors.append(
                    "U-Net expected_image_size width differs from metadata expected input width"
                )

        if metadata.default_threshold is None:
            errors.append("U-Net metadata.default_threshold is required")

        config_threshold = config.get("default_threshold")
        if config_threshold is None:
            errors.append("U-Net config default_threshold is required")
        else:
            if (
                isinstance(config_threshold, bool)
                or not isinstance(config_threshold, int | float)
                or not 0 <= float(config_threshold) <= 1
            ):
                errors.append("U-Net default_threshold must be numeric in [0, 1]")
            elif metadata.default_threshold is None or not math.isclose(
                float(config_threshold), metadata.default_threshold, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(
                    "U-Net config default_threshold differs from metadata.default_threshold"
                )

        calibrated = config.get("calibrated_analysis")
        if not isinstance(calibrated, Mapping):
            return errors
        calibrated_threshold = calibrated.get("threshold")
        if (
            isinstance(calibrated_threshold, bool)
            or not isinstance(calibrated_threshold, int | float)
            or metadata.default_threshold is None
            or not math.isclose(
                float(calibrated_threshold),
                metadata.default_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            errors.append(
                "U-Net calibrated threshold differs from metadata.default_threshold"
            )
        calibrated_min_area = calibrated.get("min_area_px")
        if (
            isinstance(calibrated_min_area, bool)
            or not isinstance(calibrated_min_area, int)
            or calibrated_min_area < 0
            or calibrated_min_area != metadata.default_min_area_px
        ):
            errors.append(
                "U-Net calibrated min_area_px differs from metadata.default_min_area_px"
            )
        calibrated_bottom = calibrated.get("bottom_crop_px")
        if (
            isinstance(calibrated_bottom, bool)
            or not isinstance(calibrated_bottom, int)
            or calibrated_bottom != metadata.inference_invalid_bottom_px
        ):
            errors.append(
                "U-Net calibrated bottom_crop_px differs from "
                "metadata.inference_invalid_bottom_px"
            )
        if calibrated.get("threshold_comparison") != config.get(
            "threshold_comparison"
        ):
            errors.append(
                "U-Net calibrated threshold_comparison differs from inference config"
            )
        return errors

    @staticmethod
    def _unet_expected_image_size(
        config: Mapping[str, Any],
    ) -> tuple[int, int] | None:
        value = config.get("expected_image_size")
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            or any(item <= 0 for item in value)
        ):
            return None
        return value[0], value[1]

    @staticmethod
    def _unet_expected_metadata(
        adapter_path: str,
        config: Mapping[str, Any],
    ) -> dict[str, int]:
        if adapter_path != "app.inference.adapters.unet:UNetAdapter":
            return {}
        expected_size = ModelRegistryService._unet_expected_image_size(config)
        if expected_size is None:
            return {}
        expected_height, expected_width = expected_size
        return {
            "expected_input_height": expected_height,
            "expected_input_width": expected_width,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resolve_adapter_class(adapter_path: str) -> type[SegmentationAdapter]:
        match = _ADAPTER_RE.fullmatch(adapter_path)
        if match is None:
            raise ImportError(f"invalid adapter path: {adapter_path}")
        module = importlib.import_module(match.group("module"))
        adapter_class = getattr(module, match.group("name"), None)
        if not isinstance(adapter_class, type):
            raise ImportError(f"adapter class does not exist: {adapter_path}")
        return cast(type[SegmentationAdapter], adapter_class)

    @staticmethod
    def _adapter_source_path(adapter_path: str) -> Path:
        match = _ADAPTER_RE.fullmatch(adapter_path)
        if match is None:
            raise ImportError(f"invalid adapter path: {adapter_path}")
        spec = importlib.util.find_spec(match.group("module"))
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            raise ImportError("adapter implementation must be a Python source module")
        source_path = Path(spec.origin).resolve()
        if not source_path.is_file():
            raise ImportError(f"adapter source does not exist: {source_path}")
        return source_path

    @staticmethod
    def _validate_adapter_path(adapter_path: str) -> str | None:
        match = _ADAPTER_RE.fullmatch(adapter_path)
        if match is None:
            return f"invalid adapter path: {adapter_path}"
        try:
            spec = importlib.util.find_spec(match.group("module"))
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            return f"adapter module cannot be inspected: {type(exc).__name__}: {exc}"
        if spec is None:
            return f"adapter module does not exist: {match.group('module')}"
        if spec.origin and spec.origin.endswith(".py"):
            try:
                tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeError) as exc:
                return f"adapter module cannot be parsed: {type(exc).__name__}: {exc}"
            exported_names = {
                node.name
                for node in tree.body
                if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            }
            if match.group("name") not in exported_names:
                return f"adapter class does not exist: {adapter_path}"
        return None

    @staticmethod
    def _fallback_metadata(model_id: str, reason: str) -> ModelMetadata:
        from app.contracts.enums import ModelFamily, ModelVariant, QualityTier

        return ModelMetadata(
            model_id=model_id,
            family=ModelFamily.UNET,
            variant=ModelVariant.GENERAL,
            quality_tier=QualityTier.BALANCED,
            version="invalid",
            status=ModelStatus.UNAVAILABLE,
            supports_box_prompt=False,
            preprocess_profile="invalid",
            postprocess_profile="invalid",
            health_error=reason,
        )
