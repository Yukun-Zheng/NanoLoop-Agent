"""Application-facing inference facade."""

from __future__ import annotations

from contextlib import suppress

from app.contracts.enums import (
    DevicePreference,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.execution import InferenceExecutionEvidence
from app.contracts.inference import SegmentationOutput, SegmentationRequest
from app.contracts.models import (
    ModelBundleReference,
    ModelCandidate,
    ModelHealth,
    ModelMetadata,
    ModelRecommendationRequest,
)
from app.core.errors import (
    InferenceExecutionError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from app.inference.cache import AdapterCache, AdapterLoadError
from app.inference.execution import deterministic_inference, resolve_device
from app.inference.registry import ModelArtifactProvenance, ModelRegistryService


class InferenceGateway:
    """Single public entry point for model discovery, selection, health, and execution."""

    def __init__(self, registry: ModelRegistryService, cache: AdapterCache | None = None) -> None:
        self.registry = registry
        self.cache = cache or AdapterCache()

    def list_models(self, only_ready: bool = False) -> list[ModelMetadata]:
        return self.registry.list_models(only_ready=only_ready)

    def recommend(self, request: ModelRecommendationRequest) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for metadata in self.registry.list_models(only_ready=True):
            memory_mb = metadata.metric_context.get("gpu_memory_mb")
            if (
                request.max_gpu_memory_mb is not None
                and isinstance(memory_mb, int | float)
                and memory_mb > request.max_gpu_memory_mb
            ):
                continue

            score = 0.45
            reasons: list[str] = []
            if metadata.variant == request.target_profile:
                score += 0.25
                reasons.append("target profile match")
            elif metadata.variant == ModelVariant.GENERAL:
                score += 0.08
                reasons.append("general-purpose fallback")

            preferred_tier = {
                "speed": QualityTier.FAST,
                "balance": QualityTier.BALANCED,
                "accuracy": QualityTier.ACCURATE,
            }[request.prefer]
            if metadata.quality_tier == preferred_tier:
                score += 0.2
                reasons.append(f"{request.prefer} preference match")
            elif metadata.quality_tier == QualityTier.BALANCED:
                score += 0.08
                reasons.append("balanced quality tier")

            if request.roi_mode == RoiMode.BOXES:
                if metadata.supports_box_prompt:
                    score += 0.1
                    reasons.append("supports box prompts")
                else:
                    score -= 0.2
                    reasons.append("box prompts require ROI post-filtering")

            score = round(max(0.0, min(1.0, score)), 4)
            candidates.append(
                ModelCandidate(model_id=metadata.model_id, score=score, reasons=reasons)
            )
        return sorted(candidates, key=lambda item: (-item.score, item.model_id))

    def predict(
        self,
        model_id: str,
        request: SegmentationRequest,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
        model_bundle: ModelBundleReference | None = None,
    ) -> SegmentationOutput:
        if model_bundle is None:
            metadata = self.registry.get_metadata(model_id)
            if metadata.status != ModelStatus.READY:
                raise ModelNotReadyError(
                    details={
                        "model_id": model_id,
                        "status": metadata.status.value,
                        "reason": metadata.health_error,
                    }
                )

        device = resolve_device(request.device.value)
        resolved_request = request.model_copy(
            update={"device": DevicePreference(device)},
        )
        try:
            if model_bundle is None:
                bundle = self.registry.validate_bundle(
                    model_id,
                    expected_model_version=expected_model_version,
                    expected_adapter_path=expected_adapter_path,
                    expected_weight_sha256=expected_weight_sha256,
                    expected_config_sha256=expected_config_sha256,
                    expected_model_card_sha256=expected_model_card_sha256,
                    expected_adapter_sha256=expected_adapter_sha256,
                )
            else:
                bundle = self.registry.open_bundle(model_bundle)
                if bundle.metadata.model_id != model_id:
                    raise ModelNotReadyError(
                        details={
                            "model_id": model_id,
                            "reason": "model_bundle_id_mismatch",
                        }
                    )
                self._validate_frozen_expectations(
                    model_id,
                    bundle.provenance,
                    expected_model_version=expected_model_version,
                    expected_adapter_path=expected_adapter_path,
                    expected_weight_sha256=expected_weight_sha256,
                    expected_config_sha256=expected_config_sha256,
                    expected_model_card_sha256=expected_model_card_sha256,
                    expected_adapter_sha256=expected_adapter_sha256,
                )
            with deterministic_inference(request.seed, device=device) as controls:
                with self.cache.lease(
                    model_id,
                    device=device,
                    factory=lambda _: self.registry.create_adapter(bundle),
                    fingerprint=bundle.provenance.cache_key,
                ) as adapter:
                    # Adapter load consumes only the immutable snapshot. Recheck that snapshot
                    # after load without reopening any mutable registry source.
                    self.registry.verify_bundle(bundle)
                    output = adapter.predict(resolved_request)
                    if not isinstance(output, SegmentationOutput):
                        raise TypeError("adapter returned a non-SegmentationOutput value")
                    backend = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
                evidence = InferenceExecutionEvidence(
                    actual_device=device,
                    python_random_seeded=controls.python_random_seeded,
                    numpy_random_seeded=controls.numpy_random_seeded,
                    torch_deterministic_algorithms=(
                        controls.torch_deterministic_algorithms
                    ),
                    global_inference_serialized=controls.global_inference_serialized,
                    backend=backend,
                )
            return output.model_copy(update={"execution": evidence})
        except (ModelNotFoundError, ModelNotReadyError):
            raise
        except AdapterLoadError as exc:
            cause = exc.__cause__ or exc
            with suppress(ModelNotFoundError):
                self.registry.mark_unavailable(
                    model_id, f"adapter load failed: {type(cause).__name__}: {cause}"
                )
            raise InferenceExecutionError(
                details={"model_id": model_id, "stage": "load", "device": device}
            ) from cause
        except Exception as exc:
            raise InferenceExecutionError(
                details={"model_id": model_id, "stage": "predict", "device": device}
            ) from exc

    def freeze_model_bundle(
        self,
        model_id: str,
        *,
        expected_model_version: str | None = None,
        expected_adapter_path: str | None = None,
        expected_weight_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_model_card_sha256: str | None = None,
        expected_adapter_sha256: str | None = None,
    ) -> ModelBundleReference:
        """Publish every model load input before a run is durably queued."""

        bundle = self.registry.validate_bundle(
            model_id,
            expected_model_version=expected_model_version,
            expected_adapter_path=expected_adapter_path,
            expected_weight_sha256=expected_weight_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_model_card_sha256=expected_model_card_sha256,
            expected_adapter_sha256=expected_adapter_sha256,
        )
        return bundle.reference.model_copy(deep=True)

    @staticmethod
    def _validate_frozen_expectations(
        model_id: str,
        provenance: ModelArtifactProvenance,
        **expected: str | None,
    ) -> None:
        current = {
            "expected_model_version": provenance.model_version,
            "expected_adapter_path": provenance.adapter_path,
            "expected_weight_sha256": provenance.weight_sha256,
            "expected_config_sha256": provenance.config_sha256,
            "expected_model_card_sha256": provenance.model_card_sha256,
            "expected_adapter_sha256": provenance.adapter_sha256,
        }
        mismatches = sorted(
            name.removeprefix("expected_")
            for name, value in expected.items()
            if value is not None and current[name] != value
        )
        if mismatches:
            raise ModelNotReadyError(
                "模型 bundle 与已冻结运行配置不一致",
                details={
                    "model_id": model_id,
                    "reason": "run_artifact_mismatch",
                    "artifacts": mismatches,
                },
            )

    def health(self) -> list[ModelHealth]:
        health_by_id = {item.model_id: item for item in self.registry.health()}
        with self.cache.health_snapshot() as adapters:
            for adapter in adapters:
                registry_health = health_by_id.get(adapter.metadata.model_id)
                if registry_health is not None and registry_health.status != ModelStatus.READY:
                    continue
                try:
                    runtime_health = adapter.health()
                except Exception as exc:  # a health probe must not take down the API
                    metadata = adapter.metadata
                    runtime_health = ModelHealth(
                        model_id=metadata.model_id,
                        status=ModelStatus.UNAVAILABLE,
                        error_summary=f"health check failed: {type(exc).__name__}: {exc}",
                    )
                health_by_id[runtime_health.model_id] = runtime_health
        return [health_by_id[key] for key in sorted(health_by_id)]
