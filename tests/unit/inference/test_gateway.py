from __future__ import annotations

import importlib
from pathlib import Path
from threading import Event, Thread

import pytest
import yaml

from app.contracts.enums import DevicePreference, ModelStatus, ModelVariant, RoiMode
from app.contracts.inference import SegmentationRequest
from app.contracts.models import ModelHealth, ModelRecommendationRequest
from app.core.errors import InferenceExecutionError, ModelNotFoundError, ModelNotReadyError
from app.inference.cache import AdapterCache
from app.inference.gateway import InferenceGateway
from app.inference.registry import ModelRegistryService
from tests.unit.inference.fakes import FakeAdapter
from tests.unit.inference.helpers import build_registry, model_entry


def request(tmp_path: Path) -> SegmentationRequest:
    return SegmentationRequest(
        image_id="image-1",
        image_path=tmp_path / "image.png",
        run_dir=tmp_path / "run",
        roi_mode=RoiMode.FULL_IMAGE,
        device=DevicePreference.CPU,
    )


def test_predict_is_lazy_and_reuses_cached_fake(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "ready")
    instances: list[FakeAdapter] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class TrackedFake(FakeAdapter):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                instances.append(self)

        return TrackedFake

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    gateway = InferenceGateway(registry, AdapterCache())

    assert instances == []
    first = gateway.predict("ready", request(tmp_path))
    second = gateway.predict("ready", request(tmp_path))

    assert first.binary_mask_path.name == "fake-mask.png"
    assert second.binary_mask_path == first.binary_mask_path
    assert first.execution is not None
    assert first.execution.actual_device == "cpu"
    assert first.execution.python_random_seeded is True
    assert first.execution.numpy_random_seeded is True
    assert first.execution.global_inference_serialized is True
    assert first.execution.backend.endswith("TrackedFake")
    assert len(instances) == 1
    assert instances[0].load_calls == 1
    assert instances[0].predict_calls == 2


def test_frozen_run_hashes_reject_registry_refresh_and_cache_by_fingerprint(
    tmp_path: Path,
) -> None:
    entry = model_entry(tmp_path, "ready")
    instances: list[FakeAdapter] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class TrackedFake(FakeAdapter):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                instances.append(self)

        return TrackedFake

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    gateway = InferenceGateway(registry, AdapterCache())
    frozen = registry.get_metadata("ready")
    expected = {
        "expected_model_version": frozen.version,
        "expected_adapter_path": frozen.adapter_path,
        "expected_weight_sha256": frozen.weight_sha256,
        "expected_config_sha256": frozen.config_sha256,
        "expected_model_card_sha256": frozen.model_card_sha256,
    }
    gateway.predict("ready", request(tmp_path), **expected)

    (tmp_path / entry["config_path"]).write_text("fixture: changed\n", encoding="utf-8")
    registry.refresh()

    with pytest.raises(ModelNotReadyError) as captured:
        gateway.predict("ready", request(tmp_path), **expected)

    assert captured.value.details["reason"] == "run_artifact_mismatch"
    assert captured.value.details["artifacts"] == ["config_sha256"]
    assert len(instances) == 1

    gateway.predict("ready", request(tmp_path))
    assert len(instances) == 2
    assert instances[0].predict_calls == 1
    assert instances[1].predict_calls == 1

    current = registry.get_metadata("ready")
    current_expected = {
        "expected_model_version": current.version,
        "expected_adapter_path": current.adapter_path,
        "expected_weight_sha256": current.weight_sha256,
        "expected_config_sha256": current.config_sha256,
        "expected_model_card_sha256": current.model_card_sha256,
    }
    registry_payload = yaml.safe_load((tmp_path / "registry.yaml").read_text(encoding="utf-8"))
    registry_payload["models"][0]["metadata"]["version"] = "test-2"
    (tmp_path / "registry.yaml").write_text(
        yaml.safe_dump(registry_payload, sort_keys=False),
        encoding="utf-8",
    )
    registry.refresh()

    with pytest.raises(ModelNotReadyError) as version_error:
        gateway.predict("ready", request(tmp_path), **current_expected)

    assert version_error.value.details["artifacts"] == ["model_version"]
    gateway.predict("ready", request(tmp_path))
    assert len(instances) == 3


def test_adapter_cannot_mutate_nested_registry_config(tmp_path: Path) -> None:
    entry = model_entry(
        tmp_path,
        "nested-config",
        config={"nested": {"values": [1]}},
    )
    observed: list[list[int]] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class MutatingConfigFake(FakeAdapter):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                nested = self.config["nested"]
                assert isinstance(nested, dict)
                values = nested["values"]
                assert isinstance(values, list)
                observed.append(list(values))
                values.append(999)

        return MutatingConfigFake

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    gateway = InferenceGateway(registry, AdapterCache())

    gateway.predict("nested-config", request(tmp_path))
    gateway.cache.clear()
    gateway.predict("nested-config", request(tmp_path))

    assert observed == [[1], [1]]
    registration = registry.get_registration("nested-config")
    assert registration.config == {"nested": {"values": [1]}}


def test_in_place_artifact_mutation_marks_model_unavailable(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "ready")
    registry = build_registry(tmp_path, [entry])
    gateway = InferenceGateway(registry)
    (tmp_path / entry["weight_path"]).write_bytes(b"mutated in place")

    with pytest.raises(ModelNotReadyError) as captured:
        gateway.predict("ready", request(tmp_path))

    assert captured.value.details["reason"] == "artifact_integrity_mismatch"
    assert captured.value.details["artifacts"] == ["weight_sha256"]
    assert registry.get_metadata("ready").status == ModelStatus.UNAVAILABLE


def test_source_mutated_and_restored_during_load_cannot_change_snapshot(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "ready")
    weight_path = tmp_path / entry["weight_path"]
    original_weight = weight_path.read_bytes()
    instances: list[FakeAdapter] = []
    loaded_bytes: list[bytes] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class MutatingLoadFake(FakeAdapter):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                instances.append(self)

            def load(self, device: str) -> None:
                weight_path.write_bytes(b"changed during adapter.load")
                try:
                    loaded_bytes.append(self.weight_bytes)
                    super().load(device)
                finally:
                    weight_path.write_bytes(original_weight)

        return MutatingLoadFake

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    cache = AdapterCache()
    gateway = InferenceGateway(registry, cache)
    gateway.predict("ready", request(tmp_path))

    assert weight_path.read_bytes() == original_weight
    assert loaded_bytes == [original_weight]
    assert len(instances) == 1
    assert instances[0].weight_path != weight_path
    assert instances[0].predict_calls == 1


def test_snapshot_path_swap_during_load_cannot_change_pinned_weight_bytes(
    tmp_path: Path,
) -> None:
    entry = model_entry(tmp_path, "ready")
    original_weight = (tmp_path / entry["weight_path"]).read_bytes()
    observed: list[bytes] = []
    snapshot_path: list[Path] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class SwappingLoadFake(FakeAdapter):
            def load(self, device: str) -> None:
                target = snapshot_path[0]
                backup = target.with_name("weights.original")
                target.replace(backup)
                target.write_bytes(b"malicious replacement")
                target.chmod(0o444)
                try:
                    observed.append(self.weight_bytes)
                    super().load(device)
                finally:
                    target.unlink()
                    backup.replace(target)

        return SwappingLoadFake

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    gateway = InferenceGateway(registry, AdapterCache())
    frozen_bundle = registry.validate_bundle("ready")
    snapshot_path.append(frozen_bundle.snapshot_weight_path)

    gateway.predict("ready", request(tmp_path))

    assert observed == [original_weight]


def test_frozen_bundle_ignores_config_and_card_source_mutation_after_queue(
    tmp_path: Path,
) -> None:
    entry = model_entry(tmp_path, "ready", config={"marker": "queued"})
    observed_configs: list[dict[str, object]] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class RecordingConfigFake(FakeAdapter):
            def load(self, device: str) -> None:
                observed_configs.append(dict(self.config))
                super().load(device)

        return RecordingConfigFake

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    gateway = InferenceGateway(registry, AdapterCache())
    frozen = registry.get_metadata("ready")
    bundle = gateway.freeze_model_bundle(
        "ready",
        expected_model_version=frozen.version,
        expected_adapter_path=frozen.adapter_path,
        expected_weight_sha256=frozen.weight_sha256,
        expected_config_sha256=frozen.config_sha256,
        expected_model_card_sha256=frozen.model_card_sha256,
        expected_adapter_sha256=frozen.adapter_sha256,
    )
    (tmp_path / entry["config_path"]).write_text("marker: changed\n", encoding="utf-8")
    (tmp_path / entry["model_card_path"]).write_text("# changed\n", encoding="utf-8")
    registry.refresh()

    gateway.predict(
        "ready",
        request(tmp_path),
        expected_model_version=frozen.version,
        expected_adapter_path=frozen.adapter_path,
        expected_weight_sha256=frozen.weight_sha256,
        expected_config_sha256=frozen.config_sha256,
        expected_model_card_sha256=frozen.model_card_sha256,
        expected_adapter_sha256=frozen.adapter_sha256,
        model_bundle=bundle,
    )

    assert observed_configs == [{"marker": "queued"}]


def test_frozen_bundle_executes_snapshotted_adapter_source_after_module_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "handoff_adapter.py"

    def adapter_source(marker: str) -> str:
        return f'''\
from app.contracts.enums import ModelStatus
from app.contracts.inference import SegmentationOutput
from app.contracts.models import ModelHealth

MARKER = "{marker}"

class HandoffAdapter:
    def __init__(self, *, metadata, weight_path, weight_bytes, config, weight_sha256=None):
        self._metadata = metadata
        self.weight_path = weight_path
        self.weight_bytes = weight_bytes
        self.config = config
        self.weight_sha256 = weight_sha256

    @property
    def metadata(self):
        return self._metadata

    def load(self, device):
        self.device = device

    def predict(self, request):
        return SegmentationOutput(
            width=8,
            height=6,
            binary_mask_path=request.run_dir / "frozen-mask.png",
            warnings=[MARKER],
            runtime_ms=1,
        )

    def health(self):
        return ModelHealth(model_id=self.metadata.model_id, status=ModelStatus.READY)

    def unload(self):
        self.device = None
'''

    module_path.write_text(adapter_source("queued-adapter"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    entry = model_entry(tmp_path, "ready")
    entry["adapter_path"] = "handoff_adapter:HandoffAdapter"
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"schema_version": "test", "models": [entry]}, sort_keys=False),
        encoding="utf-8",
    )
    registry = ModelRegistryService(registry_path, snapshot_root=tmp_path / "snapshots")
    gateway = InferenceGateway(registry, AdapterCache())
    frozen = registry.get_metadata("ready")
    bundle = gateway.freeze_model_bundle("ready")

    module_path.write_text(adapter_source("changed-adapter"), encoding="utf-8")
    importlib.invalidate_caches()
    registry_path.write_text("models: registry-is-now-unusable\n", encoding="utf-8")
    recovered_registry = ModelRegistryService(
        registry_path,
        snapshot_root=tmp_path / "snapshots",
    )
    recovered_gateway = InferenceGateway(recovered_registry, AdapterCache())
    output = recovered_gateway.predict(
        "ready",
        request(tmp_path),
        expected_model_version=frozen.version,
        expected_adapter_path=frozen.adapter_path,
        expected_weight_sha256=frozen.weight_sha256,
        expected_config_sha256=frozen.config_sha256,
        expected_model_card_sha256=frozen.model_card_sha256,
        expected_adapter_sha256=frozen.adapter_sha256,
        model_bundle=bundle,
    )

    assert output.warnings == ["queued-adapter"]


def test_registry_unavailable_status_wins_over_cached_adapter_health(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "ready")
    registry = build_registry(tmp_path, [entry])
    gateway = InferenceGateway(registry)
    gateway.predict("ready", request(tmp_path))

    registry.mark_unavailable("ready", "operator disabled runtime")

    health = gateway.health()[0]
    assert health.status == ModelStatus.UNAVAILABLE
    assert health.error_summary == "operator disabled runtime"


def test_health_probe_and_concurrent_unload_share_the_adapter_lifecycle_lock(
    tmp_path: Path,
) -> None:
    entry = model_entry(tmp_path, "ready")
    registry = build_registry(tmp_path, [entry])
    metadata = registry.get_metadata("ready")
    probe_started = Event()
    release_probe = Event()
    unload_started = Event()
    unload_finished = Event()

    class BlockingHealthAdapter(FakeAdapter):
        def health(self) -> ModelHealth:
            probe_started.set()
            if not release_probe.wait(timeout=2):
                raise TimeoutError("test did not release health probe")
            return super().health()

    adapter = BlockingHealthAdapter(
        metadata=metadata,
        weight_path=tmp_path / entry["weight_path"],
        config={},
        weight_sha256=metadata.weight_sha256,
    )
    cache = AdapterCache()
    cache.get_or_load("ready", device="cpu", factory=lambda _: adapter)
    gateway = InferenceGateway(registry, cache)
    observed_health: list[ModelHealth] = []

    health_worker = Thread(target=lambda: observed_health.extend(gateway.health()))

    def unload() -> None:
        unload_started.set()
        cache.unload("ready", device="cpu")
        unload_finished.set()

    health_worker.start()
    assert probe_started.wait(timeout=1)
    unload_worker = Thread(target=unload)
    unload_worker.start()
    assert unload_started.wait(timeout=1)
    assert not unload_finished.wait(timeout=0.05)
    assert adapter.unload_calls == 0

    release_probe.set()
    health_worker.join(timeout=1)
    unload_worker.join(timeout=1)

    assert not health_worker.is_alive()
    assert not unload_worker.is_alive()
    assert unload_finished.is_set()
    assert adapter.unload_calls == 1
    assert len(observed_health) == 1
    assert cache.loaded() == []


def test_unknown_and_unavailable_models_map_to_domain_errors(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "unavailable", status="unavailable")
    registry = build_registry(tmp_path, [entry])
    gateway = InferenceGateway(registry)

    try:
        gateway.predict("unknown", request(tmp_path))
    except ModelNotFoundError:
        pass
    else:  # pragma: no cover - assertion aid
        raise AssertionError("ModelNotFoundError was not raised")

    try:
        gateway.predict("unavailable", request(tmp_path))
    except ModelNotReadyError as exc:
        assert exc.details["status"] == ModelStatus.UNAVAILABLE.value
    else:  # pragma: no cover - assertion aid
        raise AssertionError("ModelNotReadyError was not raised")


def test_load_failure_is_wrapped_and_updates_health(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "load-failure", config={"fail_load": True})
    registry = build_registry(tmp_path, [entry])
    gateway = InferenceGateway(registry)

    try:
        gateway.predict("load-failure", request(tmp_path))
    except InferenceExecutionError as exc:
        assert exc.details["stage"] == "load"
        assert isinstance(exc.__cause__, RuntimeError)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("InferenceExecutionError was not raised")

    health = gateway.health()[0]
    assert health.status == ModelStatus.UNAVAILABLE
    assert "fake load failed" in (health.error_summary or "")


def test_predict_failure_is_wrapped_without_exposing_traceback(tmp_path: Path) -> None:
    entry = model_entry(tmp_path, "predict-failure", config={"fail_predict": True})
    instances: list[FakeAdapter] = []

    def resolver(_: str) -> type[FakeAdapter]:
        class TrackedFailure(FakeAdapter):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                instances.append(self)

        return TrackedFailure

    registry = build_registry(tmp_path, [entry], resolver=resolver)
    cache = AdapterCache()
    gateway = InferenceGateway(registry, cache)

    try:
        gateway.predict("predict-failure", request(tmp_path))
    except InferenceExecutionError as exc:
        assert exc.details == {
            "model_id": "predict-failure",
            "stage": "predict",
            "device": "cpu",
        }
        assert isinstance(exc.__cause__, RuntimeError)
        assert "fake predict failed" not in str(exc.details)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("InferenceExecutionError was not raised")

    assert registry.get_metadata("predict-failure").status == ModelStatus.READY
    assert cache.loaded() == []
    assert instances[0].unload_calls == 1


def test_recommendation_is_deterministic_and_uses_only_ready_models(tmp_path: Path) -> None:
    accurate = model_entry(
        tmp_path,
        "accurate-dense",
        family="sam2",
        variant="dense_particle",
        tier="accurate",
        supports_box_prompt=True,
    )
    balanced = model_entry(
        tmp_path,
        "balanced-general",
        variant="general",
        tier="balanced",
    )
    unavailable = model_entry(
        tmp_path,
        "unavailable-fast",
        variant="dense_particle",
        tier="fast",
        status="unavailable",
    )
    registry = build_registry(tmp_path, [balanced, unavailable, accurate])
    gateway = InferenceGateway(registry)
    recommendation_request = ModelRecommendationRequest(
        image_id="image-1",
        roi_mode=RoiMode.BOXES,
        target_profile=ModelVariant.DENSE_PARTICLE,
        prefer="accuracy",
    )

    first = gateway.recommend(recommendation_request)
    second = gateway.recommend(recommendation_request)

    assert first == second
    assert [candidate.model_id for candidate in first] == [
        "accurate-dense",
        "balanced-general",
    ]
    assert first[0].score > first[1].score
