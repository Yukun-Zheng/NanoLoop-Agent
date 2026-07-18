from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from time import sleep

from app.contracts.enums import ModelFamily, ModelStatus, ModelVariant, QualityTier
from app.contracts.models import ModelMetadata
from app.inference.cache import AdapterCache, AdapterLoadError
from tests.unit.inference.fakes import FakeAdapter


def metadata(model_id: str) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        family=ModelFamily.UNET,
        variant=ModelVariant.GENERAL,
        quality_tier=QualityTier.BALANCED,
        version="test",
        status=ModelStatus.READY,
        supports_box_prompt=False,
        preprocess_profile="fixture",
        postprocess_profile="fixture",
    )


def fake(model_id: str, *, fail_load: bool = False) -> FakeAdapter:
    return FakeAdapter(
        metadata=metadata(model_id),
        weight_path=Path("fixture.weights"),
        config={"fail_load": fail_load},
    )


def test_cache_loads_once_and_reuses_adapter() -> None:
    cache = AdapterCache()
    created: list[FakeAdapter] = []

    def factory(model_id: str) -> FakeAdapter:
        adapter = fake(model_id)
        created.append(adapter)
        return adapter

    first = cache.get_or_load("model", device="cpu", factory=factory)
    second = cache.get_or_load("model", device="cpu", factory=factory)

    assert first is second
    assert len(created) == 1
    assert created[0].load_calls == 1


def test_lru_eviction_and_clear_unload_adapters() -> None:
    cache = AdapterCache(max_size=1)
    first = fake("first")
    second = fake("second")
    cache.get_or_load("first", device="cpu", factory=lambda _: first)
    cache.get_or_load("second", device="cpu", factory=lambda _: second)

    assert first.unload_calls == 1
    assert cache.get("first", "cpu") is None

    cache.clear()

    assert second.unload_calls == 1
    assert cache.loaded() == []


def test_failed_load_is_not_cached_and_preserves_cause() -> None:
    cache = AdapterCache()
    adapter = fake("broken", fail_load=True)

    try:
        cache.get_or_load("broken", device="cpu", factory=lambda _: adapter)
    except AdapterLoadError as exc:
        assert isinstance(exc.__cause__, RuntimeError)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("AdapterLoadError was not raised")

    assert cache.get("broken", "cpu") is None
    assert adapter.unload_calls == 1


def test_artifact_fingerprint_separates_cached_adapters() -> None:
    cache = AdapterCache()
    created: list[FakeAdapter] = []

    def factory(model_id: str) -> FakeAdapter:
        adapter = fake(model_id)
        created.append(adapter)
        return adapter

    first = cache.get_or_load(
        "model", device="cpu", factory=factory, fingerprint="artifact-a"
    )
    second = cache.get_or_load(
        "model", device="cpu", factory=factory, fingerprint="artifact-b"
    )

    assert first is not second
    assert len(created) == 2
    assert cache.get("model", "cpu", fingerprint="artifact-a") is first
    assert cache.get("model", "cpu", fingerprint="artifact-b") is second
    assert cache.get("model", "cpu") is second

    cache.unload("model", device="cpu", fingerprint="artifact-a")

    assert first.unload_calls == 1
    assert second.unload_calls == 0
    assert cache.get("model", "cpu", fingerprint="artifact-a") is None


def test_lease_serializes_prediction_for_one_cached_adapter() -> None:
    cache = AdapterCache()
    adapter = fake("model")
    worker_started = Event()
    worker_acquired = Event()

    def use_adapter() -> None:
        worker_started.set()
        with cache.lease("model", device="cpu", factory=lambda _: adapter):
            worker_acquired.set()

    with cache.lease("model", device="cpu", factory=lambda _: adapter):
        worker = Thread(target=use_adapter)
        worker.start()
        assert worker_started.wait(timeout=1)
        assert not worker_acquired.wait(timeout=0.05)

    assert worker_acquired.wait(timeout=1)
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert adapter.load_calls == 1


def test_unload_waits_for_active_lease() -> None:
    cache = AdapterCache()
    adapter = fake("model")
    unload_started = Event()
    unload_finished = Event()

    def unload_adapter() -> None:
        unload_started.set()
        cache.unload("model", device="cpu")
        unload_finished.set()

    with cache.lease("model", device="cpu", factory=lambda _: adapter):
        worker = Thread(target=unload_adapter)
        worker.start()
        assert unload_started.wait(timeout=1)
        assert not unload_finished.wait(timeout=0.05)
        assert adapter.unload_calls == 0

    assert unload_finished.wait(timeout=1)
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert adapter.unload_calls == 1
    assert cache.get("model", "cpu") is None


def test_exception_escaping_lease_invalidates_exact_fingerprint() -> None:
    cache = AdapterCache()
    first = fake("model")

    try:
        with cache.lease(
            "model",
            device="cpu",
            factory=lambda _: first,
            fingerprint="artifact-a",
        ):
            raise RuntimeError("mutable adapter state is now unknown")
    except RuntimeError:
        pass
    else:  # pragma: no cover - assertion aid
        raise AssertionError("RuntimeError was not raised")

    assert first.unload_calls == 1
    assert cache.get("model", "cpu", fingerprint="artifact-a") is None

    second = fake("model")
    with cache.lease(
        "model",
        device="cpu",
        factory=lambda _: second,
        fingerprint="artifact-a",
    ) as reloaded:
        assert reloaded is second

    assert second.load_calls == 1


def test_retiring_active_adapter_blocks_same_fingerprint_reload() -> None:
    cache = AdapterCache()
    first = fake("model")
    second = fake("model")
    unload_finished = Event()
    replacement_loaded = Event()
    replacement_used: list[FakeAdapter] = []

    def unload_active() -> None:
        cache.unload("model", device="cpu", fingerprint="artifact-a")
        unload_finished.set()

    def use_replacement() -> None:
        with cache.lease(
            "model",
            device="cpu",
            factory=lambda _: (replacement_loaded.set(), second)[1],
            fingerprint="artifact-a",
        ) as adapter:
            replacement_used.append(adapter)  # type: ignore[arg-type]

    with cache.lease(
        "model",
        device="cpu",
        factory=lambda _: first,
        fingerprint="artifact-a",
    ):
        unloader = Thread(target=unload_active)
        unloader.start()
        for _ in range(100):
            if cache.get("model", "cpu", fingerprint="artifact-a") is None:
                break
            sleep(0.001)
        assert cache.get("model", "cpu", fingerprint="artifact-a") is None

        replacement = Thread(target=use_replacement)
        replacement.start()
        assert not replacement_loaded.wait(timeout=0.05)
        assert not unload_finished.is_set()

    assert unload_finished.wait(timeout=1)
    assert replacement_loaded.wait(timeout=1)
    unloader.join(timeout=1)
    replacement.join(timeout=1)
    assert not unloader.is_alive()
    assert not replacement.is_alive()
    assert replacement_used == [second]
    assert first.unload_calls == 1
