from __future__ import annotations

import random
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
from pytest import MonkeyPatch

from app.inference import execution


def test_explicit_device_is_preserved_without_loading_torch() -> None:
    assert execution.resolve_device("cpu") == "cpu"
    assert execution.resolve_device("cuda") == "cuda"
    assert execution.resolve_device("mps") == "mps"


def test_auto_device_falls_back_to_cpu_without_torch(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "_load_torch", lambda: None)

    assert execution.resolve_device("auto") == "cpu"


def test_deterministic_context_repeats_values_and_restores_rng_state(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "_load_torch", lambda: None)
    random.seed(999)
    np.random.seed(999)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    with execution.deterministic_inference(42, device="cpu") as first_controls:
        first = (random.random(), float(np.random.random()))
    with execution.deterministic_inference(42, device="cpu") as second_controls:
        second = (random.random(), float(np.random.random()))

    assert first == second
    assert first_controls.seed == 42
    assert first_controls.torch_deterministic_algorithms is False
    assert second_controls.global_inference_serialized is True
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]


def test_torch_controls_and_cuda_environment_are_restored(
    monkeypatch: MonkeyPatch,
) -> None:
    deterministic_calls: list[tuple[bool, bool]] = []
    cuda_seed_calls: list[int] = []

    @contextmanager
    def fork_rng(**_kwargs: object) -> Any:
        yield

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        manual_seed_all=cuda_seed_calls.append,
    )
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    fake_torch = SimpleNamespace(
        cuda=cuda,
        random=SimpleNamespace(fork_rng=fork_rng),
        backends=SimpleNamespace(cudnn=cudnn),
        are_deterministic_algorithms_enabled=lambda: False,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        manual_seed=lambda _seed: None,
        use_deterministic_algorithms=lambda enabled, warn_only: (
            deterministic_calls.append((enabled, warn_only))
        ),
    )
    monkeypatch.setattr(execution, "_load_torch", lambda: fake_torch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with execution.deterministic_inference(17, device="cuda") as controls:
        assert controls.torch_deterministic_algorithms is True
        assert cudnn.deterministic is True
        assert cudnn.benchmark is False
        assert execution.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    assert cuda_seed_calls == [17]
    assert deterministic_calls == [(True, False), (False, False)]
    assert cudnn.deterministic is False
    assert cudnn.benchmark is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in execution.os.environ
