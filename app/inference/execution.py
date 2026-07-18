"""Resolved device and process-wide deterministic inference controls."""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from threading import RLock
from typing import Any

import numpy as np

_DETERMINISTIC_INFERENCE_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class DeterminismControls:
    seed: int
    python_random_seeded: bool
    numpy_random_seeded: bool
    torch_deterministic_algorithms: bool
    global_inference_serialized: bool = True


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` once before adapter cache/load and preserve explicit choices."""

    if requested != "auto":
        return requested
    torch = _load_torch()
    if torch is None:
        return "cpu"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    return "cpu"


@contextmanager
def deterministic_inference(seed: int, *, device: str) -> Iterator[DeterminismControls]:
    """Seed, serialize, enforce deterministic Torch ops, then restore global RNG state.

    Python, NumPy, and Torch expose process-global RNG/configuration. Serializing this
    small boundary prevents different model adapters from changing each other's seeds.
    Torch's strict deterministic mode raises if an operation has no deterministic
    implementation instead of silently returning a non-reproducible result.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    with _DETERMINISTIC_INFERENCE_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch = _load_torch()
        try:
            if torch is None:
                yield DeterminismControls(
                    seed=seed,
                    python_random_seeded=True,
                    numpy_random_seeded=True,
                    torch_deterministic_algorithms=False,
                )
                return
            with _torch_deterministic_context(torch, seed=seed, device=device):
                yield DeterminismControls(
                    seed=seed,
                    python_random_seeded=True,
                    numpy_random_seeded=True,
                    torch_deterministic_algorithms=True,
                )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@contextmanager
def _torch_deterministic_context(
    torch: Any,
    *,
    seed: int,
    device: str,
) -> Iterator[None]:
    cuda = getattr(torch, "cuda", None)
    cuda_devices: list[int] = []
    cublas_key = "CUBLAS_WORKSPACE_CONFIG"
    previous_cublas = os.environ.get(cublas_key)
    if device == "cuda" and cuda is not None and bool(cuda.is_available()):
        os.environ.setdefault(cublas_key, ":4096:8")
        cuda_devices = list(range(int(cuda.device_count())))
    fork_rng = getattr(getattr(torch, "random", None), "fork_rng", None)
    if not callable(fork_rng):
        raise RuntimeError("Torch runtime does not expose random.fork_rng")

    previous_enabled = bool(torch.are_deterministic_algorithms_enabled())
    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    previous_warn_only = bool(warn_only_getter()) if callable(warn_only_getter) else False
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    previous_cudnn_deterministic = getattr(cudnn, "deterministic", None)
    previous_cudnn_benchmark = getattr(cudnn, "benchmark", None)

    try:
        with fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if cuda_devices:
                assert cuda is not None
                cuda.manual_seed_all(seed)
            mps = getattr(torch, "mps", None)
            mps_manual_seed = getattr(mps, "manual_seed", None)
            if device == "mps" and callable(mps_manual_seed):
                mps_manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=False)
            if cudnn is not None:
                cudnn.deterministic = True
                cudnn.benchmark = False
            try:
                yield
            finally:
                if cudnn is not None:
                    if previous_cudnn_deterministic is not None:
                        cudnn.deterministic = previous_cudnn_deterministic
                    if previous_cudnn_benchmark is not None:
                        cudnn.benchmark = previous_cudnn_benchmark
                torch.use_deterministic_algorithms(
                    previous_enabled,
                    warn_only=previous_warn_only,
                )
    finally:
        if cuda_devices:
            if previous_cublas is None:
                os.environ.pop(cublas_key, None)
            else:
                os.environ[cublas_key] = previous_cublas


def _load_torch() -> Any | None:
    try:
        if find_spec("torch") is None:
            return None
        return importlib.import_module("torch")
    except (ImportError, ValueError):
        return None


__all__ = ["DeterminismControls", "deterministic_inference", "resolve_device"]
