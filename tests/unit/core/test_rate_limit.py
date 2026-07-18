from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from app.core.rate_limit import TokenBucketLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_limiter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TokenBucketLimiter(0, 60)
    with pytest.raises(ValueError, match="capacity"):
        TokenBucketLimiter(True, 60)
    with pytest.raises(ValueError, match="window_seconds"):
        TokenBucketLimiter(1, 0)
    with pytest.raises(ValueError, match="window_seconds"):
        TokenBucketLimiter(1, float("inf"))


def test_bucket_exhaustion_and_refill_are_deterministic() -> None:
    clock = FakeClock()
    limiter = TokenBucketLimiter(2, 10, clock=clock)

    first = limiter.consume("service")
    second = limiter.consume("service")
    rejected = limiter.consume("service")

    assert (first.allowed, first.limit, first.remaining, first.retry_after_seconds) == (
        True,
        2,
        1,
        None,
    )
    assert second.allowed is True
    assert second.remaining == 0
    assert rejected.allowed is False
    assert rejected.remaining == 0
    assert rejected.retry_after_seconds == 5

    clock.advance(4)
    almost_ready = limiter.consume("service")
    assert almost_ready.allowed is False
    assert almost_ready.retry_after_seconds == 1

    clock.advance(1)
    refilled = limiter.consume("service")
    assert refilled.allowed is True
    assert refilled.remaining == 0

    clock.advance(100)
    capped = limiter.consume("service")
    assert capped.allowed is True
    assert capped.remaining == 1


def test_decision_is_immutable() -> None:
    decision = TokenBucketLimiter(1, 60, clock=lambda: 0.0).consume("service")

    with pytest.raises(FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]


def test_only_three_fixed_buckets_can_be_allocated() -> None:
    limiter = TokenBucketLimiter(1, 60, clock=lambda: 0.0)

    for bucket in ("authenticated", "anonymous", "service"):
        limiter.consume(bucket)
    assert limiter.bucket_count == 3

    with pytest.raises(ValueError, match="unsupported"):
        limiter.consume(cast(Any, "attacker-controlled"))
    assert limiter.bucket_count == 3


def test_concurrent_consumers_cannot_exceed_capacity() -> None:
    limiter = TokenBucketLimiter(40, 60, clock=lambda: 0.0)

    with ThreadPoolExecutor(max_workers=16) as executor:
        decisions = list(executor.map(lambda _: limiter.consume("authenticated"), range(200)))

    assert sum(decision.allowed for decision in decisions) == 40
    assert limiter.bucket_count == 1


def test_clock_must_remain_finite() -> None:
    limiter = TokenBucketLimiter(1, 60, clock=lambda: float("nan"))

    with pytest.raises(ValueError, match="clock"):
        limiter.consume("service")
    assert limiter.bucket_count == 0
