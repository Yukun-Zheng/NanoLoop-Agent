from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from app.core.rate_limit import BoundedKeyedTokenBucketLimiter, TokenBucketLimiter


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


def test_keyed_limiter_rejects_invalid_configuration_and_keys() -> None:
    with pytest.raises(ValueError, match="capacity"):
        BoundedKeyedTokenBucketLimiter(0, 60, max_buckets=1)
    with pytest.raises(ValueError, match="window_seconds"):
        BoundedKeyedTokenBucketLimiter(1, float("inf"), max_buckets=1)
    with pytest.raises(ValueError, match="max_buckets"):
        BoundedKeyedTokenBucketLimiter(1, 60, max_buckets=0)

    limiter = BoundedKeyedTokenBucketLimiter(1, 60, max_buckets=1)
    with pytest.raises(ValueError, match="key"):
        limiter.consume("")
    with pytest.raises(ValueError, match="key"):
        limiter.consume("x" * 257)
    assert limiter.bucket_count == 0


def test_keyed_limiter_isolated_buckets_refill_independently() -> None:
    clock = FakeClock()
    limiter = BoundedKeyedTokenBucketLimiter(1, 10, max_buckets=4, clock=clock)

    assert limiter.consume("peer:192.0.2.1").allowed is True
    assert limiter.consume("peer:192.0.2.1").allowed is False
    assert limiter.consume("peer:192.0.2.2").allowed is True

    clock.advance(10)
    assert limiter.consume("peer:192.0.2.1").allowed is True
    assert limiter.consume("peer:192.0.2.2").allowed is True
    assert limiter.bucket_count == 2


def test_keyed_limiter_evicts_the_least_recently_used_bucket() -> None:
    limiter = BoundedKeyedTokenBucketLimiter(
        1,
        60,
        max_buckets=2,
        clock=lambda: 0.0,
    )

    assert limiter.consume("peer:a").allowed is True
    assert limiter.consume("peer:b").allowed is True
    assert limiter.consume("peer:a").allowed is False  # refresh a; b is now LRU
    assert limiter.consume("peer:c").allowed is True

    assert limiter.bucket_count == 2
    assert limiter.consume("peer:a").allowed is False
    assert limiter.consume("peer:b").allowed is True  # b was evicted and starts fresh
    assert limiter.bucket_count == 2


def test_keyed_limiter_remains_strictly_bounded_under_key_churn() -> None:
    limiter = BoundedKeyedTokenBucketLimiter(
        1,
        60,
        max_buckets=32,
        clock=lambda: 0.0,
    )

    for index in range(10_000):
        assert limiter.consume(f"peer:{index}").allowed is True
        assert limiter.bucket_count <= limiter.max_buckets

    assert limiter.bucket_count == 32


def test_keyed_limiter_is_thread_safe_for_same_and_distinct_keys() -> None:
    limiter = BoundedKeyedTokenBucketLimiter(
        40,
        60,
        max_buckets=16,
        clock=lambda: 0.0,
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        same_key = list(
            executor.map(lambda _: limiter.consume("principal:one"), range(200))
        )
    assert sum(decision.allowed for decision in same_key) == 40

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda index: limiter.consume(f"peer:{index}"), range(1_000)))
    assert limiter.bucket_count == 16
