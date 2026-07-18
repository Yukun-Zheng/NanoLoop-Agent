"""Constant-memory token-bucket primitives for the single-process API."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Literal, TypeAlias

RateLimitBucket: TypeAlias = Literal["authenticated", "anonymous", "service"]
_BUCKET_NAMES = frozenset({"authenticated", "anonymous", "service"})


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int | None


@dataclass(slots=True)
class _BucketState:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    """Apply one bounded token bucket to each of three fixed caller classes."""

    def __init__(
        self,
        capacity: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be positive")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self._capacity = capacity
        self._window_seconds = float(window_seconds)
        self._refill_per_second = capacity / self._window_seconds
        self._clock = clock
        self._buckets: dict[RateLimitBucket, _BucketState] = {}
        self._lock = Lock()

    @property
    def bucket_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def consume(self, bucket: RateLimitBucket) -> RateLimitDecision:
        if bucket not in _BUCKET_NAMES:
            raise ValueError("unsupported rate-limit bucket")
        now = self._clock()
        if not math.isfinite(now):
            raise ValueError("clock must return a finite value")

        with self._lock:
            state = self._buckets.get(bucket)
            if state is None:
                state = _BucketState(tokens=float(self._capacity), updated_at=now)
                self._buckets[bucket] = state
            else:
                elapsed = max(0.0, now - state.updated_at)
                state.tokens = min(
                    float(self._capacity),
                    state.tokens + elapsed * self._refill_per_second,
                )
                state.updated_at = max(state.updated_at, now)

            if state.tokens + 1e-12 >= 1.0:
                state.tokens = max(0.0, state.tokens - 1.0)
                return RateLimitDecision(
                    allowed=True,
                    limit=self._capacity,
                    remaining=min(self._capacity, math.floor(state.tokens)),
                    retry_after_seconds=None,
                )

            retry_after = max(
                1,
                math.ceil((1.0 - state.tokens) / self._refill_per_second),
            )
            return RateLimitDecision(
                allowed=False,
                limit=self._capacity,
                remaining=0,
                retry_after_seconds=retry_after,
            )


class BoundedKeyedTokenBucketLimiter:
    """Apply isolated token buckets while retaining at most ``max_buckets`` keys.

    Keys must be derived by trusted application code, never copied from bearer credentials. When
    the bound is reached, the least-recently-used bucket is evicted. Eviction deliberately favors
    availability over creating a shared overflow bucket that one caller could exhaust for every
    other caller.
    """

    def __init__(
        self,
        capacity: int,
        window_seconds: float,
        *,
        max_buckets: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be positive")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        if (
            isinstance(max_buckets, bool)
            or not isinstance(max_buckets, int)
            or max_buckets <= 0
        ):
            raise ValueError("max_buckets must be positive")
        self._capacity = capacity
        self._window_seconds = float(window_seconds)
        self._refill_per_second = capacity / self._window_seconds
        self._max_buckets = max_buckets
        self._clock = clock
        self._buckets: OrderedDict[str, _BucketState] = OrderedDict()
        self._lock = Lock()

    @property
    def bucket_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    @property
    def max_buckets(self) -> int:
        return self._max_buckets

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def consume(self, key: str) -> RateLimitDecision:
        """Consume one token for a bounded, application-derived key."""

        if not isinstance(key, str) or not key or len(key) > 256:
            raise ValueError("rate-limit key must contain 1-256 characters")
        now = self._clock()
        if not math.isfinite(now):
            raise ValueError("clock must return a finite value")

        with self._lock:
            state = self._buckets.get(key)
            if state is None:
                if len(self._buckets) >= self._max_buckets:
                    self._buckets.popitem(last=False)
                state = _BucketState(tokens=float(self._capacity), updated_at=now)
                self._buckets[key] = state
            else:
                self._buckets.move_to_end(key)
                elapsed = max(0.0, now - state.updated_at)
                state.tokens = min(
                    float(self._capacity),
                    state.tokens + elapsed * self._refill_per_second,
                )
                state.updated_at = max(state.updated_at, now)

            if state.tokens + 1e-12 >= 1.0:
                state.tokens = max(0.0, state.tokens - 1.0)
                return RateLimitDecision(
                    allowed=True,
                    limit=self._capacity,
                    remaining=min(self._capacity, math.floor(state.tokens)),
                    retry_after_seconds=None,
                )

            retry_after = max(
                1,
                math.ceil((1.0 - state.tokens) / self._refill_per_second),
            )
            return RateLimitDecision(
                allowed=False,
                limit=self._capacity,
                remaining=0,
                retry_after_seconds=retry_after,
            )
