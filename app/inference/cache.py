"""Thread-safe, bounded cache for lazily loaded model adapters."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Condition, RLock

from app.inference.adapters.base import SegmentationAdapter

logger = logging.getLogger(__name__)

AdapterFactory = Callable[[str], SegmentationAdapter]
CacheKey = tuple[str, str, str]
CacheEntry = tuple[CacheKey, SegmentationAdapter, RLock]


class AdapterLoadError(RuntimeError):
    """Internal marker that distinguishes model loading from prediction failures."""

    def __init__(self, model_id: str, device: str) -> None:
        super().__init__(f"failed to load adapter {model_id} on {device}")
        self.model_id = model_id
        self.device = device


class AdapterCache:
    """Keep a small LRU keyed by model, device, and immutable artifact fingerprint."""

    def __init__(self, *, max_size: int = 3) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._items: OrderedDict[CacheKey, SegmentationAdapter] = OrderedDict()
        self._prediction_locks: dict[CacheKey, RLock] = {}
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._retiring: set[CacheKey] = set()

    def get(
        self,
        model_id: str,
        device: str | None = None,
        *,
        fingerprint: str | None = None,
    ) -> SegmentationAdapter | None:
        with self._lock:
            if device is not None and fingerprint is not None:
                key = (model_id, device, fingerprint)
                adapter = self._items.get(key)
                if adapter is not None:
                    self._items.move_to_end(key)
                return adapter
            for key in reversed(self._items):
                if key[0] == model_id and (device is None or key[1] == device):
                    adapter = self._items[key]
                    self._items.move_to_end(key)
                    return adapter
        return None

    def get_or_load(
        self,
        model_id: str,
        *,
        device: str,
        factory: AdapterFactory,
        fingerprint: str = "",
    ) -> SegmentationAdapter:
        key = (model_id, device, fingerprint)
        evicted: list[CacheEntry] = []
        with self._condition:
            while key in self._retiring:
                self._condition.wait()
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached
            adapter: SegmentationAdapter | None = None
            try:
                adapter = factory(model_id)
                adapter.load(device)
            except Exception as exc:
                if adapter is not None:
                    try:
                        adapter.unload()
                    except Exception:  # pragma: no cover - defensive cleanup only
                        logger.exception("adapter cleanup failed after a load error")
                raise AdapterLoadError(model_id, device) from exc
            self._items[key] = adapter
            self._prediction_locks[key] = RLock()
            self._items.move_to_end(key)
            evicted = self._pop_evictions_locked()
        self._unload_entries(evicted, context="LRU eviction", suppress_errors=True)
        return adapter

    @contextmanager
    def lease(
        self,
        model_id: str,
        *,
        device: str,
        factory: AdapterFactory,
        fingerprint: str = "",
    ) -> Iterator[SegmentationAdapter]:
        """Serialize one mutable adapter and invalidate it if use raises.

        Prediction locks are acquired without holding the cache metadata lock.  The
        mapping is then rechecked, so a concurrent unload cannot hand a caller an
        already-unloaded adapter.  Any exception escaping the lease removes the exact
        artifact fingerprint before another waiter can reuse mutable model state.
        """

        key = (model_id, device, fingerprint)
        while True:
            adapter = self.get_or_load(
                model_id,
                device=device,
                factory=factory,
                fingerprint=fingerprint,
            )
            with self._lock:
                prediction_lock = self._prediction_locks.get(key)
                current = self._items.get(key)
            if prediction_lock is None or current is not adapter:
                continue
            prediction_lock.acquire()
            with self._lock:
                still_current = (
                    self._items.get(key) is adapter
                    and self._prediction_locks.get(key) is prediction_lock
                )
            if still_current:
                break
            prediction_lock.release()
        try:
            yield adapter
        except BaseException:
            with self._condition:
                if (
                    self._items.get(key) is adapter
                    and self._prediction_locks.get(key) is prediction_lock
                ):
                    self._items.pop(key)
                    self._prediction_locks.pop(key)
                    self._retiring.add(key)
                    invalidated: CacheEntry | None = (key, adapter, prediction_lock)
                else:
                    invalidated = None
            if invalidated is not None:
                self._unload_entries(
                    [invalidated],
                    context="invalidating a failed lease",
                    suppress_errors=True,
                )
            raise
        finally:
            prediction_lock.release()

    def loaded(self) -> list[SegmentationAdapter]:
        with self._lock:
            return list(self._items.values())

    @contextmanager
    def health_snapshot(self) -> Iterator[tuple[SegmentationAdapter, ...]]:
        """Lease the currently loaded adapters for lifecycle-safe health probes.

        The metadata lock is never held while waiting for a prediction lock or while adapter code
        runs. Entries are acquired in cache-key order and revalidated after acquisition. A
        concurrent unload may retire an entry first, in which case it is omitted after unloading;
        otherwise unload waits until the caller leaves this context.
        """

        with self._lock:
            candidates = sorted(
                (
                    (key, adapter, self._prediction_locks[key])
                    for key, adapter in self._items.items()
                ),
                key=lambda entry: entry[0],
            )

        acquired: list[RLock] = []
        adapters: list[SegmentationAdapter] = []
        try:
            for key, adapter, prediction_lock in candidates:
                prediction_lock.acquire()
                with self._lock:
                    still_current = (
                        self._items.get(key) is adapter
                        and self._prediction_locks.get(key) is prediction_lock
                        and key not in self._retiring
                    )
                if not still_current:
                    prediction_lock.release()
                    continue
                acquired.append(prediction_lock)
                adapters.append(adapter)
            yield tuple(adapters)
        finally:
            for prediction_lock in reversed(acquired):
                prediction_lock.release()

    def unload(
        self,
        model_id: str,
        *,
        device: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        with self._condition:
            keys = [
                key
                for key in self._items
                if key[0] == model_id
                and (device is None or key[1] == device)
                and (fingerprint is None or key[2] == fingerprint)
            ]
            entries = self._retire_keys_locked(keys)
        self._unload_entries(entries, context="explicit unload", suppress_errors=False)

    def clear(self) -> None:
        with self._condition:
            entries = self._retire_keys_locked(list(self._items))
        self._unload_entries(entries, context="clearing the cache", suppress_errors=True)

    def _pop_evictions_locked(self) -> list[CacheEntry]:
        entries: list[CacheEntry] = []
        while len(self._items) > self.max_size:
            key, adapter = self._items.popitem(last=False)
            prediction_lock = self._prediction_locks.pop(key)
            self._retiring.add(key)
            entries.append((key, adapter, prediction_lock))
        return entries

    def _retire_keys_locked(self, keys: list[CacheKey]) -> list[CacheEntry]:
        entries: list[CacheEntry] = []
        for key in keys:
            adapter = self._items.pop(key, None)
            prediction_lock = self._prediction_locks.pop(key, None)
            if adapter is None or prediction_lock is None:
                continue
            self._retiring.add(key)
            entries.append((key, adapter, prediction_lock))
        return entries

    def _unload_entries(
        self,
        entries: list[CacheEntry],
        *,
        context: str,
        suppress_errors: bool,
    ) -> None:
        first_error: Exception | None = None
        for key, adapter, prediction_lock in entries:
            try:
                with prediction_lock:
                    adapter.unload()
            except Exception as error:  # lifecycle cleanup must always release retirement
                if suppress_errors:
                    logger.exception("adapter cleanup failed while %s", context)
                elif first_error is None:
                    first_error = error
            finally:
                with self._condition:
                    self._retiring.discard(key)
                    self._condition.notify_all()
        if first_error is not None:
            raise first_error
