"""Deterministic character budgets for small-model Agent context."""

from __future__ import annotations

import json
from typing import Any


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def bounded_mapping(value: dict[str, Any], max_chars: int) -> dict[str, Any]:
    encoded = compact_json(value)
    if len(encoded) <= max_chars:
        return value
    marker: dict[str, Any] = {
        "truncated": True,
        "available_keys": sorted(str(key) for key in value)[:40],
        "preview": "",
    }
    if len(compact_json(marker)) > max_chars:
        return {"truncated": True}
    low = 0
    high = len(encoded)
    while low < high:
        middle = (low + high + 1) // 2
        marker["preview"] = encoded[:middle]
        if len(compact_json(marker)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    marker["preview"] = encoded[:low]
    return marker


def bounded_newest_mappings(
    values: list[dict[str, Any]],
    *,
    max_items: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Keep newest mappings inside one shared serialized-character budget."""

    selected: list[dict[str, Any]] = []
    for item in reversed(values[-max_items:]):
        candidate = [item, *selected]
        if len(compact_json(candidate)) <= max_chars:
            selected = candidate
            continue
        if not selected:
            selected = [bounded_mapping(item, max(1, max_chars - 2))]
        break
    return selected


def bounded_texts(
    values: list[str],
    *,
    max_items: int,
    max_chars: int,
    keep_newest: bool,
) -> list[str]:
    """Keep ordered text items within one shared serialized-character budget."""

    source = values[-max_items:] if keep_newest else values[:max_items]
    selected: list[str] = []
    iterable = reversed(source) if keep_newest else iter(source)
    for item in iterable:
        candidate = [item, *selected] if keep_newest else [*selected, item]
        if len(compact_json(candidate)) <= max_chars:
            selected = candidate
            continue
        if not selected:
            selected = [_bounded_text(item, max(1, max_chars - 2))]
        break
    return selected


def _bounded_text(value: str, max_json_chars: int) -> str:
    if len(compact_json(value)) <= max_json_chars:
        return value
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(compact_json(value[:middle])) <= max_json_chars:
            low = middle
        else:
            high = middle - 1
    return value[:low]
