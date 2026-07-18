"""Safe invocation helpers for optional sync or async application providers."""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.core.errors import ApiNotImplementedError


async def invoke(provider: object, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(provider, method_name, None)
    if not callable(method):
        raise ApiNotImplementedError(
            details={"provider": type(provider).__name__, "method": method_name}
        )
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    result = await run_in_threadpool(partial(method, *args, **kwargs))
    if inspect.isawaitable(result):
        return await result
    return result
