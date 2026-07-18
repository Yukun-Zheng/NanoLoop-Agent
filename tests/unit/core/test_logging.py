from __future__ import annotations

import io
import logging

from app.core.logging import JsonFormatter, configure_logging


def test_configure_logging_is_idempotent_and_preserves_external_handlers() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    external = logging.StreamHandler(io.StringIO())
    try:
        root.addHandler(external)

        configure_logging("INFO", stream=io.StringIO())
        configure_logging("WARNING", stream=io.StringIO())

        assert external in root.handlers
        owned = [
            handler
            for handler in root.handlers
            if getattr(handler, "_nanoloop_json_handler", False)
        ]
        assert len(owned) == 1
        assert isinstance(owned[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
    finally:
        for handler in tuple(root.handlers):
            root.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
