from __future__ import annotations

import io
import json
import logging

from app.core.logging import JsonFormatter, log_context


def test_json_logging_includes_only_bound_context_and_selected_extras() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.contract.logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    with log_context(request_id="req_test", job_id="job_1"):
        logger.info(
            "contract_event",
            extra={"event": "tested", "status_code": 200, "secret": "must-not-leak"},
        )

    record = json.loads(stream.getvalue())
    assert record["request_id"] == "req_test"
    assert record["job_id"] == "job_1"
    assert record["event"] == "tested"
    assert record["status_code"] == 200
    assert "secret" not in record
