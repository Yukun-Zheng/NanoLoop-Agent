from __future__ import annotations

import json

from httpx import Response

from tests.integration.api_contract.conftest import ApiHarness


def _assert_multipart_rejection(response: Response, *, reason: str) -> None:
    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["data"] is None
    assert payload["request_id"] == response.headers["x-request-id"]
    assert payload["error"]["code"] == "INVALID_MULTIPART"
    assert payload["error"]["details"]["reason"] == reason


def _analysis_metadata(filename: str = "sample.png") -> str:
    return json.dumps(
        {
            "job_name": "multipart limits",
            "images": [
                {
                    "filename": filename,
                    "sample_id": "sample_1",
                    "scale": {"mode": "pixel_only"},
                }
            ],
        }
    )


def test_analysis_rejects_twenty_first_file_before_endpoint_binding(
    api_harness: ApiHarness,
) -> None:
    files = [
        ("files", (f"sample-{index}.png", b"x", "image/png"))
        for index in range(21)
    ]

    response = api_harness.client.post(
        "/api/v1/analyses",
        files=files,
        data={"metadata_json": _analysis_metadata()},
    )

    _assert_multipart_rejection(response, reason="parser_rejected")
    assert response.json()["error"]["details"]["max_files"] == 20


def test_analysis_rejects_unknown_part_even_within_total_count(
    api_harness: ApiHarness,
) -> None:
    response = api_harness.client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("sample.png", b"x", "image/png")),
            ("unexpected", (None, "value")),
        ],
        data={"metadata_json": _analysis_metadata()},
    )

    _assert_multipart_rejection(response, reason="parser_rejected")
    assert response.json()["error"]["details"]["max_fields"] == 1


def test_analysis_rejects_oversized_non_file_part(api_harness: ApiHarness) -> None:
    response = api_harness.client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("sample.png", b"x", "image/png")),
            ("metadata_json", (None, "x" * (256 * 1024 + 1))),
        ],
    )

    _assert_multipart_rejection(response, reason="parser_rejected")
    assert response.json()["error"]["details"]["max_text_part_bytes"] == 256 * 1024


def test_knowledge_ingestion_rejects_second_file(api_harness: ApiHarness) -> None:
    response = api_harness.client.post(
        "/api/v1/knowledge/documents",
        files=[
            ("file", ("first.pdf", b"first", "application/pdf")),
            ("file", ("second.pdf", b"second", "application/pdf")),
        ],
        data={"metadata_json": "{}"},
    )

    _assert_multipart_rejection(response, reason="parser_rejected")
    assert response.json()["error"]["details"]["max_files"] == 1


def test_corrected_mask_rejects_any_text_field(api_harness: ApiHarness) -> None:
    response = api_harness.client.post(
        "/api/v1/runs/run_1/corrected-mask",
        files=[
            ("file", ("mask.png", b"mask", "image/png")),
            ("metadata_json", (None, "{}")),
        ],
    )

    _assert_multipart_rejection(response, reason="parser_rejected")
    assert response.json()["error"]["details"]["max_fields"] == 0


def test_analysis_rejects_single_file_under_wrong_field_name(
    api_harness: ApiHarness,
) -> None:
    response = api_harness.client.post(
        "/api/v1/analyses",
        files={"wrong": ("sample.png", b"x", "image/png")},
        data={"metadata_json": _analysis_metadata()},
    )

    _assert_multipart_rejection(response, reason="unexpected_part")
