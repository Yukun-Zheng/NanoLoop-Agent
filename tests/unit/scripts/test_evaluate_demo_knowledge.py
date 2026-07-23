from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_demo_knowledge import (
    apply_evaluation_contract,
    build_parser,
    evaluate_one,
    load_questions,
    load_scope_map,
    normalize_api_base,
)


def _question(*, query_type: str = "material_knowledge") -> dict[str, object]:
    return {
        "query_id": "q001",
        "query_type": query_type,
        "expected_outcome": "OK",
        "expected_answer_contains_any": ["supported"],
        "require_citations": True,
        "expected_asset_ids": ["asset_relevant"],
    }


def _response(*, doc_id: str = "doc_relevant") -> dict[str, object]:
    return {
        "query_type": "material_knowledge",
        "outcome_code": "OK",
        "confidence": "medium",
        "answer": "supported answer [C1]",
        "citations": [
            {
                "citation_id": "C1",
                "doc_id": doc_id,
                "chunk_id": "chunk_1",
                "excerpt": "supported evidence",
                "retrieval_score": 0.8,
            }
        ],
        "data_evidence": [],
        "limitations": [],
        "needs_clarification": False,
    }


def test_evaluator_binds_citations_to_expected_curated_assets() -> None:
    passed = evaluate_one(
        _question(),
        _response(),
        request_id="req_1",
        asset_to_doc={"asset_relevant": "doc_relevant"},
    )
    wrong_source = evaluate_one(
        _question(),
        _response(doc_id="doc_wrong"),
        request_id="req_2",
        asset_to_doc={"asset_relevant": "doc_relevant"},
    )

    assert passed["passed"] is True
    assert wrong_source["passed"] is False
    assert any("expected curated sources" in error for error in wrong_source["errors"])


def test_mixed_evaluator_requires_numeric_evidence_and_causal_boundary() -> None:
    question = _question(query_type="mixed")
    response = _response()
    response["query_type"] = "mixed"

    result = evaluate_one(
        question,
        response,
        request_id="req_1",
        asset_to_doc={"asset_relevant": "doc_relevant"},
    )

    assert result["passed"] is False
    assert "mixed answer requires data_evidence" in result["errors"]
    assert any("causal boundary" in error for error in result["errors"])


def _comparison_question() -> dict[str, object]:
    return {
        **_question(query_type="mixed"),
        "query_id": "q027",
        "evidence_requirement": {
            "tool_name": "compare_groups",
            "validated_arguments": {
                "intent": "compare_groups",
                "metric": "number_density_um2",
                "group_by": "sample",
                "order": "desc",
            },
            "required_units": {
                "number_density_um2": "um^-2",
                "value": "um^-2",
            },
            "required_numeric_fields": ["value"],
            "min_source_runs": 2,
            "min_rows": 2,
            "aggregate_equals_row_count": "group_count",
            "required_row_values": {
                "metric": "number_density_um2",
                "group_by": "sample",
            },
            "distinct_row_field": "group",
            "sort": {"field": "value", "order": "desc"},
            "require_row_source_coverage": True,
        },
    }


def _comparison_response() -> dict[str, object]:
    response = _response()
    response["query_type"] = "mixed"
    response["limitations"] = [
        "文献中的一般规律不能直接证明当前样品的因果机理。"
    ]
    response["data_evidence"] = [
        {
            "tool_name": "compare_groups",
            "validated_arguments": {
                "intent": "compare_groups",
                "metric": "number_density_um2",
                "group_by": "sample",
                "order": "desc",
                "run_ids": ["run_1", "run_2"],
            },
            "rows": [
                {
                    "group": "sample_b",
                    "group_by": "sample",
                    "metric": "number_density_um2",
                    "value": 12.0,
                    "run_ids": ["run_2"],
                },
                {
                    "group": "sample_a",
                    "group_by": "sample",
                    "metric": "number_density_um2",
                    "value": 4.0,
                    "run_ids": ["run_1"],
                },
            ],
            "aggregates": {"group_count": 2},
            "units": {
                "number_density_um2": "um^-2",
                "value": "um^-2",
            },
            "source_run_ids": ["run_1", "run_2"],
        }
    ]
    return response


def test_mixed_evidence_contract_checks_metric_units_scope_and_sorting() -> None:
    result = evaluate_one(
        _comparison_question(),
        _comparison_response(),
        request_id="req_1",
        asset_to_doc={"asset_relevant": "doc_relevant"},
        request_scope={"run_ids": ["run_1", "run_2"]},
    )

    assert result["passed"] is True


def test_mixed_evidence_contract_rejects_non_semantic_placeholder_data() -> None:
    response = _comparison_response()
    response["data_evidence"] = [
        {
            "tool_name": "anything",
            "validated_arguments": {
                "intent": "anything",
                "metric": "anything",
                "group_by": "sample",
                "order": "desc",
                "run_ids": ["run_fake"],
            },
            "rows": [
                {
                    "group": "same",
                    "group_by": "sample",
                    "metric": "anything",
                    "value": "high",
                    "run_ids": ["run_fake"],
                },
                {
                    "group": "same",
                    "group_by": "sample",
                    "metric": "anything",
                    "value": "low",
                    "run_ids": ["run_fake"],
                },
            ],
            "aggregates": {"value": "high"},
            "units": {"number_density_um2": "words", "value": "words"},
            "source_run_ids": ["run_fake"],
        }
    ]

    result = evaluate_one(
        _comparison_question(),
        response,
        request_id="req_1",
        asset_to_doc={"asset_relevant": "doc_relevant"},
        request_scope={"run_ids": ["run_1", "run_2"]},
    )

    assert result["passed"] is False
    errors = "\n".join(result["errors"])
    assert "evidence tool expected compare_groups" in errors
    assert "validated argument metric" in errors
    assert "unit for value expected" in errors
    assert "lacks finite numeric field value" in errors
    assert "do not match requested run_ids" in errors
    assert "must contain distinct strings" in errors
    assert "sort field value must be finite numeric" in errors


def test_demo_evaluator_defaults_to_all_questions_and_auto_routing() -> None:
    args = build_parser().parse_args(
        ["--job-id", "job_1", "--output", "result.json"]
    )

    assert args.exclude_mixed is False
    assert args.request_query_type == "auto"
    assert args.expect_rag_health == "healthy"


def test_demo_evaluator_rejects_plain_http_to_remote_hosts() -> None:
    assert normalize_api_base("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    try:
        normalize_api_base("http://example.test:8000")
    except ValueError as error:
        assert "requires HTTPS" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("remote plain HTTP must be rejected")


def test_scope_map_and_checked_in_contract_are_complete(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(
        json.dumps({"q025": {"image_id": "img_1", "run_ids": ["run_1"]}}),
        encoding="utf-8",
    )
    assert load_scope_map(scope_path) == {
        "q025": {"image_id": "img_1", "run_ids": ["run_1"]}
    }

    repository = tmp_path
    package = repository / "package"
    package.mkdir()
    questions_path = repository / "questions.jsonl"
    questions_path.write_text(
        json.dumps({"query_id": "q001", "question": "question"}) + "\n",
        encoding="utf-8",
    )
    (package / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_asset_ids": {"q001": ["asset_1"]},
                "scope_requirements": {},
                "evidence_requirements": {},
            }
        ),
        encoding="utf-8",
    )

    questions = apply_evaluation_contract(load_questions(questions_path), package)

    assert questions[0]["expected_asset_ids"] == ["asset_1"]
