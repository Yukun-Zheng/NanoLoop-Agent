"""Run the curated NanoLoop knowledge-question set against a live API."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import urlsplit

import httpx

_CITATION_MARKER = re.compile(r"\[(C\d+)\]")
_MIXED_CAUSAL_BOUNDARY = "文献中的一般规律不能直接证明当前样品的因果机理"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_EVIDENCE_REQUIREMENT_FIELDS = {
    "tool_name",
    "validated_arguments",
    "required_units",
    "required_numeric_fields",
    "min_source_runs",
    "min_rows",
    "aggregate_equals_row_count",
    "required_row_values",
    "distinct_row_field",
    "sort",
    "require_row_source_coverage",
}


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        questions.append(value)
    if not questions:
        raise ValueError("question file is empty")
    return questions


def apply_evaluation_contract(
    questions: list[dict[str, Any]],
    package: Path,
) -> list[dict[str, Any]]:
    path = package.expanduser().resolve(strict=True) / "evaluation_contract.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("evaluation contract must be an object with schema_version=1")
    expected = value.get("expected_asset_ids")
    requirements = value.get("scope_requirements", {})
    evidence_requirements = value.get("evidence_requirements", {})
    if (
        not isinstance(expected, dict)
        or not isinstance(requirements, dict)
        or not isinstance(evidence_requirements, dict)
    ):
        raise ValueError("evaluation contract mappings are invalid")
    question_ids = {str(question.get("query_id")) for question in questions}
    if set(expected) != question_ids:
        raise ValueError("evaluation contract must cover exactly the question set")
    mixed_ok_ids = {
        str(question.get("query_id"))
        for question in questions
        if question.get("query_type") == "mixed"
        and question.get("expected_outcome") == "OK"
    }
    if set(evidence_requirements) != mixed_ok_ids:
        raise ValueError(
            "evidence requirements must cover exactly the mixed OK question set"
        )
    output: list[dict[str, Any]] = []
    for question in questions:
        query_id = str(question["query_id"])
        assets = expected[query_id]
        requirement = requirements.get(query_id)
        evidence_requirement = evidence_requirements.get(query_id)
        if not isinstance(assets, list) or any(
            not isinstance(asset_id, str) or not asset_id for asset_id in assets
        ):
            raise ValueError(f"{query_id}: expected_asset_ids must be strings")
        if requirement not in {None, "image", "job"}:
            raise ValueError(f"{query_id}: invalid scope requirement")
        if evidence_requirement is not None:
            _validate_evidence_contract(query_id, evidence_requirement)
        output.append(
            {
                **question,
                "expected_asset_ids": assets,
                **({"scope_requirement": requirement} if requirement else {}),
                **(
                    {"evidence_requirement": evidence_requirement}
                    if evidence_requirement is not None
                    else {}
                ),
            }
        )
    return output


def _non_empty_string_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and bool(key)
        and isinstance(item, str)
        and bool(item)
        for key, item in value.items()
    )


def _validate_evidence_contract(query_id: str, value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{query_id}: evidence requirement must be an object")
    unknown = set(value) - _EVIDENCE_REQUIREMENT_FIELDS
    if unknown:
        raise ValueError(
            f"{query_id}: unsupported evidence requirement fields {sorted(unknown)}"
        )
    tool_name = value.get("tool_name")
    arguments = value.get("validated_arguments")
    units = value.get("required_units")
    numeric_fields = value.get("required_numeric_fields")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError(f"{query_id}: evidence tool_name must be a non-empty string")
    if not isinstance(arguments, dict) or any(
        not isinstance(key, str) or not key for key in arguments
    ):
        raise ValueError(f"{query_id}: validated_arguments must be an object")
    if not _non_empty_string_mapping(units) or not units:
        raise ValueError(f"{query_id}: required_units must be a non-empty string mapping")
    if (
        not isinstance(numeric_fields, list)
        or not numeric_fields
        or any(not isinstance(field, str) or not field for field in numeric_fields)
    ):
        raise ValueError(
            f"{query_id}: required_numeric_fields must be non-empty strings"
        )
    for field in ("min_source_runs", "min_rows"):
        count = value.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{query_id}: {field} must be a non-negative integer")
    count_field = value.get("aggregate_equals_row_count")
    if count_field is not None and (
        not isinstance(count_field, str) or not count_field
    ):
        raise ValueError(
            f"{query_id}: aggregate_equals_row_count must be a non-empty string"
        )
    row_values = value.get("required_row_values", {})
    if not isinstance(row_values, dict) or any(
        not isinstance(key, str) or not key for key in row_values
    ):
        raise ValueError(f"{query_id}: required_row_values must be an object")
    distinct_field = value.get("distinct_row_field")
    if distinct_field is not None and (
        not isinstance(distinct_field, str) or not distinct_field
    ):
        raise ValueError(f"{query_id}: distinct_row_field must be a non-empty string")
    sort = value.get("sort")
    if sort is not None and (
        not isinstance(sort, dict)
        or set(sort) != {"field", "order"}
        or not isinstance(sort.get("field"), str)
        or not sort["field"]
        or sort.get("order") not in {"asc", "desc"}
    ):
        raise ValueError(f"{query_id}: sort must define field and asc/desc order")
    coverage = value.get("require_row_source_coverage", False)
    if not isinstance(coverage, bool):
        raise ValueError(
            f"{query_id}: require_row_source_coverage must be a boolean"
        )


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _field_has_finite_value(
    field: str,
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
) -> bool:
    if _is_finite_number(aggregates.get(field)):
        return True
    return any(_is_finite_number(row.get(field)) for row in rows)


def _row_source_ids(rows: list[dict[str, Any]]) -> set[str] | None:
    result: set[str] = set()
    for row in rows:
        run_id = row.get("run_id")
        run_ids = row.get("run_ids")
        if isinstance(run_id, str) and run_id:
            result.add(run_id)
            continue
        if not isinstance(run_ids, list) or any(
            not isinstance(item, str) or not item for item in run_ids
        ):
            return None
        result.update(run_ids)
    return result


def _validate_mixed_evidence(
    question: Mapping[str, Any],
    evidence: list[Any],
    request_scope: Mapping[str, Any],
) -> list[str]:
    query_id = str(question.get("query_id"))
    requirement = question.get("evidence_requirement")
    if not isinstance(requirement, dict):
        return [f"{query_id}: mixed question is missing its evidence contract"]
    if len(evidence) != 1:
        return [f"{query_id}: expected exactly one data_evidence record"]
    item = evidence[0]
    if not isinstance(item, dict):
        return [f"{query_id}: data_evidence record must be an object"]

    errors: list[str] = []
    if item.get("tool_name") != requirement["tool_name"]:
        errors.append(
            f"{query_id}: evidence tool expected {requirement['tool_name']}, "
            f"observed {item.get('tool_name')}"
        )
    arguments = item.get("validated_arguments")
    if not isinstance(arguments, dict):
        errors.append(f"{query_id}: evidence validated_arguments must be an object")
        arguments = {}
    for field, expected in requirement["validated_arguments"].items():
        if arguments.get(field) != expected:
            errors.append(
                f"{query_id}: validated argument {field} expected {expected!r}, "
                f"observed {arguments.get(field)!r}"
            )

    source_run_ids_value = item.get("source_run_ids")
    if (
        not isinstance(source_run_ids_value, list)
        or any(
            not isinstance(run_id, str) or not run_id
            for run_id in source_run_ids_value
        )
        or len(set(source_run_ids_value)) != len(source_run_ids_value)
    ):
        errors.append(f"{query_id}: source_run_ids must be unique non-empty strings")
        source_run_ids: list[str] = []
    else:
        source_run_ids = source_run_ids_value
    if len(source_run_ids) < requirement["min_source_runs"]:
        errors.append(
            f"{query_id}: expected at least {requirement['min_source_runs']} source runs"
        )

    requested_run_ids_value = request_scope.get("run_ids", [])
    requested_run_ids = (
        requested_run_ids_value
        if isinstance(requested_run_ids_value, list)
        else []
    )
    if requested_run_ids and set(source_run_ids) != set(requested_run_ids):
        errors.append(f"{query_id}: evidence source runs do not match requested run_ids")
    argument_run_ids = arguments.get("run_ids")
    if set(source_run_ids) != (
        set(argument_run_ids) if isinstance(argument_run_ids, list) else set()
    ):
        errors.append(
            f"{query_id}: validated run_ids do not match evidence source_run_ids"
        )
    requested_image_id = request_scope.get("image_id")
    if requested_image_id is not None and arguments.get("image_id") != requested_image_id:
        errors.append(f"{query_id}: validated image_id does not match requested image_id")

    rows_value = item.get("rows")
    aggregates_value = item.get("aggregates")
    if not isinstance(rows_value, list) or any(
        not isinstance(row, dict) for row in rows_value
    ):
        errors.append(f"{query_id}: evidence rows must be a list of objects")
        rows: list[dict[str, Any]] = []
    else:
        rows = rows_value
    if not isinstance(aggregates_value, dict):
        errors.append(f"{query_id}: evidence aggregates must be an object")
        aggregates: dict[str, Any] = {}
    else:
        aggregates = aggregates_value
    if len(rows) < requirement["min_rows"]:
        errors.append(f"{query_id}: evidence has fewer than required rows")
    count_field = requirement.get("aggregate_equals_row_count")
    if isinstance(count_field, str) and aggregates.get(count_field) != len(rows):
        errors.append(
            f"{query_id}: aggregate {count_field} must equal the row count"
        )

    units = item.get("units")
    if not _non_empty_string_mapping(units) or not units:
        errors.append(f"{query_id}: evidence units must be a non-empty string mapping")
        units = {}
    for field, expected_unit in requirement["required_units"].items():
        if units.get(field) != expected_unit:
            errors.append(
                f"{query_id}: unit for {field} expected {expected_unit!r}, "
                f"observed {units.get(field)!r}"
            )
    for field in requirement["required_numeric_fields"]:
        if not _field_has_finite_value(field, rows, aggregates):
            errors.append(f"{query_id}: evidence lacks finite numeric field {field}")

    for field, expected in requirement.get("required_row_values", {}).items():
        if any(row.get(field) != expected for row in rows):
            errors.append(
                f"{query_id}: every row must have {field}={expected!r}"
            )
    distinct_field = requirement.get("distinct_row_field")
    if isinstance(distinct_field, str):
        values = [row.get(distinct_field) for row in rows]
        if any(not isinstance(value, str) or not value for value in values) or len(
            set(values)
        ) != len(values):
            errors.append(
                f"{query_id}: row field {distinct_field} must contain distinct strings"
            )
    sort = requirement.get("sort")
    if isinstance(sort, dict):
        field = str(sort["field"])
        values = [row.get(field) for row in rows]
        if not all(_is_finite_number(value) for value in values):
            errors.append(f"{query_id}: sort field {field} must be finite numeric")
        else:
            numeric = [
                float(value) for value in values if _is_finite_number(value)
            ]
            ordered = (
                all(left <= right for left, right in pairwise(numeric))
                if sort["order"] == "asc"
                else all(left >= right for left, right in pairwise(numeric))
            )
            if not ordered:
                errors.append(
                    f"{query_id}: rows are not sorted {sort['order']} by {field}"
                )
    if requirement.get("require_row_source_coverage"):
        row_sources = _row_source_ids(rows)
        if row_sources is None or row_sources != set(source_run_ids):
            errors.append(
                f"{query_id}: row run IDs do not exactly cover source_run_ids"
            )
    return errors


def request_headers() -> dict[str, str]:
    key = os.getenv("NANOLOOP_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


def normalize_api_base(
    value: str,
    *,
    allow_insecure_remote_http: bool = False,
) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("api base must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("api base must not contain credentials, query, or fragment")
    if (
        parsed.scheme == "http"
        and parsed.hostname not in _LOOPBACK_HOSTS
        and not allow_insecure_remote_http
    ):
        raise ValueError("remote evaluation requires HTTPS")
    return value.strip().rstrip("/")


def response_envelope(response: httpx.Response) -> tuple[str, dict[str, Any]]:
    try:
        body = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"non-JSON API response: {response.status_code} {response.text[:300]}"
        ) from error
    if not isinstance(body, dict):
        raise RuntimeError("API response must be an object")
    if response.is_error or body.get("status") == "error":
        raise RuntimeError(
            f"query failed ({response.status_code}): "
            f"{json.dumps(body.get('error') or body, ensure_ascii=False)}"
        )
    data = body.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("query success response has no data object")
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("API response has no request_id")
    return request_id, data


def load_scope_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("scope map must be a JSON object keyed by query_id")
    output: dict[str, dict[str, Any]] = {}
    for query_id, raw_scope in value.items():
        if not isinstance(query_id, str) or not isinstance(raw_scope, dict):
            raise ValueError("scope map entries must be query_id -> object")
        unknown = set(raw_scope) - {"image_id", "run_ids"}
        if unknown:
            raise ValueError(f"{query_id}: unsupported scope fields {sorted(unknown)}")
        image_id = raw_scope.get("image_id")
        run_ids = raw_scope.get("run_ids", [])
        if image_id is not None and (not isinstance(image_id, str) or not image_id):
            raise ValueError(f"{query_id}: image_id must be a non-empty string")
        if not isinstance(run_ids, list) or any(
            not isinstance(run_id, str) or not run_id for run_id in run_ids
        ):
            raise ValueError(f"{query_id}: run_ids must be non-empty strings")
        output[query_id] = {
            **({"image_id": image_id} if image_id is not None else {}),
            **({"run_ids": run_ids} if run_ids else {}),
        }
    return output


def runtime_asset_mapping(
    client: httpx.Client,
    package: Path,
) -> tuple[dict[str, str], str]:
    manifest_path = package.expanduser().resolve(strict=True) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("curated package manifest has no documents")
    expected: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("curated package document must be an object")
        asset_id = document.get("asset_id")
        digest = document.get("sha256")
        if not isinstance(asset_id, str) or not isinstance(digest, str):
            raise ValueError("curated package document is missing asset_id/sha256")
        expected[asset_id] = digest

    request_id, data = response_envelope(client.get("/api/v1/knowledge/documents"))
    rows = data.get("documents")
    if not isinstance(rows, list):
        raise RuntimeError("knowledge catalogue has no documents list")
    by_sha = {
        row.get("sha256"): row.get("doc_id")
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "ready"
        and isinstance(row.get("sha256"), str)
        and isinstance(row.get("doc_id"), str)
    }
    missing = sorted(asset_id for asset_id, digest in expected.items() if digest not in by_sha)
    if missing:
        raise RuntimeError(f"ready knowledge catalogue is missing curated assets: {missing}")
    mapping: dict[str, str] = {}
    for asset_id, digest in expected.items():
        doc_id = by_sha[digest]
        if not isinstance(doc_id, str):  # narrowed by the catalogue comprehension
            raise RuntimeError(f"curated asset {asset_id} has an invalid runtime doc_id")
        mapping[asset_id] = doc_id
    return mapping, request_id


def evaluate_one(
    question: dict[str, Any],
    data: dict[str, Any],
    *,
    request_id: str,
    asset_to_doc: dict[str, str] | None = None,
    scope_error: str | None = None,
    request_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if scope_error:
        errors.append(scope_error)
    expected_outcome = question.get("expected_outcome")
    if data.get("outcome_code") != expected_outcome:
        errors.append(
            f"outcome expected {expected_outcome}, observed {data.get('outcome_code')}"
        )
    expected_type = question.get("query_type")
    if data.get("query_type") != expected_type:
        errors.append(
            f"query_type expected {expected_type}, observed {data.get('query_type')}"
        )
    citations = data.get("citations", [])
    if not isinstance(citations, list):
        errors.append("citations is not a list")
        citations = []
    require_citations = question.get("require_citations") is True
    if expected_outcome == "OK" and require_citations and not citations:
        errors.append("expected at least one citation")
    if expected_outcome == "INSUFFICIENT_EVIDENCE" and citations:
        errors.append("insufficient-evidence answer must not contain citations")
    answer = str(data.get("answer", ""))
    if not answer.strip():
        errors.append("answer is empty")
    expected_tokens = question.get("expected_answer_contains_any", [])
    token_match = isinstance(expected_tokens, list) and any(
        isinstance(token, str) and token.casefold() in answer.casefold()
        for token in expected_tokens
    )
    if expected_outcome == "OK" and expected_tokens and not token_match:
        errors.append(f"answer contains none of expected tokens: {expected_tokens}")
    returned_citation_ids: list[str] = []
    returned_doc_ids: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            errors.append("one or more citations are not objects")
            continue
        required_fields = ("citation_id", "doc_id", "chunk_id", "excerpt")
        missing = [field for field in required_fields if not citation.get(field)]
        if missing:
            errors.append(f"citation missing fields: {missing}")
            continue
        score = citation.get("retrieval_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            errors.append("citation retrieval_score must be in [0, 1]")
        returned_citation_ids.append(str(citation["citation_id"]))
        returned_doc_ids.append(str(citation["doc_id"]))
    if len(set(returned_citation_ids)) != len(returned_citation_ids):
        errors.append("citation_id values must be unique")
    answer_markers = set(_CITATION_MARKER.findall(answer))
    if set(returned_citation_ids) != answer_markers:
        errors.append(
            "answer citation markers must exactly match returned citation_id values"
        )

    expected_assets = question.get("expected_asset_ids", [])
    if expected_outcome == "OK" and expected_assets:
        if asset_to_doc is None:
            errors.append("expected source mapping was not checked")
        else:
            expected_docs = {
                asset_to_doc[asset_id]
                for asset_id in expected_assets
                if asset_id in asset_to_doc
            }
            if not expected_docs:
                errors.append("question expected assets are absent from runtime mapping")
            elif not expected_docs.intersection(returned_doc_ids):
                errors.append(
                    "citations contain none of the expected curated sources: "
                    f"{expected_assets}"
                )

    evidence = data.get("data_evidence", [])
    if not isinstance(evidence, list):
        errors.append("data_evidence is not a list")
        evidence = []
    if question.get("query_type") == "mixed" and expected_outcome == "OK":
        if not evidence:
            errors.append("mixed answer requires data_evidence")
        errors.extend(
            _validate_mixed_evidence(question, evidence, request_scope or {})
        )
        limitations = data.get("limitations", [])
        if not isinstance(limitations, list) or not any(
            _MIXED_CAUSAL_BOUNDARY in str(item) for item in limitations
        ):
            errors.append("mixed answer is missing the literature-to-sample causal boundary")
    if data.get("needs_clarification") is True and expected_outcome == "OK":
        errors.append("expected OK answer must not require clarification")
    if expected_outcome == "INSUFFICIENT_EVIDENCE" and data.get("confidence") != "low":
        errors.append("insufficient-evidence answer must have low confidence")
    return {
        "query_id": question.get("query_id"),
        "request_id": request_id,
        "passed": not errors,
        "errors": errors,
        "observed": {
            "query_type": data.get("query_type"),
            "outcome_code": data.get("outcome_code"),
            "confidence": data.get("confidence"),
            "citation_count": len(citations),
            "citations": citations,
            "data_evidence": evidence,
            "answer": answer,
            "limitations": data.get("limitations", []),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("demo_data/rag/questions.jsonl"),
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("demo_data/rag"),
        help="Curated package used to bind expected asset IDs to runtime doc IDs.",
    )
    parser.add_argument(
        "--exclude-mixed",
        action="store_true",
        help="Run only the retrieval-only subset. The default evaluates all 30 questions.",
    )
    parser.add_argument(
        "--request-query-type",
        choices=("auto", "expected"),
        default="auto",
        help="AUTO is the default so the evaluator also tests routing.",
    )
    parser.add_argument("--scope-map", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument(
        "--expect-rag-health",
        choices=("healthy", "degraded", "unavailable"),
        default="healthy",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-insecure-remote-http", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions = apply_evaluation_contract(
        load_questions(args.questions.expanduser().resolve(strict=True)),
        args.package,
    )
    selected = [
        question
        for question in questions
        if not args.exclude_mixed or question.get("query_type") != "mixed"
    ]
    scope_map = load_scope_map(args.scope_map)
    results: list[dict[str, Any]] = []
    base = normalize_api_base(
        args.api_base,
        allow_insecure_remote_http=args.allow_insecure_remote_http,
    )
    try:
        with httpx.Client(
            base_url=base,
            headers=request_headers(),
            timeout=args.timeout,
        ) as client:
            health_before_request_id, health_before = response_envelope(
                client.get("/api/v1/health")
            )
            before_rag = health_before.get("rag_index")
            before_status = (
                before_rag.get("status") if isinstance(before_rag, dict) else None
            )
            if before_status != args.expect_rag_health:
                raise RuntimeError(
                    "RAG health mismatch before evaluation: "
                    f"expected {args.expect_rag_health}, observed {before_status}"
                )
            asset_to_doc, catalogue_request_id = runtime_asset_mapping(
                client,
                args.package,
            )
            for question in selected:
                query_id = str(question["query_id"])
                scope = dict(scope_map.get(query_id, {}))
                if question.get("query_type") == "mixed" and args.run_id:
                    scope.setdefault("run_ids", list(dict.fromkeys(args.run_id)))
                if (
                    question.get("scope_requirement") == "image"
                    and args.image_id
                ):
                    scope.setdefault("image_id", args.image_id)
                scope_error = None
                if (
                    question.get("scope_requirement") == "image"
                    and not scope.get("image_id")
                ):
                    scope_error = (
                        "question requires an explicit image_id via --image-id or --scope-map"
                    )
                payload = {
                    "question": question["question"],
                    "query_type": (
                        "auto"
                        if args.request_query_type == "auto"
                        else question["query_type"]
                    ),
                    "material_context": question.get("material_context"),
                    **scope,
                }
                request_id, data = response_envelope(
                    client.post(f"/api/v1/analyses/{args.job_id}/query", json=payload)
                )
                results.append(
                    evaluate_one(
                        question,
                        data,
                        request_id=request_id,
                        asset_to_doc=asset_to_doc,
                        scope_error=scope_error,
                        request_scope=scope,
                    )
                )
            health_after_request_id, health_after = response_envelope(
                client.get("/api/v1/health")
            )
            after_rag = health_after.get("rag_index")
            after_status = (
                after_rag.get("status") if isinstance(after_rag, dict) else None
            )
            if after_status != args.expect_rag_health:
                raise RuntimeError(
                    "RAG health changed during evaluation: "
                    f"expected {args.expect_rag_health}, observed {after_status}"
                )
        failed = [item for item in results if not item["passed"]]
        report = {
            "status": "passed" if not failed else "failed",
            "job_id": args.job_id,
            "question_count": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "mixed_included": not args.exclude_mixed,
            "request_query_type": args.request_query_type,
            "source_mapping_checked": True,
            "catalogue_request_id": catalogue_request_id,
            "health_before": {
                "request_id": health_before_request_id,
                "data": health_before,
            },
            "health_after": {
                "request_id": health_after_request_id,
                "data": health_after,
            },
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {key: report[key] for key in report if key != "results"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not failed else 2
    except Exception as error:
        error_payload = {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(
            json.dumps(error_payload, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
