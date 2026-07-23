"""Run the curated NanoLoop knowledge-question set against a live API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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


def request_headers() -> dict[str, str]:
    key = os.getenv("NANOLOOP_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


def response_data(response: httpx.Response) -> dict[str, Any]:
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
    return data


def evaluate_one(question: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    expected_outcome = question.get("expected_outcome")
    if data.get("outcome_code") != expected_outcome:
        errors.append(
            f"outcome expected {expected_outcome}, observed {data.get('outcome_code')}"
        )
    expected_type = question.get("query_type")
    if data.get("query_type") != expected_type:
        errors.append(f"query_type expected {expected_type}, observed {data.get('query_type')}")
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
    expected_tokens = question.get("expected_answer_contains_any", [])
    if expected_outcome == "OK" and isinstance(expected_tokens, list) and expected_tokens:
        if not any(
            isinstance(token, str) and token.casefold() in answer.casefold()
            for token in expected_tokens
        ):
            errors.append(f"answer contains none of expected tokens: {expected_tokens}")
    unknown_citations = [
        citation
        for citation in citations
        if not isinstance(citation, dict) or not citation.get("doc_id")
    ]
    if unknown_citations:
        errors.append("one or more citations have no doc_id")
    return {
        "query_id": question.get("query_id"),
        "passed": not errors,
        "errors": errors,
        "observed": {
            "query_type": data.get("query_type"),
            "outcome_code": data.get("outcome_code"),
            "confidence": data.get("confidence"),
            "citation_count": len(citations),
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
    parser.add_argument("--include-mixed", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions = load_questions(args.questions.expanduser().resolve(strict=True))
    selected = [
        question
        for question in questions
        if args.include_mixed or question.get("query_type") != "mixed"
    ]
    results: list[dict[str, Any]] = []
    base = args.api_base.rstrip("/")
    try:
        with httpx.Client(base_url=base, headers=request_headers(), timeout=args.timeout) as client:
            for question in selected:
                payload = {
                    "question": question["question"],
                    "query_type": question["query_type"],
                    "material_context": question.get("material_context"),
                }
                data = response_data(
                    client.post(f"/api/v1/analyses/{args.job_id}/query", json=payload)
                )
                results.append(evaluate_one(question, data))
        failed = [item for item in results if not item["passed"]]
        report = {
            "status": "passed" if not failed else "failed",
            "job_id": args.job_id,
            "question_count": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "mixed_included": args.include_mixed,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {key: report[key] for key in report if key != "results"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not failed else 2
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
