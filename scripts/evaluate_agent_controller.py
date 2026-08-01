#!/usr/bin/env python3
"""Evaluate the configured local Agent controller on the fixed decision suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agent.evaluation import (
    evaluate_agent_controller,
    load_agent_evaluation_suite,
    write_agent_evaluation_report,
)
from app.agent.model_provider import build_agent_decision_model
from app.core.config import Settings

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SUITE = _REPOSITORY_ROOT / "agent-evals" / "controller-v1.json"
_DEFAULT_OUTPUT = _REPOSITORY_ROOT / "outputs" / "agent-evals" / "controller-report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed NanoLoop Agent controller evaluation."
    )
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.minimum_pass_rate <= 1:
        parser.error("--minimum-pass-rate must be between 0 and 1")

    settings = Settings()
    model = build_agent_decision_model(settings)
    health = model.health()
    if health.status == "unavailable":
        print(
            json.dumps(
                {
                    "status": "unavailable",
                    "provider": model.identity.provider,
                    "model": model.identity.model,
                    "detail": health.detail,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    suite = load_agent_evaluation_suite(args.suite)
    report = evaluate_agent_controller(model, suite)
    write_agent_evaluation_report(report, args.output)
    accepted = (
        report.schema_valid_rate == 1
        and report.tool_arguments_valid_rate == 1
        and report.pass_rate >= args.minimum_pass_rate
        and report.unexpected_write_count == 0
    )
    print(
        json.dumps(
            {
                "status": "passed" if accepted else "failed",
                "suite_id": report.suite_id,
                "suite_version": report.suite_version,
                "prompt_id": report.prompt_id,
                "system_prompt_chars": report.system_prompt_chars,
                "tool_contract_chars": report.tool_contract_chars,
                "max_raw_state_chars": report.max_raw_state_chars,
                "mean_raw_state_chars": report.mean_raw_state_chars,
                "provider": report.model.provider,
                "model": report.model.model,
                "case_count": report.case_count,
                "schema_valid_rate": report.schema_valid_rate,
                "tool_arguments_valid_rate": report.tool_arguments_valid_rate,
                "pass_rate": report.pass_rate,
                "unexpected_write_count": report.unexpected_write_count,
                "report_path": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
