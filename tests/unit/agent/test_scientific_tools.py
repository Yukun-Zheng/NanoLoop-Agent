from __future__ import annotations

from typing import Any, cast

from app.agent.scientific_tools import (
    build_scientific_tool_registry,
    scientific_tool_specs,
)
from app.contracts.agent_runtime import (
    AgentToolRisk,
    AgentToolTransport,
)


def test_scientific_registry_exposes_only_bounded_reviewed_capabilities() -> None:
    registry = build_scientific_tool_registry(
        session_factory=cast(Any, lambda: None),
        inference_gateway=object(),
        analysis_service=cast(Any, object()),
        data_tools=cast(Any, object()),
        file_store=cast(Any, object()),
        file_access=cast(Any, object()),
        api_prefix="/api/v1",
    )

    specs = {spec.name: spec for spec in registry.specs()}
    public_specs = {spec.name: spec for spec in scientific_tool_specs()}

    assert set(specs) == {
        "inspect_job",
        "inspect_runs",
        "recommend_models",
        "query_results",
        "create_analysis_runs",
        "create_review_run",
        "generate_scientific_report",
        "export_reproducibility_bundle",
    }
    assert public_specs == specs
    assert all(spec.transport is AgentToolTransport.PYTHON for spec in specs.values())
    assert all(
        spec.requires_approval
        for spec in specs.values()
        if spec.risk is not AgentToolRisk.READ_ONLY
    )
    assert all(
        not spec.requires_approval
        for spec in specs.values()
        if spec.risk is AgentToolRisk.READ_ONLY
    )
