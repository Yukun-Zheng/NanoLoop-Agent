"""Actual executor provenance recorded separately from the queued run contract."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.contracts.analysis_config import ExecutionBuildProvenance
from app.contracts.common import ContractModel
from app.contracts.enums import DevicePreference


class InferenceExecutionEvidence(ContractModel):
    """Observed device and controls at the adapter invocation boundary."""

    actual_device: Literal["cpu", "cuda", "mps"]
    python_random_seeded: bool
    numpy_random_seeded: bool
    torch_deterministic_algorithms: bool
    global_inference_serialized: bool
    backend: str = Field(min_length=1, max_length=500)


class ExecutionRuntimeProvenance(ContractModel):
    schema_version: Literal[1] = 1
    executor_build: ExecutionBuildProvenance
    build_identity_matches_contract: bool
    requested_device: DevicePreference
    actual_device: Literal["cpu", "cuda", "mps", "not_applicable"]
    seed: int
    python_random_seeded: bool
    numpy_random_seeded: bool
    torch_deterministic_algorithms: bool
    global_inference_serialized: bool
    backend: str = Field(min_length=1, max_length=500)
    model_bundle_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    warnings: list[str] = Field(default_factory=list)
    executed_at: datetime


def scientific_build_mismatches(
    contract_creator: ExecutionBuildProvenance,
    executor: ExecutionBuildProvenance,
) -> list[str]:
    """Return execution-sensitive build fields that differ or cannot be verified."""

    fields = (
        "python_version",
        "dependency_contract_sha256",
        "installed_dependencies_sha256",
        "application_source_sha256",
    )
    return [
        field
        for field in fields
        if getattr(contract_creator, field, None) != getattr(executor, field, None)
        or getattr(contract_creator, field, None) is None
    ]


__all__ = [
    "ExecutionRuntimeProvenance",
    "InferenceExecutionEvidence",
    "scientific_build_mismatches",
]
