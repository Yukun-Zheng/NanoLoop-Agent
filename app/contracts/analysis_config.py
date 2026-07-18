"""Resolved scientific settings and reproducible execution provenance contracts."""

import hashlib
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.contracts.common import ContractModel


class PreprocessProfile(ContractModel):
    profile_id: str = "sem_gray_v1"
    grayscale: bool = True
    normalization: Literal["percentile", "minmax", "none"] = "percentile"
    lower_percentile: float = Field(default=1.0, ge=0, le=100)
    upper_percentile: float = Field(default=99.0, ge=0, le=100)
    roi_context_px: int = Field(default=16, ge=0, le=1024)

    @model_validator(mode="after")
    def validate_percentiles(self) -> "PreprocessProfile":
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be below upper_percentile")
        return self


class PostprocessProfile(ContractModel):
    profile_id: str = "default_v1"
    min_area_px: int = Field(default=8, ge=0)
    fill_holes: bool = True
    watershed_enabled: bool = False
    exclude_border: bool = True
    connectivity: Literal[1, 2] = 2
    instance_iou_threshold: float = Field(default=0.7, ge=0, le=1)


class MorphometryConfig(ContractModel):
    perimeter_neighborhood: Literal[4, 8] = 8


class QualityGateConfig(ContractModel):
    foreground_ratio_review_low: float = Field(default=0.0001, ge=0, le=1)
    foreground_ratio_warn_high: float = Field(default=0.30, ge=0, le=1)
    foreground_ratio_review_high: float = Field(default=0.40, ge=0, le=1)
    confidence_warn_below: float = Field(default=0.50, ge=0, le=1)
    fragment_ratio_warn_above: float = Field(default=0.30, ge=0, le=1)
    edge_touch_ratio_warn_above: float = Field(default=0.20, ge=0, le=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "QualityGateConfig":
        if self.foreground_ratio_warn_high > self.foreground_ratio_review_high:
            raise ValueError("foreground warn threshold cannot exceed review threshold")
        return self


class ExecutionBuildProvenance(ContractModel):
    """Best-effort identity of the application build that created a run contract."""

    application_version: str
    git_commit: str = "unknown"
    docker_image_tag: str = "unknown"
    python_version: str
    dependency_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    installed_dependencies_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    application_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def _application_source_sha256() -> str:
    """Hash the exact backend source tree, including relative file identities."""

    package_root = Path(__file__).resolve().parents[1]
    source_files = sorted(
        package_root.rglob("*.py"),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in source_files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _installed_dependencies_sha256() -> str:
    """Hash resolved installed distribution versions, not only declared ranges."""

    installed: set[str] = set()
    for distribution in metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            name = None
        if name:
            installed.add(f"{name.casefold()}=={distribution.version}")
    payload = "\n".join(sorted(installed)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_execution_build_provenance() -> ExecutionBuildProvenance:
    """Capture the same stable software identity used by export manifests."""

    try:
        application_version = metadata.version("nanoloop-agent")
        requirements = sorted(metadata.requires("nanoloop-agent") or [])
    except metadata.PackageNotFoundError:
        application_version = "0.1.0"
        requirements = []
    requirement_bytes = "\n".join(requirements).encode("utf-8")
    return ExecutionBuildProvenance(
        application_version=application_version,
        git_commit=os.environ.get("NANOLOOP_GIT_COMMIT") or "unknown",
        docker_image_tag=os.environ.get("NANOLOOP_IMAGE_TAG") or "unknown",
        python_version=platform.python_version(),
        dependency_contract_sha256=hashlib.sha256(requirement_bytes).hexdigest(),
        installed_dependencies_sha256=_installed_dependencies_sha256(),
        application_source_sha256=_application_source_sha256(),
    )
