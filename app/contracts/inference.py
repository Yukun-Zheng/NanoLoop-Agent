"""Internal contract shared by analysis and model adapter implementations."""

from pathlib import Path

from pydantic import Field

from app.contracts.analyses import ROIBox
from app.contracts.common import ContractModel
from app.contracts.enums import DevicePreference, RoiMode
from app.contracts.execution import InferenceExecutionEvidence


class SegmentationRequest(ContractModel):
    image_id: str
    image_path: Path
    image_bytes: bytes | None = Field(default=None, exclude=True, repr=False)
    run_dir: Path
    roi_mode: RoiMode
    boxes: list[ROIBox] = Field(default_factory=list, max_length=20)
    threshold: float | None = Field(default=None, ge=0, le=1)
    min_area_px: int = Field(default=0, ge=0)
    roi_context_px: int = Field(default=16, ge=0, le=4096)
    device: DevicePreference = DevicePreference.AUTO
    seed: int = 42


class InstancePrediction(ContractModel):
    instance_index: int = Field(ge=1)
    bbox: tuple[int, int, int, int]
    area_px: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    mask_score: float | None = Field(default=None, ge=0, le=1)


class SegmentationOutput(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    probability_path: Path | None = None
    binary_mask_path: Path
    instances_path: Path | None = None
    overlay_path: Path | None = None
    instances: list[InstancePrediction] = Field(default_factory=list)
    model_scores: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    runtime_ms: int = Field(ge=0)
    execution: InferenceExecutionEvidence | None = None
