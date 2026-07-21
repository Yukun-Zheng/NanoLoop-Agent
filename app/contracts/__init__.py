"""Public NanoLoop contracts.

Import from this module only for the stable high-level DTO surface. Module-level imports remain
available for less common internal contracts.
"""

from app.contracts.analyses import (
    AnalysisJobDTO,
    BoxSetDTO,
    CreateAnalysisMetadata,
    CreateRunsRequest,
    ImageAssetDTO,
    ImageSummaryDTO,
    ROIBox,
    SegmentationRunDTO,
)
from app.contracts.common import ApiErrorPayload, ApiResponse, HealthData
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityStatus,
    QualityTier,
    QueryType,
    RoiMode,
)
from app.contracts.identity import AuthMode, PrincipalContext, PrincipalKind, PrincipalRole
from app.contracts.models import ModelMetadata

__all__ = [
    "AnalysisJobDTO",
    "ApiErrorPayload",
    "ApiResponse",
    "AuthMode",
    "BoxSetDTO",
    "CreateAnalysisMetadata",
    "CreateRunsRequest",
    "HealthData",
    "ImageAssetDTO",
    "ImageSummaryDTO",
    "JobStatus",
    "ModelFamily",
    "ModelMetadata",
    "ModelStatus",
    "ModelVariant",
    "PrincipalContext",
    "PrincipalKind",
    "PrincipalRole",
    "QualityStatus",
    "QualityTier",
    "QueryType",
    "ROIBox",
    "RoiMode",
    "SegmentationRunDTO",
]
