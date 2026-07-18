"""Frozen machine-readable enum values from the v2.0 shared contract."""

from enum import StrEnum


class JobStatus(StrEnum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    READY_FOR_CONFIGURATION = "READY_FOR_CONFIGURATION"
    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    SEGMENTING = "SEGMENTING"
    POSTPROCESSING = "POSTPROCESSING"
    QUALITY_CHECKING = "QUALITY_CHECKING"
    ANALYZING = "ANALYZING"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"


class ModelFamily(StrEnum):
    UNET = "unet"
    YOLO_SEG = "yolo_seg"
    SAM2 = "sam2"


class ModelVariant(StrEnum):
    GENERAL = "general"
    SMALL_PARTICLE = "small_particle"
    LARGE_PARTICLE = "large_particle"
    DENSE_PARTICLE = "dense_particle"
    LOW_CONTRAST = "low_contrast"


class QualityTier(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class RoiMode(StrEnum):
    FULL_IMAGE = "full_image"
    BOXES = "boxes"


class QualityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class QueryType(StrEnum):
    AUTO = "auto"
    ANALYSIS_DATA = "analysis_data"
    MATERIAL_KNOWLEDGE = "material_knowledge"
    MIXED = "mixed"


class ModelStatus(StrEnum):
    READY = "ready"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class ApiStatus(StrEnum):
    SUCCESS = "success"
    ACCEPTED = "accepted"
    ERROR = "error"


class ScaleMode(StrEnum):
    NM_PER_PIXEL = "nm_per_pixel"
    PIXEL_ONLY = "pixel_only"


class DevicePreference(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class KnowledgeDocumentStatus(StrEnum):
    READY = "ready"
    INDEXING = "indexing"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class KnowledgeSourceType(StrEnum):
    PAPER = "paper"
    REPORT = "report"
    MATERIAL_NOTE = "material_note"
    OTHER = "other"
