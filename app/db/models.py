"""Relational metadata and audit records; large artifacts stay in FileStore."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.contracts.common import utc_now
from app.contracts.enums import (
    JobStatus,
    KnowledgeDocumentStatus,
    ModelStatus,
    QualityStatus,
)
from app.db.base import Base

JsonObject = dict[str, Any]


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default=JobStatus.CREATED.value, nullable=False)
    config_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))

    images: Mapped[list["ImageAsset"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    runs: Mapped[list["SegmentationRun"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    queries: Mapped[list["QueryLog"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class ImageAsset(TimestampMixin, Base):
    __tablename__ = "image_assets"
    __table_args__ = (
        UniqueConstraint("job_id", "filename", name="job_filename"),
        UniqueConstraint("job_id", "sha256", name="job_sha256"),
        CheckConstraint("width > 0", name="width_positive"),
        CheckConstraint("height > 0", name="height_positive"),
        CheckConstraint("bit_depth > 0", name="bit_depth_positive"),
        CheckConstraint(
            "scale_nm_per_pixel IS NULL OR scale_nm_per_pixel > 0",
            name="scale_positive_or_null",
        ),
    )

    image_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_jobs.job_id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    bit_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_id: Mapped[str] = mapped_column(String(120), nullable=False)
    material_name: Mapped[str | None] = mapped_column(String(255))
    material_formula: Mapped[str | None] = mapped_column(String(255))
    experiment_conditions_json: Mapped[JsonObject] = mapped_column(
        JSON, default=dict, nullable=False
    )
    analysis_roi_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    scale_nm_per_pixel: Mapped[float | None] = mapped_column(Float)
    box_revision: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    job: Mapped[AnalysisJob] = relationship(back_populates="images")
    boxes: Mapped[list["ROIBoxRecord"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    runs: Mapped[list["SegmentationRun"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ROIBoxRecord(TimestampMixin, Base):
    __tablename__ = "roi_boxes"
    __table_args__ = (
        CheckConstraint("x1 >= 0 AND y1 >= 0", name="origin_nonnegative"),
        CheckConstraint("x1 < x2", name="x_ordered"),
        CheckConstraint("y1 < y2", name="y_ordered"),
        CheckConstraint("revision >= 1", name="revision_positive"),
        UniqueConstraint("image_id", "revision", "box_id", name="image_revision_box"),
        Index("ix_roi_boxes_image_revision", "image_id", "revision"),
    )

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    box_id: Mapped[str] = mapped_column(String(64), nullable=False)
    image_id: Mapped[str] = mapped_column(
        ForeignKey("image_assets.image_id", ondelete="CASCADE"), nullable=False
    )
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)

    image: Mapped[ImageAsset] = relationship(back_populates="boxes")


class ROIBoxRevisionRecord(Base):
    __tablename__ = "roi_box_revisions"
    __table_args__ = (
        CheckConstraint("revision >= 0", name="revision_nonnegative"),
        CheckConstraint("box_count >= 0", name="box_count_nonnegative"),
        UniqueConstraint("image_id", "revision", name="image_box_revision"),
        Index("ix_roi_box_revisions_image_revision", "image_id", "revision"),
    )

    revision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(
        ForeignKey("image_assets.image_id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    box_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ModelRegistryRecord(TimestampMixin, Base):
    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    family: Mapped[str] = mapped_column(String(40), nullable=False)
    variant: Mapped[str] = mapped_column(String(60), nullable=False)
    quality_tier: Mapped[str] = mapped_column(String(40), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    adapter: Mapped[str] = mapped_column(String(500), nullable=False)
    weight_path: Mapped[str | None] = mapped_column(Text)
    config_path: Mapped[str | None] = mapped_column(Text)
    model_card_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(40), default=ModelStatus.UNAVAILABLE.value, nullable=False
    )
    metadata_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    health_error: Mapped[str | None] = mapped_column(Text)
    weight_sha256: Mapped[str | None] = mapped_column(String(64))

    runs: Mapped[list["SegmentationRun"]] = relationship(back_populates="model")


class SegmentationRun(TimestampMixin, Base):
    __tablename__ = "segmentation_runs"
    __table_args__ = (
        CheckConstraint(
            "threshold IS NULL OR (threshold >= 0 AND threshold <= 1)", name="threshold"
        ),
        CheckConstraint("runtime_ms IS NULL OR runtime_ms >= 0", name="runtime_nonnegative"),
        Index("ix_segmentation_runs_job_status", "job_id", "status"),
    )

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    image_id: Mapped[str] = mapped_column(
        ForeignKey("image_assets.image_id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[str] = mapped_column(
        ForeignKey("model_registry.model_id", ondelete="RESTRICT"), nullable=False
    )
    roi_mode: Mapped[str] = mapped_column(String(40), nullable=False)
    box_revision: Mapped[int | None] = mapped_column(Integer)
    threshold: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(40), default=JobStatus.CREATED.value, nullable=False)
    inference_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    run_config_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    execution_json: Mapped[JsonObject | None] = mapped_column(JSON)
    paths_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    runtime_ms: Mapped[int | None] = mapped_column(Integer)
    parent_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("segmentation_runs.run_id", ondelete="SET NULL")
    )
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)

    job: Mapped[AnalysisJob] = relationship(back_populates="runs")
    image: Mapped[ImageAsset] = relationship(back_populates="runs")
    model: Mapped[ModelRegistryRecord] = relationship(back_populates="runs")
    particles: Mapped[list["ParticleRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    summary: Mapped["ImageSummary | None"] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )
    status_events: Mapped[list["RunStatusEvent"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunStatusEvent.event_id",
    )


class RunStatusEvent(Base):
    __tablename__ = "run_status_events"
    __table_args__ = (Index("ix_run_status_events_run_event", "run_id", "event_id"),)

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("segmentation_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    from_status: Mapped[str | None] = mapped_column(String(40))
    to_status: Mapped[str] = mapped_column(String(40), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    run: Mapped[SegmentationRun] = relationship(back_populates="status_events")


class ParticleRecord(Base):
    __tablename__ = "particle_records"
    __table_args__ = (
        CheckConstraint("instance_index >= 1", name="instance_index_positive"),
        CheckConstraint("area_px >= 0", name="area_nonnegative"),
        CheckConstraint("perimeter_px >= 0", name="perimeter_nonnegative"),
        UniqueConstraint("run_id", "instance_index", name="run_instance_index"),
    )

    particle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("segmentation_runs.run_id", ondelete="CASCADE"), index=True, nullable=False
    )
    instance_index: Mapped[int] = mapped_column(Integer, nullable=False)
    area_px: Mapped[float] = mapped_column(Float, nullable=False)
    perimeter_px: Mapped[float] = mapped_column(Float, nullable=False)
    equivalent_diameter_px: Mapped[float] = mapped_column(Float, nullable=False)
    equivalent_diameter_nm: Mapped[float | None] = mapped_column(Float)
    circularity: Mapped[float | None] = mapped_column(Float)
    bbox_json: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)

    run: Mapped[SegmentationRun] = relationship(back_populates="particles")


class ImageSummary(Base):
    __tablename__ = "image_summaries"
    __table_args__ = (
        CheckConstraint("particle_count >= 0", name="particle_count_nonnegative"),
        CheckConstraint("roi_area_px >= 0", name="roi_area_nonnegative"),
        CheckConstraint("coverage_ratio >= 0 AND coverage_ratio <= 1", name="coverage_ratio"),
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("segmentation_runs.run_id", ondelete="CASCADE"), primary_key=True
    )
    particle_count: Mapped[int] = mapped_column(Integer, nullable=False)
    roi_area_px: Mapped[int] = mapped_column(Integer, nullable=False)
    number_density_px2: Mapped[float] = mapped_column(Float, nullable=False)
    number_density_um2: Mapped[float | None] = mapped_column(Float)
    mean_equivalent_diameter_px: Mapped[float | None] = mapped_column(Float)
    mean_equivalent_diameter_nm: Mapped[float | None] = mapped_column(Float)
    coverage_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    perimeter_density_px: Mapped[float] = mapped_column(Float, nullable=False)
    perimeter_density_um: Mapped[float | None] = mapped_column(Float)
    quality_status: Mapped[str] = mapped_column(
        String(40), default=QualityStatus.PASS.value, nullable=False
    )
    quality_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)

    run: Mapped[SegmentationRun] = relationship(back_populates="summary")


class KnowledgeDocument(TimestampMixin, Base):
    __tablename__ = "knowledge_documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(80), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer)
    citation_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(40), default=KnowledgeDocumentStatus.INDEXING.value, nullable=False
    )
    metadata_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (Index("ix_knowledge_chunks_doc_page", "doc_id", "page_start"),)

    chunk_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    material_tags_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    vector_id: Mapped[int | None] = mapped_column(Integer, unique=True)

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    query_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_jobs.job_id", ondelete="CASCADE"), index=True, nullable=False
    )
    image_id: Mapped[str | None] = mapped_column(
        ForeignKey("image_assets.image_id", ondelete="SET NULL")
    )
    query_type: Mapped[str] = mapped_column(String(40), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    request_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    answer_json: Mapped[JsonObject] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    job: Mapped[AnalysisJob] = relationship(back_populates="queries")
