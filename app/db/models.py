"""Relational metadata and audit records; large artifacts stay in FileStore."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DDL,
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.contracts.common import utc_now
from app.contracts.enums import (
    JobStatus,
    KnowledgeDocumentStatus,
    ModelStatus,
    QualityStatus,
)
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.db.base import Base

JsonObject = dict[str, Any]
_HEX_32_GLOB = "[0-9a-f]" * 32


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        UniqueConstraint("job_id", "tenant_id", name="uq_analysis_jobs_job_tenant"),
        ForeignKeyConstraint(
            ["owner_principal_id", "tenant_id"],
            ["principals.principal_id", "principals.tenant_id"],
            name="fk_analysis_jobs_owner_principal_tenant",
            ondelete="RESTRICT",
        ),
        Index("ix_analysis_jobs_tenant_created", "tenant_id", "created_at"),
        Index(
            "ix_analysis_jobs_tenant_owner_created",
            "tenant_id",
            "owner_principal_id",
            "created_at",
        ),
    )

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.tenant_id", ondelete="RESTRICT"),
        nullable=False,
    )
    owner_principal_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )
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
        UniqueConstraint("image_id", "job_id", name="uq_image_assets_image_job"),
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
        back_populates="image",
        cascade="all, delete-orphan",
        foreign_keys="SegmentationRun.image_id",
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
        ForeignKeyConstraint(
            ["image_id", "job_id"],
            ["image_assets.image_id", "image_assets.job_id"],
            name="fk_segmentation_runs_image_job",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["parent_run_id", "job_id"],
            ["segmentation_runs.run_id", "segmentation_runs.job_id"],
            name="fk_segmentation_runs_parent_job",
        ),
        UniqueConstraint("run_id", "job_id", name="uq_segmentation_runs_run_job"),
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
    image_id: Mapped[str] = mapped_column(String(64), nullable=False)
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
    image: Mapped[ImageAsset] = relationship(
        back_populates="runs",
        foreign_keys=[image_id],
    )
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
    __table_args__ = (
        ForeignKeyConstraint(
            ["image_id", "job_id"],
            ["image_assets.image_id", "image_assets.job_id"],
            name="fk_query_logs_image_job",
        ),
        ForeignKeyConstraint(
            ["job_id", "actor_tenant_id"],
            ["analysis_jobs.job_id", "analysis_jobs.tenant_id"],
            name="fk_query_logs_job_actor_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["actor_principal_id", "actor_tenant_id"],
            ["principals.principal_id", "principals.tenant_id"],
            name="fk_query_logs_actor_principal_tenant",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["actor_credential_id", "actor_principal_id"],
            ["api_credentials.credential_id", "api_credentials.principal_id"],
            name="fk_query_logs_actor_credential_principal",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "actor_role IN ('tenant_admin', 'analyst', 'viewer')",
            name="actor_role_known",
        ),
        CheckConstraint(
            "actor_auth_mode IN "
            "('disabled', 'shared_key', 'principal', 'legacy_unknown')",
            name="actor_auth_mode_known",
        ),
        CheckConstraint(
            "(actor_auth_mode = 'principal' AND actor_credential_id IS NOT NULL) OR "
            "(actor_auth_mode IN ('disabled', 'shared_key', 'legacy_unknown') "
            "AND actor_credential_id IS NULL)",
            name="actor_credential_shape",
        ),
        CheckConstraint(
            "actor_auth_mode = 'principal' OR "
            f"(actor_tenant_id = '{LEGACY_TENANT_ID}' "
            f"AND actor_principal_id = '{LEGACY_PRINCIPAL_ID}' "
            "AND actor_role = 'tenant_admin')",
            name="compatibility_actor_shape",
        ),
        Index(
            "ix_query_logs_actor_created",
            "actor_tenant_id",
            "actor_principal_id",
            "created_at",
        ),
    )

    query_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    image_id: Mapped[str | None] = mapped_column(
        ForeignKey("image_assets.image_id", ondelete="SET NULL")
    )
    query_type: Mapped[str] = mapped_column(String(40), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    request_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    answer_json: Mapped[JsonObject] = mapped_column(JSON, nullable=False)
    actor_tenant_id: Mapped[str] = mapped_column(String(36), nullable=False)
    actor_principal_id: Mapped[str] = mapped_column(String(36), nullable=False)
    actor_credential_id: Mapped[str | None] = mapped_column(String(36))
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_auth_mode: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    job: Mapped[AnalysisJob] = relationship(
        back_populates="queries",
        foreign_keys=[job_id],
    )


class Tenant(TimestampMixin, Base):
    """Tenant lifecycle state used by principal authentication and authorization."""

    __tablename__ = "tenants"
    __table_args__ = (
        CheckConstraint("length(tenant_id) = 36", name="tenant_id_length"),
        CheckConstraint(
            f"tenant_id GLOB 'tnt_{_HEX_32_GLOB}'",
            name="tenant_id_canonical",
        ),
        CheckConstraint("length(slug) BETWEEN 1 AND 63", name="slug_length"),
        CheckConstraint("slug = lower(slug)", name="slug_lowercase"),
        CheckConstraint(
            "slug NOT GLOB '*[^a-z0-9-]*' "
            "AND substr(slug, 1, 1) GLOB '[a-z0-9]' "
            "AND substr(slug, -1, 1) GLOB '[a-z0-9]'",
            name="slug_canonical",
        ),
        CheckConstraint(
            "length(trim(display_name)) BETWEEN 1 AND 255",
            name="display_name_length",
        ),
        CheckConstraint("enabled IN (0, 1)", name="enabled_boolean"),
        CheckConstraint("version >= 1", name="version_positive"),
        Index("ix_tenants_enabled", "enabled"),
    )

    tenant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class Principal(TimestampMixin, Base):
    """Human or service identity scoped to exactly one tenant."""

    __tablename__ = "principals"
    __table_args__ = (
        CheckConstraint("length(principal_id) = 36", name="principal_id_length"),
        CheckConstraint(
            f"principal_id GLOB 'prn_{_HEX_32_GLOB}'",
            name="principal_id_canonical",
        ),
        CheckConstraint("length(handle) BETWEEN 1 AND 64", name="handle_length"),
        CheckConstraint("handle = lower(handle)", name="handle_lowercase"),
        CheckConstraint(
            "handle NOT GLOB '*[^a-z0-9._-]*' "
            "AND substr(handle, 1, 1) GLOB '[a-z0-9]' "
            "AND substr(handle, -1, 1) GLOB '[a-z0-9]'",
            name="handle_canonical",
        ),
        CheckConstraint(
            "length(trim(display_name)) BETWEEN 1 AND 255",
            name="display_name_length",
        ),
        CheckConstraint("kind IN ('user', 'service')", name="kind_known"),
        CheckConstraint(
            "role IN ('tenant_admin', 'analyst', 'viewer')",
            name="role_known",
        ),
        CheckConstraint("enabled IN (0, 1)", name="enabled_boolean"),
        CheckConstraint("version >= 1", name="version_positive"),
        UniqueConstraint("tenant_id", "handle", name="uq_principals_tenant_handle"),
        UniqueConstraint(
            "principal_id",
            "tenant_id",
            name="uq_principals_principal_tenant",
        ),
        Index("ix_principals_tenant_enabled", "tenant_id", "enabled"),
    )

    principal_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.tenant_id", ondelete="RESTRICT"), nullable=False
    )
    handle: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class ApiCredential(TimestampMixin, Base):
    """Revocable bearer credential metadata; raw credential tokens are never stored."""

    __tablename__ = "api_credentials"
    __table_args__ = (
        CheckConstraint("length(credential_id) = 36", name="credential_id_length"),
        CheckConstraint(
            f"credential_id GLOB 'crd_{_HEX_32_GLOB}'",
            name="credential_id_canonical",
        ),
        CheckConstraint("length(trim(label)) BETWEEN 1 AND 120", name="label_length"),
        CheckConstraint("length(token_digest) = 32", name="token_digest_32_bytes"),
        CheckConstraint("enabled IN (0, 1)", name="enabled_boolean"),
        CheckConstraint("version >= 1", name="version_positive"),
        UniqueConstraint("token_digest", name="uq_api_credentials_token_digest"),
        UniqueConstraint(
            "credential_id",
            "principal_id",
            name="uq_api_credentials_credential_principal",
        ),
        Index("ix_api_credentials_principal_enabled", "principal_id", "enabled"),
    )

    credential_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    principal_id: Mapped[str] = mapped_column(
        ForeignKey("principals.principal_id", ondelete="RESTRICT"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    token_digest: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(credential_id={self.credential_id!r}, "
            f"principal_id={self.principal_id!r}, token_digest=<redacted>)"
        )


class IdentityAuditEvent(Base):
    """Append-only audit fact for identity lifecycle changes."""

    __tablename__ = "identity_audit_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ("
            "'tenant.created', 'tenant.enabled', 'tenant.disabled', "
            "'principal.created', 'principal.enabled', 'principal.disabled', "
            "'credential.issued', 'credential.enabled', 'credential.disabled', "
            "'credential.revoked')",
            name="event_type_known",
        ),
        CheckConstraint(
            "actor_kind IN ('operator_cli', 'principal', 'migration', 'system')",
            name="actor_kind_known",
        ),
        CheckConstraint(
            "(actor_kind = 'principal' AND actor_principal_id IS NOT NULL) OR "
            "(actor_kind <> 'principal' AND actor_principal_id IS NULL)",
            name="actor_principal_shape",
        ),
        CheckConstraint(
            "credential_id IS NULL OR principal_id IS NOT NULL",
            name="credential_requires_principal",
        ),
        Index("ix_identity_audit_tenant_occurred", "tenant_id", "occurred_at"),
        Index("ix_identity_audit_principal_occurred", "principal_id", "occurred_at"),
        Index("ix_identity_audit_credential_occurred", "credential_id", "occurred_at"),
    )

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.tenant_id", ondelete="RESTRICT"), nullable=False
    )
    principal_id: Mapped[str | None] = mapped_column(
        ForeignKey("principals.principal_id", ondelete="RESTRICT")
    )
    credential_id: Mapped[str | None] = mapped_column(
        ForeignKey("api_credentials.credential_id", ondelete="RESTRICT")
    )
    actor_principal_id: Mapped[str | None] = mapped_column(
        ForeignKey("principals.principal_id", ondelete="RESTRICT")
    )
    actor_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    metadata_json: Mapped[JsonObject] = mapped_column(JSON, default=dict, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


# Alembic owns production schema changes and installs the same triggers in revision
# f5c1d8a4b2e9. These metadata listeners cover isolated/test databases built with
# ``Base.metadata.create_all()`` so their direct-SQL integrity guarantees do not differ.
event.listen(
    IdentityAuditEvent.__table__,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        """
        CREATE TRIGGER trg_identity_audit_events_no_update
        BEFORE UPDATE ON identity_audit_events
        BEGIN
            SELECT RAISE(ABORT, 'identity audit events are append-only');
        END
        """
    ).execute_if(dialect="sqlite"),
)

# Production and demo databases must use Alembic, whose identity migration creates these rows and
# records the matching audit facts. Isolated tests still build disposable SQLite schemas with
# ``Base.metadata.create_all()``; seed only the fixed compatibility records there so callers can
# explicitly create jobs for disabled/shared-key compatibility identities. Deliberately do not
# synthesize audit events in test-only schemas because no migration occurred.
event.listen(
    Base.metadata,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        f"""
        INSERT OR IGNORE INTO tenants
            (tenant_id, slug, display_name, enabled, version, created_at, updated_at)
        VALUES
            ('{LEGACY_TENANT_ID}', 'legacy-local', 'Legacy local tenant', 1, 1,
             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
    ).execute_if(dialect="sqlite"),
)
event.listen(
    Base.metadata,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        f"""
        INSERT OR IGNORE INTO principals
            (principal_id, tenant_id, handle, display_name, kind, role, enabled, version,
             created_at, updated_at)
        VALUES
            ('{LEGACY_PRINCIPAL_ID}', '{LEGACY_TENANT_ID}', 'legacy-local',
             'Legacy local service', 'service', 'tenant_admin', 1, 1,
             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
    ).execute_if(dialect="sqlite"),
)
event.listen(
    IdentityAuditEvent.__table__,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        """
        CREATE TRIGGER trg_identity_audit_events_no_delete
        BEFORE DELETE ON identity_audit_events
        BEGIN
            SELECT RAISE(ABORT, 'identity audit events are append-only');
        END
        """
    ).execute_if(dialect="sqlite"),
)
