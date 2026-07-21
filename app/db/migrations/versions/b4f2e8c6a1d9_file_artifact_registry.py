"""immutable tenant-scoped file artifact registry

Revision ID: b4f2e8c6a1d9
Revises: e7b3c1d9a5f2
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b4f2e8c6a1d9"
down_revision: str | None = "e7b3c1d9a5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_HEX_32_GLOB = "[0-9a-f]" * 32
_HEX_64_GLOB = "[0-9a-f]" * 64


def upgrade() -> None:
    # Both checks intentionally precede the first DDL statement. The registry is empty at
    # introduction and never invents rows by scanning paths or interpreting historical JSON.
    _assert_sqlite_foreign_keys_clean(boundary="before")
    _assert_existing_run_relationships()

    with op.batch_alter_table("segmentation_runs", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_segmentation_runs_run_image_job",
            ["run_id", "image_id", "job_id"],
        )

    op.create_table(
        "file_artifacts",
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=True),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("artifact_kind", sa.String(length=32), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "length(artifact_id) = 36",
            name=op.f("ck_file_artifacts_artifact_id_length"),
        ),
        sa.CheckConstraint(
            f"artifact_id GLOB 'art_{_HEX_32_GLOB}'",
            name=op.f("ck_file_artifacts_artifact_id_canonical"),
        ),
        sa.CheckConstraint(
            "artifact_kind IN "
            "('original_image', 'run_artifact', 'analysis_export', "
            "'corrected_mask_input')",
            name=op.f("ck_file_artifacts_artifact_kind_known"),
        ),
        sa.CheckConstraint(
            "(artifact_kind = 'original_image' AND image_id IS NOT NULL "
            "AND run_id IS NULL) OR "
            "(artifact_kind IN ('run_artifact', 'corrected_mask_input') "
            "AND image_id IS NOT NULL AND run_id IS NOT NULL) OR "
            "(artifact_kind = 'analysis_export' AND image_id IS NULL AND run_id IS NULL)",
            name=op.f("ck_file_artifacts_artifact_relationship_shape"),
        ),
        sa.CheckConstraint(
            "length(storage_path) BETWEEN 1 AND 4096 "
            "AND substr(storage_path, 1, 1) <> '/' "
            "AND substr(storage_path, -1, 1) <> '/' "
            "AND instr(storage_path, char(92)) = 0 "
            "AND instr(storage_path, char(0)) = 0 "
            "AND instr(storage_path, char(10)) = 0 "
            "AND instr(storage_path, char(13)) = 0 "
            "AND storage_path NOT LIKE '%//%' "
            "AND storage_path <> '.' AND storage_path <> '..' "
            "AND storage_path NOT LIKE './%' "
            "AND storage_path NOT LIKE '../%' "
            "AND storage_path NOT LIKE '%/./%' "
            "AND storage_path NOT LIKE '%/../%' "
            "AND storage_path NOT LIKE '%/.' "
            "AND storage_path NOT LIKE '%/..'",
            name=op.f("ck_file_artifacts_storage_path_managed_relative"),
        ),
        sa.CheckConstraint(
            "length(trim(filename)) BETWEEN 1 AND 255 "
            "AND filename = trim(filename) "
            "AND filename NOT IN ('.', '..') "
            "AND instr(filename, '/') = 0 "
            "AND instr(filename, char(92)) = 0 "
            "AND instr(filename, char(0)) = 0 "
            "AND instr(filename, char(10)) = 0 "
            "AND instr(filename, char(13)) = 0",
            name=op.f("ck_file_artifacts_filename_basename"),
        ),
        sa.CheckConstraint(
            "length(media_type) BETWEEN 3 AND 255 "
            "AND media_type = lower(media_type) "
            "AND instr(media_type, '/') > 1 "
            "AND instr(media_type, ' ') = 0 "
            "AND instr(media_type, char(0)) = 0 "
            "AND instr(media_type, char(10)) = 0 "
            "AND instr(media_type, char(13)) = 0",
            name=op.f("ck_file_artifacts_media_type_canonical"),
        ),
        sa.CheckConstraint(
            "length(sha256) = 64",
            name=op.f("ck_file_artifacts_sha256_length"),
        ),
        sa.CheckConstraint(
            f"sha256 GLOB '{_HEX_64_GLOB}'",
            name=op.f("ck_file_artifacts_sha256_canonical"),
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name=op.f("ck_file_artifacts_size_bytes_nonnegative"),
        ),
        sa.CheckConstraint(
            "state IN ('active', 'consumed', 'revoked')",
            name=op.f("ck_file_artifacts_state_known"),
        ),
        sa.CheckConstraint(
            "state <> 'consumed' OR artifact_kind = 'corrected_mask_input'",
            name=op.f("ck_file_artifacts_consumed_kind"),
        ),
        sa.CheckConstraint(
            "(state = 'active' AND consumed_at IS NULL AND revoked_at IS NULL) OR "
            "(state = 'consumed' AND consumed_at IS NOT NULL AND revoked_at IS NULL "
            "AND consumed_at >= created_at) OR "
            "(state = 'revoked' AND consumed_at IS NULL AND revoked_at IS NOT NULL "
            "AND revoked_at >= created_at)",
            name=op.f("ck_file_artifacts_state_timestamp_shape"),
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["analysis_jobs.job_id"],
            name="fk_file_artifacts_job",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["image_id", "job_id"],
            ["image_assets.image_id", "image_assets.job_id"],
            name="fk_file_artifacts_image_job",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "job_id"],
            ["segmentation_runs.run_id", "segmentation_runs.job_id"],
            name="fk_file_artifacts_run_job",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "image_id", "job_id"],
            [
                "segmentation_runs.run_id",
                "segmentation_runs.image_id",
                "segmentation_runs.job_id",
            ],
            name="fk_file_artifacts_run_image_job",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("artifact_id", name=op.f("pk_file_artifacts")),
        sa.UniqueConstraint(
            "storage_path",
            name="uq_file_artifacts_storage_path",
        ),
    )
    op.create_index(
        "ix_file_artifacts_job_kind_state",
        "file_artifacts",
        ["job_id", "artifact_kind", "state"],
        unique=False,
    )
    _create_registry_triggers()
    _assert_sqlite_foreign_keys_clean(boundary="after")


def downgrade() -> None:
    # Refuse before the first DDL statement: dropping audit facts is never an implicit downgrade.
    _assert_sqlite_foreign_keys_clean(boundary="before")
    _assert_registry_empty()

    op.execute("DROP TRIGGER IF EXISTS trg_file_artifacts_terminal_state")
    op.execute("DROP TRIGGER IF EXISTS trg_file_artifacts_immutable_facts")
    op.drop_index("ix_file_artifacts_job_kind_state", table_name="file_artifacts")
    op.drop_table("file_artifacts")

    with op.batch_alter_table("segmentation_runs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_segmentation_runs_run_image_job",
            type_="unique",
        )

    _assert_sqlite_foreign_keys_clean(boundary="after")


def _assert_existing_run_relationships() -> None:
    mismatch_count = int(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT count(*)
                FROM segmentation_runs AS run
                LEFT JOIN image_assets AS image
                  ON image.image_id = run.image_id AND image.job_id = run.job_id
                WHERE image.image_id IS NULL
                """
            )
        )
        or 0
    )
    if mismatch_count:
        raise RuntimeError("file artifact migration refused: segmentation run image/job mismatch")


def _assert_registry_empty() -> None:
    artifact_count = int(op.get_bind().scalar(sa.text("SELECT count(*) FROM file_artifacts")) or 0)
    if artifact_count:
        raise RuntimeError(
            "file artifact downgrade refused because registered artifact facts exist"
        )


def _create_registry_triggers() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return
    op.execute(
        """
        CREATE TRIGGER trg_file_artifacts_immutable_facts
        BEFORE UPDATE ON file_artifacts
        WHEN NEW.artifact_id IS NOT OLD.artifact_id
          OR NEW.job_id IS NOT OLD.job_id
          OR NEW.image_id IS NOT OLD.image_id
          OR NEW.run_id IS NOT OLD.run_id
          OR NEW.artifact_kind IS NOT OLD.artifact_kind
          OR NEW.storage_path IS NOT OLD.storage_path
          OR NEW.filename IS NOT OLD.filename
          OR NEW.media_type IS NOT OLD.media_type
          OR NEW.sha256 IS NOT OLD.sha256
          OR NEW.size_bytes IS NOT OLD.size_bytes
          OR NEW.created_at IS NOT OLD.created_at
        BEGIN
            SELECT RAISE(ABORT, 'file artifact facts are immutable');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_file_artifacts_terminal_state
        BEFORE UPDATE OF state, consumed_at, revoked_at ON file_artifacts
        WHEN OLD.state IN ('consumed', 'revoked')
          AND (NEW.state IS NOT OLD.state
               OR NEW.consumed_at IS NOT OLD.consumed_at
               OR NEW.revoked_at IS NOT OLD.revoked_at)
        BEGIN
            SELECT RAISE(ABORT, 'terminal file artifact state is immutable');
        END
        """
    )


def _assert_sqlite_foreign_keys_clean(*, boundary: str) -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"SQLite foreign key check failed {boundary} file artifact migration")
