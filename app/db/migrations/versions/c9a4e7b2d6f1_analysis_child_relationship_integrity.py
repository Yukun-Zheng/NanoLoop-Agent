"""analysis child relationship integrity

Revision ID: c9a4e7b2d6f1
Revises: a6d2e9f4c7b1
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9a4e7b2d6f1"
down_revision: str | None = "a6d2e9f4c7b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    _assert_relationships_consistent()

    with op.batch_alter_table("image_assets", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_image_assets_image_job",
            ["image_id", "job_id"],
        )

    with op.batch_alter_table("segmentation_runs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_segmentation_runs_image_id_image_assets",
            type_="foreignkey",
        )
        batch_op.create_unique_constraint(
            "uq_segmentation_runs_run_job",
            ["run_id", "job_id"],
        )
        batch_op.create_foreign_key(
            "fk_segmentation_runs_image_job",
            "image_assets",
            ["image_id", "job_id"],
            ["image_id", "job_id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_segmentation_runs_parent_job",
            "segmentation_runs",
            ["parent_run_id", "job_id"],
            ["run_id", "job_id"],
        )

    with op.batch_alter_table("query_logs", recreate="always") as batch_op:
        batch_op.create_foreign_key(
            "fk_query_logs_image_job",
            "image_assets",
            ["image_id", "job_id"],
            ["image_id", "job_id"],
        )

    _assert_sqlite_foreign_keys_clean()


def downgrade() -> None:
    with op.batch_alter_table("query_logs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_query_logs_image_job",
            type_="foreignkey",
        )

    with op.batch_alter_table("segmentation_runs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "fk_segmentation_runs_parent_job",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_segmentation_runs_image_job",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "uq_segmentation_runs_run_job",
            type_="unique",
        )
        batch_op.create_foreign_key(
            "fk_segmentation_runs_image_id_image_assets",
            "image_assets",
            ["image_id"],
            ["image_id"],
            ondelete="CASCADE",
        )

    with op.batch_alter_table("image_assets", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_image_assets_image_job",
            type_="unique",
        )

    _assert_sqlite_foreign_keys_clean()


def _assert_relationships_consistent() -> None:
    """Reject ambiguous legacy relationships before the first DDL statement."""

    bind = op.get_bind()
    run_image_mismatches = int(
        bind.scalar(
            sa.text(
                """
                SELECT count(*)
                FROM segmentation_runs AS run
                LEFT JOIN image_assets AS image ON image.image_id = run.image_id
                WHERE image.image_id IS NULL OR image.job_id != run.job_id
                """
            )
        )
        or 0
    )
    if run_image_mismatches:
        raise RuntimeError(
            "analysis relationship migration refused: segmentation run image/job mismatch"
        )

    query_image_mismatches = int(
        bind.scalar(
            sa.text(
                """
                SELECT count(*)
                FROM query_logs AS query
                LEFT JOIN image_assets AS image ON image.image_id = query.image_id
                WHERE query.image_id IS NOT NULL
                  AND (image.image_id IS NULL OR image.job_id != query.job_id)
                """
            )
        )
        or 0
    )
    if query_image_mismatches:
        raise RuntimeError(
            "analysis relationship migration refused: query image/job mismatch"
        )

    parent_job_mismatches = int(
        bind.scalar(
            sa.text(
                """
                SELECT count(*)
                FROM segmentation_runs AS child
                LEFT JOIN segmentation_runs AS parent
                    ON parent.run_id = child.parent_run_id
                WHERE child.parent_run_id IS NOT NULL
                  AND (parent.run_id IS NULL OR parent.job_id != child.job_id)
                """
            )
        )
        or 0
    )
    if parent_job_mismatches:
        raise RuntimeError(
            "analysis relationship migration refused: review parent crosses jobs"
        )


def _assert_sqlite_foreign_keys_clean() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(
            "SQLite foreign key check failed after analysis relationship migration"
        )
