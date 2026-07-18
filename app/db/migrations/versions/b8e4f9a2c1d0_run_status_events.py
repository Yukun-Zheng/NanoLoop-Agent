"""run status events

Revision ID: b8e4f9a2c1d0
Revises: 53eaa43adc19
Create Date: 2026-07-18
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "b8e4f9a2c1d0"
down_revision: str | None = "53eaa43adc19"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_status_events",
        sa.Column("event_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("from_status", sa.String(length=40), nullable=True),
        sa.Column("to_status", sa.String(length=40), nullable=False),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["segmentation_runs.run_id"],
            name=op.f("fk_run_status_events_run_id_segmentation_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_run_status_events")),
    )
    op.create_index(
        "ix_run_status_events_run_event",
        "run_status_events",
        ["run_id", "event_id"],
        unique=False,
    )
    # Existing databases cannot reconstruct past transitions. Preserve an honest
    # one-event snapshot at the last known update time so every legacy run still
    # has an auditable starting point after the migration.
    op.execute(
        """
        INSERT INTO run_status_events (
            run_id, from_status, to_status, error_code, error_message, created_at
        )
        SELECT run_id, NULL, status, error_code, error_message, updated_at
        FROM segmentation_runs
        """
    )


def downgrade() -> None:
    op.drop_index("ix_run_status_events_run_event", table_name="run_status_events")
    op.drop_table("run_status_events")
