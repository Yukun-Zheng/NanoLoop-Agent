"""box revision ledger

Revision ID: d9f3b6c2a1e7
Revises: c4d7a1e6f2b9
Create Date: 2026-07-18
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "d9f3b6c2a1e7"
down_revision: str | None = "c4d7a1e6f2b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "roi_box_revisions",
        sa.Column("revision_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("box_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("box_count >= 0", name=op.f("ck_roi_box_revisions_box_count_nonnegative")),
        sa.CheckConstraint("revision >= 0", name=op.f("ck_roi_box_revisions_revision_nonnegative")),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["image_assets.image_id"],
            name=op.f("fk_roi_box_revisions_image_id_image_assets"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("revision_id", name=op.f("pk_roi_box_revisions")),
        sa.UniqueConstraint("image_id", "revision", name="image_box_revision"),
    )
    op.create_index(
        "ix_roi_box_revisions_image_revision",
        "roi_box_revisions",
        ["image_id", "revision"],
        unique=False,
    )
    op.execute(
        """
        INSERT INTO roi_box_revisions (image_id, revision, box_count, created_at)
        SELECT image_id, 0, 0, created_at FROM image_assets
        """
    )
    op.execute(
        """
        INSERT OR IGNORE INTO roi_box_revisions (image_id, revision, box_count, created_at)
        SELECT image_id, revision, COUNT(*), MIN(created_at)
        FROM roi_boxes
        GROUP BY image_id, revision
        """
    )
    op.execute(
        """
        INSERT OR IGNORE INTO roi_box_revisions (image_id, revision, box_count, created_at)
        SELECT image_id, box_revision, 0, updated_at
        FROM image_assets
        """
    )


def downgrade() -> None:
    op.drop_index("ix_roi_box_revisions_image_revision", table_name="roi_box_revisions")
    op.drop_table("roi_box_revisions")
