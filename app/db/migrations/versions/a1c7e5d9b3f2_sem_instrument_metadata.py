"""Persist normalized SEM instrument metadata and scale provenance.

Revision ID: a1c7e5d9b3f2
Revises: f8a2c6d4e9b1
"""

import sqlalchemy as sa
from alembic import op

revision: str = "a1c7e5d9b3f2"
down_revision: str | None = "f8a2c6d4e9b1"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "image_assets",
        sa.Column(
            "sem_metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "image_assets",
        sa.Column(
            "scale_source",
            sa.String(length=32),
            nullable=False,
            server_default="none",
        ),
    )
    op.execute(
        """
        UPDATE image_assets
        SET scale_source = 'manual'
        WHERE scale_nm_per_pixel IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_column("image_assets", "scale_source")
    op.drop_column("image_assets", "sem_metadata_json")
