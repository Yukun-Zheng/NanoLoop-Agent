"""run execution provenance

Revision ID: e2a7c4d8f1b3
Revises: d9f3b6c2a1e7
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2a7c4d8f1b3"
down_revision: str | None = "d9f3b6c2a1e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows predate runtime provenance and must remain explicitly
    # unknown. Backfilling the run-creation build would misrepresent the worker
    # that actually executed the model.
    op.add_column(
        "segmentation_runs",
        sa.Column("execution_json", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("segmentation_runs", "execution_json")
