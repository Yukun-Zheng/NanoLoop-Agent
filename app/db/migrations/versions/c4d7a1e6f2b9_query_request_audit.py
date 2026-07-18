"""query request audit

Revision ID: c4d7a1e6f2b9
Revises: b8e4f9a2c1d0
Create Date: 2026-07-18
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "c4d7a1e6f2b9"
down_revision: str | None = "b8e4f9a2c1d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("query_logs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "request_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("query_logs") as batch_op:
        batch_op.drop_column("request_json")
