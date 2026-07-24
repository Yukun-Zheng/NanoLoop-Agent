"""persistent evidence-bearing chat conversations

Revision ID: f8a2c6d4e9b1
Revises: b4f2e8c6a1d9
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f8a2c6d4e9b1"
down_revision: str | None = "b4f2e8c6a1d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by", "tenant_id"],
            ["principals.principal_id", "principals.tenant_id"],
            name="fk_chat_conversations_creator_tenant",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id"],
            ["analysis_jobs.job_id", "analysis_jobs.tenant_id"],
            name="fk_chat_conversations_job_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_chat_conversations_tenant_job_updated",
        "chat_conversations",
        ["tenant_id", "job_id", "updated_at"],
    )
    op.create_table(
        "chat_messages",
        sa.Column("message_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("query_type", sa.String(length=40), nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=True),
        sa.Column("run_ids_json", sa.JSON(), nullable=False),
        sa.Column("material_context_json", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.String(length=16), nullable=True),
        sa.Column("outcome_code", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "query_type IN "
            "('auto', 'general_chat', 'analysis_data', 'material_knowledge', 'mixed')",
            name="chat_message_query_type_known",
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="chat_message_role_known",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["image_assets.image_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index(
        "ix_chat_messages_conversation_created",
        "chat_messages",
        ["conversation_id", "created_at"],
    )
    op.create_table(
        "chat_turn_evidence",
        sa.Column("message_id", sa.String(length=64), nullable=False),
        sa.Column("citations_json", sa.JSON(), nullable=False),
        sa.Column("data_evidence_json", sa.JSON(), nullable=False),
        sa.Column("tool_calls_json", sa.JSON(), nullable=False),
        sa.Column("limitations_json", sa.JSON(), nullable=False),
        sa.Column("llm_provider", sa.String(length=80), nullable=False),
        sa.Column("llm_model", sa.String(length=255), nullable=True),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("generation_time_ms", sa.Integer(), nullable=False),
        sa.Column("prompt_template_id", sa.String(length=100), nullable=False),
        sa.Column("prompt_template_sha256", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_messages.message_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )


def downgrade() -> None:
    op.drop_table("chat_turn_evidence")
    op.drop_index(
        "ix_chat_messages_conversation_created",
        table_name="chat_messages",
    )
    op.drop_table("chat_messages")
    op.drop_index(
        "ix_chat_conversations_tenant_job_updated",
        table_name="chat_conversations",
    )
    op.drop_table("chat_conversations")
