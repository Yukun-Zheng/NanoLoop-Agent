"""add durable bounded-agent control-plane state

Revision ID: c2f6a8d4e1b7
Revises: a1c7e5d9b3f2
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2f6a8d4e1b7"
down_revision: str | None = "a1c7e5d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_tasks",
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=False),
        sa.Column("created_credential_id", sa.String(length=36), nullable=True),
        sa.Column("auth_mode", sa.String(length=16), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=False),
        sa.Column("current_step", sa.Text(), nullable=True),
        sa.Column("budget_json", sa.JSON(), nullable=False),
        sa.Column("context_json", sa.JSON(), nullable=False),
        sa.Column("observations_json", sa.JSON(), nullable=False),
        sa.Column("user_inputs_json", sa.JSON(), nullable=False),
        sa.Column("pending_action_json", sa.JSON(), nullable=True),
        sa.Column("next_wakeup_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("waiting_question", sa.Text(), nullable=True),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("final_evidence_refs_json", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("model_provider", sa.String(length=80), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('created', 'running', 'waiting_for_approval', "
            "'waiting_for_input', 'waiting_for_external', "
            "'completed', 'failed', 'cancelled')",
            name=op.f("ck_agent_tasks_agent_task_status_known"),
        ),
        sa.CheckConstraint(
            "auth_mode IN ('disabled', 'shared_key', 'principal')",
            name=op.f("ck_agent_tasks_agent_task_auth_mode_known"),
        ),
        sa.CheckConstraint(
            "(auth_mode = 'principal' AND created_credential_id IS NOT NULL) OR "
            "(auth_mode IN ('disabled', 'shared_key') AND created_credential_id IS NULL)",
            name=op.f("ck_agent_tasks_agent_task_credential_shape"),
        ),
        sa.CheckConstraint(
            "step_count >= 0",
            name=op.f("ck_agent_tasks_agent_task_step_count_nonnegative"),
        ),
        sa.CheckConstraint(
            "failure_count >= 0",
            name=op.f("ck_agent_tasks_agent_task_failure_count_nonnegative"),
        ),
        sa.CheckConstraint(
            "version >= 1",
            name=op.f("ck_agent_tasks_agent_task_version_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["created_credential_id"],
            ["api_credentials.credential_id"],
            name=op.f("fk_agent_tasks_created_credential_id_api_credentials"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by", "tenant_id"],
            ["principals.principal_id", "principals.tenant_id"],
            name=op.f("fk_agent_tasks_creator_tenant"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id"],
            ["analysis_jobs.job_id", "analysis_jobs.tenant_id"],
            name=op.f("fk_agent_tasks_job_tenant"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("task_id", name=op.f("pk_agent_tasks")),
    )
    op.create_index(
        "ix_agent_tasks_tenant_job_updated",
        "agent_tasks",
        ["tenant_id", "job_id", "updated_at"],
        unique=False,
    )

    op.create_table(
        "agent_task_events",
        sa.Column("event_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["agent_tasks.task_id"],
            name=op.f("fk_agent_task_events_task_id_agent_tasks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_agent_task_events")),
        sa.UniqueConstraint(
            "task_id",
            "sequence",
            name="uq_agent_task_events_task_sequence",
        ),
    )
    op.create_index(
        "ix_agent_task_events_task_created",
        "agent_task_events",
        ["task_id", "created_at"],
        unique=False,
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_task_events_no_update
        BEFORE UPDATE ON agent_task_events
        BEGIN
            SELECT RAISE(ABORT, 'agent task events are append-only');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_task_events_no_delete
        BEFORE DELETE ON agent_task_events
        BEGIN
            SELECT RAISE(ABORT, 'agent task events are append-only');
        END
        """
    )

    op.create_table(
        "agent_approvals",
        sa.Column("approval_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("action_id", sa.String(length=64), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("tool_arguments_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=36), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name=op.f("ck_agent_approvals_agent_approval_status_known"),
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by"],
            ["principals.principal_id"],
            name=op.f("fk_agent_approvals_resolved_by_principals"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["agent_tasks.task_id"],
            name=op.f("fk_agent_approvals_task_id_agent_tasks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("approval_id", name=op.f("pk_agent_approvals")),
        sa.UniqueConstraint("action_id", name="uq_agent_approvals_action"),
    )
    op.create_index(
        "ix_agent_approvals_task_status",
        "agent_approvals",
        ["task_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_approvals_task_status", table_name="agent_approvals")
    op.drop_table("agent_approvals")
    op.drop_index("ix_agent_task_events_task_created", table_name="agent_task_events")
    op.drop_table("agent_task_events")
    op.drop_index("ix_agent_tasks_tenant_job_updated", table_name="agent_tasks")
    op.drop_table("agent_tasks")
