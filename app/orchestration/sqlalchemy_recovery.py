"""SQLAlchemy persistence adapter for process-start run recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.contracts.enums import JobStatus
from app.core.errors import JobStateConflictError, ResourceNotFoundError
from app.core.state_machine import TERMINAL_STATUSES, aggregate_job_status, ensure_transition
from app.db.models import AnalysisJob, RunStatusEvent, SegmentationRun
from app.db.session import Database
from app.orchestration.recovery import (
    RECOVERY_STATUSES,
    RESTART_ERROR_CODE,
    STALE_ACTIVE_STATUSES,
    RecoverableRun,
    RecoveryRequeueBlockedError,
)

SessionFactory = Callable[[], Session]
RunIdFactory = Callable[[], str]
CORRECTED_MASK_REQUEUE_REASON = (
    "corrected-mask recovery requires the original external artifact; "
    "a JSON-only replacement would not be reproducible"
)


class SqlAlchemyRunRecoveryStore:
    """Recover durable runs with one short transaction per mutation.

    Failing a stale run always passes through the shared state machine. Requeue never rewinds an
    immutable row from an active stage to ``QUEUED``: it marks that row failed and creates a child
    whose scientific inputs are deep copies of the parent configuration. Corrected-mask runs are
    failed without a child because their external mask artifact cannot be represented by that copy.
    """

    def __init__(
        self,
        database_or_session_factory: Database | SessionFactory,
        *,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        if isinstance(database_or_session_factory, Database):
            self._session_factory: SessionFactory = database_or_session_factory.session_factory
        else:
            self._session_factory = database_or_session_factory
        self._run_id_factory = run_id_factory or self._new_run_id

    def list_by_status(self, statuses: tuple[JobStatus, ...]) -> list[RecoverableRun]:
        normalized = tuple(dict.fromkeys(statuses))
        if not normalized:
            return []
        with self._transaction() as session:
            rows = session.execute(
                select(SegmentationRun.run_id, SegmentationRun.status)
                .where(SegmentationRun.status.in_([status.value for status in normalized]))
                .order_by(SegmentationRun.created_at, SegmentationRun.run_id)
            ).all()
            return [
                RecoverableRun(run_id=run_id, status=JobStatus(status)) for run_id, status in rows
            ]

    def mark_failed(
        self,
        run_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        with self._transaction() as session:
            record = self._locked_run(session, run_id)
            current = JobStatus(record.status)
            if current == JobStatus.FAILED:
                self._aggregate_job(session, record.job_id, error_code=record.error_code)
                return
            if current in TERMINAL_STATUSES:
                raise JobStateConflictError(
                    "终态运行不能被启动恢复覆盖",
                    details={"run_id": run_id, "status": current.value},
                )
            if current not in RECOVERY_STATUSES:
                raise JobStateConflictError(
                    "运行不属于可恢复状态",
                    details={"run_id": run_id, "status": current.value},
                )

            # Every supported queued/active stage explicitly permits -> FAILED. This guard makes a
            # future state-machine tightening fail closed instead of bypassing it.
            ensure_transition(current, JobStatus.FAILED)
            record.status = JobStatus.FAILED.value
            record.error_code = error_code
            record.error_message = error_message
            session.add(
                RunStatusEvent(
                    run_id=record.run_id,
                    from_status=current.value,
                    to_status=JobStatus.FAILED.value,
                    error_code=error_code,
                    error_message=error_message,
                )
            )
            session.flush()
            self._aggregate_job(session, record.job_id, error_code=error_code)

    def requeue(self, run: RecoverableRun) -> str:
        if run.status not in STALE_ACTIVE_STATUSES:
            raise JobStateConflictError(
                "只有遗留活跃运行可创建恢复子运行",
                details={"run_id": run.run_id, "status": run.status.value},
            )
        ensure_transition(run.status, JobStatus.FAILED)

        blocked_reason: str | None = None
        child_run_id: str | None = None
        with self._transaction() as session:
            # This must be the transaction's first database statement. SQLite ignores
            # SELECT ... FOR UPDATE, while a conditional UPDATE obtains the writer lock and makes
            # the stale -> FAILED claim atomic. A competing caller resumes after this transaction
            # commits, observes no returned row, and reads the winner's child below.
            claimed_run_id = session.scalar(
                update(SegmentationRun)
                .where(
                    SegmentationRun.run_id == run.run_id,
                    SegmentationRun.status == run.status.value,
                )
                .values(
                    status=JobStatus.FAILED.value,
                    error_code=RESTART_ERROR_CODE,
                    error_message=(
                        f"Run was left in {run.status.value} when the process restarted; "
                        "restart recovery claim acquired"
                    ),
                )
                .returning(SegmentationRun.run_id)
                .execution_options(synchronize_session=False)
            )
            parent = session.get(SegmentationRun, run.run_id)
            if parent is None:
                raise ResourceNotFoundError(details={"resource": "run", "run_id": run.run_id})

            if claimed_run_id is None:
                current = JobStatus(parent.status)
                if current != JobStatus.FAILED or parent.error_code != RESTART_ERROR_CODE:
                    raise JobStateConflictError(
                        "运行状态已变化，不能按旧快照创建恢复子运行",
                        details={
                            "run_id": run.run_id,
                            "expected_status": run.status.value,
                            "status": current.value,
                            "error_code": parent.error_code,
                        },
                    )
                if parent.run_config_json.get("review_source") == "corrected_mask":
                    blocked_reason = CORRECTED_MASK_REQUEUE_REASON
                else:
                    existing_child = self._recovery_child(session, parent.run_id)
                    if existing_child is not None:
                        child_run_id = existing_child
                    else:
                        raise JobStateConflictError(
                            "恢复失败运行没有可复用的恢复子运行",
                            details={"run_id": run.run_id, "status": current.value},
                        )
            elif parent.run_config_json.get("review_source") == "corrected_mask":
                blocked_reason = CORRECTED_MASK_REQUEUE_REASON
                parent.status = JobStatus.FAILED.value
                parent.error_code = RESTART_ERROR_CODE
                parent.error_message = (
                    f"Run was left in {run.status.value} when the process restarted; "
                    f"automatic requeue was blocked because {blocked_reason}"
                )
                session.add(
                    RunStatusEvent(
                        run_id=parent.run_id,
                        from_status=run.status.value,
                        to_status=JobStatus.FAILED.value,
                        error_code=RESTART_ERROR_CODE,
                        error_message=parent.error_message,
                    )
                )
                session.flush()
                self._aggregate_job(
                    session,
                    parent.job_id,
                    error_code=RESTART_ERROR_CODE,
                )
            else:
                child_run_id = self._validated_new_run_id(session)
                child = SegmentationRun(
                    run_id=child_run_id,
                    job_id=parent.job_id,
                    image_id=parent.image_id,
                    model_id=parent.model_id,
                    roi_mode=parent.roi_mode,
                    box_revision=parent.box_revision,
                    threshold=parent.threshold,
                    status=JobStatus.QUEUED.value,
                    inference_json=deepcopy(parent.inference_json),
                    run_config_json=deepcopy(parent.run_config_json),
                    paths_json={},
                    runtime_ms=None,
                    parent_run_id=parent.run_id,
                    error_code=None,
                    error_message=None,
                )
                parent.status = JobStatus.FAILED.value
                parent.error_code = RESTART_ERROR_CODE
                parent.error_message = (
                    f"Run was left in {run.status.value} when the process restarted; "
                    f"replacement child {child_run_id} was queued"
                )
                session.add(child)
                session.flush()
                session.add_all(
                    [
                        RunStatusEvent(
                            run_id=parent.run_id,
                            from_status=run.status.value,
                            to_status=JobStatus.FAILED.value,
                            error_code=RESTART_ERROR_CODE,
                            error_message=parent.error_message,
                        ),
                        RunStatusEvent(
                            run_id=child.run_id,
                            from_status=None,
                            to_status=JobStatus.QUEUED.value,
                        ),
                    ]
                )
                self._aggregate_job(session, parent.job_id, error_code=RESTART_ERROR_CODE)

        if blocked_reason is not None:
            raise RecoveryRequeueBlockedError(run.run_id, blocked_reason)
        if child_run_id is None:  # pragma: no cover - both branches above assign an outcome
            raise RuntimeError("requeue completed without a replacement run")
        return child_run_id

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _locked_run(session: Session, run_id: str) -> SegmentationRun:
        record = session.scalar(
            select(SegmentationRun).where(SegmentationRun.run_id == run_id).with_for_update()
        )
        if record is None:
            raise ResourceNotFoundError(details={"resource": "run", "run_id": run_id})
        return record

    @staticmethod
    def _recovery_child(session: Session, parent_run_id: str) -> str | None:
        return session.scalar(
            select(SegmentationRun.run_id)
            .where(
                SegmentationRun.parent_run_id == parent_run_id,
            )
            .order_by(SegmentationRun.created_at, SegmentationRun.run_id)
            .limit(1)
        )

    def _validated_new_run_id(self, session: Session) -> str:
        run_id = self._run_id_factory()
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id.strip()) > 64:
            raise ValueError("run_id_factory must return a non-empty string of at most 64 chars")
        normalized = run_id.strip()
        if session.get(SegmentationRun, normalized) is not None:
            raise JobStateConflictError(
                "恢复子运行 ID 冲突",
                details={"run_id": normalized},
            )
        return normalized

    @staticmethod
    def _aggregate_job(
        session: Session,
        job_id: str,
        *,
        error_code: str | None,
    ) -> None:
        statuses = [
            JobStatus(value)
            for value in session.scalars(
                select(SegmentationRun.status).where(SegmentationRun.job_id == job_id)
            ).all()
        ]
        aggregate = aggregate_job_status(statuses)
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise ResourceNotFoundError(details={"resource": "job", "job_id": job_id})
        # Job status is a derived aggregate, not a rewind of any immutable run state.
        job.status = aggregate.value
        has_failed_run = any(status == JobStatus.FAILED for status in statuses)
        job.error_code = error_code if has_failed_run else None
        session.flush()

    @staticmethod
    def _new_run_id() -> str:
        return f"run_{uuid4().hex}"
