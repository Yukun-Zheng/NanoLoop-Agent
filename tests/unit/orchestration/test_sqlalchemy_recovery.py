from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Lock, Thread, get_ident

import pytest
from sqlalchemy import event, func, select

from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityTier,
    RoiMode,
)
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import Settings
from app.core.errors import JobStateConflictError, ResourceNotFoundError
from app.db.base import Base
from app.db.models import (
    AnalysisJob,
    ImageAsset,
    ModelRegistryRecord,
    RunStatusEvent,
    SegmentationRun,
)
from app.db.session import Database
from app.orchestration import (
    RECOVERY_STATUSES,
    RESTART_ERROR_CODE,
    STALE_ACTIVE_STATUSES,
    InlineDispatcher,
    RecoverableRun,
    RecoveryRequeueBlockedError,
    SqlAlchemyRunRecoveryStore,
    StaleRunPolicy,
    StartupRecovery,
)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    configured = Settings(app_env="test", database_url=f"sqlite:///{tmp_path / 'recovery.db'}")
    database = Database(configured)
    Base.metadata.create_all(database.engine)
    try:
        yield database
    finally:
        database.dispose()


def _seed_runs(
    database: Database,
    statuses: list[tuple[str, JobStatus]],
    *,
    job_id: str = "job-recovery",
    job_status: JobStatus = JobStatus.AGGREGATING,
    review_source: str = "model_inference",
) -> None:
    now = datetime.now(UTC)
    image_id = f"image-{job_id}"
    model_id = f"model-{job_id}"
    with database.session() as session:
        session.add(
            AnalysisJob(
                job_id=job_id,
                tenant_id=LEGACY_TENANT_ID,
                owner_principal_id=LEGACY_PRINCIPAL_ID,
                name="recovery fixture",
                status=job_status.value,
                config_json={"fixture": True},
            )
        )
        session.add(
            ImageAsset(
                image_id=image_id,
                job_id=job_id,
                filename="fixture.tif",
                storage_path=f"{job_id}/input/{image_id}/original.tif",
                sha256="a" * 64,
                width=128,
                height=96,
                bit_depth=16,
                sample_id="sample-recovery",
                experiment_conditions_json={},
                analysis_roi_json={"fixture": True},
                box_revision=3,
            )
        )
        session.add(
            ModelRegistryRecord(
                model_id=model_id,
                family=ModelFamily.UNET.value,
                variant=ModelVariant.GENERAL.value,
                quality_tier=QualityTier.BALANCED.value,
                version="test",
                adapter="tests.fake:FakeAdapter",
                status=ModelStatus.READY.value,
            )
        )
        session.flush()
        for index, (run_id, status) in enumerate(statuses):
            session.add(
                SegmentationRun(
                    run_id=run_id,
                    job_id=job_id,
                    image_id=image_id,
                    model_id=model_id,
                    roi_mode=RoiMode.BOXES.value,
                    box_revision=3,
                    threshold=0.42,
                    status=status.value,
                    inference_json={"threshold": 0.42, "min_area_px": 7},
                    run_config_json={
                        "model_id": model_id,
                        "scientific": {"profile": "fixture", "revision": 3},
                        "review_source": review_source,
                    },
                    paths_json={"pred_mask_path": f"old/{run_id}.png"},
                    runtime_ms=11,
                    created_at=now + timedelta(seconds=index),
                    updated_at=now + timedelta(seconds=index),
                )
            )


def test_list_by_status_is_filtered_and_deterministic(database: Database) -> None:
    _seed_runs(
        database,
        [
            ("run-queued", JobStatus.QUEUED),
            ("run-segmenting", JobStatus.SEGMENTING),
            ("run-completed", JobStatus.COMPLETED),
        ],
    )
    store = SqlAlchemyRunRecoveryStore(database.session_factory)

    records = store.list_by_status((JobStatus.QUEUED, JobStatus.SEGMENTING))

    assert records == [
        RecoverableRun("run-queued", JobStatus.QUEUED),
        RecoverableRun("run-segmenting", JobStatus.SEGMENTING),
    ]
    assert store.list_by_status(()) == []


def test_all_stale_active_stages_fail_through_state_machine_and_aggregate_job(
    database: Database,
) -> None:
    statuses = [(f"run-{status.value.lower()}", status) for status in STALE_ACTIVE_STATUSES]
    _seed_runs(database, statuses)
    store = SqlAlchemyRunRecoveryStore(database)

    for run_id, _status in statuses:
        store.mark_failed(
            run_id,
            error_code=RESTART_ERROR_CODE,
            error_message="interrupted during restart",
        )

    with database.session() as session:
        runs = session.scalars(select(SegmentationRun).order_by(SegmentationRun.run_id)).all()
        job = session.get(AnalysisJob, "job-recovery")
        assert job is not None
        assert {JobStatus(run.status) for run in runs} == {JobStatus.FAILED}
        assert all(run.error_code == RESTART_ERROR_CODE for run in runs)
        assert all(run.error_message == "interrupted during restart" for run in runs)
        assert JobStatus(job.status) == JobStatus.FAILED
        assert job.error_code == RESTART_ERROR_CODE
        events = session.scalars(select(RunStatusEvent).order_by(RunStatusEvent.run_id)).all()
        assert len(events) == len(statuses)
        assert all(event.to_status == JobStatus.FAILED.value for event in events)


def test_failure_aggregation_preserves_partial_success(database: Database) -> None:
    _seed_runs(
        database,
        [
            ("run-completed", JobStatus.COMPLETED),
            ("run-stale", JobStatus.ANALYZING),
        ],
    )
    store = SqlAlchemyRunRecoveryStore(database)

    store.mark_failed(
        "run-stale",
        error_code=RESTART_ERROR_CODE,
        error_message="restart",
    )

    with database.session() as session:
        job = session.get(AnalysisJob, "job-recovery")
        assert job is not None
        assert JobStatus(job.status) == JobStatus.COMPLETED_WITH_WARNINGS
        assert job.error_code == RESTART_ERROR_CODE


def test_terminal_or_unknown_runs_are_never_overwritten(database: Database) -> None:
    _seed_runs(database, [("run-completed", JobStatus.COMPLETED)])
    store = SqlAlchemyRunRecoveryStore(database)

    with pytest.raises(JobStateConflictError):
        store.mark_failed(
            "run-completed",
            error_code=RESTART_ERROR_CODE,
            error_message="must not overwrite",
        )
    with pytest.raises(ResourceNotFoundError):
        store.mark_failed(
            "missing",
            error_code=RESTART_ERROR_CODE,
            error_message="missing",
        )

    with database.session() as session:
        completed = session.get(SegmentationRun, "run-completed")
        assert completed is not None
        assert JobStatus(completed.status) == JobStatus.COMPLETED
        assert completed.error_code is None


def test_requeue_fails_parent_and_clones_immutable_scientific_inputs(
    database: Database,
) -> None:
    _seed_runs(database, [("run-parent", JobStatus.SEGMENTING)])
    store = SqlAlchemyRunRecoveryStore(database, run_id_factory=lambda: "run-child")

    child_id = store.requeue(RecoverableRun("run-parent", JobStatus.SEGMENTING))

    assert child_id == "run-child"
    with database.session() as session:
        parent = session.get(SegmentationRun, "run-parent")
        child = session.get(SegmentationRun, "run-child")
        job = session.get(AnalysisJob, "job-recovery")
        assert parent is not None and child is not None and job is not None
        assert JobStatus(parent.status) == JobStatus.FAILED
        assert parent.error_code == RESTART_ERROR_CODE
        assert parent.threshold == 0.42
        assert parent.box_revision == 3
        assert child.parent_run_id == parent.run_id
        assert JobStatus(child.status) == JobStatus.QUEUED
        assert child.job_id == parent.job_id
        assert child.image_id == parent.image_id
        assert child.model_id == parent.model_id
        assert child.roi_mode == parent.roi_mode
        assert child.box_revision == parent.box_revision
        assert child.threshold == parent.threshold
        assert child.inference_json == parent.inference_json
        assert child.run_config_json == parent.run_config_json
        assert child.paths_json == {}
        assert child.runtime_ms is None
        assert child.error_code is None
        assert JobStatus(job.status) == JobStatus.QUEUED
        parent_events = session.scalars(
            select(RunStatusEvent).where(RunStatusEvent.run_id == parent.run_id)
        ).all()
        child_events = session.scalars(
            select(RunStatusEvent).where(RunStatusEvent.run_id == child.run_id)
        ).all()
        assert [event.to_status for event in parent_events] == [JobStatus.FAILED.value]
        assert [event.to_status for event in child_events] == [JobStatus.QUEUED.value]

    assert store.requeue(RecoverableRun("run-parent", JobStatus.SEGMENTING)) == "run-child"
    with database.session() as session:
        child_count = session.scalar(
            select(func.count())
            .select_from(SegmentationRun)
            .where(SegmentationRun.parent_run_id == "run-parent")
        )
        assert child_count == 1


def test_concurrent_sqlite_requeue_has_one_cas_winner_and_one_child(
    database: Database,
) -> None:
    _seed_runs(database, [("run-parent", JobStatus.SEGMENTING)])
    id_lock = Lock()
    id_calls = 0

    def next_run_id() -> str:
        nonlocal id_calls
        with id_lock:
            id_calls += 1
            return f"run-child-{id_calls}"

    store = SqlAlchemyRunRecoveryStore(database, run_id_factory=next_run_id)
    claim_barrier = Barrier(2)
    synchronized_threads: set[int] = set()
    synchronization_lock = Lock()

    def synchronize_first_claim(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if not statement.lstrip().upper().startswith("UPDATE SEGMENTATION_RUNS SET"):
            return
        thread_id = get_ident()
        with synchronization_lock:
            is_first_claim = thread_id not in synchronized_threads
            synchronized_threads.add(thread_id)
        if is_first_claim:
            claim_barrier.wait(timeout=2)

    event.listen(database.engine, "before_cursor_execute", synchronize_first_claim)
    results: list[str] = []
    errors: list[BaseException] = []
    result_lock = Lock()

    def requeue() -> None:
        try:
            result = store.requeue(RecoverableRun("run-parent", JobStatus.SEGMENTING))
        except BaseException as error:  # retain the exact worker failure for the main assertion
            with result_lock:
                errors.append(error)
        else:
            with result_lock:
                results.append(result)

    workers = [Thread(target=requeue), Thread(target=requeue)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
    finally:
        event.remove(database.engine, "before_cursor_execute", synchronize_first_claim)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert sorted(results) == ["run-child-1", "run-child-1"]
    assert id_calls == 1
    with database.session() as session:
        parent = session.get(SegmentationRun, "run-parent")
        children = session.scalars(
            select(SegmentationRun)
            .where(SegmentationRun.parent_run_id == "run-parent")
            .order_by(SegmentationRun.run_id)
        ).all()
        parent_events = session.scalars(
            select(RunStatusEvent).where(RunStatusEvent.run_id == "run-parent")
        ).all()
        child_events = session.scalars(
            select(RunStatusEvent).where(RunStatusEvent.run_id == "run-child-1")
        ).all()

    assert parent is not None
    assert JobStatus(parent.status) == JobStatus.FAILED
    assert [child.run_id for child in children] == ["run-child-1"]
    assert [event.to_status for event in parent_events] == [JobStatus.FAILED.value]
    assert [event.to_status for event in child_events] == [JobStatus.QUEUED.value]


def test_requeue_collision_rolls_back_parent_failure(database: Database) -> None:
    _seed_runs(
        database,
        [
            ("run-parent", JobStatus.SEGMENTING),
            ("run-existing", JobStatus.QUEUED),
        ],
    )
    store = SqlAlchemyRunRecoveryStore(database, run_id_factory=lambda: "run-existing")

    with pytest.raises(JobStateConflictError):
        store.requeue(RecoverableRun("run-parent", JobStatus.SEGMENTING))

    with database.session() as session:
        parent = session.get(SegmentationRun, "run-parent")
        assert parent is not None
        assert JobStatus(parent.status) == JobStatus.SEGMENTING
        assert parent.error_code is None


def test_corrected_mask_run_is_failed_instead_of_requeued_without_its_artifact(
    database: Database,
) -> None:
    _seed_runs(
        database,
        [("run-corrected", JobStatus.POSTPROCESSING)],
        review_source="corrected_mask",
    )
    store = SqlAlchemyRunRecoveryStore(database, run_id_factory=lambda: "run-lossy-child")

    with pytest.raises(RecoveryRequeueBlockedError, match="external artifact"):
        store.requeue(RecoverableRun("run-corrected", JobStatus.POSTPROCESSING))
    # A loser/retry observes the committed FAILED row, remains fail-closed, and does not append a
    # duplicate status event or synthesize a child.
    with pytest.raises(RecoveryRequeueBlockedError, match="external artifact"):
        store.requeue(RecoverableRun("run-corrected", JobStatus.POSTPROCESSING))

    with database.session() as session:
        parent = session.get(SegmentationRun, "run-corrected")
        child = session.get(SegmentationRun, "run-lossy-child")
        events = session.scalars(
            select(RunStatusEvent).where(RunStatusEvent.run_id == "run-corrected")
        ).all()
        assert parent is not None
        assert JobStatus(parent.status) == JobStatus.FAILED
        assert parent.error_code == RESTART_ERROR_CODE
        assert "automatic requeue was blocked" in (parent.error_message or "")
        assert parent.paths_json == {"pred_mask_path": "old/run-corrected.png"}
        assert child is None
        assert [event.to_status for event in events] == [JobStatus.FAILED.value]


def test_startup_requeue_reports_corrected_mask_failure_for_operator_attention(
    database: Database,
) -> None:
    _seed_runs(
        database,
        [("run-corrected", JobStatus.QUALITY_CHECKING)],
        review_source="corrected_mask",
    )
    executed: list[str] = []
    recovery = StartupRecovery(
        SqlAlchemyRunRecoveryStore(
            database,
            run_id_factory=lambda: "run-lossy-child",
        ),
        InlineDispatcher(executed.append),
        stale_policy=StaleRunPolicy.REQUEUE,
    )

    report = recovery.recover()

    assert executed == []
    assert report.failed_stale_run_ids == ("run-corrected",)
    assert report.requeued_run_ids == ()
    assert report.submitted_run_ids == ()
    assert report.requires_attention
    assert report.errors == (
        (
            "run-corrected",
            "requeue_blocked: corrected-mask recovery requires the original external "
            "artifact; a JSON-only replacement would not be reproducible",
        ),
    )


def test_requeue_never_clones_an_unrelated_failed_run(database: Database) -> None:
    _seed_runs(database, [("run-failed", JobStatus.FAILED)])
    with database.session() as session:
        failed = session.get(SegmentationRun, "run-failed")
        assert failed is not None
        failed.error_code = "INFERENCE_FAILED"
        failed.error_message = "ordinary analysis failure"
    store = SqlAlchemyRunRecoveryStore(database, run_id_factory=lambda: "run-child")

    with pytest.raises(JobStateConflictError):
        store.requeue(RecoverableRun("run-failed", JobStatus.SEGMENTING))

    with database.session() as session:
        child = session.get(SegmentationRun, "run-child")
        assert child is None


def test_startup_recovery_default_fail_policy_is_database_backed(database: Database) -> None:
    _seed_runs(
        database,
        [
            ("run-queued", JobStatus.QUEUED),
            ("run-stale", JobStatus.PREPROCESSING),
        ],
    )
    executed: list[str] = []
    recovery = StartupRecovery(
        SqlAlchemyRunRecoveryStore(database),
        InlineDispatcher(executed.append),
    )

    report = recovery.recover()

    assert executed == ["run-queued"]
    assert report.queued_run_ids == ("run-queued",)
    assert report.failed_stale_run_ids == ("run-stale",)
    with database.session() as session:
        stale = session.get(SegmentationRun, "run-stale")
        job = session.get(AnalysisJob, "job-recovery")
        assert stale is not None and job is not None
        assert JobStatus(stale.status) == JobStatus.FAILED
        assert JobStatus(job.status) == JobStatus.QUEUED


def test_startup_requeue_commits_child_before_dispatch(database: Database) -> None:
    _seed_runs(database, [("run-parent", JobStatus.QUALITY_CHECKING)])
    observed: list[tuple[str, JobStatus]] = []

    def task(run_id: str) -> None:
        with database.session() as session:
            child = session.get(SegmentationRun, run_id)
            assert child is not None
            observed.append((run_id, JobStatus(child.status)))

    recovery = StartupRecovery(
        SqlAlchemyRunRecoveryStore(database, run_id_factory=lambda: "run-child"),
        InlineDispatcher(task),
        stale_policy=StaleRunPolicy.REQUEUE,
    )

    report = recovery.recover()

    assert observed == [("run-child", JobStatus.QUEUED)]
    assert report.requeued_run_ids == ("run-child",)
    assert report.submitted_run_ids == ("run-child",)
    assert report.errors == ()


def test_store_query_covers_exact_startup_recovery_statuses(database: Database) -> None:
    _seed_runs(
        database,
        [(f"run-{index}", status) for index, status in enumerate(RECOVERY_STATUSES)],
    )
    store = SqlAlchemyRunRecoveryStore(database)

    records = store.list_by_status(RECOVERY_STATUSES)

    assert {record.status for record in records} == set(RECOVERY_STATUSES)
