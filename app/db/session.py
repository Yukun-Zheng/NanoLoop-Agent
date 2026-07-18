"""Engine and session factory with SQLite safety pragmas."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.errors import JobStateConflictError
from app.db.models import SegmentationRun

IMMUTABLE_RUN_FIELDS = (
    "job_id",
    "image_id",
    "model_id",
    "roi_mode",
    "box_revision",
    "threshold",
    "inference_json",
    "run_config_json",
    "parent_run_id",
)


@event.listens_for(Session, "before_flush")
def protect_immutable_run_configuration(
    session: Session,
    _flush_context: object,
    _instances: object,
) -> None:
    """Reject mutation of scientific inputs after a run row has been created."""

    for instance in session.dirty:
        if not isinstance(instance, SegmentationRun):
            continue
        state = inspect(instance)
        changed = [
            field
            for field in IMMUTABLE_RUN_FIELDS
            if state.attrs[field].history.has_changes()
        ]
        if changed:
            raise JobStateConflictError(
                "运行配置不可修改；请创建新的 run",
                details={"run_id": instance.run_id, "immutable_fields": changed},
            )


def create_database_engine(settings: Settings) -> Engine:
    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )

    if settings.database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def configure_sqlite(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


class Database:
    """Own an engine and explicit SQLAlchemy session factory."""

    def __init__(self, settings: Settings) -> None:
        self.engine = create_database_engine(settings)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
