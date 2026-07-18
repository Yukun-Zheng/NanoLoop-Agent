from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.session import Database
from app.main import _state_directory_lock, create_app
from app.operations.backup import BackupPreconditionError, StateDirectoryLock


def test_api_holds_shared_state_lock_for_complete_lifespan(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'nanoloop.db'}",
        output_root=tmp_path / "outputs",
        model_registry_path=tmp_path / "registry.yaml",
        model_snapshot_root=tmp_path / "model-snapshots",
        knowledge_source_dir=tmp_path / "knowledge-sources",
        faiss_index_path=tmp_path / "knowledge-index" / "faiss.index",
    )
    database = Database(settings)
    application = create_app(
        settings=settings,
        database=database,
        inference_gateway=object(),
        analysis_creation_service=object(),  # type: ignore[arg-type]
        analysis_application_service=object(),  # type: ignore[arg-type]
        knowledge_application_service=object(),  # type: ignore[arg-type]
        knowledge_source_store=object(),  # type: ignore[arg-type]
        query_application_service=object(),  # type: ignore[arg-type]
    )

    with StateDirectoryLock(tmp_path, exclusive=True):
        pass
    with TestClient(application):
        with pytest.raises(BackupPreconditionError, match="locked by another process"):
            StateDirectoryLock(tmp_path, exclusive=True).acquire()
        with StateDirectoryLock(tmp_path, exclusive=False):
            pass
    with StateDirectoryLock(tmp_path, exclusive=True):
        pass

    database.dispose()


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite:///:memory:",
        "sqlite:///file:memorydb?mode=memory&uri=true",
        "postgresql://localhost/nanoloop",
    ],
)
def test_non_file_sqlite_state_has_no_process_lock(database_url: str) -> None:
    assert _state_directory_lock(Settings(database_url=database_url)) is None
