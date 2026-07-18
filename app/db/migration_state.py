"""Read-only Alembic revision checks shared by runtime health diagnostics."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


@lru_cache(maxsize=1)
def expected_alembic_heads() -> tuple[str, ...]:
    """Return the packaged migration heads used by ``alembic upgrade head``."""

    config = Config()
    migrations = Path(__file__).resolve().parent / "migrations"
    config.set_main_option("script_location", str(migrations))
    return tuple(sorted(ScriptDirectory.from_config(config).get_heads()))
