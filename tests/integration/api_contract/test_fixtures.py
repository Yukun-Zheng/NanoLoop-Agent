from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.contracts.analyses import BoxSetDTO
from app.contracts.common import ApiResponse, HealthData
from app.contracts.models import ModelListData
from app.contracts.queries import UnifiedQueryResponse

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "api"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_checked_in_success_fixtures_match_public_contracts() -> None:
    ApiResponse[HealthData].model_validate(_load("health.degraded.json"))
    ApiResponse[ModelListData].model_validate(_load("models.unavailable.json"))
    ApiResponse[BoxSetDTO].model_validate(_load("boxes.revision.json"))
    ApiResponse[UnifiedQueryResponse].model_validate(_load("query.insufficient.json"))


def test_checked_in_error_fixture_matches_shared_envelope() -> None:
    response = ApiResponse[dict[str, object]].model_validate(
        _load("error.box-revision-conflict.json")
    )

    assert response.data is None
    assert response.error is not None
    assert response.error.code == "BOX_REVISION_CONFLICT"
