from pathlib import Path

import pytest

from scripts.mvp_fixture_smoke import run_fixture_mvp


@pytest.mark.integration
def test_fixture_profile_runs_real_backend_mvp_without_external_model(tmp_path: Path) -> None:
    result = run_fixture_mvp(tmp_path / "fixture-state")

    assert result["mode"] == "engineering_fixture_not_scientific"
    assert result["run_status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
    assert result["particle_count"] == 3
    assert result["configuration_schema_version"] == 3
    assert len(result["model_bundle_id"]) == 64
    assert result["execution_backend"].endswith(".DeterministicFixtureAdapter")
    assert len(result["export_sha256"]) == 64
    assert len(result["export_selection_sha256"]) == 64
