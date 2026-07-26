"""Bounds for valid-looking but pathological public collection inputs."""

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from app.contracts.analyses import CreateRunsRequest
from app.contracts.enums import KnowledgeSourceType, RoiMode
from app.contracts.knowledge import IngestDocumentMetadata, RetrievalRequest
from app.contracts.limits import MAX_MATERIAL_ALIAS_CHARS, MAX_MATERIAL_ALIASES
from app.contracts.models import ModelRecommendationRequest
from app.contracts.queries import MaterialContext


def _metadata(**updates: object) -> IngestDocumentMetadata:
    values: dict[str, object] = {
        "title": "Bounded source",
        "source_type": KnowledgeSourceType.PAPER,
        "citation_text": "Citation",
        "license_note": "Licensed for test",
    }
    values.update(updates)
    return IngestDocumentMetadata.model_validate(values)


@pytest.mark.parametrize(
    "factory",
    [
        lambda aliases: _metadata(material_aliases=aliases),
        lambda aliases: RetrievalRequest(query="evidence", material_aliases=aliases),
        lambda aliases: MaterialContext(aliases=aliases),
    ],
)
def test_material_alias_collections_have_a_stable_public_bound(
    factory: Callable[[list[str]], object],
) -> None:
    with pytest.raises(ValidationError):
        factory(["alias"] * (MAX_MATERIAL_ALIASES + 1))
    with pytest.raises(ValidationError):
        factory(["x" * (MAX_MATERIAL_ALIAS_CHARS + 1)])


def test_material_aliases_are_trimmed_and_empty_values_are_rejected() -> None:
    assert _metadata(material_aliases=["  TiO2  "]).material_aliases == ["TiO2"]
    with pytest.raises(ValidationError):
        _metadata(material_aliases=["   "])


def test_run_model_assignments_require_one_exact_model_per_image() -> None:
    request = CreateRunsRequest(
        image_ids=["img_1", "img_2"],
        model_ids=["model_small", "model_large"],
        model_assignments={"img_1": "model_small", "img_2": "model_large"},
        roi_mode=RoiMode.FULL_IMAGE,
    )
    assert request.model_assignments == {
        "img_1": "model_small",
        "img_2": "model_large",
    }

    with pytest.raises(ValidationError, match="exactly one model"):
        CreateRunsRequest(
            image_ids=["img_1", "img_2"],
            model_ids=["model_small"],
            model_assignments={"img_1": "model_small"},
            roi_mode=RoiMode.FULL_IMAGE,
        )

    with pytest.raises(ValidationError, match="exactly the models"):
        CreateRunsRequest(
            image_ids=["img_1", "img_2"],
            model_ids=["model_small", "unused_model"],
            model_assignments={"img_1": "model_small", "img_2": "model_small"},
            roi_mode=RoiMode.FULL_IMAGE,
        )


def test_unassigned_run_request_still_limits_model_comparisons_to_three() -> None:
    with pytest.raises(ValidationError, match="at most 3 models"):
        CreateRunsRequest(
            image_ids=["img_1"],
            model_ids=["model_1", "model_2", "model_3", "model_4"],
            roi_mode=RoiMode.FULL_IMAGE,
        )


def test_auto_model_profile_requires_job_context() -> None:
    with pytest.raises(ValidationError, match="auto_profile requires job_id"):
        ModelRecommendationRequest(
            image_id="img_1",
            roi_mode=RoiMode.FULL_IMAGE,
            auto_profile=True,
        )
