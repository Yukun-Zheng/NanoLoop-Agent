"""Bounds for valid-looking but pathological public collection inputs."""

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from app.contracts.enums import KnowledgeSourceType
from app.contracts.knowledge import IngestDocumentMetadata, RetrievalRequest
from app.contracts.limits import MAX_MATERIAL_ALIAS_CHARS, MAX_MATERIAL_ALIASES
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
