from __future__ import annotations

from typing import cast

from sqlalchemy import ForeignKeyConstraint, Table, UniqueConstraint
from sqlalchemy.orm import configure_mappers

from app.db.models import ImageAsset, QueryLog, SegmentationRun


def test_analysis_relationship_models_declare_composite_scope_constraints() -> None:
    image_unique = _unique_constraint(ImageAsset, "uq_image_assets_image_job")
    run_unique = _unique_constraint(SegmentationRun, "uq_segmentation_runs_run_job")
    run_image = _foreign_key(SegmentationRun, "fk_segmentation_runs_image_job")
    run_parent = _foreign_key(SegmentationRun, "fk_segmentation_runs_parent_job")
    query_image = _foreign_key(QueryLog, "fk_query_logs_image_job")

    assert tuple(image_unique.columns.keys()) == ("image_id", "job_id")
    assert tuple(run_unique.columns.keys()) == ("run_id", "job_id")
    assert _foreign_key_shape(run_image) == (
        ("image_id", "job_id"),
        ("image_assets.image_id", "image_assets.job_id"),
        "CASCADE",
    )
    assert _foreign_key_shape(run_parent) == (
        ("parent_run_id", "job_id"),
        ("segmentation_runs.run_id", "segmentation_runs.job_id"),
        None,
    )
    assert _foreign_key_shape(query_image) == (
        ("image_id", "job_id"),
        ("image_assets.image_id", "image_assets.job_id"),
        None,
    )


def test_composite_image_relationship_keeps_orm_mappers_unambiguous() -> None:
    configure_mappers()

    assert {column.key for column in SegmentationRun.image.property.local_columns} == {
        "image_id",
    }


def _unique_constraint(
    model: type[ImageAsset] | type[SegmentationRun],
    name: str,
) -> UniqueConstraint:
    constraint = next(
        constraint
        for constraint in cast(Table, model.__table__).constraints
        if isinstance(constraint, UniqueConstraint) and constraint.name == name
    )
    return constraint


def _foreign_key(
    model: type[SegmentationRun] | type[QueryLog],
    name: str,
) -> ForeignKeyConstraint:
    constraint = next(
        constraint
        for constraint in cast(Table, model.__table__).constraints
        if isinstance(constraint, ForeignKeyConstraint) and constraint.name == name
    )
    return constraint


def _foreign_key_shape(
    constraint: ForeignKeyConstraint,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    return (
        tuple(constraint.columns.keys()),
        tuple(element.target_fullname for element in constraint.elements),
        constraint.ondelete,
    )
