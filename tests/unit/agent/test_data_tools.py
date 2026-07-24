from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import update

from app.agent.data_tools import SqlAlchemyDataToolService
from app.agent.unified_query import DataQuery
from app.contracts.enums import (
    JobStatus,
    ModelFamily,
    ModelStatus,
    ModelVariant,
    QualityStatus,
    QualityTier,
    RoiMode,
)
from app.contracts.identity import LEGACY_PRINCIPAL_ID, LEGACY_TENANT_ID
from app.core.config import Settings
from app.db.base import Base
from app.db.models import (
    AnalysisJob,
    ImageAsset,
    ImageSummary,
    ModelRegistryRecord,
    ParticleRecord,
    Principal,
    SegmentationRun,
    Tenant,
)
from app.db.session import Database

_FOREIGN_TENANT_ID = f"tnt_{'a' * 32}"
_FOREIGN_PRINCIPAL_ID = f"prn_{'b' * 32}"


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    instance = Database(Settings(database_url=f"sqlite:///{tmp_path / 'data-tools.db'}"))
    Base.metadata.create_all(instance.engine)
    _seed_experiment(instance)
    try:
        yield instance
    finally:
        instance.dispose()


@pytest.fixture
def service(database: Database) -> SqlAlchemyDataToolService:
    return SqlAlchemyDataToolService(database.session_factory)


def test_job_overview_is_persisted_and_auditable(service: SqlAlchemyDataToolService) -> None:
    result = service.answer(_query("任务概览"))

    assert result.outcome_code == "OK"
    assert result.confidence == "medium"
    evidence = result.evidence[0]
    assert evidence.tool_name == "get_job_overview"
    assert evidence.validated_arguments == {
        "job_id": "job_1",
        "image_id": None,
        "run_ids": [],
        "intent": "overview",
    }
    assert evidence.aggregates["selected_run_count"] == 3
    assert evidence.aggregates["completed_run_count"] == 2
    assert evidence.aggregates["particle_count_total"] == 6
    assert evidence.aggregates["review_required_count"] == 1
    assert evidence.source_run_ids == ["run_a", "run_b"]
    assert evidence.units["coverage_ratio"] == "ratio"
    assert any("run_pending" in warning for warning in evidence.quality_warnings)
    assert any("low_confidence" in warning for warning in evidence.quality_warnings)
    assert result.tool_calls[0].outcome == "success"
    assert result.tool_calls[0].source_run_ids == ["run_a", "run_b"]


def test_particle_count_respects_image_and_run_filters(
    service: SqlAlchemyDataToolService,
) -> None:
    image_result = service.answer(_query("颗粒数是多少？", image_id="img_a"))
    image_evidence = image_result.evidence[0]
    assert image_evidence.aggregates["particle_count"] == 2
    assert image_evidence.source_run_ids == ["run_a"]
    assert image_evidence.rows[0]["stored_particle_rows"] == 2

    run_result = service.answer(_query("颗粒数是多少？", run_ids=("run_b", "run_b")))
    run_evidence = run_result.evidence[0]
    assert run_evidence.aggregates["particle_count"] == 4
    assert run_evidence.validated_arguments["run_ids"] == ["run_b"]
    assert run_evidence.source_run_ids == ["run_b"]


def test_particle_number_density_uses_area_metric_not_particle_count(
    service: SqlAlchemyDataToolService,
) -> None:
    direct = service.answer(
        _query(
            "我们这张图的颗粒密度较高，文献上有哪些可能原因？",
            image_id="img_a",
        )
    )
    evidence = direct.evidence[0]

    assert direct.outcome_code == "OK"
    assert evidence.validated_arguments["intent"] == "number_density"
    assert evidence.validated_arguments["metric"] == "number_density_um2"
    assert evidence.aggregates["number_density_um2"] == pytest.approx(80_000)
    assert evidence.aggregates["particle_count"] == 2
    assert evidence.aggregates["roi_area_um2"] == pytest.approx(0.000025)
    assert evidence.rows[0]["number_density_px2"] == pytest.approx(0.02)
    assert evidence.rows[0]["number_density_um2"] == pytest.approx(80_000)
    assert evidence.units["number_density_um2"] == "um^-2"
    assert "相同物理尺度" in direct.answer


def test_particle_number_density_ranking_does_not_rank_raw_counts(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("哪组颗粒数密度最高，这种差异可能和什么材料因素有关？"))
    evidence = result.evidence[0]

    assert result.outcome_code == "OK"
    assert evidence.validated_arguments["metric"] == "number_density_um2"
    assert [row["group"] for row in evidence.rows] == ["sample_A", "sample_B"]
    assert [row["value"] for row in evidence.rows] == pytest.approx([80_000, 160_000 / 3])
    assert all(
        row["weighting"] == "total_particle_count_over_total_roi_area" for row in evidence.rows
    )
    assert evidence.units["value"] == "um^-2"
    assert "number_density_um2" in result.answer
    assert "number_density_px2" not in result.answer


def test_particle_number_density_ranking_uses_total_physical_roi_area(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("哪组颗粒数密度最高？"))
    evidence = result.evidence[0]

    assert result.outcome_code == "OK"
    assert evidence.tool_name == "rank_samples"
    assert [row["group"] for row in evidence.rows] == ["sample_A", "sample_B"]
    assert all(
        row["aggregation"] == "total_particle_count_over_total_roi_area" for row in evidence.rows
    )
    assert all(row["weight_unit"] == "um^2" for row in evidence.rows)
    assert evidence.units["value"] == "um^-2"
    assert "number_density_um2" in result.answer
    assert "number_density_px2" not in result.answer


def test_particle_number_density_comparison_requires_compatible_area_scale(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            update(SegmentationRun)
            .where(SegmentationRun.run_id.in_(("run_a", "run_b")))
            .values(
                run_config_json={
                    "schema_version": 2,
                    "provenance_status": "complete",
                    "scale_nm_per_pixel": None,
                }
            )
        )
        connection.execute(
            update(ImageSummary)
            .where(ImageSummary.run_id.in_(("run_a", "run_b")))
            .values(number_density_um2=None)
        )

    result = service.answer(_query("哪组颗粒数密度最高？"))

    assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert result.needs_clarification is True
    assert "物理面积尺度" in result.answer


def test_perimeter_density_uses_particle_perimeters_and_physical_scale(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("周长密度是多少？"))

    assert result.outcome_code == "OK"
    evidence = result.evidence[0]
    assert evidence.validated_arguments["intent"] == "perimeter_density"
    assert evidence.validated_arguments["metric"] == "perimeter_density_um"
    assert evidence.aggregates["perimeter_density_um"] == pytest.approx(30)
    assert evidence.aggregates["roi_area_um2"] == pytest.approx(0.0001)
    by_run = {row["run_id"]: row for row in evidence.rows}
    assert by_run["run_a"]["perimeter_density_px"] == pytest.approx(0.02)
    assert by_run["run_a"]["perimeter_density_um"] == pytest.approx(40)
    assert by_run["run_b"]["perimeter_density_um"] == pytest.approx(80 / 3)
    assert evidence.units["perimeter_density_um"] == "um^-1"
    assert "总颗粒周长/总 ROI 面积" in result.answer


def test_perimeter_density_ranking_uses_total_perimeter_over_area(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("哪组周长密度最高？"))

    assert result.outcome_code == "OK"
    evidence = result.evidence[0]
    assert [row["group"] for row in evidence.rows] == ["sample_A", "sample_B"]
    assert [row["value"] for row in evidence.rows] == pytest.approx([40, 80 / 3])
    assert all(
        row["aggregation"] == "total_perimeter_over_total_roi_area"
        for row in evidence.rows
    )
    assert all(row["weight_unit"] == "um^2" for row in evidence.rows)
    assert evidence.units["value"] == "um^-1"


def test_perimeter_density_comparison_requires_compatible_length_scale(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            update(SegmentationRun)
            .where(SegmentationRun.run_id.in_(("run_a", "run_b")))
            .values(
                run_config_json={
                    "schema_version": 2,
                    "provenance_status": "complete",
                    "scale_nm_per_pixel": None,
                }
            )
        )
        connection.execute(
            update(ImageSummary)
            .where(ImageSummary.run_id.in_(("run_a", "run_b")))
            .values(perimeter_density_um=None)
        )

    result = service.answer(_query("哪组周长密度最高？"))

    assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert result.needs_clarification is True
    assert "物理长度尺度" in result.answer


def test_mean_diameter_uses_particle_rows_not_summary_placeholders(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("平均粒径是多少？"))

    evidence = result.evidence[0]
    assert evidence.aggregates["mean_equivalent_diameter_px"] == pytest.approx(170 / 6)
    assert evidence.aggregates["mean_equivalent_diameter_nm"] == pytest.approx(85 / 6)
    by_run = {row["run_id"]: row for row in evidence.rows}
    assert by_run["run_a"]["mean_equivalent_diameter_px"] == 15
    assert by_run["run_b"]["mean_equivalent_diameter_px"] == 35
    assert evidence.units["mean_equivalent_diameter_nm"] == "nm"
    assert evidence.source_run_ids == ["run_a", "run_b"]
    assert any(
        "physical diameter exists for 3/4" in warning for warning in evidence.quality_warnings
    )
    assert any(
        "recomputed from pixel diameter and scale=0.5" in warning
        for warning in evidence.quality_warnings
    )


def test_physical_diameter_does_not_drift_when_live_image_scale_changes(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    with database.session() as session:
        image = session.get(ImageAsset, "img_b")
        assert image is not None
        image.scale_nm_per_pixel = None

    aggregate = service.answer(_query("平均粒径是多少？"))
    distribution = service.answer(_query("平均粒径分布 4 bins"))
    one_image = service.answer(_query("平均粒径是多少？", image_id="img_b"))

    assert aggregate.outcome_code == "OK"
    assert aggregate.evidence[0].aggregates["mean_equivalent_diameter_nm"] == pytest.approx(85 / 6)
    assert distribution.outcome_code == "OK"
    assert one_image.outcome_code == "OK"
    assert one_image.evidence[0].aggregates["mean_equivalent_diameter_nm"] == pytest.approx(17.5)
    assert "17.5 nm" in one_image.answer


def test_legacy_run_scale_uses_live_metadata_with_explicit_warning(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            update(SegmentationRun)
            .where(SegmentationRun.run_id == "run_a")
            .values(run_config_json={})
        )
        connection.execute(
            update(ImageAsset).where(ImageAsset.image_id == "img_a").values(scale_nm_per_pixel=0.25)
        )

    result = service.answer(_query("平均粒径是多少？", image_id="img_a"))

    evidence = result.evidence[0]
    assert evidence.aggregates["mean_equivalent_diameter_nm"] == pytest.approx(3.75)
    assert any(
        "legacy run configuration has no complete frozen scale" in warning
        and "fallback=0.25 nm/px" in warning
        for warning in evidence.quality_warnings
    )


def test_legacy_density_rows_and_aggregate_use_the_same_effective_scale(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            update(SegmentationRun)
            .where(SegmentationRun.run_id == "run_a")
            .values(run_config_json={})
        )
        connection.execute(
            update(ImageAsset)
            .where(ImageAsset.image_id == "img_a")
            .values(scale_nm_per_pixel=0.25)
        )

    result = service.answer(_query("颗粒数密度是多少？", image_id="img_a"))

    evidence = result.evidence[0]
    assert result.outcome_code == "OK"
    assert evidence.aggregates["number_density_um2"] == pytest.approx(320_000)
    assert evidence.rows[0]["number_density_um2"] == pytest.approx(320_000)
    assert any(
        "persisted number_density_um2=80000 differs" in warning
        and "effective scale-derived value=320000" in warning
        for warning in evidence.quality_warnings
    )


def test_particle_level_statistics_fail_closed_on_missing_detail_row(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    with database.session() as session:
        particle = session.get(ParticleRecord, "p_a2")
        assert particle is not None
        session.delete(particle)

    mean = service.answer(_query("平均粒径是多少？", image_id="img_a"))
    distribution = service.answer(_query("平均粒径分布 4 bins", image_id="img_a"))
    count = service.answer(_query("颗粒数是多少？", image_id="img_a"))
    density = service.answer(_query("颗粒数密度是多少？", image_id="img_a"))
    overview = service.answer(_query("任务概览", image_id="img_a"))

    for result in (mean, distribution, count, density, overview):
        assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
        assert result.tool_calls[0].outcome == "insufficient_data"
        assert "run_a=1/2" in result.answer
        assert any(
            "summary particle_count=2 but stored particle rows=1" in warning
            for warning in result.evidence[0].quality_warnings
        )


def test_coverage_is_roi_area_weighted(service: SqlAlchemyDataToolService) -> None:
    result = service.answer(_query("覆盖率是多少？"))

    evidence = result.evidence[0]
    assert evidence.aggregates["coverage_ratio"] == pytest.approx(0.425)
    assert evidence.aggregates["coverage_percent"] == pytest.approx(42.5)
    assert evidence.aggregates["roi_area_px"] == 400
    assert evidence.units == {
        "coverage_ratio": "ratio",
        "coverage_percent": "%",
        "roi_area_px": "px^2",
    }


def test_review_results_include_reasons_and_support_a_negative_answer(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("有哪些需复核结果？"))

    evidence = result.evidence[0]
    assert evidence.aggregates["review_required_count"] == 1
    assert evidence.rows[0]["run_id"] == "run_b"
    assert evidence.rows[0]["reasons"] == ["low_confidence"]
    assert evidence.rows[0]["recommendations"] == ["manual_review"]
    assert evidence.source_run_ids == ["run_a", "run_b"]

    negative = service.answer(_query("有哪些需复核结果？", image_id="img_a"))
    assert negative.outcome_code == "OK"
    assert negative.evidence[0].aggregates["review_required_count"] == 0
    assert negative.evidence[0].source_run_ids == ["run_a"]
    assert negative.tool_calls[0].outcome == "success"


@pytest.mark.parametrize(
    ("job_id", "question", "image_id", "run_ids"),
    [
        ("job_1", "颗粒数是多少？", None, ("run_other",)),
        ("job_1", "覆盖率是多少？", "img_other", ()),
        ("job_2", "任务概览", None, ()),
        ("missing", "任务概览", None, ()),
    ],
)
def test_foreign_or_missing_scope_is_an_error_without_leaking_data(
    service: SqlAlchemyDataToolService,
    job_id: str,
    question: str,
    image_id: str | None,
    run_ids: tuple[str, ...],
) -> None:
    result = service.answer(
        DataQuery(
            job_id=job_id,
            tenant_id=LEGACY_TENANT_ID,
            question=question,
            image_id=image_id,
            run_ids=run_ids,
        )
    )

    assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert result.evidence[0].rows == []
    assert result.evidence[0].source_run_ids == []
    assert result.tool_calls[0].outcome == "error"


def test_non_completed_selection_returns_insufficient_not_fabricated_zero(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("颗粒数是多少？", run_ids=("run_pending",)))

    assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert result.evidence[0].aggregates == {}
    assert result.evidence[0].source_run_ids == []
    assert result.tool_calls[0].outcome == "insufficient_data"
    assert any("run_status=QUEUED" in warning for warning in result.evidence[0].quality_warnings)


def test_explicit_sample_ranking_is_deterministic(service: SqlAlchemyDataToolService) -> None:
    result = service.answer(_query("哪组颗粒数最高？"))

    evidence = result.evidence[0]
    assert evidence.tool_name == "rank_samples"
    assert evidence.validated_arguments["group_by"] == "sample"
    assert evidence.validated_arguments["metric"] == "particle_count"
    assert [row["group"] for row in evidence.rows] == ["sample_B", "sample_A"]
    assert [row["value"] for row in evidence.rows] == [4.0, 2.0]
    assert all(row["aggregation"] == "sum_across_runs" for row in evidence.rows)
    assert all(row["observation_unit"] == "run" for row in evidence.rows)
    assert all(row["weight_total"] == 1 for row in evidence.rows)
    assert evidence.source_run_ids == ["run_a", "run_b"]


def test_compare_groups_uses_explicit_metric_and_single_model(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("比较不同样品的覆盖率"))

    evidence = result.evidence[0]
    assert result.outcome_code == "OK"
    assert evidence.tool_name == "compare_groups"
    assert evidence.validated_arguments["group_by"] == "sample"
    assert evidence.validated_arguments["statistic"] == "mean"
    assert [row["group"] for row in evidence.rows] == ["sample_B", "sample_A"]
    assert [row["value"] for row in evidence.rows] == [0.5, 0.2]
    assert all(row["observation_unit"] == "run" for row in evidence.rows)
    assert all(row["weighting"] == "unweighted_run_values" for row in evidence.rows)


def test_ranking_rejects_mixed_model_results(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    _add_second_model_run(database)

    result = service.answer(_query("哪组颗粒数最高？", run_ids=("run_a", "run_a2")))

    assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert result.needs_clarification is True
    assert "多个模型" in result.answer
    assert result.evidence[0].source_run_ids == ["run_a", "run_a2"]


def test_describe_particle_distribution_uses_particle_rows(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("平均粒径分布 4 bins"))

    evidence = result.evidence[0]
    assert result.outcome_code == "OK"
    assert evidence.tool_name == "describe_distribution"
    assert evidence.validated_arguments["metric"] == "mean_equivalent_diameter_nm"
    assert len(evidence.rows) == 6
    assert evidence.aggregates["count"] == 6
    assert evidence.aggregates["median"] == pytest.approx(12.5)
    assert evidence.aggregates["q1"] == pytest.approx(10)
    assert evidence.aggregates["q3"] == pytest.approx(18.75)
    assert len(evidence.aggregates["histogram"]) == 4


def test_particle_distribution_keeps_exact_aggregates_but_bounds_audit_rows(
    database: Database,
) -> None:
    with database.session() as session:
        summary = session.get(ImageSummary, "run_a")
        assert summary is not None
        summary.particle_count = 302
        summary.number_density_px2 = 3.02
        session.add_all(
            _particle(
                f"p_bulk_{index:04d}",
                "run_a",
                index,
                float(index),
                None,
            )
            for index in range(3, 303)
        )

    bounded = SqlAlchemyDataToolService(
        database.session_factory,
        distribution_evidence_limit=25,
    )
    first = bounded.answer(_query("平均粒径分布 8 bins", image_id="img_a"))
    second = bounded.answer(_query("平均粒径分布 8 bins", image_id="img_a"))

    evidence = first.evidence[0]
    assert first.outcome_code == "OK"
    assert evidence.aggregates["count"] == 302
    assert evidence.aggregates["evidence_rows_total"] == 302
    assert evidence.aggregates["evidence_rows_returned"] <= 25
    assert len(evidence.rows) <= 25
    assert sum(item["count"] for item in evidence.aggregates["histogram"]) == 302
    assert evidence.rows == second.evidence[0].rows
    assert any("returned" in warning for warning in evidence.quality_warnings)


def test_anomaly_query_lists_quality_and_consistency_findings(
    service: SqlAlchemyDataToolService,
) -> None:
    result = service.answer(_query("有哪些异常结果？"))

    evidence = result.evidence[0]
    assert result.outcome_code == "OK"
    assert evidence.tool_name == "find_anomalies"
    assert evidence.aggregates["anomaly_count"] == 1
    assert evidence.rows[0]["run_id"] == "run_b"
    assert evidence.rows[0]["reasons"] == ["low_confidence"]


def test_compare_models_requires_two_selected_runs_on_one_image(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    _add_second_model_run(database)

    result = service.answer(
        _query(
            "不同模型的覆盖率对比",
            image_id="img_a",
            run_ids=("run_a", "run_a2"),
        )
    )

    evidence = result.evidence[0]
    assert result.outcome_code == "OK"
    assert evidence.tool_name == "compare_models"
    assert evidence.validated_arguments["image_id"] == "img_a"
    assert [row["model_id"] for row in evidence.rows] == ["model_2", "model_1"]
    assert [row["value"] for row in evidence.rows] == [0.3, 0.2]


def test_default_numeric_query_refuses_to_double_count_alternative_runs(
    service: SqlAlchemyDataToolService,
    database: Database,
) -> None:
    _add_second_model_run(database)

    ambiguous = service.answer(_query("颗粒数是多少？"))
    explicit = service.answer(_query("颗粒数是多少？", run_ids=("run_a", "run_b")))

    assert ambiguous.outcome_code == "INSUFFICIENT_EVIDENCE"
    assert ambiguous.needs_clarification is True
    assert "img_a=[run_a, run_a2]" in ambiguous.answer
    assert explicit.outcome_code == "OK"
    assert explicit.evidence[0].aggregates["particle_count"] == 6


def test_unknown_or_ambiguous_intent_asks_for_clarification_without_querying_sql(
    service: SqlAlchemyDataToolService,
) -> None:
    unknown = service.answer(_query("帮我看看"))
    ambiguous_rank = service.answer(_query("哪组最高？"))

    for result in (unknown, ambiguous_rank):
        assert result.outcome_code == "INSUFFICIENT_EVIDENCE"
        assert result.needs_clarification
        assert result.evidence[0].tool_name == "resolve_data_intent"
        assert result.tool_calls[0].outcome == "insufficient_data"


def _query(
    question: str,
    *,
    image_id: str | None = None,
    run_ids: tuple[str, ...] = (),
) -> DataQuery:
    return DataQuery(
        job_id="job_1",
        tenant_id=LEGACY_TENANT_ID,
        question=question,
        image_id=image_id,
        run_ids=run_ids,
    )


def _seed_experiment(database: Database) -> None:
    with database.session() as session:
        session.add(
            Tenant(
                tenant_id=_FOREIGN_TENANT_ID,
                slug="data-tools-foreign",
                display_name="Data tools foreign",
            )
        )
        session.flush()
        session.add(
            Principal(
                principal_id=_FOREIGN_PRINCIPAL_ID,
                tenant_id=_FOREIGN_TENANT_ID,
                handle="data-tools-foreign",
                display_name="Data tools foreign",
                kind="user",
                role="analyst",
            )
        )
        session.flush()
        session.add_all(
            [
                AnalysisJob(
                    job_id="job_1",
                    tenant_id=LEGACY_TENANT_ID,
                    owner_principal_id=LEGACY_PRINCIPAL_ID,
                    name="data tools fixture",
                    status=JobStatus.COMPLETED_WITH_WARNINGS.value,
                    config_json={},
                ),
                AnalysisJob(
                    job_id="job_2",
                    tenant_id=_FOREIGN_TENANT_ID,
                    owner_principal_id=_FOREIGN_PRINCIPAL_ID,
                    name="foreign fixture",
                    status=JobStatus.COMPLETED.value,
                    config_json={},
                ),
            ]
        )
        session.add_all(
            [
                _image("img_a", "job_1", "a.tif", "sample_A", "TiO2"),
                _image("img_b", "job_1", "b.tif", "sample_B", "Fe2O3"),
                _image("img_other", "job_2", "other.tif", "other", "ZnO"),
            ]
        )
        session.add(
            ModelRegistryRecord(
                model_id="model_1",
                family=ModelFamily.UNET.value,
                variant=ModelVariant.GENERAL.value,
                quality_tier=QualityTier.BALANCED.value,
                version="1",
                adapter="tests.fake:Adapter",
                status=ModelStatus.READY.value,
                metadata_json={},
            )
        )
        session.flush()
        session.add_all(
            [
                _run("run_a", "job_1", "img_a", JobStatus.COMPLETED),
                _run(
                    "run_b",
                    "job_1",
                    "img_b",
                    JobStatus.COMPLETED_WITH_WARNINGS,
                ),
                _run("run_pending", "job_1", "img_a", JobStatus.QUEUED),
                _run("run_other", "job_2", "img_other", JobStatus.COMPLETED),
            ]
        )
        session.flush()
        session.add_all(
            [
                _summary(
                    "run_a",
                    particle_count=2,
                    roi_area_px=100,
                    coverage_ratio=0.2,
                    quality_status=QualityStatus.PASS,
                ),
                _summary(
                    "run_b",
                    particle_count=4,
                    roi_area_px=300,
                    coverage_ratio=0.5,
                    quality_status=QualityStatus.REVIEW_REQUIRED,
                    reasons=["low_confidence"],
                    recommendations=["manual_review"],
                ),
                _summary(
                    "run_other",
                    particle_count=1,
                    roi_area_px=50,
                    coverage_ratio=0.9,
                    quality_status=QualityStatus.PASS,
                ),
            ]
        )
        session.add_all(
            [
                _particle("p_a1", "run_a", 1, 10, 5),
                _particle("p_a2", "run_a", 2, 20, 10),
                _particle("p_b1", "run_b", 1, 20, 10),
                _particle("p_b2", "run_b", 2, 30, 15),
                _particle("p_b3", "run_b", 3, 40, 20),
                _particle("p_b4", "run_b", 4, 50, None),
                _particle("p_o1", "run_other", 1, 999, 999),
            ]
        )


def _add_second_model_run(database: Database) -> None:
    with database.session() as session:
        session.add(
            ModelRegistryRecord(
                model_id="model_2",
                family=ModelFamily.YOLO_SEG.value,
                variant=ModelVariant.GENERAL.value,
                quality_tier=QualityTier.BALANCED.value,
                version="1",
                adapter="tests.fake:Adapter",
                status=ModelStatus.READY.value,
                metadata_json={},
            )
        )
        session.add(
            _run(
                "run_a2",
                "job_1",
                "img_a",
                JobStatus.COMPLETED,
                model_id="model_2",
            )
        )
        session.flush()
        session.add(
            _summary(
                "run_a2",
                particle_count=3,
                roi_area_px=100,
                coverage_ratio=0.3,
                quality_status=QualityStatus.PASS,
            )
        )
        session.add_all(
            [
                _particle("p_a2_1", "run_a2", 1, 11, 5.5),
                _particle("p_a2_2", "run_a2", 2, 21, 10.5),
                _particle("p_a2_3", "run_a2", 3, 31, 15.5),
            ]
        )


def _image(
    image_id: str,
    job_id: str,
    filename: str,
    sample_id: str,
    material_formula: str,
) -> ImageAsset:
    return ImageAsset(
        image_id=image_id,
        job_id=job_id,
        filename=filename,
        storage_path=f"{job_id}/input/{image_id}/original.tif",
        sha256=(image_id[-1] if image_id[-1].isalnum() else "a") * 64,
        width=100,
        height=100,
        bit_depth=8,
        sample_id=sample_id,
        material_name=material_formula,
        material_formula=material_formula,
        experiment_conditions_json={},
        analysis_roi_json={
            "schema_version": 1,
            "coordinate_space": "original_px",
            "valid_rect": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "invalid_rects": [],
            "source": "none",
            "revision": 1,
        },
        scale_nm_per_pixel=0.5,
    )


def _run(
    run_id: str,
    job_id: str,
    image_id: str,
    status: JobStatus,
    *,
    model_id: str = "model_1",
) -> SegmentationRun:
    return SegmentationRun(
        run_id=run_id,
        job_id=job_id,
        image_id=image_id,
        model_id=model_id,
        roi_mode=RoiMode.FULL_IMAGE.value,
        status=status.value,
        inference_json={},
        run_config_json={
            "schema_version": 2,
            "provenance_status": "complete",
            "scale_nm_per_pixel": 0.5,
        },
        paths_json={},
    )


def _summary(
    run_id: str,
    *,
    particle_count: int,
    roi_area_px: int,
    coverage_ratio: float,
    quality_status: QualityStatus,
    reasons: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> ImageSummary:
    return ImageSummary(
        run_id=run_id,
        particle_count=particle_count,
        roi_area_px=roi_area_px,
        number_density_px2=particle_count / roi_area_px,
        number_density_um2=(particle_count / roi_area_px) * (1000 / 0.5) ** 2,
        mean_equivalent_diameter_px=999,
        mean_equivalent_diameter_nm=999,
        coverage_ratio=coverage_ratio,
        perimeter_density_px=0,
        perimeter_density_um=None,
        quality_status=quality_status.value,
        quality_json={
            "status": quality_status.value,
            "reasons": reasons or [],
            "recommendations": recommendations or [],
        },
    )


def _particle(
    particle_id: str,
    run_id: str,
    instance_index: int,
    diameter_px: float,
    diameter_nm: float | None,
) -> ParticleRecord:
    return ParticleRecord(
        particle_id=particle_id,
        run_id=run_id,
        instance_index=instance_index,
        area_px=1,
        perimeter_px=1,
        equivalent_diameter_px=diameter_px,
        equivalent_diameter_nm=diameter_nm,
        circularity=1,
        bbox_json=[0, 0, 1, 1],
        confidence=1,
    )
