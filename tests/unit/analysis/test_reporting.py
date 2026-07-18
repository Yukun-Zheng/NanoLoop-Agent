import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.analysis.config import (
    MorphometryConfig,
    PostprocessProfile,
    QualityGateConfig,
    capture_execution_build_provenance,
)
from app.analysis.morphometry import measure
from app.analysis.postprocessing import NormalizedInstance
from app.analysis.reporting import JobExportSnapshot, ReportWriter
from app.analysis.transforms import TransformRecord
from app.contracts.analyses import (
    AnalysisJobDTO,
    AnalysisROI,
    BoxSetDTO,
    ImageAssetDTO,
    InferenceOptions,
    PixelRect,
    QualityReportDTO,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.enums import JobStatus, QualityStatus, QueryType, RoiMode
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.identity import AuthMode
from app.contracts.queries import (
    QueryActorDTO,
    QueryAuditRecordDTO,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from app.core.identity import legacy_principal_context
from app.storage.file_store import LocalFileStore
from app.storage.paths import StoragePaths


def test_write_reports_and_export_have_versioned_auditable_files(tmp_path: Path) -> None:
    file_store = LocalFileStore(
        StoragePaths(tmp_path / "outputs"),
        max_upload_bytes=1_000_000,
        token_secret=b"x" * 32,
    )
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True
    instance = NormalizedInstance(
        instance_index=1,
        mask=mask,
        bbox=(10, 10, 20, 20),
        area_px=100,
        confidence=0.9,
        touches_roi_boundary=False,
    )
    result = measure(
        run_id="run_1",
        instances=[instance],
        roi_mask=np.ones((32, 32), dtype=bool),
        scale_nm_per_pixel=1.0,
        config=MorphometryConfig(),
    )
    analysis_roi = AnalysisROI(valid_rect=PixelRect(x1=0, y1=0, x2=32, y2=32))
    configuration = RunConfiguration(
        schema_version=2,
        provenance_status="complete",
        provenance_warnings=[],
        model_id="unet-small-balanced-v1",
        model_version="1.0.0",
        adapter_path="tests.fake:FakeAdapter",
        weight_sha256="1" * 64,
        config_sha256="2" * 64,
        model_card_sha256="3" * 64,
        roi_mode=RoiMode.FULL_IMAGE,
        analysis_roi=analysis_roi,
        inference=InferenceOptions(),
        preprocess_profile="sem_gray_v1",
        postprocess_profile="small_particle_v1",
        image_sha256="a" * 64,
        scale_nm_per_pixel=1.0,
        resolved_postprocess=PostprocessProfile(profile_id="small_particle_v1"),
        resolved_morphometry=MorphometryConfig(perimeter_neighborhood=8),
        resolved_quality_gate=QualityGateConfig(),
        execution_build=capture_execution_build_provenance(),
        created_at=datetime.now(UTC),
    )
    quality = QualityReportDTO(status=QualityStatus.PASS)
    assert configuration.execution_build is not None
    execution = ExecutionRuntimeProvenance(
        executor_build=configuration.execution_build,
        build_identity_matches_contract=True,
        requested_device=configuration.inference.device,
        actual_device="cpu",
        seed=configuration.inference.seed,
        python_random_seeded=True,
        numpy_random_seeded=True,
        torch_deterministic_algorithms=False,
        global_inference_serialized=True,
        backend="tests.fake.FakeAdapter",
        executed_at=datetime.now(UTC),
    )
    writer = ReportWriter(file_store)
    files = writer.write_run_reports(
        job_id="job_1",
        image_id="img_1",
        run_id="run_1",
        configuration=configuration,
        execution=execution,
        transform=TransformRecord(
            original_width=32,
            original_height=32,
            crop=analysis_roi.valid_rect,
        ),
        morphometry=result,
        quality=quality,
    )

    particles_path = file_store.paths.root / files.particles_csv_path
    assert particles_path.read_bytes().startswith(b"\xef\xbb\xbf")
    summary_path = file_store.paths.root / files.image_summary_path
    assert json.loads(summary_path.read_text())["schema_version"] == "1.0"

    exported = writer.build_job_export("job_1")
    with zipfile.ZipFile(exported.path) as archive:
        names = set(archive.namelist())
        assert "export_manifest.json" in names
        assert "images/img_1/runs/run_1/particles.csv" in names
        manifest = json.loads(archive.read("export_manifest.json"))
    assert manifest["schema_version"] == "1.0"
    assert "images/img_1/runs/run_1/execution_provenance.json" in names
    assert len(manifest["files"]) == 6

    now = datetime.now(UTC)
    snapshot = JobExportSnapshot(
        job=AnalysisJobDTO(
            job_id="job_1",
            name="Export test",
            status=JobStatus.COMPLETED,
            created_at=now,
            updated_at=now,
        ),
        images=(
            ImageAssetDTO(
                image_id="img_1",
                job_id="job_1",
                filename='=HYPERLINK("https://attacker.invalid")',
                sha256="a" * 64,
                width=32,
                height=32,
                bit_depth=8,
                sample_id="+sample-a",
                material_name="@Test material",
                scale_nm_per_pixel=1.0,
                analysis_roi=analysis_roi,
            ),
        ),
        runs=(
            SegmentationRunDTO(
                run_id="run_1",
                job_id="job_1",
                image_id="img_1",
                model_id=configuration.model_id,
                status=JobStatus.COMPLETED,
                roi_mode=RoiMode.FULL_IMAGE,
                threshold=0.5,
                inference=configuration.inference,
                configuration=configuration,
                execution=execution,
                summary=result.image_summary,
                quality=quality,
                runtime_ms=12,
                created_at=now,
                updated_at=now,
            ),
        ),
        queries=(
            QueryAuditRecordDTO(
                query_id="query_1",
                job_id="job_1",
                image_id="img_1",
                actor=QueryActorDTO.from_principal(
                    legacy_principal_context(AuthMode.DISABLED)
                ),
                request=UnifiedQueryRequest(
                    question="有多少颗粒？",
                    query_type=QueryType.ANALYSIS_DATA,
                    image_id="img_1",
                    run_ids=["run_1"],
                ),
                response=UnifiedQueryResponse(
                    query_type=QueryType.ANALYSIS_DATA,
                    answer="共有 1 个颗粒。",
                    confidence="high",
                ),
                created_at=now,
            ),
        ),
        box_revisions=(BoxSetDTO(image_id="img_1", revision=0, boxes=[]),),
    )
    exported = writer.build_job_export("job_1", snapshot=snapshot)
    issued_path = exported.path
    issued_bytes = issued_path.read_bytes()
    with zipfile.ZipFile(exported.path) as archive:
        names = set(archive.namelist())
        assert {
            "job_summary.json",
            "run_summary.csv",
            "sample_summary.csv",
            "audit_summary.json",
            "software_manifest.json",
            "charts/particle_count_by_run.svg",
            "charts/coverage_ratio_by_run.svg",
            "query_history.jsonl",
            "rag_citations.json",
            "images/img_1/boxes_revision_000.json",
        } <= names
        run_summary = archive.read("run_summary.csv").decode("utf-8-sig")
        sample_summary = archive.read("sample_summary.csv").decode("utf-8-sig")
        audit = json.loads(archive.read("audit_summary.json"))
        software = json.loads(archive.read("software_manifest.json"))
        history = [
            json.loads(line)
            for line in archive.read("query_history.jsonl").decode("utf-8").splitlines()
        ]

    assert "'=HYPERLINK" in run_summary
    assert "'+sample-a" in run_summary
    assert "'@Test material" in run_summary
    assert "'+sample-a" in sample_summary
    assert "unet-small-balanced-v1" in sample_summary
    assert audit["run_lineage"][0]["run_id"] == "run_1"
    assert audit["run_lineage"][0]["status_history"] == []
    assert audit["run_lineage"][0]["configuration_provenance_status"] == "complete"
    assert audit["run_lineage"][0]["image_sha256"] == "a" * 64
    assert audit["run_lineage"][0]["scale_nm_per_pixel"] == 1.0
    assert audit["run_lineage"][0]["resolved_postprocess"]["profile_id"] == ("small_particle_v1")
    assert audit["run_lineage"][0]["resolved_morphometry"]["perimeter_neighborhood"] == 8
    assert audit["run_lineage"][0]["resolved_quality_gate"]["foreground_ratio_review_low"] == 0.0001
    assert audit["run_lineage"][0]["execution_build"]["application_version"]
    assert audit["run_lineage"][0]["execution"]["actual_device"] == "cpu"
    assert audit["query_ids"] == ["query_1"]
    assert history[0]["request"]["run_ids"] == ["run_1"]
    assert len(software["dependency_contract_sha256"]) == 64
    assert len(software["installed_dependencies_sha256"]) == 64
    assert len(software["application_source_sha256"]) == 64
    assert (
        audit["run_lineage"][0]["execution_build"]["application_source_sha256"]
        == configuration.execution_build.application_source_sha256
    )

    repeated = writer.build_job_export("job_1", snapshot=snapshot)
    assert repeated.path == issued_path
    assert repeated.sha256 == exported.sha256
    assert issued_path.read_bytes() == issued_bytes
    assert file_store.resolve_file_token(exported.file_token) == issued_path

    changed_query = snapshot.queries[0].model_copy(
        update={
            "response": snapshot.queries[0].response.model_copy(
                update={"answer": "共有 1 个颗粒，已复核。"}
            )
        }
    )
    changed = writer.build_job_export(
        "job_1",
        snapshot=JobExportSnapshot(
            job=snapshot.job,
            images=snapshot.images,
            runs=snapshot.runs,
            queries=(changed_query,),
            box_revisions=snapshot.box_revisions,
        ),
    )
    assert changed.path != issued_path
    assert issued_path.read_bytes() == issued_bytes
    assert file_store.resolve_file_token(exported.file_token) == issued_path
