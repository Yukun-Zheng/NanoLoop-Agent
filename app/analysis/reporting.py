"""Versioned run reports and export assembly over the FileStore abstraction."""

import csv
import io
import json
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from xml.sax.saxutils import escape

from app.analysis.morphometry import MorphometryResult
from app.analysis.transforms import TransformRecord
from app.contracts.analyses import (
    AnalysisJobDTO,
    BoxSetDTO,
    ImageAssetDTO,
    QualityReportDTO,
    RunConfiguration,
    SegmentationRunDTO,
)
from app.contracts.analysis_config import capture_execution_build_provenance
from app.contracts.enums import QualityStatus
from app.contracts.execution import ExecutionRuntimeProvenance
from app.contracts.queries import QueryAuditRecordDTO
from app.storage.file_store import LocalFileStore, StoredFile

_EXPORT_LOCK = threading.Lock()
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _csv_safe_row(row: dict[str, object]) -> dict[str, object]:
    """Neutralize spreadsheet formulas without mutating authoritative metadata."""

    return {
        key: f"'{value}"
        if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES)
        else value
        for key, value in row.items()
    }


@dataclass(frozen=True, slots=True)
class RunReportFiles:
    run_config_path: str
    execution_provenance_path: str
    transform_path: str
    particles_csv_path: str
    image_summary_path: str
    quality_report_path: str

    def as_paths_json(self) -> dict[str, str]:
        return {
            "run_config_path": self.run_config_path,
            "execution_provenance_path": self.execution_provenance_path,
            "transform_path": self.transform_path,
            "particles_csv_path": self.particles_csv_path,
            "image_summary_path": self.image_summary_path,
            "quality_report_path": self.quality_report_path,
        }


@dataclass(frozen=True, slots=True)
class JobExportSnapshot:
    """Database-backed public state used to build job-level export reports."""

    job: AnalysisJobDTO
    images: tuple[ImageAssetDTO, ...]
    runs: tuple[SegmentationRunDTO, ...]
    queries: tuple[QueryAuditRecordDTO, ...] = ()
    box_revisions: tuple[BoxSetDTO, ...] = ()
    image_storage_paths: tuple[str, ...] = ()
    run_artifact_paths: tuple[tuple[str, str], ...] = ()


class ReportWriter:
    _CANONICAL_RUN_ARTIFACT_FILENAMES = frozenset(
        {
            "execution_provenance.json",
            "image_summary.json",
            "instances.json",
            "labeled_particles.png",
            "overlay.png",
            "particles.csv",
            "pred_mask.png",
            "probability.npy",
            "quality_report.json",
            "run_config.json",
            "transform.json",
        }
    )

    PARTICLE_COLUMNS = (
        "particle_id",
        "run_id",
        "instance_index",
        "area_px",
        "perimeter_px",
        "equivalent_diameter_px",
        "equivalent_diameter_nm",
        "circularity",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "confidence",
    )

    def __init__(self, file_store: LocalFileStore) -> None:
        self.file_store = file_store

    def write_run_reports(
        self,
        *,
        job_id: str,
        image_id: str,
        run_id: str,
        configuration: RunConfiguration,
        execution: ExecutionRuntimeProvenance,
        transform: TransformRecord,
        morphometry: MorphometryResult,
        quality: QualityReportDTO,
    ) -> RunReportFiles:
        paths = self.file_store.paths
        self.file_store.create_run_dir(job_id, image_id, run_id)
        run_config_path = paths.run_artifact(job_id, image_id, run_id, "run_config.json")
        execution_path = paths.run_artifact(job_id, image_id, run_id, "execution_provenance.json")
        transform_path = paths.run_artifact(job_id, image_id, run_id, "transform.json")
        particles_path = paths.run_artifact(job_id, image_id, run_id, "particles.csv")
        summary_path = paths.run_artifact(job_id, image_id, run_id, "image_summary.json")
        quality_path = paths.run_artifact(job_id, image_id, run_id, "quality_report.json")

        self.file_store.atomic_write_json(
            run_config_path,
            self._contract_payload(configuration.model_dump(mode="json")),
        )
        self.file_store.atomic_write_json(
            execution_path,
            self._contract_payload(execution.model_dump(mode="json")),
        )
        self.file_store.atomic_write_json(
            transform_path,
            self._contract_payload(transform.model_dump(mode="json")),
        )
        self.file_store.atomic_write_bytes(
            particles_path,
            self._particles_csv(morphometry).encode("utf-8-sig"),
        )
        self.file_store.atomic_write_json(
            summary_path,
            morphometry.image_summary.model_dump(mode="json"),
        )
        self.file_store.atomic_write_json(
            quality_path,
            quality.model_dump(mode="json"),
        )
        return RunReportFiles(
            run_config_path=paths.relative_path(run_config_path),
            execution_provenance_path=paths.relative_path(execution_path),
            transform_path=paths.relative_path(transform_path),
            particles_csv_path=paths.relative_path(particles_path),
            image_summary_path=paths.relative_path(summary_path),
            quality_report_path=paths.relative_path(quality_path),
        )

    def build_job_export(
        self,
        job_id: str,
        *,
        run_ids: set[str] | None = None,
        snapshot: JobExportSnapshot | None = None,
    ) -> StoredFile:
        # Job-level summaries share stable filenames. Serialize export assembly so
        # concurrent selections cannot overwrite a report while another ZIP reads it.
        with _EXPORT_LOCK:
            return self._build_job_export_locked(
                job_id,
                run_ids=run_ids,
                snapshot=snapshot,
            )

    def _build_job_export_locked(
        self,
        job_id: str,
        *,
        run_ids: set[str] | None,
        snapshot: JobExportSnapshot | None,
    ) -> StoredFile:
        if snapshot is not None:
            self._write_job_reports(job_id, snapshot=snapshot, run_ids=run_ids)
        job_dir = self.file_store.paths.require_managed(self.file_store.paths.job_dir(job_id))
        if not job_dir.is_dir():
            raise FileNotFoundError(job_id)
        files = self._declared_export_files(
            job_id,
            job_dir=job_dir,
            snapshot=snapshot,
            run_ids=run_ids,
        )
        # The exact selected member bytes define one immutable archive. Repeated
        # exports reuse it, while a changed snapshot publishes at a new path so
        # every previously issued token keeps resolving to its original bytes.
        return self.file_store.build_zip(job_id, files, filename=None)

    def _write_job_reports(
        self,
        job_id: str,
        *,
        snapshot: JobExportSnapshot,
        run_ids: set[str] | None,
    ) -> None:
        if snapshot.job.job_id != job_id:
            raise ValueError("export snapshot job_id does not match the requested job")
        if any(image.job_id != job_id for image in snapshot.images):
            raise ValueError("export snapshot contains an image from another job")
        if any(run.job_id != job_id for run in snapshot.runs):
            raise ValueError("export snapshot contains a run from another job")
        image_ids = {image.image_id for image in snapshot.images}
        if any(revision.image_id not in image_ids for revision in snapshot.box_revisions):
            raise ValueError("export snapshot contains boxes for an unknown image")

        for revision in snapshot.box_revisions:
            self.file_store.atomic_write_json(
                self.file_store.paths.boxes_revision(
                    job_id,
                    revision.image_id,
                    revision.revision,
                ),
                {
                    **revision.model_dump(mode="json"),
                    "coordinate_space": "original_px",
                },
            )

        selected = self._selected_runs(snapshot, run_ids=run_ids)
        images = {image.image_id: image for image in snapshot.images}
        status_counts = Counter(run.status.value for run in selected)
        self.file_store.atomic_write_json(
            self.file_store.paths.job_summary(job_id),
            {
                "job": snapshot.job.model_dump(mode="json"),
                "image_count": len(snapshot.images),
                "selected_run_count": len(selected),
                "selected_run_ids": [run.run_id for run in selected],
                "run_status_counts": dict(sorted(status_counts.items())),
                "quality_status_counts": dict(
                    sorted(
                        Counter(
                            run.quality.status.value for run in selected if run.quality is not None
                        ).items()
                    )
                ),
            },
        )
        self.file_store.atomic_write_bytes(
            self.file_store.paths.run_summary(job_id),
            self._run_summary_csv(selected, images).encode("utf-8-sig"),
        )
        sample_rows = self._sample_summary_rows(selected, images)
        self.file_store.atomic_write_bytes(
            self.file_store.paths.sample_summary(job_id),
            self._sample_summary_csv(sample_rows).encode("utf-8-sig"),
        )
        self.file_store.atomic_write_json(
            self.file_store.paths.audit_summary(job_id),
            {
                "job_id": job_id,
                "query_ids": [query.query_id for query in snapshot.queries],
                "run_lineage": [
                    {
                        "run_id": run.run_id,
                        "parent_run_id": run.parent_run_id,
                        "image_id": run.image_id,
                        "model_id": run.model_id,
                        "model_version": run.configuration.model_version,
                        "adapter_path": run.configuration.adapter_path,
                        "weight_sha256": run.configuration.weight_sha256,
                        "config_sha256": run.configuration.config_sha256,
                        "model_card_sha256": run.configuration.model_card_sha256,
                        "adapter_sha256": run.configuration.adapter_sha256,
                        "model_bundle": (
                            run.configuration.model_bundle.model_dump(mode="json")
                            if run.configuration.model_bundle is not None
                            else None
                        ),
                        "configuration_provenance_status": (run.configuration.provenance_status),
                        "configuration_provenance_warnings": (
                            run.configuration.provenance_warnings
                        ),
                        "image_sha256": run.configuration.image_sha256,
                        "scale_nm_per_pixel": run.configuration.scale_nm_per_pixel,
                        "resolved_postprocess": (
                            run.configuration.resolved_postprocess.model_dump(mode="json")
                            if run.configuration.resolved_postprocess is not None
                            else None
                        ),
                        "resolved_morphometry": (
                            run.configuration.resolved_morphometry.model_dump(mode="json")
                            if run.configuration.resolved_morphometry is not None
                            else None
                        ),
                        "resolved_quality_gate": (
                            run.configuration.resolved_quality_gate.model_dump(mode="json")
                            if run.configuration.resolved_quality_gate is not None
                            else None
                        ),
                        "execution_build": (
                            run.configuration.execution_build.model_dump(mode="json")
                            if run.configuration.execution_build is not None
                            else None
                        ),
                        "execution": (
                            run.execution.model_dump(mode="json")
                            if run.execution is not None
                            else None
                        ),
                        "box_revision": run.box_revision,
                        "review_source": run.configuration.review_source,
                        "status": run.status.value,
                        "quality_status": (
                            run.quality.status.value if run.quality is not None else None
                        ),
                        "error_code": run.error_code,
                        "status_history": [
                            event.model_dump(mode="json") for event in run.status_history
                        ],
                    }
                    for run in selected
                ],
            },
        )
        self._write_query_projections(job_id, snapshot.queries)
        self.file_store.atomic_write_json(
            self.file_store.paths.software_manifest(job_id),
            self._software_manifest(),
        )
        chart_rows = [
            (
                run.run_id,
                float(run.summary.particle_count),
                float(run.summary.coverage_ratio),
            )
            for run in selected
            if run.summary is not None
        ]
        self.file_store.atomic_write_bytes(
            self.file_store.paths.chart_file(job_id, "particle_count_by_run.svg"),
            self._bar_chart_svg(
                [(run_id, particle_count) for run_id, particle_count, _ in chart_rows],
                title="Particle count by run",
                value_label="particles",
            ).encode("utf-8"),
        )
        self.file_store.atomic_write_bytes(
            self.file_store.paths.chart_file(job_id, "coverage_ratio_by_run.svg"),
            self._bar_chart_svg(
                [(run_id, coverage) for run_id, _, coverage in chart_rows],
                title="Coverage ratio by run",
                value_label="ratio",
            ).encode("utf-8"),
        )

    def _write_query_projections(
        self,
        job_id: str,
        queries: tuple[QueryAuditRecordDTO, ...],
    ) -> None:
        history_lines = [
            self._json_line(
                {
                    "schema_version": "1.0",
                    "query_id": query.query_id,
                    "created_at": query.created_at.isoformat(),
                    "request": query.request.model_dump(mode="json"),
                    "response": query.response.model_dump(mode="json"),
                }
            )
            for query in queries
        ]
        self.file_store.atomic_write_bytes(
            self.file_store.paths.query_history(job_id),
            b"".join(history_lines),
        )
        self.file_store.atomic_write_json(
            self.file_store.paths.rag_citations(job_id),
            {
                "job_id": job_id,
                "queries": [
                    {
                        "query_id": query.query_id,
                        "created_at": query.created_at.isoformat(),
                        "citations": [
                            citation.model_dump(mode="json")
                            for citation in query.response.citations
                        ],
                    }
                    for query in queries
                ],
            },
        )

    @staticmethod
    def _json_line(payload: dict[str, object]) -> bytes:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _run_summary_csv(
        runs: list[SegmentationRunDTO],
        images: dict[str, ImageAssetDTO],
    ) -> str:
        columns = (
            "run_id",
            "parent_run_id",
            "image_id",
            "filename",
            "sample_id",
            "material_name",
            "model_id",
            "model_version",
            "status",
            "quality_status",
            "particle_count",
            "roi_area_px",
            "number_density_px2",
            "number_density_um2",
            "mean_equivalent_diameter_px",
            "mean_equivalent_diameter_nm",
            "coverage_ratio",
            "runtime_ms",
        )
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            image = images.get(run.image_id)
            summary = run.summary
            writer.writerow(
                _csv_safe_row(
                    {
                        "run_id": run.run_id,
                        "parent_run_id": run.parent_run_id,
                        "image_id": run.image_id,
                        "filename": image.filename if image else None,
                        "sample_id": image.sample_id if image else None,
                        "material_name": image.material_name if image else None,
                        "model_id": run.model_id,
                        "model_version": run.configuration.model_version,
                        "status": run.status.value,
                        "quality_status": run.quality.status.value if run.quality else None,
                        "particle_count": summary.particle_count if summary else None,
                        "roi_area_px": summary.roi_area_px if summary else None,
                        "number_density_px2": summary.number_density_px2 if summary else None,
                        "number_density_um2": summary.number_density_um2 if summary else None,
                        "mean_equivalent_diameter_px": (
                            summary.mean_equivalent_diameter_px if summary else None
                        ),
                        "mean_equivalent_diameter_nm": (
                            summary.mean_equivalent_diameter_nm if summary else None
                        ),
                        "coverage_ratio": summary.coverage_ratio if summary else None,
                        "runtime_ms": run.runtime_ms,
                    }
                )
            )
        return buffer.getvalue()

    @staticmethod
    def _sample_summary_rows(
        runs: list[SegmentationRunDTO],
        images: dict[str, ImageAssetDTO],
    ) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str], list[SegmentationRunDTO]] = defaultdict(list)
        for run in runs:
            image = images.get(run.image_id)
            if image is not None and run.summary is not None:
                grouped[(image.sample_id, run.model_id)].append(run)

        rows: list[dict[str, object]] = []
        quality_rank = {
            QualityStatus.PASS: 0,
            QualityStatus.WARN: 1,
            QualityStatus.REVIEW_REQUIRED: 2,
        }
        for (sample_id, model_id), group in sorted(grouped.items()):
            summaries = [run.summary for run in group if run.summary is not None]
            total_particles = sum(summary.particle_count for summary in summaries)
            total_roi = sum(summary.roi_area_px for summary in summaries)
            diameter_px_weight = sum(
                summary.particle_count
                for summary in summaries
                if summary.mean_equivalent_diameter_px is not None
            )
            diameter_nm_weight = sum(
                summary.particle_count
                for summary in summaries
                if summary.mean_equivalent_diameter_nm is not None
            )
            statuses = [run.quality.status for run in group if run.quality is not None]
            rows.append(
                {
                    "sample_id": sample_id,
                    "model_id": model_id,
                    "image_count": len({run.image_id for run in group}),
                    "run_count": len(group),
                    "particle_count_total": total_particles,
                    "roi_area_px_total": total_roi,
                    "number_density_px2": total_particles / total_roi if total_roi else None,
                    "coverage_ratio_weighted": (
                        sum(summary.coverage_ratio * summary.roi_area_px for summary in summaries)
                        / total_roi
                        if total_roi
                        else None
                    ),
                    "mean_equivalent_diameter_px_weighted": (
                        sum(
                            (summary.mean_equivalent_diameter_px or 0) * summary.particle_count
                            for summary in summaries
                            if summary.mean_equivalent_diameter_px is not None
                        )
                        / diameter_px_weight
                        if diameter_px_weight
                        else None
                    ),
                    "mean_equivalent_diameter_nm_weighted": (
                        sum(
                            (summary.mean_equivalent_diameter_nm or 0) * summary.particle_count
                            for summary in summaries
                            if summary.mean_equivalent_diameter_nm is not None
                        )
                        / diameter_nm_weight
                        if diameter_nm_weight
                        else None
                    ),
                    "worst_quality_status": (
                        max(statuses, key=quality_rank.__getitem__).value if statuses else None
                    ),
                }
            )
        return rows

    @staticmethod
    def _sample_summary_csv(rows: list[dict[str, object]]) -> str:
        columns = (
            "sample_id",
            "model_id",
            "image_count",
            "run_count",
            "particle_count_total",
            "roi_area_px_total",
            "number_density_px2",
            "coverage_ratio_weighted",
            "mean_equivalent_diameter_px_weighted",
            "mean_equivalent_diameter_nm_weighted",
            "worst_quality_status",
        )
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_safe_row(row) for row in rows)
        return buffer.getvalue()

    @staticmethod
    def _software_manifest() -> dict[str, object]:
        try:
            requirements = sorted(metadata.requires("nanoloop-agent") or [])
        except metadata.PackageNotFoundError:
            requirements = []
        build = capture_execution_build_provenance().model_dump(mode="json")
        return {**build, "dependencies": requirements}

    @staticmethod
    def _bar_chart_svg(
        rows: list[tuple[str, float]],
        *,
        title: str,
        value_label: str,
    ) -> str:
        width = 960
        row_height = 34
        top = 74
        bottom = 34
        height = max(180, top + bottom + row_height * max(len(rows), 1))
        label_width = 250
        chart_width = width - label_width - 90
        maximum = max((value for _, value in rows), default=0.0)
        scale = chart_width / maximum if maximum > 0 else 0.0
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            f'<text x="28" y="38" font-family="sans-serif" font-size="22" '
            f'font-weight="700" fill="#172033">{escape(title)}</text>',
        ]
        if not rows:
            elements.append(
                '<text x="28" y="96" font-family="sans-serif" font-size="15" '
                'fill="#5b6475">No completed run summaries available.</text>'
            )
        for index, (label, value) in enumerate(rows):
            y = top + index * row_height
            bar_width = max(0.0, value * scale)
            compact_label = label if len(label) <= 28 else f"{label[:25]}..."
            elements.extend(
                [
                    f'<text x="28" y="{y + 20}" font-family="monospace" font-size="12" '
                    f'fill="#39445a">{escape(compact_label)}</text>',
                    f'<rect x="{label_width}" y="{y + 5}" width="{bar_width:.2f}" '
                    'height="19" rx="3" fill="#0f766e"/>',
                    f'<text x="{label_width + bar_width + 8:.2f}" y="{y + 20}" '
                    'font-family="sans-serif" font-size="12" fill="#172033">'
                    f"{value:.6g} {escape(value_label)}</text>",
                ]
            )
        elements.append("</svg>")
        return "\n".join(elements) + "\n"

    @staticmethod
    def _selected_runs(
        snapshot: JobExportSnapshot,
        *,
        run_ids: set[str] | None,
    ) -> list[SegmentationRunDTO]:
        return sorted(
            (run for run in snapshot.runs if run_ids is None or run.run_id in run_ids),
            key=lambda run: run.run_id,
        )

    def _declared_export_files(
        self,
        job_id: str,
        *,
        job_dir: Path,
        snapshot: JobExportSnapshot | None,
        run_ids: set[str] | None,
    ) -> list[Path]:
        """Collect only durable files declared by the export snapshot.

        Corrected-mask uploads are first staged below ``input/review_mask_*``.
        A recursive job-directory scan would expose those capability-addressed
        temporary files, as well as arbitrary crash residue, in every export.
        """

        if snapshot is None:
            # The snapshot-free entry point is retained for low-level report
            # tests and callers that only need canonical run reports. It never
            # walks input staging or arbitrary job-level files.
            return [
                path
                for path in sorted(job_dir.rglob("*"))
                if path.is_file()
                and self._is_canonical_run_artifact(
                    path,
                    job_dir=job_dir,
                    run_ids=run_ids,
                )
            ]

        paths = self.file_store.paths
        selected = self._selected_runs(snapshot, run_ids=run_ids)
        selected_by_id = {run.run_id: run for run in selected}
        snapshot_run_ids = {run.run_id for run in snapshot.runs}
        declared: set[Path] = set()

        def include_if_present(path: str | Path) -> None:
            managed = paths.require_managed(path)
            if managed.is_file():
                declared.add(managed)

        for fixed_path in (
            paths.job_manifest(job_id),
            paths.job_config(job_id),
            paths.job_summary(job_id),
            paths.run_summary(job_id),
            paths.sample_summary(job_id),
            paths.audit_summary(job_id),
            paths.software_manifest(job_id),
            paths.query_history(job_id),
            paths.rag_citations(job_id),
            paths.chart_file(job_id, "particle_count_by_run.svg"),
            paths.chart_file(job_id, "coverage_ratio_by_run.svg"),
        ):
            include_if_present(fixed_path)

        image_input_dirs = {
            paths.require_managed(paths.input_dir(job_id, image.image_id))
            for image in snapshot.images
        }
        for storage_path in snapshot.image_storage_paths:
            managed = paths.require_managed(storage_path)
            if managed.parent not in image_input_dirs or not managed.name.startswith("original."):
                raise ValueError("export snapshot contains an undeclared image storage path")
            include_if_present(managed)

        for image in snapshot.images:
            include_if_present(paths.image_metadata(job_id, image.image_id))
        for revision in snapshot.box_revisions:
            include_if_present(paths.boxes_revision(job_id, revision.image_id, revision.revision))

        for run in selected:
            for filename in self._CANONICAL_RUN_ARTIFACT_FILENAMES:
                include_if_present(paths.run_artifact(job_id, run.image_id, run.run_id, filename))
            if run.configuration.review_source == "corrected_mask":
                include_if_present(
                    paths.run_artifact(job_id, run.image_id, run.run_id, "corrected_mask.png")
                )

        for artifact_run_id, artifact_path in sorted(snapshot.run_artifact_paths):
            if artifact_run_id not in snapshot_run_ids:
                raise ValueError("export snapshot contains artifacts for an unknown run")
            selected_run = selected_by_id.get(artifact_run_id)
            if selected_run is None:
                continue
            managed = paths.require_managed(artifact_path)
            run_dir = paths.require_managed(
                paths.run_dir(job_id, selected_run.image_id, selected_run.run_id)
            )
            try:
                relative = managed.relative_to(run_dir)
            except ValueError as error:
                raise ValueError("export snapshot contains an artifact outside its run") from error
            if not relative.parts or any(
                part.startswith(".") or part.endswith(".tmp") for part in relative.parts
            ):
                raise ValueError("export snapshot contains a transient run artifact")
            include_if_present(managed)

        return sorted(declared)

    def _is_canonical_run_artifact(
        self,
        path: Path,
        *,
        job_dir: Path,
        run_ids: set[str] | None,
    ) -> bool:
        parts = path.relative_to(job_dir).parts
        if len(parts) != 5 or parts[0] != "images" or parts[2] != "runs":
            return False
        run_id = parts[3]
        return (run_ids is None or run_id in run_ids) and parts[
            4
        ] in self._CANONICAL_RUN_ARTIFACT_FILENAMES

    def _particles_csv(self, result: MorphometryResult) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=self.PARTICLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for particle in result.particles:
            x1, y1, x2, y2 = particle.bbox
            writer.writerow(
                _csv_safe_row(
                    {
                        "particle_id": particle.particle_id,
                        "run_id": particle.run_id,
                        "instance_index": particle.instance_index,
                        "area_px": particle.area_px,
                        "perimeter_px": particle.perimeter_px,
                        "equivalent_diameter_px": particle.equivalent_diameter_px,
                        "equivalent_diameter_nm": particle.equivalent_diameter_nm,
                        "circularity": particle.circularity,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "confidence": particle.confidence,
                    }
                )
            )
        return buffer.getvalue()

    @staticmethod
    def _contract_payload(payload: dict[str, object]) -> dict[str, object]:
        """Keep contract and file schema versions distinct at the JSON document boundary."""

        prepared = dict(payload)
        contract_version = prepared.pop("schema_version", None)
        if contract_version is not None:
            prepared["contract_schema_version"] = contract_version
        return prepared


def resolve_report_path(file_store: LocalFileStore, relative_path: str) -> Path:
    """Resolve a stored report key without allowing it to escape managed storage."""

    return file_store.paths.require_managed(relative_path, must_exist=True)
