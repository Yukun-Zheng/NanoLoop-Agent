"""Fail-closed validation for the private RAG acceptance asset contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.embedding_snapshot_fingerprint import (
    directory_tree_sha256 as _canonical_embedding_tree_sha256,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,99}$")
_QUERY_ID = re.compile(r"^q[0-9]{3,}$")
_APPROVED_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
_APPROVED_EMBEDDING_REVISION = "13942ee7a1615d20a84b41e800c63775e174f97f"
_APPROVED_EMBEDDING_DIMENSION = 512
_APPROVED_EMBEDDING_TREE_SHA256 = (
    "b24a9bacfdd51e203abc9e060567ae565fa094f95a400a6fbbae502904e1afd4"
)
_EMBEDDING_PROBE = "二氧化钛纳米颗粒的晶型与光催化性能有什么关系？"
_DECISIONS = {
    "DISCOVERY_ONLY",
    "METADATA_ONLY",
    "REVIEW_REQUIRED",
    "CANDIDATE_FULLTEXT",
    "ACCEPT_FULLTEXT",
}
_EXPECTED_OUTCOMES = {"OK", "INSUFFICIENT_EVIDENCE"}
_RAG_QUERY_TYPES = {"material_knowledge", "mixed"}
_CORPUS_FIELDS = {
    "asset_id",
    "decision",
    "title",
    "authors",
    "year",
    "source_url",
    "doi",
    "license",
    "license_url",
    "license_evidence_url",
    "citation_text",
    "material_names",
    "formula",
    "aliases",
    "file_sha256",
    "allowed_for_demo",
    "page_text_verified",
    "reviewed_by",
    "reviewed_at",
    "file_path",
}


@dataclass(frozen=True, slots=True)
class ValidatedAcceptancePackage:
    root: Path
    corpus_rows: tuple[dict[str, str], ...]
    questions: tuple[dict[str, Any], ...]
    embedding: dict[str, Any]

    @property
    def accepted_rows(self) -> tuple[dict[str, str], ...]:
        return tuple(row for row in self.corpus_rows if row["decision"] == "ACCEPT_FULLTEXT")


class AssetValidationError(ValueError):
    """The acceptance package cannot safely be used for a real run."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(f"- {issue}" for issue in self.issues))


class _SentenceTransformerModel(Protocol):
    def get_embedding_dimension(self) -> int | None: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


def _sentence_transformer_factory() -> Callable[..., object]:
    """Resolve the optional runtime only when validating a runnable package."""

    module = importlib.import_module("sentence_transformers")
    factory = module.SentenceTransformer
    return cast(Callable[..., object], factory)


def _load_sentence_transformer(local_dir: Path) -> _SentenceTransformerModel:
    factory = _sentence_transformer_factory()
    model = factory(
        str(local_dir),
        device="cpu",
        revision=_APPROVED_EMBEDDING_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    return cast(_SentenceTransformerModel, model)


def _validate_embedding_runtime(path: Path, local_dir: Path) -> list[str]:
    """Load and encode from the fingerprinted local snapshot without network access."""

    try:
        model = _load_sentence_transformer(local_dir)
    except Exception as error:
        return [
            f"{path}: offline SentenceTransformer load failed from {local_dir}: "
            f"{type(error).__name__}: {error}"
        ]

    issues: list[str] = []
    try:
        try:
            observed_dimension = model.get_embedding_dimension()
        except AttributeError:
            # SentenceTransformers <5 used the longer method name.
            observed_dimension = model.get_sentence_embedding_dimension()
    except Exception as error:
        issues.append(
            f"{path}: embedding dimension inspection failed: "
            f"{type(error).__name__}: {error}"
        )
    else:
        if observed_dimension != _APPROVED_EMBEDDING_DIMENSION:
            issues.append(
                f"{path}: loaded embedding dimension must be "
                f"{_APPROVED_EMBEDDING_DIMENSION}, observed {observed_dimension!r}"
            )

    try:
        encoded = model.encode(
            [_EMBEDDING_PROBE],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = np.asarray(encoded, dtype=np.float32)
    except Exception as error:
        issues.append(
            f"{path}: offline embedding probe failed: {type(error).__name__}: {error}"
        )
        return issues

    expected_shape = (1, _APPROVED_EMBEDDING_DIMENSION)
    if vector.shape != expected_shape:
        issues.append(
            f"{path}: embedding probe must return shape {expected_shape}, "
            f"observed {vector.shape}"
        )
        return issues
    if not bool(np.isfinite(vector).all()):
        issues.append(f"{path}: embedding probe returned non-finite values")
        return issues
    norm = float(np.linalg.norm(vector[0]))
    if not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-5):
        issues.append(
            f"{path}: normalized embedding probe must have unit norm, observed {norm:.8g}"
        )
    return issues


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetValidationError([f"{path}: invalid JSON: {error}"]) from error
    if not isinstance(value, dict):
        raise AssetValidationError([f"{path}: expected one JSON object"])
    return value


def _jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    values: list[dict[str, Any]] = []
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], [f"{path}: cannot read: {error}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            issues.append(f"{path}:{line_number}: invalid JSON: {error.msg}")
            continue
        if not isinstance(value, dict):
            issues.append(f"{path}:{line_number}: expected one JSON object")
            continue
        values.append(value)
    return values, issues


def _is_https_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
    )


def _license_is_reviewed(value: str) -> bool:
    normalized = value.strip().casefold()
    return bool(normalized) and not any(
        marker in normalized for marker in ("not reviewed", "unknown", "todo", "tbd")
    )


def _parse_bool(value: str, *, location: str, issues: list[str]) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    issues.append(f"{location}: expected true or false")
    return None


def _safe_source_path(root: Path, value: str, *, location: str, issues: list[str]) -> Path:
    relative = PurePosixPath(value)
    unsafe_part = any(part in {"", ".", ".."} for part in relative.parts)
    if not value or relative.is_absolute() or unsafe_part:
        issues.append(f"{location}: file_path must be a normalized relative path")
        return root / "__invalid__"
    candidate = (root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        issues.append(f"{location}: file_path escapes corpus/sources")
    return candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(root: Path) -> str:
    try:
        return _canonical_embedding_tree_sha256(root)
    except (ValueError, RuntimeError) as error:
        raise AssetValidationError([f"{root}: {error}"]) from error


def _validate_corpus(
    root: Path,
    *,
    require_runnable: bool,
    verify_files: bool,
) -> tuple[tuple[dict[str, str], ...], list[str]]:
    path = root / "corpus" / "corpus-manifest.csv"
    issues: list[str] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = set(reader.fieldnames or [])
            if fieldnames != _CORPUS_FIELDS:
                missing = sorted(_CORPUS_FIELDS - fieldnames)
                extra = sorted(fieldnames - _CORPUS_FIELDS)
                issues.append(f"{path}: schema mismatch; missing={missing}, extra={extra}")
            rows = list(reader)
    except (OSError, csv.Error) as error:
        return (), [f"{path}: invalid CSV: {error}"]

    normalized_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    sources_root = (root / "corpus" / "sources").resolve()
    for line_number, raw in enumerate(rows, start=2):
        location = f"{path}:{line_number}"
        if None in raw or any(value is None for value in raw.values()):
            issues.append(f"{location}: row does not match the header column count")
            continue
        row = {key: value.strip() for key, value in raw.items() if key is not None}
        asset_id = row.get("asset_id", "")
        if not _ASSET_ID.fullmatch(asset_id):
            issues.append(f"{location}: invalid asset_id {asset_id!r}")
        elif asset_id in seen:
            issues.append(f"{location}: duplicate asset_id {asset_id}")
        seen.add(asset_id)
        decision = row.get("decision", "")
        if decision not in _DECISIONS:
            issues.append(f"{location}: invalid decision {decision!r}")
        if not row.get("title"):
            issues.append(f"{location}: title is required")
        if not row.get("formula"):
            issues.append(f"{location}: formula is required")
        if not row.get("aliases"):
            issues.append(f"{location}: aliases are required")
        allowed = _parse_bool(
            row.get("allowed_for_demo", ""), location=f"{location}:allowed_for_demo", issues=issues
        )
        page_verified = _parse_bool(
            row.get("page_text_verified", ""),
            location=f"{location}:page_text_verified",
            issues=issues,
        )
        if decision != "ACCEPT_FULLTEXT" and allowed is True:
            issues.append(f"{location}: only ACCEPT_FULLTEXT may be allowed_for_demo")
        file_path = _safe_source_path(
            sources_root, row.get("file_path", ""), location=location, issues=issues
        )
        supplied_sha = row.get("file_sha256", "")
        if supplied_sha and not _SHA256.fullmatch(supplied_sha):
            issues.append(f"{location}: file_sha256 must be a lowercase SHA-256 digest")

        if decision == "ACCEPT_FULLTEXT":
            required = (
                "authors",
                "year",
                "source_url",
                "license",
                "license_url",
                "license_evidence_url",
                "citation_text",
                "file_sha256",
                "reviewed_by",
                "reviewed_at",
            )
            missing = [field for field in required if not row.get(field)]
            if missing:
                issues.append(f"{location}: ACCEPT_FULLTEXT missing {missing}")
            if row.get("license") and not _license_is_reviewed(row["license"]):
                issues.append(f"{location}: ACCEPT_FULLTEXT license is still unreviewed")
            if row.get("year"):
                try:
                    year = int(row["year"])
                except ValueError:
                    issues.append(f"{location}: year must be an integer")
                else:
                    if not 1000 <= year <= 3000:
                        issues.append(f"{location}: year must be between 1000 and 3000")
            if allowed is not True:
                issues.append(f"{location}: ACCEPT_FULLTEXT requires allowed_for_demo=true")
            if page_verified is not True:
                issues.append(f"{location}: ACCEPT_FULLTEXT requires page_text_verified=true")
            for field in ("source_url", "license_url", "license_evidence_url"):
                if row.get(field) and not _is_https_url(row[field]):
                    issues.append(f"{location}: {field} must be an HTTPS URL without credentials")
            if row.get("reviewed_at"):
                try:
                    date.fromisoformat(row["reviewed_at"])
                except ValueError:
                    issues.append(f"{location}: reviewed_at must be YYYY-MM-DD")
            if verify_files or require_runnable:
                if not file_path.is_file():
                    issues.append(f"{location}: accepted source file is missing: {file_path}")
                elif _SHA256.fullmatch(supplied_sha):
                    observed = file_sha256(file_path)
                    if observed != supplied_sha:
                        issues.append(
                            f"{location}: source SHA mismatch; expected {supplied_sha}, "
                            f"observed {observed}"
                        )
        normalized_rows.append(row)

    accepted = [row for row in normalized_rows if row.get("decision") == "ACCEPT_FULLTEXT"]
    if require_runnable and not 5 <= len(accepted) <= 10:
        issues.append(
            f"{path}: runnable acceptance requires 5-10 ACCEPT_FULLTEXT rows; found {len(accepted)}"
        )
    return tuple(normalized_rows), issues


def _validate_embedding(
    root: Path,
    *,
    require_runnable: bool,
) -> tuple[dict[str, Any], list[str]]:
    path = root / "embedding" / "model-manifest.json"
    try:
        value = _json_object(path)
    except AssetValidationError as error:
        return {}, list(error.issues)
    issues: list[str] = []
    status = value.get("status")
    if status not in {"candidate_unverified", "verified"}:
        issues.append(f"{path}: status must be candidate_unverified or verified")
    verified = value.get("verified") is True
    if verified != (status == "verified"):
        issues.append(f"{path}: status and verified must agree")
    if not isinstance(value.get("dimension"), int) or value["dimension"] <= 0:
        issues.append(f"{path}: dimension must be a positive integer")
    if not isinstance(value.get("normalize"), bool):
        issues.append(f"{path}: normalize must be boolean")
    if require_runnable:
        if value.get("model") != _APPROVED_EMBEDDING_MODEL:
            issues.append(
                f"{path}: runnable embedding model must be "
                f"{_APPROVED_EMBEDDING_MODEL!r}"
            )
        if value.get("revision") != _APPROVED_EMBEDDING_REVISION:
            issues.append(
                f"{path}: runnable embedding revision must be "
                f"{_APPROVED_EMBEDDING_REVISION}"
            )
        if value.get("dimension") != _APPROVED_EMBEDDING_DIMENSION:
            issues.append(
                f"{path}: runnable embedding dimension must be "
                f"{_APPROVED_EMBEDDING_DIMENSION}"
            )
        if value.get("normalize") is not True:
            issues.append(f"{path}: runnable embedding requires normalize=true")
        if not verified:
            issues.append(f"{path}: embedding snapshot is not verified")
        revision = value.get("revision")
        tree_sha = value.get("tree_sha256")
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            issues.append(f"{path}: revision must be an immutable 40-character commit digest")
        if not isinstance(tree_sha, str) or not _SHA256.fullmatch(tree_sha):
            issues.append(f"{path}: tree_sha256 must be a lowercase SHA-256 digest")
        elif tree_sha != _APPROVED_EMBEDDING_TREE_SHA256:
            issues.append(
                f"{path}: runnable embedding tree SHA must be "
                f"{_APPROVED_EMBEDDING_TREE_SHA256}"
            )
        for field in ("license", "license_url", "local_dir", "verified_by", "verified_at"):
            if not value.get(field):
                issues.append(f"{path}: verified embedding missing {field}")
        license_name = value.get("license")
        if isinstance(license_name, str) and not _license_is_reviewed(license_name):
            issues.append(f"{path}: embedding license is still unreviewed")
        if value.get("license_url") and not _is_https_url(str(value["license_url"])):
            issues.append(f"{path}: license_url must be an HTTPS URL without credentials")
        verified_at = value.get("verified_at")
        if isinstance(verified_at, str) and verified_at:
            try:
                date.fromisoformat(verified_at)
            except ValueError:
                issues.append(f"{path}: verified_at must be YYYY-MM-DD")
        local_value = value.get("local_dir")
        if isinstance(local_value, str) and local_value:
            local_dir = Path(local_value).expanduser().resolve()
            if not local_dir.is_dir():
                issues.append(f"{path}: local_dir is not a directory: {local_dir}")
            else:
                try:
                    observed = directory_tree_sha256(local_dir)
                except (AssetValidationError, OSError) as error:
                    issues.append(f"{path}: cannot fingerprint local_dir: {error}")
                else:
                    if observed != _APPROVED_EMBEDDING_TREE_SHA256:
                        issues.append(
                            f"{path}: embedding tree SHA mismatch; expected "
                            f"{_APPROVED_EMBEDDING_TREE_SHA256}, observed {observed}"
                        )
                    elif tree_sha == _APPROVED_EMBEDDING_TREE_SHA256:
                        issues.extend(_validate_embedding_runtime(path, local_dir))
    return value, issues


def _validate_questions(
    root: Path,
    rows: tuple[dict[str, str], ...],
    *,
    require_runnable: bool,
) -> tuple[tuple[dict[str, Any], ...], list[str]]:
    path = root / "evaluation" / "questions.jsonl"
    questions, issues = _jsonl_objects(path)
    assets = {row["asset_id"]: row for row in rows}
    seen: set[str] = set()
    required = {
        "query_id",
        "question",
        "language",
        "query_type",
        "material_context",
        "case_type",
        "relevant_asset_ids",
        "relevant_pages",
        "expected_outcome",
        "must_not_return_asset_ids",
        "annotation_status",
        "annotated_by",
        "reviewed_by",
    }
    observed_only_fields = {
        "passed",
        "final_passed",
        "retrieved_doc_ids",
        "citation_correct",
        "leaked_material",
        "fabricated_fact",
    }
    for line_number, question in enumerate(questions, start=1):
        location = f"{path}:{line_number}"
        missing = sorted(required - question.keys())
        if missing:
            issues.append(f"{location}: missing fields {missing}")
            continue
        leaked_observations = sorted(observed_only_fields & question.keys())
        if leaked_observations:
            issues.append(
                f"{location}: expected question contains observed result fields "
                f"{leaked_observations}"
            )
        query_id = question.get("query_id")
        if not isinstance(query_id, str) or not _QUERY_ID.fullmatch(query_id):
            issues.append(f"{location}: invalid query_id")
        elif query_id in seen:
            issues.append(f"{location}: duplicate query_id {query_id}")
        seen.add(str(query_id))
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            issues.append(f"{location}: question must be non-empty")
        if question.get("query_type") not in _RAG_QUERY_TYPES:
            issues.append(f"{location}: query_type must be material_knowledge or mixed")
        outcome = question.get("expected_outcome")
        if outcome not in _EXPECTED_OUTCOMES:
            issues.append(f"{location}: invalid expected_outcome {outcome!r}")
        relevant = question.get("relevant_asset_ids")
        forbidden = question.get("must_not_return_asset_ids")
        pages = question.get("relevant_pages")
        for field, value in (
            ("relevant_asset_ids", relevant),
            ("must_not_return_asset_ids", forbidden),
            ("relevant_pages", pages),
        ):
            if not isinstance(value, list):
                issues.append(f"{location}: {field} must be a list")
        if isinstance(relevant, list) and isinstance(forbidden, list):
            relevant_set = {item for item in relevant if isinstance(item, str)}
            forbidden_set = {item for item in forbidden if isinstance(item, str)}
            unknown = sorted((relevant_set | forbidden_set) - assets.keys())
            if unknown:
                issues.append(f"{location}: unknown asset IDs {unknown}")
            overlap = sorted(relevant_set & forbidden_set)
            if overlap:
                issues.append(
                    f"{location}: assets cannot be both relevant and forbidden: {overlap}"
                )
        status = question.get("annotation_status")
        if status not in {"draft", "final"}:
            issues.append(f"{location}: annotation_status must be draft or final")
        annotated_by = question.get("annotated_by")
        reviewed_by = question.get("reviewed_by")
        if not isinstance(annotated_by, str) or not annotated_by.strip():
            issues.append(f"{location}: annotated_by is required")
        if status == "final":
            if not isinstance(reviewed_by, str) or not reviewed_by.strip():
                issues.append(f"{location}: final annotation requires reviewed_by")
            elif reviewed_by == annotated_by:
                issues.append(f"{location}: annotator and reviewer must be different people")
            if outcome == "OK":
                if not relevant:
                    issues.append(f"{location}: OK requires at least one relevant_asset_id")
                if not pages or any(not isinstance(page, int) or page < 1 for page in pages):
                    issues.append(f"{location}: OK requires verified positive source pages")
        if require_runnable:
            if status != "final":
                issues.append(f"{location}: runnable acceptance requires final annotations")
            if isinstance(relevant, list):
                unaccepted = sorted(
                    asset_id
                    for asset_id in relevant
                    if isinstance(asset_id, str)
                    and assets.get(asset_id, {}).get("decision") != "ACCEPT_FULLTEXT"
                )
                if unaccepted:
                    issues.append(f"{location}: relevant assets are not accepted: {unaccepted}")
    if len(questions) < 20:
        issues.append(f"{path}: at least 20 questions are required; found {len(questions)}")
    if require_runnable:
        accepted_assets = {
            row["asset_id"] for row in rows if row["decision"] == "ACCEPT_FULLTEXT"
        }
        covered_assets = {
            asset_id
            for question in questions
            if question.get("annotation_status") == "final"
            and question.get("expected_outcome") == "OK"
            for asset_id in question.get("relevant_asset_ids", [])
            if isinstance(asset_id, str)
        }
        uncovered = sorted(accepted_assets - covered_assets)
        if uncovered:
            issues.append(f"{path}: accepted assets lack final OK questions: {uncovered}")
    return tuple(questions), issues


def _validate_package_metadata(
    root: Path,
    rows: tuple[dict[str, str], ...],
    questions: tuple[dict[str, Any], ...],
    embedding: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    ledger_path = root / "asset-ledger.json"
    judgment_path = root / "evaluation" / "judgment-schema.json"
    try:
        ledger = _json_object(ledger_path)
    except AssetValidationError as error:
        issues.extend(error.issues)
        ledger = {}
    summary = ledger.get("corpus_summary")
    if not isinstance(summary, dict):
        issues.append(f"{ledger_path}: corpus_summary must be an object")
    else:
        expected_summary = {
            "total_candidates": len(rows),
            "candidate_fulltext": sum(
                row["decision"] == "CANDIDATE_FULLTEXT" for row in rows
            ),
            "accepted_fulltext": sum(row["decision"] == "ACCEPT_FULLTEXT" for row in rows),
        }
        for field, expected in expected_summary.items():
            if summary.get(field) != expected:
                issues.append(
                    f"{ledger_path}: corpus_summary.{field} must be {expected}, "
                    f"observed {summary.get(field)!r}"
                )
    if ledger.get("embedding_status") != embedding.get("status"):
        issues.append(f"{ledger_path}: embedding_status does not match model-manifest.json")
    if ledger.get("real_acceptance_completed") is not False:
        issues.append(f"{ledger_path}: checked-in scaffold must not claim real acceptance")
    try:
        judgment = _json_object(judgment_path)
    except AssetValidationError as error:
        issues.extend(error.issues)
        judgment = {}
    if judgment.get("expected_source") != "questions.jsonl":
        issues.append(f"{judgment_path}: expected_source must be questions.jsonl")
    pattern = judgment.get("observed_output_pattern")
    if not isinstance(pattern, str) or "results" not in pattern:
        issues.append(f"{judgment_path}: observed_output_pattern is missing")
    if any(question.get("final_passed") is True for question in questions):
        issues.append(f"{judgment_path}: checked-in expectations cannot pre-pass questions")
    return issues


def validate_acceptance_package(
    package: str | Path,
    *,
    require_runnable: bool = False,
    verify_files: bool = False,
) -> ValidatedAcceptancePackage:
    root = Path(package).expanduser().resolve()
    if not root.is_dir():
        raise AssetValidationError([f"{root}: package directory does not exist"])
    rows, corpus_issues = _validate_corpus(
        root, require_runnable=require_runnable, verify_files=verify_files
    )
    embedding, embedding_issues = _validate_embedding(root, require_runnable=require_runnable)
    questions, question_issues = _validate_questions(
        root, rows, require_runnable=require_runnable
    )
    metadata_issues = _validate_package_metadata(root, rows, questions, embedding)
    issues = [*corpus_issues, *embedding_issues, *question_issues, *metadata_issues]
    if issues:
        raise AssetValidationError(issues)
    return ValidatedAcceptancePackage(
        root=root,
        corpus_rows=rows,
        questions=questions,
        embedding=embedding,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a NanoLoop RAG acceptance package")
    parser.add_argument("--package", default="rag-acceptance-v1")
    parser.add_argument("--require-runnable", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        package = validate_acceptance_package(
            args.package,
            require_runnable=args.require_runnable,
            verify_files=args.verify_files,
        )
    except AssetValidationError as error:
        print(json.dumps({"valid": False, "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "status": "runnable" if args.require_runnable else "schema-valid-draft",
                "corpus_rows": len(package.corpus_rows),
                "accepted_fulltext": len(package.accepted_rows),
                "questions": len(package.questions),
                "embedding_verified": package.embedding.get("verified") is True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
