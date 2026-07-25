#!/usr/bin/env python3
"""Run the sealed-validation NanoLoop-MSBI A-E curriculum in order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = ("a", "b", "c", "d", "e")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _assert_sealed_manifest_root(manifest_root: Path) -> None:
    required = {
        "train.jsonl",
        "validation.jsonl",
        "split-manifest.json",
    }
    missing = sorted(
        name for name in required if not (manifest_root / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"missing training manifest files: {missing}")
    forbidden = tuple(manifest_root.glob("*independent*"))
    if forbidden:
        raise RuntimeError(
            "independent-test artifacts must not be present on the training host: "
            + ", ".join(str(path) for path in forbidden)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_root = args.manifest_root.resolve()
    output_root = args.output_root.resolve()
    _assert_sealed_manifest_root(manifest_root)
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "formal-run-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "python": sys.executable,
        "repo_root": str(repo_root),
        "manifest_root": str(manifest_root),
        "split_manifest_sha256": _sha256(
            manifest_root / "split-manifest.json"
        ),
        "independent_test_transferred": False,
        "device": args.device,
        "stages": [],
    }
    _write_manifest(manifest_path, manifest)

    previous_best: Path | None = None
    try:
        for stage in STAGES:
            _assert_sealed_manifest_root(manifest_root)
            config = (
                repo_root
                / "model_artifacts"
                / "training_configs"
                / f"msbi-stage-{stage}.yaml"
            )
            stage_output = output_root / f"stage-{stage}"
            command = [
                sys.executable,
                str(repo_root / "scripts" / "models" / "train_msbi.py"),
                "--config",
                str(config),
                "--output-dir",
                str(stage_output),
                "--device",
                args.device,
            ]
            if previous_best is not None:
                command.extend(["--init-checkpoint", str(previous_best)])
            stage_record: dict[str, Any] = {
                "stage": stage.upper(),
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
                "config": str(config),
                "config_sha256": _sha256(config),
                "output_dir": str(stage_output),
                "init_checkpoint": (
                    str(previous_best) if previous_best is not None else None
                ),
                "init_checkpoint_sha256": (
                    _sha256(previous_best) if previous_best is not None else None
                ),
                "command": command,
            }
            manifest["stages"].append(stage_record)
            _write_manifest(manifest_path, manifest)
            subprocess.run(
                command,
                cwd=repo_root,
                check=True,
                env={
                    **os.environ,
                    "NANOLOOP_MSBI_MANIFEST_ROOT": str(manifest_root),
                    "NANOLOOP_MSBI_OUTPUT_ROOT": str(output_root),
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            (
                                str(repo_root),
                                os.environ.get("PYTHONPATH"),
                            ),
                        )
                    ),
                },
            )
            previous_best = stage_output / "best.pt"
            run_manifest = stage_output / "run-manifest.json"
            if not previous_best.is_file() or not run_manifest.is_file():
                raise RuntimeError(f"stage {stage.upper()} did not produce complete evidence")
            run_evidence = json.loads(run_manifest.read_text(encoding="utf-8"))
            if "completed_at" not in run_evidence:
                raise RuntimeError(f"stage {stage.upper()} has no completion marker")
            stage_record.update(
                {
                    "status": "complete",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "best_checkpoint_sha256": _sha256(previous_best),
                    "best_validation_loss": run_evidence["best_validation_loss"],
                }
            )
            _write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(UTC).isoformat()
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    if previous_best is None:
        raise RuntimeError("formal curriculum completed without a checkpoint")
    manifest["final_checkpoint"] = str(previous_best)
    manifest["final_checkpoint_sha256"] = _sha256(previous_best)
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
