"""Download, freeze, verify, and fingerprint the NanoLoop demo embedding snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "13942ee7a1615d20a84b41e800c63775e174f97f"
LICENSE_NAME = "MIT"
LICENSE_URL = "https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/main/LICENSE"


def directory_tree_sha256(root: Path) -> str:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("embedding snapshot contains no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def total_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verified-by", required=True)
    parser.add_argument("--verified-at", required=True, help="YYYY-MM-DD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve(strict=False)
    manifest_path = args.manifest.expanduser().resolve(strict=False)
    try:
        if output.exists():
            if not args.force:
                raise ValueError(f"output directory already exists: {output}")
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        resolved = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(output),
        )
        if Path(resolved).resolve() != output:
            raise RuntimeError(f"snapshot resolved to unexpected directory: {resolved}")
        downloaded_seconds = time.perf_counter() - started

        load_started = time.perf_counter()
        model = SentenceTransformer(str(output), device=args.device, local_files_only=True)
        vector = model.encode(
            ["钙钛矿氧化物中的纳米颗粒析出"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cold_start_seconds = time.perf_counter() - load_started
        if getattr(vector, "shape", None) is None or vector.shape[0] != 1:
            raise RuntimeError("embedding smoke test returned an invalid batch")
        dimension = int(vector.shape[1])
        if dimension != 512:
            raise RuntimeError(f"unexpected embedding dimension: {dimension}")

        tree_sha = directory_tree_sha256(output)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "verified",
            "verified": True,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "dimension": dimension,
            "normalize": True,
            "max_length": int(getattr(model, "max_seq_length", 512)),
            "pooling": "model_defined",
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "local_dir": str(output),
            "tree_sha256": tree_sha,
            "size_bytes": total_size(output),
            "resource": {
                "device": args.device,
                "download_seconds": round(downloaded_seconds, 3),
                "cold_start_seconds": round(cold_start_seconds, 3),
            },
            "verified_by": args.verified_by,
            "verified_at": args.verified_at,
            "notes": (
                "Immutable local snapshot prepared on a network-enabled asset machine; "
                "NanoLoop runtime must load this directory with local_files_only=True."
            ),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
