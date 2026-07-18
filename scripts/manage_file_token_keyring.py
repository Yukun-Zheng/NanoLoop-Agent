#!/usr/bin/env python3
"""Non-interactive operator controls for the protected file-token v2 key ring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.storage.file_token_keyring_store import (  # noqa: E402
    FileTokenV2KeyRingStore,
    FileTokenV2KeyRingStoreError,
)
from app.storage.file_tokens_v2 import FileTokenV2KeyRing  # noqa: E402

_DEFAULT_KEYRING_PATH = Path("data/.file_token_v2_keyring.json")


def _emit(payload: Mapping[str, object], *, error: bool = False) -> None:
    print(
        json.dumps(payload, ensure_ascii=True, sort_keys=True),
        file=sys.stderr if error else sys.stdout,
    )


def _safe_status(keyring: FileTokenV2KeyRing) -> dict[str, object]:
    retained_kids = sorted(keyring.retained_kids)
    return {
        "active_kid": keyring.active_kid,
        "retained_kids": retained_kids,
        "count": len(retained_kids),
    }


def _selected_path(value: str | None) -> Path:
    selected = (
        value
        or os.environ.get("FILE_TOKEN_V2_KEYRING_PATH")
        or os.environ.get("NANOLOOP_FILE_TOKEN_V2_KEYRING_PATH")
    )
    return Path(selected) if selected else _DEFAULT_KEYRING_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or rotate the protected file-token v2 key ring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show non-secret key-ring metadata")
    status.add_argument("--path", help="override FILE_TOKEN_V2_KEYRING_PATH")

    rotate = subparsers.add_parser("rotate", help="atomically activate one new signing key")
    rotate.add_argument("--path", help="override FILE_TOKEN_V2_KEYRING_PATH")
    rotate.add_argument("--new-kid", required=True, help="fresh public key identifier")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = FileTokenV2KeyRingStore(_selected_path(args.path))
        keyring = store.load() if args.command == "status" else store.rotate(new_kid=args.new_kid)
    except FileTokenV2KeyRingStoreError as error:
        _emit(
            {
                "error": {
                    "code": error.code,
                    "message": "file-token v2 key-ring operation failed",
                },
                "status": "error",
            },
            error=True,
        )
        return 1
    except Exception:
        _emit(
            {
                "error": {
                    "code": "unexpected_error",
                    "message": "file-token v2 key-ring operation failed",
                },
                "status": "error",
            },
            error=True,
        )
        return 1

    _emit(_safe_status(keyring))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
