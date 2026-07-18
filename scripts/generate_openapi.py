"""Generate the checked-in OpenAPI v1 handoff artifact deterministically."""

from __future__ import annotations

import json
from pathlib import Path

from app.main import create_app


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs" / "api" / "openapi-v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = create_app().openapi()
    destination.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination.relative_to(root)}")


if __name__ == "__main__":
    main()
