"""Headless smoke check for every Streamlit workspace page."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGES = (
    "Connection",
    "Project",
    "ROI & models",
    "Runs & results",
    "Ask NanoLoop",
    "Knowledge base",
)


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    for page in PAGES:
        app = AppTest.from_file(str(app_path), default_timeout=10)
        app.run()
        if page != "Connection":
            app.session_state["navigation"] = page
            app.run()
        if app.exception:
            raise RuntimeError(f"Streamlit page {page!r} raised: {app.exception}")
    print(f"Streamlit AppTest passed for {len(PAGES)} pages.")


if __name__ == "__main__":
    main()
