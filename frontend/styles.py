"""Compact, accessible visual system for the NanoLoop science workbench."""

from __future__ import annotations

from typing import Any

PAGE_CONFIG: dict[str, Any] = {
    "page_title": "NanoLoop 科研工作台",
    "page_icon": "◌",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get help": None,
        "Report a bug": None,
        "About": "NanoLoop Agent — 可追溯 SEM 分析与证据驱动科研工作台。",
    },
}

WORKBENCH_CSS = """
<style>
:root {
  --nl-ink: #14232b;
  --nl-muted: #5e7078;
  --nl-line: #d7e0e3;
  --nl-surface: #ffffff;
  --nl-canvas: #f4f7f7;
  --nl-teal: #087f78;
  --nl-teal-dark: #075c59;
  --nl-good: #177245;
  --nl-warn: #9a5b00;
  --nl-bad: #b42318;
  --nl-live: #1f5a96;
}

.stApp { background: var(--nl-canvas); color: var(--nl-ink); }
.block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #102a31; }
[data-testid="stSidebar"] * { color: #edf6f5; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
  color: #14232b;
  background: #ffffff;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #14232b; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }

h1, h2, h3 { letter-spacing: -0.025em; color: var(--nl-ink); }
h1 { font-size: clamp(1.9rem, 2.8vw, 2.75rem); }
h2 { font-size: 1.35rem; margin-top: 1.2rem; }
h3 { font-size: 1.05rem; }
p, li, label { line-height: 1.5; }

.nl-brand { padding: .4rem .1rem 1rem; }
.nl-brand-mark {
  display: inline-grid; place-items: center; width: 2.2rem; height: 2.2rem;
  border: 1px solid rgba(255,255,255,.5); border-radius: 50%;
  font-size: 1.3rem; margin-right: .55rem;
}
.nl-brand strong { font-size: 1.03rem; letter-spacing: .02em; }
.nl-brand small { display: block; margin: .4rem 0 0 2.85rem; color: #b9d0d1; }

.nl-hero {
  background: linear-gradient(120deg, #fafdfe 0%, #edf7f5 100%);
  border: 1px solid var(--nl-line); border-left: 5px solid var(--nl-teal);
  border-radius: 12px; padding: 1.15rem 1.35rem; margin-bottom: 1rem;
}
.nl-eyebrow {
  color: var(--nl-teal-dark); font-size: .73rem; font-weight: 760;
  letter-spacing: .11em; text-transform: uppercase; margin-bottom: .3rem;
}
.nl-hero h1, .nl-hero h2 { margin: 0 0 .35rem; }
.nl-hero p { color: var(--nl-muted); margin: 0; max-width: 78ch; }

.nl-card {
  background: var(--nl-surface); border: 1px solid var(--nl-line);
  border-radius: 10px; padding: .9rem 1rem; height: 100%;
  box-shadow: 0 1px 2px rgba(20,35,43,.035);
}
.nl-card-title { font-weight: 720; margin-bottom: .25rem; }
.nl-card-copy { color: var(--nl-muted); font-size: .9rem; }

.nl-status {
  display: inline-flex; align-items: center; gap: .38rem; border-radius: 999px;
  font-size: .78rem; font-weight: 700; padding: .18rem .55rem;
  border: 1px solid currentColor; white-space: nowrap;
}
.nl-status::before {
  content: ""; width: .48rem; height: .48rem;
  border-radius: 50%; background: currentColor;
}
.nl-status-good { color: var(--nl-good); background: #eef9f2; }
.nl-status-warn { color: var(--nl-warn); background: #fff7e6; }
.nl-status-bad { color: var(--nl-bad); background: #fff0ee; }
.nl-status-live { color: var(--nl-live); background: #edf5ff; }
.nl-status-neutral { color: #5f6f75; background: #f3f6f7; }

.nl-note {
  border-left: 3px solid var(--nl-teal); padding: .55rem .75rem;
  background: #f2faf8; color: #365158; font-size: .9rem; margin: .45rem 0;
}
.nl-citation {
  border: 1px solid var(--nl-line); border-radius: 8px; padding: .7rem .8rem;
  background: #fbfdfd; margin-bottom: .55rem;
}
.nl-citation-id { color: var(--nl-teal-dark); font-weight: 760; font-size: .8rem; }
.nl-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: .82rem; color: #2e4b53; word-break: break-all;
}
.nl-muted { color: var(--nl-muted); }
.nl-divider { height: 1px; background: var(--nl-line); margin: .9rem 0; }

[data-testid="stMetric"] {
  background: var(--nl-surface); border: 1px solid var(--nl-line);
  border-radius: 9px; padding: .65rem .8rem;
}
[data-testid="stMetricLabel"] { color: var(--nl-muted); }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border: 1px solid var(--nl-line); border-radius: 8px; overflow: hidden;
}
.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
  border-radius: 7px; font-weight: 650; min-height: 2.35rem;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--nl-teal); border-color: var(--nl-teal); color: white;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background: var(--nl-teal-dark); border-color: var(--nl-teal-dark);
}
[data-testid="stFileUploaderDropzone"] { background: #f9fbfb; border-color: #b9cacc; }

@media (max-width: 760px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .nl-hero { padding: .95rem 1rem; }
}
</style>
"""


def apply_styles(streamlit: Any) -> None:
    streamlit.markdown(WORKBENCH_CSS, unsafe_allow_html=True)


__all__ = ["PAGE_CONFIG", "WORKBENCH_CSS", "apply_styles"]
