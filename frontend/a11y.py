"""Accessibility helpers for the NanoLoop Streamlit workbench.

These helpers keep WCAG 2.2 AA concerns (focus visibility, screen-reader-only
text, skip navigation, status announcements) out of the visual layer so that
pages and components can adopt them consistently.

The functions in this module are intentionally pure HTML/string builders so
they are easy to test outside of Streamlit's runtime.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

__all__ = [
    "aria_describedby",
    "aria_label",
    "focus_hint",
    "render_skip_link",
    "render_sr_only",
    "render_status_announcement",
    "sr_only_text",
]


def sr_only_text(text: str) -> str:
    """Wrap *text* in a visually hidden but screen-reader-announced span."""

    return f'<span class="nl-sr-only">{html.escape(text)}</span>'


def render_sr_only(streamlit: Any, text: str) -> None:
    """Render screen-reader-only text inside a Streamlit container."""

    streamlit.markdown(sr_only_text(text), unsafe_allow_html=True)


def aria_label(text: str) -> str:
    """Return an ``aria-label`` HTML attribute fragment."""

    return f'aria-label="{html.escape(text, quote=True)}"'


def aria_describedby(element_id: str) -> str:
    """Return an ``aria-describedby`` HTML attribute fragment."""

    return f'aria-describedby="{html.escape(element_id, quote=True)}"'


def render_skip_link(
    streamlit: Any,
    *,
    target_id: str,
    label: str = "跳转到主内容",
) -> None:
    """Render a keyboard-reachable skip link at the top of the page.

    The link is visually hidden until focused, satisfying WCAG 2.4.1.
    The *target_id* should reference a container that exists on the page
    (for example the hero section rendered by ``section_header``).
    """

    streamlit.markdown(
        (
            f'<a href="#{html.escape(target_id, quote=True)}" '
            f'class="nl-skip-link">{html.escape(label)}</a>'
        ),
        unsafe_allow_html=True,
    )


def focus_hint(streamlit: Any, message: str) -> None:
    """Render a polite hint that is announced to screen readers.

    Useful after a destructive action completes, to draw attention back to
    the workflow instead of forcing the user to hunt for the next control.
    """

    streamlit.markdown(
        (
            '<div role="status" aria-live="polite" aria-atomic="true" '
            f'class="nl-sr-only">{html.escape(message)}</div>'
        ),
        unsafe_allow_html=True,
    )


def render_status_announcement(
    streamlit: Any,
    *,
    role: str,
    title: str,
    body: str,
    tone: str = "neutral",
    actions: Mapping[str, str] | None = None,
) -> None:
    """Render a structured status panel that is announced to assistive tech.

    Parameters
    ----------
    role:
        ARIA role; usually ``"status"`` (polite) or ``"alert"`` (assertive).
    title:
        Short heading for the announcement.
    body:
        Longer descriptive text.
    tone:
        One of ``"good" | "warn" | "bad" | "live" | "neutral"`` matching the
        design system's status colors.
    actions:
        Optional mapping of ``{label: anchor_or_button_html}``. Callers are
        responsible for producing the HTML; this helper only wraps it in a
        navigable list.
    """

    action_html = ""
    if actions:
        items = "".join(
            f'<li class="nl-announcement-action">{value}</li>'
            for value in actions.values()
        )
        action_html = f'<ul class="nl-announcement-actions">{items}</ul>'

    streamlit.markdown(
        (
            f'<div role="{html.escape(role, quote=True)}" '
            f'class="nl-announcement nl-announcement-{html.escape(tone, quote=True)}" '
            'aria-live="polite" aria-atomic="true">'
            f'<div class="nl-announcement-title">{html.escape(title)}</div>'
            f'<div class="nl-announcement-body">{html.escape(body)}</div>'
            f"{action_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
