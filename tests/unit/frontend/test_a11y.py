"""Accessibility (a11y) compliance tests for the NanoLoop frontend.

Covers WCAG 2.2 AA concerns:
  - Colour contrast ratios (WCAG 1.4.3 / 1.4.6)
  - Skip-link HTML structure (WCAG 2.4.1)
  - Screen-reader-only text (WCAG 1.3.1)
  - ARIA live regions (WCAG 4.1.3)
  - Focus-visible CSS (WCAG 2.4.7)
  - Reduced-motion support (WCAG 2.3.3)

These tests are pure-Python: they parse CSS values and call a11y helper
functions without launching Streamlit, so they run fast in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from frontend import a11y
from frontend.styles import WORKBENCH_CSS


# ---------------------------------------------------------------------------
# Helpers: WCAG colour-contrast maths
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` or ``#rgb`` into an ``(r, g, b)`` tuple (0-255)."""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Compute the WCAG relative luminance for an sRGB colour."""

    def _channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    rs = _channel(r / 255)
    gs = _channel(g / 255)
    bs = _channel(b / 255)
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Return the WCAG contrast ratio between *fg_hex* and *bg_hex*."""

    r1, g1, b1 = _hex_to_rgb(fg_hex)
    r2, g2, b2 = _hex_to_rgb(bg_hex)
    l1 = _relative_luminance(r1, g1, b1)
    l2 = _relative_luminance(r2, g2, b2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _extract_css_var(name: str, css: str = WORKBENCH_CSS) -> str:
    """Extract a ``--var: #hex`` value from the CSS string."""

    m = re.search(rf"--{name}\s*:\s*(#[0-9a-fA-F]{{3,6}})", css)
    assert m, f"CSS variable --{name} not found in WORKBENCH_CSS"
    return m.group(1)


# ---------------------------------------------------------------------------
# Colour contrast (WCAG 1.4.3 — AA requires 4.5:1 for normal text, 3:1 large)
# ---------------------------------------------------------------------------


class TestColourContrast:
    """Verify that every foreground/background pair meets WCAG AA."""

    @pytest.fixture
    def vars(self) -> dict[str, str]:
        names = [
            "nl-ink", "nl-muted", "nl-line", "nl-surface", "nl-canvas",
            "nl-teal", "nl-teal-dark", "nl-good", "nl-warn", "nl-bad",
            "nl-live",
        ]
        return {n: _extract_css_var(n) for n in names}

    def test_ink_on_canvas(self, vars):
        """Primary text on page background."""
        assert contrast_ratio(vars["nl-ink"], vars["nl-canvas"]) >= 4.5

    def test_ink_on_surface(self, vars):
        """Primary text on card background."""
        assert contrast_ratio(vars["nl-ink"], vars["nl-surface"]) >= 4.5

    def test_muted_on_canvas(self, vars):
        """Muted/caption text on page background."""
        assert contrast_ratio(vars["nl-muted"], vars["nl-canvas"]) >= 4.5

    def test_muted_on_surface(self, vars):
        """Muted/caption text on card background."""
        assert contrast_ratio(vars["nl-muted"], vars["nl-surface"]) >= 4.5

    def test_teal_dark_on_surface(self, vars):
        """Eyebrow / accent text on white cards."""
        assert contrast_ratio(vars["nl-teal-dark"], vars["nl-surface"]) >= 4.5

    def test_good_status_text(self, vars):
        """Status badge text colour on its background."""
        assert contrast_ratio(vars["nl-good"], "#eef9f2") >= 4.5

    def test_warn_status_text(self, vars):
        assert contrast_ratio(vars["nl-warn"], "#fff7e6") >= 4.5

    def test_bad_status_text(self, vars):
        assert contrast_ratio(vars["nl-bad"], "#fff0ee") >= 4.5

    def test_live_status_text(self, vars):
        assert contrast_ratio(vars["nl-live"], "#edf5ff") >= 4.5

    def test_white_on_teal_button(self, vars):
        """Primary button: white text on teal background."""
        assert contrast_ratio("#ffffff", vars["nl-teal"]) >= 4.5

    def test_white_on_teal_dark_button(self, vars):
        """Primary button hover: white text on dark teal."""
        assert contrast_ratio("#ffffff", vars["nl-teal-dark"]) >= 4.5

    def test_teal_on_canvas_large(self, vars):
        """Teal used for large text / UI elements (3:1 threshold)."""
        assert contrast_ratio(vars["nl-teal"], vars["nl-canvas"]) >= 3.0


# ---------------------------------------------------------------------------
# Skip link (WCAG 2.4.1)
# ---------------------------------------------------------------------------


class TestSkipLink:
    def test_renders_anchor_with_class(self):
        """The skip link must be an <a> with the nl-skip-link class."""
        # We capture the markdown output via a fake Streamlit.
        rendered: list[str] = []

        class FakeST:
            def markdown(self, html_str: str, *, unsafe_allow_html: bool = False):
                rendered.append(html_str)

        a11y.render_skip_link(
            FakeST(),
            target_id="nl-main-content",
            label="跳转到主内容",
        )
        assert len(rendered) == 1
        html_str = rendered[0]
        assert 'href="#nl-main-content"' in html_str
        assert "nl-skip-link" in html_str
        assert "跳转到主内容" in html_str

    def test_label_is_escaped(self):
        """User-supplied labels must be HTML-escaped."""
        rendered: list[str] = []

        class FakeST:
            def markdown(self, html_str: str, *, unsafe_allow_html: bool = False):
                rendered.append(html_str)

        a11y.render_skip_link(
            FakeST(),
            target_id="test",
            label='<script>alert(1)</script>',
        )
        assert "<script>" not in rendered[0]
        assert "&lt;script&gt;" in rendered[0]


# ---------------------------------------------------------------------------
# Screen-reader-only text (WCAG 1.3.1)
# ---------------------------------------------------------------------------


class TestSrOnly:
    def test_wraps_text_in_span(self):
        result = a11y.sr_only_text("隐藏但可朗读")
        assert result.startswith('<span class="nl-sr-only">')
        assert "隐藏但可朗读" in result
        assert result.endswith("</span>")

    def test_escapes_html(self):
        result = a11y.sr_only_text("<b>bold</b>")
        assert "<b>" not in result
        assert "&lt;b&gt;" in result


# ---------------------------------------------------------------------------
# ARIA live regions (WCAG 4.1.3)
# ---------------------------------------------------------------------------


class TestStatusAnnouncement:
    def _render(self, **kwargs) -> str:
        rendered: list[str] = []

        class FakeST:
            def markdown(self, html_str: str, *, unsafe_allow_html: bool = False):
                rendered.append(html_str)

        defaults = {
            "role": "status",
            "title": "测试标题",
            "body": "测试内容",
            "tone": "neutral",
        }
        defaults.update(kwargs)
        a11y.render_status_announcement(FakeST(), **defaults)
        return rendered[0]

    def test_has_aria_live(self):
        html_str = self._render()
        assert 'aria-live="polite"' in html_str
        assert 'aria-atomic="true"' in html_str

    def test_alert_role(self):
        html_str = self._render(role="alert")
        assert 'role="alert"' in html_str

    def test_tone_class(self):
        html_str = self._render(tone="bad")
        assert "nl-announcement-bad" in html_str

    def test_actions_rendered(self):
        html_str = self._render(
            actions={"刷新": '<a href="#">刷新</a>'}
        )
        assert "nl-announcement-actions" in html_str
        assert '<a href="#">刷新</a>' in html_str


# ---------------------------------------------------------------------------
# Focus-visible CSS (WCAG 2.4.7)
# ---------------------------------------------------------------------------


class TestFocusVisibleCSS:
    def test_focus_visible_rule_exists(self):
        """The CSS must include a :focus-visible outline rule."""
        assert ":focus-visible" in WORKBENCH_CSS
        assert "outline" in WORKBENCH_CSS

    def test_focus_visible_has_important(self):
        """Focus outline must use !important to survive overrides."""
        # Locate the universal focus-visible rule for interactive elements,
        # not the skip-link or sidebar input rules which come earlier.
        idx = WORKBENCH_CSS.find("input:focus-visible,")
        assert idx != -1, "universal input:focus-visible rule not found"
        block = WORKBENCH_CSS[idx : idx + 400]
        assert "!important" in block

    def test_skip_link_focus_rule(self):
        assert ".nl-skip-link:focus" in WORKBENCH_CSS or ".nl-skip-link:focus-visible" in WORKBENCH_CSS


# ---------------------------------------------------------------------------
# Reduced motion (WCAG 2.3.3)
# ---------------------------------------------------------------------------


class TestReducedMotion:
    def test_prefers_reduced_motion_exists(self):
        assert "prefers-reduced-motion" in WORKBENCH_CSS

    def test_reduces_animation_duration(self):
        idx = WORKBENCH_CSS.find("prefers-reduced-motion")
        block = WORKBENCH_CSS[idx : idx + 300]
        assert "animation-duration" in block
        assert "0.01ms" in block
