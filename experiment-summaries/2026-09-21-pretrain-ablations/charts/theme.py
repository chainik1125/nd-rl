"""Shared dark Altair theme + helpers for the blog's charts.

Every chart is Altair, rendered to static PNG + SVG via vl-convert at build time,
themed to match blog.css (--bg #0b0b0b, light text).

FONT SIZES ARE PER-CHART, and this is the important, non-obvious bit. A chart is
authored at some Vega `width` (px), then embedded as an <img> that the browser
scales to fit the ~COLUMN-px content column. So the text a reader actually sees:

    on-screen px = font_px * COLUMN / chart_width

A narrow chart (e.g. a 450px bar chart) is scaled *up* to the column, making its
text look big; a 720px scatter is scaled up less, so the same font looks small.
To make every chart's text identical on screen, scale the fonts with the chart's
width via fonts(width) and pass that width to save(). The reference is the
Quotidian-water bar chart (width 450) — the size we want everywhere.

DON'T hardcode font sizes in a chart, and don't reach for matplotlib/seaborn
(their inches->raster->squish model is what made text tiny in the first place).

Run a chart script from posts/plastic-straws/ so the charts/<name> paths and the
`from theme import ...` sibling import both resolve:
    python3 charts/table_bars.py
"""
import os

import altair as alt

# Two palettes. Dark is the site default (blog.css is dark); light is for
# cross-posts to white-background platforms (Substack, LessWrong). Select with
# CHART_THEME=light; the dark branch is byte-identical to before so the committed
# site charts never change. WHISK is the range-whisker rule color (only the
# bar_buy_absolute flight range uses it) — the old #cccccc vanishes on white.
if os.environ.get("CHART_THEME") == "light":
    BG, FG, MUTED, GRID, WHISK = "#ffffff", "#1a1d24", "#6b7280", "#e5e7eb", "#9aa0ab"
else:
    BG, FG, MUTED, GRID, WHISK = "#1a1d24", "#ccd0d8", "#8b919d", "#2e333d", "#cccccc"
FONT = "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"

# Category palettes carried over from the old charts so colors stay familiar.
CAT = {"food": "#c1432e", "energy": "#2e7dc1", "goods": "#7a5fb0",
       "digital": "#3f8f5b", "travel": "#d98a1f"}
HUE = {"carbon": "#c1432e", "water": "#2e7dc1", "recycle": "#3f8f5b",
       "dollar": "#7a5fb0"}
# Lab brand hues (eval-awareness repo plots/style.py). OpenAI's black is lifted to a mid grey that
# stays distinct from FG (the pooled line / text color); Kimi's gold is left at full brightness.
LAB = {
    "Anthropic": "#d97757", "OpenAI": "#9aa4b5", "Google": "#34a853", "Alibaba": "#c026d3",
    "DeepSeek": "#4d6bfe", "Moonshot": "#fec230", "Z.ai": "#dc2626", "MiniMax": "#22b8d8",
}
# Delta superscripts on the heatmaps: red = awareness went up, green = down.
UP, DOWN = "#e0524f", "#3fbf7f"

# --- font scaling ----------------------------------------------------------
COLUMN = 752        # blog content column (blog.css: main 50rem max, ~24px pad/side)
_REF_WIDTH = 450    # the Quotidian-water bar chart — the on-screen size we match
_REF = {"title": 17, "subtitle": 12, "axis_title": 14, "axis_label": 13,
        "label": 11, "legend": 12}


def fonts(width):
    """Per-chart font sizes so on-screen text matches the reference at any width.
    Use fonts(width)['label'] for in-chart mark_text; pass the same width to
    save() for the axis / title / legend chrome. Never hardcode font sizes."""
    k = width / _REF_WIDTH
    return {name: round(sz * k, 1) for name, sz in _REF.items()}


def fmt(v):
    """Compact number label: 1,900 / 0.093 / 0.0001."""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.4g}"


def save(chart, name, width):
    """Apply the dark theme + width-scaled fonts; write <name>.svg + .png (@2x).
    `width` is the chart's Vega width (for a vconcat, the panel width)."""
    f = fonts(width)
    styled = (
        chart
        .configure(background=BG, font=FONT)
        .configure_view(stroke=None)
        .configure_axis(labelColor=FG, titleColor=FG, gridColor=GRID,
                        domainColor=MUTED, tickColor=MUTED,
                        labelFontSize=f["axis_label"], titleFontSize=f["axis_title"])
        .configure_title(color=FG, subtitleColor=MUTED, fontSize=f["title"],
                         subtitleFontSize=f["subtitle"], anchor="start")
        .configure_legend(labelColor=FG, titleColor=FG,
                          labelFontSize=f["legend"], titleFontSize=f["legend"])
    )
    # Light theme writes to charts/light/ so it sits alongside — never over —
    # the dark charts the site serves.
    sub = "charts/light" if os.environ.get("CHART_THEME") == "light" else "charts"
    os.makedirs(sub, exist_ok=True)
    styled.save(f"{sub}/{name}.svg")
    styled.save(f"{sub}/{name}.png", scale_factor=2.0)
    print(f"wrote {sub}/{name}.png + .svg")
