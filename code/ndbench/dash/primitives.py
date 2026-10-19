"""Static-HTML chart/page primitives for the ndbench dashboard (plan §3 WP6 row: "CSS tokens +
primitives ported from `rl_dash.py`").

Ported from `/scratch/cowifornia/nd-rl/ab/dash/nd-rl/code/tools/rl_dash.py` (commit `cf26d14`,
branch `leon/rl-dashboards`; not on this branch, plan §0), which is not itself part of ndbench
(there is nothing to vendor byte-identically -- the plan asks WP6 to *port the look and
primitives*, not reuse the module). `esc`/`fmt`/`tick_fmt`/`nice_ticks`/`Series`/`_legend`/
`_table`/`figure`/`missing_fig`/`line_chart`/`heatmap`/`columns`/`table_fig`/`page`/`CSS`/`JS` are
carried over near-verbatim (renamed/trimmed where `rl_dash.py`'s version was specific to RL-run
metrics.jsonl records, which ndbench has no equivalent of). Added for this WP: `Series.band` (a
shaded confidence band under a line, for pass@k bootstrap CIs), `Bar`/`bar_chart` (categorical
values with optional CI whiskers, for ratios/coefficients/slopes), and `venn2` (a two-set solved-set
Venn, plan §2 B8).

Every chart is inline SVG with the data embedded as a `<script type="application/json">` blob (no
network requests, no external URLs); `JS` wires up a shared hover tooltip and crosshair. Colours are
CSS custom properties on `:root`, redefined for dark mode; `body` sets an explicit background.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

Rec = dict[str, Any]

# ---------------------------------------------------------------- formatting


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) else None


def fmt(v: Any, kind: str = "g") -> str:
    x = _num(v)
    if x is None or math.isnan(x):
        return "missing"
    if kind == "pct":
        # two decimals below 1% so e.g. 0.14% vs 0.15% doesn't print as "0.1% -> 0.2%"
        return f"{100 * x:.2f}%" if 0 < abs(100 * x) < 1 else f"{100 * x:.1f}%"
    if kind == "int":
        return f"{x:,.0f}"
    if kind == "sci":
        return f"{x:.3g}"
    if kind == "secs":
        return f"{x:.1f} s"
    if kind == "f3":
        return f"{x:.3f}"
    return f"{x:.4g}"


def _decimals_needed(step: float, max_d: int = 6) -> int:
    """The smallest number of decimal digits that represents `step` exactly (up to float noise):
    a `log10`-floor heuristic alone picks too few for steps like `0.25` (`log10(0.25)` floors to
    `-1`, i.e. 1 decimal place, which rounds `0.25`/`0.75` to `0.2`/`0.8` under round-half-to-even
    -- confirmed on the real glue R-ratio chart's y-axis). `0.1`/`0.2`/`0.5`/`1`/`2`/`2.5`/`5`/`10`
    are `nice_ticks`'s only possible step *shapes* (any power of ten times one of those), so this
    just checks increasing decimal counts until rounding to that many places is a no-op.
    """
    if step <= 0:
        return 0
    for d in range(max_d + 1):
        if abs(round(step, d) - step) < 1e-9 * max(1.0, abs(step)):
            return d
    return max_d


def tick_fmt(t: float, step: float, kind: str) -> str:
    """Tick label with as many decimals as the tick step needs."""
    if kind == "pct":
        d = max(0, -math.floor(math.log10(step * 100) + 1e-9)) if step > 0 else 0
        return f"{100 * t:.{d}f}%"
    if kind == "sci" and (abs(t) >= 1e5 or 0 < abs(t) < 1e-3):
        return f"{t:.2g}"
    d = _decimals_needed(step)
    return f"{t:,.{d}f}"


def nice_ticks(lo: float, hi: float, n: int = 5, integer: bool = False) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    if integer:
        step = max(1.0, round(step))
    t0 = math.floor(lo / step) * step
    ticks = []
    t = t0
    while t <= hi + step * 0.5:
        ticks.append(round(t, 12))
        t += step
    return ticks


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- charts (inline SVG)

W, H = 640, 250
WW = 1280  # viewBox width of a full-row chart (text keeps its size)
ML, MR, MT, MB = 58, 14, 10, 38
_ids: Counter[str] = Counter()


def _cid(prefix: str) -> str:
    _ids[prefix] += 1
    return f"{prefix}{_ids[prefix]}"


@dataclass
class Series:
    name: str
    pts: list[tuple[float, float]]
    color: str  # a CSS value, e.g. var(--s1)
    muted: bool = False
    #: an optional shaded confidence band, `(x, lo, hi)` per point (same `x`s as `pts` ideally).
    band: list[tuple[float, float, float]] | None = None


@dataclass
class Bar:
    name: str
    value: float
    lo: float | None = None
    hi: float | None = None
    color: str = "var(--s1)"


def _legend(series: list[Series], kind: str = "line") -> str:
    if len(series) < 2:
        return ""
    items = []
    for s in series:
        if kind == "line":
            # a literal dashed vs. solid line swatch (not just a colour block) so a legend with
            # both solid and dashed series of the same colour palette (e.g. the probes chart's
            # "accuracy" vs. "R^2" series, same per-model colours) is still legible without
            # relying on the reader to guess which line is which from the chart alone.
            dash = ' stroke-dasharray="6 4"' if s.muted else ""
            key = (
                '<svg width="16" height="10" class="lkeyline" aria-hidden="true">'
                f'<line x1="1" y1="5" x2="15" y2="5" stroke="{s.color}" stroke-width="2"{dash}/>'
                "</svg>"
            )
        else:
            key = f'<span class="bkey" style="background:{s.color}"></span>'
        items.append(f'<span class="li">{key}{esc(s.name)}</span>')
    return f'<div class="legend">{"".join(items)}</div>'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        '<details><summary>Table</summary><div class="tw">'
        f"<table><tr>{th}</tr>{body}</table></div></details>"
    )


def figure(title: str, sub: str, body: str, note: str = "", wide: bool = False) -> str:
    n = f'<p class="note">{note}</p>' if note else ""
    cls = "card wide" if wide else "card"
    cap = f'<figcaption><h3>{esc(title)}</h3><p class="sub">{esc(sub)}</p></figcaption>'
    return f'<figure class="{cls}">{cap}{body}{n}</figure>'


def missing_fig(title: str, what: str) -> str:
    return figure(
        title,
        what,
        '<div class="missing">no data yet</div>',
    )


def line_table_rows(
    series: list[Series], *, x_label: str = "x", x_fmt: str = "g", y_fmt: str = "g"
) -> tuple[list[str], list[list[str]]]:
    """`(headers, rows)` of `line_chart`'s data table: one row per distinct x over every
    non-empty series, one column per series (a series with no point at that x shows "missing").
    Shared with the summary page (`dash/summary.py`), which draws the chart as a paper figure but
    keeps this exact table."""
    series = [s for s in series if s.pts]
    xs = sorted({x for s in series for x, _ in s.pts})
    return (
        [x_label, *[s.name for s in series]],
        [[fmt(x, x_fmt), *[fmt(dict(s.pts).get(x), y_fmt) for s in series]] for x in xs],
    )


def line_table(
    series: list[Series], *, x_label: str = "x", x_fmt: str = "g", y_fmt: str = "g"
) -> str:
    return _table(*line_table_rows(series, x_label=x_label, x_fmt=x_fmt, y_fmt=y_fmt))


def line_chart(
    title: str,
    sub: str,
    series: list[Series],
    *,
    x_label: str = "x",
    y_fmt: str = "g",
    y_min: float | None = None,
    y_max: float | None = None,
    note: str = "",
    wide: bool = False,
    x_fmt: str = "g",
    log_x: bool = False,
    x_tick_labels: list[str] | None = None,
) -> str:
    """`x_tick_labels[i]` (when given) replaces the x-tick label at `x == i` (e.g. a layer index)
    with a real category name, and ticks are placed at every integer in range instead of
    `nice_ticks`'s own step size -- without this, a 5-point integer x-axis (plan: "layer 0..4")
    gets fractional ticks like `0.5`/`1.5` from `nice_ticks` picking a step of `0.5`, which label
    no real category (confirmed on the real probes chart)."""
    empty = [s for s in series if not s.pts]
    series = [s for s in series if s.pts]
    if not series:
        return missing_fig(title, sub)
    if empty:
        miss = ", ".join(s.name for s in empty)
        note = f"Missing (not logged): {esc(miss)}." + (f" {note}" if note else "")
    fig_w = WW if wide else W
    xs = sorted({x for s in series for x, _ in s.pts})

    def xform(x: float) -> float:
        return math.log2(max(x, 1e-9)) if log_x else x

    ys = [y for s in series for _, y in s.pts]
    for s in series:
        if s.band:
            ys.extend(b[1] for b in s.band)
            ys.extend(b[2] for b in s.band)
    lo = min(ys) if y_min is None else y_min
    hi = max(ys) if y_max is None else y_max
    if y_min is None and lo > 0 and lo < 0.35 * hi:
        lo = 0.0
    yt = nice_ticks(lo, hi)
    lo, hi = min(yt[0], lo), max(yt[-1], hi)
    xs_t = [xform(x) for x in xs]
    x0, x1 = xs_t[0], (xs_t[-1] if xs_t[-1] > xs_t[0] else xs_t[0] + 1)
    pw, ph = fig_w - ML - MR, H - MT - MB

    def px(x: float) -> float:
        return ML + (xform(x) - x0) / (x1 - x0) * pw

    def py(y: float) -> float:
        return MT + ph - (y - lo) / ((hi - lo) or 1) * ph

    g = []
    ystep = yt[1] - yt[0] if len(yt) > 1 else 1.0
    for t in yt:
        g.append(
            f'<line class="grid" x1="{ML}" x2="{fig_w - MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>'
        )
        g.append(
            f'<text class="tick" x="{ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">'
            f"{esc(tick_fmt(t, ystep, y_fmt))}</text>"
        )
    if x_tick_labels is not None:
        for i, label in enumerate(x_tick_labels):
            t = xform(float(i))
            if x0 <= t <= x1:
                g.append(
                    f'<text class="tick" x="{ML + (t - x0) / (x1 - x0) * pw:.1f}" '
                    f'y="{H - MB + 16}" text-anchor="middle">{esc(label)}</text>'
                )
    else:
        for t in nice_ticks(x0, x1, 12 if wide else 7, integer=(x_fmt == "int" and not log_x)):
            if x0 <= t <= x1:
                label = fmt(2**t, x_fmt) if log_x else fmt(t, x_fmt)
                g.append(
                    f'<text class="tick" x="{ML + (t - x0) / (x1 - x0) * pw:.1f}" '
                    f'y="{H - MB + 16}" text-anchor="middle">{esc(label)}</text>'
                )
    g.append(f'<line class="axis" x1="{ML}" x2="{fig_w - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    g.append(
        f'<text class="alabel" x="{ML + pw / 2}" y="{H - 4}" text-anchor="middle">'
        f"{esc(x_label)}</text>"
    )
    marks = []
    for s in series:
        if s.band:
            band_sorted = sorted(s.band, key=lambda b: b[0])
            top = " ".join(f"{px(x):.1f},{py(hi_):.1f}" for x, _lo, hi_ in band_sorted)
            bot = " ".join(f"{px(x):.1f},{py(lo_):.1f}" for x, lo_, _hi in reversed(band_sorted))
            marks.append(f'<polygon class="band" points="{top} {bot}" style="fill:{s.color}"/>')
    for s in series:
        segs: list[list[tuple[float, float]]] = [[]]
        for x, y in s.pts:
            segs[-1].append((x, y))
        cls = "ln muted" if s.muted else "ln"
        for seg in segs:
            if not seg:
                continue
            if len(seg) == 1:
                marks.append(
                    f'<circle class="dot" cx="{px(seg[0][0]):.1f}" cy="{py(seg[0][1]):.1f}" r="4" '
                    f'style="fill:{s.color}"/>'
                )
            else:
                d = " ".join(
                    f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                    for i, (x, y) in enumerate(seg)
                )
                marks.append(f'<path class="{cls}" d="{d}" style="stroke:{s.color}"/>')
    data = {
        "xl": x_label,
        "xs": [fmt(x, x_fmt) for x in xs],
        "px": [round(px(x), 1) for x in xs],
        "s": [
            {"n": s.name, "c": s.color, "v": [fmt(dict(s.pts).get(x), y_fmt) for x in xs]}
            for s in series
        ],
    }
    svg = (
        f'<svg viewBox="0 0 {fig_w} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}"
        f'<line class="cross" x1="0" x2="0" y1="{MT}" y2="{MT + ph}" visibility="hidden"/>'
        f'<rect class="hit" x="{ML}" y="{MT}" width="{pw}" height="{ph}"/></svg>'
    )
    blob = json.dumps(data).replace("</", "<\\/")
    js = f'<script type="application/json">{blob}</script>'
    table = line_table(series, x_label=x_label, x_fmt=x_fmt, y_fmt=y_fmt)
    leg = _legend(series + [Series(f"{s.name} (missing)", [], "transparent") for s in empty])
    body = f'{leg}<div class="plot" data-chart="line">{svg}{js}</div>{table}'
    return figure(title, sub, body, note, wide)


def heatmap(
    title: str,
    sub: str,
    grid: dict[Any, dict[str, float]],
    *,
    y_label: str = "L",
    x_label: str = "x",
    v_fmt: str = "pct",
    v_max: float | None = None,
    counts: dict[Any, dict[str, int]] | None = None,
    note: str = "",
    wide: bool = False,
) -> str:
    """`grid[col][row_label] = value`. Columns sort by their own value (numeric if possible); rows
    sort by float(row_label) (short at the bottom). `counts[col][row_label] = n_items`, when
    given, is shown alongside the value in both the hover tooltip and the fallback table (plan §2
    B3's skill x L heat-map: "pass@1 and pass@32 ... with cell counts")."""
    grid = {c: v for c, v in grid.items() if v}
    if not grid:
        return missing_fig(title, sub)
    fig_w = WW if wide else W
    cols = sorted(grid, key=lambda c: (isinstance(c, str), c))
    rows = sorted({r for v in grid.values() for r in v}, key=lambda s: float(s))
    vals = [v for g in grid.values() for v in g.values()]
    vmax = v_max if v_max is not None else (max(vals) if vals else 1.0) or 1.0
    pw, ph = fig_w - ML - MR, H - MT - MB
    cw, chh = pw / len(cols), ph / len(rows)

    def _n_at(col: Any, row: str) -> int | None:
        return counts.get(col, {}).get(row) if counts is not None else None

    cells = []
    for i, col in enumerate(cols):
        for j, row in enumerate(rows):
            v = grid[col].get(row)
            x, y = ML + i * cw, MT + ph - (j + 1) * chh
            cls = (
                "na" if v is None else f"q{min(8, max(0, int((v / vmax if vmax else 0) * 8.999)))}"
            )
            tip_rows = [[f"{col}, {y_label} {row}", "", None], ["", fmt(v, v_fmt), None]]
            n = _n_at(col, row)
            if n is not None:
                tip_rows.append(["n items", str(n), None])
            tip = json.dumps(tip_rows)
            cells.append(
                f'<rect class="{cls}" x="{x:.2f}" y="{y:.2f}" width="{max(cw - 1, 0.6):.2f}" '
                f'height="{chh - 2:.2f}" data-tip="{esc(tip)}"/>'
            )
    g = [
        f'<text class="tick" x="{ML - 6}" y="{MT + ph - (j + 0.5) * chh + 4:.1f}" '
        f'text-anchor="end">{esc(row)}</text>'
        for j, row in enumerate(rows)
    ]
    step = max(1, len(cols) // (12 if wide else 7))
    for i in range(0, len(cols), step):
        g.append(
            f'<text class="tick" x="{ML + (i + 0.5) * cw:.1f}" y="{H - MB + 16}" '
            f'text-anchor="middle">{esc(cols[i])}</text>'
        )
    g.append(
        f'<text class="alabel" x="{ML + pw / 2}" y="{H - 4}" text-anchor="middle">'
        f"{esc(x_label)}</text>"
    )
    g.append(
        f'<text class="alabel" x="12" y="{MT + ph / 2}" transform="rotate(-90 12 {MT + ph / 2})" '
        f'text-anchor="middle">{esc(y_label)}</text>'
    )
    scale = (
        '<div class="scale"><span>0</span>'
        + "".join(f'<i class="q{k}"></i>' for k in range(9))
        + f"<span>{esc(fmt(vmax, v_fmt))}</span>"
        + '<span class="na-key"><i class="na"></i> no data</span></div>'
    )
    svg = (
        f'<svg viewBox="0 0 {fig_w} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(cells)}</svg>"
    )

    def _cell_text(col: Any, row: str) -> str:
        v = fmt(grid[col].get(row), v_fmt)
        n = _n_at(col, row)
        return f"{v} (n={n})" if n is not None else v

    table = _table(
        [x_label, *[f"{y_label} {r}" for r in rows]],
        [[str(c), *[_cell_text(c, r) for r in rows]] for c in cols],
    )
    return figure(title, sub, f'{scale}<div class="plot">{svg}</div>{table}', note, wide)


def columns(
    title: str,
    sub: str,
    xs: list[str],
    stacks: list[Series],
    *,
    y_fmt: str = "g",
    x_label: str = "x",
    note: str = "",
    wide: bool = False,
) -> str:
    """Stacked columns: each `Series.pts[i] = (i, value)`."""
    stacks = [s for s in stacks if any(v for _, v in s.pts)]
    if not stacks or not xs:
        return missing_fig(title, sub)
    fig_w = WW if wide else W
    vals = [{int(i): v for i, v in s.pts} for s in stacks]
    tot = [sum(v.get(i, 0.0) for v in vals) for i in range(len(xs))]
    hi = max(tot) or 1.0
    yt = nice_ticks(0, hi)
    hi = max(yt[-1], hi)
    pw, ph = fig_w - ML - MR, H - MT - MB
    slot = pw / len(xs)
    bw = min(24.0, max(slot - 2, 1.0))

    def py(y: float) -> float:
        return MT + ph - y / hi * ph

    g = []
    ystep = yt[1] - yt[0] if len(yt) > 1 else 1.0
    for t in yt:
        g.append(
            f'<line class="grid" x1="{ML}" x2="{fig_w - MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>'
        )
        g.append(
            f'<text class="tick" x="{ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">'
            f"{esc(tick_fmt(t, ystep, y_fmt))}</text>"
        )
    step = max(1, len(xs) // 8)
    for i in range(0, len(xs), step):
        g.append(
            f'<text class="tick" x="{ML + (i + 0.5) * slot:.1f}" y="{H - MB + 16}" '
            f'text-anchor="middle">{esc(xs[i])}</text>'
        )
    g.append(
        f'<text class="alabel" x="{ML + pw / 2}" y="{H - 4}" text-anchor="middle">'
        f"{esc(x_label)}</text>"
    )
    marks = []
    for i in range(len(xs)):
        base = 0.0
        x = ML + (i + 0.5) * slot - bw / 2
        tip = [[f"{x_label} {xs[i]}", "", None]] + [
            [s.name, fmt(v.get(i), y_fmt), s.color] for s, v in zip(stacks, vals, strict=True)
        ]
        for s, v in zip(stacks, vals, strict=True):
            hgt = v.get(i, 0.0)
            if not hgt:
                continue
            y0, y1 = py(base), py(base + hgt)
            base += hgt
            marks.append(
                f'<rect class="seg" x="{x:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                f'height="{max(y0 - y1, 0.5):.1f}" style="fill:{s.color}" '
                f'data-tip="{esc(json.dumps(tip))}"/>'
            )
    g.append(f'<line class="axis" x1="{ML}" x2="{fig_w - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    svg = (
        f'<svg viewBox="0 0 {fig_w} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}</svg>"
    )
    table = _table(
        [x_label, *[s.name for s in stacks]],
        [[xs[i], *[fmt(v.get(i), y_fmt) for v in vals]] for i in range(len(xs))],
    )
    return figure(
        title, sub, f'{_legend(stacks, "bar")}<div class="plot">{svg}</div>{table}', note, wide
    )


def bar_chart(
    title: str,
    sub: str,
    bars: list[Bar],
    *,
    y_fmt: str = "g",
    y_min: float | None = None,
    y_max: float | None = None,
    note: str = "",
    wide: bool = False,
) -> str:
    """One bar per `Bar`, with an optional `(lo, hi)` whisker; a category axis of bar names."""
    bars = [b for b in bars if not (isinstance(b.value, float) and math.isnan(b.value))]
    if not bars:
        return missing_fig(title, sub)
    fig_w = WW if wide else W
    vals = [b.value for b in bars]
    for b in bars:
        if b.lo is not None:
            vals.append(b.lo)
        if b.hi is not None:
            vals.append(b.hi)
    lo_v = min(vals) if y_min is None else y_min
    hi_v = max(vals) if y_max is None else y_max
    if y_min is None and lo_v > 0:
        lo_v = 0.0
    if y_max is None and hi_v < 0:
        hi_v = 0.0
    yt = nice_ticks(lo_v, hi_v)
    lo_v, hi_v = min(yt[0], lo_v), max(yt[-1], hi_v)
    pw, ph = fig_w - ML - MR, H - MT - MB
    slot = pw / len(bars)
    bw = min(48.0, max(slot - 10, 6.0))

    def py(y: float) -> float:
        return MT + ph - (y - lo_v) / ((hi_v - lo_v) or 1) * ph

    g = []
    ystep = yt[1] - yt[0] if len(yt) > 1 else 1.0
    for t in yt:
        g.append(
            f'<line class="grid" x1="{ML}" x2="{fig_w - MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>'
        )
        g.append(
            f'<text class="tick" x="{ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">'
            f"{esc(tick_fmt(t, ystep, y_fmt))}</text>"
        )
    base_y = py(0.0) if lo_v <= 0 <= hi_v else py(lo_v)
    marks = []
    for i, b in enumerate(bars):
        x = ML + (i + 0.5) * slot - bw / 2
        y1 = py(b.value)
        top, bottom = (y1, base_y) if y1 < base_y else (base_y, y1)
        tip = [[b.name, "", None], ["value", fmt(b.value, y_fmt), b.color]]
        if b.lo is not None and b.hi is not None:
            tip.append(["95% CI", f"{fmt(b.lo, y_fmt)} .. {fmt(b.hi, y_fmt)}", None])
        marks.append(
            f'<rect class="seg" x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" '
            f'height="{max(bottom - top, 0.5):.1f}" style="fill:{b.color}" '
            f'data-tip="{esc(json.dumps(tip))}"/>'
        )
        if b.lo is not None and b.hi is not None:
            cx = x + bw / 2
            y_lo, y_hi = py(b.lo), py(b.hi)
            marks.append(
                f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" '
                f'y1="{y_hi:.1f}" y2="{y_lo:.1f}"/>'
            )
            for y_w in (y_lo, y_hi):
                marks.append(
                    f'<line class="whisker" x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" '
                    f'y1="{y_w:.1f}" y2="{y_w:.1f}"/>'
                )
        g.append(
            f'<text class="tick" x="{ML + (i + 0.5) * slot:.1f}" y="{H - MB + 16}" '
            f'text-anchor="middle">{esc(b.name)}</text>'
        )
    g.append(
        f'<line class="axis" x1="{ML}" x2="{fig_w - MR}" y1="{base_y:.1f}" y2="{base_y:.1f}"/>'
    )
    svg = (
        f'<svg viewBox="0 0 {fig_w} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}</svg>"
    )
    table = bar_table(bars, y_fmt=y_fmt)
    return figure(title, sub, f'<div class="plot">{svg}</div>{table}', note, wide)


def bar_table_rows(bars: list[Bar], *, y_fmt: str = "g") -> tuple[list[str], list[list[str]]]:
    """`(headers, rows)` of `bar_chart`'s data table (NaN-valued bars dropped, as in the chart)."""
    bars = [b for b in bars if not (isinstance(b.value, float) and math.isnan(b.value))]
    return (
        ["name", "value", "ci lo", "ci hi"],
        [[b.name, fmt(b.value, y_fmt), fmt(b.lo, y_fmt), fmt(b.hi, y_fmt)] for b in bars],
    )


def bar_table(bars: list[Bar], *, y_fmt: str = "g") -> str:
    return _table(*bar_table_rows(bars, y_fmt=y_fmt))


def hbar_chart(
    title: str,
    sub: str,
    bars: list[Bar],
    *,
    x_fmt: str = "g",
    x_min: float | None = None,
    x_max: float | None = None,
    log_x: bool = False,
    note: str = "",
    wide: bool = False,
) -> str:
    """One horizontal bar per `Bar` (value along the x-axis, category labels down the left edge),
    with an optional `(lo, hi)` whisker -- the summary dashboard's EDL-per-class chart (many short
    category labels, a value that can span orders of magnitude) reads better this way than
    `bar_chart`'s vertical categorical columns. `log_x` mirrors `line_chart`'s own log-x convention
    (log2 internally, tick labels show the real value)."""
    bars = [b for b in bars if not (isinstance(b.value, float) and math.isnan(b.value))]
    if not bars:
        return missing_fig(title, sub)
    fig_w = WW if wide else W
    label_w = min(220, max(90, fig_w // 4))
    row_h = 24
    top, bot = 10, 30
    total_h = top + row_h * len(bars)
    h = total_h + bot
    pw = fig_w - label_w - MR - 8

    def xform(x: float) -> float:
        return math.log2(max(x, 1e-9)) if log_x else x

    vals = [b.value for b in bars]
    for b in bars:
        if b.lo is not None:
            vals.append(b.lo)
        if b.hi is not None:
            vals.append(b.hi)
    lo_v = (
        x_min
        if x_min is not None
        else (min((v for v in vals if v > 0), default=0.0) if log_x else 0.0)
    )
    hi_v = x_max if x_max is not None else max(vals)
    lo_t, hi_t = xform(lo_v), xform(hi_v if hi_v > lo_v else lo_v + 1.0)

    def px(x: float) -> float:
        return label_w + (xform(x) - lo_t) / ((hi_t - lo_t) or 1) * pw

    g = []
    for t in nice_ticks(lo_t, hi_t, 6):
        xpix = label_w + (t - lo_t) / ((hi_t - lo_t) or 1) * pw
        label = fmt(2**t, x_fmt) if log_x else fmt(t, x_fmt)
        g.append(f'<line class="grid" x1="{xpix:.1f}" x2="{xpix:.1f}" y1="{top}" y2="{total_h}"/>')
        g.append(
            f'<text class="tick" x="{xpix:.1f}" y="{total_h + 16}" text-anchor="middle">'
            f"{esc(label)}</text>"
        )
    marks = []
    for i, b in enumerate(bars):
        y = top + i * row_h + row_h / 2
        x0, x1 = px(lo_v), px(b.value)
        left, right = (x0, x1) if x1 > x0 else (x1, x0)
        tip = [[b.name, "", None], ["value", fmt(b.value, x_fmt), b.color]]
        if b.lo is not None and b.hi is not None:
            tip.append(["95% CI", f"{fmt(b.lo, x_fmt)} .. {fmt(b.hi, x_fmt)}", None])
        w = max(right - left, 1.0)
        marks.append(
            f'<rect class="seg" x="{left:.1f}" y="{y - 7:.1f}" width="{w:.1f}" height="14" '
            f'style="fill:{b.color}" data-tip="{esc(json.dumps(tip))}"/>'
        )
        if b.lo is not None and b.hi is not None:
            x_lo, x_hi = px(b.lo), px(b.hi)
            marks.append(
                f'<line class="whisker" x1="{x_lo:.1f}" x2="{x_hi:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
            )
            for xw in (x_lo, x_hi):
                marks.append(
                    f'<line class="whisker" x1="{xw:.1f}" x2="{xw:.1f}" '
                    f'y1="{y - 5:.1f}" y2="{y + 5:.1f}"/>'
                )
        g.append(
            f'<text class="tick" x="{label_w - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">'
            f"{esc(b.name)}</text>"
        )
    g.append(f'<line class="axis" x1="{label_w}" x2="{label_w}" y1="{top}" y2="{total_h}"/>')
    svg = (
        f'<svg viewBox="0 0 {fig_w} {h}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}</svg>"
    )
    table = _table(
        ["name", "value", "ci lo", "ci hi"],
        [[b.name, fmt(b.value, x_fmt), fmt(b.lo, x_fmt), fmt(b.hi, x_fmt)] for b in bars],
    )
    return figure(title, sub, f'<div class="plot">{svg}</div>{table}', note, wide)


@dataclass
class DotRow:
    label: str
    #: `(series_name, value, color, censored)`. `value=None` means this series' own `none_label`
    #: applies at this row (plan: "families where nothing reaches 50% ... are marked 'none'" --
    #: generalised to any dot_plot caller's own "did not reach X" label, e.g. "fails at smallest
    #: size"). `censored=True` draws a "this value or more" marker next to an otherwise-ordinary
    #: dot (plan: a horizon that never dropped below threshold within the tested range is a lower
    #: bound on the true horizon, not the horizon itself).
    points: list[tuple[str, float | None, str, bool]]


#: Vertical pixel offsets for up to this many same-row dots, evenly spread so same-coloured runs
#: of base/rl/armC (or similar) don't render exactly on top of each other at the same x (plan:
#: "Dots overlap: base and rl are hidden under armC. Offset each model vertically within its row").
_DOT_ROW_SPREAD = 6.0


def _row_offsets(n: int) -> list[float]:
    if n <= 1:
        return [0.0] * n
    step = 2 * _DOT_ROW_SPREAD / (n - 1)
    return [-_DOT_ROW_SPREAD + i * step for i in range(n)]


def _dot_legend_items(rows: list[DotRow]) -> list[tuple[str, str]]:
    legend_items: list[tuple[str, str]] = []
    for row in rows:
        for n, v, c, _z in row.points:
            if v is not None and (n, c) not in legend_items:
                legend_items.append((n, c))
    return legend_items


def dot_table_rows(
    rows: list[DotRow], *, x_fmt: str = "g", log_x: bool = False, none_label: str = "none"
) -> tuple[list[str], list[list[str]]]:
    """`(headers, rows)` of `dot_plot`'s data table: one row per `DotRow`, one column per series
    that has at least one real value; `none_label` / `<= 0` (under `log_x`) / `>= v` (censored)
    exactly as the chart's tooltips show them."""
    rows = [r for r in rows if r.points]
    model_names = [n for n, _c in _dot_legend_items(rows)]
    table_rows = []
    for row in rows:
        by_name = {n: (v, z) for n, v, _c, z in row.points}
        cells = []
        for n in model_names:
            v, z = by_name.get(n, (None, False))
            if v is None:
                cells.append(none_label if n in by_name else "")
            elif log_x and v <= 0:
                cells.append("<= 0")
            else:
                cells.append((">= " if z else "") + fmt(v, x_fmt))
        table_rows.append([row.label, *cells])
    return ["row", *model_names], table_rows


def dot_table(
    rows: list[DotRow], *, x_fmt: str = "g", log_x: bool = False, none_label: str = "none"
) -> str:
    return _table(*dot_table_rows(rows, x_fmt=x_fmt, log_x=log_x, none_label=none_label))


def dot_plot(
    title: str,
    sub: str,
    rows: list[DotRow],
    *,
    x_label: str = "x",
    x_fmt: str = "g",
    x_max: float | None = None,
    x_min: float | None = None,
    log_x: bool = False,
    none_label: str = "none",
    note: str = "",
    wide: bool = False,
) -> str:
    """A horizontal dot plot: one category row per `DotRow.label`, with one coloured dot per
    `(series_name, value, color, censored)` point at that row's own x position -- the summary
    dashboard's "one row per family/class, one dot per model" charts. A `None` value renders as a
    small marker (tooltip: `none_label`) at the row's left edge instead of a dot; `censored=True`
    draws a small `>=` mark after the dot. `log_x` mirrors `line_chart`'s own convention (log2
    internally, real-value tick labels); a value `<= 0` under `log_x` (undefined on a log scale,
    e.g. a slightly negative null-control estimate) is clamped to the axis minimum and tagged
    `<= 0` rather than silently misplaced or dropped.
    """
    rows = [r for r in rows if r.points]
    if not rows:
        return missing_fig(title, sub)
    fig_w = WW if wide else W
    label_w = min(260, max(110, fig_w // 3))
    row_h = 26
    top, bot = 10, 30
    total_h = top + row_h * len(rows)
    h = total_h + bot
    pw = fig_w - label_w - MR - 8

    def xform(x: float) -> float:
        return math.log2(max(x, 1e-9)) if log_x else x

    pos_vals = [v for r in rows for _n, v, _c, _z in r.points if v is not None and v > 0]
    all_vals = [v for r in rows for _n, v, _c, _z in r.points if v is not None]
    vals = pos_vals if log_x else all_vals
    default_lo = (0.1 if log_x else 0.0) if not vals else min(vals)
    lo_v = x_min if x_min is not None else default_lo
    hi_v = x_max if x_max is not None else (max(vals) if vals else 1.0) or 1.0
    if log_x:
        lo_t, hi_t = xform(lo_v), xform(hi_v if hi_v > lo_v else lo_v + 1.0)
        yt_t = nice_ticks(lo_t, hi_t, 6)
    else:
        yt = nice_ticks(lo_v, hi_v, 6)
        lo_v, hi_v = min(yt[0], lo_v), max(yt[-1], hi_v)
        lo_t, hi_t = lo_v, hi_v
        yt_t = yt

    def px(x: float) -> float:
        return label_w + (xform(x) - lo_t) / ((hi_t - lo_t) or 1) * pw

    def clamp(v: float) -> tuple[float, bool]:
        """`(plotted_value, was_clamped)`: under `log_x`, a non-positive value has no real log
        position, so it is pinned to the axis floor and flagged for a "<= 0" display instead."""
        if log_x and v <= 0:
            return lo_v, True
        return v, False

    g = []
    for t in yt_t:
        if t < lo_t - 1e-9 or t > hi_t + 1e-9:
            continue
        xpix = label_w + (t - lo_t) / ((hi_t - lo_t) or 1) * pw
        label = fmt(2**t, x_fmt) if log_x else fmt(t, x_fmt)
        g.append(f'<line class="grid" x1="{xpix:.1f}" x2="{xpix:.1f}" y1="{top}" y2="{total_h}"/>')
        g.append(
            f'<text class="tick" x="{xpix:.1f}" y="{total_h + 16}" text-anchor="middle">'
            f"{esc(label)}</text>"
        )
    g.append(
        f'<text class="alabel" x="{label_w + pw / 2:.1f}" y="{h - 4}" text-anchor="middle">'
        f"{esc(x_label)}</text>"
    )
    marks = []
    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        g.append(f'<line class="grid" x1="{label_w}" x2="{fig_w - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        g.append(
            f'<text class="tick" x="{label_w - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">'
            f"{esc(row.label)}</text>"
        )
        offsets = _row_offsets(len(row.points))
        for (n, v, c, censored), y_off in zip(row.points, offsets, strict=True):
            yi = y + y_off
            if v is None:
                tip = json.dumps([[row.label, "", None], [n, none_label, c]])
                marks.append(
                    f'<line class="whisker" x1="{label_w + 4:.1f}" x2="{label_w + 12:.1f}" '
                    f'y1="{yi:.1f}" y2="{yi:.1f}" style="stroke:{c}" data-tip="{esc(tip)}"/>'
                )
                continue
            plotted, clamped = clamp(v)
            val_text = "<= 0" if clamped else (f">= {fmt(v, x_fmt)}" if censored else fmt(v, x_fmt))
            tip = json.dumps([[row.label, "", None], [n, val_text, c]])
            marks.append(
                f'<circle class="dot" cx="{px(plotted):.1f}" cy="{yi:.1f}" r="5" style="fill:{c}" '
                f'data-tip="{esc(tip)}"/>'
            )
            if censored and not clamped:
                marks.append(
                    f'<text class="tick" x="{px(plotted) + 9:.1f}" y="{yi + 4:.1f}" '
                    f'data-tip="{esc(tip)}">&#8805;</text>'
                )
    g.append(f'<line class="axis" x1="{label_w}" x2="{label_w}" y1="{top}" y2="{total_h}"/>')
    svg = (
        f'<svg viewBox="0 0 {fig_w} {h}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}</svg>"
    )
    legend_items = _dot_legend_items(rows)
    leg = "".join(
        f'<span class="li"><span class="lkey" style="background:{c}"></span>{esc(n)}</span>'
        for n, c in legend_items
    )
    table = dot_table(rows, x_fmt=x_fmt, log_x=log_x, none_label=none_label)
    body = f'<div class="legend">{leg}</div><div class="plot">{svg}</div>{table}'
    return figure(title, sub, body, note, wide)


def venn2(
    title: str,
    sub: str,
    label_a: str,
    label_b: str,
    only_a: int,
    only_b: int,
    both: int,
    *,
    note: str = "",
) -> str:
    if only_a + only_b + both == 0:
        return missing_fig(title, sub)
    cx_a, cx_b, cy, r = W * 0.36, W * 0.60, H * 0.48, 72.0
    svg = (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">'
        f'<circle cx="{cx_a}" cy="{cy}" r="{r}" class="venn-a"/>'
        f'<circle cx="{cx_b}" cy="{cy}" r="{r}" class="venn-b"/>'
        f'<text x="{cx_a - r * 0.55:.1f}" y="{cy:.1f}" text-anchor="middle" '
        f'class="vlabel">{only_a}</text>'
        f'<text x="{(cx_a + cx_b) / 2:.1f}" y="{cy:.1f}" text-anchor="middle" '
        f'class="vlabel">{both}</text>'
        f'<text x="{cx_b + r * 0.55:.1f}" y="{cy:.1f}" text-anchor="middle" '
        f'class="vlabel">{only_b}</text>'
        f'<text x="{cx_a:.1f}" y="{cy - r - 14:.1f}" text-anchor="middle" '
        f'class="tick">{esc(label_a)}</text>'
        f'<text x="{cx_b:.1f}" y="{cy - r - 14:.1f}" text-anchor="middle" '
        f'class="tick">{esc(label_b)}</text>'
        "</svg>"
    )
    table = _table(
        ["group", "n"],
        [[f"{label_a} only", str(only_a)], [f"{label_b} only", str(only_b)], ["both", str(both)]],
    )
    return figure(title, sub, f'<div class="plot">{svg}</div>{table}', note)


def table_fig(
    title: str,
    sub: str,
    headers: list[str],
    rows: list[list[str]],
    note: str = "",
    wide: bool = True,
) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return figure(
        title,
        sub,
        f'<div class="tw"><table class="txt"><tr>{th}</tr>{body}</table></div>',
        note,
        wide,
    )


# ---------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;
--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--ctl:#898781;--q0:#cde2fb;
--q1:#b7d3f6;--q2:#9ec5f4;--q3:#86b6ef;--q4:#5598e7;--q5:#2a78d6;--q6:#1c5cab;--q7:#104281;
--q8:#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--ctl:#898781;--q0:#0d366b;--q1:#104281;--q2:#184f95;--q3:#1c5cab;
--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4;--q8:#cde2fb}}:root[data-theme="dark"]{
color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--ctl:#898781;--q0:#0d366b;--q1:#104281;
--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4;
--q8:#cde2fb}*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}header{padding:20px 16px 8px;
max-width:1360px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}header .sub{color:var(--ink2);
margin:0 0 10px;word-break:break-all}nav{display:flex;flex-wrap:wrap;gap:6px}
nav a{color:var(--ink2);text-decoration:none;padding:4px 10px;border-radius:6px;
border:1px solid var(--ring)}nav a.on{color:var(--ink);font-weight:600;
background:var(--surface)}main{max-width:1360px;margin:0 auto;padding:8px 16px 40px;
display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:14px}
.card{margin:0;background:var(--surface);border:1px solid var(--ring);border-radius:10px;
padding:12px 14px;min-width:0}.card.wide{grid-column:1/-1}figcaption h3{font-size:15px;margin:0}
figcaption .sub,.note{color:var(--ink2);margin:2px 0 6px;font-size:12.5px}.note{margin-top:8px}
svg{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}.tick{fill:var(--muted);font-size:11px;
font-variant-numeric:tabular-nums}.alabel{fill:var(--ink2);font-size:11.5px}.ln{fill:none;
stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.ln.muted{stroke-dasharray:6 4}
.dot{stroke:var(--surface);
stroke-width:2}.cross{stroke:var(--axis);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.band{opacity:.16;stroke:none}.whisker{stroke:var(--ink2);stroke-width:1.3}
.seg{stroke:var(--surface);stroke-width:1}.seg:hover,rect[data-tip]:hover{opacity:.8}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:2px 0 4px;color:var(--ink2);
font-size:12.5px}.li{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.legend svg.lkeyline{flex:none;width:16px;height:10px;display:inline-block}
.lkey{width:14px;height:3px;
border-radius:2px;display:inline-block}.bkey{width:10px;height:10px;border-radius:2px;
display:inline-block}.q0{fill:var(--q0);background:var(--q0)}.q1{fill:var(--q1);
background:var(--q1)}.q2{fill:var(--q2);background:var(--q2)}.q3{fill:var(--q3);
background:var(--q3)}.q4{fill:var(--q4);background:var(--q4)}.q5{fill:var(--q5);
background:var(--q5)}.q6{fill:var(--q6);background:var(--q6)}.q7{fill:var(--q7);
background:var(--q7)}.q8{fill:var(--q8);background:var(--q8)}.na{fill:transparent;
stroke:var(--grid);stroke-width:1}i.na{border:1px solid var(--grid)}.scale{display:flex;
align-items:center;gap:2px;font-size:11.5px;color:var(--ink2);margin:0 0 4px}
.scale i{display:inline-block;width:16px;height:10px}.scale span{margin:0 6px}
.na-key{display:inline-flex;gap:4px;align-items:center}.missing{color:var(--muted);
border:1px dashed var(--grid);border-radius:8px;padding:28px;text-align:center}
.venn-a{fill:var(--s1);fill-opacity:.35;stroke:var(--s1);stroke-width:1.5}
.venn-b{fill:var(--s2);fill-opacity:.35;stroke:var(--s2);stroke-width:1.5;
mix-blend-mode:multiply}.vlabel{fill:var(--ink);font-size:15px;font-weight:600;
font-variant-numeric:tabular-nums}
details{margin-top:6px;font-size:12.5px}summary{cursor:pointer;color:var(--ink2)}
.tw{overflow-x:auto;max-height:420px}table{border-collapse:collapse;
font-variant-numeric:tabular-nums;font-size:12.5px}td,th{border-bottom:1px solid var(--grid);
padding:3px 8px;text-align:right;white-space:nowrap}th{position:sticky;top:0;
background:var(--surface);color:var(--ink2);font-weight:600}
td:first-child,th:first-child{text-align:left}table.txt td,table.txt th{text-align:left;
white-space:normal}.tip{position:fixed;display:none;pointer-events:none;
background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
padding:6px 9px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9}.tr{display:flex;
align-items:center;gap:6px;white-space:nowrap}.tr b{font-variant-numeric:tabular-nums}
.tr .k{width:12px;height:3px;border-radius:2px;display:inline-block}
"""

#: Charts re-render on resize (plan §3's WP6 constraint): rebuilding every SVG from its embedded
#: JSON blob is simplest here since the primitives above already compute pixel positions in
#: Python; the JS side only needs the hover/tooltip wiring, which is resize-safe as-is (it reads
#: current DOM/CTM state on each pointer event, not cached pixel coordinates), so no JS-side
#: relayout is needed for the crosshair/tooltip to keep working after a resize.
JS = """
(function(){const tip=document.createElement('div');tip.className='tip';
document.body.appendChild(tip);function show(e,rows){tip.replaceChildren();
for(const [n,v,c] of rows){const r=document.createElement('div');r.className='tr';if(c){
const k=document.createElement('span');k.className='k';k.style.background=c;r.appendChild(k);}
if(v){const b=document.createElement('b');b.textContent=v;r.appendChild(b);}
const t=document.createElement('span');t.textContent=n;r.appendChild(t);tip.appendChild(r);}
tip.style.display='block';const w=tip.offsetWidth,h=tip.offsetHeight;
let x=e.clientX+14,y=e.clientY+14;if(x+w>innerWidth-8)x=e.clientX-w-14;
if(y+h>innerHeight-8)y=e.clientY-h-14;tip.style.left=x+'px';tip.style.top=y+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('.plot[data-chart="line"]').forEach(p=>{
const d=JSON.parse(p.querySelector('script').textContent);
const svg=p.querySelector('svg'),hit=svg.querySelector('.hit'),cr=svg.querySelector('.cross');
if(!hit||!cr)return;
hit.addEventListener('pointermove',e=>{const pt=svg.createSVGPoint();pt.x=e.clientX;
pt.y=e.clientY;const q=pt.matrixTransform(svg.getScreenCTM().inverse());let b=0,bd=1e9;
d.px.forEach((x,i)=>{const dd=Math.abs(x-q.x);if(dd<bd){bd=dd;b=i;}});
cr.setAttribute('x1',d.px[b]);cr.setAttribute('x2',d.px[b]);
cr.setAttribute('visibility','visible');
show(e,[[d.xl+' '+d.xs[b],'',null]].concat(d.s.map(s=>[s.n,s.v[b],s.c])));});
hit.addEventListener('pointerleave',()=>{hide();cr.setAttribute('visibility','hidden');});});
document.querySelectorAll('[data-tip]').forEach(el=>{
el.addEventListener('pointermove',e=>show(e,JSON.parse(el.dataset.tip)));
el.addEventListener('pointerleave',hide);});})();
"""


def page(title: str, sub: str, active: str, figs: list[str], nav: list[tuple[str, str]]) -> str:
    """`nav`: `[(href, name), ...]`; the entry whose `href` matches `active` gets the "on" class."""
    links = []
    for href, name in nav:
        cls = ' class="on"' if href == active else ""
        links.append(f'<a{cls} href="{esc(href)}">{esc(name)}</a>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
        f'<header><h1>{esc(title)}</h1><p class="sub">{esc(sub)}</p>'
        f"<nav>{''.join(links)}</nav></header><main>{''.join(figs)}</main>"
        f"<script>{JS}</script></body></html>"
    )
