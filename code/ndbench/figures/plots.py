"""One NeurIPS-style figure per summary benchmark, plus the Figure 1 overview grid.

Every function takes the `ChartData` that `dash/summary.py` computed for that benchmark (the same
object whose values the summary page tabulates) and only *draws* it: series, bars and dot rows are
plotted as given, CIs are the bands/whiskers the data carries, and the few figure-only panels
(textbook_law, the skills comparisons, EDL against its starting loss) read values `summary.py`
already computed for its takeaway or caption. Sorting rows for display is the only reordering.

Chart types (rationale in `STYLE.md`): pass@k and other curves are line plots with markers and
shaded 95% CI bands; per-family / per-class results are dot plots (Cleveland), with base and rl
joined by a thin connector where the gap is the claim; glue and the skills comparisons are grouped
bars with capped CIs; steering is a booktabs table on the page (`figures/tables.py`) with a
forest-plot variant here.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator

from ..dash import primitives as P
from ..dash.chart_data import ChartData
from . import style as S

#: Display order of models: base, then the RL models, never the view's alphabetical order.
MODEL_ORDER: tuple[str, ...] = ("base", "rl", "armC")


def _order(names: Sequence[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(names, key=lambda m: (rank.get(m, len(rank)), m))


def _plot_line(
    ax: Axes,
    pts: Sequence[tuple[float, float]],
    st: S.ModelStyle,
    *,
    band: Sequence[tuple[float, float, float]] | None = None,
    markers: bool = True,
    marker_size: float = 3.6,
    zorder: float = 3,
) -> None:
    pts = sorted(pts)
    if band:
        b = sorted(band)
        ax.fill_between(
            [x for x, _lo, _hi in b],
            [lo for _x, lo, _hi in b],
            [hi for _x, _lo, hi in b],
            color=st.color,
            alpha=S.BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )
        for k in (1, 2):
            ax.plot(
                [t[0] for t in b],
                [t[k] for t in b],
                color=st.color,
                alpha=S.BAND_EDGE_ALPHA,
                linewidth=0.45,
                zorder=1.5,
            )
    kw = st.line_kw(markers=False)
    if markers:
        kw.update(st.marker_kw(marker_size))
    ax.plot([x for x, _ in pts], [y for _, y in pts], zorder=zorder, **kw)


def _top_legend(fig: Figure, handles: Sequence[Any], ncol: int | None = None) -> None:
    fig.legend(
        handles=list(handles),
        loc="outside upper center",
        ncol=ncol or len(handles),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )


def _ylim_zero(ax: Axes, top: float | None = None) -> None:
    lo, hi = ax.get_ylim()
    ax.set_ylim(0.0, top if top is not None else hi)


def _integer_x(ax: Axes, nbins: int = 7) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins, steps=[1, 2, 5, 10]))


def _series(cd: ChartData) -> dict[str, P.Series]:
    return {s.name: s for s in cd.series if s.pts}


# ======================================================================== 1. textbook pass@k


def _draw_passk(
    ax: Axes, series: Sequence[P.Series], *, bands: bool, size: float = 3.6, every: int = 4
) -> float:
    by = {s.name: s for s in series if s.pts}
    k_max = 1.0
    for m in _order(list(by)):
        s = by[m]
        _plot_line(ax, s.pts, S.model_style(m), band=s.band if bands else None, marker_size=size)
        k_max = max(k_max, max(x for x, _ in s.pts))
    S.log2_k_axis(ax, k_max, every=every)
    S.pct_axis(ax)
    _ylim_zero(ax)
    S.light_grid(ax)
    return k_max


def fig_yue(cd: ChartData, *, narrow: bool = False) -> Figure:
    law = cd.extra.get("law")
    panels: list[tuple[str, Any, list[P.Series]]] = [
        ("textbook_dev", cd.extra.get("n", "?"), cd.series)
    ]
    if law:
        panels.append(("textbook_law", law.get("n", "?"), law["series"]))
    if narrow:
        fig = S.new_figure(S.NARROW_W, 0.45 + 1.75 * len(panels))
        axs = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    else:
        fig = S.new_figure(S.FULL_W, 2.05)
        axs = fig.subplots(1, len(panels), squeeze=False)[0]
    for i, (ax, (name, n, series)) in enumerate(zip(axs, panels, strict=True)):
        _draw_passk(ax, series, bands=True)
        ax.set_xlabel("k (samples per theorem)")
        if i == 0 or narrow:
            ax.set_ylabel("pass@k")
        S.panel_title(ax, "ab"[i] if len(panels) > 1 else None, f"{name}, n = {n} theorems")
    models = _order(list({s.name for _n, _k, ss in panels for s in ss if s.pts}))
    _top_legend(fig, S.legend_handles([S.model_style(m) for m in models]))
    return fig


# ======================================================================== 2. families horizon


def _row_offsets(n: int, spread: float = 0.24) -> list[float]:
    if n <= 1:
        return [0.0] * n
    return [-spread + 2 * spread * i / (n - 1) for i in range(n)]


def fig_families(cd: ChartData) -> Figure:
    # Rows sorted by the best horizon any model reaches (ties keep the table's order), so the
    # families where something holds up sit at the top.
    def best(r: P.DotRow) -> float:
        return max((v for _n, v, _c, _z in r.points if v is not None), default=0.0)

    rows = sorted(cd.rows, key=lambda r: -best(r))
    models = _order(list({n for r in rows for n, _v, _c, _z in r.points}))
    offs = dict(zip(models, _row_offsets(len(models)), strict=True))
    fig = S.new_figure(S.HALF_W, 0.62 + 0.2 * len(rows))
    ax = fig.add_subplot()
    x_hi = max([best(r) for r in rows] + [1.0])
    for i, row in enumerate(rows):
        for name, v, _c, censored in row.points:
            st = S.model_style(name)
            y = i + offs[name]
            if v is None:
                hollow = S.variant(st, name, filled=False).marker_kw(3.6)
                ax.plot([0.0], [y], linestyle="none", label=f"_{name}", **hollow)
                continue
            ax.plot([v], [y], linestyle="none", zorder=4, label=f"_{name}", **st.marker_kw(4.2))
            if censored:
                ax.annotate(
                    "",
                    xy=(v + 0.09 * x_hi, y),
                    xytext=(v + 0.025 * x_hi, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "lw": 0.7,
                        "color": st.color,
                        "mutation_scale": 5,
                    },
                )
    ax.set_yticks(range(len(rows)), [r.label for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.06 * x_hi - 0.3, x_hi * 1.14 + 0.3)
    _integer_x(ax)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    S.light_grid(ax, "x")
    for i in range(len(rows)):
        ax.axhline(i, color=S.GRID, linewidth=0.4, zorder=0)
    ax.set_xlabel("written lines at n* (unseen theorems)")
    handles = S.legend_handles([S.model_style(m) for m in models], kind="marker")
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            label="fails at smallest size",
            **S.ModelStyle("x", S.REF, "o", filled=False).marker_kw(3.6),
        )
    )
    _top_legend(fig, handles, ncol=2)
    return fig


# ===================================================================================== 3. glue


def _bar_lookup(bars: Sequence[P.Bar]) -> dict[tuple[str, str], P.Bar]:
    out: dict[tuple[str, str], P.Bar] = {}
    for b in bars:
        group, _sep, model = b.name.partition(": ")
        out[(group, model)] = b
    return out


def _grouped_bars(
    ax: Axes,
    groups: Sequence[str],
    models: Sequence[str],
    value: Callable[[str, str], P.Bar | None],
    *,
    width: float = 0.38,
    label_small: Callable[[float], str] | None = None,
    small_frac: float = 0.06,
    y_top: float | None = None,
    label_size: float = 6.0,
) -> None:
    span = y_top if y_top is not None else 1.0
    for j, m in enumerate(models):
        st = S.model_style(m)
        off = (j - (len(models) - 1) / 2) * width
        for i, g in enumerate(groups):
            b = value(g, m)
            if b is None or b.value != b.value:
                continue
            kw = st.bar_kw()
            if i:
                kw.pop("label", None)
            ax.bar(i + off, b.value, width * 0.9, zorder=2, **kw)
            top = b.value
            if b.lo is not None and b.hi is not None and b.lo == b.lo and b.hi == b.hi:
                ax.errorbar(
                    i + off,
                    b.value,
                    yerr=[[max(b.value - b.lo, 0.0)], [max(b.hi - b.value, 0.0)]],
                    fmt="none",
                    ecolor=S.INK2,
                    elinewidth=0.6,
                    capsize=1.6,
                    capthick=0.6,
                    zorder=3,
                )
                top = max(top, b.hi)
            if label_small is not None and b.value < small_frac * span:
                ax.text(
                    i + off,
                    top + 0.02 * span,
                    label_small(b.value),
                    ha="center",
                    va="bottom",
                    fontsize=label_size,
                    rotation=90,
                    color=S.INK2,
                )
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.tick_params(axis="x", length=0)


def _fmt_pct_small(v: float) -> str:
    return "0" if v == 0 else P.fmt(v, "pct")


def _fmt_r(v: float) -> str:
    return "0" if abs(v) < 5e-4 else f"{v:.3f}"


def fig_glue(cd: ChartData) -> Figure:
    lookup = _bar_lookup(cd.bars)
    ops = list(cd.extra.get("ops") or dict.fromkeys(g for g, _m in lookup))
    counts: dict[str, int] = cd.extra.get("counts", {})
    models = _order(list({m for _g, m in lookup}))
    tops = [b.hi if b.hi is not None and b.hi == b.hi else b.value for b in cd.bars]
    y_top = max([1.08, *(t * 1.08 for t in tops)])
    fig = S.new_figure(S.HALF_W, 2.1)
    ax = fig.add_subplot()
    _grouped_bars(ax, ops, models, lambda g, m: lookup.get((g, m)), label_small=_fmt_r, y_top=y_top)
    ax.set_xticklabels([f"{op}\n(n = {counts[op]})" if op in counts else op for op in ops])
    ax.axhline(1.0, color=S.REF, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax.text(len(ops) - 0.42, 1.0, "R = 1", ha="right", va="bottom", fontsize=S.FS_TICK, color=S.REF)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_ylabel("compositionality ratio R")
    S.light_grid(ax)
    _top_legend(fig, S.legend_handles([S.model_style(m) for m in models], kind="bar"))
    return fig


# =================================================================================== 4. skills

_SKILL_SHORT: dict[str, str] = {
    "no special skill": "no special skill",
    "needs case split": "case split",
    "needs classical": "classical",
    "case_split": "case split",
    "classical": "classical",
}
_SKILL_TICK: dict[str, str] = {
    "no special skill": "none",
    "case_split": "case\nsplit",
    "classical": "classical",
}


def _skill_style(name: str) -> S.ModelStyle:
    """All three skill lines are rl: one hue (rl's), told apart by shade, marker and dash."""
    rl = S.model_style("rl")
    label = _SKILL_SHORT.get(name, name)
    if name == "needs case split":
        return S.variant(
            rl, label, color=S.shade(rl.color, -0.45), marker="D", linestyle=(0, (4, 1.6))
        )
    if name == "needs classical":
        return S.variant(
            rl, label, color=S.shade(rl.color, 0.38), marker="^", linestyle=(0, (1, 1.2))
        )
    return S.variant(rl, label)


def fig_skills(cd: ChartData, *, narrow: bool = False) -> Figure:
    comparisons: list[dict[str, Any]] = list(cd.extra.get("comparisons", []))
    axs: list[Axes]
    if narrow and comparisons:
        fig = S.new_figure(S.NARROW_W, 4.0)
        gs = fig.add_gridspec(2, len(comparisons), height_ratios=[1.15, 1.0])
        top = fig.add_subplot(gs[0, :])
        axs = [top] + [fig.add_subplot(gs[1, j], sharey=top) for j in range(len(comparisons))]
    else:
        fig = S.new_figure(S.NARROW_W if narrow else S.FULL_W, 2.1)
        ratios = [2.25] + [1.3] * len(comparisons)
        axs = list(
            fig.subplots(
                1, len(ratios), squeeze=False, sharey=True, gridspec_kw={"width_ratios": ratios}
            )[0]
        )
    ax = axs[0]
    y_span = max(
        [y for s in cd.series for _x, y in s.pts]
        + [hi for s in cd.series for _x, _lo, hi in (s.band or [])]
        + [
            float(v)
            for c in comparisons
            for r in c["by_skill"].values()
            for m, v in r.items()
            if m != "n_items"
        ]
        + [0.01]
    )
    for s in cd.series:
        if s.pts:
            _plot_line(ax, s.pts, _skill_style(s.name), band=s.band)
    _integer_x(ax)
    ax.set_xlabel("L (written lines)")
    ax.set_ylabel("pass@1")
    S.pct_axis(ax)
    S.light_grid(ax)
    S.panel_title(ax, "a" if comparisons else None, "rl, by required skill")
    ax.legend(
        handles=S.legend_handles([_skill_style(s.name) for s in cd.series if s.pts]),
        loc="upper right",
        handlelength=2.2,
    )
    models_seen: list[str] = []
    for i, c in enumerate(comparisons, start=1):
        a, b = c["models"]
        by_skill: dict[str, dict[str, Any]] = c["by_skill"]
        groups = list(by_skill)
        axc = axs[i]

        def val(g: str, m: str, by_skill: dict[str, dict[str, Any]] = by_skill) -> P.Bar | None:
            v = by_skill.get(g, {}).get(m)
            return None if v is None else P.Bar(f"{g}: {m}", float(v))

        _grouped_bars(
            axc,
            groups,
            [a, b],
            val,
            width=0.36,
            label_small=_fmt_pct_small,
            small_frac=0.03,
            y_top=y_span,
        )
        axc.set_xticklabels([_SKILL_TICK.get(g, g) for g in groups])
        S.light_grid(axc)
        axc.tick_params(axis="y", labelleft=narrow and i == 1)
        if narrow and i == 1:
            S.pct_axis(axc)
            axc.set_ylabel("pass@1")
        sep = "\n" if narrow else ", "
        S.panel_title(axc, "abcdefg"[i], f"{a} vs {b}{sep}n = {c['n_common']:,}")
        models_seen += [m for m in (a, b) if m not in models_seen]
    _ylim_zero(ax)
    if comparisons:
        _top_legend(
            fig, S.legend_handles([S.model_style(m) for m in _order(models_seen)], kind="bar")
        )
    return fig


# ===================================================================== 5-7. simple line figures


def _lines_figure(
    cd: ChartData,
    styles: Callable[[str], S.ModelStyle],
    *,
    x_label: str,
    legend_loc: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right",
    y_label: str = "pass@1",
    width: float = S.HALF_W,
    height: float = 1.95,
    integer_x: bool = True,
) -> Figure:
    fig = S.new_figure(width, height)
    ax = fig.add_subplot()
    for s in cd.series:
        if s.pts:
            _plot_line(ax, s.pts, styles(s.name), band=s.band)
    if integer_x:
        _integer_x(ax)
    S.pct_axis(ax)
    _ylim_zero(ax)
    S.light_grid(ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(
        handles=S.legend_handles([styles(s.name) for s in cd.series if s.pts]), loc=legend_loc
    )
    return fig


def fig_subst(cd: ChartData) -> Figure:
    return _lines_figure(cd, S.model_style, x_label="s (connectives per metavariable)")


def _padding_style(name: str) -> S.ModelStyle:
    model, _sep, kind = name.partition(" ")
    st = S.model_style(model)
    if kind == "natural":
        return S.variant(st, f"{model}, natural theorems", filled=False, linestyle=(0, (4, 1.6)))
    return S.variant(st, f"{model}, padded")


def fig_padding(cd: ChartData) -> Figure:
    return _lines_figure(cd, _padding_style, x_label="W (written lines)")


def fig_prefix(cd: ChartData) -> Figure:
    return _lines_figure(cd, S.model_style, x_label="lines left to write (L - j)")


# ====================================================================================== 8. EDL


def _clean_label(label: str) -> str:
    return label.replace(" (null control)", "")


def fig_edl(cd: ChartData, *, narrow: bool = False) -> Figure:
    per: dict[str, dict[str, dict[str, float]]] = cd.extra.get("per", {})
    null = cd.extra.get("null_control")
    rho = cd.extra.get("rho")
    rows = cd.rows
    models = _order(list({n for r in rows for n, _v, _c, _z in r.points}))
    if narrow:
        fig = S.new_figure(S.NARROW_W, 0.62 + 0.17 * len(rows) + 2.2)
        axs = fig.subplots(2, 1, gridspec_kw={"height_ratios": [0.17 * len(rows) + 0.3, 2.0]})
    else:
        fig = S.new_figure(S.FULL_W, 0.62 + 0.17 * len(rows))
        axs = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3.0, 2.0]})
    ax = axs[0]
    pos = [v for r in rows for _n, v, _c, _z in r.points if v is not None and v > 0]
    pos += [
        per[c][m][f]
        for c in per
        for m in per[c]
        for f in ("lo", "hi")
        if per[c][m][f] == per[c][m][f] and per[c][m][f] > 0
    ]
    floor = min(pos) / 2.2 if pos else 0.1
    offs = dict(zip(models, _row_offsets(len(models), 0.13), strict=True))
    for i, row in enumerate(rows):
        cls = _clean_label(row.label)
        vals = {n: v for n, v, _c, _z in row.points if v is not None}
        if len(vals) >= 2:
            xs = [max(v, floor) for v in vals.values()]
            ax.plot([min(xs), max(xs)], [i, i], color="#bdbdbd", linewidth=1.0, zorder=1)
        for name, v in vals.items():
            st = S.model_style(name)
            y = i + offs[name]
            d = per.get(cls, {}).get(name, {})
            lo, hi = d.get("lo", math.nan), d.get("hi", math.nan)
            if lo == lo and hi == hi:
                ax.plot(
                    [max(lo, floor), max(hi, floor)],
                    [y, y],
                    color=st.color,
                    linewidth=0.8,
                    zorder=2,
                )
            ax.plot(
                [max(v, floor)],
                [y],
                linestyle="none",
                zorder=4,
                label=f"_{name}",
                **st.marker_kw(4.0),
            )
            if v <= 0:
                ax.text(floor * 1.12, y, "≤ 0", fontsize=6.0, va="center", ha="left", color=S.INK2)
        if null is not None and cls == null and i > 0:
            ax.axhline(i - 0.5, color="#9a9a9a", linewidth=0.5, zorder=0)
    S.plain_log_axis(ax, "x")
    ax.set_xlim(left=floor / 1.3)
    ax.set_yticks(range(len(rows)), [r.label for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    S.light_grid(ax, "x")
    ax.set_xlabel("EDL* per example (nats, log scale)")
    S.panel_title(
        ax, "a", "EDL* per class" if narrow else "EDL* per class (whiskers: two data orders)"
    )

    axb = axs[1]
    n_points = 0
    for name in models:
        st = S.model_style(name)
        xs, ys = [], []
        for cls, d in per.items():
            if cls == null or name not in d:
                continue
            x, y = d[name]["l0"], d[name]["edl"]
            if x == x and y == y and x > 0 and y > 0:
                xs.append(x)
                ys.append(y)
        n_points += len(xs)
        axb.plot(xs, ys, linestyle="none", zorder=3, label=f"_{name}", **st.marker_kw(4.0))
    S.panel_title(axb, "b", "EDL* vs starting loss")
    if not n_points:  # log-log needs positive values (e.g. every EDL* is 0)
        axb.text(
            0.5,
            0.5,
            "no positive EDL*",
            transform=axb.transAxes,
            ha="center",
            va="center",
            fontsize=S.FS_TICK,
            color=S.REF,
        )
        axb.set_xticks([])
        axb.set_yticks([])
        _top_legend(fig, S.legend_handles([S.model_style(m) for m in models], kind="marker"))
        return fig
    for cls, d in per.items():
        if cls == null or not all(m in d for m in models[:2]) or len(models) < 2:
            continue
        a, b = d[models[0]], d[models[1]]
        if min(a["l0"], b["l0"], a["edl"], b["edl"]) > 0:
            axb.plot(
                [a["l0"], b["l0"]], [a["edl"], b["edl"]], color="#bdbdbd", linewidth=0.7, zorder=1
            )
    S.plain_log_axis(axb, "x")
    S.plain_log_axis(axb, "y")
    S.light_grid(axb, "both")
    axb.set_xlabel("initial test loss (nats per example)")
    axb.set_ylabel("EDL* per example (nats)")
    if isinstance(rho, float) and rho == rho:
        axb.text(
            0.04,
            0.96,
            f"Spearman ρ = {rho:.2f}",
            transform=axb.transAxes,
            ha="left",
            va="top",
            fontsize=S.FS_TICK,
        )
    _top_legend(fig, S.legend_handles([S.model_style(m) for m in models], kind="marker"))
    return fig


# ================================================================================ 9. likelihood


def _two_line(label: str) -> str:
    head, sep, tail = label.partition(" (n=")
    if not sep:
        return label
    n = tail.rstrip(")")
    return f"{head}\nn = {int(n):,}" if n.isdigit() else f"{head}\nn = {n}"


def _dumbbell(
    ax: Axes, rows: Sequence[P.DotRow], *, labels: Sequence[str], size: float = 4.2
) -> list[str]:
    models = _order(list({n for r in rows for n, _v, _c, _z in r.points}))
    for i, row in enumerate(rows):
        vals = {n: v for n, v, _c, _z in row.points if v is not None}
        if len(vals) >= 2:
            ax.plot(
                [min(vals.values()), max(vals.values())],
                [i, i],
                color="#bdbdbd",
                linewidth=1.1,
                zorder=1,
            )
        for name in _order(list(vals)):
            ax.plot(
                [vals[name]],
                [i],
                linestyle="none",
                zorder=3,
                label=f"_{name}",
                **S.model_style(name).marker_kw(size),
            )
    ax.set_yticks(range(len(rows)), list(labels))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    S.light_grid(ax, "x")
    return models


def fig_likelihood(cd: ChartData) -> Figure:
    rows = cd.rows
    fig = S.new_figure(S.HALF_W, 0.72 + 0.33 * len(rows))
    ax = fig.add_subplot()
    models = _dumbbell(ax, rows, labels=[_two_line(r.label) for r in rows])
    ax.set_xlabel("median log p per written line (nats)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    lo = min((v for r in rows for _n, v, _c, _z in r.points if v is not None), default=-1.0)
    ax.set_xlim(min(lo * 1.06, lo - 0.5), 0.4)
    _top_legend(fig, S.legend_handles([S.model_style(m) for m in models], kind="marker"))
    return fig


# ================================================================================= 10. steering


def _steer_style(name: str) -> S.ModelStyle:
    if name.startswith("base +"):
        return S.variant(S.model_style("base"), name, filled=False)
    if name.startswith("rl -"):
        return S.variant(S.model_style("rl"), name, filled=False)
    return S.model_style("rl" if name == "rl" else "base")


def fig_steering(cd: ChartData) -> Figure:
    """Forest-plot variant of the steering comparison (the page shows it as Table 1)."""
    bars = cd.bars
    fig = S.new_figure(S.HALF_W, 0.62 + 0.27 * len(bars))
    ax = fig.add_subplot()
    for i, b in enumerate(bars):
        st = _steer_style(b.name)
        if b.lo is not None and b.hi is not None:
            ax.plot([b.lo, b.hi], [i, i], color=st.color, linewidth=0.9, zorder=2)
            for x in (b.lo, b.hi):
                ax.plot([x, x], [i - 0.12, i + 0.12], color=st.color, linewidth=0.9, zorder=2)
        ax.plot([b.value], [i], linestyle="none", zorder=3, **st.marker_kw(4.2))
    ax.set_yticks(
        range(len(bars)), [b.name.replace("delta", "δ").replace(" a=", " α=") for b in bars]
    )
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    S.pct_axis(ax, "x")
    ax.set_xlim(left=0)
    S.light_grid(ax, "x")
    ax.set_xlabel("pass@1 (95% CI)")
    return fig


# ================================================================================== 11. weights

_PASS1 = S.ModelStyle("pass@1", S.INK, "o")
_PASS128 = S.ModelStyle("pass@128", S.REF, "s", linestyle=(0, (4, 1.6)), filled=False)


def _draw_weights(ax: Axes, cd: ChartData, *, size: float = 3.6, annotate: bool = True) -> None:
    by = _series(cd)
    xs = sorted({x for s in by.values() for x, _ in s.pts})
    if xs and xs[-1] > 1.0:
        ax.axvspan(1.0, xs[-1] + 0.1, color=S.SHADE, linewidth=0, zorder=0)
        if annotate:
            ax.text(
                (1.0 + xs[-1]) / 2 + 0.05,
                0.97,
                "extrapolation",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=S.FS_TICK - 0.5,
                color=S.INK2,
            )
    for name, st in (("pass@128", _PASS128), ("pass@1", _PASS1)):
        if name in by:
            _plot_line(ax, by[name].pts, st, band=by[name].band, marker_size=size)
    ticks = [x for x in xs if abs(x * 4 - round(x * 4)) < 1e-9]
    ax.xaxis.set_major_locator(FixedLocator(ticks or xs))
    labels = []
    for t in ticks or xs:
        lab = f"{t:g}"
        if annotate and t == 0:
            lab += "\nbase"
        elif annotate and t == 1:
            lab += "\nrl"
        labels.append(lab)
    ax.set_xticklabels(labels)
    if xs:
        ax.set_xlim(xs[0] - 0.06, xs[-1] + 0.06)
    S.pct_axis(ax)
    _ylim_zero(ax)
    S.light_grid(ax)


def fig_weights(cd: ChartData) -> Figure:
    fig = S.new_figure(S.HALF_W, 2.0)
    ax = fig.add_subplot()
    _draw_weights(ax, cd)
    ax.set_xlabel("interpolation weight α")
    ax.set_ylabel("pass@k")
    by = _series(cd)
    ax.legend(
        handles=S.legend_handles(
            [st for n, st in (("pass@1", _PASS1), ("pass@128", _PASS128)) if n in by]
        ),
        loc="upper left",
    )
    return fig


# =================================================================================== 12. probes


def _probe_parts(cd: ChartData) -> tuple[list[tuple[str, P.Series]], P.Series | None]:
    models: list[tuple[str, P.Series]] = []
    ref: P.Series | None = None
    for s in cd.series:
        if not s.pts:
            continue
        if s.name.startswith("next_rule probe: "):
            models.append((s.name.split(": ", 1)[1], s))
        else:
            ref = s
    return sorted(models, key=lambda t: MODEL_ORDER.index(t[0]) if t[0] in MODEL_ORDER else 99), ref


def _draw_probes(
    ax: Axes, cd: ChartData, *, size: float = 3.6, label_ref: bool = True
) -> list[str]:
    models, ref = _probe_parts(cd)
    if ref is not None:
        y = ref.pts[0][1]
        ax.axhline(y, color=S.REF, linewidth=0.8, linestyle=(0, (4, 1.6)), zorder=1)
        if label_ref:
            ax.text(
                0.98,
                y,
                "position-only lookup",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=S.FS_TICK - 0.5,
                color=S.REF,
            )
    for m, s in models:
        _plot_line(ax, s.pts, S.model_style(m), marker_size=size)
    labels: list[str] = list(cd.opts.get("x_tick_labels") or [])
    xs = sorted({x for _m, s in models for x, _ in s.pts})
    if labels:
        ax.xaxis.set_major_locator(FixedLocator([float(i) for i in range(len(labels))]))
        ax.set_xticklabels(labels)
    if xs:
        ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25)
    S.pct_axis(ax)
    S.light_grid(ax)
    return [m for m, _s in models]


def fig_probes(cd: ChartData) -> Figure:
    fig = S.new_figure(S.HALF_W, 2.0)
    ax = fig.add_subplot()
    models = _draw_probes(ax, cd)
    ax.set_xlabel("layer")
    ax.set_ylabel("next_rule probe accuracy")
    ax.legend(handles=S.legend_handles([S.model_style(m) for m in models]), loc="lower right")
    return fig


# ============================================================================ Figure 1 overview


def _mini_missing(ax: Axes) -> None:
    ax.text(
        0.5,
        0.5,
        "not available",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=S.FS_SMALL,
        color=S.REF,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _mini_lines(
    ax: Axes, cd: ChartData, styles: Callable[[str], S.ModelStyle], x_label: str
) -> None:
    for s in cd.series:
        if s.pts:
            _plot_line(ax, s.pts, styles(s.name), marker_size=2.4)
    _integer_x(ax)
    S.pct_axis(ax, nbins=4)
    _ylim_zero(ax)
    S.light_grid(ax)
    ax.set_xlabel(x_label, labelpad=1)


def _mini_families(ax: Axes, cd: ChartData) -> None:
    """Longest horizon (written lines at n*) any family reaches, per model (0 = fails at the
    smallest unseen size everywhere): the max over Figure 3's rows."""
    best: dict[str, tuple[float, bool]] = {}
    for r in cd.rows:
        for n, v, _c, z in r.points:
            cur = best.get(n, (0.0, False))
            if v is not None and v > cur[0]:
                best[n] = (v, z)
            else:
                best.setdefault(n, cur)
    models = _order(list(best))
    for i, m in enumerate(models):
        v, censored = best[m]
        ax.bar(i, v, 0.62, zorder=2, **S.model_style(m).bar_kw())
        ax.text(
            i,
            v,
            ("≥" if censored else "") + f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=S.FS_SMALL - 0.3,
        )
    ax.set_xticks(range(len(models)), models)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    _ylim_zero(ax, max([v for v, _z in best.values()] + [1.0]) * 1.25)
    S.light_grid(ax)


def _mini_glue(ax: Axes, cd: ChartData) -> None:
    lookup = _bar_lookup(cd.bars)
    ops = list(cd.extra.get("ops") or dict.fromkeys(g for g, _m in lookup))
    models = _order(list({m for _g, m in lookup}))
    tops = [b.hi if b.hi is not None and b.hi == b.hi else b.value for b in cd.bars]
    y_top = max([1.1, *(t * 1.06 for t in tops if t == t)])
    _grouped_bars(
        ax,
        ops,
        models,
        lambda g, m: lookup.get((g, m)),
        width=0.38,
        label_small=_fmt_r,
        small_frac=0.04,
        y_top=y_top,
        label_size=5.5,
    )
    ax.set_xticklabels(ops, rotation=30, ha="right", rotation_mode="anchor")
    ax.axhline(1.0, color=S.REF, linewidth=0.5, linestyle=(0, (1, 1.5)), zorder=1)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    S.light_grid(ax)


def _mini_skills(ax: Axes, cd: ChartData) -> None:
    comparisons: list[dict[str, Any]] = list(cd.extra.get("comparisons", []))
    if not comparisons:
        _mini_missing(ax)
        return
    c = comparisons[0]
    a, b = c["models"]
    by_skill: dict[str, dict[str, Any]] = c["by_skill"]
    groups = list(by_skill)

    def val(g: str, m: str) -> P.Bar | None:
        v = by_skill.get(g, {}).get(m)
        return None if v is None else P.Bar(f"{g}: {m}", float(v))

    span = max(
        [float(v) for r in by_skill.values() for m, v in r.items() if m != "n_items"] + [0.01]
    )
    _grouped_bars(
        ax,
        groups,
        [a, b],
        val,
        width=0.38,
        label_small=_fmt_pct_small,
        small_frac=0.03,
        y_top=span,
        label_size=5.5,
    )
    ax.set_xticklabels(
        [_SKILL_TICK.get(g, g).replace("\n", " ") for g in groups],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    S.pct_axis(ax, nbins=4)
    S.light_grid(ax)


def _mini_edl(ax: Axes, cd: ChartData) -> None:
    """rl's EDL* against base's, one point per class (null control excluded): below the diagonal
    means rl's is lower."""
    null = cd.extra.get("null_control")
    pts = []
    for r in cd.rows:
        vals = {n: v for n, v, _c, _z in r.points if v is not None}
        if _clean_label(r.label) == null or "base" not in vals or "rl" not in vals:
            continue
        if vals["base"] > 0 and vals["rl"] > 0:
            pts.append((vals["base"], vals["rl"]))
    if not pts:
        _mini_missing(ax)
        return
    lo = min(min(p) for p in pts) / 2
    hi = max(max(p) for p in pts) * 2
    ax.plot([lo, hi], [lo, hi], color=S.REF, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    st = S.model_style("rl")
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        linestyle="none",
        zorder=3,
        label="_rl",
        **st.marker_kw(3.0),
    )
    S.plain_log_axis(ax, "x", numticks=4)
    S.plain_log_axis(ax, "y", numticks=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("base", labelpad=1)
    ax.set_ylabel("rl", labelpad=1)
    ax.text(
        0.95,
        0.06,
        "y = x",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=S.FS_SMALL - 0.5,
        color=S.REF,
    )


_LIK_SHORT: dict[str, str] = {
    "reserve (neutral)": "reserve",
    "family witnesses": "family",
    "base's own verified proofs": "base's own",
    "neutral minimal proofs": "neutral*",
    "rl's own proofs": "rl's own*",
}


def _mini_likelihood(ax: Axes, cd: ChartData) -> None:
    labels = [_LIK_SHORT.get(r.label.partition(" (n=")[0], r.label) for r in cd.rows]
    _dumbbell(ax, cd.rows, labels=labels, size=3.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(axis="y", pad=1.5)


def _mini_steering(ax: Axes, cd: ChartData) -> None:
    short = []
    span = max([b.hi if b.hi is not None else b.value for b in cd.bars] + [0.01])
    for i, b in enumerate(cd.bars):
        st = _steer_style(b.name)
        kw = st.bar_kw()
        if not st.filled:  # the steered variants: same hue, lighter
            kw["facecolor"] = S.shade(st.color, 0.45)
            kw["hatch"] = kw["hatch"] or None
        ax.bar(i, b.value, 0.62, zorder=2, **kw)
        top = b.value
        if b.lo is not None and b.hi is not None:
            ax.errorbar(
                i,
                b.value,
                yerr=[[max(b.value - b.lo, 0)], [max(b.hi - b.value, 0)]],
                fmt="none",
                ecolor=S.INK2,
                elinewidth=0.5,
                capsize=1.2,
                capthick=0.5,
                zorder=3,
            )
            top = max(top, b.hi)
        if b.value < 0.03 * span:
            ax.text(
                i,
                top + 0.02 * span,
                P.fmt(b.value, "pct"),
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
                color=S.INK2,
            )
        name = b.name
        short.append(
            "base"
            if name.startswith("base (")
            else "base+δ"
            if name.startswith("base +")
            else "rl−δ"
            if name.startswith("rl -")
            else name
        )
    ax.set_xticks(range(len(short)), short)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(short) - 0.4)
    S.pct_axis(ax, nbins=4)
    _ylim_zero(ax)
    S.light_grid(ax)


def _mini_passk(ax: Axes, cd: ChartData) -> None:
    _draw_passk(ax, cd.series, bands=False, size=2.4, every=16)
    ax.set_xlabel("k", labelpad=1)


def _mini_subst(ax: Axes, cd: ChartData) -> None:
    _mini_lines(ax, cd, S.model_style, "s")


def _mini_padding(ax: Axes, cd: ChartData) -> None:
    _mini_lines(ax, cd, _padding_style, "W (lines)")


def _mini_prefix(ax: Axes, cd: ChartData) -> None:
    _mini_lines(ax, cd, S.model_style, "lines left")


def _mini_weights(ax: Axes, cd: ChartData) -> None:
    _draw_weights(ax, cd, size=2.4, annotate=False)
    ax.set_xlabel("α", labelpad=1)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0, 1.5]))
    ax.set_xticklabels(["0", "0.5", "1", "1.5"])


def _mini_probes(ax: Axes, cd: ChartData) -> None:
    _draw_probes(ax, cd, size=2.4, label_ref=False)
    ax.set_xlabel("layer", labelpad=1)


#: (key, panel title, drawer) in the order of the later figures.
OVERVIEW_PANELS: tuple[tuple[str, str, Callable[[Axes, ChartData], None]], ...] = (
    ("yue", "Textbook pass@k", _mini_passk),
    ("families", "Family horizon (lines)", _mini_families),
    ("glue", "Glue ratio R", _mini_glue),
    ("skills", "Skills pass@1", _mini_skills),
    ("subst", "Laws vs substitution s", _mini_subst),
    ("padding", "Padded proofs pass@1", _mini_padding),
    ("prefix", "Prefix completion", _mini_prefix),
    ("edl", "EDL*, rl vs base", _mini_edl),
    ("likelihood", "Log p per line (nats)", _mini_likelihood),
    ("steering", "Steering pass@1", _mini_steering),
    ("weights", "Weight interpolation", _mini_weights),
    ("probes", "Probe accuracy", _mini_probes),
)


def fig_overview(charts: Sequence[ChartData], *, narrow: bool = False) -> Figure:
    by = {c.key: c for c in charts}
    panels = OVERVIEW_PANELS
    if narrow:
        fig = S.new_figure(S.NARROW_W, 8.6)
        axs = fig.subplots(6, 2)
    else:
        fig = S.new_figure(S.FULL_W, 4.55)
        axs = fig.subplots(3, 4)
    for i, (ax, (key, title, draw)) in enumerate(zip(axs.flat, panels, strict=True)):
        cd = by.get(key)
        ax.tick_params(length=2, pad=1.5)
        if cd is None or cd.kind == "missing":
            _mini_missing(ax)
        else:
            draw(ax, cd)
        # after drawing: the drawers set tick labels and axis labels at the figure-text sizes.
        ax.tick_params(labelsize=S.FS_SMALL)
        ax.xaxis.label.set_size(S.FS_SMALL)
        ax.yaxis.label.set_size(S.FS_SMALL)
        S.panel_title(ax, "abcdefghijkl"[i], title, size=S.FS_TICK)
    models = _order(
        list(
            {
                n
                for c in charts
                for n in [s.name for s in c.series] + [n for r in c.rows for n, *_ in r.points]
                if n in S.MODEL_STYLES
            }
            or {"base", "rl"}
        )
    )
    _top_legend(fig, S.legend_handles([S.model_style(m) for m in models]))
    return fig


# ===================================================================================== registry

#: key -> (figure function, width in inches). The page shows each figure at a fixed scale, so
#: its text appears at one size across figures. Full-width figures also have a stacked `narrow`
#: layout (`NARROW`), which the page serves to phones instead of shrinking the wide one.
FIGURES: dict[str, tuple[Callable[[ChartData], Figure], float]] = {
    "yue": (fig_yue, S.FULL_W),
    "families": (fig_families, S.HALF_W),
    "glue": (fig_glue, S.HALF_W),
    "skills": (fig_skills, S.FULL_W),
    "subst": (fig_subst, S.HALF_W),
    "padding": (fig_padding, S.HALF_W),
    "prefix": (fig_prefix, S.HALF_W),
    "edl": (fig_edl, S.FULL_W),
    "likelihood": (fig_likelihood, S.HALF_W),
    "steering": (fig_steering, S.HALF_W),
    "weights": (fig_weights, S.HALF_W),
    "probes": (fig_probes, S.HALF_W),
}
NARROW: dict[str, Callable[[ChartData], Figure]] = {
    "yue": lambda cd: fig_yue(cd, narrow=True),
    "skills": lambda cd: fig_skills(cd, narrow=True),
    "edl": lambda cd: fig_edl(cd, narrow=True),
}


def render(cd: ChartData, *, narrow: bool = False) -> Figure | None:
    """The figure for one benchmark (`None` when it has no data, or when `narrow` is asked for a
    figure that has no narrow layout). Call inside `paper_style()`."""
    if cd.kind == "missing" or cd.key not in FIGURES:
        return None
    if narrow:
        fn = NARROW.get(cd.key)
        return fn(cd) if fn is not None else None
    return FIGURES[cd.key][0](cd)


def svg_data_uri(fig: Figure) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(S.svg_bytes(fig)).decode("ascii")


def figure_uri(cd: ChartData, *, narrow: bool = False) -> tuple[str, float] | None:
    """`(data URI of the SVG, width in inches)` for the summary page, or `None` without data."""
    with S.paper_style():
        fig = render(cd, narrow=narrow)
        if fig is None:
            return None
        try:
            return svg_data_uri(fig), fig.get_figwidth()
        finally:
            S.close(fig)


def overview_uri(charts: Sequence[ChartData], *, narrow: bool = False) -> tuple[str, float] | None:
    if all(c.kind == "missing" for c in charts):
        return None
    with S.paper_style():
        fig = fig_overview(charts, narrow=narrow)
        try:
            return svg_data_uri(fig), fig.get_figwidth()
        finally:
            S.close(fig)


def write_figure(fig: Figure, base: Path) -> list[Path]:
    return S.save(fig, base)
