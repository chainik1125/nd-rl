"""Figures for the Dan-only results page (`dash/dan_only.py`, `figures --dan-only`): Dan
Pandori's twelve best-state checkpoints only, no base/rl/armC anywhere.

Every function takes the `ChartData` `dash/dan_only.py` computed and only draws it, in the
`STYLE.md` rules (NeurIPS widths, Times-like serif, Okabe-Ito): Dan's cap-12 family is reddish
purple and cap 6 sky blue (`style.register_family`, done once in `figures/plots.py`); within a
family fz (Stage-1, pretrained) is the darker shade with solid bars and t1 (after the T1 RL
ladder) the full hue with hatched bars; seeds 1 and 2 are dashed and dotted lines of their seed-0
colour. Non-model series (pass@1 vs pass@128, position-only baselines, Dan's published values,
R = 1) are black or grey. Legends use display names ("pretrained (cap 12)"), the data tables the run
labels ("dan12-fz-s0").

Shared drawing helpers come from `figures/plots.py` (its private `_plot_line`, `_draw_passk`,
`_grouped_bars`, ... -- reused, not copied), so a Dan figure and the paper figure of the same
benchmark draw the same marks.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable, Sequence
from typing import Any, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator

from ..analysis.dan_views import MAX_TRAIN_PREMISES
from ..dash import primitives as P
from ..dash.chart_data import ChartData
from . import plots as PL
from . import style as S

# --------------------------------------------------------------------------------- labels


#: Readers' names for Dan's role codes: fz is the pretrained Stage-1 checkpoint, t1 the same
#: checkpoint after his T1 RL ladder.
ROLE_NAME: dict[str, str] = {"fz": "pretrained", "t1": "RL"}


def display(label: str) -> str:
    """`"dan12-fz-s0"` -> `"pretrained (cap 12)"`; seeds are told apart by line style, not the
    name."""
    family, role, _seed = S.parse_model_label(label)
    if family.startswith("dan") and family[3:].isdigit() and role:
        return f"{ROLE_NAME.get(role, role)} (cap {family[3:]})"
    return label


def dstyle(label: str) -> S.ModelStyle:
    """`style.model_style(label)` under its display name (for legends)."""
    return S.variant(S.model_style(label), display(label))


def _role_order(names: Sequence[str]) -> list[str]:
    """Cap 12 before cap 6, fz before t1, then seed."""

    def key(n: str) -> tuple[int, int, int, str]:
        family, role, seed = S.parse_model_label(n)
        cap = int(family[3:]) if family[3:].isdigit() else 0
        return (-cap, 0 if role == "fz" else 1, seed or 0, n)

    return sorted(names, key=key)


def _seed0(names: Sequence[str]) -> list[str]:
    """One legend entry per (family, role): the seed-0 (or lowest-seed) label."""
    out: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for n in _role_order(names):
        family, role, _seed = S.parse_model_label(n)
        if (family, role) not in seen:
            seen.add((family, role))
            out.append(n)
    return out


def _seed_handles(names: Sequence[str]) -> list[Any]:
    seeds = sorted({s for n in names if (s := S.parse_model_label(n)[2])})
    return [
        Line2D(
            [],
            [],
            color=S.REF,
            linewidth=0.9,
            linestyle=S.SEED_LINESTYLES[s % len(S.SEED_LINESTYLES)],
            label=f"seed {s}",
        )
        for s in seeds
    ]


def _model_legend(
    fig: Figure,
    names: Sequence[str],
    *,
    kind: str = "line",
    ncol: int | None = None,
    extra: Sequence[Any] = (),
) -> None:
    handles = S.legend_handles([dstyle(n) for n in _seed0(names)], kind=kind)
    if kind == "line":
        handles += _seed_handles(names)
    handles += list(extra)
    PL._top_legend(fig, handles, ncol=ncol or min(len(handles), 4))  # noqa: SLF001


def _ref_handle(
    label: str, *, marker: str = "", ls: Any = (0, (4, 1.6)), color: str = S.REF
) -> Any:
    if marker:
        return Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=4.0,
            markerfacecolor=S.WHITE,
            markeredgecolor=color,
            markeredgewidth=0.9,
            label=label,
        )
    return Line2D([], [], color=color, linewidth=0.8, linestyle=ls, label=label)


def _share_y(axs: Sequence[Axes], *, top: float | None = None) -> None:
    hi = top if top is not None else max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(0.0, hi)


# ======================================================================= 1.1 premise ceiling

#: The most premises any training theorem has.
MAXP = MAX_TRAIN_PREMISES
_PREMISE_X = (-0.5, 7.5)
_TRAIN_COLOR = "#8a8a8a"


def _shade_beyond(ax: Axes, *, label: bool = False, x_hi: float = _PREMISE_X[1]) -> None:
    """Grey background over the premise counts no training theorem has."""
    ax.axvspan(MAXP + 0.5, x_hi, color=S.SHADE, linewidth=0, zorder=0)
    if label:
        ax.text(
            (MAXP + 0.5 + x_hi) / 2,
            0.96,
            "not in\ntraining",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=S.FS_TICK - 1,
            color=S.INK2,
        )


def _premise_axis(ax: Axes) -> None:
    ax.set_xlim(*_PREMISE_X)
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, 8))))
    S.light_grid(ax)


def _draw_train(ax: Axes, cd: ChartData, *, size: float = S.FS_TICK) -> None:
    """Share of the cap-12 checkpoints' training theorems (Stage-1 set and RL targets pooled) with
    each number of premises."""
    train = cast("dict[str, dict[int, int]]", cd.extra.get("train") or {})
    pooled: dict[int, int] = {}
    for label, counts in train.items():
        if label.startswith("cap-6"):
            continue
        for k, n in counts.items():
            pooled[k] = pooled.get(k, 0) + n
    tot = sum(pooled.values())
    _shade_beyond(ax)
    if tot:
        ks = sorted(pooled)
        ax.bar(ks, [pooled[k] / tot for k in ks], 0.7, color=_TRAIN_COLOR, zorder=2, linewidth=0)
        ax.text(
            (MAXP + 0.5 + _PREMISE_X[1]) / 2,
            0.5,
            "none",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=size,
            color=S.INK2,
        )
    _premise_axis(ax)
    S.pct_axis(ax, nbins=4)
    PL._ylim_zero(ax)  # noqa: SLF001


def _ceiling_panel(ax: Axes, series: Sequence[P.Series], *, size: float = 3.0) -> list[str]:
    _shade_beyond(ax)
    names: list[str] = []
    for s in series:
        if s.pts:
            PL._plot_line(ax, s.pts, S.model_style(s.name), marker_size=size)  # noqa: SLF001
            names.append(s.name)
    _premise_axis(ax)
    S.pct_axis(ax, nbins=4)
    ax.set_ylim(0, 1.02)
    return names


def fig_ceiling(cd: ChartData, *, narrow: bool = False) -> Figure:
    panels = cast("dict[str, list[P.Series]]", cd.extra.get("suite_panels") or {})
    order = [s for s in cast("list[str]", cd.extra.get("order") or list(panels)) if s in panels]
    n = 1 + len(order)
    cols = 2 if narrow else 4
    rows = math.ceil(n / cols)
    if narrow:
        fig = S.new_figure(S.NARROW_W, 0.55 + 1.6 * rows)
    else:
        fig = S.new_figure(S.FULL_W, 0.45 + 1.6 * rows)
    axs = list(fig.subplots(rows, cols, squeeze=False).flat)
    _draw_train(axs[0], cd)
    S.panel_title(axs[0], "a", "training theorems", size=S.FS_LABEL - 0.5)
    axs[0].set_ylabel("share")
    names: list[str] = []
    for i, suite in enumerate(order, start=1):
        ax = axs[i]
        names += _ceiling_panel(ax, panels[suite])
        S.panel_title(ax, "abcdefghij"[i], suite, size=S.FS_LABEL - 0.5)
        if i % cols == 0:
            ax.set_ylabel("pass@1")
    for i, ax in enumerate(axs[:n]):
        if i >= n - cols:
            ax.set_xlabel("premises")
    for ax in axs[n:]:
        ax.axis("off")
    handles = S.legend_handles([dstyle(m) for m in _seed0(names)])
    handles.append(
        Patch(
            facecolor=S.SHADE,
            edgecolor="none",
            label=f"beyond training (> {MAXP} premises)"
            if narrow
            else "more premises than any training theorem",
        )
    )
    PL._top_legend(fig, handles, ncol=2 if narrow else 3)  # noqa: SLF001
    return fig


def _mini_ceiling(ax: Axes, cd: ChartData) -> None:
    pooled = cast("dict[str, list[tuple[float, float]]]", cd.extra.get("pooled") or {})
    _ceiling_panel(ax, [P.Series(m, pts, "var(--s1)") for m, pts in pooled.items()], size=2.4)
    ax.set_xlabel("premises", labelpad=1)


# ===================================================================== 2. textbook pass@k


def _panels(cd: ChartData) -> dict[str, ChartData]:
    return cast("dict[str, ChartData]", cd.extra.get("panels") or {})


_CAP_TITLE = {"dan12": "cap 12", "dan6": "cap 6"}


def fig_textbook(cd: ChartData, *, narrow: bool = False) -> Figure:
    panels = _panels(cd)
    fams = [f for f in ("dan12", "dan6") if f in panels]
    n = max(len(fams), 1)
    if narrow:
        fig = S.new_figure(S.NARROW_W, 0.55 + 1.75 * n)
        axs = list(fig.subplots(n, 1, squeeze=False)[:, 0])
    else:
        fig = S.new_figure(S.FULL_W, 2.3)
        axs = list(fig.subplots(1, n, squeeze=False, sharey=True)[0])
    names: list[str] = []
    for i, (ax, fam) in enumerate(zip(axs, fams, strict=True)):
        series = panels[fam].series
        if not any(s.pts for s in series):
            PL._mini_missing(ax)  # noqa: SLF001
        else:
            PL._draw_passk(ax, series, bands=True)  # noqa: SLF001
            names += [s.name for s in series if s.pts]
        ax.set_xlabel("k (samples per theorem)")
        if i == 0 or narrow:
            ax.set_ylabel("pass@k")
        S.panel_title(
            ax,
            "ab"[i] if n > 1 else None,
            f"{_CAP_TITLE.get(fam, fam)}, n = {cd.extra.get('n', '?')} theorems",
        )
    _share_y(axs)
    _model_legend(fig, names, ncol=3 if narrow else 6)
    return fig


def _mini_textbook(ax: Axes, cd: ChartData) -> None:
    pair = cast("list[P.Series]", cd.extra.get("pair") or [])
    PL._draw_passk(ax, pair, bands=False, size=2.4, every=16)  # noqa: SLF001
    ax.set_xlabel("k", labelpad=1)


# ============================================================== textbook72 (overview panel)


def _mini_textbook72(ax: Axes, cd: ChartData) -> None:
    """Theorems solved of 72 at k = 256 per cap and role: bar = seed mean, dots = seeds, black
    ticks = Dan's published per-seed counts."""
    t72 = cast("dict[str, dict[str, Any]]", cd.extra.get("t72") or {})
    groups = [g for g in ("dan6", "dan12") if any(k.startswith(g + "-") for k in t72)]
    width = 0.36
    for gi, fam in enumerate(groups):
        for j, role in enumerate(("fz", "t1")):
            key = f"{fam}-{role}"
            info = t72.get(key)
            if not info:
                continue
            st = S.model_style(f"{fam}-{role}-s0")
            x = gi + (j - 0.5) * width
            vals = cast("list[float]", info["ours"])
            ax.bar(x, sum(vals) / len(vals), width * 0.9, zorder=2, **st.bar_kw())
            for v in vals:
                ax.plot(
                    [x],
                    [v],
                    linestyle="none",
                    zorder=4,
                    marker="o",
                    markersize=1.9,
                    markerfacecolor=S.WHITE,
                    markeredgecolor=S.INK,
                    markeredgewidth=0.4,
                )
            for v in cast("list[float]", info.get("dan") or []):
                ax.plot(
                    [x - width * 0.42, x + width * 0.42],
                    [v, v],
                    color=S.INK,
                    linewidth=0.5,
                    zorder=3,
                )
    ax.set_xticks(range(len(groups)), [_CAP_TITLE[g] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 72)
    ax.yaxis.set_major_locator(FixedLocator([0, 24, 48, 72]))
    S.light_grid(ax)


# ============================================================================== 3. families


def fig_families(cd: ChartData) -> Figure:
    order = list(cd.extra.get("row_order") or [r.label for r in cd.rows])
    by_label = {r.label: r for r in cd.rows}
    models = _role_order(list({n for r in cd.rows for n, _v, _c, _z in r.points}))
    offs = dict(zip(models, PL._row_offsets(len(models)), strict=True))  # noqa: SLF001
    fig = S.new_figure(S.HALF_W, 0.72 + 0.2 * len(order))
    ax = fig.add_subplot()
    x_hi = float(cd.extra.get("x_hi") or 1.0)
    _dot_rows(ax, order, by_label, offs, x_hi, size=4.2)
    _out_of_range(ax, order, cd, f"out of range: every item has ≥ {MAXP + 1} premises")
    ax.set_xlabel("written lines at n* (unseen theorems)")
    handles = S.legend_handles([dstyle(m) for m in models], kind="marker")
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            label="fails at smallest size",
            **S.ModelStyle("x", S.REF, "o", filled=False).marker_kw(3.6),
        )
    )
    PL._top_legend(fig, handles, ncol=2)  # noqa: SLF001
    return fig


def _out_of_range(ax: Axes, order: Sequence[str], cd: ChartData, text: str) -> None:
    """Write `text` across the rows of families the premise ceiling leaves no item in."""
    for label in cast("list[str]", cd.extra.get("out_of_range") or []):
        if label in order:
            ax.text(
                0.0,
                order.index(label),
                text,
                ha="left",
                va="center",
                fontsize=S.FS_TICK - 1,
                color=S.REF,
                style="italic",
            )


def _dot_rows(
    ax: Axes,
    order: Sequence[str],
    by_label: dict[str, P.DotRow],
    offs: dict[str, float],
    x_hi: float,
    *,
    size: float,
) -> None:
    for i, label in enumerate(order):
        row = by_label.get(label)
        pts = {n: (v, z) for n, v, _c, z in row.points} if row is not None else {}
        vals = [v for v, _z in pts.values() if v is not None]
        if len(pts) >= 2:
            lo = min([0.0 if v is None else v for v, _z in pts.values()])
            hi = max([0.0 if v is None else v for v, _z in pts.values()])
            ax.plot([lo, hi], [i, i], color="#cfcfcf", linewidth=0.9, zorder=1)
        del vals
        for name, (v, censored) in pts.items():
            st = S.model_style(name)
            y = i + offs.get(name, 0.0)
            if v is None:
                ax.plot(
                    [0.0],
                    [y],
                    linestyle="none",
                    zorder=4,
                    **S.variant(st, name, filled=False).marker_kw(size * 0.86),
                )
                continue
            ax.plot([v], [y], linestyle="none", zorder=4, **st.marker_kw(size))
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
    ax.set_yticks(range(len(order)), list(order))
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-0.06 * x_hi - 0.3, x_hi * 1.16 + 0.3)
    PL._integer_x(ax)  # noqa: SLF001
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    S.light_grid(ax, "x")


def _mini_families(ax: Axes, cd: ChartData) -> None:
    order = list(cd.extra.get("row_order") or [r.label for r in cd.rows])
    by_label = {r.label: r for r in cd.rows}
    models = _role_order(list({n for r in cd.rows for n, _v, _c, _z in r.points}))
    offs = dict(zip(models, PL._row_offsets(len(models), spread=0.18), strict=True))  # noqa: SLF001
    _dot_rows(ax, order, by_label, offs, float(cd.extra.get("x_hi") or 1.0), size=2.8)
    _out_of_range(ax, order, cd, "out of range")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.set_yticklabels([_short_family(o) for o in order])
    ax.tick_params(axis="y", pad=1.5, labelsize=S.FS_SMALL - 0.5)
    ax.set_xlabel("lines at n*", labelpad=1)


def _short_family(label: str) -> str:
    return {"ILTP-style": "ILTP", "reorder_and": "reord_and", "and_extract": "and_extr"}.get(
        label, label
    )


# ==================================================================== 4/8. simple line figs


def fig_laws(cd: ChartData) -> Figure:
    return PL._lines_figure(cd, dstyle, x_label="s (connectives per metavariable)")  # noqa: SLF001


def fig_prefix(cd: ChartData) -> Figure:
    return PL._lines_figure(cd, dstyle, x_label="lines left to write (L - j)")  # noqa: SLF001


def _mini_lines(ax: Axes, cd: ChartData, x_label: str) -> None:
    for s in cd.series:
        if s.pts:
            PL._plot_line(ax, s.pts, S.model_style(s.name), band=s.band, marker_size=2.4)  # noqa: SLF001
    PL._integer_x(ax, nbins=5)  # noqa: SLF001
    S.pct_axis(ax, nbins=4)
    PL._ylim_zero(ax)  # noqa: SLF001
    S.light_grid(ax)
    ax.set_xlabel(x_label, labelpad=1)


# ================================================================================= 5. glue


def fig_glue(cd: ChartData) -> Figure:
    lookup = PL._bar_lookup(cd.bars)  # noqa: SLF001
    ops = list(cd.extra.get("ops") or dict.fromkeys(g for g, _m in lookup))
    counts = cast("dict[str, int]", cd.extra.get("counts") or {})
    models = _role_order(list(dict.fromkeys(m for _g, m in lookup)))
    tops = [b.hi if b.hi is not None and b.hi == b.hi else b.value for b in cd.bars]
    y_top = max([1.1, *(t * 1.08 for t in tops)])
    fig = S.new_figure(S.HALF_W, 2.1)
    ax = fig.add_subplot()
    PL._grouped_bars(
        ax,
        ops,
        models,
        lambda g, m: lookup.get((g, m)),  # noqa: SLF001
        label_small=PL._fmt_r,
        y_top=y_top,
        styles=S.model_style,
    )  # noqa: SLF001
    ax.set_xticklabels([f"{op}\n(n = {counts[op]})" if op in counts else op for op in ops])
    ax.axhline(1.0, color=S.REF, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax.text(len(ops) - 0.42, 1.0, "R = 1", ha="right", va="bottom", fontsize=S.FS_TICK, color=S.REF)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_ylabel("compositionality ratio R")
    S.light_grid(ax)
    _model_legend(fig, models, kind="bar")
    return fig


def _mini_glue(ax: Axes, cd: ChartData) -> None:
    lookup = PL._bar_lookup(cd.bars)  # noqa: SLF001
    ops = list(cd.extra.get("ops") or dict.fromkeys(g for g, _m in lookup))
    models = _role_order(list(dict.fromkeys(m for _g, m in lookup)))
    tops = [b.hi if b.hi is not None and b.hi == b.hi else b.value for b in cd.bars]
    y_top = max([1.1, *(t * 1.06 for t in tops if t == t)])
    PL._grouped_bars(
        ax,
        ops,
        models,
        lambda g, m: lookup.get((g, m)),
        width=0.38,  # noqa: SLF001
        label_small=None,
        y_top=y_top,
        styles=S.model_style,
    )
    ax.set_xticklabels(ops, rotation=30, ha="right", rotation_mode="anchor")
    ax.axhline(1.0, color=S.REF, linewidth=0.5, linestyle=(0, (1, 1.5)), zorder=1)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    S.light_grid(ax)


# =============================================================================== 6. skills

SKILL_ORDER: tuple[str, ...] = ("plain", "case_split", "classical", "classical+case_split")
_SKILL_TICK = {
    "plain": "plain",
    "case_split": "case\nsplit",
    "classical": "classical",
    "classical+case_split": "classical\n+ case",
}


def _skill_variant(model: str, group: str) -> S.ModelStyle:
    """One model's hue for every skill line, told apart by shade, marker and dash."""
    st = S.model_style(model)
    label = group.replace("_", " ")
    if group == "case_split":
        return S.variant(
            st, label, color=S.shade(st.color, -0.3), marker="D", linestyle=(0, (4, 1.6))
        )
    if group == "classical":
        return S.variant(
            st, label, color=S.shade(st.color, 0.35), marker="^", linestyle=(0, (1, 1.2))
        )
    if group == "classical+case_split":
        return S.variant(
            st,
            label,
            color=S.shade(st.color, 0.15),
            marker="v",
            linestyle=(0, (5, 1.5, 1, 1.5)),
            filled=False,
        )
    return S.variant(st, label, linestyle="-")


def fig_skills(cd: ChartData, *, narrow: bool = False) -> Figure:
    lookup = PL._bar_lookup(cd.bars)  # noqa: SLF001
    groups = [g for g in SKILL_ORDER if any(gg == g for gg, _m in lookup)]
    models = _role_order(list(dict.fromkeys(m for _g, m in lookup)))
    by_len = cast("dict[str, list[P.Series]]", cd.extra.get("by_length") or {})
    line_models = [m for m in models if by_len.get(m)]
    n_line = len(line_models)
    if narrow:
        fig = S.new_figure(S.NARROW_W, 0.75 + 1.7 * (1 + n_line))
        axs = list(fig.subplots(1 + n_line, 1, squeeze=False)[:, 0])
    else:
        fig = S.new_figure(S.FULL_W, 2.25)
        ratios = [1.25] + [1.0] * n_line
        axs = list(
            fig.subplots(1, 1 + n_line, squeeze=False, gridspec_kw={"width_ratios": ratios})[0]
        )
    ax = axs[0]
    PL._grouped_bars(
        ax,
        groups,
        models,
        lambda g, m: lookup.get((g, m)),
        width=0.36,  # noqa: SLF001
        label_small=PL._fmt_pct_small,
        small_frac=0.03,
        y_top=1.0,  # noqa: SLF001
        styles=S.model_style,
    )
    ax.set_xticklabels([_SKILL_TICK.get(g, g) for g in groups], fontsize=S.FS_TICK - 0.5)
    S.pct_axis(ax)
    ax.set_ylabel("pass@1")
    S.light_grid(ax)
    S.panel_title(ax, "a" if n_line else None, "mean over each group's items")
    for i, m in enumerate(line_models, start=1):
        axl = axs[i]
        styles = []
        for s in sorted(
            by_len[m], key=lambda s: SKILL_ORDER.index(s.name) if s.name in SKILL_ORDER else 99
        ):
            st = _skill_variant(m, s.name)
            PL._plot_line(axl, s.pts, st, marker_size=3.0)  # noqa: SLF001
            styles.append(st)
        PL._integer_x(axl)  # noqa: SLF001
        S.pct_axis(axl)
        S.light_grid(axl)
        axl.set_xlabel("L (written lines)")
        if narrow:
            axl.set_ylabel("pass@1")
        S.panel_title(axl, "abc"[i], f"{display(m)}, by length")
        axl.legend(
            handles=S.legend_handles(styles),
            loc="best",
            handlelength=2.2,
            fontsize=S.FS_TICK - 0.5,
        )
    _share_y(axs, top=max([1.0] + [ax.get_ylim()[1] for ax in axs[1:]]) if not narrow else None)
    _model_legend(fig, models, kind="bar")
    return fig


def _mini_skills(ax: Axes, cd: ChartData) -> None:
    lookup = PL._bar_lookup(cd.bars)  # noqa: SLF001
    groups = [g for g in SKILL_ORDER if any(gg == g for gg, _m in lookup)]
    models = _role_order(list(dict.fromkeys(m for _g, m in lookup)))
    PL._grouped_bars(
        ax,
        groups,
        models,
        lambda g, m: lookup.get((g, m)),
        width=0.38,  # noqa: SLF001
        label_small=None,
        y_top=1.0,
        styles=S.model_style,
    )
    short = {
        "plain": "plain",
        "case_split": "case",
        "classical": "class.",
        "classical+case_split": "cl.+case",
    }
    ax.set_xticklabels(
        [short.get(g, g) for g in groups], rotation=30, ha="right", rotation_mode="anchor"
    )
    S.pct_axis(ax, nbins=4)
    PL._ylim_zero(ax)  # noqa: SLF001
    S.light_grid(ax)


# ============================================================================== 7. padding

PAD_SHORT = {"P2": "P2", "P1-shared": "P1-shared", "P1-disjoint": "P1-disjoint"}
#: Pad types told apart by line style and marker fill within a model's colour.
_PAD_LINE: dict[str, dict[str, Any]] = {
    "P1-shared": {"linestyle": "-"},
    "P1-disjoint": {"linestyle": (0, (4, 1.6)), "filled": False},
    "P2": {"linestyle": (0, (1, 1.2)), "marker": "^"},
}


def _pad_style(model: str, pad_type: str) -> S.ModelStyle:
    return S.variant(S.model_style(model), model, **_PAD_LINE.get(pad_type, {}))


def _pad_handles(types: Sequence[str]) -> list[Any]:
    """Grey legend entries for the pad types (the colour is the model's)."""
    return S.legend_handles(
        [S.variant(S.ModelStyle(t, S.REF, "o"), t, **_PAD_LINE.get(t, {})) for t in types]
    )


def fig_padding(cd: ChartData, *, narrow: bool = False) -> Figure:
    """(a) pass@1 against total premises, one line per (model, pad type), the premise counts
    beyond training shaded; (b) P2 against pads added."""
    by: dict[str, dict[str, P.Series]] = {}
    for s in cd.series:
        t, _sep, m = s.name.partition(": ")
        by.setdefault(m, {})[t] = s
    p2 = cast("list[P.Series]", cd.extra.get("p2") or [])
    if narrow:
        fig = S.new_figure(S.NARROW_W, 0.75 + 1.75 * 2)
        axs = list(fig.subplots(2, 1, squeeze=False)[:, 0])
    else:
        fig = S.new_figure(S.FULL_W, 2.15)
        axs = list(fig.subplots(1, 2, squeeze=False, gridspec_kw={"width_ratios": [1.45, 1.0]})[0])
    ax = axs[0]
    _shade_beyond(ax, label=True)
    models = _role_order(list(by))
    types = [t for t in ("P1-shared", "P1-disjoint", "P2") if any(t in by[m] for m in models)]
    for m in models:
        for t in types:
            s = by[m].get(t)
            if s is not None and s.pts:
                PL._plot_line(ax, s.pts, _pad_style(m, t), marker_size=3.2)  # noqa: SLF001
    ax.set_xlim(0.5, _PREMISE_X[1])
    ax.xaxis.set_major_locator(FixedLocator(list(range(1, 8))))
    S.pct_axis(ax)
    ax.set_ylim(0, 1.02)
    S.light_grid(ax)
    ax.set_xlabel("premises in all (the theorem's own + P1 pads)")
    ax.set_ylabel("pass@1")
    S.panel_title(ax, "a", "by total premises, padded items", size=S.FS_LABEL - 0.5)
    axb = axs[1]
    for s in p2:
        if s.pts:
            PL._plot_line(axb, s.pts, S.model_style(s.name), marker_size=3.2)  # noqa: SLF001
    PL._integer_x(axb)  # noqa: SLF001
    S.pct_axis(axb)
    axb.set_ylim(0, 1.02)
    S.light_grid(axb)
    axb.set_xlabel("P2 pads added m (+1 ∧I line each)")
    if narrow:
        axb.set_ylabel("pass@1")
    S.panel_title(axb, "b", "P2: a longer proof and goal", size=S.FS_LABEL - 0.5)
    handles = S.legend_handles([dstyle(m) for m in _seed0(models)]) + _pad_handles(types)
    PL._top_legend(fig, handles, ncol=3 if narrow else 5)  # noqa: SLF001
    return fig


def _mini_padding(ax: Axes, cd: ChartData) -> None:
    """Both P1 kinds pooled, against total premises, per model."""
    pooled = cast("dict[str, list[tuple[float, float]]]", cd.extra.get("mini") or {})
    _shade_beyond(ax)
    for model, pts in pooled.items():
        if pts:
            PL._plot_line(  # noqa: SLF001
                ax,
                pts,
                S.variant(S.model_style(model), model, linestyle=(0, (1, 1.2)), filled=False),
                marker_size=2.4,
            )
    ax.set_xlim(0.5, _PREMISE_X[1])
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 5, 7]))
    S.pct_axis(ax, nbins=4)
    ax.set_ylim(0, 1.02)
    S.light_grid(ax)
    ax.set_xlabel("premises", labelpad=1)


# ======================================================================= 2b. failure modes


def _mini_failures(ax: Axes, cd: ChartData) -> None:
    lookup = PL._bar_lookup(cd.bars)  # noqa: SLF001
    groups = list(dict.fromkeys(g for g, _m in lookup))
    models = _role_order(list(dict.fromkeys(m for _g, m in lookup)))
    PL._grouped_bars(  # noqa: SLF001
        ax,
        groups,
        models,
        lambda g, m: lookup.get((g, m)),
        width=0.38,
        label_small=None,
        y_top=1.0,
        styles=S.model_style,
    )
    ax.set_xticklabels(groups, rotation=30, ha="right", rotation_mode="anchor")
    S.pct_axis(ax, nbins=4)
    PL._ylim_zero(ax)  # noqa: SLF001
    S.light_grid(ax)


# =================================================================================== 11a. EDL


def fig_edl(cd: ChartData, *, narrow: bool = False) -> Figure:
    """`plots.fig_edl`'s drawing (EDL* per class with the order spread; EDL* against starting
    loss), with the legend under display names."""
    fig = PL.fig_edl(cd, narrow=narrow)
    for leg in list(fig.legends):
        leg.remove()
    models = _role_order(list({n for r in cd.rows for n, _v, _c, _z in r.points}))
    _model_legend(fig, models, kind="marker")
    return fig


def _mini_edl(ax: Axes, cd: ChartData) -> None:
    """RL's EDL* against pretrained's, one point per class (null control excluded): below the
    diagonal means RL's is lower."""
    null = cd.extra.get("null_control")
    models = _role_order(list({n for r in cd.rows for n, _v, _c, _z in r.points}))
    fz = next((m for m in models if S.parse_model_label(m)[1] == "fz"), None)
    t1 = next((m for m in models if S.parse_model_label(m)[1] == "t1"), None)
    pts = []
    for r in cd.rows:
        vals = {n: v for n, v, _c, _z in r.points if v is not None}
        if r.label.replace(" (null control)", "") == null or fz not in vals or t1 not in vals:
            continue
        if vals[fz] > 0 and vals[t1] > 0:
            pts.append((vals[fz], vals[t1]))
    if not pts or t1 is None:
        PL._mini_missing(ax)  # noqa: SLF001
        return
    lo = min(min(p) for p in pts) / 2
    hi = max(max(p) for p in pts) * 2
    ax.plot([lo, hi], [lo, hi], color=S.REF, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        linestyle="none",
        zorder=3,
        **S.model_style(t1).marker_kw(3.0),
    )
    S.plain_log_axis(ax, "x", numticks=4)
    S.plain_log_axis(ax, "y", numticks=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("pretrained", labelpad=1)
    ax.set_ylabel("RL", labelpad=1)


# ============================================================================ 9. held-out


def _forest(
    ax: Axes,
    bars: Sequence[P.Bar],
    published: dict[str, float],
    *,
    size: float,
    labels: Sequence[str],
) -> None:
    for i, b in enumerate(bars):
        st = S.model_style(b.name)
        if b.lo is not None and b.hi is not None:
            ax.plot([b.lo, b.hi], [i, i], color=st.color, linewidth=0.9, zorder=2)
            for x in (b.lo, b.hi):
                ax.plot([x, x], [i - 0.1, i + 0.1], color=st.color, linewidth=0.9, zorder=2)
        ax.plot([b.value], [i], linestyle="none", zorder=3, **st.marker_kw(size))
        if b.name in published:
            ax.plot(
                [published[b.name]],
                [i + 0.26],
                linestyle="none",
                zorder=3,
                marker="v",
                markersize=size * 0.8,
                markerfacecolor=S.WHITE,
                markeredgecolor=S.INK,
                markeredgewidth=0.8,
            )
    ax.set_yticks(range(len(bars)), list(labels))
    ax.set_ylim(len(bars) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    S.light_grid(ax, "x")


def fig_heldout(cd: ChartData) -> Figure:
    published = cast("dict[str, float]", cd.extra.get("published") or {})
    bars = sorted(cd.bars, key=lambda b: _role_order([x.name for x in cd.bars]).index(b.name))
    fig = S.new_figure(S.HALF_W, 0.85 + 0.32 * len(bars))
    ax = fig.add_subplot()
    _forest(ax, bars, published, size=4.6, labels=[display(b.name) for b in bars])
    vals = [v for b in bars for v in (b.lo, b.hi, b.value) if v is not None]
    vals += list(published.values())
    lo = math.floor(min(vals) * 100 / 2) * 2 / 100 - 0.01
    ax.set_xlim(lo, 1.0)
    S.pct_axis(ax, "x")
    ax.set_xlabel("held-out greedy accuracy (95% CI)")
    PL._top_legend(  # noqa: SLF001
        fig, [_ref_handle("Dan's reported value", marker="v", color=S.INK)], ncol=1
    )
    return fig


def _mini_heldout(ax: Axes, cd: ChartData) -> None:
    published = cast("dict[str, float]", cd.extra.get("published") or {})
    bars = sorted(cd.bars, key=lambda b: _role_order([x.name for x in cd.bars]).index(b.name))
    _forest(
        ax,
        bars,
        published,
        size=3.2,
        labels=[ROLE_NAME.get(S.parse_model_label(b.name)[1] or "", b.name) for b in bars],
    )
    vals = [v for b in bars for v in (b.lo, b.hi, b.value) if v is not None]
    ax.set_xlim(math.floor(min(vals + list(published.values())) * 50) / 50 - 0.01, 1.0)
    S.pct_axis(ax, "x", nbins=3)


# ===================================================================== 10. textbook_law


def fig_textbook_law(cd: ChartData) -> Figure:
    fig = S.new_figure(S.HALF_W, 2.0)
    ax = fig.add_subplot()
    PL._draw_passk(ax, cd.series, bands=True, every=2)  # noqa: SLF001
    ax.set_xlabel("k (samples per theorem)")
    ax.set_ylabel("pass@k")
    ax.legend(
        handles=S.legend_handles([dstyle(s.name) for s in cd.series if s.pts]), loc="center right"
    )
    return fig


def _mini_textbook_law(ax: Axes, cd: ChartData) -> None:
    PL._draw_passk(ax, cd.series, bands=False, size=2.4, every=4)  # noqa: SLF001
    ax.set_xlabel("k", labelpad=1)


# ======================================================================= 11. likelihood


def fig_likelihood(cd: ChartData) -> Figure:
    rows = cd.rows
    fig = S.new_figure(S.HALF_W, 0.8 + 0.36 * len(rows))
    ax = fig.add_subplot()
    models = PL._dumbbell(ax, rows, labels=[PL._two_line(r.label) for r in rows])  # noqa: SLF001
    ax.set_xlabel("median log p per written line (nats)")
    vals = [v for r in rows for _n, v, _c, _z in r.points if v is not None]
    lo = min(vals, default=-1.0)
    ax.set_xlim(min(lo * 1.1, lo - 0.4), 0.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    _model_legend(fig, models, kind="marker")
    return fig


def _mini_likelihood(ax: Axes, cd: ChartData) -> None:
    labels = [r.label.partition(" (n=")[0].split(" ")[0] for r in cd.rows]
    PL._dumbbell(ax, cd.rows, labels=labels, size=3.2)  # noqa: SLF001
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="y", pad=1.5)


# ============================================================================== 12. KL


def _kl_style(name: str) -> S.ModelStyle:
    model = name.split("|", 1)[1] if "|" in name else name
    return S.model_style(model)


def _kl_draw(ax: Axes, cd: ChartData, *, size: float) -> list[str]:
    rows = cd.rows
    names: list[str] = []
    for i, row in enumerate(rows):
        vals = [v for _n, v, _c, _z in row.points if v is not None]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [i, i], color="#cfcfcf", linewidth=0.9, zorder=1)
        for name, v, _c, _z in row.points:
            if v is None:
                continue
            ax.plot([v], [i], linestyle="none", zorder=3, **_kl_style(name).marker_kw(size))
            if name not in names:
                names.append(name)
    ax.set_yticks(range(len(rows)), [r.label for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    hi = max([v for r in rows for _n, v, _c, _z in r.points if v is not None] + [0.01])
    ax.set_xlim(0, hi * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    S.light_grid(ax, "x")
    return names


def fig_kl(cd: ChartData) -> Figure:
    fig = S.new_figure(S.HALF_W, 0.95 + 0.165 * len(cd.rows))
    ax = fig.add_subplot()
    names = _kl_draw(ax, cd, size=3.8)
    if len(cd.rows) > 1:  # the pooled row sits apart from the per-kind rows
        ax.axhline(0.5, color=S.INK2, linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", labelsize=S.FS_TICK - 0.5)
    ax.set_xlabel("KL per token (nats)")
    PL._top_legend(fig, _kl_handles(names), ncol=1)  # noqa: SLF001
    return fig


def _kl_handles(names: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            # Nimbus Roman has no U+2016; draw the norm bars with mathtext (STIX)
            label=n.split("|", 1)[0].replace("‖", r"$\Vert$"),
            **_kl_style(n).marker_kw(4.2),
        )
        for n in names
    ]


def fig_kl_wide(cd: ChartData) -> Figure:
    """Full-width layout: the pooled column, then one column per action kind (largest KL
    first), KL per token on y, one marker per direction joined by a grey line."""
    rows = cd.rows
    fig = S.new_figure(S.FULL_W, 2.3)
    ax = fig.add_subplot()
    names: list[str] = []
    for i, row in enumerate(rows):
        vals = [v for _n, v, _c, _z in row.points if v is not None]
        if len(vals) >= 2:
            ax.plot([i, i], [min(vals), max(vals)], color="#cfcfcf", linewidth=0.9, zorder=1)
        for name, v, _c, _z in row.points:
            if v is None:
                continue
            ax.plot([i], [v], linestyle="none", zorder=3, **_kl_style(name).marker_kw(4.0))
            if name not in names:
                names.append(name)
    ax.set_xticks(
        range(len(rows)),
        [r.label for r in rows],
        rotation=40,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="x", length=0, pad=5, labelsize=S.FS_TICK - 0.5)
    if len(rows) > 1:  # the pooled column sits apart from the per-kind columns
        ax.axvline(0.5, color=S.INK2, linewidth=0.5, zorder=0)
    hi = max([v for r in rows for _n, v, _c, _z in r.points if v is not None] + [0.01])
    ax.set_ylim(0, hi * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    S.light_grid(ax, "y")
    ax.set_ylabel("KL per token (nats)")
    PL._top_legend(fig, _kl_handles(names), ncol=2)  # noqa: SLF001
    return fig


def _mini_kl(ax: Axes, cd: ChartData) -> None:
    """The pooled row and the four kinds with the largest KL."""
    rows = cd.rows[:5]
    _kl_draw(ax, ChartData(cd.key, cd.title, "dot", rows=rows), size=3.0)
    ax.set_yticklabels(["all" if r.label == "all tokens" else r.label for r in rows])
    ax.tick_params(axis="y", pad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))


# ========================================================================= 13. steering


def _steer_bar_style(name: str) -> tuple[S.ModelStyle, bool]:
    """`(style, steered)` for a steering condition named "pretrained", "pretrained + δ", "RL − δ" or
    "RL"."""
    model = name.split("|", 1)[1]
    return S.model_style(model), ("δ" in name)


def _mini_steering(ax: Axes, cd: ChartData) -> None:
    lookup = PL._bar_lookup(cd.bars)  # noqa: SLF001
    groups = list(dict.fromkeys(g for g, _m in lookup))
    conds = list(dict.fromkeys(m for _g, m in lookup))
    width = 0.8 / max(len(conds), 1)
    for j, cond in enumerate(conds):
        st, steered = _steer_bar_style(cond)
        off = (j - (len(conds) - 1) / 2) * width
        for i, g in enumerate(groups):
            b = lookup.get((g, cond))
            if b is None:
                continue
            kw = st.bar_kw()
            kw.pop("label", None)
            if steered:
                kw["facecolor"] = S.shade(st.color, 0.45)
            ax.bar(i + off, b.value, width * 0.9, zorder=2, **kw)
            if b.lo is not None and b.hi is not None:
                ax.errorbar(
                    i + off,
                    b.value,
                    yerr=[[b.value - b.lo], [b.hi - b.value]],
                    fmt="none",
                    ecolor=S.INK2,
                    elinewidth=0.5,
                    capsize=1.0,
                    capthick=0.5,
                    zorder=3,
                )
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    S.pct_axis(ax, nbins=4)
    PL._ylim_zero(ax)  # noqa: SLF001
    S.light_grid(ax)


# ===================================================================== 14. interpolation

_PASS1 = S.ModelStyle("pass@1", S.INK, "o")
_PASS128 = S.ModelStyle("pass@128", S.REF, "s", linestyle=(0, (4, 1.6)), filled=False)


def _draw_weights(ax: Axes, cd: ChartData, *, size: float = 3.6, annotate: bool = True) -> None:
    by = PL._series(cd)  # noqa: SLF001
    xs = sorted({x for s in by.values() for x, _ in s.pts})
    if xs and xs[-1] > 1.0:
        ax.axvspan(1.0, xs[-1] + 0.1, color=S.SHADE, linewidth=0, zorder=0)
        if annotate:
            ax.text(
                (1.0 + xs[-1]) / 2 + 0.05,
                0.04,
                "extrapo-\nlation",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=S.FS_TICK - 0.5,
                color=S.INK2,
            )
    for name, st in (("pass@128", _PASS128), ("pass@1", _PASS1)):
        if name in by:
            PL._plot_line(ax, by[name].pts, st, band=by[name].band, marker_size=size)  # noqa: SLF001
    ticks = [x for x in xs if abs(x * 4 - round(x * 4)) < 1e-9]
    ax.xaxis.set_major_locator(FixedLocator(ticks or xs))
    labels = []
    for t in ticks or xs:
        lab = f"{t:g}"
        if annotate and t == 0:
            lab += "\npretrained"
        elif annotate and t == 1:
            lab += "\nRL"
        labels.append(lab)
    ax.set_xticklabels(labels)
    if xs:
        ax.set_xlim(xs[0] - 0.06, xs[-1] + 0.06)
    S.pct_axis(ax)
    PL._ylim_zero(ax)  # noqa: SLF001
    S.light_grid(ax)


def fig_weights(cd: ChartData) -> Figure:
    fig = S.new_figure(S.HALF_W, 2.0)
    ax = fig.add_subplot()
    _draw_weights(ax, cd)
    ax.set_xlabel("interpolation weight α")
    ax.set_ylabel("pass@k")
    by = PL._series(cd)  # noqa: SLF001
    ax.legend(
        handles=S.legend_handles(
            [st for n, st in (("pass@1", _PASS1), ("pass@128", _PASS128)) if n in by]
        ),
        loc="lower center",
    )
    return fig


def _mini_weights(ax: Axes, cd: ChartData) -> None:
    _draw_weights(ax, cd, size=2.4, annotate=False)
    ax.set_xlabel("α", labelpad=1)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.set_xticklabels(["0", "0.5", "1"])


# ============================================================================ 15. probes


def _probe_draw(ax: Axes, cd: ChartData, metric: str, *, size: float, label_ref: bool) -> list[str]:
    baselines = cast("dict[str, list[tuple[str, float]]]", cd.extra.get("baselines") or {})
    names: list[str] = []
    for s in cd.series:
        if not s.pts or not s.name.startswith(metric + ": "):
            continue
        model = s.name.split(": ", 1)[1]
        PL._plot_line(ax, s.pts, S.model_style(model), marker_size=size)  # noqa: SLF001
        names.append(model)
    refs = baselines.get(metric, [])
    for j, (label, y) in enumerate(refs):
        ls = (0, (4, 1.6)) if j == len(refs) - 1 else (0, (1, 1.3))
        ax.axhline(y, color=S.REF, linewidth=0.8, linestyle=ls, zorder=1)
        if label_ref:
            ax.text(
                0.98,
                y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom" if j == len(refs) - 1 else "top",
                fontsize=S.FS_TICK - 1,
                color=S.REF,
            )
    xs = sorted({x for s in cd.series for x, _ in s.pts})
    if xs:
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.set_xticklabels([f"L{int(x)}" for x in xs])
        ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25)
    S.light_grid(ax)
    return names


def fig_probes(cd: ChartData, *, narrow: bool = False) -> Figure:
    if narrow:
        fig = S.new_figure(S.NARROW_W, 3.95)
        axs = list(fig.subplots(2, 1, squeeze=False)[:, 0])
    else:
        fig = S.new_figure(S.FULL_W, 2.1)
        axs = list(fig.subplots(1, 2, squeeze=False)[0])
    names = _probe_draw(axs[0], cd, "next_rule", size=3.6, label_ref=False)
    S.pct_axis(axs[0])
    axs[0].set_ylabel("held-out accuracy")
    S.panel_title(axs[0], "a", "next_rule (rule of the next action)")
    _probe_draw(axs[1], cd, "actions_to_go", size=3.6, label_ref=False)
    axs[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[1].set_ylabel("held-out R²")
    S.panel_title(axs[1], "b", "actions_to_go (actions left)")
    for ax in axs:
        ax.set_xlabel("layer")
    _model_legend(
        fig,
        names,
        ncol=2 if narrow else 4,
        extra=[
            _ref_handle("lookup: step + premises", ls=(0, (1, 1.3))),
            _ref_handle("lookup: + frame depth", ls=(0, (4, 1.6))),
        ],
    )
    return fig


def _mini_probes(ax: Axes, cd: ChartData) -> None:
    _probe_draw(ax, cd, "next_rule", size=2.4, label_ref=False)
    xs = sorted({x for s in cd.series for x, _ in s.pts})
    ax.xaxis.set_major_locator(FixedLocator(xs[::2] if len(xs) > 4 else xs))
    ax.set_xticklabels([f"{int(x)}" for x in (xs[::2] if len(xs) > 4 else xs)])
    S.pct_axis(ax, nbins=4)
    ax.set_xlabel("layer", labelpad=1)


# ======================================================================= Figure 1 overview

#: (key, panel title, drawer), in the order of the page's later figures.
OVERVIEW_PANELS: tuple[tuple[str, str, Callable[[Axes, ChartData], None]], ...] = (
    ("do_ceiling", "Pass@1 by premises", _mini_ceiling),
    ("do_textbook", "Textbook pass@k", _mini_textbook),
    ("do_table1", "Textbook72 solved", _mini_textbook72),
    ("do_failures", "How attempts end", _mini_failures),
    ("do_textbook_law", "textbook_law, unseen", _mini_textbook_law),
    ("do_heldout", "Held-out greedy", _mini_heldout),
    ("do_families", "Family horizon", _mini_families),
    ("do_laws", "Laws vs s", lambda ax, cd: _mini_lines(ax, cd, "s")),
    ("do_glue", "Glue ratio R", _mini_glue),
    ("do_prefix", "Prefix completion", lambda ax, cd: _mini_lines(ax, cd, "lines left")),
    ("do_skills", "Skills pass@1", _mini_skills),
    ("do_padding", "Padding (P1) pass@1", _mini_padding),
    ("do_edl", "EDL*, RL vs pretrained", _mini_edl),
    ("do_likelihood", "Log p per line (nats)", _mini_likelihood),
    ("do_weights", "Weight interpolation", _mini_weights),
    ("do_kl", "KL per token", _mini_kl),
    ("do_steering", "Steering pass@1", _mini_steering),
    ("do_probes", "Probe accuracy", _mini_probes),
)
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _legend_cell(ax: Axes, models: Sequence[str]) -> None:
    ax.axis("off")
    handles: list[Any] = []
    for m in _seed0(models):
        st = dstyle(m)
        handles.append(
            Patch(
                facecolor=st.color,
                edgecolor=S.WHITE if st.hatch else st.color,
                hatch=st.hatch,
                linewidth=0.0,
                label=st.label,
            )
        )
    handles += [
        _ref_handle("seed (dots, c)", marker="o", color=S.INK),
        Line2D([], [], color=S.INK, linewidth=0.6, label="Dan's count (c)"),
        Patch(facecolor=S.SHADE, edgecolor="none", label="> 3 premises (a, l)"),
        Line2D([], [], color=S.INK, marker="o", markersize=2.6, linewidth=0.9, label="pass@1 (o)"),
        Line2D(
            [],
            [],
            color=S.REF,
            linestyle=(0, (4, 1.6)),
            marker="s",
            markersize=2.6,
            markerfacecolor=S.WHITE,
            linewidth=0.9,
            label="pass@128 (o)",
        ),
        Line2D(
            [],
            [],
            color=S.REF,
            linestyle=(0, (1, 1.2)),
            marker="o",
            markersize=2.6,
            markerfacecolor=S.WHITE,
            linewidth=0.9,
            label="P1 pads (l)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=S.FS_SMALL,
        handlelength=1.6,
        borderaxespad=0.0,
        labelspacing=0.35,
    )


def fig_overview(charts: Sequence[ChartData], *, narrow: bool = False) -> Figure:
    by = {c.key: c for c in charts}
    n_cells = len(OVERVIEW_PANELS) + 1
    if narrow:
        rows = math.ceil(n_cells / 2)
        fig = S.new_figure(S.NARROW_W, 1.275 * rows)
        axs = fig.subplots(rows, 2)
    else:
        rows = math.ceil(n_cells / 4)
        fig = S.new_figure(S.FULL_W, 1.4 * rows)
        axs = fig.subplots(rows, 4)
    flat = list(axs.flat)
    models: list[str] = []
    for c in charts:
        models += [s.name for s in c.series] + [n for r in c.rows for n, *_ in r.points]
        models += [b.name.rpartition(": ")[2] for b in c.bars]
    models = [m for m in dict.fromkeys(models) if S.parse_model_label(m)[0] in ("dan12", "dan6")]
    for i, (ax, (key, title, draw)) in enumerate(zip(flat, OVERVIEW_PANELS, strict=False)):
        cd = by.get(key)
        ax.tick_params(length=2, pad=1.5)
        if cd is None or cd.kind == "missing":
            PL._mini_missing(ax)  # noqa: SLF001
            if cd is not None and cd.extra.get("missing_label"):
                ax.texts[-1].set_text(str(cd.extra["missing_label"]))
        else:
            draw(ax, cd)
        ax.tick_params(labelsize=S.FS_SMALL)
        ax.xaxis.label.set_size(S.FS_SMALL)
        ax.yaxis.label.set_size(S.FS_SMALL)
        S.panel_title(ax, _LETTERS[i], title, size=S.FS_TICK)
    _legend_cell(flat[len(OVERVIEW_PANELS)], models or ["dan12-fz-s0", "dan12-t1-s0"])
    for ax in flat[len(OVERVIEW_PANELS) + 1 :]:
        ax.axis("off")
    return fig


# =============================================================================== registry

#: key -> (figure function, width in inches); full-width figures also have a stacked layout
#: (`NARROW`) the page serves below 600 px. Tables have no figure.
FIGURES: dict[str, tuple[Callable[[ChartData], Figure], float]] = {
    "do_ceiling": (fig_ceiling, S.FULL_W),
    "do_textbook": (fig_textbook, S.FULL_W),
    "do_families": (fig_families, S.HALF_W),
    "do_laws": (fig_laws, S.HALF_W),
    "do_glue": (fig_glue, S.HALF_W),
    "do_skills": (fig_skills, S.FULL_W),
    "do_padding": (fig_padding, S.FULL_W),
    "do_prefix": (fig_prefix, S.HALF_W),
    "do_heldout": (fig_heldout, S.HALF_W),
    "do_textbook_law": (fig_textbook_law, S.HALF_W),
    "do_edl": (fig_edl, S.FULL_W),
    "do_likelihood": (fig_likelihood, S.HALF_W),
    "do_kl": (fig_kl_wide, S.FULL_W),
    "do_weights": (fig_weights, S.HALF_W),
    "do_probes": (fig_probes, S.FULL_W),
}
NARROW: dict[str, Callable[[ChartData], Figure]] = {
    "do_ceiling": lambda cd: fig_ceiling(cd, narrow=True),
    "do_textbook": lambda cd: fig_textbook(cd, narrow=True),
    "do_skills": lambda cd: fig_skills(cd, narrow=True),
    "do_padding": lambda cd: fig_padding(cd, narrow=True),
    "do_edl": lambda cd: fig_edl(cd, narrow=True),
    "do_probes": lambda cd: fig_probes(cd, narrow=True),
    "do_kl": fig_kl,
}


def width_of(key: str) -> float | None:
    entry = FIGURES.get(key)
    return entry[1] if entry is not None else None


def render(cd: ChartData, *, narrow: bool = False) -> Figure | None:
    """The figure for one chart (`None` without data, for a table, or for `narrow` on a figure
    with no stacked layout). Call inside `style.paper_style()`."""
    if cd.kind in ("missing", "table"):
        return None
    if narrow:
        fn = NARROW.get(cd.key)
        return fn(cd) if fn is not None else None
    entry = FIGURES.get(cd.key)
    return entry[0](cd) if entry is not None else None


def _uri(fig: Figure) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(S.svg_bytes(fig)).decode("ascii")


def figure_uri(cd: ChartData, *, narrow: bool = False) -> tuple[str, float] | None:
    with S.paper_style():
        fig = render(cd, narrow=narrow)
        if fig is None:
            return None
        try:
            return _uri(fig), fig.get_figwidth()
        finally:
            S.close(fig)


def overview_uri(charts: Sequence[ChartData], *, narrow: bool = False) -> tuple[str, float] | None:
    if all(c.kind == "missing" for c in charts):
        return None
    with S.paper_style():
        fig = fig_overview(charts, narrow=narrow)
        try:
            return _uri(fig), fig.get_figwidth()
        finally:
            S.close(fig)
