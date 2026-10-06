"""Figures for the thinking-trace experiment, in the ndbench NeurIPS style
(`code/ndbench/figures/STYLE.md`: 5.5 in text width, Times-like serif, thin axes, Okabe-Ito colours
with a marker and line style per model, PDF with TrueType fonts + SVG + 300-dpi PNG).

    python -m code.thinking.figures lengthbin --out DIR/fig_lengthbin_baseline \
        --group "Pretraining only=S0,S1,S2" --group "Round 8=R0,R1,R2"
    python -m code.thinking.figures budget --out DIR/fig_budget --group "T2=EVAL_s0,EVAL_s1" \
        --arms plain,logical --sets charles_long_reach,textbook_dev
    python -m code.thinking.figures tokens --out DIR/fig_tokens --group "T0=..." --group "T2=..." \
        --arms plain,logical --sets charles_long_reach

A *group* is one model (one line); its comma-separated paths are one source per seed.  Each command
writes `<out>.{pdf,svg,png}`, `<out>.json` (every plotted value, per seed, with its source) and
`<out>.caption.md` (a NeurIPS-style caption: bold title, what is plotted, n, estimator).

Figures:
  * `lengthbin` -- Dmitry's progress figure, "Solve rate by reference-length bin": textbook72 at
    pass@k (default: every attempt in the source, i.e. pass@256 for k-256 reads), the 58 dev
    problems in Dan's bins of the reference proof length (ND lines, `reference_lines` of
    `textbook_dev.jsonl`; bins 1-5 / 6-10 / 11-15 / 16+ inclusive, `bs_analysis.py:22@4dcd0e4f`) and
    the 14 train problems (no reference) as a separate group behind a divider.  Large marker = mean
    over seeds, small dots = single seeds, value labels in %.
  * `budget` -- pass@1 against the per-step think budget B (one panel per set).
  * `tokens` -- pass@k against total *sampled* tokens per theorem (think + `</think>` + action,
    rejected redraws included; prompt prefill excluded): the point for k attempts sits at
    x = k x (mean sampled tokens per attempt over the set), y = mean unbiased pass@k.

Sources (one per seed; the kind is detected from the path):
  * a `code.thinking.eval` output directory (`summary.json` + `rows_<arm>_B<B>.jsonl`), or one
    `rows_*.jsonl` file of it;
  * a fork `guided_eval.py` read: `<out>.rows.jsonl.gz`, `<out>.json` or the `<out>` prefix;
  * a per-theorem table, `.jsonl[.gz]` or `.csv`, with `name` and `solved` (Dan's `state_eval.py`
    `--out` rows) or `n_ok` [+ `k`];
  * Dan's `state_eval.py --summary` JSON (`artifacts/bs/eval/<ckpt>__tb72.json`): solved counts per
    `reference_lines` value (`by_len`), enough for `lengthbin` only.

This module never imports torch or the vendored fork, so it runs anywhere matplotlib does.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from ..ndbench.figures import style as S

# ------------------------------------------------------------------------------ textbook72 bins

#: Dan's bins (`bs_analysis.py:22@4dcd0e4f`): inclusive bounds on `reference_lines`.
BINS: tuple[tuple[str, int, int], ...] = (
    ("1–5", 1, 5),
    ("6–10", 6, 10),
    ("11–15", 11, 15),
    ("16+", 16, 10**9),
)
TRAIN = "train14"
BIN_ORDER: tuple[str, ...] = (*(b for b, _lo, _hi in BINS), TRAIN)
#: Bin sizes of textbook72 (dev58 by reference length, then train14); checked on load.
TB72_SIZES: dict[str, int] = {"1–5": 5, "6–10": 24, "11–15": 16, "16+": 13, TRAIN: 14}
DEFAULT_FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"


def default_tb72_dir() -> Path:
    fork = os.environ.get("FORK") or os.environ.get("ND_TAKEHOME") or DEFAULT_FORK
    return Path(fork) / "data" / "eval_only" / "textbook72"


def bin_of(ref: int | None) -> str:
    """Reference length (ND lines) -> bin label; `None` (no reference proof) -> train14."""
    if ref is None:
        return TRAIN
    for name, lo, hi in BINS:
        if lo <= ref <= hi:
            return name
    raise ValueError(f"reference length {ref} is in no bin")


def reference_lengths(tb72_dir: Path | str | None = None) -> dict[str, int | None]:
    """name -> `reference_lines` for the 58 dev problems, `None` for the 14 train problems (they
    have no reference proof).  The MANIFEST defines no length; the field is the line count of the
    reference proof that came with the dev split (Dmitry's cap-comparison evaluation data)."""
    d = Path(tb72_dir) if tb72_dir is not None else default_tb72_dir()
    out: dict[str, int | None] = {}
    for line in (d / "textbook_dev.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["name"]] = int(r["reference_lines"])
    for line in (d / "textbook_train.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["name"]] = None
    return out


# ------------------------------------------------------------------------------------- sources


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n attempts with c successes (Chen et al. 2021); `eval.pass_at_k`."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def ks_upto(n: int) -> list[int]:
    """1, 2, 4, ... up to n, plus n itself (`eval.ks_upto`)."""
    out, k = [], 1
    while k <= n:
        out.append(k)
        k *= 2
    if out[-1] != n:
        out.append(n)
    return out


@dataclass
class Thm:
    """One theorem of one read: n attempts, c accepted; per-attempt sampled tokens if known."""

    name: str
    file: str
    n: int
    c: int
    tokens: list[int] | None = None


@dataclass
class Source:
    """One seed's read.  `thms` per theorem, or (Dan's summaries) only `by_len`: solved and n per
    reference length (`None` = no reference)."""

    path: str
    kind: str
    cell: str | None = None
    thms: list[Thm] = field(default_factory=lambda: list[Thm]())
    by_len: dict[int | None, tuple[int, int]] | None = None
    k: int | None = None


def _open_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            yield from f
    else:
        with open(path) as f:
            yield from f


def _thm_from_row(r: dict[str, Any], file_default: str) -> Thm:
    name = str(r["name"])
    file = str(r.get("file") or file_default)
    tokens = r.get("tokens") if isinstance(r.get("tokens"), list) else None
    if "n_ok" in r:
        c = int(r["n_ok"])
        n = int(r.get("k") or (len(r["ok"]) if isinstance(r.get("ok"), list) else 0))
        if n <= 0:
            raise ValueError(f"row {name}: n_ok without k or ok[]")
    elif "solved" in r:
        # Dan's state_eval rows: solved at the read's k, attempts not recorded -> one "attempt"
        c, n = int(bool(r["solved"])), 1
    else:
        raise ValueError(f"row {name}: needs n_ok (+ k) or solved")
    return Thm(name, file, n, c, [int(t) for t in tokens] if tokens else None)


def _rows_file(path: Path, kind: str, cell: str | None, file_default: str = "") -> Source:
    thms = [_thm_from_row(json.loads(x), file_default) for x in _open_text(path) if x.strip()]
    ks = {t.n for t in thms}
    return Source(str(path), kind, cell, thms, None, ks.pop() if len(ks) == 1 else None)


def eval_cells(directory: Path | str) -> list[str]:
    """The (arm, budget) cells of a `code.thinking.eval` output directory, e.g. ['plain_B0']."""
    return sorted(
        p.name[len("rows_") : -len(".jsonl")] for p in Path(directory).glob("rows_*.jsonl")
    )


def cell_budget(cell: str) -> tuple[str, int]:
    """'logical_B64' -> ('logical', 64)."""
    arm, _sep, b = cell.rpartition("_B")
    return arm, int(b)


def load_source(path: Path | str, cell: str | None = None) -> Source:
    """One seed's results from any supported source (see the module docstring).  `cell` picks the
    (arm, budget) of a `code.thinking.eval` directory, e.g. 'plain_B0'; it may be omitted when the
    directory holds only one."""
    p = Path(path)
    if p.is_dir():
        cells = eval_cells(p)
        if not cells:
            raise ValueError(f"{p}: no rows_*.jsonl (not a code.thinking.eval output directory)")
        if cell is None:
            if len(cells) != 1:
                raise ValueError(f"{p}: several cells {cells}; choose one with cell=")
            cell = cells[0]
        if cell not in cells:
            raise ValueError(f"{p}: no cell {cell!r} (has {cells})")
        return _rows_file(p / f"rows_{cell}.jsonl", "thinking.eval", cell)
    if p.name.startswith("rows_") and p.suffix == ".jsonl":
        return _rows_file(p, "thinking.eval", p.name[len("rows_") : -len(".jsonl")])
    gz = Path(str(p) + ".rows.jsonl.gz")
    if p.name.endswith(".rows.jsonl.gz") or (not p.exists() and gz.exists()):
        f = p if p.name.endswith(".rows.jsonl.gz") else gz
        return _rows_file(f, "guided_eval", _guided_arm(f))
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if "by_len" in d:
            by_len: dict[int | None, tuple[int, int]] = {}
            for key, v in d["by_len"].items():
                ref = None if key in ("None", "null", None) else int(key)
                by_len[ref] = (int(v["solved"]), int(v["n"]))
            return Source(str(p), "state_eval summary", None, [], by_len, d.get("k"))
        if "cells" in d:
            return load_source(p.parent, cell)
        rows = Path(str(p)[: -len(".json")] + ".rows.jsonl.gz")
        if rows.exists():
            return _rows_file(rows, "guided_eval", d.get("arm"))
        raise ValueError(f"{p}: unrecognised JSON (no by_len, cells or sibling .rows.jsonl.gz)")
    if p.suffix == ".csv":
        with open(p, newline="") as f:
            recs = list(csv.DictReader(f))
        thms = []
        for r in recs:
            row: dict[str, Any] = {"name": r["name"], "file": r.get("file") or ""}
            if r.get("n_ok") not in (None, ""):
                row["n_ok"], row["k"] = int(r["n_ok"]), int(r["k"])
            else:
                row["solved"] = str(r["solved"]).strip().lower() in ("1", "true", "yes")
            thms.append(_thm_from_row(row, ""))
        ks = {t.n for t in thms}
        return Source(str(p), "table", None, thms, None, ks.pop() if len(ks) == 1 else None)
    if p.name.endswith((".jsonl", ".jsonl.gz")):
        return _rows_file(p, "table", None)
    raise ValueError(f"{p}: unsupported source")


def _guided_arm(rows: Path) -> str | None:
    summ = Path(str(rows)[: -len(".rows.jsonl.gz")] + ".json")
    if summ.exists():
        return json.loads(summ.read_text()).get("arm")
    return None


def solved_value(t: Thm, k: int | None) -> float:
    """Expected solved indicator at pass@k (k = None: every attempt, i.e. c > 0)."""
    if k is None or k == t.n:
        return float(t.c > 0)
    if t.n == 1 and t.c in (0, 1) and k > 1:
        raise ValueError(f"{t.name}: a solved/unsolved table cannot give pass@{k}")
    if k > t.n:
        raise ValueError(f"{t.name}: pass@{k} needs at least {k} attempts (has {t.n})")
    return pass_at_k(t.n, t.c, k)


def bin_counts(
    src: Source,
    refs: dict[str, int | None],
    k: int | None = None,
    *,
    check_sizes: dict[str, int] | None = TB72_SIZES,
) -> dict[str, tuple[float, int]]:
    """bin -> (expected solved, n) for one seed.  Theorems not in `refs` are ignored (a read may
    cover more files); every name in `refs` must be present.  `check_sizes` asserts the bin sizes
    (textbook72's 5/24/16/13 + 14)."""
    out = {b: [0.0, 0] for b in BIN_ORDER}
    if src.by_len is not None:
        if k is not None and src.k is not None and k != src.k:
            raise ValueError(f"{src.path}: a by_len summary has pass@{src.k} only, not pass@{k}")
        for ref, (solved, n) in src.by_len.items():
            b = bin_of(ref)
            out[b][0] += solved
            out[b][1] += n
    else:
        seen = {t.name: t for t in src.thms if t.name in refs}
        missing = sorted(set(refs) - set(seen))
        if missing:
            raise ValueError(
                f"{src.path}: {len(missing)} of {len(refs)} theorems missing, e.g. {missing[:3]}"
            )
        for name, t in seen.items():
            b = bin_of(refs[name])
            out[b][0] += solved_value(t, k)
            out[b][1] += 1
    res = {b: (float(v[0]), int(v[1])) for b, v in out.items()}
    if check_sizes is not None:
        got = {b: n for b, (_s, n) in res.items()}
        if got != check_sizes:
            raise ValueError(f"{src.path}: bin sizes {got} != textbook72's {check_sizes}")
    return res


# ------------------------------------------------------------------------------------ styling


#: Group order -> fixed encoding: the paper's base / rl / armC first (blue circles, vermillion
#: squares, bluish-green triangles), then the family hues of `style.FAMILY_HUES`.
_GROUP_STYLES: tuple[tuple[str, str, Any], ...] = (
    (S.OKABE_ITO["blue"], "o", "-"),
    (S.OKABE_ITO["vermillion"], "s", "-"),
    (S.OKABE_ITO["bluish_green"], "^", (0, (4, 1.6))),
    (S.OKABE_ITO["reddish_purple"], "D", (0, (1, 1.3))),
    (S.OKABE_ITO["sky_blue"], "v", (0, (5, 1.5, 1, 1.5))),
    (S.OKABE_ITO["orange"], "P", "-"),
    (S.OKABE_ITO["black"], "X", (0, (4, 1.6))),
)
#: Arm -> line style in the TTS figures (colour stays the model's).
ARM_LINESTYLES: dict[str, Any] = {
    "plain": "-",
    "structural": (0, (1, 1.3)),
    "logical": (0, (4, 1.6)),
}


def group_style(i: int, label: str) -> S.ModelStyle:
    color, marker, ls = _GROUP_STYLES[i % len(_GROUP_STYLES)]
    return S.ModelStyle(label, color, marker, linestyle=ls)


def text_color(color: str) -> str:
    """Value labels in a model's hue, darkened so light hues (sky blue, orange) stay legible."""
    return S.shade(color, -0.25)


@dataclass
class Group:
    label: str
    paths: list[str]


def parse_group(spec: str) -> Group:
    """'Round 8=a.json,b.json' -> Group('Round 8', ['a.json', 'b.json'])."""
    label, sep, rest = spec.partition("=")
    if not sep or not label.strip() or not rest.strip():
        raise argparse.ArgumentTypeError(f"--group wants 'label=path[,path...]', got {spec!r}")
    return Group(label.strip(), [p.strip() for p in rest.split(",") if p.strip()])


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def _write(fig: Figure, out: Path, data: dict[str, Any], caption: str) -> list[Path]:
    paths = S.save(fig, out)
    S.close(fig)
    j = out.with_suffix(".json")
    j.write_text(json.dumps(data, indent=1) + "\n")
    c = out.with_suffix(".caption.md")
    c.write_text(caption + "\n")
    return [*paths, j, c]


def _legend(fig: Figure, handles: Sequence[Any], max_cols: int = 3) -> None:
    """Legend above the panels, wrapped at `max_cols` entries a row (one row overflows the
    text width beyond 3 entries; a half-width figure fits 2)."""
    fig.legend(
        handles=list(handles),
        loc="outside upper center",
        ncol=min(len(handles), max_cols),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )


# ------------------------------------------------------------------------- 1. length-bin figure

LENGTHBIN_TITLE = "Solve rate by reference-length bin"


def lengthbin_data(
    groups: Sequence[Group],
    *,
    refs: dict[str, int | None] | None = None,
    k: int | None = None,
    cell: str | None = None,
    check_sizes: dict[str, int] | None = TB72_SIZES,
) -> dict[str, Any]:
    """Per group, per seed and bin: solved (expected at pass@k), n, rate; plus the seed mean."""
    refs = refs if refs is not None else reference_lengths()
    out: dict[str, Any] = {"bins": list(BIN_ORDER), "k": k, "groups": []}
    ks_seen: set[int | None] = set()
    for g in groups:
        seeds = []
        for p in g.paths:
            src = load_source(p, cell)
            counts = bin_counts(src, refs, k, check_sizes=check_sizes)
            ks_seen.add(k if k is not None else src.k)
            seeds.append(
                {
                    "path": src.path,
                    "kind": src.kind,
                    "cell": src.cell,
                    "k": src.k,
                    "bins": {
                        b: {"solved": s, "n": n, "rate": s / n if n else float("nan")}
                        for b, (s, n) in counts.items()
                    },
                }
            )
        mean = {b: _mean([sd["bins"][b]["rate"] for sd in seeds]) for b in BIN_ORDER}
        n = {b: seeds[0]["bins"][b]["n"] for b in BIN_ORDER}
        out["groups"].append({"label": g.label, "seeds": seeds, "mean": mean, "n": n})
    ks_known = {x for x in ks_seen if x is not None}
    out["k_effective"] = ks_known.pop() if len(ks_known) == 1 else None
    return out


def _label_slots(means: Sequence[float]) -> list[str]:
    """Where each model's value label goes at one x: above the highest, below the lowest, to the
    right for the rest; a lone model goes above."""
    if len(means) == 1:
        return ["above"]
    order = sorted(range(len(means)), key=lambda i: means[i])
    slots = ["right"] * len(means)
    slots[order[-1]] = "above"
    slots[order[0]] = "below"
    return slots


def fig_lengthbin(data: dict[str, Any], *, title: str | None = LENGTHBIN_TITLE) -> Figure:
    """Draw `lengthbin_data`: one line per group over the four dev bins, the train14 group behind
    a divider; large marker = seed mean, small dots = seeds, value labels in %."""
    groups = data["groups"]
    G = len(groups)
    xs = {b: float(i) for i, b in enumerate(BIN_ORDER[:-1])}
    divider = len(BINS) - 0.5 + 0.1
    xs[TRAIN] = divider + 0.75
    fig = S.new_figure(S.FULL_W, 2.45)
    ax = fig.add_subplot(1, 1, 1)
    styles = [group_style(i, g["label"]) for i, g in enumerate(groups)]
    off = [(i - (G - 1) / 2) * 0.12 for i in range(G)]
    # each model's seed dots sit in a narrow column on its outer side, clear of the mean marker
    side = [-1.0 if i < (G - 1) / 2 else 1.0 for i in range(G)]
    for gi, (g, st) in enumerate(zip(groups, styles, strict=True)):
        dev = list(BIN_ORDER[:-1])
        ax.plot(
            [xs[b] + off[gi] for b in dev],
            [g["mean"][b] for b in dev],
            color=st.color,
            linestyle=st.linestyle,
            linewidth=1.2,
            zorder=3,
        )
        for b in BIN_ORDER:
            x = xs[b] + off[gi]
            ax.plot([x], [g["mean"][b]], linestyle="none", zorder=4, **st.marker_kw(6.0))
            seeds = [sd["bins"][b]["rate"] for sd in g["seeds"]]
            if len(seeds) > 1:
                ax.plot(
                    [
                        x + side[gi] * 0.12 + 0.03 * (si - (len(seeds) - 1) / 2)
                        for si in range(len(seeds))
                    ],
                    seeds,
                    linestyle="none",
                    marker=st.marker,
                    markersize=2.3,
                    markerfacecolor=st.color,
                    markeredgecolor="none",
                    alpha=0.55,
                    zorder=2.5,
                )
    for b in BIN_ORDER:
        means = [g["mean"][b] for g in groups]
        for gi, (slot, g, st) in enumerate(zip(_label_slots(means), groups, styles, strict=True)):
            x, y = xs[b] + off[gi], g["mean"][b]
            if slot == "below" and y < 0.1:
                slot = "right"  # keep clear of the tick labels
            dx, dy, ha, va = {
                "above": (0, 5.5, "center", "bottom"),
                "below": (0, -5.5, "center", "top"),
                "right": (
                    13.0 if side[gi] > 0 and len(g["seeds"]) > 1 else 6.5,
                    0,
                    "left",
                    "center",
                ),
            }[slot]
            ax.annotate(
                _pct(y),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=S.FS_TICK,
                color=text_color(st.color),
                zorder=5,
            )
    ax.axvline(divider, color=S.INK2, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    n = groups[0]["n"] if groups else dict.fromkeys(BIN_ORDER, 0)
    ticks = [xs[b] for b in BIN_ORDER]
    labels = [f"{b}\nn = {n[b]}" for b in BIN_ORDER[:-1]] + [f"train14\nn = {n[TRAIN]}"]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.45, xs[TRAIN] + 0.55)
    ax.set_ylim(-0.04, 1.12)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    ax.yaxis.set_major_formatter(S.PctFormatter())
    S.light_grid(ax)
    k = data.get("k") or data.get("k_effective")
    ax.set_ylabel(f"% solved at pass@{k}" if k else "% solved")
    ax.set_xlabel("Reference proof length (ND lines), textbook dev58")
    ax.xaxis.set_label_coords((1.5 + 0.45) / (xs[TRAIN] + 1.0), -0.19)
    ax.text(
        xs[TRAIN],
        -0.19,
        "no reference",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=S.FS_LABEL,
    )
    if title:
        S.panel_title(ax, None, title)
    seed_n = sorted({len(g["seeds"]) for g in groups})
    seed_txt = f"{seed_n[0]} seeds" if len(seed_n) == 1 else "seeds"
    handles: list[Any] = [
        Line2D([], [], color=st.color, linestyle=st.linestyle, label=st.label, **st.marker_kw(5.0))
        for st in styles
    ]
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=2.3,
            markerfacecolor=S.REF,
            markeredgecolor="none",
            label=f"single seeds (large marker: mean of {seed_txt})",
        )
    )
    _legend(fig, handles)
    return fig


def lengthbin_caption(data: dict[str, Any], *, title: str = LENGTHBIN_TITLE) -> str:
    k = data.get("k") or data.get("k_effective")
    parts = []
    for g in data["groups"]:
        vals = " / ".join(_pct(g["mean"][b]) for b in BIN_ORDER[:-1])
        parts.append(
            f"{g['label']} ({len(g['seeds'])} seeds) {vals}, train14 {_pct(g['mean'][TRAIN])}"
        )
    n = data["groups"][0]["n"] if data["groups"] else {}
    sizes = ", ".join(f"{b} n = {n.get(b, '?')}" for b in BIN_ORDER[:-1])
    return (
        f"**{title}.** Share of textbook72 problems solved at pass@{k} "
        f"({k} samples per theorem; a problem counts as solved if any sample is accepted by Lean). "
        f"The 58 dev problems are binned by the length of their reference proof in ND lines "
        f"({sizes}); the 14 train-split problems have no reference proof and are shown "
        f"separately (n = {n.get(TRAIN, '?')}). Large markers: mean over seeds; small dots: single "
        f"seeds. Point estimates, no CI. Seed means per bin (1–5 / 6–10 / 11–15 / 16+): "
        + "; ".join(parts)
        + "."
    )


# --------------------------------------------------------------------- shared TTS-figure loading


def _cells_for(path: Path | str, arms: Sequence[str] | None) -> list[str]:
    p = Path(path)
    if p.is_dir():
        cells = eval_cells(p)
        return [c for c in cells if arms is None or cell_budget(c)[0] in arms]
    src = load_source(p)
    if src.cell is None:
        raise ValueError(f"{p}: the arm of this source is unknown")
    c = src.cell if "_B" in src.cell else f"{src.cell}_B0"
    return [c] if arms is None or cell_budget(c)[0] in arms else []


def _load_cell(path: Path | str, cell: str) -> Source:
    p = Path(path)
    if p.is_dir():
        return load_source(p, cell)
    return load_source(p)


def _select(thms: Sequence[Thm], set_name: str) -> list[Thm]:
    if set_name == "all":
        return list(thms)
    sel = [t for t in thms if t.file == set_name]
    if not sel:
        files = sorted({t.file for t in thms})
        raise ValueError(f"no theorems of set {set_name!r} (files: {files})")
    return sel


def _set_n(thms: Sequence[Thm], set_name: str) -> int:
    return len(_select(thms, set_name))


# ------------------------------------------------------------------- 2. pass@1 against budget B


def budget_data(
    groups: Sequence[Group], *, arms: Sequence[str], sets: Sequence[str]
) -> dict[str, Any]:
    """Per group, arm, set and seed: pass@1 (mean c/n over the set's theorems) at each budget B."""
    out: dict[str, Any] = {"sets": list(sets), "arms": list(arms), "series": []}
    for g in groups:
        for arm in arms:
            seeds = []
            for p in g.paths:
                pts: dict[str, dict[str, float]] = {s: {} for s in sets}
                n_set: dict[str, int] = {}
                for c in _cells_for(p, [arm]):
                    _a, B = cell_budget(c)
                    src = _load_cell(p, c)
                    for s in sets:
                        th = _select(src.thms, s)
                        pts[s][str(B)] = _mean([t.c / t.n for t in th])
                        n_set[s] = len(th)
                if any(pts[s] for s in sets):
                    seeds.append({"path": str(p), "pass1": pts, "n": n_set})
            if not seeds:
                continue
            mean: dict[str, dict[str, float]] = {}
            for s in sets:
                Bs = sorted({int(b) for sd in seeds for b in sd["pass1"][s]})
                mean[s] = {
                    str(B): _mean(
                        [sd["pass1"][s][str(B)] for sd in seeds if str(B) in sd["pass1"][s]]
                    )
                    for B in Bs
                }
            out["series"].append(
                {"group": g.label, "arm": arm, "seeds": seeds, "mean": mean, "n": seeds[0]["n"]}
            )
    return out


def _budget_x(Bs: Sequence[int]) -> tuple[dict[int, float], float | None]:
    """B -> x position: log2(B) for B > 0, B = 0 one octave left of the smallest positive B
    (returns the x of the axis break between them, or None if there is no B = 0)."""
    pos = sorted(b for b in Bs if b > 0)
    xs = {b: math.log2(b) for b in pos}
    brk = None
    if 0 in Bs:
        x0 = (math.log2(pos[0]) if pos else 1.0) - 1.25
        xs[0] = x0
        brk = x0 + 0.62
    return xs, brk


def fig_budget(data: dict[str, Any]) -> Figure:
    sets = data["sets"]
    series = data["series"]
    groups = list(dict.fromkeys(s["group"] for s in series))
    gstyle = {g: group_style(i, g) for i, g in enumerate(groups)}
    allB = sorted({int(b) for s in series for st in sets for b in s["mean"][st]})
    xs, brk = _budget_x(allB)
    ncol = len(sets)
    w = S.FULL_W if ncol > 1 else S.HALF_W + 0.6
    fig = S.new_figure(w, 2.15)
    axs = fig.subplots(1, ncol, squeeze=False, sharey=True)[0]
    for si, (ax, st_name) in enumerate(zip(axs, sets, strict=True)):
        for s in series:
            gs = gstyle[s["group"]]
            ls = ARM_LINESTYLES.get(s["arm"], gs.linestyle)
            m = s["mean"][st_name]
            Bs = sorted(int(b) for b in m)
            if not Bs:
                continue
            if Bs == [0]:  # a no-think model: its B = 0 value as a reference across all B
                ax.axhline(m["0"], color=gs.color, linestyle=ls, linewidth=0.7, alpha=0.6, zorder=2)
            ax.plot(
                [xs[b] for b in Bs],
                [m[str(b)] for b in Bs],
                color=gs.color,
                linestyle=ls,
                linewidth=1.2,
                zorder=3,
                **gs.marker_kw(4.2),
            )
            for sd in s["seeds"]:
                pts = sd["pass1"][st_name]
                bs = sorted(int(b) for b in pts)
                if len(s["seeds"]) > 1:
                    ax.plot(
                        [xs[b] for b in bs],
                        [pts[str(b)] for b in bs],
                        linestyle="none",
                        marker=gs.marker,
                        markersize=2.2,
                        markerfacecolor=gs.color,
                        markeredgecolor="none",
                        alpha=0.5,
                        zorder=2.5,
                    )
        ax.xaxis.set_major_locator(FixedLocator([xs[b] for b in allB]))
        ax.set_xticklabels([str(b) for b in allB])
        ax.xaxis.set_minor_formatter(NullFormatter())
        lo = min(xs.values()) if xs else 0.0
        hi = max(xs.values()) if xs else 1.0
        ax.set_xlim(lo - 0.4, hi + 0.4)
        if brk is not None:
            _axis_break(ax, brk)
        S.pct_axis(ax)
        S.light_grid(ax)
        ax.set_xlabel("think budget B (tokens per step)")
        if si == 0:
            ax.set_ylabel("pass@1")
        n = series[0]["n"].get(st_name, "?") if series else "?"
        S.panel_title(ax, "abcdefgh"[si] if ncol > 1 else None, f"{st_name}, n = {n}")
    top = max(
        [v for s in series for st in sets for v in s["mean"][st].values()]
        + [v for s in series for sd in s["seeds"] for st in sets for v in sd["pass1"][st].values()]
        + [0.01]
    )
    axs[0].set_ylim(0, min(1.0, top) * 1.08)
    handles: list[Any] = [
        Line2D([], [], color=gstyle[g].color, label=g, **gstyle[g].marker_kw(4.2)) for g in groups
    ]
    arms = list(dict.fromkeys(s["arm"] for s in series))
    if len(arms) > 1:
        handles += [
            Line2D([], [], color=S.INK2, linestyle=ARM_LINESTYLES.get(a, "-"), label=a)
            for a in arms
        ]
    _legend(fig, handles, max_cols=3 if ncol > 1 else 2)
    return fig


def _axis_break(ax: Axes, x: float) -> None:
    """Two short slanted strokes on the x axis at data-x `x` (B = 0 is not on the log axis)."""
    kw: dict[str, Any] = {
        "transform": ax.get_xaxis_transform(),
        "color": S.INK,
        "linewidth": 0.6,
        "clip_on": False,
        "zorder": 6,
    }
    for d in (-0.06, 0.06):
        ax.plot([x + d - 0.05, x + d + 0.05], [-0.03, 0.03], **kw)


def budget_caption(data: dict[str, Any]) -> str:
    parts = []
    for s in data["series"]:
        for st in data["sets"]:
            m = s["mean"][st]
            vals = ", ".join(f"B = {b}: {_pct(m[b])}" for b in sorted(m, key=int))
            parts.append(f"{s['group']} {s['arm']} on {st} ({len(s['seeds'])} seeds): {vals}")
    return (
        "**pass@1 against the per-step think budget.** pass@1 (mean share of accepted attempts per "
        "theorem) against the maximum number of think tokens per environment step B (`</think>` "
        "forced at B; B = 0 is no think block). Solid lines: plain sampling; dashed: guided "
        "logical redraws. Large markers: mean over seeds; small dots: single seeds. Point "
        "estimates, no CI. " + "; ".join(parts) + "."
    )


# ------------------------------------------------------- 3. pass@k against sampled tokens per thm


def tokens_data(
    groups: Sequence[Group],
    *,
    arms: Sequence[str],
    sets: Sequence[str],
    budgets: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Per group, cell (arm, B), set and seed: for k = 1, 2, 4, ..., K the mean unbiased pass@k and
    the expected sampled tokens per theorem, k x (sampled tokens per attempt, mean over the set)."""
    out: dict[str, Any] = {"sets": list(sets), "series": []}
    for g in groups:
        cells: list[str] = []
        for p in g.paths:
            for c in _cells_for(p, arms):
                if (budgets is None or cell_budget(c)[1] in budgets) and c not in cells:
                    cells.append(c)
        for c in sorted(cells, key=lambda c: (arms.index(cell_budget(c)[0]), cell_budget(c)[1])):
            seeds = []
            for p in g.paths:
                if c not in _cells_for(p, arms):
                    continue
                src = _load_cell(p, c)
                per: dict[str, Any] = {}
                for s in sets:
                    th = _select(src.thms, s)
                    if any(t.tokens is None for t in th):
                        raise ValueError(f"{p}: per-attempt token counts missing")
                    K = min(t.n for t in th)
                    tpa = sum(sum(t.tokens or []) for t in th) / sum(t.n for t in th)
                    per[s] = {
                        "n": len(th),
                        "k_max": K,
                        "tokens_per_attempt": tpa,
                        "pts": [
                            [k * tpa, _mean([pass_at_k(t.n, t.c, k) for t in th])]
                            for k in ks_upto(K)
                        ],
                        "ks": ks_upto(K),
                    }
                seeds.append({"path": str(p), "sets": per})
            if not seeds:
                continue
            mean: dict[str, list[list[float]]] = {}
            for s in sets:
                ks = min((sd["sets"][s]["ks"] for sd in seeds), key=len)
                mean[s] = [
                    [
                        _mean([sd["sets"][s]["pts"][i][0] for sd in seeds]),
                        _mean([sd["sets"][s]["pts"][i][1] for sd in seeds]),
                    ]
                    for i in range(len(ks))
                ]
            arm, B = cell_budget(c)
            out["series"].append(
                {
                    "group": g.label,
                    "arm": arm,
                    "budget": B,
                    "cell": c,
                    "seeds": seeds,
                    "mean": mean,
                    "n": {s: seeds[0]["sets"][s]["n"] for s in sets},
                }
            )
    return out


def _tokens_fmt(v: float, _pos: int | None = None) -> str:
    for div, suf in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            q = v / div
            return f"{q:g}{suf}"
    return f"{v:g}"


def fig_tokens(data: dict[str, Any]) -> Figure:
    sets = data["sets"]
    series = data["series"]
    groups = list(dict.fromkeys(s["group"] for s in series))
    gstyle = {g: group_style(i, g) for i, g in enumerate(groups)}
    budgets = sorted({s["budget"] for s in series})
    ncol = len(sets)
    w = S.FULL_W if ncol > 1 else S.HALF_W + 0.6
    fig = S.new_figure(w, 2.2)
    axs = fig.subplots(1, ncol, squeeze=False, sharey=True)[0]
    for si, (ax, st_name) in enumerate(zip(axs, sets, strict=True)):
        for s in series:
            gs = gstyle[s["group"]]
            ls = ARM_LINESTYLES.get(s["arm"], gs.linestyle)
            t = budgets.index(s["budget"]) / max(1, len(budgets) - 1) if len(budgets) > 1 else 0.0
            color = S.shade(gs.color, -0.3 + 0.6 * t) if len(budgets) > 1 else gs.color
            pts = s["mean"][st_name]
            if len(s["seeds"]) > 1:
                for sd in s["seeds"]:
                    sp = sd["sets"][st_name]["pts"]
                    ax.plot(
                        [p[0] for p in sp],
                        [p[1] for p in sp],
                        color=color,
                        linestyle=ls,
                        linewidth=0.5,
                        alpha=0.4,
                        zorder=2,
                    )
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                linestyle=ls,
                linewidth=1.2,
                zorder=3,
                marker=gs.marker,
                markersize=3.2,
                markerfacecolor=color if s["arm"] == "plain" else S.WHITE,
                markeredgecolor=S.WHITE if s["arm"] == "plain" else color,
                markeredgewidth=0.5 if s["arm"] == "plain" else 0.8,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_tokens_fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
        S.pct_axis(ax)
        S.light_grid(ax, "both")
        ax.set_xlabel("sampled tokens per theorem")
        if si == 0:
            ax.set_ylabel("pass@k")
        n = series[0]["n"].get(st_name, "?") if series else "?"
        S.panel_title(ax, "abcdefgh"[si] if ncol > 1 else None, f"{st_name}, n = {n}")
    top = max(
        [p[1] for s in series for st in sets for p in s["mean"][st]]
        + [p[1] for s in series for sd in s["seeds"] for st in sets for p in sd["sets"][st]["pts"]]
        + [0.01]
    )
    axs[0].set_ylim(0, min(1.0, top) * 1.08)
    handles: list[Any] = [
        Line2D([], [], color=gstyle[g].color, label=g, **gstyle[g].marker_kw(4.2)) for g in groups
    ]
    arms = list(dict.fromkeys(s["arm"] for s in series))
    handles += [
        Line2D(
            [],
            [],
            color=S.INK2,
            linestyle=ARM_LINESTYLES.get(a, "-"),
            marker="o",
            markersize=3.2,
            markerfacecolor=S.INK2 if a == "plain" else S.WHITE,
            markeredgecolor=S.WHITE if a == "plain" else S.INK2,
            label=a,
        )
        for a in arms
    ]
    if len(budgets) > 1:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                label=f"darker → lighter: B = {budgets[0]} → {budgets[-1]}",
            )
        )
    _legend(fig, handles, max_cols=3 if ncol > 1 else 2)
    return fig


def tokens_caption(data: dict[str, Any]) -> str:
    parts = []
    for s in data["series"]:
        for st in data["sets"]:
            last = s["mean"][st][-1]
            tpa = _mean([sd["sets"][st]["tokens_per_attempt"] for sd in s["seeds"]])
            parts.append(
                f"{s['group']} {s['arm']} B = {s['budget']} on {st}: {tpa:.0f} tokens per attempt, "
                f"pass@{s['seeds'][0]['sets'][st]['ks'][-1]} {_pct(last[1])}"
            )
    return (
        "**pass@k against sampled tokens per theorem.** Mean unbiased pass@k (k = 1, 2, 4, ...) "
        "against the expected number of tokens sampled per theorem to draw k attempts: k times "
        "the mean sampled tokens per attempt over the set, counting think tokens, `</think>`, "
        "action tokens and every rejected redraw (prompt prefill excluded). Solid lines with "
        "filled markers: plain sampling; dashed with hollow markers: guided logical redraws. "
        "Thick lines: mean over seeds; thin lines: single seeds. Point estimates, no CI. "
        + "; ".join(parts)
        + "."
    )


# ------------------------------------------------------------------------------------------ CLI


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("lengthbin", help="Dmitry's solve-rate-by-reference-length figure")
    bu = sub.add_parser("budget", help="pass@1 against the per-step think budget B")
    to = sub.add_parser("tokens", help="pass@k against sampled tokens per theorem")
    for p in (lb, bu, to):
        p.add_argument("--out", required=True, help="output base path (no suffix)")
        p.add_argument(
            "--group",
            action="append",
            required=True,
            type=parse_group,
            help="'label=path[,path...]', one path per seed; repeat per model, in legend order",
        )
    lb.add_argument("--tb72", default=None, help="textbook72 directory (default: $FORK's)")
    lb.add_argument("--k", type=int, default=None, help="pass@k (default: every attempt)")
    lb.add_argument("--cell", default=None, help="code.thinking.eval cell, e.g. plain_B0")
    lb.add_argument("--title", default=LENGTHBIN_TITLE, help="panel title ('' for none)")
    for p in (bu, to):
        p.add_argument("--arms", default="plain,logical")
        p.add_argument(
            "--sets", default="all", help="comma-separated set (file) names, one panel each"
        )
    to.add_argument("--budgets", default=None, help="comma-separated B to include (default all)")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with S.paper_style():
        if a.cmd == "lengthbin":
            refs = reference_lengths(a.tb72)
            data = lengthbin_data(a.group, refs=refs, k=a.k, cell=a.cell)
            files = _write(
                fig_lengthbin(data, title=a.title or None),
                out,
                data,
                lengthbin_caption(data, title=a.title or LENGTHBIN_TITLE),
            )
        elif a.cmd == "budget":
            data = budget_data(a.group, arms=a.arms.split(","), sets=a.sets.split(","))
            files = _write(fig_budget(data), out, data, budget_caption(data))
        else:
            bud = [int(b) for b in a.budgets.split(",")] if a.budgets else None
            data = tokens_data(a.group, arms=a.arms.split(","), sets=a.sets.split(","), budgets=bud)
            files = _write(fig_tokens(data), out, data, tokens_caption(data))
    for f in files:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
