"""The 2x2x2 factorial's charts, from charts/runs.csv and charts/passk.csv (see extract.py).

  by-cell.png         dev theorems solved at >= 7 lines (the autoresearch metric, of 1,108): bar =
                      3-seed mean, dots = seeds, for naive, naive + each piece, best, best - each
  passk-holdout.png   pass@k on 250 held-out theorems (all >= 7 lines), k = 1..256
  passk-textbook.png  pass@k on the 72 textbook problems

Each piece keeps one color everywhere (Lean format blue, pretraining orange, RL aqua; naive grey,
best light). pass@k is the unbiased estimator 1 - C(n-c, k) / C(n, k) from n = 256 samples,
averaged over problems, then over the 3 seeds.
Run from the summary dir:  python3 charts/factorial.py
"""

import csv
import itertools
import math
import statistics
import sys
from collections import defaultdict

import altair as alt

sys.path.insert(0, "../2026-09-21-pretrain-ablations/charts")
from theme import FG, MUTED, fonts, save

PIECE = {"lean": "#3987e5", "pre": "#d95926", "rl": "#199e70"}  # validated on the dark surface
# cell id (tok-pre-rl) -> (label, color, which side of the chart)
CELLS = {
    "abs-naive-ei": ("naive", MUTED, "both"),
    "lean-naive-ei": ("naive + Lean format", PIECE["lean"], "add"),
    "abs-best-ei": ("naive + best pretraining", PIECE["pre"], "add"),
    "abs-naive-leon": ("naive + best RL", PIECE["rl"], "add"),
    "lean-best-leon": ("best (all three)", FG, "both"),
    "abs-best-leon": ("best − Lean format", PIECE["lean"], "remove"),
    "lean-naive-leon": ("best − best pretraining", PIECE["pre"], "remove"),
    "lean-best-ei": ("best − best RL", PIECE["rl"], "remove"),
}
ORDER = [v[0] for v in CELLS.values()]
# One domain for every panel: hconcat merges same-field color scales, so per-panel domains
# would be reordered against their ranges.
COLOR = alt.Color(
    "cell:N", scale=alt.Scale(domain=ORDER, range=[v[1] for v in CELLS.values()]), legend=None
)
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LABEL_GAP = 4.5  # percentage points between stacked end-of-line labels

runs = list(csv.DictReader(open("charts/runs.csv")))
cell_of = {r["run"]: f"{r['tok']}-{r['pre']}-{r['rl']}" for r in runs}


def bars():
    width = 520
    f = fonts(width)
    seeds = [
        {"cell": CELLS[cell_of[r["run"]]][0], "metric": int(r["metric"]), "seed": r["seed"]}
        for r in runs
    ]
    by = defaultdict(list)
    for s in seeds:
        by[s["cell"]].append(s["metric"])
    means = [
        {"cell": c, "mean": statistics.mean(v), "top": max(v), "txt": f"{statistics.mean(v):.0f}"}
        for c, v in by.items()
    ]
    x = alt.X("cell:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=-35, labelLimit=260))
    y_title = "dev theorems solved at >= 7 lines (of 1,108)"
    chart = (
        alt.Chart(alt.Data(values=means))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=34)
        .encode(x=x, y=alt.Y("mean:Q", title=y_title), color=COLOR)
        + alt.Chart(alt.Data(values=means))  # above the highest seed dot, clear of the dots
        .mark_text(color=FG, dy=-12, fontSize=f["label"])
        .encode(x=x, y="top:Q", text="txt:N")
        + alt.Chart(alt.Data(values=seeds))
        .mark_circle(size=40, color=FG, opacity=0.8)
        .encode(x=x, y="metric:Q")
    ).properties(
        width=width,
        height=320,
        title=alt.Title(
            "Where the gains come from",
            subtitle="bar = 3-seed mean, dots = seeds; same data, 2,400 s budget per run",
        ),
    )
    save(chart, "by-cell", width)


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curves(pool, name, title):
    per_run = defaultdict(list)  # run -> [(n, c)]
    for r in csv.DictReader(open("charts/passk.csv")):
        if r["pool"] == pool:
            per_run[r["run"]].append((int(r["n"]), int(r["n_ok"])))
    by_cell = defaultdict(list)  # cell -> [per-run curve]
    for run, probs in per_run.items():
        by_cell[cell_of[run]].append(
            [statistics.mean(pass_at_k(n, c, k) for n, c in probs) for k in KS]
        )
    width = 330
    f = fonts(2 * width)
    panels = []
    for side, panel_title in (
        ("add", "adding one piece to naive"),
        ("remove", "removing one piece from best"),
    ):
        pts = []
        for cell, (label, _color, where) in CELLS.items():
            if where not in (side, "both") or cell not in by_cell:
                continue
            for i, k in enumerate(KS):
                pct = 100 * statistics.mean(c[i] for c in by_cell[cell])
                pts.append({"cell": label, "k": k, "pct": pct, "end": k == KS[-1]})
        ends = sorted((p for p in pts if p["end"]), key=lambda p: p["pct"])
        for lo, hi in itertools.pairwise(ends):  # end labels at least LABEL_GAP apart
            hi["label_y"] = max(hi["pct"], lo.setdefault("label_y", lo["pct"]) + LABEL_GAP)
        base = alt.Chart(alt.Data(values=pts)).encode(
            x=alt.X(
                "k:Q",
                scale=alt.Scale(type="log", base=2, domain=[1, 256]),
                axis=alt.Axis(values=KS),
                title="k (samples per problem)",
            ),
            y=alt.Y("pct:Q", title=f"pass@k, % of {title}", scale=alt.Scale(domain=[0, 100])),
            color=COLOR,
        )
        panels.append(
            (
                base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
                + base.transform_filter("datum.end")
                .mark_text(align="left", dx=6, fontSize=f["label"])
                .encode(text="cell:N", y="label_y:Q")
            ).properties(width=width, height=300, title=panel_title)
        )
    save(alt.hconcat(*panels, spacing=130), name, 2 * width)


bars()
curves("holdout250", "passk-holdout", "250 held-out theorems")
curves("textbook72", "passk-textbook", "72 textbook problems")
