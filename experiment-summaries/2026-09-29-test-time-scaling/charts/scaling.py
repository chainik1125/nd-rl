"""Solve rate vs sampled tokens per theorem on the transfer holdout, one line per test-time method.

Model: combined-pipeline c005 (186 pretrain + Leon's RL + guided RL sampling), final.pt, seeds 0-2.
Pool: the holdout half of transfer.jsonl, 1,177 theorems of reference length 7-14; the right panel
counts the 114 with reference length > 10. Every point is a 3-seed mean (x and y).

  plain       i.i.d. samples, k = 1..64 (pass@k, unbiased estimator); x = k * tokens per sample
  resample    method 1 read at cap 10 rejections per sample, k = 1..64, same axes
  beam        B = 1..16 beams, E = 4 segments per beam node; x = tokens spent per theorem
  smc         N = 2..64 particles
  best-first  one anytime run per seed (alpha 0.5, E 8, 16k budget): at budget b a theorem
              counts if it was solved within b tokens, and spends min(its tokens, b)

Sampled tokens count every token drawn from the model, including rejected segments; prefill
(re-encoding prefixes) is excluded and reported in the README's time table.

Run from the summary dir:  uv run --with altair --with vl-convert-python python charts/scaling.py
(reads ../../artifacts/test_time_scaling/hold_*_s<seed>/; writes charts/scaling.csv and
scaling.png/.svg)
"""

import csv
import json
import statistics
import sys

import altair as alt

sys.path.insert(0, "../2026-09-21-pretrain-ablations/charts")
sys.path.insert(0, "../../code/experiments/current/test_time_scaling")
from analyze import at_cap, pass_at_k  # noqa: E402
from theme import FG, MUTED, fonts, save  # noqa: E402

ART = "../../artifacts/test_time_scaling"
SEEDS = [0, 1, 2]
KS = [1, 2, 4, 8, 16, 32, 64]
BUDGETS = [500, 1000, 2000, 4000, 8000, 16000]
LONG = 10
CAP = 10
# Categorical slots 1-5 (dark steps), validated on the #1a1d24 surface; order is fixed.
COLOR = {
    "plain": "#3987e5",
    "resample": "#d95926",
    "beam": "#199e70",
    "smc": "#c98500",
    "best-first": "#d55181",
}
WIDTH, HEIGHT = 360, 300


def load(part, seed, name):
    with open(f"{ART}/hold_{part}_s{seed}/{name}.json") as f:
        return json.load(f)["theorems"]


def iid(ths, cap=None):
    """(k, tokens per theorem, % solved, expected solved with L > 10) for k in KS."""
    per = [[at_cap(s, cap) if cap is not None else (s[0], s[1]) for s in t["samples"]] for t in ths]
    n = len(per[0])
    tok = sum(x[1] for p in per for x in p) / (len(per) * n)
    out = []
    for k in KS:
        pk = [pass_at_k(n, sum(x[0] for x in p), k) for p in per]
        long = sum(q for q, t in zip(pk, ths, strict=True) if t["L"] > LONG)
        out.append((f"k={k}", k * tok, 100 * sum(pk) / len(pk), long))
    return out


def search(ths, setting):
    tok = sum(t["tok"] for t in ths) / len(ths)
    solved = [t["solved"] for t in ths]
    long = sum(s for s, t in zip(solved, ths, strict=True) if t["L"] > LONG)
    return [(setting, tok, 100 * sum(solved) / len(ths), long)]


def anytime(ths):
    out = []
    for b in BUDGETS:
        ok = [t["solved"] and t["tok"] <= b for t in ths]
        long = sum(o for o, t in zip(ok, ths, strict=True) if t["L"] > LONG)
        tok = sum(min(t["tok"], b) for t in ths) / len(ths)
        out.append((f"budget {b}", tok, 100 * sum(ok) / len(ths), long))
    return out


def per_seed(seed):
    return {
        "plain": iid(load("a", seed, "plain")),
        "resample": iid(load("a", seed, "guided"), cap=CAP),
        "beam": [x for b in (1, 2, 4, 8, 16) for x in search(load("b", seed, f"beam{b}x4"), f"B={b}")],
        "smc": [x for n in (2, 4, 8, 16, 32, 64) for x in search(load("b", seed, f"smc{n}"), f"N={n}")],
        "best-first": anytime(load("a", seed, "bfs_a05")),
    }


seeds = [per_seed(s) for s in SEEDS]
rows = []
for method in COLOR:
    for i, (setting, *_) in enumerate(seeds[0][method]):
        pts = [s[method][i] for s in seeds]
        rows.append(
            {
                "method": method,
                "setting": setting,
                "tokens": round(statistics.mean(p[1] for p in pts)),
                "solved_pct": round(statistics.mean(p[2] for p in pts), 2),
                "solved_pct_min": round(min(p[2] for p in pts), 2),
                "solved_pct_max": round(max(p[2] for p in pts), 2),
                "long_solved": round(statistics.mean(p[3] for p in pts), 2),
            }
        )
with open("charts/scaling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

F = fonts(WIDTH)
data = alt.Data(values=rows)
x = alt.X(
    "tokens:Q",
    scale=alt.Scale(type="log"),
    title="Sampled tokens per theorem (log scale)",
    axis=alt.Axis(values=[250, 1000, 4000, 16000], format="~s"),
)
color = alt.Color(
    "method:N",
    scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
    legend=alt.Legend(title=None, orient="bottom", columns=5),
)
tooltip = ["method:N", "setting:N", "tokens:Q", "solved_pct:Q", "long_solved:Q"]


def panel(field, title, ytitle, labels):
    """`labels`: direct line-end labels (only where lines separate; the legend always shows)."""
    base = alt.Chart(data).encode(
        x=x,
        y=alt.Y(f"{field}:Q", title=ytitle, scale=alt.Scale(zero=False)),
        color=color,
        detail="method:N",
        tooltip=tooltip,
    )
    lines = base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=64)
    ends = (
        alt.Chart(data)
        .transform_joinaggregate(maxtok="max(tokens)", groupby=["method"])
        .transform_filter("datum.tokens == datum.maxtok")
        .mark_text(align="left", dx=7, fontSize=F["label"], color=FG)
        .encode(x=x, y=f"{field}:Q", text="method:N")
    )
    return (lines + ends if labels else lines).properties(
        width=WIDTH, height=HEIGHT, title=alt.Title(title, color=FG)
    )


chart = alt.hconcat(
    panel("solved_pct", "All 1,177 holdout theorems", "Solved (%)", labels=False),
    panel("long_solved", "The 114 with reference length > 10", "Solved (theorems)", labels=True),
    spacing=70,
).properties(
    title=alt.Title(
        "Test-time scaling of c005 on the transfer holdout (3-seed means)",
        subtitle="plain = i.i.d. samples; resample = method 1, cap 10 rejections per sample",
        subtitleColor=MUTED,
    )
)
save(chart, "scaling", WIDTH)
