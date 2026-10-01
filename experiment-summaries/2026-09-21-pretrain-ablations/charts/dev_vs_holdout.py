"""Dev vs holdout score, one point per experiment, one panel per era that has holdout scores.

Both axes are % of theorems >= 7 lines solved (dev pool 1,108, holdout 1,177), 3-seed means, so
the dashed y = x line is "holdout agrees with dev". Only runs with all 3 seeds scored on holdout.
Holdout re-scores the same checkpoints on theorems the loop never saw, so it tests whether keeps
overfit the dev theorems; it cannot catch seed luck, which carries over to both. 166-fable2
(broken compile, ~14%) is left off, as in the other charts.

Run from the summary dir:  python3 charts/dev_vs_holdout.py
"""
import csv
import statistics

import altair as alt

from theme import HUE, MUTED, fonts, save

WIDTH, HEIGHT = 560, 360
DEV_N, HOLDOUT_N = 1108, 1177
OFF_SCALE = {"166-fable2_lean-compile-dynamic"}
ERAS = {"token": "Token Era (012–097)", "lean": "Lean Era (100–184)"}
COLOR = {"keep": HUE["carbon"], "discard": MUTED}

rows = [{**r, "dev_pct": 100 * float(r["dev"]) / DEV_N,
         "holdout_pct": 100 * float(r["holdout"]) / HOLDOUT_N}
        for r in csv.DictReader(open("charts/runs.csv"))
        if r["holdout_seeds"] == "3" and r["run"] not in OFF_SCALE]
f = fonts(WIDTH)


def panel(era, title):
    pts = [r for r in rows if r["era"] == era]
    x, y = [p["dev_pct"] for p in pts], [p["holdout_pct"] for p in pts]
    sub = (f"R = {statistics.correlation(x, y):.3f}, "
           f"Spearman = {statistics.correlation(x, y, method='ranked'):.3f}, n = {len(pts)}")
    lo, hi = min(x + y) - 1, max(x + y) + 1
    scale = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    diag = alt.Chart(alt.Data(values=[{"v": lo}, {"v": hi}])).mark_line(
        color=MUTED, strokeDash=[4, 4], strokeWidth=1.2).encode(
        x=alt.X("v:Q", scale=scale), y=alt.Y("v:Q", scale=scale))
    diag_label = alt.Chart(alt.Data(values=[{"v": hi, "t": "holdout = dev"}])).mark_text(
        align="right", dx=-4, dy=-8, color=MUTED, fontSize=f["label"]).encode(
        x="v:Q", y="v:Q", text="t:N")
    dots = alt.Chart(alt.Data(values=pts)).mark_circle(size=42).encode(
        x=alt.X("dev_pct:Q", title="Dev: Theorems ≥ 7 Lines Solved (%)", scale=scale),
        y=alt.Y("holdout_pct:Q", title="Holdout: Theorems ≥ 7 Lines Solved (%)", scale=scale),
        color=alt.Color("status:N", title=None,
                        scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
                        legend=alt.Legend(orient="top-left", direction="vertical", offset=6)),
        opacity=alt.condition("datum.status == 'keep'", alt.value(0.95), alt.value(0.55)),
        tooltip=["run:N", "dev_pct:Q", "holdout_pct:Q"],
    )
    return (diag + diag_label + dots).properties(
        width=WIDTH, height=HEIGHT, title=alt.TitleParams(text=title, subtitle=sub))


panels = [panel(era, title) for era, title in ERAS.items() if any(r["era"] == era for r in rows)]
save(alt.vconcat(*panels, spacing=28), "dev-vs-holdout", WIDTH)
