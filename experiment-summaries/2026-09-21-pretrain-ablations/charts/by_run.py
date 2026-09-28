"""Pretrain loss, dev score and holdout score against run number.

Top: final pretrain val loss. Bottom: % of theorems >= 7 lines solved after RL, on dev (the pool
the loop kept runs by, 1,108) and on the transfer holdout (never looked at during the loop, 1,177).
Percentages because the two pools differ in size. Kept runs are bold and joined by a line (the
incumbent's path); discards are faint. The rule at 098 is the protocol switch to the Lean tokenizer
and reward: loss changes units there and scores are not comparable across it. Holdout only exists
for the Lean era. Same two off-scale Lean runs dropped as in loss_vs_score.py.

Run from the summary dir:  python3 charts/by_run.py
"""
import csv

import altair as alt

from theme import HUE, MUTED, fonts, save

WIDTH, HEIGHT = 560, 190
DEV_N, HOLDOUT_N = 1108, 1177
OFF_SCALE = {"166-fable2_lean-compile-dynamic", "162-fable_lean-tail-ascending"}
SERIES = {"Dev": HUE["carbon"], "Holdout": HUE["water"]}
X_SCALE, X_AXIS = alt.Scale(domain=[10, 186], nice=False), alt.Axis(tickCount=9, format="03d")
X = alt.X("num:Q", title="Run", scale=X_SCALE, axis=X_AXIS)
X_TOP = alt.X("num:Q", title=None, scale=X_SCALE, axis=X_AXIS)  # shares the bottom's "Run" title

runs = [r for r in csv.DictReader(open("charts/runs.csv")) if r["run"] not in OFF_SCALE]
loss = [{"num": int(r["num"]), "loss": float(r["loss"]), "status": r["status"], "run": r["run"]}
        for r in runs]
score = [{"num": int(r["num"]), "series": "Dev", "pct": 100 * float(r["dev"]) / DEV_N,
          "status": r["status"], "run": r["run"]} for r in runs]
score += [{"num": int(r["num"]), "series": "Holdout", "pct": 100 * float(r["holdout"]) / HOLDOUT_N,
           "status": r["status"], "run": r["run"]} for r in runs if r["holdout"]]

f = fonts(WIDTH)
KEEP = "datum.status == 'keep'"
era = alt.Chart(alt.Data(values=[{"x": 98.5}])).mark_rule(
    color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q")
era_label = alt.Chart(alt.Data(values=[{"x": 98.5, "t": "Lean tokenizer + reward →"}])).mark_text(
    align="left", dx=5, baseline="top", color=MUTED, fontSize=f["label"]).encode(
    x="x:Q", y=alt.value(4), text="t:N")


def layer(rows, x, y, color):
    """Faint dots for every run, bold dots + a line through the keeps."""
    base = alt.Chart(alt.Data(values=rows))
    enc = {"x": x, "y": y, "color": color, "tooltip": ["run:N", y.shorthand]}
    dots = base.mark_circle().encode(
        **enc, size=alt.condition(KEEP, alt.value(46), alt.value(22)),
        opacity=alt.condition(KEEP, alt.value(1.0), alt.value(0.35)))
    path = base.transform_filter(KEEP).mark_line(strokeWidth=1.6, interpolate="step-after").encode(
        x=x, y=y, color=color, detail="series:N" if "series" in rows[0] else alt.Undefined)
    return path + dots


top = (layer(loss, X_TOP, alt.Y("loss:Q", title="Pretrain Val Loss", scale=alt.Scale(zero=False),
                         axis=alt.Axis(format=".2f", tickCount=4)),
             alt.value(HUE["dollar"]))
       + era + era_label).properties(width=WIDTH, height=HEIGHT, title=alt.TitleParams(
           text="Pretrain Loss vs. RL Score, Run by Run",
           subtitle="Bold = kept run; the line follows the incumbent."))
# Manual key in the empty lower right: a real legend in a vconcat lands beside the top panel.
key_rows = [{"num": 140, "pct": 20 - 6 * i, "series": s} for i, s in enumerate(SERIES)]
key = alt.Chart(alt.Data(values=key_rows))
key = key.mark_circle(size=46, opacity=1).encode(x=X, y="pct:Q", color=alt.Color(
    "series:N", scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())), legend=None)) + \
    key.mark_text(align="left", dx=8, fontSize=f["legend"]).encode(
        x=X, y="pct:Q", text="series:N", color=alt.value(MUTED))
bottom = (layer(score, X, alt.Y("pct:Q", title="Theorems ≥ 7 Lines Solved (%)",
                             scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=5)),
                alt.Color("series:N", title=None,
                          scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                          legend=None))
          + era + key).properties(width=WIDTH, height=HEIGHT + 40)

save(alt.vconcat(top, bottom, spacing=14), "by-run", WIDTH)
