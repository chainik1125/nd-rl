"""Metric and end-to-end wall-clock per combined-pipeline iteration, one dot per seed.

Left: dev-transfer theorems solved at L_true >= 7 (of 1,108), the autoresearch metric; the bar
is the 3-seed mean. Right: seconds from harness start to summary, including pretraining. c001 and
c002 loaded 186-claude's pretrained weights, so their 300 s of pretraining is added back; c003+
pretrained on the pod. The dashed line is the frozen harness's 2,400 s TOTAL_SECS.

Run from the summary dir:  python3 charts/by_iteration.py
Reads the committed results/*/results.json of the experiment's working folder.
"""

import glob
import json
import sys

import altair as alt

sys.path.insert(0, "../2026-09-21-pretrain-ablations/charts")
from theme import FG, HUE, MUTED, fonts, save  # noqa: E402

RES = "../../code/experiments/current/combined_pipeline_20260928/results"
LABEL = {
    "c001": "c001 186 + harness EI",
    "c002": "c002 + Leon's RL",
    "c003": "c003 + full pool",
    "c004": "c004 + guided sampling",
    "c005": "c005 guided, in budget",
}
LOADED_PRETRAIN_SECS = {"c001": 300, "c002": 300}
WIDTH, HEIGHT = 330, 300
f = fonts(2 * WIDTH)

seeds, means = [], []
for fn in sorted(glob.glob(f"{RES}/*/results.json")):
    cid = fn.split("/")[-2].split("-")[0]
    r = json.load(open(fn))
    for s, m in r["seed_metrics"].items():
        secs = r["total_secs"][s] + LOADED_PRETRAIN_SECS.get(cid, 0)
        seeds.append({"it": LABEL[cid], "seed": s, "metric": m, "secs": secs})
    means.append({"it": LABEL[cid], "metric": r["metric"], "txt": f"{r['metric']:.0f}"})
order = list(LABEL.values())
x = alt.X("it:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=220))

dots = alt.Chart(alt.Data(values=seeds)).mark_circle(size=90, color=HUE["water"], opacity=0.9)
tip = [alt.Tooltip("it:N", title="iteration"), alt.Tooltip("seed:N"), alt.Tooltip("metric:Q")]
left = (
    dots.encode(
        x=x,
        y=alt.Y("metric:Q", title="dev theorems solved, >= 7 lines", scale=alt.Scale(zero=False)),
        tooltip=tip,
    )
    + alt.Chart(alt.Data(values=means))
    .mark_tick(color=FG, thickness=2, size=22)
    .encode(x=x, y="metric:Q")
    + alt.Chart(alt.Data(values=means))
    .mark_text(color=FG, dx=15, align="left", fontSize=f["label"])
    .encode(x=x, y="metric:Q", text="txt:N")
).properties(width=WIDTH, height=HEIGHT, title="Metric (bar = 3-seed mean)")

right = (
    dots.encode(
        x=x,
        y=alt.Y("secs:Q", title="seconds, harness start to summary", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("it:N", title="iteration"), alt.Tooltip("seed:N"), "secs:Q"],
    )
    + alt.Chart(alt.Data(values=[{"y": 2400}]))
    .mark_rule(color=MUTED, strokeDash=[4, 4])
    .encode(y="y:Q")
    + alt.Chart(alt.Data(values=[{"y": 2400, "t": "2,400 s budget"}]))
    .mark_text(color=MUTED, align="right", dy=-7, fontSize=f["label"])
    .encode(y="y:Q", text="t:N", x=alt.value(WIDTH - 4))
).properties(width=WIDTH, height=HEIGHT, title="End-to-end wall clock incl. pretraining")

save(alt.hconcat(left, right), "by-iteration", 2 * WIDTH)
