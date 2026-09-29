"""pass@k on Dmitry's 72 textbook problems for c005, by inference method, k = 1..256.

  plain                  i.i.d. samples (what the metric uses)
  resample               redraw a line that fails the line check, <= 10 redraws per proof
  resample, no repeats   the same, but a redraw never repeats a line already rejected there

c005 (combined-pipeline run: 186 pretraining + Leon's RL + Lean format), seeds 0-2, T 0.8, 256
samples per problem. Resampling ran at a cap of 32 redraws and is read at 10. pass@k is the
unbiased estimator averaged over problems, then over seeds. The dashed line is Dmitry's ~30% goal.
The textbook72 manifest asks for dev58 and train14 to be reported separately too: see the CSV.

Run from the summary dir:  uv run --with altair --with vl-convert-python python charts/textbook_tts.py
(reads ../../artifacts/test_time_scaling/tb_s<seed>/; writes charts/textbook-tts.csv and
textbook-test-time.png/.svg)
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
KS = [2**i for i in range(9)]
CAP = 10
GOAL = 30
METHODS = {  # label -> (result file, color); slots 1-3 of the validated dark palette
    "plain": ("plain", "#3987e5"),
    "resample": ("guided", "#d95926"),
    "resample, no repeats": ("swor", "#199e70"),
}
SPLITS = {"all 72": None, "dev58": "textbook_dev", "train14": "rl_train"}
WIDTH, HEIGHT = 420, 300


def curve(ths, method):
    per = [[(s[0], s[1]) if method == "plain" else at_cap(s, CAP) for s in t["samples"]] for t in ths]
    n = len(per[0])
    return {k: 100 * statistics.mean(pass_at_k(n, sum(x[0] for x in p), k) for p in per) for k in KS}


rows = []
for label, (fname, _) in METHODS.items():
    for split, role in SPLITS.items():
        curves = []
        for seed in (0, 1, 2):
            with open(f"{ART}/tb_s{seed}/{fname}.json") as f:
                ths = json.load(f)["theorems"]
            curves.append(curve([t for t in ths if role is None or t["role"] == role], fname))
        for k in KS:
            vals = [c[k] for c in curves]
            rows.append(
                {
                    "method": label,
                    "split": split,
                    "k": k,
                    "pass": round(statistics.mean(vals), 2),
                    "seed_min": round(min(vals), 2),
                    "seed_max": round(max(vals), 2),
                }
            )
with open("charts/textbook-tts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

F = fonts(WIDTH)
data = alt.Data(values=[r for r in rows if r["split"] == "all 72"])
x = alt.X("k:Q", scale=alt.Scale(type="log", base=2), axis=alt.Axis(values=KS), title="Samples per problem (k)")
y = alt.Y("pass:Q", title="Solved (%)", scale=alt.Scale(domain=[0, 60]))
scale = alt.Scale(domain=list(METHODS), range=[c for _, c in METHODS.values()])
color = alt.Color("method:N", scale=scale, legend=alt.Legend(title=None, orient="bottom"))
base = alt.Chart(data).encode(x=x, y=y, color=color, tooltip=["method:N", "k:Q", "pass:Q", "seed_min:Q", "seed_max:Q"])
band = base.mark_area(opacity=0.18).encode(
    y="seed_min:Q", y2="seed_max:Q", color=alt.Color("method:N", scale=scale, legend=None)
)
lines = base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=64)
goal = alt.Chart(alt.Data(values=[{"g": GOAL}])).mark_rule(strokeDash=[6, 4], color=MUTED).encode(y="g:Q")
goal_text = (
    alt.Chart(alt.Data(values=[{"g": GOAL, "k": 256, "t": "Dmitry's ~30% goal"}]))
    .mark_text(align="right", dy=12, color=MUTED, fontSize=F["label"])  # empty lower right
    .encode(x="k:Q", y="g:Q", text="t:N")
)
chart = (band + lines + goal + goal_text).resolve_legend(color="independent").properties(
    width=WIDTH,
    height=HEIGHT,
    title=alt.Title(
        "pass@k on the 72 textbook problems, c005 (3-seed mean, band = seed range)",
        subtitle="resample = redraw a line that fails the line check, up to 10 times per proof",
        subtitleColor=MUTED,
        color=FG,
    ),
)
save(chart, "textbook-test-time", WIDTH)
