"""The proof-state experiment's chart, from charts/state_runs.csv (see state_extract.py).

  state-format.png   dev theorems solved at >= 7 lines (of 1,108), per pretraining recipe and arm:
                     bar = 3-seed median, dots = seeds (whole-proof EI's take-off failures are the
                     low dots, which is why the bar is the median, not the mean)

Arms: whole proof (C0, the post's neutral grey), whole proof with 6x RL samples (C0k192), proof
state (SN, the post's Lean-format blue: it is a format change), proof so far + state (SH).
The three hues pass validate_palette.js on the dark and light surfaces.
Run from the summary dir:  python3 charts/state_format.py
"""

import csv
import statistics
import sys
from collections import defaultdict

import altair as alt

sys.path.insert(0, "../2026-09-21-pretrain-ablations/charts")
from theme import FG, MUTED, fonts, save

ARMS = {
    "C0": ("whole proof", MUTED),
    "C0k192": ("whole proof, 6× RL samples", "#c98500"),
    "SN": ("proof state", "#3987e5"),
    "SH": ("proof so far + state", "#d55181"),
}
LABELS = [v[0] for v in ARMS.values()]
RECIPES = ["198", "207", "201", "204"]  # autoresearch score order (681, 635, 604, 548)
RECIPE_NAME = {"198": "198 (incumbent)", "207": "207", "201": "201", "204": "204"}

rows = [r for r in csv.DictReader(open("charts/state_runs.csv")) if r["status"] == "ok"]
seeds = [
    {"recipe": RECIPE_NAME[r["recipe"]], "arm": ARMS[r["arm"]][0], "metric": int(r["metric"])}
    for r in rows
]
by = defaultdict(list)
for s in seeds:
    by[s["recipe"], s["arm"]].append(s["metric"])
bars = [
    {"recipe": rc, "arm": arm, "median": statistics.median(v), "top": max(v),
     "txt": f"{statistics.median(v):.0f}"}
    for (rc, arm), v in by.items()
]

width = 620
f = fonts(width)
x = alt.X("recipe:N", sort=[RECIPE_NAME[r] for r in RECIPES], title="pretraining recipe",
          axis=alt.Axis(labelAngle=0))
off = alt.XOffset("arm:N", sort=LABELS, scale=alt.Scale(paddingInner=0.12))
color = alt.Color("arm:N", sort=LABELS,
                  scale=alt.Scale(domain=LABELS, range=[v[1] for v in ARMS.values()]),
                  legend=alt.Legend(title=None, orient="top", direction="horizontal",
                                    symbolType="square", columns=2, labelLimit=400))
y = alt.Y("median:Q", title="dev theorems solved at >= 7 lines (of 1,108)",
          scale=alt.Scale(domain=[0, 900]))
chart = (
    alt.Chart(alt.Data(values=bars))
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(x=x, xOffset=off, y=y, color=color)
    + alt.Chart(alt.Data(values=bars))
    .mark_text(color=FG, dy=-9, fontSize=f["label"])
    .encode(x=x, xOffset=off, y="top:Q", text="txt:N")
    + alt.Chart(alt.Data(values=seeds))
    .mark_circle(size=36, color=FG, opacity=0.85)
    .encode(x=x, xOffset=alt.XOffset("arm:N", sort=LABELS), y="metric:Q")
).properties(
    width=width,
    height=340,
    title=alt.Title(
        "Showing the model the proof state",
        subtitle="bar = 3-seed median, dots = seeds; same data and plain expert iteration in every run",
    ),
)
save(chart, "state-format", width)
