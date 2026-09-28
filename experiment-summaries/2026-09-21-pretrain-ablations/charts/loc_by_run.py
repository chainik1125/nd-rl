"""Size of pretrain.py against run number.

pretrain.py is the one file agents edit (model, optimizer, schedule, batching), on top of frozen
fork modules it imports, so its size tracks how much machinery the loop has piled on. Two series:
every line, and code lines (no blanks, comments or docstrings). Kept runs are bold and joined by a
line (the incumbent's path); discards are faint. The rule at 098 is the protocol switch to the Lean
tokenizer and reward. Columns come from extract.py.

Run from the summary dir:  python3 charts/loc_by_run.py
"""
import csv

import altair as alt

from theme import HUE, MUTED, fonts, save

WIDTH, HEIGHT = 560, 260
SERIES = {"All lines": HUE["dollar"], "Code lines": HUE["recycle"]}
COLUMN = {"All lines": "lines", "Code lines": "sloc"}
X = alt.X("num:Q", title="Run", scale=alt.Scale(domain=[10, 186], nice=False),
          axis=alt.Axis(tickCount=9, format="03d"))
Y = alt.Y("n:Q", title="Lines in pretrain.py", scale=alt.Scale(domain=[0, 460]),
          axis=alt.Axis(tickCount=5))
COLOR = alt.Color("series:N", scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                  legend=None)

runs = list(csv.DictReader(open("charts/runs.csv")))
rows = [{"num": int(r["num"]), "series": s, "n": int(r[col]), "status": r["status"], "run": r["run"]}
        for r in runs for s, col in COLUMN.items()]

f = fonts(WIDTH)
KEEP = "datum.status == 'keep'"
base = alt.Chart(alt.Data(values=rows))
dots = base.mark_circle().encode(
    x=X, y=Y, color=COLOR, tooltip=["run:N", "series:N", "n:Q"],
    size=alt.condition(KEEP, alt.value(46), alt.value(22)),
    opacity=alt.condition(KEEP, alt.value(1.0), alt.value(0.35)))
path = base.transform_filter(KEEP).mark_line(strokeWidth=1.6, interpolate="step-after").encode(
    x=X, y=Y, color=COLOR, detail="series:N")
# Direct labels above each series' left end instead of a legend.
first = min(int(r["num"]) for r in runs)
label_rows = [{"num": first, "n": int(runs[0][col]), "series": s} for s, col in COLUMN.items()]
labels = alt.Chart(alt.Data(values=label_rows)).mark_text(
    align="left", dy=-12, fontSize=f["legend"]).encode(x=X, y=Y, text="series:N", color=COLOR)
era = alt.Chart(alt.Data(values=[{"x": 98.5}])).mark_rule(
    color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q")
era_label = alt.Chart(alt.Data(values=[{"x": 98.5, "t": "Lean tokenizer + reward →"}])).mark_text(
    align="left", dx=5, baseline="top", color=MUTED, fontSize=f["label"]).encode(
    x="x:Q", y=alt.value(4), text="t:N")

chart = (path + dots + labels + era + era_label).properties(
    width=WIDTH, height=HEIGHT, title=alt.TitleParams(
        text="pretrain.py Grew 2.4× Over the Loop",
        subtitle="Bold = kept run; the line follows the incumbent. Code lines exclude blanks, "
                 "comments and docstrings."))
save(chart, "loc-by-run", WIDTH)
