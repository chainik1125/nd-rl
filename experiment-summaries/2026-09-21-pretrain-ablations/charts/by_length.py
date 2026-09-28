"""Dev theorems proved, by the shortest known proof length, run by run.

Length is min(the model's shortest verified proof after pruning uncited lines, the reference
proof's n_lines), per run (extract.py's len columns). The n_lines label alone is loose: the model
beats it on ~27% of solved theorems. The per-run min moves bins between runs, so there is no
fixed denominator, and the y axis is a count (3-seed mean), on a symlog scale so 270 and 7 both
read.
13 lines is left out (never more than ~1 theorem). 11 and 12 get only the incumbent line, no
discard dots, because six dot clouds are unreadable. Kept runs are bold and joined by a line
(the incumbent's path, carried on to the last run); discards are faint. Same era rule and the
same two off-scale Lean runs dropped as in by_run.py.

Run from the summary dir:  python3 charts/by_length.py
"""
import csv

import altair as alt

from theme import MUTED, fonts, save

WIDTH, HEIGHT = 560, 320
LENGTHS = range(7, 13)
NO_DOTS = {"11 lines", "12 lines"}
# Ordered ramp, short proofs warm -> long proofs cool, all readable on the dark background.
COLOR = {"7 lines": "#f2c14e", "8 lines": "#e8833a", "9 lines": "#d1495b", "10 lines": "#b46ee0",
         "11 lines": "#5b8def", "12 lines": "#3fbfbf"}
OFF_SCALE = {"166-fable2_lean-compile-dynamic", "162-fable_lean-tail-ascending"}
LAST = 185  # just past the last run, so the incumbent line visibly holds to the end
X = alt.X("num:Q", title="Run", scale=alt.Scale(domain=[10, LAST + 18], nice=False),  # room for labels
          axis=alt.Axis(values=list(range(20, LAST, 20)), format="03d"))
Y = alt.Y("count:Q", title="Dev Theorems Proved (log scale)",
          scale=alt.Scale(type="symlog", domain=[0, 400]),
          axis=alt.Axis(values=[0, 1, 3, 10, 30, 100, 300]))
C = alt.Color("length:N", title=None, sort=list(COLOR),
              scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
              legend=None)  # lines are labelled at their right ends instead

runs = [r for r in csv.DictReader(open("charts/runs.csv")) if r["run"] not in OFF_SCALE]
rows = [{"num": int(r["num"]), "length": f"{n} lines", "count": float(r[f"len{n}"]),
         "status": r["status"], "run": r["run"]}
        for r in runs for n in LENGTHS]
keeps = [r for r in rows if r["status"] == "keep"]
incumbent = {r["length"]: r for r in keeps}  # rows are in run order, so this is the last keep
path = keeps + [{**r, "num": LAST} for r in incumbent.values()]

f = fonts(WIDTH)
faint = alt.Chart(alt.Data(values=[r for r in rows if r["status"] == "discard" and r["length"] not in NO_DOTS])).mark_circle(
    size=16, opacity=0.3).encode(x=X, y=Y, color=C, tooltip=["run:N", "length:N", "count:Q"])
line = alt.Chart(alt.Data(values=path)).mark_line(strokeWidth=1.8, interpolate="step-after").encode(
    x=X, y=Y, color=C, detail="length:N")
bold = alt.Chart(alt.Data(values=keeps)).mark_circle(size=40, opacity=1).encode(
    x=X, y=Y, color=C, tooltip=["run:N", "length:N", "count:Q"])
era = alt.Chart(alt.Data(values=[{"x": 98.5}])).mark_rule(
    color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q")
era_label = alt.Chart(alt.Data(values=[{"x": 98.5, "t": "Lean tokenizer + reward →"}])).mark_text(
    align="left", dx=5, baseline="top", color=MUTED, fontSize=f["label"]).encode(
    x="x:Q", y=alt.value(4), text="t:N")

end_labels = alt.Chart(alt.Data(values=[{**r, "num": LAST} for r in incumbent.values()])).mark_text(
    align="left", dx=4, fontSize=f["label"]).encode(x=X, y=Y, color=C, text="length:N")

chart = (faint + line + bold + era + era_label + end_labels).properties(
    width=WIDTH, height=HEIGHT,
    title=alt.TitleParams(text="Longer Proofs, Run by Run",
                          subtitle=["Theorems binned by min(model's shortest proof, reference proof).",
                                    "Bold = kept run; the line follows the incumbent. 3-seed mean."]))
save(chart, "by-length", WIDTH)
