"""Does pretraining loss predict the RL score? One point per experiment, one panel per era.

x is the final pretrain val loss, y the loop's metric (dev theorems at >= 7 lines, 3-seed mean).
The eras get separate panels and x scales because their tokenizers differ, so their losses are
in different units. Two Lean runs are left off because they'd squash the axis and dominate R:
166-fable2 (torch.compile broke two seeds, loss ~4, score 162) and 162-fable (a curriculum ending
on the longest proofs, loss 0.41, score 633). The subtitle gives the R with them included.

The two high-leverage points both changed the training objective, which inflates the plain CE val
loss by construction: 017 label smoothing (token era, 25x the mean leverage) and 173 focal loss
(Lean era, 8x). 017 is plotted but left out of the fit and R; 173 stays in both.

Run from the summary dir:  python3 charts/loss_vs_score.py
"""
import csv
import statistics

import altair as alt

from theme import HUE, MUTED, fonts, save

WIDTH, HEIGHT = 560, 230
OFF_SCALE = {"166-fable2_lean-compile-dynamic", "162-fable_lean-tail-ascending"}
LABEL = {"030_muon-hidden": "Muon", "081-claude_polar-express": "Polar Express",
         "012_baseline-3seeds": "baseline", "100-claude_lean-baseline-2400": "Lean baseline",
         "153-claude_lean-width384": "width 384",
         "017_label-smoothing-002": "label smoothing", "173-fable_lean-focal-loss": "focal loss"}
RIGHT_EDGE = {"017_label-smoothing-002", "173-fable_lean-focal-loss"}  # label to the left instead
NO_FIT = {"017_label-smoothing-002"}
EXCLUDED = "excluded from fit"
COLOR = {"keep": HUE["carbon"], "discard": MUTED, EXCLUDED: HUE["water"]}

rows = [{**r, "loss": float(r["loss"]), "dev": float(r["dev"])}
        for r in csv.DictReader(open("charts/runs.csv"))]
for r in rows:
    r["label"] = LABEL.get(r["run"], "")
    r["side"] = "right" if r["run"] in RIGHT_EDGE else "left"  # the label's text-align
    if r["run"] in NO_FIT:
        r["status"] = EXCLUDED


def pearson(pts):
    return statistics.correlation([p["loss"] for p in pts], [p["dev"] for p in pts])


f = fonts(WIDTH)


def panel(era, title):
    pts = [r for r in rows if r["era"] == era and r["run"] not in OFF_SCALE]
    fitted = [r for r in pts if r["status"] != EXCLUDED]
    everything = [r for r in rows if r["era"] == era]
    sub = f"R = {pearson(fitted):+.2f}, n = {len(fitted)}"
    if len(fitted) < len(pts):
        sub += f"  (R = {pearson(pts):+.2f} with label smoothing)"
    if len(everything) > len(pts):
        sub += f"  (R = {pearson(everything):+.2f} with the 2 broken runs)"
    base = alt.Chart(alt.Data(values=pts))
    dots = base.mark_circle(size=42).encode(
        x=alt.X("loss:Q", title="Final Pretrain Val Loss",
                scale=alt.Scale(zero=False, nice=False, padding=8),
                axis=alt.Axis(format=".3f", tickCount=6)),
        y=alt.Y("dev:Q", title="Dev Theorems (≥ 7 Lines)", scale=alt.Scale(zero=False),
                axis=alt.Axis(tickCount=5)),
        color=alt.Color("status:N", title=None,
                        scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
                        legend=alt.Legend(orient="top-right", direction="vertical", offset=4)),
        opacity=alt.condition("datum.status == 'discard'", alt.value(0.55), alt.value(0.95)),
        tooltip=["run:N", "loss:Q", "dev:Q"],
    )
    fit = base.transform_filter(f"datum.status != '{EXCLUDED}'").transform_regression(
        "loss", "dev").mark_line(
        color=MUTED, strokeDash=[4, 4], strokeWidth=1.2).encode(x="loss:Q", y="dev:Q")
    label_color = alt.Color("status:N", scale=alt.Scale(domain=list(COLOR),
                            range=list(COLOR.values())), legend=None)
    labels = [base.transform_filter(f"datum.label != '' && datum.side == '{side}'").mark_text(
        # right-edge labels sit below-left, clear of the fit line that ends at their point
        align=side, dx=6 if side == "left" else -6, dy=-6 if side == "left" else 14,
        fontSize=f["label"]).encode(
        x="loss:Q", y="dev:Q", text="label:N", color=label_color)
        for side in ("left", "right")]
    return alt.layer(fit, dots, *labels).properties(
        width=WIDTH, height=HEIGHT,
        title=alt.TitleParams(text=title, subtitle=sub))


chart = alt.vconcat(
    panel("token", "Token Era (012–097)"),
    panel("lean", "Lean Era (100–184)"),
    spacing=28,
).resolve_scale(x="independent", y="independent")
save(chart, "loss-vs-score", WIDTH)
