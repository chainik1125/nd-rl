"""pass@k curves, k = 1..256, for the post's last keep and five big-jump keeps before it.

Models: 153 (the incumbent through 184, the post's last run) and the biggest dev jumps over their
baseline before it, spread over the loop: 030 +65 and 081 +48 pre-Lean; 101 +17, 115 +25 and
129 +15 Lean (100's +159 is the protocol switch, not a jump). final.pt (after RL), 3 seeds each,
same sampler/temperature as the harness eval.

Pools:
  holdout250  a fixed 250 of the 1,177 transfer-holdout theorems (all >= 7 lines), k = 256,
              scored by score_holdout.py --pool holdout250 (same 250 under both forks)
  textbook72  58 dev + 14 train textbook problems, k = 256, scored by --pool textbook
Pre-Lean (030, 081) proofs are judged by nd_verify alone; Lean-era ones must also pass Lean 4,
so their curves are not on the same footing.

pass@k per problem is the unbiased estimator 1 - C(n - c, k) / C(n, k) with n = 256 samples and
c successes (Chen et al. 2021), averaged over problems (unsolved ones count 0), then over seeds.

Run from the summary dir:  python3 charts/passk.py   (writes charts/passk.csv, passk-<pool>.png/.svg)
"""
import csv
import json
import math
import os
import statistics

import altair as alt

from theme import fonts, save

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HOLDOUT = f"{ROOT}/artifacts/autoresearch/holdout"
MODELS = {  # run -> short label; ordered by run number
    "030_muon-hidden": "030 Muon (pre-Lean)",
    "081-claude_polar-express": "081 Polar Express (pre-Lean)",
    "101-claude_lean-depth-6": "101 depth 6",
    "115-fable_lean-alibi-half": "115 ALiBi half",
    "129-claude_lean-lr-floor-03": "129 LR floor 0.3",
    "153-claude_lean-width384": "153 width 384 (post's last keep)",
}
PRELEAN = {"030_muon-hidden", "081-claude_polar-express"}
POOLS = {"holdout250": "Transfer Holdout (250 Theorems ≥ 7 Lines)",
         "textbook": "Textbook Problems (72)"}
KS = [2**i for i in range(9)]
# Hot -> cold with run order (early runs hot, late runs cold). Neighbours are close in hue, so each
# line also carries its run number at the k = 256 end; identity never rests on color alone.
RAMP = ["#d55753", "#eb7c33", "#ecaa0b", "#00bedb", "#2389e2", "#765fca"]

def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k)


def curve(result):
    n = result["attempts_per_theorem"]
    counts = [p["n_ok"] for p in result["problems"].values()]  # textbook lists solved ones only
    return {k: 100 * sum(pass_at_k(n, c, k) for c in counts) / result["n"] for k in KS}


rows = []
for run, label in MODELS.items():
    for pool in POOLS:
        seeds = [json.load(open(f"{HOLDOUT}/{run}_s{s}.{pool}.json")) for s in range(3)
                 if os.path.exists(f"{HOLDOUT}/{run}_s{s}.{pool}.json")]
        if not seeds:
            continue
        assert {r["attempts_per_theorem"] for r in seeds} == {256}, run
        curves = [curve(r) for r in seeds]
        rows += [{"run": run, "model": label, "pool": pool, "k": k, "seeds": len(seeds),
                  "pass": round(statistics.mean(c[k] for c in curves), 2),
                  "seed_min": round(min(c[k] for c in curves), 2),
                  "seed_max": round(max(c[k] for c in curves), 2),
                  "prelean": run in PRELEAN} for k in KS]

with open("charts/passk.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

WIDTH, HEIGHT = 520, 340
f = fonts(WIDTH)
color = alt.Color("model:N", title=None, sort=list(MODELS.values()),
                  scale=alt.Scale(domain=list(MODELS.values()), range=RAMP),
                  legend=alt.Legend(orient="bottom", columns=3, labelLimit=0, symbolType="stroke"))


def end_labels(pts, lo, hi):
    """Run numbers at k = 256, nudged apart so none overlap (min gap ~ one label height)."""
    ends = sorted((r for r in pts if r["k"] == KS[-1]), key=lambda r: r["pass"])
    gap, y = 0.05 * (hi - lo), lo
    out = []
    for r in ends:
        y = max(r["pass"], y + gap if out else r["pass"])
        out.append({**r, "label_y": y, "num": r["run"][:3]})
    return out


def chart(pool, title):
    pts = [r for r in rows if r["pool"] == pool]
    lo, hi = math.floor(min(r["pass"] for r in pts)) - 1, math.ceil(max(r["pass"] for r in pts)) + 1
    x = alt.X("k:Q", title="k (Samples per Problem)",
              scale=alt.Scale(type="log", base=2, domain=[1, 256], nice=False),
              axis=alt.Axis(values=KS, format="d"))
    y_scale = alt.Scale(domain=[lo, hi], nice=False)
    base = alt.Chart(alt.Data(values=pts)).encode(
        x=x, y=alt.Y("pass:Q", title="pass@k (%)", scale=y_scale), color=color,
        strokeDash=alt.condition("datum.prelean", alt.value([5, 3]), alt.value([1, 0])),
        tooltip=["model:N", "k:Q", "pass:Q", "seed_min:Q", "seed_max:Q"],
    )
    labels = alt.Chart(alt.Data(values=end_labels(pts, lo, hi))).mark_text(
        align="left", dx=8, fontSize=f["label"]).encode(
        x=x, y=alt.Y("label_y:Q", scale=y_scale), text="num:N", color=color)
    seeds = sorted({r["seeds"] for r in pts})
    sub = (f"Mean over {'/'.join(map(str, seeds))} seeds of final.pt. "
           "Dashed = pre-Lean, judged by nd_verify only.")
    layers = base.mark_line(strokeWidth=2) + base.mark_circle(size=40, opacity=1) + labels
    save(layers.properties(width=WIDTH, height=HEIGHT,
                           title=alt.TitleParams(text=title, subtitle=sub)),
         f"passk-{pool}", WIDTH)


for pool, title in POOLS.items():
    if any(r["pool"] == pool for r in rows):
        chart(pool, title)
