"""How much RL brings back proof by contradiction (sigma) after it is removed from pretraining.

Reads rl_curve.json (written by extract_rl_curve.py on the VPS from create-or-elicit's ladders) and writes
rl_needed.png / rl_needed.pdf next to this file. Any arm present in the JSON is drawn, so new ladders (for example a
0.1 % dose arm) appear once they are added to the extraction script and the JSON is regenerated.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
DATA = json.loads((HERE / "rl_curve.json").read_text())

# Categorical slots 1-4 of the validated reference palette, fixed by arm, never by rank: full, knockout, 1 % dose,
# then 0.1 % dose. The first three pass all-pairs CVD (>= 9.2); with the fourth, adjacent pairs pass (>= 9.1). Aqua
# and yellow are below 3:1 on the surface, so every arm is also named in text.
COLOR = {"full": "#2a78d6", "knockout": "#eb6834", "dose1": "#1baf7a", "dose01": "#eda100"}
ROW = {"full": "full exposure\n(σ in 10.9 % of\npretraining)", "dose1": "1 % σ dose", "dose01": "0.1 % σ dose",
       "knockout": "knockout\n(σ removed)"}
ORDER = [a for a in ("full", "dose1", "dose01", "knockout") if any(o["arm"] == a for o in DATA)]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
XLIM, XTICKS = (0.015, 22), [0.03, 0.1, 0.3, 1, 3, 10]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

# The capability to reproduce: what the full-exposure bases solve before any RL update (round 1 = their samples).
full_r1 = [o["rounds"][0]["heldout_cum"] for o in DATA if o["arm"] == "full"]
THR = st.median(full_r1)


def log_x(ax):
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS, [f"{t:g}×" for t in XTICKS])
    ax.minorticks_off()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.42))

# (a) trajectories ------------------------------------------------------------------------------------------
log_x(a)
a.grid(axis="y", color=GRID, lw=0.8)
a.axhspan(min(full_r1), max(full_r1), color=COLOR["full"], alpha=0.07, lw=0)
a.axhline(THR, color=MUTED, lw=1, ls=(0, (4, 3)))
ends = {}
for o in DATA:
    xs = [r["x_train"] for r in o["rounds"]]
    ys = [r["heldout_cum"] for r in o["rounds"]]
    c = COLOR[o["arm"]]
    n_pre = 16 if o["name"] == "la_ko_s1" else len(xs)
    a.plot(xs[:n_pre], ys[:n_pre], color=c, lw=1.6, marker="o", ms=2.6, ls="-",
           alpha=0.9, solid_capstyle="round")
    if n_pre < len(xs):  # knockout s1's exploratory rounds 17-32 (17-24 create-or-elicit, 25-32 coe-dose01)
        a.plot(xs[n_pre - 1:], ys[n_pre - 1:], color=c, lw=1.6, marker="o", ms=2.6, ls=":", mfc=SURFACE)
    ends.setdefault(o["arm"], []).append((xs[-1], ys[-1]))
# Full and dose lines interleave, so only the knockout cluster is labelled at its end; the legend names every arm.
ko = ends.get("knockout", [])
if ko:
    a.text(max(p[0] for p in ko) * 1.12, max(p[1] for p in ko), "knockout", color=TEXT, fontsize=8.5, va="center")
a.set_ylim(-3, 100)
a.set_xlabel("cumulative RL compute ÷ pretraining compute (FLOPs)")
a.set_ylabel("held-out theorems that need σ,\nsolved so far (of 106)")
a.set_title("a   Held-out σ-covered theorems as RL proceeds", loc="left", fontsize=10, color=TEXT)
handles = [Line2D([], [], color=COLOR[arm], lw=1.8, marker="o", ms=3, label=ROW[arm].replace("\n", " "))
           for arm in ORDER]
handles += [Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)),
                   label=f"full bases before RL: median {THR:g} (band: range {min(full_r1)}–{max(full_r1)})"),
            Line2D([], [], color=MUTED, lw=1.4, ls=":", label=f"knockout s1, exploratory rounds 17–{max(len(o['rounds']) for o in DATA if o['name'] == 'la_ko_s1')}")]
a.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.6, handlelength=2.4)

# (b) RL needed to reach the threshold ---------------------------------------------------------------------
log_x(b)
ypos = {arm: len(ORDER) - 1 - i for i, arm in enumerate(ORDER)}
b.set_yticks([ypos[a_] for a_ in ORDER], [ROW[a_] for a_ in ORDER])
b.set_ylim(-0.7, len(ORDER) - 0.3)
b.tick_params(axis="y", length=0, labelcolor=TEXT)
for arm in ORDER:
    lads = [o for o in DATA if o["arm"] == arm]
    y0 = ypos[arm]
    if arm == "full":
        b.text(XLIM[0] * 1.15, y0, f"the reference: their median before RL\n(3 of {len(lads)} seeds already ≥ {THR:g})",
               va="center", ha="left", color=MUTED, fontsize=8)
        continue
    hits, misses = [], []
    for k, o in enumerate(sorted(lads, key=lambda o: o["seed"])):
        jitter = (k - (len(lads) - 1) / 2) * 0.09
        hit = next((r for r in o["rounds"] if r["heldout_cum"] >= THR), None)
        if hit:
            hits.append(hit["x_train"])
            b.plot([hit["x_train"]], [y0 + jitter], "o", ms=8, color=COLOR[arm], mec=SURFACE, mew=1.5, zorder=3)
        else:
            last = o["rounds"][-1]
            misses.append((last["x_train"], last["heldout_cum"]))
            b.plot([o["rounds"][0]["x_train"], last["x_train"]], [y0 + jitter] * 2, color=COLOR[arm], lw=1.2,
                   alpha=0.35, solid_capstyle="round")
            b.plot([last["x_train"]], [y0 + jitter], ">", ms=7, color=COLOR[arm], mec=SURFACE, mew=1.2, zorder=3)
    if hits:
        b.text(max(hits) * 1.35, y0, f"reached at {min(hits):.1f}–{max(hits):.1f}×\n({len(hits)} of {len(lads)} seeds)",
               va="center", color=TEXT, fontsize=8.5)
    if misses:
        best = max(m[1] for m in misses)
        b.text(XLIM[0] * 1.15, y0 - 0.38,
               f"not reached by {min(m[0] for m in misses):.1f}–{max(m[0] for m in misses):.1f}× "
               f"({len(misses)} of {len(lads)} seeds; best {best} of 106)", va="center", color=TEXT, fontsize=8.5)
b.set_xlabel("RL compute ÷ pretraining compute needed")
b.set_title(f"b   RL needed to reach the full bases' level ({THR:g} of 106)", loc="left", fontsize=10, color=TEXT)

fig.text(0.01, -0.07,
         "9.56M ALiBiGPT bases, T1 expert iteration, k 32 per round, judged by Lean (create-or-elicit; the 0.1 % arm and knockout "
         "s1's rounds 25–32: coe-dose01). Held-out theorems "
         "are never trained on; a theorem counts once any round's sample solves it.\nx: sampling (decoded and re-read "
         "state tokens) × 2N plus fine-tuning × 6N, including the 20,000 replayed pretraining records per round, over "
         "Stage-1's 6N·D (rl-compute-share's definitions;\nexcluding replay lowers x by ≈ 14 %). Round 1 is the base's own "
         f"samples, so it sits at ≈ 0.1×. Knockout values are lower bounds: those ladders stopped at round 16 (s1: "
         f"{max(len(o['rounds']) for o in DATA if o['name'] == 'la_ko_s1')}).",
         fontsize=7.4, color=MUTED, va="top")
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"rl_needed.{ext}", dpi=200, bbox_inches="tight")
print("threshold", THR, "full r1", sorted(full_r1))

# Headline: RL needed against sigma's share of pretraining ------------------------------------------------------
# One dot per seed at the cumulative RL compute after the first round whose samples have solved >= THR held-out
# members. Hollow: already there in the base's own round-1 samples (no update yet). Triangles: lower bounds.
SHARE = {"full": 10.9, "dose1": 1.0, "dose01": 0.1, "knockout": None}
ZERO_X = 0.025  # where "0 %" sits on the log axis, behind a break mark
fig2, h = plt.subplots(figsize=(6.6, 4.6))
h.set_xscale("log")
h.set_yscale("log")
h.set_xlim(0.016, 22)
h.set_ylim(0.06, 40)
xt = [ZERO_X] + [SHARE[a_] for a_ in ORDER if SHARE[a_]]
h.set_xticks(sorted(xt), [{ZERO_X: "0 %\n(removed)", 0.1: "0.1 %", 1.0: "1 %", 10.9: "10.9 %\n(all of K12)"}[t]
                          for t in sorted(xt)])
h.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1×", "0.3×", "1×", "3×", "10×", "30×"])
h.minorticks_off()
h.grid(axis="y", color=GRID, lw=0.8)
h.set_axisbelow(True)
h.text(0.05, 0.06, "//", color=MUTED, ha="center", va="center", fontsize=9, clip_on=False)  # axis break
for arm in ORDER:
    lads = sorted((o for o in DATA if o["arm"] == arm), key=lambda o: (o["name"].startswith("old"), o["seed"]))
    x0 = SHARE[arm] or ZERO_X
    spread = [0.8 + 0.4 * k / max(1, len(lads) - 1) for k in range(len(lads))]  # multiplicative jitter
    hits, firsts, misses = [], 0, []
    for f, o in zip(spread, lads):
        hit = next((r for r in o["rounds"] if r["heldout_cum"] >= THR), None)
        if hit:
            hits.append(hit["x_train"])
            first = hit["round"] == 1
            firsts += first
            h.plot([x0 * f], [hit["x_train"]], "o", ms=8, color=COLOR[arm], mfc=SURFACE if first else COLOR[arm],
                   mec=COLOR[arm] if first else SURFACE, mew=1.6 if first else 1.2, zorder=3)
        else:
            last = o["rounds"][-1]
            misses.append((last["x_train"], last["heldout_cum"], o["name"]))
            h.plot([x0 * f] * 2, [0.09, last["x_train"]], color=COLOR[arm], lw=1.1, alpha=0.35)
            h.plot([x0 * f], [last["x_train"]], "^", ms=8, color=COLOR[arm], mec=SURFACE, mew=1.2, zorder=3)
    if hits and len(hits) == len(lads) and arm != "full":
        m = st.median(hits)
        h.plot([x0 * 0.72, x0 * 1.38], [m, m], color=MUTED, lw=1.2, zorder=2)
    if arm == "full":
        rest = sorted(x for x in hits if x > 0.2)
        h.text(x0 / 1.6, 0.17, f"{firsts} of {len(lads)} seeds at the bar\nbefore any update (hollow);\n"
               f"{len(rest)} reach it at {min(rest):.2f}–{max(rest):.2f}×", ha="right", va="center", color=TEXT,
               fontsize=8)
    elif hits:
        h.text(x0 * 1.5, st.median(hits), f"{min(hits):.2f}–{max(hits):.2f}×\n(median {st.median(hits):.2f}×)",
               va="center", color=TEXT, fontsize=8.5)
    if misses:
        short = [m for m in misses if m[2] != "la_ko_s1"]
        s1 = [m for m in misses if m[2] == "la_ko_s1"]
        txt = f"not reached by {min(m[0] for m in short):.1f}–{max(m[0] for m in short):.1f}×\n({len(short)} seeds, 16 rounds)"
        if s1:
            txt += f"\nnor by {s1[0][0]:.1f}× (s1, {len(next(o for o in DATA if o['name'] == 'la_ko_s1')['rounds'])} rounds;\nbest {s1[0][1]} of 106)"
        h.text(x0 * 1.45, 6.5, txt, va="center", color=TEXT, fontsize=8.5)
h.set_xlabel("σ's share of the pretraining proofs (log scale)")
h.set_ylabel(f"RL compute ÷ pretraining compute\nneeded to solve {THR:g} of 106 held-out σ theorems")
h.set_title("RL needed to bring back proof by contradiction,\nby how much of it pretraining contained", loc="left",
            fontsize=10.5, color=TEXT)
fig2.text(0.01, -0.02,
          f"One dot per seed (9.56M ALiBiGPT, T1 expert iteration, Lean judge). Bar: {THR:g} of 106 strict σ-covered "
          "held-out theorems,\nthe full bases' median before RL. Grey ticks: medians. Only the contrast with the "
          "knockout is resolved (3 / 3 vs 0 / 6,\np = 0.0033); 1 % vs 0.1 % is descriptive at 3 seeds and one σ draw "
          "per dose (p = 0.10). Replay carries σ at each\narm's own rate, so the dose arms' recovery is not RL alone.",
          fontsize=7.4, color=MUTED, va="top")
for ext in ("png", "pdf"):
    fig2.savefig(HERE / f"rl_needed_headline.{ext}", dpi=200, bbox_inches="tight")
