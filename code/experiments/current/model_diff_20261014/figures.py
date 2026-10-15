# ruff: noqa: E402,E501,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Figures for the write-up -> figs/*.png.  uv run -q --with matplotlib --with torch python figures.py
f1_hardpop_curves.png   hard-pop accuracy over RL rounds (v4 trajectories) + v2/v3 endpoints
f2_grpo_timeline.png    spine-3 ladder targets newly solved per 10 rounds, 4 GRPO runs
f3_kr_cells.png         pop accuracy by (k, r) cell: pretrained vs GRPO wall vs GRPO no-wall vs EI
f4_decoder_cos.png      crosscoder decoder-cosine distributions: GRPO pre/fin, EI pre/fin, two pretrain seeds
"""

import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

os.makedirs("figs", exist_ok=True)
S1, S2, S3, S4, S5, S6, S7, S8 = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "legend.frameon": False,
    }
)
HP = json.load(open("out/hardpop.json"))

# f1: hard-pop accuracy over rounds
fig, ax = plt.subplots(figsize=(7.5, 4))
runs = [
    ("v4_untied-ei_s0", "EI v4 s0", S3, ["005", "010", "015", "020", "030"]),
    ("v4_untied-ei_s1", "EI v4 s1", S6, ["005", "010", "015", "020", "030"]),
    ("v4_untied-grpo_s0", "GRPO v4 s0", S2, ["020", "040", "060", "080", "100", "120", "140"]),
    ("v4_untied-grpo_s1", "GRPO v4 s1", S4, ["020", "040", "060", "080", "100", "120", "140"]),
]
for key, lab, col, rounds in runs:
    xs = [0] + [int(r) for r in rounds]
    ys = [HP[f"{key}/pretrain.pt"]["hard"]] + [HP[f"{key}/rl_round_{r}.pt"]["hard"] for r in rounds]
    ax.plot(
        xs,
        [100 * y for y in ys],
        color=col,
        lw=2,
        marker="o",
        ms=4,
        label=lab,
        solid_capstyle="round",
    )
    if lab.startswith("EI"):
        ax.annotate(
            lab,
            (xs[-1], 100 * ys[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=INK2,
            fontsize=9,
        )
ax.axhline(100 * HP["v3_untied-grpo_s0/final.pt"]["hard"], color=S1, lw=2, ls=(0, (1, 2)))
ax.annotate(
    "GRPO v3 s0 final (the wall)",
    (2, 100 * HP["v3_untied-grpo_s0/final.pt"]["hard"]),
    xytext=(0, 4),
    textcoords="offset points",
    color=INK2,
    fontsize=9,
)
ax.set_xlabel("RL round (oracle call)")
ax.set_ylabel("hard-pop accuracy (%)")
ax.set_ylim(0, 100)
ax.set_xlim(-3, 175)
ax.set_title(
    "Hard pops (depth after close ≠ max(0, 2 − r)): EI learns them, GRPO mostly doesn't",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.legend(loc="center right", fontsize=9)
fig.tight_layout()
fig.savefig("figs/f1_hardpop_curves.png", dpi=150)
plt.close(fig)

# f2: GRPO spine-3 timeline
import contextlib
import io
import sys

E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
sys.path.insert(0, f"{E}/analysis_20261014/wall")
with contextlib.redirect_stdout(io.StringIO()):
    import spine as SP
R = {r["name"]: r for r in map(json.loads, open(f"{E}/data/rl.jsonl"))}
sp3 = {n for n, r in R.items() if r["src"].startswith("ladder") and SP.spine(SP.goal(r)) == 3}
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
G = [
    ("GRPO v3 s0", f"{A}/v3/long-rl-20261014-v3_untied-grpo_s0/found_targets.jsonl", S1),
    ("GRPO v3 s1", f"{A}/v3/long-rl-20261014-v3_untied-grpo_s1/found_targets.jsonl", S2),
    ("GRPO v4 s0", "ckpt/runs/long-rl-20261014-v4_untied-grpo_s0/found_targets.jsonl", S3),
    ("GRPO v4 s1", "ckpt/runs/long-rl-20261014-v4_untied-grpo_s1/found_targets.jsonl", S4),
]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for lab, fn, col in G:
    first = {}
    for r in map(json.loads, open(fn)):
        if r["name"] in sp3:
            first[r["name"]] = min(first.get(r["name"], 10**9), r["round"])
    c = collections.Counter(v // 10 * 10 for v in first.values())
    xs = list(range(0, 180, 10))
    cum = 0
    ys = []
    for x in xs:
        cum += c[x]
        ys.append(cum)
    ax.plot([x + 10 for x in xs], ys, color=col, lw=2, label=f"{lab} ({len(first)})")
ax.set_xlabel("RL round")
ax.set_ylabel("spine-3 targets solved (cumulative)")
ax.set_title(
    "1,805 ladder targets with goal A→(B→(C→D)): one GRPO run of four takes off",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.legend(fontsize=9, loc="center right")
fig.tight_layout()
fig.savefig("figs/f2_grpo_timeline.png", dpi=150)
plt.close(fig)

# f3: (k, r) cells
cells = ["k1r1", "k1r2", "k2r1", "k1r3", "k2r2", "k3r1"]
pre_keys = [k for k in HP if k.endswith("pretrain.pt")]
groups = [
    ("8 pretrained models (mean)", [HP[k] for k in pre_keys], "#a7a59d"),
    (
        "GRPO, no wall (3 finals)",
        [
            HP[k]
            for k in (
                "v3_untied-grpo_s1/final.pt",
                "v4_untied-grpo_s0/final.pt",
                "v4_untied-grpo_s1/final.pt",
            )
        ],
        S2,
    ),
    ("GRPO v3 s0 (wall)", [HP["v3_untied-grpo_s0/final.pt"]], S1),
    (
        "EI (v2 s0, v2 s1, v4 s0)",
        [
            HP[k]
            for k in (
                "v2_untied-ei_s0/final.pt",
                "v2_untied-ei_s1/final.pt",
                "v4_untied-ei_s0/final.pt",
            )
        ],
        S3,
    ),
]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
w = 0.2
for gi, (lab, rows, col) in enumerate(groups):
    ys = [100 * sum(r[c] for r in rows) / len(rows) for c in cells]
    ax.bar(
        [i + (gi - 1.5) * w for i in range(len(cells))], ys, width=w * 0.92, color=col, label=lab
    )
ax.set_xticks(range(len(cells)))
ax.set_xticklabels(
    ["k=1 r=1\n(heuristic ok)", "k=1 r=2", "k=2 r=1", "k=1 r=3", "k=2 r=2", "k=3 r=1"]
)
ax.set_ylabel("pop accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title(
    "Pop accuracy by k (depth returned to) and r (close index in a run)",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.legend(fontsize=8.5, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig("figs/f3_kr_cells.png", dpi=150)
plt.close(fig)

# f4: decoder cosine distributions
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for run, lab, col in (
    ("runs/g0_L3", "GRPO s0 (wall) pre vs final", S1),
    ("runs/g1_L3", "GRPO s1 (no wall) pre vs final", S2),
    ("runs/e1_L3", "EI s1 pre vs final", S3),
    ("runs/x01pre_L3", "two pretraining seeds", S8),
):
    if not os.path.exists(f"{run}/cc.pt"):
        continue
    cc = torch.load(f"{run}/cc.pt", map_location="cpu")
    cs = cc["cos"][cc["freq"] > 0]
    h = torch.histc(cs, 40, -1, 1)
    xs = torch.linspace(-1 + 1 / 40, 1 - 1 / 40, 40)
    ax.plot(xs, h / h.sum() * 100, color=col, lw=2, label=lab)
ax.set_xlabel("cosine between a latent's two decoder directions (layer 3)")
ax.set_ylabel("% of alive latents")
ax.set_title(
    "Crosscoders find no model-exclusive latents; what moves is the decoder cosine",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("figs/f4_decoder_cos.png", dpi=150)
plt.close(fig)
print("ok")
