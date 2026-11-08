#!/usr/bin/env python3
"""The headline figure from ladder.json: decoder cosine and strict RL-only count against RL hours,
expert iteration (two seeds) against the continued-pretraining control, with the two reference
levels (Leon's 0.29x GRPO end; two independent pretrains of the same seed).

  uv run --with matplotlib python figure.py   # writes figures/cos_vs_rl.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# the dataviz reference palette, slots 1-3 (validated all-pairs), text and grid tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def series(rows, run_prefixes, seed):
    pts = [
        r
        for r in rows
        if not r["planted"]
        and r["run"].endswith(f"s{seed}")
        and r["run"].startswith(run_prefixes)
        and r["rl_secs"] > 0
    ]
    return sorted(pts, key=lambda r: r["rl_secs"])


def main():
    rows = json.loads((HERE / "ladder.json").read_text())
    ei = {s: series(rows, ("v4", "v5"), s) for s in (0, 1)}
    cpt = series(rows, ("cpt",), 0)
    twin = {
        s: next(r for r in rows if r["run"] == f"v2_untied-ei_s{s}" and r["stem"] == "final")
        for s in (0, 1)
    }
    null = [r["cos_w"] for r in rows if r["run"].startswith("v2") and r["stem"] == "pretrain"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=1)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.set_xlabel("RL GPU-hours (fine-tune rounds for the control)")
        a.set_xlim(0, 7.3)

    line = dict(lw=2, solid_capstyle="round", solid_joinstyle="round", ms=7, mec=SURFACE, mew=1.5)
    for s, c in ((0, BLUE), (1, ORANGE)):
        h = [r["rl_hours"] for r in ei[s]]
        ax.plot(
            h,
            [r["cos_w"] for r in ei[s]],
            "o-",
            color=c,
            label=f"expert iteration, seed {s}",
            **line,
        )
        ax2.plot(
            h,
            [r["n_rl"] for r in ei[s]],
            "o-",
            color=c,
            label=f"expert iteration, seed {s}",
            **line,
        )
        ax.plot(
            [twin[s]["rl_hours"]],
            [twin[s]["cos_w"]],
            "s",
            color=c,
            ms=9,
            mfc=SURFACE,
            mew=2,
            label=f"independent 3 h run, seed {s}" if s == 0 else None,
        )
    ax.plot(
        [r["rl_hours"] for r in cpt],
        [r["cos_w"] for r in cpt],
        "^--",
        color=AQUA,
        label="continued pretraining, no RL (control)",
        **line,
    )
    ax2.plot(
        [r["rl_hours"] for r in cpt],
        [r["n_rl"] for r in cpt],
        "^--",
        color=AQUA,
        label="continued pretraining, no RL (control)",
        **line,
    )

    # reference levels
    ax.axhline(0.945, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(
        7.2,
        0.952,
        "Leon's GRPO end, 0.29x pretraining FLOPs: 0.945",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK2,
    )
    lo, hi = min(null), max(null)
    ax.axhspan(lo, hi, color=INK2, alpha=0.12, lw=0)
    ax.text(
        7.2,
        hi + 0.008,
        "two independent pretrains of the same seed",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK2,
    )
    ax2.axhspan(4, 5, color=INK2, alpha=0.12, lw=0)
    ax2.text(
        7.2,
        5.15,
        "count between two independent pretrains",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK2,
    )

    # direct end labels
    for s in (0, 1):
        last = ei[s][-1]
        ax.annotate(
            f"{last['cos_w']:.2f}",
            (last["rl_hours"], last["cos_w"]),
            xytext=(6, 8 if s == 0 else -12),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    ax.annotate(
        f"control ends at {cpt[-1]['cos_w']:.2f}, EI at {ei[0][7]['cos_w']:.2f}",
        (cpt[-1]["rl_hours"], cpt[-1]["cos_w"]),
        xytext=(16, -70),
        textcoords="offset points",
        fontsize=8.5,
        color=INK2,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
    )
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("decoder cosine, base vs checkpoint")
    ax.set_title(
        "The base's features rotate with training steps, not with RL",
        loc="left",
        fontsize=12,
        color=INK,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax2.set_ylim(-0.5, 18)
    ax2.set_ylabel("strict RL-only latents (of 8,192)")
    ax2.set_title("Strict RL-only latents: one seed, late", loc="left", fontsize=12, color=INK)
    fig.text(
        0.01,
        0.01,
        "Crosscoders at layer 3 of 6, Leon's recipe (BatchTopK, 8,192 latents, k 32, 40M tokens), "
        "one per base-checkpoint pair.\n"
        "Long-RL untied expert iteration (long_rl_20261014), two seeds; control: same fine-tune "
        "recipe, on pretraining records only.",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "cos_vs_rl.png", dpi=200, facecolor=SURFACE)
    fig.savefig(HERE / "figures" / "cos_vs_rl.pdf", facecolor=SURFACE)
    print("wrote figures/cos_vs_rl.{png,pdf}")


if __name__ == "__main__":
    main()
