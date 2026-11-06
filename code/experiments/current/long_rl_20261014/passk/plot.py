#!/usr/bin/env python3
"""pass@k curves from passk/results/*.json: uv run --with numpy --with matplotlib passk/plot.py

pass@k per theorem is the unbiased estimator 1 - C(n - c, k) / C(n, k) over its n attempts (c correct), averaged over
theorems; bands are 95% theorem-bootstrap intervals (2,000 resamples).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [  # (result tag, label, color): the validated first three categorical slots
    ("pretrain", "pretrain (no RL)", "#2a78d6"),
    ("v2_final", "EI, 3 h RL (v2 final)", "#eb6834"),
    ("v5_round020", "EI, ~5 h RL (v5 round 20)", "#1baf7a"),
]
SETS = [("textbook72", "textbook72 (72 problems)"), ("holdout250", "holdout250 (transfer holdout, 250)")]
KS = 2 ** np.arange(9)  # 1..256


def passk(n, c, k):
    """Unbiased pass@k for one theorem: 1 - prod_{i=n-c+1}^{n} (1 - k / i)."""
    return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def curves(rows):
    per = np.array([[passk(r["n_tried"], r["n_ok"], k) for k in KS] for r in rows])  # theorems x ks
    boot = np.random.default_rng(0).integers(0, len(per), (2000, len(per)))
    lo, hi = np.percentile(per[boot].mean(1), [2.5, 97.5], axis=0)
    return per.mean(0), lo, hi


def main():
    res = {t: json.load(open(f"{HERE}/results/{t}.json")) for t, *_ in MODELS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    lines = ["| set | model | " + " | ".join(f"pass@{k}" for k in KS) + " |", "|---|---|" + "---|" * len(KS)]
    for ax, (s, title) in zip(axes, SETS, strict=True):
        for tag, label, color in MODELS:
            mean, lo, hi = curves(res[tag]["sets"][s]["rows"])
            ax.plot(KS, mean, color=color, lw=2, marker="o", ms=4, label=label)
            ax.fill_between(KS, lo, hi, color=color, alpha=0.15, lw=0)
            lines.append(f"| {s} | {label} | " + " | ".join(f"{m:.3f}" for m in mean) + " |")
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("k (attempts per theorem, T = 0.8)")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("pass@k (Lean-checked)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("long_rl untied-ei seed 1: pretrain vs expert iteration", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{HERE}/passk.png", dpi=150)
    open(f"{HERE}/results_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
