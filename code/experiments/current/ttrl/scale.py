#!/usr/bin/env python3
"""The 5x run: dev solved (recorded 64-sample baseline + targets newly solved) against test-time
compute in units of x = the seed's pretrain + RL seconds, per arm, averaged over seeds.

  python3 scale.py artifacts/ttrl --summaries <dir with summary_c005_s{0,1,2}.json> --chart <png>
"""

import argparse
import json
import os

ARMS = {"control": "sample more", "selftrain": "targets-only EI", "ttrl": "TTRL"}
COLORS = {"control": "#9a9a9a", "selftrain": "#e08a2e", "ttrl": "#2a6fdb"}
MULTIPLES = (1, 2, 3, 4, 5)


def solved_at(run, secs):
    """Targets solved by the last round that finished within `secs` of loop time."""
    done = [r["targets_solved"] for r in run["rounds"] if r["loop_secs"] <= secs]
    return done[-1] if done else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--chart")
    a = ap.parse_args()
    rows, curves = {}, {}
    for s in (0, 1, 2):
        with open(os.path.join(a.summaries, f"summary_c005_s{s}.json")) as f:
            m = json.load(f)
        x, base = m["pretrain"]["secs"] + m["rl"]["secs"], m["eval"]["solved"]
        for arm in ARMS:
            path = os.path.join(a.root, f"devall_s{s}_{arm}", f"{arm}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                run = json.load(f)
            rows[(s, arm)] = [base + solved_at(run, k * x) for k in MULTIPLES]
            curves[(s, arm)] = (
                [0] + [r["loop_secs"] / x for r in run["rounds"]],
                [base] + [base + r["targets_solved"] for r in run["rounds"]],
            )
        print(f"seed {s}: x = {x:.0f} s, baseline {base}")
    print("\n| arm | seed | " + " | ".join(f"{k}x" for k in MULTIPLES) + " |")
    print("| --- | --- |" + " --- |" * len(MULTIPLES))
    for arm, label in ARMS.items():
        seeds = [s for s in (0, 1, 2) if (s, arm) in rows]
        for s in seeds:
            print(f"| {label} | {s} | " + " | ".join(map(str, rows[(s, arm)])) + " |")
        if len(seeds) > 1:
            mean = [sum(rows[(s, arm)][i] for s in seeds) / len(seeds) for i in range(5)]
            print(f"| {label} | mean | " + " | ".join(f"{v:.0f}" for v in mean) + " |")
    if a.chart:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for s, ax in zip((0, 1, 2), axes, strict=True):
            for arm, label in ARMS.items():
                if (s, arm) in curves:
                    xs, ys = curves[(s, arm)]
                    ax.step(xs, ys, where="post", color=COLORS[arm], label=label)
                    ax.annotate(
                        f"{ys[-1]}", (xs[-1], ys[-1]), xytext=(4, -3), textcoords="offset points"
                    )
            ax.set_title(f"seed {s}")
            ax.set_xlabel("test-time compute (x = pretrain + RL time)")
            ax.set_xlim(0, max(MULTIPLES) + 0.3)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, loc="lower right")
        axes[0].set_ylabel("dev theorems solved (of 1,108)")
        fig.tight_layout()
        fig.savefig(a.chart, dpi=150)


if __name__ == "__main__":
    main()
