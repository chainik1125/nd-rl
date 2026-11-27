"""M1 figure: arm-A pass@1 and reach rate per cell (5.5 in wide, serif 7-8 pt, PDF/SVG/PNG via ndbench.figures.style).

Reads summary.json from analyze.py. Panel a: arm-A pass@1 (%) with 95 % prompt-bootstrap intervals for the cells present
(dpo stored T 1.0, C1, C4, C2, s2300 stored T 1.0, C3); the stored bars have no interval in summary.json unless
`--stored-ci` is given, so they are drawn as reference lines. Panel b: reach rate (%) of dpo baseline / C1b, s0500 stored /
S5R, s2300 stored. Usage: fig_temp.py --summary summary.json --out BASE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # ./code
from ndbench.figures import style as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True, help="base path without suffix")
    a = ap.parse_args(argv)
    res = json.load(open(a.summary))
    C = S.OKABE_ITO
    col = {
        "C1": C["blue"] if "blue" in C else "#0072B2",
        "C4": "#56B4E9",
        "C2": "#009E73",
        "C3": "#D55E00",
    }
    arms = res.get("arm_a", {})
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.4)
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
        S.panel_title(ax1, "a", "arm A (218 prompts), pass@1")
        labels, vals, errs, cols = [], [], [[], []], []
        for k in ("C1", "C4", "C2", "C3"):
            if k in arms:
                b = arms[k]
                labels.append(
                    b["cell"].split(" ", 1)[0]
                    + "\n"
                    + b["cell"].split(" ", 1)[1].replace(" T", "\nT", 1)
                )
                vals.append(b["pass1_pct"] / 100)
                errs[0].append((b["pass1_pct"] - b["pass1_lo95"]) / 100)
                errs[1].append((b["pass1_hi95"] - b["pass1_pct"]) / 100)
                cols.append(col[k])
        x = list(range(len(vals)))
        if vals:
            ax1.bar(x, vals, yerr=errs, color=cols, width=0.6, error_kw={"lw": 0.6, "capsize": 1.5})
        ax1.set_xticks(x, labels, fontsize=S.FS_SMALL)
        st = res.get("stored", {})
        # label the reference lines inside the axes (they floated past its edge)
        x_right = ax1.get_xlim()[1]
        for key, name, ls in (
            ("dpo_pass1_pct", "dpo, T 1.0", "--"),
            ("s2300_pass1_pct", "s2300, T 1.0", ":"),
        ):
            if key in st:
                ax1.axhline(st[key] / 100, color=S.REF, lw=0.6, ls=ls)
                ax1.text(
                    x_right - 0.02,
                    st[key] / 100 + 0.015,
                    name,
                    fontsize=S.FS_SMALL,
                    ha="right",
                    color=S.REF,
                )
        ax1.set_ylim(0, 1)
        S.pct_axis(ax1)
        S.panel_title(ax2, "b", "48 reach prompts, pass@1")
        names, v2, c2 = [], [], []
        r = res.get("reach_C1b")
        if r:
            names += [
                "dpo T 1.0\n(" + r["baseline"].split(" (")[0].replace("E0'", "E0'") + ")",
                "dpo\ncard (C1)",
            ]
            v2 += [r["baseline_rate_pct"] / 100, r["rate_pct"] / 100]
            c2 += ["#999999", col["C1"]]
            names += ["s2300\nT 1.0"]
            v2 += [r["s2300_stored_rate_pct"] / 100]
            c2 += ["#999999"]
        s = res.get("reach_S5R")
        if s:
            names += ["s0500\nT 1.0", "s0500\ncard (S5R)"]
            v2 += [s["stored32_rate_pct"] / 100, s["rate_pct"] / 100]
            c2 += ["#999999", col["C1"]]
        if v2:
            ax2.bar(list(range(len(v2))), v2, color=c2, width=0.6)
            ax2.set_ylim(0, 1.15 * max(v2))  # headroom for the value labels
            for i, v in enumerate(v2):  # value labels: the E0' and C1 bars are too short to see
                ax2.text(
                    i, v + 0.01, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=S.FS_SMALL
                )
            ax2.set_xticks(list(range(len(v2))), names, fontsize=S.FS_SMALL)
        S.pct_axis(ax2)
        for ax in (ax1, ax2):
            S.light_grid(ax)
        S.save(fig, Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
