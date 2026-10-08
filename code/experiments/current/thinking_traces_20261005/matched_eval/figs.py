"""Figures for the 2026-10-07 RL comparison (thinking vs no-think, GRPO vs EI, forcing, graded).

    code/.venv/bin/python -m code.experiments.current.thinking_traces_20261005.matched_eval.figs \
        [--out experiment-summaries/2026-10-05-thinking-traces/figures]

Reads the stored eval rows under /scratch/cowifornia/nd-rl/think/runs/eval/ (CPU only). Points are
pass@k (unbiased, Chen et al. 2021) with 95% percentile intervals from `code.thinking.stats`
(theorem bootstrap; one seed per arm). The 13-20 reads use the 986 items of the 1,000-item eval
subsample that are unchanged in the final sealed set. Panels whose rows are missing are skipped.
Palette: the dataviz reference categorical slots 1-4 (validated: CVD and normal-vision pass; slots
3-4 below 3:1 contrast, so every series is direct-labelled and has its own marker).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from matplotlib.ticker import NullFormatter

from code.ndbench.figures import style as S
from code.thinking import stats

R = Path("/scratch/cowifornia/nd-rl/think/runs")
E = R / "eval"
M = E / "matched"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#6b6b6b"


def valid_1320() -> set[str]:
    fin = {}
    with open(R / "pools/test_13_20/test_13_20.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fin[r["name"]] = r
    ok = set()
    with open(M / "sets/t1320_n1000.jsonl") as f:
        for line in f:
            r = json.loads(line)
            g = fin.get(r["name"])
            if (
                g
                and g["prompt"] == r["prompt"]
                and g.get("reference_lines") == r.get("reference_lines")
            ):
                ok.add(r["name"])
    return ok


class Rows:
    """Loads one rows file as a one-seed arm, filtered to `keep` names when given."""

    def __init__(self, keep: set[str] | None, tmp: Path) -> None:
        self.keep, self.tmp, self.n = keep, tmp, 0

    def arm(self, path: Path) -> stats.Arm | None:
        if not path.exists():
            return None
        if self.keep is not None:
            self.n += 1
            out = self.tmp / f"{self.n}_{path.parent.name}_{path.name}"
            with open(path) as f, open(out, "w") as g:
                g.writelines(line for line in f if json.loads(line)["name"] in self.keep)
            path = out
        return stats.load_arm(path.parent.name, [str(path)])


def est(arm: stats.Arm | None, k: int = 1) -> tuple[float, float, float] | None:
    if arm is None:
        return None
    e = stats.ci(arm, k, draws=2000)
    return e.est, e.lo, e.hi


def errbar(ax, xs, ests, color, marker, label, ls="-", dx=0.0, dy=0.0) -> None:
    pts = [(x + dx, e) for x, e in zip(xs, ests, strict=True) if e is not None]
    if not pts:
        return
    x = [p[0] for p in pts]
    y = [p[1][0] for p in pts]
    lo = [p[1][0] - p[1][1] for p in pts]
    hi = [p[1][2] - p[1][0] for p in pts]
    ax.errorbar(
        x,
        y,
        yerr=[lo, hi],
        color=color,
        marker=marker,
        ms=4.2,
        lw=1.4,
        ls=ls,
        capsize=1.6,
        elinewidth=0.7,
        label=label,
        mec="white",
        mew=0.6,
    )
    ax.annotate(
        label,
        (x[-1], y[-1]),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=S.FS_TICK,
        color=S.INK2,
        va="center",
    )


def fig_curves(rows: Rows, out: Path) -> None:
    """13-20 pass@1 over GRPO iterations: thinking (B 256, B 0) vs no-think, per seed."""
    seeds = {
        0: (
            [39, 49, 52, 60, 70, 78],
            lambda i: (
                M
                / (
                    "t1320_grpoT1v1e_it39"
                    if i == 39
                    else "t1320_grpoT1v1e_it78"
                    if i == 78
                    else f"diag/t1320_grpoT1v1e_it{i}"
                )
            ),
            lambda i: M / f"t1320_grpoT0_it{i}",
        ),
        1: (
            [10, 20, 25, 30, 35, 39, 45, 50, 55, 60, 65, 70, 75, 78],
            lambda i: (
                M
                / (
                    f"t1320_grpoT1v1es1_it{i}"
                    if i in (39, 78)
                    else f"curve/t1320_grpoT1v1es1_it{i}"
                )
            ),
            lambda i: M / f"t1320_grpoT0s1_it{i}",
        ),
    }
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.4, layout="constrained")
        axs = fig.subplots(1, 2, sharey=True)
        for ax, (seed, (its, think_dir, t0_dir)) in zip(axs, seeds.items(), strict=True):
            t256 = [est(rows.arm(think_dir(i) / "rows_plain_B256.jsonl")) for i in its]
            t0 = [est(rows.arm(think_dir(i) / "rows_plain_B0.jsonl")) for i in its]
            nt_its = [39, 78]
            nt = [est(rows.arm(t0_dir(i) / "rows_plain_B0.jsonl")) for i in nt_its]
            if seed == 0:
                ax.axvspan(47, 55, color=S.SHADE, zorder=0)
                ax.text(
                    51,
                    0.97,
                    "unstable\nit 47-55",
                    ha="center",
                    va="top",
                    fontsize=S.FS_SMALL,
                    color=GRAY,
                    transform=ax.get_xaxis_transform(),
                )
            errbar(ax, its, t256, BLUE, "o", "thinking, B 256", dy=7 if seed == 0 else 0)
            errbar(ax, its, t0, AQUA, "^", "thinking, B 0", dy=-7 if seed == 0 else 0)
            errbar(ax, nt_its, nt, ORANGE, "s", "no-think GRPO", ls="--")
            S.panel_title(ax, "ab"[seed], f"seed {seed}")
            ax.set_xlabel("GRPO iteration (39 = 1 EI round of attempts)")
            ax.set_xlim(0, 100)
            S.pct_axis(ax)
            S.light_grid(ax)
        axs[0].set_ylabel("pass@1, 13-20-line proofs")
        S.save(fig, out / "fig_rl_t1320_curves")
        S.close(fig)


def fig_matched(rows_all: Rows, rows_1320: Rows, out: Path) -> None:
    """Matched-attempt bars: EI vs GRPO x no-think vs thinking (B 256), 3 test sets, pass@1/@5."""
    arms = [
        ("EI no-think", AQUA, "eiT0", "B0"),
        ("EI thinking", YELLOW, "eiT1v1e", "B256"),
        ("GRPO no-think", ORANGE, "grpoT0", "B0"),
        ("GRPO thinking", BLUE, "grpoT1v1e", "B256"),
    ]
    sets = [
        ("hard12 test (1,000)", "h12", rows_all),
        ("textbook72", "tb72", rows_all),
        ("13-20 lines (986)", "t1320", rows_1320),
    ]
    pts = [("1 round", "r1", "it39"), ("2 rounds", "r2", "it78")]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 3.6, layout="constrained")
        axs = fig.subplots(2, 3)
        for row, k in enumerate((1, 5)):
            for col, (title, s, rows) in enumerate(sets):
                ax = axs[row][col]
                for j, (lab, color, tag, b) in enumerate(arms):
                    for i, (_, r, it) in enumerate(pts):
                        step = r if tag.startswith("ei") else it
                        e = est(rows.arm(M / f"{s}_{tag}_{step}" / f"rows_plain_{b}.jsonl"), k)
                        if e is None:
                            continue
                        x = i + (j - 1.5) * 0.19
                        ax.bar(
                            x,
                            e[0],
                            width=0.17,
                            color=color,
                            edgecolor="white",
                            linewidth=0.6,
                            label=lab if (row, col, i) == (0, 0, 0) else None,
                        )
                        ax.errorbar(
                            x,
                            e[0],
                            yerr=[[e[0] - e[1]], [e[2] - e[0]]],
                            color=S.INK2,
                            lw=0.7,
                            capsize=1.2,
                        )
                ax.set_xticks([0, 1], [p[0] for p in pts])
                S.pct_axis(ax)
                S.light_grid(ax)
                if row == 0:
                    S.panel_title(ax, None, title)
                if col == 0:
                    ax.set_ylabel(f"pass@{k}")
        fig.legend(loc="outside upper center", ncol=4, frameon=False, fontsize=S.FS_TICK)
        S.save(fig, out / "fig_rl_matched")
        S.close(fig)


def fig_forcing(rows: Rows, out: Path) -> None:
    """Budget forcing: pass@1 vs the forced minimum think tokens per step (cap 256)."""
    mins = [0, 64, 96, 128]
    models = [
        ("Stage-1 T1v1", ORANGE, "s", "stage1_T1v1"),
        ("RL'd (GRPO, B 256)", BLUE, "o", "b256_it78"),
    ]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.3, layout="constrained")
        axs = fig.subplots(1, 2)
        for ax, (title, s) in zip(
            axs, [("textbook72", "tb72"), ("hard12 test (300)", "h12s")], strict=True
        ):
            for lab, color, mk, tag in models:
                es = [
                    est(
                        rows.arm(
                            E
                            / "followup"
                            / f"force_{s}_{tag}_B256_min{m}"
                            / "rows_plain_B256.jsonl"
                        )
                    )
                    for m in mins
                ]
                errbar(ax, mins, es, color, mk, lab)
            S.panel_title(ax, None, title)
            ax.set_xlabel("forced minimum think tokens per step")
            ax.set_xticks(mins)
            ax.set_xlim(-8, 175)
            S.pct_axis(ax)
            S.light_grid(ax)
        axs[0].set_ylabel("pass@1")
        S.save(fig, out / "fig_forcing")
        S.close(fig)


def fig_graded(rows: Rows, out: Path) -> None:
    """Stage-1 effort curve: graded-length traces (T1v1g4) vs original T1v1, T0 as reference."""
    levels = [0, 16, 64, 256]
    panels = [
        (
            "hard12 test (1,000)",
            M / "h12_T1v1g4_stage1",
            E / "h12_T1v1_s0",
            [E / "h12_T0_s1", E / "h12_T0_s2"],
        ),
        (
            "textbook72",
            M / "tb72_T1v1g4_stage1",
            E / "tb72_T1v1_s0",
            [E / "tb72_T0_s0", E / "tb72_T0_s1", E / "tb72_T0_s2"],
        ),
    ]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.3, layout="constrained")
        axs = fig.subplots(1, 2)
        for ax, (title, g4, t1, t0s) in zip(axs, panels, strict=True):
            xs = list(range(len(levels)))
            errbar(
                ax,
                xs,
                [est(rows.arm(g4 / f"rows_plain_B{b}.jsonl")) for b in levels],
                BLUE,
                "o",
                "graded traces (T1v1g4)",
                dy=-8 if "textbook" in title else 0,
            )
            errbar(
                ax,
                [0, 3],
                [est(rows.arm(t1 / f"rows_plain_B{b}.jsonl")) for b in (0, 256)],
                AQUA,
                "^",
                "original traces (T1v1)",
                ls=":",
            )
            arms0 = [p / "rows_plain_B0.jsonl" for p in t0s if (p / "rows_plain_B0.jsonl").exists()]
            if arms0:
                e0 = stats.ci(stats.load_arm("T0", [str(p) for p in arms0]), 1, draws=2000)
                ax.axhline(e0.est, color=GRAY, ls="--", lw=0.9)
                ax.text(
                    0.99 if "hard12" in title else 0.01,
                    e0.est,
                    f" T0, no-think model ({len(arms0)} seeds) ",
                    transform=ax.get_yaxis_transform(),
                    fontsize=S.FS_SMALL,
                    color=GRAY,
                    va="top" if "hard12" in title else "bottom",
                    ha="right" if "hard12" in title else "left",
                )
            ax.set_xticks(xs, [f"e{b}" for b in levels])
            ax.set_xlabel("effort level (think budget per step)")
            ax.set_xlim(-0.3, 4.4)
            S.panel_title(ax, None, title)
            S.pct_axis(ax)
            S.light_grid(ax)
        axs[0].set_ylabel("pass@1, Stage 1 (before RL)")
        S.save(fig, out / "fig_graded_stage1")
        S.close(fig)


def fig_tokens(rows: Rows, out: Path) -> None:
    """13-20 pass@k at matched sampled tokens per problem: thinking (B 256) vs no-think, it39."""
    panels = [
        ("seed 0, it 39", M / "t1320_grpoT1v1e_it39", M / "t1320_grpoT0_it39"),
        ("seed 1, it 39", M / "t1320_grpoT1v1es1_it39", M / "t1320_grpoT0s1_it39"),
    ]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.3, layout="constrained")
        axs = fig.subplots(1, 2, sharey=True)
        for ax, (title, think, nothink) in zip(axs, panels, strict=True):
            for lab, color, mk, p in [
                ("thinking, B 256", BLUE, "o", think / "rows_plain_B256.jsonl"),
                ("no-think GRPO", ORANGE, "s", nothink / "rows_plain_B0.jsonl"),
            ]:
                a = rows.arm(p)
                if a is None or a.tokens is None:
                    continue
                per = float(a.tokens.sum() / a.n.sum())
                grid = [
                    x
                    for x in (300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5500)
                    if per <= x <= per * 8
                ]
                es = stats.at_tokens(a, grid, draws=1000)
                errbar(ax, grid, [(e.est, e.lo, e.hi) for e in es], color, mk, lab)
            ax.set_xscale("log")
            ax.set_xticks([300, 1000, 3000], ["300", "1k", "3k"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("sampled tokens per problem")
            S.panel_title(ax, None, title)
            S.pct_axis(ax)
            S.light_grid(ax)
        axs[0].set_ylabel("pass@k at matched tokens, 13-20")
        S.save(fig, out / "fig_rl_t1320_tokens")
        S.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    a = ap.parse_args(argv)
    out = Path(a.out)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        keep = valid_1320()
        r1320, rall = Rows(keep, tmp), Rows(None, tmp)
        fig_curves(r1320, out)
        fig_matched(rall, r1320, out)
        fig_forcing(rall, out)
        fig_graded(rall, out)
        fig_tokens(r1320, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
