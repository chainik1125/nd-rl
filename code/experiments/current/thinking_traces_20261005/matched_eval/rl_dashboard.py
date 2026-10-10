"""RL monitoring dashboard: one GRPO line's metrics.jsonl rows as a NeurIPS-style figure.

    ND_OFFLINE=1 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.rl_dashboard \
        --runs RUNDIR_stage1,RUNDIR_stage2 --label grpo_think_v_s0 [--out DIR]

`--runs` are the consecutive stages of one line (each a `code.thinking.grpo` output directory,
or its metrics.jsonl); their iterations are plotted end to end, with a light line at each stage
boundary. Panels: (a) the training batch's pass rate per target proof length; (b) the share of
groups that give a gradient, overall and per length; (c) where the gradient comes from: each
length's share of the |advantage|-weighted token mass (stacked); (d) the shares of that mass on
think-block tokens and on rejected (redrawn) draws; (e) rejected draws per draw, with the most
common causes of the last iteration; (f) the long-context canary against its base (action NLL,
P(':'), the top wrong token), with the alarm multiple. (a), (c), (d) and (e) need `--monitor`
rows (`grpo.monitor_metrics`); without them (a) and (b) fall back to the overall batch numbers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from code.ndbench.figures import style as S

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
Row = dict[str, Any]


def load_line(runs: Sequence[str]) -> tuple[list[Row], list[int]]:
    """Rows of consecutive stages, re-indexed end to end -> (rows with `x`, stage boundaries)."""
    rows: list[Row] = []
    bounds: list[int] = []
    off = 0
    for r in runs:
        p = Path(r)
        p = p / "metrics.jsonl" if p.is_dir() else p
        stage = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        for x in stage:
            x["x"] = off + int(x["iter"]) - int(stage[0]["iter"]) + 1
        rows += stage
        if stage:
            off = rows[-1]["x"]
            bounds.append(off)
    return rows, bounds[:-1]


def length_colours(lengths: Sequence[str]) -> dict[str, str]:
    """One sequential hue, light (short proofs) to dark (long)."""
    ks = sorted(lengths, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0))
    n = len(ks)
    return {k: S.shade(BLUE, 0.55 - 0.9 * i / max(1, n - 1)) for i, k in enumerate(ks)}


def _end_label(
    ax: Any, x: Sequence[float], y: Sequence[float], text: str, c: str, dy: float = 0
) -> None:
    ok = [(a, b) for a, b in zip(x, y, strict=True) if np.isfinite(b)]
    if ok:
        ax.annotate(
            text,
            ok[-1],
            xytext=(3, dy),
            textcoords="offset points",
            va="center",
            fontsize=S.FS_TICK,
            color=S.INK2,
        )
        ax.plot([ok[-1][0]], [ok[-1][1]], marker="o", ms=2.2, color=c, ls="none")


def draw(rows: list[Row], bounds: list[int], label: str, base: Path) -> None:
    mon = [r for r in rows if "monitor" in r]
    lens = sorted({k for r in mon for k in r["monitor"]["by_len"]}, key=lambda s: (len(s), s))
    col = length_colours(lens)
    xs = [r["x"] for r in rows]
    mx = [r["x"] for r in mon]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 4.1, layout="constrained")
        axs = fig.subplots(2, 3, sharex=True)
        (a, b, c), (d, e, f) = axs
        # (a) batch pass rate per length
        if mon:
            for k in lens:
                y = [r["monitor"]["by_len"].get(k, {}).get("pass") for r in mon]
                ya = [np.nan if v is None else v for v in y]
                a.plot(mx, ya, color=col[k], lw=1.1)
                _end_label(a, mx, ya, f"{k} lines", col[k])
        else:
            a.plot(xs, [r["reward_pass1"] for r in rows], color=BLUE, lw=1.1)
        S.panel_title(a, "a", "batch pass rate by proof length")
        S.pct_axis(a)
        # (b) groups with a gradient
        b.plot(xs, [1 - r["groups_skipped"] for r in rows], color=S.INK, lw=1.3)
        _end_label(b, xs, [1 - r["groups_skipped"] for r in rows], "all", S.INK)
        for k in lens:
            y = [r["monitor"]["by_len"].get(k, {}).get("with_grad", np.nan) for r in mon]
            b.plot(mx, y, color=col[k], lw=0.8, alpha=0.9)
        S.panel_title(b, "b", "groups that give a gradient")
        S.pct_axis(b)
        # (c) gradient share per length, stacked
        if mon:
            ys = [
                [r["monitor"]["by_len"].get(k, {}).get("grad_share", 0.0) for r in mon]
                for k in lens
            ]
            c.stackplot(mx, ys, colors=[col[k] for k in lens], lw=0)
            cum = np.cumsum(np.array(ys), axis=0)
            for i, k in enumerate(lens):
                mid = cum[i][-1] - ys[i][-1] / 2
                if ys[i][-1] > 0.06:
                    c.annotate(
                        k,
                        (mx[-1], mid),
                        xytext=(3, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=S.FS_TICK,
                        color=S.INK2,
                    )
            c.set_ylim(0, 1)
        S.panel_title(c, "c", "gradient share by proof length")
        S.pct_axis(c)
        # (d) think and rejected-draw shares of the gradient
        if mon:
            th = [r["monitor"]["mass_think"] for r in mon]
            rj = [r["monitor"]["mass_rejected"] for r in mon]
            d.plot(mx, th, color=AQUA, lw=1.1)
            d.plot(mx, rj, color=ORANGE, lw=1.1, ls=(0, (3.5, 1.6)))
            _end_label(d, mx, th, "think tokens", AQUA)
            _end_label(d, mx, rj, "rejected draws", ORANGE)
            d.set_ylim(0, 1)
        S.panel_title(d, "d", "share of the gradient on ...")
        S.pct_axis(d)
        # (e) rejections per draw
        if mon:
            rpd = [r["monitor"]["rej_per_draw"] for r in mon]
            e.plot(mx, rpd, color=ORANGE, lw=1.1)
            causes = mon[-1]["monitor"].get("rej_causes") or {}
            txt = "\n".join(
                f"{k.split(':', 1)[-1][:22]} {100 * v:.0f}%" for k, v in list(causes.items())[:3]
            )
            if txt:
                e.text(
                    0.98,
                    0.95,
                    txt,
                    transform=e.transAxes,
                    ha="right",
                    va="top",
                    fontsize=S.FS_SMALL,
                    color=S.INK2,
                )
            e.set_ylim(bottom=0)
        S.panel_title(e, "e", "rejected draws per draw")
        S.pct_axis(e)
        # (f) canary drift
        can = [r for r in rows if r.get("canary") and r["canary"].get("base")]
        if can:
            cx = [r["x"] for r in can]
            for key, name, cc, ls, dy in (
                ("nll", "action NLL", S.INK, "-", 5),
                ("colon", "P(':')", ORANGE, (0, (3.5, 1.6)), 0),
                ("wrong_max", "top wrong token", AQUA, (0, (1, 1.3)), -5),
            ):
                y = [
                    r["canary"][key] / r["canary"]["base"][key]
                    if r["canary"]["base"][key] > 0
                    else np.nan
                    for r in can
                ]
                f.plot(cx, y, color=cc, lw=1.1, ls=ls)
                _end_label(f, cx, y, name, cc, dy)
            mult = (can[-1].get("alarm_cfg") or {}).get("canary_mult") or 0
            if mult:
                f.axhline(mult, color=S.REF, lw=0.6, ls=(0, (2, 2)))
            f.axhline(1.0, color=S.REF, lw=0.5)
            f.set_yscale("log")
        S.panel_title(f, "f", "long-context canary / its base")
        if not mon:
            for ax in (c, d, e):
                ax.text(
                    0.5,
                    0.5,
                    "needs --monitor rows",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=S.FS_TICK,
                    color=S.REF,
                )
        for ax in (a, b, c, d, e, f):
            S.light_grid(ax)
            for x in bounds:
                ax.axvline(x + 0.5, color=S.GRID, lw=0.8, zorder=0)
        fig.supxlabel(f"GRPO iteration ({label})", fontsize=S.FS_LABEL)
        S.save(fig, base)
        S.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated stage directories, in order")
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--out", default="experiment-summaries/2026-10-05-thinking-traces/figures/rl_dash"
    )
    a = ap.parse_args(argv)
    rows, bounds = load_line(a.runs.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    draw(rows, bounds, a.label, out / f"dash_{a.label}")
    print(f"wrote {out / ('dash_' + a.label)}.{{pdf,svg,png}} ({len(rows)} iterations)")


if __name__ == "__main__":
    main()
