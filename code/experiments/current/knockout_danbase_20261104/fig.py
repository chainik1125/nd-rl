"""Figures for knockout_danbase, from numbers.json, on Dan's chart (his bar, his measure, his x).

    code/.venv/bin/python -m code.experiments.current.knockout_danbase_20261104.fig [--figdir DIR]

(a) `rl_needed_danbase`: Dan's layout. RL ÷ pretraining compute (his rule) to reach his bar (14
    of the 106 strict σ theorems, his cumulative k 32 count) against σ's share of the
    pretraining proofs; one marker per seed. Grey: his EI ladders (dan_axis.json). Colour: GRPO on
    his own bases: the knockout with his 4,495 targets (primary), the knockout with our σ-rich pool
    (secondary), the full base with his targets (reference). Filled circle: first crossing;
    up-triangle on a thin line: not reached by that x (hollow: stopped there by the 15 GPU-h cap,
    short of his round-16 x); hollow circle on the floor: at the bar at round 1 (the base).
(b) `cum_vs_x`: his measure (the cumulative count over the matched draws) against x on his rule,
    our GRPO lines beside his EI ladders on the same bases, with the bar.
(c) `sigma_danbase`: σ's share of the distinct Lean-accepted training proofs (Dan's coe_sigma, a
    10-iteration window) and the snapshot first-32 count, against x on his rule.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from matplotlib.lines import Line2D

from code.ndbench.figures import style as S

HERE = Path(__file__).resolve().parent
OI = S.OKABE_ITO
COL = {
    "primary": OI["vermillion"],
    "secondary": OI["reddish_purple"],
    "reference": OI["blue"],
    "dose": OI["vermillion"],
}  # the dose lines are the primary recipe on his dose bases
LAB = {"primary": "GRPO on his base, his 4,495 targets",
       "secondary": "GRPO, knockout base, our σ-rich pool",
       "reference": "GRPO, full K12 base, his targets"}  # fmt: skip
DAN_X = {"knockout": 0.02, "dose01": 0.1, "dose1": 1.0, "full": 10.9}
FLOOR = 0.012
SURF = S.WHITE


def _spread(n: int, k: int, w: float = 0.14) -> float:
    return 1.0 + w * ((k - (n - 1) / 2) / max(1, n - 1)) * 2


def _dan(ax, dan: list[dict]) -> None:
    for arm, x0 in DAN_X.items():
        lads = sorted((o for o in dan if o["arm"] == arm), key=lambda o: o["name"])
        hits = []
        for k, o in enumerate(lads):
            x = x0 * _spread(len(lads), k, 0.12) * (0.8 if arm in ("knockout", "full") else 1.0)
            c = o["cross"]
            kw = {"ms": 3.0, "color": S.REF, "mec": SURF, "mew": 0.4, "zorder": 3}
            if c is not None and c["round"] == 1:
                ax.plot([x], [FLOOR], "o", ms=3.0, mfc=SURF, mec=S.REF, mew=0.7, zorder=3)
            elif c is not None:
                ax.plot([x], [c["x_dan"]], "o", **kw)
                hits.append(c["x_dan"])
            else:
                y = o["rounds"][-1]["x_dan"]
                ax.plot([x, x], [FLOOR, y], color=S.REF, lw=0.5, alpha=0.35, zorder=1)
                ax.plot([x], [y], "^", **kw)
        if hits and len(hits) == len(lads):
            m = st.median(hits)
            xm = x0 * (0.8 if arm in ("knockout", "full") else 1.0)
            ax.plot([xm / 1.16, xm * 1.16], [m, m], color=S.REF, lw=0.7, zorder=2)


def _ours(ax, x: float, v: dict, color: str, mk: str = "o") -> None:
    c = v["cross_dan"]
    if c is not None and c["round"] == 1:
        ax.plot([x], [FLOOR], mk, ms=4.4, mfc=SURF, mec=color, mew=0.9, zorder=5)
    elif c is not None:
        ax.plot([x], [c["x_dan"]], mk, ms=4.4, color=color, mec=SURF, mew=0.5, zorder=5)
    else:
        y = max(v["x_dan"], FLOOR * 1.5)
        ls = "-" if v["status"] == "done" else ":"  # dotted: still running
        ax.plot([x, x], [FLOOR, y], color=color, lw=0.6, alpha=0.6, ls=ls, zorder=2)
        if (v.get("stop_reason") or "").startswith("GPU-h cap"):  # short of his ladder's end
            ax.plot([x], [y], "^", ms=4.9, mfc=SURF, mec=color, mew=0.9, zorder=5)
        else:
            ax.plot([x], [y], "^", ms=4.9, color=color, mec=SURF, mew=0.5, zorder=5)


def headline(N: dict, figdir: Path) -> None:
    fig = S.new_figure(S.FULL_W, 3.6, layout="constrained")
    ax = fig.add_subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.012, 22)
    ax.set_ylim(FLOOR * 0.7, 40)
    _dan(ax, N["dan"])
    lines = N["lines"]
    for role, x0, mk in (("primary", 0.02 * 1.25, "o"), ("secondary", 0.02 * 1.55, "D"),
                         ("reference", 10.9 * 1.25, "o")):  # fmt: skip
        tags = [t for t, v in lines.items() if v["arm"] == role]
        for k, t in enumerate(sorted(tags)):
            _ours(ax, x0 * _spread(len(tags), k, 0.08), lines[t], COL[role], mk)
    for dose in ("dose01", "dose1"):  # amendment 1: GRPO beside his EI at his two doses
        tags = sorted(t for t, v in lines.items() if v["arm"] == "dose" and v.get("dose") == dose)
        for k, t in enumerate(tags):
            _ours(ax, DAN_X[dose] * 1.3 * _spread(len(tags), k, 0.08), lines[t], COL["dose"])
    ax.axhline(FLOOR * 1.9, color=S.INK2, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.text(-0.012, FLOOR, "at bar\nat round 1", ha="right", va="center", fontsize=S.FS_SMALL,
            color=S.INK2, clip_on=False, transform=ax.get_yaxis_transform())  # fmt: skip
    ax.set_xticks([0.02, 0.1, 1.0, 10.9], ["0 %\n(knockout)", "0.1 %", "1 %", "10.9 %\n(K12)"])
    ax.minorticks_off()
    yt = [0.03, 0.1, 0.3, 1, 3, 10]
    ax.set_yticks(yt, [f"{v:g}×" for v in yt])
    S.light_grid(ax, "y")
    ax.set_xlabel("σ's share of the pretraining proofs (log scale)")
    ax.set_ylabel(f"RL ÷ pretraining compute (Dan's rule)\nto reach his bar ({N['bar']} of 106)")
    h = [Line2D([], [], ls="none", marker="o", color=S.REF, ms=3.0, label="Dan's EI ladders"),
         Line2D([], [], ls="none", marker="o", color=COL["primary"], ms=4.4, label=LAB["primary"]),
         Line2D([], [], ls="none", marker="D", color=COL["secondary"], ms=4.0,
                label=LAB["secondary"]),
         Line2D([], [], ls="none", marker="o", color=COL["reference"], ms=4.4,
                label=LAB["reference"]),
         Line2D([], [], ls="none", marker="^", color=S.INK2, ms=4.4,
                label="not reached by that x, at his round-16 x"),
         Line2D([], [], ls="none", marker="^", mfc=SURF, mec=S.INK2, mew=0.9, ms=4.4,
                label="not reached: stopped by the 15 GPU-h cap, short of his ladder")]  # fmt: skip
    ax.legend(handles=h, fontsize=S.FS_SMALL, frameon=False, loc="upper right")
    S.save(fig, figdir / "rl_needed_danbase")
    S.close(fig)


def cum_vs_x(N: dict, figdir: Path) -> None:
    fig = S.new_figure(S.FULL_W, 4.2, layout="constrained")
    axs = fig.subplots(2, 2, sharey=True).ravel()
    panels = (("knockout", ("primary", "secondary"), "knockout base (K12, σ removed)"),
              ("dose01", ("dose",), "0.1 % σ dose base"), ("dose1", ("dose",), "1 % σ dose base"),
              ("full", ("reference",), "full K12 base"))  # fmt: skip
    for ax, (arm, roles, title) in zip(axs, panels, strict=True):
        for o in N["dan"]:
            if o["arm"] == arm:
                ax.plot([r["x_dan"] for r in o["rounds"]], [r["y"] for r in o["rounds"]], "-",
                        color=S.REF, lw=0.6, alpha=0.7)  # fmt: skip
        for _t, v in sorted(N["lines"].items()):
            if v["arm"] == "dose" and v.get("dose") != arm:
                continue
            if v["arm"] in roles and v["matched"]:
                xs = [max(m["x_dan"], 0.004) for m in v["matched"]]
                ax.plot(xs, [m["cum"] for m in v["matched"]], marker="o", ms=2.0, lw=0.9,
                        color=COL[v["arm"]],
                        ls="-" if v["arm"] != "secondary" else "--")  # fmt: skip
        ax.axhline(N["bar"], color=S.INK2, lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("x (Dan's rule)", fontsize=S.FS_SMALL)
        S.panel_title(ax, None, title, size=S.FS_SMALL)
        S.light_grid(ax, "both")
        ax.tick_params(labelsize=S.FS_SMALL)
    for ax in axs[::2]:
        ax.set_ylabel("cumulative solved of the 106\n(Dan's measure)", fontsize=S.FS_SMALL)
    axs[0].legend(handles=[Line2D([], [], color=S.REF, lw=0.6, label="Dan's EI"),
                           Line2D([], [], color=COL["primary"], lw=0.9, label="GRPO, his targets"),
                           Line2D([], [], color=COL["secondary"], lw=0.9, ls="--",
                                  label="GRPO, our pool")],
                  fontsize=S.FS_SMALL, frameon=False, loc="upper left")  # fmt: skip
    S.save(fig, figdir / "cum_vs_x")
    S.close(fig)


def sigma_fig(N: dict, figdir: Path) -> None:
    fig = S.new_figure(S.FULL_W, 2.3, layout="constrained")
    a, b = fig.subplots(1, 2)
    for _t, v in sorted(N["lines"].items()):
        col = COL[v["arm"]]
        ls = "--" if v["arm"] == "secondary" else "-"
        xs = {c["it"]: c["x_dan"] for c in v["curve"]}
        sg = v.get("sigma") or []
        pts = []
        for i in range(0, len(sg), 10):
            w = sg[i : i + 10]
            n = sum(x["accepted"] for x in w)
            near = [xs[k] for k in xs if k <= w[-1]["it"]]
            if n and near:
                pts.append((max(near), 100 * sum(x["sigma"] for x in w) / n))
        if pts:
            a.plot(*zip(*pts, strict=True), ls, color=col, lw=0.8)
        cv = [c for c in v["curve"] if c["it"] > 0]
        if cv:
            b.plot([c["x_dan"] for c in cv], [c["first32"] for c in cv], ls, color=col, lw=0.8)
    a.set_ylabel("σ share of accepted\ntraining proofs (%)", fontsize=S.FS_SMALL)
    b.set_ylabel("snapshot first-32 count", fontsize=S.FS_SMALL)
    b.axhline(N["bar"], color=S.INK2, lw=0.6, ls="--")
    for ax, ttl in ((a, "(a) training proofs"), (b, "(b) held-out σ theorems")):
        ax.set_xscale("log")
        ax.set_xlabel("x (Dan's rule)", fontsize=S.FS_SMALL)
        S.panel_title(ax, None, ttl, size=S.FS_SMALL)
        S.light_grid(ax, "both")
        ax.tick_params(labelsize=S.FS_SMALL)
    S.save(fig, figdir / "sigma_danbase")
    S.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=str(HERE / "figs"))
    a = ap.parse_args()
    figdir = Path(a.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    N = json.loads((HERE / "numbers.json").read_text())
    with S.paper_style():
        headline(N, figdir)
        cum_vs_x(N, figdir)
        sigma_fig(N, figdir)


if __name__ == "__main__":
    main()
