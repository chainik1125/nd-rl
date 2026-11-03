"""Figures for knockout_grpo, from numbers.json (Dan's headline layout, our R axis).

    code/.venv/bin/python -m code.experiments.current.knockout_grpo_20261103.fig [--figdir DIR]

(a) `rl_needed_grpo`: RL ÷ pretraining compute needed to reach the bar, against σ's share of the
    pretraining proofs; one marker per seed. Filled circle: first crossing (interpolated in log R);
    down-triangle: reached by the first checkpoint (an upper bound); hollow circle on the floor: at
    the bar before RL; up-triangle on a thin line: not reached by that R. Colour: GRPO with our
    pool; black-edged squares: GRPO with Dan's pool (Amendment 2); grey: Dan's EI ladders
    recomputed with our R rule (dan_axis.py). Short ticks: medians where every seed reached.
(b) `e32_vs_r`: E32 on the strict 106 against R per line, with the bar.
(c) `sigma_train`: σ's share of the distinct Lean-accepted training proofs (5-iteration pooled
    window) and E32 on the strict 106, against GRPO iteration, for the lines with a Lean dump.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import textwrap
from pathlib import Path

from matplotlib.lines import Line2D

from code.ndbench.figures import style as S

from .common import OUT

HERE = Path(__file__).resolve().parent
OI = S.OKABE_ITO
ARMS = ("full", "d1", "d01", "ko", "kos")
COLOR = {"full": OI["blue"], "d1": OI["bluish_green"], "d01": OI["orange"],
         "ko": OI["reddish_purple"], "kos": OI["vermillion"]}  # fmt: skip
NAME = {"full": "S1g3 (2.48 % σ)", "d1": "1 % σ dose", "d01": "0.1 % σ dose",
        "ko": "knockout, σ plans in narration", "kos": "ko_strict (0 %)"}  # fmt: skip
X = {"full": 2.48, "d1": 1.0, "d01": 0.1, "ko": 0.042, "kos": 0.02}  # where each arm sits
DAN_X = {"full": 10.9, "dose1": 1.0, "dose01": 0.1, "knockout": 0.02}
DAN_OFF = {"full": 1.0, "dose1": 1.32, "dose01": 1.7, "knockout": 1 / 1.32}  # beside ours
Y_LO, Y_HI = 0.0006, 30.0
FLOOR = 0.00092  # "at the bar before RL" row
LEFT_LABEL = {"d1"}  # arms whose range label sits left of the cluster
SURF = S.WHITE
CAPTION = [
    "One marker per seed. Bar: {bar:.1f} of 106 held-out theorems that need classical proof by",
    "contradiction (σ, Dan's strict set), as E32 (the expected number a 32-sample draw solves),",
    "the median of our three S1g3 bases before RL. y: (6 × trained + 2 × sampled tokens) ÷",
    "(6 × Stage-1 tokens), prompt prefill excluded. Filled: first crossing, interpolated in log y",
    "between checkpoints; ▼: at the first checkpoint (an upper bound); hollow: at the bar before",
    "RL; ▲ on a thin line: not reached by that y{running}; cross on a dotted line: the recipe's",
    "think-ban guard ended the line there (censored, uninformative). Colour: GRPO with",
    "programmatic thinking traces on our pool ({pool_lo:.1f}–{pool_hi:.1f} % of targets",
    "σ-covered); black edge: the same GRPO on Dan's {dan_n:,} EI targets ({dan_pct:.1f} %); grey:",
    "Dan's T1 expert-iteration ladders on this y (his own x adds re-read prefill and replay, so it",
    "is {r_lo:.1f}–{r_hi:.1f} × these values). Differences from Dan: GRPO, not EI; a thinking",
    "model (programmatic search traces); no pretraining replay; S1g3 bases (σ {full:.2f} %, not",
    "{k12:.1f} %); doses of {d01:.1f} % and {d1:.1f} % of the proofs; ko_strict also removes the σ",
    "plans that {ko_plan:.1f} % of ko's records write in their narration.",
]


def caption(N: dict, running: int) -> str:
    f = N["facts"]
    sh, pl = f["sigma_plus_share_pct"], f["pool_strict_pct"]
    return " ".join(CAPTION).format(
        bar=N["bar_e32"], pool_lo=min(pl["L2_pool"], pl["L3_pool"]),
        pool_hi=max(pl["L2_pool"], pl["L3_pool"]), dan_n=f["dan_pool_n"],
        dan_pct=pl["dan_rl_targets"], r_lo=f["dan_x_over_ours"][0], r_hi=f["dan_x_over_ours"][1],
        full=sh["full"], k12=sh["k12"], d01=sh["d01"], d1=sh["d1"], ko_plan=f["ko_plan_pct"],
        running=f" (open: still running, {running} lines)" if running else "",
    )  # fmt: skip


def _spread(n: int, k: int, w: float = 0.16) -> float:
    """Multiplicative x jitter for seed k of n."""
    return 1.0 + w * ((k - (n - 1) / 2) / max(1, n - 1)) * 2


def _ours(ax, x: float, c: dict, color: str, square: bool = False) -> float | None:
    ms = 3.6 if square else 4.4
    mk = "s" if square else "o"
    mec = S.INK if square else SURF
    if c["status"] == "at_bar_before_rl":
        ax.plot([x], [FLOOR], mk, ms=ms, mfc=SURF, mec=color, mew=0.9, zorder=4)
        return None
    if c["status"] == "reached":
        if c.get("R_upper_only") and not square:
            mk = "v"
        ax.plot([x], [c["R"]], mk, ms=ms, color=color, mec=mec, mew=0.5, zorder=5)
        return c["R"]
    if c["status"] == "not_reached" and c.get("censored"):  # ended by the think-ban guard
        ax.plot([x, x], [FLOOR, c["R_max"]], color=color, lw=0.6, alpha=0.5, ls=":", zorder=2)
        ax.plot([x], [c["R_max"]], "X", ms=ms + 0.6, color=color, mec=mec, mew=0.5, zorder=5)
    elif c["status"] == "not_reached":
        ax.plot([x, x], [FLOOR, c["R_max"]], color=color, lw=0.6, alpha=0.5, zorder=2)
        ax.plot([x], [c["R_max"]], "^", ms=ms + 0.5, color=color, mec=mec, mew=0.5, zorder=5)
    return None


def _rng(vals: list[float]) -> str:
    lo, hi = f"{min(vals):.2g}", f"{max(vals):.2g}"
    return f"{lo}×" if lo == hi else f"{lo}–{hi}×"


def _cross(run: dict) -> dict:
    """The run's crossing, marked censored when the think-ban guard ended it."""
    return {
        **run["cross"],
        "censored": run.get("think_ban_stop"),
        "running": not run.get("line_done"),
    }


def _miss_note(ax, x: float, groups: list[tuple[str, list[dict]]]) -> None:
    """One text block above an arm's up-triangles: per group, 'n of m not reached by R'."""
    lines, tops = [], []
    for name, cs in groups:
        nr = [c for c in cs if c["status"] == "not_reached"]
        miss = [c["R_max"] for c in nr if not c.get("censored") and not c.get("running")]
        run = [c["R_max"] for c in nr if not c.get("censored") and c.get("running")]
        cens = [c["R_max"] for c in nr if c.get("censored")]
        parts = [f"{len(miss)} of {len(cs)} not by {_rng(miss)}"] if miss else []
        parts += [f"{len(run)} of {len(cs)} running ({_rng(run)})"] if run else []
        parts += [f"{len(cens)} censored ({_rng(cens)})"] if cens else []
        if parts:
            lines += [f"{name}:"] + [f"   {p}" for p in parts]
            tops.append(max(miss + run + cens))
    if lines:
        ax.text(x, max(tops) * 1.25, "\n".join(lines), fontsize=S.FS_SMALL, ha="left",
                va="bottom", color=S.INK2)  # fmt: skip


def _dan(ax, N: dict) -> None:
    for arm, x0 in DAN_X.items():
        lads = [o for o in N.get("dan_axis", []) if o["arm"] == arm]
        lads.sort(key=lambda o: o["name"])
        hits = []
        for k, o in enumerate(lads):
            x = x0 * DAN_OFF[arm] * _spread(len(lads), k, 0.12)
            kw = {"ms": 3.0, "color": S.REF, "mec": SURF, "mew": 0.4, "zorder": 3}
            c = o["cross"]
            if c is not None and c["round"] == 1:
                ax.plot([x], [FLOOR], "o", ms=3.0, mfc=SURF, mec=S.REF, mew=0.7, zorder=3)
            elif c is not None:
                ax.plot([x], [c["x_ours"]], "o", **kw)
                hits.append(c["x_ours"])
            else:
                y = o["last"]["x_ours"]
                ax.plot([x, x], [FLOOR, y], color=S.REF, lw=0.5, alpha=0.35, zorder=1)
                ax.plot([x], [y], "^", **kw)
        if hits and len(hits) == len(lads):
            m = st.median(hits)
            xm = x0 * DAN_OFF[arm]
            ax.plot([xm / 1.16, xm * 1.16], [m, m], color=S.REF, lw=0.7, zorder=2)
        miss = [o["last"]["x_ours"] for o in lads if o["cross"] is None]
        if miss:
            ax.text(x0 * 1.2, max(miss) * 1.05, f"Dan's EI: {len(miss)} of {len(lads)} not\n"
                    f"reached by {min(miss):.1f}–{max(miss):.1f}×", fontsize=S.FS_SMALL,
                    ha="left", va="top", color=S.REF)  # fmt: skip


def _dump_labels(fig, ax, path: Path) -> None:
    """Every text the figure draws (for checking against numbers.md before reporting)."""
    out = {
        "texts": [t.get_text() for t in ax.texts],
        "xticks": [t.get_text() for t in ax.get_xticklabels()],
        "legend": [t.get_text() for t in ax.get_legend().get_texts()],
        "ylabel": ax.get_ylabel(),
        "caption": [t.get_text() for t in fig.texts],
    }
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))


def headline(N: dict, figdir: Path, provisional: bool) -> None:
    fig = S.new_figure(S.FULL_W, 4.55, layout="none")
    ax = fig.add_axes((0.115, 0.37, 0.86, 0.585))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0125, 17)
    ax.set_ylim(Y_LO, Y_HI)
    sh = N["facts"]["sigma_plus_share_pct"]
    X.update(full=sh["full"], d1=sh["d1"], d01=sh["d01"])  # each dose at its corpus's share
    DAN_X["full"] = sh["k12"]
    NAME["full"] = f"S1g3 ({sh['full']:.2f} % σ)"
    xt = [X["kos"], X["ko"], X["d01"], X["d1"], X["full"], DAN_X["full"]]
    lab = ["0 %\nko_strict", "0 %, σ plans\nin narration", f"{sh['d01']:.1f} %",
           f"{sh['d1']:.1f} %",
           f"{sh['full']:.2f} %\n(S1g3)", f"{sh['k12']:.1f} %\n(K12, Dan)"]  # fmt: skip  # noqa: E501
    ax.set_xticks(xt, lab)
    ax.minorticks_off()
    yt = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    ax.set_yticks(yt, [f"{v:g}×" for v in yt])
    S.light_grid(ax, "y")
    ax.text(-0.012, FLOOR, "at bar\nbefore RL", ha="right", va="center", fontsize=S.FS_SMALL,
            color=S.INK2, clip_on=False, transform=ax.get_yaxis_transform())  # fmt: skip
    ax.axhline(FLOOR * 1.9, color=S.INK2, lw=0.5, ls=(0, (2, 2)), zorder=0)
    _dan(ax, N)
    runs = N["runs"]
    for arm in ARMS:
        x0 = X[arm]
        tags = [f"{arm}_s{s}" for s in range(3) if f"{arm}_s{s}" in runs]
        vals = [_ours(ax, x0 * _spread(len(tags), k), _cross(runs[t]), COLOR[arm])
                for k, t in enumerate(tags)]  # fmt: skip
        hits = [v for v in vals if v is not None]
        reached = [runs[t]["cross"]["status"] in ("reached", "at_bar_before_rl") for t in tags]
        if hits and all(reached) and len(hits) > 1:
            m = st.median(hits)
            ax.plot([x0 / 1.2, x0 * 1.2], [m, m], color=S.INK, lw=0.7, zorder=4)
            left = arm in LEFT_LABEL
            ax.text(x0 / 1.27 if left else x0 * 1.27, m, f"{min(hits):.2g}–{max(hits):.2g}×",
                    fontsize=S.FS_SMALL, va="center", ha="right" if left else "left")  # fmt: skip
        dps = [f"{arm}_s{s}_dp" for s in range(3) if f"{arm}_s{s}_dp" in runs]
        for k, t in enumerate(dps):
            _ours(ax, x0 * 1.3 * _spread(len(dps), k, 0.1), _cross(runs[t]), COLOR[arm], True)
        _miss_note(ax, x0 / 1.1, [("our pool", [_cross(runs[t]) for t in tags]),
                                  ("Dan's pool", [_cross(runs[t]) for t in dps])])  # fmt: skip
    ax.set_xlabel("σ's share of the pretraining proofs (log scale)")
    ax.set_ylabel(f"RL ÷ pretraining compute (our rule) needed\nto reach the bar, "
                  f"{N['bar_e32']:.1f} of 106 strict σ theorems")  # fmt: skip
    h = [Line2D([], [], ls="none", marker="o", color=COLOR[a], mec=SURF, label=NAME[a], ms=4.4)
         for a in ARMS]  # fmt: skip
    h += [Line2D([], [], ls="none", marker="s", mfc=SURF, mec=S.INK, label="black edge: Dan's pool", ms=3.6),  # noqa: E501
          Line2D([], [], ls="none", marker="o", color=S.REF, label="Dan's EI on our rule", ms=3.0),
          Line2D([], [], ls="none", marker="^", color=S.INK2, label="not reached by", ms=4.4),
          Line2D([], [], ls="none", marker="X", color=S.INK2, label="think-ban stop (censored)",
                 ms=5.0)]  # fmt: skip
    ax.legend(handles=h, fontsize=S.FS_SMALL, frameon=False, loc="upper right", ncol=2,
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)  # fmt: skip
    n_run = sum(1 for x in runs.values() if x["cross"]["status"] == "not_reached"
                and not x.get("line_done"))  # fmt: skip
    cap = caption(N, n_run).replace(" %", "\u00a0%").replace(" ×", "\u00a0×")
    if provisional or n_run:
        cap = f"PROVISIONAL: {n_run} lines still running. " + cap
    cap = "\n".join(textwrap.wrap(cap, 132))
    fig.text(0.015, 0.012, cap, fontsize=S.FS_SMALL, va="bottom", ha="left", color=S.INK,
             linespacing=1.15)  # fmt: skip
    S.save(fig, figdir / "rl_needed_grpo")
    _dump_labels(fig, ax, figdir / "rl_needed_grpo.labels.json")
    S.close(fig)


def curves(N: dict, figdir: Path) -> None:
    fig = S.new_figure(S.FULL_W, 1.95, layout="constrained")
    axs = fig.subplots(1, len(ARMS), sharey=True)
    for ax, arm in zip(axs, ARMS, strict=True):
        ax.set_xscale("log")
        for s in range(3):
            for v, ls in (("", "-"), ("_dp", ":")):
                x = N["runs"].get(f"{arm}_s{s}{v}")
                if not x or "curve" not in x:
                    continue
                R = [r["R"] for r in x["curve"]]
                y = [r["e32"] for r in x["curve"]]
                ax.plot(R, y, ls, color=S.shade(COLOR[arm], -0.25 * s), lw=0.8)
        ax.axhline(N["bar_e32"], color=S.REF, lw=0.6, ls="--")
        S.panel_title(ax, None, NAME[arm].replace("knockout, σ plans in narration", "ko (σ plans)"),
                      size=S.FS_SMALL)  # fmt: skip
        ax.set_xlabel("R", fontsize=S.FS_SMALL)
        ax.set_xlim(0.002, 1.5)
        ax.tick_params(labelsize=S.FS_SMALL)
        S.light_grid(ax, "both")
    axs[0].set_ylabel("E32 on the strict 106", fontsize=S.FS_SMALL)
    axs[-1].legend(handles=[Line2D([], [], color=S.INK, lw=0.8, label="our pool"),
                            Line2D([], [], color=S.INK, lw=0.8, ls=":", label="Dan's pool")],
                   fontsize=S.FS_SMALL, frameon=False, loc="upper left")  # fmt: skip
    S.save(fig, figdir / "e32_vs_r")
    S.close(fig)


def _pooled(rows: list[dict], w: int = 5) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for i in range(len(rows)):
        win = rows[max(0, i - w + 1) : i + 1]
        n = sum(r["accepted_distinct"] for r in win)
        xs.append(rows[i]["c"])
        ys.append(100.0 * sum(r["sigma"] for r in win) / max(1, n))
    return xs, ys


def sigma_train(N: dict, figdir: Path) -> None:
    tags = [t for t in ("kos_s0", "kos_s1", "kos_s2", "kos_s0_dp", "kos_s1_dp", "kos_s2_dp",
                        "full_s0_dp") if (OUT / "train_sigma" / f"{t}.json").exists()]  # fmt: skip
    if not tags:
        return
    fig = S.new_figure(S.FULL_W, 2.1, layout="constrained")
    a, b = fig.subplots(1, 2)
    for t in tags:
        d = json.loads((OUT / "train_sigma" / f"{t}.json").read_text())
        rows = [{**r, "c": r["iter"]} for r in d["L2"]]
        rows += [{**r, "c": 39 + r["iter"]} for r in d["L3"]]
        arm, s = t.split("_")[0], int(t.split("_")[1][1])
        col = S.shade(COLOR[arm], -0.25 * s)
        ls = ":" if t.endswith("_dp") else "-"
        xs, ys = _pooled(rows)
        a.plot(xs, ys, ls, color=col, lw=0.9, label=t)
        cur = N["runs"].get(t, {}).get("curve", [])
        b.plot([r["c"] for r in cur], [r["e32"] for r in cur], ls, color=col, lw=0.9)
    a.set_ylabel("σ share of accepted\ntraining proofs (%)", fontsize=S.FS_SMALL)
    b.set_ylabel("E32 on the strict 106", fontsize=S.FS_SMALL)
    b.axhline(N["bar_e32"], color=S.REF, lw=0.6, ls="--")
    for ax, (letter, ttl) in zip((a, b), (("a", "training proofs (5-iteration window)"),
                                          ("b", "held-out σ theorems")), strict=True):  # fmt: skip
        ax.set_xlabel("GRPO iteration (L2 1–39, then L3)", fontsize=S.FS_SMALL)
        ax.axvline(39.5, color=S.GRID, lw=0.6)
        ax.tick_params(labelsize=S.FS_SMALL)
        S.light_grid(ax, "y")
        S.panel_title(ax, letter, ttl, size=S.FS_SMALL)
    a.legend(fontsize=S.FS_SMALL, frameon=False, loc="upper left", ncol=2, handlelength=1.6)
    S.save(fig, figdir / "sigma_train")
    S.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=str(HERE / "figs"))
    ap.add_argument("--provisional", action="store_true")
    a = ap.parse_args()
    N = json.loads((HERE / "numbers.json").read_text())
    with S.paper_style():
        headline(N, Path(a.figdir), a.provisional)
        curves(N, Path(a.figdir))
        sigma_train(N, Path(a.figdir))


if __name__ == "__main__":
    main()
