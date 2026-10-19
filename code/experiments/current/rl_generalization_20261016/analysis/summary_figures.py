"""The experiment summary's two figures, read directly from the eval rows with mini_results.py's
loaders (nothing in results/ is read or written).

    cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=2 PYTHONPATH=. nice code/.venv/bin/python \
        code/experiments/current/rl_generalization_20261016/analysis/summary_figures.py [--out DIR]

Writes `fig_g3_line` and `fig_controls` (.pdf, .svg, .png) into the summary folder
(`experiment-summaries/2026-10-16-rl-generalization-testbed/figures/` by default) and prints every
plotted number, so the summary can quote them.

1. G3 (PREREGISTRATION.md § 3, informational): L3 test_1k pass@1 against 13–20 pass@1, both on a
   logit axis. Full-scale grpo_think_vs_mb_kkl0 seed 0 at it 0/10/20/30/39 (it0 is the shared
   grpo_think_vs L2 checkpoint) with its OLS line (the line the 09:27Z deviation, ba275d73, cited);
   the other full-scale points G3 has when run (other seeds, unlabelled) and the refit on all of
   them, as in results/gate_mini20.md; the four 10 + 10 and four 20 + 20 references; the base
   S1g3_s0.
   Reads and k exactly as mini_results.g3_view (l3test k8; t1320 k4, k16 at it39; the mini runs'
   MINI_READS k8).
2. The controls: per seed, the paired fork difference c_s − r_s in logit of 13–20 pass@1 (the same
   L2-mini; 95% problem bootstrap, attempt and problem noise only), for kl1 and g64 at 10 + 10 and
   20 + 20, next to the gate's unpaired Δ with its one-sided 95% upper bound U (§ 3), against the
   expected shifts E (kl1 −0.19, g64 −0.255) and ½E. Panel (b) also shows the full-scale g64
   contrast per seed (kg64 − merged recipe, both L3 at KL 0.02; t1320 k16, as
   results/g64_replicate.md): seed 0, the pilot behind E, and the replicate seed (2, in place of
   failed seed 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from code.experiments.current.rl_generalization_20261016.analysis import mini_results as mr
from code.ndbench.figures import style as st

SUMMARY = mr.REPO / "experiment-summaries" / "2026-10-16-rl-generalization-testbed" / "figures"
EV = mr.R / "eval/v2"
REGIMES = ("mini", "mini20")
TAG = {"mini": "10 + 10", "mini20": "20 + 20"}
S0_ITS = ("it0", "it10", "it20", "it30", "it39")
# label offsets (points) of the seed-0 trajectory's ends; its L3 test rises with every checkpoint,
# so the points in between need no labels (with seeds 1 and 2 drawn, theirs would collide)
LABEL_AT = {"it0": (-4, 5, "right"), "it39": (-4, 5, "right")}


def expit(x: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-x))


def load() -> tuple[mr.Loader, mr.Est]:
    keep = mr.register_sets()
    for key in REGIMES:  # registers each regime's arms (base, references, controls) with v2_results
        rg = mr.REGIMES[key]
        found, roll = mr.discover(EV, rg)
        mr.register(rg, found, roll)
    ld = mr.Loader(EV, keep)
    return ld, mr.Est(ld, mr.stats.DRAWS)


# ------------------------------------------------------------------------------------------ G3


def g3_points(ld: mr.Loader) -> list[tuple[int, str, float, float]]:
    """(seed, checkpoint, L3 test pass@1, 13–20 pass@1) of the full-scale line, as g3_view."""
    pts = []
    for s in mr.G3_SEEDS:
        cks = [("it0", mr.G3_IT0, 4), *((f"it{i}", f"{mr.G3_ARM}_it{i}", 4) for i in mr.G3_ITS)]
        for name, arm, kt in [*cks, ("it39", mr.G3_ARM, 16)]:
            x, y = mr.p_of(ld, arm, "l3test", s, 1, 8), mr.p_of(ld, arm, "t1320", s, 1, kt)
            if x is not None and y is not None:
                pts.append((s, name, x, y))
    return pts


def fig_g3(ld: mr.Loader, out: Path) -> None:
    pts = g3_points(ld)
    s0 = [p for p in pts if p[0] == 0]
    assert [p[1] for p in s0] == list(S0_ITS), f"seed-0 trajectory incomplete: {s0}"
    other = [p for p in pts if p[0] != 0]
    fit5 = mr.ols([(mr.logit(x), mr.logit(y)) for *_, x, y in s0])
    fitall = mr.ols([(mr.logit(x), mr.logit(y)) for *_, x, y in pts])
    assert fit5 is not None and fitall is not None
    res5 = [mr.logit(y) - fit5[0] - fit5[1] * mr.logit(x) for *_, x, y in s0]
    resall = [mr.logit(y) - fitall[0] - fitall[1] * mr.logit(x) for *_, x, y in pts]
    print("G3 full-scale points (seed, ckpt, L3 test %, 13–20 %):")
    for s, n, x, y in pts:
        print(f"  s{s} {n}: {100 * x:.1f} {100 * y:.1f}")
    print(
        f"  s0 line ({len(s0)} pts): a {fit5[0]:+.3f} b {fit5[1]:+.3f}, residual SD "
        f"{mr.sd(res5):.3f}"
    )
    print(
        f"  all-points line ({len(pts)} pts): a {fitall[0]:+.3f} b {fitall[1]:+.3f}, residual SD "
        f"{mr.sd(resall):.3f}"
    )
    mini = {}
    for key in REGIMES:
        rg = mr.REGIMES[key]
        mini[key] = []
        for s in mr.REF_SEEDS:
            x, y = mr.p_of(ld, rg.ref, "l3test", s, 1), mr.p_of(ld, rg.ref, "t1320", s, 1)
            if x is not None and y is not None:
                mini[key].append((s, x, y))
        for fit, nm in ((fit5, "s0 line"), (fitall, "all-points line")):
            r = [mr.logit(y) - fit[0] - fit[1] * mr.logit(x) for _, x, y in mini[key]]
            print(
                f"  {TAG[key]} references vs the {nm}: "
                + ", ".join(f"{v:+.3f}" for v in r)
                + f"; mean {mr.mean(r):+.3f}"
            )
    bx, by = mr.p_of(ld, mr.BASE, "l3test", 0, 1), mr.p_of(ld, mr.BASE, "t1320", 0, 1)
    assert bx is not None and by is not None
    print(
        f"  base S1g3_s0: L3 test {100 * bx:.1f}%, 13–20 {100 * by:.1f}%; residual vs s0 line "
        f"{mr.logit(by) - fit5[0] - fit5[1] * mr.logit(bx):+.3f}"
    )

    full = st.OKABE_ITO["black"]
    c10, c20, cb = st.OKABE_ITO["sky_blue"], st.OKABE_ITO["vermillion"], st.OKABE_ITO["blue"]
    with st.paper_style():
        fig = st.new_figure(st.FULL_W, 2.75)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        lo_fit, hi_fit = (mr.logit(f(p[2] for p in s0)) for f in (min, max))
        xs_in = np.linspace(lo_fit, hi_fit, 50)
        xs_lo = np.linspace(mr.logit(0.18), lo_fit, 80)
        xs_hi = np.linspace(hi_fit, mr.logit(0.975), 20)
        for xs, ls in ((xs_in, "-"), (xs_lo, (0, (3, 2))), (xs_hi, (0, (3, 2)))):
            ax.plot(expit(xs), expit(fit5[0] + fit5[1] * xs), color=full, lw=0.9, ls=ls, zorder=1)
        xs_all = np.concatenate([xs_lo, xs_in, xs_hi])
        ax.plot(
            expit(xs_all),
            expit(fitall[0] + fitall[1] * xs_all),
            color=st.REF,
            lw=0.7,
            ls=(0, (1, 1.4)),
            zorder=1,
        )
        ax.plot(
            [p[2] for p in s0],
            [p[3] for p in s0],
            ls="none",
            marker="s",
            ms=4.0,
            mfc=full,
            mec=st.WHITE,
            mew=0.5,
            zorder=3,
        )
        for _, n, x, y in s0:
            if n not in LABEL_AT:
                continue
            dx, dy, ha = LABEL_AT[n]
            ax.annotate(
                f"s0 {n}",
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=st.FS_SMALL,
                ha=ha,
                va="center",
                color=st.INK2,
            )
        if other:
            ax.plot(
                [p[2] for p in other],
                [p[3] for p in other],
                ls="none",
                marker="s",
                ms=4.0,
                mfc=st.WHITE,
                mec=full,
                mew=0.8,
                zorder=3,
            )
        for key, col, mk in (("mini", c10, "o"), ("mini20", c20, "^")):
            ax.plot(
                [p[1] for p in mini[key]],
                [p[2] for p in mini[key]],
                ls="none",
                marker=mk,
                ms=4.2,
                mfc=col,
                mec=st.WHITE,
                mew=0.5,
                zorder=4,
            )
        ax.plot([bx], [by], ls="none", marker="D", ms=4.0, mfc=cb, mec=st.WHITE, mew=0.5, zorder=4)
        ax.annotate(
            "base S1g3_s0",
            (bx, by),
            xytext=(5, -2),
            textcoords="offset points",
            fontsize=st.FS_TICK,
            va="top",
        )
        xt = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        yt = [0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
        ax.set_xticks(xt, [f"{100 * v:g}%" for v in xt])
        ax.set_yticks(yt, [f"{100 * v:g}%" for v in yt])
        ax.minorticks_off()
        ax.set_xlim(0.18, 0.975)
        ax.set_ylim(0.07, 0.45)
        st.light_grid(ax, "both")
        ax.set_xlabel("L3 test_1k pass@1 (logit axis)")
        ax.set_ylabel("13–20 pass@1 (logit axis)")
        handles = [
            Line2D([], [], color=full, lw=0.9, marker="s", ms=4.0, mfc=full, mec=st.WHITE),
            Line2D([], [], color=st.REF, lw=0.7, ls=(0, (1, 1.4))),
            Line2D([], [], ls="none", marker="s", ms=4.0, mfc=st.WHITE, mec=full, mew=0.8),
            Line2D([], [], ls="none", marker="o", ms=4.2, mfc=c10, mec=st.WHITE),
            Line2D([], [], ls="none", marker="^", ms=4.2, mfc=c20, mec=st.WHITE),
        ]
        labels = [
            "full scale, KL 0, seed 0 (it 0–39) and its line",
            f"refit with every full-scale point ({len(pts)})",
            "full scale, KL 0, seed"
            + ("s " if len({p[0] for p in other}) > 1 else " ")
            + " and ".join(str(x) for x in sorted({p[0] for p in other}))
            + " (it 0–39)",
            "mini 10 + 10 references (s0–s3)",
            "mini 20 + 20 references (s0–s3)",
        ]
        ax.legend(handles, labels, loc="upper left", fontsize=st.FS_TICK)
        st.save(fig, out / "fig_g3_line")
        st.close(fig)


# ------------------------------------------------------------------------------------ controls


Ctl = dict[tuple[str, str], tuple[float, mr.CtrlResult, list[tuple[int, float, float, float]]]]


def control_data(ld: mr.Loader, est: mr.Est) -> Ctl:
    out: Ctl = {}
    for key in REGIMES:
        rg = mr.REGIMES[key]
        xr = mr.x_of(ld, rg.ref, list(mr.REF_SEEDS), *mr.PRIMARY)
        assert xr is not None
        s_ref = mr.sd(xr)
        for c in rg.controls:
            res = mr.classify(ld, rg, c, list(mr.REF_SEEDS), s_ref)
            per: list[tuple[int, float, float, float]] = []
            for s in res.seeds:
                d = est.logit_diff(c.arm, rg.ref, s, *mr.PRIMARY)
                assert d is not None
                per.append((s, *d))
            out[(key, c.name)] = (c.E, res, per)
            print(
                f"{TAG[key]} {c.name}: class {res.cls}, n_c {len(res.seeds)}, Δ {res.delta:+.3f}, "
                f"U {res.U:+.3f}, s_ref {s_ref:.4f}; paired: "
                + "; ".join(f"s{s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for s, d, lo, hi in per)
            )
    return out


def replicate_data(est: mr.Est) -> list[tuple[int, float, float, float]]:
    """Full-scale kg64 − merged, 13–20 pass@1 in logit, at the pilot seed and every replicate seed
    with reads (mini_results.replicate_md's statistic)."""
    out = []
    for s in (mr.REP_PILOT, *mr.REP_TRY):
        d = est.logit_diff(mr.REP_ARM, mr.REP_PAIR, s, *mr.PRIMARY, mr.REP_K["t1320"])
        if d is not None:
            out.append((s, *d))
    print(
        "full-scale g64 (kg64 − mb): "
        + "; ".join(f"s{s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for s, d, lo, hi in out)
    )
    return out


def fig_controls(data: Ctl, rep: list[tuple[int, float, float, float]], out: Path) -> None:
    seed_col = st.OKABE_ITO["blue"]
    with st.paper_style():
        fig = st.new_figure(st.FULL_W, 2.55)
        axes = fig.subplots(1, 2, sharey=True, width_ratios=[2.45, 3.3])
        for ax, (letter, name, title) in zip(
            axes,
            (("a", "kl1", "KL anchor 0.1 (L3 fork)"), ("b", "g64", "64 × 64 groups (L3 fork)")),
            strict=True,
        ):
            E = data[("mini", name)][0]
            full = name == "g64" and bool(rep)
            x0, x1 = -0.5, 2.3 if full else 1.45
            ax.hlines(0, x0, x1, color=st.REF, lw=0.6, zorder=1)
            ax.hlines(E, x0, x1, color=st.INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
            ax.hlines(E / 2, x0, x1, color=st.INK, lw=0.8, ls=(0, (1, 1.3)), zorder=1)
            for yv, txt in ((E, f"E {E:+.3f}"), (E / 2, f"½E {E / 2:+.4f}")):
                ax.text(x1 + 0.03, yv, txt.replace("-", "−"), va="center", fontsize=st.FS_SMALL)
            for i, key in enumerate(REGIMES):
                _, res, per = data[(key, name)]
                n = len(per)
                offs = np.linspace(-0.27, 0.07, n) if n > 1 else np.array([-0.1])
                for (s, d, lo, hi), dx in zip(per, offs, strict=True):
                    ax.errorbar(
                        i + dx,
                        d,
                        yerr=[[d - lo], [hi - d]],
                        fmt="o",
                        ms=3.4,
                        color=seed_col,
                        mfc=seed_col,
                        mec=st.WHITE,
                        mew=0.4,
                        elinewidth=0.7,
                        capsize=1.4,
                        zorder=3,
                    )
                    ax.annotate(
                        f"s{s}",
                        (i + dx, hi),
                        xytext=(0, 1.5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=st.INK2,
                    )
                gx = i + 0.25
                ax.plot([gx, gx], [res.delta, res.U], color=st.INK, lw=0.9, zorder=3)
                ax.plot([gx - 0.04, gx + 0.04], [res.U, res.U], color=st.INK, lw=0.9, zorder=3)
                ax.plot(
                    [gx],
                    [res.delta],
                    marker="D",
                    ms=4.2,
                    color=st.INK,
                    mec=st.WHITE,
                    mew=0.5,
                    ls="none",
                    zorder=4,
                )
            ticks = [
                f"{TAG[k]}\nn_c {len(data[(k, name)][2])}: {data[(k, name)][1].cls}"
                for k in REGIMES
            ]
            if full:
                offs = np.linspace(-0.12, 0.12, len(rep)) if len(rep) > 1 else np.array([0.0])
                for (s, d, lo, hi), dx in zip(rep, offs, strict=True):
                    ax.errorbar(
                        2 + dx,
                        d,
                        yerr=[[d - lo], [hi - d]],
                        fmt="s",
                        ms=3.6,
                        color=st.INK2,
                        mfc=st.WHITE,
                        mec=st.INK,
                        mew=0.8,
                        elinewidth=0.7,
                        capsize=1.4,
                        zorder=3,
                    )
                    ax.annotate(
                        f"s{s}",
                        (2 + dx, hi),
                        xytext=(0, 1.5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=st.INK2,
                    )
                ticks.append("full scale,\nseeds " + " / ".join(str(r[0]) for r in rep))
            ax.set_xticks(list(range(len(ticks))), ticks)
            ax.set_xlim(-0.5, x1 + 0.5)
            st.panel_title(ax, letter, title)
            st.light_grid(ax, "y")
        axes[0].set_ylabel("Δ logit, 13–20 pass@1\n(control − reference)")
        handles = [
            Line2D([], [], ls="none", marker="o", ms=3.4, mfc=seed_col, mec=st.WHITE),
            Line2D([], [], ls="none", marker="D", ms=4.2, mfc=st.INK, mec=st.WHITE),
            Line2D([], [], ls="none", marker="s", ms=3.6, mfc=st.WHITE, mec=st.INK, mew=0.8),
            Line2D([], [], color=st.INK, lw=0.8, ls=(0, (4, 2))),
            Line2D([], [], color=st.INK, lw=0.8, ls=(0, (1, 1.3))),
        ]
        labels = [
            "paired fork difference, one seed (95% problem bootstrap)",
            "gate Δ, bar to its one-sided 95% bound U",
            "full scale: kg64 − merged recipe, one seed (95% problem bootstrap)",
            "E, the full-scale seed-0 shift",
            "½E, the gate's threshold",
        ]
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=st.FS_TICK)
        st.save(fig, out / "fig_controls")
        st.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--out", default=str(SUMMARY))
    a = ap.parse_args(argv)
    out = Path(a.out)
    ld, est = load()
    fig_g3(ld, out)
    fig_controls(control_data(ld, est), replicate_data(est), out)
    print(f"wrote {out}/fig_g3_line.* and fig_controls.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
