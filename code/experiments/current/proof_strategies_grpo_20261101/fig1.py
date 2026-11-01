"""Figure 1 of the high-RL regime paper, "capability emergence": one row, 5.5 in wide.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \\
        code.experiments.current.proof_strategies_grpo_20261101.fig1 [--draws 4000]

(a) Charles149 pass@k, base S1g3 vs the recipe's L3 end, means over seeds 0-2, 95% problem x seed
    bootstrap bands (`code.thinking.stats.ci_curve`).
(b) Two Lean-accepted RL proofs (L3 end; the shortest clean proof over seeds 0-2) of Charles149
    problems the base solves in 0 and 4 of 3,072 attempts (seeds 0-2), in Fitch notation, the
    strategy marked; the header gives RL's success rate over its 3 x 1,024 attempts. Their Lean
    texts are re-checked by `lean_check` here (the run stops if Lean rejects one).
(c) Emergence against R = RL / Stage-1 FLOPs (seed-0 checkpoints): the per-attempt solve rate on
    the Charles149 problems the base never solves at k 1,024, and the per-attempt rate of accepted
    13-20-line proofs that use a working proof by contradiction; logistic fits in log10 R, the
    base at its own x position left of an axis break, Yue et al.'s median R as a dotted line.
(d) A placeholder for RL trajectory generalization (pending).
Writes docs/papers/high-rl-regime/figures/fig1_emergence.{pdf,png} and fig1_numbers.json here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from code.ndbench.figures import style as S
from code.thinking import stats

from ..high_rl_passk_20261030.passk_figs import BASE_C, FIGDIR, KEEP, RL_HUE, YUE_MEDIAN, pow2
from . import common as C
from .classify import read_name

K = 1024
RL2_C = S.OKABE_ITO["blue"]
XB = 1e-5  # panel (c): the x position that stands for the base (left of the axis break)
#: The examples: (problem, strategy predicate, title, lines to mark, notes by line)
EXAMPLES = [
    (
        "textbook_49b7d961bb8d378750b8",
        "contradiction.working",
        "Proof by contradiction",
        {4, 8, 12, 13},
        {4: "negated goal", 8: "MT"},
    ),
    (
        "textbook_5b38b6ab0a04f4c4bb28",
        "disj_syll",
        "Disjunctive syllogism, twice",
        {9, 10, 15, 16},
        {9: "case refuted", 15: "case refuted"},
    ),
]
RULE = {
    "PR": "PR",
    "AS": "AS",
    "R": "R",
    "ANDI": r"$\wedge$I",
    "ANDE1": r"$\wedge$E",
    "ANDE2": r"$\wedge$E",
    "IMPE": r"$\to$E",
    "IMPI": r"$\to$I",
    "ORI1": r"$\vee$I",
    "ORI2": r"$\vee$I",
    "ORE": r"$\vee$E",
    "NEGE": r"$\neg$E",
    "NEGI": r"$\neg$I",
    "BOTE": r"$\bot$E",
    "DN": r"$\neg\neg$E",
}


def mtext(f: tuple, top: bool = True) -> str:
    """formula tuple -> mathtext body (outer parentheses dropped)."""
    t = f[0]
    if t == "atom":
        return str(f[1])
    if t == "bot":
        return r"\bot"
    if t == "not":
        return r"\neg " + mtext(f[1], False)
    op = {"and": r"\wedge", "or": r"\vee", "imp": r"\to"}[t]
    body = f"{mtext(f[1], False)} {op} {mtext(f[2], False)}"
    return body if top else f"({body})"


def _box(s: int, e: int) -> str:
    return str(s) if s == e else f"{s}–{e}"


def cites(ln: dict[str, Any]) -> str:
    r = ln["refs"]
    if ln["rule"] in ("IMPI", "NEGI"):
        return _box(r[0], r[1])
    if ln["rule"] == "ORE":
        return f"{r[0]}, {_box(r[1], r[2])}, {_box(r[3], r[4])}"
    return ", ".join(map(str, r))


def _row(stem: str, seed: int, name: str) -> dict[str, Any]:
    with gzip.open(
        C.OUT / "reads" / f"{read_name('charles149', stem, seed, K)}.json.gz", "rt"
    ) as f:
        return dict(json.load(f)[name])


def pick_example(name: str, pred: str) -> dict[str, Any]:
    """The shortest RL proof (L3 end, any of seeds 0-2) of `name` with the strategy and no dead
    lines (ties: most frequent), with every Lean text of its read that converts to it; RL and
    base success counts per seed."""
    rl = {s: _row(C.END, s, name) for s in (0, 1, 2)}
    ok = [
        (p[2]["Lp"], -p[1], s, p)
        for s, r in rl.items()
        for p in r["proofs"]
        if (pred in p[2]["sub"] or pred in p[2]["u"]) and p[2]["L"] == p[2]["Lp"]
    ]
    _, _, seed, (nd, count, lab) = min(ok, key=lambda x: x[:3])
    texts = [
        t
        for r in C.iter_rows(C.read_path("charles149", C.END, seed, K), {name})
        for t in r["accepted"]
        if C.lean_to_nd(t) == nd
    ]
    return {
        "name": name,
        "prompt": rl[0]["prompt"],
        "nd": nd,
        "count": count,
        "seed": seed,
        "k": rl[0]["k"],
        "rl_n_ok_seeds": [rl[s]["n_ok"] for s in (0, 1, 2)],
        "base_n_ok_seeds": [_row("s1g3", s, name)["n_ok"] for s in (0, 1, 2)],
        "labels": lab,
        "lean_texts": texts,
    }


def lean_recheck(examples: list[dict[str, Any]]) -> list[bool]:
    from code.thinking.lean import lean_check

    os.environ.setdefault("LEAN_GATE_LOG", os.path.join(tempfile.mkdtemp(), "lean_gate.jsonl"))
    items = [(C.lean_statement(e["prompt"]), t) for e in examples for t in e["lean_texts"][:3]]
    return lean_check(items)


def draw_proof(
    ax: Any, ex: dict[str, Any], title: str, mark: set[int], notes: dict[int, str]
) -> None:
    """One proof in Fitch notation on an axes whose data units are points (set by the caller)."""
    C.dan_strategies()
    from patterns import parse  # type: ignore[import-not-found]

    W, H = ax.get_xlim()[1], ax.get_ylim()[1]
    fs, lead, ind = S.FS_SMALL, 7.0, 5.0
    lines = parse(ex["nd"])
    ax.text(0, H, title, fontsize=S.FS_TICK, va="top", ha="left", color=RL_HUE, fontweight="bold")
    b = ex["base_n_ok_seeds"]
    rl_pct = 100 * sum(ex["rl_n_ok_seeds"]) / (3 * ex["k"])
    ax.text(
        0,
        H - 8.5,
        f"RL {rl_pct:.0f}%, base {sum(b)} of {3 * ex['k']:,}",
        fontsize=fs,
        va="top",
        ha="left",
        color=S.INK2,
    )
    y0 = H - 22.0
    base_y: dict[int, float] = {}
    for i, ln in enumerate(lines):
        d, y = ln["depth"], y0 - i * lead
        base_y[ln["idx"]] = y
        col = RL_HUE if ln["idx"] in mark else S.INK
        ax.text(7.5, y, str(ln["idx"]), fontsize=fs, va="baseline", ha="right", color=S.INK2)
        ftext = ax.text(
            11.5 + ind * d,
            y,
            f"${mtext(ln['formula'])}$",
            fontsize=fs,
            va="baseline",
            ha="left",
            color=col,
        )
        just = RULE[ln["rule"]] + ("" if ln["rule"] in ("PR", "AS") else " " + cites(ln))
        ax.text(
            W,
            y,
            just,
            fontsize=fs,
            va="baseline",
            ha="right",
            color=RL_HUE if ln["idx"] in mark else S.INK2,
        )
        if ln["idx"] in notes:
            # the note starts 3 pt after the rendered formula
            bb = ftext.get_window_extent(ax.figure.canvas.get_renderer())
            x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
            ax.text(
                x_end + 3.0,
                y,
                notes[ln["idx"]],
                fontsize=fs - 0.5,
                va="baseline",
                ha="left",
                color=RL_HUE,
                style="italic",
            )
    # Fitch bars: a box runs from its assumption to the last line before the depth drops or a
    # sibling assumption opens at the same depth; a short rule under each assumption
    for i, ln in enumerate(lines):
        if ln["rule"] != "AS":
            continue
        d, end = ln["depth"], ln["idx"]
        for nxt in lines[i + 1 :]:
            if nxt["depth"] < d or (nxt["depth"] == d and nxt["rule"] == "AS"):
                break
            end = nxt["idx"]
        xb = 10.0 + ind * (d - 1)
        ax.plot(
            [xb, xb],
            [base_y[ln["idx"]] + 4.7, base_y[end] - 1.4],
            color=S.INK,
            lw=0.5,
            solid_capstyle="butt",
        )
        ax.plot(
            [xb, xb + 9.0],
            [base_y[ln["idx"]] - 1.5] * 2,
            color=S.INK,
            lw=0.5,
            solid_capstyle="butt",
        )


def logistic_curve(x: np.ndarray, f: dict[str, float]) -> np.ndarray:
    return f["lo"] + (f["hi"] - f["lo"]) / (1 + np.exp(-(x - f["mid_log10R"]) / f["scale_decades"]))


def _pow10(v: float, _pos: Any = None) -> str:
    return f"$10^{{{int(round(math.log10(v)))}}}$"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--draws", type=int, default=4000)
    a = ap.parse_args(argv)
    N = json.loads((C.HERE / "numbers.json").read_text())
    out: dict[str, Any] = {}

    # (a) data
    keep = KEEP["charles149"]
    assert keep is not None

    def arm(stem: str) -> stats.Arm:
        x = stats.load_arm(stem, [C.read_path("charles149", stem, s, K) for s in (0, 1, 2)])
        x = x.subset(sorted(keep & set(x.names)), stem)
        assert len(x.names) == 149 and x.kmax >= K
        return x

    ks = pow2(K)
    arms = {"base": arm("s1g3"), "rl": arm(C.END)}
    curves = {}
    for nm, x in arms.items():
        est = stats.ci_curve(x, ks, draws=a.draws, seed=0)
        curves[nm] = {
            "k": ks,
            "mean": [100 * e.est for e in est],
            "lo": [100 * e.lo for e in est],
            "hi": [100 * e.hi for e in est],
        }
    out["a"] = curves

    # (b) data, with the Lean re-check
    exs = [pick_example(nm, pred) for nm, pred, _, _, _ in EXAMPLES]
    verdicts = lean_recheck(exs)
    out["b"] = [
        {kk: v for kk, v in e.items() if kk != "labels"} | {"u": e["labels"]["u"]} for e in exs
    ]
    out["b_lean"] = verdicts
    if not all(verdicts) or not all(e["lean_texts"] for e in exs):
        raise SystemExit(f"Lean re-check failed: {verdicts}")
    sealed = __import__("code.thinking.data", fromlist=["EvalIndex"]).EvalIndex(C.TB72)
    hits = [e["name"] for e in exs if sealed.hit(e["prompt"])]
    if hits:
        raise SystemExit(f"example in a textbook72 class: {hits}")

    # (c) data
    q3c, q3t = N["q3"]["charles149"], N["q3"]["t1320s250"]
    s1 = [(p["R"], p["never_rate"]) for p in q3c["points"]]
    s2 = [(p["R"], p["A"]["work_red"]) for p in q3t["points"]]
    out["c"] = {
        "charles149_never_rate": s1,
        "charles149_never_n": q3c["base"]["never_n"],
        "t1320s250_work_red_A": s2,
        "t1320s250_base_work_red_A": q3t["base"]["A"]["work_red"],
        "fits": {
            "never_rate": q3c["fits"]["never_rate"],
            "work_red_A_t1320s250": q3t["fits"]["A:work_red"],
        },
    }

    with S.paper_style():
        FW, FH = S.FULL_W, 2.3
        fig = S.new_figure(FW, FH)
        fig.set_layout_engine("none")

        def at(left: float, bottom: float, w: float, h: float) -> Any:
            return fig.add_axes((left / FW, bottom / FH, w / FW, h / FH))

        ax = at(0.40, 0.36, 0.90, 1.70)
        p1, p2 = at(1.42, 0.06, 1.02, 1.93), at(2.54, 0.06, 1.22, 1.93)
        cx = at(4.10, 0.36, 0.93, 1.70)
        dx_ = at(5.10, 0.36, 0.40, 1.70)
        # (a)
        for nm, col, ls, lab in (
            ("base", BASE_C, (0, (3, 1.5)), "base"),
            ("rl", RL_HUE, "-", "RL"),
        ):
            c = curves[nm]
            ax.fill_between(c["k"], c["lo"], c["hi"], color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(c["k"], c["mean"], color=col, ls=ls, lw=1.2, zorder=4, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, K)
        ax.xaxis.set_major_locator(FixedLocator([1, 8, 64, 1024]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}"))
        ax.xaxis.set_minor_locator(FixedLocator(ks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(25, 100)
        ax.set_xlabel("k (attempts)")
        ax.set_ylabel("pass@k (% of Charles149)")
        S.light_grid(ax)
        ax.legend(loc="lower right", fontsize=S.FS_SMALL)
        S.panel_title(ax, "a", "pass@k")
        # (b)
        for pax, ex, (_, _, title, mark, notes) in (
            (p1, exs[0], EXAMPLES[0]),
            (p2, exs[1], EXAMPLES[1]),
        ):
            w_in, h_in = pax.get_position().width * FW, pax.get_position().height * FH
            pax.set_xlim(0, 72 * w_in)
            pax.set_ylim(0, 72 * h_in)
            pax.set_axis_off()
            draw_proof(pax, ex, title, mark, notes)
        fig.text(
            1.42 / FW,
            (2.06 + 3.0 / 72) / FH,
            "(b) RL proofs where the base almost never succeeds",
            fontsize=S.FS_LABEL,
            va="bottom",
            ha="left",
        )
        # (c)
        xs = np.linspace(math.log10(6e-5), math.log10(0.6), 200)
        for pts, fit, col, mk, mfc in (
            (s1, out["c"]["fits"]["never_rate"], RL_HUE, "o", RL_HUE),
            (s2, out["c"]["fits"]["work_red_A_t1320s250"], RL2_C, "s", "white"),
        ):
            cx.plot(10**xs, logistic_curve(xs, fit), color=col, lw=0.9, zorder=3)
            cx.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                ls="none",
                marker=mk,
                ms=2.6,
                mfc=mfc,
                mec=col,
                mew=0.7,
                zorder=4,
            )
        # the base sits at its own x position, left of a break in the log axis
        cx.plot(
            [XB * 0.8],
            [0.0],
            ls="none",
            marker="o",
            ms=2.6,
            mfc=RL_HUE,
            mec=RL_HUE,
            mew=0.7,
            zorder=5,
            clip_on=False,
        )
        cx.plot(
            [XB * 1.25],
            [out["c"]["t1320s250_base_work_red_A"]],
            ls="none",
            marker="s",
            ms=2.6,
            mfc="white",
            mec=RL2_C,
            mew=0.7,
            zorder=5,
            clip_on=False,
        )
        tr = cx.get_xaxis_transform()
        for x0 in (2.9e-5, 3.5e-5):
            cx.plot(
                [x0, x0 * 1.25],
                [-0.025, 0.025],
                transform=tr,
                color=S.INK,
                lw=0.6,
                clip_on=False,
                zorder=6,
            )
        cx.plot([YUE_MEDIAN] * 2, [-1, 30], color=S.INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
        cx.text(
            YUE_MEDIAN * 1.3,
            7.0,
            "Yue et al. median",
            rotation=90,
            fontsize=S.FS_SMALL,
            va="bottom",
            ha="left",
            color=S.INK2,
        )
        cx.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=RL_HUE,
                    marker="o",
                    ms=2.6,
                    lw=0.9,
                    label="base-unsolved,\nCharles149",
                ),
                Line2D(
                    [],
                    [],
                    color=RL2_C,
                    marker="s",
                    ms=2.6,
                    mfc="white",
                    lw=0.9,
                    label="working reductio,\n13–20 lines",
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=S.FS_SMALL,
            handlelength=1.2,
            borderaxespad=0.0,
            borderpad=0.15,
            frameon=False,
        )
        cx.set_xscale("log")
        cx.set_xlim(6e-6, 0.8)
        cx.set_ylim(-1, 46)
        cx.yaxis.set_major_locator(FixedLocator([0, 10, 20, 30]))
        cx.xaxis.set_major_locator(FixedLocator([XB, 1e-3, 1e-1]))
        cx.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "base" if v < 2e-5 else _pow10(v)))
        cx.xaxis.set_minor_locator(FixedLocator([1e-4, 1e-2]))
        cx.xaxis.set_minor_formatter(NullFormatter())
        cx.set_xlabel("RL / pretraining FLOPs")
        cx.set_ylabel("% of attempts")
        S.light_grid(cx)
        S.panel_title(cx, "c", "emergence")
        # (d)
        dx_.set_axis_off()
        dx_.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=dx_.transAxes,
                fill=False,
                ec=S.REF,
                lw=0.7,
                ls=(0, (3, 2)),
            )
        )
        dx_.text(
            0.5,
            0.5,
            "RL traj.\ngen.\n(pending)",
            transform=dx_.transAxes,
            ha="center",
            va="center",
            fontsize=S.FS_SMALL,
            color=S.INK2,
        )
        S.panel_title(dx_, "d", "")
        paths = []
        base = FIGDIR / "fig1_emergence"
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {"Creator": "proof_strategies_grpo_20261101", "CreationDate": None}
        for ext, kw in ((".pdf", {"metadata": meta}), (".png", {"dpi": 300})):
            fig.savefig(base.with_suffix(ext), **kw)
            paths.append(str(base.with_suffix(ext)))
        S.close(fig)
    out["paths"] = paths
    (C.HERE / "fig1_numbers.json").write_text(json.dumps(out, indent=1))
    print("wrote", paths, "Lean", verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
