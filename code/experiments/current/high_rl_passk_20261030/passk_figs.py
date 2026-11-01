"""Base vs RL pass@k figures for the high-RL regime paper (F1-F4) and the numbers behind them.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \
        code.experiments.current.high_rl_passk_20261030.passk_figs [--draws 2000] \
        [--recount-stage1] [--extra-f3 POINTS.json] [--figdir DIR] [--only f1,f2,f3,f4,numbers]

Reads existing eval rows and GRPO metrics only (read-only on every run directory), CPU only.
pass@k is the unbiased estimate 1 - C(n-c, k)/C(n, k) per problem, averaged over problems and
seeds (`code.thinking.stats`); bands and intervals are its hierarchical bootstrap (problems, and
seeds where an arm has several), 95% percentile.

Writes `passk_base_vs_rl`, `gap_vs_k`, `gap_vs_rl_compute` (PDF + PNG) into `--figdir`
(default: docs/papers/high-rl-regime/figures/) and `numbers.json` / `numbers.md` next to this
file. README.md holds the definitions and the findings.

F3's points come from `F3_RUNS` (a checkpoint = metrics segments for compute + read paths for the
gaps). To add an arm (e.g. the planned low-RL arm), append a `Point` to `f3_points()`, or pass
`--extra-f3 FILE.json`: a list of {"label", "family", "set", "x", "gap1", "gapk"} with x the FLOP
ratio and gaps in points (family "v2", "v1" or any new name, which gets its own marker).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from code.ndbench.figures import style as S
from code.thinking import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIGDIR = REPO / "docs/papers/high-rl-regime/figures"

R = Path("/scratch/cowifornia/nd-rl/think/runs")  # ynez world (canonical): v2 recipe
V2 = R / "eval/v2"
RS = R / "eval/rebuild_sanity"
G2 = R / "grpo_v2"
SR = Path("/scratch/cowifornia/nd-rl/think_serrano_20261011/runs")  # pre-crash world: v1 RL
SM = SR / "eval/matched"

YUE_MEDIAN = 1.2e-4  # Dan's rl-compute-share review: Yue et al.'s RL / pretraining FLOP median

# ------------------------------------------------------------------------------------ problem sets


def _names(path: Path) -> set[str]:
    with open(path) as f:
        return {json.loads(line)["name"] for line in f if line.strip()}


def valid_1320() -> set[str]:
    """The 986 valid 13-20-line items (`thinking_traces_20261005/matched_eval/figs.valid_1320`)."""
    fin = {}
    with open(R / "pools/test_13_20/test_13_20.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fin[r["name"]] = r
    ok = set()
    with open(R / "eval/matched/sets/t1320_n1000.jsonl") as f:
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


KEEP: dict[str, set[str] | None] = {
    "charles149": _names(R / "pools/charles_1002/charles_clean.jsonl"),
    "t1320": valid_1320(),
    "t2130": None,
    "tb72": None,
}
READ = {"charles149": "charles173", "t1320": "t1320", "t2130": "t2130", "tb72": "tb72"}
SET_TITLE = {
    "charles149": "Charles149 (source OOD)",
    "t1320": "13–20 lines (length OOD)",
    "t2130": "21–30 lines",
    "tb72": "textbook72",
}

# -------------------------------------------------------------------------------------- read paths


def v2_rows(st: str, arm: str, seed: int, k: int) -> Path:
    """A v2 read: `arm` is the eval-name stem after the set ('s1g3', 'grpo_think_vs_L2', ...)."""
    return V2 / f"{READ[st]}_{arm}_s{seed}_plain_k{k}" / "rows_plain_B64.jsonl"


#: The v2 chain (seed 0 unless noted): name -> (eval stem, {set: k of the read}).
V2_ARMS: dict[str, tuple[str, dict[str, int]]] = {
    "base": ("s1g3", {"charles149": 64, "t1320": 8, "t2130": 8}),
    "mini20 L2it8": ("grpo_think_vs_mini20_L2it8", {"charles149": 16, "t1320": 8}),
    "L2 end": ("grpo_think_vs_L2", {"charles149": 64, "t1320": 8, "t2130": 8}),
    "L3 it10": ("grpo_think_vs_mb_kkl0_it10", {"charles149": 16, "t1320": 4, "t2130": 4}),
    "L3 it20": ("grpo_think_vs_mb_kkl0_it20", {"charles149": 16, "t1320": 4, "t2130": 4}),
    "L3 it30": ("grpo_think_vs_mb_kkl0_it30", {"charles149": 16, "t1320": 4, "t2130": 4}),
    "L3 end": ("grpo_think_vs_mb_kkl0", {"charles149": 64, "t1320": 16, "t2130": 8, "tb72": 64}),
    "ki78 end": ("grpo_think_vs_mb_ki78", {"charles149": 64, "t1320": 16, "t2130": 8}),
}
CHAIN = ["mini20 L2it8", "L2 end", "L3 it10", "L3 it20", "L3 it30", "L3 end", "ki78 end"]
THREE_SEEDS = {"L2 end", "L3 it10", "L3 it20", "L3 it30", "L3 end"}


def v2_paths(name: str, st: str, seeds: Sequence[int]) -> list[Path]:
    stem, ks = V2_ARMS[name]
    if st == "tb72" and name == "base":
        return [RS / f"tb72_S1g3_s{s}" / "rows_plain_B256.jsonl" for s in seeds]
    return [v2_rows(st, stem, s, ks[st]) for s in seeds]


#: v1 (serrano) reads: name -> {set: rows path}.  T1v1 arms at B256, T0 arms at B0.
def v1_paths() -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for base, b in (("T1v1", 256), ("T0", 0)):
        for s in (0, 1):
            out[f"v1 base {base}_s{s}"] = {
                "tb72": SR / f"eval/tb72_{base}_s{s}/rows_plain_B{b}.jsonl",
                "t1320": SM / f"t1320_{base}_s{s}/rows_plain_B{b}.jsonl",
            }
        run = "grpoT1v1e" if base == "T1v1" else "grpoT0"
        for s in (0, 1):
            sfx = "" if s == 0 else "s1"
            for it in (39, 78):
                out[f"v1 GRPO {base}_s{s} it{it}"] = {
                    st: SM / f"{st}_{run}{sfx}_it{it}/rows_plain_B{b}.jsonl"
                    for st in ("tb72", "t1320")
                }
    out["v1 EI T1v1_s0 r2"] = {"tb72": SM / "tb72_eiT1v1e_r2/rows_plain_B256.jsonl"}
    return out


V1 = v1_paths()

# ------------------------------------------------------------------------------------------- arms

_ARM_CACHE: dict[tuple[str, tuple[str, ...]], stats.Arm] = {}


def arm(label: str, paths: Sequence[Path], st: str) -> stats.Arm:
    key = (st, tuple(str(p) for p in paths))
    if key not in _ARM_CACHE:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        a = stats.load_arm(label, list(paths))
        keep = KEEP[st]
        if keep is not None:
            a = a.subset(sorted(keep & set(a.names)), label)
            if len(a.names) != len(keep):
                raise ValueError(f"{label} on {st}: {len(a.names)} of {len(keep)} problems")
        _ARM_CACHE[key] = a
    return _ARM_CACHE[key]


def v2_arm(name: str, st: str, seeds: Sequence[int] | None = None) -> stats.Arm:
    if seeds is None:
        seeds = (0, 1, 2) if name in THREE_SEEDS or (name == "base" and st == "tb72") else (0,)
    return arm(f"{name} s{''.join(map(str, seeds))}", v2_paths(name, st, seeds), st)


def v1_arm(name: str, st: str) -> stats.Arm:
    return arm(name, [V1[name][st]], st)


def mean_curve(a: stats.Arm) -> np.ndarray:
    """pass@k for k = 1..kmax, mean over seeds and problems (fraction)."""
    return a.curve().mean(axis=(0, 1))


def pow2(K: int) -> list[int]:
    ks = [2**i for i in range(int(math.log2(K)) + 1)]
    return ks if ks[-1] == K else ks + [K]


def band(a: stats.Arm, ks: Sequence[int], draws: int) -> tuple[np.ndarray, np.ndarray]:
    est = stats.ci_curve(a, ks, draws=draws, seed=0)
    return np.array([e.lo for e in est]), np.array([e.hi for e in est])


def _seed_w(dr: stats.Draws, S_: int) -> np.ndarray:
    return dr.seed_weights(S_)


def _means(V: np.ndarray, U: np.ndarray, W: np.ndarray) -> np.ndarray:
    return ((U @ V) * W).sum(1) / (V.shape[0] * W.sum(1))


def diff_curve(
    a: stats.Arm, b: stats.Arm, ks: Sequence[int], draws: int, paired_seeds: bool = False
) -> dict[str, np.ndarray]:
    """a - b (points) at each k, one problem resample shared by both arms and every k; seeds
    resampled independently unless `paired_seeds` (same Stage-1 seed order in both arms)."""
    if a.names != b.names:
        raise ValueError(f"{a.label} vs {b.label}: different problems")
    dr = stats.make_draws(a.T, draws=draws, seed=0)
    Ua = _seed_w(dr, a.S)
    Ub = Ua if paired_seeds else _seed_w(dr, b.S)
    est, lo, hi, p = [], [], [], []
    for k in ks:
        Va, Vb = a.values(k), b.values(k)
        d = _means(Va, Ua, dr.W) - _means(Vb, Ub, dr.W)
        est.append(Va.mean() - Vb.mean())
        q = np.quantile(d, [0.025, 0.975])
        lo.append(q[0])
        hi.append(q[1])
        p.append(stats.boot_p(d))
    return {
        "k": np.array(ks),
        "est": 100 * np.array(est),
        "lo": 100 * np.array(lo),
        "hi": 100 * np.array(hi),
        "p": np.array(p),
    }


# ------------------------------------------------------------------------------------ RL compute


def _rows_upto(metrics: Path, upto: int | None) -> list[dict[str, Any]]:
    """metrics.jsonl rows in file order up to the last row of iteration `upto` (all rows if None):
    rolled-back iterations count, since their compute was spent."""
    with open(metrics) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if upto is None:
        return rows
    last = max(i for i, r in enumerate(rows) if r["iter"] == upto)
    return rows[: last + 1]


def rl_tokens(segments: Sequence[tuple[Path, int | None]]) -> dict[str, float]:
    """Sampled and trained tokens over GRPO stages (run dir, last iteration or None)."""
    samp = trained = att = 0.0
    for d, upto in segments:
        a = json.loads((d / "args.json").read_text())
        per_it = int(a["prompts"]) * int(a["group"])
        for r in _rows_upto(d / "metrics.jsonl", upto):
            samp += float(r["tokens_per_attempt"]) * per_it
            trained += float(r["trained_tokens"])
            att += per_it
    return {"sampled": samp, "trained": trained, "attempts": att}


STAGE1_CACHE = HERE / "stage1_tokens.json"
STAGE1 = {  # model -> (train.json, data pickle)
    "S1g3_s0": (R / "stage1/S1g3_s0/train.json", R / "data/S1g3.pkl"),
    "T1v1_s0": (SR / "stage1/T1v1_s0/train.json", SR / "data/T1v1.pkl"),
    "T1v1_s1": (SR / "stage1/T1v1_s1/train.json", SR / "data/T1v1.pkl"),
    "T0_s0": (SR / "stage1/T0_s0/train.json", SR / "data/T0.pkl"),
    "T0_s1": (SR / "stage1/T0_s1/train.json", SR / "data/T0.pkl"),
}


def stage1_tokens(recount: bool) -> dict[str, dict[str, float]]:
    """Tokens Stage-1 trained on: epochs (train.json) x tokens per epoch (state + completion of
    every pair in the data pickle, no padding).  Cached in stage1_tokens.json."""
    if STAGE1_CACHE.exists() and not recount:
        return json.loads(STAGE1_CACHE.read_text())
    per_pickle: dict[str, tuple[int, int, int]] = {}
    out: dict[str, dict[str, float]] = {}
    for model, (tj, pk) in STAGE1.items():
        if str(pk) not in per_pickle:
            with open(pk, "rb") as f:
                data = pickle.load(f)
            P = Q = N = 0
            for rec in data["records"]:
                for t in rec:
                    P, Q, N = P + len(t[0]), Q + len(t[1]), N + 1
            per_pickle[str(pk)] = (P, Q, N)
        P, Q, N = per_pickle[str(pk)]
        t = json.loads(tj.read_text())
        ep = float(t["epochs"])
        out[model] = {
            "epochs": ep,
            "steps": t["steps"],
            "pairs_per_epoch": N,
            "state_tokens_per_epoch": P,
            "completion_tokens_per_epoch": Q,
            "tokens": ep * (P + Q),
            "completion_tokens": ep * Q,
            "n_params": t["n_params"],
            "secs": t["secs"],
        }
    STAGE1_CACHE.write_text(json.dumps(out, indent=1) + "\n")
    return out


def flop_ratio(tok: dict[str, float], s1_tokens: float) -> float:
    """(6 x trained-token passes + 2 x sampled tokens) / (6 x Stage-1 tokens); N cancels (same
    network); prefill, old-log-prob, KL-anchor and canary passes excluded."""
    return (6 * tok["trained"] + 2 * tok["sampled"]) / (6 * s1_tokens)


def v2_segments(name: str, seed: int = 0) -> list[tuple[Path, int | None]]:
    L2 = (G2 / f"grpo_think_vs_s{seed}_L2", None)
    if name == "base":
        return []
    if name == "mini20 L2it8":
        return [(G2 / f"grpo_think_vs_mini20_s{seed}_L2", 8)]
    if name == "L2 end":
        return [L2]
    if name.startswith("L3 it"):
        return [L2, (G2 / f"grpo_think_vs_mb_kkl0_s{seed}_L3", int(name[5:]))]
    if name == "L3 end":
        return [L2, (G2 / f"grpo_think_vs_mb_kkl0_s{seed}_L3", None)]
    if name == "ki78 end":
        return [L2, (G2 / f"grpo_think_vs_mb_ki78_s{seed}_L3", None)]
    raise ValueError(name)


def v1_segments(name: str) -> tuple[list[tuple[Path, int | None]], str]:
    """'v1 GRPO T1v1_s0 it39' -> (segments, Stage-1 model)."""
    _, _, model, it = name.split()
    base, s = model.split("_s")
    run = ("grpo_T1v1e" if base == "T1v1" else "grpo_T0") + ("" if s == "0" else "_s1")
    return [(SR / "restart3_grpofix" / run, int(it[2:]))], model


# ------------------------------------------------------------------------------------------ plots

RL_HUE = S.OKABE_ITO["vermillion"]
BASE_C = "#7a7a7a"
EI_C = S.OKABE_ITO["bluish_green"]
V1_C = S.OKABE_ITO["blue"]


def ramp(i: int, n: int) -> str:
    """Light (little RL) to dark (most RL), one hue."""
    t = 0.62 - (0.62 + 0.5) * (i / max(1, n - 1))
    return S.shade(RL_HUE, t)


CHAIN_COLOR = {name: ramp(i, len(CHAIN)) for i, name in enumerate(CHAIN)}
CHAIN_MARK = dict(zip(CHAIN, ("v", "o", "o", "o", "o", "s", "D"), strict=True))


def draw_curve(
    ax: Any,
    a: stats.Arm,
    *,
    color: str,
    label: str,
    marker: str = "o",
    ls: Any = "-",
    bands: bool = False,
    draws: int = 2000,
    lw: float = 1.1,
) -> dict[str, Any]:
    c = 100 * mean_curve(a)
    K = len(c)
    kk = np.arange(1, K + 1)
    ax.plot(kk, c, color=color, ls=ls, lw=lw, zorder=3)
    mk = pow2(K)
    ax.plot(
        mk,
        c[np.array(mk) - 1],
        ls="none",
        marker=marker,
        ms=3.0,
        color=color,
        mec="white",
        mew=0.4,
        zorder=4,
    )
    out: dict[str, Any] = {"label": label, "k": mk, "pass": [float(c[k - 1]) for k in mk], "S": a.S}
    if bands:
        lo, hi = band(a, mk, draws)
        ax.fill_between(mk, 100 * lo, 100 * hi, color=color, alpha=0.16, lw=0, zorder=2)
        out["lo"], out["hi"] = (100 * lo).tolist(), (100 * hi).tolist()
    return out


def legend_line(color: str, label: str, marker: str = "o", ls: Any = "-") -> Any:
    from matplotlib.lines import Line2D

    return Line2D(
        [], [], color=color, ls=ls, lw=1.1, marker=marker, ms=3.0, mec="white", mew=0.4, label=label
    )


def crossing(a: stats.Arm, b: stats.Arm) -> float | None:
    """First k where pass@k(a) - pass@k(b) turns from > 0 to <= 0 (linear in log k)."""
    K = min(a.kmax, b.kmax)
    d = mean_curve(a)[:K] - mean_curve(b)[:K]
    for i in range(1, K):
        if d[i - 1] > 0 >= d[i]:
            x0, x1 = math.log(i), math.log(i + 1)
            return math.exp(x0 + (x1 - x0) * d[i - 1] / (d[i - 1] - d[i]))
    return None


def save(fig: Any, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(paths[0], metadata={"Creator": "high_rl_passk_20261030", "CreationDate": None})
    fig.savefig(paths[1], dpi=300)
    return paths


LABELS = {
    "base": "base (Stage-1 S1g3)",
    "mini20 L2it8": "mini L2 it8",
    "L2 end": "L2 end",
    "L3 it10": "L3 it10",
    "L3 it20": "L3 it20",
    "L3 it30": "L3 it30",
    "L3 end": "L3 end (recipe)",
    "ki78 end": "L3 ×2 (ki78)",
}


def fig1(draws: int, figdir: Path, N: dict[str, Any]) -> list[Path]:
    out: dict[str, Any] = {}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.45)
        gs = fig.add_gridspec(1, 3)
        axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
        # (a) Charles149, seed-0 chain
        ax = axs[0]
        res = {
            "base": draw_curve(
                ax,
                v2_arm("base", "charles149"),
                color=BASE_C,
                label="base",
                ls=(0, (3, 1.5)),
                bands=True,
                draws=draws,
            )
        }
        for name in CHAIN:
            a = v2_arm(name, "charles149", (0,))
            res[name] = draw_curve(
                ax,
                a,
                color=CHAIN_COLOR[name],
                label=name,
                marker=CHAIN_MARK[name],
                bands=name in ("L2 end", "L3 end"),
                draws=draws,
            )
        out["a_charles149_s0"] = res
        S.log2_k_axis(ax, 64, every=4)
        ax.set_ylim(30, 92)
        S.panel_title(ax, "a", "Charles149, seed-0 chain")
        ax.set_ylabel("pass@k (% of set)")
        # (b) 13-20 lines, 3-seed means
        ax = axs[1]
        res = {
            "base": draw_curve(
                ax,
                v2_arm("base", "t1320"),
                color=BASE_C,
                label="base",
                ls=(0, (3, 1.5)),
                bands=True,
                draws=draws,
            )
        }
        for name in ("L2 end", "L3 it10", "L3 it20", "L3 it30", "L3 end"):
            a = v2_arm(name, "t1320")
            res[name] = draw_curve(
                ax,
                a,
                color=CHAIN_COLOR[name],
                label=name,
                marker=CHAIN_MARK[name],
                bands=name in ("L2 end", "L3 end"),
                draws=draws,
            )
        out["b_t1320_3seed"] = res
        S.log2_k_axis(ax, 16, every=2)
        ax.set_ylim(0, 80)
        S.panel_title(ax, "b", "13–20 lines, 3 RL seeds")
        # (c) v1 textbook72
        ax = axs[2]
        base1 = v1_arm("v1 base T1v1_s0", "tb72")
        g39 = v1_arm("v1 GRPO T1v1_s0 it39", "tb72")
        g78 = v1_arm("v1 GRPO T1v1_s0 it78", "tb72")
        ei2 = v1_arm("v1 EI T1v1_s0 r2", "tb72")
        res: dict[str, Any] = {
            "base": draw_curve(
                ax,
                base1,
                color=BASE_C,
                label="base T1v1",
                ls=(0, (3, 1.5)),
                bands=True,
                draws=draws,
            ),
            "GRPO it39": draw_curve(
                ax, g39, color=ramp(1, 4), label="GRPO it39", marker="o", draws=draws
            ),
            "GRPO it78": draw_curve(
                ax, g78, color=ramp(3, 4), label="GRPO it78", marker="s", bands=True, draws=draws
            ),
            "EI r2": draw_curve(
                ax, ei2, color=EI_C, label="EI r2", marker="^", ls=(0, (4, 1.6)), draws=draws
            ),
        }
        kx = crossing(g78, base1)
        res["crossing_it78_vs_base_k"] = kx
        res["crossing_it39_vs_base_k"] = crossing(g39, base1)
        if kx is not None:
            y = float(np.interp(math.log(kx), np.log(np.arange(1, 65)), 100 * mean_curve(g78)[:64]))
            ax.plot([kx], [y], marker="o", ms=7, mfc="none", mec=S.INK, mew=0.8, zorder=6)
            ax.annotate(
                f"crossing\nk ≈ {kx:.0f}",
                (kx, y),
                xytext=(kx * 5.0, y - 19),
                fontsize=S.FS_TICK,
                ha="center",
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": S.INK},
            )
        out["c_v1_tb72"] = res
        S.log2_k_axis(ax, 256, every=4)
        ax.set_ylim(10, 75)
        S.panel_title(ax, "c", "v1 recipe, textbook72")
        ax.legend(
            handles=[
                legend_line(BASE_C, "base T1v1", ls=(0, (3, 1.5))),
                legend_line(ramp(1, 4), "GRPO it39"),
                legend_line(ramp(3, 4), "GRPO it78", marker="s"),
                legend_line(EI_C, "EI round 2", marker="^", ls=(0, (4, 1.6))),
            ],
            loc="upper left",
            fontsize=S.FS_SMALL,
        )
        for ax in axs:
            ax.set_xlabel("k (attempts per problem)")
            S.light_grid(ax)
        handles = [legend_line(BASE_C, "base S1g3", ls=(0, (3, 1.5)))] + [
            legend_line(CHAIN_COLOR[n], LABELS[n], marker=CHAIN_MARK[n]) for n in CHAIN
        ]
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=8,
            fontsize=S.FS_SMALL,
            handlelength=1.5,
            columnspacing=0.8,
            handletextpad=0.3,
        )
        paths = save(fig, figdir / "passk_base_vs_rl")
        S.close(fig)
    N["F1"] = out
    return paths


F2_SETS = (
    ("charles149", "Charles149 (base s0)", S.OKABE_ITO["vermillion"], "s"),
    ("tb72", "textbook72 (base 3 seeds, B256)", S.OKABE_ITO["reddish_purple"], "D"),
    ("t1320", "13–20 lines (base s0)", S.OKABE_ITO["blue"], "o"),
    ("t2130", "21–30 lines (base s0)", S.OKABE_ITO["sky_blue"], "^"),
)


def fig2(draws: int, figdir: Path, N: dict[str, Any]) -> list[Path]:
    out: dict[str, Any] = {}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.1)
        ax = fig.add_subplot(1, 1, 1)
        ax.axhline(0, color=S.REF, lw=0.7, zorder=1)
        for st, lab, col, mk in F2_SETS:
            rl, base = v2_arm("L3 end", st), v2_arm("base", st)
            K = min(rl.kmax, base.kmax)
            ks = list(range(1, K + 1))
            d = diff_curve(rl, base, ks, draws, paired_seeds=(st == "tb72"))
            ax.fill_between(d["k"], d["lo"], d["hi"], color=col, alpha=0.15, lw=0, zorder=2)
            ax.plot(d["k"], d["est"], color=col, lw=1.1, zorder=3)
            m = np.array(pow2(K)) - 1
            ax.plot(
                d["k"][m],
                d["est"][m],
                ls="none",
                marker=mk,
                ms=3.2,
                color=col,
                mec="white",
                mew=0.4,
                zorder=4,
                label=lab,
            )
            out[st] = {
                kk: [float(d["est"][i]), float(d["lo"][i]), float(d["hi"][i])]
                for i, kk in enumerate(ks)
            }
        # contrast: the v1 GRPO run that crosses its base (textbook72, single seeds)
        g78, b1 = v1_arm("v1 GRPO T1v1_s0 it78", "tb72"), v1_arm("v1 base T1v1_s0", "tb72")
        ks = list(range(1, 65))
        d = diff_curve(g78, b1, ks, draws)
        ax.plot(
            d["k"],
            d["est"],
            color=S.INK2,
            lw=0.9,
            ls=(0, (2, 1.4)),
            zorder=3,
            label="contrast: v1 GRPO it78, textbook72",
        )
        out["v1_tb72_grpo_it78"] = {
            kk: [float(d["est"][i]), float(d["lo"][i]), float(d["hi"][i])]
            for i, kk in enumerate(ks)
        }
        S.log2_k_axis(ax, 64, every=2)
        ax.set_xlabel("k (attempts per problem)")
        ax.set_ylabel("RL end − base (points)")
        S.light_grid(ax)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=S.FS_TICK)
        paths = save(fig, figdir / "gap_vs_k")
        S.close(fig)
    N["F2"] = out
    return paths


# ---------------------------------------------------------------------------------------- F3


@dataclass
class Point:
    label: str
    family: str  # "v2", "v1 T1v1", "v1 T0", or a new family (own marker)
    set: str
    x: float  # RL FLOPs / Stage-1 FLOPs
    gap1: float  # points
    gapk: float
    k: int
    ci1: tuple[float, float] = (math.nan, math.nan)
    cik: tuple[float, float] = (math.nan, math.nan)
    meta: dict[str, Any] = field(default_factory=dict)


F3_SETS = (("t1320", 4), ("charles149", 16), ("tb72", 64))  # the largest k every point shares


def f3_points(draws: int, s1: dict[str, dict[str, float]]) -> list[Point]:
    pts: list[Point] = []
    for st, k in F3_SETS:
        base = v2_arm("base", st, (0,))
        names = CHAIN if st != "tb72" else ["L3 end"]
        for name in names:
            if st not in V2_ARMS[name][1]:
                continue
            rl = v2_arm(name, st, (0,))
            d = diff_curve(rl, base, [1, k], draws)
            tok = rl_tokens(v2_segments(name, 0))
            pts.append(
                Point(
                    name,
                    "v2",
                    st,
                    flop_ratio(tok, s1["S1g3_s0"]["tokens"]),
                    float(d["est"][0]),
                    float(d["est"][1]),
                    k,
                    (float(d["lo"][0]), float(d["hi"][0])),
                    (float(d["lo"][1]), float(d["hi"][1])),
                    {"tokens": tok},
                )
            )
        if st == "charles149":
            continue  # no v1 base read on Charles149
        for name in V1:
            if not name.startswith("v1 GRPO") or st not in V1[name]:
                continue
            model = name.split()[2]
            bname = f"v1 base {model}"
            if not V1[name][st].exists() or not V1[bname][st].exists():
                continue
            rl, base1 = v1_arm(name, st), v1_arm(bname, st)
            d = diff_curve(rl, base1, [1, k], draws)
            seg, m = v1_segments(name)
            tok = rl_tokens(seg)
            pts.append(
                Point(
                    name,
                    "v1 " + model.split("_")[0],
                    st,
                    flop_ratio(tok, s1[m]["tokens"]),
                    float(d["est"][0]),
                    float(d["est"][1]),
                    k,
                    (float(d["lo"][0]), float(d["hi"][0])),
                    (float(d["lo"][1]), float(d["hi"][1])),
                    {"tokens": tok, "stage1": m},
                )
            )
    return pts


# ------------------------------------------------------------------------------- Dan's EI data

#: Read-only copies of Dan's bucket files (hf://buckets/dan-pandori/nd-rl/{support-curves,trajectory}/artifacts/).
DAN = Path("/scratch/cowifornia/nd-rl/dan_data")
DAN_CKPTS = ("pend", "r1", "r4", "r8")  # trajectory: end of pretraining (r0) and T1 ladder rounds
DAN_SET = {"tb72": "tb72", "h250": "h250"}


def dan_traj_arm(st: str, ckpt: str, seeds: Sequence[int] = (0, 1, 2)) -> stats.Arm:
    """Dan's trajectory reads (best-cap12, sample seed 1, k 256) as a stats.Arm."""
    per = []
    for s in seeds:
        with open(DAN / "trajectory" / f"s{s}_{ckpt}__{DAN_SET[st]}_x1.jsonl") as f:
            per.append(
                {(r := json.loads(line))["name"]: (int(r["n_tried"]), int(r["n_ok"])) for line in f}
            )
    names = sorted(per[0])
    if any(sorted(p) != names for p in per):
        raise ValueError(f"trajectory {st} {ckpt}: seeds cover different theorems")
    n = np.array([[p[x][0] for x in names] for p in per], dtype=np.int64)
    c = np.array([[p[x][1] for x in names] for p in per], dtype=np.int64)
    label = f"Dan {'r0' if ckpt == 'pend' else ckpt}"
    return stats.Arm(label, names, [st] * len(names), n, c, None, [], [None] * len(seeds))


def dan_ratio(rnd: int, seed: int) -> float:
    """Cumulative (6 x fine-tune tokens + 2 x generated tokens) of ladder rounds 1..rnd over
    6 x Stage-1 train tokens, from Dan's compute.json (his fine-tune tokens include the state
    prompt; GRPO's trained tokens do not)."""
    cj = json.loads((DAN / "trajectory/compute.json").read_text())
    tr = sum(cj[f"T1 ladder|{seed}|{r}"]["train_tokens"] for r in range(1, rnd + 1))
    gen = sum(cj[f"T1 ladder|{seed}|{r}"]["gen_tokens"] for r in range(1, rnd + 1))
    return (6 * tr + 2 * gen) / (6 * cj[f"stage1|{seed}"]["train_tokens"])


def dan_f3_points(draws: int) -> list[Point]:
    pts = []
    base = dan_traj_arm("tb72", "pend")
    for ck in DAN_CKPTS[1:]:
        rl = dan_traj_arm("tb72", ck)
        d = diff_curve(rl, base, [1, 64], draws, paired_seeds=True)
        rnd = int(ck[1:])
        x = float(np.mean([dan_ratio(rnd, s) for s in (0, 1, 2)]))
        pts.append(
            Point(
                f"Dan EI {ck}",
                "Dan EI cap12",
                "tb72",
                x,
                float(d["est"][0]),
                float(d["est"][1]),
                64,
                (float(d["lo"][0]), float(d["hi"][0])),
                (float(d["lo"][1]), float(d["hi"][1])),
                {"tokens": {"sampled": math.nan, "trained": math.nan}, "seeds": 3},
            )
        )
    return pts


def _sc_rows(model: str) -> dict[str, tuple[int, int, int]]:
    out = {}
    for sh in (0, 1):
        with open(DAN / "support-curves" / f"s1_{model}_T08_s0.s{sh}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                out[r["name"]] = (int(r["n_tried"]), int(r["n_ok"]), int(r["L_true"]))
    return out


def _passk_any(n: np.ndarray, c: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """T x len(ks): unbiased pass@k where k <= n; above n (only rows that stopped early after 50
    successes) the plug-in 1 - (1 - c/n)^k."""
    from math import lgamma

    out = np.zeros((len(n), len(ks)))
    for t, (nt, ct) in enumerate(zip(n.tolist(), c.tolist(), strict=True)):
        for j, k in enumerate(ks.tolist()):
            if k > nt:
                out[t, j] = 1.0 - (1.0 - ct / nt) ** k
            elif nt - ct < k:
                out[t, j] = 1.0
            else:
                out[t, j] = 1.0 - math.exp(
                    lgamma(nt - ct + 1)
                    - lgamma(nt - ct - k + 1)
                    - lgamma(nt + 1)
                    + lgamma(nt - k + 1)
                )
    return out


def fig4(draws: int, figdir: Path, N: dict[str, Any]) -> list[Path]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35)
        axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
        # (a) support-curves: 383 transfer theorems, seed 0, T 0.8, up to 10,000 attempts
        ax = axs[0]
        B, E = _sc_rows("base"), _sc_rows("ei")
        names = sorted(B)
        assert sorted(E) == names
        ks = np.unique(np.round(np.logspace(0, 4, 41)).astype(int))
        res = {}
        for lab, rows, col, mk, ls in (
            ("base (Stage 1, cap 6)", B, BASE_C, "o", (0, (3, 1.5))),
            ("EI, 8 rounds × k 32", E, EI_C, "^", "-"),
        ):
            n = np.array([rows[x][0] for x in names])
            c = np.array([rows[x][1] for x in names])
            V = _passk_any(n, c, ks)
            m = 100 * V.mean(0)
            idx = rng.integers(0, len(names), size=(draws, len(names)))
            bs = 100 * V[idx].mean(1)
            lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
            ax.fill_between(ks, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(ks, m, color=col, ls=ls, lw=1.1, zorder=3)
            sel = [i for i, k in enumerate(ks) if k in (1, 10, 100, 1000, 10000)]
            ax.plot(
                ks[sel],
                m[sel],
                ls="none",
                marker=mk,
                ms=3.0,
                color=col,
                mec="white",
                mew=0.4,
                zorder=4,
            )
            res[lab] = {
                int(k): [float(m[i]), float(lo[i]), float(hi[i])]
                for i, k in enumerate(ks)
                if i in sel
            }
            res[lab]["solved_any"] = int((c > 0).sum())
        out["a_support_curves"] = res
        S.plain_log_axis(ax, "x")
        ax.set_xlim(0.8, 1.3e4)
        S.panel_title(ax, "a", "support-curves, 383 transfer thms")
        ax.set_ylabel("pass@k (% of set)")
        ax.legend(
            handles=[
                legend_line(BASE_C, "base (cap 6)", ls=(0, (3, 1.5))),
                legend_line(EI_C, "EI r8", marker="^"),
            ],
            loc="upper left",
            fontsize=S.FS_SMALL,
        )
        # (b, c) trajectory, best-cap12, 3 seeds: r0 vs ladder rounds
        for ax, st, letter, title in (
            (axs[1], "tb72", "b", "trajectory, textbook72"),
            (axs[2], "h250", "c", "trajectory, holdout250"),
        ):
            res = {}
            for i, ck in enumerate(DAN_CKPTS):
                a = dan_traj_arm(st, ck)
                if ck == "pend":
                    res["r0"] = draw_curve(
                        ax, a, color=BASE_C, label="r0", ls=(0, (3, 1.5)), bands=True, draws=draws
                    )
                else:
                    res[ck] = draw_curve(
                        ax,
                        a,
                        color=ramp(i, len(DAN_CKPTS)),
                        label=ck,
                        marker=("v", "o", "s")[i - 1],
                        bands=ck == "r8",
                        draws=draws,
                    )
            out[f"{letter}_trajectory_{st}"] = res
            S.log2_k_axis(ax, 256, every=4)
            S.panel_title(ax, letter, title)
        axs[1].legend(
            handles=[legend_line(BASE_C, "r0 (end of pretraining)", ls=(0, (3, 1.5)))]
            + [
                legend_line(ramp(i, 4), f"EI round {ck[1:]}", marker=("v", "o", "s")[i - 1])
                for i, ck in enumerate(DAN_CKPTS)
                if i
            ],
            loc="lower right",
            fontsize=S.FS_SMALL,
        )
        for ax in axs:
            ax.set_xlabel("k (attempts per problem)")
            S.light_grid(ax)
        paths = save(fig, figdir / "dan_large_k")
        S.close(fig)
    N["F4"] = out
    return paths


FAMILY_STYLE = {
    "v2": (RL_HUE, "o"),
    "v1 T1v1": (V1_C, "s"),
    "v1 T0": (S.OKABE_ITO["sky_blue"], "D"),
    "Dan EI cap12": (S.OKABE_ITO["reddish_purple"], "^"),
}
EXTRA_MARKERS = ("v", "P", "X", "h")


def fig3(pts: list[Point], figdir: Path, N: dict[str, Any]) -> list[Path]:
    fams = list(dict.fromkeys(p.family for p in pts))
    extra = [f for f in fams if f not in FAMILY_STYLE]
    for i, f in enumerate(extra):
        FAMILY_STYLE[f] = (S.OKABE_ITO["orange"], EXTRA_MARKERS[i % len(EXTRA_MARKERS)])
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35)
        axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
        for ax, (st, k) in zip(axs, F3_SETS, strict=True):
            ax.axhline(0, color=S.REF, lw=0.7, zorder=1)
            ax.axvline(YUE_MEDIAN, color=S.INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
            ax.text(
                YUE_MEDIAN * 1.25,
                0.97,
                "Yue et al.\nmedian\n(FLOP ratio)",
                transform=ax.get_xaxis_transform(),
                fontsize=S.FS_SMALL,
                va="top",
                ha="left",
                color=S.INK2,
            )
            for fam in fams:
                col, mk = FAMILY_STYLE[fam]
                ps = sorted((p for p in pts if p.set == st and p.family == fam), key=lambda p: p.x)
                if not ps:
                    continue
                runs: dict[str, list[Point]] = {}
                for p in ps:
                    runs.setdefault(
                        p.label.rsplit(" it", 1)[0] if fam.startswith("v1") else fam, []
                    ).append(p)
                for rp in runs.values():
                    xs = [p.x for p in rp]
                    ax.plot(xs, [p.gap1 for p in rp], color=col, lw=0.8, zorder=3)
                    ax.plot(xs, [p.gapk for p in rp], color=col, lw=0.8, ls=(0, (3, 1.5)), zorder=3)
                    ax.plot(
                        xs,
                        [p.gap1 for p in rp],
                        ls="none",
                        marker=mk,
                        ms=3.6,
                        color=col,
                        mec="white",
                        mew=0.4,
                        zorder=4,
                    )
                    ax.plot(
                        xs,
                        [p.gapk for p in rp],
                        ls="none",
                        marker=mk,
                        ms=3.6,
                        mfc="white",
                        mec=col,
                        mew=0.9,
                        zorder=4,
                    )
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 4)
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

            ax.xaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _p: f"$10^{{{int(round(math.log10(v)))}}}$")
            )
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("RL / Stage-1 FLOPs")
            S.light_grid(ax)
            S.panel_title(
                ax, "abc"[F3_SETS.index((st, k))], f"{SET_TITLE[st].split(' (')[0]}, k = 1 and {k}"
            )
        axs[0].set_ylabel("RL − base (points)")
        handles = []
        for fam in fams:
            col, mk = FAMILY_STYLE[fam]
            name = {
                "v2": "v2 recipe chain (seed 0)",
                "v1 T1v1": "v1 GRPO from T1v1 (it39, it78)",
                "v1 T0": "v1 GRPO from T0, no think (it39, it78)",
                "Dan EI cap12": "Dan's EI ladder, cap 12 (r1, r4, r8; 3 seeds)",
            }.get(fam, fam)
            handles.append(legend_line(col, name, marker=mk))
        from matplotlib.lines import Line2D

        handles += [
            Line2D(
                [],
                [],
                color=S.INK2,
                marker="o",
                ls="-",
                lw=0.8,
                ms=3.6,
                mec="white",
                label="pass@1",
            ),
            Line2D(
                [],
                [],
                color=S.INK2,
                marker="o",
                ls=(0, (3, 1.5)),
                lw=0.8,
                ms=3.6,
                mfc="white",
                mec=S.INK2,
                label="pass@k (k in title)",
            ),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=S.FS_SMALL)
        paths = save(fig, figdir / "gap_vs_rl_compute")
        S.close(fig)
    N["F3"] = [
        {
            "label": p.label,
            "family": p.family,
            "set": p.set,
            "x": p.x,
            "k": p.k,
            "gap1": p.gap1,
            "ci1": list(p.ci1),
            "gapk": p.gapk,
            "cik": list(p.cik),
            **p.meta,
        }
        for p in pts
    ]
    return paths


# ------------------------------------------------------------------------------------- numbers


def headline(draws: int) -> dict[str, Any]:
    """Bootstrap 95% intervals on the headline gaps (points)."""
    out: dict[str, Any] = {}
    specs = [
        ("charles149", "L3 end", (0,), (1, 5, 16, 64)),
        ("charles149", "L3 end", (0, 1, 2), (1, 5, 16, 64)),
        ("charles149", "L2 end", (0, 1, 2), (1, 5, 16, 64)),
        ("charles149", "ki78 end", (0,), (1, 5, 16, 64)),
        ("t1320", "L3 end", (0, 1, 2), (1, 5, 8)),
        ("t1320", "L2 end", (0, 1, 2), (1, 5, 8)),
        ("t2130", "L3 end", (0, 1, 2), (1, 5, 8)),
        ("tb72", "L3 end", (0, 1, 2), (1, 5, 64)),
    ]
    for st, name, seeds, ks in specs:
        rl = v2_arm(name, st, seeds)
        base = v2_arm("base", st)
        d = diff_curve(rl, base, list(ks), draws, paired_seeds=(st == "tb72"))
        cb, cr = 100 * mean_curve(base), 100 * mean_curve(rl)
        out[f"{st}: {rl.label} − {base.label}"] = {
            int(k): {
                "base": float(cb[k - 1]),
                "rl": float(cr[k - 1]),
                "gap": float(d["est"][i]),
                "lo": float(d["lo"][i]),
                "hi": float(d["hi"][i]),
                "p": float(d["p"][i]),
            }
            for i, k in enumerate(ks)
        }
    # v1 textbook72 contrast
    for name in (
        "v1 GRPO T1v1_s0 it39",
        "v1 GRPO T1v1_s0 it78",
        "v1 GRPO T1v1_s1 it78",
        "v1 GRPO T0_s1 it78",
        "v1 EI T1v1_s0 r2",
    ):
        model = "T1v1_s0" if "EI" in name else name.split()[2]
        rl, base = v1_arm(name, "tb72"), v1_arm(f"v1 base {model}", "tb72")
        ks = (1, 5, 16, 32, 64)
        d = diff_curve(rl, base, list(ks), draws)
        cb, cr = 100 * mean_curve(base), 100 * mean_curve(rl)
        out[f"tb72: {name} − v1 base {model}"] = {
            int(k): {
                "base": float(cb[k - 1]),
                "rl": float(cr[k - 1]),
                "gap": float(d["est"][i]),
                "lo": float(d["lo"][i]),
                "hi": float(d["hi"][i]),
                "p": float(d["p"][i]),
            }
            for i, k in enumerate(ks)
        }
    return out


def _ok_lists(path: Path, keep: set[str] | None) -> dict[str, list[int]]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if keep is None or r["name"] in keep:
                out[r["name"]] = r["ok"]
    return out


def rl_only(
    name_rl: str,
    st: str,
    seed: int,
    n: int,
    base_path: Path | None = None,
    rl_path: Path | None = None,
) -> dict[str, Any]:
    """Problems RL solves in its first n attempts and the base never does in n: RL's own success
    rate on them (over all of RL's attempts), and the base's 95% upper bound per attempt."""
    keep = KEEP[st]
    bp = base_path or v2_paths("base", st, (0,))[0]
    rp = rl_path or v2_paths(name_rl, st, (seed,))[0]
    B, Rr = _ok_lists(bp, keep), _ok_lists(rp, keep)
    assert set(B) == set(Rr)
    ro = [x for x in B if sum(B[x][:n]) == 0 and sum(Rr[x][:n]) > 0]
    bo = [x for x in B if sum(B[x][:n]) > 0 and sum(Rr[x][:n]) == 0]
    p = np.array([sum(Rr[x]) / len(Rr[x]) for x in ro])
    nb = min(len(B[x]) for x in B)
    return {
        "n_problems": len(B),
        "attempts_matched": n,
        "rl_only": len(ro),
        "rl_only_pct": 100 * len(ro) / len(B),
        "base_only": len(bo),
        "base_only_pct": 100 * len(bo) / len(B),
        "rl_rate_on_rl_only_mean": float(p.mean()) if len(p) else None,
        "rl_rate_on_rl_only_median": float(np.median(p)) if len(p) else None,
        "rl_rate_quartiles": [float(q) for q in np.quantile(p, [0.25, 0.75])] if len(p) else None,
        "rl_only_with_rate_ge_0.5": int((p >= 0.5).sum()),
        "rl_only_with_rate_ge_0.1": int((p >= 0.1).sum()),
        "rl_attempts": min(len(Rr[x]) for x in Rr),
        "base_attempts": nb,
        "base_rate_upper95_if_0_of_n": 1 - 0.05 ** (1 / n),
    }


def matched_tokens(draws: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    X = [500, 1000, 2000, 4000, 8000, 16000, 32000]
    for st, name, seeds in (
        ("charles149", "L3 end", (0,)),
        ("charles149", "L3 end", (0, 1, 2)),
        ("t1320", "L3 end", (0, 1, 2)),
        ("t2130", "L3 end", (0, 1, 2)),
        ("tb72", "L3 end", (0, 1, 2)),
    ):
        rl, base = v2_arm(name, st, seeds), v2_arm("base", st)
        tpa = {"rl": rl.tokens_per_attempt().tolist(), "base": base.tokens_per_attempt().tolist()}
        ests = stats.paired_diff_at_tokens(
            rl, base, X, draws=draws, seed=0, paired_seeds=(st == "tb72")
        )
        lv_r = stats.at_tokens(rl, X, draws=200, seed=0)
        lv_b = stats.at_tokens(base, X, draws=200, seed=0)
        out[f"{st}: {rl.label} − {base.label}"] = {
            "tokens_per_attempt": tpa,
            "rows": [
                {
                    "tokens": x,
                    "rl": 100 * r.est,
                    "base": 100 * b.est,
                    "gap": 100 * e.est,
                    "lo": 100 * e.lo,
                    "hi": 100 * e.hi,
                    "p": e.p_boot,
                    "in_range": bool(e.in_range and r.in_range and b.in_range),
                    "clamped_share": e.clamped,
                }
                for x, e, r, b in zip(X, ests, lv_r, lv_b, strict=True)
            ],
        }
    return out


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _ci(e: float, lo: float, hi: float) -> str:
    return f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_md(N: dict[str, Any], path: Path) -> None:
    L = ["# Numbers behind the high-RL pass@k figures", ""]
    L += [
        "Generated by `passk_figs.py`; do not edit. Values are % of the set (pass@k) or points",
        "(gaps); [lo, hi] are 95% bootstrap intervals.",
        "",
        "## Headline gaps: RL − base",
        "",
        _row("set and arms", "k", "base", "RL", "gap [95%]", "p"),
        _row(*["---"] * 6),
    ]
    for key, rows in N["headline"].items():
        for k, v in rows.items():
            gap = _ci(v["gap"], v["lo"], v["hi"])
            L.append(_row(key, k, f"{v['base']:.1f}", f"{v['rl']:.1f}", gap, f"{v['p']:.3g}"))
    L += [
        "",
        "## RL-only problems (solved by RL in the first n attempts, never by the base in n)",
        "",
        "RL's own rate: its success rate over all its attempts on those problems. Base bound: the",
        "95% upper bound on a per-attempt rate with 0 successes in n.",
        "",
        _row(
            "comparison",
            "n",
            "RL-only",
            "base-only",
            "RL's own rate: mean / median [IQR]",
            "≥ 0.5",
            "≥ 0.1",
            "base bound",
        ),
        _row(*["---"] * 8),
    ]
    for key, v in N["rl_only"].items():
        q = v["rl_rate_quartiles"] or [math.nan, math.nan]
        rate = (
            f"{_pct(v['rl_rate_on_rl_only_mean'])} / {_pct(v['rl_rate_on_rl_only_median'])} "
            f"[{_pct(q[0])}, {_pct(q[1])}]"
        )
        L.append(
            _row(
                key,
                v["attempts_matched"],
                f"{v['rl_only']} ({v['rl_only_pct']:.1f}%)",
                f"{v['base_only']} ({v['base_only_pct']:.1f}%)",
                rate,
                v["rl_only_with_rate_ge_0.5"],
                v["rl_only_with_rate_ge_0.1"],
                _pct(v["base_rate_upper95_if_0_of_n"]),
            )
        )
    L += ["", "## Matched sampled tokens per problem (`stats.paired_diff_at_tokens`)", ""]
    for key, v in N["matched_tokens"].items():
        tr = ", ".join(f"{x:.0f}" for x in v["tokens_per_attempt"]["rl"])
        tb = ", ".join(f"{x:.0f}" for x in v["tokens_per_attempt"]["base"])
        L += [
            f"**{key}** (sampled tokens per attempt: RL {tr}; base {tb})",
            "",
            _row("tokens per problem", "base", "RL", "gap [95%]", "p", "both in range"),
            _row(*["---"] * 6),
        ]
        for r in v["rows"]:
            rng = "yes" if r["in_range"] else "no (clamped)"
            L.append(
                _row(
                    f"{r['tokens']:,}",
                    f"{r['base']:.1f}",
                    f"{r['rl']:.1f}",
                    _ci(r["gap"], r["lo"], r["hi"]),
                    f"{r['p']:.3g}",
                    rng,
                )
            )
        L.append("")
    L += [
        "## F3 points: RL / Stage-1 FLOP ratio and gaps",
        "",
        _row(
            "point",
            "family",
            "set",
            "FLOP ratio",
            "sampled tokens",
            "trained tokens",
            "gap pass@1 [95%]",
            "k",
            "gap pass@k [95%]",
        ),
        _row(*["---"] * 9),
    ]
    for p in N["F3"]:
        t = p["tokens"]
        tok_s = "–" if math.isnan(t["sampled"]) else f"{t['sampled'] / 1e6:.1f}M"
        tok_t = "–" if math.isnan(t["trained"]) else f"{t['trained'] / 1e6:.1f}M"
        L.append(
            _row(
                p["label"],
                p["family"],
                p["set"],
                f"{p['x']:.3g}",
                tok_s,
                tok_t,
                _ci(p["gap1"], *p["ci1"]),
                p["k"],
                _ci(p["gapk"], *p["cik"]),
            )
        )
    L += [
        "",
        "## Stage-1 tokens",
        "",
        _row(
            "model",
            "epochs",
            "tokens (state + completion)",
            "completion tokens",
            "params",
            "train secs",
        ),
        _row(*["---"] * 6),
    ]
    for m, v in N["stage1"].items():
        L.append(
            _row(
                m,
                f"{v['epochs']:.3f}",
                f"{v['tokens'] / 1e6:.1f}M",
                f"{v['completion_tokens'] / 1e6:.1f}M",
                f"{v['n_params']:,}",
                f"{v['secs']:.0f}",
            )
        )
    if "F4" in N:
        L += ["", "## F4: Dan's EI data", ""]
        for lab, r in N["F4"]["a_support_curves"].items():
            vals = ", ".join(
                f"@{k} {v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"
                for k, v in r.items()
                if k != "solved_any"
            )
            L.append(
                f"- support-curves, {lab}: {vals}; solved at least once: {r['solved_any']} / 383"
            )
        for panel in ("b_trajectory_tb72", "c_trajectory_h250"):
            for name, r in N["F4"][panel].items():
                vals = ", ".join(f"@{k} {v:.1f}" for k, v in zip(r["k"], r["pass"], strict=True))
                L.append(f"- {panel}, {name} (3 seeds): {vals}")
    L += ["", "## F1 curves (pass@k at powers of two)", ""]
    for panel, res in N["F1"].items():
        L += [f"**{panel}**", ""]
        for name, r in res.items():
            if not isinstance(r, dict):
                L.append(f"- {name}: {r}")
                continue
            vals = ", ".join(f"@{k} {v:.1f}" for k, v in zip(r["k"], r["pass"], strict=True))
            L.append(f"- {name} ({r['S']} seed{'s' if r['S'] > 1 else ''}): {vals}")
        L.append("")
    path.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--figdir", type=Path, default=FIGDIR)
    ap.add_argument("--recount-stage1", action="store_true")
    ap.add_argument("--extra-f3", type=Path, default=None)
    ap.add_argument("--only", default="", help="comma list of f1,f2,f3,numbers (default: all)")
    a = ap.parse_args(argv)
    only = set(filter(None, a.only.split(",")))
    N: dict[str, Any] = {}
    s1 = stage1_tokens(a.recount_stage1)
    N["stage1"] = s1
    paths: list[Path] = []
    if not only or "f1" in only:
        paths += fig1(a.draws, a.figdir, N)
    if not only or "f2" in only:
        paths += fig2(a.draws, a.figdir, N)
    if not only or "f3" in only:
        pts = f3_points(a.draws, s1) + dan_f3_points(a.draws)
        if a.extra_f3 is not None:
            for e in json.loads(a.extra_f3.read_text()):
                pts.append(
                    Point(
                        e["label"],
                        e["family"],
                        e["set"],
                        float(e["x"]),
                        float(e["gap1"]),
                        float(e["gapk"]),
                        int(e.get("k", dict(F3_SETS)[e["set"]])),
                        meta={
                            "tokens": e.get("tokens", {"sampled": math.nan, "trained": math.nan})
                        },
                    )
                )
        paths += fig3(pts, a.figdir, N)
    if not only or "f4" in only:
        paths += fig4(a.draws, a.figdir, N)
    if not only or "numbers" in only:
        N["headline"] = headline(max(a.draws, 4000))
        N["rl_only"] = {
            "Charles149, L3 end s0 vs base s0": rl_only("L3 end", "charles149", 0, 64),
            "Charles149, L2 end s0 vs base s0": rl_only("L2 end", "charles149", 0, 64),
            "Charles149, ki78 end s0 vs base s0": rl_only("ki78 end", "charles149", 0, 64),
            "13–20, L3 end s0 vs base s0": rl_only("L3 end", "t1320", 0, 8),
            "21–30, L3 end s0 vs base s0": rl_only("L3 end", "t2130", 0, 8),
            "textbook72, L3 end s0 vs base s0 (B256)": rl_only(
                "L3 end", "tb72", 0, 64, base_path=v2_paths("base", "tb72", (0,))[0]
            ),
            "v1 textbook72, GRPO T1v1 s0 it78 vs T1v1 s0": rl_only(
                "",
                "tb72",
                0,
                64,
                base_path=V1["v1 base T1v1_s0"]["tb72"],
                rl_path=V1["v1 GRPO T1v1_s0 it78"]["tb72"],
            ),
        }
        N["matched_tokens"] = matched_tokens(max(a.draws, 4000))
    if not only:
        (HERE / "numbers.json").write_text(json.dumps(N, indent=1, default=float) + "\n")
        write_md(N, HERE / "numbers.md")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
