"""GRPO − EI by proof length: is the human-written / generated split a length or a source effect?

    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.length_split \
        [--draws 4000] [--v2 RUNS/eval/v2] [--out FIGDIR] [--table PATH]

Existing reads only (no sampling): the thinking models of GRPO recipe v2 read at B64 with the plain
arm, through `v2_results`'s loaders, sets and intervals. Two comparisons, each on every Stage-1
seed both arms have (rerun as reads land): GRPO + verifier (stop) − EI + verifier, and GRPO − EI.
Sets: textbook72 and Charles149 (human-written); the 986 valid 13–20 items and the 1,000-item
21–30 sample (generated, `cut_composite`).

**Proof length** is the ND line count of the set's reference proof, premise lines included, QED
not: the convention of `reference_lines`, checked here against every stored 13–20, 21–30 and
Charles149 reference proof. textbook72 stores `reference_lines` without proofs for its 58 dev
problems and nothing for its 14 train problems; those 14 take the shortest Lean-verified proof
that either headline arm found (all seeds, plain and guided reads; a read's `steps` less the QED
step, a convention checked on Charles149, whose references are search-minimal). That gives one
length per problem, shared by both arms of every comparison. A robustness variant uses the
shortest known proof everywhere (the reference or a shorter found proof).

**Intervals.** Per set and bin: `stats.ci` and `stats.paired_diff` (hierarchical bootstrap over
seeds and problems, seeds paired: both arms start from the same Stage-1 models). Pooled
human-written bins concatenate textbook72 and Charles149. The length-matched contrast
(human − generated GRPO − EI difference at equal length) resamples problems within each set and
seeds once for every set (one checkpoint per arm and seed reads every set), and reweights the
generated problems to the human-written per-line length mix of the range. The difficulty view
splits each read's attempts in two: difficulty is the mean pass rate of both arms on the even
attempts, the difference is read on the odd ones, so the two share no sampling noise; difficulty
still depends on both arms' training, so it is endogenous and only a secondary axis.

Writes `fig_length_split_<pair>`, `fig_length_split_delta_<pair>` (pair: `vs`, `plain`) and
`fig_length_split_difficulty` (PDF/SVG/PNG) to `--out`, and the tables to `--table`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from code.experiments.current.thinking_traces_20261005.matched_eval import v2_results as V
from code.ndbench.figures import style as S
from code.thinking import stats

F64 = NDArray[np.float64]
Bin = tuple[int, int]

TB = Path("/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72")
HUMAN, GENERATED = ("tb72", "charles149"), ("t1320", "t2130")
SETS = HUMAN + GENERATED
NAME = {
    "tb72": "textbook72",
    "charles149": "Charles149",
    "t1320": "13–20 lines",
    "t2130": "21–30 lines",
    "human": "human-written (textbook72 + Charles149)",
    "generated": "generated (13–20, 21–30)",
}
SOURCE = {s: "human" for s in HUMAN} | {s: "generated" for s in GENERATED}
# Source colours for the difference figures (no arm colour there), checked with the dataviz
# validator (adjacent CVD ΔE 14, contrast >= 3:1 on white); marker and line style differ too.
SRC_COL = {"human": "#9b3fa8", "generated": "#1a8a5a"}
SRC_MK = {"human": "o", "generated": "s"}
SET_MK = {"tb72": "^", "charles149": "D"}
OPEN = 99  # the open upper end of a last bin
ALL: Bin = (0, OPEN)
#: Bins per set (and for the pooled human-written series): every human-written bin holds >= 10
#: problems, every generated bin ~250.
BINS: dict[str, tuple[Bin, ...]] = {
    "tb72": ((1, 6), (7, 10), (11, 14), (15, OPEN)),
    "charles149": ((1, 6), (7, 9), (10, 12), (13, 15), (16, OPEN)),
    "t1320": ((13, 14), (15, 16), (17, 18), (19, 20)),
    "t2130": ((21, 22), (23, 24), (25, 26), (27, 30)),
    "human": ((1, 6), (7, 9), (10, 12), (13, 15), (16, 20), (21, OPEN)),
}
#: Line ranges of the length-matched human − generated contrast (`ALL`: unmatched, every length).
MATCH: tuple[Bin, ...] = ((13, 15), (16, 20), (13, 20), (21, 30), ALL)
#: Headline arms whose accepted proofs give the lengths textbook72's train problems lack.
FALLBACK = ("grpo_think_vs", "ei_think_v")
#: Difficulty bins (mean pass rate of both arms on the even attempts): exactly 0, (lo, hi], 1.
D_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 0.2),
    (0.2, 0.5),
    (0.5, 0.8),
    (0.8, 1.0),
    (1.0, 1.0),
)
XMAX = 33.5


@dataclass(frozen=True)
class Pair:
    key: str
    a: str  # GRPO arm
    b: str  # EI arm

    @property
    def la(self) -> str:
        return V.SPEC[self.a].label

    @property
    def lb(self) -> str:
        return V.SPEC[self.b].label

    @property
    def title(self) -> str:
        return f"{self.la} − {self.lb}" + (
            "" if V.SPEC[self.a].key.endswith("_vs") else " (no verifier)"
        )


PAIRS = (Pair("vs", "grpo_think_vs", "ei_think_v"), Pair("plain", "grpo_think", "ei_think"))


@dataclass(frozen=True)
class Length:
    lines: int
    ref: bool  # False: no reference proof; the shortest proof a headline arm found


@dataclass
class Cell:
    """One (set or source, bin): its problems and pass@1 / pass@5 of both arms and their difference
    (`a1`, `b1`, `d1`, `a5`, `b5`, `d5`)."""

    group: str
    bin: Bin
    n: int
    n_fallback: int
    x: float  # mean length (or difficulty) of the bin's problems
    seeds: list[int]
    est: dict[str, stats.Estimate]


@dataclass
class Matched:
    """GRPO − EI on human-written and on length-matched generated problems, and their gap."""

    bin: Bin
    k: int
    n_h: int
    n_g: int
    seeds: list[int]
    dh: stats.Estimate
    dg: stats.Estimate
    gap: stats.Estimate


@dataclass
class Result:
    pair: Pair
    cells: list[Cell] = field(default_factory=lambda: [])
    matched: list[Matched] = field(default_factory=lambda: [])
    robust: list[Matched] = field(default_factory=lambda: [])
    difficulty: list[Cell] = field(default_factory=lambda: [])
    notes: list[str] = field(default_factory=lambda: [])

    def series(self, group: str) -> list[Cell]:
        return sorted(
            (c for c in self.cells if c.group == group and c.bin != ALL), key=lambda c: c.bin
        )


# ---------------------------------------------------------------------------------- lengths


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def nd_lines(proof: str) -> int:
    """ND lines of `N1 ... : PR ; N2 ... ; QED` (premise lines included, QED not)."""
    return sum(bool(re.match(r"N\d+\s", part.strip())) for part in proof.split(";"))


def references(
    keep: dict[str, set[str] | None], sets: Sequence[str] = SETS
) -> dict[str, dict[str, int]]:
    """Reference length per problem of each of `sets` (only those files are read); the stored
    proofs are recounted and must give the stored `reference_lines`."""
    out: dict[str, dict[str, int]] = {}
    for s, path in (
        ("t1320", V.MATCHED / "sets/t1320_n1000.jsonl"),
        ("t2130", V.MATCHED / "sets/t2130_n1000.jsonl"),
        ("charles149", V.R / "pools/charles_1002/charles_clean.jsonl"),
    ):
        if s not in sets:
            continue
        ref: dict[str, int] = {}
        for r in jsonl(path):
            n = nd_lines(r["reference_proof"])
            if n != int(r["reference_lines"]):
                raise ValueError(
                    f"{path}: {r['name']}: {n} proof lines, {r['reference_lines']} stored"
                )
            names = keep[s]
            if names is None or r["name"] in names:
                ref[r["name"]] = n
        out[s] = ref
    if "tb72" in sets:
        out["tb72"] = {
            r["name"]: int(r["reference_lines"]) for r in jsonl(TB / "textbook_dev.jsonl")
        }
    return out


def shortest_found(ld: V.Loader, s: str, arms: Sequence[str] = FALLBACK) -> dict[str, int]:
    """Per problem of set `s`: the fewest ND lines of an accepted attempt in `arms`' reads (every
    seed, plain and guided). A read's `steps` counts accepted steps, the closing QED included."""
    best: dict[str, int] = {}
    for arm in arms:
        for ra in ("plain", "logical") if V.SET[s].guided else ("plain",):
            for seed in V.SEEDS:
                p = V.rows_path(V.SPEC[arm], V.SET[s], seed, ra, ld.v2)
                if p is None or not p.exists():
                    continue
                for r in jsonl(p):
                    for ok, st in zip(r["ok"], r["steps"], strict=True):
                        if ok:
                            best[r["name"]] = min(best.get(r["name"], st - 1), st - 1)
    return best


def lengths(
    refs: dict[str, dict[str, int]], found: dict[str, dict[str, int]], known: bool = False
) -> dict[str, dict[str, Length]]:
    """name -> length per set: the reference (`known`: or a shorter found proof), else for
    textbook72's train problems the shortest found proof (problems no arm solved get none)."""
    out: dict[str, dict[str, Length]] = {}
    for s in SETS:
        f = found[s]
        out[s] = {n: Length(min(v, f.get(n, v)) if known else v, True) for n, v in refs[s].items()}
    for r in jsonl(TB / "textbook_train.jsonl"):
        if r["name"] in found["tb72"]:
            out["tb72"][r["name"]] = Length(found["tb72"][r["name"]], False)
    return out


def length_checks(
    refs: dict[str, dict[str, int]],
    found: dict[str, dict[str, int]],
    L: dict[str, dict[str, Length]],
) -> list[str]:
    md = ["## Proof lengths", ""]
    md += [
        "| set | problems | reference length | shortest found proof (no reference) | no length |",
        "|---|---|---|---|---|",
    ]
    train = [r["name"] for r in jsonl(TB / "textbook_train.jsonl")]
    total = {"tb72": len(refs["tb72"]) + len(train)} | {s: len(refs[s]) for s in SETS[1:]}
    for s in SETS:
        n_ref = sum(x.ref for x in L[s].values())
        n_fb = len(L[s]) - n_ref
        md.append(f"| {NAME[s]} | {total[s]} | {n_ref} | {n_fb} | {total[s] - len(L[s])} |")
    md += [""]
    md += [
        "- The stored 13–20, 21–30 and Charles149 reference proofs, recounted (premise lines in,",
        "  QED out), give the stored `reference_lines` for every problem (986, 1,000, 149).",
    ]
    for s, what in (
        ("charles149", "search-minimal reference"),
        ("tb72", "stored reference (dev58, no proofs)"),
        ("t1320", "reference"),
        ("t2130", "reference"),
    ):
        d = Counter(found[s][n] - v for n, v in refs[s].items() if n in found[s])
        solved = sum(d.values())
        shorter = sum(v for k, v in d.items() if k < 0)
        longer = sum(v for k, v in d.items() if k > 0)
        md.append(
            f"- {NAME[s]}: shortest proof the headline arms found − {what}, over the {solved} "
            f"problems they solved: = 0 for {d[0]}, shorter for {shorter}, longer for {longer}."
        )
    fb = sorted(x.lines for x in L["tb72"].values() if not x.ref)
    md += [
        "- So `steps` − 1 counts ND lines as `reference_lines` does (Charles149: never below the",
        "  minimum, equal on most). The shorter found proofs mostly use Lean's ¬A = A → False or",
        "  beat a non-minimal stored reference (textbook72's dev references are not all minimal).",
        f"- textbook72 train14, shortest found lengths: {', '.join(map(str, fb))} lines.",
        "",
    ]
    return md


# ---------------------------------------------------------------------------------- estimates


def in_bin(lines: int, b: Bin) -> bool:
    return b[0] <= lines <= b[1]


def bin_label(b: Bin) -> str:
    if b == ALL:
        return "all"
    if b[1] == OPEN:
        return f"≥{b[0]}"
    if b[0] == 1:
        return f"≤{b[1]}"
    return f"{b[0]}–{b[1]}" if b[0] != b[1] else str(b[0])


def estimate(
    x: stats.Arm, y: stats.Arm, names: Sequence[str], draws: int, ks: Sequence[int]
) -> dict[str, stats.Estimate]:
    xs, ys = x.subset(names), y.subset(names)
    out: dict[str, stats.Estimate] = {}
    for k in ks:
        out[f"a{k}"] = stats.ci(xs, k, draws=draws)
        out[f"b{k}"] = stats.ci(ys, k, draws=draws)
        out[f"d{k}"] = stats.paired_diff(xs, ys, k, paired_seeds=True, draws=draws)
    return out


def concat(arms: Sequence[stats.Arm], tags: Sequence[str], label: str) -> stats.Arm:
    """One arm over the problems of several sets (names `set:name`), seed by seed."""
    if len({a.S for a in arms}) != 1:
        raise ValueError(f"{label}: seed counts differ")
    return stats.Arm(
        label,
        [f"{t}:{n}" for a, t in zip(arms, tags, strict=True) for n in a.names],
        [f for a in arms for f in a.files],
        np.hstack([a.n for a in arms]),
        np.hstack([a.c for a in arms]),
        None,
        [p for a in arms for p in a.paths],
        [None] * arms[0].S,
    )


def tagged(L: dict[str, dict[str, Length]], sets: Sequence[str]) -> dict[str, Length]:
    return {f"{s}:{n}": v for s in sets for n, v in L[s].items()}


def seeds_of(ld: V.Loader, pair: Pair, sets: Sequence[str]) -> list[int]:
    have = [set(V.common_seeds(ld, [pair.a, pair.b], s)) for s in sets]
    return sorted(set.intersection(*have)) if have else []


def bins_of(
    x: stats.Arm,
    y: stats.Arm,
    L: dict[str, Length],
    group: str,
    seeds: list[int],
    draws: int,
) -> list[Cell]:
    out = []
    for b in (*BINS[group], ALL):
        names = (
            list(x.names) if b == ALL else [n for n in x.names if n in L and in_bin(L[n].lines, b)]
        )
        if not names:
            continue
        lens = [L[n].lines for n in names if n in L]
        out.append(
            Cell(
                group,
                b,
                len(names),
                sum(not L[n].ref for n in names if n in L),
                float(np.mean(lens)),
                seeds,
                estimate(x, y, names, draws, (1, 5)),
            )
        )
    return out


def by_length(
    ld: V.Loader, pair: Pair, L: dict[str, dict[str, Length]], res: Result, draws: int
) -> None:
    have: dict[str, list[int]] = {}
    for s in SETS:
        seeds = V.common_seeds(ld, [pair.a, pair.b], s)
        x, y = ld.arm(pair.a, s, seeds), ld.arm(pair.b, s, seeds)
        if x is None or y is None:
            res.notes.append(f"{NAME[s]}: no seed with both arms read yet.")
            continue
        have[s] = seeds
        res.cells += bins_of(x, y, L[s], s, seeds, draws)
    hs = [s for s in HUMAN if s in have]
    seeds = sorted(set.intersection(*(set(have[s]) for s in hs))) if hs else []
    if seeds:
        xs = [ld.arm(pair.a, s, seeds) for s in hs]
        ys = [ld.arm(pair.b, s, seeds) for s in hs]
        if all(a is not None for a in xs + ys):
            X = concat([a for a in xs if a is not None], hs, "human a")
            Y = concat([a for a in ys if a is not None], hs, "human b")
            res.cells += bins_of(X, Y, tagged(L, hs), "human", seeds, draws)


def _estimate(label: str, est: float, d: F64, k: int, T: int) -> stats.Estimate:
    lo, hi = np.quantile(d, [0.025, 0.975])
    return stats.Estimate(
        label,
        est,
        float(lo),
        float(hi),
        float(np.std(d, ddof=1)),
        k=k,
        T=T,
        p_boot=stats.boot_p(d),
        draws=len(d),
        samples=d,
    )


def matched(
    ld: V.Loader, pair: Pair, L: dict[str, dict[str, Length]], draws: int, seed: int = 0
) -> list[Matched]:
    """Human-written − generated GRPO − EI difference at equal length, for each range of `MATCH`:
    generated problems weighted to the human-written per-line mix (lines without generated
    problems dropped on both sides). Problems are resampled within each set, seeds once for all."""
    sets = [s for s in SETS if V.common_seeds(ld, [pair.a, pair.b], s)]
    if not any(s in HUMAN for s in sets) or not any(s in GENERATED for s in sets):
        return []
    seeds = seeds_of(ld, pair, sets)
    if not seeds:
        return []
    xs = [ld.arm(pair.a, s, seeds) for s in sets]
    ys = [ld.arm(pair.b, s, seeds) for s in sets]
    if any(a is None for a in xs + ys):
        return []
    X = concat([a for a in xs if a is not None], sets, "a")
    Y = concat([a for a in ys if a is not None], sets, "b")
    W = np.hstack(
        [
            stats.make_draws(a.T, draws=draws, seed=seed + j).W
            for j, a in enumerate(a for a in xs if a is not None)
        ]
    )
    dr = stats.Draws(W, np.random.default_rng([seed, 1]), "both")
    U = dr.seed_weights(X.S)
    Lt = tagged(L, sets)
    lines = np.array([Lt[n].lines if n in Lt else -1 for n in X.names])
    human = np.array([SOURCE[n.split(":")[0]] == "human" for n in X.names])
    out = []
    for k in (1, 5):
        D = X.values(k) - Y.values(k)
        UD = U @ D

        def gmean(w: F64, D: F64 = D, UD: F64 = UD) -> tuple[float, F64]:
            ww = W * w
            return float((D.mean(0) * w).sum() / w.sum()), (UD * ww).sum(1) / (X.S * ww.sum(1))

        for b in MATCH:
            inr = (lines >= max(b[0], 1)) & (lines <= b[1])
            h, g = inr & human, inr & ~human
            if b == ALL:
                wh, wg = h.astype(float), g.astype(float)
            else:
                hc, gc = Counter(lines[h].tolist()), Counter(lines[g].tolist())
                wh = np.array([float(hh and gc[ln] > 0) for hh, ln in zip(h, lines, strict=True)])
                wg = np.array(
                    [hc[ln] / gc[ln] if gg else 0.0 for gg, ln in zip(g, lines, strict=True)]
                )
            if wh.sum() == 0 or wg.sum() == 0:
                continue
            (eh, sh), (eg, sg) = gmean(wh), gmean(wg)
            out.append(
                Matched(
                    b,
                    k,
                    int((wh > 0).sum()),
                    int((wg > 0).sum()),
                    seeds,
                    _estimate("human", eh, sh, k, int((wh > 0).sum())),
                    _estimate("generated", eg, sg, k, int((wg > 0).sum())),
                    _estimate("human − generated", eh - eg, sh - sg, k, int(inr.sum())),
                )
            )
    return out


def half_arm(ld: V.Loader, arm: str, s: str, seeds: Sequence[int], parity: int) -> stats.Arm:
    """The arm on the even (`parity` 0) or odd (1) attempts of each read (set `s`)."""
    keep = ld.keep[s]
    per: list[dict[str, tuple[str, list[int]]]] = []
    paths = []
    for seed in seeds:
        p = V.rows_path(V.SPEC[arm], V.SET[s], seed, "plain", ld.v2)
        assert p is not None
        paths.append(str(p))
        per.append(
            {
                r["name"]: (r["file"], r["ok"][parity::2])
                for r in jsonl(p)
                if keep is None or r["name"] in keep
            }
        )
    names = sorted(per[0])
    return stats.Arm(
        f"{arm} {s} half {parity}",
        names,
        [per[0][n][0] for n in names],
        np.array([[len(by[n][1]) for n in names] for by in per], dtype=np.int64),
        np.array([[sum(by[n][1]) for n in names] for by in per], dtype=np.int64),
        None,
        paths,
        [None] * len(seeds),
    )


def by_difficulty(ld: V.Loader, pair: Pair, draws: int) -> list[Cell]:
    """GRPO − EI pass@1 on the odd attempts, binned by both arms' mean pass rate on the even ones,
    per source (sets pooled at the seeds they share)."""
    out = []
    for src, group in (("human", HUMAN), ("generated", GENERATED)):
        sets = [s for s in group if V.common_seeds(ld, [pair.a, pair.b], s)]
        seeds = seeds_of(ld, pair, sets)
        if not sets or not seeds:
            continue
        halves = {
            (arm, par): concat([half_arm(ld, arm, s, seeds, par) for s in sets], sets, arm)
            for arm in (pair.a, pair.b)
            for par in (0, 1)
        }
        ea, eb = halves[(pair.a, 0)], halves[(pair.b, 0)]
        diff = (ea.c / ea.n + eb.c / eb.n).mean(0) / 2
        xa, xb = halves[(pair.a, 1)], halves[(pair.b, 1)]
        for lo, hi in D_BINS:
            m = (diff == lo) if lo == hi else (diff > lo) & (diff <= hi) & (diff < 1)
            names = [n for n, ok in zip(xa.names, m, strict=True) if ok]
            if not names:
                continue
            out.append(
                Cell(
                    src,
                    (int(100 * lo), int(100 * hi)),
                    len(names),
                    0,
                    float(diff[m].mean()),
                    seeds,
                    estimate(xa, xb, names, draws, (1,)),
                )
            )
    return out


# ------------------------------------------------------------------------------------ figures


def _band(ax: Axes, cells: Sequence[Cell], key: str, col: str, mk: str, **kw: Any) -> None:
    if not cells:
        return
    x = [c.x for c in cells]
    ax.fill_between(
        x,
        [c.est[key].lo for c in cells],
        [c.est[key].hi for c in cells],
        color=col,
        alpha=S.BAND_ALPHA * 1.6,
        lw=0,
        zorder=1,
    )
    ax.plot(
        x,
        [c.est[key].est for c in cells],
        color=col,
        marker=mk,
        ms=3.4,
        mfc=col,
        mec=S.WHITE,
        mew=0.5,
        zorder=3,
        **kw,
    )


def _len_axis(ax: Axes) -> None:
    ax.set_xlim(0.5, XMAX)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    S.light_grid(ax, "y")


def _points(v: float, _pos: float | None = None) -> str:
    return f"{100 * v:+.0f}".replace("-", "−") if round(100 * v) else "0"


def _seeds_note(res: Result, sets: Sequence[str]) -> str:
    per = {s: res.series(s)[0].seeds for s in sets if res.series(s)}
    if not per:
        return "no reads yet"
    if len({tuple(v) for v in per.values()}) == 1:
        sd = next(iter(per.values()))
        return f"seeds {','.join(map(str, sd))} pooled" if len(sd) > 1 else f"seed {sd[0]} only"
    return "seeds: " + "; ".join(f"{NAME[s]} {','.join(map(str, v))}" for s, v in per.items())


def _counts(ax: Axes, L: dict[str, dict[str, Length]], sets: Sequence[str]) -> None:
    light = S.mix(S.INK2, S.WHITE, 0.6)
    for s in sets:
        c = Counter(min(x.lines, int(XMAX)) for x in L[s].values())
        fb = Counter(min(x.lines, int(XMAX)) for x in L[s].values() if not x.ref)
        xs = sorted(c)
        ax.bar(xs, [c[x] - fb[x] for x in xs], width=0.8, color=S.INK2, lw=0)
        ax.bar(xs, [fb[x] for x in xs], bottom=[c[x] - fb[x] for x in xs], width=0.8, color=light)
    edges = [b[0] - 0.5 for b in BINS[sets[0]][1:]] if len(sets) == 1 else [20.5]
    for e in edges:
        ax.axvline(e, color=S.REF, lw=0.4, ls=(0, (1.5, 1.5)), zorder=0)


def fig_main(res: Result, L: dict[str, dict[str, Length]], out: Path) -> None:
    pair = res.pair
    cols: list[tuple[str, ...]] = [("tb72",), ("charles149",), GENERATED]
    titles = ["textbook72", "Charles149", "generated"]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 4.4, layout="constrained")
        axs = fig.subplots(3, 3, sharex=True, squeeze=False, height_ratios=[1.0, 1.0, 0.55])
        for r in range(2):
            for j in range(1, 3):
                axs[r][j].sharey(axs[r][0])
                axs[r][j].tick_params(labelleft=False)
        for j, sets in enumerate(cols):
            for r, k in enumerate((1, 5)):
                ax: Axes = axs[r][j]
                for key, col, mk in ((f"a{k}", V.BLUE, "o"), (f"b{k}", V.ORANGE, "s")):
                    for s in sets:
                        cells = res.series(s)
                        _band(ax, cells, key, col, mk)
                for s in sets:  # a set with no read of both arms yet
                    if not res.series(s):
                        lo, hi = BINS[s][0][0], min(BINS[s][-1][1], int(XMAX))
                        x0 = (lo + hi) / 2 if len(sets) > 1 else 0.5 * (1 + XMAX)
                        ax.text(
                            x0,
                            0.5,
                            "reads\npending",
                            ha="center",
                            va="center",
                            fontsize=S.FS_SMALL,
                            color=S.REF,
                        )
                if len(sets) > 1:
                    ax.axvline(20.5, color=S.REF, lw=0.4, ls=(0, (1.5, 1.5)), zorder=0)
                    for x0, lab in ((16.5, "13–20"), (25.5, "21–30")):
                        ax.text(
                            x0, 0.97, lab, ha="center", va="top", fontsize=S.FS_SMALL, color=S.REF
                        )
                S.panel_title(ax, "abcdef"[3 * r + j], f"{titles[j]}, pass@{k}", size=S.FS_TICK)
                ax.set_ylim(0, 1)
                S.pct_axis(ax, nbins=5)
                _len_axis(ax)
            ax = axs[2][j]
            _counts(ax, L, sets)
            S.panel_title(ax, "ghi"[j], "problems per line", size=S.FS_TICK)
            _len_axis(ax)
        axs[0][0].set_ylabel("pass@1")
        axs[1][0].set_ylabel("pass@5")
        axs[2][0].set_ylabel("problems")
        fig.supxlabel(
            "proof length (ND lines of the reference proof); points at each bin's mean length, "
            "dotted lines at bin edges\nlight bars: textbook72 problems without a reference "
            "(length of the shortest proof either headline arm found)",
            fontsize=S.FS_TICK,
        )
        handles = [
            Line2D([], [], color=V.BLUE, marker="o", ms=4, mec=S.WHITE, label=pair.la),
            Line2D([], [], color=V.ORANGE, marker="s", ms=4, mec=S.WHITE, label=pair.lb),
            Line2D(
                [],
                [],
                color=S.REF,
                lw=5,
                alpha=0.3,
                label=f"95% interval ({_seeds_note(res, SETS)})",
            ),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        S.save(fig, out / f"fig_length_split_{pair.key}")
        S.close(fig)


def fig_delta(res: Result, L: dict[str, dict[str, Length]], out: Path) -> None:
    pair = res.pair
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 5.2, layout="constrained")
        axs = fig.subplots(3, 1, sharex=True, squeeze=False, height_ratios=[1.25, 1.0, 0.7])[:, 0]
        for r, k in enumerate((1, 5)):
            ax: Axes = axs[r]
            ax.axvspan(12.5, 30.5, color=S.SHADE, lw=0, zorder=0)
            ax.axhline(0, color=S.INK, lw=0.6, zorder=2)
            key = f"d{k}"
            human = res.series("human")
            gen = [c for s in GENERATED for c in res.series(s)]
            _band(ax, human, key, SRC_COL["human"], SRC_MK["human"])
            for s in GENERATED:
                _band(ax, res.series(s), key, SRC_COL["generated"], SRC_MK["generated"])
            for s in HUMAN:
                cells = res.series(s)
                ax.plot(
                    [c.x for c in cells],
                    [c.est[key].est for c in cells],
                    ls=(0, (1, 1.4)),
                    lw=0.7,
                    color=SRC_COL["human"],
                    marker=SET_MK[s],
                    ms=3.2,
                    mfc=S.WHITE,
                    mec=SRC_COL["human"],
                    mew=0.7,
                    zorder=2,
                )
            lab = "first try" if k == 1 else "within 5 tries"
            S.panel_title(ax, "ab"[r], f"{pair.title}: Δ pass@{k} ({lab})", size=S.FS_TICK)
            ax.set_ylabel(f"Δ pass@{k} (points)")
            ax.yaxis.set_major_formatter(_points)
            _len_axis(ax)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - (0.16 if r == 0 else 0.0) * (hi - lo), hi)
            ax.text(
                30.3,
                hi,
                "lengths both sources cover",
                ha="right",
                va="top",
                fontsize=S.FS_SMALL,
                color=S.REF,
            )
            if r == 0:  # problems per bin, one row per source
                for cells, y, src in ((human, 0.035, "human"), (gen, 0.105, "generated")):
                    for c in cells:
                        ax.text(
                            c.x,
                            y,
                            str(c.n),
                            transform=ax.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            fontsize=S.FS_SMALL,
                            color=SRC_COL[src],
                        )
                ax.text(
                    0.7,
                    0.07,
                    "n:",
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="bottom",
                    fontsize=S.FS_SMALL,
                    color=S.REF,
                )
        ax = axs[2]
        ax.axvspan(12.5, 30.5, color=S.SHADE, lw=0, zorder=0)
        ax.plot([9.5, 15.5], [300, 300], color=S.INK2, lw=1.2, solid_capstyle="butt")
        ax.text(9.3, 300, "RL curriculum lengths", ha="right", va="center", fontsize=S.FS_SMALL)
        styles = {"tb72": (0, (2.5, 1.2)), "charles149": "-", "t1320": "-", "t2130": "-"}
        for s, ls in styles.items():
            c = Counter(min(x.lines, int(XMAX)) for x in L[s].values())
            xs = np.arange(1, int(XMAX) + 1)
            ys = np.array([c.get(int(x), 0) for x in xs], dtype=float)
            ys[ys == 0] = np.nan
            ax.step(xs, ys, where="mid", color=SRC_COL[SOURCE[s]], lw=1.0, ls=ls)
        fb = sorted(x.lines for x in L["tb72"].values() if not x.ref)
        ax.plot(fb, [0.75] * len(fb), ls="none", marker="|", ms=4, color=SRC_COL["human"], mew=0.6)
        ax.set_yscale("log")
        ax.set_ylim(0.6, 4000)
        ax.set_yticks([1, 10, 100], ["1", "10", "100"])
        S.panel_title(ax, "c", "problems at each length (log scale)", size=S.FS_TICK)
        ax.set_ylabel("problems")
        _len_axis(ax)
        ax.legend(
            handles=[
                Line2D([], [], color=SRC_COL["human"], ls=(0, (2.5, 1.2)), label="textbook72"),
                Line2D([], [], color=SRC_COL["human"], label="Charles149"),
                Line2D([], [], color=SRC_COL["generated"], label="13–20, 21–30"),
                Line2D(
                    [],
                    [],
                    color=SRC_COL["human"],
                    marker="|",
                    ls="none",
                    ms=4,
                    mew=0.6,
                    label="textbook72, no reference",
                ),
            ],
            loc="upper right",
            ncol=4,
            fontsize=S.FS_SMALL,
            handlelength=1.6,
            columnspacing=0.9,
        )
        ax.set_xlabel(
            "proof length (ND lines of the reference proof; points at each bin's mean length)"
        )
        handles = [
            Line2D(
                [], [], color=SRC_COL["human"], marker="o", ms=4, mec=S.WHITE, label=NAME["human"]
            ),
            Line2D(
                [],
                [],
                color=SRC_COL["generated"],
                marker="s",
                ms=4,
                mec=S.WHITE,
                label=NAME["generated"],
            ),
            Line2D(
                [],
                [],
                color=SRC_COL["human"],
                marker="^",
                ms=3.4,
                mfc=S.WHITE,
                ls=(0, (1, 1.4)),
                lw=0.7,
                label="textbook72 alone (no band)",
            ),
            Line2D(
                [],
                [],
                color=SRC_COL["human"],
                marker="D",
                ms=3.2,
                mfc=S.WHITE,
                ls=(0, (1, 1.4)),
                lw=0.7,
                label="Charles149 alone (no band)",
            ),
            Line2D([], [], color=S.REF, lw=5, alpha=0.3, label="95% interval"),
            Line2D([], [], ls="none", label=f"({_seeds_note(res, SETS)})"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        S.save(fig, out / f"fig_length_split_delta_{pair.key}")
        S.close(fig)


def fig_difficulty(results: Sequence[Result], out: Path, min_n: int = 5) -> None:
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.4, layout="constrained")
        axs = fig.subplots(1, len(results), sharey=True, squeeze=False)[0]
        for j, res in enumerate(results):
            ax: Axes = axs[j]
            ax.axhline(0, color=S.INK, lw=0.6, zorder=2)
            for src in ("human", "generated"):
                cells = sorted(
                    (c for c in res.difficulty if c.group == src and c.n >= min_n),
                    key=lambda c: c.x,
                )
                _band(ax, cells, "d1", SRC_COL[src], SRC_MK[src])
            seeds = sorted({sd for c in res.difficulty for sd in c.seeds})
            S.panel_title(
                ax,
                "ab"[j],
                f"{res.pair.title}, seed{'s' if len(seeds) > 1 else ''} "
                f"{','.join(map(str, seeds))}",
                size=S.FS_TICK,
            )
            ax.set_xlim(-0.03, 1.03)
            S.pct_axis(ax, "x", nbins=5)
            S.light_grid(ax, "y")
        axs[0].set_ylabel("Δ pass@1, odd attempts (points)")
        axs[0].yaxis.set_major_formatter(_points)
        fig.supxlabel(
            "difficulty: both arms' mean pass@1 on the even attempts (bins: 0, ≤20%, ≤50%, ≤80%, "
            f"<100%, 100%; bins under {min_n} problems not drawn)",
            fontsize=S.FS_TICK,
        )
        handles = [
            Line2D([], [], color=SRC_COL[s], marker=SRC_MK[s], ms=4, mec=S.WHITE, label=NAME[s])
            for s in ("human", "generated")
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
        S.save(fig, out / "fig_length_split_difficulty")
        S.close(fig)


# ------------------------------------------------------------------------------------- tables


def _cell_row(c: Cell, ks: Sequence[int]) -> str:
    n = f"{c.n} ({c.n_fallback}†)" if c.n_fallback else str(c.n)
    cells = []
    for k in ks:
        cells += [V.fmt(c.est[f"a{k}"]), V.fmt(c.est[f"b{k}"]), V.fmt(c.est[f"d{k}"], True)]
    return (
        f"| {NAME[c.group]} | {bin_label(c.bin)} | {n} | {','.join(map(str, c.seeds))} | "
        + " | ".join(cells)
        + " |"
    )


def _matched_rows(ms: Sequence[Matched]) -> list[str]:
    md = [
        "| lines | human n | generated n | seeds | k | Δ human-written | Δ generated "
        "(length-matched) | human − generated | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(ms, key=lambda m: (m.k, MATCH.index(m.bin))):
        lab = "all (unmatched)" if m.bin == ALL else bin_label(m.bin)
        p = m.gap.p_boot
        md.append(
            f"| {lab} | {m.n_h} | {m.n_g} | {','.join(map(str, m.seeds))} | pass@{m.k} | "
            f"{V.fmt(m.dh, True)} | {V.fmt(m.dg, True)} | {V.fmt(m.gap, True)} | "
            f"{'—' if p is None else f'{p:.3f}'} |"
        )
    return md


def tables(results: Sequence[Result], checks: list[str], ld: V.Loader, draws: int) -> list[str]:
    md = [
        "# GRPO − EI by proof length (generated by `length_split.py`; do not edit by hand)",
        "",
        "Thinking models, GRPO recipe v2, plain reads at B64 (ynez). pass@k in % of the bin, 95%",
        f"hierarchical-bootstrap intervals over seeds and problems ({draws} draws), seeds paired",
        "(both arms start from the same Stage-1 models). Δ = GRPO arm − EI arm, in points.",
        "Figures: `fig_length_split_*` in",
        "`experiment-summaries/2026-10-05-thinking-traces/figures/ynez/`.",
        "",
    ]
    md += checks
    md += [
        "## Bins",
        "",
        "| set | bins (lines) |",
        "|---|---|",
    ]
    md += [f"| {NAME[s]} | {', '.join(bin_label(b) for b in BINS[s])} |" for s in BINS]
    md += ["", "† problems whose length is the shortest found proof (textbook72 train14).", ""]
    for i, res in enumerate(results, 1):
        p = res.pair
        md += [f"## {i}. {p.title}", ""]
        md += [f"### {i}a. Per set and length bin (pooled human-written rows last)", ""]
        md += [
            f"| set | lines | n | seeds | {p.la} pass@1 | {p.lb} pass@1 | Δ pass@1 | "
            f"{p.la} pass@5 | {p.lb} pass@5 | Δ pass@5 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g in (*SETS, "human"):
            for c in sorted((c for c in res.cells if c.group == g), key=lambda c: c.bin == ALL):
                md.append(_cell_row(c, (1, 5)))
        md += ["", f"### {i}b. Human-written − generated at equal length", ""]
        md += [
            "Generated problems reweighted to the human-written per-line mix of the range",
            "(problems resampled within each set, seeds once for every set). ≈ 0: length explains",
            "the split; > 0: human-written problems gain more from GRPO at the same length.",
            "",
        ]
        md += _matched_rows(res.matched) if res.matched else ["(not available yet)"]
        md += [
            "",
            f"### {i}c. Robustness: the same with the shortest known proof as the length",
            "",
            "(the reference, or a shorter Lean-verified proof either headline arm found)",
            "",
        ]
        md += _matched_rows(res.robust) if res.robust else ["(not available yet)"]
        md += ["", f"### {i}d. Secondary: by difficulty (endogenous; cross-fit halves)", ""]
        md += [
            "Difficulty = both arms' mean pass@1 on the even attempts (seeds pooled); pass@1 and Δ",
            "on the odd attempts. Difficulty is itself a product of the two arms' training, so",
            "this conditions on an outcome; read it as descriptive, not causal.",
            "",
            f"| source | difficulty | n | seeds | mean difficulty | {p.la} pass@1 | "
            f"{p.lb} pass@1 | Δ pass@1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in res.difficulty:
            lo, hi = c.bin
            lab = f"{lo}%" if lo == hi else f"({lo}%, {hi}%" + (")" if hi == 100 else "]")
            md.append(
                f"| {NAME[c.group]} | {lab} | {c.n} | {','.join(map(str, c.seeds))} | "
                f"{100 * c.x:.0f}% | {V.fmt(c.est['a1'])} | {V.fmt(c.est['b1'])} | "
                f"{V.fmt(c.est['d1'], True)} |"
            )
        if res.notes:
            md += ["", *[f"- {n}" for n in res.notes]]
        md += [""]
    if ld.missing:
        runs = Counter(V._run_of(p) for p in set(ld.missing))
        md += [f"Rows not there yet ({sum(runs.values())} files; run: files):", ""]
        md += [", ".join(f"`{r}`: {n}" for r, n in sorted(runs.items())), ""]
    return md


# --------------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=stats.DRAWS)
    ap.add_argument("--v2", default=str(V.E / "v2"), help="the v2 reads")
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures/ynez")
    ap.add_argument(
        "--table",
        default="code/experiments/current/thinking_traces_20261005/matched_eval/length_split_ynez.md",
    )
    a = ap.parse_args(argv)
    ld = V.Loader(Path(a.v2), V.subsets())
    refs = references(ld.keep)
    found = {s: shortest_found(ld, s) for s in SETS}
    L, Lk = lengths(refs, found), lengths(refs, found, known=True)
    out = Path(a.out)
    results = []
    for pair in PAIRS:
        res = Result(pair)
        by_length(ld, pair, L, res, a.draws)
        res.matched = matched(ld, pair, L, a.draws)
        res.robust = matched(ld, pair, Lk, a.draws)
        res.difficulty = by_difficulty(ld, pair, a.draws)
        fig_main(res, L, out)
        fig_delta(res, L, out)
        results.append(res)
        print(f"{pair.title}: {len(res.cells)} cells, {len(res.matched)} matched contrasts")
    fig_difficulty(results, out)
    md = tables(results, length_checks(refs, found, L), ld, a.draws)
    Path(a.table).write_text("\n".join(md) + "\n")
    print(f"wrote {a.table} and {out}/fig_length_split_*")


if __name__ == "__main__":
    main()
