"""Uncertainty for the pass@k reads of the thinking-trace experiment.

    python -m code.thinking.stats table --group "T0=DIR_s0,DIR_s1,DIR_s2" \
        --group "T1 B256=DIR_s0/rows_plain_B256.jsonl,..." --k 1,5 --tokens 1000,3000,10000 \
        --out FILE.md [--ref T0] [--pair "T1 B256 vs T1 B0"] [--exclude textbook_dev] \
        [--clusters SET.jsonl,... --cluster-by family,knob] [--draws 4000] [--seed 0] [--json F]

**Arms.** An arm is one recipe read one way (e.g. T1 at think budget B 256) with one source per
training seed; sources are anything `figures.load_source` reads with per-theorem rows (a
`code.thinking.eval` directory with one cell, or one `rows_<cell>.jsonl` of it).  Its pass@k is the
mean over seeds and theorems of the per-theorem unbiased estimate 1 - C(n-c, k) / C(n, k)
(Chen et al. 2021) from n attempts with c accepted.

**Bootstrap.** Every draw resamples seeds with replacement, independently per arm (different arms
are different models), and theorems with replacement, with one theorem resample shared by every
arm and seed, so that arm differences are paired on theorems.  This is Owen's pigeonhole bootstrap
for a crossed seed x theorem design; it slightly over-counts the seed x theorem residual, so it is
conservative in that term.  Two arms read from the same checkpoints (T1 at B 0 and B 256) also share
the seed resample (`paired_seeds`, detected from `ckpt` in the eval directory's summary.json).
Optional clusters, such as generated (family, knob) cells, are resampled as units instead of single
theorems.  Intervals are percentiles; the bootstrap p of a difference is twice the smaller tail
mass at 0.

**Few seeds.** With S = 3 the seed level has 10 distinct resamples, and the bootstrap variance of a
mean understates the seed variance by (S - 1) / S.  `components` gives the method-of-moments
decomposition of a crossed random-effects model, Var = sum(sigma2_seed / S) + Var_t(d_t) / T,
with a Satterthwaite t interval as a cross-check; it is also the basis of `mde`,
`seeds_needed` and `theorems_needed`.

**Matched tokens.** pass@k at X sampled tokens per theorem (think + `</think>` + action + rejected
redraws, prompt prefill excluded, as in `figures.tokens_data`).  For each seed,
k* = X / (mean sampled tokens per attempt over the set), and pass@k* is interpolated linearly in
log k between integer k (log k and log tokens differ by a constant); `grid="pow2"` interpolates
between k = 1, 2, 4, ... as the tokens figure draws it.  The arm value is the mean over seeds.
Tokens per attempt are recomputed on every theorem resample.  When k* leaves [1, n], it is clamped
(to pass@1 or pass@n) and the share of clamped draws is reported.

This module never imports torch or the vendored fork.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .figures import Group, Source, load_source, parse_group

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
Bool = NDArray[np.bool_]

DRAWS = 4000
LEVEL = 0.95
LEVELS = ("both", "theorems", "seeds")
GRIDS = ("all", "pow2")

# ------------------------------------------------------------------------------------------ arms


@dataclass
class Arm:
    """One arm: S seeds x T theorems (`names` sorted).  `tokens[s, t]` is the sum of sampled tokens
    over the attempts of theorem t in seed s (None if the source has no per-attempt counts)."""

    label: str
    names: list[str]
    files: list[str]
    n: I64
    c: I64
    tokens: F64 | None
    paths: list[str]
    models: list[str | None]
    _curve: F64 | None = field(default=None, repr=False)

    @property
    def S(self) -> int:
        return int(self.n.shape[0])

    @property
    def T(self) -> int:
        return int(self.n.shape[1])

    @property
    def kmax(self) -> int:
        """The largest k every theorem of every seed supports (the read's k)."""
        return int(self.n.min())

    def curve(self) -> F64:
        """S x T x kmax: unbiased pass@k for k = 1..kmax."""
        if self._curve is None:
            self._curve = passk_curve(self.n, self.c, self.kmax)
        return self._curve

    def values(self, k: int | None) -> F64:
        """S x T per-theorem pass@k (k None: every attempt, i.e. pass@kmax)."""
        return self.curve()[:, :, _k_index(self, k)]

    def tokens_per_attempt(self) -> F64:
        """Per seed: sampled tokens per attempt over the arm's theorems."""
        if self.tokens is None:
            raise ValueError(f"{self.label}: per-attempt token counts missing")
        return self.tokens.sum(1) / self.n.sum(1)

    def subset(self, names: Sequence[str], label: str | None = None) -> Arm:
        """The same arm restricted to `names` (kept in this arm's order)."""
        want = set(names)
        idx = [i for i, x in enumerate(self.names) if x in want]
        missing = want - set(self.names)
        if missing:
            raise ValueError(f"{self.label}: no theorems {sorted(missing)[:3]}")
        return Arm(
            self.label if label is None else label,
            [self.names[i] for i in idx],
            [self.files[i] for i in idx],
            self.n[:, idx],
            self.c[:, idx],
            None if self.tokens is None else self.tokens[:, idx],
            self.paths,
            self.models,
        )


def passk_curve(n: I64, c: I64, K: int) -> F64:
    """`[..., k - 1]` = unbiased pass@k for k = 1..K, elementwise over n >= K attempts and c
    successes: 1 - prod_{i<k} (n - c - i) / (n - i)."""
    if K < 1 or bool((n < K).any()):
        raise ValueError(f"pass@k curve up to {K} needs at least {K} attempts everywhere")
    i = np.arange(K)
    num = np.clip((n - c)[..., None] - i, 0, None).astype(np.float64)
    den = (n[..., None] - i).astype(np.float64)
    return np.asarray(1.0 - np.cumprod(num / den, axis=-1), dtype=np.float64)


def _k_index(arm: Arm, k: int | None) -> int:
    K = arm.kmax
    if k is None:
        return K - 1
    if not 1 <= k <= K:
        raise ValueError(f"{arm.label}: pass@{k} needs 1 <= k <= {K} (the read's attempts)")
    return k - 1


def _model_of(src: Source) -> str | None:
    """The checkpoint a `code.thinking.eval` read came from (summary.json / args.json `ckpt`)."""
    if src.kind != "thinking.eval":
        return None
    d = Path(src.path).parent
    for name in ("summary.json", "args.json"):
        f = d / name
        if f.exists():
            try:
                ck = json.loads(f.read_text()).get("ckpt")
            except (OSError, ValueError):
                continue
            if ck:
                return str(ck)
    return None


def load_arm(
    label: str,
    paths: Sequence[str | Path],
    *,
    cell: str | None = None,
    sets: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> Arm:
    """One arm from one source per seed.  `sets` keeps only those files (sets), `exclude` drops
    files.  Every seed must cover the same theorems."""
    if not paths:
        raise ValueError(f"{label}: no sources")
    srcs: list[Source] = []
    per: list[dict[str, Any]] = []
    for p in paths:
        src = load_source(p, cell)
        if not src.thms:
            raise ValueError(
                f"{src.path}: no per-theorem rows (a by_len summary cannot be resampled)"
            )
        by: dict[str, Any] = {}
        for t in src.thms:
            if (sets is not None and t.file not in sets) or t.file in exclude:
                continue
            if t.name in by:
                raise ValueError(f"{src.path}: theorem {t.name} appears twice")
            by[t.name] = t
        if not by:
            raise ValueError(f"{src.path}: no theorems left after the set filter")
        srcs.append(src)
        per.append(by)
    names = sorted(per[0])
    for src, by in zip(srcs[1:], per[1:], strict=True):
        if sorted(by) != names:
            diff = sorted(set(by) ^ set(names))
            raise ValueError(f"{label}: {src.path} covers other theorems, e.g. {diff[:3]}")
    n = np.array([[by[x].n for x in names] for by in per], dtype=np.int64)
    c = np.array([[by[x].c for x in names] for by in per], dtype=np.int64)
    has_tok = all(by[x].tokens is not None for by in per for x in names)
    tokens = (
        np.array([[float(sum(by[x].tokens)) for x in names] for by in per], dtype=np.float64)
        if has_tok
        else None
    )
    files = [str(per[0][x].file) for x in names]
    return Arm(
        label, names, files, n, c, tokens, [s.path for s in srcs], [_model_of(s) for s in srcs]
    )


def _as_arm(x: Arm | Sequence[str | Path], label: str) -> Arm:
    return x if isinstance(x, Arm) else load_arm(label, x)


def _check_aligned(*arms: Arm) -> None:
    for a in arms[1:]:
        if a.names != arms[0].names:
            diff = sorted(set(a.names) ^ set(arms[0].names))
            raise ValueError(
                f"{a.label} and {arms[0].label} cover different theorems "
                f"({len(a.names)} vs {len(arms[0].names)}), e.g. {diff[:3]}"
            )


def same_models(a: Arm, b: Arm) -> bool:
    """True if both arms are reads of the same checkpoints, seed by seed."""
    return a.S == b.S and all(m is not None for m in a.models) and list(a.models) == list(b.models)


def cluster_index(names: Sequence[str], clusters: Mapping[str, str] | None) -> I64 | None:
    """Theorem -> cluster id (0..G-1); a theorem absent from `clusters` is its own cluster."""
    if clusters is None:
        return None
    ids: dict[str, int] = {}
    out = []
    for nm in names:
        key = clusters.get(nm, f"\0{nm}")
        out.append(ids.setdefault(key, len(ids)))
    return np.array(out, dtype=np.int64)


def clusters_from_sets(paths: Sequence[str | Path], by: Sequence[str]) -> dict[str, str]:
    """name -> 'file:field=value,...' for every row of the given set files (jsonl)."""
    out: dict[str, str] = {}
    for p in paths:
        stem = Path(p).name.removesuffix(".jsonl")
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r["name"])] = stem + ":" + ",".join(f"{f}={r.get(f)}" for f in by)
    return out


# ------------------------------------------------------------------------------------- resampling


@dataclass
class Draws:
    """One set of bootstrap draws shared by every arm of a comparison: theorem weights `W`
    (draws x T resample counts, cluster counts copied to their theorems) and an RNG for the
    per-arm seed resamples."""

    W: F64
    rng: np.random.Generator
    levels: str

    @property
    def B(self) -> int:
        return int(self.W.shape[0])

    def seed_weights(self, S: int) -> F64:
        if self.levels == "theorems" or S == 1:
            return np.ones((self.B, S))
        return self.rng.multinomial(S, np.full(S, 1.0 / S), size=self.B).astype(np.float64)


def make_draws(
    T: int,
    *,
    draws: int = DRAWS,
    seed: int = 0,
    clusters: I64 | None = None,
    levels: str = "both",
) -> Draws:
    """The theorem resample depends only on (T, draws, seed, clusters), so separate calls with the
    same arguments share it."""
    if levels not in LEVELS:
        raise ValueError(f"levels must be one of {LEVELS}")
    rng = np.random.default_rng(seed)
    if levels == "seeds":
        wts = np.ones((draws, T))
    elif clusters is None:
        wts = rng.multinomial(T, np.full(T, 1.0 / T), size=draws).astype(np.float64)
    else:
        G = int(clusters.max()) + 1
        cw = rng.multinomial(G, np.full(G, 1.0 / G), size=draws).astype(np.float64)
        wts = cw[:, clusters]
    return Draws(wts, np.random.default_rng([seed, 1]), levels)


def _mean_draws(V: F64, U: F64, W: F64) -> F64:
    """Per draw: sum_{s,t} U[b,s] V[s,t] W[b,t] / (S * sum_t W[b,t])."""
    S = V.shape[0]
    return ((U @ V) * W).sum(1) / (S * W.sum(1))


def _interp_logk(P: F64, grid: I64, kstar: F64) -> tuple[F64, Bool]:
    """Row b of P (draws x len(grid)) is a pass@k curve on `grid`; its value at kstar[b], linear in
    log k, clamped to the grid's ends (second output: clamped)."""
    out = (kstar < grid[0]) | (kstar > grid[-1])
    if len(grid) == 1:
        return P[:, 0].copy(), out
    lg = np.log(grid.astype(np.float64))
    lk = np.log(np.clip(kstar, grid[0], grid[-1]))
    j = np.clip(np.searchsorted(lg, lk, side="right") - 1, 0, len(grid) - 2)
    w = (lk - lg[j]) / (lg[j + 1] - lg[j])
    rows = np.arange(P.shape[0])
    return P[rows, j] * (1.0 - w) + P[rows, j + 1] * w, out


def k_grid(K: int, grid: str = "all") -> I64:
    """k values the matched-token interpolation runs over: every k <= K, or 1, 2, 4, ..., K."""
    if grid == "all":
        return np.arange(1, K + 1, dtype=np.int64)
    if grid == "pow2":
        ks, k = [], 1
        while k <= K:
            ks.append(k)
            k *= 2
        if ks[-1] != K:
            ks.append(K)
        return np.array(ks, dtype=np.int64)
    raise ValueError(f"grid must be one of {GRIDS}")


def _tokens_values(arm: Arm, X: Sequence[float], U: F64, W: F64, grid: str) -> tuple[F64, Bool]:
    """Per draw and target X: mean over (resampled) seeds of pass@k* with k* = X / tokens per
    attempt; also whether any used seed was clamped."""
    if arm.tokens is None:
        raise ValueError(f"{arm.label}: per-attempt token counts missing")
    g = k_grid(arm.kmax, grid)
    curve = arm.curve()[:, :, g - 1]
    B, S = U.shape
    sw = W.sum(1)
    vals = np.zeros((B, len(X)))
    clamp = np.zeros((B, len(X)), dtype=bool)
    for s in range(S):
        P = (W @ curve[s]) / sw[:, None]
        tpa = (W @ arm.tokens[s]) / (W @ arm.n[s].astype(np.float64))
        for j, x in enumerate(X):
            v, out = _interp_logk(P, g, x / tpa)
            vals[:, j] += U[:, s] * v
            clamp[:, j] |= out & (U[:, s] > 0)
    return vals / S, clamp


def _theorem_values_at_tokens(arm: Arm, x: float, grid: str) -> F64:
    """S x T per-theorem pass@k* at x tokens (k* from each seed's observed tokens per attempt)."""
    g = k_grid(arm.kmax, grid)
    curve = arm.curve()[:, :, g - 1]
    tpa = arm.tokens_per_attempt()
    out = np.zeros((arm.S, arm.T))
    for s in range(arm.S):
        kstar = np.full(arm.T, x / tpa[s])
        out[s], _ = _interp_logk(curve[s], g, kstar)
    return out


# ------------------------------------------------------------------------------------- estimates


@dataclass
class Estimate:
    """A point estimate with its bootstrap percentile interval (fractions, not %).  For a
    difference: `p_boot` and the theorem sign test (`n_pos`, `n_neg`, `p_sign`)."""

    label: str
    est: float
    lo: float
    hi: float
    se: float
    k: int | None = None
    tokens: float | None = None
    T: int = 0
    seeds: tuple[int, ...] = ()
    paired_seeds: bool = False
    p_boot: float | None = None
    n_pos: int | None = None
    n_neg: int | None = None
    p_sign: float | None = None
    clamped: float = 0.0
    in_range: bool = True
    draws: int = 0
    samples: F64 | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "samples"}
        d["seeds"] = list(self.seeds)
        return d


def _interval(d: F64, level: float) -> tuple[float, float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(d, [a, 1.0 - a])
    return float(lo), float(hi), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0


def boot_p(d: F64) -> float:
    """Two-sided bootstrap p for 'difference = 0': twice the smaller share of draws on either side
    of 0 (ties count on both sides), capped at 1; 0 means below 1 / draws."""
    return float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))))


def sign_test(d: Sequence[float] | F64, eps: float = 1e-12) -> tuple[int, int, float]:
    """Exact two-sided sign test on per-theorem differences (ties dropped): (n+, n-, p)."""
    arr = np.asarray(d, dtype=np.float64)
    pos, neg = int((arr > eps).sum()), int((arr < -eps).sum())
    m = pos + neg
    if m == 0:
        return pos, neg, 1.0
    tail = sum(math.comb(m, i) for i in range(min(pos, neg) + 1))
    return pos, neg, min(1.0, 2 * tail / 2**m)


def _setup(
    arms: Sequence[Arm],
    draws: int,
    seed: int,
    clusters: Mapping[str, str] | None,
    levels: str,
) -> Draws:
    _check_aligned(*arms)
    return make_draws(
        arms[0].T,
        draws=draws,
        seed=seed,
        clusters=cluster_index(arms[0].names, clusters),
        levels=levels,
    )


def ci(
    arm: Arm | Sequence[str | Path],
    k: int | None,
    *,
    draws: int = DRAWS,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
    levels: str = "both",
    level: float = LEVEL,
) -> Estimate:
    """pass@k of one arm (mean over seeds and theorems) with its hierarchical-bootstrap interval.
    `arm` is an `Arm` or one source path per seed; k None is pass@n (every attempt)."""
    a = _as_arm(arm, "arm")
    dr = _setup([a], draws, seed, clusters, levels)
    V = a.values(k)
    d = _mean_draws(V, dr.seed_weights(a.S), dr.W)
    lo, hi, se = _interval(d, level)
    return Estimate(
        a.label, float(V.mean()), lo, hi, se, k=k, T=a.T, seeds=(a.S,), draws=draws, samples=d
    )


def ci_curve(
    arm: Arm,
    ks: Sequence[int],
    *,
    draws: int = DRAWS,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
    levels: str = "both",
    level: float = LEVEL,
) -> list[Estimate]:
    """`ci` at several k, from one set of draws (for bands on a pass@k curve)."""
    dr = _setup([arm], draws, seed, clusters, levels)
    U = dr.seed_weights(arm.S)
    out = []
    for k in ks:
        V = arm.values(k)
        d = _mean_draws(V, U, dr.W)
        lo, hi, se = _interval(d, level)
        out.append(
            Estimate(
                arm.label, float(V.mean()), lo, hi, se, k=k, T=arm.T, seeds=(arm.S,), draws=draws
            )  # fmt: skip
        )
    return out


def _seed_pair(a: Arm, b: Arm, dr: Draws, paired_seeds: bool | None) -> tuple[F64, F64, bool]:
    paired = same_models(a, b) if paired_seeds is None else paired_seeds
    if paired and a.S != b.S:
        raise ValueError(f"paired seeds need equal seed counts ({a.S} vs {b.S})")
    Ua = dr.seed_weights(a.S)
    Ub = Ua if paired else dr.seed_weights(b.S)
    return Ua, Ub, paired


def paired_diff(
    arm_a: Arm | Sequence[str | Path],
    arm_b: Arm | Sequence[str | Path],
    k: int | None,
    *,
    paired_seeds: bool | None = None,
    draws: int = DRAWS,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
    levels: str = "both",
    level: float = LEVEL,
) -> Estimate:
    """pass@k(a) - pass@k(b) on the same theorems, one theorem resample shared by both arms.  Seeds
    are resampled independently unless the arms are reads of the same checkpoints (`paired_seeds`
    None: detected; True/False forces).  Adds the bootstrap p and a sign test over theorems of the
    seed-mean difference (conditional on these models)."""
    a, b = _as_arm(arm_a, "a"), _as_arm(arm_b, "b")
    dr = _setup([a, b], draws, seed, clusters, levels)
    Ua, Ub, paired = _seed_pair(a, b, dr, paired_seeds)
    Va, Vb = a.values(k), b.values(k)
    d = _mean_draws(Va, Ua, dr.W) - _mean_draws(Vb, Ub, dr.W)
    lo, hi, se = _interval(d, level)
    pos, neg, ps = sign_test(Va.mean(0) - Vb.mean(0))
    return Estimate(
        f"{a.label} − {b.label}",
        float(Va.mean() - Vb.mean()),
        lo,
        hi,
        se,
        k=k,
        T=a.T,
        seeds=(a.S, b.S),
        paired_seeds=paired,
        p_boot=boot_p(d),
        n_pos=pos,
        n_neg=neg,
        p_sign=ps,
        draws=draws,
        samples=d,
    )


def at_tokens(
    arm: Arm | Sequence[str | Path],
    tokens: Sequence[float],
    *,
    grid: str = "all",
    draws: int = DRAWS,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
    levels: str = "both",
    level: float = LEVEL,
) -> list[Estimate]:
    """pass@k at matched sampled tokens per theorem, one estimate per target in `tokens`."""
    a = _as_arm(arm, "arm")
    dr = _setup([a], draws, seed, clusters, levels)
    pt, pclamp = _tokens_values(a, tokens, np.ones((1, a.S)), np.ones((1, a.T)), grid)
    d, clamp = _tokens_values(a, tokens, dr.seed_weights(a.S), dr.W, grid)
    out = []
    for j, x in enumerate(tokens):
        lo, hi, se = _interval(d[:, j], level)
        out.append(
            Estimate(
                a.label,
                float(pt[0, j]),
                lo,
                hi,
                se,
                tokens=float(x),
                T=a.T,
                seeds=(a.S,),
                clamped=float(clamp[:, j].mean()),
                in_range=not bool(pclamp[0, j]),
                draws=draws,
                samples=d[:, j],
            )
        )
    return out


def paired_diff_at_tokens(
    arm_a: Arm | Sequence[str | Path],
    arm_b: Arm | Sequence[str | Path],
    tokens: Sequence[float],
    *,
    grid: str = "all",
    paired_seeds: bool | None = None,
    draws: int = DRAWS,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
    levels: str = "both",
    level: float = LEVEL,
) -> list[Estimate]:
    """a - b at matched sampled tokens per theorem, paired on theorems as in `paired_diff`."""
    a, b = _as_arm(arm_a, "a"), _as_arm(arm_b, "b")
    dr = _setup([a, b], draws, seed, clusters, levels)
    Ua, Ub, paired = _seed_pair(a, b, dr, paired_seeds)
    one_a, one_b, one_t = np.ones((1, a.S)), np.ones((1, b.S)), np.ones((1, a.T))
    pa, ca = _tokens_values(a, tokens, one_a, one_t, grid)
    pb, cb = _tokens_values(b, tokens, one_b, one_t, grid)
    da, cla = _tokens_values(a, tokens, Ua, dr.W, grid)
    db, clb = _tokens_values(b, tokens, Ub, dr.W, grid)
    d, clamp = da - db, cla | clb
    out = []
    for j, x in enumerate(tokens):
        lo, hi, se = _interval(d[:, j], level)
        per_t = _theorem_values_at_tokens(a, x, grid).mean(0)
        per_t = per_t - _theorem_values_at_tokens(b, x, grid).mean(0)
        pos, neg, ps = sign_test(per_t)
        out.append(
            Estimate(
                f"{a.label} − {b.label}",
                float(pa[0, j] - pb[0, j]),
                lo,
                hi,
                se,
                tokens=float(x),
                T=a.T,
                seeds=(a.S, b.S),
                paired_seeds=paired,
                p_boot=boot_p(d[:, j]),
                n_pos=pos,
                n_neg=neg,
                p_sign=ps,
                clamped=float(clamp[:, j].mean()),
                in_range=not bool(ca[0, j] or cb[0, j]),
                draws=draws,
                samples=d[:, j],
            )
        )
    return out


def informative(arms: Sequence[Arm], k: int | None) -> dict[str, int]:
    """Theorems at 0% (or 100%) pass@k in every seed of every arm carry no information about any
    difference between them; the rest are 'informative'."""
    _check_aligned(*arms)
    V = np.concatenate([a.values(k) for a in arms], axis=0)
    zero = int((V <= 0.0).all(0).sum())
    one = int((V >= 1.0).all(0).sum())
    return {
        "n": int(V.shape[1]),
        "all_0": zero,
        "all_100": one,
        "informative": V.shape[1] - zero - one,
    }


# ------------------------------------------------------------------- analytic variance and power


@dataclass
class Components:
    """Method-of-moments variance decomposition of an arm mean or an arm difference in a crossed
    seed x theorem random-effects model V[s,t] = mu + seed_s + theorem_t + residual_st.

    `seed_var[i]`: sigma2 of the seed effect of each independent seed population (one for a single
    arm or a paired-seed difference, two for an unpaired difference); `res_ms[i]`: its residual mean
    square; `theorem_var`: Var_t of the true per-theorem value (difference), the residual share
    removed.  Var(est) = sum(seed_var / S) + deff * (theorem_var + sum(res_ms / S)) / T, where
    `deff` is the design effect of clustered theorems (cluster-robust over independent theorem
    variance of the mean; 1 without clusters)."""

    est: float
    se: float
    df: float
    lo: float
    hi: float
    p: float
    seed_var: list[float]
    res_ms: list[float]
    theorem_var: float
    S: list[int]
    T: int
    paired_seeds: bool = False
    deff: float = 1.0
    clusters: int | None = None

    def var(self, S: int | None = None, T: int | None = None) -> float:
        """Projected variance of the estimate with S seeds per arm and T theorems."""
        Ts = self.T if T is None else T
        out = 0.0
        res = 0.0
        for i, (sv, ms) in enumerate(zip(self.seed_var, self.res_ms, strict=True)):
            Si = self.S[i] if S is None else S
            out += (0.0 if math.isnan(sv) else sv) / Si
            res += (0.0 if math.isnan(ms) else ms) / Si
        return out + self.deff * (self.theorem_var + res) / Ts

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _decompose(V: F64) -> tuple[float, float, F64]:
    """(sigma2_seed, residual MS, theorem means) of one S x T matrix; NaN with one seed."""
    S, T = V.shape
    cm = V.mean(0)
    if S < 2 or T < 2:
        return math.nan, math.nan, cm
    rm = V.mean(1)
    resid = V - rm[:, None] - cm[None, :] + V.mean()
    ms = float((resid**2).sum() / ((S - 1) * (T - 1)))
    sv = max(0.0, float(np.var(rm, ddof=1)) - ms / T)
    return sv, ms, cm


def components(
    arm_a: Arm,
    arm_b: Arm | None,
    k: int | None,
    *,
    paired_seeds: bool | None = None,
    clusters: Mapping[str, str] | None = None,
    levels: str = "both",
    level: float = LEVEL,
) -> Components:
    """Analytic SE, Satterthwaite t interval and t-test p for pass@k of `arm_a`, or of
    `arm_a - arm_b` (paired on theorems; seeds paired if the arms share checkpoints).  With
    `clusters`, the theorem term is the cluster-robust variance of the mean of the per-theorem
    seed means (G / (G - 1) * sum_g (total_g - size_g * mean)^2 / T^2, G - 1 df).  `levels` as in
    `make_draws`: 'seeds' conditions on the theorems (variance of the seed means only),
    'theorems' on the seeds (the theorem term only); the power fields stay those of 'both'."""
    if levels not in LEVELS:
        raise ValueError(f"levels must be one of {LEVELS}")
    if arm_b is None:
        mats = [arm_a.values(k)]
        est = float(mats[0].mean())
        paired = False
    else:
        _check_aligned(arm_a, arm_b)
        paired = same_models(arm_a, arm_b) if paired_seeds is None else paired_seeds
        Va, Vb = arm_a.values(k), arm_b.values(k)
        est = float(Va.mean() - Vb.mean())
        mats = [Va - Vb] if paired else [Va, Vb]
    dec = [_decompose(V) for V in mats]
    S = [int(V.shape[0]) for V in mats]
    T = int(mats[0].shape[1])
    cm = dec[0][2] if len(dec) == 1 else dec[0][2] - dec[1][2]
    var_t = float(np.var(cm, ddof=1)) if T > 1 else 0.0
    seed_var = [d[0] for d in dec]
    res_ms = [d[1] for d in dec]
    res = sum(0.0 if math.isnan(ms) else ms / s for ms, s in zip(res_ms, S, strict=True))
    theorem_var = max(0.0, var_t - res)
    terms = [(sv / s, s - 1) for sv, s in zip(seed_var, S, strict=True) if not math.isnan(sv)]
    v_th, df_th, deff, n_cl = var_t / T, T - 1, 1.0, None
    idx = cluster_index(arm_a.names, clusters)
    if idx is not None and int(idx.max()) + 1 < T:
        n_cl = g = int(idx.max()) + 1
        tot = np.bincount(idx, weights=cm, minlength=g)
        size = np.bincount(idx, minlength=g).astype(np.float64)
        v_th = g / (g - 1) * float(((tot - size * cm.mean()) ** 2).sum()) / T**2
        df_th = g - 1
        deff = v_th / (var_t / T) if var_t > 0 else 1.0
    terms.append((v_th, df_th))
    if levels == "theorems":
        terms = [(v_th, df_th)]
    elif levels == "seeds":
        terms = [
            (float(np.var(V.mean(1), ddof=1)) / V.shape[0], V.shape[0] - 1)
            for V in mats
            if V.shape[0] > 1
        ]
    v = sum(x for x, _ in terms)
    den = sum(x * x / df for x, df in terms if df > 0 and x > 0)
    df = v * v / den if den > 0 else math.inf
    se = math.sqrt(v)
    q = t_ppf(0.5 + level / 2, df)
    p = 2.0 * (1.0 - t_cdf(abs(est) / se, df)) if se > 0 else (1.0 if est == 0 else 0.0)
    return Components(
        est, se, df, est - q * se, est + q * se, p, seed_var, res_ms, theorem_var, S, T, paired,
        deff, n_cl,
    )  # fmt: skip


def pooled_components(
    arms: Sequence[Arm], comparisons: Sequence[tuple[Arm, Arm]], k: int | None
) -> Components:
    """A generic two-independent-arm design for power: the seed variance and residual MS averaged
    over `arms` (each estimated from its own seeds, so 2 df apiece with 3 seeds; pass independent
    models only), and the per-theorem difference variance averaged over `comparisons`."""
    sv, ms = [], []
    for a in arms:
        s, m, _cm = _decompose(a.values(k))
        if not math.isnan(s):
            sv.append(s)
            ms.append(m)
    if not sv:
        raise ValueError("pooling needs arms with at least 2 seeds")
    tv = [components(a, b, k, paired_seeds=False).theorem_var for a, b in comparisons]
    S = round(sum(a.S for a in arms) / len(arms))
    s_bar, m_bar = sum(sv) / len(sv), sum(ms) / len(ms)
    th = sum(tv) / len(tv) if tv else 0.0
    out = Components(
        math.nan, 0.0, math.nan, math.nan, math.nan, math.nan,
        [s_bar, s_bar], [m_bar, m_bar], th, [S, S], arms[0].T,
    )  # fmt: skip
    out.se = math.sqrt(out.var())
    return out


def mde(
    comp: Components,
    S: int | None = None,
    T: int | None = None,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true difference a two-sided z-test at `alpha` detects with probability `power`
    at S seeds per arm and T theorems (defaults: the design measured)."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * math.sqrt(comp.var(S, T))


def seeds_needed(
    comp: Components, target: float, T: int | None = None, *, max_seeds: int = 10_000, **kw: float
) -> int | None:
    """Fewest seeds per arm for `mde` <= target at T theorems; None if more than `max_seeds`."""
    for s in range(2, max_seeds + 1):
        if mde(comp, s, T, **kw) <= target:
            return s
    return None


def theorems_needed(
    comp: Components, target: float, S: int | None = None, *, max_t: int = 10**7, **kw: float
) -> int | None:
    """Fewest theorems for `mde` <= target at S seeds per arm; None if the seed term alone
    exceeds the target (more theorems cannot help) or more than `max_t` are needed."""
    lo, hi = 2, max_t
    if mde(comp, S, hi, **kw) > target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if mde(comp, S, mid, **kw) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------- Student t (no scipy)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x)
    bt = math.exp(lbt + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if math.isinf(df):
        return NormalDist().cdf(t)
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (bisection on `t_cdf`); df = inf gives the normal quantile."""
    if math.isinf(df) or df > 1e7:
        return NormalDist().inv_cdf(q)
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q:
        lo *= 2.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ------------------------------------------------------------------------------------- the table


def _pp(v: float, signed: bool = False) -> str:
    s = f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"
    return s.replace("-", "−")


def _pval(p: float | None, draws: int = 0) -> str:
    if p is None:
        return "–"
    if p == 0.0 and draws:
        return f"<{1 / draws:.4f}"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def _ci_cell(e: Estimate, *, signed: bool = False, unit: str = "%") -> str:
    if not e.in_range:
        return "–"
    flag = "†" if e.clamped > 0.05 else ""
    return f"{_pp(e.est, signed)}{unit} [{_pp(e.lo, signed)}, {_pp(e.hi, signed)}]{flag}"


def _tok_label(x: float) -> str:
    return f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"


def _kname(k: int | None, K: int) -> str:
    return f"pass@{K if k is None else k}"


@dataclass
class TableSpec:
    groups: list[Group]
    ks: list[int | None]
    tokens: list[float]
    ref: str | None = None
    pairs: list[tuple[str, str]] = field(default_factory=lambda: list[tuple[str, str]]())
    cell: str | None = None
    sets: list[str] | None = None
    exclude: list[str] = field(default_factory=lambda: list[str]())
    clusters: dict[str, str] | None = None
    draws: int = DRAWS
    seed: int = 0
    levels: str = "both"
    grid: str = "all"
    level: float = LEVEL
    mde_target: float = 0.03
    title: str = "pass@k with hierarchical-bootstrap intervals"
    command: str = ""


def table(spec: TableSpec) -> tuple[str, dict[str, Any]]:
    """Markdown tables (per arm, paired differences, matched tokens, informative theorems, power)
    and every number as a dict."""
    arms = [
        load_arm(g.label, g.paths, cell=spec.cell, sets=spec.sets, exclude=spec.exclude)
        for g in spec.groups
    ]
    _check_aligned(*arms)
    by = {a.label: a for a in arms}
    if len(by) != len(arms):
        raise ValueError("group labels must be unique")
    ref = by[spec.ref] if spec.ref else arms[0]
    comps = [(a, ref) for a in arms if a is not ref]
    for x, y in spec.pairs:
        if x not in by or y not in by:
            raise ValueError(f"--pair {x!r} vs {y!r}: unknown group")
        comps.append((by[x], by[y]))
    kw: dict[str, Any] = {
        "draws": spec.draws,
        "seed": spec.seed,
        "clusters": spec.clusters,
        "levels": spec.levels,
        "level": spec.level,
    }
    K = min(a.kmax for a in arms)
    T = arms[0].T
    files = sorted(set(arms[0].files))
    n_clusters = None
    if spec.clusters is not None:
        ci_idx = cluster_index(arms[0].names, spec.clusters)
        n_clusters = int(ci_idx.max()) + 1 if ci_idx is not None else None
    pct = round(100 * spec.level)
    lines: list[str] = [f"### {spec.title}", ""]
    if spec.command:
        lines += ["```", spec.command, "```", ""]
    lines.append(
        f"n = {T} theorems ({', '.join(files)}); {spec.draws} draws, RNG seed {spec.seed}; "
        f"resampled: {spec.levels}"
        + (f", theorems in {n_clusters} clusters" if n_clusters else "")
        + f". Brackets: {pct}% percentile intervals."
    )
    lines.append("")
    data: dict[str, Any] = {"spec": {k: v for k, v in spec.__dict__.items() if k != "clusters"}}
    data["spec"]["groups"] = [g.__dict__ for g in spec.groups]
    data["n_theorems"] = T
    data["n_clusters"] = n_clusters
    data["arms"] = []
    # ---- per arm
    hasT = all(a.tokens is not None for a in arms)
    head = "| arm | seeds | " + " | ".join(_kname(k, K) for k in spec.ks)
    head += " | tokens / attempt |" if hasT else " |"
    lines += [head, "|" + "---|" * (2 + len(spec.ks) + (1 if hasT else 0))]
    for a in arms:
        es = [ci(a, k, **kw) for k in spec.ks]
        row = f"| {a.label} | {a.S} | " + " | ".join(_ci_cell(e) for e in es)
        tpa = float(a.tokens_per_attempt().mean()) if a.tokens is not None else None
        row += f" | {tpa:.0f} |" if tpa is not None else " |"
        lines.append(row)
        data["arms"].append(
            {
                "label": a.label,
                "paths": a.paths,
                "models": a.models,
                "seeds": a.S,
                "tokens_per_attempt": tpa,
                "ci": [e.as_dict() for e in es],
            }
        )
    lines.append("")
    # ---- paired differences
    data["diffs"] = []
    boot: dict[tuple[str, int | None], tuple[Estimate, Components]] = {}
    if comps:
        lines += [
            f"Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean "
            f"difference is +/−. t-check: analytic SE, Satterthwaite t {pct}% interval.",
            "",
            "| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |",
            "|---|---|---|---|---|---|---|",
        ]
        for a, b in comps:
            for k in spec.ks:
                e = paired_diff(a, b, k, **kw)
                cp = components(
                    a, b, k, clusters=spec.clusters, levels=spec.levels, level=spec.level
                )
                boot[(e.label, k)] = (e, cp)
                seeds = f"{a.S} vs {b.S}" + (" (paired)" if e.paired_seeds else "")
                lines.append(
                    f"| {e.label} | {_kname(k, K)} | {_ci_cell(e, signed=True, unit='')} | "
                    f"{_pval(e.p_boot, spec.draws)} | {e.n_pos}/{e.n_neg} ({_pval(e.p_sign)}) | "
                    f"[{_pp(cp.lo, True)}, {_pp(cp.hi, True)}] ({cp.df:.1f}) | {seeds} |"
                )
                data["diffs"].append({**e.as_dict(), "analytic": cp.as_dict()})
        lines.append("")
    # ---- matched tokens
    data["tokens"] = []
    data["token_diffs"] = []
    if spec.tokens and hasT:
        lines += [
            "pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of "
            "the read's range; † more than 5% of draws clamped to pass@1 or pass@n).",
            "",
            "| arm | " + " | ".join(_tok_label(x) for x in spec.tokens) + " |",
            "|---|" + "---|" * len(spec.tokens),
        ]
        for a in arms:
            es = at_tokens(a, spec.tokens, grid=spec.grid, **kw)
            lines.append(f"| {a.label} | " + " | ".join(_ci_cell(e) for e in es) + " |")
            data["tokens"].append({"label": a.label, "at": [e.as_dict() for e in es]})
        if comps:
            lines += [
                "",
                "| difference (pp) | " + " | ".join(_tok_label(x) for x in spec.tokens) + " |",
                "|---|" + "---|" * len(spec.tokens),
            ]
            for a, b in comps:
                es = paired_diff_at_tokens(a, b, spec.tokens, grid=spec.grid, **kw)
                cells = [
                    _ci_cell(e, signed=True, unit="")
                    + (f" p {_pval(e.p_boot, spec.draws)}" if e.in_range else "")
                    for e in es
                ]
                lines.append(f"| {es[0].label} | " + " | ".join(cells) + " |")
                data["token_diffs"].append([e.as_dict() for e in es])
        lines.append("")
    # ---- informative theorems
    info_ks: list[int | None] = list(dict.fromkeys([*spec.ks, None]))
    data["informative"] = {}
    lines += [
        "Informative theorems (not 0% or not 100% in every seed of every arm above):",
        "",
        "| k | 0% everywhere | 100% everywhere | informative |",
        "|---|---|---|---|",
    ]
    for k in info_ks:
        inf = informative(arms, k)
        data["informative"][_kname(k, K)] = inf
        lines.append(
            f"| {_kname(k, K)} | {inf['all_0']} | {inf['all_100']} | {inf['informative']} |"
        )
    lines.append("")
    # ---- variance components and power
    data["power"] = []
    if comps:
        tgt = spec.mde_target
        lines += [
            f"Power (analytic components; two-sided α 0.05, 80% power, z approximation): "
            f"minimum detectable difference (MDE) of this design, and the seeds per arm (at "
            f"n = {T}) or theorems (at these seeds) needed for an MDE of {_pp(tgt)} pp. Seed SDs "
            f"come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread "
            f"no more than theorem-level noise predicts"
            + (" (theorem terms are cluster-robust)." if n_clusters else "."),
            "",
            "| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic "
            f"(pp) | MDE (pp) | seeds for {_pp(tgt)} pp | theorems for {_pp(tgt)} pp |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a, b in comps:
            for k in spec.ks:
                e, cp = boot[(f"{a.label} − {b.label}", k)]
                sds = " / ".join("–" if math.isnan(v) else _pp(math.sqrt(v)) for v in cp.seed_var)
                one_seed = any(math.isnan(v) for v in cp.seed_var)
                ns = None if one_seed else seeds_needed(cp, tgt)
                nt = theorems_needed(cp, tgt)
                ns_txt = "n/a (1 seed)" if one_seed else (str(ns) if ns is not None else ">10k")
                lines.append(
                    f"| {a.label} − {b.label} | {_kname(k, K)} | {sds} | "
                    f"{_pp(math.sqrt(cp.theorem_var))} | {_pp(e.se)} / {_pp(cp.se)} | "
                    f"{_pp(mde(cp))} | {ns_txt} | {nt if nt is not None else 'never'} |"
                )
                data["power"].append(
                    {
                        "label": f"{a.label} − {b.label}",
                        "k": k,
                        "components": cp.as_dict(),
                        "se_boot": e.se,
                        "mde": mde(cp),
                        "seeds_needed": ns,
                        "theorems_needed": nt,
                    }
                )
        lines.append("")
    return "\n".join(lines), data


# ------------------------------------------------------------------------------------------ CLI


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ks(s: str) -> list[int | None]:
    out: list[int | None] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(None if x in ("n", "all", "max") else int(x))
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="markdown tables with hierarchical-bootstrap intervals")
    t.add_argument(
        "--group",
        action="append",
        required=True,
        type=parse_group,
        help="'label=path[,path...]', one source per seed; repeat per arm",
    )
    t.add_argument("--k", default="1,5", help="comma-separated k ('n' = every attempt)")
    t.add_argument("--tokens", default="", help="comma-separated sampled tokens per theorem")
    t.add_argument("--out", default=None, help="markdown output (default: stdout)")
    t.add_argument("--json", default=None, help="also write every number here")
    t.add_argument("--ref", default=None, help="reference group for differences (default: first)")
    t.add_argument(
        "--pair", action="append", default=[], help="extra difference 'A vs B' (group labels)"
    )
    t.add_argument("--cell", default=None, help="code.thinking.eval cell for directory sources")
    t.add_argument("--sets", default=None, help="keep only these files (comma-separated)")
    t.add_argument("--exclude", default="", help="drop these files (comma-separated)")
    t.add_argument(
        "--clusters", default=None, help="set .jsonl files naming each theorem's cluster"
    )
    t.add_argument("--cluster-by", default="family,knob", help="fields that define a cluster")
    t.add_argument("--draws", type=int, default=DRAWS)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--levels", choices=LEVELS, default="both")
    t.add_argument("--grid", choices=GRIDS, default="all")
    t.add_argument("--mde-target", type=float, default=0.03, help="difference for 'needed' columns")
    t.add_argument("--title", default="pass@k with hierarchical-bootstrap intervals")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    a = parser().parse_args(args)
    pairs = []
    for p in a.pair:
        x, sep, y = p.partition(" vs ")
        if not sep:
            raise SystemExit(f"--pair wants 'A vs B', got {p!r}")
        pairs.append((x.strip(), y.strip()))
    clusters = None
    if a.clusters:
        clusters = clusters_from_sets(a.clusters.split(","), a.cluster_by.split(","))
    spec = TableSpec(
        groups=a.group,
        ks=_ks(a.k),
        tokens=_floats(a.tokens),
        ref=a.ref,
        pairs=pairs,
        cell=a.cell,
        sets=a.sets.split(",") if a.sets else None,
        exclude=[x for x in a.exclude.split(",") if x],
        clusters=clusters,
        draws=a.draws,
        seed=a.seed,
        levels=a.levels,
        grid=a.grid,
        mde_target=a.mde_target,
        title=a.title,
        command="python -m code.thinking.stats " + shlex.join(args),
    )
    md, data = table(spec)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md + "\n")
        print(a.out)
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(data, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
