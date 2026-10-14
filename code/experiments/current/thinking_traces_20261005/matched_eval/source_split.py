"""What in "source" drives the GRPO − EI split? Structural features and closeness to the RL pools.

    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.source_split \
        [--draws 4000] [--v2 RUNS/eval/v2] [--out FIGDIR] [--table PATH]

Follows `length_split` (same reads, pairs, sets, lengths and intervals; existing reads only). GRPO
gains on human-written problems (textbook72, Charles149) and loses on generated ones (13–20, 21–30)
at every length; this asks which problem properties carry that difference.

**Features** (gap-analysis taggers, `gap_analysis/common.py` via `hard712/ext.py`, the code behind
the curriculum report's shares). Statement: premises, goal shape ((A → B) → C apart), atoms,
connectives, classical-only (not intuitionistically provable, G4ip) and a ¬¬ subformula. Proof:
case split (∨E), ex falso (⊥E), ¬¬E (reductio / classical), a ¬-subproof inside another subproof,
box depth. Proofs are the set's reference proofs; textbook72 has none, so its proofs are the
shortest Lean-verified proof either headline arm found (all seeds, plain and guided reads),
translated back to ND by `shared_code.lean_seq.inverse`. One proof per problem, shared by every
arm; textbook72 problems no arm solved have statement features only.

**Δ by feature**: GRPO − EI pass@1 / pass@5 per category within each source (`stats.paired_diff`,
seeds paired). **Reweighting**: the human-written − generated gap after reweighting the generated
problems to the human-written mix of one feature (post-stratification) or of every feature at once
(raking on the marginals). Human-written problems in a category with fewer than `MIN_SUPPORT`
generated problems cannot be matched and are left out of that contrast (counted in the table).
The weights are refitted on every bootstrap draw (problems resampled within each set, seeds once
for every set and both arms), so the intervals include the uncertainty of the human-written mix.

**Closeness to the RL pools** (`runs/pools/curriculum/{L2,L3}/pool.jsonl`, 10,000 problems), model
independent: `sub`, the share of the statement's subformula shapes with >= 2 connectives (atoms
renamed by first occurrence) that occur in some pool statement; `nn`, minus the distance to the
nearest pool problem in a standardised statement-and-proof feature signature; `skel`, whether the
proof's box skeleton occurs among the pool's reference proofs. Δ per closeness quartile (quartiles
of all four sets) within each source, the within-source slope of Δ on the closeness rank, and the
gap after reweighting the generated problems to the human-written closeness quartiles.

Writes `fig_source_split_features_<pair>`, `fig_source_split_reweight` and
`fig_source_split_closeness` (PDF/SVG/PNG) to `--out`, and the tables to `--table`.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from code.experiments.current.thinking_traces_20261005.hard712 import ext
from code.experiments.current.thinking_traces_20261005.matched_eval import length_split as LSP
from code.experiments.current.thinking_traces_20261005.matched_eval import v2_results as V
from code.ndbench.figures import style as S
from code.shared_code import lean_seq
from code.thinking import stats

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
GC = ext.gap_common()
POOLS = [V.R / f"pools/curriculum/{b}/pool.jsonl" for b in ("L2", "L3")]
HUMAN, GENERATED, SETS = LSP.HUMAN, LSP.GENERATED, LSP.SETS
SRC_COL, SRC_MK = LSP.SRC_COL, LSP.SRC_MK
MIN_SUPPORT = 5  # generated problems a category needs before it can carry human-written weight
RAKE_ITERS = 40
MIN_ESS = 30  # reweighted contrasts with fewer effective generated problems are greyed


# ------------------------------------------------------------------------------------- proofs


def lean_tokens(text: str) -> list[str]:
    """A read's accepted Lean text -> `lean_seq` symbols (`.1`, `.2`, `.elim` unglued)."""
    out: list[str] = []
    for t in text.split():
        m = re.fullmatch(r"([nh]\d+)(\.1|\.2|\.elim)", t)
        out += [m.group(1), m.group(2)] if m else [t]
    return out


def found_proofs(ld: V.Loader, s: str) -> dict[str, str]:
    """Per problem of set `s`: the shortest accepted proof of the headline arms' reads (every seed,
    plain and guided), as ND text; ties go to the lexicographically first ND text."""
    best: dict[str, tuple[int, str]] = {}
    for arm in LSP.FALLBACK:
        for ra in ("plain", "logical") if V.SET[s].guided else ("plain",):
            for seed in V.SEEDS:
                p = V.rows_path(V.SPEC[arm], V.SET[s], seed, ra, ld.v2)
                if p is None or not p.exists():
                    continue
                for r in LSP.jsonl(p):
                    for t in r["accepted"]:
                        nd = lean_seq.inverse(lean_tokens(t))
                        cand = (LSP.nd_lines(nd), nd)
                        if r["name"] not in best or cand < best[r["name"]]:
                            best[r["name"]] = cand
    return {n: v[1] for n, v in best.items()}


@dataclass(frozen=True)
class Problem:
    prompt: str
    proof: str | None
    ref: bool  # the proof is the set's reference proof


def problems(ld: V.Loader) -> dict[str, dict[str, Problem]]:
    out: dict[str, dict[str, Problem]] = {}
    for s, path in (
        ("t1320", V.MATCHED / "sets/t1320_n1000.jsonl"),
        ("t2130", V.MATCHED / "sets/t2130_n1000.jsonl"),
        ("charles149", V.R / "pools/charles_1002/charles_clean.jsonl"),
    ):
        keep = ld.keep[s]
        out[s] = {
            r["name"]: Problem(r["prompt"], r["reference_proof"], True)
            for r in LSP.jsonl(path)
            if keep is None or r["name"] in keep
        }
    found = found_proofs(ld, "tb72")
    out["tb72"] = {
        r["name"]: Problem(r["prompt"], found.get(r["name"]), False)
        for f in ("textbook_dev", "textbook_train")
        for r in LSP.jsonl(LSP.TB / f"{f}.jsonl")
    }
    return out


# ----------------------------------------------------------------------------------- features


@dataclass(frozen=True)
class Feature:
    key: str
    title: str
    cats: tuple[str, ...]
    of: Callable[[dict[str, Any], dict[str, Any] | None], str | None]
    group: str = "statement"  # "statement", "proof" or "case split" (∨E proofs only)


def _bin(v: int, edges: Sequence[int], labels: Sequence[str]) -> str:
    for e, lab in zip(edges, labels, strict=False):
        if v <= e:
            return lab
    return labels[-1]


def _goal(t: dict[str, Any]) -> str:
    if t["goal_imp_imp_ante"]:
        return "(A→B)→C"
    return {"atom": "atom", "not": "¬A", "and": "A∧B", "or": "A∨B", "imp": "A→B", "bot": "⊥"}[
        t["goal_type"]
    ]


def _yes(v: Any) -> str:
    return "yes" if v else "no"


def _p(
    f: Callable[[dict[str, Any]], str],
) -> Callable[[dict[str, Any], dict[str, Any] | None], str | None]:
    """A proof feature (None without a proof)."""
    return lambda t, p: None if p is None else f(p)


def _ore(
    f: Callable[[dict[str, Any]], str],
) -> Callable[[dict[str, Any], dict[str, Any] | None], str | None]:
    """A feature of the case splits (None without a proof or without ∨E)."""
    return lambda t, p: None if p is None or not p["p_has_ORE"] else f(p)


FEATURES: tuple[Feature, ...] = (
    Feature(
        "prem",
        "premises",
        ("0", "1", "2", "3+"),
        lambda t, p: _bin(t["n_prem"], (0, 1, 2), ("0", "1", "2", "3+")),
    ),
    Feature(
        "goal", "goal", ("atom", "¬A", "A∧B", "A∨B", "A→B", "(A→B)→C", "⊥"), lambda t, p: _goal(t)
    ),
    Feature(
        "atoms",
        "atoms",
        ("1–2", "3", "4"),
        lambda t, p: _bin(t["n_atoms"], (2, 3), ("1–2", "3", "4")),
    ),
    Feature(
        "size",
        "connectives",
        ("≤8", "9–12", "13–16", "17+"),
        lambda t, p: _bin(t["n_conn"], (8, 12, 16), ("≤8", "9–12", "13–16", "17+")),
    ),
    Feature(
        "classical",
        "classical-only statement",
        ("no", "yes"),
        lambda t, p: _yes(t["classical_only"]),
    ),
    Feature("dneg", "¬¬ in statement", ("no", "yes"), lambda t, p: _yes(t["has_dneg"])),
    Feature("pad", "padding (A∨A, A∨(A∧B), …)", ("no", "yes"), lambda t, p: _yes(t["pad"])),
    Feature("case", "case split (∨E)", ("no", "yes"), _p(lambda p: _yes(p["p_has_ORE"])), "proof"),
    Feature(
        "exfalso", "ex falso (⊥E)", ("no", "yes"), _p(lambda p: _yes(p["p_has_BOTE"])), "proof"
    ),
    Feature(
        "dn",
        "¬¬E (reductio / classical)",
        ("no", "yes"),
        _p(lambda p: _yes(p["p_has_DN"])),
        "proof",
    ),
    Feature(
        "nneg",
        "nested ¬-subproof",
        ("no", "yes"),
        _p(lambda p: _yes(p["p_negbox_nested"])),
        "proof",
    ),
    Feature(
        "depth",
        "box depth",
        ("0–1", "2", "3+"),
        _p(lambda p: _bin(p["p_box_depth"], (1, 2), ("0–1", "2", "3+"))),
        "proof",
    ),
    Feature(
        "n_ore",
        "∨E steps",
        ("1", "2", "3+"),
        _ore(lambda p: _bin(p["p_n_ORE"], (1, 2), ("1", "2", "3+"))),
        "case split",
    ),
    Feature(
        "ore_prem",
        "∨E on a premise",
        ("no", "yes"),
        _ore(lambda p: _yes(p["p_ore_on_prem"])),
        "case split",
    ),
    Feature(
        "ore_triv",
        "∨E with a trivial branch",
        ("no", "yes"),
        _ore(lambda p: _yes(p["triv_ore"])),
        "case split",
    ),
    Feature(
        "ore_lemma",
        "∨E result reused",
        ("no", "yes"),
        _ore(lambda p: _yes(p["p_ore_lemma"])),
        "case split",
    ),
)
FEAT = {f.key: f for f in FEATURES}


def padding(prompt: str) -> int:
    """Idempotent or absorption subformulas: A∨A, A∧A, A→A, A∨(A∧B), A∧(A∨B) (either order)."""
    prem, goal = GC.parse_prompt(prompt)
    n = 0
    for f in [*prem, goal]:
        for g in GC.subs(f):
            if g[0] not in ("and", "or", "imp"):
                continue
            a, b = g[1], g[2]
            dual = {"and": "or", "or": "and"}.get(g[0])
            n += a == b or (
                dual is not None
                and ((b[0] == dual and a in b[1:]) or (a[0] == dual and b in a[1:]))
            )
    return n


def trivial_ore(proof: str) -> int:
    """A ∨E with a branch whose result is its own assumption (A∨A, A∨(A∧B) and the like)."""
    for ln in GC.parse_proof(proof):
        if ln["rule"] == "ORE":
            _, s1, e1, s2, e2 = ln["refs"]
            if s1 == e1 or s2 == e2:
                return 1
    return 0


@dataclass
class Tagged:
    thm: dict[str, Any]
    proof: dict[str, Any] | None
    cat: dict[str, str | None]


def tag(
    prompt: str,
    proof: str | None,
    thm: dict[str, Any] | None = None,
    pf: dict[str, Any] | None = None,
) -> Tagged:
    """Features of one problem (`thm`, `pf`: the taggers' outputs when already computed)."""
    t = dict(GC.thm_features(prompt) if thm is None else thm, pad=padding(prompt))
    p = None
    if proof is not None:
        p = dict(
            GC.proof_features(prompt, proof) if pf is None else pf, triv_ore=trivial_ore(proof)
        )
    return Tagged(t, p, {f.key: f.of(t, p) for f in FEATURES})


# ---------------------------------------------------------------------------------- closeness


def _shape(f: Any, names: dict[str, str]) -> str:
    if f[0] == "atom":
        return names.setdefault(f[1], f"a{len(names)}")
    if f[0] == "bot":
        return "F"
    if f[0] == "not":
        return "~" + _shape(f[1], names)
    op = {"and": "&", "or": "v", "imp": ">"}[f[0]]
    return f"({_shape(f[1], names)}{op}{_shape(f[2], names)})"


def _n_conn(f: Any) -> int:
    return 0 if f[0] in ("atom", "bot") else 1 + sum(_n_conn(g) for g in f[1:])


def shapes(prompt: str, min_conn: int = 2) -> set[str]:
    """Subformula shapes of a statement with >= `min_conn` connectives, atoms renamed in order of
    first occurrence within each subformula."""
    prem, goal = GC.parse_prompt(prompt)
    return {_shape(g, {}) for f in [*prem, goal] for g in GC.subs(f) if _n_conn(g) >= min_conn}


SIG_THM = (
    "n_prem",
    "n_atoms",
    "n_neg",
    "n_and",
    "n_or",
    "n_imp",
    "fdepth",
    "goal_depth",
    "imp_nest",
    "n_dneg",
    "classical_only",
    "goal_atom",
    "goal_not",
    "goal_and",
    "goal_or",
    "goal_imp",
    "goal_imp_imp_ante",
)
SIG_PROOF = (
    "p_lines",
    "p_box_depth",
    "p_n_AS",
    "p_n_ORE",
    "p_n_BOTE",
    "p_n_DN",
    "p_n_NEGI",
    "p_n_IMPI",
    "p_n_IMPE",
    "p_n_NEGE",
    "p_n_ANDI",
    "p_n_ANDE",
    "p_n_ORI",
)


def signature(t: dict[str, Any], p: dict[str, Any]) -> list[float]:
    return [float(t[k]) for k in SIG_THM] + [float(p[k]) for k in SIG_PROOF]


@dataclass
class Pool:
    shapes: set[str]
    skeletons: Counter[str]
    sig: F64  # standardised pool signatures
    mu: F64
    sd: F64
    tagged: list[Tagged]


def load_pool() -> Pool:
    rows = [r for p in POOLS for r in LSP.jsonl(p)]
    tg = [tag(r["prompt"], r["reference_proof"], r["gap_thm"], r["gap_proof"]) for r in rows]
    sig = np.array([signature(x.thm, x.proof) for x in tg if x.proof is not None])
    mu, sd = sig.mean(0), sig.std(0)
    sd[sd == 0] = 1.0
    return Pool(
        set().union(*(shapes(r["prompt"]) for r in rows)),
        Counter(r["skeleton"] for r in rows),
        (sig - mu) / sd,
        mu,
        sd,
        tg,
    )


def closeness(pool: Pool, prompt: str, x: Tagged) -> dict[str, float]:
    sh = shapes(prompt)
    out = {"sub": len(sh & pool.shapes) / len(sh) if sh else float("nan")}
    if x.proof is None:
        out |= {"nn": float("nan"), "skel": float("nan")}
        return out
    v = (np.array(signature(x.thm, x.proof)) - pool.mu) / pool.sd
    out["nn"] = -float(np.sqrt(((pool.sig - v) ** 2).sum(1)).min())
    out["skel"] = float(pool.skeletons.get(x.proof["p_skeleton"], 0) > 0)
    return out


CLOSE = {
    "sub": "subformula shapes in the pool",
    "nn": "nearest pool problem (−distance)",
    "skel": "proof skeleton in the pool",
}


# ------------------------------------------------------------------------------------ the joint


@dataclass
class Joint:
    """GRPO − EI on every problem of several sets, with one bootstrap: problems resampled within
    each set, seeds once for every set and both arms (one checkpoint per arm and seed reads every
    set)."""

    names: list[str]  # "set:name"
    set_of: list[str]
    seeds: list[int]
    S: int
    W: F64
    UD: dict[int, F64]
    Dm: dict[int, F64]

    @property
    def T(self) -> int:
        return len(self.names)

    def mask(self, keep: Callable[[str, str], bool]) -> F64:
        return np.array(
            [
                float(keep(s, n.split(":", 1)[1]))
                for s, n in zip(self.set_of, self.names, strict=True)
            ]
        )

    def mean(self, k: int, w: F64, wb: F64 | None = None) -> tuple[float, F64]:
        """Point estimate with weights `w` (T), draws with `wb` (draws x T, default `w`)."""
        ww = self.W * (w if wb is None else wb)
        d = (self.UD[k] * ww).sum(1) / (self.S * np.maximum(ww.sum(1), 1e-12))
        return float((self.Dm[k] * w).sum() / w.sum()), d


def joint(ld: V.Loader, pair: LSP.Pair, draws: int, seed: int = 0) -> Joint | None:
    sets = [s for s in SETS if V.common_seeds(ld, [pair.a, pair.b], s)]
    seeds = LSP.seeds_of(ld, pair, sets)
    if not any(s in HUMAN for s in sets) or not any(s in GENERATED for s in sets) or not seeds:
        return None
    xs = [ld.arm(pair.a, s, seeds) for s in sets]
    ys = [ld.arm(pair.b, s, seeds) for s in sets]
    xa = [a for a in xs if a is not None]
    ya = [a for a in ys if a is not None]
    if len(xa) != len(sets) or len(ya) != len(sets):
        return None
    X, Y = LSP.concat(xa, sets, "a"), LSP.concat(ya, sets, "b")
    W = np.hstack([stats.make_draws(a.T, draws=draws, seed=seed + j).W for j, a in enumerate(xa)])
    U = stats.Draws(W, np.random.default_rng([seed, 1]), "both").seed_weights(X.S)
    UD, Dm = {}, {}
    for k in (1, 5):
        D = X.values(k) - Y.values(k)
        UD[k], Dm[k] = U @ D, D.mean(0)
    set_of = [n.split(":", 1)[0] for n in X.names]
    return Joint(X.names, set_of, seeds, X.S, W, UD, Dm)


def onehot(codes: I64, C: int) -> F64:
    m = np.zeros((len(codes), C))
    ok = codes >= 0
    m[np.flatnonzero(ok), codes[ok]] = 1.0
    return m


def rake(Wh: F64, Wg: F64, Hs: Sequence[F64], Gs: Sequence[F64], iters: int = RAKE_ITERS) -> F64:
    """Generated weights (rows of `Wg`, draws x Tg) whose category shares on every feature (one-hot
    `Gs[j]`, Tg x C) match the human-written shares under `Wh` (one-hot `Hs[j]`)."""
    w = np.ones_like(Wg)
    targets = [(Wh @ H) / Wh.sum(1, keepdims=True) for H in Hs]
    for _ in range(1 if len(Hs) == 1 else iters):
        worst = 0.0
        for tgt, G in zip(targets, Gs, strict=True):
            ww = Wg * w
            cur = (ww @ G) / ww.sum(1, keepdims=True)
            ratio = np.where(cur > 0, tgt / np.maximum(cur, 1e-300), 0.0)
            worst = max(worst, float(np.abs(np.where(cur > 0, ratio - 1, 0)).max()))
            w = w * (ratio @ G.T)
        if worst < 1e-6:
            break
    return w


@dataclass
class Gap:
    section: str
    label: str
    k: int
    n_h: int
    n_h_all: int
    n_g: int
    ess: float
    dh: stats.Estimate
    dg: stats.Estimate
    gap: stats.Estimate


@dataclass(frozen=True)
class Plan:
    section: str
    label: str
    keys: tuple[str, ...] = ()
    only: tuple[tuple[str, int], ...] = ()  # (feature, category index) both sources must have


def gap_after(
    J: Joint, k: int, plan: Plan, codes: dict[str, I64], ncat: dict[str, int]
) -> Gap | None:
    """Human-written − generated GRPO − EI gap, the generated problems raked to the human-written
    mix of `plan.keys` (none: unadjusted), on the problems in `plan.only`. Human problems in a
    category without `MIN_SUPPORT` generated problems (or with a missing feature) are left out."""
    label, keys = plan.label, plan.keys
    human = np.array([LSP.SOURCE[s] == "human" for s in J.set_of])
    inside = np.ones(J.T, dtype=bool)
    for f, c in plan.only:
        inside &= codes[f] == c
    elig_h, elig_g = human & inside, ~human & inside
    n_h_all = int(elig_h.sum())
    for f in keys:
        c = codes[f]
        elig_h &= c >= 0
        elig_g &= c >= 0
    for f in keys:
        c = codes[f]
        sup = np.bincount(c[elig_g], minlength=ncat[f]) >= MIN_SUPPORT
        elig_h &= np.where(c >= 0, sup[np.maximum(c, 0)], False)
    hi, gi = np.flatnonzero(elig_h), np.flatnonzero(elig_g)
    if not len(hi) or not len(gi):
        return None
    Wh, Wg = J.W[:, hi], J.W[:, gi]
    Hs = [onehot(codes[f][hi], ncat[f]) for f in keys]
    Gs = [onehot(codes[f][gi], ncat[f]) for f in keys]
    wg = rake(Wh, Wg, Hs, Gs) if keys else np.ones_like(Wg)
    w0 = rake(np.ones((1, len(hi))), np.ones((1, len(gi))), Hs, Gs)[0] if keys else np.ones(len(gi))
    full = np.zeros(J.T)
    full[gi] = w0
    fullb = np.zeros_like(J.W)
    fullb[:, gi] = wg
    eg, sg = J.mean(k, full, fullb)
    eh, sh = J.mean(k, elig_h.astype(float))
    T = len(hi) + len(gi)
    return Gap(
        plan.section,
        label,
        k,
        len(hi),
        n_h_all,
        len(gi),
        float(w0.sum() ** 2 / (w0**2).sum()),
        LSP._estimate("human", eh, sh, k, len(hi)),
        LSP._estimate("generated", eg, sg, k, len(gi)),
        LSP._estimate("gap", eh - eg, sh - sg, k, T),
    )


def slope(J: Joint, k: int, x: F64, mask: F64) -> stats.Estimate:
    """Within `mask`: least-squares slope of the per-problem GRPO − EI difference on `x` (a
    closeness rank in [0, 1]), per bootstrap draw."""
    d = J.UD[k] / J.S
    w = J.W * mask
    sw = w.sum(1, keepdims=True)
    xm = (w * x).sum(1, keepdims=True) / sw
    dm = (w * d).sum(1, keepdims=True) / sw
    b = ((w * (x - xm) * (d - dm)).sum(1)) / ((w * (x - xm) ** 2).sum(1))
    m = mask > 0
    x0, d0 = x[m], J.Dm[k][m]
    est = float(((x0 - x0.mean()) * (d0 - d0.mean())).sum() / ((x0 - x0.mean()) ** 2).sum())
    return LSP._estimate("slope", est, b, k, int(m.sum()))


def plans() -> list[Plan]:
    """The reweighting contrasts, in figure order."""
    st = [f.key for f in FEATURES if f.group == "statement"]
    pr = [f.key for f in FEATURES if f.group == "proof"]
    yes = FEAT["case"].cats.index("yes")
    out = [Plan("all problems", "unadjusted")]
    out += [Plan("one feature", f.title, (f.key,)) for f in FEATURES if f.group != "case split"]
    out += [
        Plan("several features", "case split + premises + connectives", ("case", "prem", "size")),
        Plan("several features", f"proof features ({len(pr)})", tuple(pr)),
        Plan("several features", f"statement + proof features ({len(st) + len(pr)})", (*st, *pr)),
    ]
    out += [Plan("case-split problems only", "∨E problems: unadjusted", (), (("case", yes),))]
    out += [
        Plan("case-split problems only", f"∨E problems: {f.title}", (f.key,), (("case", yes),))
        for f in FEATURES
        if f.group == "case split"
    ]
    out += [Plan("closeness to the RL pools", CLOSE[m], (f"close_{m}",)) for m in CLOSE]
    return out


# ------------------------------------------------------------------------------------ analysis


@dataclass
class Cell:
    src: str
    feat: str
    cat: str
    n: int
    seeds: list[int]
    est: dict[str, stats.Estimate]


@dataclass
class Result:
    pair: LSP.Pair
    cells: list[Cell] = field(default_factory=lambda: [])
    gaps: list[Gap] = field(default_factory=lambda: [])
    close_cells: list[Cell] = field(default_factory=lambda: [])
    slopes: list[tuple[str, str, stats.Estimate]] = field(default_factory=lambda: [])
    notes: list[str] = field(default_factory=lambda: [])


def source_arms(
    ld: V.Loader, pair: LSP.Pair, src: str
) -> tuple[stats.Arm, stats.Arm, list[int], list[str]] | None:
    group = HUMAN if src == "human" else GENERATED
    sets = [s for s in group if V.common_seeds(ld, [pair.a, pair.b], s)]
    seeds = LSP.seeds_of(ld, pair, sets)
    if not sets or not seeds:
        return None
    xs = [ld.arm(pair.a, s, seeds) for s in sets]
    ys = [ld.arm(pair.b, s, seeds) for s in sets]
    xa = [a for a in xs if a is not None]
    ya = [a for a in ys if a is not None]
    if len(xa) != len(sets) or len(ya) != len(sets):
        return None
    return LSP.concat(xa, sets, src), LSP.concat(ya, sets, src), seeds, sets


def analyse(
    ld: V.Loader,
    pair: LSP.Pair,
    tags: dict[str, dict[str, Tagged]],
    close: dict[str, dict[str, dict[str, float]]],
    draws: int,
) -> Result:
    res = Result(pair)
    qs = {
        m: np.nanquantile([c[m] for s in SETS for c in close[s].values()], [0.25, 0.5, 0.75])
        for m in CLOSE
    }
    for src in ("human", "generated"):
        got = source_arms(ld, pair, src)
        if got is None:
            res.notes.append(f"{src}: no seed with both arms read on every set yet")
            continue
        X, Y, seeds, sets = got
        for f in FEATURES:
            for c in f.cats:
                names = [
                    n for n in X.names if tags[n.split(":")[0]][n.split(":", 1)[1]].cat[f.key] == c
                ]
                if names:
                    res.cells.append(
                        Cell(
                            src,
                            f.key,
                            c,
                            len(names),
                            seeds,
                            LSP.estimate(X, Y, names, draws, (1, 5)),
                        )
                    )
        for m in CLOSE:
            vals = {n: close[n.split(":")[0]][n.split(":", 1)[1]][m] for n in X.names}
            for q in range(4) if m != "skel" else range(2):
                if m == "skel":
                    names = [n for n, v in vals.items() if v == q]
                else:
                    lo = -np.inf if q == 0 else qs[m][q - 1]
                    hi = np.inf if q == 3 else qs[m][q]
                    names = [n for n, v in vals.items() if lo < v <= hi]
                if names:
                    res.close_cells.append(
                        Cell(
                            src,
                            m,
                            str(q),
                            len(names),
                            seeds,
                            LSP.estimate(X, Y, names, draws, (1,)),
                        )
                    )
    J = joint(ld, pair, draws)
    if J is None:
        res.notes.append("no seed shared by both sources yet: no human − generated contrasts")
        return res
    codes: dict[str, I64] = {}
    ncat: dict[str, int] = {}
    for f in FEATURES:
        idx = {c: i for i, c in enumerate(f.cats)}
        cat_of = [
            tags[s][n.split(":", 1)[1]].cat[f.key] for s, n in zip(J.set_of, J.names, strict=True)
        ]
        codes[f.key] = np.array([-1 if c is None else idx[c] for c in cat_of], dtype=np.int64)
        ncat[f.key] = len(f.cats)
    for m in CLOSE:
        v = np.array(
            [close[s][n.split(":", 1)[1]][m] for s, n in zip(J.set_of, J.names, strict=True)]
        )
        if m == "skel":
            codes[f"close_{m}"] = np.where(np.isnan(v), -1, v).astype(np.int64)
            ncat[f"close_{m}"] = 2
        else:
            q = np.searchsorted(qs[m], v, side="left")
            codes[f"close_{m}"] = np.where(np.isnan(v), -1, q).astype(np.int64)
            ncat[f"close_{m}"] = 4
    for k in (1, 5):
        for plan in plans():
            g = gap_after(J, k, plan, codes, ncat)
            if g is not None:
                res.gaps.append(g)
    human = np.array([LSP.SOURCE[s] == "human" for s in J.set_of])
    for m in ("sub", "nn"):
        v = np.array(
            [close[s][n.split(":", 1)[1]][m] for s, n in zip(J.set_of, J.names, strict=True)]
        )
        for src, sm in (("human", human), ("generated", ~human)):
            ok = sm & ~np.isnan(v)
            r = np.zeros(J.T)
            order = np.argsort(np.argsort(v[ok]))
            r[ok] = order / max(1, ok.sum() - 1)
            res.slopes.append((src, m, slope(J, 1, r, ok.astype(float))))
    return res


# ------------------------------------------------------------------------------------- figures


GROUP_TITLE = {
    "statement": "statement",
    "proof": "proof",
    "case split": "∨E detail (∨E proofs only)",
}


def _src_handles() -> list[Any]:
    return [
        Line2D(
            [],
            [],
            color=SRC_COL[s],
            marker=SRC_MK[s],
            ls="none",
            ms=4,
            mec=S.WHITE,
            label=LSP.NAME[s],
        )
        for s in ("human", "generated")
    ]


def _seed_text(seeds: Sequence[int]) -> str:
    return f"seed{'s' if len(seeds) > 1 else ''} {','.join(map(str, seeds))}"


def fig_features(results: Sequence[Result], out: Path) -> None:
    n_rows = sum(len(f.cats) for f in FEATURES) + len(GROUP_TITLE)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 0.75 + 0.112 * n_rows, layout="constrained")
        axs = fig.subplots(1, len(results), sharey=True, squeeze=False)[0]
        ticks: list[float] = []
        labels: list[str] = []
        for j, res in enumerate(results):
            ax: Axes = axs[j]
            y = 0.0
            ticks, labels = [], []
            group = ""
            for f in FEATURES:
                if f.group != group:
                    group = f.group
                    ticks.append(y)
                    labels.append(f"— {GROUP_TITLE[group]} —")
                    y -= 1
                for c in f.cats:
                    ticks.append(y)
                    labels.append(f"{f.title}: {c}" if c == f.cats[0] else c)
                    for src, dy in (("human", 0.18), ("generated", -0.18)):
                        cell = next(
                            (
                                x
                                for x in res.cells
                                if x.src == src and x.feat == f.key and x.cat == c
                            ),
                            None,
                        )
                        if cell is None:
                            continue
                        e = cell.est["d1"]
                        ax.errorbar(
                            [e.est],
                            [y + dy],
                            xerr=[[e.est - e.lo], [e.hi - e.est]],
                            fmt=SRC_MK[src],
                            ms=2.8,
                            color=SRC_COL[src],
                            mec=S.WHITE,
                            mew=0.4,
                            elinewidth=0.7,
                            capsize=0,
                        )
                        ax.text(
                            1.01,
                            y + dy,
                            str(cell.n),
                            transform=ax.get_yaxis_transform(),
                            fontsize=S.FS_SMALL - 1.5,
                            color=SRC_COL[src],
                            va="center",
                            ha="left",
                        )
                    y -= 1
                ax.axhline(y + 0.5, color=S.GRID, lw=0.5)
            ax.axvline(0, color=S.INK, lw=0.6)
            ax.set_xlim(-0.35, 0.45)
            ax.xaxis.set_major_formatter(LSP._points)
            S.light_grid(ax, "x")
            ax.set_ylim(y + 0.5, 0.6)
            seeds = sorted({sd for c in res.cells for sd in c.seeds})
            S.panel_title(ax, "ab"[j], f"{res.pair.title}, {_seed_text(seeds)}", size=S.FS_TICK)
            ax.set_xlabel("Δ pass@1, GRPO − EI (points)")
        axs[0].set_yticks(ticks, labels, fontsize=S.FS_SMALL - 0.5)
        fig.legend(handles=_src_handles(), loc="outside upper center", ncol=2, frameon=False)
        fig.supxlabel(
            "right margins: problems per cell; bars: 95% intervals; proof features from the "
            "reference proof (textbook72: shortest found proof)",
            fontsize=S.FS_SMALL,
        )
        S.save(fig, out / "fig_source_split_features")
        S.close(fig)


def fig_reweight(results: Sequence[Result], out: Path) -> None:
    order = [(p.section, p.label) for p in plans()]
    rows: list[tuple[str, str] | str] = []
    for sec, lab in order:
        if not rows or (isinstance(rows[-1], tuple) and rows[-1][0] != sec):
            rows.append(sec)
        rows.append((sec, lab))
    rows = rows[1:]  # no header above the unadjusted gap
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 0.85 + 0.135 * len(rows), layout="constrained")
        axs = fig.subplots(1, len(results), sharey=True, squeeze=False)[0]
        for j, res in enumerate(results):
            ax: Axes = axs[j]
            gaps = {(g.section, g.label): g for g in res.gaps if g.k == 1}
            base = gaps.get(order[0])
            for i, row in enumerate(rows):
                if isinstance(row, str):
                    ax.axhline(-i + 0.5, color=S.GRID, lw=0.5)
                    continue
                g = gaps.get(row)
                if g is None:
                    continue
                e = g.gap
                first = row == order[0] or row[1] == "∨E problems: unadjusted"
                col = S.INK if first or g.ess >= MIN_ESS else S.mix(S.INK2, S.WHITE, 0.55)
                ax.errorbar(
                    [e.est],
                    [-i],
                    xerr=[[e.est - e.lo], [e.hi - e.est]],
                    fmt="o",
                    ms=3,
                    color=col,
                    mfc=col if first else S.WHITE,
                    elinewidth=0.8,
                    capsize=0,
                )
                note = f"{g.n_h}/{g.n_h_all}" + ("" if first else f", {g.ess:.0f}")
                ax.text(
                    1.01,
                    -i,
                    note,
                    transform=ax.get_yaxis_transform(),
                    fontsize=S.FS_SMALL - 1.5,
                    color=S.REF,
                    va="center",
                )
            if base is not None:
                ax.axvline(base.gap.est, color=S.REF, lw=0.5, ls=(0, (2, 2)))
            ax.axvline(0, color=S.INK, lw=0.6)
            ax.set_ylim(-len(rows) + 0.5, 0.6)
            ax.xaxis.set_major_formatter(LSP._points)
            S.light_grid(ax, "x")
            S.panel_title(ax, "ab"[j], res.pair.title, size=S.FS_TICK)
            ax.set_xlabel("gap (points)")
        labels = [f"— {r} —" if isinstance(r, str) else r[1] for r in rows]
        axs[0].set_yticks([-i for i in range(len(rows))], labels, fontsize=S.FS_SMALL - 0.5)
        fig.supxlabel(
            "gap = human-written − generated Δ pass@1, generated problems reweighted to the "
            "human-written mix of the feature(s) on the left\nright margins: human-written "
            "problems kept / eligible, effective number of generated problems (grey: under "
            f"{MIN_ESS}, uninformative);\ndashed: unadjusted gap; filled: unadjusted; bars: 95% "
            "intervals",
            fontsize=S.FS_SMALL,
        )
        S.save(fig, out / "fig_source_split_reweight")
        S.close(fig)


def fig_closeness(results: Sequence[Result], out: Path) -> None:
    ms = list(CLOSE)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 0.9 + 1.45 * len(results), layout="constrained")
        axs = fig.subplots(len(results), len(ms), sharey="row", squeeze=False)
        for r, res in enumerate(results):
            for j, m in enumerate(ms):
                ax: Axes = axs[r][j]
                ax.axhline(0, color=S.INK, lw=0.6)
                for src, dx in (("human", -0.08), ("generated", 0.08)):
                    cells = sorted(
                        (c for c in res.close_cells if c.src == src and c.feat == m),
                        key=lambda c: int(c.cat),
                    )
                    if not cells:
                        continue
                    x = [int(c.cat) + dx for c in cells]
                    e = [c.est["d1"] for c in cells]
                    ax.errorbar(
                        x,
                        [v.est for v in e],
                        yerr=[[v.est - v.lo for v in e], [v.hi - v.est for v in e]],
                        fmt=SRC_MK[src],
                        ls="-",
                        lw=0.9,
                        ms=3,
                        color=SRC_COL[src],
                        mec=S.WHITE,
                        mew=0.4,
                        elinewidth=0.7,
                        capsize=0,
                    )
                    for xx, c, v in zip(x, cells, e, strict=True):
                        up = src == "human"
                        ax.text(
                            xx,
                            v.hi if up else v.lo,
                            str(c.n),
                            fontsize=S.FS_SMALL - 1,
                            color=SRC_COL[src],
                            ha="center",
                            va="bottom" if up else "top",
                        )
                ax.margins(y=0.14)
                if m == "skel":
                    ax.set_xticks([0, 1], ["not in pool", "in pool"])
                    ax.set_xlim(-0.5, 1.5)
                else:
                    ax.set_xticks([0, 1, 2, 3], ["Q1\nfar", "Q2", "Q3", "Q4\nclose"])
                    ax.set_xlim(-0.5, 3.5)
                ax.yaxis.set_major_formatter(LSP._points)
                S.light_grid(ax, "y")
                S.panel_title(ax, "abcdef"[3 * r + j], CLOSE[m], size=S.FS_SMALL)
            axs[r][0].set_ylabel(f"{res.pair.title}\nΔ pass@1 (points)", fontsize=S.FS_SMALL)
        handles = [
            Line2D([], [], color=SRC_COL[s], marker=SRC_MK[s], ms=4, mec=S.WHITE, label=LSP.NAME[s])
            for s in ("human", "generated")
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
        fig.supxlabel(
            "closeness to the RL curriculum pools (L2, L3; quartiles over all four sets); numbers: "
            "problems per point; bars: 95% intervals",
            fontsize=S.FS_SMALL,
        )
        S.save(fig, out / "fig_source_split_closeness")
        S.close(fig)


# -------------------------------------------------------------------------------------- tables


def share_table(
    tags: dict[str, dict[str, Tagged]], pool: Pool, close: dict[str, dict[str, dict[str, float]]]
) -> list[str]:
    cols = [*SETS, "pool"]
    md = [
        "## Feature shares per set (% of the set's problems with the feature; pool = L2 + L3)",
        "",
    ]
    md += [
        "| feature | category | "
        + " | ".join(LSP.NAME[s] if s != "pool" else "RL pools" for s in cols)
        + " |"
    ]
    md += ["|---|---|" + "---|" * len(cols)]
    for f in FEATURES:
        for c in f.cats:
            cells = []
            for s in cols:
                xs = pool.tagged if s == "pool" else list(tags[s].values())
                have = [x for x in xs if x.cat[f.key] is not None]
                n = sum(x.cat[f.key] == c for x in have)
                cells.append(f"{100 * n / len(have):.0f}" if have else "—")
            md.append(f"| {f.title} | {c} | " + " | ".join(cells) + " |")
    md += ["", "Closeness to the pools, mean per set:", ""]
    md += [
        "| measure | " + " | ".join(LSP.NAME[s] for s in SETS) + " |",
        "|---|" + "---|" * len(SETS),
    ]
    for m, title in CLOSE.items():
        vals = [np.nanmean([c[m] for c in close[s].values()]) for s in SETS]
        fmt = "{:.2f}" if m != "skel" else "{:.0%}"
        md.append(f"| {title} | " + " | ".join(fmt.format(v) for v in vals) + " |")
    md += [""]
    return md


def tables(
    results: Sequence[Result],
    checks: list[str],
    shares: list[str],
    ld: V.Loader,
    draws: int,
) -> list[str]:
    md = [
        "# What in source drives GRPO − EI? (generated by `source_split.py`; do not edit by hand)",
        "",
        "Thinking models, GRPO recipe v2, plain reads at B64 (ynez). Δ = GRPO arm − EI arm in",
        f"points of pass@k; 95% hierarchical-bootstrap intervals ({draws} draws). human-written =",
        "textbook72 + Charles149; generated = 13–20 + 21–30. Figures `fig_source_split_*` in",
        "`experiment-summaries/2026-10-05-thinking-traces/figures/ynez/`.",
        "",
    ]
    md += checks + shares
    for i, res in enumerate(results, 1):
        p = res.pair
        md += [f"## {i}. {p.title}", "", f"### {i}a. Δ by feature, within each source", ""]
        md += [
            "| feature | category | source | n | seeds | Δ pass@1 | Δ pass@5 |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in FEATURES:
            for c in f.cats:
                for src in ("human", "generated"):
                    cell = next(
                        (x for x in res.cells if x.src == src and x.feat == f.key and x.cat == c),
                        None,
                    )
                    if cell is None:
                        continue
                    md.append(
                        f"| {f.title} | {c} | {src} | {cell.n} | "
                        f"{','.join(map(str, cell.seeds))} | "
                        f"{V.fmt(cell.est['d1'], True)} | {V.fmt(cell.est['d5'], True)} |"
                    )
        md += ["", f"### {i}b. Human-written − generated gap after reweighting", ""]
        md += [
            "Generated problems reweighted to the human-written mix of the listed feature(s)",
            f"(categories need >= {MIN_SUPPORT} generated problems; human-written problems outside",
            "them are left out: 'human kept'). ESS: effective number of generated problems.",
            "",
            "| problems | reweighted on | k | human kept | ESS | Δ human-written | "
            "Δ generated (reweighted) | human − generated | p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for g in res.gaps:
            pb = g.gap.p_boot
            md.append(
                f"| {g.section} | {g.label} | pass@{g.k} | {g.n_h}/{g.n_h_all} | {g.ess:.0f} | "
                f"{V.fmt(g.dh, True)} | {V.fmt(g.dg, True)} | {V.fmt(g.gap, True)} | "
                f"{'—' if pb is None else f'{pb:.3f}'} |"
            )
        md += ["", f"### {i}c. Closeness to the RL pools", ""]
        md += [
            "Quartiles over all four sets (Q1 farthest); `skel`: 0 = skeleton not in the pool.",
            "",
            "| measure | bin | source | n | Δ pass@1 |",
            "|---|---|---|---|---|",
        ]
        for c in res.close_cells:
            md.append(
                f"| {CLOSE[c.feat]} | {'Q' + str(int(c.cat) + 1) if c.feat != 'skel' else c.cat} | "
                f"{c.src} | {c.n} | {V.fmt(c.est['d1'], True)} |"
            )
        md += [
            "",
            "Within-source slope of Δ pass@1 on the closeness rank (0 = farthest, 1 = closest);",
            "the hypothesis that EI gains with closeness predicts a negative slope.",
            "",
            "| source | measure | n | slope (points per full rank range) | p |",
            "|---|---|---|---|---|",
        ]
        for src, m, e in res.slopes:
            md.append(
                f"| {src} | {CLOSE[m]} | {e.T} | {V.fmt(e, True)} | "
                f"{'—' if e.p_boot is None else f'{e.p_boot:.3f}'} |"
            )
        if res.notes:
            md += ["", *[f"- {n}" for n in res.notes]]
        md += [""]
    if ld.missing:
        runs = Counter(V._run_of(p) for p in set(ld.missing))
        md += [f"Rows not there yet ({sum(runs.values())} files; run: files):", ""]
        md += [", ".join(f"`{r}`: {n}" for r, n in sorted(runs.items())), ""]
    return md


def proof_checks(
    ld: V.Loader, probs: dict[str, dict[str, Problem]], tags: dict[str, dict[str, Tagged]]
) -> list[str]:
    """How the textbook72 proofs were obtained, and how far features from a shortest found proof
    agree with features from the reference proof (on Charles149, which has both)."""
    md = ["## Proofs behind the proof features", ""]
    tb = probs["tb72"]
    md.append(
        f"- textbook72: {sum(p.proof is not None for p in tb.values())} of {len(tb)} problems "
        "have a shortest found proof (no reference proofs exist); the rest get statement features "
        "only."
    )
    found = found_proofs(ld, "charles149")
    agree: dict[str, list[bool]] = {f.key: [] for f in FEATURES if f.group == "proof"}
    for n, p in probs["charles149"].items():
        if n not in found:
            continue
        alt = tag(p.prompt, found[n])
        for k in agree:
            agree[k].append(alt.cat[k] == tags["charles149"][n].cat[k])
    n_ok = len(next(iter(agree.values())))
    md.append(
        f"- Charles149, {n_ok} problems solved by a headline arm: a proof feature taken from the "
        "shortest found proof agrees with the reference proof's for "
        + ", ".join(f"{FEAT[k].title} {100 * np.mean(v):.0f}%" for k, v in agree.items())
        + "."
    )
    md += [""]
    return md


# ---------------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=stats.DRAWS)
    ap.add_argument("--v2", default=str(V.E / "v2"), help="the v2 reads")
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures/ynez")
    ap.add_argument(
        "--table",
        default="code/experiments/current/thinking_traces_20261005/matched_eval/source_split_ynez.md",
    )
    a = ap.parse_args(argv)
    ld = V.Loader(Path(a.v2), V.subsets())
    probs = problems(ld)
    tags = {s: {n: tag(p.prompt, p.proof) for n, p in probs[s].items()} for s in SETS}
    pool = load_pool()
    close = {
        s: {n: closeness(pool, probs[s][n].prompt, x) for n, x in tags[s].items()} for s in SETS
    }
    results = [analyse(ld, pair, tags, close, a.draws) for pair in LSP.PAIRS]
    out = Path(a.out)
    fig_features(results, out)
    fig_reweight(results, out)
    fig_closeness(results, out)
    md = tables(results, proof_checks(ld, probs, tags), share_table(tags, pool, close), ld, a.draws)
    Path(a.table).write_text("\n".join(md) + "\n")
    print(f"wrote {a.table} and {out}/fig_source_split_*")


if __name__ == "__main__":
    main()
