"""Strategy shares for the corpus, K12, Charles's reference proofs and every read; base vs RL,
RL-only problems, emergence against R, and Dan's EI ladders. Units in PREREG.md.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \\
        code.experiments.current.proof_strategies_grpo_20261101.analyze [--draws 2000]

Reads the label files that `classify` wrote under `common.OUT`; writes numbers.json and
numbers.md next to this file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from . import common as C
from .classify import READS, labels, read_name

Lab = dict[str, Any]

#: The tracked strategies (PREREG.md): id -> (name, predicate on one proof's labels).
TRACK: dict[str, tuple[str, Callable[[Lab], bool]]] = {
    "work_red": ("proof by contradiction, working", lambda x: "contradiction.working" in x["sub"]),
    "free_red": ("  of which free", lambda x: "contradiction.free" in x["sub"]),
    "templ_red": (
        "proof by contradiction, template only",
        lambda x: "contradiction" in x["u"] and "contradiction.working" not in x["sub"],
    ),
    "mt": ("modus tollens", lambda x: "mod_tollens" in x["u"]),
    "ds": ("disjunctive syllogism", lambda x: "disj_syll" in x["u"]),
    "work_neg": ("proof of a negation, working", lambda x: "neg_intro.working" in x["sub"]),
    "gen_cases": ("proof by cases, genuine", lambda x: "cases.genuine" in x["sub"]),
    "lem": ("excluded middle", lambda x: "lem" in x["u"]),
    "dne": ("double-negation elimination", lambda x: "dne" in x["u"]),
    "efq": ("ex falso", lambda x: "ex_falso" in x["u"]),
    "any_dn": (
        "any classical step (#10-#12)",
        lambda x: bool({"contradiction", "dne", "lem"} & set(x["u"])),
    ),
}
RISING = ["work_red", "mt", "ds", "work_neg", "gen_cases"]
OPENINGS = ["forward", "goal_intro", "goal_cases", "goal_reductio", "other_box", "none"]
YUE = 1.2e-4


# ------------------------------------------------------------------------------ loading
def load_read(st: str, stem: str, seed: int, k: int) -> dict[str, Any]:
    with gzip.open(C.OUT / "reads" / f"{read_name(st, stem, seed, k)}.json.gz", "rt") as f:
        return dict(json.load(f))


def load_labels(name: str) -> list[Lab]:
    with gzip.open(C.OUT / "labels" / f"{name}.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in f]


class Read:
    """Per-problem counts of one read: n_ok, k, attempts per strategy, distinct proofs."""

    def __init__(self, rows: dict[str, Any], keys: list[str]):
        self.names = sorted(rows)
        self.k = np.array([rows[n]["k"] for n in self.names], float)
        self.n_ok = np.array([rows[n]["n_ok"] for n in self.names], float)
        self.c: dict[str, np.ndarray] = {}
        self.d: dict[str, np.ndarray] = {}
        self.ndist = np.array([len(rows[n]["proofs"]) for n in self.names], float)
        for key in keys:
            f = TRACK[key][1] if key in TRACK else _extra(key)
            self.c[key] = np.array(
                [sum(c for _, c, lab in rows[n]["proofs"] if f(lab)) for n in self.names], float
            )
            self.d[key] = np.array(
                [sum(1 for _, _, lab in rows[n]["proofs"] if f(lab)) for n in self.names], float
            )
        # order features, attempt-weighted sums
        self.dead = np.array(
            [sum(c * (lab["L"] - lab["Lp"]) for _, c, lab in rows[n]["proofs"]) for n in self.names]
        )
        self.Lp = np.array(
            [sum(c * lab["Lp"] for _, c, lab in rows[n]["proofs"]) for n in self.names]
        )
        self.gf = np.array(
            [sum(c * lab["gf"] for _, c, lab in rows[n]["proofs"]) for n in self.names]
        )
        self.fpl = np.array(
            [sum(c * lab["fpl"] for _, c, lab in rows[n]["proofs"]) for n in self.names]
        )
        self.fp = np.array(
            [sum(c * lab["fp"] for _, c, lab in rows[n]["proofs"]) for n in self.names]
        )
        self.prompts = [rows[n]["prompt"] for n in self.names]
        self.op = {
            o: np.array(
                [sum(c for _, c, lab in rows[n]["proofs"] if lab["op"] == o) for n in self.names],
                float,
            )
            for o in OPENINGS
        }

    def idx(self, names: list[str]) -> np.ndarray:
        pos = {n: i for i, n in enumerate(self.names)}
        return np.array([pos[n] for n in names], int)


def _extra(key: str) -> Callable[[Lab], bool]:
    """'u:<strategy>' -> uses that strategy."""
    s = key.split(":", 1)[1]
    return lambda x: s in x["u"]


ALLU = [
    f"u:{s}"
    for s in (
        "direct",
        "and_intro",
        "cond_proof",
        "or_intro",
        "cases",
        "disj_syll",
        "neg_intro",
        "mod_tollens",
        "ex_falso",
        "contradiction",
        "dne",
        "lem",
        "lemma_reuse",
    )
]
KEYS = list(TRACK) + ALLU


# ------------------------------------------------------------------------------ metrics
def unit_A(r: Read, key: str, sel: np.ndarray | None = None) -> float:
    s = slice(None) if sel is None else sel
    return float(100 * np.mean(r.c[key][s] / r.k[s]))


def unit_B(r: Read, key: str, sel: np.ndarray | None = None) -> float:
    s = np.arange(len(r.names)) if sel is None else sel
    s = s[r.n_ok[s] > 0]
    return float(100 * np.mean(r.c[key][s] / r.n_ok[s])) if len(s) else float("nan")


def unit_Bpool(r: Read, key: str, sel: np.ndarray | None = None) -> float:
    s = slice(None) if sel is None else sel
    return float(100 * r.c[key][s].sum() / max(r.n_ok[s].sum(), 1))


def unit_C(r: Read, key: str, sel: np.ndarray | None = None) -> float:
    s = slice(None) if sel is None else sel
    return float(100 * np.mean(r.c[key][s] > 0))


def unit_D(r: Read, key: str, sel: np.ndarray | None = None) -> float:
    """share of distinct (ND) proofs, pooled: Dan's unit for the EI ladders."""
    s = slice(None) if sel is None else sel
    return float(100 * r.d[key][s].sum() / max(r.ndist[s].sum(), 1))


def boot_diff(
    fa: Callable[[np.ndarray], float],
    fb: Callable[[np.ndarray], float],
    n: int,
    draws: int,
    seed: int = 0,
) -> tuple[float, float, float]:
    """fa(sel) - fb(sel) with a 95% problem-bootstrap interval (both arms share the draw)."""
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    est = fa(full) - fb(full)
    vals = []
    for _ in range(draws):
        sel = rng.integers(0, n, n)
        v = fa(sel) - fb(sel)
        if not math.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def b_matched(sel: np.ndarray, x: Read, b: Read, r: Read, key: str) -> float:
    """unit B of `x` on the problems of `sel` that both `b` and `r` solve."""
    sel = sel[(b.n_ok[sel] > 0) & (r.n_ok[sel] > 0)]
    return unit_B(x, key, sel)


def corpus_table(rows: list[Lab]) -> dict[str, float]:
    n = len(rows)
    out = {"n": n}
    for key in KEYS:
        f = TRACK[key][1] if key in TRACK else _extra(key)
        out[key] = 100 * sum(f(r) for r in rows) / n
    out["dead_lines_per_proof"] = sum(r["L"] - r["Lp"] for r in rows) / n
    out["Lp_mean"] = sum(r["Lp"] for r in rows) / n
    out["gf"] = 100 * sum(r["gf"] for r in rows) / n
    out["box_free"] = 100 * sum(r["bf"] for r in rows) / n
    for o in OPENINGS:
        out[f"op:{o}"] = 100 * sum(r["op"] == o for r in rows) / n
    return out


def read_table(r: Read, sel: np.ndarray | None = None) -> dict[str, Any]:
    s = slice(None) if sel is None else sel
    ok = r.n_ok[s].sum()
    return {
        "n_problems": int(len(r.names) if sel is None else len(sel)),
        "solved_pct": float(100 * np.mean(r.n_ok[s] > 0)),
        "attempt_success_pct": float(100 * np.mean(r.n_ok[s] / r.k[s])),
        "accepted_attempts": int(ok),
        "distinct_proofs": int(r.ndist[s].sum()),
        "A": {key: unit_A(r, key, sel) for key in KEYS},
        "B": {key: unit_B(r, key, sel) for key in KEYS},
        "Bpool": {key: unit_Bpool(r, key, sel) for key in KEYS},
        "C": {key: unit_C(r, key, sel) for key in KEYS},
        "D": {key: unit_D(r, key, sel) for key in KEYS},
        "dead_per_proof": float(r.dead[s].sum() / max(ok, 1)),
        "Lp_mean": float(r.Lp[s].sum() / max(ok, 1)),
        "gf_pct": float(100 * r.gf[s].sum() / max(ok, 1)),
        "fwd_prefix_live": float(r.fpl[s].sum() / max(ok, 1)),
        "fwd_prefix": float(r.fp[s].sum() / max(ok, 1)),
        "opening_B": {o: unit_Bop(r, o, sel) for o in OPENINGS},
    }


def unit_Bop(r: Read, o: str, sel: np.ndarray | None = None) -> float:
    """opening move o: share of accepted proofs, macro over solved problems (unit B)."""
    s = np.arange(len(r.names)) if sel is None else sel
    s = s[r.n_ok[s] > 0]
    return float(100 * np.mean(r.op[o][s] / r.n_ok[s])) if len(s) else float("nan")


# ------------------------------------------------------------------------------ curve fits
def fit_compare(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """4-parameter logistic in x = log10 R against a straight line; Gaussian AIC."""
    n = len(x)
    A = np.vstack([np.ones(n), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rss_lin = float(np.sum((A @ coef - y) ** 2))
    best = (math.inf, 0.0, 0.0, 0.0, 0.0)
    for m in np.linspace(x.min(), x.max(), 161):
        for s in np.geomspace(0.03, 3.0, 80):
            g = 1 / (1 + np.exp(-(x - m) / s))
            B = np.vstack([np.ones(n), g]).T
            cf, *_ = np.linalg.lstsq(B, y, rcond=None)
            rss = float(np.sum((B @ cf - y) ** 2))
            if rss < best[0]:
                best = (rss, float(m), float(s), float(cf[0]), float(cf[1]))
    rss_log, m, s, lo, amp = best
    aic = lambda rss, p: n * math.log(max(rss, 1e-12) / n) + 2 * p  # noqa: E731
    return {
        "aic_linear": aic(rss_lin, 2),
        "aic_logistic": aic(rss_log, 4),
        "delta_aic": aic(rss_lin, 2) - aic(rss_log, 4),
        "mid_log10R": m,
        "mid_R": 10**m,
        "scale_decades": s,
        "lo": lo,
        "hi": lo + amp,
        "slope_per_decade": float(coef[1]),
        "midpoint_inside": bool(x.min() < m < x.max()),
    }


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args(argv)
    out: dict[str, Any] = {"units": "see PREREG.md; % unless noted", "fork": C.FORK_COMMIT}
    TH = json.loads((C.OUT / "theorems.json").read_text())

    # ---- Q1: corpora
    corpus = load_labels("corpus_s1g3")
    k12 = load_labels("k12")
    with open(C.CHARLES_CLEAN) as f:
        charles = [json.loads(x) for x in f]
    ref = [labels(r["reference_proof"], r["prompt"]) for r in charles]
    assert all(x is not None for x in ref)
    out["q1"] = {
        "corpus_s1g3": corpus_table(corpus),
        "k12": corpus_table(k12),
        "charles149_reference": corpus_table([x for x in ref if x is not None]),
        "charles149_classical_only_pct": 100
        * sum(TH[r["prompt"]]["classical"] for r in charles)
        / len(charles),
        "charles149_classical_only_field_agrees": sum(
            TH[r["prompt"]]["classical"] == r["classical_only"] for r in charles
        ),
    }
    with gzip.open(C.OUT / "labels/corpus_theorems.jsonl.gz", "rt") as f:
        cth = {r["id"]: r for r in map(json.loads, f)}
    ccl = [r for r in corpus if cth[r["id"]]["classical"]]
    cin = [r for r in corpus if not cth[r["id"]]["classical"]]
    out["q1"]["corpus_classical_only"] = corpus_table(ccl)
    out["q1"]["corpus_intuitionistic"] = corpus_table(cin)
    out["q1"]["corpus_classical_only_pct"] = 100 * len(ccl) / len(corpus)
    pools = {}
    for nm in ("L2", "L3", "L3_3p"):
        rows = load_labels(f"pool_{nm}")
        t = corpus_table([r for r in rows if "fail" not in r])
        t["classical_only_pct"] = 100 * sum(r["thm"]["classical"] for r in rows) / len(rows)
        pools[nm] = t
    out["q1"]["grpo_pools"] = pools
    # the corpus by source: same proof as the K12 generator's or a different search proof
    stats = json.loads((C.TRACES / "e256.stats.json").read_text())
    out["q1"]["corpus_same_as_ref"] = stats["counts"]["same_as_ref"]

    # ---- Q2: charles149 per checkpoint
    R = C.compute_R()
    reads: dict[tuple[str, str, int], Read] = {}
    for st, stem, seed, k in READS:
        p = C.OUT / "reads" / f"{read_name(st, stem, seed, k)}.json.gz"
        if p.exists():
            reads[(st, stem, seed)] = Read(load_read(st, stem, seed, k), KEYS)
    base = reads[("charles149", "s1g3", 0)]
    n = len(base.names)
    q2: dict[str, Any] = {}
    for stem in C.CHAIN:
        r = reads[("charles149", stem, 0)]
        assert r.names == base.names
        row = read_table(r)
        row["R"] = R.get(stem)
        both = np.where((base.n_ok > 0) & (r.n_ok > 0))[0]
        row["matched_n"] = len(both)
        row["B_matched"] = {key: unit_B(r, key, both) for key in KEYS}
        row["B_matched_base"] = {key: unit_B(base, key, both) for key in KEYS}
        row["opening_B_matched"] = {o: unit_Bop(r, o, both) for o in OPENINGS}
        row["opening_B_matched_base"] = {o: unit_Bop(base, o, both) for o in OPENINGS}
        if stem != "s1g3":
            row["diff_A"] = {
                key: boot_diff(
                    lambda s, r=r, key=key: unit_A(r, key, s),
                    lambda s, key=key: unit_A(base, key, s),
                    n,
                    a.draws,
                )
                for key in RISING + ["templ_red", "any_dn"]
            }
            row["diff_B_matched"] = {
                key: boot_diff(
                    lambda s, r=r, key=key: b_matched(s, r, base, r, key),
                    lambda s, r=r, key=key: b_matched(s, base, base, r, key),
                    n,
                    a.draws,
                )
                for key in RISING + ["templ_red", "any_dn"]
            }
        q2[stem] = row
    out["q2_charles149"] = q2

    # seeds 0-2 (base s and RL end s), pooled as means over seeds
    seeds = {}
    for stem in ("s1g3", C.END):
        rs = [reads[("charles149", stem, s)] for s in (0, 1, 2)]
        seeds[stem] = {
            "A": {key: float(np.mean([unit_A(x, key) for x in rs])) for key in KEYS},
            "B": {key: float(np.mean([unit_B(x, key) for x in rs])) for key in KEYS},
            "C": {key: float(np.mean([unit_C(x, key) for x in rs])) for key in KEYS},
            "per_seed_A": {key: [unit_A(x, key) for x in rs] for key in RISING},
            "per_seed_B": {key: [unit_B(x, key) for x in rs] for key in RISING},
            "dead_per_proof": [read_table(x)["dead_per_proof"] for x in rs],
            "gf_pct": [read_table(x)["gf_pct"] for x in rs],
            "Lp_mean": [read_table(x)["Lp_mean"] for x in rs],
        }
    out["q2_seeds"] = seeds

    # by theorem class: classical-only problems vs the rest
    cl = np.array([TH[p]["classical"] for p in base.prompts])
    out["q2_classical"] = {}
    for st in ("charles149", "t1320s250"):
        b0 = reads[(st, "s1g3", 0)]
        clx = np.array([TH[p]["classical"] for p in b0.prompts])
        out["q2_classical"][st] = {
            "n_classical_only": int(clx.sum()),
            **{
                stem: {
                    "classical_only": read_table(reads[(st, stem, 0)], np.where(clx)[0]),
                    "intuitionistic": read_table(reads[(st, stem, 0)], np.where(~clx)[0]),
                }
                for stem in ("s1g3", "grpo_think_vs_L2", C.END)
            },
        }
    # t1320s250: matched problems, base vs L3 end
    tb, te = reads[("t1320s250", "s1g3", 0)], reads[("t1320s250", C.END, 0)]
    both_t = np.where((tb.n_ok > 0) & (te.n_ok > 0))[0]
    nt = len(tb.names)

    def bmt(sl: np.ndarray, x: Read, key: str) -> float:
        sl = sl[(tb.n_ok[sl] > 0) & (te.n_ok[sl] > 0)]
        return unit_B(x, key, sl)

    out["q2_t1320s250"] = {
        "base": read_table(tb),
        "end": read_table(te),
        "matched_n": len(both_t),
        "B_matched_base": {key: unit_B(tb, key, both_t) for key in KEYS},
        "B_matched_end": {key: unit_B(te, key, both_t) for key in KEYS},
        "opening_B_matched_base": {o: unit_Bop(tb, o, both_t) for o in OPENINGS},
        "opening_B_matched_end": {o: unit_Bop(te, o, both_t) for o in OPENINGS},
        "diff_B_matched": {
            key: boot_diff(
                lambda sl, key=key: bmt(sl, te, key),
                lambda sl, key=key: bmt(sl, tb, key),
                nt,
                a.draws,
            )
            for key in RISING + ["templ_red", "any_dn"]
        },
        "diff_A": {
            key: boot_diff(
                lambda sl, key=key: unit_A(te, key, sl),
                lambda sl, key=key: unit_A(tb, key, sl),
                nt,
                a.draws,
            )
            for key in RISING + ["templ_red", "any_dn"]
        },
    }

    # RL-only problems: seed 0 at k1024, and pooled over seeds 0-2
    rl = reads[("charles149", C.END, 0)]
    ro = np.where((base.n_ok == 0) & (rl.n_ok > 0))[0]
    bo = np.where((base.n_ok > 0) & (rl.n_ok == 0))[0]
    bs = np.where(base.n_ok > 0)[0]
    bsum = sum(reads[("charles149", "s1g3", s)].n_ok for s in (0, 1, 2))
    rsum = sum(reads[("charles149", C.END, s)].n_ok for s in (0, 1, 2))
    ro3 = np.where((bsum == 0) & (rsum > 0))[0]
    never = np.where(base.n_ok == 0)[0]
    out["q2_rl_only"] = {
        "seed0": {
            "n": len(ro),
            "base_only": len(bo),
            "names": [base.names[i] for i in ro],
            "classical_only": int(cl[ro].sum()),
            "rl_on_rl_only": read_table(rl, ro),
            "rl_on_base_solved": read_table(rl, bs),
            "base_on_base_solved": read_table(base, bs),
            "theorem_level": {
                key: int(sum(TH[base.prompts[i]][key] for i in ro))
                for key in ("classical", "dneg", "inconsistent")
            },
        },
        "pooled3": {
            "n": len(ro3),
            "names": [base.names[i] for i in ro3],
            "classical_only": int(cl[ro3].sum()),
            "rl_seeds_on_rl_only": [
                read_table(reads[("charles149", C.END, s)], ro3) for s in (0, 1, 2)
            ],
        },
        "set_classical_only_pct": float(100 * cl.mean()),
        "base_never_n": len(never),
    }

    # ---- Q3: emergence vs R (charles149 seed-0 chain, and t1320s250)
    teaser = json.loads(
        (C.REPO / "code/experiments/current/high_rl_passk_20261030/teaser_numbers.json").read_text()
    )
    gaps = {c["checkpoint"]: (c["gap1"][0], c["gapK"][0]) for c in teaser["checkpoints"]}
    q3: dict[str, Any] = {}
    for st in ("charles149", "t1320s250"):
        b = reads[(st, "s1g3", 0)]
        nev = np.where(b.n_ok == 0)[0]
        pts = []
        for stem in C.CHAIN[1:]:
            r = reads.get((st, stem, 0))
            if r is None:
                continue
            assert r.names == b.names
            pts.append(
                {
                    "stem": stem,
                    "R": R[stem],
                    "A": {
                        key: unit_A(r, key) for key in RISING + ["templ_red", "any_dn", "free_red"]
                    },
                    "B": {
                        key: unit_B(r, key) for key in RISING + ["templ_red", "any_dn", "free_red"]
                    },
                    "C": {key: unit_C(r, key) for key in RISING + ["any_dn"]},
                    "never_rate": float(100 * np.mean(r.n_ok[nev] / r.k[nev])),
                    "never_solved_pct": float(100 * np.mean(r.n_ok[nev] > 0)),
                    "pass1": float(100 * np.mean(r.n_ok / r.k)),
                    "gf_pct": read_table(r)["gf_pct"],
                    "dead_per_proof": read_table(r)["dead_per_proof"],
                    "opening_B": {o: unit_Bop(r, o) for o in OPENINGS},
                    "gap1": gaps.get(stem, (None, None))[0] if st == "charles149" else None,
                    "gapK": gaps.get(stem, (None, None))[1] if st == "charles149" else None,
                }
            )
        x = np.log10([p["R"] for p in pts])
        fits = {}
        for key in RISING + ["any_dn"]:
            fits[f"A:{key}"] = fit_compare(x, np.array([p["A"][key] for p in pts]))
            fits[f"B:{key}"] = fit_compare(x, np.array([p["B"][key] for p in pts]))
        fits["never_rate"] = fit_compare(x, np.array([p["never_rate"] for p in pts]))
        fits["pass1"] = fit_compare(x, np.array([p["pass1"] for p in pts]))
        base_lv = {
            "A": {key: unit_A(b, key) for key in RISING + ["templ_red", "any_dn", "free_red"]},
            "B": {key: unit_B(b, key) for key in RISING + ["templ_red", "any_dn", "free_red"]},
            "C": {key: unit_C(b, key) for key in RISING + ["any_dn"]},
            "never_n": len(nev),
            "pass1": float(100 * np.mean(b.n_ok / b.k)),
            "gf_pct": read_table(b)["gf_pct"],
            "dead_per_proof": read_table(b)["dead_per_proof"],
            "opening_B": {o: unit_Bop(b, o) for o in OPENINGS},
        }
        sp = {}
        if st == "charles149":
            for key in RISING + ["any_dn"]:
                da = [p["A"][key] - base_lv["A"][key] for p in pts]
                sp[key] = {
                    "rho_gap1": spearman(da, [p["gap1"] for p in pts]),
                    "rho_gapK": spearman(da, [p["gapK"] for p in pts]),
                }
            nv = [p["never_rate"] for p in pts]
            sp["never_rate"] = {
                "rho_gap1": spearman(nv, [p["gap1"] for p in pts]),
                "rho_gapK": spearman(nv, [p["gapK"] for p in pts]),
            }
        # the pre-registered selection rule for panel (c)
        fold = {
            key: (pts[[p["stem"] for p in pts].index(C.END)]["A"][key] / base_lv["A"][key])
            if base_lv["A"][key]
            else math.inf
            for key in RISING
        }
        elig = {key: v for key, v in fold.items() if base_lv["B"][key] < 10}
        q3[st] = {
            "base": base_lv,
            "points": pts,
            "fits": fits,
            "spearman": sp,
            "fold_A_end": fold,
            "panel_c_choice": max(elig, key=lambda kk: elig[kk]) if elig else "never_rate",
        }
    out["q3"] = q3

    # ---- other sets: base vs L2 end vs L3 end, means over seeds 0-2
    other = {}
    for st in ("charles173", "l2test", "l3test", "t1320", "t2130"):
        row = {}
        for stem in ("s1g3", "grpo_think_vs_L2", C.END):
            rs = [reads[(st, stem, s)] for s in (0, 1, 2) if (st, stem, s) in reads]
            if not rs:
                continue
            row[stem] = {
                "k": int(rs[0].k[0]),
                "pass1": float(np.mean([100 * np.mean(x.n_ok / x.k) for x in rs])),
                "A": {
                    key: float(np.mean([unit_A(x, key) for x in rs])) for key in RISING + ["any_dn"]
                },
                "B": {
                    key: float(np.mean([unit_B(x, key) for x in rs])) for key in RISING + ["any_dn"]
                },
                "gf_pct": float(np.mean([read_table(x)["gf_pct"] for x in rs])),
                "dead_per_proof": float(np.mean([read_table(x)["dead_per_proof"] for x in rs])),
            }
        other[st] = row
    out["other_sets"] = other

    # ---- Q4: Dan's EI ladders (working reductio among new distinct proofs per round)
    dan = json.loads((C.FORK / "artifacts/ps/rl_found.json").read_text())
    q4 = {}
    for arm, v in dan.items():
        new = {int(kk): x for kk, x in v["new_by_round"].items()}
        w = {int(kk): x for kk, x in v["working_by_round"].items()}
        q4[arm] = {
            "r1_share": 100 * w[1] / new[1],
            "r16_share": 100 * w[16] / new[16],
            "r2_16_share": 100
            * sum(w[i] for i in range(2, 17))
            / sum(new[i] for i in range(2, 17)),
            "r1_count": w[1],
            "r16_count": w[16],
            "r2_16_count": sum(w[i] for i in range(2, 17)),
        }
    out["q4_dan_ei"] = q4
    ours = {}
    for st in ("charles149", "t1320s250"):
        b, e = reads[(st, "s1g3", 0)], reads[(st, C.END, 0)]
        ours[st] = {
            unit: {
                "base": f(b, "work_red"),
                "end": f(e, "work_red"),
                "fold": f(e, "work_red") / f(b, "work_red") if f(b, "work_red") else math.inf,
            }
            for unit, f in (("A", unit_A), ("B", unit_B), ("Bpool", unit_Bpool), ("D", unit_D))
        }
        ours[st]["distinct_count"] = {
            "base": int(b.d["work_red"].sum()),
            "end": int(e.d["work_red"].sum()),
        }
    out["q4_ours"] = ours

    (C.HERE / "numbers.json").write_text(json.dumps(out, indent=1))
    write_md(out)
    print("wrote numbers.json and numbers.md")
    return 0


NAMES = {
    "work_red": "working #10",
    "free_red": "free #10",
    "templ_red": "template-only #10",
    "mt": "MT",
    "ds": "DS",
    "work_neg": "working ¬I",
    "gen_cases": "genuine cases",
    "any_dn": "any DN step",
    "lem": "LEM",
    "dne": "DNE",
    "efq": "ex falso",
}
COLS = ["work_red", "templ_red", "mt", "ds", "work_neg", "gen_cases", "any_dn"]


def _fold(a: float, b: float) -> float:
    return a / b if b else float("inf")


def _ci(t: list[float]) -> str:
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def write_md(N: dict[str, Any]) -> None:
    """numbers.md: every number the README and summary quote, generated from numbers.json."""
    L = [
        "# Numbers: proof strategies of the GRPO + thinking-trace models",
        "",
        "Generated by `analyze.py` from `numbers.json`. Units are defined in "
        "[PREREG.md](PREREG.md): **A** = % of attempts that are Lean-accepted and use the "
        "strategy (mean over problems); **B** = % of a solved problem's accepted attempts that "
        "use it (mean over solved problems); **C** = % of problems with at least one such proof; "
        "**D** = % of distinct proofs (pooled). Intervals are 95 % problem-bootstrap. "
        "#10 = proof by contradiction (classical reductio).",
        "",
    ]
    q1 = N["q1"]
    L += [
        "## Q1. Pretraining corpus vs K12, textbook proofs and the GRPO pools (% of proofs)",
        "",
        "| set | proofs | classical-only theorems | "
        + " | ".join(NAMES[c] for c in COLS)
        + " | opens with ¬goal | opens forward |",
        "|---|---:|---:|" + "---:|" * (len(COLS) + 2),
    ]
    rows = [
        (
            "S1g3 Stage-1 corpus (search proofs of K12 theorems)",
            q1["corpus_s1g3"],
            q1["corpus_classical_only_pct"],
        ),
        ("  on its classical-only theorems", q1["corpus_classical_only"], 100.0),
        ("  on its intuitionistic theorems", q1["corpus_intuitionistic"], 0.0),
        ("K12 generator proofs (Dan's table)", q1["k12"], None),
        (
            "Charles149 reference proofs",
            q1["charles149_reference"],
            q1["charles149_classical_only_pct"],
        ),
    ]
    rows += [
        (f"GRPO pool {k} (generator proofs)", v, v["classical_only_pct"])
        for k, v in q1["grpo_pools"].items()
    ]
    for nm, t, cl in rows:
        L.append(
            f"| {nm} | {t['n']:,} | {'' if cl is None else f'{cl:.1f}'} | "
            + " | ".join(f"{t[c]:.2f}" for c in COLS)
            + f" | {t['op:goal_reductio']:.1f} | {t['op:forward']:.1f} |"
        )
    L += [
        "",
        "K12 classical-only theorems: 4.02 % (Dan). The corpus's search proofs equal the "
        f"generator's proof on {q1['corpus_same_as_ref']:,} of {q1['corpus_s1g3']['n']:,} records.",
        "",
    ]
    # Q2
    q2 = N["q2_charles149"]
    L += [
        "## Q2. Charles149, seed-0 checkpoints (k 1,024)",
        "",
        "| checkpoint | R | pass@1 | B: "
        + " | ".join(NAMES[c] for c in COLS)
        + " | A: working #10 | A: DS | dead lines / proof | opens with ¬goal (B) |",
        "|---|---:|---:|" + "---:|" * (len(COLS) + 4),
    ]
    for stem, r in q2.items():
        L.append(
            f"| {C.short(stem)} | {r['R'] or 0:.2g} | {r['attempt_success_pct']:.1f} | "
            + " | ".join(f"{r['B'][c]:.1f}" for c in COLS)
            + f" | {r['A']['work_red']:.1f} | {r['A']['ds']:.1f} | {r['dead_per_proof']:.2f} | "
            f"{r['opening_B']['goal_reductio']:.1f} |"
        )
    L += [
        "",
        "**Like for like (B-matched): the problems both the base and the checkpoint solve; "
        "change in points.**",
        "",
        "| checkpoint | matched | " + " | ".join(NAMES[c] for c in COLS) + " |",
        "|---|---:|" + "---|" * len(COLS),
    ]
    for stem in ("grpo_think_vs_mini20_L2it8", "grpo_think_vs_L2", C.END, "grpo_think_vs_mb_ki78"):
        r = q2[stem]
        L.append(
            f"| {C.short(stem)} | {r['matched_n']} | "
            + " | ".join(_ci(r["diff_B_matched"][c]) for c in COLS)
            + " |"
        )
    t = N["q2_t1320s250"]
    L.append(
        f"| 13–20 lines (t1320s250, k 512): L3 end | {t['matched_n']} | "
        + " | ".join(_ci(t["diff_B_matched"][c]) for c in COLS)
        + " |"
    )
    L += [
        "",
        "**Per attempt (A), L3 end − base, points.**",
        "",
        "| set | " + " | ".join(NAMES[c] for c in COLS) + " |",
        "|---|" + "---|" * len(COLS),
        "| Charles149 | " + " | ".join(_ci(q2[C.END]["diff_A"][c]) for c in COLS) + " |",
        "| 13–20 lines | " + " | ".join(_ci(t["diff_A"][c]) for c in COLS) + " |",
        "",
    ]
    sd = N["q2_seeds"]
    L += [
        "**Seeds 0–2 (Charles149, means over seeds; per-seed values in numbers.json).**",
        "",
        "| arm | "
        + " | ".join(f"B {NAMES[c]}" for c in RISING)
        + " | "
        + " | ".join(f"A {NAMES[c]}" for c in RISING)
        + " |",
        "|---|" + "---:|" * (2 * len(RISING)),
    ]
    for stem in ("s1g3", C.END):
        L.append(
            f"| {C.short(stem)} | "
            + " | ".join(f"{sd[stem]['B'][c]:.1f}" for c in RISING)
            + " | "
            + " | ".join(f"{sd[stem]['A'][c]:.1f}" for c in RISING)
            + " |"
        )
    L += [
        "",
        "**By theorem class (seed 0): B, % of accepted proofs.**",
        "",
        "| set | checkpoint | class | problems | solved % | working #10 | free #10 "
        "| template-only #10 | any DN |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for st, v in N["q2_classical"].items():
        for stem in ("s1g3", "grpo_think_vs_L2", C.END):
            for cls in ("classical_only", "intuitionistic"):
                x = v[stem][cls]
                L.append(
                    f"| {st} | {C.short(stem)} | {cls.replace('_', '-')} | {x['n_problems']} | "
                    f"{x['solved_pct']:.1f} | "
                    + " | ".join(f"{x['B'][c]:.1f}" for c in ("work_red", "free_red", "templ_red"))
                    + f" | {x['B']['any_dn']:.1f} |"
                )
    ro = N["q2_rl_only"]
    s0 = ro["seed0"]
    L += [
        "",
        "**RL-only problems (Charles149, k 1,024).**",
        "",
        f"- Seed 0: {s0['n']} RL-only, {s0['base_only']} base-only; {s0['classical_only']} of "
        f"the {s0['n']} are classical-only "
        f"({100 * s0['classical_only'] / max(s0['n'], 1):.0f} %, against "
        f"{ro['set_classical_only_pct']:.1f} % of the set). "
        f"The base s0 never solves {ro['base_never_n']} problems.",
        "- Pooled over seeds 0–2 (base never in 3,072 attempts, RL at least once): "
        f"{ro['pooled3']['n']} problems, "
        f"{ro['pooled3']['classical_only']} classical-only.",
        "",
        "| proofs | problems | attempt success % | "
        + " | ".join(f"B {NAMES[c]}" for c in COLS)
        + " | pruned lines |",
        "|---|---:|---:|" + "---:|" * (len(COLS) + 1),
    ]
    for lab, nm in (
        ("rl_on_rl_only", "RL end on RL-only"),
        ("rl_on_base_solved", "RL end on base-solved"),
        ("base_on_base_solved", "base on base-solved"),
    ):
        x = s0[lab]
        L.append(
            f"| {nm} | {x['n_problems']} | {x['attempt_success_pct']:.1f} | "
            + " | ".join(f"{x['B'][c]:.1f}" for c in COLS)
            + f" | {x['Lp_mean']:.1f} |"
        )
    # openings
    L += [
        "",
        "**Opening move (B, % of accepted proofs; matched problems for RL rows).**",
        "",
        "| set | arm | " + " | ".join(OPENINGS[:5]) + " |",
        "|---|---|" + "---:|" * 5,
    ]
    L.append(
        "| Charles149 | base | "
        + " | ".join(f"{q2['s1g3']['opening_B'][o]:.1f}" for o in OPENINGS[:5])
        + " |"
    )
    L.append(
        "| Charles149 | L3 end (matched) | "
        + " | ".join(f"{q2[C.END]['opening_B_matched'][o]:.1f}" for o in OPENINGS[:5])
        + " |"
    )
    L.append(
        "| 13–20 lines | base (matched) | "
        + " | ".join(f"{t['opening_B_matched_base'][o]:.1f}" for o in OPENINGS[:5])
        + " |"
    )
    L.append(
        "| 13–20 lines | L3 end (matched) | "
        + " | ".join(f"{t['opening_B_matched_end'][o]:.1f}" for o in OPENINGS[:5])
        + " |"
    )
    # Q3
    L += [
        "",
        "## Q3. Emergence against R (seed-0 chain, 16 RL checkpoints)",
        "",
        "Logistic (4 parameters) in log10 R against a straight line in log10 R; ΔAIC > 0 "
        'favours the logistic. Pre-registered rule for "S-shaped": ΔAIC ≥ 2 and the midpoint '
        "inside the observed R range.",
        "",
        "| set | quantity | base | L3 end | ΔAIC | midpoint R | width (decades) | S-shaped |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for st in ("charles149", "t1320s250"):
        q = N["q3"][st]
        end = next(p for p in q["points"] if p["stem"] == C.END)
        for key, f in q["fits"].items():
            unit, _, k2 = key.partition(":")
            if key == "never_rate":
                b, e = 0.0, end["never_rate"]
                nm = (
                    f"solve rate on the {q['base']['never_n']} problems the base never solves "
                    "(% of attempts)"
                )
            elif key == "pass1":
                b, e, nm = q["base"]["pass1"], end["pass1"], "pass@1"
            else:
                b, e = q["base"][unit][k2], end[unit][k2]
                nm = f"{unit}: {NAMES[k2]}"
            ok = f["delta_aic"] >= 2 and f["midpoint_inside"]
            L.append(
                f"| {st} | {nm} | {b:.2f} | {e:.2f} | {f['delta_aic']:+.1f} | {f['mid_R']:.3g} | "
                f"{f['scale_decades']:.2f} | {'yes' if ok else 'no'} |"
            )
        L.append(
            f"| {st} | pre-registered panel (c) pick | | | | | | {q['panel_c_choice']} "
            "(fold in A: "
            + ", ".join(f"{NAMES[k]} {v:.2f}×" for k, v in q["fold_A_end"].items())
            + ") |"
        )
    sp = N["q3"]["charles149"]["spearman"]
    L += [
        "",
        "Spearman ρ across the 16 Charles149 checkpoints, of the RL − base change in A (or the "
        "base-unsolved "
        "solve rate) against the pass@1 and pass@1024 gaps of `teaser_numbers.json`:",
        "",
        "| quantity | ρ with pass@1 gap | ρ with pass@1024 gap |",
        "|---|---:|---:|",
    ]
    for k, v in sp.items():
        L.append(f"| {NAMES.get(k, k)} | {v['rho_gap1']:.2f} | {v['rho_gapK']:.2f} |")
    L += [
        "",
        "Per-checkpoint points (R, A, B, base-unsolved rate) are in numbers.json under `q3`.",
        "",
    ]
    # other sets
    L += [
        "## Other sets (base vs L2 end vs L3 end, means over seeds 0–2; B, % of accepted proofs)",
        "",
        "| set | arm | k | pass@1 | " + " | ".join(NAMES[c] for c in RISING + ["any_dn"]) + " |",
        "|---|---|---:|---:|" + "---:|" * (len(RISING) + 1),
    ]
    for st, v in N["other_sets"].items():
        for stem, r in v.items():
            L.append(
                f"| {st} | {C.short(stem)} | {r['k']} | {r['pass1']:.1f} | "
                + " | ".join(f"{r['B'][c]:.1f}" for c in RISING + ["any_dn"])
                + " |"
            )
    # Q4
    L += [
        "",
        "## Q4. Against Dan's EI ladders (working #10)",
        "",
        "Dan: share of each round's new distinct Lean-accepted proofs (round 1 = the pretrained "
        "model) and "
        "counts, from his `rl_found.json`.",
        "",
        "| ladder | round 1 % | round 16 % | share fold | round 1 count | round 16 count "
        "| count fold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, v in N["q4_dan_ei"].items():
        L.append(
            f"| {arm} | {v['r1_share']:.2f} | {v['r16_share']:.2f} | "
            f"{_fold(v['r16_share'], v['r1_share']):.1f} | {v['r1_count']} | "
            f"{v['r16_count']} | {_fold(v['r16_count'], v['r1_count']):.1f} |"
        )
    L += [
        "",
        "Ours, base → L3 end (seed 0):",
        "",
        "| set | unit | base | L3 end | fold |",
        "|---|---|---:|---:|---:|",
    ]
    for st, v in N["q4_ours"].items():
        for unit in ("A", "B", "Bpool", "D"):
            L.append(
                f"| {st} | {unit} | {v[unit]['base']:.2f} | {v[unit]['end']:.2f} | "
                f"{v[unit]['fold']:.1f} |"
            )
        L.append(
            f"| {st} | distinct proofs with working #10 (count) | {v['distinct_count']['base']} | "
            f"{v['distinct_count']['end']} | "
            f"{_fold(v['distinct_count']['end'], v['distinct_count']['base']):.1f} |"
        )
    (C.HERE / "numbers.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
