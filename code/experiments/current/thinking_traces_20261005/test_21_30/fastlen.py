"""Bounded minimal-proof-length search: the fork's `minlen.Search` with a cross-budget memo.

The fork's `minlen.py` memoises `best(G, A, b)` per budget b, so the same sub-problem (goal G,
citable set A) is re-solved for every budget it is asked with. Here the memo is keyed by (G, A)
[plus box depth / context flags only when a restriction needs them] and stores either the exact
optimum (found within some budget, hence optimal: the branch-and-bound returns the minimum <=
budget) or the largest budget at which no proof exists. Rules, the restricted formula space S, lemma
steps and the linearisation are inherited unchanged from `minlen.Search`, so a label means exactly
what a `minlen` label means ("no proof of length <= L-1 in S"; every proof found is re-checked with
nd_verify). `extra` adds formulas to S (e.g. the reference proof's lines, for shortening).

    fast_minlen(prompt, bound, time_limit, extra=()) -> {min_lines_ub, proof, timeout, bound, calls,
    lb} lb = the largest total length L such that "no proof <= L" was established (n_prem + budget),
    i.e. the true minimal length (in S) is >= lb + 1 when min_lines_ub is None.
"""

from __future__ import annotations

import sys
import time

sys.dont_write_bytecode = True
FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
if FORK not in sys.path:
    sys.path.insert(0, FORK)

import minlen as _ml  # noqa: E402  (fork, read-only)
from nd_verify import verify_text  # noqa: E402

Timeout = _ml.Timeout
parse_thm = _ml.parse_thm
subformulas = _ml.subformulas
BOT = _ml.BOT


class FastSearch(_ml.Search):
    def __init__(self, prem, concl, deadline=None, extra=(), **kw):
        super().__init__(prem, concl, deadline=deadline, **kw)
        if extra:
            S = set(self.S)
            for f in extra:
                subformulas(f, S)
            self.S = S
            self.conj = [f for f in S if f[0] == "and"]
            self.imps = [f for f in S if f[0] == "imp"]
            self.disj = [f for f in S if f[0] == "or"]
        self.memo2 = {}
        self.need_d = self.max_depth is not None

    def best(self, G, A, b, d=0, H=frozenset()):
        if b <= 0:
            return None
        if G in A:
            return (1, ("R", G))
        key = (G, A, d if self.need_d else None, H if self.track_h else None)
        e = self.memo2.get(key)
        if e is not None:
            if e[1] is not None:  # exact optimum known
                return e[1] if e[1][0] <= b else None
            if b <= e[0]:  # no proof <= e[0]
                return None
        self.calls += 1
        if self.deadline and (self.calls & 1023) == 0 and time.time() > self.deadline:
            raise Timeout()
        res = self._search(G, A, b, d, H)
        if res is not None:
            self.memo2[key] = (None, res)
        else:
            old = self.memo2.get(key)
            if old is None or old[1] is None:
                self.memo2[key] = (max(b, old[0] if old else 0), None)
        return res

    def run_bound(self, bound, start=None):
        """Iterative deepening on the total from `start` (default n_prem + 1) to `bound`; -> ((n,
        text) | None, lb) where lb is the largest total certified to have no proof (or n_prem if
        none certified)."""
        n_prem = len(self.prem)
        lb = n_prem
        if self.concl in self.prem:
            return None, lb
        for total in range(start or n_prem + 1, bound + 1):
            r = self.best(self.concl, frozenset(self.prem), total - n_prem)
            if r:
                return (n_prem + r[0], self.to_text(r[1])), lb
            lb = total
        return None, lb


def fast_minlen(prompt, bound=12, time_limit=30.0, extra=(), start=None, **kw):
    prem, concl = parse_thm(prompt)
    s = FastSearch(
        prem, concl, deadline=time.time() + time_limit if time_limit else None, extra=extra, **kw
    )
    lb = len(prem)
    t0 = time.time()
    try:
        r, lb = s.run_bound(bound, start=start)
    except Timeout:
        # lb: the deepest total whose search completed before the timeout is not tracked exactly
        # here; the caller re-derives it from staged bounds. Report timeout.
        return {
            "min_lines_ub": None,
            "proof": None,
            "timeout": True,
            "bound": bound,
            "calls": s.calls,
            "lb": None,
            "secs": time.time() - t0,
        }
    if r is None:
        return {
            "min_lines_ub": None,
            "proof": None,
            "timeout": False,
            "bound": bound,
            "calls": s.calls,
            "lb": lb,
            "secs": time.time() - t0,
        }
    n, text = r
    ok, reason, nl = verify_text(prompt + " " + text)
    if not ok or nl != n:
        return {
            "min_lines_ub": None,
            "proof": text,
            "timeout": False,
            "bound": bound,
            "error": f"{reason} nl={nl} n={n}",
            "calls": s.calls,
            "lb": None,
            "secs": time.time() - t0,
        }
    return {
        "min_lines_ub": n,
        "proof": text,
        "timeout": False,
        "bound": bound,
        "calls": s.calls,
        "lb": n - 1,
        "secs": time.time() - t0,
    }


def staged(prompt, stages=((10, 5.0), (12, 30.0), (14, 120.0)), extra=()):
    """Staged search like the fork's long-pool labelling, but each stage starts where the previous
    one certified. -> dict(L=exact label or None, lb=certified 'no proof <= lb', proof, timeout,
    secs)."""
    prem, concl = parse_thm(prompt)
    lb = len(prem)
    secs = 0.0
    for bound, tl in stages:
        r = fast_minlen(prompt, bound, tl, extra=extra, start=lb + 1)
        secs += r["secs"]
        if r.get("error"):
            return dict(L=None, lb=lb, proof=None, timeout=False, secs=secs, error=r["error"])
        if r["timeout"]:
            return dict(L=None, lb=lb, proof=None, timeout=True, secs=secs, stage=bound)
        if r["min_lines_ub"] is not None:
            return dict(
                L=r["min_lines_ub"],
                lb=r["min_lines_ub"] - 1,
                proof=r["proof"],
                timeout=False,
                secs=secs,
            )
        lb = r["lb"]
    return dict(L=None, lb=lb, proof=None, timeout=False, secs=secs)


class VocabSearch(FastSearch):
    """FastSearch with the candidate formula set S replaced by an explicit vocabulary (e.g. the
    reference proof's formulas): eliminations (ANDE / IMPE / ORE / NEGE partners), lemmas and
    reductio only range over it. A proof found is a real, nd_verify-checked proof; "none <= b" only
    covers proofs whose intermediate formulas are in the vocabulary."""

    def __init__(self, prem, concl, vocab, deadline=None, **kw):
        super().__init__(prem, concl, deadline=deadline, **kw)
        S = set(vocab) | set(prem) | {concl, BOT}
        self.S = S
        self.conj = [f for f in S if f[0] == "and"]
        self.imps = [f for f in S if f[0] == "imp"]
        self.disj = [f for f in S if f[0] == "or"]


def vocab_shorten(prompt, vocab, bound, time_limit):
    """Shortest proof of length <= bound whose intermediate formulas are in `vocab`; -> dict like
    fast_minlen."""
    prem, concl = parse_thm(prompt)
    s = VocabSearch(prem, concl, vocab, deadline=time.time() + time_limit if time_limit else None)
    t0 = time.time()
    try:
        # one branch-and-bound call at the full bound (finds the minimum <= bound in the vocabulary)
        n_prem = len(prem)
        r = s.best(concl, frozenset(prem), bound - n_prem)
    except Timeout:
        return {
            "min_lines_ub": None,
            "proof": None,
            "timeout": True,
            "calls": s.calls,
            "secs": time.time() - t0,
        }
    if r is None:
        return {
            "min_lines_ub": None,
            "proof": None,
            "timeout": False,
            "calls": s.calls,
            "secs": time.time() - t0,
        }
    n = len(prem) + r[0]
    text = s.to_text(r[1])
    ok, reason, nl = verify_text(prompt + " " + text)
    if not ok or nl != n:
        return {
            "min_lines_ub": None,
            "proof": text,
            "timeout": False,
            "error": f"{reason} nl={nl} n={n}",
            "calls": s.calls,
            "secs": time.time() - t0,
        }
    return {
        "min_lines_ub": n,
        "proof": text,
        "timeout": False,
        "calls": s.calls,
        "secs": time.time() - t0,
    }
