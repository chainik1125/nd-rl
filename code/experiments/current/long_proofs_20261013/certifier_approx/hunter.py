"""Approximate certifier building blocks: minlen's goal-directed DP with semantic pruning, plus knobs that make it
an INCOMPLETE but cheap hunter for short proofs.

  Hunter(prem, concl, lem='all'|'sub'|'lit'|'none', max_depth=None, memo_cap=None, seed=None)
    .levels(lo, hi)  -> dict: iterative deepening on body length lo..hi, one level at a time.
        found=m  : a verified proof with m body lines exists (true upper bound on the minimum)
        exh=E    : every level <= E was exhausted (no truncation) in THIS search space
        stop     : 'found' | 'exhausted' | 'memo_cap' | 'timeout'

With lem='all', max_depth=None, no cap and no seed, the space is exactly minlen's (vendor/fork/minlen.py) and
an exhaustion is a minlen-equivalent certificate.  Restrictions and the memo cap make exhaustion unsound; such
runs only matter when they FIND a proof (a sound refutation of a certificate) and their miss rate is measured
empirically (calibrate.py).
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'approach_d_theorem_first'))
import certify as C  # noqa: E402  (widens nd_verify atoms, imports minlen)
from minlen import Search, Timeout, parse_thm, BOT  # noqa: E402
from minlen_sem import SemSearch  # noqa: E402
from nd_verify import verify_text  # noqa: E402


class Budget(Exception):
    pass


def is_literal(f):
    return f[0] == 'atom' or f == BOT or (f[0] == 'not' and f[1][0] == 'atom')


class Hunter(SemSearch):
    def __init__(self, prem, concl, lem='all', max_depth=None, memo_cap=None, seed=None, deadline=None):
        super().__init__(prem, concl, deadline=deadline, lemmas=lem != 'none', max_depth=max_depth)
        sub = set()
        from minlen import subformulas
        for f in prem + [concl]:
            subformulas(f, sub)
        if lem == 'all':
            self.lem = list(self.S)
        elif lem == 'sub':          # subformulas and their single negations, no double negations
            self.lem = [f for f in self.S if f in sub or (f[0] == 'not' and f[1] in sub)]
        elif lem == 'lit':
            self.lem = [f for f in self.S if is_literal(f)]
        elif lem == 'none':
            self.lem = []
        else:
            raise ValueError(lem)
        self.lem_mode = lem
        self.memo_cap = memo_cap
        self.negs = list(self.S)
        if seed is not None:
            rng = random.Random(seed)
            for lst in (self.lem, self.conj, self.imps, self.disj, self.negs):
                rng.shuffle(lst)

    def best(self, G, A, b, d=0, H=frozenset()):
        if self.memo_cap and len(self.memo) > self.memo_cap:
            raise Budget()
        return super().best(G, A, b, d, H)

    # ---- copy of minlen.Search._search with: lemma candidates from self.lem, NEGE candidates from self.negs,
    #      and the FORBID machinery dropped (never used here) ----
    def _search(self, G, A, b, d=0, H=frozenset()):
        bestc, bestt = b + 1, None
        can_box = self.max_depth is None or d < self.max_depth

        def consider(c, t):
            nonlocal bestc, bestt
            if c < bestc:
                bestc, bestt = c, t

        if b >= 1:
            for Cj in self.conj:
                if Cj[1] == G or Cj[2] == G:
                    r = self.inp(Cj, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ('ANDE1' if Cj[1] == G else 'ANDE2', G, r[1]))
            for I in self.imps:
                if I[2] == G:
                    r = self.inp(I, A, min(b, bestc) - 1, d, H)
                    if r:
                        r2 = self.inp(I[1], A | {I}, min(b, bestc) - 1 - r[0], d, H)
                        if r2:
                            consider(r[0] + r2[0] + 1, ('IMPE', G, r[1], r2[1]))
            nn = ('not', ('not', G))
            r = self.inp(nn, A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ('DN', G, r[1]))
            if G != BOT:
                r = self.inp(BOT, A, min(b, bestc) - 1, d, H)
                if r:
                    consider(r[0] + 1, ('BOTE', G, r[1]))
        if G[0] == 'and' and b >= 1:
            r = self.inp(G[1], A, min(b, bestc) - 1, d, H)
            if r:
                r2 = self.inp(G[2], A | {G[1]}, min(b, bestc) - 1 - r[0], d, H)
                if r2:
                    consider(r[0] + r2[0] + 1, ('ANDI', G, r[1], r2[1]))
        if G[0] == 'or' and b >= 1:
            r = self.inp(G[1], A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ('ORI1', G, r[1]))
            r = self.inp(G[2], A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ('ORI2', G, r[1]))
        if G[0] == 'imp' and b >= 2:
            a, c = G[1], G[2]
            if a == c and can_box:
                consider(2, ('IMPI', G, a, None))
            elif b >= 3 and can_box:
                r = self.best(c, A | {a}, min(b, bestc) - 2, d + 1, H)
                if r:
                    consider(r[0] + 2, ('IMPI', G, a, r[1]))
        if G[0] == 'not' and b >= 3 and can_box:
            r = self.best(BOT, A | {G[1]}, min(b, bestc) - 2, d + 1, H)
            if r:
                consider(r[0] + 2, ('NEGI', G, G[1], r[1]))
        if G == BOT and b >= 1:
            for X in self.negs:
                NX = ('not', X)
                if NX not in self.S and NX not in A:
                    continue
                if NX in A:
                    r = self.inp(X, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ('NEGE', BOT, r[1], ('R0', NX)))
                elif X in A:
                    r = self.inp(NX, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ('NEGE', BOT, ('R0', X), r[1]))
                else:
                    r = self.inp(X, A, min(b, bestc) - 2, d, H)
                    if r:
                        r2 = self.inp(NX, A | {X}, min(b, bestc) - 1 - r[0], d, H)
                        if r2:
                            consider(r[0] + r2[0] + 1, ('NEGE', BOT, r[1], r2[1]))
        if G != BOT and G[0] != 'not' and b >= 4 and can_box:
            r = self.best(BOT, A | {('not', G)}, min(b, bestc) - 3, d + 1, H)
            if r:
                consider(r[0] + 3, ('RAA', G, r[1]))
        if b >= 3 and can_box:
            for D in self.disj:
                if D == G:
                    continue
                r = self.inp(D, A, min(b, bestc) - 3, d, H)
                if not r:
                    continue
                a, c = D[1], D[2]
                A2 = A | {D}
                ca = (1, None) if a == G else None
                if ca is None:
                    ra = self.best(G, A2 | {a}, min(b, bestc) - r[0] - 3, d + 1, H)
                    ca = (ra[0] + 1, ra[1]) if ra else None
                if ca is None:
                    continue
                cb = (1, None) if c == G else None
                if cb is None:
                    rb = self.best(G, A2 | {c}, min(b, bestc) - r[0] - ca[0] - 1, d + 1, H)
                    cb = (rb[0] + 1, rb[1]) if rb else None
                if cb is None:
                    continue
                consider(r[0] + ca[0] + cb[0] + 1, ('ORE', G, r[1], a, ca[1], c, cb[1]))
        if self.lem and b >= 2:
            for L in self.lem:
                if L in A or L == G:
                    continue
                r = self.best(L, A, min(b, bestc) - 1, d, H)
                if not r:
                    continue
                r2 = self.best(G, A | {L}, min(b, bestc) - r[0], d, H)
                if r2:
                    consider(r[0] + r2[0], ('LEMMA', G, L, r[1], r2[1]))
        return (bestc, bestt) if bestt is not None else None

    def levels(self, lo, hi, prompt=None):
        """Deepen body length from lo to hi. Returns dict(found, proof, exh, stop, secs, memo, per_level)."""
        A0 = frozenset(self.prem)
        out = {'found': None, 'proof': None, 'exh': lo - 1, 'stop': 'exhausted', 'per_level': []}
        t0 = time.time()
        try:
            for body in range(lo, hi + 1):
                t1 = time.time()
                r = self.best(self.concl, A0, body)
                out['per_level'].append((body, round(time.time() - t1, 3), len(self.memo)))
                if r:
                    text = self.to_text(r[1])
                    if prompt is not None:
                        ok, reason, nl = verify_text(prompt + ' ' + text)
                        assert ok and nl == len(self.prem) + r[0], (reason, nl, r[0])
                    out.update(found=r[0], proof=text, stop='found')
                    break
                out['exh'] = body
        except Budget:
            out['stop'] = 'memo_cap'
        except Timeout:
            out['stop'] = 'timeout'
        out['secs'] = round(time.time() - t0, 3)
        out['memo'] = len(self.memo)
        return out


def hunt(prompt, lo, hi, secs=None, **kw):
    prem, concl = parse_thm(prompt)
    h = Hunter(prem, concl, deadline=time.time() + secs if secs else None, **kw)
    r = h.levels(lo, hi, prompt)
    r['S'] = len(h.S)
    return r


if __name__ == '__main__':
    p = sys.argv[1]
    for kw in ({}, {'lem': 'none'}, {'lem': 'lit'}, {'lem': 'sub'}, {'lem': 'none', 'max_depth': 2},
               {'memo_cap': 200_000}, {'memo_cap': 200_000, 'seed': 1}):
        r = hunt(p, 1, int(sys.argv[2]) if len(sys.argv) > 2 else 15, secs=60, **kw)
        print(kw, {k: v for k, v in r.items() if k not in ('proof', 'per_level')})
