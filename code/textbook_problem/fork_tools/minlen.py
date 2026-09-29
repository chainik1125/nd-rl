#!/usr/bin/env python3
"""Bounded minimal-proof-length search for the ND take-home logic (evaluation / labelling only).

  python minlen.py --in data/transfer.jsonl --out data/transfer_minlen.jsonl --bound 8 --time 20 --procs 32
  python minlen.py --selftest

Goal-directed dynamic programme over the 14 rules with the verifier's Fitch box semantics:
  best(G, A, b) = minimal number of NEW lines (<= b) that derive a line G at the current level, given
  the set A of citable formulas (premises, outer lines, hypotheses, lemmas) — or None.
Options: R (G in A), ANDI, ANDE1/2, IMPE, IMPI (box), ORI1/2, ORE (box pair), NEGE, NEGI (box), BOTE, DN,
classical reductio (box ~G .. F, NEGI, DN), and an explicit lemma step (derive L in S, then G with L
available), which restores sharing between sub-derivations.  Hypotheses / lemmas / rule partners are
restricted to S = subformulas of the theorem, their negations and double negations.
Total length = number of premises + best(conclusion).  Iterative deepening on the total up to --bound.
Every proof found is linearised and CHECKED WITH nd_verify; the label is
  min_lines_ub  : length of the shortest proof found (a verifier-valid proof of that length exists),
  None          : no proof of length <= bound in this search space, or timeout (flag `timeout`).
The search space is restricted (formula set S, only the hypotheses/lemmas in S are added to A), so
"None" is NOT a proof that no <= bound-line proof exists; min_lines_ub is a true upper bound on the
minimal length and, when it is <= 6, a proof that the theorem does not need a 7+-line proof.
Never used to generate training data.
"""
import argparse, json, os, sys, time, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from nd_verify.verify import parse_formula
from gen import fstr

BOT = ('bot',)
FORBID = ('DN', 'ORE_DERIVED', 'ORE_DERIVED_LOOSE', 'ORE_IN_IMPI', 'ANDE_NEGI_HYP', 'ORE_IN_ORE')   # rule restrictions for the necessity oracle (necessity.py)


class Timeout(Exception):
    pass


def subformulas(f, acc):
    acc.add(f)
    if f[0] == 'not':
        subformulas(f[1], acc)
    elif f[0] in ('and', 'or', 'imp'):
        subformulas(f[1], acc); subformulas(f[2], acc)
    return acc


def parse_thm(prompt):
    toks = prompt.split()
    i = 1
    prem = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1; continue
            break
    concl, i = parse_formula(toks, i + 1)
    return prem, concl


class Search:
    def __init__(self, prem, concl, deadline=None, lemmas=True, max_depth=None, no_derived_ore=False, forbid=()):
        self.prem, self.concl = prem, concl
        self.max_depth = max_depth
        forbid = set(forbid)
        if no_derived_ore:
            forbid.add('ORE_DERIVED_LOOSE')
        unknown = forbid - set(FORBID)
        assert not unknown, f'unknown --forbid {unknown}; known: {FORBID}'
        self.forbid = forbid
        self.forbid_dn = 'DN' in forbid
        # ORE restriction: LOOSE = ORE only over premise disjunctions (patterns.derived_ore == False);
        # STRICT = additionally allow ORE over an open hypothesis or over ( X v X ) (patterns.derived_ore_strict == False)
        self.ore_mode = 'loose' if 'ORE_DERIVED_LOOSE' in forbid else ('strict' if 'ORE_DERIVED' in forbid else None)
        # run-2 restrictions: ORE_IN_IMPI = no ORE inside an IMPI box; ORE_IN_ORE = no ORE inside an ORE branch;
        # ANDE_NEGI_HYP = no ANDE1/2 whose input is the hypothesis of an enclosing NEGI box.  Context is carried in H as
        # tagged entries: ('#impi',) / ('#ore',) flags and ('#neghyp', formula) markers; they never match formulas.
        self.ctx_impi = 'ORE_IN_IMPI' in forbid; self.ctx_ore = 'ORE_IN_ORE' in forbid; self.ctx_neg = 'ANDE_NEGI_HYP' in forbid
        self.track_h = self.ore_mode == 'strict' or self.ctx_impi or self.ctx_ore or self.ctx_neg
        sub = set()
        for f in prem + [concl]:
            subformulas(f, sub)
        S = set(sub)
        S |= {('not', f) for f in sub}
        S |= {('not', ('not', f)) for f in sub}
        S.add(BOT)
        self.S = S
        self.conj = [f for f in S if f[0] == 'and']
        self.imps = [f for f in S if f[0] == 'imp']
        self.disj = [f for f in S if f[0] == 'or']
        self.memo = {}
        self.deadline = deadline
        self.lemmas = lemmas
        self.calls = 0

    def best(self, G, A, b, d=0, H=frozenset()):
        """-> (cost, template) or None. A: frozenset of citable formulas; d: current box depth; H: open hypotheses
        (only part of the memo key when the strict ORE restriction is active)."""
        if b <= 0:
            return None
        key = (G, A, b, d, H if self.track_h else None)
        if key in self.memo:
            return self.memo[key]
        self.calls += 1
        if self.deadline and (self.calls & 1023) == 0 and time.time() > self.deadline:
            raise Timeout()
        res = None
        if G in A:
            res = (1, ('R', G))
        else:
            res = self._search(G, A, b, d, H)
        self.memo[key] = res
        return res

    def inp(self, G, A, b, d=0, H=frozenset()):
        """A rule INPUT: cost 0 if G is already citable, else a derivation of it."""
        if G in A:
            return (0, ('R0', G))
        return self.best(G, A, b, d, H)

    def _search(self, G, A, b, d=0, H=frozenset()):
        bestc, bestt = b + 1, None
        can_box = self.max_depth is None or d < self.max_depth

        def consider(c, t):
            nonlocal bestc, bestt
            if c < bestc:
                bestc, bestt = c, t

        # one-step rules from available formulas first (cheap)
        # ANDE
        if b >= 1:
            for C in self.conj:
                if C[1] == G or C[2] == G:
                    if self.ctx_neg and ('#neghyp', C) in H:
                        continue
                    r = self.inp(C, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ('ANDE1' if C[1] == G else 'ANDE2', G, r[1]))
        # IMPE
        if b >= 1:
            for I in self.imps:
                if I[2] == G:
                    r = self.inp(I, A, min(b, bestc) - 1, d, H)
                    if r:
                        r2 = self.inp(I[1], A | {I}, min(b, bestc) - 1 - r[0], d, H)
                        if r2:
                            consider(r[0] + r2[0] + 1, ('IMPE', G, r[1], r2[1]))
        # DN
        if b >= 1 and not self.forbid_dn:
            nn = ('not', ('not', G))
            r = self.inp(nn, A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ('DN', G, r[1]))
        # BOTE
        if G != BOT and b >= 1:
            r = self.inp(BOT, A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ('BOTE', G, r[1]))
        # intro rules
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
                r = self.best(c, A | {a}, min(b, bestc) - 2, d + 1, H | {a} | ({('#impi',)} if self.ctx_impi else set()))
                if r:
                    consider(r[0] + 2, ('IMPI', G, a, r[1]))
        if G[0] == 'not' and b >= 3 and can_box:
            r = self.best(BOT, A | {G[1]}, min(b, bestc) - 2, d + 1, H | {G[1]} | ({('#neghyp', G[1])} if self.ctx_neg else set()))
            if r:
                consider(r[0] + 2, ('NEGI', G, G[1], r[1]))
        if G == BOT and b >= 1:
            for X in self.S:
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
        # classical reductio: assume ~G, derive F, NEGI -> ~~G, DN
        if G != BOT and G[0] != 'not' and b >= 4 and can_box and not self.forbid_dn:
            r = self.best(BOT, A | {('not', G)}, min(b, bestc) - 3, d + 1, H | {('not', G)})
            if r:
                consider(r[0] + 3, ('RAA', G, r[1]))
        # ORE over a disjunction in S
        if b >= 3 and can_box and not (self.ctx_impi and ('#impi',) in H) and not (self.ctx_ore and ('#ore',) in H):
            for D in self.disj:
                if D == G:
                    continue
                if self.ore_mode == 'loose' and D not in self.prem:
                    continue
                if self.ore_mode == 'strict' and D not in self.prem and D not in H and D[1] != D[2]:
                    continue
                r = self.inp(D, A, min(b, bestc) - 3, d, H)
                if not r:
                    continue
                a, c = D[1], D[2]
                A2 = A | {D}
                ca = (1, None) if a == G else None
                if ca is None:
                    ra = self.best(G, A2 | {a}, min(b, bestc) - r[0] - 3, d + 1, H | {a} | ({('#ore',)} if self.ctx_ore else set()))
                    ca = (ra[0] + 1, ra[1]) if ra else None
                if ca is None:
                    continue
                cb = (1, None) if c == G else None
                if cb is None:
                    rb = self.best(G, A2 | {c}, min(b, bestc) - r[0] - ca[0] - 1, d + 1, H | {c} | ({('#ore',)} if self.ctx_ore else set()))
                    cb = (rb[0] + 1, rb[1]) if rb else None
                if cb is None:
                    continue
                consider(r[0] + ca[0] + cb[0] + 1, ('ORE', G, r[1], a, ca[1], c, cb[1]))
        # lemma: derive some L in S first, then G with L available
        if self.lemmas and b >= 2:
            for L in self.S:
                if L in A or L == G:
                    continue
                r = self.best(L, A, min(b, bestc) - 1, d, H)
                if not r:
                    continue
                r2 = self.best(G, A | {L}, min(b, bestc) - r[0], d, H)
                if r2:
                    consider(r[0] + r2[0], ('LEMMA', G, L, r[1], r2[1]))
        return (bestc, bestt) if bestt is not None else None

    # ---------- linearisation ----------
    def emit(self, t, A, depth, lines):
        """A: dict formula -> line index (0-based into lines). Appends lines; returns index of the line for t's goal."""
        op = t[0]
        if op == 'R0':
            return A[t[1]]
        if op == 'R':
            lines.append((depth, t[1], 'R', [A[t[1]]])); return len(lines) - 1
        if op in ('ANDE1', 'ANDE2', 'DN', 'BOTE', 'ORI1', 'ORI2'):
            i = self.emit(t[2], A, depth, lines)
            lines.append((depth, t[1], op, [i])); return len(lines) - 1
        if op == 'ANDI':
            i = self.emit(t[2], A, depth, lines)
            A2 = dict(A); A2[t[1][1]] = i
            j = self.emit(t[3], A2, depth, lines)
            lines.append((depth, t[1], 'ANDI', [i, j])); return len(lines) - 1
        if op == 'IMPE':
            i = self.emit(t[2], A, depth, lines)
            A2 = dict(A); A2[lines[i][1]] = i
            j = self.emit(t[3], A2, depth, lines)
            lines.append((depth, t[1], 'IMPE', [i, j])); return len(lines) - 1
        if op == 'NEGE':
            i = self.emit(t[2], A, depth, lines)
            A2 = dict(A); A2[lines[i][1]] = i
            j = self.emit(t[3], A2, depth, lines)
            lines.append((depth, BOT, 'NEGE', [i, j])); return len(lines) - 1
        if op in ('IMPI', 'NEGI'):
            hyp = t[2]
            lines.append((depth + 1, hyp, 'AS', [])); s = len(lines) - 1
            if t[3] is None:
                e = s
            else:
                A2 = dict(A); A2[hyp] = s
                e = self.emit(t[3], A2, depth + 1, lines)
            lines.append((depth, t[1], op, [s, e])); return len(lines) - 1
        if op == 'RAA':
            G = t[1]; hyp = ('not', G)
            lines.append((depth + 1, hyp, 'AS', [])); s = len(lines) - 1
            A2 = dict(A); A2[hyp] = s
            e = self.emit(t[2], A2, depth + 1, lines)
            lines.append((depth, ('not', hyp), 'NEGI', [s, e])); n = len(lines) - 1
            lines.append((depth, G, 'DN', [n])); return len(lines) - 1
        if op == 'ORE':
            G, td, a, ta, c, tc = t[1], t[2], t[3], t[4], t[5], t[6]
            d = self.emit(td, A, depth, lines)
            A2 = dict(A); A2[lines[d][1]] = d
            lines.append((depth + 1, a, 'AS', [])); s1 = len(lines) - 1
            if ta is None:
                e1 = s1
            else:
                A3 = dict(A2); A3[a] = s1
                e1 = self.emit(ta, A3, depth + 1, lines)
            lines.append((depth + 1, c, 'AS', [])); s2 = len(lines) - 1
            if tc is None:
                e2 = s2
            else:
                A3 = dict(A2); A3[c] = s2
                e2 = self.emit(tc, A3, depth + 1, lines)
            lines.append((depth, G, 'ORE', [d, s1, e1, s2, e2])); return len(lines) - 1
        if op == 'LEMMA':
            i = self.emit(t[3], A, depth, lines)
            A2 = dict(A); A2[t[2]] = i
            return self.emit(t[4], A2, depth, lines)
        raise ValueError(op)

    def to_text(self, template):
        lines = [(0, p, 'PR', []) for p in self.prem]
        A = {}
        for i, p in enumerate(self.prem):
            A.setdefault(p, i)
        self.emit(template, A, 0, lines)
        toks = []
        for i, (d, f, rule, refs) in enumerate(lines):
            toks += [f'N{i+1}'] + ['|'] * d + [fstr(f), ':', rule] + [f'N{r+1}' for r in refs] + [';']
        toks.append('QED')
        return ' '.join(toks)

    def run(self, bound):
        """Iterative deepening on the total line count. -> (n_lines, proof_text) or None."""
        n_prem = len(self.prem)
        if self.concl in self.prem:
            return None
        for total in range(n_prem + 1, bound + 1):
            r = self.best(self.concl, frozenset(self.prem), total - n_prem)
            if r:
                return n_prem + r[0], self.to_text(r[1])
        return None


def minlen(prompt, bound=8, time_limit=20.0, lemmas=True, max_depth=None, no_derived_ore=False, forbid=()):
    prem, concl = parse_thm(prompt)
    s = Search(prem, concl, deadline=time.time() + time_limit if time_limit else None, lemmas=lemmas, max_depth=max_depth, no_derived_ore=no_derived_ore, forbid=forbid)
    try:
        r = s.run(bound)
    except Timeout:
        return {'min_lines_ub': None, 'proof': None, 'timeout': True, 'bound': bound, 'calls': s.calls}
    if r is None:
        return {'min_lines_ub': None, 'proof': None, 'timeout': False, 'bound': bound, 'calls': s.calls}
    n, text = r
    ok, reason, nl = verify_text(prompt + ' ' + text)
    if not ok or nl != n:
        return {'min_lines_ub': None, 'proof': text, 'timeout': False, 'bound': bound, 'error': f'{reason} nl={nl} n={n}', 'calls': s.calls}
    return {'min_lines_ub': n, 'proof': text, 'timeout': False, 'bound': bound, 'calls': s.calls}


def _work(args):
    rec, bound, tl, md, ndo, forbid = args
    t0 = time.time()
    r = minlen(rec['prompt'], bound, tl, max_depth=md, no_derived_ore=ndo, forbid=forbid)
    r['secs'] = time.time() - t0
    return r


def selftest():
    cases = [  # (prompt, expected min_lines_ub or None within bound 8)
        ('THM ( P > Q ) , P SEQ Q PRF', 3),
        ('THM SEQ ( P > P ) PRF', 2),
        ('THM ( ~ ( ~ P ) ) SEQ P PRF', 2),
        ('THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF', 6),
        ('THM ( P v Q ) , ( ~ P ) SEQ Q PRF', 7),
        ('THM ( P > Q ) SEQ ( ( ~ Q ) > ( ~ P ) ) PRF', 7),
        ('THM P , ( ~ P ) SEQ Q PRF', 4),
        ('THM ( P & Q ) SEQ ( Q & P ) PRF', 4),
        ('THM ( P v Q ) SEQ ( Q v P ) PRF', 6),           # ORE with ORI in each branch
        ('THM ( ( P > Q ) > P ) SEQ P PRF', None),   # Peirce: 9+ lines in this search space? bound 8
        ('THM SEQ ( P v ( ~ P ) ) PRF', None),         # excluded middle needs > 8 lines
        ('THM ( P > ( Q > R ) ) SEQ ( ( P & Q ) > R ) PRF', 7),
        ('THM ( ( P & Q ) > R ) SEQ ( P > ( Q > R ) ) PRF', 7),   # export; validation file's reference bound is 8
    ]
    ok_all = True
    for prompt, exp in cases:
        r = minlen(prompt, 8, 30)
        got = r['min_lines_ub']
        flag = 'ok' if got == exp else 'MISMATCH'
        if got != exp:
            ok_all = False
        print(f'{flag:9s} exp {exp} got {got} calls {r["calls"]} {prompt}')
        if r['proof']:
            v = verify_text(prompt + ' ' + r['proof'])
            assert v[0], v
            print('   ', r['proof'])
    # rule restrictions (--forbid): (prompt, forbid, expected min_lines_ub at bound 10, pattern the found proof must / must not contain)
    from patterns import classify
    rcases = [
        ('THM ( ~ ( ~ P ) ) SEQ P PRF', ('DN',), None, None),
        ('THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ P PRF', (), 7, ('reductio', True)),            # negimp_to_pos: classical-only
        ('THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ P PRF', ('DN',), None, None),
        ('THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF', ('DN',), 6, ('reductio', False)),     # intuitionistic: unaffected
        ('THM ( ( P v Q ) & R ) SEQ ( Q v P ) PRF', (), 7, ('derived_ore_strict', True)),
        ('THM ( ( P v Q ) & R ) SEQ ( Q v P ) PRF', ('ORE_DERIVED',), None, None),
        ('THM ( ( P v Q ) & R ) SEQ ( Q v P ) PRF', ('ORE_DERIVED_LOOSE',), None, None),
        ('THM ( P v Q ) SEQ ( Q v P ) PRF', ('ORE_DERIVED',), 6, ('derived_ore', False)),   # ORE over a premise: allowed
        ('THM SEQ ( ( P v Q ) > ( Q v P ) ) PRF', ('ORE_DERIVED',), 7, ('derived_ore_strict', False)),  # ORE over a hypothesis: allowed in strict mode
        ('THM SEQ ( ( P v Q ) > ( Q v P ) ) PRF', ('ORE_DERIVED_LOOSE',), None, None),
        ('THM ( P & ( Q v R ) ) SEQ ( ( P & Q ) v ( P & R ) ) PRF', (), 10, ('derived_ore_strict', True)),   # distribution
        ('THM ( P & ( Q v R ) ) SEQ ( ( P & Q ) v ( P & R ) ) PRF', ('ORE_DERIVED',), None, None),
        ('THM ( P > ( Q v R ) ) , P , ( ~ Q ) SEQ R PRF', (), 9, ('derived_ore_strict', True)),
        ('THM ( P > ( Q v R ) ) , P , ( ~ Q ) SEQ R PRF', ('ORE_DERIVED',), None, None),
        ('THM ( P > ( Q v R ) ) , P , ( ~ Q ) SEQ R PRF', ('DN',), 9, ('derived_ore_strict', True)),
    ]
    from patterns2 import classify2
    rcases += [
        ('THM ( P v Q ) , ( P > R ) SEQ ( ( Q > R ) > R ) PRF', (), 9, None),                     # IMPI box containing an ORE
        ('THM ( P v Q ) , ( P > R ) SEQ ( ( Q > R ) > R ) PRF', ('ORE_IN_IMPI',), 11, None),   # ORE moved outside: +2 lines
        ('THM ( P v Q ) SEQ ( R > ( Q v P ) ) PRF', ('ORE_IN_IMPI',), 9, None),                    # ORE outside then IMPI: allowed (+1 line)
        ('THM ( ~ P ) SEQ ( ~ ( P & Q ) ) PRF', (), 5, None),
        ('THM ( ~ P ) SEQ ( ~ ( P & Q ) ) PRF', ('ANDE_NEGI_HYP',), 7, None),                    # detour: assume ~~(P&Q), DN, ANDE (+2 lines)
        ('THM ( P & ( ~ Q ) ) SEQ ( ~ Q ) PRF', ('ANDE_NEGI_HYP',), 2, None),                     # ANDE on a premise: unaffected
        ('THM ( ( P v Q ) v R ) SEQ ( P v ( Q v R ) ) PRF', (), 12, None),                        # nested ORE
        ('THM ( ( P v Q ) v R ) SEQ ( P v ( Q v R ) ) PRF', ('ORE_IN_ORE',), None, None),
        ('THM ( P v Q ) , ( R v S ) SEQ ( ( Q v P ) & ( S v R ) ) PRF', ('ORE_IN_ORE',), 13, None),   # sequential OREs: allowed
    ]
    P2 = {'ORE_IN_IMPI': 'impi_ore', 'ANDE_NEGI_HYP': 'negi_ande_hyp', 'ORE_IN_ORE': 'nested_ore'}
    for prompt, fb, exp, pat in rcases:
        r = minlen(prompt, 13, 120, forbid=fb)
        got = r['min_lines_ub']; good = got == exp and not r['timeout']
        if r['proof']:
            assert verify_text(prompt + ' ' + r['proof'])[0]
            if pat:
                good = good and classify(r['proof'])[pat[0]] == pat[1]
            for fbk, p2 in P2.items():          # a proof found under a run-2 restriction must not contain that pattern
                if fbk in fb:
                    good = good and not classify2(r['proof'])[p2]
        ok_all = ok_all and good
        print(f"{'ok' if good else 'MISMATCH':9s} forbid {fb} exp {exp} got {got} {prompt}")
    print('SELFTEST', 'PASS' if ok_all else 'FAIL')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp')
    ap.add_argument('--out')
    ap.add_argument('--bound', type=int, default=8)
    ap.add_argument('--time', type=float, default=20.0)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--max_depth', type=int, default=None, help='restricted search: no boxes deeper than this')
    ap.add_argument('--no_derived_ore', action='store_true', help='restricted search: ORE only over premise disjunctions (= --forbid ORE_DERIVED_LOOSE)')
    ap.add_argument('--forbid', nargs='*', default=[], help=f'rule restrictions: {FORBID}. DN also disables the reductio template; ORE_DERIVED forbids ORE over a disjunction that is not a premise, not an open hypothesis and not ( X v X ) (the complement of patterns.derived_ore_strict); ORE_DERIVED_LOOSE forbids ORE over any non-premise line')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(a.procs) as pool, open(a.out, 'w') as fo:
        for i, (rec, r) in enumerate(zip(recs, pool.imap(_work, [(r, a.bound, a.time, a.max_depth, a.no_derived_ore, tuple(a.forbid)) for r in recs], chunksize=4))):
            out = {k: rec[k] for k in ('name', 'thm', 'prompt') if k in rec}
            out['gen_lines'] = rec.get('n_lines', rec.get('gen_lines'))
            out.update(r)
            fo.write(json.dumps(out) + '\n')
            if (i + 1) % 200 == 0:
                print(f'{i+1}/{len(recs)} {time.time()-t0:.0f}s', flush=True)
    rs = [json.loads(l) for l in open(a.out)]
    found = [r for r in rs if r['min_lines_ub'] is not None]
    print(f'{len(rs)} theorems; labelled {len(found)}; timeouts {sum(r.get("timeout", False) for r in rs)}; errors {sum("error" in r for r in rs)}')
    print('min_lines_ub hist', dict(sorted(collections.Counter(r['min_lines_ub'] for r in found).items())))
    gl = collections.Counter((r['gen_lines'], r['min_lines_ub'] is not None and r['min_lines_ub'] <= 6) for r in rs)
    print('by generating length: (gen_lines, has <=6 proof) ->', dict(sorted(gl.items(), key=lambda x: (x[0][0] or 0, x[0][1]))))


if __name__ == '__main__':
    main()
