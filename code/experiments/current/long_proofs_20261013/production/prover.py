"""A small complete-enough ND prover for the clause family (premises/conclusions built from literals with one
connective). DPLL-style: forward-saturate with one-line rules, then case-split with ORE over a disjunction or,
as a last resort, over an excluded-middle instance ( x v ( ~ x ) ) derived by reductio. The proof it writes is
NOT minimal; minlen (certify.py) decides the minimal length. Every proof is checked with nd_verify.

    prove_text(prompt) -> 'N1 ... QED' or None
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C  # noqa: E402
from minlen import parse_thm  # noqa: E402

BOT = ('bot',)


def neg(f):
    return f[1] if f[0] == 'not' else ('not', f)


class Prover:
    def __init__(self, prem, concl):
        self.prem, self.concl = prem, concl
        self.lines = []  # (depth, formula, rule, refs 0-based)
        self.atoms = sorted(set().union(*[C.atoms(f, set()) for f in prem + [concl]]))

    def add(self, depth, f, rule, refs):
        self.lines.append((depth, f, rule, refs))
        return len(self.lines) - 1

    def nege(self, d, i, j):
        """F from lines i, j holding x and ~x in either order."""
        if self.lines[j][1] == ('not', self.lines[i][1]):
            return self.add(d, BOT, 'NEGE', [i, j])
        assert self.lines[i][1] == ('not', self.lines[j][1]), (self.lines[i], self.lines[j])
        return self.add(d, BOT, 'NEGE', [j, i])

    def negi(self, d, s, e, hyp):
        """Close box (s, e) deriving F: gives neg(hyp) (NEGI, plus DN when hyp is itself a negation)."""
        i = self.add(d, ('not', hyp), 'NEGI', [s, e])
        return self.add(d, hyp[1], 'DN', [i]) if hyp[0] == 'not' else i

    # ---- forward saturation with cheap rules; mutates `known` (formula -> line index) ----
    def saturate(self, known, d):
        changed = True
        while changed:
            changed = False
            for f, i in list(known.items()):
                if f[0] == 'and':
                    for k, g in ((1, f[1]), (2, f[2])):
                        if g not in known:
                            known[g] = self.add(d, g, f'ANDE{k}', [i]); changed = True
                elif f[0] == 'imp':
                    if f[1] in known and f[2] not in known:
                        known[f[2]] = self.add(d, f[2], 'IMPE', [i, known[f[1]]]); changed = True
                    elif neg(f[2]) in known and neg(f[1]) not in known and f[1] != f[2]:
                        # contrapositive box: AS a ; b ; F ; NEGI ~a
                        s = self.add(d + 1, f[1], 'AS', [])
                        b = self.add(d + 1, f[2], 'IMPE', [i, s])
                        e = self.nege(d + 1, b, known[neg(f[2])])
                        known[neg(f[1])] = self.negi(d, s, e, f[1]); changed = True
                elif f[0] == 'not' and f[1][0] == 'not' and f[1][1] not in known:
                    known[f[1][1]] = self.add(d, f[1][1], 'DN', [i]); changed = True
                elif f[0] == 'not' and f[1][0] == 'and':
                    a, b = f[1][1], f[1][2]
                    for x, y in ((a, b), (b, a)):
                        if x in known and neg(y) not in known and y not in known:
                            s = self.add(d + 1, y, 'AS', [])
                            c = self.add(d + 1, f[1], 'ANDI', [known[a], known[b]] if a in known and b in known else
                                         ([known[x], s] if x == a else [s, known[x]]))
                            e = self.add(d + 1, BOT, 'NEGE', [c, i])
                            known[neg(y)] = self.negi(d, s, e, y); changed = True
                            break

    def contradiction(self, known, d):
        """Try to derive F at depth d from `known`. Returns line index or None."""
        if BOT in known:
            return known[BOT]
        for f, i in known.items():
            if neg(f) in known and f[0] != 'not':
                return self.nege(d, i, known[neg(f)])
        for f, i in known.items():
            if f[0] == 'and' and f[1] in known and f[2] in known and neg(f) in known:
                return self.add(d, BOT, 'NEGE', [i, known[neg(f)]])
            if f[0] == 'not' and f[1][0] == 'and' and f[1][1] in known and f[1][2] in known:
                c = self.add(d, f[1], 'ANDI', [known[f[1][1]], known[f[1][2]]])
                return self.add(d, BOT, 'NEGE', [c, i])
            if f[0] == 'or' and neg(f[1]) in known and neg(f[2]) in known:
                s1 = self.add(d + 1, f[1], 'AS', []); e1 = self.nege(d + 1, s1, known[neg(f[1])])
                s2 = self.add(d + 1, f[2], 'AS', []); e2 = self.nege(d + 1, s2, known[neg(f[2])])
                return self.add(d, BOT, 'ORE', [i, s1, e1, s2, e2])
        return None

    def box(self, hyp, G, known, d):
        """Open a box at depth d+1 with hypothesis hyp, prove G inside; returns (start, end) line indices."""
        k2 = dict(known)
        s = self.add(d + 1, hyp, 'AS', [])
        k2[hyp] = s
        e = self.prove(G, k2, d + 1)
        if e is None:
            return None
        if e != len(self.lines) - 1 or self.lines[e][0] != d + 1:
            e = self.add(d + 1, G, 'R', [e])
        return s, e

    def prove(self, G, known, d):
        """Returns index of a citable line with formula G at depth <= d, or None."""
        if G in known:
            return known[G]
        self.saturate(known, d)
        if G in known:
            return known[G]
        c = self.contradiction(known, d)
        if c is not None:
            known[BOT] = c
            return c if G == BOT else self.add(d, G, 'BOTE', [c])
        if G[0] == 'and':
            i = self.prove(G[1], known, d)
            if i is None:
                return None
            j = self.prove(G[2], known, d)
            if j is None:
                return None
            return self.add(d, G, 'ANDI', [i, j])
        if G[0] == 'or' and (G[1] in known or G[2] in known):
            return self.add(d, G, 'ORI1', [known[G[1]]]) if G[1] in known else self.add(d, G, 'ORI2', [known[G[2]]])
        if G[0] == 'imp':
            r = self.box(G[1], G[2], known, d)
            return None if r is None else self.add(d, G, 'IMPI', list(r))
        if G[0] == 'not' and G[1][0] == 'and':
            r = self.box(G[1], BOT, known, d)
            return None if r is None else self.add(d, G, 'NEGI', list(r))
        # case split: ORE over a known disjunction with neither side decided
        for f, i in known.items():
            if f[0] == 'or' and f[1] not in known and f[2] not in known:
                return self.ore(i, f, G, known, d)
        # last resort: excluded middle on an undecided atom
        for x in self.atoms:
            a = ('atom', x)
            if a not in known and neg(a) not in known:
                lem = ('or', a, ('not', a))
                s = self.add(d + 1, ('not', lem), 'AS', [])
                s2 = self.add(d + 2, a, 'AS', [])
                o1 = self.add(d + 2, lem, 'ORI1', [s2])
                e2 = self.add(d + 2, BOT, 'NEGE', [o1, s])
                na = self.add(d + 1, ('not', a), 'NEGI', [s2, e2])
                o2 = self.add(d + 1, lem, 'ORI2', [na])
                e = self.add(d + 1, BOT, 'NEGE', [o2, s])
                nn = self.add(d, ('not', ('not', lem)), 'NEGI', [s, e])
                i = self.add(d, lem, 'DN', [nn])
                return self.ore(i, lem, G, known, d)
        return None

    def ore(self, i, f, G, known, d):
        r1 = self.box(f[1], G, known, d)
        if r1 is None:
            return None
        r2 = self.box(f[2], G, known, d)
        if r2 is None:
            return None
        return self.add(d, G, 'ORE', [i, *r1, *r2])

    def text(self):
        toks = []
        for i, (d, f, rule, refs) in enumerate(self.lines):
            toks += [f'N{i+1}'] + ['|'] * d + [C.fstr(f), ':', rule] + [f'N{r+1}' for r in refs] + [';']
        return ' '.join(toks + ['QED'])


def prove_text(prompt):
    prem, concl = parse_thm(prompt)
    p = Prover(prem, concl)
    known = {}
    for f in prem:
        known.setdefault(f, p.add(0, f, 'PR', []))
    e = p.prove(concl, known, 0)
    if e is None:
        return None
    if p.lines[e][0] != 0 or e != len(p.lines) - 1:
        p.add(0, concl, 'R', [e])
    body = p.text()
    ok, reason, nl = C.V.verify_text(prompt + ' ' + body)
    if not ok:
        return None
    return body


if __name__ == '__main__':
    import json
    n_ok = n = 0
    for line in open(sys.argv[1]):
        n += 1
        r = json.loads(line)
        t = prove_text(r['prompt'])
        if t is None:
            print('FAIL', r['prompt'])
        else:
            n_ok += 1
            print(t.count(';'), r['prompt'][:80])
    print(f'{n_ok}/{n} proved')
