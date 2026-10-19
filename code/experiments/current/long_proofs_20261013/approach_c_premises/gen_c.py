"""Approach C, premise-driven: theorems built as a gate DAG over atoms, with the proof built alongside.

Each premise is a "gate" that turns available atoms into a new atom (or a disjunction, or a negation).
The conclusion is an atom (or a conjunction of two) that only the full DAG can reach; optional IMPI /
NEGI wrapping moves source atoms into the goal as hypotheses. Disjunction gates force an ORE whose two
branches do distinct, branch-specific work. Every atom has a unique producer per branch, so there is no
alternative route, and each premise is checked to be semantically necessary (dropping it makes the
sequent invalid by truth table) and non-redundant (not implied by the others).

    python gen_c.py OUT.jsonl N --seed 0 [--knobs ...]
"""
import argparse
import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vendor', 'fork'))
sys.path.insert(0, os.path.join(HERE, '..'))
import nd_verify  # noqa: E402

V = sys.modules['nd_verify.verify']
LETTERS = ['P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
V.ATOM_NAMES.update(LETTERS)
from certify import fstr, truth_table, atoms as f_atoms  # noqa: E402

BOT = ('bot',)


def A(n):
    return ('atom', n)


def imp(a, b):
    return ('imp', a, b)


def conj(a, b):
    return ('and', a, b)


def neg(a):
    return ('not', a)


def contains_conj(f, t):
    """t is reachable from f by ANDE steps (f == t counts)."""
    if f == t:
        return True
    return f[0] == 'and' and (contains_conj(f[1], t) or contains_conj(f[2], t))


def reach(f, t):
    """t is reachable from f by a chain of IMPE (on the consequent) and ANDE steps."""
    if contains_conj(f, t):
        return True
    return f[0] == 'imp' and reach(f[2], t)


class Fail(Exception):
    pass


class Ctx:
    def __init__(self, pb, depth, avail, gates):
        self.pb, self.depth, self.avail, self.gates = pb, depth, dict(avail), list(gates)
        self.inprog = set()

    def child(self):
        c = Ctx(self.pb, self.depth + 1, self.avail, self.gates)
        return c


class PB:
    """Fitch proof builder with a small backward prover for the Horn-like part (IMPE / ANDE / ANDI)."""

    def __init__(self, prems):
        self.lines = []
        for p in prems:
            self.add(0, p, 'PR', [])

    def add(self, d, f, rule, refs):
        self.lines.append((d, f, rule, refs))
        return len(self.lines)

    def text(self):
        toks = []
        for i, (d, f, rule, refs) in enumerate(self.lines):
            toks += [f'N{i + 1}'] + ['|'] * d + [fstr(f), ':', rule] + [f'N{r}' for r in refs] + [';']
        return ' '.join(toks + ['QED'])

    def ensure(self, c, t):
        if t in c.avail:
            return c.avail[t]
        if t in c.inprog:
            raise Fail(t)
        c.inprog.add(t)
        try:
            # ANDE from an available conjunction
            for f, i in list(c.avail.items()):
                if f[0] == 'and' and contains_conj(f, t):
                    return self.ande(c, f, i, t)
            if t[0] == 'and':
                i = self.ensure(c, t[1])
                j = self.ensure(c, t[2])
                c.avail[t] = self.add(c.depth, t, 'ANDI', [i, j])
                return c.avail[t]
            # a gate (premise or available implication) whose consequent chain reaches t
            for g in c.gates:
                if g[0] == 'imp' and reach(g, t) and g not in c.inprog:
                    snap = len(self.lines)
                    try:
                        return self.fire(c, g, t)
                    except Fail:
                        del self.lines[snap:]
                        for k in [k for k, v in c.avail.items() if v > snap]:
                            del c.avail[k]
            raise Fail(t)
        finally:
            c.inprog.discard(t)

    def ande(self, c, f, i, t):
        while f != t:
            if contains_conj(f[1], t):
                f = f[1]
                i = c.avail.get(f) or self.add(c.depth, f, 'ANDE1', [i])
            else:
                f = f[2]
                i = c.avail.get(f) or self.add(c.depth, f, 'ANDE2', [i])
            c.avail[f] = i
        return i

    def fire(self, c, g, t):
        i = self.ensure(c, g)
        f = g
        while not contains_conj(f, t):
            j = self.ensure(c, f[1])
            f = f[2]
            i = c.avail.get(f) or self.add(c.depth, f, 'IMPE', [i, j])
            c.avail[f] = i
        return self.ande(c, f, i, t)

    # ---- plan execution ----
    def run(self, c, plan):
        """Executes a plan node, returns the line index of its result formula."""
        kind = plan[0]
        if kind == 'ens':
            return self.ensure(c, plan[1])
        if kind == 'impi':
            _, h, sub, goal = plan
            cc = c.child()
            s = self.add(cc.depth, h, 'AS', [])
            cc.avail[h] = s
            cc.gates = cc.gates + ([h] if h[0] == 'imp' else [])
            e = self.run(cc, sub)
            c.avail[goal] = self.add(c.depth, goal, 'IMPI', [s, e])
            return c.avail[goal]
        if kind == 'negi':
            _, h, sub, goal = plan      # sub derives F
            cc = c.child()
            s = self.add(cc.depth, h, 'AS', [])
            cc.avail[h] = s
            e = self.run(cc, sub)
            c.avail[goal] = self.add(c.depth, goal, 'NEGI', [s, e])
            return c.avail[goal]
        if kind == 'bot':
            _, x = plan                 # derive F from x and ~x
            i = self.ensure(c, x)
            j = self.ensure(c, neg(x))
            return self.add(c.depth, BOT, 'NEGE', [i, j])
        if kind == 'dead':
            _, x, goal = plan           # F then BOTE goal
            b = self.run(c, ('bot', x))
            c.avail[goal] = self.add(c.depth, goal, 'BOTE', [b])
            return c.avail[goal]
        if kind == 'split':
            _, D, pa, pb_, goal = plan
            i = self.ensure(c, D)
            ca = c.child()
            sa = self.add(ca.depth, D[1], 'AS', [])
            ca.avail[D[1]] = sa
            ea = self.run(ca, pa)
            cb = c.child()
            sb = self.add(cb.depth, D[2], 'AS', [])
            cb.avail[D[2]] = sb
            eb = self.run(cb, pb_)
            c.avail[goal] = self.add(c.depth, goal, 'ORE', [i, sa, ea, sb, eb])
            return c.avail[goal]
        raise ValueError(kind)


# ---------------- random theorem generation ----------------

class Gen:
    def __init__(self, rng, k_atoms, n_prem, n_hyp, p_split=0.3, p_dead=0.3, gate_types=None,
                 goal_kind='atom', max_split_depth=2, p_disj_gate=0.5):
        self.rng, self.k, self.n_prem, self.n_hyp = rng, k_atoms, n_prem, n_hyp
        self.p_split, self.p_dead, self.msd, self.p_disj_gate = p_split, p_dead, max_split_depth, p_disj_gate
        self.gate_types = gate_types or ['imp1', 'imp2', 'imp3', 'conj2', 'conj3', 'mix']
        self.goal_kind = goal_kind
        self.free = [A(x) for x in LETTERS[:k_atoms]]
        rng.shuffle(self.free)
        self.prem = []

    def take(self):
        if not self.free:
            raise Fail('atoms')
        return self.free.pop()

    def budget_left(self):
        return self.n_prem - len(self.prem)

    def gate(self, out, sources, must=None):
        """Add a premise gate producing `out`; inputs from `sources` (plus `must`), or fresh atoms that
        become sub-goals. Returns (plan-free) list of sub-goals to derive."""
        rng = self.rng
        gt = rng.choice(self.gate_types)
        n_in = {'imp1': 1, 'imp2': 2, 'imp3': 3, 'conj2': 2, 'conj3': 3, 'mix': 3}[gt]
        ins = []
        if must is not None:
            ins.append(must)
        src = [s for s in sources if s != must]
        rng.shuffle(src)
        subgoals = []
        while len(ins) < n_in:
            # fresh sub-goal atom if budget allows, else a source
            if self.budget_left() > 1 + len(subgoals) and self.free and rng.random() < 0.6:
                a = self.take()
                ins.append(a)
                subgoals.append(a)
            elif src:
                ins.append(src.pop())
            elif self.free and self.budget_left() > 1 + len(subgoals):
                a = self.take()
                ins.append(a)
                subgoals.append(a)
            else:
                break
        if not ins:
            raise Fail('no inputs')
        rng.shuffle(ins)
        n = len(ins)
        if gt in ('imp1', 'imp2', 'imp3'):
            f = out
            for a in reversed(ins):
                f = imp(a, f)
        elif gt in ('conj2', 'conj3'):
            if n == 1:
                f = imp(ins[0], out)
            else:
                c = ins[0]
                for a in ins[1:]:
                    c = conj(c, a)
                f = imp(c, out)
        else:  # mix: (a & b) > (c > out) when 3 inputs
            if n == 3:
                f = imp(conj(ins[0], ins[1]), imp(ins[2], out))
            elif n == 2:
                f = imp(conj(ins[0], ins[1]), out)
            else:
                f = imp(ins[0], out)
        self.prem.append(f)
        return subgoals

    def need(self, t, sources, sdepth=0, must=None):
        """Plan deriving atom t from `sources` (atoms available here). Adds premises."""
        rng = self.rng
        if (sdepth < self.msd and self.budget_left() >= 3 and len(self.free) >= 2 and must is None
                and rng.random() < self.p_split):
            a, b = self.take(), self.take()
            D = ('or', a, b)
            if sources and rng.random() < self.p_disj_gate:
                x = rng.choice(sources)
                self.prem.append(imp(x, D))
            else:
                self.prem.append(D)
            # branch a: must use a
            pa = self.need(t, sources + [a], sdepth + 1, must=a)
            if rng.random() < self.p_dead and sources and self.budget_left() >= 1:
                x = rng.choice(sources)
                self.prem.append(imp(b, neg(x)))
                pb_ = ('dead', x, t)
            else:
                pb_ = self.need(t, sources + [b], sdepth + 1, must=b)
            return ('split', D, pa, pb_, t)
        if self.budget_left() < 1:
            raise Fail('budget')
        subgoals = self.gate(t, sources, must)
        plans = [self.need(s, sources, sdepth) for s in subgoals]
        return ('seq', plans, ('ens', t)) if plans else ('ens', t)

    def build(self):
        rng = self.rng
        hyps = [self.take() for _ in range(self.n_hyp)]
        src_prem = []
        n_src = 0 if self.n_hyp else rng.choice([1, 1, 2])
        for _ in range(n_src):
            a = self.take()
            src_prem.append(a)
            self.prem.append(a)
        sources = hyps + src_prem
        if self.goal_kind == 'and':
            g1, g2 = self.take(), self.take()
            plan = ('seq', [self.need(g1, sources), self.need(g2, sources)], ('ens', conj(g1, g2)))
            goal = conj(g1, g2)
        elif self.goal_kind == 'neg':
            # goal ~h: assume h, derive x and ~x
            h = self.take()
            x = self.take()
            sources = sources + [h]
            p1 = self.need(x, sources)
            # gate producing ~x, must use h
            self.prem.append(imp(h, neg(x))) if rng.random() < 0.5 else self.prem.append(imp(conj(h, rng.choice(sources)), neg(x)))
            plan = ('negi', h, ('seq', [p1], ('bot', x)), neg(h))
            goal = neg(h)
            hyps = [hh for hh in hyps]
        else:
            g = self.take()
            plan = self.need(g, sources)
            goal = g
        for h in reversed(hyps):
            goal = imp(h, goal)
            plan = ('impi', h, plan, goal)
        rng.shuffle(self.prem)
        return self.prem, goal, plan


def run_plan(pb, c, plan):
    if plan[0] == 'seq':
        for p in plan[1]:
            run_plan(pb, c, p)
        return run_plan(pb, c, plan[2])
    if plan[0] in ('impi', 'negi'):
        # the sub-plan may be a seq: wrap through PB.run by converting
        kind, h, sub, goal = plan
        cc = c.child()
        s = pb.add(cc.depth, h, 'AS', [])
        cc.avail[h] = s
        e = run_plan(pb, cc, sub)
        c.avail[goal] = pb.add(c.depth, goal, kind.upper(), [s, e])
        return c.avail[goal]
    if plan[0] == 'split':
        _, D, pa, pb_, goal = plan
        i = pb.ensure(c, D)
        ca = c.child()
        sa = pb.add(ca.depth, D[1], 'AS', [])
        ca.avail[D[1]] = sa
        ea = run_plan(pb, ca, pa)
        cb = c.child()
        sb = pb.add(cb.depth, D[2], 'AS', [])
        cb.avail[D[2]] = sb
        eb = run_plan(pb, cb, pb_)
        c.avail[goal] = pb.add(c.depth, goal, 'ORE', [i, sa, ea, sb, eb])
        return c.avail[goal]
    return pb.run(c, plan)


def premises_each_necessary(prem, concl):
    sat, valid = truth_table(prem, concl)
    if not (sat and valid):
        return False
    for i in range(len(prem)):
        rest = prem[:i] + prem[i + 1:]
        _, v = truth_table(rest, concl)
        if v:
            return False            # premise i not needed for validity
        _, v2 = truth_table(rest, prem[i])
        if v2:
            return False            # premise i implied by the others
    return True


def make_one(rng, **kw):
    g = Gen(rng, **kw)
    prem, goal, plan = g.build()
    if len(prem) > 10 or len(set(prem)) != len(prem) or goal in prem:
        raise Fail('prem')
    pb = PB(prem)
    c = Ctx(pb, 0, {}, prem)
    for i, p in enumerate(prem):
        c.avail.setdefault(p, i + 1)
    run_plan(pb, c, plan)
    thm = 'THM ' + (' , '.join(fstr(p) for p in prem) + ' ' if prem else '') + 'SEQ ' + fstr(goal) + ' PRF'
    text = thm + ' ' + pb.text()
    ok, reason, nl = V.verify_text(text)
    if not ok:
        raise Fail('verify: ' + reason)
    if not premises_each_necessary(prem, goal):
        raise Fail('redundant premise')
    return {'text': text, 'n_prem': len(prem), 'n_body': nl - len(prem),
            'n_atoms': len(set().union(*[f_atoms(f, set()) for f in prem + [goal]]))}


def sample_kw(rng, a):
    return dict(k_atoms=rng.randint(a.min_atoms, 10), n_prem=rng.randint(a.min_prem, a.max_prem),
                n_hyp=rng.randint(a.min_hyp, a.max_hyp), p_split=a.p_split, p_dead=a.p_dead,
                gate_types=a.gates.split(','), goal_kind=rng.choice(a.goal_kinds.split(',')),
                max_split_depth=a.max_split_depth, p_disj_gate=a.p_disj_gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min_atoms', type=int, default=4)
    ap.add_argument('--min_prem', type=int, default=3)
    ap.add_argument('--max_prem', type=int, default=10)
    ap.add_argument('--min_hyp', type=int, default=0)
    ap.add_argument('--max_hyp', type=int, default=3)
    ap.add_argument('--p_split', type=float, default=0.3)
    ap.add_argument('--p_dead', type=float, default=0.3)
    ap.add_argument('--p_disj_gate', type=float, default=0.5)
    ap.add_argument('--max_split_depth', type=int, default=2)
    ap.add_argument('--gates', default='imp1,imp2,imp3,conj2,conj3,mix')
    ap.add_argument('--goal_kinds', default='atom,and,neg')
    ap.add_argument('--min_body', type=int, default=0, help='only keep written proofs with at least this many body lines')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen, kept, tries, fails = set(), 0, 0, {}
    with open(a.out, 'w') as f:
        while kept < a.n and tries < a.n * 200:
            tries += 1
            kw = sample_kw(rng, a)
            try:
                r = make_one(rng, **kw)
            except Fail as e:
                k = str(e).split(':')[0]
                fails[k] = fails.get(k, 0) + 1
                continue
            if r['n_body'] < a.min_body:
                fails['short'] = fails.get('short', 0) + 1
                continue
            key = r['text'].split(' PRF ')[0]
            if key in seen:
                continue
            seen.add(key)
            r['knobs'] = {k: v for k, v in kw.items() if k != 'gate_types'}
            f.write(json.dumps(r) + '\n')
            kept += 1
    print(f'kept {kept} of {tries} tries; fails {fails}', file=sys.stderr)


if __name__ == '__main__':
    main()
