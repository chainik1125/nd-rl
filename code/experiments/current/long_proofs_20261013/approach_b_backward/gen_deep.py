"""Deep backward generator of long, hard-to-shortcut ND theorems (approach B).

Why gen.py's long proofs shortcut (diagnosed on data/smoke_out.jsonl): it invents premises on demand
inside boxes, and premises are global, so a fact added to satisfy one branch makes the case split that
led there redundant; it also re-derives in each branch what is derivable once outside, and its
premises often subsume each other. Here every proof is built top-down from a plan whose work is
unsharable by construction, and every premise is checked to be semantically necessary.

Plan (abstract proof tree for an atomic goal G under hypotheses H):
  case   : fresh atoms A, B; fact (A v B); each branch proves G under its own hypothesis   (3 + branches)
  imp    : fresh atom X; fact (X > G); prove X                                            (1 + child)
  leaf   : fact (h > G) for a hypothesis h in scope, or G itself as a premise atom (top level)
  bote   : fact (~ h) for a hypothesis h in scope: F by NEGE, G by BOTE                  (2)
  root   : the conclusion may be built by ANDI / IMPI / ORI on top of atomic goals.
Intermediate atoms are fresh per branch so nothing can be hoisted out of a case split.

Facts are then PACKED into 3-10 premises: facts used under the same hypothesis h may be guarded,
(h > fact), and facts with the same guard are grouped into random conjunction trees. Each guard
costs one necessary IMPE, each conjunction node two necessary ANDE lines, so line count grows
without spending atoms (atoms are capped at 10). Unpacking is emitted once, in the outermost scope
where it is possible, so the written proof has nothing minlen could hoist.

Filters: nd_verify valid; premises consistent and the sequent valid (truth table); every premise
NECESSARY (dropping it makes the sequent invalid); conclusion not a premise; every premise cited.

    python gen_deep.py OUT.jsonl --n 200 --seed 0 [--lo 20 --hi 40] [--atoms 3-10] [--prem 3-10]
"""
import argparse
import collections
import itertools
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import certify  # noqa: E402  (shared scoreboard: check(), fstr, truth_table)

BOT = ('bot',)
LETTERS = certify.LETTERS


class Fail(Exception):
    pass


class Plan:
    def __init__(self, rng, n_atoms, p_case, p_imp, p_bote, p_guard, max_depth):
        self.rng = rng
        self.free = [('atom', a) for a in rng.sample(LETTERS, n_atoms)]
        self.all_atoms = list(self.free)
        self.facts = []          # (formula, hyps_in_scope_when_used)
        self.p_case, self.p_imp, self.p_bote, self.p_guard = p_case, p_imp, p_bote, p_guard
        self.max_depth = max_depth

    def atom(self):
        if not self.free:
            raise Fail('atoms')
        return self.free.pop()

    def fact(self, f, hyps, forced_guard=None):
        self.facts.append((f, tuple(hyps), forced_guard))
        return len(self.facts) - 1

    def build(self, G, hyps, budget, depth):
        """Abstract tree deriving atom G under hypotheses hyps with about `budget` lines."""
        r = self.rng
        x = r.random()
        if budget >= 5 and depth < self.max_depth and x < self.p_case and len(self.free) >= 2:
            A, B = self.atom(), self.atom()
            # a nested split's disjunction must be guarded by the innermost hypothesis, else it is a
            # global fact that bypasses the enclosing branch (the enclosing split becomes unnecessary)
            fid = self.fact(('or', A, B), hyps, hyps[-1] if hyps else None)
            b1 = r.randint(0, budget - 3)
            return ('case', fid, A, B, self.build(G, hyps + [A], b1, depth + 1),
                    self.build(G, hyps + [B], budget - 3 - b1, depth + 1))
        if budget >= 2 and x < self.p_case + self.p_imp and self.free:
            X = self.atom()
            fid = self.fact(('imp', X, G), hyps)
            return ('imp', fid, self.build(X, hyps, budget - 1, depth))
        if hyps and r.random() < self.p_bote:
            h = hyps[-1]
            return ('bote', self.fact(('not', h), hyps), h)
        if hyps:
            h = hyps[-1]
            return ('leaf_imp', self.fact(('imp', h, G), hyps), h)
        return ('leaf_atom', self.fact(G, hyps))


def conj_tree(rng, leaves):
    """Random binary conjunction tree over leaf formulas; returns (formula, {leaf: path}) where path
    is the list of ANDE rules to extract that leaf."""
    nodes = [(f, {f: []}) for f in leaves]
    while len(nodes) > 1:
        i, j = rng.sample(range(len(nodes)), 2)
        (fa, pa), (fb, pb) = nodes[i], nodes[j]
        paths = {f: ['ANDE1'] + p for f, p in pa.items()}
        paths.update({f: ['ANDE2'] + p for f, p in pb.items()})
        nodes = [n for k, n in enumerate(nodes) if k not in (i, j)] + [(('and', fa, fb), paths)]
    return nodes[0]


class Emitter:
    def __init__(self, premises, unpack):
        """unpack: guard formula (or None) -> list of (premise_formula, guard, {leaf: ande_path})."""
        self.lines = []          # (depth, formula, rule, refs 1-based)
        self.unpack = unpack
        self.env = {}            # formula -> line number (citable in the current scope; we rebuild per box)
        for p in premises:
            self.add(0, p, 'PR', [])

    def add(self, depth, f, rule, refs):
        self.lines.append((depth, f, rule, refs))
        self.env[f] = len(self.lines)
        return len(self.lines)

    def do_unpack(self, guard, depth):
        for pf, g, paths in self.unpack.get(guard, []):
            if g is None:
                cur, src = pf, pf
            else:
                src = pf[2]
                self.add(depth, src, 'IMPE', [self.env[pf], self.env[g]])
                cur = src
            # extract every leaf: walk the tree emitting ANDE at each internal node once
            self.extract(cur, depth)

    def extract(self, f, depth):
        if f[0] == 'and' and f in self.env and self.is_internal(f):
            self.add(depth, f[1], 'ANDE1', [self.env[f]])
            self.add(depth, f[2], 'ANDE2', [self.env[f]])
            self.extract(f[1], depth)
            self.extract(f[2], depth)

    def is_internal(self, f):
        return f in self.internal

    def emit(self, t, G, depth, saved):
        k = t[0]
        if k == 'case':
            _, fid, A, B, ta, tb = t
            D = ('or', A, B)
            dl = self.env[D]
            env0 = dict(self.env)
            s1 = self.add(depth + 1, A, 'AS', [])
            self.do_unpack(A, depth + 1)
            e1 = self.emit(ta, G, depth + 1, saved)
            self.env = dict(env0)
            s2 = self.add(depth + 1, B, 'AS', [])
            self.do_unpack(B, depth + 1)
            e2 = self.emit(tb, G, depth + 1, saved)
            self.env = env0
            return self.add(depth, G, 'ORE', [dl, s1, e1, s2, e2])
        if k == 'imp':
            _, fid, child, X = t
            xl = self.emit(child, X, depth, saved)
            return self.add(depth, G, 'IMPE', [self.env[('imp', X, G)], xl])
        if k == 'bote':
            _, fid, h = t
            fl = self.add(depth, BOT, 'NEGE', [self.env[h], self.env[('not', h)]])
            return self.add(depth, G, 'BOTE', [fl])
        if k == 'leaf_imp':
            _, fid, h = t
            return self.add(depth, G, 'IMPE', [self.env[('imp', h, G)], self.env[h]])
        if k == 'leaf_atom':
            return self.env[G]
        raise ValueError(k)


def generate(rng, n_atoms, lo, hi, prem_lo, prem_hi, knobs):
    budget = rng.randint(lo, hi)
    plan = Plan(rng, n_atoms, knobs['p_case'], knobs['p_imp'], knobs['p_bote'], knobs['p_guard'], knobs['max_depth'])
    # root shape
    G = plan.atom()
    root_kind = rng.random()
    root = None
    if root_kind < knobs['p_root_and'] and len(plan.free) >= 3:
        G2 = plan.atom()
        b1 = rng.randint(budget // 4, 3 * budget // 4)
        t1 = plan.build(G, [], b1, 0)
        t2 = plan.build(G2, [], budget - 1 - b1, 0)
        concl, root = ('and', G, G2), ('andi', t1, G, t2, G2)
    elif root_kind < knobs['p_root_and'] + knobs['p_root_imp'] and len(plan.free) >= 2:
        A = plan.atom()
        t = plan.build(G, [A], budget - 2, 1)
        concl, root = ('imp', A, G), ('impi', A, t, G)
    else:
        t = plan.build(G, [], budget, 0)
        concl, root = G, ('plain', t, G)
    # shuffle conjuncts: ( b v a ) from a premise fact ( a v b ) (5 lines)
    # (fresh atoms only: a disjunction over a hypothesis atom lets minlen case-split on it at the top
    # level and derive intermediates without the planned structure; seen as 15->14 and 11->10 shortcuts)
    shuffles = []
    for _ in range(rng.randint(0, knobs['max_shuffle'])):
        if len(plan.free) < 2:
            break
        a, b = plan.atom(), plan.atom()
        plan.fact(('or', a, b), [])
        shuffles.append((a, b))
        concl = ('and', concl, ('or', b, a))
    # annotate 'imp' nodes with X (the child's goal) for the emitter
    root = annotate(root, plan.facts)
    # ---- pack facts into premises
    facts = plan.facts
    if len({f for f, _, _ in facts}) != len(facts):
        raise Fail('duplicate fact')
    groups = collections.defaultdict(list)       # guard -> list of fact formulas
    for f, hyps, forced in facts:
        guard = forced
        if guard is None and hyps and rng.random() < knobs['p_guard']:
            guard = rng.choice(hyps)
        groups[guard].append(f)
    if not necessary_premises([f if g is None else ('imp', g, f) for g, fs in groups.items() for f in fs], concl):
        raise Fail('unnecessary fact')
    # number of premises: split each group into random conjunction trees
    n_facts = len(facts)
    n_prem_target = rng.randint(prem_lo, prem_hi)
    if n_prem_target > n_facts:
        n_prem_target = n_facts
    premises, unpack, internal = [], collections.defaultdict(list), set()
    # distribute premise count across groups proportionally (at least 1 per group)
    gkeys = list(groups)
    rng.shuffle(gkeys)
    alloc = {g: 1 for g in gkeys}
    extra = n_prem_target - len(gkeys)
    while extra > 0:
        g = rng.choice(gkeys)
        if alloc[g] < len(groups[g]):
            alloc[g] += 1
            extra -= 1
        elif all(alloc[k] >= len(groups[k]) for k in gkeys):
            break
    for g in gkeys:
        fs = list(groups[g])
        rng.shuffle(fs)
        parts = [[] for _ in range(alloc[g])]
        for i, f in enumerate(fs):
            parts[i % alloc[g]].append(f)
        for part in parts:
            if len(set(part)) != len(part):
                raise Fail('dup fact in a tree')
            tree, paths = conj_tree(rng, part)
            for leaf in part:
                node = tree
                for step in paths[leaf]:
                    internal.add(node)
                    node = node[1] if step == 'ANDE1' else node[2]
            pf = tree if g is None else ('imp', g, tree)
            premises.append(pf)
            unpack[g].append((pf, g, paths))
    if len(set(premises)) != len(premises):
        raise Fail('duplicate premise')
    if not (prem_lo <= len(premises) <= prem_hi):
        raise Fail('premise count')
    rng.shuffle(premises)
    # ---- emit
    em = Emitter(premises, unpack)
    em.internal = internal
    em.do_unpack(None, 0)
    kind = root[0]
    if kind == 'andi':
        _, t1, G1, t2, G2 = root
        l1 = em.emit(t1, G1, 0, None)
        l2 = em.emit(t2, G2, 0, None)
        cur = ('and', G1, G2)
        last = em.add(0, cur, 'ANDI', [l1, l2])
    elif kind == 'impi':
        _, A, t, G1 = root
        s = em.add(1, A, 'AS', [])
        em.do_unpack(A, 1)
        e = em.emit(t, G1, 1, None)
        cur = ('imp', A, G1)
        last = em.add(0, cur, 'IMPI', [s, e])
    else:
        _, t, G1 = root
        cur = G1
        last = em.emit(t, G1, 0, None)
    for a, b in shuffles:
        dl = em.env[('or', a, b)]
        s1 = em.add(1, a, 'AS', [])
        e1 = em.add(1, ('or', b, a), 'ORI2', [s1])
        s2 = em.add(1, b, 'AS', [])
        e2 = em.add(1, ('or', b, a), 'ORI1', [s2])
        sl = em.add(0, ('or', b, a), 'ORE', [dl, s1, e1, s2, e2])
        cur = ('and', cur, ('or', b, a))
        last = em.add(0, cur, 'ANDI', [last, sl])
    if last != len(em.lines) or cur != concl:
        raise Fail('conclusion not last')
    toks = []
    for i, (d, f, rule, refs) in enumerate(em.lines):
        toks += [f'N{i + 1}'] + ['|'] * d + [certify.fstr(f), ':', rule] + [f'N{r}' for r in refs] + [';']
    prompt = 'THM ' + (' , '.join(certify.fstr(p) for p in premises) + ' ' if premises else '') + 'SEQ ' + certify.fstr(concl) + ' PRF'
    return prompt + ' ' + ' '.join(toks) + ' QED', premises, concl


def annotate(root, facts):
    def ann(t):
        k = t[0]
        if k == 'case':
            return ('case', t[1], t[2], t[3], ann(t[4]), ann(t[5]))
        if k == 'imp':
            X = facts[t[1]][0][1]
            return ('imp', t[1], ann(t[2]), X)
        return t
    if root[0] == 'andi':
        return ('andi', ann(root[1]), root[2], ann(root[3]), root[4])
    if root[0] == 'impi':
        return ('impi', root[1], ann(root[2]), root[3])
    return ('plain', ann(root[1]), root[2])


def necessary_premises(prem, concl):
    """Every premise is needed: dropping any one makes the sequent invalid."""
    for i in range(len(prem)):
        rest = prem[:i] + prem[i + 1:]
        _, valid = certify.truth_table(rest, concl)
        if valid:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lo', type=int, default=20, help='line budget range for the plan (body lines, roughly)')
    ap.add_argument('--hi', type=int, default=40)
    ap.add_argument('--atoms', default='3-10')
    ap.add_argument('--prem', default='3-10')
    ap.add_argument('--body', default='20-200', help='keep only written body lengths in this range')
    ap.add_argument('--p_case', type=float, default=0.45)
    ap.add_argument('--p_imp', type=float, default=0.4)
    ap.add_argument('--p_bote', type=float, default=0.1)
    ap.add_argument('--p_guard', type=float, default=0.5)
    ap.add_argument('--p_root_and', type=float, default=0.2)
    ap.add_argument('--p_root_imp', type=float, default=0.2)
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--max_shuffle', type=int, default=0, help='up to this many ( b v a ) conjuncts from ( a v b ) premises')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    alo, ahi = map(int, a.atoms.split('-'))
    plo, phi = map(int, a.prem.split('-'))
    blo, bhi = map(int, a.body.split('-'))
    knobs = {k: getattr(a, k) for k in ('p_case', 'p_imp', 'p_bote', 'p_guard', 'p_root_and', 'p_root_imp', 'max_depth', 'max_shuffle')}
    st = collections.Counter()
    seen = set()
    t0 = time.time()
    with open(a.out, 'w') as out:
        while st['kept'] < a.n:
            st['tries'] += 1
            try:
                text, prem, concl = generate(rng, rng.randint(alo, ahi), a.lo, a.hi, plo, phi, knobs)
            except Fail as e:
                st['fail:' + str(e)] += 1
                continue
            c = certify.check(text)
            if c is None:
                st['check:' + certify.check.last_reason.split(':')[0]] += 1
                if 'invalid' in certify.check.last_reason:
                    print('BUG', certify.check.last_reason, text, file=sys.stderr)
                continue
            if not (blo <= c['n_body'] <= bhi):
                st['body range'] += 1
                continue
            if not necessary_premises(prem, concl):
                st['unnecessary premise'] += 1
                continue
            if c['key'] in seen:
                st['dup'] += 1
                continue
            seen.add(c['key'])
            st['kept'] += 1
            out.write(json.dumps({'text': text, 'n_body': c['n_body'], 'n_prem': c['n_prem'], 'n_atoms': c['n_atoms'],
                                  'max_depth': c['max_depth'], 'seed': a.seed}) + '\n')
    st['secs'] = round(time.time() - t0, 1)
    print(json.dumps(dict(st)))


if __name__ == '__main__':
    main()
