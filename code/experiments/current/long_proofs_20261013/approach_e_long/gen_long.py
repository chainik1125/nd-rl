"""Long ND theorems (35-80 written body lines) from <= 10 atoms by RECYCLING atoms behind guards (approach E).

Why gen_deep (approach B) caps at ~34 lines: every case split / IMPE step spends fresh atoms, so with 10 atoms
the plan tree has ~12 facts. Here atoms are recycled between scopes that can never be open at the same time.

Scopes. A scope is opened by a GUARD atom c: the hypothesis of a case-split branch (box), or an atom m that was
just derived (a "seq" block). Every fact a scope needs lives in its PACKAGE, (c > conj-tree of items), and a
package sits inside its parent scope's package, so a fact used under path (c1, ..., ck) is only reachable after
IMPE with c1, then c2, ..., then ck. Unlocking costs one IMPE per scope and two ANDE per conjunction node.

Atom recycling. When choosing new atoms (split disjuncts, IMPE intermediates, seq atoms) for a scope, only atoms
mentioned by facts of the scope and its ANCESTORS (plus the goals and guards on the path) are forbidden. Sibling
and cousin subtrees therefore reuse the same atoms in different roles; nothing from one branch helps the other
because its facts are locked behind a guard that is not derivable there (sibling disjuncts and path guards are
forbidden as goals and hypotheses in the subtree).

Plan nodes (goal g at scope p):
  split : fact (c1 v c2) in p; child scopes p+c1, p+c2 each prove g                      ORE + 2 AS + unlocks
  imp   : fact (x > g) in p; prove x in p                                                  1 IMPE
  seq   : prove m in p; package (m > ...) in p opens scope p+m, which proves g             1 IMPE unlock
  leaf  : fact g in p's package (0 extra lines: the ANDE/IMPE that extracts it is the line)
  hleaf : fact (h > g) with h an outer guard on the path                                  1 IMPE
  bote  : fact ~h, h the scope's guard (only as the whole content of a split branch)      NEGE + BOTE
Root: atom G, (G1 & G2) (two root trees, ANDI) or (H > G) (IMPI box opening scope H).
Premises: the root scope's items, packed into 3-10 conjunction trees; if there are too few, items are LIFTED out
of a child package into its parent, wrapped by the child's guard: (c > item).

Filters: every fact necessary under its full path guard (dropping it makes the sequent invalid, bitmask truth
tables); certify.check (verified, consistent, all premises cited, no duplicate premises); no dead lines (every
non-final line is cited); gen_deep.necessary_premises on the final premises.

    python gen_long.py OUT.jsonl --n 200 --seed 0 [--body 35-80] [--per_len 0] [--atoms 8-10] [--prem 3-10]
--per_len N keeps at most N records per written body length (even spread over --body).
"""
import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'approach_b_backward'))
import certify  # noqa: E402
from gen_deep import necessary_premises, conj_tree  # noqa: E402

BOT = ('bot',)
LETTERS = certify.LETTERS


class Fail(Exception):
    pass


# ---------------------------------------------------------------- truth tables as bitmasks (fast filters)
def tt_masks(names):
    n = len(names)
    rows = 1 << n
    full = (1 << rows) - 1
    m = {}
    for i, a in enumerate(names):
        bits = 0
        for r in range(rows):
            if (r >> i) & 1:
                bits |= 1 << r
        m[a] = bits
    return m, full


def tt(f, m, full, cache):
    if f in cache:
        return cache[f]
    k = f[0]
    if k == 'atom':
        v = m[f[1]]
    elif k == 'bot':
        v = 0
    elif k == 'not':
        v = full & ~tt(f[1], m, full, cache)
    else:
        a, b = tt(f[1], m, full, cache), tt(f[2], m, full, cache)
        v = a & b if k == 'and' else a | b if k == 'or' else (full & ~a) | b
    cache[f] = v
    return v


def all_necessary(prem, concl):
    """(consistent, valid, every premise necessary) via bitmask truth tables."""
    names = sorted(set().union(*[certify.atoms(f, set()) for f in prem + [concl]]))
    m, full = tt_masks(names)
    cache = {}
    ps = [tt(p, m, full, cache) for p in prem]
    c = tt(concl, m, full, cache)
    n = len(ps)
    pre = [full] * (n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] & ps[i]
    suf = [full] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] & ps[i]
    if pre[n] == 0:
        return False, False, False
    if pre[n] & ~c & full:
        return True, False, False
    for i in range(n):
        if (pre[i] & suf[i + 1]) & ~c & full == 0:
            return True, True, False
    return True, True, True


# ---------------------------------------------------------------- plan
class Scope:
    def __init__(self, parent, guard):
        self.parent, self.guard = parent, guard
        self.local = set()          # atoms forbidden as new atoms in this scope's subtree
        self.items = []             # ('fact', f) | ('scope', Scope) | ('wrap', guard, item)
        self.children = []
        self.splits, self.seqs = [], []   # child scope pairs of case splits here; scopes opened by seq atoms
        if guard is not None:
            self.local.add(guard[1])
        if parent is not None:
            parent.children.append(self)
            parent.items.append(('scope', self))

    def forb(self):
        s, out = self, set()
        while s is not None:
            out |= s.local
            s = s.parent
        return out

    def path(self):
        s, out = self, []
        while s is not None and s.guard is not None:
            out.append(s.guard)
            s = s.parent
        return out[::-1]

    def add_fact(self, f):
        self.items.append(('fact', f))
        self.local |= certify.atoms(f, set())


class Planner:
    def __init__(self, rng, atoms, knobs):
        self.rng, self.atoms, self.k = rng, atoms, knobs
        self.facts = []             # (formula, scope)
        self.picks = []             # every newly chosen atom (split disjuncts, IMPE intermediates, seq atoms)

    def fact(self, scope, f):
        scope.add_fact(f)
        self.facts.append((f, scope))

    def free(self, scope):
        fb = scope.forb()
        return [a for a in self.atoms if a[1] not in fb]

    def clone(self, src, parent, guard):
        """Copy src's scope subtree (facts and nested scopes) under parent with a new guard."""
        new = Scope(parent, guard)
        for it in src.items:
            if it[0] == 'fact':
                self.fact(new, it[1])
            else:
                self.clone(it[1], new, it[1].guard)    # Scope() appends ('scope', copy) to new.items
        pairs = {id(a): b for a, b in zip(src.children, new.children)}
        new.splits = [(pairs[id(a)], pairs[id(b)]) for a, b in src.splits]
        new.seqs = [pairs[id(a)] for a in src.seqs]
        return new

    def build(self, scope, g, budget, depth, is_branch=False, twin=False):
        r, k = self.rng, self.k
        free = self.free(scope)
        x = r.random()
        root_scope = scope.guard is None
        if budget >= 24 and depth + 2 < k['max_depth'] and len(free) >= 4 and r.random() < k['p_twin']:
            # TWIN: (A v B); under A split (X v Y) with subtrees T1 | T2; under B split (Y v X) with T1 under Y and
            # T2 under X. T1 and T2 are written twice (in different boxes, so no line can be shared) but their
            # formulas enter S once. They never mention A, B, X, Y (no hleaf, no bote on the twin guards).
            A, B, X, Y = r.sample(free, 4)
            self.picks += [A, B, X, Y]
            self.fact(scope, ('or', A, B))
            sA, sB = Scope(scope, A), Scope(scope, B)
            scope.splits.append((sA, sB))
            self.fact(sA, ('or', X, Y))
            sAX, sAY = Scope(sA, X), Scope(sA, Y)
            sA.splits.append((sAX, sAY))
            rest = (budget - 15) // 2
            b1 = r.randint(rest // 4, rest - rest // 4)
            t1 = self.build(sAX, g, b1, depth + 2, False, True)
            t2 = self.build(sAY, g, rest - b1, depth + 2, False, True)
            self.fact(sB, ('or', Y, X))
            sBY = self.clone(sAX, sB, Y)
            sBX = self.clone(sAY, sB, X)
            sB.splits.append((sBY, sBX))
            return ('split', g, A, B, ('split', g, X, Y, t1, t2), ('split', g, Y, X, t1, t2))
        if is_branch and r.random() < k['p_bote'] and budget <= k['bote_max']:
            self.fact(scope, ('not', scope.guard))
            return ('bote', g, scope.guard)
        if budget >= 6 and depth < k['max_depth'] and len(free) >= 2 and x < k['p_split']:
            c1, c2 = r.sample(free, 2)
            self.picks += [c1, c2]
            self.fact(scope, ('or', c1, c2))
            s1, s2 = Scope(scope, c1), Scope(scope, c2)
            scope.splits.append((s1, s2))
            rest = budget - 5
            b1 = r.randint(rest // 4, rest - rest // 4) if rest > 0 else 0
            t1 = self.build(s1, g, b1, depth + 1, True, twin)
            # two bote branches let the ORE derive F once and BOTE once (seen as 17 -> 16 lines)
            t2 = self.build(s2, g, rest - b1, depth + 1, t1[0] != 'bote', twin)
            return ('split', g, c1, c2, t1, t2)
        if budget >= 8 and depth < k['max_depth'] and free and x < k['p_split'] + k['p_seq']:
            m = r.choice(free)
            self.picks.append(m)
            scope.local.add(m[1])
            rest = budget - 2
            b1 = r.randint(rest // 3, rest - rest // 3)
            t1 = self.build(scope, m, b1, depth, False, twin)
            sm = Scope(scope, m)
            scope.seqs.append(sm)
            t2 = self.build(sm, g, rest - b1, depth + 1, False, twin)
            return ('seq', g, m, t1, t2)
        if (budget >= 2 or root_scope) and free and x < k['p_split'] + k['p_seq'] + k['p_imp'] or (root_scope and free):
            xa = r.choice(free)
            self.picks.append(xa)
            self.fact(scope, ('imp', xa, g))
            return ('imp', g, xa, self.build(scope, xa, budget - 2, depth, False, twin))
        if root_scope:
            raise Fail('root leaf')
        path = scope.path()
        if len(path) >= 2 and not twin and r.random() < k['p_hleaf']:
            h = r.choice(path[:-1])
            self.fact(scope, ('imp', h, g))
            return ('hleaf', g, h)
        self.fact(scope, g)
        return ('leaf', g)


# ---------------------------------------------------------------- packing
def lift(rng, root, target):
    """Move items out of child packages into the parent (wrapped by the child's guard) until the root has
    >= target items. Only from packages that keep >= 1 item."""
    def scopes(s):
        yield s
        for c in s.children:
            yield from scopes(c)
    guard = 0
    while len(root.items) < target and guard < 100:
        guard += 1
        cands = [s for s in scopes(root) if s.parent is not None and len(s.items) >= 2]
        if not cands:
            break
        # prefer shallow scopes so lifted items reach the root quickly
        cands.sort(key=lambda s: len(s.path()))
        s = cands[0] if rng.random() < 0.7 else rng.choice(cands)
        it = s.items.pop(rng.randrange(len(s.items)))
        s.parent.items.append(('wrap', s.guard, it))


class Packer:
    def __init__(self, rng, share):
        self.rng, self.share = rng, share
        self.scope_items = {}       # id(scope) -> item formulas directly in its package
        self.internal = set()       # conjunction nodes of packages (to be ANDE-extracted)
        self.wrappers = set()       # guard implications (c > X) that are unlocked on scope entry
        self.made = set()

    def item_formula(self, it):
        f = self._item_formula(it)
        if f[0] != 'atom':
            # any non-atomic item twice (a fact, or a package like (W > U) from a seq) lets an ORE over two
            # sibling branches derive it once and share the rest (seen as 20 -> 17 lines)
            if f in self.made and not self.share:
                raise Fail('repeated item')
            self.made.add(f)
        return f

    def _item_formula(self, it):
        if it[0] == 'fact':
            return it[1]
        if it[0] == 'wrap':
            f = ('imp', it[1], self.item_formula(it[2]))
            self.wrappers.add(f)
            return f
        s = it[1]
        fs = [self.item_formula(i) for i in s.items]
        self.scope_items[id(s)] = fs
        f = ('imp', s.guard, self.conj(fs))
        self.wrappers.add(f)
        return f

    def conj(self, leaves):
        if len(set(leaves)) != len(leaves):
            raise Fail('dup item in package')
        if len(leaves) == 1:
            return leaves[0]
        tree, paths = conj_tree(self.rng, leaves)
        for leaf in leaves:
            node = tree
            for step in paths[leaf]:
                self.internal.add(node)
                node = node[1] if step == 'ANDE1' else node[2]
        return tree


# ---------------------------------------------------------------- emission
class Emitter:
    def __init__(self, premises, internal, wrappers):
        self.lines = []
        self.env = {}
        self.internal, self.wrappers = internal, wrappers
        for p in premises:
            self.add(0, p, 'PR', [])

    def add(self, depth, f, rule, refs):
        self.lines.append((depth, f, rule, refs))
        self.env[f] = len(self.lines)
        return len(self.lines)

    def extract(self, f, depth):
        if f in self.internal:
            self.add(depth, f[1], 'ANDE1', [self.env[f]])
            self.add(depth, f[2], 'ANDE2', [self.env[f]])
            self.extract(f[1], depth)
            self.extract(f[2], depth)

    def open_root(self):
        for f in list(self.env):
            self.extract(f, 0)

    def open_scope(self, c, depth):
        cands = [f for f in self.env if f in self.wrappers and f[1] == c]
        for f in cands:
            self.add(depth, f[2], 'IMPE', [self.env[f], self.env[c]])
            self.extract(f[2], depth)

    def emit(self, t, depth):
        k = t[0]
        if k == 'split':
            _, g, c1, c2, t1, t2 = t
            dl = self.env[('or', c1, c2)]
            env0 = dict(self.env)
            s1 = self.add(depth + 1, c1, 'AS', [])
            self.open_scope(c1, depth + 1)
            e1 = self.emit(t1, depth + 1)
            self.env = dict(env0)
            s2 = self.add(depth + 1, c2, 'AS', [])
            self.open_scope(c2, depth + 1)
            e2 = self.emit(t2, depth + 1)
            self.env = env0
            return self.add(depth, g, 'ORE', [dl, s1, e1, s2, e2])
        if k == 'imp':
            _, g, xa, child = t
            xl = self.emit(child, depth)
            return self.add(depth, g, 'IMPE', [self.env[('imp', xa, g)], xl])
        if k == 'seq':
            _, g, m, t1, t2 = t
            self.emit(t1, depth)
            self.open_scope(m, depth)
            return self.emit(t2, depth)
        if k == 'leaf':
            return self.env[t[1]]
        if k == 'hleaf':
            _, g, h = t
            return self.add(depth, g, 'IMPE', [self.env[('imp', h, g)], self.env[h]])
        if k == 'bote':
            _, g, h = t
            fl = self.add(depth, BOT, 'NEGE', [self.env[h], self.env[('not', h)]])
            return self.add(depth, g, 'BOTE', [fl])
        raise ValueError(k)


def generate(rng, n_atoms, budget, prem_lo, prem_hi, knobs):
    atoms = [('atom', a) for a in rng.sample(LETTERS, n_atoms)]
    pl = Planner(rng, atoms, knobs)
    root = Scope(None, None)
    x = rng.random()
    if x < knobs['p_root_and']:
        G1, G2 = rng.sample(atoms, 2)
        root.local |= {G1[1], G2[1]}
        b1 = rng.randint(budget // 3, 2 * budget // 3)
        t1 = pl.build(root, G1, b1, 0)
        t2 = pl.build(root, G2, budget - 1 - b1, 0)
        concl, plan = ('and', G1, G2), ('andi', t1, t2)
    elif x < knobs['p_root_and'] + knobs['p_root_imp']:
        H, G = rng.sample(atoms, 2)
        root.local |= {G[1], H[1]}
        sh = Scope(root, H)
        t = pl.build(sh, G, budget - 2, 1)
        concl, plan = ('imp', H, G), ('impi', H, t)
    else:
        G = rng.choice(atoms)
        root.local.add(G[1])
        t = pl.build(root, G, budget, 0)
        concl, plan = G, ('plain', t)
    # every fact necessary under its full path guard
    guarded = []
    for f, s in pl.facts:
        for c in reversed(s.path()):
            f = ('imp', c, f)
        guarded.append(f)
    if len(set(guarded)) != len(guarded):
        raise Fail('duplicate guarded fact')
    # no non-atomic fact twice anywhere (identical content in sibling branches lets an ORE derive it once and
    # share everything after it: seen as 11 -> 8 lines)
    nonatomic = [f for f, _ in pl.facts if f[0] != 'atom']
    if len(set(nonatomic)) != len(nonatomic) and not knobs['share']:
        raise Fail('repeated fact')
    cons, valid, nec = all_necessary(guarded, concl)
    if not (cons and valid):
        raise Fail('facts inconsistent or invalid')
    if not nec:
        raise Fail('unnecessary fact')
    # pack
    n_prem = rng.randint(prem_lo, prem_hi)
    lift(rng, root, max(n_prem, prem_lo) if rng.random() < knobs['p_lift_more'] else prem_lo)
    pk = Packer(rng, knobs['share'])
    top = [pk.item_formula(it) for it in root.items]
    pk.scope_items[id(root)] = top
    if knobs['share']:
        sibling_share_check(root, pk.scope_items)
    if len(top) < prem_lo:
        raise Fail('too few root items')
    n_prem = min(n_prem, len(top))
    rng.shuffle(top)
    parts = [[] for _ in range(n_prem)]
    for i, f in enumerate(top):
        parts[i % n_prem].append(f)
    premises = [pk.conj(p) for p in parts]
    if len(set(premises)) != len(premises):
        raise Fail('duplicate premise')
    if not (prem_lo <= len(premises) <= prem_hi):
        raise Fail('premise count')
    rng.shuffle(premises)
    em = Emitter(premises, pk.internal, pk.wrappers)
    em.open_root()
    if plan[0] == 'andi':
        l1 = em.emit(plan[1], 0)
        l2 = em.emit(plan[2], 0)
        last = em.add(0, concl, 'ANDI', [l1, l2])
    elif plan[0] == 'impi':
        H = plan[1]
        s = em.add(1, H, 'AS', [])
        em.open_scope(H, 1)
        e = em.emit(plan[2], 1)
        last = em.add(0, concl, 'IMPI', [s, e])
    else:
        last = em.emit(plan[1], 0)
    if last != len(em.lines) or em.lines[-1][1] != concl or em.lines[-1][0] != 0:
        raise Fail('conclusion not last')
    cited = {r for _, _, _, refs in em.lines for r in refs}
    if any(i + 1 not in cited for i in range(len(em.lines) - 1)):
        raise Fail('dead line')
    toks = []
    for i, (d, f, rule, refs) in enumerate(em.lines):
        toks += [f'N{i + 1}'] + ['|'] * d + [certify.fstr(f), ':', rule] + [f'N{r}' for r in refs] + [';']
    prompt = 'THM ' + ' , '.join(certify.fstr(p) for p in premises) + ' SEQ ' + certify.fstr(concl) + ' PRF'
    generate.reuse = len(pl.picks) - len(set(pl.picks))
    return prompt + ' ' + ' '.join(toks) + ' QED', premises, concl


def sibling_share_check(root, scope_items):
    """With --share, a non-atomic item may occur in several scopes, but never so that one case split's two
    branches can both derive it: then an ORE over that split derives it once and everything after it is shared.
    D(s) = items reachable in scope s without a new hypothesis: its own items, those of scopes its seq atoms
    open, and those derivable in both branches of a split at s."""
    def D(s):
        out = {f for f in scope_items[id(s)] if f[0] != 'atom'}
        for sm in s.seqs:
            out |= D(sm)
        for s1, s2 in s.splits:
            both = D(s1) & D(s2)
            if both:
                raise Fail('shared by sibling branches')
        return out
    D(root)
    for c in root.children:       # the IMPI root scope is a child without a split
        if not any(c in pair for pair in root.splits) and c not in root.seqs:
            D(c)


def s_size(premises, concl):
    sub = set()
    for f in premises + [concl]:
        _subf(f, sub)
    return len(sub | {('not', f) for f in sub} | {('not', ('not', f)) for f in sub} | {BOT})


def _subf(f, acc):
    acc.add(f)
    if f[0] == 'not':
        _subf(f[1], acc)
    elif f[0] in ('and', 'or', 'imp'):
        _subf(f[1], acc)
        _subf(f[2], acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--budget', default='30-90', help='plan line budget range (roughly body lines)')
    ap.add_argument('--atoms', default='8-10')
    ap.add_argument('--prem', default='3-10')
    ap.add_argument('--body', default='35-80', help='keep only written body lengths in this range')
    ap.add_argument('--per_len', type=int, default=0, help='max records per written body length (0 = no cap)')
    ap.add_argument('--max_S', type=int, default=230, help="max minlen |S| (certify_fast's widest build holds 239)")
    ap.add_argument('--p_split', type=float, default=0.5)
    ap.add_argument('--p_seq', type=float, default=0.15)
    ap.add_argument('--p_imp', type=float, default=0.3)
    ap.add_argument('--p_bote', type=float, default=0.15)
    ap.add_argument('--bote_max', type=int, default=6)
    ap.add_argument('--p_hleaf', type=float, default=0.15)
    ap.add_argument('--p_root_and', type=float, default=0.15)
    ap.add_argument('--p_root_imp', type=float, default=0.15)
    ap.add_argument('--p_lift_more', type=float, default=1.0)
    ap.add_argument('--max_depth', type=int, default=6)
    ap.add_argument('--p_twin', type=float, default=0.0, help='TWIN node probability (implies --share 1)')
    ap.add_argument('--share', type=int, default=0, help='1: allow repeated items outside sibling branches (smaller S)')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    alo, ahi = map(int, a.atoms.split('-'))
    plo, phi = map(int, a.prem.split('-'))
    blo, bhi = map(int, a.body.split('-'))
    glo, ghi = map(int, a.budget.split('-'))
    knobs = {k: getattr(a, k) for k in ('p_split', 'p_seq', 'p_imp', 'p_bote', 'bote_max', 'p_hleaf', 'p_root_and',
                                         'p_root_imp', 'p_lift_more', 'max_depth', 'share', 'p_twin')}
    if a.p_twin > 0:
        knobs['share'] = 1
    st = collections.Counter()
    per = collections.Counter()
    seen = set()
    t0 = time.time()
    with open(a.out, 'w') as out:
        while st['kept'] < a.n:
            st['tries'] += 1
            try:
                text, prem, concl = generate(rng, rng.randint(alo, ahi), rng.randint(glo, ghi), plo, phi, knobs)
            except Fail as e:
                st['fail:' + str(e)] += 1
                continue
            # cheap pre-checks before the slow ones
            n_body = text.count(' ; ') - len(prem)
            if not (blo <= n_body <= bhi):
                st['body range'] += 1
                continue
            if a.per_len and per[n_body] >= a.per_len:
                st['length full'] += 1
                continue
            S = s_size(prem, concl)
            if S > a.max_S:
                st['S too big'] += 1
                continue
            c = certify.check(text)
            if c is None:
                st['check:' + certify.check.last_reason.split(':')[0]] += 1
                if 'invalid' in certify.check.last_reason:
                    print('BUG', certify.check.last_reason, text, file=sys.stderr)
                continue
            assert c['n_body'] == n_body, (c['n_body'], n_body)
            if c['key'] in seen:
                st['dup'] += 1
                continue
            if not necessary_premises(prem, concl):
                st['unnecessary premise'] += 1
                continue
            seen.add(c['key'])
            per[n_body] += 1
            st['kept'] += 1
            out.write(json.dumps({'text': text, 'n_body': c['n_body'], 'n_prem': c['n_prem'], 'n_atoms': c['n_atoms'],
                                  'max_depth': c['max_depth'], 'S': S, 'reuse': generate.reuse,
                                  'seed': a.seed}) + '\n')
            out.flush()
    st['secs'] = round(time.time() - t0, 1)
    print(json.dumps(dict(st)))


if __name__ == '__main__':
    main()
