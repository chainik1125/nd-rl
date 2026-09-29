"""First-order (predicate) natural deduction: terms, formulas, substitution,
instance matching, random derivation sampler, and Fitch linearization.

Relational FOL (no function symbols). Terms are variables or constants
(constants double as eigen-parameters). Quantifier rules use the
Lemmon/Halbach eigenvariable conventions (match Carr's nd_pack):
  ALLI : conclude (all x A) from a line A[x:=p] where parameter p is fresh
         (does not occur in any premise/assumption that line depends on, nor
         in the conclusion). Cites that one line.
  ALLE : from (all x A) conclude any instance A[x:=t].
  EXI  : from a witness B = A[x:=t] conclude (exists x A).
  EXE  : from (exists x A) and a subproof [assume A[x:=p] ... derive C]
         conclude C, where p is fresh (not in (exists x A), C, or outer
         open assumptions). Cites the existential line + the subproof box.

Formula tuples:
  ('atom', Pname, (term,...)) | ('bot',)
  ('not',A) | ('and',A,B) | ('or',A,B) | ('imp',A,B)
  ('all', vname, A) | ('ex', vname, A)
Term tuples: ('v', name) variable | ('c', name) constant/parameter.
"""
import random

VARS = ['x', 'y', 'z', 'w']
CONSTS = ['a', 'b', 'c', 'd', 'e']          # also the eigen-parameter pool
PREDS = {'P': 1, 'Q': 1, 'R': 2, 'S': 2}    # name -> arity
BOT = ('bot',)


def v(n): return ('v', n)
def c(n): return ('c', n)
def atom(p, *terms): return ('atom', p, tuple(terms))
def f_not(a): return ('not', a)
def f_and(a, b): return ('and', a, b)
def f_or(a, b): return ('or', a, b)
def f_imp(a, b): return ('imp', a, b)
def f_all(x, a): return ('all', x, a)
def f_ex(x, a): return ('ex', x, a)


# --- rendering to tokens ---------------------------------------------------
def term_str(t):
    return t[1]  # variable or constant name is the token


def formula_str(f):
    k = f[0]
    if k == 'bot':
        return 'F'
    if k == 'atom':
        if not f[2]:
            return f[1]
        return f[1] + ' ( ' + ' '.join(term_str(t) for t in f[2]) + ' )'
    if k == 'not':
        return '( ~ ' + formula_str(f[1]) + ' )'
    if k in ('and', 'or', 'imp'):
        op = {'and': '&', 'or': 'v', 'imp': '>'}[k]
        return '( ' + formula_str(f[1]) + ' ' + op + ' ' + formula_str(f[2]) + ' )'
    if k == 'all':
        return '( A ' + f[1] + ' ' + formula_str(f[2]) + ' )'
    if k == 'ex':
        return '( E ' + f[1] + ' ' + formula_str(f[2]) + ' )'
    raise ValueError(k)


def formula_tokens(f):
    return formula_str(f).split()


def formula_size(f):
    k = f[0]
    if k == 'bot':
        return 1
    if k == 'atom':
        return 1 + len(f[2])
    if k == 'not':
        return 1 + formula_size(f[1])
    if k in ('and', 'or', 'imp'):
        return 1 + formula_size(f[1]) + formula_size(f[2])
    if k in ('all', 'ex'):
        return 1 + formula_size(f[2])


# --- free variables / constants -------------------------------------------
def free_vars(f, bound=frozenset()):
    k = f[0]
    if k == 'bot':
        return set()
    if k == 'atom':
        return {t[1] for t in f[2] if t[0] == 'v' and t[1] not in bound}
    if k == 'not':
        return free_vars(f[1], bound)
    if k in ('and', 'or', 'imp'):
        return free_vars(f[1], bound) | free_vars(f[2], bound)
    if k in ('all', 'ex'):
        return free_vars(f[2], bound | {f[1]})


def consts(f):
    k = f[0]
    if k == 'bot':
        return set()
    if k == 'atom':
        return {t[1] for t in f[2] if t[0] == 'c'}
    if k == 'not':
        return consts(f[1])
    if k in ('and', 'or', 'imp'):
        return consts(f[1]) | consts(f[2])
    if k in ('all', 'ex'):
        return consts(f[2])


def all_var_names(f):
    """All variable names occurring (free or bound) — for capture avoidance."""
    k = f[0]
    if k == 'bot':
        return set()
    if k == 'atom':
        return {t[1] for t in f[2] if t[0] == 'v'}
    if k == 'not':
        return all_var_names(f[1])
    if k in ('and', 'or', 'imp'):
        return all_var_names(f[1]) | all_var_names(f[2])
    if k in ('all', 'ex'):
        return {f[1]} | all_var_names(f[2])


# --- capture-avoiding substitution f[x := t] -------------------------------
def _term_sub(term, x, t):
    return t if (term[0] == 'v' and term[1] == x) else term


def subst(f, x, t):
    """Substitute term t for free variable x in f (capture-avoiding)."""
    k = f[0]
    if k == 'bot':
        return f
    if k == 'atom':
        return ('atom', f[1], tuple(_term_sub(tm, x, t) for tm in f[2]))
    if k == 'not':
        return ('not', subst(f[1], x, t))
    if k in ('and', 'or', 'imp'):
        return (k, subst(f[1], x, t), subst(f[2], x, t))
    if k in ('all', 'ex'):
        if f[1] == x:
            return f  # x bound here, no free occurrence below
        # capture check: if t contains variable f[1] free, rename bound var
        if t[0] == 'v' and t[1] == f[1]:
            fresh = _fresh_var(all_var_names(f) | {t[1], x})
            renamed = subst(f[2], f[1], ('v', fresh))
            return (k, fresh, subst(renamed, x, t))
        return (k, f[1], subst(f[2], x, t))


def _fresh_var(used):
    for n in VARS + [f'x{i}' for i in range(10)]:
        if n not in used:
            return n
    raise RuntimeError('out of variables')


# --- instance matching: is C an instance A[x := t]? ------------------------
def match_instance(A, x, C):
    """Return the term t such that A[x:=t] == C (generalizing the positions
    where x is free in A), or None if C is not such an instance. Bound x in A
    is treated as not substitutable."""
    found = [None]

    def walk(a, cc, bound):
        if a[0] != cc[0]:
            # a may be the variable x (free) matching anything in cc
            if a[0] == 'atom':
                return False
            return False
        k = a[0]
        if k == 'bot':
            return True
        if k == 'atom':
            if a[1] != cc[1] or len(a[2]) != len(cc[2]):
                return False
            for ta, tc in zip(a[2], cc[2]):
                if ta == ('v', x) and x not in bound:
                    if found[0] is None:
                        found[0] = tc
                    elif found[0] != tc:
                        return False
                else:
                    if ta != tc:
                        return False
            return True
        if k == 'not':
            return walk(a[1], cc[1], bound)
        if k in ('and', 'or', 'imp'):
            return walk(a[1], cc[1], bound) and walk(a[2], cc[2], bound)
        if k in ('all', 'ex'):
            if a[1] != cc[1]:
                return False
            return walk(a[2], cc[2], bound | {a[1]})
        return False

    if walk(A, C, frozenset()):
        # if x does not occur free in A, t is unconstrained but C must == A
        return found[0] if found[0] is not None else ('v', x)
    return None


def theorem_id(premises, conclusion):
    return ' , '.join(formula_str(p) for p in premises) + ' |- ' + formula_str(conclusion)


def canon_premises(fs):
    return tuple(sorted(set(fs), key=formula_str))


def prompt_tokens(premises, conclusion):
    toks = ['THM']
    for i, p in enumerate(premises):
        if i > 0:
            toks.append(',')
        toks += formula_tokens(p)
    toks.append('SEQ')
    toks += formula_tokens(conclusion)
    toks.append('PRF')
    return toks


# === derivation trees, sampler, linearizer ================================
from dataclasses import dataclass, field
from typing import Optional

_uid = [0]


@dataclass
class Node:
    rule: str
    concl: tuple
    children: list = field(default_factory=list)
    discharge: Optional[tuple] = None   # box assumption formula (IMPI/NEGI/EXE) or (a,b) for ORE
    uid: int = 0

    def __post_init__(self):
        _uid[0] += 1
        self.uid = _uid[0]


def leaf(f):
    return Node('AS', f)


DISCHARureLESS = None


def open_assumptions(node, discharged=frozenset()):
    if node.rule == 'AS':
        return set() if node.concl in discharged else {node.concl}
    if node.rule in ('IMPI', 'NEGI'):
        return open_assumptions(node.children[0], discharged | {node.discharge})
    if node.rule == 'ORE':
        a, b = node.discharge
        o = open_assumptions(node.children[0], discharged)
        o |= open_assumptions(node.children[1], discharged | {a})
        o |= open_assumptions(node.children[2], discharged | {b})
        return o
    if node.rule == 'EXE':
        o = open_assumptions(node.children[0], discharged)
        o |= open_assumptions(node.children[1], discharged | {node.discharge})
        return o
    out = set()
    for ch in node.children:
        out |= open_assumptions(ch, discharged)
    return out


def tree_lines(node):
    if node.rule == 'AS':
        return 0
    if node.rule in ('IMPI', 'NEGI'):
        return 2 + tree_lines(node.children[0])
    if node.rule == 'ORE':
        return 3 + sum(tree_lines(c) for c in node.children)
    if node.rule == 'EXE':
        return 2 + tree_lines(node.children[0]) + tree_lines(node.children[1])
    return 1 + sum(tree_lines(c) for c in node.children)


@dataclass
class Line:
    idx: int
    depth: int
    formula: tuple
    rule: str
    refs: list


def linearize(tree):
    premises = canon_premises(open_assumptions(tree))
    lines = []
    prem_idx = {}
    for f in premises:
        lines.append(Line(len(lines) + 1, 0, f, 'PR', []))
        prem_idx[f] = len(lines)

    def emit(node, depth, hyp):
        if node.rule == 'AS':
            if node.concl in hyp:
                return hyp[node.concl]
            raise ValueError('unresolved assumption ' + formula_str(node.concl))
        if node.rule in ('IMPI', 'NEGI'):
            a = node.discharge
            lines.append(Line(len(lines) + 1, depth + 1, a, 'AS', []))
            s = len(lines)
            h2 = dict(hyp); h2[a] = s
            e = emit(node.children[0], depth + 1, h2)
            if e < s:
                lines.append(Line(len(lines) + 1, depth + 1, lines[e - 1].formula, 'R', [e])); e = len(lines)
            lines.append(Line(len(lines) + 1, depth, node.concl, node.rule, [s, e]))
            return len(lines)
        if node.rule == 'ORE':
            a, b = node.discharge
            t0, t1, t2 = node.children
            j = emit(t0, depth, hyp)
            lines.append(Line(len(lines) + 1, depth + 1, a, 'AS', [])); s1 = len(lines)
            h1 = dict(hyp); h1[a] = s1
            e1 = emit(t1, depth + 1, h1)
            if e1 < s1:
                lines.append(Line(len(lines) + 1, depth + 1, lines[e1 - 1].formula, 'R', [e1])); e1 = len(lines)
            lines.append(Line(len(lines) + 1, depth + 1, b, 'AS', [])); s2 = len(lines)
            h2 = dict(hyp); h2[b] = s2
            e2 = emit(t2, depth + 1, h2)
            if e2 < s2:
                lines.append(Line(len(lines) + 1, depth + 1, lines[e2 - 1].formula, 'R', [e2])); e2 = len(lines)
            lines.append(Line(len(lines) + 1, depth, node.concl, 'ORE', [j, s1, e1, s2, e2]))
            return len(lines)
        if node.rule == 'EXE':
            t0, sub = node.children
            j = emit(t0, depth, hyp)
            a = node.discharge
            lines.append(Line(len(lines) + 1, depth + 1, a, 'AS', [])); s = len(lines)
            h2 = dict(hyp); h2[a] = s
            e = emit(sub, depth + 1, h2)
            if e < s:
                lines.append(Line(len(lines) + 1, depth + 1, lines[e - 1].formula, 'R', [e])); e = len(lines)
            lines.append(Line(len(lines) + 1, depth, node.concl, 'EXE', [j, s, e]))
            return len(lines)
        # local rules (incl ALLI/ALLE/EXI): cite children in order
        refs = [emit(c, depth, hyp) for c in node.children]
        lines.append(Line(len(lines) + 1, depth, node.concl, node.rule, refs))
        return len(lines)

    emit(tree, 0, dict(prem_idx))
    return premises, tree.concl, lines


def proof_to_tokens(premises, conclusion, lines):
    toks = ['THM']
    for i, p in enumerate(premises):
        if i:
            toks.append(',')
        toks += formula_tokens(p)
    toks.append('SEQ'); toks += formula_tokens(conclusion); toks.append('PRF')
    for ln in lines:
        toks.append(f'N{ln.idx}')
        toks += ['|'] * ln.depth
        toks += formula_tokens(ln.formula)
        toks.append(':'); toks.append(ln.rule)
        toks += [f'N{r}' for r in ln.refs]
        toks.append(';')
    toks.append('QED')
    return toks


# === random proof sampler =================================================
def rand_atom(rng, terms_pool):
    p = rng.choice(list(PREDS))
    ar = PREDS[p]
    return ('atom', p, tuple(rng.choice(terms_pool) for _ in range(ar)))


def rand_formula(rng, depth, terms_pool):
    if depth <= 0 or rng.random() < 0.45:
        if rng.random() < 0.05:
            return BOT
        return rand_atom(rng, terms_pool)
    r = rng.random()
    if r < 0.22:
        return f_not(rand_formula(rng, depth - 1, terms_pool))
    if r < 0.42:
        return f_and(rand_formula(rng, depth - 1, terms_pool), rand_formula(rng, depth - 1, terms_pool))
    if r < 0.60:
        return f_or(rand_formula(rng, depth - 1, terms_pool), rand_formula(rng, depth - 1, terms_pool))
    if r < 0.78:
        return f_imp(rand_formula(rng, depth - 1, terms_pool), rand_formula(rng, depth - 1, terms_pool))
    # quantifier over a variable
    x = rng.choice(VARS)
    body = rand_formula(rng, depth - 1, terms_pool + [('v', x)])
    if x not in free_vars(body):  # ensure non-vacuous mostly
        body = f_or(body, ('atom', rng.choice([p for p, a in PREDS.items() if a == 1]), (('v', x),)))
    return (rng.choice(['all', 'ex']), x, body)


def _replace_term(f, old, newt):
    """replace all occurrences of term `old` (a const/var term) with newt."""
    k = f[0]
    if k == 'bot':
        return f
    if k == 'atom':
        return ('atom', f[1], tuple(newt if t == old else t for t in f[2]))
    if k == 'not':
        return ('not', _replace_term(f[1], old, newt))
    if k in ('and', 'or', 'imp'):
        return (k, _replace_term(f[1], old, newt), _replace_term(f[2], old, newt))
    if k in ('all', 'ex'):
        return (k, f[1], _replace_term(f[2], old, newt))


class Sampler:
    def __init__(self, rng, consts_inv=None, var_inv=None, max_fdepth=2):
        self.rng = rng
        self.consts = consts_inv or CONSTS[:3]
        self.vars = var_inv or VARS[:2]
        self.max_fdepth = max_fdepth
        self.pool = []
        self.tpool = [('c', x) for x in self.consts] + [('v', x) for x in self.vars]

    def seed(self, n=3):
        for _ in range(n):
            self.pool.append(leaf(rand_formula(self.rng, self.max_fdepth, self.tpool)))

    def _pick(self, pred):
        cands = [t for t in self.pool if pred(t)]
        return self.rng.choice(cands) if cands else None

    def step(self):
        rng = self.rng
        rules = ['ANDI', 'ANDE', 'IMPE', 'IMPI', 'ORI', 'NEGE', 'NEGI', 'DN',
                 'ALLE', 'EXI', 'ALLI', 'BOTE']
        rule = rng.choice(rules)
        try:
            t = getattr(self, '_r_' + rule)()
        except (IndexError, ValueError):
            t = None
        if t is not None:
            self.pool.append(t)
        return t

    def _r_ANDI(self):
        a, b = self._pick(lambda t: True), self._pick(lambda t: True)
        if a is None or b is None:
            return None
        return Node('ANDI', f_and(a.concl, b.concl), [a, b])

    def _r_ANDE(self):
        t = self._pick(lambda t: t.concl[0] == 'and')
        if t is None:
            return None
        if self.rng.random() < 0.5:
            return Node('ANDE1', t.concl[1], [t])
        return Node('ANDE2', t.concl[2], [t])

    def _r_IMPE(self):
        t1 = self._pick(lambda t: t.concl[0] == 'imp')
        if t1 is None:
            return None
        t2 = self._pick(lambda t: t.concl == t1.concl[1])
        if t2 is None:
            return None
        return Node('IMPE', t1.concl[2], [t1, t2])

    def _r_IMPI(self):
        t = self._pick(lambda t: True)
        if t is None:
            return None
        opens = sorted(open_assumptions(t), key=formula_str)
        a = self.rng.choice(opens) if opens and self.rng.random() < 0.9 else rand_formula(self.rng, 1, self.tpool)
        return Node('IMPI', f_imp(a, t.concl), [t], discharge=a)

    def _r_ORI(self):
        t = self._pick(lambda t: True)
        if t is None:
            return None
        b = rand_formula(self.rng, 1, self.tpool)
        if self.rng.random() < 0.5:
            return Node('ORI1', f_or(t.concl, b), [t])
        return Node('ORI2', f_or(b, t.concl), [t])

    def _r_NEGE(self):
        t1 = self._pick(lambda t: True)
        if t1 is None:
            return None
        t2 = self._pick(lambda t: t.concl == ('not', t1.concl))
        if t2 is None:
            return None
        return Node('NEGE', BOT, [t1, t2])

    def _r_NEGI(self):
        t = self._pick(lambda t: t.concl == BOT)
        if t is None:
            return None
        opens = sorted(open_assumptions(t), key=formula_str)
        a = self.rng.choice(opens) if opens else rand_formula(self.rng, 1, self.tpool)
        return Node('NEGI', f_not(a), [t], discharge=a)

    def _r_BOTE(self):
        t = self._pick(lambda t: t.concl == BOT)
        if t is None:
            return None
        return Node('BOTE', rand_formula(self.rng, 1, self.tpool), [t])

    def _r_DN(self):
        t = self._pick(lambda t: t.concl[0] == 'not' and t.concl[1][0] == 'not')
        if t is None:
            return None
        return Node('DN', t.concl[1][1], [t])

    def _r_ALLE(self):
        t = self._pick(lambda t: t.concl[0] == 'all')
        if t is None:
            return None
        x, A = t.concl[1], t.concl[2]
        term = self.rng.choice(self.tpool)
        return Node('ALLE', subst(A, x, term), [t])

    def _r_EXI(self):
        t = self._pick(lambda t: len(consts(t.concl)) > 0)
        if t is None:
            return None
        ct = self.rng.choice(sorted(consts(t.concl)))
        x = self.rng.choice([vv for vv in VARS if vv not in all_var_names(t.concl)] or ['x'])
        A = _replace_term(t.concl, ('c', ct), ('v', x))
        return Node('EXI', ('ex', x, A), [t])

    def _r_ALLI(self):
        # eigenparameter p in concl, fresh w.r.t. open assumptions
        for t in self.rng.sample(self.pool, len(self.pool)):
            cs = consts(t.concl)
            if not cs:
                continue
            oac = set().union(*[consts(o) for o in open_assumptions(t)]) if open_assumptions(t) else set()
            free = [p for p in cs if p not in oac]
            if not free:
                continue
            p = self.rng.choice(free)
            x = self.rng.choice([vv for vv in VARS if vv not in all_var_names(t.concl)] or ['x'])
            A = _replace_term(t.concl, ('c', p), ('v', x))
            return Node('ALLI', ('all', x, A), [t])
        return None


def sample_tree(rng, n_steps=10, consts_inv=None, var_inv=None, max_fdepth=2,
                min_lines=2, max_lines=30):
    s = Sampler(rng, consts_inv=consts_inv, var_inv=var_inv, max_fdepth=max_fdepth)
    s.seed(rng.randint(2, 4))
    for _ in range(n_steps):
        s.step()
    cands = [t for t in s.pool if t.rule != 'AS']
    cands = [t for t in cands if min_lines <= tree_lines(t) <= max_lines]
    # prefer trees that use quantifier rules
    def uses_q(t):
        if t.rule in ('ALLI', 'ALLE', 'EXI', 'EXE'):
            return True
        return any(uses_q(c) for c in t.children)
    q = [t for t in cands if uses_q(t)]
    pool = q if (q and rng.random() < 0.85) else cands
    if not pool:
        return None
    pool.sort(key=tree_lines)
    return pool[-1] if rng.random() < 0.6 else rng.choice(pool)
