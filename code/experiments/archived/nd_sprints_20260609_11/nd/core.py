"""Core natural-deduction objects: formulas, derivation trees, random sampler,
and linearization to Fitch-style line proofs.

Logic: propositional ND over {~, &, v, >} plus falsum F, with 10 rules:
  ANDI, ANDE1, ANDE2, IMPE, IMPI, ORI1, ORI2, ORE, NEGE, NEGI, BOTE, DN
(ANDE1/2 and ORI1/2 are the two projections/injections of AndE / OrI; counting
each connective rule once this is the standard 10-rule set: botE, negE, negI,
andI, andE, impE, impI, orI, orE, DN.)

Formulas are nested tuples:
  ('atom','P') | ('bot',) | ('not',A) | ('and',A,B) | ('or',A,B) | ('imp',A,B)
"""
import random
from dataclasses import dataclass, field
from typing import Optional

ATOMS = ['P', 'Q', 'R', 'S']

BOT = ('bot',)


def atom(name):
    return ('atom', name)


def f_not(a):
    return ('not', a)


def f_and(a, b):
    return ('and', a, b)


def f_or(a, b):
    return ('or', a, b)


def f_imp(a, b):
    return ('imp', a, b)


def formula_str(f):
    """Render a formula as a token string (space-separated, fully parenthesized)."""
    t = f[0]
    if t == 'atom':
        return f[1]
    if t == 'bot':
        return 'F'
    if t == 'not':
        return f"( ~ {formula_str(f[1])} )"
    op = {'and': '&', 'or': 'v', 'imp': '>'}[t]
    return f"( {formula_str(f[1])} {op} {formula_str(f[2])} )"


def formula_tokens(f):
    return formula_str(f).split()


def formula_size(f):
    """Number of connectives + atoms (tree size)."""
    t = f[0]
    if t in ('atom', 'bot'):
        return 1
    if t == 'not':
        return 1 + formula_size(f[1])
    return 1 + formula_size(f[1]) + formula_size(f[2])


def random_formula(rng, max_depth=2, atoms=None):
    atoms = atoms or ATOMS
    if max_depth == 0 or rng.random() < 0.40:
        # leaf: mostly atoms, occasionally bot
        if rng.random() < 0.06:
            return BOT
        return atom(rng.choice(atoms))
    r = rng.random()
    if r < 0.28:
        return f_not(random_formula(rng, max_depth - 1, atoms))
    if r < 0.52:
        return f_and(random_formula(rng, max_depth - 1, atoms),
                     random_formula(rng, max_depth - 1, atoms))
    if r < 0.76:
        return f_or(random_formula(rng, max_depth - 1, atoms),
                    random_formula(rng, max_depth - 1, atoms))
    return f_imp(random_formula(rng, max_depth - 1, atoms),
                 random_formula(rng, max_depth - 1, atoms))


# ---------------------------------------------------------------------------
# Derivation trees
# ---------------------------------------------------------------------------

_node_counter = [0]


@dataclass
class Node:
    rule: str                      # 'AS' for assumption leaves, else rule name
    concl: tuple                   # formula
    children: list = field(default_factory=list)   # list[Node]
    discharge: Optional[tuple] = None  # formula discharged (IMPI/NEGI); ORE: (A,B)
    uid: int = 0

    def __post_init__(self):
        _node_counter[0] += 1
        self.uid = _node_counter[0]


def open_assumptions(node, _discharged=None):
    """Multiset (as set of formulas) of undischarged assumption-leaf formulas."""
    if _discharged is None:
        _discharged = frozenset()
    if node.rule == 'AS':
        return set() if node.concl in _discharged else {node.concl}
    out = set()
    if node.rule == 'IMPI' or node.rule == 'NEGI':
        d = _discharged | {node.discharge}
        out |= open_assumptions(node.children[0], d)
    elif node.rule == 'ORE':
        a, b = node.discharge
        out |= open_assumptions(node.children[0], _discharged)
        out |= open_assumptions(node.children[1], _discharged | {a})
        out |= open_assumptions(node.children[2], _discharged | {b})
    else:
        for c in node.children:
            out |= open_assumptions(c, _discharged)
    return out


def tree_n_lines(node):
    """Approximate number of Fitch lines this tree will linearize to
    (rule lines + AS hypothesis lines; premises counted separately)."""
    if node.rule == 'AS':
        return 0  # premises / hypothesis lines counted by discharging parent
    n = 1
    if node.rule in ('IMPI', 'NEGI'):
        n += 1  # the hypothesis line
        n += tree_n_lines(node.children[0])
    elif node.rule == 'ORE':
        n += 2  # two hypothesis lines
        for c in node.children:
            n += tree_n_lines(c)
    else:
        for c in node.children:
            n += tree_n_lines(c)
    return n


def rule_profile(node):
    """Set of rules used in the tree."""
    if node.rule == 'AS':
        return set()
    out = {node.rule}
    for c in node.children:
        out |= rule_profile(c)
    return out


DISCHARGING = {'IMPI', 'NEGI', 'ORE'}


# ---------------------------------------------------------------------------
# Random derivation sampler (pool-based bottom-up composition)
# ---------------------------------------------------------------------------

RULES = ['ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE',
         'NEGE', 'NEGI', 'BOTE', 'DN']

# Oversample discharging rules (per proposal) and useful local rules.
DEFAULT_WEIGHTS = {
    'ANDI': 1.0, 'ANDE1': 0.7, 'ANDE2': 0.7, 'IMPE': 1.4, 'IMPI': 1.6,
    'ORI1': 0.6, 'ORI2': 0.6, 'ORE': 1.2, 'NEGE': 1.2, 'NEGI': 1.4,
    'BOTE': 0.5, 'DN': 0.7,
}


def leaf(f):
    return Node('AS', f)


class Sampler:
    def __init__(self, rng, atoms=None, max_fdepth=2, weights=None):
        self.rng = rng
        self.atoms = atoms or ATOMS
        self.max_fdepth = max_fdepth
        self.weights = weights or DEFAULT_WEIGHTS
        self.pool = []

    def rf(self, max_depth=None):
        return random_formula(self.rng, max_depth or self.max_fdepth, self.atoms)

    def _pool_with(self, pred):
        cands = [t for t in self.pool if pred(t)]
        return self.rng.choice(cands) if cands else None

    def _take(self, t):
        self.pool.remove(t)
        return t

    def _pick_or_leaf(self, pred=None, mk=None):
        """Pick a pool tree satisfying pred (consuming it), else a fresh leaf mk()."""
        if pred is not None and self.pool and self.rng.random() < 0.8:
            t = self._pool_with(pred)
            if t is not None:
                return self._take(t)
        return leaf(mk())

    def step(self):
        rng = self.rng
        rules, ws = zip(*self.weights.items())
        rule = rng.choices(rules, weights=ws)[0]
        try:
            t = self._apply(rule)
        except IndexError:
            return None
        if t is not None:
            self.pool.append(t)
        return t

    def _apply(self, rule):
        rng = self.rng
        if rule == 'ANDI':
            t1 = self._pick_or_leaf(lambda t: True, self.rf)
            t2 = self._pick_or_leaf(lambda t: True, self.rf)
            return Node('ANDI', f_and(t1.concl, t2.concl), [t1, t2])
        if rule in ('ANDE1', 'ANDE2'):
            t = self._pick_or_leaf(lambda t: t.concl[0] == 'and',
                                   lambda: f_and(self.rf(1), self.rf(1)))
            a, b = t.concl[1], t.concl[2]
            return Node(rule, a if rule == 'ANDE1' else b, [t])
        if rule == 'IMPE':
            t1 = self._pick_or_leaf(lambda t: t.concl[0] == 'imp',
                                    lambda: f_imp(self.rf(1), self.rf(1)))
            a, b = t1.concl[1], t1.concl[2]
            t2 = self._pick_or_leaf(lambda t: t.concl == a, lambda: a)
            return Node('IMPE', b, [t1, t2])
        if rule == 'IMPI':
            t = self._pick_or_leaf(lambda t: True, self.rf)
            opens = sorted(open_assumptions(t))
            if opens and rng.random() < 0.9:
                a = rng.choice(opens)
            else:
                a = self.rf(1)  # vacuous discharge
            return Node('IMPI', f_imp(a, t.concl), [t], discharge=a)
        if rule in ('ORI1', 'ORI2'):
            t = self._pick_or_leaf(lambda t: True, self.rf)
            b = self.rf(1)
            c = f_or(t.concl, b) if rule == 'ORI1' else f_or(b, t.concl)
            return Node(rule, c, [t], )
        if rule == 'ORE':
            t1 = self._pick_or_leaf(lambda t: len(open_assumptions(t)) > 0, self.rf)
            c = t1.concl
            opens1 = sorted(open_assumptions(t1))
            a = rng.choice(opens1) if opens1 else c  # leaf case: discharge itself
            t2 = self._pick_or_leaf(lambda t: t.concl == c and t.uid != t1.uid,
                                    lambda: c)
            opens2 = sorted(open_assumptions(t2))
            b = rng.choice(opens2) if opens2 else c
            t0 = self._pick_or_leaf(lambda t: t.concl == ('or', a, b),
                                    lambda: f_or(a, b))
            return Node('ORE', c, [t0, t1, t2], discharge=(a, b))
        if rule == 'NEGE':
            t1 = self._pick_or_leaf(lambda t: True, self.rf)
            a = t1.concl
            t2 = self._pick_or_leaf(lambda t: t.concl == ('not', a),
                                    lambda: f_not(a))
            return Node('NEGE', BOT, [t1, t2])
        if rule == 'NEGI':
            t = self._pick_or_leaf(lambda t: t.concl == BOT, lambda: BOT)
            opens = sorted(open_assumptions(t))
            if opens and rng.random() < 0.92:
                a = rng.choice(opens)
            else:
                a = self.rf(1)
            return Node('NEGI', f_not(a), [t], discharge=a)
        if rule == 'BOTE':
            t = self._pick_or_leaf(lambda t: t.concl == BOT, lambda: BOT)
            return Node('BOTE', self.rf(), [t])
        if rule == 'DN':
            t = self._pick_or_leaf(lambda t: t.concl[0] == 'not' and t.concl[1][0] == 'not',
                                   lambda: f_not(f_not(self.rf(1))))
            return Node('DN', t.concl[1][1], [t])
        raise ValueError(rule)


def sample_tree(rng, n_steps=8, **kw):
    """Run the sampler for n_steps rule applications, return a random pool tree
    weighted toward larger trees."""
    s = Sampler(rng, **kw)
    for _ in range(n_steps):
        s.step()
    if not s.pool:
        return None
    trees = [t for t in s.pool if t.rule != 'AS']
    if not trees:
        return None
    # weight by size so we usually return the most composed tree
    trees.sort(key=tree_n_lines)
    return trees[-1] if rng.random() < 0.8 else rng.choice(trees)


# ---------------------------------------------------------------------------
# Linearization to Fitch-style proofs
# ---------------------------------------------------------------------------

@dataclass
class Line:
    idx: int          # 1-based
    depth: int        # 0 = top level
    formula: tuple
    rule: str         # 'PR', 'AS', or rule name
    refs: list        # line indices; for box refs, (start,end) flattened


def canon_premises(fs):
    return tuple(sorted(set(fs), key=formula_str))


def linearize(tree):
    """Linearize a derivation tree into Fitch lines.

    Returns (premises, conclusion, lines).
    Raises ValueError on malformed trees (shouldn't happen)."""
    premises = canon_premises(open_assumptions(tree))
    lines = []
    prem_idx = {}
    for f in premises:
        lines.append(Line(len(lines) + 1, 0, f, 'PR', []))
        prem_idx[f] = len(lines)

    def emit(node, depth, hyp_idx):
        """hyp_idx: dict formula -> line idx for active assumption hypotheses
        (premises and open subproof hypotheses). Returns concl line idx."""
        if node.rule == 'AS':
            if node.concl in hyp_idx:
                return hyp_idx[node.concl]
            raise ValueError(f"unresolved assumption {formula_str(node.concl)}")
        if node.rule in ('IMPI', 'NEGI'):
            a = node.discharge
            lines.append(Line(len(lines) + 1, depth + 1, a, 'AS', []))
            s = len(lines)
            h2 = dict(hyp_idx)
            h2[a] = s
            e = emit(node.children[0], depth + 1, h2)
            if e < s:
                # box conclusion resolved to a line outside the box (premise or
                # outer hypothesis): reiterate it inside so the box is well-formed
                lines.append(Line(len(lines) + 1, depth + 1, node.children[0].concl
                                  if node.children[0].rule == 'AS' else lines[e - 1].formula,
                                  'R', [e]))
                e = len(lines)
            lines.append(Line(len(lines) + 1, depth, node.concl, node.rule, [s, e]))
            return len(lines)
        if node.rule == 'ORE':
            a, b = node.discharge
            t0, t1, t2 = node.children
            j = emit(t0, depth, hyp_idx)
            lines.append(Line(len(lines) + 1, depth + 1, a, 'AS', []))
            s1 = len(lines)
            h1 = dict(hyp_idx)
            h1[a] = s1
            e1 = emit(t1, depth + 1, h1)
            if e1 < s1:
                lines.append(Line(len(lines) + 1, depth + 1, lines[e1 - 1].formula,
                                  'R', [e1]))
                e1 = len(lines)
            lines.append(Line(len(lines) + 1, depth + 1, b, 'AS', []))
            s2 = len(lines)
            h2 = dict(hyp_idx)
            h2[b] = s2
            e2 = emit(t2, depth + 1, h2)
            if e2 < s2:
                lines.append(Line(len(lines) + 1, depth + 1, lines[e2 - 1].formula,
                                  'R', [e2]))
                e2 = len(lines)
            lines.append(Line(len(lines) + 1, depth, node.concl, 'ORE',
                              [j, s1, e1, s2, e2]))
            return len(lines)
        # local rules
        refs = [emit(c, depth, hyp_idx) for c in node.children]
        lines.append(Line(len(lines) + 1, depth, node.concl, node.rule, refs))
        return len(lines)

    emit(tree, 0, dict(prem_idx))
    return premises, tree.concl, lines


def proof_to_tokens(premises, conclusion, lines):
    """Full example token string: prompt + completion.

    Prompt:  THM <prem1> , <prem2> ... SEQ <concl> PRF
    Lines:   N<i> |*depth <formula> : <rule> <refs> ;
    End:     QED
    """
    toks = ['THM']
    for i, p in enumerate(premises):
        if i > 0:
            toks.append(',')
        toks += formula_tokens(p)
    toks.append('SEQ')
    toks += formula_tokens(conclusion)
    toks.append('PRF')
    for ln in lines:
        toks.append(f'N{ln.idx}')
        toks += ['|'] * ln.depth
        toks += formula_tokens(ln.formula)
        toks.append(':')
        toks.append(ln.rule)
        toks += [f'N{r}' for r in ln.refs]
        toks.append(';')
    toks.append('QED')
    return toks


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


def theorem_id(premises, conclusion):
    """Canonical id of a sequent for dedup/holdout."""
    return ' , '.join(formula_str(p) for p in premises) + ' |- ' + formula_str(conclusion)
