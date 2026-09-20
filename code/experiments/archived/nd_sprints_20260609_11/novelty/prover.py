"""Bounded, sound, classical propositional ND prover.

Produces derivation trees (core.Node) so every proof is linearized and checked
by the INDEPENDENT verifier (verify_text) before being returned — soundness is
inherited from the verifier, completeness is best-effort within a depth bound.

Used to compute formal proof/theorem measures:
  - is_provable / min proof length (iterative deepening)
  - proof multiplicity (count distinct verified proofs up to a cap)
  - lemma value (how many other targets a theorem unlocks / shortens)

Search = goal-directed (backward) with:
  - an elimination-rule forward closure of the available assumptions,
  - introduction rules driven by goal shape,
  - classical reductio (assume ~G, derive F) for completeness,
  - case analysis (ORE) on any derivable disjunction.
Memoized by (goal, frozenset(assumptions), depth).
"""
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.core import (Node, leaf, f_and, f_or, f_imp, f_not, BOT, formula_str,
                     formula_size, linearize, proof_to_tokens, canon_premises,
                     open_assumptions)
from nd.verify import verify_text


def _closure(assumps, max_size, cap=400):
    """Forward elimination closure: from assumption formulas, derive new
    formulas via ANDE/IMPE/NEGE/BOTE/DN, each as a Node. Returns dict
    formula -> Node (shortest-ish derivation found). Bounded by formula size
    and number of facts."""
    facts = {}
    for a in assumps:
        facts[a] = leaf(a)
    changed = True
    while changed and len(facts) < cap:
        changed = False
        cur = list(facts.items())
        for f, nd in cur:
            t = f[0]
            # ANDE
            if t == 'and':
                for proj, sub in ((1, 'ANDE1'), (2, 'ANDE2')):
                    g = f[proj]
                    if g not in facts:
                        facts[g] = Node(sub, g, [nd]); changed = True
            # DN
            if t == 'not' and f[1][0] == 'not':
                g = f[1][1]
                if g not in facts:
                    facts[g] = Node('DN', g, [nd]); changed = True
            # BOTE: from F derive anything? only on demand (skip in closure)
            for f2, nd2 in cur:
                # IMPE
                if t == 'imp' and f2 == f[1]:
                    g = f[2]
                    if g not in facts and formula_size(g) <= max_size:
                        facts[g] = Node('IMPE', g, [nd, nd2]); changed = True
                # NEGE -> F
                if t == 'not' and f2 == f[1]:
                    if BOT not in facts:
                        facts[BOT] = Node('NEGE', BOT, [nd2, nd]); changed = True
            if len(facts) >= cap:
                break
    return facts


class Prover:
    def __init__(self, max_depth=5, max_fsize=14, multiplicity_cap=32):
        self.max_depth = max_depth
        self.max_fsize = max_fsize
        self.mcap = multiplicity_cap
        self.memo = {}

    def prove(self, goal, assumps, depth):
        """Return a Node deriving goal from assumps (a frozenset), or None."""
        key = (goal, assumps, depth)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = None  # guard against loops
        res = self._prove(goal, assumps, depth)
        self.memo[key] = res
        return res

    def _prove(self, goal, assumps, depth):
        if goal in assumps:
            return leaf(goal)
        clo = _closure(assumps, self.max_fsize)
        if goal in clo:
            return clo[goal]
        if depth <= 0:
            return None

        # introduction by goal shape
        t = goal[0]
        if t == 'and':
            l = self.prove(goal[1], assumps, depth - 1)
            if l is not None:
                r = self.prove(goal[2], assumps, depth - 1)
                if r is not None:
                    return Node('ANDI', goal, [l, r])
        elif t == 'imp':
            sub = self.prove(goal[2], assumps | {goal[1]}, depth - 1)
            if sub is not None:
                return Node('IMPI', goal, [sub], discharge=goal[1])
        elif t == 'not':
            sub = self.prove(BOT, assumps | {goal[1]}, depth - 1)
            if sub is not None:
                return Node('NEGI', goal, [sub], discharge=goal[1])
        elif t == 'or':
            for inj, rule in ((goal[1], 'ORI1'), (goal[2], 'ORI2')):
                s = self.prove(inj, assumps, depth - 1)
                if s is not None:
                    return Node(rule, goal, [s])

        # backward modus ponens: prove goal via an available implication X->goal
        # by proving its antecedent X (introduces antecedents like P&Q)
        imps = [(f, nd) for f, nd in clo.items()
                if f[0] == 'imp' and f[2] == goal and f[1] not in clo]
        imps.sort(key=lambda x: formula_size(x[0][1]))
        for f, nd in imps:
            x = self.prove(f[1], assumps, depth - 1)
            if x is not None:
                return Node('IMPE', goal, [nd, x])

        # case analysis on any derivable disjunction in closure
        for f, nd in list(clo.items()):
            if f[0] == 'or' and f[1] not in assumps and f[2] not in assumps:
                a, b = f[1], f[2]
                la = self.prove(goal, assumps | {a}, depth - 1)
                if la is None:
                    continue
                rb = self.prove(goal, assumps | {b}, depth - 1)
                if rb is None:
                    continue
                return Node('ORE', goal, [nd, la, rb], discharge=(a, b))

        # classical reductio: assume ~goal, derive F, NEGI -> ~~goal, DN -> goal
        if t != 'not' and goal != BOT:
            ng = f_not(goal)
            contra = self.prove(BOT, assumps | {ng}, depth - 1)
            if contra is not None:
                nng = Node('NEGI', f_not(ng), [contra], discharge=ng)
                return Node('DN', goal, [nng])

        # to prove F: complementary pair in closure, or sub-prove H for an
        # available ~H (key for classical theorems like excluded middle/Peirce)
        if goal == BOT:
            for f, nd in list(clo.items()):
                neg = f_not(f)
                if neg in clo:
                    return Node('NEGE', BOT, [nd, clo[neg]])
            # for each available negation ~H, try to prove H
            negs = [(f, nd) for f, nd in clo.items() if f[0] == 'not']
            negs.sort(key=lambda x: formula_size(x[0][1]))
            for nf, nnd in negs:
                H = nf[1]
                if H in clo:
                    continue
                hd = self.prove(H, assumps, depth - 1)
                if hd is not None:
                    return Node('NEGE', BOT, [hd, nnd])
        return None

    # --- public measure helpers ---
    def min_proof(self, premises, conclusion, max_depth=None):
        """Iterative deepening; return (lines, tokens) of a verified minimal-ish
        proof, or None. 'minimal' = fewest search-depth then linearized lines."""
        md = max_depth or self.max_depth
        A = frozenset(premises)
        best = None
        for d in range(1, md + 1):
            self.memo = {}
            nd = self.prove(conclusion, A, d)
            if nd is not None:
                try:
                    prem, concl, lines = linearize(nd)
                except ValueError:
                    continue
                # the tree may use assumptions not in premises (shouldn't); guard
                opens = open_assumptions(nd)
                if not opens.issubset(set(premises)):
                    continue
                toks = proof_to_tokens(canon_premises(premises), conclusion, lines) \
                    if False else proof_to_tokens(prem, concl, lines)
                ok, _, nl = verify_text(' '.join(toks))
                if ok:
                    best = (nl, ' '.join(toks))
                    break
        return best

    def is_provable(self, premises, conclusion, max_depth=None):
        return self.min_proof(premises, conclusion, max_depth) is not None
