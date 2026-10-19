"""Proof-first generator of compact 7-12-line propositional ND problems (hard712).

    from code.experiments.current.thinking_traces_20261005.hard712.pfgen import sample
    cand = sample(seed)   # dict(prompt, proof, recipe, seed, n_lines, subst) or None

Each candidate is built proof first; the theorem is read off the finished proof.

1. **Skeleton.** A recipe (`RECIPES`) fixes the box nesting at the top of the proof and the kind
   of work inside: an implication box around a negation box (`IN`, `INR`, `INO`), a negation box
   around a negation box (`NN`), reductio around a negation box (`CN`), reductio with an
   elimination chain or a case split inside (`CW`, `CO`), a case split inside an implication or
   negation box (`IO`, `NO`), two implication boxes with elimination work at depth 2 (`II`), a
   goal (A>B)>C (`HO`), recipes with a biconditional premise (`B*`), and goal-shape recipes
   (`G*`) where only the goal's shape is fixed and the builder chooses the boxes.
2. **Filling.** Box hypotheses are small formulas (0-2 connectives) over 2-3 atoms. The work in
   the boxes is built backward by a randomised normal-proof builder (`Builder.prove`):
   introduction rules for compound goals; elimination paths (ANDE / IMPE chains) out of available
   formulas; contradictions X / ~X with X built from the hypotheses; case splits whose branches
   both use their case; reductio. A fact that is not available may be supplied by a new premise
   (at most 2, statement <= 7 connectives), and every new premise is shaped so that using it
   consumes an open hypothesis (no premise restates the goal).
3. **Substitution.** With probability `P_SUBST` one atom of the finished proof is replaced, in
   every line, by a literal or a one-connective formula (uniform substitution keeps an ND proof
   valid). This is where most of the statement-level variety comes from.
4. **Read-off.** The theorem is (premises used, root formula). The tree is linearised with sharing
   and hoisting (`ptree.linearize`) and re-checked by nd_verify.

No formula is taken from any data file. All randomness is drawn from `random.Random(seed)`: a
seed gives the same candidate on every run. The gates are in `pfgates.py`.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from typing import Any

sys.dont_write_bytecode = True

from .ptree import (  # noqa: E402
    BOT,
    ELIM_RULES,
    And,
    F,
    Imp,
    Node,
    Not,
    Or,
    atoms,
    conn,
    deps,
    fdepth,
    linearize,
    subs,
    walk,
)

ATOM_NAMES = ("P", "Q", "R", "S")
MAX_PREM = 2
MAX_CONN = 7
MAX_FDEPTH = 3
P_SUBST = 0.7
P_SUBST2 = 0.35
P_NO_PREMISE = 0.6

Move = Callable[[], "Node | None"]


class Fail(Exception):
    pass


def degenerate(f: F) -> bool:
    """A subformula X o X (same operands) or ~~X."""
    for g in subs(f):
        if g[0] in ("and", "or", "imp") and g[1] == g[2]:
            return True
        if g[0] == "not" and g[1][0] == "not":
            return True
    return False


class Builder:
    """One construction: atoms, premises, hypotheses, and the backward prover."""

    def __init__(self, rng: random.Random, n_atoms: int, max_prem: int = MAX_PREM) -> None:
        self.rng = rng
        self.atoms = rng.sample(ATOM_NAMES, n_atoms)
        self.max_prem = max_prem
        self.prem: list[F] = []
        self.hypf: dict[int, F] = {}
        self.nh = 0
        self.reserved = 0  # connectives of the goal (fixed by the recipe)
        self.calls = 0
        self.max_calls = 500
        self.allow_bote = False
        self.allow_raa = True
        # per-construction jitter on the builder's move weights: different proof styles
        self.jit = {
            k: rng.uniform(0.3, 3.0)
            for k in ("neg", "pos", "negprem", "ore", "intro", "path", "newprem", "raa")
        }

    # ------------------------------------------------------------------ state
    def snap(self) -> int:
        return len(self.prem)

    def restore(self, s: int) -> None:
        del self.prem[s:]

    def hyp(self, f: F) -> int:
        self.nh += 1
        self.hypf[self.nh] = f
        return self.nh

    def hnode(self, h: int) -> Node:
        return Node("HYP", self.hypf[h], ref=h)

    def budget_left(self) -> int:
        return MAX_CONN - self.reserved - sum(conn(p) for p in self.prem)

    def add_prem(self, f: F) -> Node:
        if f in self.prem:
            return Node("PR", f, ref=self.prem.index(f))
        if len(self.prem) >= self.max_prem or conn(f) > self.budget_left() or degenerate(f):
            raise Fail("premise")
        self.prem.append(f)
        return Node("PR", f, ref=len(self.prem) - 1)

    def tick(self) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise Fail("calls")

    def pick(self, opts: list[tuple[float, Any]]) -> Any:
        opts = [o for o in opts if o[0] > 0]
        return self.rng.choices([o for _, o in opts], [w for w, _ in opts])[0]

    # ------------------------------------------------------------------ formulas
    def atom(self) -> F:
        return ("atom", self.rng.choice(self.atoms))

    def lit(self, p_neg: float = 0.35) -> F:
        a = self.atom()
        return Not(a) if self.rng.random() < p_neg else a

    def small(self, maxc: int, p_leaf: float = 0.4) -> F:
        """Random formula with at most maxc connectives, no double negation, no X o X."""
        for _ in range(8):
            f = self._small(maxc, p_leaf)
            if not degenerate(f):
                return f
        return self.atom()

    def _small(self, maxc: int, p_leaf: float) -> F:
        if maxc <= 0 or self.rng.random() < p_leaf:
            return self.atom()
        op = self.rng.choice(["not", "and", "or", "imp"])
        if op == "not":
            sub = self._small(maxc - 1, p_leaf + 0.25)
            return sub if sub[0] == "not" else Not(sub)
        k = self.rng.randint(0, maxc - 1)
        return (op, self._small(k, p_leaf + 0.2), self._small(maxc - 1 - k, p_leaf + 0.2))

    def lit_or_small(self, p_neg: float = 0.3, p_small: float = 0.15) -> F:
        return self.small(1, 0.0) if self.rng.random() < p_small else self.lit(p_neg)

    def distinct_lits(self, p_neg: float = 0.3) -> tuple[F, F]:
        for _ in range(10):
            a, b = self.lit_or_small(p_neg), self.lit_or_small(p_neg)
            if a != b and a != Not(b) and b != Not(a):
                return a, b
        raise Fail("lits")

    def hypform(self, kinds: dict[str, float]) -> F:
        k = self.pick([(w, k) for k, w in kinds.items()])
        if k == "atom":
            return self.atom()
        if k == "lit":
            return self.lit(0.5)
        if k == "small2":
            return self.small(2, 0.15)
        a, b = self.distinct_lits()
        return {
            "conj": lambda: And(a, b),
            "disj": lambda: Or(a, b),
            "imp": lambda: Imp(a, b),
            "negc": lambda: Not(self.rng.choice([And, Or, Imp])(a, b)),
        }[k]()

    # ------------------------------------------------------------------ availability
    def closure(self, H: tuple[int, ...]) -> dict[F, Node]:
        """Formulas available in the context: premises, open hypotheses, and what single-step
        eliminations (ANDE, IMPE, NEGE -> F) yield from them, to a fixpoint."""
        av: dict[F, Node] = {}
        for i, p in enumerate(self.prem):
            av.setdefault(p, Node("PR", p, ref=i))
        for h in H:
            av.setdefault(self.hypf[h], self.hnode(h))
        for _ in range(4):
            changed = False
            for f, n in list(av.items()):
                if f[0] == "and":
                    for k, r in ((1, "ANDE1"), (2, "ANDE2")):
                        if f[k] not in av:
                            av[f[k]] = Node(r, f[k], (n,))
                            changed = True
                elif f[0] == "imp" and f[1] in av and f[2] not in av:
                    av[f[2]] = Node("IMPE", f[2], (n, av[f[1]]))
                    changed = True
                elif f[0] == "not" and f[1] in av and BOT not in av:
                    av[BOT] = Node("NEGE", BOT, (av[f[1]], n))
                    changed = True
            if not changed:
                break
        return av

    @staticmethod
    def paths(f: F, G: F, depth: int = 3) -> list[list[tuple[str, F]]]:
        """Elimination paths from f down to G through & and > (each step: rule, formula)."""
        if f == G:
            return [[]]
        if depth == 0:
            return []
        out = []
        if f[0] == "and":
            for k, r in ((1, "ANDE1"), (2, "ANDE2")):
                out += [[(r, f), *p] for p in Builder.paths(f[k], G, depth - 1)]
        elif f[0] == "imp":
            out += [[("IMPE", f), *p] for p in Builder.paths(f[2], G, depth - 1)]
        return out

    def run_path(
        self, n: Node, path: list[tuple[str, F]], H: tuple[int, ...], fuel: int, prefer: tuple
    ) -> Node:
        for r, f in path:
            if r == "IMPE":
                a = self.prove(f[1], H, fuel - 1, prefer)
                n = Node("IMPE", f[2], (n, a))
            else:
                n = Node(r, f[1] if r == "ANDE1" else f[2], (n,))
            fuel -= 1
        return n

    # ------------------------------------------------------------------ generic prover
    def attempt(self, fn: Move) -> Node | None:
        s = self.snap()
        try:
            r = fn()
        except Fail:
            r = None
        if r is None:
            self.restore(s)
        return r

    def first(self, moves: list[tuple[float, Move]]) -> Node | None:
        """Try the moves in a weight-proportional random order; the first success wins."""
        moves = [m for m in moves if m[0] > 0]
        while moves:
            i = self.rng.choices(range(len(moves)), [w for w, _ in moves])[0]
            fn = moves.pop(i)[1]
            r = self.attempt(fn)
            if r is not None:
                return r
        return None

    def prove(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple = ()) -> Node:
        """A derivation of G from the premises and the hypotheses H, built backward; raises Fail.
        `prefer` names hypotheses the derivation should consume first."""
        self.tick()
        if fuel <= 0:
            raise Fail("fuel")
        av = self.closure(H)
        if G in av:
            return av[G]
        if BOT in av and G != BOT:
            if not self.allow_bote:
                raise Fail("inconsistent context")
            return Node("BOTE", G, (av[BOT],))
        r = self.first(self.moves(G, H, fuel, prefer, av))
        if r is None:
            raise Fail(f"no move for {G}")
        return r

    def moves(
        self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]
    ) -> list[tuple[float, Move]]:
        mv: list[tuple[float, Move]] = []
        j = self.jit
        if G == BOT:
            mv.append((3.0 * j["neg"], lambda: self.m_bot_avail_neg(H, fuel, prefer, av)))
            mv.append((1.5 * j["pos"], lambda: self.m_bot_avail_pos(H, fuel, prefer, av)))
            mv.append((2.0 * j["negprem"], lambda: self.m_bot_new_negprem(H, fuel, prefer, av)))
            mv.append((0.7 * j["ore"], lambda: self.m_ore(G, H, fuel, prefer, av)))
            return mv
        t = G[0]
        intro = {
            "imp": lambda: self.m_impi(G, H, fuel, prefer),
            "not": lambda: self.m_negi(G, H, fuel, prefer),
            "and": lambda: self.m_andi(G, H, fuel, prefer),
            "or": lambda: self.m_ori(G, H, fuel, prefer),
        }.get(t)
        if intro is not None:
            mv.append(((3.0 if t != "or" else 1.5) * j["intro"], intro))
        mv.append((2.0 * j["path"], lambda: self.m_path_avail(G, H, fuel, prefer, av)))
        mv.append((1.3 * j["newprem"], lambda: self.m_path_newprem(G, H, fuel, prefer, av)))
        mv.append((0.8 * j["ore"], lambda: self.m_ore(G, H, fuel, prefer, av)))
        if self.allow_raa and t in ("atom", "or"):
            mv.append((0.5 * j["raa"], lambda: self.m_raa(G, H, fuel, prefer)))
        return mv

    # intro moves
    def m_impi(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple) -> Node:
        h = self.hyp(G[1])
        body = self.prove(G[2], (*H, h), fuel - 2, (h, *prefer))
        if h not in deps(body):
            raise Fail("vacuous implication")
        return Node("IMPI", G, (body,), (h,))

    def m_negi(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple) -> Node:
        h = self.hyp(G[1])
        body = self.prove(BOT, (*H, h), fuel - 2, (h, *prefer))
        if h not in deps(body):
            raise Fail("negation box ignores its hypothesis")
        return Node("NEGI", G, (body,), (h,))

    def m_raa(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple) -> Node:
        h = self.hyp(Not(G))
        body = self.prove(BOT, (*H, h), fuel - 3, (h, *prefer))
        if h not in deps(body):
            raise Fail("reductio ignores its hypothesis")
        return Node("DN", G, (Node("NEGI", Not(Not(G)), (body,), (h,)),))

    def m_andi(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple) -> Node:
        a = self.prove(G[1], H, fuel - 1, prefer)
        b = self.prove(G[2], H, fuel - 1, prefer)
        return Node("ANDI", G, (a, b))

    def m_ori(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple) -> Node | None:
        sides = [1, 2]
        self.rng.shuffle(sides)
        for k in sides:
            r = self.attempt(lambda k=k: self.prove(G[k], H, fuel - 1, prefer))
            if r is not None:
                return Node("ORI1" if k == 1 else "ORI2", G, (r,))
        return None

    # elimination moves
    def m_path_avail(
        self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]
    ) -> Node | None:
        opts = [(f, p) for f in av if f != G for p in self.paths(f, G) if p]
        opts = [(f, p) for f, p in opts if any(r == "IMPE" for r, _ in p)]
        if not opts:
            return None
        f, p = self.rng.choice(opts)
        return self.run_path(av[f], p, H, fuel, prefer)

    def antecedent(self, H: tuple[int, ...], prefer: tuple, av: dict[F, Node]) -> F:
        """A small formula for the left of a new premise X > ...: an open hypothesis, a
        conjunction or disjunction with one, a negation that needs its own negation box, or a
        literal that needs its own derivation."""
        hyps = [self.hypf[h] for h in prefer if h in H] or [self.hypf[h] for h in H]
        small_av = [f for f in av if f != BOT and conn(f) <= 1]
        opts: list[tuple[float, Callable[[], F]]] = []
        if hyps:
            opts.append((3.0, lambda: self.rng.choice(hyps)))
            opts.append((0.25, lambda: Or(self.rng.choice(hyps), self.lit())))
            opts.append((0.15, lambda: Or(self.lit(), self.rng.choice(hyps))))
            if len(small_av) >= 2:
                opts.append((1.0, lambda: And(self.rng.choice(hyps), self.rng.choice(small_av))))
                opts.append((0.8, lambda: And(self.rng.choice(small_av), self.rng.choice(hyps))))
        opts.append((1.2, lambda: Not(self.atom())))
        opts.append((0.5, lambda: Not(self.small(1, 0.0))))
        opts.append((0.6, lambda: self.lit()))
        f = self.pick(opts)()
        if degenerate(f):
            raise Fail("degenerate antecedent")
        return f

    def m_path_newprem(
        self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]
    ) -> Node:
        """A new premise with an elimination path to G: X > G, (X & Y) > G, X > (G & W),
        X > (Y > G), ..."""
        X = self.antecedent(H, prefer, av)
        if X == G:
            raise Fail("trivial antecedent")
        shape = self.pick([(3.0, "imp"), (0.8, "imp_and"), (0.7, "curried"), (0.5, "and_cons")])
        if shape == "imp":
            prem = Imp(X, G)
        elif shape == "imp_and":
            Y = self.antecedent(H, prefer, av)
            prem = Imp(And(X, Y), G)
        elif shape == "curried":
            Y = self.antecedent(H, prefer, av)
            prem = Imp(X, Imp(Y, G))
        else:
            W = self.lit()
            prem = Imp(X, And(G, W) if self.rng.random() < 0.5 else And(W, G))
        p = self.add_prem(prem)
        path = self.paths(prem, G)
        if not path:
            raise Fail("no path")
        return self.run_path(p, path[0], H, fuel, prefer)

    def m_ore(self, G: F, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]) -> Node:
        disj = [f for f in av if f[0] == "or" and f[1] != G and f[2] != G]
        if disj and self.rng.random() < 0.7:
            D = self.rng.choice(disj)
            dn = av[D]
        else:
            a = self.pick([(2.0, lambda: self.lit(0.4)), (0.5, lambda: self.small(1, 0.0))])()
            b = self.pick([(2.0, lambda: self.lit(0.4)), (0.5, lambda: self.small(1, 0.0))])()
            if a == b or a == Not(b) or b == Not(a):
                raise Fail("degenerate disjunction")
            D = Or(a, b)
            dn = self.add_prem(D)
        h1, h2 = self.hyp(D[1]), self.hyp(D[2])
        b1 = self.prove(G, (*H, h1), fuel - 3, (h1, *prefer))
        b2 = self.prove(G, (*H, h2), fuel - 3, (h2, *prefer))
        for h, b in ((h1, b1), (h2, b2)):
            if h not in deps(b):
                raise Fail("case branch ignores its case")
        return Node("ORE", G, (dn, b1, b2), (h1, h2))

    # contradiction moves (G = F)
    def m_bot_avail_neg(
        self, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]
    ) -> Node | None:
        negs = [f for f in av if f[0] == "not" and f[1] not in av]
        if not negs:
            return None
        pref = {self.hypf[h] for h in prefer}
        f = self.rng.choices(negs, [3.0 if f in pref else 1.0 for f in negs])[0]
        x = self.prove(f[1], H, fuel - 1, prefer)
        return Node("NEGE", BOT, (x, av[f]))

    def m_bot_avail_pos(
        self, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]
    ) -> Node | None:
        """An available formula X (preferably a hypothesis) clashes with ~X, which is reached by
        an elimination path from a new premise W > ~X (W built from the context)."""
        pref = [self.hypf[h] for h in prefer if h in H]
        pos = [f for f in av if f != BOT and Not(f) not in av and f[0] != "not"]
        if not pos:
            return None
        X = self.rng.choice([f for f in pos if f in pref] or pos)
        W = self.antecedent(H, prefer, av)
        if W in (X, Not(X)):
            raise Fail("degenerate")
        p = self.add_prem(Imp(W, Not(X)))
        w = self.prove(W, H, fuel - 1, prefer)
        return Node("NEGE", BOT, (av[X], Node("IMPE", Not(X), (p, w))))

    def m_bot_new_negprem(
        self, H: tuple[int, ...], fuel: int, prefer: tuple, av: dict[F, Node]
    ) -> Node:
        """A new premise ~X, where X is built by introduction rules from the hypotheses."""
        hyps = [self.hypf[h] for h in prefer if h in H] or [self.hypf[h] for h in H]
        if not hyps:
            raise Fail("no hypothesis")
        others = [f for f in av if f != BOT and conn(f) <= 1]
        a = self.rng.choice(hyps)
        k = self.pick([(3.0, "and"), (1.0, "or"), (1.0, "neg")])
        if k == "and":
            b = self.rng.choice([x for x in others if x != a] or [self.lit()])
            X = And(a, b) if self.rng.random() < 0.5 else And(b, a)
        elif k == "or":
            b = self.lit()
            X = Or(a, b) if self.rng.random() < 0.5 else Or(b, a)
        else:
            X = Not(self.atom())
        if degenerate(X):
            raise Fail("degenerate")
        p = self.add_prem(Not(X))
        x = self.prove(X, H, fuel - 1, prefer)
        return Node("NEGE", BOT, (x, p))


# ---------------------------------------------------------------------- recipes
HYP_ATOMIC = {"atom": 0.75, "lit": 0.25}
HYP_ANY = {
    "atom": 0.35,
    "lit": 0.15,
    "conj": 0.14,
    "disj": 0.12,
    "imp": 0.12,
    "negc": 0.06,
    "small2": 0.06,
}


def n_elims(n: Node) -> int:
    return sum(x.rule in ELIM_RULES for x in walk(n))


def _check_inner(bot: Node, must: tuple[int, ...], min_el: int) -> None:
    d = deps(bot)
    if any(h not in d for h in must):
        raise Fail("inner box ignores a hypothesis")
    if n_elims(bot) < min_el:
        raise Fail("too little work")


def _pos(b: Builder, kinds: dict[str, float]) -> F:
    """A hypothesis that is not a negation (so ~B never becomes ~~)."""
    f = b.hypform(kinds)
    return f[1] if f[0] == "not" else f


def r_in(b: Builder) -> Node:
    """I(N()): goal A > ~B; the inner contradiction uses both hypotheses (>= 2 eliminations)."""
    A, B = b.hypform(HYP_ANY), _pos(b, {"atom": 0.8, "conj": 0.1, "disj": 0.1})
    if A in (B, Not(B)):
        raise Fail("degenerate")
    goal = Imp(A, Not(B))
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    bot = b.prove(BOT, (ha, hb), 8, (hb, ha))
    _check_inner(bot, (ha, hb), 2)
    return Node("IMPI", goal, (Node("NEGI", Not(B), (bot,), (hb,)),), (ha,))


def r_ino(b: Builder) -> Node:
    """I(N(O)): goal A > ~B; the inner contradiction is a case split using A and B."""
    A, B = b.hypform(HYP_ANY), _pos(b, HYP_ATOMIC)
    if A in (B, Not(B)):
        raise Fail("degenerate")
    goal = Imp(A, Not(B))
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    bot = b.m_ore(BOT, (ha, hb), 9, (hb, ha), b.closure((ha, hb)))
    _check_inner(bot, (ha, hb), 2)
    return Node("IMPI", goal, (Node("NEGI", Not(B), (bot,), (hb,)),), (ha,))


def _use_neg_lemma(b: Builder, nl: Node, Z: F) -> Node:
    """Feed the lemma nl (~B, proved by a negation box) to IMPE through a new premise."""
    if Z in (nl.f, Not(nl.f)):
        raise Fail("degenerate")
    shape = b.pick([(3.0, "imp"), (0.6, "imp_and")])
    if shape == "imp":
        p = b.add_prem(Imp(nl.f, Z))
        return Node("IMPE", Z, (p, nl))
    W = b.lit()
    p = b.add_prem(Imp(nl.f, And(Z, W)))
    return Node("ANDE1", Z, (Node("IMPE", And(Z, W), (p, nl)),))


def r_inr(b: Builder) -> Node:
    """I(N()) with the inner negation reused: goal A > Z; in the box, ~B is proved by a negation
    box (using A and B) and then fed to IMPE."""
    A, B = b.hypform(HYP_ANY), _pos(b, HYP_ATOMIC)
    Z = b.pick([(2.0, b.lit(0.4)), (0.5, b.small(1, 0.0))])
    if len({A, B, Z}) < 3 or Not(A) == B:
        raise Fail("degenerate")
    goal = Imp(A, Z)
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    bot = b.prove(BOT, (ha, hb), 7, (hb, ha))
    _check_inner(bot, (ha, hb), 2)
    nl = Node("NEGI", Not(B), (bot,), (hb,))
    return Node("IMPI", goal, (_use_neg_lemma(b, nl, Z),), (ha,))


def r_nn(b: Builder) -> Node:
    """N(N()): goal ~A; inside, ~B comes from a negation box using A and B, and the outer
    contradiction needs ~B (through ~B > ~A, ~B > Y, or a formula built from ~B that clashes with
    a negated premise)."""
    A, B = _pos(b, HYP_ANY), _pos(b, HYP_ATOMIC)
    if A == B:
        raise Fail("degenerate")
    goal = Not(A)
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    bot = b.prove(BOT, (ha, hb), 7, (hb, ha))
    _check_inner(bot, (ha, hb), 2)
    nl = Node("NEGI", Not(B), (bot,), (hb,))
    k = b.pick([(3.0, "imp_neg_goal"), (2.0, "imp_clash"), (1.0, "ori"), (1.0, "andi")])
    if k == "imp_neg_goal":
        p = b.add_prem(Imp(Not(B), Not(A)))
        outer = Node("NEGE", BOT, (b.hnode(ha), Node("IMPE", Not(A), (p, nl))))
    elif k == "imp_clash":
        av = b.closure((ha,))
        cands = [f for f in av if f != BOT and conn(f) <= 2 and f != Not(B)]
        c = b.rng.choice(cands)
        Y = c[1] if c[0] == "not" else Not(c)
        p = b.add_prem(Imp(Not(B), Y))
        y = Node("IMPE", Y, (p, nl))
        outer = Node("NEGE", BOT, (y, av[c]) if c[0] == "not" else (av[c], y))
    elif k == "ori":
        W = b.lit()
        X = Or(Not(B), W) if b.rng.random() < 0.5 else Or(W, Not(B))
        p = b.add_prem(Not(X))
        outer = Node("NEGE", BOT, (Node("ORI1" if X[1] == Not(B) else "ORI2", X, (nl,)), p))
    else:
        X = And(A, Not(B)) if b.rng.random() < 0.5 else And(Not(B), A)
        p = b.add_prem(Not(X))
        x = Node("ANDI", X, (b.hnode(ha), nl) if X[1] == A else (nl, b.hnode(ha)))
        outer = Node("NEGE", BOT, (x, p))
    return Node("NEGI", goal, (outer,), (ha,))


def _goal_c(b: Builder) -> F:
    """A reductio goal: an atom or a disjunction (sometimes with a compound side)."""
    a, c = b.distinct_lits(0.25)
    g = b.pick([(0.5, b.atom()), (0.35, Or(a, c)), (0.15, b.small(2, 0.0))])
    if g[0] == "not":
        g = g[1]
    if degenerate(g):
        raise Fail("degenerate")
    return g


def _raa(goal: F, body: Node, h: int) -> Node:
    return Node("DN", goal, (Node("NEGI", Not(Not(goal)), (body,), (h,)),))


def r_cn(b: Builder) -> Node:
    """C(N()): goal G by reductio; inside, a negation box on B uses the reductio hypothesis ~G,
    and its result ~B is then needed to reach the contradiction."""
    G = _goal_c(b)
    B = _pos(b, {"atom": 0.7, "conj": 0.1, "disj": 0.1, "imp": 0.1})
    if B in (G, Not(G)):
        raise Fail("degenerate")
    b.reserved = conn(G)
    hg, hb = b.hyp(Not(G)), b.hyp(B)
    bot = b.prove(BOT, (hg, hb), 6, (hb, hg))
    _check_inner(bot, (hg, hb), 2)
    nl = Node("NEGI", Not(B), (bot,), (hb,))
    if G[0] == "or" and b.rng.random() < 0.4:
        side = b.rng.choice([1, 2])
        g = Node("ORI1" if side == 1 else "ORI2", G, (_use_neg_lemma(b, nl, G[side]),))
    else:
        g = _use_neg_lemma(b, nl, G)
    return _raa(G, Node("NEGE", BOT, (g, b.hnode(hg))), hg)


def r_cw(b: Builder) -> Node:
    """Reductio whose body is an elimination chain (>= 3 eliminations) from the assumed ~G."""
    G = _goal_c(b)
    b.reserved = conn(G)
    hg = b.hyp(Not(G))
    bot = b.prove(BOT, (hg,), 7, (hg,))
    _check_inner(bot, (hg,), 3)
    return _raa(G, bot, hg)


def r_co(b: Builder) -> Node:
    """Reductio with a case split inside: ~G is assumed, a disjunction is split, and each branch
    reaches F using ~G."""
    G = _goal_c(b)
    b.reserved = conn(G)
    hg = b.hyp(Not(G))
    bot = b.m_ore(BOT, (hg,), 9, (hg,), b.closure((hg,)))
    if hg not in deps(bot):
        raise Fail("split ignores ~G")
    return _raa(G, bot, hg)


def r_io(b: Builder) -> Node:
    """I(O(|)): goal A > G with a case split inside the box: on the assumption itself when A is a
    disjunction, otherwise on a premise disjunction; both branches use their case."""
    A = Or(*b.distinct_lits(0.3)) if b.rng.random() < 0.5 else b.hypform(HYP_ANY)
    G = b.pick(
        [
            (0.35, b.atom()),
            (0.15, b.lit(1.0)),
            (0.3, Or(*b.distinct_lits(0.3))),
            (0.1, And(*b.distinct_lits(0.3))),
            (0.1, b.small(2, 0.0)),
        ]
    )
    if G == A or (A[0] == "or" and G in (A[1], A[2])) or degenerate(G):
        raise Fail("degenerate")
    goal = Imp(A, G)
    b.reserved = conn(goal)
    ha = b.hyp(A)
    body = b.m_ore(G, (ha,), 9, (ha,), b.closure((ha,)))
    if ha not in deps(body):
        raise Fail("vacuous")
    return Node("IMPI", goal, (body,), (ha,))


def r_iox(b: Builder) -> Node:
    """I(O(|)) with crossed routing: goal A > (X v Y); inside the box a disjunction is split and
    one branch reaches X (Or.inl), the other Y (Or.inr)."""
    A = b.hypform(HYP_ANY)
    X, Y = b.distinct_lits(0.3)
    G = Or(X, Y)
    if A in (X, Y, G):
        raise Fail("degenerate")
    goal = Imp(A, G)
    b.reserved = conn(goal)
    ha = b.hyp(A)
    av = b.closure((ha,))
    disj = [f for f in av if f[0] == "or" and f != G]
    if disj and b.rng.random() < 0.6:
        D = b.rng.choice(disj)
        dn = av[D]
    else:
        D = Or(*b.distinct_lits(0.4))
        if D == G:
            raise Fail("degenerate")
        dn = b.add_prem(D)
    h1, h2 = b.hyp(D[1]), b.hyp(D[2])
    first = b.rng.random() < 0.5
    x = b.prove(X if first else Y, (ha, h1), 5, (h1, ha))
    y = b.prove(Y if first else X, (ha, h2), 5, (h2, ha))
    for h, n in ((h1, x), (h2, y)):
        if h not in deps(n):
            raise Fail("case branch ignores its case")
    r1 = Node("ORI1" if first else "ORI2", G, (x,))
    r2 = Node("ORI2" if first else "ORI1", G, (y,))
    body = Node("ORE", G, (dn, r1, r2), (h1, h2))
    if ha not in deps(body):
        raise Fail("vacuous")
    return Node("IMPI", goal, (body,), (ha,))


def r_iio(b: Builder) -> Node:
    """I(I(O(|))): goal A > (B > G) with a case split at depth 2 using both hypotheses."""
    A, B = b.hypform(HYP_ANY), b.hypform(HYP_ANY)
    G = b.pick([(0.5, b.atom()), (0.2, b.lit(1.0)), (0.3, Or(*b.distinct_lits(0.3)))])
    if len({A, B, G}) < 3:
        raise Fail("degenerate")
    goal = Imp(A, Imp(B, G))
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    body = b.m_ore(G, (ha, hb), 8, (hb, ha), b.closure((ha, hb)))
    _check_inner(body, (ha, hb), 2)
    return Node("IMPI", goal, (Node("IMPI", goal[2], (body,), (hb,)),), (ha,))


def r_no(b: Builder) -> Node:
    """N(O(|)): goal ~A with a case split inside the negation box."""
    A = Or(*b.distinct_lits(0.3)) if b.rng.random() < 0.55 else _pos(b, HYP_ANY)
    goal = Not(A)
    b.reserved = conn(goal)
    ha = b.hyp(A)
    body = b.m_ore(BOT, (ha,), 9, (ha,), b.closure((ha,)))
    if ha not in deps(body):
        raise Fail("vacuous")
    return Node("NEGI", goal, (body,), (ha,))


def r_ii(b: Builder) -> Node:
    """I(I()): goal A > (B > G); G comes from elimination work at depth 2 that uses both
    hypotheses."""
    A, B = b.hypform(HYP_ANY), b.hypform(HYP_ANY)
    G = b.pick(
        [
            (0.5, b.atom()),
            (0.2, b.lit(1.0)),
            (0.15, And(*b.distinct_lits(0.3))),
            (0.15, b.small(2, 0.0)),
        ]
    )
    if len({A, B, G}) < 3 or degenerate(G):
        raise Fail("degenerate")
    goal = Imp(A, Imp(B, G))
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    body = b.prove(G, (ha, hb), 7, (hb, ha))
    _check_inner(body, (ha, hb), 2)
    return Node("IMPI", goal, (Node("IMPI", goal[2], (body,), (hb,)),), (ha,))


def r_ho(b: Builder) -> Node:
    """Goal (A > B) > C: the assumed implication is applied (IMPE on a hypothesis) to an
    antecedent that needs its own work."""
    A = b.pick([(0.4, b.atom()), (0.3, Not(b.atom())), (0.3, b.hypform(HYP_ANY))])
    Bc, C = b.lit(0.3), b.pick([(0.7, b.lit(0.3)), (0.3, b.small(1, 0.0))])
    if len({A, Bc, C}) < 3:
        raise Fail("degenerate")
    goal = Imp(Imp(A, Bc), C)
    b.reserved = conn(goal)
    h = b.hyp(Imp(A, Bc))
    body = b.prove(C, (h,), 9, (h,))
    if h not in deps(body):
        raise Fail("vacuous")
    if not any(
        x.rule == "IMPE" and x.kids[0].rule == "HYP" and x.kids[0].ref == h for x in walk(body)
    ):
        raise Fail("assumed implication not applied")
    return Node("IMPI", goal, (body,), (h,))


def _goal_shape(b: Builder, shape: str) -> F:
    s = b.small
    if shape == "imp_neg":
        return Imp(s(2, 0.5), Not(_pos(b, HYP_ANY)))
    if shape == "imp_imp":
        return Imp(s(1, 0.4), Imp(s(1, 0.4), s(1, 0.5)))
    if shape == "neg":
        return Not(b.small(3, 0.0))
    if shape == "or":
        return Or(s(2, 0.3), s(2, 0.3))
    if shape == "ho":
        return Imp(Imp(s(1, 0.4), s(1, 0.5)), s(2, 0.4))
    if shape == "imp":
        return Imp(s(2, 0.2), s(2, 0.3))
    if shape == "bicond":
        x, y = s(1, 0.6), s(1, 0.6)
        return And(Imp(x, y), Imp(y, x))
    raise ValueError(shape)


def _r_goal(shape: str) -> Callable[[Builder], Node]:
    def run(b: Builder) -> Node:
        G = _goal_shape(b, shape)
        if degenerate(G) or G[0] == "atom":
            raise Fail("degenerate goal")
        b.reserved = conn(G)
        if b.reserved > MAX_CONN:
            raise Fail("goal too big")
        return b.prove(G, (), 11)

    run.__doc__ = f"Goal of shape {shape!r}; the builder chooses the boxes."
    return run


def r_g_split_ante(b: Builder) -> Node:
    """Goal (X v Y) > Z with random small X, Y, Z: the assumed disjunction is split inside the
    box, and both branches use their case."""
    X, Y, Z = b.small(1, 0.5), b.small(1, 0.5), b.small(2, 0.3)
    if len({X, Y, Z}) < 3 or Z in (X, Y) or degenerate(Or(X, Y)):
        raise Fail("degenerate")
    A = Or(X, Y)
    goal = Imp(A, Z)
    b.reserved = conn(goal)
    if b.reserved > MAX_CONN:
        raise Fail("goal too big")
    ha = b.hyp(A)
    av = b.closure((ha,))
    body = b.m_ore(Z, (ha,), 10, (ha,), {A: av[A]})
    return Node("IMPI", goal, (body,), (ha,))


def r_g_imp_split(b: Builder) -> Node:
    """Goal A > Z with random A, Z; the box's first move is a case split (on an available
    disjunction or a premise one)."""
    A, Z = b.small(2, 0.3), b.small(2, 0.3)
    if A == Z or Z[0] == "imp" or degenerate(Imp(A, Z)):
        raise Fail("degenerate")
    goal = Imp(A, Z)
    b.reserved = conn(goal)
    if b.reserved > MAX_CONN:
        raise Fail("goal too big")
    ha = b.hyp(A)
    body = b.m_ore(Z, (ha,), 10, (ha,), b.closure((ha,)))
    if ha not in deps(body):
        raise Fail("vacuous")
    return Node("IMPI", goal, (body,), (ha,))


def r_g_neg_split(b: Builder) -> Node:
    """Goal ~A with random compound A; the contradiction in the box is a case split."""
    A = b.small(3, 0.0)
    if A[0] == "not" or degenerate(A):
        raise Fail("degenerate")
    goal = Not(A)
    b.reserved = conn(goal)
    if b.reserved > MAX_CONN:
        raise Fail("goal too big")
    ha = b.hyp(A)
    body = b.m_ore(BOT, (ha,), 10, (ha,), b.closure((ha,)))
    if ha not in deps(body):
        raise Fail("vacuous")
    return Node("NEGI", goal, (body,), (ha,))


def r_g_inr(b: Builder) -> Node:
    """INR with random parts: goal A > Z, inside ~B by a negation box (using A and B), then ~B
    is fed to an elimination."""
    A, B = b.small(2, 0.35), _pos(b, {"atom": 0.6, "small2": 0.4})
    Z = b.small(1, 0.4)
    if len({A, B, Z}) < 3 or Not(A) == B or degenerate(Imp(A, Z)):
        raise Fail("degenerate")
    goal = Imp(A, Z)
    b.reserved = conn(goal)
    ha, hb = b.hyp(A), b.hyp(B)
    bot = b.prove(BOT, (ha, hb), 7, (hb, ha))
    _check_inner(bot, (ha, hb), 2)
    nl = Node("NEGI", Not(B), (bot,), (hb,))
    av = b.closure((ha,))
    av[Not(B)] = nl
    k = b.pick([(2.0, "lemma_prem"), (1.0, "path")])
    if k == "lemma_prem":
        z = _use_neg_lemma(b, nl, Z)
    else:
        z = b.m_path_newprem(Z, (ha,), 4, (ha,), av)
        if not any(x is nl for x in walk(z)):
            raise Fail("lemma unused")
    return Node("IMPI", goal, (z,), (ha,))


def _with_bicond(rec: Callable[[Builder], Node]) -> Callable[[Builder], Node]:
    def run(b: Builder) -> Node:
        x, y = b.distinct_lits(0.3)
        bic = And(Imp(x, y), Imp(y, x))
        b.add_prem(bic)
        root = rec(b)
        used = {n.rule for n in walk(root) if n.kids and n.kids[0].f == bic}
        if not {"ANDE1", "ANDE2"} <= used:
            raise Fail("a direction of the biconditional is unused")
        return root

    run.__doc__ = f"{rec.__name__} with a biconditional premise (X > Y) & (Y > X)."
    return run


RECIPES: dict[str, Callable[[Builder], Node]] = {
    "IN": r_in,
    "INO": r_ino,
    "INR": r_inr,
    "NN": r_nn,
    "CN": r_cn,
    "CW": r_cw,
    "CO": r_co,
    "IO": r_io,
    "IOX": r_iox,
    "IIO": r_iio,
    "NO": r_no,
    "II": r_ii,
    "HO": r_ho,
    "G_imp_neg": _r_goal("imp_neg"),
    "G_imp_imp": _r_goal("imp_imp"),
    "G_neg": _r_goal("neg"),
    "G_or": _r_goal("or"),
    "G_ho": _r_goal("ho"),
    "G_imp": _r_goal("imp"),
    "G_bicond": _r_goal("bicond"),
    "B_IN": _with_bicond(r_in),
    "B_II": _with_bicond(r_ii),
    "B_NO": _with_bicond(r_no),
    "B_CW": _with_bicond(r_cw),
    "B_NN": _with_bicond(r_nn),
    "B_IO": _with_bicond(r_io),
    "B_IOX": _with_bicond(r_iox),
    "G_split_ante": r_g_split_ante,
    "G_imp_split": r_g_imp_split,
    "G_neg_split": r_g_neg_split,
    "G_inr": r_g_inr,
    "B_G_split_ante": _with_bicond(r_g_split_ante),
    "B_G_imp_split": _with_bicond(r_g_imp_split),
    "B_G_inr": _with_bicond(r_g_inr),
}

#: Default sampling weights over recipes, set from the pilot's per-recipe yields and feature mix
#: (README, "Mix control"): case-split recipes up, reductio-heavy goal shapes down.
WEIGHTS: dict[str, float] = {
    "IN": 1.0,
    "INO": 1.5,
    "INR": 2.0,
    "NN": 2.0,
    "CN": 0.8,
    "CW": 0.3,
    "CO": 0.4,
    "IO": 2.5,
    "IOX": 2.0,
    "IIO": 1.0,
    "NO": 2.0,
    "II": 1.0,
    "HO": 0.6,
    "G_imp_neg": 1.2,
    "G_imp_imp": 1.2,
    "G_neg": 3.0,
    "G_or": 0.3,
    "G_ho": 0.5,
    "G_imp": 1.5,
    "G_bicond": 0.2,
    "G_split_ante": 4.0,
    "G_imp_split": 4.0,
    "G_neg_split": 3.0,
    "G_inr": 4.0,
    "B_IN": 1.0,
    "B_II": 1.0,
    "B_NO": 1.0,
    "B_IO": 1.2,
    "B_IOX": 1.0,
    "B_CW": 0.2,
    "B_NN": 1.0,
    "B_G_split_ante": 1.5,
    "B_G_imp_split": 1.5,
    "B_G_inr": 1.5,
}


# ---------------------------------------------------------------------- substitution
def subst_f(f: F, m: dict[str, F]) -> F:
    if f[0] == "atom":
        return m.get(f[1], f)
    if f[0] == "bot":
        return f
    if f[0] == "not":
        return Not(subst_f(f[1], m))
    return (f[0], subst_f(f[1], m), subst_f(f[2], m))


def subst_tree(root: Node, m: dict[str, F]) -> Node:
    memo: dict[int, Node] = {}

    def go(n: Node) -> Node:
        k = id(n)
        if k not in memo:
            memo[k] = Node(n.rule, subst_f(n.f, m), tuple(go(c) for c in n.kids), n.hyps, n.ref)
        return memo[k]

    return go(root)


def substitution(b: Builder, prem: list[F], goal: F) -> dict[str, F] | None:
    """A random atom -> small formula map that keeps the statement within the shape limits
    (<= MAX_CONN connectives, no ~~, no X o X, at most 3 atoms), or None."""
    stmt = [*prem, goal]
    used = sorted(set().union(*(atoms(f) for f in stmt)))
    a = b.rng.choice(used)
    pool = list(dict.fromkeys(used + b.atoms))
    for _ in range(6):
        x, y = ("atom", b.rng.choice(pool)), ("atom", b.rng.choice(pool))
        s = b.pick(
            [
                (2.0, Not(x)),
                (1.0, And(x, y)),
                (1.0, Or(x, y)),
                (1.0, Imp(x, y)),
                (0.5, Imp(Not(x), y)),
                (0.4, And(Not(x), y)),
                (0.4, Or(x, Not(y))),
            ]
        )
        new = [subst_f(f, {a: s}) for f in stmt]
        if sum(conn(f) for f in new) > MAX_CONN or max(fdepth(f) for f in new) > MAX_FDEPTH:
            continue
        if any(degenerate(f) for f in new) or len(set().union(*(atoms(f) for f in new))) > 3:
            continue
        return {a: s}
    return None


# ---------------------------------------------------------------------- discharge
P_DISCHARGE = 0.8


def discharge(b: Builder, root: Node, prem: list[F], hypf: dict[int, F]) -> tuple[Node, list[F]]:
    """Deduction theorem on the tree: premises become antecedents of the goal. One premise P:
    |- P > G (or Q |- P > G). Two premises: |- P1 > (P2 > G) or |- (P1 & P2) > G (each premise
    leaf becomes ANDE of the new hypothesis). Raises Fail when the statement would exceed the
    connective limit."""
    n = len(prem)
    opts = [[i] for i in range(n)] + ([list(range(n))] * 2 if n == 2 else [])
    which = b.rng.choice(opts)
    rest = [i for i in range(n) if i not in which]
    G = root.f
    conj = len(which) == 2 and b.rng.random() < 0.4
    if conj:
        P = And(prem[which[0]], prem[which[1]])
        goal = Imp(P, G)
    else:
        goal = G
        for i in reversed(which):
            goal = Imp(prem[i], goal)
    if (
        sum(conn(prem[i]) for i in rest) + conn(goal) > MAX_CONN
        or fdepth(goal) > MAX_FDEPTH
        or degenerate(goal)
    ):
        raise Fail("discharge budget")
    hs = {}
    if conj:
        h = b.hyp(P)
        hypf[h] = P
        hn = Node("HYP", P, ref=h)
        leaf = {
            which[0]: Node("ANDE1", prem[which[0]], (hn,)),
            which[1]: Node("ANDE2", prem[which[1]], (hn,)),
        }
    else:
        leaf = {}
        for i in which:
            hs[i] = b.hyp(prem[i])
            hypf[hs[i]] = prem[i]
            leaf[i] = Node("HYP", prem[i], ref=hs[i])
    remap = {old: new for new, old in enumerate(rest)}
    memo: dict[int, Node] = {}

    def go(x: Node) -> Node:
        k = id(x)
        if k not in memo:
            if x.rule == "PR":
                memo[k] = leaf[x.ref] if x.ref in leaf else Node("PR", x.f, ref=remap[x.ref])
            else:
                memo[k] = Node(x.rule, x.f, tuple(go(c) for c in x.kids), x.hyps, x.ref)
        return memo[k]

    body = go(root)
    if conj:
        out = Node("IMPI", goal, (body,), (h,))
    else:
        out = body
        f = G
        for i in reversed(which):
            f = Imp(prem[i], f)
            out = Node("IMPI", f, (out,), (hs[i],))
    return out, [prem[i] for i in rest]


# ---------------------------------------------------------------------- sampling
def fstr(f: F) -> str:
    if f[0] == "atom":
        return f[1]
    if f[0] == "bot":
        return "F"
    if f[0] == "not":
        return f"( ~ {fstr(f[1])} )"
    op = {"and": "&", "or": "v", "imp": ">"}[f[0]]
    return f"( {fstr(f[1])} {op} {fstr(f[2])} )"


def prompt_of(prem: list[F], goal: F) -> str:
    ps = " , ".join(fstr(p) for p in prem)
    return f"THM {ps} SEQ {fstr(goal)} PRF" if prem else f"THM SEQ {fstr(goal)} PRF"


def prune_premises(root: Node, prem: list[F]) -> list[F]:
    """Drop premises the tree never cites (e.g. a biconditional premise the recipe did not use)
    and renumber the premise leaves in place."""
    used = sorted({x.ref for x in walk(root) if x.rule == "PR"})
    remap = {old: new for new, old in enumerate(used)}
    seen: set[int] = set()
    for x in walk(root):
        if x.rule == "PR" and id(x) not in seen:
            seen.add(id(x))
            x.ref = remap[x.ref]
    return [prem[i] for i in used]


def sample(
    seed: int, recipe: str | None = None, weights: dict[str, float] | None = None
) -> dict[str, Any] | None:
    """One candidate from `seed` (deterministic): dict(prompt, proof, recipe, seed, n_lines,
    subst) or None when the construction fails or does not linearise."""
    rng = random.Random(seed)
    w = weights or WEIGHTS
    name = recipe or rng.choices(list(w), list(w.values()))[0]
    b = Builder(rng, rng.choices([2, 3], [0.4, 0.6])[0])
    b.allow_bote = rng.random() < 0.1
    if not name.startswith("B_") and rng.random() < P_NO_PREMISE:
        b.max_prem = 0  # everything must come from the goal's own hypotheses
    try:
        root = RECIPES[name](b)
    except (Fail, IndexError, RecursionError):
        return None
    prem = prune_premises(root, list(b.prem))
    hypf = dict(b.hypf)
    sub = None
    subs_done: dict[str, F] = {}
    for p_sub in (P_SUBST, P_SUBST2):
        if rng.random() >= p_sub:
            break
        sub = substitution(b, prem, root.f)
        if sub is None:
            break
        root = subst_tree(root, sub)
        prem = [subst_f(p, sub) for p in prem]
        hypf = {h: subst_f(f, sub) for h, f in hypf.items()}
        subs_done.update(sub)
    sub = subs_done or None
    dis = 0
    if prem and rng.random() < P_DISCHARGE:
        try:
            n0 = len(prem)
            root, prem = discharge(b, root, prem, hypf)
            dis = n0 - len(prem)
        except Fail:
            pass
    p = linearize(root, prem, hypf)
    if p is None:
        return None
    return {
        "prompt": prompt_of(prem, root.f),
        "proof": p.body(),
        "recipe": name,
        "seed": seed,
        "n_lines": len(p.lines),
        "subst": None if sub is None else {k: fstr(v) for k, v in sub.items()},
        "discharged": dis,
    }
