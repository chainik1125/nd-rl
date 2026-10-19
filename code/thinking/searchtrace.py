"""Search traces: thinking traces that record an honest proof search (Stream-of-Search style).

    python -m code.thinking.searchtrace --source K12 --level 256 --out e256.jsonl --workers 16

One trace per theorem, in the progtrace format ({id, source, mode, prompt, steps: [{think, action,
min_level}], gen}, ASCII surface), checked by `trace_dsl.validate(rec, lean=False, strict=True)`.
Unlike `progtrace`, **the actions are the search's own**: the proof is built step by step from the
theorem alone.  The search never sees a reference proof (`prove` takes the prompt, `Search` the
environment state); a reference, when the source has one, is read afterwards for statistics.

**One step.**  At a decision state (focused goal G, hypotheses in scope) the search is a depth-first
backward search over reductions of G, the DSL's goal lines `ctx ⊢ G := SKEL`:
  * candidates (`gen`): the assumption (G in scope), introduction by G's connective (pair, Or.inl,
    Or.inr, a box `fun ( x : A ) => ...`), elimination spines of a hypothesis (up to three
    applications / projections ending in G, in False then `.elim`, or in a disjunction to split on),
    case splits (`Or.elim`), the classical detour `Classical.byContradiction ⊢ ¬¬G` and ex falso
    `⊢ False .elim`.  A subgoal that is already a hypothesis is closed by its name.  Progress
    guards (not claims): no subgoal already on the current path, no split on a disjunction whose
    side is in scope, no classical detour when ¬G is in scope, no ex falso when False is.
  * order: the proof's STRATEGY (below); it never depends on a reference.
  * each candidate is checked by truth tables: a hole with a classical countermodel is dead
    (complete for this fragment), and the line says so: `⊢ G := SKEL <no> ρ` (short form) or
    `⊢ G := SKEL` / `⊢ H <no> ρ` (explicit form), where ρ is the countermodel, a minimal partial
    assignment (`P ( ¬ Q )`) that makes every hypothesis true and the hole false under Kleene's
    three-valued evaluation.  `trace_dsl` checks it (the DSL extension of this module).
  * a candidate that closes the goal outright (no open subgoal) is examined first ("closers";
    `--closers-high-only`: only at levels >= 128); then the first candidate whose holes are all
    valid is followed: its first open subgoal becomes the next goal line, until a reduction
    yields one environment action (an atomic `have`, a box opener, an `Or.elim` opener, `exact`).
    That action is committed.  A subgoal whose candidates all fail sends the search back to its
    parent's next candidate (shown by restating the parent goal).
  * at higher effort, bounded lookahead into the committed box: its goal under the binder is
    searched one or two levels (`x : A ⊢ B := ...`).
Premise restatements and `exact` of a last line that is the goal are forced: empty think.

**Strategies** (one per proof from sha256 of the id; another one if the proof fails or runs over
`--max-steps`, counted): `dfs` (introductions first, hypotheses oldest first), `hyp` (hypotheses
first, newest first), `best` (subgoals closable in one step or small first; the classical detour
and ex falso last), `rand` (a per-theorem
random order of the kinds and of candidates within a kind, stable across steps).  The strategy is
in the record's `gen`, never in the text.

**Effort = search budget.**  A level b fixes the record's token cap (240 at b = 256, else b), the
candidates examined per goal while recording, the lookahead depth, the dead-end form and whether
dead ends carry countermodels (`budget_for`).  The record is written as the search runs; once a
unit (one candidate's lines) does not fit, recording stops and the search finishes unrecorded (the
natural length is still counted).  Level 0 is an empty think and the action of the same search.
The written action is always the search's (valid, never a certified-dead move), and by default the
same at every level: one canonical proof per theorem (`--emit-plain` writes it as K12-style
records for the no-think control).

**Labels** (`min_level` per step, coordinator 2026-10-08): the commit-on-record action of level b
is what a search that trusts only its own record commits to: the first candidate its record shows
viable, else (from the deepest goal it reached) the strategy's top unverified guess, followed by
top guesses down to an action; at level 0 the top guess from the root.  It is labelled with the
oracle (`judge_action`: the proof can still be completed).  `min_level` is the smallest level of the
ladder (`--levels`, default 0,64,256) whose commit-on-record action is viable (null: none is).
`--level auto` (deferred) writes the proof that takes, at each step, the trace and
commit-on-record action of `min_level`.

Output: the JSONL, plus `<out stem>.stats.json`.  `--analysis-only` writes no traces (sealed sets).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
import random
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import progtrace as pt

FORMAT = "searchtrace/1"
OUT_DIR = "/scratch/cowifornia/nd-rl/think/traces/search"
STRATEGIES = ("dfs", "hyp", "best", "rand")
DEFAULT_LEVELS = (0, 64, 256)  # coordinator 2026-10-08: e16 dropped from the default ladder
# closers-first (a candidate that closes the goal outright is examined first) at every level: the
# written action then does not depend on the level, so every level writes the same (canonical)
# proof and a no-think control can train on it.  False: only levels >= 128 (proofs may differ).
SETTINGS: dict[str, bool] = {"closers_all": True}
UNCAPPED = 1 << 30  # the uncapped analysis level ("inf"): top-tier search, a safety cap
SAFETY_CAP = 4096
MAXN = 64
MAX_STEPS = 40
NODE_CAP = 3000  # candidate examinations per step search
DEPTH_CAP = 12
SPINE_MAX = 3

Formula = Any
Node = Any
Local = tuple[tuple[str, Formula], ...]
BOT: Formula = ("bot",)
KINDS = ("assume", "intro", "elim", "case", "classical", "efq", "split_late", "box_late")


def td() -> Any:
    return pt.td()


def ftoks(f: Formula) -> list[str]:
    return pt.ftoks(f)


def fsize(f: Formula) -> int:
    return 1 if f[0] in ("atom", "bot") else 1 + sum(fsize(x) for x in f[1:])


# ------------------------------------------------------------------------------- countermodels
# Kleene values over all 81 partial assignments of P Q R S at once: (true mask, false mask); bit a
# is the assignment with atom i unassigned / true / false for digit i of a in base 3 = 0 / 1 / 2.
NPA = 81
ALL81 = (1 << NPA) - 1
_DIG = [[(a // 3**i) % 3 for i in range(4)] for a in range(NPA)]
_KA = {
    at: (
        sum(1 << a for a in range(NPA) if _DIG[a][i] == 1),
        sum(1 << a for a in range(NPA) if _DIG[a][i] == 2),
    )
    for i, at in enumerate("PQRS")
}
# preference: fewest tokens (a negative literal is 4), then fewest literals, then index
_PREF = sorted(
    range(NPA),
    key=lambda a: (sum({0: 0, 1: 1, 2: 4}[d] for d in _DIG[a]), sum(d > 0 for d in _DIG[a]), a),
)
_KM: dict[Formula, tuple[int, int]] = {}


def kmask(f: Formula) -> tuple[int, int]:
    m = _KM.get(f)
    if m is not None:
        return m
    k = f[0]
    if k == "atom":
        m = _KA[f[1]]
    elif k == "bot":
        m = (0, ALL81)
    elif k == "not":
        t, fl = kmask(f[1])
        m = (fl, t)
    else:
        (t1, f1), (t2, f2) = kmask(f[1]), kmask(f[2])
        if k == "and":
            m = (t1 & t2, f1 | f2)
        elif k == "or":
            m = (t1 | t2, f1 & f2)
        else:
            m = (f1 | t2, t1 & f2)
    _KM[f] = m
    return m


_CM: dict[tuple[frozenset[Formula], Formula], dict[str, bool] | None] = {}


def countermodel(hyps: Iterable[Formula], g: Formula) -> dict[str, bool] | None:
    """The preferred partial assignment making every hypothesis true and g false (Kleene)."""
    hs = frozenset(hyps)
    key = (hs, g)
    if key in _CM:
        return _CM[key]
    m = ALL81 & kmask(g)[1]
    for h in hs:
        m &= kmask(h)[0]
    out = None
    if m:
        a = next(x for x in _PREF if m >> x & 1)
        out = {at: _DIG[a][i] == 1 for i, at in enumerate("PQRS") if _DIG[a][i]}
    _CM[key] = out
    return out


def fmt_countermodel(val: dict[str, bool]) -> list[str]:
    out: list[str] = []
    for at in "PQRS":
        if at in val:
            out += [at] if val[at] else ["(", "¬", at, ")"]
    return out


# ------------------------------------------------------------------------------ budgets, strategy
@dataclass(frozen=True)
class Budget:
    """The search budget of one effort level (`budget_for`)."""

    level: int  # effort level; 0 = no record; UNCAPPED = the analysis run
    cap: int  # think-body token cap
    scan: int  # candidates examined per goal while recording
    closers: bool  # first look for a candidate that closes the goal outright (no open holes)
    lookahead: int  # levels of lookahead into a committed box
    evidence: bool  # dead ends carry their countermodel
    short: bool  # dead ends in the one-line form whatever the strategy
    max_binders: int = 4

    @property
    def name(self) -> str:
        return "inf" if self.level == UNCAPPED else str(self.level)


def budget_for(level: int, cap: int | None = None) -> Budget:
    """Search parameters by level (tiers <=16, <=64, <=256, above).  The token cap follows the
    level (240 at 256, as progtrace; `cap` overrides it)."""
    lo = SETTINGS["closers_all"]  # closers-first below 128 too (default: one proof per theorem)
    if level == 0:
        return Budget(0, 0, 0, lo, 0, False, True)
    if level == UNCAPPED:
        return Budget(level, cap or SAFETY_CAP, 10**6, True, 2, True, False, 16)
    c = cap if cap is not None else (240 if level == 256 else level)
    if level <= 16:
        return Budget(level, c, 2, lo, 0, False, True)
    if level <= 64:
        return Budget(level, c, 4, lo, 0, True, True)
    if level <= 256:
        return Budget(level, c, 8, True, 1, True, False)
    return Budget(level, c, 10**6, True, 2, True, False, 8)


def parse_level(s: str) -> int | str:
    if s == "auto":
        return "auto"
    if s in ("inf", "uncapped"):
        return UNCAPPED
    return int(s)


def h64(*parts: Any) -> int:
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


@dataclass(frozen=True)
class Strategy:
    name: str
    seed: int  # per theorem
    perm: tuple[str, ...] = ("intro", "elim")  # rand: the order of these kinds
    explicit: bool = True  # dead ends in the two-line form (when the budget allows)


def make_strategy(name: str, pid: str, seed: int = 0) -> Strategy:
    s = h64("searchtrace", seed, pid, name)
    if name == "rand":
        perm = ["intro", "elim"]
        random.Random(s).shuffle(perm)
        return Strategy(name, s, tuple(perm), bool(s >> 7 & 1))
    return Strategy(name, s, explicit=name in ("dfs", "hyp"))


def strategy_order(pid: str, seed: int = 0) -> list[str]:
    """The sampled strategy first, then the others (the retry order), from the id alone."""
    first = STRATEGIES[h64("searchtrace/strategy", seed, pid) % len(STRATEGIES)]
    rest = sorted((s for s in STRATEGIES if s != first), key=lambda s: h64("retry", seed, pid, s))
    return [first, *rest]


# ------------------------------------------------------------------------------- the state view
class Ctx:
    """What the search sees of one state: names in scope, hypotheses, the focused goal."""

    def __init__(self, env: Any) -> None:
        T = td()
        self.env = env
        self.sc: dict[str, Formula] = T.scope(env)
        self.hyps = list(dict.fromkeys(self.sc.values()))
        self.names = sorted((k for k in self.sc if k.startswith("n")), key=lambda k: int(k[1:]))
        self.rank = {n: i for i, n in enumerate(self.names)}
        self.avail: dict[Formula, str] = {}
        for n in self.names:
            self.avail.setdefault(self.sc[n], n)
        fr = env.frames[-1]
        self.goal: Formula = fr.goal
        self.last: str | None = fr.last
        self.nh, self.nb = T.next_names(env)
        self.base = pt.FULL
        for h in self.hyps:
            self.base &= pt.tt(h)

    def valid(self, local: Iterable[Formula], f: Formula) -> bool:
        m = self.base
        for a in local:
            m &= pt.tt(a)
        return m & ~pt.tt(f) & pt.FULL == 0

    def consistent(self) -> bool:
        return self.base != 0

    def relevant(self, d: Formula, local: Iterable[Formula], g: Formula) -> bool:
        """d lies in some minimal support of g: a set S of the other hypotheses has S |/= g but
        S + d |= g (exact by subsets up to 12 other hypotheses, else: g fails without d)."""
        others = [h for h in dict.fromkeys([*self.hyps, *local]) if h != d]
        key = (frozenset(others), d, g)
        hit = _REL.get(key)
        if hit is not None:
            return hit
        gm, dm = pt.tt(g), pt.tt(d)
        if len(others) > 12:
            out = not self.valid_without(d, local, g)
        else:
            ms = [pt.tt(h) for h in others]
            out = False
            stack = [(0, pt.FULL)]
            while stack and not out:
                i, m = stack.pop()
                if m & ~gm & pt.FULL and not (m & dm & ~gm & pt.FULL):
                    out = True
                elif i < len(ms) and m & ~gm & pt.FULL:  # supersets of an S |= g never qualify
                    stack.append((i + 1, m))
                    stack.append((i + 1, m & ms[i]))
        _REL[key] = out
        return out

    def valid_without(self, drop: Formula, local: Iterable[Formula], f: Formula) -> bool:
        """f follows from the hypotheses (and `local`) with every copy of `drop` removed."""
        m = pt.FULL
        for h in [*self.hyps, *local]:
            if h != drop:
                m &= pt.tt(h)
        return m & ~pt.tt(f) & pt.FULL == 0


_REL: dict[tuple[frozenset[Formula], Formula, Formula], bool] = {}


# ---------------------------------------------------------------------------------- candidates
@dataclass(eq=False)
class Cand:
    kind: str
    node: Node  # skeleton; lambda binders are placeholders "?0", "?1"
    holes: list[tuple[Local, Formula]]  # (binders, formula); binders may be placeholders
    src: int = 0  # rank of the hypothesis (spines, splits)
    sub: int = 0  # introduction sub-order or spine length
    key: str = ""
    dead: list[int] = field(default_factory=list)  # indices of dead holes
    names: dict[str, str] | None = None  # placeholder -> name, once written
    failed: bool = False
    cost: int = 0  # best-first: 1 per hole closable in one step, else 2 + its size

    @property
    def open(self) -> int:
        return len(self.holes)

    @property
    def size(self) -> int:
        return sum(fsize(f) for _, f in self.holes)


def gen(
    ctx: Ctx, H: Formula, local: Local, path: Sequence[Formula], strat: Strategy, root: bool
) -> list[Cand]:
    """Every candidate reduction of H (scope + `local`), in the strategy's order."""
    avail = dict(ctx.avail)
    lrank: dict[str, int] = {}
    for i, (x, a) in enumerate(local):
        avail[a] = x  # a binder shadows an older name for the same formula
        lrank[x] = len(ctx.names) + i
    blocked = set(path) | {H}
    out: list[Cand] = []
    seen: set[str] = set()

    def add(c: Cand) -> None:
        c.key = " ".join(str(t) for t in pt.r_skel(c.node))
        if c.key not in seen:
            seen.add(c.key)
            out.append(c)

    class _Loop(Exception):
        pass

    known = set(ctx.hyps) | {a for _, a in local}

    def arg(f: Formula, holes: list[tuple[Local, Formula]], loc: Local) -> Node:
        for x, a in reversed(loc):
            if a == f and x.startswith("?"):
                return ("name", x)
        if f in avail:
            return ("name", avail[f])
        # a goal already on the path, under binders that add no new hypothesis: a loop
        if f in blocked and all(a in known for _, a in loc):
            raise _Loop
        holes.append((loc, f))
        return ("hole", f)

    if root and H in avail:
        add(Cand("assume", ("name", avail[H]), []))
    # introductions
    k = H[0]
    try:
        if k == "and":
            hs: list[tuple[Local, Formula]] = []
            add(Cand("intro", ("pair", arg(H[1], hs, local), arg(H[2], hs, local)), hs))
    except _Loop:
        pass
    if k == "or":
        for sub, (tag, side) in enumerate((("inl", H[1]), ("inr", H[2]))):
            try:
                hs = []
                add(Cand("intro", (tag, arg(side, hs, local)), hs, sub=sub))
            except _Loop:
                pass
    if k in ("imp", "not"):
        a, b = H[1], (H[2] if k == "imp" else BOT)
        loc = (*local, ("?0", a))
        try:
            hs = []
            body = arg(b, hs, loc)
            # in an inconsistent context a box whose binder is already a hypothesis adds
            # nothing (and is how the search loops across steps): a last resort there
            late = a in known and ctx.valid([x for _, x in local], BOT)
            add(Cand("box_late" if late else "intro", ("lam", "?0", a, body), hs))
        except _Loop:
            pass
    # elimination spines from hypotheses (binders first in rank: newest)
    srcs = [(n, f, ctx.rank[n]) for f, n in ctx.avail.items() if avail[f] == n]
    srcs += [(x, a, lrank[x]) for x, a in local if avail.get(a) == x]
    for n, F, rk in srcs:
        stack: list[tuple[Node, Formula, int, list[tuple[Local, Formula]]]] = [
            (("name", n), F, 0, [])
        ]
        while stack:
            t, R, ln, hs = stack.pop()
            if ln and R in avail:  # already a name in scope: its own spine continues from there
                continue
            if ln and R == H:
                add(Cand("elim", t, hs, src=rk, sub=ln))
            elif R == BOT and H != BOT:
                add(Cand("elim", ("proj", t, ".elim"), hs, src=rk, sub=ln + 1))
            elif R[0] == "or" and ln <= 1 and H != R:
                x, y = R[1], R[2]
                lf = [a for _, a in local]
                # relevance: the disjunction lies in some minimal support of the goal; an
                # irrelevant split is kept as a last resort, after ex falso
                rel = ctx.relevant(R if ln == 0 else F, lf, H)
                if x not in avail and y not in avail:
                    loc1, loc2 = (*local, ("?0", x)), (*local, ("?1", y))
                    h2 = list(hs)
                    b1, b2 = arg(H, h2, loc1), arg(H, h2, loc2)
                    add(
                        Cand(
                            "case" if rel else "split_late",
                            ("orelim", t, ("lam", "?0", x, b1), ("lam", "?1", y, b2)),
                            h2,
                            src=rk,
                            sub=ln,
                        )
                    )
            if ln >= SPINE_MAX or (ln and R == H):
                continue
            nxt: list[tuple[Node, Formula, int, list[tuple[Local, Formula]]]] = []
            if R[0] in ("imp", "not"):
                try:
                    h2 = list(hs)
                    a_node = arg(R[1], h2, local)
                    nxt.append((("app", t, a_node), R[2] if R[0] == "imp" else BOT, ln + 1, h2))
                except _Loop:
                    pass
            elif R[0] == "and":
                nxt.append((("proj", t, ".1"), R[1], ln + 1, list(hs)))
                nxt.append((("proj", t, ".2"), R[2], ln + 1, list(hs)))
            stack.extend(reversed(nxt))
    # classical detour, ex falso
    nng = ("not", ("not", H))
    if H != BOT and not (H[0] == "not" and H[1][0] == "not") and ("not", H) not in avail:
        if nng in avail:  # double-negation elimination of a hypothesis: an elimination
            add(Cand("elim", ("dne", ("name", avail[nng])), [], src=ctx.rank.get(avail[nng], 0)))
        elif nng not in blocked:
            add(Cand("classical", ("dne", ("hole", nng)), [(local, nng)]))
    if H != BOT and BOT not in avail and BOT not in blocked:
        add(Cand("efq", ("proj", ("hole", BOT), ".elim"), [(local, BOT)]))
    one = one_step(avail)
    for c in out:
        c.cost = sum(1 if f in one else 2 + fsize(f) for _, f in c.holes)
    # an inconsistent context gives truth tables nothing to refute: proximity orders first
    proximity = ctx.valid([a for _, a in local], BOT)
    return order(out, strat, proximity)


def one_step(avail: dict[Formula, str]) -> set[Formula]:
    """Formulas one elimination or introduction away from the names in scope (best-first)."""
    out: set[Formula] = set()
    for f in avail:
        k = f[0]
        if k == "imp" and f[1] in avail or k == "not" and f[1] in avail:
            out.add(f[2] if k == "imp" else BOT)
        elif k == "and":
            out.update((f[1], f[2]))
    return out


def closes_now(c: Cand) -> bool:
    return c.open == 0


KR = {
    "dfs": {"assume": 0, "intro": 1, "elim": 2, "case": 3, "classical": 4, "efq": 5},
    "hyp": {"assume": 0, "elim": 1, "intro": 2, "case": 3, "classical": 4, "efq": 5},
}
LATE = ("split_late", "box_late")
for _k in KR:
    KR[_k].update(split_late=6, box_late=7)


def order(cs: list[Cand], strat: Strategy, proximity: bool = False) -> list[Cand]:
    out = _order(cs, strat)
    if proximity:
        pos = {id(c): i for i, c in enumerate(out)}
        out.sort(key=lambda c: (c.kind in LATE, c.cost, pos[id(c)]))
    return out


def _order(cs: list[Cand], strat: Strategy) -> list[Cand]:
    n = strat.name
    idx = {id(c): i for i, c in enumerate(cs)}
    if n == "dfs":
        return sorted(cs, key=lambda c: (KR["dfs"][c.kind], c.src, c.sub, idx[id(c)]))
    if n == "hyp":
        return sorted(cs, key=lambda c: (KR["hyp"][c.kind], -c.src, c.sub, idx[id(c)]))
    if n == "best":
        return sorted(
            cs,
            key=lambda c: (
                c.kind in LATE,
                c.kind in ("classical", "efq"),  # detours last, as in every strategy
                c.cost,
                c.open,
                KR["dfs"][c.kind],
                idx[id(c)],
            ),
        )
    rank = {"assume": 0, **{k: i + 1 for i, k in enumerate(strat.perm)}, "case": 3}
    rank.update(classical=4, efq=5, split_late=6, box_late=7)
    return sorted(cs, key=lambda c: (rank[c.kind], h64(strat.seed, c.key)))


# ----------------------------------------------------------------------- terms -> the next action
def infer(node: Node, sc: dict[str, Formula]) -> Formula:
    k = node[0]
    if k == "name":
        return sc[node[1]]
    if k == "hole":
        return node[1]
    if k == "app":
        f = infer(node[1], sc)
        return f[2] if f[0] == "imp" else BOT
    if k == "proj" and node[2] in (".1", ".2"):
        f = infer(node[1], sc)
        return f[1] if node[2] == ".1" else f[2]
    raise ValueError(f"cannot infer {k}")


def first_work(
    node: Node, want: Formula, sc: dict[str, Formula]
) -> tuple[str, Any, Formula] | None:
    """The first thing the forward proof of `node` (of type `want`) needs: ("hole", None, F) to
    search, or ("act", subterm, its type), one environment action.  Box bodies come later."""
    k = node[0]
    if k == "name":
        return None
    if k == "hole":
        return ("hole", None, node[1])
    if k == "lam":
        return ("act", node, want)
    if k == "orelim":
        d = node[1]
        if d[0] != "name":
            return first_work(d, infer(d, sc), sc)
        return ("act", node, want)
    kids: list[tuple[Node, Formula]]
    if k == "app":
        tf = infer(node[1], sc)
        kids = [(node[1], tf), (node[2], tf[1])]
    elif k == "proj":
        kids = [(node[1], BOT if node[2] == ".elim" else infer(node[1], sc))]
    elif k == "pair":
        kids = [(node[1], want[1]), (node[2], want[2])]
    elif k in ("inl", "inr"):
        kids = [(node[1], want[1] if k == "inl" else want[2])]
    elif k == "dne":
        kids = [(node[1], ("not", ("not", want)))]
    else:
        raise ValueError(k)
    for c, t in kids:
        r = first_work(c, t, sc)
        if r is not None:
            return r
    return ("act", node, want)


def render_action(ctx: Ctx, node: Node, want: Formula) -> list[str]:
    nh, nb = ctx.nh, ctx.nb
    k = node[0]
    head = ["have", nh, ":", *ftoks(want), ":="]
    if k == "lam":
        return [*head, "(", "fun", "(", nb, ":", *ftoks(node[2]), ")", "=>", "by"]
    if k == "orelim":
        a = node[2][2]
        return [*head, "Or.elim", node[1][1], "(", "fun", "(", nb, ":", *ftoks(a), ")", "=>", "by"]
    if k == "name":
        if node[1] == ctx.last and ctx.sc[node[1]] == ctx.goal and want == ctx.goal:
            return ["exact", node[1]]
        term = [node[1]]
    elif k == "app":
        term = [node[1][1], node[2][1]]
    elif k == "proj":
        term = [node[1][1], node[2]]
    elif k == "pair":
        term = ["⟨", node[1][1], ",", node[2][1], "⟩"]
    elif k in ("inl", "inr"):
        term = ["Or.inl" if k == "inl" else "Or.inr", node[1][1]]
    elif k == "dne":
        term = ["Classical.byContradiction", "(", "fun", "hh", "=>", node[1][1], "hh", ")"]
    else:
        raise ValueError(k)
    return [*head, *term, ";"]


# ------------------------------------------------------------------------------------ recording
class Full(Exception):
    """A unit does not fit the record's budget (tokens, binders or names)."""


def subst(node: Node, m: dict[str, str]) -> Node:
    if not isinstance(node, tuple):
        return node
    k = node[0]
    if k == "name":
        return ("name", m.get(node[1], node[1]))
    if k == "hole":
        return node
    if k == "lam":
        return ("lam", m.get(node[1], node[1]), node[2], subst(node[3], m))
    if k == "proj":
        return ("proj", subst(node[1], m), node[2])
    return (k, *(subst(x, m) for x in node[1:]))


def placeholders(node: Node) -> list[str]:
    out: list[str] = []
    if isinstance(node, tuple):
        if node[0] == "lam" and str(node[1]).startswith("?"):
            out.append(node[1])
        for x in node[1:]:
            if isinstance(x, tuple):
                out += placeholders(x)
    return out


def placeholders_real(node: Node) -> list[str]:
    """Every lambda binder name in a (named) skeleton."""
    out: list[str] = []
    if isinstance(node, tuple):
        if node[0] == "lam":
            out.append(node[1])
        for x in node[1:]:
            if isinstance(x, tuple):
                out += placeholders_real(x)
    return out


class Recorder:
    """The think block as the search writes it.  Mirrors `trace_dsl.check_step` (anchoring,
    binders, flagged dead goals) and writes only true claims, in units that fit or close it."""

    def __init__(self, ctx: Ctx, bud: Budget, explicit: bool) -> None:
        self.T = td()
        self.ctx = ctx
        self.bud = bud
        self.explicit = explicit and not bud.short
        self.lines: list[list[str]] = []
        self.ntok = 0  # body tokens: line tokens plus the <nl> separators
        self.binders: dict[str, Formula] = {}
        self.holes: set[tuple[Local, Formula]] = {((), ctx.goal)}
        self.flagged: set[tuple[Local, Formula]] = set()
        self.open_invalid: list[tuple[Local, Formula]] = []
        self.closed = False
        self.would = 0  # tokens of the natural record past the point where it closed
        self.n_units = 0
        self.ev: collections.Counter[str] = collections.Counter()

    # ---- names
    def assign(self, c: Cand, prefer_nb: bool) -> dict[str, str]:
        m: dict[str, str] = {}
        taken = set(self.binders) | set(self.ctx.sc)
        nb = int(self.ctx.nb[1:])
        for i, ph in enumerate(placeholders(c.node)):
            if i == 0 and prefer_nb and self.ctx.nb not in taken:
                x = self.ctx.nb
            else:
                k = nb + 1
                while f"n{k}" in taken or f"n{k}" in m.values():
                    k += 1
                x = f"n{k}"
            m[ph] = x
            taken.add(x)
        return m

    # ---- claims
    def _ctx(self, local: Local) -> list[str]:
        out: list[str] = []
        for x, a in local:
            out += [x, ":", *ftoks(a)]
        return out

    def line(
        self, local: Local, g: Formula, node: Node = None, no: bool = False, ev: Any = None
    ) -> list[str]:
        ln = [*self._ctx(local), "⊢", *ftoks(g)]
        if node is not None:
            ln += [":=", *pt.r_skel(node)]
        if no:
            ln.append("<no>")
            if ev:
                ln += fmt_countermodel(ev)
        return ln

    def _emit(self, ln: list[str]) -> None:
        self.ntok += len(ln) + (1 if self.lines else 0)
        self.lines.append(ln)
        if self.ntok > self.bud.cap:
            raise Full("tokens")

    def reduce(
        self, local: Local, g: Formula, node: Node, no: bool = False, ev: Any = None
    ) -> list[tuple[Local, Formula]]:
        T = self.T
        if (local, g) not in self.holes:
            raise AssertionError(f"unanchored {g}")
        if any(int(x[1:]) > MAXN for x in placeholders_real(node)):
            raise Full("names")
        ln = self.line(local, g, node, no, ev)
        sk = ln[ln.index(":=") + 1 : ln.index("<no>") if no else len(ln)]
        p = T.Skel(sk)
        if p.skel() != node or p.i != len(sk):
            raise AssertionError(f"render round trip: {ln}")
        binders = dict(self.binders)
        ch = T.Checker(self.ctx.sc, binders)
        ch.check(node, g, list(local))  # never writes an ill-typed reduction (TraceError)
        holes = [(tuple(h[0]), h[1]) for h in ch.holes]
        dead = [h for h in holes if not self.ctx.valid([a for _, a in h[0]], h[1])]
        new = [x for x in binders if x not in self.binders]
        if any(int(x[1:]) > MAXN for x in new) or len(binders) > self.bud.max_binders:
            raise Full("binders")
        if no:
            if not dead:
                raise AssertionError("<no> on a sound reduction")
            if ev is not None:
                T_ = self.T
                ok = any(T_.refutes(ev, self.ctx.hyps + [a for _, a in h[0]], h[1]) for h in dead)
                if not ok:
                    raise AssertionError("countermodel refutes no hole")
            self.flagged.update(dead)
        else:
            self.open_invalid += dead
        self.binders = binders
        self.holes.update(holes)
        self._emit(ln)
        return holes

    def dead_goal(self, local: Local, g: Formula, ev: Any = None) -> None:
        if (local, g) not in self.holes:
            raise AssertionError("unanchored dead goal")
        hyps = self.ctx.hyps + [a for _, a in local]
        if self.ctx.valid([a for _, a in local], g):
            raise AssertionError("<no> on a provable goal")
        if ev is not None and not self.T.refutes(ev, hyps, g):
            raise AssertionError("countermodel does not refute")
        self.flagged.add((local, g))
        self._emit(self.line(local, g, None, True, ev))

    # ---- units
    def unit(self, write: Any, cost: Any) -> bool:
        """Run `write()` as one unit; roll back and close the record if it does not fit."""
        if self.closed:
            self.would += cost()
            return False
        snap = (
            len(self.lines),
            self.ntok,
            dict(self.binders),
            set(self.holes),
            set(self.flagged),
            list(self.open_invalid),
        )
        try:
            write()
            self.n_units += 1
            return True
        except Full as e:
            n, self.ntok, self.binders, self.holes, self.flagged, self.open_invalid = snap
            del self.lines[n:]
            self.closed = True
            self.ev["closed_" + str(e)] += 1
            self.would += cost()
            return False

    def close(self, why: str) -> None:
        if not self.closed:
            self.closed = True
            self.ev["closed_" + why] += 1

    def finish(self) -> list[list[str]]:
        bad = [h for h in dict.fromkeys(self.open_invalid) if h not in self.flagged]
        if bad:
            raise AssertionError("unflagged dead goals")
        return self.lines

    @property
    def body(self) -> int:
        return self.ntok


# -------------------------------------------------------------------------------------- search
class Stuck(Exception):
    pass


@dataclass
class Frame:
    H: Formula
    path: tuple[Formula, ...]
    cands: list[Cand]
    n_rec: int = 0  # candidates examined while recording (in order)
    verified: list[Cand] = field(default_factory=list)  # recorded viable, not failed


@dataclass
class StepOut:
    action: list[str]  # the written action (the search's)
    lines: list[list[str]]
    guess: list[str] | None  # the commit-on-record action
    info: dict[str, Any]


class Search:
    """One state's search at one budget (the environment state is all it reads)."""

    def __init__(self, env: Any, strat: Strategy, bud: Budget) -> None:
        self.ctx = Ctx(env)
        self.strat = strat
        self.bud = bud
        self.rec = Recorder(self.ctx, bud, strat.explicit) if bud.level else None
        self.stack: list[Frame] = []
        # where the record closed: (frame, candidates recorded there, those verified viable)
        self.snap: tuple[Frame, int, list[Cand]] | None = None
        self.after_action = False
        self.nodes = 0
        self.info: collections.Counter[str] = collections.Counter()
        self.final: tuple[Cand, Node, Formula] | None = None

    # ---- recording helpers
    def _closing(self) -> None:
        """The record has just closed: remember where (the commit-on-record frontier)."""
        if self.snap is None and not self.after_action and self.stack:
            fr = self.stack[-1]
            self.snap = (fr, fr.n_rec, [c for c in fr.verified if not c.failed])

    def write_cand(self, fr: Frame, c: Cand, local: Local) -> bool:
        rec = self.rec
        if rec is None:
            return False
        was_closed = rec.closed
        if not rec.closed and fr.n_rec >= self.bud.scan:
            rec.close("scan")
        dead = bool(c.dead)
        prefer = not dead and not local and c.kind in ("intro", "case", *LATE)
        m = rec.assign(c, prefer)
        node = subst(c.node, m)
        hl = [(tuple((m.get(x, x), a) for x, a in loc), f) for loc, f in c.holes]
        evs = [
            countermodel(self.ctx.hyps + [a for _, a in hl[i][0]], hl[i][1])
            if self.bud.evidence
            else None
            for i in c.dead
        ]
        explicit = rec.explicit and dead

        def write() -> None:
            if not dead:
                rec.reduce(local, fr.H, node)
            elif not explicit:
                rec.reduce(local, fr.H, node, no=True, ev=evs[0])
            else:
                rec.reduce(local, fr.H, node)
                for i, ev in zip(c.dead, evs, strict=True):
                    rec.dead_goal(hl[i][0], hl[i][1], ev)

        def cost() -> int:
            n = len(rec.line(local, fr.H, node, dead and not explicit, evs[0] if evs else None)) + 1
            if explicit:
                n += sum(
                    len(rec.line(hl[i][0], hl[i][1], None, True, ev)) + 1
                    for i, ev in zip(c.dead, evs, strict=True)
                )
            return n

        ok = rec.unit(write, cost)
        if ok:
            fr.n_rec += 1
            c.names = m
            self.info["rec_dead" if dead else "rec_viable"] += 1
            if dead:
                self.info["rec_dead_efq" if c.kind == "efq" else "rec_dead_other"] += 1
                self.info["rec_countermodels"] += sum(e is not None for e in evs)
        elif not was_closed:
            self._closing()
        return ok

    # ---- the search
    def run(self) -> StepOut:
        ctx = self.ctx
        G = ctx.goal
        if ctx.last is not None and ctx.sc.get(ctx.last) == G:
            act = ["exact", ctx.last]
            return StepOut(act, [], act, {"forced": "exact"})
        found = self.dfs(G, (G,), 0)
        if found is None:
            raise Stuck("no action")
        c, node, want = found
        act = render_action(ctx, node, want)
        self.after_action = True
        if self.rec is not None and self.bud.lookahead and node[0] in ("lam", "orelim"):
            self.lookahead(c, node)
        guess = self.commit_on_record(act)
        lines = self.rec.finish() if self.rec is not None else []
        info: dict[str, Any] = dict(self.info)
        info.update(
            nodes=self.nodes,
            tokens=self.rec.body if self.rec else 0,
            natural=(self.rec.body + self.rec.would) if self.rec else 0,
            exhausted=bool(self.rec and self.rec.closed),
            action_kind=node[0],
            chosen_kind=c.kind,
        )
        return StepOut(act, lines, guess, info)

    def examine(self, fr: Frame, c: Cand, local: Local) -> None:
        self.nodes += 1
        if self.nodes > NODE_CAP:
            raise Stuck("node cap")
        c.dead = [
            i for i, (loc, f) in enumerate(c.holes) if not self.ctx.valid([a for _, a in loc], f)
        ]
        if self.write_cand(fr, c, local) and not c.dead:
            fr.verified.append(c)
        if c.dead:
            self.info["dead_examined"] += 1
            self.info["dead_examined_efq" if c.kind == "efq" else "dead_examined_other"] += 1

    def dfs(
        self, H: Formula, path: tuple[Formula, ...], depth: int
    ) -> tuple[Cand, Node, Formula] | None:
        """Search goal H (no local binders): with `closers`, first a candidate that closes H
        outright; then the candidates in order, the first viable one followed, the next one tried
        when its subgoal search fails."""
        if depth > DEPTH_CAP:
            return None
        cands = gen(self.ctx, H, (), path, self.strat, root=depth == 0)
        if self.bud.closers:
            first = next((c for c in cands if closes_now(c)), None)
            if first is not None and first is not cands[0]:
                cands = [first, *(c for c in cands if c is not first)]
                self.info["closer_first"] += 1
        fr = Frame(H, path, cands)
        self.stack.append(fr)
        if self.rec is None and self.snap is None:
            self.snap = (fr, 0, [])
        try:
            for c in cands:
                self.examine(fr, c, ())
                if c.dead:
                    continue
                r = self.follow(c, fr, depth)
                if r is not None:
                    return r
                c.failed = True
                if c in fr.verified:
                    fr.verified.remove(c)
                self.info["backtracks"] += 1
            return None
        finally:
            self.stack.pop()

    def follow(self, c: Cand, fr: Frame, depth: int) -> tuple[Cand, Node, Formula] | None:
        w = first_work(c.node, fr.H, self.ctx.sc)
        if w is None:  # the root closed by a name: a reiteration
            return (c, c.node, fr.H)
        if w[0] == "act":
            return (c, w[1], w[2])
        x = w[2]
        return self.dfs(x, (*fr.path, x), depth + 1)

    # ---- lookahead into the committed box
    def lookahead(self, c: Cand, node: Node) -> None:
        rec = self.rec
        if rec is None or rec.closed or c.names is None:
            return
        m = c.names
        boxes = [(tuple((m.get(x, x), a) for x, a in loc), f) for loc, f in c.holes if loc]
        for i, (loc, f) in enumerate(boxes):
            if i >= self.bud.lookahead or rec.closed:
                break
            self.look_into(loc, f, (f,), self.bud.lookahead - i)

    def look_into(self, local: Local, X: Formula, path: tuple[Formula, ...], levels: int) -> None:
        rec = self.rec
        assert rec is not None
        cands = gen(self.ctx, X, local, path, self.strat, root=True)
        fr = Frame(X, path, cands)
        for c in cands:
            if rec.closed:
                return
            self.examine(fr, c, local)
            if c.dead:
                continue
            self.info["lookahead_lines"] += 1
            if levels > 1 and c.names is not None:
                same = [f for loc, f in c.holes if len(loc) == len(local)]
                if same and same[0] not in path:
                    self.look_into(local, same[0], (*path, same[0]), levels - 1)
            return

    # ---- the commit-on-record action (labels)
    def commit_on_record(self, written: list[str]) -> list[str] | None:
        if self.snap is None and self.rec is not None:
            return written  # the record reached the action
        ctx = self.ctx
        if self.snap is None:  # level 0 before any frame: the root
            fr = Frame(ctx.goal, (ctx.goal,), gen(ctx, ctx.goal, (), (ctx.goal,), self.strat, True))
            n_rec, ok = 0, []
        else:
            fr, n_rec, ok = self.snap
        c = ok[0] if ok else (fr.cands[n_rec] if n_rec < len(fr.cands) else None)
        goal, path = fr.H, fr.path
        for _ in range(DEPTH_CAP):
            if c is None:
                return None
            w = first_work(c.node, goal, ctx.sc)
            if w is None:
                return render_action(ctx, c.node, goal)
            if w[0] == "act":
                return render_action(ctx, w[1], w[2])
            goal = w[2]
            path = (*path, goal)
            cs = gen(ctx, goal, (), path, self.strat, root=False)
            c = cs[0] if cs else None
        return None


# ------------------------------------------------------------------------------------ one proof
def ascii_lines(lines: Sequence[Sequence[str]]) -> str:
    return "\n".join(pt.ascii_line(x) for x in lines)


def viable(env: Any, act: list[str] | None) -> bool:
    if act is None:
        return False
    T = td()
    if not T.canonical_names_ok(env, act):
        return False
    try:
        why, _ = T.judge_action(env, act)
    except Exception:
        return False
    return why is None


class ProofFail(Exception):
    pass


def prove(
    prompt: str,
    strat: Strategy,
    level: int | str,
    ladder: Sequence[int] = DEFAULT_LEVELS,
    max_steps: int = MAX_STEPS,
    labels: bool = True,
    cap: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The search entry point: a complete proof of the theorem `prompt`, one think block per step.
    -> (steps [{think, action, min_level[, viable_levels]}], per-step info).  `level` is an effort
    level, UNCAPPED, or "auto" (each step at its min_level).  Raises ProofFail."""
    T = td()
    env = T.Env(prompt, canon=True, base=0, assign=False)
    steps: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    auto = level == "auto"
    buds = {b: budget_for(b) for b in ladder}  # labels: the ladder's own budgets
    main_bud = None if auto else budget_for(int(level), cap)
    for k in range(len(env.prem)):  # premises: restated in order (forced)
        nh = T.next_names(env)[0]
        act = ["have", nh, ":", *ftoks(env.prem[k]), ":=", f"h{k + 1}", ";"]
        steps.append({"think": "", "action": pt.ascii_line(act), "min_level": 0})
        infos.append(
            {
                "kind": "premise",
                "think_tokens": 0,
                "state_tokens": len(env.state_tokens()),
                "action_tokens": len(act),
            }
        )
        ok, why = env.apply(act)
        if not ok:
            raise ProofFail(f"premise: {why}")
    while not env.done:
        if len(steps) >= max_steps:
            raise ProofFail("over max_steps")
        st_tok = len(env.state_tokens())
        info: dict[str, Any] = {
            "kind": "decision",
            "state_tokens": st_tok,
            "consistent": Ctx(env).consistent(),
        }
        outs: dict[int, StepOut] = {}
        try:
            if auto or labels:
                for b in ladder:
                    outs[b] = Search(env, strat, buds[b]).run()
            main: StepOut | None = None
            if main_bud is not None:
                if main_bud.level in outs and main_bud == buds[main_bud.level]:
                    main = outs[main_bud.level]
                else:
                    main = Search(env, strat, main_bud).run()
        except Stuck as e:
            raise ProofFail(f"stuck: {e}") from None
        forced = (main or outs[ladder[0]]).info.get("forced")
        if forced:
            info["kind"] = "exact"
        vl = [b for b in ladder if viable(env, outs[b].guess)] if outs else []
        min_level = vl[0] if vl else None
        if main is not None:
            out, act = main, main.action
        elif min_level is not None:  # auto: the step at its min_level, its own commit
            out = outs[min_level]
            act = out.guess or out.action
            info["level"] = min_level
        else:
            out = outs[ladder[-1]]
            act = out.action
            info["level"] = ladder[-1]
        if not viable(env, act):
            if any(td().is_name(x) is False and x[:1] == "n" and x[1:].isdigit() for x in act):
                raise ProofFail("names exhausted")
            raise ProofFail(f"generator bug: non-viable action {' '.join(act)}")
        lines = out.lines
        think = ["<think>", *pt.join_lines(lines), "</think>"]
        soft = T.check_step(env, think, act, {})
        if soft:
            raise ProofFail(f"generator bug: soft finding {soft[0]}")
        step: dict[str, Any] = {"think": ascii_lines(lines), "action": pt.ascii_line(act)}
        if forced:
            step["min_level"] = 0
        elif outs:
            step["min_level"] = min_level
            step["viable_levels"] = vl
        steps.append(step)
        info.update(out.info)
        info["think_tokens"] = len(pt.join_lines(lines))
        info["action_tokens"] = len(act)
        info["min_level"] = step.get("min_level")
        if outs and not forced:
            info["actions_by_level"] = {str(b): pt.ascii_line(o.action) for b, o in outs.items()}
            info["guess_by_level"] = {
                str(b): (pt.ascii_line(o.guess) if o.guess else None) for b, o in outs.items()
            }
            info["tokens_by_level"] = {str(b): o.info.get("tokens", 0) for b, o in outs.items()}
        infos.append(info)
        ok, why = env.apply(act)
        if not ok:
            raise ProofFail(f"apply: {why}")
    return steps, infos


def ref_steps(prompt: str, proof: str) -> list[list[str]] | None:
    """The reference's actions (statistics only; never passed to the search)."""
    try:
        steps, _, _ = td().decompose(prompt, proof, canon=True)
    except Exception:
        return None
    return [list(a) for _, a, _ in steps]


def trace_for(
    pid: str,
    prompt: str,
    level: int | str,
    ladder: Sequence[int] = DEFAULT_LEVELS,
    seed: int = 0,
    max_steps: int = MAX_STEPS,
    labels: bool = True,
    cap: int | None = None,
    strategy: str | None = None,
    source: str = "",
    reference: str | None = None,
    ref_slack: int | None = None,
    complete: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """-> (record, per-step info, proof info).  The search sees `prompt` only.  `reference` (a proof
    in the source's ND format) is used after the search, for statistics; with `ref_slack` (opt-in)
    also to reject a proof longer than reference + slack (a selection filter, never a search
    input).  `complete` (opt-in, `searchcomplete`): when every strategy fails, the sampled one is
    re-run with the planner fallback.  Raises ProofFail when every strategy fails (and the
    fallback, if on)."""
    order_ = [strategy] if strategy else strategy_order(pid, seed)
    errors: list[str] = []
    ref = ref_steps(prompt, reference) if reference else None
    for name in order_:
        strat = make_strategy(name, pid, seed)
        cap_steps = max_steps
        if ref_slack is not None and ref is not None:
            cap_steps = min(max_steps, len(ref) + ref_slack)
        try:
            steps, infos = prove(prompt, strat, level, ladder, cap_steps, labels, cap)
        except ProofFail as e:
            errors.append(f"{name}: {e}")
            continue
        rec: dict[str, Any] = {
            "id": pid,
            "source": source,
            "mode": "search",
            "prompt": prompt,
            "steps": steps,
            "gen": {
                "format": FORMAT,
                "strategy": name,
                "sampled": order_[0],
                "level": level if isinstance(level, str) else budget_for(level).name,
                "levels": list(ladder),
                "seed": seed,
            },
        }
        pinfo: dict[str, Any] = {"strategy": name, "tries": len(errors) + 1, "errors": errors}
        _proof_stats(pinfo, rec, ref)
        return rec, infos, pinfo
    if complete and not isinstance(level, str):
        # opt-in; searchcomplete imports this module and, on first use, the fork's minlen
        searchcomplete = importlib.import_module(f"{__package__}.searchcomplete")
        name = order_[0]
        try:
            steps, infos, cst = searchcomplete.prove_complete(
                prompt, make_strategy(name, pid, seed), int(level), ladder, labels, cap
            )
        except ProofFail as e:  # the fallback's reason first (batch stats keep 300 characters)
            errors.insert(0, f"complete/{name}: {e}")
            raise ProofFail("; ".join(errors)) from None
        n_plan = sum(bool(i.get("planner")) for i in infos)
        n_bare = sum(bool(s.get("planner")) for s in steps)
        summary = {
            "plans": cst["plans"],
            "plan_steps": n_plan,
            "plan_steps_without_record": n_bare,
            "discarded_steps": cst["discarded_steps"],
            "triggers": cst["triggers"],
            "root_fallbacks": cst["root_fallbacks"],
            "escalations": cst["escalations"],
        }
        rec = {
            "id": pid,
            "source": source,
            "mode": "search",
            "prompt": prompt,
            "steps": steps,
            "gen": {
                "format": FORMAT,
                "strategy": name,
                "sampled": order_[0],
                "level": budget_for(int(level)).name,
                "levels": list(ladder),
                "seed": seed,
                "complete": summary,
            },
        }
        pinfo = {
            "strategy": name,
            "tries": len(errors) + 1,
            "errors": errors,
            "complete": {**summary, "planner_calls": cst["planner_calls"]},
            "planner_secs": cst["planner_secs"],
        }
        _proof_stats(pinfo, rec, ref)
        return rec, infos, pinfo
    raise ProofFail("; ".join(errors))


def _proof_stats(pinfo: dict[str, Any], rec: dict[str, Any], ref: list[list[str]] | None) -> None:
    """Statistics only (never written into the record): the proof's ND line count, and its
    comparison with the reference when there is one."""
    T = td()
    env = T.Env(rec["prompt"], canon=True, base=0, assign=False)
    for s in rec["steps"]:
        env.apply(T.tokens(s["action"]))
    nd = env.nd()
    pinfo["nd_lines"] = nd.count(" : ") if env.done else None
    if ref is not None:
        acts = [pt.ascii_line(a) for a in ref]
        pinfo["ref_steps"] = len(ref)
        pinfo["overhead"] = len(rec["steps"]) - len(ref)
        pinfo["same_as_ref"] = acts == [s["action"] for s in rec["steps"]]


# ---------------------------------------------------------------------------------- batch run
LEN_BINS = ((0, 6, "<=6"), (7, 9, "7-9"), (10, 12, "10-12"), (13, 20, "13-20"), (21, 99, ">20"))
THRESH = (16, 64, 256, 512, 1024)


def len_bin(n: int) -> str:
    return next((lab for lo, hi, lab in LEN_BINS if lo <= n <= hi), "?")


def block_kind(think: str) -> str:
    """e16 characterisation: empty, one viable reduction, contains a refuted one, other."""
    if not think:
        return "empty"
    lines = think.split("\n")
    if any("<no>" in ln for ln in lines):
        return "refuted"
    if len(lines) == 1:
        return "one_candidate"
    return "several_viable"


_W: dict[str, Any] = {}


def _init(kw: dict[str, Any]) -> None:
    _W.update(kw)


def _one(job: tuple[str, str, str | None, int]) -> tuple[str, Any, list[Any], dict[str, Any], str]:
    pid, prompt, proof, n_lines = job
    t0 = time.time()
    try:
        rec, infos, pinfo = trace_for(
            pid,
            prompt,
            _W["level"],
            _W["ladder"],
            _W["seed"],
            _W["max_steps"],
            _W["labels"],
            _W["cap"],
            _W["strategy"],
            _W["source"],
            proof,
            _W["ref_slack"],
            _W.get("complete", False),
        )
    except ProofFail as e:
        return pid, None, [], {"wall": time.time() - t0}, f"fail: {str(e)[:300]}"
    except Exception as e:  # a generator bug: counted, never written
        return pid, None, [], {"wall": time.time() - t0}, f"crash: {type(e).__name__}: {e}"[:300]
    v = td().validate(rec, lean=False, strict=True)
    pinfo["wall"] = time.time() - t0
    if not v["ok"]:
        return pid, None, infos, pinfo, f"validate: {str(v['error'])[:200]}"
    rec["n_lines"] = n_lines
    pinfo["pair_max"] = v["pair_tokens_max"]
    return pid, rec, infos, pinfo, ""


def _pct(xs: Sequence[float], qs: Sequence[float] = (0.5, 0.9, 0.99)) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    out: dict[str, Any] = {"n": len(s), "mean": round(sum(s) / len(s), 2)}
    for q in qs:
        out[f"p{round(q * 100)}"] = s[min(len(s) - 1, int(q * len(s)))]
    out["max"] = s[-1]
    return out


class Stats:
    """Counts and distributions over one run (`report`)."""

    def __init__(self, level: Any, ladder: Sequence[int]) -> None:
        self.level, self.ladder = level, list(ladder)
        self.c: collections.Counter[str] = collections.Counter()
        self.fail: collections.Counter[str] = collections.Counter()
        self.fail_ids: list[dict[str, str]] = []
        self.wall: list[float] = []
        self.think_step: list[int] = []
        self.natural_step: list[int] = []
        self.natural_by_bin: dict[str, list[int]] = collections.defaultdict(list)
        self.proof_x: list[float] = []
        self.proof_y: list[float] = []
        self.think_proof: list[int] = []
        self.overhead: list[int] = []
        self.steps: list[int] = []
        self.minlev: collections.Counter[str] = collections.Counter()
        self.minlev_by_bin: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        self.blockkind: collections.Counter[str] = collections.Counter()
        self.pair: list[int] = []
        self.strat_used: collections.Counter[str] = collections.Counter()
        self.strat_sampled: collections.Counter[str] = collections.Counter()
        self.tries: collections.Counter[int] = collections.Counter()
        self.act_kind: collections.Counter[str] = collections.Counter()
        self.chosen_kind: collections.Counter[str] = collections.Counter()
        self.nd_over: list[int] = []
        self.nd_by_id: dict[str, list[int]] = {}

    def add(
        self, pid: str, rec: Any, infos: list[Any], pinfo: dict[str, Any], err: str, n: int
    ) -> None:
        self.wall.append(pinfo.get("wall", 0.0))
        if rec is None:
            self.c["failed"] += 1
            self.fail[err.split(":")[0] + ": " + err.split(":", 1)[-1].strip()[:70]] += 1
            if len(self.fail_ids) < 200:
                self.fail_ids.append({"id": pid, "error": err})
            return
        self.c["ok"] += 1
        self.strat_used[pinfo["strategy"]] += 1
        self.strat_sampled[rec["gen"]["sampled"]] += 1
        self.tries[pinfo["tries"]] += 1
        self.pair.append(pinfo.get("pair_max", 0))
        if "overhead" in pinfo:
            self.overhead.append(pinfo["overhead"])
            self.c["same_as_ref"] += bool(pinfo["same_as_ref"])
            self.c["with_ref"] += 1
            self.c["over_ref_plus_4"] += pinfo["overhead"] > 4
        self.steps.append(len(rec["steps"]))
        if pinfo.get("nd_lines"):
            if n:
                self.nd_over.append(pinfo["nd_lines"] - n)
            if len(self.nd_by_id) < 20000:
                self.nd_by_id[pid] = [pinfo["nd_lines"], n]
        cp = pinfo.get("complete")
        if cp:
            self.c["repaired"] += 1
            self.c["repaired_discarded_steps"] += cp["discarded_steps"]
            self.c["repaired_plans"] += cp["plans"]
            self.c["repaired_root_fallback"] += bool(cp["root_fallbacks"])
            self.c["repaired_escalated"] += bool(cp["escalations"])
            for t in cp["triggers"]:
                self.c[f"repair_trigger_{t}"] += 1
        b = len_bin(n)
        tot = 0
        for st_, inf in zip(rec["steps"], infos, strict=True):
            k = inf["kind"]
            self.c[f"steps_{k}"] += 1
            tt_ = inf.get("think_tokens", 0)
            tot += tt_
            if k != "decision":
                continue
            self.think_step.append(tt_)
            nat = inf.get("natural", tt_)
            self.natural_step.append(nat)
            self.natural_by_bin[b].append(nat)
            self.c["decision_empty"] += tt_ == 0
            self.c["decision_planner"] += bool(inf.get("planner"))
            self.c["decision_planner_no_record"] += bool(st_.get("planner"))
            self.c["decision_exhausted"] += bool(inf.get("exhausted"))
            self.c["decision_with_dead"] += bool(inf.get("rec_dead"))
            self.c["decision_with_dead_nonefq"] += bool(inf.get("rec_dead_other"))
            self.c["dead_lines_efq"] += inf.get("rec_dead_efq", 0)
            self.c["dead_lines_other"] += inf.get("rec_dead_other", 0)
            self.c["countermodels"] += inf.get("rec_countermodels", 0)
            self.c["decision_backtrack"] += bool(inf.get("backtracks"))
            self.c["decision_lookahead"] += bool(inf.get("lookahead_lines"))
            self.c["decision_closer_first"] += bool(inf.get("closer_first"))
            self.c["decision_consistent"] += bool(inf.get("consistent"))
            self.c["decision_dead_examined"] += bool(inf.get("dead_examined"))
            self.act_kind[str(inf.get("action_kind"))] += 1
            self.chosen_kind[str(inf.get("chosen_kind"))] += 1
            self.blockkind[block_kind(st_["think"])] += 1
            if "min_level" in st_:
                ml = st_["min_level"]
                self.minlev[str(ml)] += 1
                self.minlev_by_bin[b][str(ml)] += 1
            gl = inf.get("guess_by_level")
            al = inf.get("actions_by_level")
            if gl and al:
                self.c["labelled"] += 1
                for lv in self.ladder:
                    if al.get(str(lv)) != al.get(str(self.ladder[-1])):
                        self.c[f"written_action_differs_{lv}_vs_{self.ladder[-1]}"] += 1
                    if gl.get(str(lv)) != gl.get(str(self.ladder[-1])):
                        self.c[f"record_action_differs_{lv}_vs_{self.ladder[-1]}"] += 1
        self.think_proof.append(tot)
        self.proof_x.append(n)
        self.proof_y.append(tot)

    def report(self, wall: float) -> dict[str, Any]:
        c = self.c
        nd = max(c["steps_decision"], 1)
        nat_share = {
            f">{t}": round(
                sum(x > t for x in self.natural_step) / max(len(self.natural_step), 1), 4
            )
            for t in THRESH
        }
        by_bin = {
            b: {
                **_pct(xs),
                **{f">{t}": round(sum(x > t for x in xs) / len(xs), 4) for t in THRESH},
            }
            for b, xs in sorted(self.natural_by_bin.items())
        }
        nml = max(sum(self.minlev.values()), 1)
        return {
            "counts": dict(c),
            "proofs": {"ok": c["ok"], "failed": c["failed"]},
            "success_rate": round(c["ok"] / max(c["ok"] + c["failed"], 1), 5),
            "failures": dict(self.fail.most_common(30)),
            "failure_examples": self.fail_ids,
            "strategy_sampled": dict(self.strat_sampled),
            "strategy_used": dict(self.strat_used),
            "tries": {str(k): v for k, v in sorted(self.tries.items())},
            "wall_per_proof_s": _pct(self.wall, (0.5, 0.9, 0.99)),
            "think_tokens_per_decision_step": _pct(self.think_step),
            "think_tokens_per_proof": _pct(self.think_proof),
            "natural_tokens_per_decision_step": _pct(self.natural_step),
            "natural_share_over": nat_share,
            "natural_by_proof_lines": by_bin,
            "exhausted_share": round(c["decision_exhausted"] / nd, 4),
            "spearman_think_vs_lines": pt._spearman(self.proof_x, self.proof_y),
            "decision_fractions": {
                k: round(c[f"decision_{k}"] / nd, 4)
                for k in (
                    "empty",
                    "with_dead",
                    "with_dead_nonefq",
                    "backtrack",
                    "lookahead",
                    "closer_first",
                    "consistent",
                    "dead_examined",
                )
            },
            "efq_share_of_dead_lines": round(
                c["dead_lines_efq"] / max(c["dead_lines_efq"] + c["dead_lines_other"], 1), 4
            ),
            "block_kind": {k: round(v / nd, 4) for k, v in self.blockkind.most_common()},
            "action_kind": dict(self.act_kind.most_common()),
            "chosen_candidate_kind": dict(self.chosen_kind.most_common()),
            "min_level": {k: round(v / nml, 4) for k, v in sorted(self.minlev.items())},
            "min_level_counts": dict(self.minlev),
            "min_level_by_proof_lines": {
                b: {k: round(v / max(sum(cc.values()), 1), 4) for k, v in sorted(cc.items())}
                for b, cc in sorted(self.minlev_by_bin.items())
            },
            "action_differs": {
                k: round(v / max(c["labelled"], 1), 4)
                for k, v in c.items()
                if k.startswith(("written_action_differs", "record_action_differs"))
            },
            "overhead_vs_reference": _pct(self.overhead, (0.1, 0.5, 0.9, 0.99)),
            "overhead_hist": dict(
                sorted(collections.Counter(max(-6, min(x, 12)) for x in self.overhead).items())
            ),
            "same_proof_as_reference": round(c["same_as_ref"] / max(c["with_ref"], 1), 4),
            "over_reference_plus_4": round(c["over_ref_plus_4"] / max(c["with_ref"], 1), 4),
            "steps_per_proof": _pct(self.steps),
            "max_pair_tokens_per_proof": _pct(self.pair),
            "repaired_by_planner": c["repaired"],
            "planner_step_share": {
                "decision_steps_from_a_plan": round(c["decision_planner"] / nd, 4),
                "decision_steps_without_record": round(c["decision_planner_no_record"] / nd, 4),
            },
            "nd_overhead_vs_source_lines": _pct(self.nd_over, (0.1, 0.5, 0.9, 0.99)),
            "nd_overhead_hist": dict(
                sorted(collections.Counter(max(-6, min(x, 12)) for x in self.nd_over).items())
            ),
            "nd_lines_by_id": self.nd_by_id if len(self.nd_by_id) <= 6000 else None,
            "wall_s": round(wall, 1),
        }


def load_jobs(src: str, start: int, limit: int, exclude: bool) -> tuple[list[Any], dict[str, int]]:
    index = None
    if exclude:
        from .data import DEFAULT_EXCLUDE, EvalIndex

        index = EvalIndex([*DEFAULT_EXCLUDE, pt.TB_TRAIN])
    jobs: list[tuple[str, str, str | None, int]] = []
    excluded: collections.Counter[str] = collections.Counter()
    stem = "k12" if src == pt.K12 else Path(src).stem
    with open(src) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit and i >= start + limit:
                break
            r = json.loads(line)
            if index is not None:
                hits = index.hit(r["prompt"])
                if hits:
                    for h in hits:
                        excluded[h] += 1
                    continue
            ref = r.get("proof") or r.get("reference_proof")
            n = int(r.get("n_lines") or r.get("reference_lines") or 0)
            jobs.append(
                (
                    r.get("name") if stem != "k12" and r.get("name") else f"{stem}_{i}",
                    r["prompt"],
                    ref,
                    n,
                )
            )
    return jobs, dict(excluded)


def run(
    jobs: Sequence[Any],
    out: Path | None,
    kw: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    import multiprocessing as mp

    t0 = time.time()
    st = Stats(kw["level"], kw["ladder"])
    tmp = out.with_suffix(out.suffix + ".tmp") if out is not None else None
    if tmp is not None:
        tmp.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    with (
        open(tmp if tmp is not None else os.devnull, "w") as fo,
        ctx.Pool(workers, _init, (kw,)) as pool,
    ):
        for k, (pid, rec, infos, pinfo, err) in enumerate(pool.imap(_one, jobs, chunksize=16)):
            st.add(pid, rec, infos, pinfo, err, jobs[k][3])
            if rec is not None and tmp is not None:
                fo.write(json.dumps(rec) + "\n")
            if (k + 1) % 2000 == 0:
                print(
                    f"  {k + 1}/{len(jobs)} {st.c['ok']} ok {st.c['failed']} failed "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    if out is not None and tmp is not None:
        os.replace(tmp, out)
    return st.report(time.time() - t0)


def diversity(jobs: Sequence[Any], level: int, seed: int = 0) -> dict[str, Any]:
    """Every strategy on the same theorems: how much do proofs and thinks differ?"""
    import itertools

    by: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for pid, prompt, _, _ in jobs:
        for s in STRATEGIES:
            try:
                rec, _, _ = trace_for(pid, prompt, level, seed=seed, labels=False, strategy=s)
            except ProofFail:
                continue
            by[pid][s] = rec

    def lev(a: Sequence[str], b: Sequence[str]) -> int:
        d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, d[0] = d[0], i
            for j, y in enumerate(b, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
        return d[-1]

    pairs: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    dist: dict[str, list[float]] = collections.defaultdict(list)
    jac: dict[str, list[float]] = collections.defaultdict(list)
    for recs in by.values():
        for a, b in itertools.combinations(STRATEGIES, 2):
            if a not in recs or b not in recs:
                continue
            k = f"{a}-{b}"
            xa = [s["action"] for s in recs[a]["steps"]]
            xb = [s["action"] for s in recs[b]["steps"]]
            ta = [s["think"] for s in recs[a]["steps"]]
            tb = [s["think"] for s in recs[b]["steps"]]
            pairs[k]["n"] += 1
            pairs[k]["same_proof"] += xa == xb
            pairs[k]["same_proof_and_thinks"] += xa == xb and ta == tb
            dist[k].append(lev(xa, xb) / max(len(xa), len(xb)))
            la = {ln for t in ta for ln in t.split("\n") if ln}
            lb = {ln for t in tb for ln in t.split("\n") if ln}
            jac[k].append(len(la & lb) / max(len(la | lb), 1))
    return {
        "theorems": len(by),
        "pairs": {
            k: {
                "n": v["n"],
                "same_proof": round(v["same_proof"] / v["n"], 4),
                "same_proof_and_thinks": round(v["same_proof_and_thinks"] / v["n"], 4),
                "action_edit_distance_norm_mean": round(sum(dist[k]) / len(dist[k]), 4),
                "think_line_jaccard_mean": round(sum(jac[k]) / len(jac[k]), 4),
            }
            for k, v in sorted(pairs.items())
        },
        "distinct_proofs_per_theorem": _pct(
            [
                len({tuple(s["action"] for s in r["steps"]) for r in recs.values()})
                for recs in by.values()
            ]
        ),
    }


def emit_plain(traces: Path, out: Path) -> int:
    """The trace file's proofs as K12-style records (the ND proof the environment renders,
    `Env.nd()`; `state_env.decompose(prompt, proof, canon=True)` gives back the same actions):
    the no-think control on identical proofs (`data.py --plain`)."""
    T = td()
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(traces) as f, open(out, "w") as fo:
        for line in f:
            rec = json.loads(line)
            env = T.Env(rec["prompt"], canon=True, base=0, assign=False)
            for s in rec["steps"]:
                ok, why = env.apply(T.tokens(s["action"]))
                assert ok, why
            nd = env.nd()
            row = {
                "id": rec["id"],
                "prompt": rec["prompt"],
                "proof": nd,
                "n_lines": nd.count(" : "),
                "source_lines": rec.get("n_lines"),
                "gen": rec["gen"],
            }
            fo.write(json.dumps(row) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default="K12", help=f"{list(pt.SOURCES)} or a JSONL path")
    ap.add_argument("--out", default=None, help="the trace JSONL (default OUT_DIR/e<level>.jsonl)")
    ap.add_argument("--level", default="256", help="an effort level, 'inf' (uncapped) or 'auto'")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)), help="the ladder")
    ap.add_argument("--cap", type=int, default=None, help="override the level's token cap")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0: all)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--ref-slack", type=int, default=None, help="opt-in: reject proofs > ref + K")
    ap.add_argument("--strategy", choices=STRATEGIES, default=None, help="force one strategy")
    ap.add_argument("--no-labels", action="store_true", help="skip min_level labels")
    ap.add_argument("--no-exclude", action="store_true", help="keep rows in eval classes")
    ap.add_argument("--analysis-only", action="store_true", help="stats only, no traces written")
    ap.add_argument("--diversity", type=int, default=0, help="all strategies on N theorems")
    ap.add_argument(
        "--closers-high-only",
        action="store_true",
        help="closers-first only at levels >= 128 (levels may then write different proofs)",
    )
    ap.add_argument(
        "--complete",
        action="store_true",
        help="when every strategy fails, re-run with the planner fallback (searchcomplete.py)",
    )
    ap.add_argument(
        "--emit-plain",
        default=None,
        help="also write the proofs as K12-style records {id, prompt, proof, n_lines} (no think)",
    )
    a = ap.parse_args(argv)
    # one module object (and one ProofFail) whether run with -m or imported by searchcomplete; the
    # fork workers inherit it
    if __name__ == "__main__":
        sys.modules.setdefault(f"{__package__}.searchtrace", sys.modules[__name__])
    SETTINGS["closers_all"] = not a.closers_high_only
    level = parse_level(a.level)
    ladder = tuple(int(x) for x in a.levels.split(","))
    if ladder[0] != 0 or list(ladder) != sorted(set(ladder)):
        ap.error("--levels must start at 0 and increase")
    src: str = pt.SOURCES.get(a.source) or a.source
    jobs, excluded = load_jobs(src, a.start, a.limit, not a.no_exclude)
    name = level if isinstance(level, str) else budget_for(level).name
    out = None if a.analysis_only else Path(a.out or f"{OUT_DIR}/e{name}.jsonl")
    stats_path = Path(a.out or f"{OUT_DIR}/e{name}.jsonl")
    print(
        f"{len(jobs)} theorems from {src} ({sum(excluded.values())} eval-class hits excluded), "
        f"level {name}, ladder {ladder}, {a.workers} workers",
        flush=True,
    )
    if a.diversity:
        rep = diversity(
            jobs[: a.diversity], int(level) if not isinstance(level, str) else 256, a.seed
        )
        sp = stats_path.with_name(stats_path.stem + ".diversity.json")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1)[:3000])
        return 0
    kw = {
        "level": level,
        "ladder": ladder,
        "seed": a.seed,
        "max_steps": a.max_steps,
        "labels": not a.no_labels,
        "cap": a.cap,
        "strategy": a.strategy,
        "source": a.source,
        "ref_slack": a.ref_slack,
        "complete": a.complete,
    }
    rep = run(jobs, out, kw, a.workers)
    rep["config"] = {
        "source": src,
        "level": name,
        "levels": list(ladder),
        "budget": budget_for(level, a.cap).__dict__ if not isinstance(level, str) else None,
        "cap": a.cap,
        "seed": a.seed,
        "max_steps": a.max_steps,
        "ref_slack": a.ref_slack,
        "strategy": a.strategy,
        "labels": not a.no_labels,
        "excluded_by_set": excluded,
        "format": FORMAT,
        "analysis_only": a.analysis_only,
        "closers_all": SETTINGS["closers_all"],
        "complete": a.complete,
    }
    if a.emit_plain and out is not None:
        n = emit_plain(out, Path(a.emit_plain))
        rep["config"]["emit_plain"] = {"path": a.emit_plain, "records": n}
    sp = stats_path.with_name(stats_path.stem + ".stats.json")
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(rep, indent=1, default=str))
    print(
        f"{rep['proofs']} -> {out}; think/step {rep['think_tokens_per_decision_step']}; "
        f"natural {rep['natural_tokens_per_decision_step']}; min_level {rep['min_level']}; "
        f"{rep['wall_s']}s; stats {sp}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
