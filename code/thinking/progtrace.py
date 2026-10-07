"""Programmatic thinking traces for the proof-state model (PROPOSAL.md section 3.2 (d), improved).

    python -m code.thinking.progtrace --source K12 --out prog_traces.jsonl --workers 24

One trace per K12 proof, in the pilot format ({id, source, mode, prompt, steps: [{think, action}]},
ASCII surface), checked by `trace_dsl.validate(rec, lean=False, strict=True)`.  **The actions are
K12's reference actions** (`state_env.decompose(prompt, proof, canon=True)`, Leon 2026-10-05): the
think blocks explain, explore and reject, but never change the proof.

A think block at a decision state is a backward search that ends at the reference action:

1. **Plan.**  The reference proof of the current frame's goal is expanded into a term over the
   names in scope (future `have`s inlined, boxes as lambdas).  The path from the goal down to the
   statement the action introduces becomes goal lines `ctx ⊢ F := SKEL`, one per level, each a
   hole of the line above (sibling subgoals stay holes).  The last line is the action's own
   reduction (`⊢ F := n2 n3`, `⊢ (A → B) := fun ( n5 : A ) => ⊢ B`, an `Or.elim` skeleton).
2. **Alternatives** at each level: the other reductions by the goal's connective, hypotheses
   whose consequent is the goal (`n ⊢ A`, `( n ⊢ A ) ⊢ B`, `( n ⊢ A ) .1`), case splits on
   disjunctions in scope, the classical detour `Classical.byContradiction ⊢ ¬¬G` and ex falso.
   A dead one (a hole with a truth-table countermodel) is shown as **backtracking**: the
   reduction, followed one or two levels (`⊢ H <no>`, or a reduction of H whose hole dies, then
   `⊢ H <no>`), then the level's goal restated with another reduction, ending at the reference.
   A dead hole `(A → B)` or `(¬ A)` at the top of the frame may instead be tried as a candidate
   box with lookahead (`have … ( fun ( x : A ) => by` / `x : A ⊢ B <no>` / `<no>`), the pattern
   of the pilot's Peirce trace.  Viable alternatives are shown occasionally, one line, and
   abandoned by restating the goal (no claim).
3. **Candidates.**  Refuted actions (swapped pair or application, `.1`/`.2`, `Or.inl`/`Or.inr`,
   wrong argument, `exact` of a line that is not last, a wrong binder, the base model's own
   distractors when present), each with the looks that show why and `<no>`; dead box candidates
   killed by lookahead; the reference candidate at a random position, sometimes followed by a
   check (looks of its premises, or lookahead into its box).  **Commit rule:** the action is the
   last unrejected candidate.

Every claim is verified while it is written (truth tables over P Q R S, the environment,
`step_check`, the validator's own type checker); every step is re-checked with
`trace_dsl.check_step` and every trace with `trace_dsl.validate(strict=True)`.  A step whose
think fails is regenerated with less exploration, then emptied (counted); a trace that still
fails is dropped (counted).

Randomised against shortcut statistics, from a per-proof seed (sha256 of seed, epoch and id):
the reference's position among alternatives and candidates, the number of dead ends, whether a
check follows a passing candidate, and whether a block is empty (forced steps -- premise
restatements and `exact` -- are mostly empty).  Exploration grows with the number of
alternatives that exist at the state, so length tracks difficulty.  Think-local binders are
capped per block (`--max-binders`), and the block body at `--think-budget` tokens.
Inconsistent contexts (e.g. inside reductio boxes) get no goal-level `<no>`: the truth tables
never certify one there, so only candidates are rejected (PROPOSAL section 2.3).

Effort levels (`--level {0,16,64,256}`, `Cfg.level`): the same proof steps with each think block
capped at that many body tokens (the Stage-1 effort token).  256 is the block above.  Lower
levels generate it unchanged (same rng), recorded as segments (plan, dead end, viable, wrong
candidate, reference, check), then COMPRESS it with a deterministic first-fit ladder, each
variant made of the block's own lines (or such a line + `<no>`) and checked by `check_step`.
16: the full block if it fits, else the plan lines, a prefix of them, looks at the cited
names, the action, nothing.  64: full, no checks/viable, shortened dead ends and refuted
candidates, then segments pruned (ex falso, the reference if the plan names it, deep dead ends,
top dead ends, wrong candidates), prefixes of the plan (+ the action), then the 16
fallbacks.  0: empty.

Output: the JSONL, plus `<out stem>.stats.json` (counts, dropped traces, think tokens per step and
per proof by line count and stratum, backtracking and dead-end fractions, reference-position
histograms, empty-block fractions, maximum pair length, Spearman rho of think length against
difficulty, wall clock).  Eval exclusion (default: `data.DEFAULT_EXCLUDE` plus textbook72 train)
drops rows whose renaming class is in an eval set; `--no-exclude` keeps every row.
"""

from __future__ import annotations

import argparse
import collections
import copy
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dan import trace_dsl
from .tok import LEVELS

FORMAT = "progtrace/1"
K12 = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"
OUT = "/scratch/cowifornia/nd-rl/think/traces/prog_traces.jsonl"
DISTRACTOR_GLOB = "/scratch/cowifornia/nd-rl/think/gen_runs/wave*/distractors.jsonl"
TB_TRAIN = (
    "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72/textbook_train.jsonl"
)
SOURCES = {"K12": K12}

Formula = Any  # trace_dsl / state_env formula tuples: ("atom", "P"), ("bot",), ("not", f), ...
Node = Any  # skeleton AST (trace_dsl.Skel's tuples) or a `Def`
Local = tuple[tuple[str, Formula], ...]

BOT: Formula = ("bot",)
TO_ASCII = {
    "⊢": "|-",
    "¬": "~",
    "∧": "&",
    "∨": "v",
    "→": ">",
    "False": "F",
    "⟨": "<|",
    "⟩": "|>",
}
# env rejections that are a semantic mistake (a malformed action is not shown as a candidate)
SEMANTIC_ENV = (
    "unbound",
    "exact not last",
    "exact goal mismatch",
    "binder mismatch",
    "box on non-arrow",
    "Or.elim on non-or",
    "PR position",
)


# env rejections that are an ordering convention of the environment, not logic (Lean accepts the
# step): skipped as candidates under `Cfg.skip_convention`
CONVENTION_ONLY = ("PR position", "exact not last")


def td() -> Any:
    """The validator module (`trace_dsl.py`), bound to the vendored fork modules."""
    return trace_dsl()


def ascii_line(toks: Sequence[str]) -> str:
    return " ".join(TO_ASCII.get(t, t) for t in toks)


def ftoks(f: Formula) -> list[str]:
    return list(td().ftoks(f))


# ----------------------------------------------------------------------------------- truth tables
# A formula's truth table over P Q R S as a 16-bit mask (valuation v: P = bit 0 ... S = bit 3).
FULL = 0xFFFF
_ATOM = {a: sum(1 << v for v in range(16) if v >> i & 1) for i, a in enumerate("PQRS")}
_TT: dict[Formula, int] = {}


def tt(f: Formula) -> int:
    m = _TT.get(f)
    if m is not None:
        return m
    k = f[0]
    if k == "atom":
        m = _ATOM[f[1]]
    elif k == "bot":
        m = 0
    elif k == "not":
        m = FULL ^ tt(f[1])
    elif k == "and":
        m = tt(f[1]) & tt(f[2])
    elif k == "or":
        m = tt(f[1]) | tt(f[2])
    else:
        m = (FULL ^ tt(f[1])) | tt(f[2])
    _TT[f] = m
    return m


def ent(hyps: Iterable[Formula], g: Formula) -> bool:
    """Classical consequence over P Q R S (as `trace_dsl.entails`, by masks)."""
    m = FULL
    for h in hyps:
        m &= tt(h)
    return m & ~tt(g) & FULL == 0


# ------------------------------------------------------------------------- the reference proof tree
@dataclass(eq=False)
class Def:
    """A future statement of the reference proof, inlined where it is used."""

    key: int  # the step index of the action that introduces it
    name: str
    formula: Formula
    inner: Node = None


@dataclass(eq=False)
class FrameRec:
    kind: str
    goal: Formula
    stmts: list[tuple[int, str, Formula, Node]] = field(default_factory=list)
    exact: str | None = None


def atomic_node(term: list[str]) -> Node:
    """An atomic `have` term (the environment's nine shapes) -> skeleton AST over names."""
    if len(term) == 1:
        return ("name", term[0])
    if term[0] == "⟨":
        return ("pair", ("name", term[1]), ("name", term[3]))
    if term[0] in ("Or.inl", "Or.inr"):
        return ("inl" if term[0] == "Or.inl" else "inr", ("name", term[1]))
    if term[0] == "Classical.byContradiction":
        return ("dne", ("name", term[5]))
    if term[1] in (".1", ".2", ".elim"):
        return ("proj", ("name", term[0]), term[1])
    return ("app", ("name", term[0]), ("name", term[1]))


def parse_have(act: list[str]) -> tuple[str, Formula, int]:
    """(defined name, declared formula, index of the term) of a `have` action."""
    f, used = td().pform(act, 3)
    return act[1], f, used + 1


def build_tree(
    prompt: str, actions: list[list[str]]
) -> tuple[list[Any], list[tuple[FrameRec, int]]]:
    """Replay the reference actions -> (environment before each step, (frame record, statement
    index) of each step).  A frame record holds its statements as (step, name, formula, node);
    a box statement's node is ("box", x, A, sub-record), an Or.elim's is ("orbox", d, (x, A,
    rec1), (y, B, rec2))."""
    T = td()
    env = T.Env(prompt, canon=True, base=0, assign=False)
    envs: list[Any] = []
    pos: list[tuple[FrameRec, int]] = []
    stack = [FrameRec("top", env.frames[-1].goal)]
    for t, act in enumerate(actions):
        envs.append(copy.deepcopy(env))
        fr = stack[-1]
        pos.append((fr, len(fr.stmts)))
        kind_before = env.frames[-1].kind
        ok, why = env.apply(act)
        if not ok:
            raise ValueError(f"reference replay failed at step {t}: {why}")
        if act[0] == "exact":
            fr.exact = act[1]
            if kind_before == "or1":
                sub2 = FrameRec("or2", fr.goal)
                y = env.frames[-1].last
                stack[-1] = sub2
                parent = stack[-2]
                pt, pn, pf, pnode = parent.stmts[-1]
                parent.stmts[-1] = (pt, pn, pf, (*pnode[:3], (y, env.frames[-1].names[y], sub2)))
            elif kind_before != "top":
                stack.pop()
            continue
        name, f, j = parse_have(act)
        t0 = act[j]
        if t0 == "(":
            x = act[j + 3]
            a, _ = T.pform(act, j + 5)
            sub = FrameRec(env.frames[-1].kind, env.frames[-1].goal)
            fr.stmts.append((t, name, f, ("box", x, a, sub)))
            stack.append(sub)
        elif t0 == "Or.elim":
            x = act[j + 5]
            a, _ = T.pform(act, j + 7)
            sub = FrameRec("or1", f)
            fr.stmts.append((t, name, f, ("orbox", ("name", act[j + 1]), (x, a, sub), None)))
            stack.append(sub)
        else:
            fr.stmts.append((t, name, f, atomic_node(act[j:-1])))
    if not env.done:
        raise ValueError("reference replay did not finish")
    return envs, pos


def expand(node: Node, defs: dict[str, Def]) -> Node:
    """Inline the future statements in `defs` into a statement's node; boxes become lambdas."""
    k = node[0]
    if k == "name":
        return defs.get(node[1], node)
    if k == "box":
        _, x, a, sub = node
        return ("lam", x, a, frame_term(sub, 0, defs))
    if k == "orbox":
        _, d, (x, a, r1), second = node
        y, b, r2 = second
        return (
            "orelim",
            expand(d, defs),
            ("lam", x, a, frame_term(r1, 0, defs)),
            ("lam", y, b, frame_term(r2, 0, defs)),
        )
    if k == "proj":
        return ("proj", expand(node[1], defs), node[2])
    if k in ("inl", "inr", "dne"):
        return (k, expand(node[1], defs))
    if k in ("app", "pair"):
        return (k, expand(node[1], defs), expand(node[2], defs))
    raise ValueError(f"unknown node {k}")


def frame_term(
    rec: FrameRec, start: int, outer: dict[str, Def], made: list[Def] | None = None
) -> Node:
    """The term proving `rec`'s goal from its statement `start` on: its exact name with every
    later statement inlined (a statement sees only the definitions before it).  `made` collects
    the frame's own Defs."""
    defs = dict(outer)
    for t, name, f, node in rec.stmts[start:]:
        d = Def(t, name, f)
        d.inner = expand(node, defs)
        defs[name] = d
        if made is not None:
            made.append(d)
    assert rec.exact is not None
    return defs.get(rec.exact, ("name", rec.exact))


def sub_defs(node: Node) -> list[Def]:
    """The Defs directly under a node (not under another Def), left to right."""
    if isinstance(node, Def):
        return [node]
    out: list[Def] = []
    for c in node[1:]:
        if isinstance(c, Def) or (isinstance(c, tuple) and c and isinstance(c[0], str)):
            out += sub_defs(c)
    return out


def find_path(root: Node, key: int, rng: random.Random) -> list[Def] | None:
    """A shortest chain of Defs from the root to the statement introduced at step `key`."""
    if not isinstance(root, Def):
        return None
    frontier: list[list[Def]] = [[root]]
    while frontier:
        hits = [p for p in frontier if p[-1].key == key]
        if hits:
            return rng.choice(hits)
        nxt = []
        for p in frontier:
            for c in sub_defs(p[-1].inner):
                nxt.append(p + [c])
        frontier = nxt
    return None


def collapse(path: list[Def]) -> list[Def]:
    """Drop reiteration levels `⊢ F := ⊢ F` (a statement whose term is another future one)."""
    return [d for i, d in enumerate(path) if i == len(path) - 1 or not isinstance(d.inner, Def)]


# --------------------------------------------------------------------------------- skeleton tokens
def r_skel(n: Node) -> list[str]:
    k = n[0]
    if k == "lam":
        return ["fun", "(", n[1], ":", *ftoks(n[2]), ")", "=>", *r_skel(n[3])]
    if k == "app":
        return [*r_prim(n[1]), *r_prim(n[2])]
    if k == "proj":
        base = n[1]
        b = r_prim(base) if base[0] in ("name", "hole") else ["(", *r_skel(base), ")"]
        return [*b, n[2]]
    return r_prim(n)


def r_prim(n: Node) -> list[str]:
    k = n[0]
    if k == "name":
        return [n[1]]
    if k == "hole":
        return ["⊢", *ftoks(n[1])]
    if k == "pair":
        return ["⟨", *r_skel(n[1]), ",", *r_skel(n[2]), "⟩"]
    if k in ("inl", "inr"):
        return ["Or.inl" if k == "inl" else "Or.inr", *r_prim(n[1])]
    if k == "dne":
        return ["Classical.byContradiction", *r_prim(n[1])]
    if k == "orelim":
        return ["Or.elim", *r_prim(n[1]), "(", *r_skel(n[2]), ")", "(", *r_skel(n[3]), ")"]
    return ["(", *r_skel(n), ")"]


def head(n: Node) -> Any:
    while n[0] == "app":
        n = n[1]
    return n[1] if n[0] == "name" else n[0]


def sig(n: Node) -> tuple[Any, ...]:
    """A reduction's head: alternatives with the reference's head are the reference."""
    k = n[0]
    if k == "app":
        return ("app", head(n))
    if k == "proj":
        return ("proj", n[2], head(n[1]))
    if k == "orelim":
        return ("orelim", head(n[1]))
    if k == "name":
        return ("name", n[1])
    return (k,)


# ------------------------------------------------------------------------------ the block builder
class Budget(Exception):
    """A segment does not fit the token budget or the binder cap."""


@dataclass
class Hole:
    local: Local
    formula: Formula
    dead: bool


@dataclass
class Seg:
    """Lines [start, end) of a block that one generator segment wrote (for effort compression)."""

    kind: str  # plan | dead | viable | ref | verify | wrong | deadbox | other
    depth: int
    start: int
    end: int
    meta: dict[str, Any] = field(default_factory=dict)


class Block:
    """One think block under construction.  Mirrors `trace_dsl.check_step`'s state (binders,
    anchored holes, flagged goals, the pending candidate) and writes only true claims."""

    def __init__(self, env: Any, action: list[str], budget: int, max_binders: int) -> None:
        T = td()
        self.T = T
        self.env = env
        self.action = action
        self.sc: dict[str, Formula] = T.scope(env)
        self.hyps = list(self.sc.values())
        self.goal = env.frames[-1].goal
        self.nh, self.nb = T.next_names(env)
        self.budget = budget
        self.max_binders = max_binders
        self.top = int(self.nb[1:]) + max_binders + 1  # highest think-local name index allowed
        self.lines: list[list[str]] = []
        self.ntok = 0
        self.binders: dict[str, Formula] = {}
        self.holes: set[tuple[Local, Formula]] = {((), self.goal)}
        self.flagged: set[tuple[Local, Formula]] = set()
        self.open_invalid: list[tuple[Local, Formula]] = []
        self.pending: tuple[list[str], str | None] | None = None
        self.survivors: list[list[str]] = []
        self.reserved_tok = 0
        self.reserved_binders = 0
        self.ev: collections.Counter[str] = collections.Counter()
        self.segs: list[Seg] = []

    def mark(self, kind: str, start: int, depth: int = -1, **meta: Any) -> None:
        """Record that lines [start, now) are one segment (no rng, no effect on the lines)."""
        if start < len(self.lines):
            self.segs.append(Seg(kind, depth, start, len(self.lines), meta))

    # ---- transactions
    def snap(self) -> dict[str, Any]:
        return {
            "lines": len(self.lines),
            "ntok": self.ntok,
            "binders": dict(self.binders),
            "holes": set(self.holes),
            "flagged": set(self.flagged),
            "open_invalid": list(self.open_invalid),
            "pending": self.pending,
            "survivors": list(self.survivors),
            "ev": collections.Counter(self.ev),
            "segs": len(self.segs),
        }

    def restore(self, s: dict[str, Any]) -> None:
        del self.lines[s["lines"] :]
        self.ntok = s["ntok"]
        self.binders = s["binders"]
        self.holes = s["holes"]
        self.flagged = s["flagged"]
        self.open_invalid = s["open_invalid"]
        self.pending = s["pending"]
        self.survivors = s["survivors"]
        self.ev = s["ev"]
        del self.segs[s["segs"] :]

    def attempt(self, fn: Callable[[], Any], optional: bool = True) -> bool:
        """Run a segment; roll it back if it breaks the budget (or any claim check)."""
        s = self.snap()
        try:
            fn()
            if optional and self.ntok + self.reserved_tok > self.budget:
                raise Budget("tokens")
            if self.ntok > self.budget:
                raise Budget("tokens")
            return True
        except (Budget, self.T.TraceError) as e:
            self.restore(s)
            self.ev["rollback_" + type(e).__name__] += 1
            return False
        except AssertionError as e:  # a claim the generator got wrong: never written
            self.restore(s)
            self.ev["rollback_assert"] += 1
            self.ev["assert:" + str(e)[:60]] += 1
            return False

    def emit(self, line: list[str]) -> None:
        self.lines.append(line)
        self.ntok += len(line) + 1  # the <nl> separator (one fewer in the block, a safe bound)
        if self.ntok > self.budget:
            raise Budget("tokens")

    # ---- names
    def used_names(self) -> set[str]:
        return set(self.binders)

    def fresh(self, prefer: str | None = None, mandatory: bool = False) -> str:
        """A think-local binder name: `prefer` if still free, else the next free from nb up."""
        n_used = len(self.binders)
        cap = self.max_binders - (0 if mandatory else self.reserved_binders)
        if n_used >= cap:
            raise Budget("binders")
        if prefer is not None and prefer not in self.binders and prefer not in self.sc:
            return prefer
        k = int(self.nb[1:]) + 1  # nb itself is kept for the action's (or a candidate's) box
        while k <= self.top:
            n = f"n{k}"
            if n not in self.binders and n not in self.sc:
                return n
            k += 1
        raise Budget("names")

    # ---- claims
    def local_hyps(self, local: Local) -> list[Formula]:
        return self.hyps + [a for _, a in local]

    def consistent(self, local: Local = ()) -> bool:
        return not ent(self.local_hyps(local), BOT)

    def reduce(self, local: Local, g: Formula, node: Node, no: bool = False) -> list[Hole]:
        """Write `local ⊢ g := node` (with `<no>` if `no`); -> its holes."""
        T = self.T
        if (local, g) not in self.holes:
            raise T.TraceError("unanchored")
        line = [*self._ctx(local), "⊢", *ftoks(g), ":=", *r_skel(node)]
        p = T.Skel(line[line.index(":=") + 1 :])
        parsed = p.skel()
        if p.i != len(p.t) or parsed != node:
            raise AssertionError(f"render round trip: {line}")
        binders = dict(self.binders)
        ch = T.Checker(self.sc, binders)
        try:
            ch.check(node, g, list(local))
            err = None
        except T.TraceError as e:
            err = str(e)
        holes = [Hole(h[0], h[1], not ent(self.local_hyps(h[0]), h[1])) for h in ch.holes]
        if no:
            if err is None and not any(h.dead for h in holes):
                raise AssertionError("<no> on a sound reduction")
            self.flagged.update((h.local, h.formula) for h in holes if h.dead)
            line.append("<no>")
        else:
            if err is not None:
                raise AssertionError(f"ill-typed reduction {err}: {line}")
            self.open_invalid += [(h.local, h.formula) for h in holes if h.dead]
        if any(int(x[1:]) > self.top for x in binders if x not in self.binders):
            raise Budget("names")
        if len(binders) > self.max_binders:
            raise Budget("binders")
        self.binders = binders
        self.holes.update((h.local, h.formula) for h in holes)
        self.emit(line)
        return holes

    def dead_goal(self, local: Local, g: Formula) -> None:
        """Write `local ⊢ g <no>` (g has a countermodel)."""
        if (local, g) not in self.holes:
            raise self.T.TraceError("unanchored")
        if ent(self.local_hyps(local), g):
            raise AssertionError("<no> on a provable goal")
        self.flagged.add((local, g))
        self.emit([*self._ctx(local), "⊢", *ftoks(g), "<no>"])
        self.ev["no_goal"] += 1

    def look(self, name: str) -> None:
        f = self.sc.get(name, self.binders.get(name))
        assert f is not None, name
        self.emit([name, ":", *ftoks(f)])

    def candidate(self, act: list[str]) -> str | None:
        """Write a candidate action; -> why it fails (None: it works)."""
        T = self.T
        self.close(False)
        if not T.canonical_names_ok(self.env, act):
            raise T.TraceError("non-canonical candidate")
        why, e2 = T.judge_action(self.env, act)
        self.pending = (act, why)
        if e2 is not None and len(e2.frames) > len(self.env.frames):
            fr = e2.frames[-1]
            ((x, a),) = fr.names.items()
            if self.binders.get(x, a) == a and x not in self.sc:
                self.binders[x] = a
                self.holes.add((((x, a),), fr.goal))
        self.emit(list(act))
        return why

    def close(self, rejected: bool) -> None:
        if self.pending is None:
            return
        act, why = self.pending
        if rejected != (why is not None):
            raise AssertionError(f"verdict mismatch on {act}: {why}")
        if not rejected:
            self.survivors.append(act)
        self.pending = None

    def reject(self) -> None:
        if self.pending is None:
            raise AssertionError("<no> without a candidate")
        self.close(True)
        self.emit(["<no>"])
        self.ev["no_candidate"] += 1

    def unflagged(self) -> list[tuple[Local, Formula]]:
        return [h for h in dict.fromkeys(self.open_invalid) if h not in self.flagged]

    def _ctx(self, local: Local) -> list[str]:
        out: list[str] = []
        for x, a in local:
            out += [x, ":", *ftoks(a)]
        return out

    def finish(self) -> list[list[str]]:
        self.close(False)
        if self.unflagged():
            raise AssertionError("unflagged dead goals")
        if self.survivors and self.survivors[-1] != self.action:
            raise AssertionError("commit rule")
        return self.lines


# --------------------------------------------------------------------------------- alternatives
@dataclass
class Alt:
    """A reduction of a goal: `build(fresh)` -> its skeleton; `holes` (extra local formulas, hole)
    decide whether it is dead before any name is spent."""

    kind: str  # intro | hyp | case | classical | efq | closed
    sg: tuple[Any, ...]
    holes: list[tuple[tuple[Formula, ...], Formula]]
    build: Callable[[Callable[[], str]], Node]
    dead: bool = False


def hyp_names(blk: Block, local: Local) -> list[tuple[str, Formula]]:
    """One name per distinct hypothesis formula (think-local binders, then n-names, then
    premises)."""
    seen: dict[Formula, str] = {}
    for x, a in reversed(local):
        seen.setdefault(a, x)
    ns = sorted(
        ((k, f) for k, f in blk.sc.items() if k.startswith("n")), key=lambda kv: int(kv[0][1:])
    )
    for k, f in ns + [(k, f) for k, f in blk.sc.items() if k.startswith("h")]:
        seen.setdefault(f, k)
    return [(n, f) for f, n in seen.items()]


def alternatives(blk: Block, local: Local, g: Formula) -> list[Alt]:
    """Every reduction of `g` the generator knows (PROPOSAL section 3.2 (d), item 2)."""
    out: list[Alt] = []
    k = g[0]

    def lam(a: Formula, body_goal: Formula) -> Callable[[Callable[[], str]], Node]:
        return lambda fresh: ("lam", fresh(), a, ("hole", body_goal))

    if k == "and":
        a, b = g[1], g[2]
        out.append(
            Alt(
                "intro",
                ("pair",),
                [((), a), ((), b)],
                lambda f, a=a, b=b: ("pair", ("hole", a), ("hole", b)),
            )
        )
    elif k == "or":
        a, b = g[1], g[2]
        out.append(Alt("intro", ("inl",), [((), a)], lambda f, a=a: ("inl", ("hole", a))))
        out.append(Alt("intro", ("inr",), [((), b)], lambda f, b=b: ("inr", ("hole", b))))
    elif k == "imp":
        out.append(Alt("intro", ("lam",), [((g[1],), g[2])], lam(g[1], g[2])))
    elif k == "not":
        out.append(Alt("intro", ("lam",), [((g[1],), BOT)], lam(g[1], BOT)))
    for n, h in hyp_names(blk, local):
        hk = h[0]
        if h == g:
            out.append(Alt("closed", ("name", n), [], lambda f, n=n: ("name", n)))
            continue
        if hk == "imp" and h[2] == g or hk == "not" and g == BOT:
            out.append(
                Alt(
                    "hyp",
                    ("app", n),
                    [((), h[1])],
                    lambda f, n=n, a=h[1]: ("app", ("name", n), ("hole", a)),
                )
            )
        elif hk == "imp" and h[2][0] == "imp" and h[2][2] == g:
            a, b = h[1], h[2][1]
            out.append(
                Alt(
                    "hyp",
                    ("app", n),
                    [((), a), ((), b)],
                    lambda f, n=n, a=a, b=b: (
                        "app",
                        ("app", ("name", n), ("hole", a)),
                        ("hole", b),
                    ),
                )
            )
        elif hk == "imp" and h[2][0] == "and" and g in (h[2][1], h[2][2]):
            p = ".1" if h[2][1] == g else ".2"
            out.append(
                Alt(
                    "hyp",
                    ("proj", p, n),
                    [((), h[1])],
                    lambda f, n=n, a=h[1], p=p: ("proj", ("app", ("name", n), ("hole", a)), p),
                )
            )
        elif hk == "and" and g in (h[1], h[2]):
            p = ".1" if h[1] == g else ".2"
            out.append(
                Alt("closed", ("proj", p, n), [], lambda f, n=n, p=p: ("proj", ("name", n), p))
            )
        elif hk == "bot":
            out.append(
                Alt(
                    "closed",
                    ("proj", ".elim", n),
                    [],
                    lambda f, n=n: ("proj", ("name", n), ".elim"),
                )
            )
        elif hk == "or":
            a, b = h[1], h[2]
            out.append(
                Alt(
                    "case",
                    ("orelim", n),
                    [((a,), g), ((b,), g)],
                    lambda f, n=n, a=a, b=b: (
                        "orelim",
                        ("name", n),
                        ("lam", f(), a, ("hole", g)),
                        ("lam", f(), b, ("hole", g)),
                    ),
                )
            )
    nng = ("not", ("not", g))
    if g != BOT and not (g[0] == "not" and g[1][0] == "not"):
        out.append(Alt("classical", ("dne",), [((), nng)], lambda f: ("dne", ("hole", nng))))
    if g != BOT:
        out.append(
            Alt(
                "efq",
                ("proj", ".elim", "hole"),
                [((), BOT)],
                lambda f: ("proj", ("hole", BOT), ".elim"),
            )
        )
    base = blk.local_hyps(local)
    for alt in out:
        alt.dead = any(not ent(base + list(extra), hf) for extra, hf in alt.holes)
    return out


def binders_needed(a: Alt) -> int:
    return {"case": 2}.get(a.kind, 1 if a.sg == ("lam",) else 0)


# ------------------------------------------------------------------------------ step generation
@dataclass
class Cfg:
    """Generation knobs (probabilities are per opportunity).  Defaults are profile v2, the
    revision after the 2026-10-05 audit (traces/audit/AUDIT.md); `Cfg.v1()` is the audited
    original."""

    profile: str = "v2"
    think_budget: int = 240  # think body tokens, tags excluded
    max_binders: int = 4  # think-local binder names per block
    p_think_premise: float = 0.0  # v1 0.07: a premise restatement is forced, its plan off-target
    p_think_exact: float = 0.10
    p_think_decision: float = 0.93
    p_think_obvious: float = 0.75
    p_alt_top: float = 0.85  # v1 0.6
    p_alt_deep: float = 0.6  # v1 0.4
    cap_alt_top: int = 3
    cap_alt_deep: int = 2
    p_efq: float = 0.15  # v1 0.25; ex falso is dead 94% of the time, a free `<no>`
    efq_fill: bool = False  # v1 True: ex falso always shown when no other dead alternative is
    p_ref_last: float = 0.7
    p_viable: float = 0.0  # v1 0.15: a working reduction abandoned with no stated reason
    p_leaf_cands: float = 0.7  # v2: stop the plan above the action, decide it by candidates
    p_leaf_cand: float = 0.8  # v2: each available wrong candidate at such a leaf
    p_leaf_zero: float = 0.35  # v2: such a leaf with no wrong candidate (the first one passes)
    p_verify_leaf: float = 0.7  # v2: a check after the passing candidate at such a leaf
    ref_last: bool = True  # v1 False: the reference candidate at a random position
    decoys_after_plan: bool = False  # v1 True: wrong candidates after the plan named the action
    skip_convention: bool = True  # v1 False: rejections that are only PR order / exact-not-last
    p_cand: float = 0.4
    p_distractor: float = 0.75
    cap_cand: int = 3
    p_box_cand: float = 0.5
    p_ref_with_wrong: float = 0.9
    p_ref_cand: float = 0.4
    p_ref_cand_obvious: float = 0.2
    p_verify: float = 0.35
    p_look: float = 0.6
    viable_max_tokens: int = 45
    dead_forms: tuple[float, float, float, float] = (0.3, 0.4, 0.2, 0.1)  # short one two box
    level: int = 256  # effort cap on think-body tokens: 0, 16, 64 or 256 (= the full block)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")

    @classmethod
    def v1(cls, **kw: Any) -> Cfg:
        """The generator as audited (FORMAT progtrace/1)."""
        base: dict[str, Any] = {
            "profile": "v1",
            "p_think_premise": 0.07,
            "p_alt_top": 0.6,
            "p_alt_deep": 0.4,
            "p_efq": 0.25,
            "efq_fill": True,
            "p_viable": 0.15,
            "p_leaf_cands": 0.0,
            "ref_last": False,
            "decoys_after_plan": True,
            "skip_convention": False,
        }
        return cls(**{**base, **kw})


def unreit(d: Def) -> Def:
    while isinstance(d.inner, Def):
        d = d.inner
    return d


def render_level(
    blk: Block,
    d: Def,
    nxt: Def | None,
    rn: dict[str, str],
    local: Local,
    prefer: dict[str, str],
) -> tuple[Node, Local | None, dict[str, str]]:
    """The reference reduction of `d`: its direct sub-Defs become holes (an in-scope reiteration
    becomes its name), binders get fresh think names.  -> (node, local of `nxt`'s hole, the
    renaming in force there)."""
    found: list[tuple[Local, dict[str, str]]] = []
    taken: dict[str, Formula] = {}

    def go(n: Node, rn: dict[str, str], loc: Local) -> Node:
        if isinstance(n, Def):
            if n is nxt or unreit(n) is nxt:
                found.append((loc, rn))
                return ("hole", n.formula)
            inner = unreit(n).inner
            if inner[0] == "name" and (inner[1] in blk.sc or inner[1] in rn):
                return ("name", rn.get(inner[1], inner[1]))
            return ("hole", n.formula)
        k = n[0]
        if k == "name":
            return ("name", rn.get(n[1], n[1]))
        if k == "lam":
            saved = blk.binders
            blk.binders = {**saved, **taken}
            try:
                x = blk.fresh(prefer.get(n[1]), mandatory=True)
            finally:
                blk.binders = saved
            taken[x] = n[2]
            return ("lam", x, n[2], go(n[3], {**rn, n[1]: x}, (*loc, (x, n[2]))))
        if k == "proj":
            return ("proj", go(n[1], rn, loc), n[2])
        if k in ("inl", "inr", "dne"):
            return (k, go(n[1], rn, loc))
        if k in ("app", "pair"):
            return (k, go(n[1], rn, loc), go(n[2], rn, loc))
        if k == "orelim":
            return ("orelim", go(n[1], rn, loc), go(n[2], rn, loc), go(n[3], rn, loc))
        raise ValueError(k)

    node = go(unreit(d).inner, rn, local)
    if found:
        return node, found[0][0], found[0][1]
    return node, None, rn


@dataclass
class Cand:
    src: str  # ref | distractor | swap | flip | wrongfn | wrongarg | ... | deadbox
    act: list[str]
    looks: list[str] = field(default_factory=list)
    box: Formula = None


class StepGen:
    """One think block for one state and its reference action."""

    def __init__(
        self,
        env: Any,
        action: list[str],
        kind: str,
        path: list[Def] | None,
        distractors: list[list[str]],
        cfg: Cfg,
        rng: random.Random,
        target: Def | None = None,
    ) -> None:
        self.env, self.action, self.kind, self.path = env, action, kind, path
        self.target = target  # the Def the action introduces (None for exact)
        self.obvious = kind == "decision" and is_obvious(action, env.frames[-1].goal, path)
        if kind == "premise" and path:
            self.path = path[:1]  # a premise step thinks at most one level ahead
        self.distractors, self.cfg, self.rng = distractors, cfg, rng
        self.info: dict[str, Any] = {}
        self._dry = False
        self.leaf_cands = False  # v2: the plan stops above the action; candidates decide it
        self.segs: list[Seg] = []  # the final block's segments, tiling its lines

    def bump(self, key: str, n: int = 1) -> None:
        if not self._dry:
            self.info[key] = self.info.get(key, 0) + n

    # ---- the plan: the goal reduced level by level down to the action
    def plan(self, blk: Block, explore: bool) -> bool:
        """-> True if the plan reached the action's own reduction."""
        path = self.path or []
        local: Local = ()
        rn: dict[str, str] = {}
        prefer: dict[str, str] = {}
        if path and self.action[0] == "have" and not self.leaf_cands:
            inner = unreit(path[-1]).inner
            if inner[0] == "lam":
                prefer[inner[1]] = blk.nb
            elif inner[0] == "orelim":
                prefer[inner[2][1]] = blk.nb
        for i, d in enumerate(path):
            nxt = path[i + 1] if i + 1 < len(path) else None
            g = d.formula
            post: list[Alt] = []
            n_pre = 0
            if explore:
                try:
                    node0, _, _ = render_level(blk, d, nxt, rn, local, prefer)
                except Budget:
                    return False
                n_pre, post = self.alts_before(blk, local, g, sig(node0), i)
            try:
                node, child_local, rn2 = render_level(blk, d, nxt, rn, local, prefer)
            except Budget:
                return False
            mark = blk.snap()
            nb0 = len(blk.binders)
            try:
                blk.reduce(local, g, node)
            except (Budget, blk.T.TraceError, AssertionError):
                blk.restore(mark)
                return False
            blk.mark("plan", mark["lines"], i)
            blk.reserved_tok = max(0, blk.reserved_tok - (blk.ntok - mark["ntok"]))
            blk.reserved_binders = max(0, blk.reserved_binders - (len(blk.binders) - nb0))
            self.bump("plan_lines")
            n_post = self.alts_after(blk, local, g, post, i) if post else 0
            if n_pre + n_post:
                self.info.setdefault("alt_pos", []).append((n_pre, n_pre + n_post + 1))
            if nxt is None:
                return True
            if child_local is None:
                return False
            local, rn = child_local, rn2
        return False

    def alts_before(
        self, blk: Block, local: Local, g: Formula, ref_sig: tuple[Any, ...], depth: int
    ) -> tuple[int, list[Alt]]:
        """Dead alternatives (backtracking) and sometimes one viable one before the reference
        reduction of g.  -> (how many dead ones were written, those to write after it)."""
        cfg, rng = self.cfg, self.rng
        top = depth == 0
        alts = [a for a in alternatives(blk, local, g) if a.sg != ref_sig]
        dead = [a for a in alts if a.dead and a.kind != "efq"]
        efq = [a for a in alts if a.dead and a.kind == "efq"]
        rng.shuffle(dead)
        if efq and ((cfg.efq_fill and not dead) or rng.random() < cfg.p_efq):
            dead.insert(rng.randrange(len(dead) + 1), efq[0])
        self.bump("alts_available", len(dead))
        p = cfg.p_alt_top if top else cfg.p_alt_deep
        cap = cfg.cap_alt_top if top else cfg.cap_alt_deep
        shown = [a for a in dead if rng.random() < p][:cap]
        viable = [
            a
            for a in alts
            if not a.dead
            and (
                a.kind in ("intro", "case", "hyp")
                or (a.kind == "classical" and g[0] in ("atom", "or"))
            )
        ]
        self.bump("viable_available", len(viable))
        pv = cfg.p_viable * (2.0 if not blk.consistent(local) else 1.0) * (0.4 if shown else 1.0)
        vshow = [rng.choice(viable)] if viable and top and rng.random() < pv else []
        if not shown and not vshow:
            return 0, []
        k = len(shown) if rng.random() < cfg.p_ref_last else rng.randrange(len(shown) + 1)
        pre, post = shown[:k], shown[k:]
        items: list[tuple[str, Alt]] = [("dead", a) for a in pre] + [("viable", a) for a in vshow]
        rng.shuffle(items)
        n_pre = 0
        for how, a in items:
            if how == "dead":
                if self.explore_dead(blk, local, g, a, depth):
                    n_pre += 1
                    self.bump("dead_ends")
            else:
                s0 = len(blk.lines)
                if blk.attempt(lambda a=a: self._viable(blk, local, g, a)):
                    blk.mark("viable", s0, depth)
                    self.bump("viable_shown")
        if n_pre:
            self.info["backtrack"] = True
        return n_pre, post

    def _viable(self, blk: Block, local: Local, g: Formula, a: Alt) -> None:
        """A working alternative, one short line, abandoned by restating the goal (no claim)."""
        blk.reduce(local, g, a.build(lambda: blk.fresh()))
        if len(blk.lines[-1]) > self.cfg.viable_max_tokens:
            raise Budget("viable alternative too long")

    def alts_after(self, blk: Block, local: Local, g: Formula, post: list[Alt], depth: int) -> int:
        """Dead alternatives after the reference reduction: one line each, with `<no>`."""
        n = 0
        for a in post:
            s0 = len(blk.lines)
            if blk.attempt(lambda a=a: blk.reduce(local, g, a.build(lambda: blk.fresh()), no=True)):
                blk.mark("dead", s0, depth, pos="post", form="short", alt=a.kind)
                n += 1
                self.bump("dead_ends")
                self.bump("dead_after_ref")
        return n

    def explore_dead(self, blk: Block, local: Local, g: Formula, a: Alt, depth: int) -> bool:
        """A dead alternative: one line with `<no>`, or followed one or two levels."""
        form = self.rng.choices(("short", "one", "two", "box"), self.cfg.dead_forms)[0]
        top = depth == 0
        s0 = len(blk.lines)
        for f in dict.fromkeys((form, "one", "short")):
            if blk.attempt(lambda f=f: self._dead(blk, local, g, a, f, top)):
                blk.mark("dead", s0, depth, pos="pre", form=f, alt=a.kind)
                self.info.setdefault("dead_forms", []).append(f)
                return True
        return False

    def _dead(self, blk: Block, local: Local, g: Formula, a: Alt, form: str, top: bool) -> None:
        if form == "short":
            blk.reduce(local, g, a.build(lambda: blk.fresh()), no=True)
            return
        holes = blk.reduce(local, g, a.build(lambda: blk.fresh()))
        followed = False
        for h in holes:
            if not h.dead or (h.local, h.formula) in blk.flagged:
                continue
            if not followed and form == "two":
                followed = self._two_level(blk, h)
                if followed:
                    continue
            if (
                not followed
                and form == "box"
                and top
                and not h.local
                and self._dead_box(blk, h.formula)
            ):
                followed = True
            blk.dead_goal(h.local, h.formula)
        if form in ("two", "box") and not followed:
            raise Budget("form unavailable")

    def _two_level(self, blk: Block, h: Hole) -> bool:
        """`⊢ H := D'`, the `<no>` of each dead hole of D', then `⊢ H <no>`."""
        subs = [
            s
            for s in alternatives(blk, h.local, h.formula)
            if s.holes and s.kind in ("intro", "hyp", "case")
        ]
        if not subs:
            return False
        s = self.rng.choice(subs)
        for x in blk.reduce(h.local, h.formula, s.build(lambda: blk.fresh())):
            if x.dead and (x.local, x.formula) not in blk.flagged:
                blk.dead_goal(x.local, x.formula)
        blk.dead_goal(h.local, h.formula)
        self.info["deep_dead_end"] = True
        return True

    def _dead_box(self, blk: Block, hf: Formula) -> bool:
        """A candidate box for hf (an implication or negation with a countermodel): it passes the
        environment and step_check, the lookahead finds its goal dead, `<no>` (the Peirce
        pattern of the pilot)."""
        if hf[0] not in ("imp", "not"):
            return False
        a = hf[1]
        act = ["have", blk.nh, ":", *ftoks(hf), ":=", "(", "fun", "(", blk.nb, ":", *ftoks(a)]
        act += [")", "=>", "by"]
        why = blk.candidate(act)
        if why is None or not why.startswith("dead"):
            raise Budget("box candidate is not dead")
        inner = hf[2] if hf[0] == "imp" else BOT
        if blk.binders.get(blk.nb) != a or (((blk.nb, a),), inner) not in blk.holes:
            raise Budget("no lookahead into the box")
        blk.dead_goal(((blk.nb, a),), inner)
        self.info["lookahead_dead"] = True
        blk.reject()
        self.info["dead_box"] = True
        return True

    # ---- candidates
    def wrong_candidates(self, blk: Block) -> list[Cand]:
        """Plausible actions that fail here (environment or step_check), with the names whose
        formulas show why; the base model's own mistakes first."""
        T = blk.T
        act, sc = self.action, blk.sc
        out: list[Cand] = []
        seen = {" ".join(act)}
        ns = [(n, f) for n, f in sc.items() if n.startswith("n")]

        def add(src: str, a: list[str], looks: Sequence[str] = ()) -> None:
            key = " ".join(a)
            if key in seen:
                return
            seen.add(key)
            try:
                if not T.canonical_names_ok(blk.env, a):
                    return
                why, _ = T.judge_action(blk.env, a)
            except Exception:
                return
            if why is None or why.startswith("dead"):
                return
            if why.startswith("env") and not any(s in why for s in SEMANTIC_ENV):
                return
            if self.cfg.skip_convention and any(s in why for s in CONVENTION_ONLY):
                return
            out.append(Cand(src, a, [x for x in looks if x in sc]))

        for d in self.distractors:
            add("distractor", d, [x for x in d[4:] if x in sc])
        if act[0] == "exact":
            for n, f in ns:
                if n != act[1]:
                    add("exact", ["exact", n], [] if f == blk.goal else [n])
            for c in out:
                c.src = "exact_look" if c.looks else c.src
            return out
        _, f, j = parse_have(act)
        pre, term = act[:j], act[j:]
        if self.kind == "premise":
            k = int(term[0][1:])
            for i, p in enumerate(blk.env.prem):
                if i + 1 != k:
                    add("premise", ["have", blk.nh, ":", *ftoks(p), ":=", f"h{i + 1}", ";"])
            return out
        if term[0] == "(":
            a = T.pform(term, 5)[0]
            for _n, g in ns:
                if g != a and g[0] != "bot":
                    add("binder", [*pre, *term[:5], *ftoks(g), ")", "=>", "by"])
                    break
            return out
        if term[0] == "Or.elim":
            d = term[1]
            for n, g in ns:
                if n != d and g[0] != "or":
                    add("orelim", [*pre, "Or.elim", n, *term[2:]], [n])
                    break
            df = sc.get(d)
            if df is not None and df[0] == "or" and df[1] != df[2]:
                add("binder", [*pre, *term[:7], *ftoks(df[2]), ")", "=>", "by"], [d])
            return out
        body = term[:-1]
        if body[0] == "⟨":
            add("swap", [*pre, "⟨", body[3], ",", body[1], "⟩", ";"], [body[3]])
        elif body[0] in ("Or.inl", "Or.inr"):
            other = "Or.inr" if body[0] == "Or.inl" else "Or.inl"
            add("flip", [*pre, other, body[1], ";"], [body[1]])
        elif body[0] == "Classical.byContradiction":
            for n, _ in ns:
                if n != body[5]:
                    add("dne", [*pre, *body[:5], n, *body[6:], ";"], [n])
                    break
        elif len(body) == 2 and body[1] in (".1", ".2"):
            add("flip", [*pre, body[0], ".2" if body[1] == ".1" else ".1", ";"], [body[0]])
        elif len(body) == 2 and body[1] == ".elim":
            for n, g in ns:
                if n != body[0] and g != BOT:
                    add("elim", [*pre, n, ".elim", ";"], [n])
                    break
        elif len(body) == 2:
            fn, arg = body
            add("swap", [*pre, arg, fn, ";"], [arg])
            for n, g in ns:
                if n not in (fn, arg) and g[0] in ("imp", "not"):
                    add("wrongfn", [*pre, n, arg, ";"], [n, arg])
                    break
            for n, g in ns:
                if n not in (fn, arg) and g != sc.get(arg):
                    add("wrongarg", [*pre, fn, n, ";"], [fn, n])
                    break
        elif len(body) == 1:
            for n, g in ns:
                if n != body[0] and g != f:
                    add("reit", [*pre, n, ";"], [n])
                    break
        return out

    def dead_boxes(self, blk: Block) -> list[Formula]:
        """Implications and negations that reductions of the goal need but that have a
        countermodel: a box candidate for one passes the environment and step_check, and only
        a look into it shows it is dead."""
        out: list[Formula] = []
        if self.kind != "decision" or not blk.consistent():
            return out
        for a in alternatives(blk, (), blk.goal):
            for extra, hf in a.holes:
                if (
                    not extra
                    and hf[0] in ("imp", "not")
                    and not ent(blk.hyps, hf)
                    and hf not in out
                ):
                    out.append(hf)
        return out

    def candidates(self, blk: Block, explore: bool, force_ref: bool) -> None:
        cfg, rng = self.cfg, self.rng
        wrong: list[Cand] = []
        decoy = (
            not cfg.decoys_after_plan
            and self.kind == "decision"
            and not self.leaf_cands
            and self.info.get("plan_reached")
        )
        zero = self.leaf_cands and rng.random() < cfg.p_leaf_zero
        if explore and not decoy and not zero:
            avail = self.wrong_candidates(blk)
            boxes = self.dead_boxes(blk)
            self.bump("cands_available", len(avail) + len(boxes))
            for c in avail:
                p = cfg.p_leaf_cand if self.leaf_cands else cfg.p_cand
                if rng.random() < (cfg.p_distractor if c.src == "distractor" else p):
                    wrong.append(c)
            if boxes and rng.random() < cfg.p_box_cand:
                wrong.append(Cand("deadbox", [], box=rng.choice(boxes)))
            if self.leaf_cands and not wrong:
                wrong = (
                    [rng.choice(avail)] if avail else [Cand("deadbox", [], box=rng.choice(boxes))]
                )
            rng.shuffle(wrong)
            wrong = wrong[: cfg.cap_cand]
        if self.kind == "exact":
            wrong = wrong[:1]
        p_ref = cfg.p_ref_with_wrong if wrong else cfg.p_ref_cand
        if not wrong and self.obvious:
            p_ref = cfg.p_ref_cand_obvious
        with_ref = force_ref or rng.random() < p_ref
        items = list(wrong)
        if with_ref:
            at = len(items) if cfg.ref_last else rng.randrange(len(items) + 1)
            items.insert(at, Cand("ref", self.action))
        n_rej = before = 0
        ref_done = False
        for c in items:
            s0 = len(blk.lines)
            if c.src == "ref":
                if blk.attempt(lambda: blk.candidate(self.action), optional=False):
                    blk.mark("ref", s0)
                    ref_done, before = True, n_rej
                    pv = cfg.p_verify_leaf if self.leaf_cands else cfg.p_verify
                    s1 = len(blk.lines)
                    if explore and rng.random() < pv and blk.attempt(lambda: self.verify(blk)):
                        blk.mark("verify", s1)
                        self.info["verify"] = True
            elif c.src == "deadbox":
                if blk.attempt(lambda c=c: self._dead_box(blk, c.box)):
                    blk.mark("deadbox", s0)
                    n_rej += 1
            elif blk.attempt(lambda c=c: self._refute(blk, c)):
                blk.mark("wrong", s0, distractor=c.src == "distractor")
                n_rej += 1
                if c.src == "distractor":
                    self.bump("distractors_used")
        # a look into the action's box without writing the candidate
        s2 = len(blk.lines)
        if (
            explore
            and not ref_done
            and rng.random() < cfg.p_verify
            and self.action[0] == "have"
            and blk.attempt(lambda: self.verify(blk, box_only=True))
        ):
            blk.mark("verify", s2)
            self.info["verify"] = True
        self.bump("rejected", n_rej)
        if ref_done:
            self.info["ref_cand"] = True
            if n_rej:
                self.info["cand_pos"] = (before, n_rej + 1)

    def _refute(self, blk: Block, c: Cand) -> None:
        if blk.candidate(c.act) is None:
            raise Budget("the candidate passes")
        for n in c.looks[:2]:
            if c.src == "exact_look" or self.rng.random() < self.cfg.p_look:
                blk.look(n)
        blk.reject()

    def verify(self, blk: Block, box_only: bool = False) -> None:
        """A check after the passing reference candidate: the formulas it cites, or a look into
        the box it opens (the box's own reduction)."""
        T = blk.T
        act = self.action
        if box_only and (act[0] != "have" or act[parse_have(act)[2]] not in ("(", "Or.elim")):
            raise Budget("not a box")
        if act[0] == "exact":
            blk.look(act[1])
            return
        _, _, j = parse_have(act)
        term = act[j:]
        if term[0] in ("(", "Or.elim"):
            if self.target is None:
                raise Budget("no target")
            inner = self.target.inner
            lam = inner if inner[0] == "lam" else inner[2]
            x, a, body = lam[1], lam[2], lam[3]
            nb = blk.nb
            if not isinstance(body, Def) or blk.binders.get(nb) != a:
                raise Budget("nothing to look into")
            loc: Local = ((nb, a),)
            body = unreit(body)
            if (loc, body.formula) not in blk.holes:
                raise Budget("no box hole")
            node, _, _ = render_level(blk, body, None, {x: nb}, loc, {})
            blk.reduce(loc, body.formula, node)
            self.info["lookahead_ok"] = True
            return
        names = list(dict.fromkeys(t for t in term if T.is_name(t) and t in blk.sc))
        if not names:
            raise Budget("nothing to check")
        for n in names[:2]:
            blk.look(n)

    # ---- the whole block
    def generate(self, explore: bool = True) -> list[list[str]]:
        cfg = self.cfg
        if (
            explore
            and cfg.p_leaf_cands
            and self.kind == "decision"
            and self.path
            and self.action[0] == "have"
        ):
            probe = Block(self.env, self.action, 10**6, cfg.max_binders)
            if (
                self.wrong_candidates(probe) or self.dead_boxes(probe)
            ) and self.rng.random() < cfg.p_leaf_cands:
                self.leaf_cands = True
                self.info["leaf_cands"] = True
                self.path = self.path[:-1] or None
        dry = self._dry_plan()
        reached = dry is not None and dry[0]
        need_ref = self.action[0] == "have" and not reached
        blk = Block(self.env, self.action, cfg.think_budget, cfg.max_binders)
        ref_tok = len(self.action) + 1
        if dry is not None and dry[1] + ref_tok > cfg.think_budget:
            self.path = self._truncate(cfg.think_budget // 2)
            dry = self._dry_plan()
            need_ref = True
            self.info["plan_truncated"] = True
        blk.reserved_tok = (dry[1] if dry else 0) + (ref_tok if need_ref else 0)
        blk.reserved_binders = dry[2] if dry else 0
        reached = self.plan(blk, explore)
        self.info["plan_reached"] = reached
        blk.reserved_binders = 0
        self.candidates(
            blk, explore, force_ref=need_ref or (self.action[0] == "have" and not reached)
        )
        lines = blk.finish()
        self.segs = tile(blk.segs, len(lines))
        self.info["other_segments"] = sum(s.kind == "other" for s in self.segs)
        self.info["binders"] = len(blk.binders)
        self.info["no_goal"] = blk.ev["no_goal"]
        self.info["no_candidate"] = blk.ev["no_candidate"]
        return lines

    def _dry_plan(self) -> tuple[bool, int, int] | None:
        """(reached, tokens, binders) of the plan alone; None without a plan."""
        if not self.path:
            return None
        dry = Block(self.env, self.action, 10**6, self.cfg.max_binders)
        self._dry = True
        try:
            reached = self.plan(dry, explore=False)
        finally:
            self._dry = False
        return reached, dry.ntok, len(dry.binders)

    def _truncate(self, limit: int) -> list[Def]:
        """The longest prefix of the plan that fits `limit` tokens."""
        full = list(self.path or [])
        for n in range(len(full), 0, -1):
            self.path = full[:n]
            dry = self._dry_plan()
            if dry is not None and dry[1] <= limit:
                return full[:n]
        return []


# ---------------------------------------------------------------------------------- one proof
def step_kind(act: list[str]) -> str:
    """premise (restating hJ), exact, or decision (as `gen.dsl.step_kind`)."""
    if act and act[0] == "exact":
        return "exact"
    if len(act) >= 3 and act[-1] == ";" and act[-3] == ":=" and act[-2].startswith("h"):
        return "premise"
    return "decision"


def proof_seed(seed: int, epoch: int, pid: str) -> int:
    h = hashlib.sha256(f"progtrace/{seed}/{epoch}/{pid}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def is_obvious(act: list[str], goal: Formula, path: list[Def] | None) -> bool:
    """The introduction the goal's connective dictates (its own box), or a reiteration of it."""
    if act[0] != "have" or not path or len(path) != 1:
        return False
    inner = unreit(path[0]).inner
    if inner[0] == "lam":
        return goal[0] in ("imp", "not")
    return inner[0] == "name"


def join_lines(lines: Sequence[Sequence[str]]) -> list[str]:
    out: list[str] = []
    for i, ln in enumerate(lines):
        out += (["<nl>"] if i else []) + list(ln)
    return out


# ----------------------------------------------------------------------- effort-level compression
def tile(segs: Sequence[Seg], n: int) -> list[Seg]:
    """The segments in line order, any uncovered line a `Seg("other", ...)`."""
    out: list[Seg] = []
    pos = 0
    for sg in sorted(segs, key=lambda x: x.start):
        if sg.start > pos:
            out.append(Seg("other", -1, pos, sg.start))
        if sg.start < pos:  # overlapping: should not happen; keep the line range disjoint
            sg = Seg(sg.kind, sg.depth, pos, sg.end, sg.meta)
            if sg.end <= sg.start:
                continue
        out.append(sg)
        pos = sg.end
    if pos < n:
        out.append(Seg("other", -1, pos, n))
    return out


def short_form(sg: Seg, ls: list[list[str]]) -> list[list[str]] | None:
    """A one-or-two line form of a segment (its first line, flagged), or None."""
    if sg.kind == "dead" and len(ls) > 1:
        first = ls[0]
        if "⊢" in first and ":=" in first and first[-1] != "<no>":
            return [[*first, "<no>"]]
    if sg.kind in ("wrong", "deadbox") and len(ls) > 2:
        return [ls[0], ["<no>"]]
    return None


def prune_order(segs: Sequence[Seg], plan_reached: bool) -> list[int]:
    """Indices of segments in the order the level-64 ladder drops them."""
    idx = range(len(segs))

    def dead(i: int) -> bool:
        return segs[i].kind == "dead"

    efq = sorted(
        (i for i in idx if dead(i) and segs[i].meta.get("alt") == "efq"),
        key=lambda i: (-segs[i].depth, -i),
    )
    done = set(efq)
    deep = sorted(
        (i for i in idx if dead(i) and segs[i].depth >= 1 and i not in done),
        key=lambda i: (-segs[i].depth, segs[i].meta.get("pos") != "post", -i),
    )
    top = sorted(
        (i for i in idx if dead(i) and segs[i].depth < 1 and i not in done),
        key=lambda i: (segs[i].meta.get("pos") != "post", -i),
    )

    def wrongs(distractor: bool) -> list[int]:
        return sorted(
            (
                i
                for i in idx
                if segs[i].kind in ("wrong", "deadbox")
                and bool(segs[i].meta.get("distractor")) == distractor
            ),
            reverse=True,
        )

    ref = [i for i in idx if segs[i].kind == "ref"] if plan_reached else []
    return efq + ref + deep + top + wrongs(False) + wrongs(True)


def decision_fallbacks(
    env: Any, action: list[str], level: int
) -> list[tuple[str, list[list[str]]]]:
    """Greedy looks at the names the action cites (each added if the block still fits `level`),
    the action alone, nothing."""
    T = td()
    sc = T.scope(env)
    term = action[parse_have(action)[2] :] if action[0] == "have" else action[1:]
    names = list(dict.fromkeys(t for t in term if T.is_name(t) and t in sc))
    looks = [[n, ":", *ftoks(sc[n])] for n in names]
    out: list[tuple[str, list[list[str]]]] = []
    kept: list[list[str]] = []
    for lk in looks:
        if len(join_lines([*kept, lk])) <= level:
            kept.append(lk)
    if kept:
        out.append(("looks", kept))
    out.append(("ref", [list(action)]))
    out.append(("empty", []))
    return out


def compress(
    env: Any,
    action: list[str],
    lines: list[list[str]],
    segs: Sequence[Seg],
    plan_reached: bool,
    level: int,
) -> tuple[list[list[str]], str]:
    """The full (e256) block `lines` -> (a coherent block of at most `level` body tokens, its
    rung).  A deterministic first-fit ladder of variants, most to least complete, each a list of
    the block's own lines (or such a line + `<no>`); a variant is taken if it fits and passes
    `trace_dsl.check_step` with no finding.  No rng."""
    T = td()
    segs = list(segs)

    def variants() -> Iterable[tuple[str, list[list[str]]]]:
        yield "full", lines
        plan = [lines[s.start] for s in segs if s.kind == "plan"]
        n_plan = len(plan)
        if level >= 64:

            def build(keep: Iterable[int], short: bool) -> list[list[str]]:
                out: list[list[str]] = []
                for i in sorted(keep):
                    ls = lines[segs[i].start : segs[i].end]
                    out += (short_form(segs[i], ls) or ls) if short else ls
                return out

            keep = [i for i, s in enumerate(segs) if s.kind not in ("verify", "viable")]
            yield "nocheck", build(keep, False)
            yield "short", build(keep, True)
            for k, i in enumerate(prune_order(segs, plan_reached), 1):
                keep.remove(i)
                yield f"prune:{k}", build(keep, True)
            for k in range(n_plan - 1, 0, -1):
                yield f"plan_cut:{k}", [*plan[:k], list(action)]
                yield f"plan_cut:{k}", plan[:k]
        else:
            if n_plan:
                yield "plan", plan
            for k in range(n_plan - 1, 0, -1):
                yield f"plan_cut:{k}", plan[:k]
        yield from decision_fallbacks(env, action, level)

    for name, v in variants():
        if len(join_lines(v)) > level or (not v and name != "empty"):
            continue
        if name != "full" and v:
            try:
                soft = T.check_step(env, ["<think>", *join_lines(v), "</think>"], action, {})
            except (T.TraceError, AssertionError, KeyError, ValueError):
                continue
            if soft:
                continue
        return [list(x) for x in v], name.split(":")[0]
    return [], "empty"


def trace_for(
    pid: str,
    prompt: str,
    proof: str,
    cfg: Cfg,
    seed: int = 0,
    epoch: int = 0,
    distractors: dict[int, list[list[str]]] | None = None,
    source: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """-> (trace record, per-step info).  Raises if the reference does not replay."""
    T = td()
    rng = random.Random(proof_seed(seed, epoch, pid))
    steps, _, _ = T.decompose(prompt, proof, canon=True)
    actions = [list(a) for _, a, _ in steps]
    envs, pos = build_tree(prompt, actions)
    out_steps: list[dict[str, str]] = []
    infos: list[dict[str, Any]] = []
    for t, (act, env) in enumerate(zip(actions, envs, strict=True)):
        kind = step_kind(act)
        rec, idx = pos[t]
        made: list[Def] = []
        root = frame_term(rec, idx, {}, made)
        target = next((d for d in made if d.key == t), None)
        path = find_path(root, t, rng) if act[0] == "have" else None
        if path is not None:
            path = collapse(path)
        goal = env.frames[-1].goal
        info: dict[str, Any] = {
            "kind": kind,
            "consistent": not ent(T.scope(env).values(), BOT),
            "path_len": len(path) if path else 0,
        }
        if kind == "decision":
            info["obvious"] = is_obvious(act, goal, path)
            p = cfg.p_think_obvious if info["obvious"] else cfg.p_think_decision
        else:
            p = cfg.p_think_premise if kind == "premise" else cfg.p_think_exact
        lines: list[list[str]] = []
        full_tokens = 0
        if rng.random() < p:
            info["fallback"] = "empty"
            for explore in (True, False):
                ds = (distractors or {}).get(t, [])
                g = StepGen(env, act, kind, path, ds, cfg, rng, target)
                try:
                    lines = g.generate(explore=explore)
                    soft = T.check_step(env, ["<think>", *join_lines(lines), "</think>"], act, {})
                    if soft:
                        raise T.TraceError(f"(strict) {soft[0]}")
                except (AssertionError, T.TraceError, Budget) as e:
                    info.setdefault("gen_errors", []).append(f"{type(e).__name__}: {e}"[:160])
                    lines = []
                    continue
                info.update(g.info)
                info["fallback"] = None if explore else "plan_only"
                if lines:
                    full_tokens = len(join_lines(lines))
                    info["rung"] = "full"
                    if cfg.level == 0:
                        lines, info["rung"] = [], "e0"
                    elif cfg.level < 256:
                        lines, info["rung"] = compress(
                            env, act, lines, g.segs, bool(g.info.get("plan_reached")), cfg.level
                        )
                break
        info["think_tokens_e256"] = full_tokens
        info["lines"] = len(lines)
        info["think_tokens"] = len(join_lines(lines))
        info["state_tokens"] = len(env.state_tokens())
        info["action_tokens"] = len(act)
        names = [int(x[1:]) for ln in lines for x in ln if T.is_name(x) and x[0] == "n"]
        info["max_name"] = max(names, default=0)
        out_steps.append(
            {"think": "\n".join(ascii_line(x) for x in lines), "action": ascii_line(act)}
        )
        infos.append(info)
    rec_out: dict[str, Any] = {
        "id": pid,
        "source": source,
        "mode": "programmatic",
        "prompt": prompt,
        "steps": out_steps,
        "gen": {"format": f"progtrace/{cfg.profile[1:]}", "seed": seed, "epoch": epoch},
    }
    if cfg.level != 256:
        rec_out["gen"]["level"] = cfg.level
    return rec_out, infos


# ---------------------------------------------------------------------------------- batch run
def load_distractors(pattern: str = DISTRACTOR_GLOB) -> dict[str, dict[int, list[list[str]]]]:
    """proof_id -> step -> the base model's rejected actions (token lists), from the gen runs'
    `distractors.jsonl`.  A row whose `ref_action` does not match is the generator's problem, not
    ours: it is still keyed by step and `trace_for` re-checks every candidate against the env."""
    T = td()
    out: dict[str, dict[int, list[list[str]]]] = collections.defaultdict(dict)
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                ds = [T.tokens(d["action"]) for d in r.get("distractors", [])]
                if ds:
                    out[r["proof_id"]][int(r["step"])] = ds
    return dict(out)


_W: dict[str, Any] = {}


def _init(cfg: Cfg, seed: int, epoch: int, distractors: dict[str, Any], source: str) -> None:
    _W.update(cfg=cfg, seed=seed, epoch=epoch, ds=distractors, source=source, T=td())


def _one(job: tuple[str, str, str, int]) -> tuple[str, dict[str, Any] | None, list[Any], str]:
    pid, prompt, proof, n_lines = job
    T = _W["T"]
    try:
        rec, infos = trace_for(
            pid, prompt, proof, _W["cfg"], _W["seed"], _W["epoch"], _W["ds"].get(pid), _W["source"]
        )
    except Exception as e:  # the reference does not replay, or a generator bug
        return pid, None, [], f"generate: {type(e).__name__}: {str(e)[:120]}"
    v = T.validate(rec, lean=False, strict=True)
    if not v["ok"]:
        return pid, None, infos, f"validate: {str(v['error'])[:160]}"
    rec["n_lines"] = n_lines
    return pid, rec, infos, ""


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (vx * vy), 4) if vx and vy else None


def _dist(xs: Sequence[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 2),
        "p50": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)],
        "max": s[-1],
    }


class Stats:
    def __init__(self) -> None:
        self.c: collections.Counter[str] = collections.Counter()
        self.dropped: collections.Counter[str] = collections.Counter()
        self.dropped_ids: list[dict[str, str]] = []
        self.fallback: collections.Counter[str] = collections.Counter()
        self.think_step: dict[str, list[int]] = collections.defaultdict(list)
        self.think_proof_by_lines: dict[int, list[int]] = collections.defaultdict(list)
        self.alt_pos: collections.Counter[str] = collections.Counter()
        self.pair_max = 0
        self.rungs: collections.Counter[str] = collections.Counter()
        self.full_decision: list[int] = []
        self.max_think = 0
        self.proof_x: list[float] = []
        self.proof_y: list[float] = []

    def add(self, pid: str, rec: dict[str, Any] | None, infos: list[Any], err: str, n: int) -> None:
        if rec is None:
            self.c["dropped"] += 1
            self.dropped[err.split(":")[0] + ": " + err.split(":", 2)[-1].strip()[:60]] += 1
            if len(self.dropped_ids) < 200:
                self.dropped_ids.append({"id": pid, "error": err})
            return
        self.c["traces"] += 1
        tot = 0
        for inf in infos:
            k = inf["kind"]
            self.c[f"steps_{k}"] += 1
            self.fallback[f"{k}:{inf.get('fallback', 'nothink')}"] += 1
            tt_ = inf["think_tokens"]
            self.max_think = max(self.max_think, tt_)
            self.c["other_segments"] += inf.get("other_segments", 0)
            if inf.get("think_tokens_e256"):
                self.rungs[f"{k}:{inf.get('rung', 'full')}"] += 1
            if k == "decision":
                self.full_decision.append(inf.get("think_tokens_e256", 0))
            tot += tt_
            self.think_step[k].append(tt_)
            if tt_ == 0:
                self.c[f"empty_{k}"] += 1
            if k == "decision" and tt_:
                self.c["decision_with_think"] += 1
                self.c["decision_dead_end"] += bool(inf.get("dead_ends"))
                self.c["decision_backtrack"] += bool(inf.get("backtrack"))
                self.c["decision_rejected_cand"] += bool(inf.get("rejected"))
                self.c["decision_lookahead_dead"] += bool(inf.get("lookahead_dead"))
                self.c["decision_leaf_cands"] += bool(inf.get("leaf_cands"))
                for a, b in inf.get("alt_pos", []):
                    self.alt_pos[f"{a}/{b}"] += 1
            body = tt_ + 2 if tt_ else 2  # <think> ... </think>; empty blocks still carry the tags
            self.pair_max = max(
                self.pair_max, inf["state_tokens"] + body + inf["action_tokens"] + 1
            )
        self.think_proof_by_lines[n].append(tot)
        self.proof_x.append(n)
        self.proof_y.append(tot)

    def report(self, wall: float) -> dict[str, Any]:
        c = self.c
        nd = max(c["decision_with_think"], 1)
        return {
            "counts": dict(c),
            "dropped": dict(self.dropped.most_common(40)),
            "dropped_examples": self.dropped_ids,
            "fallback": dict(self.fallback),
            "think_tokens_per_step": {k: _dist(v) for k, v in self.think_step.items()},
            "think_tokens_per_proof": _dist(self.proof_y),
            "think_tokens_per_proof_by_lines": {
                str(k): _dist(v) for k, v in sorted(self.think_proof_by_lines.items())
            },
            "empty_fraction": {
                k: round(c[f"empty_{k}"] / max(c[f"steps_{k}"], 1), 4)
                for k in ("decision", "premise", "exact")
            },
            "decision_fractions": {
                k: round(c[f"decision_{k}"] / nd, 4)
                for k in ("dead_end", "backtrack", "rejected_cand", "lookahead_dead", "leaf_cands")
            },
            "reference_position_after_alternatives": dict(self.alt_pos.most_common()),
            "max_pair_tokens": self.pair_max,
            "rungs": dict(self.rungs.most_common()),
            "other_segments": c["other_segments"],
            "think_tokens_e256_per_step": _dist(self.full_decision),
            "max_think_tokens": self.max_think,
            "spearman_think_vs_lines": _spearman(self.proof_x, self.proof_y),
            "wall_s": round(wall, 1),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default="K12", help=f"{list(SOURCES)} or a K12-style JSONL path")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0: all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--think-budget", type=int, default=Cfg.think_budget)
    ap.add_argument("--max-binders", type=int, default=Cfg.max_binders)
    ap.add_argument("--distractors", default=DISTRACTOR_GLOB, help="glob; '' for none")
    ap.add_argument("--no-exclude", action="store_true", help="keep rows in eval classes")
    ap.add_argument(
        "--level", type=int, choices=LEVELS, default=256, help="effort cap on think tokens"
    )
    ap.add_argument("--profile", choices=["v1", "v2"], default="v2", help="v1: as audited")
    a = ap.parse_args(argv)
    t0 = time.time()
    src: str = SOURCES.get(a.source) or a.source
    mk = Cfg.v1 if a.profile == "v1" else Cfg
    cfg = mk(think_budget=a.think_budget, max_binders=a.max_binders, level=a.level)
    ds = load_distractors(a.distractors) if a.distractors else {}
    index = None
    if not a.no_exclude:
        from .data import DEFAULT_EXCLUDE, EvalIndex

        index = EvalIndex([*DEFAULT_EXCLUDE, TB_TRAIN])
    jobs: list[tuple[str, str, str, int]] = []
    excluded: collections.Counter[str] = collections.Counter()
    with open(src) as f:
        for i, line in enumerate(f):
            if i < a.start:
                continue
            if a.limit and i >= a.start + a.limit:
                break
            r = json.loads(line)
            if index is not None:
                hits = index.hit(r["prompt"])
                if hits:
                    for h in hits:
                        excluded[h] += 1
                    continue
            jobs.append((f"k12_{i}", r["prompt"], r["proof"], int(r.get("n_lines", 0))))
    print(
        f"{len(jobs)} proofs from {src} ({sum(excluded.values())} eval-class hits excluded), "
        f"distractors for {len(ds)} proofs, {a.workers} workers",
        flush=True,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    st = Stats()
    tmp = out.with_suffix(out.suffix + ".tmp")
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with (
        open(tmp, "w") as fo,
        ctx.Pool(a.workers, _init, (cfg, a.seed, a.epoch, ds, a.source)) as pool,
    ):
        for k, (pid, rec, infos, err) in enumerate(pool.imap(_one, jobs, chunksize=64)):
            st.add(pid, rec, infos, err, jobs[k][3])
            if rec is not None:
                fo.write(json.dumps(rec) + "\n")
            if (k + 1) % 10000 == 0:
                print(
                    f"  {k + 1}/{len(jobs)} {st.c['traces']} ok {st.c['dropped']} dropped "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    os.replace(tmp, out)
    rep = st.report(time.time() - t0)
    rep["config"] = {
        "source": src,
        "seed": a.seed,
        "epoch": a.epoch,
        "level": cfg.level,
        "cfg": cfg.__dict__,
        "distractors": a.distractors,
        "excluded_by_set": dict(excluded),
        "format": f"progtrace/{cfg.profile[1:]}",
    }
    sp = out.with_name(out.stem + ".stats.json")
    sp.write_text(json.dumps(rep, indent=1, default=str))
    print(
        f"{st.c['traces']} traces, {st.c['dropped']} dropped -> {out}; "
        f"think/proof {rep['think_tokens_per_proof']}; rho {rep['spearman_think_vs_lines']}; "
        f"max pair {rep['max_pair_tokens']}; {rep['wall_s']}s; stats {sp}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
