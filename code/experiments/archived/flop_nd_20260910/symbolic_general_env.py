"""Finite theorem-derived ND action proposer, independent of strict checkers.

The proposer implements Fitch contexts and box endpoints itself. It does not
call a verifier to enumerate or filter actions. The immutable checkers are used
only to audit proposals and to accept complete emitted certificates.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math

import common
from nd_data import formula_text
from nd_environment import parse_prompt, strict_complete
from nd_verify import parse_proof_tokens

BOT = ("bot",)
RULES = ("AS", "R", "ANDE1", "ANDE2", "ANDI", "IMPE", "IMPI", "ORI1", "ORI2", "ORE", "NEGE", "NEGI", "BOTE", "DN")
OPS = ("atom", "bot", "not", "and", "or", "imp")


@lru_cache(maxsize=100000)
def subformulas(f):
    out = {f}
    for x in f[1:]:
        if isinstance(x, tuple):
            out.update(subformulas(x))
    return frozenset(out)


@lru_cache(maxsize=100000)
def formula_size(f):
    return 1 + sum(formula_size(x) for x in f[1:] if isinstance(x, tuple))


@dataclass(frozen=True)
class Action:
    rule: str
    formula: tuple
    refs: tuple[int, ...]
    depth: int


@dataclass(frozen=True)
class Line:
    formula: tuple
    rule: str
    refs: tuple[int, ...]
    context: tuple[int, ...]


@dataclass
class Box:
    start: int
    end: int
    parent: tuple[int, ...]
    depth: int


def context_prefix(a, b):
    return a == b[:len(a)]


def render_line(i, line):
    return " ".join([f"N{i}"] + ["|"] * len(line.context) + [formula_text(line.formula), ":", line.rule] +
                    [f"N{r}" for r in line.refs] + [";"])


class Environment:
    """Bounded proof state with exact line and box visibility.

    Base formula set is all subformulas of theorem premises and conclusion plus
    bottom. AS, ANDI, ORI1/2 and BOTE results are restricted to this base set.
    IMPI may additionally form A>B for any base A and base B. These temporary
    implications can be eliminated, but are not added to the assumption pool.
    Other elimination rules derive their syntactically determined result.
    NEGI output must be base; all rule outputs belong to base or the finite
    one-level implication closure. No reference certificate informs this pool.

    Each scope depth from zero through current depth is a possible destination.
    AS can open at any depth 1..current+1 up to max_scope. Action refs retain all
    legal citation alternatives; there is no result-formula deduplication.
    Repeated inferences are bounded by the externally specified line budget.
    """
    def __init__(self, prompt, max_scope=4, max_lines=24):
        self.prompt = prompt
        self.premises, self.goal = parse_prompt(prompt)
        self.base = frozenset({BOT}.union(subformulas(self.goal), *(subformulas(f) for f in self.premises)))
        self.base_order = tuple(sorted(self.base, key=formula_text))
        self.max_scope, self.max_lines = max_scope, max_lines
        self.lines, self.stack, self.boxes = [], (), {}
        self.generated = 0
        for formula in self.premises:
            self._append(Action("PR", formula, (), 0))

    def formula_allowed(self, f):
        return f in self.base or (f[0] == "imp" and f[1] in self.base and f[2] in self.base)

    def _append(self, action):
        i = len(self.lines) + 1
        if action.rule == "AS":
            parent = self.stack[:action.depth - 1]
            context = parent + (i,)
            self.boxes[i] = Box(i, i, parent, action.depth)
        else:
            context = self.stack[:action.depth]
        self.stack = context
        self.lines.append(Line(action.formula, action.rule, action.refs, context))
        for start in context:
            self.boxes[start].end = i

    def visible(self, destination):
        return [(i, line) for i, line in enumerate(self.lines, 1) if context_prefix(line.context, destination)]

    def citable_boxes(self, destination):
        return [b for b in self.boxes.values()
                if b.start not in destination and context_prefix(b.parent, destination)
                and len(self.lines[b.end - 1].context) == b.depth]

    def actions(self):
        if len(self.lines) >= self.max_lines or self.success():
            return []
        actions = set()
        def offer(rule, formula, refs, depth):
            if self.formula_allowed(formula):
                actions.add(Action(rule, formula, tuple(refs), depth))

        for depth in range(1, min(len(self.stack) + 1, self.max_scope) + 1):
            for formula in self.base_order:
                offer("AS", formula, (), depth)

        for depth in range(len(self.stack) + 1):
            destination = self.stack[:depth]
            visible = self.visible(destination)
            by_formula = {}
            for i, line in visible:
                by_formula.setdefault(line.formula, []).append(i)
            for i, line in visible:
                f = line.formula
                offer("R", f, (i,), depth)
                if f[0] == "and":
                    offer("ANDE1", f[1], (i,), depth)
                    offer("ANDE2", f[2], (i,), depth)
                if f[0] == "imp":
                    for j in by_formula.get(f[1], ()):
                        offer("IMPE", f[2], (i, j), depth)
                if f[0] == "not":
                    for j in by_formula.get(f[1], ()):
                        offer("NEGE", BOT, (j, i), depth)
                    if f[1][0] == "not":
                        offer("DN", f[1][1], (i,), depth)
                if f == BOT:
                    for g in self.base_order:
                        offer("BOTE", g, (i,), depth)
            # Introduction targets come from theorem subformulas, never refs.
            for g in self.base_order:
                if g[0] == "and":
                    for i, j in product(by_formula.get(g[1], ()), by_formula.get(g[2], ())):
                        offer("ANDI", g, (i, j), depth)
                elif g[0] == "or":
                    for i in by_formula.get(g[1], ()):
                        offer("ORI1", g, (i,), depth)
                    for i in by_formula.get(g[2], ()):
                        offer("ORI2", g, (i,), depth)
            boxes = self.citable_boxes(destination)
            cases = {}
            for box in boxes:
                a, g = self.lines[box.start - 1].formula, self.lines[box.end - 1].formula
                offer("IMPI", ("imp", a, g), (box.start, box.end), depth)
                if g == BOT:
                    negation = ("not", a)
                    if negation in self.base:
                        offer("NEGI", negation, (box.start, box.end), depth)
                cases.setdefault((a, g), []).append(box)
            for i, line in visible:
                f = line.formula
                if f[0] != "or":
                    continue
                for (a, g), lefts in cases.items():
                    if a != f[1]:
                        continue
                    for left, right in product(lefts, cases.get((f[2], g), ())):
                        offer("ORE", g, (i, left.start, left.end, right.start, right.end), depth)
        # Canonical order depends only on legal state, never on a source proof.
        return sorted(actions, key=lambda a: (RULES.index(a.rule), a.depth, formula_text(a.formula), a.refs))

    def step(self, action, check_membership=True):
        if check_membership and action not in self.actions():
            raise ValueError("Action is outside the finite legal-action support")
        self._append(action)
        self.generated += 1
        success = self.success()
        if success:
            verdict = strict_complete(self.prompt, self.body())
            if not verdict["ok"]:
                raise AssertionError(f"Proposer emitted invalid terminal certificate: {verdict}")
        done = success or len(self.lines) >= self.max_lines
        return float(success), done

    def success(self):
        return bool(self.lines and not self.stack and self.lines[-1].formula == self.goal)

    def body(self):
        return " ".join(render_line(i, line) for i, line in enumerate(self.lines, 1)) + " QED"

    def prefix(self):
        return " ".join(render_line(i, line) for i, line in enumerate(self.lines, 1))

    def pruned_body(self):
        """Retain original premises, citations, and all cited scope assumptions."""
        keep = set()
        def visit(i):
            if i in keep:
                return
            keep.add(i)
            line = self.lines[i - 1]
            for parent in line.refs + line.context:
                if parent != i:
                    visit(parent)
        for i in range(1, len(self.premises) + 1):
            visit(i)
        visit(len(self.lines))
        ordered = sorted(keep)
        renumber = {old: new + 1 for new, old in enumerate(ordered)}
        out = []
        for old in ordered:
            line = self.lines[old - 1]
            pruned = Line(line.formula, line.rule, tuple(renumber[r] for r in line.refs),
                          tuple(renumber[r] for r in line.context))
            out.append(render_line(renumber[old], pruned))
        return " ".join(out) + " QED"

    def source_actions(self, row):
        parsed = parse_proof_tokens(row["body"].split())
        if [line["formula"] for line in parsed if line["rule"] == "PR"] != self.premises:
            raise ValueError("Source premise block mismatch")
        return [Action(line["rule"], line["formula"], tuple(line["refs"]), line["depth"])
                for line in parsed if line["rule"] != "PR"]

    def replay_source(self, row, generated_cut):
        actions = self.source_actions(row)
        if not 0 <= generated_cut < len(actions):
            raise ValueError("Reset must leave at least one source action")
        for a in actions[:generated_cut]:
            self.step(a)
        self.generated = 0
        return actions
