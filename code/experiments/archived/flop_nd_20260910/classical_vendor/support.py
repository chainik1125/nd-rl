"""Generic proof nodes, bounded-search base and scoped ND serialization.

Extracted from solve_training_textbook.py; its data-reading main is excluded.
Local adaptation adds only output-size censorship to the serializer.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import itertools
import time
from nd_data import formula_text

BOT = ("bot",)


def neg(formula):
    return ("not", formula)


@dataclass(frozen=True)
class Proof:
    formula: tuple
    rule: str
    args: tuple = ()
    identity: int = 0


@lru_cache(maxsize=100000)
def proof_size(proof):
    seen = set()

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for child in node.args:
            visit(child)

    visit(proof)
    return len(seen)


class BudgetExceeded(Exception):
    pass


class Search:
    def __init__(self, seconds, max_calls, variant=0):
        self.deadline = time.monotonic() + seconds
        self.max_calls = max_calls
        self.calls = 0
        self.identities = itertools.count(1)
        self.failed = set()
        self.variant = variant

    def ref(self, formula, kind="AS"):
        return Proof(formula, kind, identity=next(self.identities))

    def closure(self, context):
        facts = {node.formula: node for node in context}

        def add(formula, rule, children):
            node = Proof(formula, rule, tuple(children))
            if formula not in facts or proof_size(node) < proof_size(facts[formula]):
                facts[formula] = node
                return True
            return False

        changed = True
        while changed:
            changed = False
            for formula, node in list(facts.items()):
                if formula[0] == "and":
                    changed |= add(formula[1], "ANDE1", (node,))
                    changed |= add(formula[2], "ANDE2", (node,))
                if formula[0] == "not" and formula[1][0] == "not":
                    changed |= add(formula[1][1], "DN", (node,))
                if formula[0] == "imp" and formula[1] in facts:
                    changed |= add(formula[2], "IMPE", (node, facts[formula[1]]))
                if neg(formula) in facts:
                    changed |= add(BOT, "NEGE", (node, facts[neg(formula)]))
        return facts

    def prove(self, goal, context, depth, trail=frozenset()):
        self.calls += 1
        if self.calls > self.max_calls or time.monotonic() > self.deadline:
            raise BudgetExceeded
        facts = self.closure(context)
        if goal in facts:
            return facts[goal]
        if BOT in facts:
            return Proof(goal, "BOTE", (facts[BOT],))
        state = (goal, frozenset(facts))
        cachekey = (state, depth)
        if depth <= 0 or state in trail or cachekey in self.failed:
            return None
        trail = trail | {state}
        available = tuple(facts.values())
        ordered_facts = list(facts.items())
        if self.variant & 1:
            ordered_facts.reverse()

        def sub(formula, ctx=available):
            return self.prove(formula, ctx, depth - 1, trail)

        if goal[0] == "and":
            left = sub(goal[1])
            if left:
                right = sub(goal[2], available + (left,))
                if right:
                    return Proof(goal, "ANDI", (left, right))
        elif goal[0] in ("imp", "not"):
            assumption = self.ref(goal[1])
            target = goal[2] if goal[0] == "imp" else BOT
            inner = sub(target, available + (assumption,))
            if inner:
                return Proof(goal, "IMPI" if goal[0] == "imp" else "NEGI", (assumption, inner))
        elif goal[0] == "or":
            alternatives = [(1, "ORI1"), (2, "ORI2")]
            if self.variant & 2:
                alternatives.reverse()
            for side, rule in alternatives:
                inner = sub(goal[side])
                if inner:
                    return Proof(goal, rule, (inner,))

        for formula, implication in ordered_facts:
            if formula[0] == "imp" and formula[2] == goal:
                antecedent = sub(formula[1])
                if antecedent:
                    return Proof(goal, "IMPE", (implication, antecedent))

        if goal == BOT:
            for formula, negative in ordered_facts:
                if formula[0] == "not":
                    positive = sub(formula[1])
                    if positive:
                        return Proof(BOT, "NEGE", (positive, negative))

        for formula, disjunction in ordered_facts:
            if formula[0] == "or":
                left_as, right_as = self.ref(formula[1]), self.ref(formula[2])
                left = sub(goal, available + (left_as,))
                if left:
                    right = sub(goal, available + (right_as,))
                    if right:
                        return Proof(goal, "ORE", (disjunction, left_as, left, right_as, right))

        if goal != BOT and neg(goal) not in facts:
            assumption = self.ref(neg(goal))
            contradiction = sub(BOT, available + (assumption,))
            if contradiction:
                double_negative = Proof(neg(neg(goal)), "NEGI", (assumption, contradiction))
                return Proof(goal, "DN", (double_negative,))
        self.failed.add(cachekey)
        return None


class CertificateOutputLimitExceeded(Exception):
    def __init__(self, unit, limit, attempted):
        self.unit, self.limit, self.attempted = unit, limit, attempted
        super().__init__(f"Certificate exceeds {limit} {unit}; attempted {attempted}")


def serialize(premises, proof, max_lines=10000, max_codec_tokens=1000000):
    lines, cache, hypothesis_lines = [], {}, {}
    codec_tokens = 2  # Sampled FINAL and terminal QED.

    def add(node, context, rule, refs):
        nonlocal codec_tokens
        index = len(lines) + 1
        if index > max_lines:
            raise CertificateOutputLimitExceeded("lines", max_lines, index)
        # Relative codec: LINE, scope bars, formula, colon/rule/semicolon,
        # then REF + unary BACK distance + ENDREF for each citation.
        next_tokens = codec_tokens + 4 + len(context) + len(formula_text(node.formula).split()) + sum(index - r + 2 for r in refs)
        if next_tokens > max_codec_tokens:
            raise CertificateOutputLimitExceeded("relative_codec_response_tokens", max_codec_tokens, next_tokens)
        codec_tokens = next_tokens
        lines.append({"index": index, "depth": len(context), "context": context, "formula": node.formula, "rule": rule, "refs": refs})
        cache.setdefault(node, []).append((context, index))
        return index

    for premise in premises:
        hypothesis_lines[premise.identity] = add(premise, (), "PR", [])

    def in_scope(saved, context):
        return saved == context[:len(saved)]

    def emit(node, context):
        for saved, index in reversed(cache.get(node, [])):
            if in_scope(saved, context):
                return index
        if node.rule in ("PR", "AS"):
            return hypothesis_lines[node.identity]
        if node.rule in ("IMPI", "NEGI"):
            assumption, inner = node.args
            child_context = context + (assumption.identity,)
            start = add(assumption, child_context, "AS", [])
            hypothesis_lines[assumption.identity] = start
            end = emit(inner, child_context)
            if lines[end - 1]["context"] != child_context or end != len(lines):
                end = add(inner, child_context, "R", [end])
            return add(node, context, node.rule, [start, end])
        if node.rule == "ORE":
            disj, left_as, left, right_as, right = node.args
            disj_line = emit(disj, context)
            refs = [disj_line]
            for assumption, branch in ((left_as, left), (right_as, right)):
                branch_context = context + (assumption.identity,)
                start = add(assumption, branch_context, "AS", [])
                hypothesis_lines[assumption.identity] = start
                end = emit(branch, branch_context)
                if lines[end - 1]["context"] != branch_context or end != len(lines):
                    end = add(branch, branch_context, "R", [end])
                refs += [start, end]
            return add(node, context, "ORE", refs)
        refs = [emit(child, context) for child in node.args]
        return add(node, context, node.rule, refs)

    final = emit(proof, ())
    if final != len(lines):
        add(proof, (), "R", [final])
    body = " ".join(" ".join([f"N{row['index']}", *("|" for _ in row["context"]), formula_text(row["formula"]), ":", row["rule"], *(f"N{ref}" for ref in row["refs"]), ";"]) for row in lines) + " QED"
    return body
