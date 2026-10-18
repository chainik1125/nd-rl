#!/usr/bin/env python3
"""Truth-table prover for the project's propositional fragment.

Atoms P, Q, R, S; connectives ``~ & v >``; ``F`` is False. The input is a sequent in the ND prompt
format, e.g. ``THM ( P > Q ) , P SEQ Q PRF``. The output is a Lean 4 (core, no Mathlib) term-mode
proof, or a countermodel if the sequent is not valid.

Method (Kalmar's completeness proof):

1. Truth table: check every valuation of the atoms that occur (at most 2^4 = 16).
2. Case-split on excluded middle (X or not X, derived with ``Classical.byContradiction``) for each
   atom.
3. In each branch every formula has a fixed truth value, so prove it, or its negation, by recursion
   on its structure. A branch where some premise is false closes by ex falso.

Only natural-deduction steps are used: and / or / implies / not introduction and elimination, ex
falso, and reductio (``Classical.byContradiction``). No tactics, no library lemmas. The proofs are
correct but long (term size ~100-400): a decision procedure, not a model of a good prover.

Usage::

    python3 code/tools/tt_prover.py "THM ( P > Q ) , P SEQ Q PRF"
    python3 code/tools/tt_prover.py --check < prompts.txt   # one prompt per line; runs `lean`
"""

import itertools
import os
import subprocess
import sys
import tempfile
from typing import Any

Formula = str | tuple[Any, ...]  # an atom, or (op, *operands) with op in ~ & v > F
ATOMS = ("P", "Q", "R", "S")
OPS = {"&": "∧", "v": "∨", ">": "→"}


def parse(toks: list[str], i: int) -> tuple[Formula, int]:
    t = toks[i]
    if t in ATOMS:
        return t, i + 1
    if t == "F":
        return ("F",), i + 1
    assert t == "(", f"unexpected token {t!r}"
    if toks[i + 1] == "~":
        a, j = parse(toks, i + 2)
        assert toks[j] == ")"
        return ("~", a), j + 1
    a, j = parse(toks, i + 1)
    op = toks[j]
    b, k = parse(toks, j + 1)
    assert op in OPS and toks[k] == ")", f"bad formula near {op!r}"
    return (op, a, b), k + 1


def parse_prompt(prompt: str) -> tuple[list[Formula], Formula]:
    toks = prompt.split()
    assert toks[0] == "THM" and toks[-1] == "PRF" and "SEQ" in toks
    body = toks[1:-1]
    s = body.index("SEQ")
    prem, i = [], 0
    while i < s:
        f, i = parse(body, i)
        prem.append(f)
        if i < s:
            assert body[i] == ","
            i += 1
    goal, j = parse(body, s + 1)
    assert j == len(body)
    return prem, goal


def lean(f: Formula) -> str:
    if isinstance(f, str):
        return f
    if f[0] == "F":
        return "False"
    if f[0] == "~":
        return f"¬{lean(f[1])}"
    return f"({lean(f[1])} {OPS[f[0]]} {lean(f[2])})"


def val(f: Formula, v: dict[str, bool]) -> bool:
    if isinstance(f, str):
        return v[f]
    if f[0] == "F":
        return False
    if f[0] == "~":
        return not val(f[1], v)
    a, b = val(f[1], v), val(f[2], v)
    return {"&": a and b, "v": a or b, ">": (not a) or b}[f[0]]


def prove(f: Formula, v: dict[str, bool], hyp: dict[str, str]) -> str:
    """Kalmar's lemma: a term of type f if f is true under v, else of type ¬f."""
    t = val(f, v)
    typ = lean(f) if t else f"¬{lean(f)}"
    if isinstance(f, str):  # atom: the case-split hypothesis
        body = hyp[f]
    elif f[0] == "F":  # ¬False
        body = "fun h : False => h"
    elif f[0] == "~":  # ¬A from ¬A; ¬¬A from A
        a = prove(f[1], v, hyp)
        body = a if t else f"fun h : {lean(f)} => h {a}"
    else:
        op, x, y = f
        a, b, va, vb = prove(x, v, hyp), prove(y, v, hyp), val(x, v), val(y, v)
        if op == "&":  # ∧I, or ¬I with ∧E
            if t:
                body = f"⟨{a}, {b}⟩"
            elif not va:
                body = f"fun h : {lean(f)} => {a} h.1"
            else:
                body = f"fun h : {lean(f)} => {b} h.2"
        elif op == "v":  # ∨I, or ¬I with ∨E
            if t:
                body = f"Or.inl {a}" if va else f"Or.inr {b}"
            else:
                body = f"fun h : {lean(f)} => Or.elim h {a} {b}"
        elif vb:  # →I, consequent true
            body = f"fun _ : {lean(x)} => {b}"
        elif not va:  # →I with ex falso, antecedent false
            body = f"fun z : {lean(x)} => False.elim ({a} z)"
        else:  # ¬(A → B): ¬I with →E
            body = f"fun h : {lean(f)} => {b} (h {a})"
    return f"({body} : {typ})"


def split(
    atoms: list[str], prem: list[Formula], goal: Formula, v: dict[str, bool], hyp: dict[str, str]
) -> str:
    """Case-split on excluded middle for each atom, then close each branch."""
    if not atoms:
        for i, p in enumerate(prem):  # a false premise: ex falso
            if not val(p, v):
                return f"(False.elim ({prove(p, v, hyp)} h{i + 1}) : {lean(goal)})"
        return prove(goal, v, hyp)
    x, rest = atoms[0], atoms[1:]
    em = (
        f"(Classical.byContradiction (fun h : ¬({x} ∨ ¬{x}) => "
        f"h (Or.inr (fun p : {x} => h (Or.inl p)))) : {x} ∨ ¬{x})"
    )
    yes = split(rest, prem, goal, {**v, x: True}, {**hyp, x: f"h{x}"})
    no = split(rest, prem, goal, {**v, x: False}, {**hyp, x: f"h{x}"})
    return f"Or.elim {em} (fun h{x} : {x} => {yes}) (fun h{x} : ¬{x} => {no})"


def countermodel(prompt: str) -> dict[str, bool] | None:
    """A valuation that makes every premise true and the goal false, if there is one."""
    prem, goal = parse_prompt(prompt)
    atoms = [a for a in ATOMS if a in prompt.split()]
    for bits in itertools.product([True, False], repeat=len(atoms)):
        v = dict(zip(atoms, bits, strict=True))
        if all(val(p, v) for p in prem) and not val(goal, v):
            return v
    return None


def solve(prompt: str) -> str | None:
    """Lean source of a proof of the sequent, or None if it is not valid."""
    if countermodel(prompt) is not None:
        return None
    prem, goal = parse_prompt(prompt)
    atoms = [a for a in ATOMS if a in prompt.split()]
    hyps = "".join(f" (h{i + 1} : {lean(p)})" for i, p in enumerate(prem))
    proof = split(atoms, prem, goal, {}, {})
    return f"theorem t (P Q R S : Prop){hyps} : {lean(goal)} :=\n  {proof}"


def lean_ok(src: str, lean_bin: str = "lean") -> tuple[bool, str]:
    """Run Lean on one theorem; (accepted, Lean's output)."""
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as fh:
        fh.write(src + "\n")
    try:
        out = subprocess.run([lean_bin, fh.name], capture_output=True, text=True)
    finally:
        os.unlink(fh.name)
    msg = (out.stdout + out.stderr).strip()
    return out.returncode == 0 and "error" not in msg and "sorry" not in msg, msg


def main(argv: list[str]) -> None:
    check = "--check" in argv
    prompts = [a for a in argv if a != "--check"] or [ln for ln in sys.stdin if ln.strip()]
    for prompt in prompts:
        src = solve(prompt.strip())
        if src is None:
            print(f"-- not valid; countermodel {countermodel(prompt.strip())}\n")
            continue
        print(src)
        if check:
            ok, msg = lean_ok(src)
            print(f"-- lean: {'accepted' if ok else 'REJECTED: ' + msg}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
