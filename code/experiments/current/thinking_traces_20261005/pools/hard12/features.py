"""Shape and logic features of a theorem and its reference proof (the hard12 pool's difficulty
analysis).

    from code.experiments.current.thinking_traces_20261005.pools.hard12.features import features
    f = features(prompt, proof)          # dict of scalar features

Pure CPU, no model. The fork is imported read-only (`nd_verify.parse_formula`,
`intuit.intuit_provable`; no bytecode is written into it). Features:

  * length / size: `n_lines` (reference lines, premises included), `n_prem`, `prompt_tokens`,
    `fdepth` (deepest formula nesting in the prompt, connectives counted, `~` included),
    `goal_depth`, `n_atoms` (distinct atoms), `box_depth` (deepest `|` nesting in the reference);
  * rule mix of the reference: `n_<RULE>` counts and `has_<RULE>` flags for every rule; `raa` (DN
    applied to a NEGI conclusion: reductio), `n_boxes` (AS lines);
  * logic: `classical_only` (not intuitionistically provable, fork `intuit.py` G4ip: every proof
    needs DN), `n_redundant_prem` (premises p with Gamma minus p |= C classically: the reference
    cites p, but it is not needed), `prem_inconsistent` (no valuation satisfies the premises),
    `n_unused_prem` (premises the reference never cites: always 0 for generator output, `Gen.finish`
    prunes them);
  * premises: `n_disj_prem` (`v` connectives anywhere in the premises), `n_prem_or` (premises whose
    main connective is `v`), `n_prem_imp`, `n_prem_neg`, `n_imp_hard_ante` (premise implications
    whose antecedent is neither a premise nor a conjunct of one);
  * goal: `goal_type` (atom / not / and / or / imp);
  * direction: `top_rule` (the rule of the conclusion line), `top_goal_directed` (the conclusion is
    the goal's introduction rule: IMPI for `>`, NEGI for `~`, ANDI for `&`, ORI for `v`),
    `first_kind` (the first written line after the premises: `as_goal` assumes the goal's antecedent
    / negand, `as_case` assumes a disjunct of a premise disjunction (a case split), `as_raa` assumes
    the goal's negation, `as_other` any other assumption, `elim` a forward step from the premises),
    `first_goal_directed` (`first_kind == 'as_goal'`).
"""

from __future__ import annotations

import itertools
import sys
from typing import Any

FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
RULES = [
    "AS",
    "R",
    "ANDI",
    "ANDE1",
    "ANDE2",
    "ORI1",
    "ORI2",
    "ORE",
    "IMPI",
    "IMPE",
    "NEGI",
    "NEGE",
    "DN",
    "BOTE",
]
ATOMS = ["P", "Q", "R", "S"]
_fork: dict[str, Any] = {}


def fork() -> dict[str, Any]:
    if not _fork:
        sys.dont_write_bytecode = True
        if FORK not in sys.path:
            sys.path.insert(0, FORK)
        from intuit import intuit_provable  # type: ignore[import-not-found]
        from nd_verify.verify import parse_formula  # type: ignore[import-not-found]

        _fork.update(parse_formula=parse_formula, intuit_provable=intuit_provable)
    return _fork


def parse_prompt(prompt: str) -> tuple[list[Any], Any]:
    pf = fork()["parse_formula"]
    toks = prompt.split()
    i = 1
    prem = []
    if toks[i] != "SEQ":
        while True:
            f, i = pf(toks, i)
            prem.append(f)
            if toks[i] == ",":
                i += 1
                continue
            break
    concl, _ = pf(toks, i + 1)
    return prem, concl


def parse_proof(proof: str) -> list[dict[str, Any]]:
    """Reference lines: {n, depth, f, rule, refs}."""
    pf = fork()["parse_formula"]
    out = []
    for seg in proof.split(" ; "):
        toks = seg.split()
        if not toks or toks[0] == "QED":
            continue
        n = int(toks[0][1:])
        i = 1
        d = 0
        while toks[i] == "|":
            d += 1
            i += 1
        f, i = pf(toks, i)
        assert toks[i] == ":", seg
        rule = toks[i + 1]
        refs = [int(t[1:]) for t in toks[i + 2 :] if t.startswith("N")]
        out.append({"n": n, "depth": d, "f": f, "rule": rule, "refs": refs})
    return out


def ev(f: Any, v: dict[str, bool]) -> bool:
    t = f[0]
    if t == "atom":
        return v[f[1]]
    if t == "bot":
        return False
    if t == "not":
        return not ev(f[1], v)
    if t == "and":
        return ev(f[1], v) and ev(f[2], v)
    if t == "or":
        return ev(f[1], v) or ev(f[2], v)
    return (not ev(f[1], v)) or ev(f[2], v)


VALS = [dict(zip(ATOMS, bits, strict=True)) for bits in itertools.product([False, True], repeat=4)]


def entails(prem: list[Any], c: Any) -> bool:
    return all(ev(c, v) for v in VALS if all(ev(p, v) for p in prem))


def fdepth(f: Any) -> int:
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + fdepth(f[1])
    return 1 + max(fdepth(f[1]), fdepth(f[2]))


def atoms(f: Any, acc: set[str]) -> set[str]:
    if f[0] == "atom":
        acc.add(f[1])
    elif f[0] != "bot":
        for g in f[1:]:
            atoms(g, acc)
    return acc


def count(f: Any, op: str) -> int:
    if f[0] in ("atom", "bot"):
        return 0
    return (f[0] == op) + sum(count(g, op) for g in f[1:])


INTRO = {"imp": "IMPI", "not": "NEGI", "and": "ANDI", "or": ("ORI1", "ORI2")}


def conjuncts(f: Any) -> list[Any]:
    return [f] + (conjuncts(f[1]) + conjuncts(f[2]) if f[0] == "and" else [])


def features(prompt: str, proof: str) -> dict[str, Any]:
    prem, goal = parse_prompt(prompt)
    lines = parse_proof(proof)
    by_n = {ln["n"]: ln for ln in lines}
    body = [ln for ln in lines if ln["rule"] != "PR"]
    out: dict[str, Any] = {
        "n_lines": len(lines),
        "n_prem": len(prem),
        "prompt_tokens": len(prompt.split()) - 3,
        "fdepth": max(fdepth(f) for f in [*prem, goal]),
        "goal_depth": fdepth(goal),
        "n_atoms": len(set().union(*(atoms(f, set()) for f in [*prem, goal]))),
        "box_depth": max(ln["depth"] for ln in lines),
    }
    for r in RULES:
        c = sum(ln["rule"] == r for ln in body)
        out[f"n_{r}"] = c
        out[f"has_{r}"] = int(c > 0)
    out["has_ORI"] = int(out["has_ORI1"] or out["has_ORI2"])
    out["has_ANDE"] = int(out["has_ANDE1"] or out["has_ANDE2"])
    out["n_boxes"] = out["n_AS"]
    out["raa"] = int(
        any(
            ln["rule"] == "DN" and by_n[ln["refs"][0]]["rule"] == "NEGI"
            for ln in body
            if ln["refs"]
        )
    )
    out["classical_only"] = int(not fork()["intuit_provable"](prompt))
    cited = {r for ln in body for r in ln["refs"]}
    n_pr = [ln["n"] for ln in lines if ln["rule"] == "PR"]
    out["n_unused_prem"] = sum(n not in cited for n in n_pr)
    out["n_redundant_prem"] = sum(entails(prem[:i] + prem[i + 1 :], goal) for i in range(len(prem)))
    out["prem_inconsistent"] = int(
        bool(prem) and not any(all(ev(p, v) for p in prem) for v in VALS)
    )
    out["n_disj_prem"] = sum(count(p, "or") for p in prem)
    out["n_prem_or"] = sum(p[0] == "or" for p in prem)
    out["n_prem_imp"] = sum(p[0] == "imp" for p in prem)
    out["n_prem_neg"] = sum(p[0] == "not" for p in prem)
    avail = {c for p in prem for c in conjuncts(p)}
    out["n_imp_hard_ante"] = sum(p[0] == "imp" and p[1] not in avail for p in prem)
    out["goal_type"] = goal[0]
    top = lines[-1]["rule"]
    out["top_rule"] = top
    intro = INTRO.get(goal[0])
    out["top_goal_directed"] = int(
        intro is not None and (top in intro if isinstance(intro, tuple) else top == intro)
    )
    first = body[0]
    if first["rule"] == "AS":
        h = first["f"]
        disj = {x for p in prem if p[0] == "or" for x in (p[1], p[2])}
        if goal[0] in ("imp", "not") and h == goal[1]:
            kind = "as_goal"
        elif h in disj:
            kind = "as_case"
        elif h == ("not", goal):
            kind = "as_raa"
        else:
            kind = "as_other"
    else:
        kind = "elim"
    out["first_kind"] = kind
    out["first_goal_directed"] = int(kind == "as_goal")
    return out


NUMERIC_EXCLUDE = {"goal_type", "top_rule", "first_kind"}
