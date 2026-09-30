"""Skill tags for a `Sequent` or proof body (plan §2 B3, §3's WP2 row).

`classical` and `needs_ore` are exact, cheap (no bounded search) tests on the theorem alone --
empirically under 0.2 ms/call even at L12-14 (measured against `ab2/sealed/sealed_L12-14.jsonl`),
so a suite builder can tag every item with both unconditionally. `necessity` and `needs_depth` are
*bounded* tests: each runs up to two full `minlen` searches (unrestricted, then restricted), and at
the theorem lengths this benchmark cares about (L>=12) a single such search often does not resolve
within a couple of seconds even before doubling it (measured: 8/10 sampled `sealed_L12-14` items
timed out at `time_s=2.0`, `bound=L_true`) -- `minlen`'s default `lemmas=True` search is genuinely
expensive at these lengths. Suite builders should treat these two as available-on-demand for a
deliberately-sized subset or a downstream analysis pass, not as something to run unconditionally
over a multi-thousand-item pool; see `suites/existing.py`'s `skill_pool` docstring for the concrete
scope decision this module's cost profile drives.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ..core import formula
from ..core.proof import split_lines
from ..core.types import Sequent
from ..vendor import ensure_takehome_on_path

Formula = formula.Formula

#: `necessity`/`needs_depth`'s three-state result. `"unknown"` covers every case the underlying
#: bounded search cannot resolve: a search timeout, or the unrestricted search itself failing to
#: reach any proof within `bound` (which -- per `minlen.py`'s own docstring -- is never proof that
#: none exists, only that this restricted search space didn't find one), or (for `necessity` only)
#: a pattern with no defined restriction (`impe_chain4`, `ori_ore` -- plan §2: "uses only", "no
#: restriction exists" per the vendored `necessity.py`'s own `FORBID` map).
Necessity = Literal["requires", "not_required", "unknown"]


def classical(seq: Sequent) -> bool:
    """True iff `seq` is *not* intuitionistically provable (vendored G4ip `intuit.py`): every
    proof of a classical-only sequent must use double-negation elimination somewhere. Exact --
    `intuit_provable` is a sound+complete decision procedure for this fragment."""
    ensure_takehome_on_path()
    import intuit as _intuit  # type: ignore[import-not-found]

    return not bool(_intuit.intuit_provable(seq.prompt()))


# --------------------------------------------------------------------------------- needs_ore


def _subformulas(f: Formula, acc: set[Formula]) -> None:
    acc.add(f)
    if f[0] == "not":
        _subformulas(f[1], acc)
    elif f[0] in ("and", "or", "imp"):
        _subformulas(f[1], acc)
        _subformulas(f[2], acc)


def _replace_disjunctions(f: Formula, mapping: Mapping[Formula, Formula]) -> Formula:
    """Recursively replace every disjunction subformula of `f` (at any depth) by its mapped
    atom, leaving every other connective in place. After this, `f` contains no `'or'` node at
    all -- including inside what were `X`/`Y` of a replaced disjunction, since axiom construction
    below applies this same transform to them too."""
    if f[0] == "or":
        return mapping[f]
    if f[0] == "not":
        return ("not", _replace_disjunctions(f[1], mapping))
    if f[0] in ("and", "imp"):
        return (f[0], _replace_disjunctions(f[1], mapping), _replace_disjunctions(f[2], mapping))
    return f  # ('atom', name) | ('bot',)


def _evaluate(f: Formula, env: Mapping[str, bool]) -> bool:
    if f[0] == "atom":
        return env[f[1]]
    if f[0] == "bot":
        return False
    if f[0] == "not":
        return not _evaluate(f[1], env)
    if f[0] == "and":
        return _evaluate(f[1], env) and _evaluate(f[2], env)
    if f[0] == "or":
        return _evaluate(f[1], env) or _evaluate(f[2], env)
    if f[0] == "imp":
        return (not _evaluate(f[1], env)) or _evaluate(f[2], env)
    raise ValueError(f"not a formula: {f!r}")


def needs_ore(seq: Sequent) -> bool:
    """True iff every ND proof of `seq` needs ORE (plan §2 B3's `case_split` test).

    Replace each distinct disjunction subformula `D = (X v Y)` occurring anywhere in the premises
    or conclusion by a fresh atom `p_D`, and add axioms `X* -> p_D`, `Y* -> p_D` (starred =
    recursively transformed the same way, so nested disjunctions get their own atoms too). `seq`
    has an ORE-free proof iff the transformed premises plus every axiom classically entail the
    transformed conclusion (plan §2: sound because ORI alone realises each axiom, complete because
    the `{not,and,imp}` fragment with DN is classically complete and each axiom is realised by a
    trivial `{AS X, ORI}` box). `needs_ore` is the negation: true iff that entailment *fails* for
    some truth assignment, i.e. no ORE-free proof can exist.

    Exhaustive truth table over every atom used (original P/Q/R/S plus one synthetic atom per
    distinct disjunction subformula) -- at most 2**16 rows per plan §2's stated bound, and in
    practice far fewer for this benchmark's formula sizes.
    """
    prem_f = [formula.parse(p) for p in seq.premises]
    concl_f = formula.parse(seq.conclusion)

    subs: set[Formula] = set()
    for f in (*prem_f, concl_f):
        _subformulas(f, subs)
    disjunctions = sorted((f for f in subs if f[0] == "or"), key=repr)
    mapping: dict[Formula, Formula] = {
        d: ("atom", f"__ore{i}__") for i, d in enumerate(disjunctions)
    }

    gamma_star = [_replace_disjunctions(p, mapping) for p in prem_f]
    concl_star = _replace_disjunctions(concl_f, mapping)
    axioms: list[Formula] = []
    for d in disjunctions:
        x, y = d[1], d[2]
        p_d = mapping[d]
        axioms.append(("imp", _replace_disjunctions(x, mapping), p_d))
        axioms.append(("imp", _replace_disjunctions(y, mapping), p_d))

    constraints = (*gamma_star, *axioms)
    atoms_used: set[str] = set()
    for f in (*constraints, concl_star):
        atoms_used |= formula.atoms(f)
    atoms_list = sorted(atoms_used)

    for bits in range(1 << len(atoms_list)):
        env = {a: bool((bits >> i) & 1) for i, a in enumerate(atoms_list)}
        if all(_evaluate(f, env) for f in constraints) and not _evaluate(concl_star, env):
            return True  # a model of the premises+axioms that falsifies the conclusion
    return False


# ------------------------------------------------------------------------- necessity / depth


def _load_necessity_maps() -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_takehome_on_path()
    import necessity as _necessity  # type: ignore[import-not-found]

    return _necessity.FORBID, _necessity.MAXDEPTH


def necessity(seq: Sequent, pattern: str, bound: int, time_s: float) -> Necessity:
    """Does every minimal-length-ish proof of `seq` (within `bound`) need `pattern`?

    Two bounded `minlen` searches (vendored `necessity.py`'s `FORBID`/`MAXDEPTH` maps): an
    unrestricted one, and one with `pattern`'s rule restriction (`--forbid`/`--max_depth`)
    applied. `"requires"` iff the unrestricted search finds a proof within `bound` AND the
    restricted search exhausts `bound` without a timeout (so no pattern-free proof exists in the
    search space); `"not_required"` iff the restricted search also finds a proof; `"unknown"`
    whenever either search times out, the unrestricted search cannot reach any proof at all within
    `bound` (not itself proof of absence, per `minlen.py`'s own docstring), or `pattern` has no
    defined restriction at all (`impe_chain4`, `ori_ore`: plan §2 calls these "uses only").

    Known patterns (vendored `necessity.FORBID`'s keys): `reductio`, `derived_ore`,
    `derived_ore_strict`, `impi_ore`, `negi_ande_hyp`, `nested_ore`, `depth4`, `impe_chain4`,
    `ori_ore`.
    """
    ensure_takehome_on_path()
    import minlen as _minlen  # type: ignore[import-not-found]

    forbid_map, maxdepth_map = _load_necessity_maps()
    if pattern not in forbid_map:
        raise ValueError(f"unknown necessity pattern {pattern!r}; known: {sorted(forbid_map)}")

    prompt = seq.prompt()
    u = _minlen.minlen(prompt, bound=bound, time_limit=time_s)
    if u["min_lines_ub"] is None:
        return "unknown"

    forbid = forbid_map[pattern]
    if forbid is None:
        return "unknown"  # "uses only": no restricted search is defined for this pattern

    r = _minlen.minlen(
        prompt,
        bound=bound,
        time_limit=time_s,
        forbid=forbid,
        max_depth=maxdepth_map.get(pattern),
    )
    if r["timeout"]:
        return "unknown"
    return "requires" if r["min_lines_ub"] is None else "not_required"


def needs_depth(seq: Sequent, max_depth: int, bound: int, time_s: float = 60.0) -> Necessity:
    """Does every proof of `seq` within `bound` need a box nesting depth greater than `max_depth`
    (plan §2 B3's `depth>=3` skill, called with `max_depth` one less than the skill's threshold)?

    Same shape as `necessity`, restricting box depth (`minlen`'s own `--max_depth`) instead of
    forbidding a rule pattern: `"requires"` iff an unrestricted proof exists within `bound` but the
    depth-restricted search exhausts `bound` without a timeout; `"not_required"` iff a
    depth-restricted proof is also found; `"unknown"` on any timeout or on total unreachability.

    `time_s` is a keyword-only addition beyond the plan's `needs_depth(seq, max_depth, bound)`
    listing (needed to actually bound the two searches); it defaults to the `Certifier` Protocol's
    own default (60.0) so positional calls matching the plan's signature keep working.
    """
    ensure_takehome_on_path()
    import minlen as _minlen  # type: ignore[import-not-found]

    prompt = seq.prompt()
    u = _minlen.minlen(prompt, bound=bound, time_limit=time_s)
    if u["min_lines_ub"] is None:
        return "unknown"

    r = _minlen.minlen(prompt, bound=bound, time_limit=time_s, max_depth=max_depth)
    if r["timeout"]:
        return "unknown"
    return "requires" if r["min_lines_ub"] is None else "not_required"


# ------------------------------------------------------------------------------- rule_uses


def rule_uses(body: str) -> dict[str, int]:
    """Rule-name histogram of `body`'s *dependency-pruned* proof (plan §2 B3's descriptive
    `uses_*` skills): `{"IMPE": 2, "ANDI": 1, ...}`. Uses the vendored `patterns.prune_proof`
    (same dependency walk as `core.proof.pruned_length`, but returns the pruned text itself rather
    than just its length); falls back to counting `body` unpruned if it does not parse as a full
    proof (e.g. a truncated sample with no `QED`)."""
    ensure_takehome_on_path()
    import patterns as _patterns  # type: ignore[import-not-found]

    pruned = _patterns.prune_proof(body)
    text = pruned if pruned is not None else body

    counts: dict[str, int] = {}
    for line in split_lines(text):
        toks = line.split()
        try:
            rule = toks[toks.index(":") + 1]
        except (ValueError, IndexError):
            continue
        counts[rule] = counts.get(rule, 0) + 1
    return counts
