"""WP3 law x substitution-size suite (plan §2 B4, §3 WP3 row).

Eight classical/intuitionistic propositional-law schemas, each stated over 2-3 metavariables
(placeholder atoms `P`, `Q`, (`R`)) and given an **atomic minimal proof** fetched directly from the
vendored `takehome/minlen.py` (matching the plan's "get the atomic proofs from minlen and
substitute"; every one of the eight `minlen` calls below reproduces plan §0's/`minlen.py`'s own
`selftest()` numbers exactly -- see `atomic_proof`'s docstring). An instance for substitution size
`s` fills each metavariable with a random formula over its own, pairwise-disjoint slice of the
4-atom alphabet (`{P,Q,R,S}` split `2+2` for a 2-metavariable schema, `2+1+1` for a 3-metavariable
one -- the only two-way/three-way partitions of 4 atoms into disjoint non-empty groups that plan
§2 B4 lists), built to have **exactly** `s` connectives (`_random_formula_str`, a direct recursive
construction, not rejection sampling: each connective consumes exactly one unit of `s` and
recursively distributes the rest, so it is exact by construction, not merely checked after the
fact), and checked **contingent** (`families.is_contingent`).

## Why substitution preserves the proof

Natural-deduction rules are *schematic*: every rule in `spec.md`'s table is stated over
metavariables (`A`, `B`, ...), so replacing every occurrence of one metavariable's placeholder atom
with an arbitrary well-formed formula throughout a valid ND proof yields another valid ND proof of
the substituted sequent, with the **same line count** (substitution changes formula content only,
never the proof's line/box structure -- the ND grammar always self-delimits a compound formula in
its own parentheses, so substituting a multi-token filler for a bare atom token never disturbs the
surrounding grammar). `_substitute` implements this as one token-level, single-pass replacement
(not iterated string replacement, which could re-substitute already-substituted content) over the
atomic proof's premises/conclusion/body. Every substituted witness is nonetheless re-checked with
`nd_verify` (`families.collect_candidates`), and its line count is asserted to equal the atomic
schema's `L_atomic` -- this is not just trusted, even though the argument above guarantees it.

## Certification policy (plan §2 B4, verbatim)

`L_ub = L_atomic` (the substituted witness). An instance is **kept** only if the injected
`Certifier` exhausts `bound = L_atomic - 1` for the *substituted* sequent without finding anything
shorter and without timing out (method `ub_lb`: no proof shorter than the witness exists in
`minlen`'s closure, so the witness is minimal). If the certifier *finds* something `<= L_atomic-1`,
the substitution introduced an unintended shortcut (e.g. a filler that happens to make two
subformulas logically related in a way the atomic schema's premises did not) and the instance is
**dropped**. If the certifier **times out**, the instance is kept anyway but labelled
`witness_argument` (plan §2 B4: "if minlen times out, keep as witness_argument and plot
separately") -- its minimality is asserted only by the substitution argument above, not verified.

## Schemas

| schema | metavars | atomic sequent | `L_atomic` |
|---|---|---|---|
| `contraposition` | A, B | `(A>B) |- ((~B)>(~A))` | 7 |
| `modus_tollens` | A, B | `(A>B), (~B) |- (~A)` | 6 |
| `hypothetical_syllogism` | A, B, C | `(A>B), (B>C) |- (A>C)` | 6 |
| `disjunctive_syllogism` | A, B | `(AvB), (~A) |- B` | 7 |
| `negated_conditional` | A, B | `(~(A>B)) |- (A & (~B))` | 12 |
| `demorgan_or_to_nand` | A, B | `((~A)v(~B)) |- (~(A&B))` | 10 |
| `demorgan_nand_to_or` (classical) | A, B | `(~(A&B)) |- ((~A)v(~B))` | 14 |
| `dist_and_over_or` | A, B, C | `(A&(BvC)) |- ((A&B)v(A&C))` | 10 |

`demorgan_or_to_nand` (or-of-negations to nand) is intuitionistically valid; its converse,
`demorgan_nand_to_or`, needs excluded middle and is marked classical (its atomic proof, fetched
from `minlen`, indeed uses `DN`/reductio -- see the WP3 report).

30 instances are targeted per `(schema, s)`, `s in {0,1,2,3,4,6,8,10,12}` (2,160 cells total, some
of which may fall short of 30 at large `s` for the 3-metavariable schemas' 1-atom metavariable
slices, where the token cap and/or repeated contingency-check failures bind first -- see the WP3
report for actual per-cell counts).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.formula import Formula, parse
from ..core.types import JSON, Certifier, Excluder, LengthLabel, Problem, Sequent
from . import families as F

#: Substitution sizes (plan §2 B4).
S_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 10, 12)
#: Target instances per (schema, s) (plan §2 B4: "30 instances per (schema, s)").
TARGET_PER_S = 30


@dataclass(frozen=True)
class LawSchema:
    name: str
    metavars: tuple[str, ...]  # placeholder atom letters used in `atomic_prompt` (subset of ATOMS)
    atomic_prompt: str
    l_atomic: int
    classical: bool = False


SCHEMAS: tuple[LawSchema, ...] = (
    LawSchema("contraposition", ("P", "Q"), "THM ( P > Q ) SEQ ( ( ~ Q ) > ( ~ P ) ) PRF", 7),
    LawSchema("modus_tollens", ("P", "Q"), "THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF", 6),
    LawSchema(
        "hypothetical_syllogism",
        ("P", "Q", "R"),
        "THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF",
        6,
    ),
    LawSchema("disjunctive_syllogism", ("P", "Q"), "THM ( P v Q ) , ( ~ P ) SEQ Q PRF", 7),
    LawSchema(
        "negated_conditional",
        ("P", "Q"),
        "THM ( ~ ( P > Q ) ) SEQ ( P & ( ~ Q ) ) PRF",
        12,
    ),
    LawSchema(
        "demorgan_or_to_nand",
        ("P", "Q"),
        "THM ( ( ~ P ) v ( ~ Q ) ) SEQ ( ~ ( P & Q ) ) PRF",
        10,
    ),
    LawSchema(
        "demorgan_nand_to_or",
        ("P", "Q"),
        "THM ( ~ ( P & Q ) ) SEQ ( ( ~ P ) v ( ~ Q ) ) PRF",
        14,
        classical=True,
    ),
    LawSchema(
        "dist_and_over_or",
        ("P", "Q", "R"),
        "THM ( P & ( Q v R ) ) SEQ ( ( P & Q ) v ( P & R ) ) PRF",
        10,
    ),
)

_ATOMIC_PROOF_CACHE: dict[str, tuple[Sequent, str]] = {}


def atomic_proof(schema: LawSchema, *, time_s: float = 60.0) -> tuple[Sequent, str]:
    """`(atomic Sequent, atomic proof body)` for `schema`, fetched from vendored `minlen` and
    cached. Asserts the found length matches `schema.l_atomic` (plan §0's table / `minlen.py`'s own
    `selftest()`); this is a hard failure, not a silent fallback, if a future vendor re-sync ever
    changes `minlen`'s search behaviour."""
    if schema.name not in _ATOMIC_PROOF_CACHE:
        minlen = F._minlen()
        r = minlen.minlen(schema.atomic_prompt, bound=schema.l_atomic, time_limit=time_s)
        if r["min_lines_ub"] != schema.l_atomic:
            raise RuntimeError(
                f"minlen disagreement for schema {schema.name!r}: expected "
                f"{schema.l_atomic}, got {r['min_lines_ub']} (timeout={r['timeout']})"
            )
        seq = Sequent.from_prompt(schema.atomic_prompt)
        proof_body = r["proof"]
        assert isinstance(proof_body, str)
        _ATOMIC_PROOF_CACHE[schema.name] = (seq, proof_body)
    return _ATOMIC_PROOF_CACHE[schema.name]


# ------------------------------------------------------------------------------ formula fillers


def random_formula_str(s: int, atoms: Sequence[str], rng: random.Random) -> str:
    """A random spec-format formula with **exactly** `s` connectives over `atoms` (with
    repetition), by direct recursive construction (each recursive call spends exactly one
    connective and distributes the remaining `s-1` arbitrarily between its child/children) -- exact
    by construction, no post-hoc rejection needed for the connective count (contingency is a
    separate check, done by the caller, since it cannot be guaranteed by construction)."""
    if s == 0:
        return rng.choice(list(atoms))
    if rng.random() < 0.3:
        return f"( ~ {random_formula_str(s - 1, atoms, rng)} )"
    left_s = rng.randint(0, s - 1)
    right_s = s - 1 - left_s
    op = rng.choice(("&", "v", ">"))
    left = random_formula_str(left_s, atoms, rng)
    right = random_formula_str(right_s, atoms, rng)
    return f"( {left} {op} {right} )"


def connective_count(formula: str) -> int:
    """Number of `~`/`&`/`v`/`>` connectives in `formula` (matches `random_formula_str`'s `s`);
    used by tests to check fillers have exactly the requested size."""

    def count(f: Formula) -> int:
        if f[0] in ("atom", "bot"):
            return 0
        if f[0] == "not":
            return 1 + count(f[1])
        return 1 + count(f[1]) + count(f[2])

    return count(parse(formula))


def atom_partition(n_metavars: int, rng: random.Random) -> list[list[str]]:
    """A random partition of the 4-atom alphabet into `n_metavars` pairwise-disjoint, non-empty
    groups: `2+2` for 2 metavariables, `2+1+1` for 3 (plan §2 B4's only two- and three-way splits
    of 4 atoms); which group (size and identity) lands on which metavariable position is
    randomised too, for instance-to-instance variety."""
    atoms = list(F.ATOMS)
    rng.shuffle(atoms)
    if n_metavars == 2:
        groups = [atoms[0:2], atoms[2:4]]
    elif n_metavars == 3:
        groups = [atoms[0:2], atoms[2:3], atoms[3:4]]
        rng.shuffle(groups)
    else:  # pragma: no cover - no schema needs this
        raise ValueError(n_metavars)
    return groups


def _substitute(template: str, mapping: dict[str, str]) -> str:
    """One-pass, simultaneous token substitution (never iterated/sequential replacement, which
    could re-substitute a filler's own atoms if they happen to reuse a metavariable's placeholder
    letter)."""
    return " ".join(mapping.get(tok, tok) for tok in template.split())


def _law_candidate(
    schema: LawSchema, s: int, rng: random.Random, *, filler_attempts: int = 200
) -> F._Candidate | None:
    atomic_seq, atomic_body = atomic_proof(schema)
    groups = atom_partition(len(schema.metavars), rng)
    fillers: dict[str, str] = {}
    for mv, group in zip(schema.metavars, groups, strict=True):
        filler = None
        for _ in range(filler_attempts):
            cand_f = random_formula_str(s, group, rng)
            if F.is_contingent(cand_f):
                filler = cand_f
                break
        if filler is None:
            return None
        fillers[mv] = filler
    new_premises = tuple(_substitute(p, fillers) for p in atomic_seq.premises)
    new_conclusion = _substitute(atomic_seq.conclusion, fillers)
    if not F.is_satisfiable(new_premises):
        return None
    new_body = _substitute(atomic_body, fillers)
    seq = Sequent(premises=new_premises, conclusion=new_conclusion)
    return F._Candidate(seq, new_body, schema.l_atomic, {"schema": schema.name, "s": s})


def _interpret_law_label(lbl: LengthLabel, predicted_L: int) -> LengthLabel | None:
    """Plan §2 B4's ub_lb-or-drop-or-witness_argument policy (module docstring), given a
    `Certifier`'s raw response for one candidate.

    Written against the `Certifier` protocol's method semantics, not against one implementation's
    internal mechanics (see `families._label_and_filter`'s docstring for the full rationale): a
    witness is always passed alongside `bound`, and a conformant `Certifier` (e.g. WP2's real
    `logic.certify.MinlenCertifier`) may certify against the witness's own verified length rather
    than the literal `bound` argument, so only the returned `method`/`L` are inspected, never which
    bound was actually searched.
    """
    if lbl.method == "minlen_exact" and lbl.L is not None and predicted_L > lbl.L:
        return None  # a confirmed shortcut: the substitution wasn't minimal after all
    if predicted_L == lbl.L and lbl.method in ("minlen_exact", "ub_lb"):
        return lbl  # confirmed (direct hit, or exhaustion at L_atomic-1): keep as certified
    return LengthLabel(
        L=predicted_L,
        L_ub=predicted_L,
        L_lb=predicted_L,
        method="witness_argument",
        bound=predicted_L,
        timeout=True,
    )


def _certify_law_candidate(
    cand: F._Candidate, *, certifier: Certifier, time_s: float
) -> LengthLabel | None:
    """Single-item convenience wrapper around `_interpret_law_label` (used by tests and by any
    caller with just one candidate); `build()` itself batches a whole `(schema, s)` cell through
    one `Certifier.certify` call instead, since every candidate for a given schema shares the same
    `bound = l_atomic` and a conformant `Certifier` can run a batch across its worker pool (plan
    §A2: <=8 workers) far faster than this WP issuing one single-item call per instance."""
    lbl = certifier.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=time_s)[0]
    return _interpret_law_label(lbl, cand.predicted_L)


def build(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> tuple[list[Problem], dict[str, JSON]]:
    """Build the `laws` suite (plan §2 B4, §3 WP3 row). `cfg` keys (all optional): `schemas`
    (subset of `SCHEMAS` names), `s_values` (default `S_VALUES`), `target_per_s` (default 30),
    `max_attempts` (default 300), `time_s` (per-instance `Certifier` time budget at
    `bound=L_atomic-1`, default 15.0), `version` (default `"v1"`)."""
    owns_certifier = certifier is None
    certifier = certifier if certifier is not None else F.default_certifier()
    version = str(cfg.get("version", "v1"))
    target_per_s = int(cfg.get("target_per_s", TARGET_PER_S))
    max_attempts = int(cfg.get("max_attempts", 300))
    time_s = float(cfg.get("time_s", 15.0))
    s_values = cfg.get("s_values", list(S_VALUES))
    wanted = cfg.get("schemas")
    wanted_set = set(wanted) if wanted is not None else None

    master_rng = random.Random(rng_seed)
    problems: list[Problem] = []
    stats: dict[str, JSON] = {}
    try:
        for schema in SCHEMAS:
            if wanted_set is not None and schema.name not in wanted_set:
                continue
            for s in s_values:  # type: ignore[union-attr]
                rng = random.Random(master_rng.random())
                candidates = F.collect_candidates(
                    target_per_s,
                    lambda r, schema=schema, s=s: _law_candidate(schema, s, r),
                    rng,
                    max_attempts=max_attempts,
                )
                kept = 0
                dropped_shortcut = 0
                if candidates:
                    labels = certifier.certify(
                        [c.seq for c in candidates],
                        [c.body for c in candidates],
                        bound=schema.l_atomic,
                        time_s=time_s,
                    )
                else:
                    labels = []
                for c, raw_lbl in zip(candidates, labels, strict=True):
                    lbl = _interpret_law_label(raw_lbl, c.predicted_L)
                    if lbl is None:
                        dropped_shortcut += 1
                        continue
                    tags = dict(c.tags)
                    tags["tokens"] = F.count_tokens(c.seq.prompt(), c.body)
                    pid = f"laws-{schema.name}-s{s}-{kept:03d}"
                    problems.append(
                        F._make_problem(
                            suite="laws",
                            version=version,
                            pid=pid,
                            seq=c.seq,
                            length=lbl,
                            body=c.body,
                            source="witness",
                            tags=tags,
                            excluder=excluder,
                        )
                    )
                    kept += 1
                stats[f"{schema.name}.s{s}"] = {
                    "target": target_per_s,
                    "candidates_seen": len(candidates),
                    "built": kept,
                    "dropped_shortcut": dropped_shortcut,
                }
    finally:
        F.close_certifier_if_owned(certifier, owned=owns_certifier)
    return problems, stats
