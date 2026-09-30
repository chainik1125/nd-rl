"""WP3 acceptance tests for `suites/laws.py` (plan §3 WP3 row).

Checks: each schema's atomic proof matches plan §0's table (fetched from `minlen`, not
hand-derived); substituted witnesses verify and preserve the atomic line count; fillers have
**exactly** the requested connective count, pairwise-disjoint atom sets per metavariable, and are
individually contingent; the substituted premises are jointly satisfiable; the `ub_lb`/drop/
`witness_argument` certification policy behaves as documented; `build()`'s tags and token cap.

Kept fast (<60s): only small `s` values and small `target_per_s` are exercised through `build()`;
`LocalMinlenCertifier` calls use generous-but-bounded time budgets that complete in well under a
second per plan §0's own findings for these atomic schemas.
"""

from __future__ import annotations

import random

from code.ndbench.suites import families as F
from code.ndbench.suites import laws as L

verify_text = F._verify_text()


def test_atomic_proofs_match_plan_table() -> None:
    for schema in L.SCHEMAS:
        seq, body = L.atomic_proof(schema)
        ok, reason, nl = verify_text(seq.prompt() + " " + body)
        assert ok, (schema.name, reason)
        assert nl == schema.l_atomic


def test_demorgan_nand_to_or_is_flagged_classical() -> None:
    by_name = {s.name: s for s in L.SCHEMAS}
    assert by_name["demorgan_nand_to_or"].classical is True
    assert by_name["demorgan_or_to_nand"].classical is False


def test_random_formula_str_has_exact_connective_count() -> None:
    rng = random.Random(0)
    for s in (0, 1, 2, 3, 4, 6, 8, 10, 12):
        for _ in range(20):
            f = L.random_formula_str(s, ["P", "Q"], rng)
            assert L.connective_count(f) == s


def test_atom_partition_is_disjoint_and_covers_expected_sizes() -> None:
    rng = random.Random(0)
    for _ in range(20):
        groups = L.atom_partition(2, rng)
        sizes = sorted(len(g) for g in groups)
        assert sizes == [2, 2]
        assert set(groups[0]) & set(groups[1]) == set()
        assert set(groups[0]) | set(groups[1]) == set(F.ATOMS)
    for _ in range(20):
        groups = L.atom_partition(3, rng)
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 1, 2]
        all_atoms: list[str] = []
        for g in groups:
            all_atoms.extend(g)
        assert len(all_atoms) == len(set(all_atoms)) == 4


def test_law_candidates_verify_and_preserve_atomic_length() -> None:
    rng = random.Random(1)
    for schema in L.SCHEMAS:
        for s in (0, 1, 3, 6):
            cand = None
            for _ in range(10):
                cand = L._law_candidate(schema, s, rng)
                if cand is not None:
                    break
            assert cand is not None, (schema.name, s)
            ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
            assert ok, (schema.name, s, reason)
            assert nl == cand.predicted_L == schema.l_atomic


def test_law_fillers_are_contingent_and_disjoint() -> None:
    """Re-derives each metavariable's filler from a candidate's substituted premises/conclusion is
    not straightforward (the atomic->substituted mapping isn't kept on `_Candidate`), so this test
    instead exercises the filler-generation step directly, the same way `_law_candidate` does."""
    rng = random.Random(2)
    for schema in L.SCHEMAS:
        groups = L.atom_partition(len(schema.metavars), rng)
        for s in (0, 1, 4, 8):
            for group in groups:
                filler = None
                for _ in range(200):
                    cand_f = L.random_formula_str(s, group, rng)
                    if F.is_contingent(cand_f):
                        filler = cand_f
                        break
                assert filler is not None, (schema.name, s, group)
                assert L.connective_count(filler) == s
                assert F.is_contingent(filler)
                used_atoms = {t for t in filler.split() if t in F.ATOMS}
                assert used_atoms <= set(group)


def test_law_candidate_premises_jointly_satisfiable() -> None:
    rng = random.Random(3)
    for schema in L.SCHEMAS:
        for s in (0, 2, 4):
            cand = None
            for _ in range(10):
                cand = L._law_candidate(schema, s, rng)
                if cand is not None:
                    break
            assert cand is not None
            assert F.is_satisfiable(cand.seq.premises)


def test_certification_policy_ub_lb_or_witness_argument_or_drop() -> None:
    cert = F.LocalMinlenCertifier()
    rng = random.Random(4)
    seen_methods = set()
    for schema in L.SCHEMAS:
        for s in (0, 1):
            cand = L._law_candidate(schema, s, rng)
            assert cand is not None
            lbl = L._certify_law_candidate(cand, certifier=cert, time_s=15.0)
            assert lbl is None or lbl.method in ("minlen_exact", "ub_lb", "witness_argument")
            if lbl is not None:
                assert lbl.L == cand.predicted_L == schema.l_atomic
                seen_methods.add(lbl.method)
    # These schemas are small/cheap (all sub-second per plan §0): every one should be certified,
    # not merely kept on a timeout fallback.
    assert seen_methods & {"minlen_exact", "ub_lb"}
    cert.close()


def test_build_smoke_tags_and_token_cap() -> None:
    cfg = {
        "schemas": ["contraposition", "modus_tollens"],
        "s_values": [0, 1, 2],
        "target_per_s": 5,
        "max_attempts": 100,
        "time_s": 15.0,
    }
    cert = F.LocalMinlenCertifier()
    problems, stats = L.build(cfg, certifier=cert, excluder=None, rng_seed=0)
    cert.close()
    assert problems
    for p in problems:
        assert p.suite == "laws"
        assert set(p.tags) >= {"schema", "s", "tokens"}
        tokens = p.tags["tokens"]
        assert isinstance(tokens, int)
        assert tokens <= F.TOKEN_CAP
        assert p.length.method in ("minlen_exact", "ub_lb", "witness_argument")
        assert p.ref is not None
    for v in stats.values():
        assert isinstance(v, dict)
        built, dropped, seen = v["built"], v["dropped_shortcut"], v["candidates_seen"]
        assert isinstance(built, int) and isinstance(dropped, int) and isinstance(seen, int)
        assert built + dropped <= seen
