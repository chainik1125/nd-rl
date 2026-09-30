"""WP3 acceptance tests for `suites/families.py` (plan §3 WP3 row).

Checks, per the plan's acceptance criteria:
* for `n <= 5`, every family's witnesses verify with the predicted `L` (all families), and
  `LocalMinlenCertifier` agrees with plan §0's table wherever that family's "exact" tier reaches
  that `n` (some families -- `reorder_and`/`reorder_or` -- have a lower exact threshold than 5, per
  §0's own timeout findings, so this test only asserts minlen-agreement up to each family's own
  `exact_max_n`, capped at 5);
* distinct pkeys per `(family, n)`;
* the token cap is respected;
* a few structural properties of `build()`'s output (tags, `ReferenceProof`, suite/version).

Kept CPU-only and fast: `LocalMinlenCertifier` is only exercised at the small `n` the "exact" tier
actually needs (all sub-second per plan §0's own timing), and `build()` smoke tests use small
`target_per_n`/tight `n_grid` overrides for the CPU-heavier families (`and_tree`, `reorder_and`,
`reorder_or`) rather than the full plan-scale grids used by the real suite build.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from code.ndbench.suites import families as F

verify_text = F._verify_text()


def _stat(v: Any) -> dict[str, Any]:
    """Narrow a `build()` `stats` value (typed `JSON` on the dataclass) to the `dict` it always
    actually is, for tests that inspect its fields."""
    assert isinstance(v, dict)
    return v


# --------------------------------------------------------------------- witnesses verify (n <= 5)


_CLEAN_FAMILIES = [
    ("and_tower", F._and_tower_candidate),
    ("or_tower", F._or_tower_candidate),
    ("imp_nest", F._imp_nest_candidate),
    ("and_extract", F._and_extract_candidate),
    ("dn_elim", F._dn_elim_candidate),
    ("dn_intro", F._dn_intro_candidate),
]


def test_clean_family_witnesses_verify_for_n_le_5() -> None:
    rng = random.Random(0)
    for _name, make in _CLEAN_FAMILIES:
        for n in range(1, 6):
            cand = make(n, rng)
            ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
            assert ok, (cand.tags, reason)
            assert nl == cand.predicted_L


def test_and_tree_witness_verifies_for_n_le_5() -> None:
    rng = random.Random(0)
    for n in range(4, 6):  # and_tree needs >= 2 leaves; plan range starts at 4
        cand = F._and_tree_candidate(n, rng)
        assert cand is not None
        ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
        assert ok, reason
        assert nl == cand.predicted_L


def test_reorder_family_witnesses_verify_for_n_le_5() -> None:
    rng = random.Random(0)
    for n in range(2, 6):
        atoms_only = n <= len(F.ATOMS)
        cand = None
        for _ in range(20):
            cand = F._reorder_and_candidate(n, rng, atoms_only=atoms_only)
            if cand is not None:
                break
        assert cand is not None
        ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
        assert ok, reason
        assert nl == cand.predicted_L == 3 * n - 2
    for n in range(2, 5):
        cand = F._reorder_or_candidate(n, rng, atoms_only=True)
        assert cand is not None
        ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
        assert ok, reason
        assert nl == cand.predicted_L


# -------------------------------------------------------------- minlen agrees (exact-tier n<=5)


def test_minlen_agrees_with_plan_table_for_exact_tier() -> None:
    """Cross-checks `LocalMinlenCertifier` against plan §0's table for every family whose "exact"
    tier reaches at least `n=1`, up to `min(exact_max_n, 5)`."""
    cert = F.LocalMinlenCertifier()
    rng = random.Random(1)
    # `and_tower`'s L = n + |Gamma| (Gamma's size is randomised per instance, plan §2 B2); the
    # others have a fixed premise count, so L(n) is a plain function of n (module docstring).
    expected_L = {
        "or_tower": lambda cand: cand.predicted_L == cand.tags["n"] + 1,
        "imp_nest": lambda cand: cand.predicted_L == 2 * cand.tags["n"] + 1,
        "and_extract": lambda cand: cand.predicted_L == cand.tags["n"] + 1,
        "dn_elim": lambda cand: cand.predicted_L == cand.tags["n"] + 1,
        "dn_intro": lambda cand: cand.predicted_L == cand.tags["n"] + 3,
    }
    for name, exact_max_n in [
        ("and_tower", 8),
        ("or_tower", 8),
        ("imp_nest", 6),
        ("and_extract", 8),
        ("dn_elim", 6),
        ("dn_intro", 6),
    ]:
        make = dict(_CLEAN_FAMILIES)[name]
        for n in range(1, min(exact_max_n, 5) + 1):
            cand = make(n, rng)
            if name == "and_tower":
                assert cand.predicted_L == n + len(cand.seq.premises)
            else:
                assert expected_L[name](cand)
            lbl = cert.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=20.0)[0]
            assert lbl.method == "minlen_exact", (name, n, lbl)
            assert cand.predicted_L == lbl.L, (name, n, lbl)
    cert.close()


def test_minlen_agrees_for_and_tree_and_reorder_small_n() -> None:
    cert = F.LocalMinlenCertifier()
    rng = random.Random(2)
    for n in (4, 5):
        cand = F._and_tree_candidate(n, rng)
        assert cand is not None
        lbl = cert.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=20.0)[0]
        assert lbl.method == "minlen_exact" and cand.predicted_L == lbl.L
    for n in (2, 3, 4):
        cand = F._reorder_and_candidate(n, rng, atoms_only=True)
        assert cand is not None
        lbl = cert.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=20.0)[0]
        assert lbl.method == "minlen_exact" and cand.predicted_L == lbl.L
    for n in (2, 3):
        cand = F._reorder_or_candidate(n, rng, atoms_only=True)
        assert cand is not None
        lbl = cert.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=20.0)[0]
        assert lbl.method == "minlen_exact" and cand.predicted_L == lbl.L
    cert.close()


def test_dn_intro_classical_shortcut_length() -> None:
    """`dn_intro`'s classical shortcut gives `L = n + 3`, not the naive `3n + 1` (module
    docstring); minlen (which searches the full rule set) must agree, not just our construction."""
    cert = F.LocalMinlenCertifier()
    rng = random.Random(3)
    for n in (1, 2, 3, 4):
        cand = F._dn_intro_candidate(n, rng)
        assert cand.predicted_L == n + 3
        assert cand.tags["classical_shortcut"] is True
        lbl = cert.certify([cand.seq], [cand.body], bound=n + 3, time_s=15.0)[0]
        assert n + 3 == lbl.L, "minlen should find the classical shortcut, not just 3n+1"
    cert.close()


# ------------------------------------------------------------------------------ ILTP-style


def test_iltp_style_witnesses_verify() -> None:
    rng = random.Random(4)
    cand = F._syj204_style_candidate(3, rng)
    ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
    assert ok, reason
    assert nl == cand.predicted_L == 3 * 3 + 1

    for n in (1, 2):
        cand = F._syj203_style_candidate(n, rng)
        assert cand is not None
        ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
        assert ok, reason
        assert nl == cand.predicted_L == (8 if n == 1 else 15)

    cand = F._iff_chain_style_candidate(2, rng)
    assert cand is not None
    ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
    assert ok, reason
    assert nl == cand.predicted_L

    cand = F._de_bruijn_style_candidate(1, rng)
    assert cand is not None
    ok, reason, nl = verify_text(cand.seq.prompt() + " " + cand.body)
    assert ok, reason
    assert nl == cand.predicted_L == 11


def test_syj204_style_matches_minlen() -> None:
    cert = F.LocalMinlenCertifier()
    rng = random.Random(5)
    for n in (1, 2, 3):
        cand = F._syj204_style_candidate(n, rng)
        lbl = cert.certify([cand.seq], [cand.body], bound=cand.predicted_L, time_s=20.0)[0]
        assert lbl.L == cand.predicted_L == 3 * n + 1
    cert.close()


# --------------------------------------------------------------------------------- token cap


def test_token_cap_is_respected_by_count_tokens() -> None:
    rng = random.Random(6)
    cand = F._and_extract_candidate(19, rng)
    tokens = F.count_tokens(cand.seq.prompt(), cand.body)
    assert tokens <= F.TOKEN_CAP


def test_collect_candidates_drops_over_cap_items() -> None:
    """A synthetic candidate exceeding `TOKEN_CAP` must never survive `collect_candidates`. Uses a
    single, deeply-nested premise (huge token count, only 2 lines -- well under `LINE_CAP` and the
    tokenizer's 64-line reference range) reiterated as the conclusion."""
    cur = "P"
    for _ in range(400):
        cur = f"( {cur} & P )"
    lb = F.LineBuilder()
    prem = lb.add(0, cur, "PR")
    lb.add(0, cur, "R", [prem])
    seq = F.Sequent(premises=(cur,), conclusion=cur)
    body = lb.render()
    assert F.count_tokens(seq.prompt(), body) > F.TOKEN_CAP

    def make_long(rng: random.Random) -> F._Candidate:
        del rng
        return F._Candidate(seq, body, len(lb), {"family": "synthetic_long", "n": 400})

    out = F.collect_candidates(1, make_long, random.Random(0), max_attempts=3)
    assert out == []


# ------------------------------------------------------------------------------- build() smoke


def test_build_smoke_distinct_pkeys_and_tags() -> None:
    cfg = {
        "families": ["and_tower", "dn_elim"],
        "n_grid": {"and_tower": [1, 2, 3], "dn_elim": [1, 2, 3]},
        "target_per_n": 8,
        "max_attempts": 200,
        "exact_time_s": 10.0,
    }
    cert = F.LocalMinlenCertifier()
    problems, stats = F.build(cfg, certifier=cert, excluder=None, rng_seed=0)
    cert.close()
    assert problems, "expected at least one Problem"
    assert all(_stat(v)["method"] == "minlen_exact" for v in stats.values())

    by_family_n: dict[tuple[str, int], list[F.Sequent]] = {}
    for p in problems:
        assert p.suite == "families"
        assert p.ref is not None
        assert set(p.tags) >= {"family", "n", "lines", "tokens"}
        tokens = p.tags["tokens"]
        assert isinstance(tokens, int)
        assert tokens <= F.TOKEN_CAP
        key = (str(p.tags["family"]), int(p.tags["n"]))  # type: ignore[arg-type]
        by_family_n.setdefault(key, []).append(p.sequent)

    for (_fam, _n), seqs in by_family_n.items():
        pkeys = [F.pkey_or_key(s) for s in seqs]
        assert len(pkeys) == len(set(pkeys)), "pkeys must be pairwise distinct within (family, n)"


def test_build_reorder_families_small_scale() -> None:
    """`reorder_and`/`reorder_or`'s `ub_lb` tier, exercised at a tiny scale (n_grid restricted to
    values within each family's own exact threshold) so the test stays well under budget."""
    cfg = {
        "families": ["reorder_and", "reorder_or"],
        "n_grid": {"reorder_and": [2, 3], "reorder_or": [2, 3]},
        "target_per_n": 4,
        "max_attempts": 100,
        "exact_time_s": 10.0,
    }
    cert = F.LocalMinlenCertifier()
    problems, stats = F.build(cfg, certifier=cert, excluder=None, rng_seed=0)
    cert.close()
    assert problems
    assert all(_stat(v)["method"] == "minlen_exact" for v in stats.values())
    for p in problems:
        assert p.length.method == "minlen_exact"
        assert p.tags["lines"] == p.length.L


def test_build_ub_lb_tier_never_keeps_a_shortcut() -> None:
    """A one-off, tightly-bounded check of the `ub_lb` tier itself: any accepted item either has
    `L == predicted` (verified no-shorter-proof) or `timeout=True` (honestly unverified) -- never
    a silently-wrong `L`."""
    cfg = {
        "families": ["reorder_and"],
        "n_grid": {"reorder_and": [5]},
        "target_per_n": 3,
        "max_attempts": 60,
        "ub_time_s": 4.0,
    }
    cert = F.LocalMinlenCertifier()
    problems, _stats = F.build(cfg, certifier=cert, excluder=None, rng_seed=0)
    cert.close()
    for p in problems:
        assert p.length.method == "ub_lb"
        if p.length.timeout:
            assert p.length.L is None
        else:
            assert p.tags["lines"] == p.length.L


# ---------------------------------------------------------- exact-tier inconclusive fallback


class _InconclusiveStubCertifier:
    """A `Certifier` that always reports "inconclusive" (WP2's `"reference"` method, `L=None`,
    `timeout=False`) regardless of input, to deterministically exercise
    `_label_and_filter`'s `has_argument` fallback without waiting on a real `minlen` timeout."""

    def certify(
        self,
        seqs: Sequence[F.Sequent],
        witnesses: Sequence[str | None],
        *,
        bound: int | None = None,
        time_s: float = 60.0,
    ) -> list[F.LengthLabel]:
        del witnesses, bound, time_s
        return [
            F.LengthLabel(L=None, L_ub=None, L_lb=None, method="reference", bound=None)
            for _ in seqs
        ]


def test_exact_tier_inconclusive_is_kept_as_witness_argument_when_family_has_one() -> None:
    """Main-agent instruction (2026-09-30): imp_nest (and the other five families with a robust
    lower-bound argument) must not drop an inconclusive exact-tier result -- it should fall back to
    `witness_argument` with `timeout=True`, never `L=None`."""
    rng = random.Random(0)
    cand = F._imp_nest_candidate(3, rng)
    stub = _InconclusiveStubCertifier()
    out = F._label_and_filter(
        [cand],
        exact=True,
        tier_above="witness_argument",
        certifier=stub,  # type: ignore[arg-type]
        exact_time_s=1.0,
        ub_time_s=1.0,
        has_argument=True,
    )
    assert len(out) == 1
    _c, lbl = out[0]
    assert lbl.method == "witness_argument"
    assert lbl.timeout is True
    assert cand.predicted_L == lbl.L


def test_exact_tier_inconclusive_is_dropped_when_family_has_no_argument() -> None:
    """`dn_intro` and `reorder_and`/`reorder_or` keep the old drop-on-inconclusive behaviour in
    their exact tier (main-agent instruction, 2026-09-30)."""
    rng = random.Random(0)
    cand = F._dn_intro_candidate(3, rng)
    stub = _InconclusiveStubCertifier()
    out = F._label_and_filter(
        [cand],
        exact=True,
        tier_above="witness_argument",
        certifier=stub,  # type: ignore[arg-type]
        exact_time_s=1.0,
        ub_time_s=1.0,
        has_argument=False,
    )
    assert out == []


def test_family_specs_has_argument_matches_main_agent_instruction() -> None:
    expected_true = {"and_tower", "and_tree", "or_tower", "imp_nest", "and_extract", "dn_elim"}
    expected_false = {"dn_intro", "reorder_and", "reorder_or"}
    by_name = {spec.name: spec for spec in F.FAMILY_SPECS}
    assert {n for n, s in by_name.items() if s.has_argument} == expected_true
    assert {n for n, s in by_name.items() if not s.has_argument} == expected_false
