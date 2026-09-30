"""`logic/skills.py` (plan §2 B3, §3 WP2 row): `classical`, `needs_ore`, `necessity`, `needs_depth`,
`rule_uses`."""

from __future__ import annotations

import itertools

import pytest

from code.ndbench.core.keys import split_thm
from code.ndbench.core.types import Sequent
from code.ndbench.logic.skills import classical, necessity, needs_depth, needs_ore, rule_uses

from ._fixtures import load_valid_proofs

# ------------------------------------------------------------------------------------ classical


def test_classical_peirce_true_contraposition_false() -> None:
    peirce = Sequent.from_prompt("THM ( ( P > Q ) > P ) SEQ P PRF")
    assert classical(peirce) is True

    contraposition = Sequent.from_prompt("THM ( P > Q ) SEQ ( ( ~ Q ) > ( ~ P ) ) PRF")
    assert classical(contraposition) is False


@pytest.mark.parametrize(
    ("prompt", "expected_classical"),
    [
        ("THM ( P > Q ) , P SEQ Q PRF", False),
        ("THM ( ~ ( ~ P ) ) SEQ P PRF", True),  # DN elimination: classical-only
        ("THM P SEQ ( ~ ( ~ P ) ) PRF", False),  # DN introduction: intuitionistically valid
        ("THM SEQ ( P v ( ~ P ) ) PRF", True),  # excluded middle
        ("THM ( P v Q ) , ( ~ P ) SEQ Q PRF", False),
    ],
)
def test_classical_matches_intuit_selftest_cases(prompt: str, expected_classical: bool) -> None:
    assert classical(Sequent.from_prompt(prompt)) is expected_classical


# ------------------------------------------------------------------------------------ needs_ore


def _seq(thm: str) -> Sequent:
    prem, concl = split_thm(thm)
    return Sequent(premises=prem, conclusion=concl)


@pytest.mark.parametrize(
    ("thm", "expected"),
    [
        # Disjunctive syllogism: genuinely needs a case split (minlen table: 7 lines, uses ORE).
        ("( P v Q ) , ( ~ P ) |- Q", True),
        # Commuting a disjunction: needs ORE with ORI in each branch (minlen selftest comment).
        ("( P v Q ) |- ( Q v P )", True),
        # No disjunction anywhere: trivially ORE-free.
        ("P |- P", False),
        # The conclusion IS the premise disjunction verbatim: derivable by R alone, no case split.
        ("( P v Q ) |- ( P v Q )", False),
        # A premise disjunction feeding straight into ANDI on both sides: still just R, no ORE.
        ("( P v Q ) |- ( ( P v Q ) & ( P v Q ) )", False),
    ],
)
def test_needs_ore_hand_cases(thm: str, expected: bool) -> None:
    assert needs_ore(_seq(thm)) is expected


@pytest.mark.slow
def test_needs_ore_required_implies_shortest_proof_uses_ore() -> None:
    """Over 500 labelled theorems drawn from the committed WP0 proof fixtures (extended with
    permuted-atom variants to reach 500 distinct sequents), whenever `needs_ore` says `True` the
    theorem's own (minlen-verified) shortest proof actually contains an `ORE` line -- the
    soundness direction plan §3's acceptance test asks for. `needs_ore` is not claimed complete in
    the other direction (a proof using ORE where a shorter ORE-free alternative exists is not a
    contradiction), so this test never checks that direction.
    """
    proofs = load_valid_proofs()
    variants: list[tuple[Sequent, str]] = []
    atom_perms = list(itertools.permutations(("P", "Q", "R", "S")))
    for fixture in proofs:
        for perm in atom_perms:
            if len(variants) >= 500:
                break
            mapping: dict[str, str] = dict(zip(("P", "Q", "R", "S"), perm, strict=True))
            thm = " ".join(mapping.get(tok, tok) for tok in fixture["thm"].split())
            body = " ".join(mapping.get(tok, tok) for tok in fixture["proof"].split())
            variants.append((_seq(thm), body))
        if len(variants) >= 500:
            break
    assert len(variants) >= 500

    checked_required = 0
    for seq, body in variants:
        if needs_ore(seq):
            checked_required += 1
            assert "ORE" in body.split(), f"needs_ore(True) but no ORE in shortest proof: {seq}"
    assert checked_required > 0  # the sample must actually exercise the "required" branch


# ------------------------------------------------------------------------------------ necessity


def test_necessity_requires_reductio_for_dn_elimination() -> None:
    seq = Sequent.from_prompt("THM ( ~ ( ~ P ) ) SEQ P PRF")
    assert necessity(seq, "reductio", bound=6, time_s=10.0) == "requires"


def test_necessity_not_required_when_pattern_free_proof_exists() -> None:
    seq = Sequent.from_prompt("THM ( P > Q ) , P SEQ Q PRF")
    assert necessity(seq, "reductio", bound=6, time_s=10.0) == "not_required"


def test_necessity_unknown_when_unreachable_within_bound() -> None:
    seq = Sequent.from_prompt("THM ( ( P > Q ) > P ) SEQ P PRF")  # Peirce, bound too small
    assert necessity(seq, "reductio", bound=3, time_s=5.0) == "unknown"


def test_necessity_unknown_for_uses_only_pattern() -> None:
    seq = Sequent.from_prompt("THM ( P > Q ) , P SEQ Q PRF")
    assert necessity(seq, "impe_chain4", bound=6, time_s=5.0) == "unknown"
    assert necessity(seq, "ori_ore", bound=6, time_s=5.0) == "unknown"


def test_necessity_requires_impi_ore_within_a_tight_bound() -> None:
    seq = Sequent.from_prompt("THM ( P v Q ) , ( P > R ) SEQ ( ( Q > R ) > R ) PRF")
    # minlen's own selftest: this needs 9 lines unrestricted, 11 with ORE forbidden inside an IMPI
    # box. At bound=9, the restricted search exhausts without reaching that 11-line workaround.
    assert necessity(seq, "impi_ore", bound=9, time_s=20.0) == "requires"
    # At a bound wide enough for the workaround (11), it is no longer required.
    assert necessity(seq, "impi_ore", bound=12, time_s=20.0) == "not_required"


def test_necessity_rejects_unknown_pattern() -> None:
    seq = Sequent(premises=("P",), conclusion="P")
    with pytest.raises(ValueError):
        necessity(seq, "not_a_real_pattern", bound=6, time_s=5.0)


# ------------------------------------------------------------------------------------ needs_depth


def test_needs_depth_requires_when_shallow_search_fails() -> None:
    # Depth-3 nested IMPI boxes (patterns.py's own test fixture): P > (Q > (R > R)).
    seq = Sequent.from_prompt("THM SEQ ( P > ( Q > ( R > R ) ) ) PRF")
    assert needs_depth(seq, max_depth=1, bound=6, time_s=10.0) == "requires"


def test_needs_depth_not_required_when_shallow_search_succeeds() -> None:
    seq = Sequent.from_prompt("THM ( P > Q ) , P SEQ Q PRF")
    assert needs_depth(seq, max_depth=2, bound=6, time_s=10.0) == "not_required"


def test_needs_depth_unknown_when_unreachable() -> None:
    seq = Sequent.from_prompt("THM ( ( P > Q ) > P ) SEQ P PRF")
    assert needs_depth(seq, max_depth=1, bound=3, time_s=5.0) == "unknown"


def test_needs_depth_default_time_s_allows_positional_plan_signature() -> None:
    # plan §2/§3 lists `needs_depth(seq, max_depth, bound)`; time_s must be an optional extra.
    seq = Sequent.from_prompt("THM ( P > Q ) , P SEQ Q PRF")
    assert needs_depth(seq, 2, 6) == "not_required"


# ------------------------------------------------------------------------------------ rule_uses


def test_rule_uses_counts_pruned_proof_rules() -> None:
    body = (
        "N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; "
        "N4 R : PR ; N5 ( Q & R ) : ANDI N3 N4 ; QED"
    )
    counts = rule_uses(body)
    assert counts == {"PR": 3, "IMPE": 1, "ANDI": 1}


def test_rule_uses_drops_unpruned_padding_lines() -> None:
    """A line the conclusion never cites (padding) is dropped by the dependency prune, so it must
    not show up in the rule histogram."""
    body = (
        "N1 P : PR ; "
        "N2 ( P v Q ) : ORI1 N1 ; "  # padding: never cited below
        "N3 P : R N1 ; "
        "QED"
    )
    counts = rule_uses(body)
    assert "ORI1" not in counts
    assert counts == {"PR": 1, "R": 1}


def test_rule_uses_falls_back_on_unparseable_body() -> None:
    # No QED / malformed: prune_proof returns None, so rule_uses falls back to the raw body.
    body = "N1 P : PR ;"
    counts = rule_uses(body)
    assert counts == {"PR": 1}
