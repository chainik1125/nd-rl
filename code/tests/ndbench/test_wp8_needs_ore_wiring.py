"""WP8 integration fix list item 7: wire WP2's real `logic.skills.needs_ore` into WP4's
`suites.glue.build_joined`, whose injected `needs_ore` has a different signature
(`Callable[[Sequent, str], bool]`, a candidate join's sequent *and* witness body, vs. `logic.
skills.needs_ore`'s `Callable[[Sequent], bool]`).

`code.ndbench.suites.glue.needs_ore_from_skills` is the adapter (ignores `body`, calls the real
`logic.skills.needs_ore(seq)`); this file checks it agrees with `logic.skills.needs_ore`
(trivially, since it is a thin wrapper) and with `test_wp4_glue.py`'s independent
`stub_needs_ore` reimplementation on the same generated CASE candidates, and that `build_joined`
actually uses it when resolved lazily through `cfg["needs_ore"]` as
`"code.ndbench.suites.glue:needs_ore_from_skills"` -- the form a real CLI build's YAML config uses
(see `configs/glue.yaml`), since `cli.cmd_build` does not resolve `needs_ore` itself (plan's WP4
row: "For `needs_ore`, accept an injected callable").
"""

from __future__ import annotations

import random

import pytest

from code.ndbench.core.keys import split_thm
from code.ndbench.core.types import Sequent
from code.ndbench.logic.skills import needs_ore
from code.ndbench.suites import glue
from code.ndbench.suites.glue import needs_ore_from_skills

from .test_wp4_glue import fixture_parts, stub_needs_ore  # noqa: F401 (fixture_parts is a fixture)


def _seq(thm: str) -> Sequent:
    prem, concl = split_thm(thm)
    return Sequent(premises=prem, conclusion=concl)


@pytest.mark.parametrize(
    ("thm", "expected"),
    [
        ("( P v Q ) , ( ~ P ) |- Q", True),
        ("( P v Q ) |- ( Q v P )", True),
        ("P |- P", False),
        ("( P v Q ) |- ( P v Q )", False),
        ("( P v Q ) |- ( ( P v Q ) & ( P v Q ) )", False),
    ],
)
def test_needs_ore_from_skills_matches_needs_ore_on_the_wp2_hand_cases(
    thm: str, expected: bool
) -> None:
    seq = _seq(thm)
    assert needs_ore_from_skills(seq, "N1 P : PR ; QED") is expected
    assert needs_ore_from_skills(seq, "N1 P : PR ; QED") is needs_ore(seq)


def test_needs_ore_from_skills_ignores_body() -> None:
    """`needs_ore_from_skills`'s result must depend only on the sequent, per plan §2 B3's
    `case_split` test being a property of the theorem alone -- passing two different (nonsense)
    bodies for the same sequent must not change the answer."""
    seq = _seq("( P v Q ) , ( ~ P ) |- Q")
    assert needs_ore_from_skills(seq, "body one") == needs_ore_from_skills(
        seq, "a very different body"
    )


def test_needs_ore_from_skills_agrees_with_the_wp4_stub_on_generated_case_joins(
    fixture_parts: list[glue._Part],  # noqa: F811
) -> None:
    """Cross-check the real wiring against WP4's independent from-scratch reimplementation
    (`stub_needs_ore`, written without importing WP2's module) on real generated CASE-dilemma
    joins, not just the small hand-picked table above."""
    by_len = glue._index_by_len(fixture_parts)
    rng = random.Random(0)
    checked = 0
    for ell in range(2, 8):
        for _ in range(50):
            found = glue._try_case_dilemma(by_len, ell, rng)
            if found is None:
                continue
            pa, ai, pb, bi = found
            seq, body, _l = glue._join_case(pa, ai, pb, bi, same=False)
            checked += 1
            assert needs_ore_from_skills(seq, body) == stub_needs_ore(seq, body)
    assert checked >= 5


def test_build_joined_resolves_needs_ore_from_cfg_string_the_way_a_cli_config_would(
    fixture_parts: list[glue._Part],  # noqa: F811
) -> None:
    """`cfg["needs_ore"] = "code.ndbench.suites.glue:needs_ore_from_skills"` (a real CLI build's
    `configs/glue.yaml` value, resolved lazily by `build_joined`'s own `_resolve_optional`, not by
    `cli.cmd_build`) must gate CASE items identically to passing the callable directly."""
    resolved = glue._resolve_optional("code.ndbench.suites.glue:needs_ore_from_skills")
    assert resolved is needs_ore_from_skills

    by_len = glue._index_by_len(fixture_parts)
    rng = random.Random(1)
    checked = 0
    for ell in range(2, 8):
        for _ in range(50):
            found = glue._try_case_dilemma(by_len, ell, rng)
            if found is None:
                continue
            pa, ai, pb, bi = found
            seq, body, _l = glue._join_case(pa, ai, pb, bi, same=False)
            checked += 1
            assert resolved(seq, body) == needs_ore_from_skills(seq, body)
    assert checked >= 5
