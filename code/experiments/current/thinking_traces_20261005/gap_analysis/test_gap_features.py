"""Regression tests for the gap-analysis feature code (synthetic theorems, none from textbook72).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m pytest -q -p no:cacheprovider \
        code/experiments/current/thinking_traces_20261005/gap_analysis/test_gap_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze as A  # noqa: E402
import common as C  # noqa: E402

# a negation box nested in an implication box
NESTED = (
    "THM ( R > ( ~ S ) ) SEQ ( S > ( ~ R ) ) PRF",
    "N1 ( R > ( ~ S ) ) : PR ; N2 | S : AS ; N3 | | R : AS ; N4 | | ( ~ S ) : IMPE N1 N3 ; "
    "N5 | | F : NEGE N2 N4 ; N6 | ( ~ R ) : NEGI N3 N5 ; N7 ( S > ( ~ R ) ) : IMPI N2 N6 ; QED",
)
# a case split whose first branch is closed by ex falso, inside nothing
CASES = (
    "THM ( Q v ( R & S ) ) , ( ~ Q ) SEQ ( S & R ) PRF",
    "N1 ( Q v ( R & S ) ) : PR ; N2 ( ~ Q ) : PR ; N3 | Q : AS ; N4 | F : NEGE N3 N2 ; "
    "N5 | ( S & R ) : BOTE N4 ; N6 | ( R & S ) : AS ; N7 | S : ANDE2 N6 ; N8 | R : ANDE1 N6 ; "
    "N9 | ( S & R ) : ANDI N7 N8 ; N10 ( S & R ) : ORE N1 N3 N5 N6 N9 ; QED",
)
# reductio wrapping a negation box
RAA = (
    "THM ( ( ~ P ) > ( Q & ( ~ Q ) ) ) SEQ P PRF",
    "N1 ( ( ~ P ) > ( Q & ( ~ Q ) ) ) : PR ; N2 | ( ~ P ) : AS ; "
    "N3 | ( Q & ( ~ Q ) ) : IMPE N1 N2 ; "
    "N4 | Q : ANDE1 N3 ; N5 | ( ~ Q ) : ANDE2 N3 ; N6 | F : NEGE N4 N5 ; "
    "N7 ( ~ ( ~ P ) ) : NEGI N2 N6 ; N8 P : DN N7 ; QED",
)


def test_nested_negation_box() -> None:
    f = C.proof_features(*NESTED)
    assert f["p_skeleton"] == "I(N())"
    assert f["p_negbox_nested"] == 1 and f["p_negbox_in_impi"] == 1
    assert f["p_box_depth"] == 2 and f["p_n_boxes"] == 2 and f["p_has_raa"] == 0


def test_case_split_features() -> None:
    f = C.proof_features(*CASES)
    assert f["p_skeleton"] == "O(|)"
    assert f["p_ore_on_prem"] == 1 and f["p_ore_branch_bote"] == 1
    assert f["p_ore_goal_compound"] == 1 and f["p_ore_both_branches_work"] == 1
    assert f["p_negbox_nested"] == 0


def test_reductio_and_theorem_features() -> None:
    f = C.proof_features(*RAA)
    assert f["p_skeleton"] == "C()" and f["p_has_raa"] == 1 and f["p_raa_top"] == 1
    t = C.thm_features(RAA[0])
    assert t["classical_only"] == 1 and t["goal_atom"] == 1 and t["n_prem"] == 1
    t2 = C.thm_features(NESTED[0])
    assert t2["classical_only"] == 0 and t2["goal_imp"] == 1 and t2["imp_nest"] == 1


def test_stats_helpers() -> None:
    # Fisher exact, two-sided, [[6, 2], [1, 11]]
    assert abs(A.fisher_p(6, 2, 1, 11) - 0.0044) < 5e-4
    assert abs(A.pass_at(3, 64, 1) - 3 / 64) < 1e-12
    assert A.bucket("done", 0) == "lean_reject" and A.bucket("fail:S:unbound", 0) == "scope"
