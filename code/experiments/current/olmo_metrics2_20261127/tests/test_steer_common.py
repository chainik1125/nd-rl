"""CPU tests of steer/steer_common.py (selection rule and every fallback, bootstrap of g, split-half, scrambles, trigger incl.
d6, shrink order, E2 pid assignment, NLL segments incl. d1).  Run: python -m pytest -q tests/test_steer_common.py"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steer"))
import steer_common as C  # noqa: E402


# --- markers, groups, NLL segments -------------------------------------------------------------------------------------
def test_find_marker_uses_last_marker_and_close_variants():
    ids = [5, 524, 27963, 1363, 7, 8, 524, 27963, 397, 9, 10, 100257]
    assert C.find_marker(ids) == 9  # after the ">\n" of the LAST marker
    assert C.find_marker([1, 2, 3]) is None
    assert C.find_marker([524, 27963, 29, 5]) == 3
    assert (
        C.find_marker([524, 27963, 999, 5]) is None
    )  # '</think' without a '>' token is not a marker


def test_group_bounds_cover_sequence_without_overlap():
    b = C.group_bounds(n_prompt=10, lead_end=4, trail_start=8, n_total=15)
    assert b == [(0, 1), (1, 4), (4, 8), (8, 10), (10, 15)]
    cover = sorted(i for a, c in b for i in range(a, c))
    assert cover == list(range(15))
    with pytest.raises(AssertionError):
        C.group_bounds(10, 11, 8, 15)


def test_nll_stats_segments_and_d1_no_marker():
    lp = -np.arange(1.0, 8.0)  # NLL 1..7
    st = C.nll_stats(lp, think_end=5)
    assert st["total"] == [28.0, 7] and st["think"] == [15.0, 5] and st["answer"] == [13.0, 2]
    st0 = C.nll_stats(lp, None)  # no marker: everything is thinking, no answer tokens
    assert st0["think"] == [28.0, 7] and st0["answer"] == [0.0, 0]
    per = {"a": st, "b": st0}
    assert C.n_excluded_answer(per, ["a", "b"]) == 1
    assert C.pool_stats(per, ["a", "b"], "answer") == pytest.approx(
        13.0 / 2
    )  # b is excluded by having 0 tokens
    assert C.pool_stats(per, ["a", "b"], "total") == pytest.approx(56.0 / 14)


# --- g, R, bootstrap, bands ----------------------------------------------------------------------------------------------
def _per(means, n=100):
    return {
        f"t{i}": {"total": [m * n, n], "think": [m * n, n], "answer": [m * 5, 5]}
        for i, m in enumerate(means)
    }


def test_g_point_estimate_and_ratio_definitions():
    dpo, s23 = _per([2.0] * 16), _per([1.0] * 16)
    steered = _per([1.8] * 16)
    tids = list(dpo)
    assert C.ratio_stat(dpo, steered, s23, tids, "total", 1) == pytest.approx(0.2)  # (2-1.8)/(2-1)
    nec = _per([1.3] * 16)
    assert C.ratio_stat(dpo, nec, s23, tids, "total", -1) == pytest.approx(0.3)  # (1.3-1)/(2-1)


def test_bootstrap_ci_covers_point_and_bands():
    rng = np.random.RandomState(1)
    dpo = _per(2.0 + 0.05 * rng.randn(16))
    s23 = _per(1.0 + 0.05 * rng.randn(16))
    steered = _per(1.5 + 0.05 * rng.randn(16))
    ci = C.bootstrap_ratio(dpo, steered, s23, list(dpo), "total", 1, B=2000, seed=0)
    assert ci["lo"] < ci["point"] < ci["hi"] and 0.4 < ci["point"] < 0.6
    assert C.band_of(ci) == "substantially (g >= 20 %)"
    # deterministic with the seed
    assert C.bootstrap_ratio(dpo, steered, s23, list(dpo), "total", 1, B=2000, seed=0) == ci
    # interval straddling an edge -> between bands
    assert C.band_of({"lo": 0.01, "hi": 0.09}) == "between bands"
    assert C.band_of({"lo": 0.06, "hi": 0.15}) == "slightly (5-20 %)"
    assert C.band_of({"lo": -0.02, "hi": 0.03}) == "barely moved (g < 5 %)"
    assert C.band_of({"lo": float("nan"), "hi": 0.1}) == "undefined"


def test_bootstrap_resamples_prompts_jointly_numerator_and_denominator():
    # steered == dpo for every prompt, so g = 0 in EVERY resample (it would not be if num/den were resampled separately
    # with different prompts)
    rng = np.random.RandomState(2)
    dpo = _per(2.0 + rng.rand(16))
    s23 = _per(1.0 + rng.rand(16))
    ci = C.bootstrap_ratio(dpo, dpo, s23, list(dpo), "total", 1, B=500, seed=3)
    assert ci["lo"] == pytest.approx(0.0, abs=1e-12) and ci["hi"] == pytest.approx(0.0, abs=1e-12)


# --- vectors ---------------------------------------------------------------------------------------------------------------
def test_scramble_preserves_norm_and_magnitudes_is_seeded_and_changes_direction():
    v = np.random.RandomState(0).randn(5120).astype(np.float32)
    s = C.scramble(v, 2001)
    assert np.allclose(np.abs(s), np.abs(v)) and np.linalg.norm(s) == pytest.approx(
        np.linalg.norm(v), rel=1e-6
    )
    assert np.array_equal(s, C.scramble(v, 2001)) and not np.array_equal(s, C.scramble(v, 2002))
    assert abs(C.cosine(v, s)) < 0.1


def test_split_half_cosine_and_conversion_and_top5_removal():
    rng = np.random.RandomState(0)
    mu = rng.randn(200)
    pp = mu[None] + 0.01 * rng.randn(32, 200)
    sh = C.split_half(pp)
    assert sh["c"] > 0.99 and sh["converted"] == pytest.approx(
        math.sqrt(2 * sh["c"] / (1 + sh["c"]))
    )
    # noise only: c near 0
    noise = C.split_half(rng.randn(32, 200))
    assert abs(noise["c"]) < 0.3 and noise["converted"] <= 0.6
    # a few huge consistent coordinates make c look high; removing the top 5 exposes the noise
    pp2 = rng.randn(32, 200)
    pp2[:, :5] += 1000.0
    sh2 = C.split_half(pp2)
    assert sh2["c"] > 0.99 and abs(sh2["c_top5_removed"]) < 0.4
    assert sorted(sh2["top5_coordinates"]) == [0, 1, 2, 3, 4]
    assert C.top_k_share(np.r_[np.ones(5) * 100, np.ones(195)]) > 0.98


# --- selection rule: every branch --------------------------------------------------------------------------------------------
def R(nll, rise):
    return {"nll_sel": nll, "damage_rise": rise}


def test_selection_picks_min_nll_among_admissible_and_respects_damage_limit():
    d_rev = 0.05  # limit 0.10
    res = {
        ("V_resp", 16, 1.0): R(1.00, 0.01),
        ("V_resp", 20, 2.0): R(0.90, 0.09),
        ("V_resp", 24, 4.0): R(0.80, 0.30),  # best NLL but inadmissible (rise 0.30 > 0.10)
        ("V_prompt", 20, 1.0): R(0.95, 0.02),
    }
    fin = C.select_final(res, d_rev)
    assert (
        fin["chosen"] == ["V_resp", 20, 2.0]
        and not fin["no_admissible"]
        and fin["n_admissible"] == 3
    )
    assert fin["damage_limit"] == pytest.approx(0.10)


def test_selection_tie_break_alpha_then_layer_then_vresp():
    d_rev = 1.0
    res = {
        ("V_resp", 24, 4.0): R(0.8000, 0.0),
        ("V_resp", 20, 2.0): R(0.8030, 0.0),  # within 0.005 of the minimum
        ("V_prompt", 20, 2.0): R(0.8040, 0.0),
        ("V_resp", 32, 2.0): R(0.8040, 0.0),
    }
    assert C.select_final(res, d_rev)["chosen"] == [
        "V_resp",
        20,
        2.0,
    ]  # smaller alpha beats the min
    res2 = {
        ("V_resp", 24, 2.0): R(0.800, 0),
        ("V_resp", 20, 2.0): R(0.802, 0),
        ("V_prompt", 16, 2.0): R(0.801, 0),
    }
    assert C.select_final(res2, 1.0)["chosen"] == ["V_prompt", 16, 2.0]  # same alpha: smaller layer
    res3 = {("V_resp", 20, 2.0): R(0.802, 0), ("V_prompt", 20, 2.0): R(0.800, 0)}
    assert C.select_final(res3, 1.0)["chosen"] == ["V_resp", 20, 2.0]  # same alpha, layer: V_resp
    res4 = {("V_resp", 20, 2.0): R(0.810, 0), ("V_resp", 20, 4.0): R(0.800, 0)}
    assert C.select_final(res4, 1.0)["chosen"] == [
        "V_resp",
        20,
        4.0,
    ]  # outside the tie band: min NLL


def test_selection_flags_minimum_at_largest_alpha():
    res = {("V_resp", 20, 1.0): R(0.9, 0), ("V_resp", 20, 8.0): R(0.7, 0)}
    fin = C.select_final(res, 1.0)
    assert fin["chosen"] == ["V_resp", 20, 8.0] and fin["at_max_alpha"]
    assert not C.select_final(
        {("V_resp", 20, 1.0): R(0.5, 0), ("V_resp", 20, 8.0): R(0.7, 0)}, 1.0
    )["at_max_alpha"]


def test_selection_fallback_no_admissible_uses_alpha1_least_damaging_layer():
    res = {
        ("V_resp", 16, 1.0): R(0.9, 0.50),
        ("V_resp", 24, 1.0): R(0.95, 0.20),
        ("V_resp", 32, 1.0): R(0.8, 0.60),
        ("V_resp", 24, 2.0): R(0.7, 0.01 + 1.0),
        ("V_prompt", 24, 1.0): R(0.7, 0.05 + 0.2),
    }
    fin = C.select_final(res, d_rev=0.01)  # limit 0.02: nothing admissible
    assert (
        fin["no_admissible"] and fin["chosen"] == ["V_resp", 24, 1.0] and fin["n_admissible"] == 0
    )


def test_choose_resp_layer_branches_d2():
    resp = {
        ("V_resp", 16, 1.0): R(0.9, 0.01),
        ("V_resp", 24, 2.0): R(0.7, 0.02),
        ("V_resp", 32, 4.0): R(0.6, 0.9),
    }
    s1 = C.choose_resp_layer(
        resp, d_rev=0.05
    )  # limit 0.1: 32 is inadmissible, best admissible is layer 24
    assert s1["layer"] == 24 and "admissible" in s1["rule"]
    none = {
        ("V_resp", 16, 1.0): R(0.9, 0.5),
        ("V_resp", 24, 1.0): R(0.7, 0.3),
        ("V_resp", 24, 2.0): R(0.6, 0.0),
    }
    s2 = C.choose_resp_layer(
        none, d_rev=0.01
    )  # nothing admissible at limit 0.02 except (24, 2.0)?  rise 0.0 -> admissible
    assert s2["layer"] == 24
    none2 = {
        ("V_resp", 16, 1.0): R(0.9, 0.5),
        ("V_resp", 24, 1.0): R(0.7, 0.3),
        ("V_resp", 32, 1.0): R(0.7, 0.4),
    }
    s3 = C.choose_resp_layer(
        none2, d_rev=0.01
    )  # none admissible: smallest rise at alpha = 1 -> 24 (d2)
    assert s3["layer"] == 24 and "none admissible" in s3["rule"]


# --- cost plan and shrink order -----------------------------------------------------------------------------------------------
def test_plan_nll_grid_shrink_order_and_budget():
    kw = dict(
        tok_sel=80_000,
        tok_dmg=80_000,
        tok_chk=80_000,
        usd_per_h=5.29,
        budget_usd=3.0,
        extra_tokens=100_000,
    )
    fast = C.plan_nll_grid(1e6, **kw)
    assert (
        fast["plan"]["step"] == "full grid"
        and fast["plan"]["settings"] == 30
        and fast["plan"]["within_budget"]
    )
    steps = [s["step"] for s in C.plan_nll_grid(1.0, **kw)["log"]]
    assert steps == [
        "full grid",
        "drop alpha 0.5",
        "drop layer 40",
        "drop V_prompt row",
    ]  # the fixed order
    slow = C.plan_nll_grid(1.0, **kw)["plan"]
    assert (
        slow["settings"] == 16
        and not slow["within_budget"]
        and 0.5 not in slow["alphas"]
        and 40 not in slow["layers"]
    )
    # projection arithmetic: (30 settings x 160k + 5 x 80k + 100k) tokens at 3,000 tok/s
    p = C.plan_nll_grid(3000.0, **kw)["log"][0]
    assert p["tokens"] == 30 * 160_000 + 5 * 80_000 + 100_000
    assert p["projected_usd"] == pytest.approx(p["tokens"] / 3000 / 3600 * 5.29, abs=1e-3)
    # intermediate: stops at the first step within budget
    mid = C.plan_nll_grid(3000.0 * 1.3, **kw)
    assert mid["plan"]["within_budget"]


# --- E2 trigger (d6) and pid assignment ---------------------------------------------------------------------------------------
def test_e2_trigger_full_and_cut_e0p_d6_and_heldout_rule():
    t = C.e2_trigger(2, 384, 0, 384, 0.30, 0.30)
    assert t["trigger"] and t["reach_hit"] and not t["held_hit"] and t["scale_s"] == 1.0
    assert not C.e2_trigger(1, 384, 0, 384, 0.30, 0.30)["trigger"]  # +1 is not enough
    # E0' cut to 192 draws: s = 2, compare with 2 x E0' passes.  1 pass in E0' -> need >= 4 in E1
    assert not C.e2_trigger(3, 384, 1, 192, 0.30, 0.30)["trigger"]
    assert C.e2_trigger(4, 384, 1, 192, 0.30, 0.30)["trigger"]
    assert C.e2_trigger(0, 384, 0, 384, 0.37, 0.30)["held_hit"]  # +7 points
    assert C.e2_trigger(0, 384, 0, 384, 0.23, 0.30)["held_hit"]  # -7 points
    assert not C.e2_trigger(0, 384, 0, 384, 0.34, 0.30)["held_hit"]


def test_split_draws_and_e2_pid_assignment_mod3():
    assert (
        C.split_draws(8) == [3, 3, 2]
        and C.split_draws(4) == [2, 1, 1]
        and sum(C.split_draws(8)) == 8
    )
    reach = [f"r{i}" for i in range(5)]
    held = [f"h{i}" for i in range(7)]
    specs = C.e2_pid_specs(reach, held, reach_n=8, held_n=2)
    for k in range(3):
        assert all(specs[k][r] == C.split_draws(8)[k] for r in reach)
    for i, h in enumerate(held):
        owners = [k for k in range(3) if h in specs[k]]
        assert owners == [i % 3] and specs[i % 3][h] == 2  # prompt i -> scramble (i mod 3) + 1
    tot = sum(sum(s.values()) for s in specs)
    assert tot == 5 * 8 + 7 * 2


def test_pass_at_k_matches_known_values():
    assert C.pass_at_k(8, 0, 8) == 0.0 and C.pass_at_k(8, 8, 1) == 1.0
    assert C.pass_at_k(8, 1, 1) == pytest.approx(1 / 8)
    assert C.pass_at_k(8, 1, 8) == pytest.approx(1.0)
    assert C.pass_at_k(4, 1, 4) == 1.0
