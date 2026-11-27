import sys
from pathlib import Path

import numpy as np

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import temp_pf_stats as ps  # noqa: E402


def make(shift_pt, n_prompts=40, n_dpo=8, n_s=4, seed=0, len_shift=0.0, ends=True):
    rng = np.random.default_rng(seed)
    by_p = {}
    for p in range(n_prompts):
        base = rng.normal(-0.4, 0.05)

        def resp(mean, ln):
            n = int(max(50, rng.normal(ln, 30)))
            lp = mean + rng.normal(0, 0.03)
            return {
                "id": f"{p}",
                "ends_eos": ends,
                "n_resp": n,
                "total_lp": lp * n,
                "mean_lp": lp,
                "answer_lp": lp * n * 0.1,
                "answer_n": int(n * 0.1),
            }

        by_p[f"p{p}"] = {
            "dpo": [resp(base, 500) for _ in range(n_dpo)],
            "s2300": [resp(base + shift_pt, 500 + len_shift) for _ in range(n_s)],
            "c1": [resp(base + 0.05, 500) for _ in range(4)],
        }
    return by_p


def test_no_shift_is_pass_or_unresolved_and_p_low_near_expected():
    f = ps.segment_stats(make(0.0), "full", reps=500)
    assert (
        0.03 < f["P_low_pt"]["share"] < 0.25
        and abs(f["P_low_pt"]["no_shift_expected"] - 1 / 9) < 1e-9
    )
    assert ps.verdict(f)["verdict"] in ("PASS", "UNRESOLVED")


def test_strong_negative_shift_is_tail():
    f = ps.segment_stats(make(-0.3), "full", reps=500)
    assert f["P_low_pt"]["share"] > 0.9 and f["S_pt"]["hi95"] < 0
    assert ps.verdict(f)["verdict"] == "TAIL"


def test_positive_shift_passes():
    f = ps.segment_stats(make(+0.2), "full", reps=500)
    assert f["S_pt"]["lo95"] > 0 and f["P_low_pt"]["share"] < 0.05
    v = ps.verdict(f)
    assert v["verdict"] == "PASS" and v["pass_p_low_threshold"] == 0.25


def test_four_comparators_use_the_35_percent_threshold():
    f = ps.segment_stats(make(0.0, n_dpo=4), "full", reps=300)
    assert f["n_dpo_median"] == 4 and ps.verdict(f)["pass_p_low_threshold"] == 0.35


def test_length_adjustment_removes_a_pure_length_effect():
    # s2300 responses 300 tokens longer with the same per-token likelihood: total LL much lower, adjusted residual ~ 0
    by_p = make(0.0, len_shift=300.0)
    f = ps.segment_stats(by_p, "full", reps=300)
    raw_total_gap = np.median(
        [
            np.mean([r["total_lp"] for r in g["s2300"]])
            - np.mean([r["total_lp"] for r in g["dpo"]])
            for g in by_p.values()
        ]
    )
    assert raw_total_gap < -50
    assert abs(f["S_pt"]["median"]) < 0.03
    assert abs(f["S_adj"]["median"]) < abs(raw_total_gap)


def test_cut_draws_are_excluded_and_empty_input_is_unresolved():
    f = ps.segment_stats(make(0.0, ends=False), "full", reps=100)
    assert f == {"n_prompts": 0}
    assert ps.verdict(f)["verdict"] == "UNRESOLVED"


def test_beta_is_the_within_prompt_slope():
    rng = np.random.default_rng(0)
    d = {}
    for p in range(10):
        n = rng.integers(100, 900, 8).astype(float)
        d[p] = (-0.5 * n + rng.normal(0, 3, 8) + 17 * p, n)
    assert abs(ps.pooled_beta(d) + 0.5) < 0.02
