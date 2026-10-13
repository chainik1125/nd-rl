"""Tests for ``code.shared_code.estimators``: worked cases from the take-home fork's runs.

Sources (fork ``dan-pandori/nd-takehome``): ``capability-defs`` = branch ``dan_capability-defs``
at da11be78 (``capability_defs/analysis/out/bracket.json``, ``capability_defs/REPORT.md``);
``create-or-elicit`` = branch ``dan_create-or-elicit`` at 8e195390
(``artifacts/coe/certificates.json``, ``review_coe/cert.log``,
``artifacts/coe/estimator/per_theorem.json``). The fork computed the Clopper–Pearson bounds with
SciPy's beta quantiles, which agree with the exact root found here to about 1e-11 (the root is the
more precise: its binomial tail misses alpha by 4e-17 at c = 1, n = 1,377,024, SciPy's by 3e-12).
"""

import math

import pytest

from code.shared_code.estimators import (
    certified_budget,
    clopper_pearson_interval,
    clopper_pearson_lower,
    clopper_pearson_upper,
    good_turing_unseen,
    k_to_solve,
    known_proof_log_p,
    log_sum_exp,
    name_base_marginal,
    pass_at_k,
    zero_success_threshold,
)


def test_pass_at_k() -> None:
    assert pass_at_k(8, 1, 4) == 0.5  # fork tests/test_grpo_adv.py: 1 - C(7, 4) / C(8, 4)
    assert pass_at_k(8, 5, 4) == 1.0  # every 4 of 8 attempts holds a success
    for n, c, k in [(16, 3, 5), (256, 1, 64), (40, 0, 7), (100, 99, 100)]:
        exact = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(exact, rel=1e-12, abs=1e-15)
    assert pass_at_k(256, 3, 1) == pytest.approx(3 / 256)
    for bad in [(8, 1, 9), (8, 1, 0), (8, 9, 4)]:
        with pytest.raises(ValueError):
            pass_at_k(*bad)


# (c, n, one-sided 95 % upper bound) as the fork's SciPy computed them
UPPER = [
    (0, 66560, 4.500698405365023e-05),  # bracket.json s1 la_transfer_478; REPORT.md:239
    (10, 17152, 0.0009887346464170711),  # bracket.json s0 textbook_245a0349bb058006b821
    (1, 1377024, 3.4450075449924603e-06),  # bracket.json s1 la_transfer_1077
    (0, 1508096, 1.9864314328428887e-06),  # bracket.json s2 la_transfer_1833
    (0, 1491420, 2.008642277075361e-06),  # certificates.json ko_s0 pool105
    (1, 1338328, 3.5446152834362385e-06),  # certificates.json ko_s2 pool105
    (3, 1360222, 5.700277586553909e-06),  # certificates.json ko_s4 pool105
    (6, 1120378, 1.0569972612257482e-05),  # certificates.json ko_s1 pool105
    (6, 186172, 6.36089736143279e-05),  # certificates.json ko_s1 contraposition_conv
    (3, 211387, 3.667949943217451e-05),  # certificates.json ko_s4 contraposition_conv
]


@pytest.mark.parametrize(("c", "n", "upper"), UPPER)
def test_clopper_pearson_upper(c: int, n: int, upper: float) -> None:
    assert clopper_pearson_upper(c, n) == pytest.approx(upper, rel=1e-10)


def test_clopper_pearson_lower_and_two_sided() -> None:
    # bracket.json "lo" (one-sided 95 %)
    assert clopper_pearson_lower(10, 17152) == pytest.approx(0.0003163462669224892, rel=1e-10)
    assert clopper_pearson_lower(1, 1377024) == pytest.approx(3.724938231449035e-08, rel=1e-10)
    # create-or-elicit's two-sided 95 % coverage intervals (estimator.txt; summary.json)
    assert clopper_pearson_interval(35, 45) == pytest.approx((0.6291, 0.8880), abs=5e-5)
    assert clopper_pearson_interval(14, 17) == pytest.approx((0.5657, 0.9620), abs=5e-5)
    assert clopper_pearson_interval(2, 2)[0] == pytest.approx(1 - 0.841886116991581, rel=1e-12)
    assert clopper_pearson_lower(0, 50) == 0.0 and clopper_pearson_upper(50, 50) == 1.0
    # the closed forms and the root agree where both apply
    assert clopper_pearson_upper(0, 512) == pytest.approx(1 - 0.05 ** (1 / 512), rel=1e-12)
    assert clopper_pearson_upper(0, 512) == pytest.approx(0.005833955601, rel=1e-9)  # REPORT.md:62


def test_k_to_solve() -> None:
    # capability-defs card passk-equal-k.md:85: "k-to-solve 1,000–3,200"
    lo, hi = k_to_solve(10, 17152)
    assert (round(lo, 1), round(hi, 1)) == (1011.4, 3161.1)
    lo, hi = k_to_solve(0, 512)  # REPORT.md:62: "k-to-solve above ≈ 170"
    assert round(lo, 2) == 171.41 and hi == math.inf


def test_known_proof_estimate() -> None:
    # capability-defs bracket.json, s1 la_transfer_478: 22 known proofs, 20 scored over all 33
    # name bases and 2 over one base (a lower bound: minus ln 33); LB08 = -35.522952587521324.
    exact = [-54.5452, -35.7795, -53.5305, -52.8469, -38.1943, -37.4175, -40.5232, -59.9894]
    exact += [-61.4291, -71.7598, -61.8543, -65.3773, -67.1754, -73.8284, -63.3487, -70.823]
    exact += [-75.6927, -66.1668, -61.9278, -61.7081]
    one_base = [name_base_marginal([x], 33) for x in (-84.7949, -79.5149)]
    terms = exact[:15] + one_base[:1] + exact[15:16] + one_base[1:] + exact[16:]
    assert known_proof_log_p(terms) == -35.522952587521324  # bit for bit, the fork's arithmetic
    # its k-to-solve bracket [1 / UB95, 1 / estimate] (0 successes in 66,560)
    assert round(1 / clopper_pearson_upper(0, 66560), 1) == 22218.8
    assert math.exp(-known_proof_log_p(terms)) == pytest.approx(2.69e15, rel=1e-2)
    assert name_base_marginal([math.log(0.2)] * 33, 33) == pytest.approx(math.log(0.2))
    assert log_sum_exp([]) == -math.inf
    assert log_sum_exp([math.log(0.1), math.log(0.2)]) == pytest.approx(math.log(0.3))


def test_good_turing() -> None:
    # create-or-elicit per_theorem.json, s1 textbook_33a5b9f5c6c6ad7582e7: 5 successes, 5
    # distinct proofs, 3 of them known -> M0 = 1, M0u = 2 / 5; E_GT = E_all / (1 - M0u).
    counts = {"y1": 1, "y2": 1, "y3": 1, "y4": 1, "y5": 1}
    known = {"y1", "y2", "y3"}
    assert good_turing_unseen(counts) == 1.0
    m0u = good_turing_unseen(counts, known)
    assert m0u == 0.4
    e_all = 0.00025982950498503805
    assert e_all / (1 - m0u) == pytest.approx(0.00043304917497506343, rel=1e-15)
    # s0 la_transfer_1909's upper bound: 3 successes, one proof twice -> f1 = 1 of N = 3
    assert good_turing_unseen({"a": 2, "b": 1}) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        good_turing_unseen({})


# (budget K, "needs n >= ..." as printed by review_coe/cert.log; K from certificates.json)
THRESHOLDS = [
    (23322.094842439517, 1397334),
    (17188.62663652703, 1029849),
    (20906.75582871953, 1252619),
    (20782.098838976504, 1245151),
    (21554.009487839106, 1291399),
]


@pytest.mark.parametrize(("budget", "needed"), THRESHOLDS)
def test_zero_success_threshold(budget: float, needed: int) -> None:
    assert round(zero_success_threshold(budget)) == needed
    n = math.ceil(zero_success_threshold(budget))
    assert clopper_pearson_upper(0, n) < 0.05 / budget <= clopper_pearson_upper(0, n - 1)


def test_certified_budget() -> None:
    # certificates.json ko_s0 per_family.excluded_middle: 0 / 525,548
    assert certified_budget(0, 525548) == pytest.approx(8771.636612975326, rel=1e-12)
    # claims_ledger.md L20: the J9 certificates "hold for K <= 20,795 / 22,983 / 25,171", and
    # s1's 1 / 1,377,024 for K <= 14,514, s2's 1 / 1,114,880 for K <= 11,751
    got = [certified_budget(0, n) for n in (1245952, 1377024, 1508096)]
    assert [round(k) for k in got] == [20795, 22983, 25171]
    assert round(certified_budget(1, 1377024)) == 14514
    assert round(certified_budget(1, 1114880)) == 11751
    # ko_s0: 0 / 1,491,420 against K = 23,322 is certified; ko_s1 (6 successes) is not
    assert certified_budget(0, 1491420) > 23322.094842439517
    assert certified_budget(6, 1120378) < 17188.62663652703
