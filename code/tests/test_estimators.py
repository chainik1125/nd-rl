"""Tests for ``code.shared_code.estimators``: worked cases from the take-home fork's runs.

Sources (fork ``dan-pandori/nd-takehome``): ``capability-defs`` = branch ``dan_capability-defs``
at da11be78 (``capability_defs/analysis/out/bracket.json``, ``capability_defs/REPORT.md``,
``claims_ledger.md``); ``create-or-elicit`` = branch ``dan_create-or-elicit`` at 8e195390
(``artifacts/coe/certificates.json``, ``review_coe/cert.log``, ``artifacts/coe/estimator/``).
The fork computed the Clopper–Pearson bounds with SciPy's beta quantiles. They agree with the root
found here to about 1e-11; at c = 1 the root is the more precise (at c = 1, n = 1,377,024 its
binomial tail misses alpha by 4e-17, SciPy's by 3e-12).
"""

import math

import pytest

import code.shared_code.estimators as est
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
    # the fork's GRPO pass@k baseline (grpo_adv.py; tests/test_grpo_adv.py): 1 - C(7, 4) / C(8, 4)
    assert pass_at_k(8, 1, 4) == 0.5
    assert pass_at_k(8, 5, 4) == 1.0  # every 4 of the 8 attempts holds a success
    for n, c, k in [(16, 3, 5), (256, 1, 64), (40, 0, 7), (100, 99, 100)]:
        exact = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(exact, rel=1e-12, abs=1e-15)
    assert pass_at_k(256, 3, 1) == pytest.approx(3 / 256)
    for bad in [(8, 1, 9), (8, 1, 0), (8, 9, 4)]:
        with pytest.raises(ValueError):
            pass_at_k(*bad)


# (c, n, one-sided 95 % upper bound, lower bound or None) as the fork's SciPy computed them
BOUNDS = [
    (0, 66560, 4.500698405365023e-05, 0.0),  # bracket.json s1 la_transfer_478
    (2, 17152, 0.0003670129001697526, 2.0718761013889604e-05),  # bracket.json s0 la_transfer_1633
    (74, 4864, 0.018430138720181982, 0.012442377812331226),  # bracket.json s1 la_transfer_706
    (907, 4864, 0.19589928359366607, 0.17732412294200856),  # bracket.json s2 la_transfer_1371
    (1, 1377024, 3.4450075449924603e-06, 3.724938231449035e-08),  # bracket.json s1 la_transfer_1077
    (0, 1508096, 1.9864314328428887e-06, 0.0),  # bracket.json s2 la_transfer_1833
    (0, 1491420, 2.008642277075361e-06, None),  # certificates.json ko_s0 pool105
    (1, 1338328, 3.5446152834362385e-06, None),  # certificates.json ko_s2 pool105
    (2, 1419656, 4.434725100051238e-06, None),  # certificates.json ko_s5 pool105
    (3, 1360222, 5.700277586553909e-06, None),  # certificates.json ko_s4 pool105
    (6, 1120378, 1.0569972612257482e-05, None),  # certificates.json ko_s1 pool105
    (6, 186172, 6.36089736143279e-05, None),  # certificates.json ko_s1 contraposition_conv
]


@pytest.mark.parametrize(("c", "n", "upper", "lower"), BOUNDS)
def test_clopper_pearson_one_sided(c: int, n: int, upper: float, lower: float | None) -> None:
    assert clopper_pearson_upper(c, n) == pytest.approx(upper, rel=1e-10)
    if lower is not None:
        assert clopper_pearson_lower(c, n) == pytest.approx(lower, rel=1e-10, abs=1e-300)


# create-or-elicit's coverage table (estimator/summary.json, pooled_bins.*.coverage): K of N
# successes with a known proof, and the two-sided 95 % interval on the unseen share 1 - K / N,
# [1 - hi(K1, N), 1 - lo(K0, N)], with K in [K0, K1].
COVERAGE = [
    (2263, 23785, 25481, 0.06352886611135111, 0.9146541538986651),  # ge1e-3
    (253, 310, 362, 0.10917866922789943, 0.3512337789133335),  # 1e-4..1e-3
    (35, 35, 45, 0.11204593887713499, 0.370887552599488),  # 1e-5..1e-4
    (2, 2, 2, 0.0, 0.841886116991581),  # lt1e-5
]


@pytest.mark.parametrize(("k0", "k1", "n", "unseen_lo", "unseen_hi"), COVERAGE)
def test_clopper_pearson_two_sided(k0: int, k1: int, n: int, unseen_lo: float, unseen_hi: float):
    assert 1 - clopper_pearson_interval(k1, n)[1] == pytest.approx(unseen_lo, rel=1e-9, abs=1e-15)
    assert 1 - clopper_pearson_interval(k0, n)[0] == pytest.approx(unseen_hi, rel=1e-9)


def test_clopper_pearson_methods_agree() -> None:
    """The binomial tail summed term by term, and by the incomplete beta function (used when both
    tails have more than EXACT_TERMS terms), agree where both apply."""
    for c, n in [(1500, 3000), (6000, 20000), (2500, 1_000_000), (4200, 5000)]:
        for p in (0.95 * c / n, c / n, min(1.05 * c / n, 0.999)):
            exact = est._binomial_cdf(c, n, p)  # c and n - c are within reach of the sum
            by_beta = est._beta_inc(n - c, c + 1, 1.0 - p, p)
            assert by_beta == pytest.approx(exact, rel=1e-8), (c, n, p)
    lo, hi = clopper_pearson_interval(500_000, 1_000_000)  # milliseconds, not minutes
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.00196, rel=1e-2)
    assert clopper_pearson_upper(0, 512) == pytest.approx(1 - 0.05 ** (1 / 512), rel=1e-12)
    assert clopper_pearson_lower(0, 50) == 0.0 and clopper_pearson_upper(50, 50) == 1.0


def test_k_to_solve() -> None:
    lo, hi = k_to_solve(2, 17152)  # bracket.json s0 la_transfer_1633: [1 / ub, 1 / lo]
    assert lo == pytest.approx(1 / 0.0003670129001697526, rel=1e-10)
    assert hi == pytest.approx(1 / 2.0718761013889604e-05, rel=1e-10)
    lo, hi = k_to_solve(0, 512)  # REPORT.md:62: "k-to-solve above ≈ 170 (UB95 3 / 512)"
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
    # its k-to-solve bracket [1 / UB95, 1 / estimate], with 0 successes in 66,560 attempts
    assert round(1 / clopper_pearson_upper(0, 66560), 1) == 22218.8
    assert math.exp(-known_proof_log_p(terms)) == pytest.approx(2.69e15, rel=1e-2)
    assert name_base_marginal([math.log(0.2)] * 33, 33) == pytest.approx(math.log(0.2))
    assert log_sum_exp([]) == -math.inf
    assert log_sum_exp([math.log(0.1), math.log(0.2)]) == pytest.approx(math.log(0.3))


def test_good_turing() -> None:
    # create-or-elicit per_theorem.json, s0 la_transfer_1909: 3 successes, 2 distinct proofs, one
    # of them known; which one was found twice is not recorded, so M0u is 0 or 1/3, and
    # E_GT = E_all / (1 - M0u) = [0.00023739815719130622, 0.0003560972357869593].
    e_all = 0.00023739815719130622
    for counts, e_gt in [
        ({"known": 2, "new": 1}, 0.0003560972357869593),
        ({"known": 1, "new": 2}, 0.00023739815719130622),
    ]:
        m0u = good_turing_unseen(counts, {"known"})
        assert e_all / (1 - m0u) == pytest.approx(e_gt, rel=1e-15)
        assert good_turing_unseen(counts) == pytest.approx(1 / 3)  # f1 = 1 either way
    assert good_turing_unseen({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}) == 1.0
    with pytest.raises(ValueError):
        good_turing_unseen({})


# (budget K from certificates.json, "needs n >= ..." as review_coe/cert.log prints it)
THRESHOLDS = [
    (23322.094842439517, 1397334),  # ko_s0
    (17188.62663652703, 1029849),  # ko_s1
    (20906.75582871953, 1252619),  # ko_s2
    (20782.098838976504, 1245151),  # ko_s3
    (21554.009487839106, 1291399),  # ko_s4
    (22297.13691133893, 1335924),  # ko_s5
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
