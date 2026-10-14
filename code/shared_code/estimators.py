"""Estimators for a model's per-theorem success probability from its sampled attempts.

The attempts at one theorem are independent draws from one sampler, each a success (a proof Lean
accepts) with probability p. The functions here estimate or bound p, or something derived from
it, from c successes in n attempts or from the log-probabilities of known proofs:

- ``pass_at_k``: the unbiased pass@k (Chen et al. 2021);
- ``clopper_pearson_upper``, ``clopper_pearson_lower``, ``clopper_pearson_interval``: exact
  binomial confidence bounds on p;
- ``k_to_solve``: an interval on 1 / p, the expected number of attempts to the first proof;
- ``known_proof_log_p``: a lower bound on log p from teacher-forced log-probabilities of known
  proofs (with ``name_base_marginal`` for samplers that draw a random name base);
- ``good_turing_unseen``: the Good–Turing estimate of the success mass on proofs not yet seen;
- ``zero_success_threshold``, ``certified_budget``: how many failed attempts certify
  p < q / K, i.e. "out of reach within K attempts".

Common to all: the attempts are i.i.d. draws from one fixed sampler (checkpoint, temperature,
decoding settings, the name base's distribution), every attempt is judged the same way, and n and
c count every attempt, not just those kept after a filter. Stopping at the first success, or
pooling attempts from different samplers, breaks the assumptions behind the intervals.

Provenance: the definitions of the take-home fork's runs ``capability-defs`` (``cd_bracket.py``,
``cd_score.py``; branch ``dan_capability-defs`` at da11be78) and ``create-or-elicit``
(``coe_estimator.py``, ``review_coe/cert.py``; branch ``dan_create-or-elicit`` at 8e195390), and
``bs_analysis.passk`` (``dan`` at 4dcd0e4f). The fork computed the beta quantiles with SciPy.
Here each bound is the root, found by bisection, of the binomial tail: summed term by term, or,
when both tails have more than ``EXACT_TERMS`` terms, by the incomplete beta function's continued
fraction; the standard library only. ``tests/test_estimators.py`` reproduces worked cases from the
runs' reports and outputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def _check_counts(c: int, n: int) -> None:
    if n < 1 or not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n and n >= 1, got c={c}, n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased estimate of pass@k, 1 − (1 − p)^k, from c successes in n attempts:
    1 − C(n − c, k) / C(n, k), computed as a product.

    Valid for 1 <= k <= n. No unbiased estimate exists for k > n: extrapolating beyond the
    attempts drawn needs a model of p, not this formula. Averaged over theorems, it estimates the
    share solved within k attempts.
    """
    _check_counts(c, n)
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# --------------------------------------------------------------------------------------------------
# Clopper–Pearson


EXACT_TERMS = 1000  # up to this many terms, the binomial tail is summed term by term


def _log(x: float, one_minus_x: float) -> float:
    """log x, from 1 - x where that is the more precise."""
    return math.log1p(-one_minus_x) if x > 0.5 else math.log(x)


def _beta_cf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (modified Lentz), which converges
    quickly for x < (a + 1) / (a + b + 2)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000_000):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= c * d
        if abs(c * d - 1.0) < 1e-16:
            return h
    raise ArithmeticError("the continued fraction did not converge")


def _beta_inc(a: float, b: float, x: float, y: float) -> float:
    """The regularized incomplete beta function I_x(a, b); y = 1 - x, given for precision."""
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * _log(x, y) + b * _log(y, x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, y) / b


def _binomial_cdf(c: int, n: int, p: float) -> float:
    """P(X <= c) for X ~ Binomial(n, p): the tail summed in log space, or, when both tails have
    more than ``EXACT_TERMS`` terms, the incomplete beta I_{1-p}(n - c, c + 1)."""
    if c < 0:
        return 0.0
    if c >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    if min(c + 1, n - c) > EXACT_TERMS:
        return _beta_inc(n - c, c + 1, 1.0 - p, p)
    if c > n // 2:  # the upper tail has fewer terms
        return 1.0 - _binomial_cdf(n - c - 1, n, 1.0 - p)
    log_odds = math.log(p) - math.log1p(-p)
    t = n * math.log1p(-p)  # log P(X = 0)
    terms = [t]
    for k in range(1, c + 1):
        t += math.log((n - k + 1) / k) + log_odds
        terms.append(t)
    m = max(terms)
    return math.exp(m) * math.fsum(math.exp(x - m) for x in terms)


def _root(f: Callable[[float], float], lo: float, hi: float) -> float:
    """The root of a decreasing ``f`` with f(lo) > 0 >= f(hi), to the precision of a double."""
    while True:
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            return mid
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid


def clopper_pearson_upper(c: int, n: int, alpha: float = 0.05) -> float:
    """The exact one-sided (1 − alpha) upper confidence bound on p from c successes in n attempts:
    the p at which P(X <= c) = alpha. It is 1 when c = n, and 1 − alpha^(1/n) when c = 0
    (≈ 3 / n at alpha 0.05, "the rule of three")."""
    _check_counts(c, n)
    if c == n:
        return 1.0
    if c == 0:
        return -math.expm1(math.log(alpha) / n)
    return _root(lambda p: _binomial_cdf(c, n, p) - alpha, c / n, 1.0)


def clopper_pearson_lower(c: int, n: int, alpha: float = 0.05) -> float:
    """The exact one-sided (1 − alpha) lower confidence bound on p: the p at which
    P(X >= c) = alpha. It is 0 when c = 0, and alpha^(1/n) when c = n."""
    _check_counts(c, n)
    if c == 0:
        return 0.0
    if c == n:
        return math.exp(math.log(alpha) / n)
    return _root(lambda p: _binomial_cdf(c - 1, n, p) - (1.0 - alpha), 0.0, c / n)


def clopper_pearson_interval(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """The exact two-sided (1 − alpha) interval on p: each bound one-sided at alpha / 2."""
    return clopper_pearson_lower(c, n, alpha / 2), clopper_pearson_upper(c, n, alpha / 2)


def k_to_solve(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """An interval on 1 / p, the expected number of attempts to the first proof:
    [1 / upper, 1 / lower], each end from a one-sided (1 − alpha) Clopper–Pearson bound (so
    jointly a (1 − 2 alpha) interval).

    With c = 0 the right end is infinite: sampling alone bounds the k-to-solve only from below,
    at about n / 3. ``capability-defs`` closed the bracket with 1 / exp(``known_proof_log_p``),
    an estimate of the right end from known proofs instead.
    """
    lower = clopper_pearson_lower(c, n, alpha)
    return 1.0 / clopper_pearson_upper(c, n, alpha), (1.0 / lower if lower > 0 else math.inf)


# --------------------------------------------------------------------------------------------------
# Known proofs


def log_sum_exp(xs: Iterable[float]) -> float:
    """log Σ exp(x), stably; −inf for no terms."""
    v = list(xs)
    m = max(v, default=-math.inf)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in v))


def name_base_marginal(log_probs: Sequence[float], n_bases: int) -> float:
    """log π(y) for a sampler that draws its name base uniformly from ``n_bases`` values, from the
    teacher-forced log π(y | b) of the bases scored: log Σ_b π(y | b) − log n_bases.

    A base not scored (for example, one whose names would pass the vocabulary's last name) counts
    as probability 0, so with fewer bases than ``n_bases`` the result is a lower bound. One base
    alone gives the bound log π(y | b) − log n_bases (``capability-defs``' stage 1, 33 bases).
    """
    return log_sum_exp(log_probs) - math.log(n_bases)


def known_proof_log_p(log_probs: Iterable[float]) -> float:
    """A lower bound on log p from known proofs: log Σ_y π(y), the sum over distinct proofs y of
    the sampler's probability of writing exactly y (teacher-forced, with names marginalised by
    ``name_base_marginal``).

    In exact arithmetic it bounds p from below: p sums π over every sequence that is a proof, and
    the known proofs are some of them (a term may itself be a lower bound). That needs the terms
    to be the sampler's own probabilities (the same temperature and decoding, no length or step
    cap the scorer ignores) and the proofs to be distinct token sequences. In practice it is an
    estimate. ``capability-defs`` found it above the sampled rate's 95 % upper bound on a few
    theorems and put that down to name conditioning (the scorer used canonical names, the sampler
    its own, ≈ 14 % of them renamed by the environment); it called a theorem elicited at budget K
    only if log p >= log 2 − log K, a factor-2 margin. ``create-or-elicit`` measured its
    calibration: above 1e-4, observed / predicted successes were 1.10–1.11; below 1e-4 it was 2–4
    times too low, the known proofs missing much of the mass.
    """
    return log_sum_exp(log_probs)


def good_turing_unseen(counts: Mapping[T, int], known: Container[T] = frozenset()) -> float:
    """The Good–Turing estimate of the probability mass, among successes, of proofs not in the
    sample: f1 / N, where N is the number of successes (``counts``: proof -> times found) and f1 the
    number of proofs found exactly once, leaving out those in ``known``.

    It needs the true multiplicity of every proof over N i.i.d. successes. With ``known`` empty it
    is the classic missing mass M0. ``create-or-elicit`` passed as ``known`` the proofs found
    independently of the sample (by other models and reads), giving M0u, and corrected an
    estimate summed over those and the sample's own proofs, E, to E / (1 − M0u) (undefined at
    M0u = 1). If the known set is independent of the sample, the share of successes whose proof it
    lacks, 1 − K / N, with ``clopper_pearson_interval(K, N)``, needs no multiplicities.
    """
    n = sum(counts.values())
    if n == 0:
        raise ValueError("no successes")
    return sum(1 for y, k in counts.items() if k == 1 and y not in known) / n


# --------------------------------------------------------------------------------------------------
# Certifying "out of reach"


def zero_success_threshold(budget: float, q: float = 0.05, alpha: float = 0.05) -> float:
    """The number of attempts, all failures, above which the one-sided (1 − alpha) upper bound on
    p falls below q / budget: ln(alpha) / ln(1 − q / budget), about 59.9 × budget at
    q = alpha = 0.05.

    Below q / K, the chance of a proof within K attempts is below about q: ``create-or-elicit``
    called a theorem out of reach at budget K (certified) when
    ``clopper_pearson_upper(c, n) < 0.05 / K``. Any number of attempts above the threshold
    certifies this with 0 successes. Valid only if the attempts did not stop early.
    """
    return math.log(alpha) / math.log1p(-q / budget)


def certified_budget(c: int, n: int, q: float = 0.05, alpha: float = 0.05) -> float:
    """The largest budget K certified out of reach by c successes in n attempts:
    q / clopper_pearson_upper(c, n, alpha). Every K below it has an upper bound under q / K."""
    return q / clopper_pearson_upper(c, n, alpha)
