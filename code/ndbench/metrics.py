"""Pure-Python + numpy statistics used across the benchmarks (plan §1's `metrics.py`).

No SciPy dependency (not in `code/pyproject.toml`): `clopper_pearson` implements the regularized
incomplete beta function itself (`_betainc`, a standard continued-fraction algorithm) and inverts
it by bisection; `wilson` uses its closed-form normal approximation (`_norm_ppf`, Acklam's
rational approximation to the inverse normal CDF). `pass_at_k` is tested against
`vendor/common.py`'s reference implementation directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from .core.types import ItemResult


class HorizonResult(TypedDict):
    """`horizon()`'s return shape."""

    n_star: int | None
    ci: tuple[int | None, int | None]
    k: int
    tau: float
    curve: list[tuple[int, float]]


class CompositionalityResult(TypedDict):
    """`compositionality_ratio()`'s return shape."""

    R: float
    R_min: float
    ci: tuple[float, float]
    ci_min: tuple[float, float]
    n_used: int


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from `c` correct of `n` samples (Chen et al. 2021), `nan` if `k > n`.

    Identical to `vendor/common.py`'s `pass_at_k` (and `ab3/lengthexp/common.py`'s).
    """
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_curve(
    rows: Sequence[ItemResult], ks: Sequence[int], B: int = 2000, seed: int = 0
) -> list[tuple[int, float, float, float]]:
    """`(k, mean, lo, hi)` for each `k` in `ks`: the mean pass@k over `rows` (one `ItemResult` per
    item, using its `n`/`c`) and a 95% item-level bootstrap CI (resample items with replacement,
    `B` times)."""
    n_items = len(rows)
    if n_items == 0:
        return [(k, float("nan"), float("nan"), float("nan")) for k in ks]
    ns = [r.n for r in rows]
    cs = [r.c for r in rows]
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_items, size=(B, n_items))
    out: list[tuple[int, float, float, float]] = []
    for k in ks:
        vals = np.array([pass_at_k(n, c, k) for n, c in zip(ns, cs, strict=True)])
        mean = float(np.nanmean(vals))
        boot_means = np.nanmean(vals[boot_idx], axis=1)
        lo, hi = (float(x) for x in np.nanpercentile(boot_means, [2.5, 97.5]))
        out.append((k, mean, lo, hi))
    return out


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-14) -> float:
    """Continued-fraction evaluation used by `_betainc` (Numerical-Recipes-style Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function `I_x(a, b)`."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betaincinv(p: float, a: float, b: float) -> float:
    """Solve `_betainc(x, a, b) == p` for `x` in `[0, 1]` by bisection."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _betainc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(c: int, n: int, a: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for `c` successes of `n` trials at level `1 - a`."""
    lo = 0.0 if c == 0 else _betaincinv(a / 2, c, n - c + 1)
    hi = 1.0 if c == n else _betaincinv(1 - a / 2, c + 1, n - c)
    return lo, hi


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def wilson(c: int, n: int, a: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for `c` successes of `n` trials at level `1 - a`."""
    if n == 0:
        return 0.0, 1.0
    z = _norm_ppf(1 - a / 2)
    phat = c / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (centre - adj) / denom, (centre + adj) / denom


def horizon(
    by_n: Mapping[int, Sequence[ItemResult]],
    k: int = 8,
    tau: float = 0.5,
    B: int = 2000,
    seed: int = 0,
) -> HorizonResult:
    """`n*_{k,tau}`: the largest `n` with mean pass@k >= `tau` for every `n' <= n` (plan §2 B2),
    its item-level bootstrap CI, and the `(n, mean pass@k)` curve used to compute it.

    `by_n[n]` is the set of items sampled at family size `n`; the per-`n` statistic is the mean of
    each item's `pass_at_k(item.n, item.c, k)`.
    """
    ns_sorted = sorted(by_n)

    def mean_pass_at(rows: Sequence[ItemResult]) -> float:
        vals = [v for r in rows if not math.isnan(v := pass_at_k(r.n, r.c, k))]
        return float(np.mean(vals)) if vals else float("nan")

    curve = [(n, mean_pass_at(by_n[n])) for n in ns_sorted]

    def n_star(means: Mapping[int, float]) -> int | None:
        best: int | None = None
        for n in ns_sorted:
            m = means.get(n, float("nan"))
            if math.isnan(m) or m < tau:
                break
            best = n
        return best

    observed = n_star(dict(curve))
    rng = np.random.default_rng(seed)
    boots: list[int] = []
    for _ in range(B):
        means_b: dict[int, float] = {}
        for n in ns_sorted:
            rows = by_n[n]
            if not rows:
                means_b[n] = float("nan")
                continue
            idx = rng.integers(0, len(rows), size=len(rows))
            means_b[n] = mean_pass_at([rows[i] for i in idx])
        boot_n = n_star(means_b)
        if boot_n is not None:
            boots.append(boot_n)
    ci: tuple[int | None, int | None] = (
        (int(np.percentile(boots, 2.5)), int(np.percentile(boots, 97.5))) if boots else (None, None)
    )
    return {"n_star": observed, "ci": ci, "k": k, "tau": tau, "curve": curve}


def compositionality_ratio(
    groups: Sequence[tuple[ItemResult, Sequence[ItemResult]]],
    mastery: float = 0.5,
    B: int = 2000,
    seed: int = 0,
) -> CompositionalityResult:
    """`R`/`R_min` (plan §2 B1): the ratio of the joined item's empirical pass@1 to the product
    (resp. minimum) of its parts' pass@1, summed over items where every part is "mastered"
    (`c/n >= mastery`), with an item-level bootstrap CI on each ratio."""

    def phat(r: ItemResult) -> float:
        return r.c / r.n if r.n else float("nan")

    rows = [
        (phat(joined), [phat(p) for p in parts])
        for joined, parts in groups
        if all(phat(p) >= mastery for p in parts)
    ]
    n_used = len(rows)
    if n_used == 0:
        nan = float("nan")
        return {"R": nan, "R_min": nan, "ci": (nan, nan), "ci_min": (nan, nan), "n_used": 0}

    def ratio(rs: Sequence[tuple[float, list[float]]], use_min: bool) -> float:
        num = sum(j for j, _ in rs)
        den = sum((min(ps) if use_min else math.prod(ps)) for _, ps in rs)
        return num / den if den else float("nan")

    R = ratio(rows, False)
    R_min = ratio(rows, True)
    rng = np.random.default_rng(seed)
    boot_R = np.empty(B)
    boot_Rmin = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_used, size=n_used)
        resampled = [rows[i] for i in idx]
        boot_R[b] = ratio(resampled, False)
        boot_Rmin[b] = ratio(resampled, True)
    lo, hi = (float(x) for x in np.nanpercentile(boot_R, [2.5, 97.5]))
    lo_m, hi_m = (float(x) for x in np.nanpercentile(boot_Rmin, [2.5, 97.5]))
    return {"R": R, "R_min": R_min, "ci": (lo, hi), "ci_min": (lo_m, hi_m), "n_used": n_used}


def hazard(rows: Sequence[tuple[int, bool]]) -> list[tuple[int, int, int, float]]:
    """Discrete-time hazard of first failure by absolute line index.

    `rows` is one `(exit_line, failed)` pair per attempt: `exit_line` is the line at which the
    attempt's fate was decided (the first bad line for a failure, or the written length for a
    proof that finished or ran out of tokens without failing at a specific line); `failed` is
    `True` iff the attempt failed exactly at `exit_line`. Returns `(line, at_risk, failed,
    hazard)` for every line from 1 to the largest `exit_line`.
    """
    if not rows:
        return []
    max_line = max(line for line, _ in rows)
    out: list[tuple[int, int, int, float]] = []
    for line in range(1, max_line + 1):
        at_risk = sum(1 for exit_line, _ in rows if exit_line >= line)
        failed = sum(1 for exit_line, f in rows if exit_line == line and f)
        out.append((line, at_risk, failed, failed / at_risk if at_risk else float("nan")))
    return out


def logit_slope(
    x: Sequence[float], n: Sequence[int], c: Sequence[int], max_iter: int = 100, tol: float = 1e-10
) -> tuple[float, float]:
    """Fit `p_i = sigmoid(a + b*x_i)` to grouped binomial data (`n_i` trials, `c_i` successes) by
    IRLS (Fisher scoring). Returns `(b, se_b)`: the slope and its standard error, for "logit slope
    per added line"/"per 100 tokens" (plan §2 B2/B4/B5)."""
    xs = np.asarray(x, dtype=float)
    ns = np.asarray(n, dtype=float)
    cs = np.asarray(c, dtype=float)
    mask = ns > 0
    xs, ns, cs = xs[mask], ns[mask], cs[mask]
    design = np.column_stack([np.ones_like(xs), xs])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = design @ beta
        p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
        w = ns * p * (1 - p)
        z = eta + (cs - ns * p) / (ns * p * (1 - p))
        xtwx = design.T @ (design * w[:, None])
        xtwz = design.T @ (w * z)
        try:
            new_beta = np.linalg.solve(xtwx, xtwz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    eta = design @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
    w = ns * p * (1 - p)
    xtwx = design.T @ (design * w[:, None])
    cov = np.linalg.pinv(xtwx)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def edl(online_nats: Sequence[float], test_nats_final: float) -> float:
    """Excess description length (plan §B7): `sum(online_nats) - n * test_nats_final`, nats."""
    return float(sum(online_nats) - len(online_nats) * test_nats_final)
