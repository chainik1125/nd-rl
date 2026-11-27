# ruff: noqa: E501
"""Paired prompt-bootstrap statistics of the M2 analysis (Turn 5 B.6). Pure numpy.

Every statistic is a function of per-prompt strict-pass shares; a bootstrap resample draws prompts (with replacement) and
recomputes every quantity on the same resample, so the intervals of f, f' and of differences are paired and joint.
"""

from __future__ import annotations

import numpy as np


def _aligned(*rate_dicts: dict[str, float]):
    common = sorted(set.intersection(*[set(d) for d in rate_dicts]))
    return common, [np.array([d[p] for p in common], dtype=float) for d in rate_dicts]


def paired_boot(arrays: list[np.ndarray], fn, reps: int = 2000, seed: int = 0):
    """fn maps the list of per-prompt arrays (resampled, same prompts in each) to a scalar. Returns (point, lo, hi, se)."""
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = fn(arrays)
    vals = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, n, n)
        vals[b] = fn([a[idx] for a in arrays])
    ok = vals[np.isfinite(vals)]
    lo, hi = (
        (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))
        if len(ok)
        else (float("nan"), float("nan"))
    )
    return float(point), lo, hi, float(np.std(ok)) if len(ok) else float("nan")


def frac_fn(a):  # a = [cell, lower, upper] -> (cell - lower) / (upper - lower), unclipped
    d = a[2].mean() - a[1].mean()
    return (a[0].mean() - a[1].mean()) / d if d != 0 else float("nan")


def recovered_fraction(
    cell: dict[str, float],
    lower: dict[str, float],
    upper: dict[str, float],
    reps: int = 2000,
    seed: int = 0,
) -> dict:
    """f = (p_cell - p_lower) / (p_upper - p_lower). lower = stored dpo, upper = stored s2300 for f; lower = stored s0500 for f'."""
    pids, arr = _aligned(cell, lower, upper)
    if not pids:
        return {"n_prompts": 0}
    f, lo, hi, se = paired_boot(arr, frac_fn, reps, seed)
    pc, pl, pu = (float(a.mean()) for a in arr)
    ratio = paired_boot(
        [arr[0], arr[2]],
        lambda a: a[0].mean() / a[1].mean() if a[1].mean() else float("nan"),
        reps,
        seed,
    )
    return {
        "n_prompts": len(pids),
        "p_cell": pc,
        "p_lower": pl,
        "p_upper": pu,
        "f": f,
        "f_lo": lo,
        "f_hi": hi,
        "f_se": se,
        "cai_ratio": ratio[0],
        "cai_ratio_lo": ratio[1],
        "cai_ratio_hi": ratio[2],
    }


def pass_diff(
    a_rates: dict[str, float], b_rates: dict[str, float], reps: int = 2000, seed: int = 0
) -> dict:
    """mean(a) - mean(b) with a paired bootstrap interval (same prompts)."""
    pids, arr = _aligned(a_rates, b_rates)
    if not pids:
        return {"n_prompts": 0}
    d, lo, hi, se = paired_boot(arr, lambda x: x[0].mean() - x[1].mean(), reps, seed)
    return {
        "n_prompts": len(pids),
        "diff": d,
        "lo": lo,
        "hi": hi,
        "se": se,
        "mean_a": float(arr[0].mean()),
        "mean_b": float(arr[1].mean()),
    }
