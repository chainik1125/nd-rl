"""Compare joint-budget enumeration to an independent one-dimensional LP solution."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(
    0, str(ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation")
)
from fit_budget_baselines import joint_mixture  # noqa: E402 - runtime source path

rng = np.random.default_rng(6100607)
maxerr = 0.0
maxbudget = 0.0
cases = 0
for _ in range(200):
    counts = np.array([1.0, 2.0, 4.0, 8.0])
    work = np.cumsum(rng.uniform(100, 1000, 4))
    accuracy = rng.uniform(0, 1, 4)
    reward = accuracy - 0.03 * (counts - 1)
    w = rng.dirichlet(np.ones(4))
    target = np.array([1.0, counts @ w, work @ w])
    A = np.array([np.ones(4), counts, work])
    part = np.linalg.lstsq(A, target, rcond=None)[0]
    null = np.linalg.svd(A)[2][-1]
    lower = -np.inf
    upper = np.inf
    for p, v in zip(part, null, strict=False):
        if v > 0:
            lower = max(lower, -p / v)
        elif v < 0:
            upper = min(upper, -p / v)
    t = upper if reward @ null >= 0 else lower
    opt = part + t * null
    expected = reward @ opt
    fixed = {
        str(int(k)): {"attempts": k, "total_forward_tokens": x, "reward": r}
        for k, x, r in zip(counts, work, reward, strict=False)
    }
    got = joint_mixture(fixed, {"attempts": target[1], "total_forward_tokens": target[2]})
    assert got["feasible"]
    actual = np.array([got["weights"][str(int(k))] for k in counts])
    maxerr = max(maxerr, abs(reward @ actual - expected))
    maxbudget = max(maxbudget, float(np.max(np.abs(A @ actual - target))))
    assert abs(reward @ actual - expected) < 1e-7
    assert np.min(actual) > -1e-10
    cases += 1
report = {
    "random_feasible_joint_budget_cases": cases,
    "max_reward_error_vs_independent_nullspace_LP": float(maxerr),
    "max_absolute_budget_residual": maxbudget,
    "seed": 6100607,
    "inputs": "synthetic only; no test outcomes",
}
Path(__file__).with_name("budget_mixture_math_independent_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
