"""Independent enumeration of policy probabilities on synthetic finite pools."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from code.shared_code.selection_metrics import fixed_pool_policy

cases = 0
max_error = 0.0
n = 4
orders = list(itertools.permutations(range(n)))
for scores in itertools.product([0.0, 0.5, 1.0], repeat=n):
    for policy in [dict(count=k) for k in range(1, n + 1)] + [
        dict(threshold=t) for t in [-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1]
    ]:
        sel = np.zeros(n)
        inc = np.zeros(n)
        attempts = 0
        for order in orders:
            seen = []
            for i in order:
                seen.append(i)
                if ("count" in policy and len(seen) == policy["count"]) or (
                    "threshold" in policy and scores[i] >= policy["threshold"]
                ):
                    break
            tied = [i for i in seen if scores[i] == max(scores[j] for j in seen)]
            for i in tied:
                sel[i] += 1 / len(tied) / len(orders)
            for i in seen:
                inc[i] += 1 / len(orders)
            attempts += len(seen) / len(orders)
        actual = fixed_pool_policy(scores, **policy)
        error = max(
            np.max(np.abs(sel - actual["selected_probabilities"])),
            np.max(np.abs(inc - actual["included_probabilities"])),
            abs(attempts - actual["expected_attempts"]),
        )
        max_error = max(max_error, float(error))
        assert error < 1e-12
        cases += 1
report = {
    "pool_size": 4,
    "score_values": [0.0, 0.5, 1.0],
    "score_vectors": 81,
    "policy_cases": cases,
    "permutations_per_case": 24,
    "max_absolute_error": max_error,
    "inputs": "synthetic only, no final-test labels opened",
}
Path(__file__).with_name("threshold_math_independent_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
