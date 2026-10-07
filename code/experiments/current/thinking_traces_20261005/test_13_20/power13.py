"""Paired-comparison power for test_13_20 from the calibration rows (600 stratified items, k 8).

    python power13.py --out_dir /scratch/cowifornia/nd-rl/think/runs/pools/test_13_20

For each model pair, `test_21_30/power.py` splits the variance of the per-theorem pass@1 difference
into attempt noise (scales as 1/k) and theorem spread, then projects the standard error to the
full set size and to k 8 / 16.  Rows are restricted to theorems of the final set; the stratified
sample is reweighted to the final set's length mix by projecting each length bin separately.
Writes power.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

T2130 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_21_30")
sys.path.insert(0, T2130)

from power import load  # noqa: E402

PAIRS = [
    ("T0_B0", "T1v1_B0"),
    ("T0_B0", "T1v1_B256"),
    ("T1v1_B0", "T1v1_B256"),
]


def components(A: dict, B: dict, keys: list) -> tuple[float, float, float, int]:
    """-> (mean diff, mean attempt variance of the difference at k_cal, variance of the true
    difference, k_cal) over `keys`."""
    d, w = [], []
    k_cal = A[keys[0]][1]
    for key in keys:
        (ca, ka), (cb, kb) = A[key], B[key]
        d.append(ca / ka - cb / kb)
        w.append(ca * (ka - ca) / (ka * ka * (ka - 1)) + cb * (kb - cb) / (kb * kb * (kb - 1)))
    n = len(d)
    mean = sum(d) / n
    var_d = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    wi = sum(w) / n
    return mean, wi, max(0.0, var_d - wi), k_cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    D = a.out_dir
    final = {}
    with open(f"{D}/test_13_20.jsonl") as f:
        for line in f:
            r = json.loads(line)
            final[r["prompt"]] = r["reference_lines"]
    rows = {
        "T0_B0": load(f"{D}/calib/T0_s0/rows_plain_B0.jsonl"),
        "T1v1_B0": load(f"{D}/calib/T1v1_s0/rows_plain_B0.jsonl"),
        "T1v1_B256": load(f"{D}/calib/T1v1_s0/rows_plain_B256.jsonl"),
    }
    n_full = collections.Counter(final.values())
    N = sum(n_full.values())
    out = dict(N_final=N, pairs={})
    for an, bn in PAIRS:
        A, B = rows[an], rows[bn]
        keys = [p for p in A if p in B and p in final]
        by = collections.defaultdict(list)
        for p in keys:
            by[final[p]].append(p)
        res = dict(n_sample=len(keys))
        mean, wi, vd, k_cal = components(A, B, keys)
        res.update(diff_pp=100 * mean, attempt_var_pp2_at_k8=1e4 * wi, theorem_var_pp2=1e4 * vd)
        grid = {}
        for k in (8, 16):
            # stratified projection: Var(mean) = sum_L (n_L / N)^2 (W_L * k_cal / k + V_L) / n_L
            vf = vp = 0.0
            for L, ps in by.items():
                if len(ps) < 2:
                    continue
                _, w_l, v_l, _ = components(A, B, ps)
                share = n_full[L] / N
                vf += share**2 * (w_l * k_cal / k) / n_full[L]
                vp += share**2 * (w_l * k_cal / k + v_l) / n_full[L]
            se_f, se_p = math.sqrt(vf), math.sqrt(vp)
            grid[f"k{k}"] = dict(
                se_fixed_pp=100 * se_f,
                se_pop_pp=100 * se_p,
                sig95_fixed_pp=196 * se_f,
                sig95_pop_pp=196 * se_p,
                mde80_fixed_pp=280 * se_f,
                mde80_pop_pp=280 * se_p,
            )
        res["projection_full_set"] = grid
        out["pairs"][f"{an}_vs_{bn}"] = res
    with open(f"{D}/power.json", "w") as f:
        json.dump(out, f, indent=1)
    for name, res in out["pairs"].items():
        g8, g16 = res["projection_full_set"]["k8"], res["projection_full_set"]["k16"]
        print(
            f"{name}: diff {res['diff_pp']:.2f} pp; k8 SE fixed {g8['se_fixed_pp']:.2f} / pop "
            f"{g8['se_pop_pp']:.2f} (MDE80 {g8['mde80_fixed_pp']:.2f} / {g8['mde80_pop_pp']:.2f}); "
            f"k16 MDE80 {g16['mde80_fixed_pp']:.2f} / {g16['mde80_pop_pp']:.2f}"
        )


if __name__ == "__main__":
    main()
