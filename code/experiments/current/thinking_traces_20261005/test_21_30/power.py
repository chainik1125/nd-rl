"""Power of a paired pass@1 comparison on the test set, from calibration rows (`code.thinking.eval`
rows_*.jsonl).

    python power.py --a EVAL_T0/rows_plain_B0.jsonl --b EVAL_T1v1/rows_plain_B256.jsonl \
        --out power.json

Per theorem i, model m has c_mi successes in k attempts; p_mi = c_mi / k is the unbiased pass@1
estimate and s2_mi = c(k-c) / (k^2 (k-1)) the unbiased variance of that estimate. The paired
difference over N theorems is D = mean_i (p_ai - p_bi). Its variance has two parts:
  * attempt noise   W(k)  = mean_i (s2_ai + s2_bi) / N, which scales as 1/k
                          (per-theorem binomial variance p(1-p)/k, unbiasedly estimated at the
                          calibration k);
  * theorem spread  B     = var_i(delta_i) / N, delta_i the true per-theorem difference,
                          estimated as var_i(p_ai - p_bi) - mean_i(s2_ai + s2_bi), floored at 0.
SE_fixed(k) = sqrt(W(k)) is the standard error for "these two checkpoints differ on this sealed set"
(the theorems are fixed, only sampling is random); SE_pop(k) = sqrt(W(k) + B) generalises to new
theorems from the same generator. The 95% two-sided significance threshold is 1.96 SE; the
difference detectable with 80% power is 2.80 SE. Extrapolation to another k rescales W by k_cal / k;
to another N rescales both terms by N_cal / N.
"""

from __future__ import annotations

import argparse
import json
import math


def load(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["prompt"]] = (r["n_ok"], r["k"])
    return out


def analyse(A: dict, B: dict) -> dict:
    keys = sorted(set(A) & set(B))
    n = len(keys)
    pa, pb, sa, sb, d = [], [], [], [], []
    for key in keys:
        (ca, ka), (cb, kb) = A[key], B[key]
        x, y = ca / ka, cb / kb
        pa.append(x)
        pb.append(y)
        sa.append(ca * (ka - ca) / (ka * ka * (ka - 1)))
        sb.append(cb * (kb - cb) / (kb * kb * (kb - 1)))
        d.append(x - y)
    k_cal = A[keys[0]][1]
    mean_d = sum(d) / n
    var_d = sum((x - mean_d) ** 2 for x in d) / (n - 1)
    w_i = sum(sa) / n + sum(sb) / n  # mean per-theorem attempt variance of the difference, at k_cal
    var_delta = max(0.0, var_d - w_i)
    res = dict(
        n=n,
        k_cal=k_cal,
        pass1_a=100 * sum(pa) / n,
        pass1_b=100 * sum(pb) / n,
        diff=100 * mean_d,
        sd_diff_per_theorem=100 * math.sqrt(var_d),
        attempt_var_per_theorem_at_kcal=1e4 * w_i,
        theorem_var_of_true_diff=1e4 * var_delta,
        frac_attempt_noise=w_i / var_d if var_d > 0 else None,
        solved_any_a=sum(x > 0 for x in pa),
        solved_any_b=sum(x > 0 for x in pb),
        solved_any_either=sum((x > 0) or (y > 0) for x, y in zip(pa, pb, strict=True)),
    )
    grid = {}
    for k in (8, 16, 32):
        for N in sorted({n, 600, 1000, 2000, 2775, 3000, 5000}):
            W = w_i * k_cal / k / N
            Bv = var_delta / N
            se_f, se_p = math.sqrt(W), math.sqrt(W + Bv)
            grid[f"k{k}_N{N}"] = dict(
                k=k,
                N=N,
                se_fixed_pp=100 * se_f,
                se_pop_pp=100 * se_p,
                sig95_fixed_pp=196 * se_f,
                sig95_pop_pp=196 * se_p,
                mde80_fixed_pp=280 * se_f,
                mde80_pop_pp=280 * se_p,
            )
    res["grid"] = grid
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyse(load(a.a), load(a.b))
    res["a"], res["b"] = a.a, a.b
    with open(a.out, "w") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=1))
    for key, g in res["grid"].items():
        print(key, {k: round(v, 3) for k, v in g.items() if k not in ("k", "N")})


if __name__ == "__main__":
    main()
