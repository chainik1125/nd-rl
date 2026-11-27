# ruff: noqa: E501
"""W pass (Turn 5 B.2 steps 2-3): weight-space statistics of the 448 matrices of dpo, s0100, s0500, s2300, the gate verdict,
and the rank-1 / rank-16 factor files the adapter builder reads.

  python3 wpass.py --models-dir /workspace/models --out runs/W/out [--device cuda] [--layers 64] [--iters 6]

Per matrix and per Delta in {d100, d500, d2300} (bf16 differences formed exactly in fp32, streamed one tensor at a time):
Frobenius norm, top singular values by randomized SVD (Halko, >= 6 subspace iterations, QR-stabilised), the sign-randomized
null sigma_1 (d2300 and d500) (Delta o S, S a seeded +-1 matrix), ratio = sigma_1 / null. For d2300 also energy sigma_1^2 / ||Delta||_F^2,
alpha = ||Delta||_F / sigma_1 (Cai Eq. 3, Frobenius reading) and the top-16 factors. Sign-aligned |cos| of the top singular
vectors across d100 / d500 / d2300, cos(Delta_500, Delta_2300) with fp64 inner products, gamma_proj, and
||dpo + gamma Delta_500 - s2300|| / ||Delta_2300|| for gamma in {3.17, 3.5, 4.6} (reported, never used to choose).
Exact-SVD check: sigma_1 of Delta_2300 of the first gate / q / k matrix and the null of the first gate matrix, exact vs
randomized, within 0.5 %, else exit 2.
Gate (median of ratio_2300 weighted by ||Delta_2300||_F): < 1.1 -> exit 20 STOP; [1.1, 1.5) noisy and >= 1.5 signal both exit 0
and write the band to <out>/w_band.txt (signal | noisy | stop). Other exits: 2 exact check failed (rerun once with --iters 12; then
with --exact-fallback), 3 input error. Turn 6 D6: top-16 + null for d2300, top-1 + null for d500, top-1 only for d100.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DELTAS,
    GAMMAS_REPORT,
    MATRIX_TYPES,
    Checkpoint,
    matrix_names,
    tensor_seed,
)

BAND_STOP, BAND_SIGNAL = 1.1, 1.5
NULL_SEED = 710100


def rsvd(A: torch.Tensor, k: int, iters: int = 6, oversample: int = 10, seed: int = 0):
    """Randomized SVD with QR-stabilised subspace iteration. Returns U [m,k], S [k], V [n,k] (A ~ U diag(S) V^T)."""
    m, n = A.shape
    q = min(k + oversample, m, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn(n, q, generator=g, dtype=A.dtype).to(A.device)
    Y = A @ omega
    for _ in range(iters):
        Y, _ = torch.linalg.qr(Y)
        Z, _ = torch.linalg.qr(A.T @ Y)
        Y = A @ Z
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ A
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], S[:k], Vh[:k].T


def sign_matrix(shape, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0, 2, shape, generator=g, dtype=torch.int8).to(torch.float32) * 2 - 1).to(
        device
    )


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def verdict(ratio_median: float) -> tuple[int, str]:
    if ratio_median < BAND_STOP:
        return (
            20,
            "STOP rank-1: sigma_1 is indistinguishable from the sign-randomized null (skip A1, A1u, A2; go to ExPO)",
        )
    if ratio_median < BAND_SIGNAL:
        return (
            11,
            "NOISY top direction (implied cosine with a planted rank-1 about 0.58-0.84 under an iid-noise model); cells run and are labelled so",
        )
    return 0, "SIGNAL-DOMINATED top direction (ratio >= 1.5)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--ckpts", default="dpo,s0100,s0500,s2300")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--exact", type=int, default=3, help="number of exact-SVD checks (0 to skip)")
    ap.add_argument("--exact-tol", type=float, default=0.005)
    ap.add_argument(
        "--exact-fallback",
        action="store_true",
        help="exact sigma_1 and null of every Delta_2300 (after a second failed check)",
    )
    a = ap.parse_args(argv)
    if a.iters < 6:
        print("iters must be >= 6", file=sys.stderr)
        return 3
    torch.set_num_threads(1 if a.device == "cpu" else torch.get_num_threads())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.ckpts.split(",")
    try:
        ck = {n: Checkpoint(Path(a.models_dir) / n) for n in names}
        mats = matrix_names(a.layers)
        for n in mats:
            for c in ck.values():
                c.shard_of(n)
        for c in ck.values():
            c.get(mats[0])  # opens the first shard: a missing checkpoint fails here, not mid-run
    except Exception as e:  # noqa: BLE001
        print(f"input error: {e!r}", file=sys.stderr)
        return 3
    dev = a.device
    rows, f2300, f_small = [], {}, {}
    acc = {"a2": 0.0, "b2": 0.0, "ab": 0.0}
    tot_g = {g: 0.0 for g in GAMMAS_REPORT}
    exact_targets = {
        f"model.layers.0.{MATRIX_TYPES[i][0]}.weight" for i in (4, 0, 1)
    }  # gate (largest), q, k
    exact_log, exact_ok = [], True
    t0 = time.time()
    for mi, name in enumerate(mats):
        W = {c: ck[c].get(name).to(torch.float32).to(dev) for c in names}
        D = {key: W[t] - W[a_] for key, (a_, t) in DELTAS.items()}
        fro = {k_: float(torch.linalg.norm(d)) for k_, d in D.items()}
        row = {"name": name, "out": D["d2300"].shape[0], "in": D["d2300"].shape[1]}
        top, null = {}, {}
        for key, d in D.items():
            seed = tensor_seed(NULL_SEED, name + key)
            k_use = a.k if key == "d2300" else 1  # Turn 6 D6: top-16 (d2300), top-1 (d500, d100)
            U, S, V = rsvd(d, k_use, a.iters, seed=seed)
            top[key] = (U, S, V)
            s1 = float(S[0])
            row[f"fro_{key}"] = fro[key]
            if key == "d100":  # top-1 only, no null
                row["sigma1_d100"] = s1
                continue
            Sg = sign_matrix(tuple(d.shape), seed + 1, dev)
            null[key] = float(rsvd(d * Sg, 1, a.iters, seed=seed + 2)[1][0])
            if (
                key == "d2300" and a.exact_fallback
            ):  # Turn 6 D4(c): exact sigma_1 and null for every Delta_2300
                s1 = float(torch.linalg.svdvals(d)[0])
                null[key] = float(torch.linalg.svdvals(d * Sg)[0])
            row[f"sigma1_{key}"] = s1
            row[f"null_{key}"] = null[key]
            row[f"ratio_{key}"] = s1 / null[key] if null[key] > 0 else float("nan")
            if key == "d2300" and name in exact_targets and a.exact and not a.exact_fallback:
                e = float(torch.linalg.svdvals(d)[0])
                checks = [("sigma1_d2300", e, float(S[0]))]
                if "gate_proj" in name:
                    en = float(torch.linalg.svdvals(d * Sg)[0])
                    checks.append(("null_d2300", en, null[key]))
                for lab, ex, rs in checks:
                    rel = abs(ex - rs) / ex
                    exact_log.append(
                        {"name": name, "what": lab, "exact": ex, "randomized": rs, "rel_err": rel}
                    )
                    exact_ok &= rel <= a.exact_tol
        row["alpha_d2300"] = row["fro_d2300"] / row["sigma1_d2300"]
        row["energy_d2300"] = (row["sigma1_d2300"] / row["fro_d2300"]) ** 2
        # sign alignment of u1 (and v1 with it) to d2300's, then |cos|
        u2, v2 = top["d2300"][0][:, 0], top["d2300"][2][:, 0]
        small = {}
        for key in ("d100", "d500"):
            u, v = top[key][0][:, 0], top[key][2][:, 0]
            s = 1.0 if float(u @ u2) >= 0 else -1.0
            small[key] = (u * s, v * s)
            row[f"cos_u_{key}_d2300"] = abs(float(u @ u2))
            row[f"cos_v_{key}_d2300"] = abs(float(v @ v2))
        row["cos_u_d100_d500"] = abs(float(small["d100"][0] @ small["d500"][0]))
        # fp64 inner products of the full deltas
        d5, d23 = D["d500"].double(), D["d2300"].double()
        ab, a2, b2 = float((d5 * d23).sum()), float((d5 * d5).sum()), float((d23 * d23).sum())
        row["cos_delta500_delta2300"] = (
            ab / (a2**0.5 * b2**0.5) if a2 > 0 and b2 > 0 else float("nan")
        )
        row["gamma_proj"] = ab / a2 if a2 > 0 else float("nan")
        acc["a2"] += a2
        acc["b2"] += b2
        acc["ab"] += ab
        for g in GAMMAS_REPORT:
            tot_g[g] += g * g * a2 - 2 * g * ab + b2
        rows.append(row)
        f2300[name + "::U"] = top["d2300"][0].cpu().contiguous()
        f2300[name + "::S"] = top["d2300"][1].cpu().contiguous()
        f2300[name + "::V"] = top["d2300"][2].cpu().contiguous()
        for key in ("d100", "d500"):
            f_small[name + f"::{key}::u"] = small[key][0].cpu().contiguous()
            f_small[name + f"::{key}::v"] = small[key][1].cpu().contiguous()
            f_small[name + f"::{key}::sigma"] = top[key][1][:1].cpu().contiguous()
        del W, D, d5, d23
        if (mi + 1) % 28 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] {mi + 1}/{len(mats)} matrices, {time.time() - t0:.0f}s",
                flush=True,
            )
    from safetensors.torch import save_file

    save_file(f2300, str(out / "factors_d2300_k16.safetensors"))
    save_file(f_small, str(out / "factors_d100_d500_r1.safetensors"))
    with open(out / "w_matrices.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wts = np.array(
        [r["fro_d2300"] ** 2 for r in rows]
    )  # weight by ||Delta||_F for the median; shares by squared norm
    wf = np.sqrt(wts)

    def wmed(col):
        return weighted_median(np.array([r[col] for r in rows]), wf)

    r2300 = wmed("ratio_d2300")
    share_low = float(wts[np.array([r["ratio_d2300"] < BAND_STOP for r in rows])].sum() / wts.sum())
    code, text = verdict(r2300)
    band = {0: "signal", 11: "noisy", 20: "stop"}[code]
    if not exact_ok and a.exact:
        code = 2
    summ = {
        "n_matrices": len(rows),
        "iters": a.iters,
        "k": a.k,
        "device": dev,
        "ckpts": names,
        "ratio_d2300_weighted_median": r2300,
        "ratio_d500_weighted_median": wmed("ratio_d500"),
        "share_norm2_d2300_in_matrices_ratio_below_1p1": share_low,
        "alpha_d2300_weighted_median": wmed("alpha_d2300"),
        "alpha_d2300_median": float(np.median([r["alpha_d2300"] for r in rows])),
        "energy_d2300_weighted_median": wmed("energy_d2300"),
        "cos_delta500_delta2300_total": acc["ab"] / (acc["a2"] ** 0.5 * acc["b2"] ** 0.5),
        "gamma_proj_total": acc["ab"] / acc["a2"],
        "dist_ratio_to_s2300": {str(g): (tot_g[g] / acc["b2"]) ** 0.5 for g in GAMMAS_REPORT},
        "norm_ratio_2300_over_500": (acc["b2"] / acc["a2"]) ** 0.5,
        "persistence_cos_u_d100_d500_weighted_median": wmed("cos_u_d100_d500"),
        "persistence_cos_u_d500_d2300_weighted_median": wmed("cos_u_d500_d2300"),
        "exact_checks": exact_log,
        "exact_ok": exact_ok,
        "verdict_code": code,
        "verdict": text,
        "bands": {"stop_below": BAND_STOP, "signal_from": BAND_SIGNAL},
    }
    (out / "w_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "exact_checks"}, indent=1))
    print(f"exact checks: {'OK' if exact_ok else 'FAILED'} ({len(exact_log)} comparisons)")
    (out / "w_band.txt").write_text(band + "\n")
    print(f"W VERDICT band={band} exit={code}: {text}")
    return (
        0 if code == 11 else code
    )  # Turn 6 D4(b): both run bands exit 0, the band is in w_band.txt; non-zero only for STOP (20) and errors


if __name__ == "__main__":
    sys.exit(main())
