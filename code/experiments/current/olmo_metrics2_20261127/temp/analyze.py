"""M1 (`temp`) analysis: every statistic of Turn 5 Part 3 sections 4 and 6, to summary.json and summary.csv. CPU, one core.

Cells (graded.jsonl of each pod job; rows need pid, chunk, i, strict, n_tok, finish, think_end):
  --c1 (dpo, T 0.6 / top_p 0.95, arm A x2 + reach xk)   --s5r (s0500 card, reach xk)   --c3 (s2300 card, arm A x2)
  --c4 (dpo pure T 0.6)   --c2 (dpo T 0.8)   --e0 (M3's E0', accepted only if --e0-accepted)   --c1b-draws k
Statistics (prompt bootstrap, 10,000 resamples, seed 0, the same resamples inside each statistic; loops, truncations and a
missing </think> count as failures):
  arm A: pass@1 (first 2 draws per prompt), pass@2, Delta vs the stored T 1.0 draws of the same checkpoint (dpo 16, s0500 8,
         s2300 8), share_RL = Delta(dpo C1) / D with D = mean_p(B_p s2300 - B_p dpo) in ONE joint bootstrap, matched gap
         G(C1) = mean_p(N_p s2300 C3 - N_p dpo C1) against G(1.0) = D, the secondary split (dpo stored draws 1-8 define
         ever-passed / never-passed, baseline from draws 9-16 only), baseline-rate bins, constraint-family breakdown,
         token mean / p50 / p90 / p99, share cut at the cap, share without </think>, pass@1 among finishing draws.
  reach: rate over draws, "any pass in k", d_p = rate(C1b) - rate(E0') (E0' only when accepted, else the proxy 0.13 %),
         share_reach = (r(C1b) - r(E0')) / (r(s2300 stored) - r(E0')), s0500 reach against its stored 32-draw rate and the
         first-k stored draws.
Usage: analyze.py --c1 G [--s5r G] [--c3 G] [--c4 G] [--c2 G] [--e0 G --e0-accepted] [--c1b-draws 8] --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import temp_common as tc  # noqa: E402

PROXY_RATE = 0.0013  # s0100 fresh T 1.0 reach rate, 2 of 1,536 (a proxy for the dpo reach rate when E0' is unavailable)
PROBLEMS = tc.DAN / "olmo_ifomega" / "problems.jsonl"


def rate(v: list[dict]) -> float:
    return float(np.mean([tc.strict_of(r) for r in v])) if v else float("nan")


def cell_arm(rows: list[dict], n: int = 2) -> dict[str, list[dict]]:
    arm = set(tc.dc.arm_a())
    return {p: v[:n] for p, v in tc.by_pid(rows).items() if p in arm}


def cell_reach(rows: list[dict], k: int) -> dict[str, list[dict]]:
    reach = set(tc.dc.reach())
    return {p: v[:k] for p, v in tc.by_pid(rows).items() if p in reach}


def vec(cell: dict, pids: list[str]) -> np.ndarray:
    return np.array([rate(cell[p]) for p in pids])


def boot(v: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    b = v[idx].mean(axis=1)
    lo, hi = tc.ci(b)
    return float(v.mean()), lo, hi


def token_stats(draws: list[dict]) -> dict:
    t = np.array([r["n_tok"] for r in draws], float)
    return {
        "n_draws": len(t),
        "mean_tok": float(t.mean()),
        "p50_tok": float(np.percentile(t, 50)),
        "p90_tok": float(np.percentile(t, 90)),
        "p99_tok": float(np.percentile(t, 99)),
        "cut_pct": 100 * float(np.mean([tc.is_cut(r) for r in draws])),
        "no_think_end_pct": 100
        * float(np.mean([not int(bool(r.get("think_end", 1))) for r in draws])),
        "pass_among_finishing_pct": 100
        * float(np.mean([tc.strict_of(r) for r in draws if not tc.is_cut(r)] or [np.nan])),
    }


def families() -> dict[str, list[str]]:
    fam: dict[str, list[str]] = {}
    for ln in open(PROBLEMS):
        r = json.loads(ln)
        if r["pid"].startswith("ifbench"):
            ids = r["instruction_id_list"]
            ids = json.loads(ids.replace("'", '"')) if isinstance(ids, str) else ids
            for f in sorted({i.split(":")[0] for i in ids}):
                fam.setdefault(f, []).append(r["pid"])
    return fam


def arm_a_block(
    label: str, rows: list[dict], base: dict[str, list[dict]], idx, pids: list[str]
) -> dict:
    cell = cell_arm(rows, 2)
    N = vec(cell, pids)
    B = vec(base, pids)
    flat = [r for v in cell.values() for r in v]
    out = {"cell": label, "n_prompts": len(pids)}
    m, lo, hi = boot(N, idx)
    out["pass1_pct"] = 100 * m
    out["pass1_lo95"], out["pass1_hi95"] = 100 * lo, 100 * hi
    out["pass2_pct"] = 100 * float(np.mean([any(tc.strict_of(r) for r in cell[p]) for p in pids]))
    m, lo, hi = boot(N - B, idx)
    out["delta_pts"], out["delta_lo95"], out["delta_hi95"] = 100 * m, 100 * lo, 100 * hi
    out["tokens"] = token_stats(flat)
    # baseline-rate bins (stored baseline per-prompt rate)
    bins = [(0.0, 0.0), (0.0001, 0.25), (0.2501, 0.75), (0.7501, 1.0)]
    out["bins"] = {}
    for lo_b, hi_b in bins:
        sel = np.where((lo_b <= B) & (hi_b >= B))[0]
        if len(sel):
            out["bins"][f"base_{lo_b:g}-{hi_b:g}"] = {
                "n_prompts": int(len(sel)),
                "baseline_pct": 100 * float(B[sel].mean()),
                "cell_pct": 100 * float(N[sel].mean()),
                "delta_pts": 100 * float((N[sel] - B[sel]).mean()),
            }
    out["families"] = {}
    pos = {p: i for i, p in enumerate(pids)}
    for f, ps in sorted(families().items()):
        sel = [pos[p] for p in ps if p in pos]
        if len(sel) >= 3:
            out["families"][f] = {
                "n_prompts": len(sel),
                "baseline_pct": 100 * float(B[sel].mean()),
                "cell_pct": 100 * float(N[sel].mean()),
                "delta_pts": 100 * float((N[sel] - B[sel]).mean()),
            }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("c1", "s5r", "c3", "c4", "c2", "e0"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--e0-accepted", action="store_true")
    ap.add_argument("--c1b-draws", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pids = sorted(tc.dc.arm_a())
    idx = tc.boot_idx(len(pids))
    st = {c: tc.stored_arm_a(c) for c in ("dpo", "s0500", "s2300")}
    Bdpo, B2300 = vec(st["dpo"], pids), vec(st["s2300"], pids)
    D = B2300 - Bdpo
    res: dict = {
        "stored": {
            "dpo_pass1_pct": 100 * Bdpo.mean(),
            "s0500_pass1_pct": 100 * vec(st["s0500"], pids).mean(),
            "s2300_pass1_pct": 100 * B2300.mean(),
            "D_pts": 100 * D.mean(),
            "D_ci95_pts": [100 * x for x in tc.ci(D[idx].mean(axis=1))],
        }
    }
    # secondary split: groups from dpo draws 1-8, baseline from draws 9-16
    ever = [p for p in pids if any(tc.strict_of(r) for r in st["dpo"][p][:8])]
    never = [p for p in pids if p not in set(ever)]
    res["split_dpo_draws_1_8"] = {
        "ever_passed": {
            "n_prompts": len(ever),
            "baseline_draws_9_16_pct": 100 * float(np.mean([rate(st["dpo"][p][8:]) for p in ever])),
        },
        "never_passed": {
            "n_prompts": len(never),
            "baseline_draws_9_16_pct": 100
            * float(np.mean([rate(st["dpo"][p][8:]) for p in never])),
        },
    }
    arm_cells = {}
    if a.c1:
        c1 = tc.jl(a.c1)
        blk = arm_a_block("C1 dpo T0.6/top_p0.95", c1, st["dpo"], idx, pids)
        N = vec(cell_arm(c1, 2), pids)
        num = N - Bdpo
        bs = (num[idx].mean(axis=1)) / (D[idx].mean(axis=1))
        lo, hi = tc.ci(bs)
        blk["share_RL"] = {
            "point": float(num.mean() / D.mean()),
            "lo95": lo,
            "hi95": hi,
            "note": "numerator and denominator in one joint prompt bootstrap",
        }
        base_late = {p: st["dpo"][p][8:] for p in pids}
        blk["split"] = {}
        for name, grp in (("ever_passed", ever), ("never_passed", never)):
            d = np.array([rate(cell_arm(c1, 2)[p]) - rate(base_late[p]) for p in grp])
            ii = tc.boot_idx(len(d))
            m, l, h = boot(d, ii)
            blk["split"][name] = {
                "n_prompts": len(grp),
                "delta_pts_vs_draws_9_16": 100 * m,
                "lo95": 100 * l,
                "hi95": 100 * h,
            }
        arm_cells["C1"] = blk
        # reach: C1b
        k = a.c1b_draws
        c1b = cell_reach(c1, k)
        reach_p = sorted(tc.dc.reach())
        r_c1 = float(np.mean([rate(c1b[p]) for p in reach_p]))
        ridx = tc.boot_idx(len(reach_p))
        rb = {
            "k": k,
            "rate_pct": 100 * r_c1,
            "passes": int(sum(tc.strict_of(r) for v in c1b.values() for r in v)),
            "draws": int(sum(len(v) for v in c1b.values())),
            "prompts_with_pass": int(sum(any(tc.strict_of(r) for r in v) for v in c1b.values())),
            "tokens": token_stats([r for v in c1b.values() for r in v]),
        }
        s2300r = float(np.mean([rate(v) for v in tc.stored_reach("s2300").values()]))
        if a.e0 and a.e0_accepted:
            e0 = cell_reach(tc.jl(a.e0), k)
            base_v, base_name = np.array([rate(e0[p]) for p in reach_p]), "E0' (first k draws)"
        else:
            base_v, base_name = (
                np.full(len(reach_p), PROXY_RATE),
                "proxy 0.13 % (s0100 fresh T 1.0)",
            )
        cv = np.array([rate(c1b[p]) for p in reach_p])
        d = cv - base_v
        m, l, h = boot(d, ridx)
        den = s2300r - base_v.mean()
        shr = (cv[ridx].mean(axis=1) - base_v[ridx].mean(axis=1)) / (
            s2300r - base_v[ridx].mean(axis=1)
        )
        lo2, hi2 = tc.ci(shr)
        rb.update(
            {
                "baseline": base_name,
                "baseline_rate_pct": 100 * float(base_v.mean()),
                "d_pts": 100 * m,
                "d_lo95": 100 * l,
                "d_hi95": 100 * h,
                "s2300_stored_rate_pct": 100 * s2300r,
                "share_reach": {
                    "point": float((cv.mean() - base_v.mean()) / den),
                    "lo95": lo2,
                    "hi95": hi2,
                },
            }
        )
        res["reach_C1b"] = rb
    for label, key, ck in (
        ("C3 s2300 T0.6/top_p0.95", "c3", "s2300"),
        ("C4 dpo T0.6/top_p1.0", "c4", "dpo"),
        ("C2 dpo T0.8/top_p1.0", "c2", "dpo"),
    ):
        if getattr(a, key):
            arm_cells[key.upper()] = arm_a_block(label, tc.jl(getattr(a, key)), st[ck], idx, pids)
    if a.c3 and a.c1:
        N3 = vec(cell_arm(tc.jl(a.c3), 2), pids)
        N1 = vec(cell_arm(tc.jl(a.c1), 2), pids)
        g = N3 - N1
        m, l, h = boot(g, idx)
        res["matched_gap"] = {
            "G_C1_pts": 100 * m,
            "lo95": 100 * l,
            "hi95": 100 * h,
            "G_T1_0_pts": 100 * float(D.mean()),
        }
    if a.s5r:
        k = a.c1b_draws
        s5 = cell_reach(tc.jl(a.s5r), k)
        stor = tc.stored_reach("s0500")
        reach_p = sorted(tc.dc.reach())
        ridx = tc.boot_idx(len(reach_p))
        Ns = np.array([rate(s5[p]) for p in reach_p])
        Bs = np.array([rate(stor[p]) for p in reach_p])
        Bk = np.array([rate(stor[p][:k]) for p in reach_p])
        m, l, h = boot(Ns - Bs, ridx)
        m2, l2, h2 = boot(Ns - Bk, ridx)
        res["reach_S5R"] = {
            "k": k,
            "rate_pct": 100 * float(Ns.mean()),
            "stored32_rate_pct": 100 * float(Bs.mean()),
            "stored_first_k_rate_pct": 100 * float(Bk.mean()),
            "d_vs_stored32_pts": 100 * m,
            "lo95": 100 * l,
            "hi95": 100 * h,
            "d_vs_first_k_pts": 100 * m2,
            "lo95_k": 100 * l2,
            "hi95_k": 100 * h2,
            "any_pass_prompts": int(sum(any(tc.strict_of(r) for r in s5[p]) for p in reach_p)),
            "stored_first_k_any_pass_prompts": int(
                sum(any(tc.strict_of(r) for r in stor[p][:k]) for p in reach_p)
            ),
            "tokens": token_stats([r for v in s5.values() for r in v]),
        }
    res["arm_a"] = arm_cells
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "set", "stat", "value", "lo95", "hi95"])
        for k, b in arm_cells.items():
            w.writerow([k, "arm A", "pass1_pct", b["pass1_pct"], b["pass1_lo95"], b["pass1_hi95"]])
            w.writerow([k, "arm A", "delta_pts", b["delta_pts"], b["delta_lo95"], b["delta_hi95"]])
            if "share_RL" in b:
                w.writerow(
                    [
                        k,
                        "arm A",
                        "share_RL",
                        b["share_RL"]["point"],
                        b["share_RL"]["lo95"],
                        b["share_RL"]["hi95"],
                    ]
                )
        if "reach_C1b" in res:
            r = res["reach_C1b"]
            w.writerow(["C1b", "reach", "rate_pct", r["rate_pct"], "", ""])
            w.writerow(["C1b", "reach", "d_pts_vs_baseline", r["d_pts"], r["d_lo95"], r["d_hi95"]])
            w.writerow(
                [
                    "C1b",
                    "reach",
                    "share_reach",
                    r["share_reach"]["point"],
                    r["share_reach"]["lo95"],
                    r["share_reach"]["hi95"],
                ]
            )
        if "reach_S5R" in res:
            r = res["reach_S5R"]
            w.writerow(["S5R", "reach", "rate_pct", r["rate_pct"], "", ""])
            w.writerow(
                ["S5R", "reach", "d_vs_stored32_pts", r["d_vs_stored32_pts"], r["lo95"], r["hi95"]]
            )
    print(
        json.dumps(
            {k: v for k, v in res.items() if k in ("stored", "split_dpo_draws_1_8")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
