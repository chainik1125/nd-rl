"""PF statistics and the pre-stated gate (CPU). No manual step: the verdict is printed by this script.

Input: hf_score.py output (per-sequence totals under dpo) joined with the manifest's group / pid. Groups: `s2300` (stored
strict-pass draws), `dpo` (stored dpo T 1.0 draws), `c1` (new card-decoded draws; DESCRIPTIVE ONLY, never in a gate).
Draws cut at the cap (no EOS) are excluded from per-token means and residuals.
For each prompt p with at least one s2300 response and one dpo response:
  per-token l = mean log p_dpo over the response tokens (EOS included when present);  total L, length n.
  S_pt   = median over prompts of [ mean l (s2300 responses) - mean l (dpo responses) ]
  P_low,pt = share of s2300 responses (pooled over prompts) with l below the minimum l of that prompt's dpo responses
  length-adjusted total: beta = pooled within-prompt slope of L on n over the dpo responses (prompt fixed effects);
     rho = L - [ mean L_dpo,p + beta (n - mean n_dpo,p) ];
  S_adj  = median over prompts of mean rho (s2300);  P_low,adj = share of s2300 responses with rho below the minimum
     rho of that prompt's dpo responses (dpo rho computed the same way).
Intervals: prompt bootstrap (10,000, seed 0), beta held fixed.  No-shift value of P_low = 1/(n_dpo + 1).
Gate (thresholds frozen in the Turn 5 design, Part 3 section 7):
  TAIL        if, on the full response, S_pt or S_adj has CI upper bound < 0, or P_low,pt >= 0.5 or P_low,adj >= 0.5;
  PASS        if S_pt and S_adj have CI upper bounds >= 0 AND P_low,pt and P_low,adj <= 0.25 (0.35 when the dpo comparators
              per prompt are 4 or fewer, --pass-p-low to override);
  UNRESOLVED  otherwise.
The answer segment (after </think>) gets the same statistics as a pre-stated SECONDARY and does not change the verdict.
Usage: temp_pf_stats.py --scores S.jsonl --manifest M.jsonl [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import temp_common as tc


def _arr(rows, key_n, key_lp):
    return np.array([r[key_lp] for r in rows], float), np.array([r[key_n] for r in rows], float)


def pooled_beta(dpo_by_p: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    num = den = 0.0
    for L, n in dpo_by_p.values():
        if len(L) >= 2:
            num += float(((L - L.mean()) * (n - n.mean())).sum())
            den += float(((n - n.mean()) ** 2).sum())
    return num / den if den > 0 else 0.0


def segment_stats(
    by_p: dict[str, dict[str, list[dict]]], seg: str, reps: int = tc.BOOT_REPS
) -> dict:
    """seg: 'full' (total_lp, n_resp) or 'answer' (answer_lp, answer_n)."""
    kl, kn = ("total_lp", "n_resp") if seg == "full" else ("answer_lp", "answer_n")
    prompts, data = [], {}
    for p, g in by_p.items():
        s = [r for r in g.get("s2300", []) if r["ends_eos"] and r[kn] > 0]
        d = [r for r in g.get("dpo", []) if r["ends_eos"] and r[kn] > 0]
        if s and d:
            prompts.append(p)
            data[p] = (_arr(s, kn, kl), _arr(d, kn, kl))
    beta = pooled_beta({p: data[p][1] for p in prompts})
    diffs, rhos, low_pt, low_adj, cnt = [], [], {}, {}, {}
    for p in prompts:
        (Ls, ns), (Ld, nd) = data[p]
        lps, lpd = Ls / ns, Ld / nd
        diffs.append(lps.mean() - lpd.mean())
        res_s = Ls - (Ld.mean() + beta * (ns - nd.mean()))
        res_d = Ld - Ld.mean() - beta * (nd - nd.mean())
        rhos.append(res_s.mean())
        low_pt[p] = int((lps < lpd.min()).sum())
        low_adj[p] = int((res_s < res_d.min()).sum())
        cnt[p] = len(Ls)
    if not prompts:
        return {"n_prompts": 0}
    diffs, rhos = np.array(diffs), np.array(rhos)
    ids = np.array(prompts)
    idx = tc.boot_idx(len(prompts), reps)
    lo_pt = np.array([low_pt[p] for p in prompts], float)
    lo_ad = np.array([low_adj[p] for p in prompts], float)
    c = np.array([cnt[p] for p in prompts], float)
    out = {
        "n_prompts": len(prompts),
        "n_s2300": int(c.sum()),
        "beta_per_token_of_length": beta,
        "n_dpo_median": float(np.median([len(data[p][1][0]) for p in prompts])),
    }
    for name, v in (("S_pt", diffs), ("S_adj", rhos)):
        b = np.median(v[idx], axis=1)
        lo, hi = tc.ci(b)
        out[name] = {"median": float(np.median(v)), "lo95": lo, "hi95": hi}
    for name, num in (("P_low_pt", lo_pt), ("P_low_adj", lo_ad)):
        b = num[idx].sum(axis=1) / c[idx].sum(axis=1)
        lo, hi = tc.ci(b)
        out[name] = {
            "share": float(num.sum() / c.sum()),
            "lo95": lo,
            "hi95": hi,
            "no_shift_expected": 1 / (out["n_dpo_median"] + 1),
        }
    return out


def verdict(full: dict, pass_p_low: float | None = None) -> dict:
    if not full.get("n_prompts"):
        return {"verdict": "UNRESOLVED", "reason": "no prompt has both s2300 and dpo responses"}
    thr = pass_p_low if pass_p_low is not None else (0.25 if full["n_dpo_median"] >= 8 else 0.35)
    tail = (
        full["S_pt"]["hi95"] < 0
        or full["S_adj"]["hi95"] < 0
        or full["P_low_pt"]["share"] >= 0.5
        or full["P_low_adj"]["share"] >= 0.5
    )
    ok = (
        full["S_pt"]["hi95"] >= 0
        and full["S_adj"]["hi95"] >= 0
        and full["P_low_pt"]["share"] <= thr
        and full["P_low_adj"]["share"] <= thr
    )
    return {
        "verdict": "TAIL" if tail else ("PASS" if ok else "UNRESOLVED"),
        "pass_p_low_threshold": thr,
        "tail_threshold": 0.5,
    }


def describe_c1(by_p, reps=None) -> dict:
    """Descriptive scale: mean per-token l and mean total L of the card-decoded and the dpo T 1.0 responses."""
    out = {}
    for g in ("dpo", "c1", "s2300"):
        rows = [
            r for gg in by_p.values() for r in gg.get(g, []) if r["ends_eos"] and r["n_resp"] > 0
        ]
        if rows:
            out[g] = {
                "n": len(rows),
                "mean_l_per_token": float(np.mean([r["total_lp"] / r["n_resp"] for r in rows])),
                "mean_total_lp": float(np.mean([r["total_lp"] for r in rows])),
                "mean_len": float(np.mean([r["n_resp"] for r in rows])),
            }
    return out


def build(scores: list[dict], manifest: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grp = {m["id"]: (m["pid"], m["group"]) for m in manifest}
    by_p: dict[str, dict[str, list[dict]]] = {}
    for r in scores:
        if r["id"] in grp:
            p, g = grp[r["id"]]
            by_p.setdefault(p, {}).setdefault(g, []).append(r)
    return by_p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pass-p-low", type=float)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    by_p = build(tc.jl(a.scores), tc.jl(a.manifest))
    full, ans = segment_stats(by_p, "full"), segment_stats(by_p, "answer")
    res = {
        "gate_full_response": verdict(full, a.pass_p_low),
        "full_response": full,
        "answer_segment_secondary_no_effect_on_verdict": ans,
        "descriptive_scale": describe_c1(by_p),
    }
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
