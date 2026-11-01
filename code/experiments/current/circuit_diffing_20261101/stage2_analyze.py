"""Stage 2 analysis: solve rates per condition and category, with paired problem-bootstrap
intervals.

    python -m code.experiments.current.circuit_diffing_20261101.stage2_analyze [--cc NAME]

For every condition in `/scratch/cowifornia/nd-rl/circuits/stage2/<cc>/`:
  - solve % on G (RL-gained) and S (shared), the mean over problems of n_ok / k;
  - RL-driver conditions: relative drop against rl_clean, drop_c = 1 - rate_c / rate_c(clean),
    and selectivity = drop_G - drop_S; the RL-only ablation is compared with the matched random
    ablations (their mean);
  - base-driver conditions: share of the RL - base gap recovered,
    (rate_c - base_clean_c) / (rl_clean_c - base_clean_c).
95% intervals: 4,000 draws, problems resampled within each category, the same draw for every
condition (paired). Writes `stage2.json` and `stage2.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import ART, HERE

ST2 = ART / "stage2"
DRAWS = 4000


def load(d: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for p in sorted(d.glob("*.json")):
        doc = json.loads(p.read_text())
        rows = doc["rows"]
        out[doc["condition"]] = {
            cat: np.array([r["n_ok"] / r["k"] for r in rows if r["cat"] == cat])
            for cat in ("G", "S")
        }
        out[doc["condition"]]["_doc"] = doc  # type: ignore[assignment]
    return out


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyse(d: Path) -> dict[str, Any]:
    data = load(d)
    conds = [c for c in data if not c.startswith("_")]
    nG, nS = len(data[conds[0]]["G"]), len(data[conds[0]]["S"])
    rng = np.random.default_rng(0)
    iG = rng.integers(0, nG, size=(DRAWS, nG))
    iS = rng.integers(0, nS, size=(DRAWS, nS))
    boot = {c: {"G": data[c]["G"][iG].mean(1), "S": data[c]["S"][iS].mean(1)} for c in conds}
    point = {c: {"G": data[c]["G"].mean(), "S": data[c]["S"].mean()} for c in conds}
    res: dict[str, Any] = {"n": {"G": nG, "S": nS}, "conditions": {}}
    for c in conds:
        doc = data[c]["_doc"]
        st = doc.get("stats") or {}
        r: dict[str, Any] = {
            "solve_pct": {k: 100 * float(point[c][k]) for k in ("G", "S")},
            "solve_ci": {k: [100 * v for v in ci(boot[c][k])] for k in ("G", "S")},
            "n_latents": doc.get("n_latents"),
            "edit_over_h": st.get("edit_norm", 0) / st["h_norm"] if st.get("h_norm") else 0.0,
        }
        if c.startswith("rl_") and c != "rl_clean" and "rl_clean" in boot:
            dr = {k: 1 - boot[c][k] / boot["rl_clean"][k] for k in ("G", "S")}
            dp = {k: 1 - point[c][k] / point["rl_clean"][k] for k in ("G", "S")}
            r["drop"] = {k: float(dp[k]) for k in ("G", "S")}
            r["drop_ci"] = {k: ci(dr[k]) for k in ("G", "S")}
            r["selectivity"] = float(dp["G"] - dp["S"])
            r["selectivity_ci"] = ci(dr["G"] - dr["S"])
            r["_sel_boot"] = dr["G"] - dr["S"]
            if "base_clean" in boot:
                rem = {}
                for k in ("G", "S"):
                    gap = point["rl_clean"][k] - point["base_clean"][k]
                    rem[k] = float((point["rl_clean"][k] - point[c][k]) / gap) if gap else None
                r["gap_removed"] = rem
                gb = boot["rl_clean"]["G"] - boot["base_clean"]["G"]
                r["gap_removed_ci_G"] = ci((boot["rl_clean"]["G"] - boot[c]["G"]) / gb)
        if c.startswith("base_") and c != "base_clean" and {"base_clean", "rl_clean"} <= set(boot):
            rec = {}
            for k in ("G", "S"):
                gap = point["rl_clean"][k] - point["base_clean"][k]
                rec[k] = float((point[c][k] - point["base_clean"][k]) / gap) if gap else None
            r["gap_recovered"] = rec
            gb = boot["rl_clean"]["G"] - boot["base_clean"]["G"]
            r["gap_recovered_ci_G"] = ci((boot[c]["G"] - boot["base_clean"]["G"]) / gb)
        res["conditions"][c] = r
    rands = [c for c in conds if c.startswith("rl_ablate_rand")]
    if "rl_ablate_rlonly" in res["conditions"] and rands:
        sel_r = np.mean([res["conditions"][c]["_sel_boot"] for c in rands], axis=0)
        sel_o = res["conditions"]["rl_ablate_rlonly"]["_sel_boot"]
        res["rlonly_minus_random_selectivity"] = {
            "point": float(
                res["conditions"]["rl_ablate_rlonly"]["selectivity"]
                - np.mean([res["conditions"][c]["selectivity"] for c in rands])
            ),
            "ci": ci(sel_o - sel_r),
            "p_le_0": float(np.mean(sel_o - sel_r <= 0)),
        }
        dG_o = 1 - boot["rl_ablate_rlonly"]["G"] / boot["rl_clean"]["G"]
        dG_r = np.mean([1 - boot[c]["G"] / boot["rl_clean"]["G"] for c in rands], axis=0)
        res["rlonly_minus_random_dropG"] = {
            "point": float(
                res["conditions"]["rl_ablate_rlonly"]["drop"]["G"]
                - np.mean([res["conditions"][c]["drop"]["G"] for c in rands])
            ),
            "ci": ci(dG_o - dG_r),
        }
    for r in res["conditions"].values():
        r.pop("_sel_boot", None)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default=None, help="a directory name under stage2/ (default: all)")
    a = ap.parse_args(argv)
    dirs = [ST2 / a.cc] if a.cc else sorted(p for p in ST2.glob("*") if p.is_dir())
    out = {d.name: analyse(d) for d in dirs if any(d.glob("*.json"))}
    (HERE / "stage2.json").write_text(json.dumps(out, indent=1))
    lines = ["# Stage 2 numbers (generated by `stage2_analyze.py`)", ""]
    for name, res in out.items():
        lines += [
            f"## {name}",
            "",
            f"G = {res['n']['G']} RL-gained problems, S = {res['n']['S']} shared problems; "
            "solve % = mean over problems of the per-problem success rate; [95% paired "
            "problem bootstrap].",
            "",
            "| condition | latents | edit / ‖h‖ | G solve % | S solve % | drop G | drop S "
            "| selectivity (drop G − drop S) | share of the RL − base gap |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c, r in res["conditions"].items():
            sg, ss = r["solve_pct"]["G"], r["solve_pct"]["S"]
            cg, cs = r["solve_ci"]["G"], r["solve_ci"]["S"]
            dg = ds = sel = gr = ""
            if "drop" in r:
                dg = f"{100 * r['drop']['G']:+.1f}%"
                ds = f"{100 * r['drop']['S']:+.1f}%"
                lo, hi = r["selectivity_ci"]
                sel = f"{100 * r['selectivity']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            for key, tag in (("gap_recovered", "recovered"), ("gap_removed", "removed")):
                if key in r:
                    g = r[key]
                    lo, hi = r[f"{key}_ci_G"]
                    s_txt = "—" if g["S"] is None else f"{100 * g['S']:.0f}%"
                    gr = f"{tag} G {100 * g['G']:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}], S {s_txt}"
            lat = "" if r["n_latents"] is None else str(r["n_latents"])
            lines.append(
                f"| {c} | {lat} | {r['edit_over_h']:.3f} "
                f"| {sg:.1f} [{cg[0]:.1f}, {cg[1]:.1f}] | {ss:.1f} [{cs[0]:.1f}, {cs[1]:.1f}] "
                f"| {dg} | {ds} | {sel} | {gr} |"
            )
        if "rlonly_minus_random_selectivity" in res:
            x = res["rlonly_minus_random_selectivity"]
            y = res["rlonly_minus_random_dropG"]
            lines += [
                "",
                f"RL-only minus matched-random selectivity: {100 * x['point']:+.1f} points "
                f"[{100 * x['ci'][0]:+.1f}, {100 * x['ci'][1]:+.1f}], bootstrap P(≤ 0) = "
                f"{x['p_le_0']:.3f}. RL-only minus random drop on G: {100 * y['point']:+.1f} "
                f"[{100 * y['ci'][0]:+.1f}, {100 * y['ci'][1]:+.1f}].",
            ]
        lines.append("")
    (HERE / "stage2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
