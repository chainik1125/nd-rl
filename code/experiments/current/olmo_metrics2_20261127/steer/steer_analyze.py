#!/usr/bin/env python3
"""M3 analysis (serrano): F6 of discourse/steer.md Turn 5.  Tables as CSV/JSON, one figure through ndbench's style.py.

    steer_analyze.py --runs DIR --inputs inputs/steer --p2 DIR --p3 DIR --out OUT

--runs holds one sub-directory per sampled cell, each with graded.jsonl (Dan's grader output, one row per draw with
pid / strict / n_tok / finish):  dev_guard  e0p  e1  [e2_s1 e2_s2 e2_s3].  Absent cells are skipped (E2 is conditional).
Everything is a percentage of the prompt set with a 95 % prompt-bootstrap interval (B = 10,000, seed 0), paired where two
cells share prompts.  A "pass" is Dan's strict checker.  Reach: E1 vs the fresh E0'.  Held-out: E1 vs the stored dpo
draws (job F_dpo_ifb, 16 per prompt).  Dev guard: both scored at the 12,000-token cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steer_common as C  # noqa: E402

B_DEFAULT = 10000
DEV_CAP = 12000
EQUIV = [  # checkpoint equivalents on the 48 reach prompts (CONTEXT.md; plain prompt, % of draws) and other references
    ("dpo", 0.0),
    ("s0100", 0.1),
    ("s0500", 4.5),
    ("s1100", 38.5),
    ("s2300", 69.4),
    ("60-word hint on dpo", 6.2),
    ("64-example LoRA on dpo", 63.5),
]


def read_graded(path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def per_prompt_counts(rows, pids, cap=None) -> dict:
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["pid"] not in pids:
            continue
        ok = bool(r.get("strict")) and (cap is None or int(r["n_tok"]) <= cap)
        d[r["pid"]][0] += 1
        d[r["pid"]][1] += int(ok)
    return dict(d)


def pass1_vec(counts: dict, pids: list) -> np.ndarray:
    return np.array(
        [counts[p][1] / counts[p][0] if p in counts and counts[p][0] else np.nan for p in pids]
    )


def passk_vec(counts: dict, pids: list, k: int) -> np.ndarray:
    return np.array(
        [
            C.pass_at_k(counts[p][0], counts[p][1], min(k, counts[p][0]))
            if p in counts and counts[p][0]
            else np.nan
            for p in pids
        ]
    )


def boot_mean(x: np.ndarray, B=B_DEFAULT, seed=0) -> dict:
    x = x[~np.isnan(x)]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return {
        "pct": 100 * float(x.mean()),
        "lo": 100 * float(np.percentile(m, 2.5)),
        "hi": 100 * float(np.percentile(m, 97.5)),
        "n_prompts": int(len(x)),
    }


def boot_diff(a: np.ndarray, b: np.ndarray, B=B_DEFAULT, seed=0) -> dict:
    """Paired prompt bootstrap of mean(a) - mean(b) in percentage points."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(B, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return {
        "pts": 100 * float((a - b).mean()),
        "lo": 100 * float(np.percentile(d, 2.5)),
        "hi": 100 * float(np.percentile(d, 97.5)),
        "n_prompts": int(len(a)),
    }


def baseline_counts(base: dict, pids, cap=None) -> dict:
    out = {}
    for p in pids:
        lst = base.get(p, [])
        out[p] = [
            len(lst),
            sum(1 for strict, n_tok, _f in lst if strict and (cap is None or n_tok <= cap)),
        ]
    return out


def length_stats(rows, pids, cap_tokens: int) -> dict:
    rs = [r for r in rows if r["pid"] in pids]
    if not rs:
        return {}
    n = np.array([int(r["n_tok"]) for r in rs])
    return {
        "draws": len(rs),
        "mean_len": float(n.mean()),
        "cap_hit_rate": float(
            np.mean([r.get("finish") == "length" or int(r["n_tok"]) >= cap_tokens for r in rs])
        ),
        "no_end_of_thinking_share": float(np.mean([not r.get("think_end", True) for r in rs])),
    }


def pooled_counts(cells: list) -> dict:
    d = defaultdict(lambda: [0, 0])
    for c in cells:
        for p, (n, k) in c.items():
            d[p][0] += n
            d[p][1] += k
    return dict(d)


def analyze(runs: Path, inputs: Path, B=B_DEFAULT) -> dict:
    man = json.loads((inputs / "manifest.json").read_text())
    base = json.loads((inputs / "dpo_baseline.json").read_text())
    reach, held, dev = man["sets"]["reach"], man["sets"]["held"], man["sets"]["dev"]
    rows = {}
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        f = d / "graded.jsonl"
        if d.is_dir() and f.exists():
            rows[d.name] = read_graded(f)
    res = {
        "cells": {k: len(v) for k, v in rows.items()},
        "reach": {},
        "held": {},
        "dev_guard": {},
        "lengths": {},
    }
    # reach ------------------------------------------------------------------------------------------------------------------
    rc = {k: per_prompt_counts(v, set(reach)) for k, v in rows.items()}
    e2 = [k for k in rows if k.startswith("e2_s")]
    if e2:
        rc["e2_pooled"] = pooled_counts([rc[k] for k in e2])
    if "e0p" in rc:
        n0 = min(v[0] for v in rc["e0p"].values())
        for k, cnt in rc.items():
            if k in ("dev_guard",) or not cnt:
                continue
            ent = {
                "draws": sum(v[0] for v in cnt.values()),
                "passes": sum(v[1] for v in cnt.values()),
                "pass1": boot_mean(pass1_vec(cnt, reach), B),
            }
            kk = min(n0, min(v[0] for v in cnt.values()))
            ent["k_common"] = kk
            ent[f"pass@{kk}"] = boot_mean(passk_vec(cnt, reach, kk), B)
            if k != "e0p":
                ent["diff_pass1_vs_e0p"] = boot_diff(
                    pass1_vec(cnt, reach), pass1_vec(rc["e0p"], reach), B
                )
                ent[f"diff_pass@{kk}_vs_e0p"] = boot_diff(
                    passk_vec(cnt, reach, kk), passk_vec(rc["e0p"], reach, kk), B
                )
                ent["prompts_with_a_pass"] = int(
                    sum(1 for p in reach if p in cnt and cnt[p][1] > 0)
                )
            res["reach"][k] = ent
    # held-out ----------------------------------------------------------------------------------------------------------------
    bh = baseline_counts(base["held"], held)
    res["held"]["dpo_stored"] = {
        "draws": sum(v[0] for v in bh.values()),
        "pass1": boot_mean(pass1_vec(bh, held), B),
    }
    for k in ["e1", *e2, "e2_pooled"]:
        src = rows.get(k) if k != "e2_pooled" else sum([rows[x] for x in e2], []) if e2 else None
        if src is None:
            continue
        cnt = per_prompt_counts(src, set(held))
        if not cnt:
            continue
        ps = [p for p in held if p in cnt]
        ent = {
            "draws": sum(v[0] for v in cnt.values()),
            "n_prompts": len(ps),
            "pass1": boot_mean(pass1_vec(cnt, ps), B),
            "diff_vs_dpo_stored": boot_diff(pass1_vec(cnt, ps), pass1_vec(bh, ps), B),
        }
        res["held"][k] = ent
    # dev guard (12,000-token cap on both sides) -------------------------------------------------------------------------------
    if "dev_guard" in rows:
        cnt = per_prompt_counts(rows["dev_guard"], set(dev), cap=DEV_CAP)
        bd = baseline_counts(base["dev"], dev, cap=DEV_CAP)
        ps = [p for p in dev if p in cnt]
        dg = rows["dev_guard"]
        ls = length_stats(dg, set(dev), DEV_CAP)
        base_len = float(np.mean([min(x[1], DEV_CAP) for p in dev for x in base["dev"][p]]))
        diff = boot_diff(pass1_vec(cnt, ps), pass1_vec(bd, ps), B)
        ratio = ls["mean_len"] / base_len if base_len else float("nan")
        res["dev_guard"] = {
            "pass1": boot_mean(pass1_vec(cnt, ps), B),
            "dpo_stored_pass1": boot_mean(pass1_vec(bd, ps), B),
            "diff": diff,
            "cap_hit_rate": ls["cap_hit_rate"],
            "mean_len": ls["mean_len"],
            "dpo_mean_len_capped": base_len,
            "length_ratio": ratio,
            "guard": {
                "cap_hits_le_10pct": ls["cap_hit_rate"] <= 0.10,
                "length_ratio_0.7_1.3": 0.7 <= ratio <= 1.3,
                "pass1_not_more_than_10pts_below": diff["pts"] >= -10.0,
            },
        }
        res["dev_guard"]["guard"]["pass"] = all(res["dev_guard"]["guard"].values())
    # lengths ------------------------------------------------------------------------------------------------------------------
    for k, v in rows.items():
        if k != "dev_guard":
            res["lengths"][k] = length_stats(v, set(reach) | set(held), 32768)
    res["checkpoint_equivalents_pct_reach"] = dict(EQUIV)
    return res


def nll_tables(p2: Path, p3: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    r3 = json.loads((p3 / "p3_report.json").read_text())
    r2 = json.loads((p2 / "p2_report.json").read_text())
    with open(out / "nll_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "vtype",
                "layer",
                "alpha",
                "nll_sel_total",
                "nll_sel_think",
                "nll_sel_answer",
                "damage_rise_total",
            ]
        )
        for k, g in r3["grid"].items():
            vt, l, a = k.split("|")
            sp = g["sel_pooled"]
            w.writerow([vt, l, a, g["nll_sel"], sp["think"], sp["answer"], g["damage_rise"]])
    mc = r3["manipulation_check"]
    with open(out / "manipulation_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "segment", "g_pct", "lo_pct", "hi_pct", "band", "damage_rise"])
        for name, segs in mc["g"].items():
            for seg, ci in segs.items():
                w.writerow(
                    [
                        name,
                        seg,
                        100 * ci["point"],
                        100 * ci["lo"],
                        100 * ci["hi"],
                        ci["band"],
                        mc["damage_rise"].get(name, "") if seg == "total" else "",
                    ]
                )
        for seg, ci in (mc.get("R_alpha1") or {}).items():
            if isinstance(ci, dict) and "point" in ci:
                w.writerow(
                    ["R_alpha1", seg, 100 * ci["point"], 100 * ci["lo"], 100 * ci["hi"], "", ""]
                )
    summ = {
        "chosen": mc["chosen"],
        "D_rev": r2["D_rev"],
        "final": r3["final"],
        "split_half": r3["split_half"],
        "plan": r3["plan"],
        "timing_tokens_per_s": r3["timing_tokens_per_s"],
    }
    (out / "nll_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def write_tables(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    with open(out / "reach.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cell",
                "draws",
                "passes",
                "pass1_pct",
                "lo",
                "hi",
                "k",
                "passk_pct",
                "lo",
                "hi",
                "diff_pass1_pts_vs_e0p",
                "lo",
                "hi",
            ]
        )
        for k, e in res["reach"].items():
            kk = e["k_common"]
            pk = e[f"pass@{kk}"]
            d = e.get("diff_pass1_vs_e0p", {})
            w.writerow(
                [
                    k,
                    e["draws"],
                    e["passes"],
                    e["pass1"]["pct"],
                    e["pass1"]["lo"],
                    e["pass1"]["hi"],
                    kk,
                    pk["pct"],
                    pk["lo"],
                    pk["hi"],
                    d.get("pts", ""),
                    d.get("lo", ""),
                    d.get("hi", ""),
                ]
            )
    with open(out / "heldout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "draws", "pass1_pct", "lo", "hi", "diff_vs_dpo_stored_pts", "lo", "hi"])
        for k, e in res["held"].items():
            d = e.get("diff_vs_dpo_stored", {})
            w.writerow(
                [
                    k,
                    e["draws"],
                    e["pass1"]["pct"],
                    e["pass1"]["lo"],
                    e["pass1"]["hi"],
                    d.get("pts", ""),
                    d.get("lo", ""),
                    d.get("hi", ""),
                ]
            )
    with open(out / "checkpoint_equivalents.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference", "reach_pass_pct_of_draws"])
        for k, v in res["checkpoint_equivalents_pct_reach"].items():
            w.writerow([k, v])


def figure(res: dict, out: Path) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # .../code
    from ndbench.figures import style as S

    cells = [(k, res["reach"][k]) for k in ("e0p", "e1", "e2_pooled") if k in res["reach"]]
    names = {
        "e0p": "dpo (E0')",
        "e1": "dpo + vector (E1)",
        "e2_pooled": "dpo + sign scrambles (E2)",
    }
    cols = {"e0p": "#555555", "e1": "#D55E00", "e2_pooled": "#0072B2"}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 1.9)
        ax = fig.add_subplot(111)
        ys = list(range(len(cells)))[::-1]
        for y, (k, e) in zip(ys, cells):
            p = e["pass1"]
            ax.errorbar(
                [p["pct"]],
                [y],
                xerr=[[p["pct"] - p["lo"]], [p["hi"] - p["pct"]]],
                fmt="o",
                color=cols[k],
                capsize=2.5,
            )
        for lab, v in (("s0100", 0.1), ("s0500", 4.5)):
            ax.axvline(v, color="#999999", ls=":", lw=0.8)
            ax.text(
                v, ys[0] + 0.55, lab, fontsize=S.FS_TICK, color="#666666", ha="left", va="bottom"
            )
        ax.set_yticks(ys)
        ax.set_yticklabels([names[k] for k, _ in cells], fontsize=S.FS_TICK)
        ax.set_xlabel(
            "Reach prompts passed (% of the prompt set; strict checker)", fontsize=S.FS_LABEL
        )
        ax.set_xlim(left=-0.5)
        ax.set_ylim(-0.6, len(cells) - 0.2)
        paths = S.save(fig, out / "reach_pass1")
        S.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--p2")
    ap.add_argument("--p3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = analyze(Path(a.runs), Path(a.inputs), a.B)
    write_tables(res, out)
    if a.p2 and a.p3:
        res["nll"] = nll_tables(Path(a.p2), Path(a.p3), out)
        (out / "analysis.json").write_text(json.dumps(res, indent=1))
    if not a.no_figure and "e0p" in res["reach"]:
        print("figure:", [str(p) for p in figure(res, out)])
    print(json.dumps({k: res[k] for k in ("cells",)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
