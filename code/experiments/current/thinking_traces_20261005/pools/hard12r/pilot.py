"""Pilot yield of the RL'd filter model at k 8: share of candidates solved <= 2, 3-6 and 7-8 of 8
times, overall and by source, reference length and structural features.

    P=code.experiments.current.thinking_traces_20261005.pools.hard12r.pilot
    python -m $P --work runs/pools/hard12r/work --out runs/pools/hard12r/work/pilot_report.json

Joins every <work>/eval/pilot_*/rows_plain_B0.jsonl to its candidate record in <work>/cands/
(by name). Features are hard12's (`features.py`, stored in each record's `feat`): `goal_type`,
`first_kind` (`elim` = the reference's first written step works forward from the premises),
`prem_inconsistent`, `fdepth` (formula depth). Groups: `prior` (hard12 candidates T0_s0 already
measured: spare in-band targets and T0 <= 1 of 32) and `fresh` (unmeasured candidates of hard12's
steered streams); each breakdown is given per group because the prior group is selected on T0.
Prints markdown tables and writes the report json.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

from code.thinking.data import read_jsonl

BINS = (("le2", 0, 2), ("3-6", 3, 6), ("7-8", 7, 8))
FILTER = (
    "/scratch/cowifornia/nd-rl/think/runs/restart3_grpofix/grpo_T0_crashed_it38/ckpts/"
    "grpo_T0_it35_rolling.pt"
)


def cell(ns: list[int]) -> dict[str, Any]:
    n = len(ns)
    out: dict[str, Any] = {"n": n}
    for b, lo, hi in BINS:
        out[f"{b}_pct"] = round(100 * sum(lo <= x <= hi for x in ns) / n, 1)
    out["zero_pct"] = round(100 * sum(x == 0 for x in ns) / n, 1)
    out["le2_nonzero_pct"] = round(100 * sum(1 <= x <= 2 for x in ns) / n, 1)
    out["pass1_pct"] = round(100 * sum(ns) / (8 * n), 1)
    return out


def group_of(r: dict[str, Any]) -> str:
    return "fresh" if r.get("src", "fresh") == "fresh" else "prior"


def breakdown(data: list[tuple[dict[str, Any], int]], key: Any) -> dict[str, Any]:
    g: dict[str, list[int]] = collections.defaultdict(list)
    for r, n_ok in data:
        g[str(key(r))].append(n_ok)
    return {v: cell(ns) for v, ns in sorted(g.items())}


def t0_bin(r: dict[str, Any]) -> str:
    t = r.get("t0_n_ok")
    if t is None:
        return "unmeasured"
    return str(t) if t <= 1 else ("2-4" if t <= 4 else "5-16")


KEYS: dict[str, Any] = {
    "src": lambda r: r.get("src", "fresh"),
    "t0_of_32": t0_bin,
    "cfg": lambda r: r.get("cfg", "strict_long"),
    "n_lines": lambda r: r["n_lines"],
    "goal_type": lambda r: r["feat"]["goal_type"],
    "forward_first": lambda r: int(r["feat"]["first_kind"] == "elim"),
    "first_kind": lambda r: r["feat"]["first_kind"],
    "prem_inconsistent": lambda r: r["feat"]["prem_inconsistent"],
    "fdepth": lambda r: r["feat"]["fdepth"],
    "t0_band_score": lambda r: f">={min(int(10 * r['band_score']), 6) / 10}",
}


def md(title: str, tab: dict[str, Any]) -> str:
    lines = [
        f"**{title}**",
        "",
        "| value | n | <= 2/8 | 3-6/8 | 7-8/8 | of which 0/8 | pass@1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, c in tab.items():
        lines.append(
            f"| {v} | {c['n']:,} | {c['le2_pct']} | {c['3-6_pct']} | {c['7-8_pct']} | "
            f"{c['zero_pct']} | {c['pass1_pct']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = {r["name"]: r for p in glob.glob(f"{a.work}/cands/*.jsonl") for r in read_jsonl(p)}
    runs = {}
    data: list[tuple[dict[str, Any], int]] = []
    for d in sorted(glob.glob(f"{a.work}/eval/pilot_*/")):
        d = d.rstrip("/")
        if not os.path.exists(f"{d}/summary.json"):
            continue
        s = json.loads(Path(f"{d}/summary.json").read_text())
        cellp = s["cells"]["plain_B0"]
        runs[os.path.basename(d)] = {
            "ckpt": s["ckpt"],
            "k": s["k"],
            "seed": s["config"]["seed"],
            "n": s["n_theorems"],
            "wall_s": round(cellp["wall_s"]),
            "tokens_per_attempt": round(cellp["all"]["tokens_per_attempt"]["tokens"], 1),
        }
        data += [(recs[r["name"]], r["n_ok"]) for r in read_jsonl(f"{d}/rows_plain_B0.jsonl")]
    rep: dict[str, Any] = {
        "written_on": "2026-10-07",
        "written_by": "agent:claude",
        "status": "pilot only: stopped after the k 8 pilot (scope change); no confirmation, "
        "no pool or test split was built",
        "filter_model": FILTER,
        "eval": "code.thinking.eval, plain arm, B 0, k 8, T 0.8, max_action 512, max_steps 96, "
        "max_rej 10, batch 1024, seed 1, Lean-judged",
        "runs": runs,
        "attempts": 8 * len(data),
        "gpu_s": sum(v["wall_s"] for v in runs.values()),
        "all": cell([n for _, n in data]),
    }
    groups = {"all": data}
    for gname in ("prior", "fresh"):
        groups[gname] = [(r, n) for r, n in data if group_of(r) == gname]
    rep["by_group"] = {
        g: {k: breakdown(xs, f) for k, f in KEYS.items()} for g, xs in groups.items()
    }
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({"runs": runs, "all": rep["all"]}, indent=1))
    for g in ("all", "prior", "fresh"):
        if groups[g]:
            print(f"\n### {g} (n = {len(groups[g]):,})\n")
            for k in KEYS:
                print(md(k, rep["by_group"][g][k]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
