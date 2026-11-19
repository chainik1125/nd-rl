#!/usr/bin/env python3
# 120 columns, as in long_rl_20261014's files
# ruff: noqa: E501
"""One row per run at a matched RL time, from the rounds.jsonl files status.py pulled (or a finished run's own).

  python summary.py [RL_MINUTES]     # default: the latest minute every full-pool run has reached

For the arm's removed rule X, at the last oracle call at or before RL_MINUTES:
  solved      RL targets with an accepted proof so far (of 18,439)
  attempts    Lean-accepted attempts so far, and how many of them use X
  X proofs    distinct accepted proofs using X so far, and the targets they are on
  ref-X       solved targets whose reference proof uses X, of those in the pool (a proxy for needing X)
The control rows report the same four numbers for each rule in turn.
"""

import glob
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "artifacts", "rule_knockout_20261118")
)
RULE = {"mp": "IMPE", "cp": "IMPI", "exfalso": "BOTE", "dn": "DN"}
# RL targets whose reference proof uses each rule (the full pool; 4,493 ladder targets have no reference proof)
REF_N = {"IMPE": 13774, "IMPI": 4224, "BOTE": 4469, "DN": 271}


def load():
    runs = {}
    for fn in glob.glob(f"{ART}/live/*/rounds.jsonl") + glob.glob(
        f"{ART}/rule-knockout-20261118_*/rounds.jsonl"
    ):
        name = os.path.basename(os.path.dirname(fn)).removeprefix("rule-knockout-20261118_")
        rs = [json.loads(line) for line in Path(fn).read_text().splitlines() if line.strip()]
        if len(rs) > len(runs.get(name, [])):  # a finished run's own file beats an older live copy
            runs[name] = rs
    return runs


def at(rs, minutes, x):
    """Cumulative numbers at the last call at or before `minutes` of RL clock."""
    upto = [c for c in rs if c["rl_secs_used"] <= minutes * 60]
    if not upto:
        return None
    c, cen = upto[-1], upto[-1]["census"]
    return {
        "minute": c["rl_secs_used"] / 60,
        "solved": c["targets_solved"],
        "accepted": sum(u["census"]["accepted"] for u in upto),
        "accepted_x": sum(u["census"]["accepted_using"].get(x, 0) for u in upto),
        "proofs_x": cen["proofs_using"].get(x, 0),
        "targets_x": cen["targets_using"].get(x, 0),
        "ref_x": cen["solved_ref_uses"].get(x, 0),
    }


def main():
    runs = load()
    full_pool = {n: rs for n, rs in runs.items() if "focus" not in n}
    minutes = (
        float(sys.argv[1])
        if len(sys.argv) > 1
        else min(rs[-1]["rl_secs_used"] for rs in full_pool.values()) / 60
    )
    print(f"At {minutes:.0f} min of RL clock (each run's last oracle call at or before it).\n")
    print(
        "| run | rule X | min | solved | accepted attempts | of them using X | X proofs (targets) | ref-X solved |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name in sorted(runs):
        arm = name.split("-")[0]
        for x in [RULE[arm]] if arm in RULE else list(RULE.values()):
            r = at(runs[name], minutes, x)
            if r is None:
                print(f"| {name} | {x} | not there yet | | | | | |")
                continue
            share = f" ({100 * r['accepted_x'] / r['accepted']:.1f}%)" if r["accepted"] else ""
            print(
                f"| {name} | {x} | {r['minute']:.0f} | {r['solved']:,} | {r['accepted']:,} | {r['accepted_x']:,}{share} | "
                f"{r['proofs_x']:,} ({r['targets_x']:,}) | {r['ref_x']:,} of {REF_N[x]:,} |"
            )


if __name__ == "__main__":
    main()
