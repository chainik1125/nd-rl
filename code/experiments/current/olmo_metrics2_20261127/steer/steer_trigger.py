#!/usr/bin/env python3
"""M3 E2 trigger (pod CPU, no GPU): decide from graded draws whether the three sign-scramble cells are run (F4 step 9, d6).

    steer_trigger.py --e1 runs/m3_e1/out/graded.jsonl --e0p runs/m3_e0p/out/graded.jsonl \\
        --inputs inputs/steer --out runs/m3_e1/out/e2_trigger.json

Rule (mechanical, no human input):  run E2  iff
    [ E1 reach passes - s x E0' reach passes >= 2 ]   with s = (E1 reach draws) / (E0' reach draws)
                                                      (s = 1 with the full 384-draw E0'; s = 2 when E0' was cut to 192)
 OR [ |held-out pass@1 of E1 - stored dpo held-out pass@1| >= 6 points ].
A pass is Dan's strict grade.  Held-out pass@1 = mean over held-out prompts of the per-prompt pass fraction; the stored
baseline is dpo_baseline.json (job F_dpo_ifb, 16 draws per prompt).  Prints `E2_TRIGGER=1|0`, writes the JSON, exits 0
(2 if an input has no rows of the expected kind).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steer_common as C  # noqa: E402


def read_graded(path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def per_prompt(rows, pids: set) -> dict:
    d = defaultdict(lambda: [0, 0])  # pid -> [draws, passes]
    for r in rows:
        if r["pid"] in pids:
            d[r["pid"]][0] += 1
            d[r["pid"]][1] += int(bool(r.get("strict")))
    return dict(d)


def decide(e1_rows, e0p_rows, baseline_held: dict, reach: set, held: set, **kw) -> dict:
    e1_reach = per_prompt(e1_rows, reach)
    e0_reach = per_prompt(e0p_rows, reach)
    e1_held = per_prompt(e1_rows, held)
    if not e1_reach or not e0_reach or not e1_held:
        raise ValueError(
            "an input has no rows of the expected kind (E1 reach / E0' reach / E1 held-out)"
        )
    e1_draws, e1_pass = sum(v[0] for v in e1_reach.values()), sum(v[1] for v in e1_reach.values())
    e0_draws, e0_pass = sum(v[0] for v in e0_reach.values()), sum(v[1] for v in e0_reach.values())
    held_p1 = sum(v[1] / v[0] for v in e1_held.values()) / len(e1_held)
    base = {
        p: sum(1 for x in lst if x[0]) / len(lst)
        for p, lst in baseline_held.items()
        if p in held and lst
    }
    base_p1 = sum(base[p] for p in e1_held if p in base) / max(
        1, sum(1 for p in e1_held if p in base)
    )
    out = C.e2_trigger(e1_pass, e1_draws, e0_pass, e0_draws, held_p1, base_p1, **kw)
    out.update(
        {
            "e1_reach_draws": e1_draws,
            "e1_reach_pass": e1_pass,
            "e0p_reach_draws": e0_draws,
            "e0p_reach_pass": e0_pass,
            "e1_held_pass1": held_p1,
            "base_held_pass1": base_p1,
            "n_e1_held_prompts": len(e1_held),
        }
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e0p", required=True)
    ap.add_argument("--inputs", required=True, help="dir with manifest.json and dpo_baseline.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    inp = Path(a.inputs)
    man = json.loads((inp / "manifest.json").read_text())
    base = json.loads((inp / "dpo_baseline.json").read_text())
    try:
        out = decide(
            read_graded(a.e1),
            read_graded(a.e0p),
            base["held"],
            set(man["sets"]["reach"]),
            set(man["sets"]["held"]),
        )
    except ValueError as e:
        print(f"E2_TRIGGER_ERROR {e}")
        return 2
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"E2_TRIGGER={int(out['trigger'])} {json.dumps(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
