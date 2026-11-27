# ruff: noqa: E501
"""Inputs of M2 (Turn 5 B.1): prompt files D / H / R, stored reference rates, seeds, MANIFEST. Serrano, CPU, reads only.

  python3 make_inputs.py [--out /scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/rank1]

pids_D.txt  the 64 prompts of pool/train.jsonl (arm-A non-reach); dev only.
pids_H.txt  arm A (218) minus the reach prompts (48; 42 are in arm A) minus D = 112 prompts. Held-out arm A.
pids_R.txt  Dan's pids_reach.txt (48), plain prompts as arm B sent them. Scored only; nothing is chosen on it.
refs.json   {dpo|s0500|s2300: {D|H|R: {pid: strict pass share}}} from Dan's stored rows (dan_curve.Rows: arm A for D and H with the
            cell's first N draws, arm B first 32 base draws for R). dpo has no arm B: the reach prompts are the ones dpo never
            passed in its stored plain tries, so its R rate is 0 by definition (CONTEXT.md).
seeds.json  the seeds of B.1.
spec_<cell>.txt  per-pid draw counts for Dan's sampler (`--pids @spec --n 1 --chunk 1`): A1 H:2 R:4, A1u H:2 R:2, A2 H:1 R:2,
            Xheld H:2 R:4, Xdev D:2 (one engine load per cell).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0,
    "/scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code/experiments/current/olmo_elicitation_20261125",
)
from common import SEEDS  # noqa: E402

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
STORED = {"dpo": 16, "s0500": 8, "s2300": 8}


def build(out: Path, dan_root: Path, pool: Path, rows_dir: Path) -> dict:
    import dan_curve as dc

    out.mkdir(parents=True, exist_ok=True)
    arm_a = dc.arm_a(dan_root)
    reach = dc.reach(dan_root)
    d = [json.loads(ln)["pid"] for ln in open(pool / "train.jsonl") if ln.strip()]
    assert len(d) == 64 and set(d) <= set(arm_a) and not (set(d) & set(reach)), (
        "D must be 64 arm-A non-reach prompts"
    )
    h = [p for p in arm_a if p not in set(reach) and p not in set(d)]
    assert len(h) == 112, len(h)
    for name, pids in (("D", d), ("H", h), ("R", reach)):
        (out / f"pids_{name}.txt").write_text("\n".join(pids) + "\n")
    # per-pid draw counts for Dan's sampler (--pids @file --n 1: "pid:n" tokens, whitespace separated; --chunk 1)
    specs = {"A1": (2, 4), "A1u": (2, 2), "A2": (1, 2), "Xheld": (2, 4)}
    for nm, (nh, nr) in specs.items():
        (out / f"spec_{nm}.txt").write_text(
            "\n".join([f"{p}:{nh}" for p in h] + [f"{p}:{nr}" for p in reach]) + "\n"
        )
    (out / "spec_Xdev.txt").write_text("\n".join(f"{p}:2" for p in d) + "\n")
    rows = dc.Rows(rows_dir)
    refs: dict = {}
    for ck in STORED:
        refs[ck] = {}
        for tag, pids in (("D", d), ("H", h)):
            refs[ck][tag] = {}
            for p in pids:
                dr = rows.draws(ck, p, "A")
                refs[ck][tag][p] = float(np.mean([x["strict"] for x in dr])) if dr else None
        if ck == "dpo":
            refs[ck]["R"] = {p: 0.0 for p in reach}
        else:
            refs[ck]["R"] = {}
            for p in reach:
                dr = rows.draws(ck, p, "B")
                refs[ck]["R"][p] = float(np.mean([x["strict"] for x in dr])) if dr else None
    missing = {
        ck: {t: [p for p, v in refs[ck][t].items() if v is None] for t in refs[ck]} for ck in refs
    }
    for ck in refs:
        for t in refs[ck]:
            refs[ck][t] = {p: v for p, v in refs[ck][t].items() if v is not None}
    (out / "refs.json").write_text(json.dumps(refs))
    (out / "seeds.json").write_text(json.dumps(SEEDS, indent=1))
    summ = {
        ck: {t: round(100 * float(np.mean(list(refs[ck][t].values()))), 2) for t in refs[ck]}
        for ck in refs
    }
    man = {
        "n": {"D": len(d), "H": len(h), "R": len(reach)},
        "stored_pass1_pct": summ,
        "missing_prompts_without_stored_draws": {
            ck: {t: len(v) for t, v in m.items()} for ck, m in missing.items()
        },
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.iterdir())
            if p.name != "MANIFEST.json"
        },
        "problems_file": "olmo_ifomega/problems.jsonl (Dan's; confirm with analyze_arms.py before the first cell)",
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OE / "metrics2" / "inputs" / "rank1"))
    ap.add_argument("--dan-root", default=str(OE / "ext" / "nd-takehome"))
    ap.add_argument("--pool", default=str(OE / "pool"))
    ap.add_argument(
        "--rows-dir",
        default=str(OE / "ext" / "nd-takehome" / "artifacts" / "olmo_ifcurve" / "rows"),
    )
    a = ap.parse_args(argv)
    print(
        json.dumps(build(Path(a.out), Path(a.dan_root), Path(a.pool), Path(a.rows_dir)), indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
