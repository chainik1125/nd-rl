#!/usr/bin/env python3
"""Inputs of the weight-interpolation run (serrano, CPU, reads only; nothing here is sealed).

    python3 make_inputs.py [--out /scratch/cowifornia/nd-rl/olmo_elicit/interp]

<out>/inputs/ (pushed to the pod):
  pids_reach.txt            Dan's 48 reach prompts (olmo_ifcurve/pids_reach.txt, his order)
  spec_reach4.txt           "pid:4" per prompt for his sampler (`--pids @spec --n 1 --chunk 1`):
                            192 requests
  prompt_sha256_reach.json  sha256 of each prompt's token ids, from the meta of metrics 2's ExPO
                            reach cell (job Xheld); must equal the fresh zero-vector dpo cell's
                            (job m3_e0p)
  MANIFEST.json             sha256 of the three files, counts, sources
<out>/refs/ (analysis only, never pushed):
  refs_reach.json           stored per-prompt strict pass shares on the reach prompts (metrics 2's
                            refs.json: s2300 32 draws per prompt, s0500, dpo 0 by construction)
                            and the stored summary points
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
DRAWS = 4
#: stored points shown beside the curve (strict checker, reach prompts); sources in PREREG.md
STORED = {
    "dpo_stored": {"rate": 0.0, "passes": 0, "draws": 768, "note": "stored plain draws"},
    "dpo_zero_vector_same_stack": {"rate": 1 / 384, "passes": 1, "draws": 384, "note": "m3_e0p"},
    "s2300_stored": {"rate": 0.694, "draws_per_prompt": 32, "note": "Dan's job B_s2300"},
    "expo_gamma_3.5": {"rate": 0.703, "draws_per_prompt": 4, "note": "metrics 2 job Xheld"},
}


def last_meta(path: Path) -> dict:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1])


def build(out: Path, dan: Path, m2: Path) -> dict:
    reach = [ln.strip() for ln in (dan / "olmo_ifcurve/pids_reach.txt").read_text().splitlines()]
    reach = [p for p in reach if p]
    assert len(reach) == 48 and len(set(reach)) == 48, len(reach)
    problems = {
        json.loads(ln)["pid"]
        for ln in (dan / "olmo_ifomega/problems.jsonl").read_text().splitlines()
    }
    assert set(reach) <= problems, "a reach prompt is missing from Dan's problems file"
    inp, refs = out / "inputs", out / "refs"
    inp.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    (inp / "pids_reach.txt").write_text("\n".join(reach) + "\n")
    (inp / "spec_reach4.txt").write_text("\n".join(f"{p}:{DRAWS}" for p in reach) + "\n")
    sha_x = last_meta(m2 / "runs/Xheld/out/meta.jsonl")["prompt_sha256_by_pid"]
    sha_e = last_meta(m2 / "runs/m3_e0p/out/meta.jsonl")["prompt_sha256_by_pid"]
    sha = {p: sha_x[p] for p in reach}
    assert all(sha_e[p] == sha[p] for p in reach), (
        "Xheld and m3_e0p tokenise a reach prompt differently"
    )
    (inp / "prompt_sha256_reach.json").write_text(json.dumps(sha, indent=1) + "\n")
    r = json.loads((m2 / "inputs/rank1/refs.json").read_text())
    ref = {ck: {p: r[ck]["R"][p] for p in reach} for ck in ("dpo", "s0500", "s2300")}
    mean = {ck: sum(v.values()) / len(v) for ck, v in ref.items()}
    assert abs(mean["s2300"] - 0.694) < 0.0005 and mean["dpo"] == 0.0, mean
    (refs / "refs_reach.json").write_text(
        json.dumps({"per_prompt": ref, "mean": mean, "stored_points": STORED}, indent=1) + "\n"
    )
    man = {
        "prompts": len(reach),
        "draws_per_prompt": DRAWS,
        "planned_draws": len(reach) * DRAWS,
        "sources": {
            "pids": str(dan / "olmo_ifcurve/pids_reach.txt"),
            "prompt_sha256": str(m2 / "runs/Xheld/out/meta.jsonl"),
            "refs": str(m2 / "inputs/rank1/refs.json"),
        },
        "stored_mean_pct": {k: round(100 * v, 2) for k, v in mean.items()},
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(inp.iterdir())
            if p.name != "MANIFEST.json"
        },
    }
    (inp / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OE / "interp"))
    ap.add_argument("--dan-root", default=str(OE / "ext/nd-takehome"))
    ap.add_argument("--metrics2", default=str(OE / "metrics2"))
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.out), Path(a.dan_root), Path(a.metrics2)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
