"""M1 baseline inventory (read only): draws per prompt, response tokens, share cut at the cap, for Dan's stored T 1.0 draws.

Usage: uv run --with numpy python -I temp_baseline_inventory.py   (imports dan_curve from the 20261125 folder; edits nothing)
"""

import gzip
import json
import sys

import numpy as np

sys.path.insert(
    0,
    "/scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code/experiments/current/olmo_elicitation_20261125",
)
import dan_curve as dc  # noqa: E402

D = dc.DAN / "artifacts"
CAP = 32768
armA, reach = dc.arm_a(), dc.reach()
R = dc.Rows(D / "olmo_ifcurve" / "rows")


def stats(label, cells):
    # cells: pid -> list of (n_tok, finish, strict)
    toks = np.array([t for v in cells.values() for t, _, _ in v])
    cut = np.array([(f == "length") or t >= CAP for v in cells.values() for t, f, _ in v])
    s = np.array([x for v in cells.values() for _, _, x in v])
    per = [len(v) for v in cells.values()]
    pp = np.mean([np.mean([x for _, _, x in v]) for v in cells.values()])
    print(
        f"{label:34s} prompts={len(cells):3d} draws/prompt={min(per)}-{max(per)} total={len(toks):5d} "
        f"mean_tok={toks.mean():7.0f} p90_tok={np.percentile(toks, 90):7.0f} cut@cap={100 * cut.mean():5.2f}% "
        f"pass@1={100 * pp:5.1f}%"
    )


def jl(p):
    with gzip.open(p, "rt") if str(p).endswith(".gz") else open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


for c in ["dpo", "s0100", "s0500", "s2300"]:
    cells = {p: [(x["n_tok"], x["finish"], x["strict"]) for x in R.draws(c, p, "A")] for p in armA}
    print("armA", c, "jobs:", dc.SOURCES["A"][c], "N =", dc.N_A.get(c, 8))
    stats(f"armA  {c}", cells)
    if c == "dpo":
        rs = {}
        for j in ("F_dpo_ifb", "F_dpo_ife"):
            for r in sorted(
                jl(D / f"olmo_ifomega/graded/{j}.jsonl"), key=lambda r: (r["chunk"], r["i"])
            ):
                if r["pid"] in reach:
                    rs.setdefault(r["pid"], []).append(
                        (r["n_tok"], r["finish"], int(bool(r["strict"])))
                    )
        rs = {p: v[:16] for p, v in rs.items()}
        print("reach dpo jobs: olmo_ifomega F_dpo_ifb + F_dpo_ife (first 16)")
        stats("reach dpo", rs)
    else:
        cells = {
            p: [(x["n_tok"], x["finish"], x["strict"]) for x in R.draws(c, p, "B")] for p in reach
        }
        print("reach", c, "jobs:", dc.SOURCES["B"][c], "N =", dc.N_B)
        stats(f"reach {c}", cells)
