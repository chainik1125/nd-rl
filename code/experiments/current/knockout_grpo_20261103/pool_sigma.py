"""Share of RL pool problems that are strict sigma-covered under Dan's rule (coe_families.py):
classical-only (G4ip, intuit.intuit_provable, cannot prove it) and no '( ~ ( ~' in the sequent."""

import json
import multiprocessing as mp
import sys

sys.path.insert(0, "/scratch/cowifornia/nd-rl/ext/nd-takehome-ps")
from intuit import intuit_provable

K = "/scratch/cowifornia/nd-rl/knockout_grpo"


def lab(p):
    return (not intuit_provable(p), "( ~ ( ~" in p)


if __name__ == "__main__":
    files = {
        "L2_pool": f"{K}/pools/L2_pool.jsonl",
        "L3_pool": f"{K}/pools/L3_pool.jsonl",
        "L3_3p": f"{K}/pools/L3_3p.jsonl",
        "dan_rl_targets": "/scratch/cowifornia/nd-rl/ext/nd-takehome-ps/data/ladder/rl_targets.jsonl",  # noqa: E501
    }
    out = {}
    with mp.Pool(8) as pool:
        for nm, f in files.items():
            ps = [json.loads(l)["prompt"] for l in open(f) if l.strip()]  # noqa: E741, SIM115
            labs = pool.map(lab, ps, chunksize=50)
            n = len(labs)
            co = sum(a for a, b in labs)
            dn = sum(b for a, b in labs)
            st = sum(a and not b for a, b in labs)
            out[nm] = {
                "n": n,
                "classical_only": co,
                "classical_only_pct": 100 * co / n,
                "dn_anywhere": dn,
                "strict_sigma_covered": st,
                "strict_pct": 100 * st / n,
            }
            print(nm, out[nm], flush=True)
    json.dump(out, open(f"{K}/pool_sigma.json", "w"), indent=1)  # noqa: SIM115
