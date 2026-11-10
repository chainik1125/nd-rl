#!/usr/bin/env python3
"""Pack a final checkpoint: policy weights, the seed's verifier, and the frozen solver settings.

    python pack.py --policy rl_s0/r27.pt --verifier rl_s0/verifier.pt --out final_s0.pt \
        [--solver '{"search": {"tau": 0.5, "tries": 4}, "plain": {"temperature": 1.0}}'] [--note ...]

`solve.py` reads `extra["verifier"]` (the separate verifier network) and `extra["solver"]`
(per-mode settings that override the library defaults; command-line flags override both).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--verifier", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver", default="{}")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    extra = dict(ck.get("extra") or {})
    extra["policy_file"] = os.path.abspath(a.policy)
    if a.verifier:
        from fr.verifier import load_verifier_spec

        extra["verifier"] = load_verifier_spec(a.verifier)
    extra["solver"] = json.loads(a.solver)
    extra["note"] = a.note
    ck["extra"] = extra
    torch.save(ck, a.out)
    n_pol = sum(t.numel() for t in ck["state"].values())
    vs = extra.get("verifier") or {}
    n_ver = sum(t.numel() for p in vs.get("parts", [vs]) for t in p.get("state", {}).values())
    print(json.dumps({"out": a.out, "policy_params": n_pol, "verifier_params": n_ver,
                      "solver": extra["solver"], "round": extra.get("round")}))  # fmt: skip


if __name__ == "__main__":
    main()
