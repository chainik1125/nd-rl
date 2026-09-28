"""Is each run_<arm>/hf_base the pinned checkpoint, and how far did run_<arm>/final move? Own code.
Run from code/: uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/review/weights_check.py"""
import json
import os
import sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "..", "..", "..", "code", "ws3"))
import ndpath  # noqa: E402

pt = torch.load(ndpath.FORK_CKPT, map_location="cpu", weights_only=False)["state"]


def load(p):
    f = safe_open(os.path.join(p, "model.safetensors"), "pt")
    return {k: f.get_tensor(k) for k in f.keys()}


out = {}
for a in ["lr3e-5", "lr1e-4", "lr3e-4", "ei"]:
    base, final = load(os.path.join(EXP, f"run_{a}", "hf_base")), load(os.path.join(EXP, f"run_{a}", "final"))
    o = {"hf_base_vs_pretrain_max_abs": max((base["gpt." + k].float() - v.float()).abs().max().item() for k, v in pt.items()),
         "hf_base_tensors": len(base), "pt_tensors": len(pt)}
    common = [k for k in base if k in final]
    o["final_tensors_changed"] = sum((final[k] != base[k]).any().item() for k in common)
    o["final_vs_base_max_abs"] = max((final[k].float() - base[k].float()).abs().max().item() for k in common)
    o["final_vs_base_rel_fro"] = (sum(((final[k].float() - base[k].float()) ** 2).sum().item() for k in common)
                                  / sum((base[k].float() ** 2).sum().item() for k in common)) ** 0.5
    out[a] = o
    print(a, o)
json.dump(out, open(os.path.join(HERE, "weights_check.json"), "w"), indent=1)
