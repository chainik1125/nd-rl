"""Is run/hf_base the pinned checkpoint, and did run/final actually move? Own code."""
import os, sys, json, torch
from safetensors import safe_open
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.join(os.path.dirname(HERE), "run")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "..", "..", "code", "ws3")); import ndpath
pt = torch.load(ndpath.FORK_CKPT, map_location="cpu", weights_only=False)["state"]
def load(d):
    f = safe_open(os.path.join(RUN, d, "model.safetensors"), "pt")
    return {k: f.get_tensor(k) for k in f.keys()}
base, final = load("hf_base"), load("final")
out = {"pt_tensors": len(pt), "hf_base_tensors": len(base), "final_tensors": len(final)}
mx = 0.0; missing = []
for k, v in pt.items():
    kk = "gpt." + k
    if kk not in base: missing.append(k); continue
    mx = max(mx, (base[kk].float() - v.float()).abs().max().item())
out["hf_base_vs_pretrain_pt_max_abs_diff"] = mx; out["pt_keys_missing_in_hf_base"] = missing
out["hf_base_extra_keys"] = sorted(set(base) - {"gpt." + k for k in pt})
d = {k: (final[k].float() - base[k].float()).abs().max().item() for k in base}
out["final_vs_hf_base_max_abs_diff"] = max(d.values()); out["final_tensors_changed"] = sum(1 for v in d.values() if v > 0)
out["final_vs_hf_base_rel_fro"] = (sum(((final[k].float() - base[k].float())**2).sum().item() for k in base) /
                                   sum((base[k].float()**2).sum().item() for k in base)) ** 0.5
out["n_params"] = sum(v.numel() for v in pt.values())
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "weights_check.json"), "w"), indent=1)
