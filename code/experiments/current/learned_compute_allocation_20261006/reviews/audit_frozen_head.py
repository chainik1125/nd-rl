"""Independent CPU audit of cached frozen features and the saved scalar head; no fit."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
W = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
B = ROOT / "artifacts/learned_compute_allocation_20261006"
D = B / "selectors"
C = D / "frozen_head_features_v1"
H = D / "frozen_head_v1"
torch.set_num_threads(2)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(line) for line in Path(p).open()]


ec = read(C / "config.json")
hc = read(H / "config.json")
for cfg, keys in [(ec, ["source_hashes"]), (hc, ["source_hashes", "input_hashes"])]:
    for key in keys:
        for path, value in cfg[key].items():
            assert sha(path) == value, (path, value)
assert sha(H / "head.pt") == hc["head_sha256"]
sys.path.insert(0, str(B / "source/nd-takehome-dan"))
from model import load_ckpt  # noqa: E402 - runtime source path

model, tok, _ = load_ckpt(ec["args"]["checkpoint"], "cpu")
dig = hashlib.sha256()
for name, p in sorted(model.named_parameters()):
    a = p.detach().contiguous().numpy()
    dig.update(name.encode())
    dig.update(str(a.dtype).encode())
    dig.update(str(a.shape).encode())
    dig.update(a.tobytes())
assert (
    dig.hexdigest()
    == ec["parameter_sha256_before"]
    == ec["parameter_sha256_after"]
    == hc["frozen_original_parameter_sha256"]
)
head = torch.load(H / "head.pt", map_location="cpu", weights_only=True)
w = head["weight"].numpy()
bias = head["bias"].item()
assert w.shape == (384,) and w.dtype == np.float64
result = {
    "original_parameters": sum(p.numel() for p in model.parameters()),
    "original_parameter_sha256": dig.hexdigest(),
    "head_parameters": 385,
    "splits": {},
}
keysets = {}
for split in ["train", "dev"]:
    cache = np.load(C / f"{split}_features.npz", allow_pickle=False)
    meta = read(C / f"{split}_metadata.json")
    assert sha(C / f"{split}_features.npz") == meta["cache_sha256"]
    assert sha(meta["public"]) == meta["public_sha256"]
    pubs = rows(meta["public"])
    assert all(r["split"] == split for r in pubs)
    lookup = {(r["theorem_id"], c["candidate_id"]): (r, c) for r in pubs for c in r["candidates"]}
    keys = list(zip(cache["theorem_ids"], cache["candidate_ids"], strict=False))
    assert len(keys) == len(set(keys)) and set(keys) == set(lookup)
    keysets[split] = {k[0] for k in keys}
    lengths = []
    for key in keys:
        r, c = lookup[key]
        lengths.append(len(tok.encode_prompt(r["prompt"])) + len(c["transition_tokens"]) + 5)
    assert lengths == cache["input_tokens"].tolist()
    priv = rows(hc["args"][f"{split}_private"])
    ys = {(r["theorem_id"], c["candidate_id"]): c["valid"] for r in priv for c in r["labels"]}
    assert set(ys) == set(keys)
    x = cache["features"].astype(np.float64)
    y = np.array([ys[k] for k in keys], dtype=np.float64)
    z = x @ w + bias
    p = 1 / (1 + np.exp(-z))
    loss = np.mean(np.logaddexp(0, z) - y * z)
    item = {
        "candidates": len(keys),
        "theorems": len(keysets[split]),
        "binary_logloss": float(loss),
        "brier": float(np.mean((p - y) ** 2)),
        "max_input_tokens": max(lengths),
    }
    if split == "train":
        grad = x.T @ (p - y) / len(y) + 1e-3 * w
        gb = np.mean(p - y)
        item.update(
            objective=float(loss + 0.5e-3 * np.dot(w, w)),
            max_absolute_gradient=float(max(np.abs(grad).max(), abs(gb))),
        )
        assert abs(item["objective"] - hc["convergence"]["objective"]) < 1e-12
        assert (
            abs(item["max_absolute_gradient"] - hc["convergence"]["max_absolute_gradient"]) < 1e-12
        )
    else:
        preds = read(H / "dev_predictions.json")
        saved = {r["theorem_id"]: r for r in preds}
        correct = []
        errs = []
        for tid in keysets[split]:
            ix = [i for i, k in enumerate(keys) if k[0] == tid]
            zz = z[ix]
            tied = np.flatnonzero(zz == zz.max())
            correct.append(y[ix][tied].mean())
            errs.extend(np.abs(zz - np.array(saved[tid]["candidate_validity_logits"])))
        item.update(accuracy=float(np.mean(correct)), max_saved_logit_error=float(max(errs)))
        assert item["accuracy"] == read(H / "dev_summary.json")["accuracy"]
        assert max(errs) < 1e-12
    result["splits"][split] = item
assert keysets["train"].isdisjoint(keysets["dev"])
# Structural CPU readout equivalence at actual public factorized inputs; not GPU/BF16 reproduction.
r = pubs[0]
sequences = [
    [tok.stoi["<st>"]]
    + tok.encode_prompt(r["prompt"])
    + [tok.stoi["<nl>"], tok.stoi["<act>"]]
    + c["transition_tokens"]
    + [tok.stoi["<nl>"], tok.stoi["<act>"]]
    for c in r["candidates"][:2]
]
maxerr = 0.0
with torch.no_grad():
    for seq in sequences:
        ids = torch.tensor([seq])
        full = model(ids)[0, -1]
        hidden = model.ln_f(model.features(ids)[0])[0, -1]
        headout = model.head(hidden)
        maxerr = max(maxerr, float((full - headout).abs().max()))
result["cpu_original_forward_vs_extracted_feature_head_max_error"] = maxerr
Path(__file__).with_name("frozen_head_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
