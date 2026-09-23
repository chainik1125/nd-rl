"""Load the ws3 fixed checkpoint with the pinned fork code in this directory.

    import sys; sys.path.insert(0, 'code/ws3/fork_pin')
    from load_pretrain import load_pretrain
    model, tok, meta = load_pretrain()          # ckpt/pretrain.pt by default

The checkpoint is a raw state dict without an architecture config (see MANIFEST.json), so the
model shape is fixed here: 4 layers, d 256, 8 heads, tokenizer mode 'abs' (99 tokens).
"""

import hashlib
import json
import os

import torch
from model import GPT
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
N_LAYER, D, N_HEAD = 4, 256, 8


def load_pretrain(path=None, device="cpu", verify=True):
    path = path or os.path.join(HERE, "ckpt", "pretrain.pt")
    with open(os.path.join(HERE, "MANIFEST.json")) as f:
        manifest = json.load(f)
    if verify:
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        want = manifest["checkpoint"]["sha256"]
        assert got == want, f"checkpoint sha256 {got} != manifest {want}"
    ck = torch.load(path, map_location=device, weights_only=True)
    tok = Tokenizer(ck.get("tok_mode", "abs"))
    model = GPT(tok.vocab_size, N_LAYER, D, N_HEAD).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    meta = {k: v for k, v in ck.items() if k != "state"}
    meta["n_params"] = model.n_params()
    return model, tok, meta


if __name__ == "__main__":
    m, t, meta = load_pretrain()
    print(json.dumps(meta, indent=1, default=str))
