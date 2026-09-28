# imports the fork (tokenizer) and the 186-claude pretrain.py, neither resolvable here
# pyright: reportMissingImports=false
"""Uploaded to the pod as pretrain.py. Instead of pretraining, returns the 186-claude incumbent's
pretrained model for this seed (its pretrain.pt, uploaded next to this file as pretrain_ckpt.pt),
so every arm of the combined-pipeline comparison starts RL from the same weights.

pretrain186.py is 186-claude's pretrain.py verbatim (sha f8451342...), imported for its model
classes only.
"""

import hashlib
import os
import time

import torch
from pretrain186 import N_HEAD, N_LAYER, TOK_MODE, ALiBiGPT, D, PeriLNBlock
from tokenizer import make_tokenizer  # fork

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "pretrain_ckpt.pt")


def pretrain(train_recs, budget_secs, device, seed, log):
    t0 = time.time()
    ck = torch.load(CKPT, map_location="cpu")
    assert ck["stage"] == "pretrain" and ck["seed"] == seed, (ck["stage"], ck["seed"], seed)
    assert ck["tok_mode"] == TOK_MODE
    tok = make_tokenizer(TOK_MODE)
    model = ALiBiGPT(tok.vocab_size, N_LAYER, D, N_HEAD)
    model.blocks = torch.nn.ModuleList(PeriLNBlock(D, N_HEAD) for _ in range(N_LAYER))
    model.load_state_dict(ck["state"])
    model = model.to(device)
    with open(CKPT, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:12]
    log(f"loaded {ck['pretrain_sha']} pretrain.pt s{seed} (sha {sha}) in {time.time() - t0:.0f}s")
    stats = {"curve": [], "loaded_ckpt_sha": sha, "source_pretrain_sha": ck["pretrain_sha"]}
    return model, tok, stats
