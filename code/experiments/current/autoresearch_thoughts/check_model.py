#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Local CPU checks of pretrain.py's model, against the fork's sampler calling convention.

  ND_FORK=<nd-takehome at 51604b3> python check_model.py

1. Training (run() on right-padded [prompt, proof] rows) and sampling (left-padded prefill, then
   one token at a time through cached()) give the same next-token logits at several thought
   counts, on real records with different numbers of lines.
2. With zero thoughts both equal a plain causal forward: run 198's network.
3. Thoughts change the output, and train_loss runs and backpropagates.
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.environ["ND_FORK"])
import pretrain as P  # noqa: E402
import torch  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402

torch.manual_seed(0)
tok = make_tokenizer(P.TOK_MODE)
with open(f"{os.environ['ND_FORK']}/data/train.jsonl") as f:
    recs = [json.loads(line) for line in itertools.islice(f, 0, 150000, 15000)]
data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in recs]
triggers = (tok.stoi[";"], tok.stoi["by"])
model = P.LineThoughtGPT(tok.vocab_size, P.N_LAYER, P.D, P.N_HEAD, triggers, seed=0)
model.blocks = torch.nn.ModuleList(P.PeriLNBlock(P.D, P.N_HEAD) for _ in range(P.N_LAYER))
aux = P.Aux(P.D, P.N_HEAD, tok.vocab_size)
for p in list(model.parameters()) + list(aux.parameters()):  # make every block and bias matter
    torch.nn.init.normal_(p, std=0.2)
model.eval()

ids, mask, _, lens = P.to_gpu(data, tok, "cpu")
B, S = ids.shape
plen = torch.tensor([len(p) for p, _ in data])
L = int(plen.max())
left = torch.full((B, L), tok.pad)
keep = torch.zeros((B, L), dtype=torch.bool)
for i, (p, _) in enumerate(data):
    left[i, L - len(p) :] = torch.tensor(p)
    keep[i, L - len(p) :] = True
pos = (keep.cumsum(1) - 1).clamp(min=0)
causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
prefill = (causal[None, None] & keep[:, None, None, :]) | torch.eye(L, dtype=torch.bool)[None, None]
n_new = int((lens - plen).min()) - 1  # proof tokens fed one at a time, in every row
lines = [(ids[i, : lens[i]] == triggers[0]).sum().item() + 1 for i in range(B)]
print(f"{B} records, lines {lines}, {n_new} proof tokens compared per row")

with torch.no_grad():
    plain = None
    for n in (0, 1, 3, 8, 16):
        model.n_thoughts = n
        full = model(ids)  # (B, S, V); row i's proof predictions start at plen[i] - 1
        caches = [{"max": L + n_new} for _ in model.blocks]
        step = [model(left, pos=pos, mask=prefill, caches=caches)[:, -1]]
        k, cur = keep, pos[:, -1]
        for j in range(n_new):
            nxt = ids[torch.arange(B), plen + j]
            cur = cur + 1
            k = torch.cat([k, torch.ones(B, 1, dtype=torch.bool)], 1)
            m = k[:, None, None, :]
            step.append(model(nxt[:, None], pos=cur[:, None], mask=m, caches=caches)[:, -1])
        want = torch.stack([full[i, plen[i] - 1 : plen[i] + n_new] for i in range(B)])
        err = (torch.stack(step, 1) - want).abs().max().item()
        print(f"n {n}: max |sampling - training| = {err:.2e}")
        assert err < 2e-3, err
        if n == 0:
            x, ar = model.emb(ids), torch.arange(S)[None]
            for b in model.blocks:
                x = b(x, None, None, model.bias(ar, ar, ar[:, None, :] <= ar[:, :, None]), None)
            plain = model.head(model.ln_f(x))
            assert (plain - full).abs().max() < 2e-3
        else:
            assert (plain - full).abs().max() > 1e-2, "thoughts are a no-op"

model.train()
for _ in range(4):
    total, main, mtp, probe = P.train_loss(model, aux, ids, mask, tok)
    total.backward()
    print(f"train_loss: total {total.item():.3f} main {main.item():.3f} probe {probe.item():.3f}")
moves = torch.tensor([tok.stoi[":="], tok.stoi["exact"]])
label = P.next_move(ids[:1], moves)[0]
at = torch.isin(ids[0], model.triggers).nonzero()[:, 0].tolist()
print("row 0 moves at its triggers:", [tok.itos[label[i]] for i in at if label[i] >= 0])
print(" ".join(tok.itos[t] for t in ids[0, plen[0] : lens[0]].tolist())[:300])
print("ok")
