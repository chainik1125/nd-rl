#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Local checks of pretrain_coconut.py on CPU, against the fork's own sampler path.

  ND_FORK=<nd-takehome at 51604b3> ND_ARM=<arm> python check_model.py

1. The training pass (prompt_pass + finish on right-padded [prompt, proof] rows) and the sampling
   pass (left-padded prefill with the thoughts inside it, then one token at a time through the
   KV cache) give the same next-token logits, for several thought counts.
2. With zero thoughts the model is run 198's: same logits as a plain causal ALiBi forward.
3. Each arm's loss runs and backpropagates on real records.
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.environ["ND_FORK"])
import pretrain_coconut as P  # noqa: E402
import torch  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402

torch.manual_seed(0)
tok = make_tokenizer(P.TOK_MODE)
with open(f"{os.environ['ND_FORK']}/data/train.jsonl") as f:
    recs = [json.loads(line) for line in itertools.islice(f, 0, 4000, 500)]
data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in recs]
model = P.ThoughtGPT(tok.vocab_size, P.N_LAYER, P.D, P.N_HEAD, data[0][0][-1], seed=0)
model.blocks = torch.nn.ModuleList(P.PeriLNBlock(P.D, P.N_HEAD) for _ in range(P.N_LAYER))
aux = P.Aux(P.D, P.N_HEAD)
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

with torch.no_grad():
    for T in (0, 1, 3, 11):
        model.n_thoughts = T
        full = model(ids)  # (B, S, V), row i's proof predictions start at plen[i] - 1
        caches = [{"max": L + n_new} for _ in model.blocks]
        step = [model(left, pos=pos, mask=prefill, caches=caches)[:, -1]]
        k, cur = keep, pos[:, -1]
        for j in range(n_new):
            nxt = ids[torch.arange(B), plen + j]
            cur = cur + 1
            k = torch.cat([k, torch.ones(B, 1, dtype=torch.bool)], 1)
            step.append(
                model(nxt[:, None], pos=cur[:, None], mask=k[:, None, None, :], caches=caches)[
                    :, -1
                ]
            )
        want = torch.stack([full[i, plen[i] - 1 : plen[i] + n_new] for i in range(B)])
        err = (torch.stack(step, 1) - want).abs().max().item()
        print(f"T {T}: max |sampling - training| = {err:.2e}")
        assert err < 2e-3, err
    model.n_thoughts = 0
    x = model.emb(ids)
    for b in model.blocks:
        x = b(x, None, None, model.causal_bias(S, "cpu"), None)
    err = (model.head(model.ln_f(x)) - model(ids)).abs().max().item()
    print(f"T 0 vs plain causal forward: {err:.2e}")
    assert err < 2e-3, err
    model.n_thoughts = 3
    assert (model(ids) - model.head(model.ln_f(x))).abs().max() > 1e-2, "thoughts are a no-op"

model.train()
plan, ends = P.plans(recs, data, tok, "cpu")
print("plans", [[P.RULES[r] for r in row if r != P.PLAN_END] for row in plan.tolist()][:3])
for frac in (0.0, 0.3, 0.9):
    total, main, mtp, arm = P.arm_loss(model, aux, ids, mask, plan, ends, frac)
    total.backward()
    print(
        f"{P.ARM} at frac {frac}: total {total.item():.3f} main {main.item():.3f} "
        f"arm {arm.item():.3f}"
    )
print("ok")
