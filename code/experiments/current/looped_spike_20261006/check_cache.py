#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Cached decoding must equal the full forward at every loop count, on left-padded batches.

ND_FORK=<nd-takehome at 51604b3> python check_cache.py
"""

import os
import sys

sys.path.insert(0, os.environ["ND_FORK"])
import pretrain_looped as P  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
V, B, L, NEW = 50, 3, 9, 6
model = P.LoopedGPT(V, P.N_LAYER, P.D, P.N_HEAD, seed=0)
model.blocks = torch.nn.ModuleList(P.PeriLNBlock(P.D, P.N_HEAD) for _ in range(P.N_LAYER))
for p in model.parameters():  # fork init is near-identity; make every block matter
    torch.nn.init.normal_(p, std=0.3)
model.eval()
idx = torch.randint(1, V, (B, L + NEW))
keep = torch.ones(B, L + NEW, dtype=torch.bool)
keep[1, :4] = keep[2, :7] = False  # left padding, as sample.generate_ids builds it
pos = (keep.cumsum(1) - 1).clamp(min=0)


def mask_for(T):
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    return (causal[None, None] & keep[:, None, None, :T]) | torch.eye(T, dtype=torch.bool)


with torch.no_grad():
    for n in (1, 2, 3, 5):
        model.n_loops = n
        full = model(idx, pos=pos, mask=mask_for(L + NEW))
        caches = [{"max": L + NEW} for _ in model.blocks]
        step = [model(idx[:, :L], pos=pos[:, :L], mask=mask_for(L), caches=caches)[:, -1]]
        for t in range(L, L + NEW):
            m = keep[:, None, None, : t + 1]
            step.append(
                model(idx[:, t : t + 1], pos=pos[:, t : t + 1], mask=m, caches=caches)[:, -1]
            )
        err = (torch.stack(step, 1) - full[:, L - 1 :]).abs().max().item()
        print(f"n_loops {n}: max |cached - full| = {err:.2e}")
        assert err < 1e-3, err
    one, model.n_loops = full, 2
    assert (model(idx, pos=pos, mask=mask_for(L + NEW)) - one).abs().max() > 1e-2, (
        "loops are a no-op"
    )
    model.train()
    seen = {model(idx[:, :4]).sum().item() for _ in range(40)}
    assert len(seen) == len(P.TRAIN_LOOPS), (len(seen), P.TRAIN_LOOPS)
print(f"ok; {sum(p.numel() for p in model.parameters()):,} params")
