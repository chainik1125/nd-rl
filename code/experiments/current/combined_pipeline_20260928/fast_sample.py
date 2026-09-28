# imports the fork, which is not resolvable here
# pyright: reportMissingImports=false
"""RL-stage sampler: the fork's sample.generate (pinned 51604b3) with two speedups, same
sampling distribution and the same Lean AND nd_verify gate. The harness eval keeps the fork's.

  - compaction (Dan's efficiency run; Leon's fast_generate): rows that emitted <eos> leave the
    decode batch and every KV cache, so a chunk stops paying for its finished rows
  - Lean overlap: each decoded chunk goes to lean_gate.gate on a worker thread while the GPU
    decodes the next chunk (the gate is Lean subprocesses, so it barely needs the GIL)

Row noise differs from the fork's (multinomial over fewer rows), not its distribution.
"""

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from lean_gate import gate  # fork

COMPACT_BELOW = 0.75  # compact once fewer than this fraction of the batch is still decoding
COMPACT_EVERY = 8


def _index_caches(caches, rows, B):
    for c in caches:
        for k, v in c.items():
            if torch.is_tensor(v) and v.size(0) == B:
                c[k] = v[rows]


@torch.no_grad()
def _decode(model, tok, prompt_ids, temperature, max_new, gen):
    """sample.generate_ids (non-greedy) with finished rows compacted out."""
    dev = next(model.parameters()).device
    B0 = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    idx = torch.full((B0, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B0, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p) :] = torch.tensor(p, device=dev)
        keep[i, L - len(p) :] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    caches = [{"max": L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    out = torch.full((B0, max_new), tok.pad, dtype=torch.long, device=dev)
    rows = torch.arange(B0, device=dev)  # live slot -> original row
    done = torch.zeros(B0, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    for t in range(max_new):
        probs = F.softmax(logits.float() / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[rows, t] = nxt
        done = done | (nxt == tok.eos)
        if t % COMPACT_EVERY == COMPACT_EVERY - 1:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < COMPACT_BELOW * done.numel():
                live = (~done).nonzero().squeeze(1)
                _index_caches(caches, live, done.numel())
                rows, nxt, cur_pos, keep, done = (
                    rows[live],
                    nxt[live],
                    cur_pos[live],
                    keep[live],
                    done[live],
                )
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(keep.size(0), 1, dtype=torch.bool, device=dev)], 1)
        logits = model(
            nxt[:, None], pos=cur_pos[:, None], mask=keep[:, None, None, :], caches=caches
        )[:, -1]
    return out.tolist()


def generate(model, tok, prompts, temperature, max_new, batch, seed):
    """Same contract as the fork's generate(greedy=False): ND proofs, Lean-rejected ones prefixed
    'LEANREJ ', in prompt order."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    futs = []
    with ThreadPoolExecutor(1) as pool:
        for s in range(0, len(order), batch):
            chunk = order[s : s + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                outs = _decode(model, tok, [ids[i] for i in chunk], temperature, max_new, gen)
            nd, texts = [], []
            for o in outs:
                nd.append(tok.decode(o))
                texts.append(tok.last_text)
            futs.append((chunk, pool.submit(gate, tok, [prompts[i] for i in chunk], nd, texts)))
        for chunk, fut in futs:
            for i, r in zip(chunk, fut.result(), strict=True):
                res[i] = r
    return res
