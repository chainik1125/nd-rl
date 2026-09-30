"""Batched greedy / temperature sampling with a KV cache. Left-pads prompts.

  generate(model, tok, prompts, greedy=True, temperature=1.0, max_new=400, batch=512, seed=0)
    -> list of proof-body strings (decoded, spec format), one per prompt.
"""
import torch, torch.nn.functional as F
from model import rope_cache


@torch.no_grad()
def generate_ids(model, tok, prompt_ids, greedy=True, temperature=1.0, max_new=400, gen=None):
    dev = next(model.parameters()).device
    B = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    # prefill: causal + key padding mask
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]  # pads attend to self (avoid NaN)
    caches = [{'max': L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    out = torch.full((B, max_new), tok.pad, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    for t in range(max_new):
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(logits.float() / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[:, t] = nxt
        done = done | (nxt == tok.eos)
        if bool(done.all()):
            break
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        mask = keep[:, None, None, :]
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=mask, caches=caches)[:, -1]
    return out.tolist()


def generate(model, tok, prompts, greedy=True, temperature=1.0, max_new=400, batch=512, seed=0):
    dev = next(model.parameters()).device
    gen = None
    if not greedy:
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = generate_ids(model, tok, [ids[i] for i in chunk], greedy, temperature, max_new, gen)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o)
        del outs
    if dev.type == 'cuda':
        torch.cuda.empty_cache()   # hand reserved memory back to co-tenant jobs
    return res
