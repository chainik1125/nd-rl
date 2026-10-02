"""Plain temperature sampling with a compacting batch (finished rows leave the batch and the KV cache).

ab3/repr copy of lengthexp/sampler.py (same distribution as lib.sample.generate: left-padded prompts, KV cache,
bf16 autocast, softmax at temperature T, torch.multinomial with a seeded generator). Changes: positions come from
the model's init_pos / next_pos (lib/model.py), so every positional variant decodes exactly as its full forward;
for pe='rope' this is the lengthexp computation (tests/test_equivalence.py checks identical samples). Optional:
  greedy=True         argmax instead of sampling
  pos_tables          one list of positions per prompt (prompt tokens first, then one per generated token, at
                      least len(prompt) + max_new long): used instead of the model's positions (the pos-rand
                      random-position evaluation check)

    sample_ids(model, tok, prompts, temperature, max_new, batch, seed, greedy=False, pos_tables=None)
        -> list of token-id lists (QED included)
"""
import torch, torch.nn.functional as F


@torch.no_grad()
def sample_ids(model, tok, prompts, temperature=0.8, max_new=1024, batch=768, seed=0, compact_every=16, compact_below=0.8,
               greedy=False, pos_tables=None):
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        tabs = None if pos_tables is None else [pos_tables[i] for i in chunk]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = _gen(model, tok, [ids[i] for i in chunk], temperature, max_new, gen, compact_every, compact_below,
                        greedy, tabs)
        for i, o in zip(chunk, outs):
            res[i] = o
        del outs
        if dev.type == 'cuda':
            torch.cuda.empty_cache()   # compaction leaves variable-size KV blocks; release them between batches
    return res


def _gen(model, tok, prompt_ids, temperature, max_new, gen, compact_every, compact_below, greedy=False, tabs=None):
    dev = next(model.parameters()).device
    B = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    if tabs is None:
        pos = model.init_pos(keep)
        tab = None
    else:                                          # column c of the left-padded layout -> the row's own position
        tab = torch.zeros((B, L + max_new), dtype=torch.long, device=dev)
        for i, (p, t) in enumerate(zip(prompt_ids, tabs)):
            assert len(t) >= len(p) + max_new, 'pos_tables rows must cover prompt + max_new'
            tab[i, L - len(p):] = torch.tensor(t[:len(p) + max_new], device=dev)
        pos = tab[:, :L]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    caches = [{'max': L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    out = torch.full((B, max_new), tok.pad, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)            # original row of each live row
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    last = idx[:, -1]
    for t in range(max_new):
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(logits.float() / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[rows, t] = nxt
        done = done | (nxt == tok.eos)
        if t % compact_every == compact_every - 1 or t == 0:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < compact_below * done.numel():
                live = (~done).nonzero().squeeze(1)
                rows, done, nxt, cur_pos, keep, last = rows[live], done[live], nxt[live], cur_pos[live], keep[live], last[live]
                if tab is not None:
                    tab = tab[live]
                for c in caches:
                    c['k'], c['v'] = c['k'][live], c['v'][live]
        if t == max_new - 1:
            break
        cur_pos = model.next_pos(cur_pos, last) if tab is None else tab[:, L + t]
        last = nxt
        keep = torch.cat([keep, torch.ones(keep.size(0), 1, dtype=torch.bool, device=dev)], 1)
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=keep[:, None, None, :], caches=caches)[:, -1]
    out = out.tolist()
    eos, pad = tok.eos, tok.pad
    return [row[:row.index(eos) + 1] if eos in row else [x for x in row if x != pad] for row in out]
