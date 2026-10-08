# imports the fork (lean_gate, nd_verify), which is not resolvable here
# pyright: reportMissingImports=false
"""Multi-attempt sampling. An episode is one theorem and up to `max_attempts` proofs of it:

  STATEMENT attempt_1 <eos> ERR [name]  STATEMENT attempt_2 <eos> ERR [name]  ...
  STATEMENT proof <eos>

After a failed attempt the environment appends the error tokens (errors.py) and the statement
again, and the model writes a whole new proof. The statement is repeated because half the heads
are ALiBi with slopes down to 1/256: by the third attempt the original statement is ~1,000 tokens
back and those heads can no longer read the premises. An episode ends at the first accepted
proof, at max_attempts, or when the next attempt would not fit in MAX_CTX.

Acceptance is the harness's rule: the fork's Lean gate AND nd_verify.
"""

import torch
from errors import classify, error_ids
from lean_gate import gate  # fork
from nd_verify import verify_text  # fork

MAX_CTX = 4608  # prefix + max_new must fit: 8 attempts of ~500 tokens
ROW_TOKENS = 1_000_000  # rows * (prefix + max_new) per decode batch; the KV cache is ~10 GB here
PREFILL_CHUNK = 128  # the ALiBi bias is (rows, heads, chunk, keys) floats: ~5 GB at the cap


def context_ids(prompt_ids, fails):
    """Token ids a new attempt is conditioned on: fails is [(attempt ids to <eos>, error ids)]."""
    out = list(prompt_ids)
    for attempt, err in fails:
        out += attempt + err + prompt_ids
    return out


def trim(ids, tok):
    """Sampled ids up to and including <eos>; all of them if the attempt never ended."""
    ids = [x for x in ids if x != tok.pad]
    return ids[: ids.index(tok.eos) + 1] if tok.eos in ids else ids


@torch.no_grad()
def generate_ids(model, tok, prefixes, temperature, max_new, gen):
    """The fork's sample.generate_ids for long prefixes: the prefill runs in chunks through the
    KV cache instead of one (rows, heads, L, L) attention bias. Sampling only."""
    dev = next(model.parameters()).device
    B, L = len(prefixes), max(len(p) for p in prefixes)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prefixes):
        idx[i, L - len(p) :] = torch.tensor(p, device=dev)
        keep[i, L - len(p) :] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    ar = torch.arange(L, device=dev)
    caches = [{"max": L + max_new} for _ in model.blocks]
    logits = None
    for s in range(0, L, PREFILL_CHUNK):
        e = min(L, s + PREFILL_CHUNK)
        causal = ar[s:e, None] >= ar[None, :e]
        mask = causal[None, None] & keep[:, None, None, :e]
        mask = mask | (ar[s:e, None] == ar[None, :e])[None, None]  # pads attend to themselves
        logits = model(idx[:, s:e], pos=pos[:, s:e], mask=mask, caches=caches)[:, -1]
    out = torch.full((B, max_new), tok.pad, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    for t in range(max_new):
        probs = torch.softmax(logits.float() / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[:, t] = nxt
        done = done | (nxt == tok.eos)
        if bool(done.all()):
            break
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        logits = model(
            nxt[:, None], pos=cur_pos[:, None], mask=keep[:, None, None, :], caches=caches
        )[:, -1]
    return out.tolist()


def generate(model, tok, prefixes, temperature, max_new, gen):
    """One sampled attempt per prefix, in length-sorted batches sized to ROW_TOKENS."""
    dev = next(model.parameters()).device
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    res, s = [None] * len(prefixes), 0
    while s < len(order):
        rows = max(1, min(1024, ROW_TOKENS // (len(prefixes[order[s]]) + max_new)))
        chunk = order[s : s + rows]
        # sorted ascending: shrink the batch until its longest prefix fits the budget too
        while len(chunk) > 1 and len(chunk) * (len(prefixes[chunk[-1]]) + max_new) > ROW_TOKENS:
            chunk = chunk[: len(chunk) // 2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            outs = generate_ids(model, tok, [prefixes[i] for i in chunk], temperature, max_new, gen)
        for i, o in zip(chunk, outs, strict=True):
            res[i] = o
        s += len(chunk)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def sample_episodes(model, tok, recs, n_ep, max_attempts, mode, temperature, seed, max_new):
    """n_ep episodes per record. mode: 'error' or 'blank' (errors.error_ids).

    Returns one list of episodes per record. An episode is
      {"fails": [(attempt ids, error ids), ...], "codes": [error code per failed attempt],
       "proof": accepted ND proof or None, "n_tok": [sampled tokens per attempt]}.
    """
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    prompt_ids = [tok.encode_prompt(r["prompt"]) for r in recs]
    eps = [
        [{"fails": [], "codes": [], "proof": None, "n_tok": []} for _ in range(n_ep)] for _ in recs
    ]
    active = [(i, j) for i in range(len(recs)) for j in range(n_ep)]
    for _ in range(max_attempts):
        if not active:
            break
        prefixes = [context_ids(prompt_ids[i], eps[i][j]["fails"]) for i, j in active]
        outs = generate(model, tok, prefixes, temperature, max_new, gen)
        prompts = [recs[i]["prompt"] for i, _ in active]
        nds, texts = [], []
        for o in outs:
            nds.append(tok.decode(o))
            texts.append(tok.last_text)
        gated = gate(tok, prompts, nds, texts)
        nxt = []
        for (i, j), o, nd, p in zip(active, outs, gated, prompts, strict=True):
            ep, ids = eps[i][j], trim(o, tok)
            ep["n_tok"].append(len(ids))
            if verify_text(p + " " + nd)[0]:
                ep["proof"] = nd
                continue
            ep["fails"].append((ids, error_ids(tok, p, ids, mode)))
            ep["codes"].append(classify(tok, p, ids)[0])
            if len(context_ids(prompt_ids[i], ep["fails"])) + max_new <= MAX_CTX:
                nxt.append((i, j))
        active = nxt
    return eps
