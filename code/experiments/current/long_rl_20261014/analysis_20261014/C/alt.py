"""Counterfactual continuation test at decision points.
For each (rec, token position j, forced token): prefix = prompt + gold[:j] + forced; sample k continuations at T=0.8,
Lean-judge the full proof. Also returns the greedy line written after the forced token (for order classification)."""
import sys, os, torch
sys.path.insert(0, '..')
import ndmodel as M
H = M.H


def run(model, tok, items, k=4, seed=0, batch=8):
    """items: list of dict(prompt, proof_ids (gold proof ids incl eos), j, forced (token id), body).
    returns per item: list of (lean_ok, nd) and greedy continuation token strings."""
    gen = torch.Generator(); gen.manual_seed(seed)
    jobs = []
    for a, it in enumerate(items):
        pre = tok.encode_prompt(it['prompt']) + it['proof_ids'][:it['j']] + [it['forced']]
        for s in range(k):
            jobs.append((a, pre, H.new_tokens(it['body'], H.MAX_NEW)))
    order = sorted(range(len(jobs)), key=lambda q: (jobs[q][2], len(jobs[q][1])))
    outs = [None] * len(jobs)
    for s in range(0, len(order), batch):
        ch = order[s:s + batch]
        cap = max(jobs[q][2] for q in ch)
        with torch.no_grad():
            res = H.generate_ids(model, tok, [jobs[q][1] for q in ch], False, H.TEMP, cap, gen)
        for q, o in zip(ch, res):
            outs[q] = o
    prompts, nds, texts = [], [], []
    for (a, pre, cap), o in zip(jobs, outs):
        it = items[a]
        o = o[:o.index(tok.eos) + 1] if tok.eos in o else o
        full = it['proof_ids'][:it['j']] + [it['forced']] + o
        nd = tok.decode(full)
        prompts.append(it['prompt']); nds.append(nd); texts.append(tok.last_text)
    gated = H.gate(tok, prompts, nds, texts)
    res = [[] for _ in items]
    for (a, _, _), nd, o in zip(jobs, gated, outs):
        res[a].append((not nd.startswith('LEAN'), nd, [tok.itos[x] for x in o[:60]]))
    return res
