#!/usr/bin/env python3
"""Validation of score_loglik.py on a pool: (1) the teacher-forced forward (right-padded, causal ALiBi) matches the
sampler's own forward (left-padded, cached prefill as in sample.generate_ids_fast) on the first action token's
distribution; (2) greedy re-decode (sample.generate_ids_fast, greedy) of each step (done attempts only)'s state reproduces the stored action
(outside masked name positions) for the highest-likelihood attempts; (3) summary of masked-name-position log-probs."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_loglik import replay, score_pairs, load_model, _tok
sys.path.insert(0, os.path.expanduser('~/runs/proof-selection/nd-takehome'))
import torch, torch.nn.functional as F
from sample import generate_ids_fast

pool, scores_dir = sys.argv[1], sys.argv[2]
n_att = int(sys.argv[3]) if len(sys.argv) > 3 else 200
stem = os.path.basename(pool)[:-6]
rows = [json.loads(l) for l in open(pool)]
lm = [json.loads(l)['scores'] for l in open(os.path.join(scores_dir, f'{stem}__loglik_mean.jsonl'))]
cands = [(lm[ri][ai], ri, ai) for ri, r in enumerate(rows) for ai in range(len(r['attempts'])) if r['attempts'][ai]['done']]
cands.sort(reverse=True)
random.seed(0)
top = cands[:n_att]
rnd = random.sample(cands, min(n_att, len(cands)))
dev = torch.device('cuda')
model = load_model(dev)
tok = _tok()

def check(sel, label):
    pairs = []
    for _, ri, ai in sel:
        r = rows[ri]; at = r['attempts'][ai]
        rp = replay(dict(at, prompt=r['prompt'], max_action=512))
        pairs += rp['steps']
    # (1) first-token distribution, sampler path vs teacher-forced path
    st = [p[0] for p in pairs]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        L = max(len(p) for p in st)
        idx = torch.full((len(st), L), tok.pad, dtype=torch.long, device=dev)
        keep = torch.zeros((len(st), L), dtype=torch.bool, device=dev)
        for i, p in enumerate(st):
            idx[i, L - len(p):] = torch.tensor(p, device=dev); keep[i, L - len(p):] = True
        pos = (keep.cumsum(1) - 1).clamp(min=0)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))[None, None] & keep[:, None, None, :]
        mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
        caches = [{'max': L + 4} for _ in model.blocks]
        with torch.no_grad():
            lg = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    lp_samp = F.log_softmax(lg.float(), -1)
    tf = score_pairs(model, pairs, dev)
    first_tf = torch.tensor([t[0][0] if t[0][0] is not None else float('nan') for t in tf])
    first_samp = torch.tensor([lp_samp[i, pairs[i][1][0]].item() if pairs[i][1] else float('nan') for i in range(len(pairs))])
    ok = ~torch.isnan(first_tf)
    d = (first_tf[ok] - first_samp[ok]).abs()
    # (2) greedy re-decode
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        outs = generate_ids_fast(model, tok, st, goals=None, greedy=True, max_new=512, early='eos', compact=True)
    exact = agree_unmasked = 0
    for (s, a, m, ended), o in zip(pairs, outs):
        g, end = tok.decode_action(o)
        gi = [tok.stoi[x] for x in g]
        if gi == a: exact += 1
        if len(gi) == len(a) and all(gi[j] == a[j] for j in range(len(a)) if j not in m): agree_unmasked += 1
    print(json.dumps(dict(sel=label, n_attempts=len(sel), n_steps=len(pairs),
                          first_tok_absdiff_max=d.max().item(), first_tok_absdiff_mean=d.mean().item(),
                          greedy_step_exact=exact / len(pairs), greedy_step_agree_unmasked=agree_unmasked / len(pairs),
                          mean_loglik_sel=sum(c[0] for c in sel) / len(sel))))



@torch.no_grad()
def cached_check(sel, label, max_pairs=600):
    """every action token's log p via the sampler's cached incremental path (left-padded prefill, then one token per
    step with the KV cache, as generate_ids_fast) vs the batched teacher-forced path; includes failed / truncated
    attempts' last actions and box transitions."""
    pairs = []
    for _, ri, ai in sel:
        r = rows[ri]; at = r['attempts'][ai]
        pairs += replay(dict(at, prompt=r['prompt'], max_action=512))['steps']
    pairs = pairs[:max_pairs]
    tf = score_pairs(model, pairs, dev)
    diffs = []
    for c0 in range(0, len(pairs), 128):
        ch = pairs[c0:c0 + 128]
        st = [p[0] for p in ch]
        tg = [list(p[1]) + ([tok.eos] if p[3] else []) for p in ch]
        A = max(len(t) for t in tg)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            L = max(len(p) for p in st)
            idx = torch.full((len(st), L), tok.pad, dtype=torch.long, device=dev)
            keep = torch.zeros((len(st), L), dtype=torch.bool, device=dev)
            for i, p in enumerate(st):
                idx[i, L - len(p):] = torch.tensor(p, device=dev); keep[i, L - len(p):] = True
            pos = (keep.cumsum(1) - 1).clamp(min=0)
            mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))[None, None] & keep[:, None, None, :]
            mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
            caches = [{'max': L + A + 1} for _ in model.blocks]
            logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
            keepb = torch.ones((len(st), L + A + 1), dtype=torch.bool, device=dev); keepb[:, :L] = keep
            cur = pos[:, -1]
            T = torch.full((len(st), A), tok.pad, dtype=torch.long, device=dev)
            for i, t in enumerate(tg):
                T[i, :len(t)] = torch.tensor(t, device=dev)
            lps = []
            for t in range(A):
                lp = F.log_softmax(logits.float(), -1)
                lps.append(lp.gather(-1, T[:, t:t + 1])[:, 0])
                cur = cur + 1
                logits = model(T[:, t:t + 1], pos=cur[:, None], mask=keepb[:, None, None, :L + t + 1], caches=caches)[:, -1]
        lps = torch.stack(lps, 1).cpu()
        for i, t in enumerate(tg):
            full = tf[c0 + i][3]
            diffs += [abs(full[j] - lps[i, j].item()) for j in range(len(t))]
    d = torch.tensor(diffs)
    n_fail_last = sum(1 for p in pairs if not p[3])
    print(json.dumps(dict(sel=label, n_pairs=len(pairs), n_tokens=len(diffs), truncated_actions=n_fail_last,
                          tf_vs_cached_absdiff_max=d.max().item(), tf_vs_cached_absdiff_mean=d.mean().item(),
                          frac_gt_0p05=(d > 0.05).float().mean().item())))


check(top, 'top_loglik_mean')
check(rnd, 'random')
allc = [(lm[ri][ai], ri, ai) for ri, r in enumerate(rows) for ai in range(len(r['attempts']))]
fails = [c for c in allc if not rows[c[1]]['attempts'][c[2]]['done']]
cached_check(random.sample(fails, min(n_att, len(fails))), 'failed_attempts')
cached_check(random.sample(cands, min(n_att, len(cands))), 'done_attempts')
