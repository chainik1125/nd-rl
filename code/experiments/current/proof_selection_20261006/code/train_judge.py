#!/usr/bin/env python3
"""Phase 4: trained likelihood judge (no head), pre-registered 05:28 UTC in logs/worker_A.md.

init = sibling generator s1; data = s0 train-split pools; scoring later by Worker B's score_loglik.py --ckpt.
  --mode sft : CE on action tokens (+eos) of Lean-valid finished attempts, (state, effective action) pairs
  --mode pair: DPO-style logistic on masked-mean-logp differences (valid vs invalid finished, same theorem) relative to
               the frozen s1 reference + lambda * token KL(ref || policy) anchor
Early stop on dev within-theorem finished-only AUROC of the masked mean log p (same quantity as B's scorer).
Checkpoints are saved in Dan's format (load_ckpt-compatible) at ~/runs/proof-selection/judge/<name>/best.pt
"""
import argparse, json, math, multiprocessing as mp, os, pickle, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn.functional as F
import pool_lib as PL
import score_loglik as SL
from model import load_ckpt
from evaluate_selection import auroc

S1 = os.path.join(PL.RUNS, 'ckpts', 'la_T1_best12_s1_r16.pt')
POOLS = ['rl_targets:16', 'dev1108:16', 'rr600:32', 'rr600:96b', 'rl_targets:48b', 'dev1108:48b']
CACHE = os.path.join(PL.RUNS, 'judge', 'cache.pkl')
TOK = SL._tok()


def _prep(job):
    at, prompt = job
    rp = SL.replay(dict(at, prompt=prompt, max_action=512))
    _, pairs = PL.replay(prompt, at['base'], at['actions'], TOK, at['has_eos'])
    eff = [(st, act + ([TOK.eos] if at['has_eos'][t] else [])) for t, (st, act) in enumerate(pairs)]
    return dict(steps=rp['steps'], eff=eff)


def build_cache():
    sp = json.load(open(os.path.join(PL.SPRINT, 'results', 'splits.json')))
    items, jobs = [], []
    for spec in POOLS:
        s, k = spec.split(':')
        lab = PL.load_labels(s, k)
        for r in PL.load_pool(s, k):
            split = 'train' if r['name'] in set(sp['train'].get(s, [])) else ('dev' if r['name'] in set(sp['dev'].get(s, [])) else None)
            if split is None:
                continue
            for a in r['attempts']:
                if a['done']:
                    items.append(dict(set=s, k=k, name=r['name'], i=a['i'], split=split, valid=lab[r['name']][a['i']]))
                    jobs.append((a, r['prompt']))
    with mp.Pool(48) as p:
        res = p.map(_prep, jobs, chunksize=256)
    for it, rr in zip(items, res):
        it.update(rr)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(items, open(CACHE, 'wb'))
    return items


def seq_batch(pairs, dev):
    """pairs: list of (state ids, target ids, mask positions) -> idx (B,T), list of (row, p0, ntgt, maskset)"""
    L = [len(s) + len(t) for s, t, _ in pairs]
    T = max(L)
    idx = torch.full((len(pairs), T), TOK.pad, dtype=torch.long)
    for r, (s, t, _) in enumerate(pairs):
        idx[r, :len(s) + len(t)] = torch.tensor(list(s) + list(t))
    return idx.to(dev)


def token_logps(model, pairs, dev, want_kl_ref=None):
    """-> list of per-pair tensors of target log p (and logp tensors for KL)."""
    idx = seq_batch(pairs, dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(idx)
    logp = F.log_softmax(logits.float(), -1)
    return idx, logp


def attempt_scores(logp, idx, pairs, owner, n_att, keep_mask=True):
    """masked mean log p per attempt (differentiable). pairs[j] = (state, tgt, mask) ; owner[j] = attempt index"""
    sums = [0.0] * n_att; cnts = [0] * n_att
    tl = logp[:, :-1].gather(-1, idx[:, 1:, None])[..., 0]
    for j, (s, t, m) in enumerate(pairs):
        p0 = len(s) - 1
        v = tl[j, p0:p0 + len(t)]
        if keep_mask and m:
            keep = torch.ones(len(t), dtype=torch.bool, device=v.device)
            keep[list(m)] = False
            v = v[keep]
        sums[owner[j]] = sums[owner[j]] + v.sum(); cnts[owner[j]] += v.numel()
    return torch.stack([s / max(c, 1) if torch.is_tensor(s) else torch.tensor(-1e9, device=logp.device) for s, c in zip(sums, cnts)])


def scored_pairs(it):
    """B's scorer view: (state, raw action ids + eos if ended, masked positions)"""
    return [(st, list(ids) + ([TOK.eos] if ended else []), m) for st, ids, m, ended in it['steps']]


@torch.no_grad()
def dev_auroc(model, dev_items, devc, max_pairs=3000):
    model.eval()
    allp, owner = [], []
    for a, it in enumerate(dev_items):
        for p in scored_pairs(it):
            allp.append(p); owner.append(a)
    order = sorted(range(len(allp)), key=lambda j: len(allp[j][0]) + len(allp[j][1]))
    sums = np.zeros(len(dev_items)); cnts = np.zeros(len(dev_items))
    i = 0
    while i < len(order):
        T = len(allp[order[i]][0]) + len(allp[order[i]][1]); j = i
        while j < len(order) and (j - i) < max_pairs and (len(allp[order[j]][0]) + len(allp[order[j]][1])) ** 2 * (j - i + 1) < 2 ** 27:
            j += 1
        j = max(j, i + 1)
        b = order[i:j]
        pr = [allp[t] for t in b]
        idx = seq_batch(pr, devc)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logp = F.log_softmax(model(idx).float(), -1)
        tl = logp[:, :-1].gather(-1, idx[:, 1:, None])[..., 0].cpu().numpy()
        for r, t in enumerate(b):
            s, tg, m = allp[t]
            p0 = len(s) - 1
            v = tl[r, p0:p0 + len(tg)]
            if m:
                v = np.delete(v, m)
            sums[owner[t]] += v.sum(); cnts[owner[t]] += len(v)
        i = j
    model.train()
    sc = sums / np.maximum(cnts, 1)
    by = {}
    for it, v in zip(dev_items, sc):
        g = by.setdefault((it['set'], it['name']), ([], [])); g[0].append(v); g[1].append(it['valid'])
    a = [auroc(x, y) for x, y in by.values()]
    a = [x for x in a if x is not None]
    vs = np.array([it['valid'] for it in dev_items])
    return float(np.mean(a)), len(a), float(sc[vs].mean()), float(sc[~vs].mean())


def save(model, path, extra):
    ck = torch.load(S1, map_location='cpu')
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, tok_mode=ck['tok_mode'],
                    cfg=ck['cfg'], extra=extra), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sft', 'pair'], required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--beta', type=float, default=10.0)
    ap.add_argument('--kl', type=float, default=1.0)
    ap.add_argument('--max_steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--recs', type=int, default=128)
    ap.add_argument('--theorems', type=int, default=16)
    ap.add_argument('--per_class', type=int, default=4)
    ap.add_argument('--build_cache', action='store_true')
    a = ap.parse_args()
    if a.build_cache:
        items = build_cache(); print('cached', len(items)); return
    lr = a.lr or (1e-4 if a.mode == 'sft' else 5e-5)
    out = os.path.join(PL.RUNS, 'judge', a.name); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a')
    def log(m):
        m = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {m}'; print(m, flush=True); logf.write(m + '\n'); logf.flush()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = torch.device('cuda')
    items = pickle.load(open(CACHE, 'rb'))
    train = [it for it in items if it['split'] == 'train']; devi = [it for it in items if it['split'] == 'dev']
    model, tok, _ = load_ckpt(S1, dev); model.train()
    ref = None
    if a.mode == 'pair':
        ref, _, _ = load_ckpt(S1, dev); ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = lambda s: lr * s / 100 if s < 100 else 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * (s - 100) / max(1, a.max_steps - 100)))
    if a.mode == 'sft':
        pool = [it for it in train if it['valid']]
        log(f'sft: {len(pool)} valid finished train attempts; lr {lr}')
    else:
        groups = {}
        for it in train:
            groups.setdefault((it['set'], it['name']), ([], []))[0 if it['valid'] else 1].append(it)
        groups = [g for g in groups.values() if g[0] and g[1]]
        log(f'pair: {len(groups)} theorems with both classes; {sum(len(g[1]) for g in groups)} invalid finished; beta {a.beta} kl {a.kl} lr {lr}')
    au0 = dev_auroc(model, devi, dev)
    log(f'step 0 (s1 start) dev within-fin AUROC {au0[0]:.4f} (n{au0[1]}) mean masked logp valid {au0[2]:.4f} invalid {au0[3]:.4f}')
    best, best_step, bad, hist, t0 = au0[0], 0, 0, [dict(step=0, dev=au0[0], lp_valid=au0[2], lp_invalid=au0[3])], time.time()
    save(model, os.path.join(out, 'best.pt'), dict(step=0))
    order = []
    for step in range(1, a.max_steps + 1):
        for g in opt.param_groups:
            g['lr'] = sched(step)
        if a.mode == 'sft':
            if len(order) < a.recs:
                order = list(range(len(pool))); rng.shuffle(order)
            bt = [pool[order.pop()] for _ in range(a.recs)]
            pairs = [(st, act, []) for it in bt for st, act in it['eff']]
            idx = seq_batch(pairs, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(idx)
            logp = F.log_softmax(logits.float(), -1)
            tl = logp[:, :-1].gather(-1, idx[:, 1:, None])[..., 0]
            msk = torch.zeros_like(tl)
            for j, (s, t, _) in enumerate(pairs):
                msk[j, len(s) - 1:len(s) - 1 + len(t)] = 1
            loss = -(tl * msk).sum() / msk.sum(); extra = ''
        else:
            if len(order) < a.theorems:
                order = list(range(len(groups))); rng.shuffle(order)
            gb = [groups[order.pop()] for _ in range(a.theorems)]
            atts, lab, thm = [], [], []
            for gi, (pos, neg) in enumerate(gb):
                for it in rng.sample(pos, min(a.per_class, len(pos))):
                    atts.append(it); lab.append(1); thm.append(gi)
                for it in rng.sample(neg, min(a.per_class, len(neg))):
                    atts.append(it); lab.append(0); thm.append(gi)
            pairs, owner = [], []
            for ai, it in enumerate(atts):
                for p in scored_pairs(it):
                    pairs.append(p); owner.append(ai)
            idx = seq_batch(pairs, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(idx)
            logp = F.log_softmax(logits.float(), -1)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                rlogp = F.log_softmax(ref(idx).float(), -1)
            s_pol = attempt_scores(logp, idx, pairs, owner, len(atts))
            with torch.no_grad():
                s_ref = attempt_scores(rlogp, idx, pairs, owner, len(atts))
            z = s_pol - s_ref
            terms = []
            for gi in range(len(gb)):
                pv = [k for k in range(len(atts)) if thm[k] == gi and lab[k] == 1]
                nv = [k for k in range(len(atts)) if thm[k] == gi and lab[k] == 0]
                d = z[pv][:, None] - z[nv][None, :]
                terms.append(-F.logsigmoid(a.beta * d).mean())
            pl = torch.stack(terms).mean()
            # token KL(ref || policy) at every scored action position (incl. masked name positions)
            pos = torch.zeros(idx.shape[0], idx.shape[1] - 1, dtype=torch.bool, device=dev)
            for j, (s, t, _) in enumerate(pairs):
                pos[j, len(s) - 1:len(s) - 1 + len(t)] = True
            rp, pp = rlogp[:, :-1][pos], logp[:, :-1][pos]
            kl = (rp.exp() * (rp - pp)).sum(-1).mean()
            loss = pl + a.kl * kl; extra = f' pair {pl.item():.4f} kl {kl.item():.4f}'
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            log(f'step {step} loss {loss.item():.4f}{extra} {time.time()-t0:.0f}s')
        if step % a.eval_every == 0 or step == a.max_steps:
            au = dev_auroc(model, devi, dev)
            hist.append(dict(step=step, dev=au[0], lp_valid=au[2], lp_invalid=au[3], loss=float(loss.detach())))
            log(f'EVAL step {step} dev within-fin AUROC {au[0]:.4f} mean masked logp valid {au[2]:.4f} invalid {au[3]:.4f}')
            if au[0] > best:
                best, best_step, bad = au[0], step, 0
                save(model, os.path.join(out, 'best.pt'), dict(step=step, mode=a.mode, dev_auroc=au[0]))
            else:
                bad += 1
            if bad >= a.patience:
                log('early stop'); break
    log(f'best dev within-fin AUROC {best:.4f} at step {best_step} (s1 start {au0[0]:.4f}); frozen')
    json.dump(dict(vars(a), lr=lr, best_dev=best, best_step=best_step, start_dev=au0[0], hist=hist,
                   train_secs=time.time() - t0), open(os.path.join(out, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
