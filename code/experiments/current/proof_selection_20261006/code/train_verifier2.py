#!/usr/bin/env python3
"""Phase 2: within-theorem ranker on FINISHED attempts (pre-registered 03:38 UTC in logs/worker_A.md).

r16 init + scalar head (optionally + Worker B's loglik_mean_masked as an extra input to the head), hist input format.
Loss: pairwise logistic over (finished valid, finished invalid) pairs of the same theorem; each theorem weighted
equally; a theorem visit samples up to --per_class valid and --per_class invalid finished attempts.  Only theorems with
both classes among finished attempts contribute.  Early stop on dev mean within-theorem finished AUROC.

Data: --pools lists `set:k` pool stems (e.g. rl_targets:16 dev1108:16 rr600:32 rr600:96b ...); attempts of the same
theorem from several pools are merged (same frozen generator).  Train/dev membership from results/splits.json.
--inv_frac f keeps a seeded nested subsample (fixed permutation, seed 12345) of the ORIGINAL pools' train
finished-invalid attempts (ladder rungs); all finished-valid attempts are kept.

  python train_verifier2.py --name r25 --inv_frac 0.25 --pools rl_targets:16 dev1108:16 rr600:32
"""
import argparse, json, math, os, pickle, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pool_lib as PL
from model import load_ckpt
from evaluate_selection import auroc

SEQS = os.path.join(PL.RUNS, 'verifier', 'seqs_hist')
ORIG = {'rl_targets': '16', 'dev1108': '16', 'rr600': '32', 'holdout250': '64', 'textbook72': '64'}
SCORE_POOLS = ['rl_targets:16', 'dev1108:16', 'rr600:32', 'holdout250:64', 'textbook72:64',
               'rr600:96b', 'rl_targets:48b', 'dev1108:48b']


class Verifier2(nn.Module):
    def __init__(self, body, use_ll):
        super().__init__()
        self.body, self.use_ll = body, use_ll
        self.vhead = nn.Linear(body.cfg['d'] + (1 if use_ll else 0), 1)
        nn.init.zeros_(self.vhead.weight); nn.init.zeros_(self.vhead.bias)

    def forward(self, x, lens, ll=None):
        h, _ = self.body.features(x)
        last = self.body.ln_f(h)[torch.arange(x.size(0), device=x.device), lens - 1].float()
        if self.use_ll:
            last = torch.cat([last, ll[:, None]], 1)
        return self.vhead(last).squeeze(-1)


def parse(spec):
    s, k = spec.split(':')
    return s, k


def load(specs, with_labels=True, ll_stats=None, finished_only=True):
    """-> list of dict(set, name, stem, i, seq, done, valid, ll)"""
    out = []
    for spec in specs:
        s, k = parse(spec)
        stem = PL.pool_stem(s, k)
        d = pickle.load(open(os.path.join(SEQS, stem + '.pkl'), 'rb'))
        pool = PL.load_pool(s, k)
        lab = PL.load_labels(s, k) if with_labels else None
        ll = PL.load_scores(s, 'loglik_mean_masked', k) if ll_stats is not None else None
        for r in pool:
            for a in r['attempts']:
                if finished_only and not a['done']:
                    continue
                v = None
                if ll is not None:
                    v = ll[r['name']][a['i']]
                    v = ll_stats[2] if not math.isfinite(v) else v
                    v = (v - ll_stats[0]) / ll_stats[1]
                out.append(dict(set=s, name=r['name'], stem=stem, k=k, i=a['i'], seq=d[r['name']][0][a['i']], done=a['done'],
                                valid=None if lab is None else lab[r['name']][a['i']], ll=v))
    return out


def ll_standardisation(specs, train_names):
    xs = []
    for spec in specs:
        s, k = parse(spec)
        ll = PL.load_scores(s, 'loglik_mean_masked', k)
        for n, v in ll.items():
            if n in train_names.get(s, ()):
                xs += [x for x in v if math.isfinite(x)]
    xs = np.array(xs)
    return float(xs.mean()), float(xs.std() + 1e-9), float(xs.min())


def collate(items, dev):
    L = max(len(it['seq']) for it in items)
    x = torch.zeros((len(items), L), dtype=torch.long)
    for r, it in enumerate(items):
        x[r, :len(it['seq'])] = torch.from_numpy(it['seq'].astype(np.int64))
    lens = torch.tensor([len(it['seq']) for it in items])
    ll = torch.tensor([it['ll'] if it['ll'] is not None else 0.0 for it in items], dtype=torch.float32)
    return x.to(dev), lens.to(dev), ll.to(dev)


@torch.no_grad()
def predict(model, items, dev, tok_budget=131072):
    model.eval()
    order = sorted(range(len(items)), key=lambda j: len(items[j]['seq']))
    out = np.zeros(len(items), np.float32)
    i = 0
    while i < len(order):
        L = len(items[order[i]]['seq']); j = i
        while j < len(order) and len(items[order[j]]['seq']) * (j - i + 1) <= tok_budget:
            j += 1
        j = max(j, i + 1)
        b = order[i:j]
        x, lens, ll = collate([items[t] for t in b], dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out[b] = model(x, lens, ll).float().cpu().numpy()
        i = j
    model.train()
    return out


def within_auroc(items, p, keyf=lambda it: (it['set'], it['name'])):
    by = {}
    for it, v in zip(items, p):
        if it['done']:
            g = by.setdefault(keyf(it), ([], []))
            g[0].append(v); g[1].append(it['valid'])
    a = [auroc(s, y) for s, y in by.values()]
    a = [x for x in a if x is not None]
    return (float(np.mean(a)) if a else float('nan')), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pools', nargs='+', required=True)
    ap.add_argument('--dev_pools', nargs='*', default=None, help='pools for the dev split (default: --pools)')
    ap.add_argument('--inv_frac', type=float, default=1.0)
    ap.add_argument('--use_ll', action='store_true')
    ap.add_argument('--theorems_per_batch', type=int, default=16)
    ap.add_argument('--per_class', type=int, default=8)
    ap.add_argument('--max_steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--score_pools', nargs='*', default=SCORE_POOLS)
    ap.add_argument('--score_only', action='store_true')
    a = ap.parse_args()
    out = os.path.join(PL.RUNS, 'verifier2', a.name)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a')
    def log(m):
        m = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {m}'; print(m, flush=True); logf.write(m + '\n'); logf.flush()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = 'cuda'
    sp = json.load(open(os.path.join(PL.SPRINT, 'results', 'splits.json')))
    trn = {s: set(v) for s, v in sp['train'].items()}; dvn = {s: set(v) for s, v in sp['dev'].items()}
    ll_stats = None
    if a.use_ll and a.score_only:
        ll_stats = tuple(json.load(open(os.path.join(out, 'll_stats.json'))))
    elif a.use_ll:
        ll_stats = ll_standardisation(a.pools, trn)
        json.dump(ll_stats, open(os.path.join(out, 'll_stats.json'), 'w'))
    body, tok, _ = load_ckpt(PL.CKPT, dev)
    model = Verifier2(body, a.use_ll).to(dev)
    scorer = f'verifier2_{a.name}'
    if not a.score_only:
        items = load(a.pools, ll_stats=ll_stats)
        train = [it for it in items if it['name'] in trn.get(it['set'], ())]
        ditems = items if a.dev_pools is None else load(a.dev_pools, ll_stats=ll_stats)
        devi = [it for it in ditems if it['name'] in dvn.get(it['set'], ())]
        dev_orig = [it for it in devi if it['k'] == ORIG[it['set']]]
        # ladder: nested subsample of ORIGINAL-pool train finished-invalid attempts
        inv_orig = [j for j, it in enumerate(train) if not it['valid'] and it['k'] == ORIG[it['set']]]
        perm = sorted(inv_orig, key=lambda j: (train[j]['set'], train[j]['name'], train[j]['k'], train[j]['i']))
        random.Random(12345).shuffle(perm)
        drop = set(perm[int(round(a.inv_frac * len(perm))):])
        train = [it for j, it in enumerate(train) if j not in drop]
        groups = {}
        for it in train:
            groups.setdefault((it['set'], it['name']), ([], []))[0 if it['valid'] else 1].append(it)
        groups = [g for g in groups.values() if g[0] and g[1]]
        n_inv = sum(1 for it in train if not it['valid']); n_inv_used = sum(len(g[1]) for g in groups)
        info = dict(dev_pools=a.dev_pools, n_train_finished=len(train), n_finished_invalid=n_inv, n_finished_invalid_in_pair_theorems=n_inv_used,
                    n_pair_theorems=len(groups), n_dev_finished=len(devi), inv_frac=a.inv_frac, pools=a.pools)
        log(json.dumps(info))
        body_p = [p for n, p in model.named_parameters() if not n.startswith('vhead')]
        opt = torch.optim.AdamW([dict(params=body_p, lr=a.lr), dict(params=model.vhead.parameters(), lr=a.head_lr)],
                                weight_decay=a.wd, betas=(0.9, 0.95))
        warm = 100; base = [a.lr, a.head_lr]
        best, best_step, bad, hist, t0 = -1.0, 0, 0, [], time.time()
        order = []
        model.train()
        for step in range(1, a.max_steps + 1):
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.max_steps - warm)))
            for g, lr in zip(opt.param_groups, base):
                g['lr'] = lr * max(f, 0.05)
            if len(order) < a.theorems_per_batch:
                order = list(range(len(groups))); rng.shuffle(order)
            gb = [groups[order.pop()] for _ in range(a.theorems_per_batch)]
            batch, spans = [], []
            for pos, neg in gb:
                p = rng.sample(pos, min(a.per_class, len(pos))); n = rng.sample(neg, min(a.per_class, len(neg)))
                spans.append((len(batch), len(p), len(n))); batch += p + n
            x, lens, ll = collate(batch, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(x, lens, ll).float()
            terms = []
            for s0, npos, nneg in spans:
                zp, zn = z[s0:s0 + npos], z[s0 + npos:s0 + npos + nneg]
                terms.append(F.softplus(-(zp[:, None] - zn[None, :])).mean())
            loss = torch.stack(terms).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % a.eval_every == 0 or step == a.max_steps:
                p = predict(model, devi, dev)
                au, nth = within_auroc(devi, p)
                po = predict(model, dev_orig, dev) if len(dev_orig) != len(devi) else p
                au_o, nth_o = within_auroc(dev_orig, po)
                hist.append(dict(step=step, loss=float(loss.detach()), dev_within=au, dev_within_orig=au_o, secs=time.time() - t0))
                log(f'step {step} loss {loss.item():.4f} dev within-fin AUROC {au:.4f} (n{nth}) orig-dev {au_o:.4f} (n{nth_o}) {time.time()-t0:.0f}s')
                if au > best:
                    best, best_step, bad = au, step, 0
                    torch.save(model.state_dict(), os.path.join(out, 'best.pt'))
                    best_orig = au_o
                else:
                    bad += 1
                if bad >= a.patience:
                    log('early stop'); break
        log(f'best dev within-fin AUROC {best:.4f} (orig-dev {best_orig:.4f}) at step {best_step}; frozen')
        json.dump(dict(vars(a), **info, best_dev_within=best, best_dev_within_orig=best_orig, best_step=best_step,
                       hist=hist, train_secs=time.time() - t0), open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    model.load_state_dict(torch.load(os.path.join(out, 'best.pt'), map_location=dev))
    for spec in a.score_pools:
        s, k = parse(spec)
        if not os.path.exists(os.path.join(SEQS, PL.pool_stem(s, k) + '.pkl')):
            log(f'skip scoring {spec} (no seqs)'); continue
        if a.use_ll and not os.path.exists(os.path.join(PL.SCORES, f'{PL.pool_stem(s, k)}__loglik_mean_masked.jsonl')):
            log(f'skip scoring {spec} (no loglik)'); continue
        its = load([spec], with_labels=False, ll_stats=ll_stats, finished_only=False)
        p = predict(model, its, dev)
        rows = {}
        for it, v in zip(its, p):
            rows.setdefault(it['name'], {})[it['i']] = float(v)
        PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s, k)}__{scorer}.jsonl'),
                       [dict(name=r['name'], scores=[rows[r['name']][i] for i in range(len(rows[r['name']]))])
                        for r in PL.load_pool(s, k)])
        log(f'scored {spec} -> {scorer}')


if __name__ == '__main__':
    main()
