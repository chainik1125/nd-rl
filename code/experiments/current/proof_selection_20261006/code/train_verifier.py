#!/usr/bin/env python3
"""Learned verifier head: a copy of the frozen-generator checkpoint (r16, `load_ckpt`) with a scalar head on the final
token's hidden state (after ln_f).  Input = verifier_data sequences (replayed (state, action) pairs, native format).

Training: BCE with positive-class weight n_neg/n_pos (balance by weighting, nothing dropped) on the train split
(results/splits.json), AdamW (body lr 1e-4, head lr 1e-3, wd 0.01), token-budget batches, cosine schedule, eval
on dev every --eval_every steps; the checkpoint with the best pooled dev AUROC is kept (early stopping).  Test pools
are scored exactly once, by the kept checkpoint, after training ends.

  python train_verifier.py --seed 0           -> ~/runs/proof-selection/verifier/s0/{best.pt,log.jsonl,summary.json}
  python train_verifier.py --seed 0 --score_only   (re-score all pools with the kept checkpoint)
"""
import argparse, json, math, os, pickle, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pool_lib as PL
from model import load_ckpt
from evaluate_selection import auroc

SEQS = os.path.join(PL.RUNS, 'verifier', 'seqs')
SEQS_DIR = [SEQS]
ALL_SETS = ['rl_targets', 'dev1108', 'rr600', 'holdout250', 'textbook72']


class Verifier(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.vhead = nn.Linear(body.cfg['d'], 1)
        nn.init.zeros_(self.vhead.weight); nn.init.zeros_(self.vhead.bias)

    def forward(self, x, lens):
        h, _ = self.body.features(x)
        h = self.body.ln_f(h)
        last = h[torch.arange(x.size(0), device=x.device), lens - 1]
        return self.vhead(last.float()).squeeze(-1)


def load_items(sets_names, with_labels=True, seqs=None):
    """sets_names: {set: list of theorem names or None (=all)} -> list of (set, name, i, seq ids, label or None)."""
    items = []
    for s, names in sets_names.items():
        d = pickle.load(open(os.path.join(seqs or SEQS_DIR[0], PL.pool_stem(s) + '.pkl'), 'rb'))
        lab = PL.load_labels(s) if with_labels else None
        done = {r['name']: [a['done'] for a in r['attempts']] for r in PL.load_pool(s)}
        for n in (names if names is not None else list(d)):
            sq, _ = d[n]
            for i, x in enumerate(sq):
                items.append((s, n, i, x, None if lab is None else lab[n][i], done[n][i]))
    return items


def batches(items, tok_budget, shuffle, rng):
    order = sorted(range(len(items)), key=lambda j: len(items[j][3]))
    bs, cur, mx = [], [], 0
    for j in order:
        L = len(items[j][3])
        if cur and max(mx, L) * (len(cur) + 1) > tok_budget:
            bs.append(cur); cur, mx = [], 0
        cur.append(j); mx = max(mx, L)
    if cur:
        bs.append(cur)
    if shuffle:
        rng.shuffle(bs)
    return bs


def theorem_batches(items, tok_budget, rng):
    """whole theorems per batch (all k attempts together), packed by padded tokens."""
    groups = {}
    for j, it in enumerate(items):
        groups.setdefault((it[0], it[1]), []).append(j)
    gl = sorted(groups.values(), key=lambda g: max(len(items[j][3]) for j in g))
    bs, cur, mx = [], [], 0
    for g in gl:
        L = max(len(items[j][3]) for j in g)
        if cur and max(mx, L) * (len(cur) + len(g)) > tok_budget:
            bs.append(cur); cur, mx = [], 0
        cur = cur + g; mx = max(mx, L)
    if cur:
        bs.append(cur)
    rng.shuffle(bs)
    return bs


def rank_loss(z, items, b):
    """mean over theorems (equal weight) of mean softplus(-(s_valid - s_invalid)) over pairs of FINISHED attempts."""
    by = {}
    for r, j in enumerate(b):
        it = items[j]
        if it[5]:
            by.setdefault((it[0], it[1]), ([], []))[0 if it[4] else 1].append(r)
    terms = []
    for pos, neg in by.values():
        if pos and neg:
            d = z[pos][:, None] - z[neg][None, :]
            terms.append(F.softplus(-d).mean())
    return torch.stack(terms).mean() if terms else z.sum() * 0, len(terms)


def within_fin_auroc(items, p):
    by = {}
    for it, v in zip(items, p):
        if it[5]:
            by.setdefault((it[0], it[1]), ([], []))[0].append(v); by[(it[0], it[1])][1].append(it[4])
    a = [auroc(s, y) for s, y in by.values()]
    a = [x for x in a if x is not None]
    return float(np.mean(a)) if a else float('nan')


def collate(items, idx, dev, pad=0):
    L = max(len(items[j][3]) for j in idx)
    x = torch.full((len(idx), L), pad, dtype=torch.long)
    lens = torch.tensor([len(items[j][3]) for j in idx])
    for r, j in enumerate(idx):
        x[r, :len(items[j][3])] = torch.from_numpy(items[j][3].astype(np.int64))
    return x.to(dev), lens.to(dev)


@torch.no_grad()
def predict(model, items, dev, tok_budget=65536):
    model.eval()
    out = np.zeros(len(items), np.float32)
    for b in batches(items, tok_budget, False, None):
        x, lens = collate(items, b, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out[b] = model(x, lens).float().cpu().numpy()
    model.train()
    return out


def write_scores(model, dev, scorer, log, sets=ALL_SETS):
    os.makedirs(PL.SCORES, exist_ok=True)
    for s in sets:
        items = load_items({s: None}, with_labels=False)
        p = predict(model, items, dev)
        rows = {}
        for (ss, n, i, _, _, _), v in zip(items, p):
            rows.setdefault(n, {})[i] = float(v)
        pool_names = [r['name'] for r in PL.load_pool(s)]
        PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s)}__{scorer}.jsonl'),
                       [dict(name=n, scores=[rows[n][i] for i in range(len(rows[n]))]) for n in pool_names])
        log(f'scored {s} -> {scorer}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--tok_budget', type=int, default=32768)
    ap.add_argument('--eval_every', type=int, default=300)
    ap.add_argument('--patience', type=int, default=6, help='evals without dev-AUROC improvement before stopping')
    ap.add_argument('--score_only', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--only_sets', nargs='*', default=None, help='restrict train/dev to these sets (smoke tests)')
    ap.add_argument('--score_sets', nargs='*', default=ALL_SETS, help='pools to score at the end / with --score_only')
    ap.add_argument('--scorer', default=None, help='score name (default verifier_s<seed> / verifier_hist_s<seed>)')
    ap.add_argument('--fmt', default='pairs', choices=['pairs', 'hist'])
    ap.add_argument('--loss', default='bce', choices=['bce', 'rank'],
                    help='rank: whole-theorem batches, 0.5*BCE + within-theorem pairwise loss on finished valid vs '
                         'finished invalid; early stop on dev mean within-theorem AUROC among finished attempts')
    a = ap.parse_args()
    SEQS_DIR[0] = SEQS + ('_hist' if a.fmt == 'hist' else '')
    tag = ('' if a.fmt == 'pairs' else 'hist_') + ('rank_' if a.loss == 'rank' else '')
    scorer = a.scorer or f'verifier_{tag}s{a.seed}'
    out = a.out or os.path.join(PL.RUNS, 'verifier', ('' if a.fmt == 'pairs' else 'hist_') + ('rank_' if a.loss == 'rank' else '') + f's{a.seed}')
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a')
    def log(m):
        m = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {m}'; print(m, flush=True); logf.write(m + '\n'); logf.flush()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = 'cuda'
    body, tok, _ = load_ckpt(PL.CKPT, dev)
    model = Verifier(body).to(dev)
    if a.score_only:
        model.load_state_dict(torch.load(os.path.join(out, 'best.pt'), map_location=dev))
        write_scores(model, dev, scorer, log, a.score_sets); return
    sp = json.load(open(os.path.join(PL.SPRINT, 'results', 'splits.json')))
    keep = lambda d: {s: v for s, v in d.items() if a.only_sets is None or s in a.only_sets}
    train = load_items(keep(sp['train']))
    devi = load_items(keep(sp['dev']))
    ytr = np.array([it[4] for it in train], bool); ydv = np.array([it[4] for it in devi], bool)
    pw = (~ytr).sum() / max(1, ytr.sum())
    log(f'seed {a.seed}: train {len(train)} attempts (pos {ytr.mean():.3f}), dev {len(devi)} (pos {ydv.mean():.3f}); pos_weight {pw:.3f}')
    body_p = [p for n, p in model.named_parameters() if not n.startswith('vhead')]
    opt = torch.optim.AdamW([dict(params=body_p, lr=a.lr), dict(params=model.vhead.parameters(), lr=a.head_lr)],
                            weight_decay=a.wd, betas=(0.9, 0.95))
    mkb = (lambda: theorem_batches(train, a.tok_budget, rng)) if a.loss == 'rank' else (lambda: batches(train, a.tok_budget, True, rng))
    nb = len(mkb())
    total = int(nb * a.epochs); warm = min(200, total // 20)
    base_lrs = [a.lr, a.head_lr]
    log(f'{nb} batches/epoch, {total} steps planned')
    best, best_step, bad, step, t0 = -1.0, 0, 0, 0, time.time()
    hist = []
    model.train()
    done = False
    while not done:
        for b in mkb():
            step += 1
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g['lr'] = lr * max(f, 0.05)
            x, lens = collate(train, b, dev)
            y = torch.tensor([float(train[j][4]) for j in b], device=dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(x, lens)
            loss = F.binary_cross_entropy_with_logits(z.float(), y, pos_weight=torch.tensor(pw, device=dev))
            if a.loss == 'rank':
                rl, _ = rank_loss(z.float(), train, b)
                loss = 0.5 * loss + rl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % a.eval_every == 0 or step == total:
                p = predict(model, devi, dev)
                au_pooled, au_wf = auroc(p, ydv), within_fin_auroc(devi, p)
                au = au_wf if a.loss == 'rank' else au_pooled
                hist.append(dict(step=step, loss=float(loss.detach()), dev_auroc=au_pooled, dev_within_fin_auroc=au_wf,
                                 secs=time.time() - t0))
                log(f'step {step}/{total} loss {loss.item():.4f} dev AUROC {au_pooled:.4f} within-fin {au_wf:.4f} {time.time()-t0:.0f}s')
                if au > best:
                    best, best_step, bad = au, step, 0
                    torch.save(model.state_dict(), os.path.join(out, 'best.pt'))
                else:
                    bad += 1
                if bad >= a.patience:
                    log('early stop'); done = True; break
            if step >= total:
                done = True; break
    log(f'best dev AUROC {best:.4f} at step {best_step}; frozen; scoring all pools once')
    json.dump(dict(vars(a), best_dev_auroc=best, best_step=best_step, steps=step, hist=hist, pos_weight=float(pw),
                   n_train=len(train), n_dev=len(devi), train_secs=time.time() - t0),
              open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    model.load_state_dict(torch.load(os.path.join(out, 'best.pt'), map_location=dev))
    write_scores(model, dev, scorer, log, a.score_sets)


if __name__ == '__main__':
    main()
