#!/usr/bin/env python3
"""Evaluate a checkpoint on a jsonl of theorems ({prompt, n_lines|gen_lines, key|thm}).

  python eval_set.py --ckpt ckpts/stage1.pt --in data/heldout.jsonl --out artifacts/h.jsonl [--k 16 --temperature 0.8 --seed 0]
Greedy by default (k=1). With --k>1, samples k proofs per theorem and reports pass@k plus
per-theorem distinct verified proofs. Prints per-length table with Wilson CIs and failure reasons.
Writes one record per theorem: {name/thm, prompt, n_lines, solved, proofs:[verified distinct], written_lens, pruned_lens, reasons}.
"""
import argparse, json, os, sys, collections, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import load_ckpt
from sample import generate
from lean_judge import verify_text, judge_many    # Lean alone decides (Dan, 2026-09-27)
from prune import pruned_length


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def judge(recs, proofs_per, lenfield):
    """recs: list of theorem records; proofs_per: list of list of proof strings. Returns rows + summary."""
    rows = []
    # one batched judging call for every sample of every theorem: after sample.generate() these are all registry hits,
    # and where they are not (a caller that never saw the gate) the Lean runs are batched (pitfall 4).
    flat = judge_many([(r['prompt'], p) for r, ps in zip(recs, proofs_per) for p in ps])
    at = 0
    for r, ps in zip(recs, proofs_per):
        good, wl, pl, reasons = [], [], [], []
        fail_example = None
        for p in ps:
            ok, reason, nl = flat[at]; at += 1
            if not ok and fail_example is None:
                fail_example = p
            if ok:
                if p not in good:
                    good.append(p)
                    wl.append(nl)
                    pl.append(pruned_length(r['prompt'], p))
            else:
                reasons.append(reason.split(' (line')[0])
        rows.append({'name': r.get('name', r.get('thm')), 'thm': r.get('thm'), 'prompt': r['prompt'],
                     lenfield: r.get(lenfield), 'solved': bool(good), 'n_ok': sum(1 for p in ps if p in good),
                     'n_tried': len(ps), 'proofs': good, 'written_lens': wl, 'pruned_lens': pl, 'reasons': reasons, 'fail_example': fail_example})
    return rows


def summarize(rows, lenfield, title=''):
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r.get(lenfield)][0] += r['solved']; by[r.get(lenfield)][1] += 1
    n = len(rows); k = sum(r['solved'] for r in rows)
    lo, hi = wilson(k, n)
    out = {'n': n, 'solved': k, 'rate': k / max(n, 1), 'ci': [lo, hi], 'by_len': {}}
    print(f'{title} SOLVED {k}/{n} = {k/max(n,1):.3f} [{lo:.3f},{hi:.3f}]')
    for L, (kk, nn) in sorted(by.items(), key=lambda x: (x[0] is None, x[0])):
        lo, hi = wilson(kk, nn)
        out['by_len'][str(L)] = {'solved': kk, 'n': nn, 'rate': kk / nn, 'ci': [lo, hi]}
        print(f'  len {L}: {kk}/{nn} = {kk/nn:.3f} [{lo:.3f},{hi:.3f}]')
    wh = collections.Counter(); ph = collections.Counter()
    for r in rows:
        for x in r['written_lens']: wh[x] += 1
        for x in r['pruned_lens']: ph[x] += 1
    out['written_hist'] = dict(sorted(wh.items())); out['pruned_hist'] = dict(sorted(ph.items()))
    # robust frontier: longest written length with >=5 distinct verified proofs (also pruned version)
    out['frontier_written'] = max([L for L, c in wh.items() if c >= 5], default=0)
    out['frontier_pruned'] = max([L for L, c in ph.items() if c >= 5], default=0)
    reasons = collections.Counter(x for r in rows for x in r['reasons'])
    out['reasons'] = dict(reasons.most_common(10))
    print('  written hist', out['written_hist']); print('  pruned hist', out['pruned_hist'])
    print('  frontier written', out['frontier_written'], 'pruned', out['frontier_pruned'])
    print('  reasons', reasons.most_common(6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--lenfield', default='n_lines')
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--summary', default=None)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok, _ = load_ckpt(a.ckpt, dev)
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    t0 = time.time()
    if a.k == 1 and a.temperature == 0:
        outs = [[p] for p in generate(model, tok, [r['prompt'] for r in recs], greedy=True, batch=a.batch)]
    else:
        prompts = [r['prompt'] for r in recs for _ in range(a.k)]
        flat = generate(model, tok, prompts, greedy=False, temperature=a.temperature, batch=a.batch, seed=a.seed)
        outs = [flat[i * a.k:(i + 1) * a.k] for i in range(len(recs))]
    print(f'generated {sum(len(o) for o in outs)} proofs in {time.time()-t0:.0f}s', flush=True)
    rows = judge(recs, outs, a.lenfield)
    summ = summarize(rows, a.lenfield, title=f'{os.path.basename(a.ckpt)} on {os.path.basename(a.inp)} k={a.k}')
    summ.update({'ckpt': a.ckpt, 'in': a.inp, 'k': a.k, 'temperature': a.temperature if a.k > 1 or a.temperature else 0, 'seed': a.seed})
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    if a.summary:
        json.dump(summ, open(a.summary, 'w'), indent=1)


if __name__ == '__main__':
    main()
