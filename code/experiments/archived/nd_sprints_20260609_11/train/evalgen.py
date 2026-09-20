"""Batched generation + verification: solve rates, pass@k, and rollout
collection for expert iteration.

For each target theorem we sample k proofs and verify each with the
independent verifier. A target is 'solved' if any sample is a valid proof of
exactly that sequent.

Usage:
  python -m train.evalgen --ckpt runs/L8/ckpt.pt --targets data/eval_ood.jsonl \
      --k 8 --temp 0.7 --out results/L8_ood_k8.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nd.tokenizer import STOI, ITOS, PAD
from nd.verify import verify_text
from train.model import GPT

QED_ID = STOI['QED']


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck['config']
    model = GPT(cfg['vocab'], block_size=cfg['block'], n_layer=cfg['layers'],
                n_head=cfg['heads'], d_model=cfg['dim']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, cfg


def get_prompt(rec):
    if 'prompt' in rec:
        return rec['prompt']
    toks = rec['text'].split()
    return ' '.join(toks[:toks.index('PRF') + 1])


def run_eval(model, records, k, temp, max_new, bs, device, top_k=None,
             progress=True, keep_fails=False):
    """Returns list of result dicts aligned with records. With keep_fails,
    each result also carries 'fail_texts' (full text of every failed rollout)
    for hindsight harvesting."""
    jobs = []  # (rec_idx, prompt_ids)
    for i, r in enumerate(records):
        ptoks = get_prompt(r).split()
        pids = [STOI[t] for t in ptoks]
        for _ in range(k):
            jobs.append((i, pids))
    # sort by prompt length for batching efficiency, remember order
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]))
    results = [{'n_ok': 0, 'best_len': None, 'best_text': None,
                'fail_reasons': {}, 'fail_texts': []} for _ in records]
    t0 = time.time()
    nb = 0
    for s in range(0, len(order), bs):
        chunk = [jobs[j] for j in order[s:s + bs]]
        prompts = [c[1] for c in chunk]
        outs = model.sample(prompts, max_new=max_new,
                            temperature=temp, top_k=top_k,
                            eos_id=QED_ID, pad_id=PAD, device=device)
        for (ri, pids), out in zip(chunk, outs):
            full = ' '.join(ITOS[t] for t in pids + out)
            ok, reason, nl = verify_text(full)
            res = results[ri]
            if ok:
                res['n_ok'] += 1
                if res['best_len'] is None or nl < res['best_len']:
                    res['best_len'] = nl
                    res['best_text'] = full
            else:
                key = reason.split('(')[0].strip()[:40]
                res['fail_reasons'][key] = res['fail_reasons'].get(key, 0) + 1
                if keep_fails:
                    res['fail_texts'].append(full)
        nb += 1
        if progress and nb % 10 == 0:
            print(f'  {s+len(chunk)}/{len(jobs)} rollouts, {time.time()-t0:.0f}s',
                  file=sys.stderr, flush=True)
    out = []
    for r, res in zip(records, results):
        rec = {'thm': r['thm'], 'n_lines': r.get('n_lines'),
               'discharge': r.get('discharge'),
               'rules': r.get('rules'), 'name': r.get('name'),
               'k': k, 'n_ok': res['n_ok'],
               'solved': res['n_ok'] > 0,
               'best_len': res['best_len'],
               'best_text': res['best_text'],
               'fail_reasons': res['fail_reasons']}
        if keep_fails:
            rec['fail_texts'] = res['fail_texts']
        out.append(rec)
    return out


def summarize(results, by='n_lines'):
    from collections import defaultdict
    groups = defaultdict(lambda: [0, 0])
    for r in results:
        key = r.get(by)
        groups[key][0] += r['solved']
        groups[key][1] += 1
    lines = []
    for key in sorted(groups, key=lambda x: (x is None, x)):
        s, n = groups[key]
        lines.append(f'{by}={key}: {s}/{n} = {s/n:.3f}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--top-k', type=int, default=None)
    ap.add_argument('--max-new', type=int, default=420)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model, cfg = load_model(args.ckpt, device)
    records = []
    with open(args.targets) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f'{len(records)} targets, k={args.k}, temp={args.temp}')
    results = run_eval(model, records, args.k, args.temp, args.max_new,
                       args.bs, device, top_k=args.top_k)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(summarize(results))
    solved = sum(r['solved'] for r in results)
    print(f'TOTAL: {solved}/{len(results)} = {solved/len(results):.3f}')


if __name__ == '__main__':
    main()
