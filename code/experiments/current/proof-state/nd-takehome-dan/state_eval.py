#!/usr/bin/env python3
"""Evaluate a state-conditioned checkpoint on a jsonl of theorems, sampling **in the environment**.

  python3 state_eval.py --ckpt ckpts/se/stage1_S_s0.pt --in data/p2/heldout.jsonl \
      --out artifacts/se/heldout_S_s0.jsonl --summary artifacts/se/heldout_S_s0.json --k 1 --temperature 0

Same record format and same summary as `eval_set.py` (`judge` / `summarize` are imported from it, so the per-length
table, the Wilson intervals and the written / pruned length histograms are computed by the same code as every
whole-proof number in this project).  Lean alone decides.  The environment's per-step diagnostics go in the summary.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import load_ckpt
from eval_set import judge, summarize
from state_sample import env_generate, env_stats_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--summary', default=None)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--max_action', type=int, default=256)
    ap.add_argument('--max_steps', type=int, default=48)
    ap.add_argument('--lenfield', default='n_lines')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    model, tok, _ = load_ckpt(a.ckpt, 'cuda')
    stats = {}
    prompts = [r['prompt'] for r in recs for _ in range(a.k)]
    t0 = time.time()
    flat = env_generate(model, tok, prompts, greedy=(a.temperature == 0), temperature=max(a.temperature, 1e-6),
                        max_action=a.max_action, max_steps=a.max_steps, batch=a.batch, seed=a.seed, stats=stats)
    per = [flat[i * a.k:(i + 1) * a.k] for i in range(len(recs))]
    rows = judge(recs, per, a.lenfield)
    summ = summarize(rows, a.lenfield, f'[{os.path.basename(a.ckpt)} on {os.path.basename(a.inp)} k={a.k} T={a.temperature}]')
    summ['ckpt'] = a.ckpt; summ['in'] = a.inp; summ['k'] = a.k; summ['temperature'] = a.temperature
    summ['seed'] = a.seed; summ['batch'] = a.batch; summ['max_action'] = a.max_action; summ['max_steps'] = a.max_steps
    summ['wall_s'] = time.time() - t0
    summ['env'] = env_stats_json(stats)
    print('env', json.dumps({k: v for k, v in summ['env'].items() if k not in ('action_declen_hist',)}))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
    if a.summary:
        os.makedirs(os.path.dirname(a.summary) or '.', exist_ok=True)
        json.dump(summ, open(a.summary, 'w'), indent=1)


if __name__ == '__main__':
    main()
