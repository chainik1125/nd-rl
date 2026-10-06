#!/usr/bin/env python3
"""Print greedy rollouts in the environment, step by step (debugging aid for run state-env).
  python3 se_trace.py --ckpt ckpts/se/stage1_SN_s1.pt --in data/p2/heldout.jsonl --n 8 --len 2"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import load_ckpt
from sample import generate_ids_fast
from state_env import Env
from state_sample import prompt_ids

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt'); ap.add_argument('--in', dest='inp'); ap.add_argument('--n', type=int, default=8)
ap.add_argument('--len', type=int, default=2); ap.add_argument('--only_fail', action='store_true')
a = ap.parse_args()
model, tok, _ = load_ckpt(a.ckpt, 'cuda')
recs = [json.loads(l) for l in open(a.inp) if json.loads(l)['n_lines'] == a.len]
shown = 0
for r in recs:
    e = Env(r['prompt'], canon=getattr(tok, 'canon', False)); lines = []
    for step in range(12):
        ids = prompt_ids(tok, e)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = generate_ids_fast(model, tok, [ids], greedy=True, max_new=256, early='eos')[0]
        act, ended = tok.decode_action(o)
        lines.append(f'   S: {tok.text(e.state_tokens())}\n   A: {tok.text(act)}')
        ok, why = e.apply(act)
        if not ok:
            lines.append(f'   -> FAIL {why}'); break
        if e.done:
            lines.append('   -> done'); break
    if a.only_fail and e.done:
        continue
    print(r['thm']); print('\n'.join(lines)); print()
    shown += 1
    if shown >= a.n:
        break
