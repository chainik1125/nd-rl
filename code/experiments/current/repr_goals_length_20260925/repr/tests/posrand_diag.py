#!/usr/bin/env python3
"""Diagnostic for the monitor's pos-rand decision (no training): held-out greedy (039 slice, 2,000 cap-6 records,
max_new 512) of a pos-rand checkpoint under three sampling-time position schemes:
  contiguous  0, 1, 2, ...                                  (the brief's "offset 0")
  random      sorted random subset of range(1024), size len(prompt) + 19 x n_lines (evaluate.py --pos-eval rand)
  grid        evenly spaced at the same density: round(j * 1024 / (len(prompt) + 19 x n_lines))
    $PY tests/with_gpu.py -- $PY tests/posrand_diag.py pretrain/posrand_s0/pretrain.pt
"""
import json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common_repr import HELDOUT_CAP6, load_model, lib_path, read_jsonl
lib_path()
from nd_verify import verify_text
from sampler import sample_ids
from evaluate import rand_tables, TOK_PER_LINE

model, tok, _ = load_model(sys.argv[1], 'cuda')
held = read_jsonl(HELDOUT_CAP6)
recs = held[::2][:2000]
prompts = [r['prompt'] for r in recs]
res = {}
for scheme in ('contiguous', 'random', 'grid'):
    tabs = None
    if scheme == 'random':
        tabs = rand_tables(model, tok, prompts, [r['n_lines'] for r in recs], 512, 0)
    elif scheme == 'grid':
        tabs = []
        for p, r in zip(prompts, recs):
            n = len(tok.encode_prompt(p)) + TOK_PER_LINE * r['n_lines']
            g = 1024 / n
            tabs.append([round(j * g) for j in range(len(tok.encode_prompt(p)) + 512)])
    ids = sample_ids(model, tok, prompts, max_new=512, batch=512, seed=0, greedy=True, pos_tables=tabs)
    ok = [verify_text(p + ' ' + tok.decode(o))[0] for p, o in zip(prompts, ids)]
    by = {}
    for r, x in zip(recs, ok):
        by.setdefault(r['n_lines'], []).append(x)
    res[scheme] = {'all': sum(ok) / len(ok), 'by_lines': {k: sum(v) / len(v) for k, v in sorted(by.items())}}
    print(scheme, json.dumps(res[scheme]), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(sys.argv[1]), 'posrand_diag.json'), 'w'), indent=1)
