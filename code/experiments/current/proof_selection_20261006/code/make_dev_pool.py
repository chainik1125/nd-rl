#!/usr/bin/env python3
"""Dev-only: turn an old pools/*.jsonl (valid ND proofs only, no actions) into a schema-section-1 pool of done attempts
by `state_env.decompose(prompt, nd, canon=True)` (base 0).  Used to develop score_loglik.py before pools_v2 exist."""
import json, os, sys
sys.path.insert(0, os.path.expanduser('~/runs/proof-selection/nd-takehome'))
from state_env import decompose
from tokenizer import make_tokenizer
src, dst, lim = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
tok = make_tokenizer('lean_staten')
n = bad = 0
with open(dst, 'w') as f:
    for li, l in enumerate(open(src)):
        if lim and li >= lim: break
        r = json.loads(l)
        atts = []
        for nd in r['proofs']:
            try:
                steps, toks, env = decompose(r['prompt'], nd, canon=True)
            except Exception as e:
                bad += 1; continue
            acts = [tok.encode_toks(a) for _, a, _ in steps]
            atts.append(dict(i=len(atts), base=0, actions=acts, states=None, done=True, steps=len(acts), nd=nd,
                             gen_tokens=sum(len(a) + 1 for a in acts)))
        n += len(atts)
        f.write(json.dumps(dict(name=r['name'], prompt=r['prompt'], set='dev', ref_lines=None, gen='s0_r16',
                                k=len(atts), sampling=dict(max_action=512), attempts=atts)) + '\n')
print('attempts', n, 'decompose failures', bad)
