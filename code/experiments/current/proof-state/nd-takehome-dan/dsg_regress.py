#!/usr/bin/env python3
"""ds-generator resume phase: check that the fast decode path adopted from run `efficiency` (2026-09-23) is a
no-op against this branch's pre-efficiency base path.  Adapted from that run's `ef_regress.py`; the only changes
are the prompt source (this repo's ladder transfer pool instead of its 200-target subset) and the gate signature.
For each checkpoint given: sample the same prompts with path='base' (row-keyed rng), with sample.py's default as
this run configures it (ND_SAMPLE_COMPACT=0), and with the fast path with compaction explicitly off, and require
identical token streams and identical decoded proofs.  Compaction is measured separately and NOT required to
match: on 2026-09-23 it flipped 1 row of 128 at decode step 162 -- once the live batch shrinks, the bf16
reduction order changes and a near-tie argmax can flip.  That is an RNG re-draw, not a wrong accept (every
sample is still checked by Lean and nd_verify), but the pre-registration required exact equivalence, so this run
turns compaction off and keeps the pre-registered batch.
  python3 dsg_regress.py <ckpt> [<ckpt> ...]"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_ckpt
import sample as S

tg = [json.loads(l) for l in open('data/ladder/transfer.jsonl')][:16]
prompts = [t['prompt'] for t in tg for _ in range(8)]
MAXNEW, BATCH = 192, 64
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
bad = 0
rep = {}
for ck in sys.argv[1:]:
    model, tok, _ = load_ckpt(ck, dev)
    outs = {}
    for name, kw in (('base', dict(path='base', rowrng=True)),
                     ('default', {}),
                     ('fast_nocompact', dict(path='fast', compact=False, rowrng=True)),
                     ('fast_compact', dict(path='fast', compact=True, rowrng=True))):
        raw = np.zeros((len(prompts), MAXNEW), dtype=np.int16)
        st = {}
        res = S.generate(model, tok, prompts, greedy=False, temperature=0.8, max_new=MAXNEW, batch=BATCH,
                         seed=0, stats=st, gate=False, raw=raw, **kw)
        outs[name] = (raw, res, st)
    ok_ids = bool((outs['base'][0] == outs['default'][0]).all()) and bool((outs['base'][0] == outs['fast_nocompact'][0]).all())
    ok_res = outs['base'][1] == outs['default'][1] == outs['fast_nocompact'][1]
    print(f'{os.path.basename(ck)}: mode {tok.mode}, {len(prompts)} samples — token streams identical {ok_ids}, '
          f'decoded strings identical {ok_res}, mean declen '
          f'{np.mean(outs["base"][2]["declen_by_prompt"]):.1f} / {np.mean(outs["default"][2]["declen_by_prompt"]):.1f}, '
          f'wall {outs["base"][2]["sample_wall_s"]:.1f}s / {outs["default"][2]["sample_wall_s"]:.1f}s; '
          f'compaction (not required to match) differs on {int((outs["base"][0] != outs["fast_compact"][0]).any(1).sum())} rows')
    dc = outs['base'][0] != outs['fast_compact'][0]
    rep[os.path.basename(ck)] = {'n': len(prompts), 'token_streams_identical': ok_ids, 'texts_identical': ok_res,
                                 'compaction_rows_differing': int(dc.any(1).sum()),
                                 'compaction_texts_identical': outs['base'][1] == outs['fast_compact'][1],
                                 'wall_base_s': outs['base'][2]['sample_wall_s'], 'wall_default_s': outs['default'][2]['sample_wall_s'],
                                 'declen_mean': float(np.mean(outs['default'][2]['declen_by_prompt']))}
    bad += (not ok_ids) or (not ok_res)
rep['verdict'] = 'PASS' if bad == 0 else f'FAIL ({bad})'
os.makedirs('artifacts/dsg', exist_ok=True)
json.dump(rep, open('artifacts/dsg/regress.json', 'w'), indent=1)
print('REGRESSION', rep['verdict'])
