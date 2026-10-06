#!/usr/bin/env python3
"""Add the phase-3 (s1/s2 generator) test pools to pools_v2/MANIFEST.json under 'pools' without touching other entries."""
import hashlib, json, os
P = os.path.expanduser('~/runs/proof-selection/pools_v2')
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
m = json.load(open(os.path.join(P, 'MANIFEST.json')))
for s in ('holdout250', 'textbook72'):
    for g in ('s1', 's2'):
        stem = f'{s}__{g}_r16__k64'
        S = json.load(open(os.path.join(P, stem + '.summary.json'))); G = json.load(open(os.path.join(P, stem + '.gen.json')))
        m['pools'][stem] = dict(file=stem + '.jsonl', sha256=sha(os.path.join(P, stem + '.jsonl')), labels=stem + '.labels.jsonl',
                                labels_sha256=sha(os.path.join(P, stem + '.labels.jsonl')), phase=3, generator=G['ckpt'],
                                generator_sha256=sha(os.path.expanduser('~/runs/proof-selection/' + G['ckpt'])), sampling_seed=S['seed'],
                                note='row field gen = s0_r16 is a collector constant; true generator above', added_by='worker B',
                                **{k: S[k] for k in ('n_theorems', 'k', 'attempts', 'finished_rate', 'validity_rate', 'pass_at_k',
                                                     'gen_tokens', 'prefill_tokens', 'replay_checked', 'replay_mismatch')},
                                wall_s=round(S['wall_s'], 1))
json.dump(m, open(os.path.join(P, 'MANIFEST.json'), 'w'), indent=1)
print('added', [k for k in m['pools'] if '__s1_' in k or '__s2_' in k])
