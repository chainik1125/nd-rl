#!/usr/bin/env python3
"""pools_v2/MANIFEST.json from the collectors' .summary.json files + sha256 of pool and label files."""
import glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL
m = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), generator=PL.CKPT,
         generator_sha256=PL.sha256(PL.CKPT), nd_takehome='branch dan @ 4dcd0e4f',
         sampling=dict(temperature=0.8, max_action=512, max_steps=96, batch=2048, early='eos', arm='SN assign=True'),
         splits='splits.json', splits_sha256=PL.sha256(os.path.join(PL.POOLS, 'splits.json')), pools={})
SPECS = [(s, None) for s in PL.SET_FILES] + [('rr600', '96b'), ('rl_targets', '48b'), ('dev1108', '48b')]
for s, kk in SPECS:
    p = PL.pool_path(s, kk)
    sf = p[:-6] + '.summary.json'
    if not os.path.exists(sf):
        continue
    S = json.load(open(sf))
    rep = os.path.join(PL.POOLS, PL.pool_stem(s, kk) + '.replay.json')
    m['pools'][PL.pool_stem(s, kk)] = dict(file=os.path.basename(p), sha256=PL.sha256(p), labels=os.path.basename(PL.labels_path(s, kk)),
                         labels_sha256=PL.sha256(PL.labels_path(s, kk)), phase=2 if kk else 1, sampling_seed=S['seed'], source=PL.SET_FILES[s],
                         n_theorems=S['n_theorems'], k=S['k'], seed=S['seed'], attempts=S['attempts'],
                         wall_s=round(S['wall_s'], 1), sample_wall_s=round(S['sample_wall_s'], 1), lean_s=round(S['lean_s'], 1),
                         finished_rate=round(S['finished_rate'], 4), validity_rate=round(S['validity_rate'], 4),
                         p_valid_given_finished=round(S['p_valid_given_finished'], 4), pass_at_k=round(S['pass_at_k'], 4),
                         gen_tokens=S['gen_tokens'], prefill_tokens=S['prefill_tokens'], ends=S['ends'],
                         replay=json.load(open(rep)) if os.path.exists(rep) else None)
json.dump(m, open(os.path.join(PL.POOLS, 'MANIFEST.json'), 'w'), indent=1)
print(json.dumps({s: {k: v[k] for k in ('n_theorems', 'k', 'finished_rate', 'validity_rate', 'pass_at_k', 'wall_s')} for s, v in m['pools'].items()}, indent=0))
