#!/usr/bin/env python3
"""Phase 2: combine per-judge masked mean log-likelihoods into cross-seed scorers (per attempt, aligned by name).
  loglik_mean_masked_xseed     = mean over judges s0, s1, s2
  loglik_mean_masked_siblings  = mean over s1, s2 (judges that never generated the s0 pool)
  completion_loglik_{xseed,siblings,s1,s2} = done ? x : -inf
Usage: make_xseed.py <stem> ..."""
import json, math, os, sys
D = os.path.expanduser('~/runs/proof-selection/scores')
P = os.path.expanduser('~/runs/proof-selection/pools_v2')
rd = lambda f: {r['name']: r['scores'] for r in map(json.loads, open(os.path.join(D, f)))}
for stem in sys.argv[1:]:
    J = {j: rd(f'{stem}__loglik_mean_masked{"" if j == "s0" else "_" + j}.jsonl') for j in ('s0', 's1', 's2')}
    done = {r['name']: [a['done'] for a in r['attempts']] for r in map(json.loads, open(os.path.join(P, stem + '.jsonl')))}
    out = {k: {} for k in ('loglik_mean_masked_xseed', 'loglik_mean_masked_siblings', 'completion_loglik_xseed',
                           'completion_loglik_siblings', 'completion_loglik_s1', 'completion_loglik_s2')}
    for n in J['s0']:
        x = [sum(v) / 3 for v in zip(J['s0'][n], J['s1'][n], J['s2'][n])]
        sb = [(a + b) / 2 for a, b in zip(J['s1'][n], J['s2'][n])]
        c = lambda xs: [v if d else -math.inf for v, d in zip(xs, done[n])]
        out['loglik_mean_masked_xseed'][n] = x; out['loglik_mean_masked_siblings'][n] = sb
        out['completion_loglik_xseed'][n] = c(x); out['completion_loglik_siblings'][n] = c(sb)
        out['completion_loglik_s1'][n] = c(J['s1'][n]); out['completion_loglik_s2'][n] = c(J['s2'][n])
    for k, m in out.items():
        with open(os.path.join(D, f'{stem}__{k}.jsonl'), 'w') as f:
            for n, sc in m.items():
                f.write(json.dumps({'name': n, 'scores': sc}) + '\n')
    print(stem, 'ok', len(J['s0']))
