#!/usr/bin/env python3
"""Fill the outcome fields of results/<id>_<slug>.json from artifacts/autoresearch/<id>_<slug>/summary.json.

  python3 record.py 003_wider-ffn keep|discard "surprise text"
"""
import glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
name, status, surprise = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None
fn = f'{HERE}/results/{name}.json'
r = json.load(open(fn))
s = json.load(open(f'{HERE}/../../../../artifacts/autoresearch/{name}/summary.json'))
assert status in ('keep', 'discard', 'crash', 'timeout')
if s['status'] != 'ok':
    assert status in ('crash', 'timeout'), f"harness status {s['status']}"
pre = s.get('pretrain', {})
r.update(metric=s['metric'], status=status, pretrain_secs=round(pre.get('secs', 0)), n_params=pre.get('n_params'),
         heldout_greedy=s.get('eval', {}).get('heldout_greedy'), lstar=s.get('eval', {}).get('lstar'),
         total_secs=round(s['total_secs']), pretrain_stats={k: v for k, v in pre.items() if k not in ('secs', 'n_params')},
         harness_sha=s['harness_sha'], pretrain_sha=s['pretrain_sha'],
         commit=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
if surprise:
    r['surprise'] = surprise
json.dump(r, open(fn, 'w'), indent=2)
print(json.dumps({k: r[k] for k in ('id', 'slug', 'metric', 'status', 'pretrain_secs', 'total_secs', 'heldout_greedy', 'lstar')}))
