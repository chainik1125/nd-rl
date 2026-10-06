#!/usr/bin/env python3
"""Checker of record for the counted proofs of Lean-format arms: every proof in found_<R>.jsonl / found_transfer_<R>.jsonl goes through the
UNMODIFIED nd2lean.py --check (official translation + Lean) and nd_verify; agreement counts -> artifacts/lf/record_<arm>_<pool>.json.
  python3 pod/lf/record.py artifacts/lf/ei_d3_seq_s0 [more arm dirs...]"""
import sys, os, glob, json, subprocess, collections
for d in sys.argv[1:]:
    arm = os.path.basename(d.rstrip('/'))
    R = max(int(f.split('_')[-1][:-5]) for f in glob.glob(f'{d}/round_*.json'))
    for pool in ('found', 'found_transfer'):
        src = f'{d}/{pool}_{R}.jsonl'; rep = f'{d}/record_{pool}_{R}.jsonl'
        p = subprocess.run(['python3', 'nd2lean.py', '--check', src, '--out', rep], capture_output=True, text=True)
        rows = [json.loads(l) for l in open(rep)]
        c = collections.Counter((r['nd_ok'], r['lean_ok']) for r in rows)
        out = {'source': src, 'round': R, 'n': len(rows), 'both_accept': c[(True, True)], 'nd_ok_lean_rej': c[(True, False)], 'nd_rej_lean_ok': c[(False, True)], 'both_reject': c[(False, False)],
               'stdout_tail': p.stdout.strip().splitlines()[-1:] if p.stdout.strip() else [], 'stderr_tail': p.stderr[-300:]}
        json.dump(out, open(f'artifacts/lf/record_{arm}_{pool}.json', 'w'), indent=1)
        print(arm, pool, out['n'], 'both accept', out['both_accept'], 'disagree', out['nd_ok_lean_rej'] + out['nd_rej_lean_ok'], flush=True)
