#!/usr/bin/env python3
"""Label-isolation test (design audit item 5): copy a pool, write a PERMUTED labels file next to it (valid and reason
shuffled across theorems and attempts), re-run a scorer command against the copy, and require byte-identical score
files.  Usage: python label_permutation_test.py --set dev1108 --cmd "python baseline_scores.py --sets dev1108"
(the command must read pools via pool_lib, which honours SEL_POOLS / SEL_SCORES)."""
import argparse, filecmp, glob, json, os, random, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL

ap = argparse.ArgumentParser()
ap.add_argument('--set', required=True)
ap.add_argument('--cmd', required=True)
ap.add_argument('--scorers', nargs='+', required=True)
ap.add_argument('--inputs', nargs='*', default=[], help='input score files (other scorers) the command reads; copied as-is')
a = ap.parse_args()
tmp = tempfile.mkdtemp(prefix='permtest_', dir=os.path.join(PL.RUNS, 'logs'))
tp, ts = os.path.join(tmp, 'pools'), os.path.join(tmp, 'scores')
os.makedirs(tp); os.makedirs(ts)
shutil.copy(PL.pool_path(a.set), tp)
for sc in a.inputs:
    shutil.copy(os.path.join(PL.SCORES, f'{PL.pool_stem(a.set)}__{sc}.jsonl'), ts)
labs = PL.read_jsonl(PL.labels_path(a.set))
flat_v = [v for r in labs for v in r['valid']]; flat_r = [x for r in labs for x in r['reason']]
rng = random.Random(1); perm = list(range(len(flat_v))); rng.shuffle(perm)
j = 0
for r in labs:
    n = len(r['valid'])
    r['valid'] = [flat_v[perm[j + t]] for t in range(n)]; r['reason'] = [flat_r[perm[j + t]] for t in range(n)]; j += n
PL.write_jsonl(os.path.join(tp, os.path.basename(PL.labels_path(a.set))), labs)
env = dict(os.environ, SEL_POOLS=tp, SEL_SCORES=ts)
subprocess.run(a.cmd, shell=True, check=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
res = {}
for sc in a.scorers:
    f = f'{PL.pool_stem(a.set)}__{sc}.jsonl'
    res[sc] = filecmp.cmp(os.path.join(PL.SCORES, f), os.path.join(ts, f), shallow=False)
print(json.dumps(dict(set=a.set, identical=res, tmp=tmp)))
sys.exit(0 if all(res.values()) else 1)
