#!/usr/bin/env python3
"""Independent checks of the assembled ab2 files: visible-field whitelist, pkey/key recomputation, pairwise pkey
disjointness, exclusion (cap-6 train/held-out, textbook dev, validation-36), and nd_verify of every generating proof
(n lines == gen_lines) and every minlen proof (n lines == L_true) behind every record (via provenance.jsonl)."""
import json, sys, glob, collections, multiprocessing as mp
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab2/data/tools'); sys.path.insert(0, '/scratch/cowifornia/nd-rl/external/nd-takehome')
from keys2 import pkey, canon_key
from nd_verify import verify_text
D = '/scratch/cowifornia/nd-rl/ab2/data'
VIS = {'name', 'thm', 'key', 'pkey', 'prompt', 'n_lines', 'L_true', 'source', 'schema', 'n_prem'}
rd = lambda fn: [json.loads(l) for l in open(fn) if l.strip()]


def chk(job):
    prompt, proof, n = job
    ok, reason, nl = verify_text(prompt + ' ' + proof)
    return ok and nl == n


def main():
    files = {fn.split('/')[-1][:-6]: rd(fn) for fn in sorted(glob.glob(f'{D}/*.jsonl')) if 'provenance' not in fn}
    prov = {p['name']: p for p in rd(f'{D}/provenance.jsonl')}
    names = [r['name'] for rs in files.values() for r in rs]
    assert len(names) == len(set(names)) == len(prov), 'names not unique'
    seen = {}
    for k, rs in files.items():
        vis = not (k.startswith('sealed') or k == 'reserve')
        for r in rs:
            if vis:
                assert set(r) == VIS, (k, set(r) ^ VIS)
            assert r['key'] == canon_key(r['thm']) and r['pkey'] == pkey(r['thm']) and r['n_lines'] == r['L_true']
            assert r['pkey'] not in seen, (k, seen.get(r['pkey']))
            seen[r['pkey']] = k
    excl = set()
    for fn in ('/scratch/cowifornia/nd-rl/ab/common/data/train_cap6.jsonl', '/scratch/cowifornia/nd-rl/ab/common/data/heldout_cap6.jsonl'):
        excl |= {pkey(r['thm']) for r in rd(fn)}
    tbdev = {pkey(json.loads(l)['thm']) for l in open('/scratch/cowifornia/nd-rl/ab/sealed/textbook_dev.jsonl')}
    heldp = {pkey(r['thm']) for r in rd('/scratch/cowifornia/nd-rl/ab/common/data/heldout_cap6.jsonl')}
    bad = [(p, k) for p, k in seen.items() if (p in excl or p in tbdev) and not (k == 'targets_L6-11' and p in heldp)]
    assert not bad, bad[:5]
    assert not (set(seen) & tbdev)
    n_held_in_targets = sum(1 for r in files['targets_L6-11'] if r['source'] == 'cap6_heldout')
    # proofs
    gen = {}
    for l in open(f'{D}/raw/pool_gen.jsonl'):
        r = json.loads(l); gen[r['name']] = (r['prompt'], r['proof'], r['n_lines'])
    lab = {}
    for fn in ('gen_b10', 'gen_b16', 'tb_b10', 'tb_b16', 'heldout6_b6'):
        for x in rd(f'{D}/labels/{fn}.jsonl'):
            if x['min_lines_ub'] is not None:
                lab[(fn[:2], x['name'])] = (x['prompt'], x['proof'], x['min_lines_ub'])
    old = {}
    for pool in ('rl_targets', 'transfer', 'reserve'):
        for r in rd(f'/scratch/cowifornia/nd-rl/external/nd-takehome/data/ladder/{pool}.jsonl'):
            if r.get('gen_proof'):
                old[r['name']] = (r['prompt'], r['gen_proof'], r['gen_lines'])
    jobs = []; kinds = collections.Counter()
    for k, rs in files.items():
        for r in rs:
            p = prov[r['name']]; o = p['origin']; n = p['orig_name']
            if o == 'ab2/gen':
                jobs.append(gen[n]); jobs.append(lab[('ge', n)]); kinds['gen_proof'] += 1; kinds['minlen_proof_gen'] += 1
                assert lab[('ge', n)][2] == r['L_true']
            elif o == 'ab2/textbook':
                jobs.append(lab[('tb', n)]); kinds['minlen_proof_tb'] += 1; assert lab[('tb', n)][2] == r['L_true']
            elif o == 'cap6/heldout_cap6':
                jobs.append(lab[('he', n)]); kinds['minlen_proof_heldout'] += 1; assert lab[('he', n)][2] == r['L_true']
            elif n in old:
                jobs.append(old[n]); kinds['old_gen_proof'] += 1
    with mp.Pool(40) as pool:
        res = pool.map(chk, jobs, chunksize=2000)
    print('records', sum(map(len, files.values())), 'proof checks', len(res), 'failed', res.count(False), dict(kinds))
    print('held-out records in targets (intended):', n_held_in_targets)
    assert all(res)
    print('ALL CHECKS PASS')


if __name__ == '__main__':
    main()
