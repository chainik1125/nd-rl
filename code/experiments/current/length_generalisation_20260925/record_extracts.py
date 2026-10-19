#!/usr/bin/env python3
"""Small extracts for the repository record (CPU only, reads existing files):

    python record_extracts.py

report/runs_summary.json    per training run: status, seed, compute used, wall secs, rounds, samples, held-out greedy,
                            checkpoint sha256 (from runs/<label>/summary.json and eval/<label>/meta.json)
report/frozen_shas.json     the frozen-file hashes recorded by C_s101 and D_s101 (only the targets file differs)
report/redraw_check.json    how many theorems of the earlier (pre-final) draw survive in the final sealed files
report/base039_context.json dev/frontier of the run-3 seed recipe on the 039 pretrain base vs the original base
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import read_jsonl, write_json  # noqa: E402
AB3 = '/scratch/cowifornia/nd-rl/ab3'


def sha(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    runs = {}
    for l in sorted(os.listdir(os.path.join(HERE, 'runs'))):
        fn = os.path.join(HERE, 'runs', l, 'summary.json')
        if not os.path.isfile(fn) or '.failed.' in l:
            continue
        s = json.load(open(fn))
        m = os.path.join(HERE, 'eval', l, 'meta.json')
        runs[l] = {'status': s['status'], 'seed': s['seed'], 'flops_budget': s.get('flops_budget'),
                   'compute_frac': (s.get('compute') or {}).get('frac'), 'rl_secs': s.get('rl_secs'),
                   'rounds': s['rl_stats'].get('rounds'), 'total_samples': s['rl_stats'].get('total_samples'),
                   'heldout_greedy': s.get('heldout_greedy'), 'budget_exhausted': s.get('budget_exhausted'),
                   'rl_py_sha256': s['shas']['rl.py'], 'targets_sha256': s['shas'].get('data/targets_L6-11.jsonl'),
                   'final_pt_sha256': json.load(open(m))['ckpt_sha256'] if os.path.isfile(m) else None,
                   'final_pt': os.path.join(HERE, 'runs', l, 'final.pt')}
    write_json(os.path.join(HERE, 'report', 'runs_summary.json'), runs)
    c = json.load(open(os.path.join(HERE, 'runs', 'C_s101', 'summary.json')))['shas']
    d = json.load(open(os.path.join(HERE, 'runs', 'D_s101', 'summary.json')))['shas']
    write_json(os.path.join(HERE, 'report', 'frozen_shas.json'),
               {'C_s101 (ab3/frozen)': c, 'D_s101 (ab3/frozen_comp)': d, 'differing': sorted(k for k in c if c[k] != d.get(k))})
    # redraw: earlier draw (testsets_predraw) vs final files
    fin = {}
    for f in ('matched_L6-14', 'sealed_L15-16', 'sealed_L17-18'):
        for r in read_jsonl(f'{AB3}/sealed/{f}.jsonl'):
            fin[r['prompt']] = r
    old_m = read_jsonl(os.path.join(HERE, 'testsets_predraw', 'matched.jsonl'))
    old_s = [r for r in read_jsonl(os.path.join(HERE, 'testsets_predraw', 'std.jsonl')) if r['L_true'] >= 15]
    write_json(os.path.join(HERE, 'report', 'redraw_check.json'), {
        'matched_predraw_theorems': len(old_m), 'matched_predraw_in_final_files': sum(r['prompt'] in fin for r in old_m),
        'sealed_L15_18_predraw_theorems': len(old_s), 'sealed_L15_18_predraw_in_final_files': sum(r['prompt'] in fin for r in old_s),
        'final_files_sha256': {f: sha(f'{AB3}/sealed/{f}.jsonl') for f in ('matched_L6-14', 'sealed_L15-16', 'sealed_L17-18')}})
    ctx = {}
    for lab, fn in [('039 base, s101', f'{AB3}/base039/runs/seed039_s101/summary.json'),
                    ('039 base, s102', f'{AB3}/base039/runs/seed039_s102/summary.json'),
                    ('039 base seed 1, s101', f'{AB3}/base039/runs/seed039s1_s101/summary.json'),
                    ('039 base seed 2, s101', f'{AB3}/base039/runs/seed039s2_s101/summary.json'),
                    ('original base, s101', f'{AB3}/seed/runs/seed_s101/summary.json'),
                    ('original base, s102', f'{AB3}/seed/runs/seed_s102/summary.json')]:
        s = json.load(open(fn))
        ctx[lab] = {'source': fn, 'source_sha256': sha(fn), 'dev_of_1200': s.get('metric'), 'frontier_of_1000': s.get('frontier'),
                    'heldout_greedy': s.get('heldout_greedy'), 'status': s.get('status')}
    write_json(os.path.join(HERE, 'report', 'base039_context.json'), ctx)
    print('wrote runs_summary.json, frozen_shas.json, redraw_check.json, base039_context.json')


if __name__ == '__main__':
    main()
