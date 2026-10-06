#!/usr/bin/env python3
"""Reviewer's Lean recheck for run noise-floor.

  python3 review_nf_lean_recheck.py sample   -> rv/leanwork/counted_sample.jsonl
        >= 200 counted (nd_verify-accepted) proofs per arm, drawn from held-out greedy,
        the frozen ladder and both coverage pools, plus the three gap-closer cells.
  python3 review_nf_lean_recheck.py gate     -> rv/leanwork/gate_disagree_distinct.jsonl
        the distinct (prompt, ND proof) pairs behind every in-loop-gate Lean-only acceptance.

Then, from a checkout that has the post-BOTE-fix nd2lean.py (e.g. ~/work/efficiency):
  python3 nd2lean.py   --check <file> --out <report>
  python3 lean_check.py --check <file> --out <report>     # allowlist + axioms + term size
"""
import json, os, random, sys, collections

R = os.environ.get('NF_REVIEW_ROOT', '/home/dan/review/noise-floor')
A = f'{R}/artifacts/nf'
OUT = f'{R}/rv/leanwork'


def _pool(fn, key='name'):
    return {json.loads(l)[key]: json.loads(l) for l in open(fn) if l.strip()}


def sample(seed=1234, per_heldout=120, per_ladder=60, per_cov=20):
    random.seed(seed)
    os.makedirs(OUT, exist_ok=True)
    tr = _pool(f'{R}/data/ladder/transfer.jsonl')

    def heldout(p, s, n):
        rows = [json.loads(l) for l in open(f'{A}/heldout_p{p}_s{s}.jsonl')]
        sel = [r for r in rows if r['solved'] and r['proofs']]
        random.shuffle(sel)
        return [{'prompt': r['prompt'], 'proof': r['proofs'][0], 'src': f'heldout_p{p}_s{s}',
                 'name': r['name']} for r in sel[:n]]

    def ladder(name, n):
        d = f'{A}/{name}'
        if not os.path.isdir(d):
            return []
        rows = [json.loads(l) for l in open(f'{d}/found_transfer_8.jsonl') if l.strip()]
        random.shuffle(rows)
        return [{'prompt': tr[r['name']]['prompt'], 'proof': r['proof'], 'src': name,
                 'name': r['name']} for r in rows[:n]]

    def cov(tag, p, s, n):
        f = f'{A}/cov_{tag}_p{p}_s{s}.s0.jsonl'
        if not os.path.exists(f):
            return []
        out = []
        for l in open(f):
            r = json.loads(l)
            for pr in r['proofs']:
                out.append({'prompt': r['prompt'], 'proof': pr['proof'] if isinstance(pr, dict) else pr,
                            'src': f'cov_{tag}_p{p}_s{s}', 'name': r['name']})
        random.shuffle(out)
        return out[:n]

    recs = []
    for p in (1, 2, 3, 4):
        for s in (0, 1):
            recs += heldout(p, s, per_heldout)
            recs += ladder(f'la_frozen_p{p}_s{s}', per_ladder)
            recs += cov('red', p, s, per_cov) + cov('req8', p, s, per_cov)
    for p in (1, 2, 3, 4):
        for s in random.sample(range(2, 13), 3):
            recs += heldout(p, s, 15)
    for nm in ('la_frozen_dsg_g1_s0', 'la_frozen_dsc_a1_s1', 'la_T1_dsc_a1_s1'):
        recs += ladder(nm, per_ladder)
    with open(f'{OUT}/counted_sample.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    per = collections.Counter()
    for r in recs:
        for p in '1234':
            for s in '01':
                if f'_p{p}_s{s}' in r['src']:
                    per[f'p{p}_s{s}'] += 1
    print(len(recs), 'records; per arm', dict(per))


def gate():
    import glob
    os.makedirs(OUT, exist_ok=True)
    seen = {}
    n = 0
    for f in sorted(glob.glob(f'{A}/gate_*.disagree.jsonl')):
        for l in open(f):
            if not l.strip():
                continue
            d = json.loads(l); n += 1
            seen[(d['prompt'], d['nd'])] = d
    with open(f'{OUT}/gate_disagree_distinct.jsonl', 'w') as fh:
        for (p, nd) in seen:
            fh.write(json.dumps({'prompt': p, 'proof': nd}) + '\n')
    print(n, 'raw gate disagreement records ->', len(seen), 'distinct (prompt, proof)')


if __name__ == '__main__':
    {'sample': sample, 'gate': gate}[sys.argv[1] if len(sys.argv) > 1 else 'sample']()
