"""Seam set (written 20-34): filter by the signature bound, Lean-check, deduplicate, write production/v2/e_seam_sig.jsonl.

    python approach_e_long/seam_finish.py [--min_sig 15] [--per_len 200] [--procs 6]
Inputs: production/v2/e_seam_raw.jsonl (gen_long.py) and certifier_scoped/results/e_seam_sig.jsonl (run_e.py).
Output records have e_raw_sig's schema (n_body, n_prem, n_atoms, S, prompt, cert_ge_sig, n_sig, n_hyp) plus 'text'
(the written proof, which make_data.py otherwise looks up in e_raw.jsonl by prompt).
Dedup is by production/canon.py's canon_key: within the file, against e_raw.jsonl and the long_rl pools.
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'certifier_scoped'), os.path.join(ROOT, 'production')]
from lean_check import lean_ok, _load  # noqa: E402
from canon import canon_key  # noqa: E402

V2 = os.path.join(ROOT, 'production', 'v2')
POOLS = os.path.join(ROOT, '..', 'long_rl_20261014', 'data')


def key_of(prompt):
    return canon_key(*_load().parse_prompt(prompt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_sig', type=int, default=15)
    ap.add_argument('--per_len', type=int, default=200)
    ap.add_argument('--procs', type=int, default=6)
    a = ap.parse_args()
    raw = [json.loads(line) for line in open(os.path.join(V2, 'e_seam_raw.jsonl'))]
    sig = [json.loads(line) for line in open(os.path.join(ROOT, 'certifier_scoped', 'results', 'e_seam_sig.jsonl'))]
    assert len(raw) == len(sig)
    recs = []
    for r, s in zip(raw, sig):
        assert r['text'].split(' PRF ', 1)[0] + ' PRF' == s['prompt']
        recs.append({**s, 'text': r['text']})
    print(f'{len(recs)} raw; cert_ge_sig >= {a.min_sig}: ', end='')
    recs = [r for r in recs if r['cert_ge_sig'] >= a.min_sig]
    print(len(recs), flush=True)

    texts = [r['text'] for r in recs]
    chunks = [texts[i:i + 200] for i in range(0, len(texts), 200)]
    with ProcessPoolExecutor(a.procs) as ex:
        res = [x for part in ex.map(lean_ok, chunks) for x in part]
    bad = [(r, m) for r, (ok, m) in zip(recs, res) if not ok]
    print(f'Lean: {len(recs)} checked, {len(recs) - len(bad)} accepted, {len(bad)} rejected', flush=True)
    for r, m in bad[:10]:
        print('  REJECT', m[:200], r['text'][:200])
    recs = [r for r, (ok, _) in zip(recs, res) if ok]

    taken = {}
    for line in open(os.path.join(V2, 'e_raw.jsonl')):
        taken.setdefault(key_of(json.loads(line)['text'].split(' PRF ', 1)[0] + ' PRF'), 'e_raw')
    for name in ('rl', 'dev', 'holdout'):
        for line in open(os.path.join(POOLS, f'{name}.jsonl')):
            taken.setdefault(key_of(json.loads(line)['prompt']), name)
    drops = collections.Counter()
    seen, kept = set(), []
    for r in recs:
        k = key_of(r['prompt'])
        if k in seen:
            drops['self'] += 1
        elif k in taken:
            drops[taken[k]] += 1
        else:
            seen.add(k)
            kept.append(r)
    print(f'dedup dropped {dict(drops)}; {len(kept)} left', flush=True)

    per = collections.Counter()
    out = []
    for r in kept:
        if per[r['n_body']] < a.per_len:
            per[r['n_body']] += 1
            out.append(r)
    with open(os.path.join(V2, 'e_seam_sig.jsonl'), 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(out)} to production/v2/e_seam_sig.jsonl; per written length {dict(sorted(per.items()))}')
    print(f'premises {dict(sorted(collections.Counter(r["n_prem"] for r in out).items()))}; '
          f'atoms {dict(sorted(collections.Counter(r["n_atoms"] for r in out).items()))}')
    rat = [r['cert_ge_sig'] / r['n_body'] for r in out]
    print(f'bound/written mean {sum(rat) / len(rat):.3f} min {min(rat):.3f}; '
          f'exact (bound == written) {sum(r["cert_ge_sig"] == r["n_body"] for r in out)}; '
          f'bound >= 20 {sum(r["cert_ge_sig"] >= 20 for r in out)}')


if __name__ == '__main__':
    main()
