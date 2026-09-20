"""Compute formal novelty/interest measures for a target set.

For each target: min proof length (prover), genus & MDL bits of that proof,
and lemma-value (how many OTHER targets in the set it unlocks/shortens at a
shallow budget). Output one row per target.

Usage: python -m novelty.score_targets --targets ../data/targets_w4.jsonl --out ../results/formal_w4.jsonl
"""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty.prover import Prover
from novelty.measures import genus_proxy, mdl_bits, parse_seq, lemma_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-depth', type=int, default=8)
    ap.add_argument('--lemma-budget', type=int, default=4)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.targets) if l.strip()]
    seqs = [parse_seq(r['prompt']) for r in recs]
    P = Prover(max_depth=args.max_depth, max_fsize=22)

    # precompute min proofs
    rows = []
    t0 = time.time()
    for i, (r, (prem, concl)) in enumerate(zip(recs, seqs)):
        P.memo = {}
        mp = P.min_proof(prem, concl, max_depth=args.max_depth)
        row = {'thm': r['thm'], 'name': r.get('name'), 'source': r.get('source'),
               'gen_lines': r.get('gen_lines'), 'n_premises': r.get('n_premises'),
               'has_solution': r.get('has_solution'),
               'provable': mp is not None}
        if mp:
            nl, txt = mp
            row['min_lines'] = nl
            row['genus'] = genus_proxy(txt)
            row['mdl_bits'] = mdl_bits(txt)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f'  min-proofs {i+1}/{len(recs)} ({time.time()-t0:.0f}s)',
                  file=sys.stderr, flush=True)

    # lemma value: each provable theorem vs the rest
    for i, (r, (prem, concl)) in enumerate(zip(recs, seqs)):
        if not rows[i]['provable']:
            rows[i]['lemma_unlocked'] = None
            rows[i]['lemma_shortened'] = None
            continue
        others = [seqs[j] for j in range(len(seqs)) if j != i]
        P.memo = {}
        lv = lemma_value(prem, concl, others, prover=P,
                         budget_depth=args.lemma_budget)
        rows[i]['lemma_unlocked'] = lv['unlocked']
        rows[i]['lemma_shortened'] = lv['shortened']
        rows[i]['lemma_value'] = lv['unlocked'] + 0.5 * lv['shortened']
        if (i + 1) % 25 == 0:
            print(f'  lemma-value {i+1}/{len(recs)} ({time.time()-t0:.0f}s)',
                  file=sys.stderr, flush=True)

    with open(args.out, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    prov = sum(r['provable'] for r in rows)
    print(f'scored {len(rows)} targets; {prov} provable; wrote {args.out}')


if __name__ == '__main__':
    main()
