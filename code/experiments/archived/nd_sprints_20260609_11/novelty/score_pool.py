"""Score a large pool of target theorems by difficulty + novelty (multiprocess).

Per target: min_lines (difficulty), genus (structural novelty of a min proof),
mdl_bits, and lemma_value over a fixed PROBE set (how many probe targets it
unlocks/shortens at a shallow budget) — the theory-predicted best seeding signal.

Usage: python -m novelty.score_pool --targets data/rl_targets.jsonl \
    --probe data/eval_transfer.jsonl --n-probe 50 --out results/pool_novelty.jsonl
"""
import argparse
import json
import sys
import os
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty.prover import Prover
from novelty.measures import genus_proxy, mdl_bits, parse_seq, lemma_value

_PROBE = None


def rec_prompt(r):
    if 'prompt' in r:
        return r['prompt']
    toks = r['text'].split()
    return ' '.join(toks[:toks.index('PRF') + 1])


def _init(probe_seqs):
    global _PROBE
    _PROBE = probe_seqs


def _score(rec):
    prem, concl = parse_seq(rec_prompt(rec))
    P = Prover(max_depth=7, max_fsize=22)
    mp = P.min_proof(prem, concl, max_depth=7)
    row = {'thm': rec['thm'], 'gen_lines': rec.get('n_lines'),
           'provable': mp is not None}
    if not mp:
        return row
    nl, txt = mp
    row['min_lines'] = nl
    row['genus'] = genus_proxy(txt)
    row['mdl_bits'] = mdl_bits(txt)
    P.memo = {}
    lv = lemma_value(prem, concl, _PROBE, prover=P, budget_depth=4)
    row['lemma_unlocked'] = lv['unlocked']
    row['lemma_shortened'] = lv['shortened']
    row['lemma_value'] = lv['unlocked'] + 0.5 * lv['shortened']
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--n-probe', type=int, default=50)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.targets) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    probe = [json.loads(l) for l in open(args.probe) if l.strip()][:args.n_probe]
    probe_seqs = [parse_seq(rec_prompt(r)) for r in probe]

    with Pool(args.workers, initializer=_init, initargs=(probe_seqs,)) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_score, recs, chunksize=8)):
            rows.append(row)
            if (i + 1) % 200 == 0:
                print(f'  {i+1}/{len(recs)}', file=sys.stderr, flush=True)
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    prov = sum(r['provable'] for r in rows)
    print(f'scored {len(rows)}; {prov} provable; wrote {args.out}')


if __name__ == '__main__':
    main()
