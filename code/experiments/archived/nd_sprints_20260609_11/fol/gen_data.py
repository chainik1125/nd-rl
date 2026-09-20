"""Generate first-order ND proof datasets. Every emitted proof is checked by
the independent FOL verifier. Records carry the distinct-constant count so we
can build the parameter token-wall experiment.

Usage:
  python -m fol.gen_data --n 200000 --consts abc --min-lines 2 --max-lines 16 \
     --seed 0 --out data/fol_pool.jsonl
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fol.core import (sample_tree, linearize, proof_to_tokens, theorem_id,
                      consts as fconsts)
from fol.verify import verify_text
from fol.tokenizer import STOI

MAX_TOKENS = 900


def _rules_used(t, acc):
    if t.rule != 'AS':
        acc.add(t.rule)
    for c in t.children:
        _rules_used(c, acc)


def gen_chunk(args):
    seed, n_attempts, consts_inv, min_lines, max_lines, steps_hi = args
    rng = random.Random(seed)
    out = []
    for _ in range(n_attempts):
        steps = rng.randint(4, steps_hi)
        t = sample_tree(rng, n_steps=steps, consts_inv=list(consts_inv),
                        var_inv=['x', 'y'], min_lines=min_lines, max_lines=max_lines)
        if t is None:
            continue
        try:
            prem, concl, lines = linearize(t)
        except ValueError:
            continue
        if concl in prem:
            continue
        if not (min_lines <= len(lines) <= max_lines):
            continue
        toks = proof_to_tokens(prem, concl, lines)
        if len(toks) > MAX_TOKENS or any(tk not in STOI for tk in toks):
            continue
        text = ' '.join(toks)
        ok, reason, nl = verify_text(text)
        if not ok:
            continue
        ru = set()
        _rules_used(t, ru)
        used_consts = set()
        for ln in lines:
            used_consts |= fconsts(ln.formula)
        out.append({'thm': theorem_id(prem, concl), 'text': text,
                    'n_lines': len(lines), 'rules': sorted(ru),
                    'n_consts': len(used_consts), 'consts': sorted(used_consts),
                    'quant': bool(ru & {'ALLI', 'ALLE', 'EXI', 'EXE'})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--consts', default='abc', help='constant inventory string')
    ap.add_argument('--min-lines', type=int, default=2)
    ap.add_argument('--max-lines', type=int, default=16)
    ap.add_argument('--steps-hi', type=int, default=22)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    seen = set()
    n_written = 0
    batch = 0
    with open(args.out, 'w') as fo, Pool(args.workers) as pool:
        while n_written < args.n:
            jobs = [(args.seed * 100003 + batch * 911 + w, 6000, args.consts,
                     args.min_lines, args.max_lines, args.steps_hi)
                    for w in range(args.workers)]
            batch += 1
            for recs in pool.imap_unordered(gen_chunk, jobs):
                for r in recs:
                    if r['thm'] in seen:
                        continue
                    seen.add(r['thm'])
                    fo.write(json.dumps(r) + '\n')
                    n_written += 1
                    if n_written >= args.n:
                        break
                if n_written >= args.n:
                    break
            print(f'... {n_written}/{args.n}', file=sys.stderr, flush=True)
    print(f'wrote {n_written} to {args.out}')


if __name__ == '__main__':
    main()
