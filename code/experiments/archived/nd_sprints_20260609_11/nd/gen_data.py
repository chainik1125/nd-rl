"""Generate stratified ND proof datasets.

Outputs JSONL records:
  {"thm": theorem_id, "text": token-string, "n_lines": int,
   "rules": [...], "discharge": bool, "n_prem": int}

Usage:
  python -m nd.gen_data --n 200000 --min-lines 2 --max-lines 8 --seed 0 --out data/train_L8.jsonl
"""
import argparse
import json
import random
import sys
import os
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.core import (sample_tree, linearize, proof_to_tokens, theorem_id,
                     rule_profile, DISCHARGING)
from nd.verify import verify_text
from nd.tokenizer import STOI, MAX_LINES

MAX_TOKENS = 640


LOCAL_ONLY_WEIGHTS = {
    'ANDI': 1.0, 'ANDE1': 0.7, 'ANDE2': 0.7, 'IMPE': 1.4, 'IMPI': 0.0,
    'ORI1': 0.6, 'ORI2': 0.6, 'ORE': 0.0, 'NEGE': 1.2, 'NEGI': 0.0,
    'BOTE': 0.5, 'DN': 0.7,
}


def gen_chunk(args):
    seed, n_attempts, min_lines, max_lines, steps_hi, local_only = args
    rng = random.Random(seed)
    weights = LOCAL_ONLY_WEIGHTS if local_only else None
    out = []
    for _ in range(n_attempts):
        steps = rng.randint(2, steps_hi)
        t = sample_tree(rng, n_steps=steps, weights=weights)
        if t is None:
            continue
        try:
            prem, concl, lines = linearize(t)
        except ValueError:
            continue
        if concl in prem:
            continue
        if not (min_lines <= len(lines) <= max_lines) or len(lines) > MAX_LINES:
            continue
        toks = proof_to_tokens(prem, concl, lines)
        if len(toks) > MAX_TOKENS or any(tk not in STOI for tk in toks):
            continue
        text = ' '.join(toks)
        ok, reason, nl = verify_text(text)
        if not ok:
            # should never happen; surface loudly
            print('GEN-VERIFY FAIL:', reason, text[:200], file=sys.stderr)
            continue
        rules = sorted(rule_profile(t))
        out.append({'thm': theorem_id(prem, concl), 'text': text,
                    'n_lines': len(lines), 'rules': rules,
                    'discharge': bool(set(rules) & DISCHARGING),
                    'n_prem': len(prem)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True, help='target number of unique-theorem examples')
    ap.add_argument('--min-lines', type=int, default=2)
    ap.add_argument('--max-lines', type=int, default=8)
    ap.add_argument('--steps-hi', type=int, default=None,
                    help='max sampler steps (default scales with max-lines)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--local-only', action='store_true',
                    help='disable discharging rules (IMPI/NEGI/ORE)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--exclude', default=None,
                    help='file with theorem ids (one per line) to exclude')
    args = ap.parse_args()

    steps_hi = args.steps_hi or max(6, args.max_lines + 6)
    excluded = set()
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude) as f:
            excluded = {l.strip() for l in f if l.strip()}

    seen = set()
    n_written = 0
    chunk_attempts = 8000
    batch_id = 0
    with open(args.out, 'w') as fo, Pool(args.workers) as pool:
        while n_written < args.n:
            jobs = [(args.seed * 1000003 + batch_id * 977 + w, chunk_attempts,
                     args.min_lines, args.max_lines, steps_hi, args.local_only)
                    for w in range(args.workers)]
            batch_id += 1
            for recs in pool.imap_unordered(gen_chunk, jobs):
                for r in recs:
                    if r['thm'] in seen or r['thm'] in excluded:
                        continue
                    seen.add(r['thm'])
                    fo.write(json.dumps(r) + '\n')
                    n_written += 1
                    if n_written >= args.n:
                        break
                if n_written >= args.n:
                    break
            print(f'... {n_written}/{args.n}', file=sys.stderr)
    print(f'wrote {n_written} examples to {args.out}')


if __name__ == '__main__':
    main()
