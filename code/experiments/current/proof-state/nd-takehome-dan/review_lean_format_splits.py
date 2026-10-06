#!/usr/bin/env python3
"""Reviewer (lean-format, phase 1): split disjointness by renaming class (own canonicaliser: min over 24 atom permutations, premises
order-insensitive), f = 0 check of the depth-3 training set (ND depth and nested-`fun` depth of the Lean rendering), cap-6 check,
sequence-length statistics of both formats, render/inverse round trip.
  python3 review_lean_format_splits.py > artifacts/review_lf/splits_stdout.txt"""
import json, sys, collections, statistics
sys.path.insert(0, '.')
from review_lean_format_recount import jl, cls, parse_nd, max_depth

FILES = {
    'train_a1': 'data/p2/train_depth3_f0_a1.jsonl', 'train_full': 'data/train.jsonl',
    'd3_targets': 'data/p2/targets_depth3.jsonl', 'd3_transfer': 'data/p2/transfer_depth3.jsonl', 'p2_heldout': 'data/p2/heldout.jsonl',
    'la_targets': 'data/ladder/rl_targets.jsonl', 'la_transfer': 'data/ladder/transfer.jsonl',
    'heldout': 'data/heldout.jsonl', 'th_transfer': 'data/transfer.jsonl',
}
C = {}
for k, fn in FILES.items():
    rows = jl(fn)
    C[k] = collections.Counter(cls(r['prompt']) for r in rows)
    print(f'{k:12s} {fn:40s} n {len(rows):>7} classes {len(C[k]):>7}', flush=True)
print('\npairwise class overlaps (training files and RL-target pools vs every evaluation pool; 0 expected):')
pairs = [('train_a1', x) for x in ('d3_targets', 'd3_transfer', 'p2_heldout')] + [('d3_targets', 'd3_transfer'), ('d3_targets', 'p2_heldout'), ('d3_transfer', 'p2_heldout')] + \
        [('train_full', x) for x in ('la_targets', 'la_transfer', 'heldout', 'th_transfer')] + [('la_targets', 'la_transfer'), ('la_targets', 'heldout'), ('la_transfer', 'heldout'), ('la_targets', 'th_transfer'), ('la_transfer', 'th_transfer')]
for a, b in pairs:
    print(f'  {a:12s} x {b:12s}: {len(set(C[a]) & set(C[b]))}')

# ---- f = 0 and cap checks on the training sets ----
from lean_tok import LeanTokenizer, prompt_tokens, proof_tokens, inverse
from tokenizer import Tokenizer
ltok = LeanTokenizer('lean_seq'); ttok = Tokenizer('abs')
for k in ('train_a1', 'train_full'):
    rows = jl(FILES[k])
    dh = collections.Counter(); fh = collections.Counter(); lh = collections.Counter(); bad_rt = 0
    for r in rows:
        L = parse_nd(r['proof']); dh[max_depth(L)] += 1; lh[len(L)] += 1
        toks = proof_tokens(r['proof'])
        # nested `fun ( n : A ) => by` depth of the Lean rendering (DN's `fun hh` not counted), own counter on the token stream
        d = m = 0; st = []
        for i, t in enumerate(toks):
            if t == '(' and i + 1 < len(toks) and toks[i + 1] == 'fun':
                isbox = toks[i + 2] == '('
                st.append(('box' if isbox else 'dn', 0)); d += isbox; m = max(m, d)
            elif t == '(' and st: st[-1] = (st[-1][0], st[-1][1] + 1)
            elif t == ')' and st:
                if st[-1][1] == 0:
                    d -= st[-1][0] == 'box'; st.pop()
                else: st[-1] = (st[-1][0], st[-1][1] - 1)
        fh[m] += 1
        strs = [x if isinstance(x, str) else f'n{x[1]}' for x in toks]
        if inverse(strs) != r['proof']: bad_rt += 1
    print(f'\n{k}: n {len(rows)} ND max-depth hist {dict(sorted(dh.items()))} | Lean nested-box depth hist {dict(sorted(fh.items()))} | ND lines hist {dict(sorted(lh.items()))} | render->inverse != original: {bad_rt}')

# ---- sequence lengths (prompt + proof tokens incl. <eos>) ----
print('\nsequence lengths, mean / median / p95 / max  (prompt, proof incl. eos)')
def stats(v):
    v = sorted(v); return f'{statistics.mean(v):6.1f} {v[len(v)//2]:4d} {v[int(.95*len(v))]:4d} {v[-1]:4d}'
for k in ('train_a1', 'train_full', 'p2_heldout', 'heldout'):
    rows = jl(FILES[k])
    lp = [len(ltok.encode_prompt(r['prompt'])) for r in rows]; lq = [len(ltok.encode_proof(r['proof'])) for r in rows]
    tp = [len(ttok.encode_prompt(r['prompt'])) for r in rows]; tq = [len(ttok.encode_proof(r['proof'])) for r in rows]
    print(f'{k:12s} lean prompt {stats(lp)} | lean proof {stats(lq)} || token prompt {stats(tp)} | token proof {stats(tq)} || total ratio {(sum(lp)+sum(lq))/(sum(tp)+sum(tq)):.3f}')
print('vocab: lean', ltok.vocab_size, 'token abs', ttok.vocab_size)
