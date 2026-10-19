#!/usr/bin/env python3
"""Necessity oracle (run 5): does a theorem REQUIRE a pattern?  Evaluation / labelling only, never training data.

  python necessity.py --in CANDS.jsonl --out LABELLED.jsonl --pattern reductio|derived_ore_strict --bound 10 --time 60 --procs 6

Per theorem, two bounded minlen searches (minlen.py, same search space, iterative deepening to --bound):
  unrestricted  -> min_lines_ub (shortest verifier-valid proof found, or None), proof
  restricted    -> r_min_lines_ub, r_timeout, with --forbid DN (reductio: DN and the reductio template disabled) or
                   --forbid ORE_DERIVED (derived_ore_strict: ORE only over a premise, an open hypothesis or ( X v X ))
A theorem `requires` the pattern iff the unrestricted search finds a proof of <= bound lines AND the restricted search
fails within the bound WITHOUT a timeout.  `requires_wide` additionally accepts theorems the unrestricted search cannot
reach within the bound when --classical_only marks them intuitionistically unprovable (reductio only: DN is the only
classical rule, so such a theorem needs DN for any proof).  `oracle_ok` checks that the unrestricted proof of a required
theorem contains the pattern (patterns.py on the pruned proof) — a consistency check of the two searches, not a proof
of soundness: both searches share minlen's restricted formula space, so a model may still find a pattern-free proof
outside it.  Every found proof is re-verified with nd_verify by minlen.
"""
import argparse, json, os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minlen import minlen
from patterns import classify
from patterns2 import classify2


def pats(proof, pattern):
    if proof is None:
        return None
    c = classify(proof); c2 = classify2(proof)
    out = {k: c[k] for k in ('reductio', 'derived_ore', 'derived_ore_strict', 'depth3')}
    out.update({k: c2[k] for k in P2}); out['impe_chain'] = c2['impe_chain']; out['ore_nesting'] = c2['ore_nesting']
    return out

FORBID = {'reductio': ('DN',), 'derived_ore_strict': ('ORE_DERIVED',), 'derived_ore': ('ORE_DERIVED_LOOSE',),
          'impi_ore': ('ORE_IN_IMPI',), 'negi_ande_hyp': ('ANDE_NEGI_HYP',), 'nested_ore': ('ORE_IN_ORE',),
          'depth4': (), 'impe_chain4': None, 'ori_ore': None}     # depth4: restricted = --max_depth 3; None: no restriction ("uses" only)
MAXDEPTH = {'depth4': 3}
P2 = ('depth4', 'impe_chain4', 'nested_ore', 'impi_ore', 'negi_ande_hyp', 'ori_ore')


def label(rec, pattern, bound, tl, classical_only=False):
    t0 = time.time()
    u = minlen(rec['prompt'], bound, tl)
    t1 = time.time()
    if FORBID[pattern] is None:      # no restriction exists: "uses" only
        r = {'min_lines_ub': None, 'proof': None, 'timeout': False, 'calls': 0, 'no_restriction': True}
    else:
        r = minlen(rec['prompt'], bound, tl, forbid=FORBID[pattern], max_depth=MAXDEPTH.get(pattern))
    t2 = time.time()
    out = {k: rec[k] for k in ('name', 'thm', 'key', 'prompt', 'schema', 'source', 'n_prem') if k in rec}
    out['gen_lines'] = rec.get('n_lines', rec.get('gen_lines'))
    out.update({'pattern': pattern, 'bound': bound, 'min_lines_ub': u['min_lines_ub'], 'proof': u['proof'], 'timeout': u['timeout'],
                'r_min_lines_ub': r['min_lines_ub'], 'r_proof': r['proof'], 'r_timeout': r['timeout'],
                'secs_u': t1 - t0, 'secs_r': t2 - t1, 'calls_u': u['calls'], 'calls_r': r['calls']})
    reach = u['min_lines_ub'] is not None
    rfail = r['min_lines_ub'] is None and not r['timeout'] and not r.get('no_restriction')
    out['requires'] = bool(reach and rfail)
    out['no_restriction'] = bool(r.get('no_restriction'))
    out['classical_only'] = bool(classical_only)
    out['requires_wide'] = bool(out['requires'] or (classical_only and rfail and pattern == 'reductio'))
    cl = pats(u['proof'], pattern)
    out['proof_pat'] = cl
    out['uses'] = bool(cl and cl[pattern])          # the shortest found proof contains the pattern
    out['oracle_ok'] = (not out['requires']) or (cl is not None and bool(cl[pattern]))
    if r['proof']:
        clr = pats(r['proof'], pattern)
        out['r_proof_pat'] = clr
        # the restricted proof must NOT contain the pattern (a check on the restriction itself)
        out['restriction_ok'] = clr is not None and not clr[pattern]
    return out


def _work(args):
    rec, pattern, bound, tl, co = args
    return label(rec, pattern, bound, tl, co)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--pattern', required=True, choices=list(FORBID))
    ap.add_argument('--bound', type=int, default=10); ap.add_argument('--time', type=float, default=60.0)
    ap.add_argument('--procs', type=int, default=6); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--classical_only', default=None, help="jsonl with fields name + classical_only (intuit.py output), or 'all' (every candidate is classical-only by construction)")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    co = {}
    if a.classical_only == 'all':
        co = {r['name']: True for r in recs}
    elif a.classical_only:
        co = {json.loads(l)['name']: json.loads(l)['classical_only'] for l in open(a.classical_only) if l.strip()}
    done = set()
    if os.path.exists(a.out):     # resumable
        done = {json.loads(l)['name'] for l in open(a.out) if l.strip()}
    todo = [r for r in recs if r['name'] not in done]
    print(f'{len(recs)} candidates, {len(done)} already labelled, {len(todo)} to do; pattern {a.pattern} forbid {FORBID[a.pattern]} bound {a.bound} time {a.time}', flush=True)
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(a.procs) as pool, open(a.out, 'a') as fo:
        for i, r in enumerate(pool.imap_unordered(_work, [(x, a.pattern, a.bound, a.time, co.get(x['name'], False)) for x in todo], chunksize=2)):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            if (i + 1) % 100 == 0:
                print(f'{i+1}/{len(todo)} {time.time()-t0:.0f}s', flush=True)
    rs = [json.loads(l) for l in open(a.out) if l.strip()]
    c = collections.Counter()
    for r in rs:
        c['n'] += 1; c['reach'] += r['min_lines_ub'] is not None; c['u_timeout'] += r['timeout']; c['r_timeout'] += r['r_timeout']
        c['requires'] += r['requires']; c['requires_wide'] += r['requires_wide']; c['oracle_bad'] += not r['oracle_ok']; c['uses'] += r.get('uses', False)
        c['restriction_bad'] += not r.get('restriction_ok', True)
    print(json.dumps(c))
    print('min_lines_ub (unrestricted):', dict(sorted(collections.Counter(str(r['min_lines_ub']) for r in rs).items())))
    print('required, by min_lines_ub:', dict(sorted(collections.Counter(r['min_lines_ub'] for r in rs if r['requires']).items())))
    print('reachable but NOT required, by restricted length:', dict(sorted(collections.Counter(str(r['r_min_lines_ub']) for r in rs if r['min_lines_ub'] is not None and not r['requires']).items())))


if __name__ == '__main__':
    main()
