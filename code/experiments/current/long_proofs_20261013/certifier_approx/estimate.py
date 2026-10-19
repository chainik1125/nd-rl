"""Approximate certifier: `estimate(prompt, n_prem, floor, secs)`.

Two halves, bracketing the minimal body length m:
  A. exact half (lower bound within minlen's space S): minlen's DP with semantic pruning, iterative deepening,
     truncated by a memo cap (memory) and a time share. Exhausting level E means m > E (in S). Reaching
     floor-1 is a full minlen-equivalent certificate.
  B. hunter half (upper bound): cheap INCOMPLETE searches (no lemmas / literal lemmas / subformula lemmas /
     randomised capped full search) deepened from E+1 to floor-1. Any proof found is verified with nd_verify
     and refutes the claim; it is a true upper bound on m.
Claim ">= floor" iff A did not find a proof, and B found none. p_false = the hunters' measured miss rate
(calib.json, from calibrate.py + analyze.py) maximised over the uncertified levels E+1..floor-1; levels beyond
the calibrated range reuse the last calibrated level and set `extrapolated: True`.

CLI:  python estimate.py IN.jsonl OUT.jsonl --floor 20 --secs 30 --procs 2
      input records need `prompt` and `n_prem` (certify.py-style records work).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hunter import Hunter, parse_thm  # noqa: E402

# Pipeline, cheapest refuters first; a verified find stops it. (name, kwargs, share of `secs`)
#   none_d2 : no lemmas, boxes <= 2 deep. Finds a proof < floor fast when one exists, but non-minimal (excess >= 1 in 78%).
#   none    : no lemmas (tree proofs). Never missed in calibration; excess over the exact minimum <= 3, >= 1 in 6.6%.
#   exact   : minlen's full space with semantic pruning, truncated by memo cap + time. Exhausting level E is a
#             minlen-equivalent certificate that m > E (within S).
#   lit/sub : lemmas restricted to literals / subformulas and their single negations. `sub` found the exact minimum in
#             every calibrated run that reached the level; its only misses were time-outs.
STAGES = [
    ('none_d2', dict(lem='none', max_depth=2, memo_cap=1_000_000), 0.05),
    ('none', dict(lem='none', memo_cap=10_000_000), 0.35),
    ('exact', dict(lem='all', memo_cap=3_000_000), 0.25),
    ('lit', dict(lem='lit', memo_cap=8_000_000), 0.15),
    ('sub', dict(lem='sub', memo_cap=6_000_000), 0.20),
]   # memo entries cost ~120 B each (measured: 20M entries = 2.4 GB RSS); stages run one at a time
RELIABLE = ('none', 'lit', 'sub')     # hunters whose per-slack miss rates are calibrated (calib.json)

_CALIB = None


def calib():
    global _CALIB
    if _CALIB is None:
        p = os.path.join(HERE, 'calib.json')
        _CALIB = json.load(open(p)) if os.path.exists(p) else {}
    return _CALIB


def p_false_for(E, floor, reached):
    """reached: {hunter: highest body level it exhausted}. For every uncertified level m in (E, floor) take the
    smallest calibrated miss bound among the reliable hunters that reached m, where miss bound = Wilson-95% upper
    bound of P(hunter finds no proof < floor | true min = floor - slack) from calibrate.py; p_false = max over m.
    Returns (p_false, covered_to): covered_to < floor - 1 means some level was reached by no reliable hunter and
    the claim at `floor` is NOT supported (p_false then refers to the claim '>= covered_to + 1')."""
    tables = calib().get('miss_ub_slack_by_hunter', {})
    covered_to = E
    worst = 0.0
    for m in range(E + 1, floor):
        best = None
        for h in RELIABLE:
            if reached.get(h, 0) >= m and h in tables:
                g = str(min(8, floor - m))
                v = tables[h].get(g)
                if v is not None and (best is None or v < best):
                    best = v
        if best is None:
            break
        covered_to = m
        worst = max(worst, best)
    return (worst if tables else None), covered_to


def estimate(prompt, n_prem, floor=20, secs=30.0):
    """-> dict(claim_ge, upper, p_false, secs, ...). claim_ge = floor when no stage found a proof < floor and the
    reliable hunters covered every level up to floor-1; else None. `covered_ge` is the floor the run does support
    (exact_exh + 1 when nothing reached further). `upper` = body length of a verified proof found (None if none);
    `exact_min` is set when the exact half found it (then upper == exact_min)."""
    t0 = time.time()
    prem, concl = parse_thm(prompt)
    hi = floor - 1
    out = {'claim_ge': None, 'upper': None, 'p_false': None, 'exact_exh': 0, 'proof': None, 'found_by': None,
           'stages': [], 'floor': floor}
    reached = {}
    for name, kw, share in STAGES:
        lo = out['exact_exh'] + 1 if name != 'exact' else 1
        h = Hunter(prem, concl, deadline=time.time() + share * secs, **kw)
        out.setdefault('S', len(h.S))
        r = h.levels(lo, hi, prompt)
        del h
        out['stages'].append((name, r['stop'], r['exh'], r['secs'], r['memo']))
        if name == 'exact':
            out['exact_exh'] = r['exh']
        else:
            reached[name] = r['exh']
        if r['found'] is not None:
            out.update(upper=r['found'], proof=r['proof'], found_by=name, p_false=0.0)
            if name == 'exact':
                out['exact_min'] = r['found']
            break
        if name == 'exact' and r['stop'] == 'exhausted':
            out.update(claim_ge=floor, p_false=0.0, exact_cert=True, covered_ge=floor)
            break
    if out['upper'] is None and not out.get('exact_cert'):
        pf, covered_to = p_false_for(out['exact_exh'], floor, reached)
        out.update(p_false=pf, covered_ge=covered_to + 1, reached=reached)
        if covered_to >= hi:
            out['claim_ge'] = floor
    out['secs'] = round(time.time() - t0, 2)
    return out


def _work(args):
    rec, floor, secs = args
    r = estimate(rec['prompt'], rec['n_prem'], floor, secs)
    keep = {k: rec[k] for k in ('key', 'prompt', 'n_prem', 'text', 'n_body', 'min_body', 'cert_ge') if k in rec}
    r.pop('proof_lines', None)
    return {**keep, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--floor', type=int, default=20)
    ap.add_argument('--secs', type=float, default=30.0)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    recs = [json.loads(line) for line in open(a.src)]
    if a.limit:
        recs = recs[:a.limit]
    t0 = time.time()
    n_claim = 0
    with ProcessPoolExecutor(a.procs) as ex, open(a.dst, 'w') as fo:
        for i, r in enumerate(ex.map(_work, [(r, a.floor, a.secs) for r in recs], chunksize=1)):
            n_claim += r['claim_ge'] is not None
            fo.write(json.dumps(r) + '\n')
            fo.flush()
            if (i + 1) % 10 == 0:
                print(f'{i+1}/{len(recs)} claims {n_claim} {time.time()-t0:.0f}s', flush=True)
    el = time.time() - t0
    print(f'{len(recs)} theorems, {n_claim} claimed >= {a.floor}, {el:.0f}s wall with {a.procs} procs '
          f'= {len(recs) / (el * a.procs / 3600):.0f} theorems per core-hour')


if __name__ == '__main__':
    main()
