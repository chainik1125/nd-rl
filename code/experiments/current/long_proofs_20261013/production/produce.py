"""Overnight producer: sample theorems (approach D's tree sampler), write a proof, label the minimal length.

One process = one seed = one output file; run several with different seeds and merge (dedupe by `key`).

    python produce.py OUT.jsonl SEED [--floor 16] [--secs 60] [--memo-cap 8000000] [--atoms 4-7] [--prems 4-10]

Every record passed certify.check (verified proof, consistent premises, all cited, no duplicates). Label from
minlen_sem (approach D: minlen plus truth-table pruning of states A |/= G; sound, same search space):
  min_body = exact minimal body length (< floor),  cert_ge = floor (search exhausted, nothing shorter),
  or neither if the search hit the time limit or memo cap (`timeout`, kept so we can retry with a better certifier).
The record's own `text` proof is the generated one (prover.py), not the minimal one; `min_proof` holds minlen's.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_tree as ST  # noqa: E402  (imports certify as C, sets up the widened atom set)
import prover  # noqa: E402
from minlen_sem import SemSearch, Timeout, parse_thm, verify_text  # noqa: E402

C = ST.C


class CappedSearch(SemSearch):
    memo_cap = 8_000_000

    def best(self, G, A, b, d=0, H=frozenset()):
        if (self.calls & 1023) == 0 and len(self.memo) > self.memo_cap:
            raise Timeout()
        return super().best(G, A, b, d, H)


def label(prompt, n_prem, floor, secs, memo_cap):
    prem, concl = parse_thm(prompt)
    t0 = time.time()
    s = CappedSearch(prem, concl, deadline=t0 + secs)
    s.memo_cap = memo_cap
    out = {'floor': floor, 'min_body': None, 'cert_ge': None, 'timeout': False, 'certifier': 'minlen_sem'}
    try:
        r = s.run(n_prem + floor - 1)
    except Timeout:
        out['timeout'] = True
        r = None
    out['cert_secs'] = round(time.time() - t0, 2)
    out['memo'] = len(s.memo)
    if r is not None:
        n, text = r
        ok, reason, nl = verify_text(prompt + ' ' + text)
        if ok and nl == n:
            out['min_body'] = n - n_prem
            out['min_proof'] = text
        else:
            out['error'] = f'{reason} nl={nl} n={n}'
    elif not out['timeout']:
        out['cert_ge'] = floor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('seed', type=int)
    ap.add_argument('--floor', type=int, default=16)
    ap.add_argument('--secs', type=float, default=60.0)
    ap.add_argument('--memo-cap', type=int, default=8_000_000)
    ap.add_argument('--atoms', default='4-7')
    ap.add_argument('--prems', default='4-10')
    ap.add_argument('--hours', type=float, default=12.0)
    a = ap.parse_args()
    ar = tuple(int(x) for x in a.atoms.split('-'))
    pr = tuple(int(x) for x in a.prems.split('-'))
    rng = random.Random(a.seed)
    seen = set()
    stats = {'draws': 0, 'kept': 0, 'cert': 0, 'timeout': 0, 'rejected': 0}
    t_end = time.time() + a.hours * 3600
    with open(a.out, 'a') as f:
        while time.time() < t_end:
            stats['draws'] += 1
            s = ST.sample(rng, ar, pr)
            if s is None:
                continue
            key = C.theorem_key(*s)
            if key in seen:
                continue
            seen.add(key)
            prompt = ST.SC.prompt_of(*s)
            body = prover.prove_text(prompt)
            c = C.check(prompt + ' ' + body) if body else None
            if c is None:
                stats['rejected'] += 1
                continue
            lab = label(c['prompt'], c['n_prem'], a.floor, a.secs, a.memo_cap)
            stats['kept'] += 1
            stats['cert'] += bool(lab['cert_ge'])
            stats['timeout'] += lab['timeout']
            f.write(json.dumps({**c, **lab, 'seed': a.seed, 'gen': 'tree'}) + '\n')
            f.flush()
            if stats['kept'] % 50 == 0:
                print(json.dumps({**stats, 't': round(time.time())}), flush=True)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
