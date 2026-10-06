#!/usr/bin/env python3
"""Inspect verified proofs found by an arm: padding check (written vs pruned length, reiteration count),
per-length counts, and pretty-printed examples of the longest proofs.
  python analyze_found.py artifacts/ei_abs_s0/found_transfer_8.jsonl [--n 5]
"""
import json, sys, collections, argparse


def pretty(prompt, proof):
    print('  ' + prompt)
    for ln in proof.replace(' ;', ' ;\n').split('\n'):
        if ln.strip():
            print('    ' + ln.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fn')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--minlen', type=int, default=0)
    a = ap.parse_args()
    rs = [json.loads(l) for l in open(a.fn)]
    print('proofs', len(rs), 'theorems', len({r['name'] for r in rs}))
    wh = collections.Counter(r['written'] for r in rs); ph = collections.Counter(r['pruned'] for r in rs)
    print('written hist', dict(sorted(wh.items()))); print('pruned  hist', dict(sorted(ph.items())))
    pad = [r for r in rs if r['written'] > r['pruned']]
    print(f'padded (written>pruned): {len(pad)}/{len(rs)} = {len(pad)/max(1,len(rs)):.3f}; mean excess {sum(r["written"]-r["pruned"] for r in pad)/max(1,len(pad)):.2f}')
    reit = sum(r['proof'].count(': R ') for r in rs)
    print('reiteration lines total', reit, 'per proof', round(reit / max(1, len(rs)), 3))
    byround = collections.Counter(r.get('round') for r in rs)
    print('by round found', dict(sorted(byround.items())))
    # longest by pruned length
    rs2 = sorted([r for r in rs if r['pruned'] >= a.minlen], key=lambda r: (-r['pruned'], -r['written']))
    for r in rs2[:a.n]:
        print(f'\n[{r["name"]}] gen_lines={r.get("gen_lines")} written={r["written"]} pruned={r["pruned"]} round={r.get("round")}')
        pretty(r['prompt'], r['proof'])


if __name__ == '__main__':
    main()
