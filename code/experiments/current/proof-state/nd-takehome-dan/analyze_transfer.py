#!/usr/bin/env python3
"""Where does the model stop? Solve rate on the transfer set broken down by properties of the theorem's
generating proof (length, box depth, rules, premises), for one or more arms' cumulative found files.
  python analyze_transfer.py --found artifacts/ei_abs_s0/found_transfer_8.jsonl artifacts/frozen_abs_s0/found_transfer_8.jsonl
"""
import argparse, json, collections


def depth_of(proof):
    return max((ln.count('|') for ln in proof.split(' ; ')), default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--found', nargs='+', required=True)
    ap.add_argument('--transfer', default='data/transfer.jsonl')
    a = ap.parse_args()
    T = [json.loads(l) for l in open(a.transfer)]
    for t in T:
        t['depth'] = depth_of(t['gen_proof'])
        t['has_ORE'] = 'ORE' in t['rules']; t['has_DN'] = 'DN' in t['rules']; t['has_NEGI'] = 'NEGI' in t['rules']
        t['n_boxes'] = t['gen_proof'].count(': AS ;')
    solved = {}
    for fn in a.found:
        s = collections.defaultdict(list)
        for l in open(fn):
            r = json.loads(l); s[r['name']].append(r)
        solved[fn] = s

    def table(key, label):
        groups = sorted({t[key] for t in T}, key=lambda x: (str(type(x)), x))
        print(f'\n| {label} | n | ' + ' | '.join(fn.split('/')[1] for fn in a.found) + ' | min written len of found proofs (first arm) |')
        print('|---|---|' + '---|' * (len(a.found) + 1))
        for g in groups:
            ts = [t for t in T if t[key] == g]
            cells = []
            for fn in a.found:
                k = sum(1 for t in ts if t['name'] in solved[fn]); cells.append(f'{100*k/len(ts):.0f}% ({k}/{len(ts)})')
            fn0 = a.found[0]
            mins = [min(x['written'] for x in solved[fn0][t['name']]) for t in ts if t['name'] in solved[fn0]]
            mh = collections.Counter(mins)
            print(f'| {g} | {len(ts)} | ' + ' | '.join(cells) + f' | {dict(sorted(mh.items()))} |')
    table('n_lines', 'generating length')
    table('depth', 'max box depth of generating proof')
    table('n_boxes', 'number of boxes (AS lines) in generating proof')
    table('n_prem', 'premises')
    table('has_ORE', 'generating proof uses ORE')
    table('has_DN', 'generating proof uses DN')
    table('has_NEGI', 'generating proof uses NEGI')


if __name__ == '__main__':
    main()
