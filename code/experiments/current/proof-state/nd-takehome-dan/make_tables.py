#!/usr/bin/env python3
"""Markdown tables for the writeup from artifacts/*.json. Prints to stdout.
  python make_tables.py --arms ei_abs_s0 frozen_abs_s0 ei_abs_long_s0 ei_abs_s1 frozen_abs_s1
"""
import argparse, json, glob, os, collections


def rounds(arm):
    fs = sorted(glob.glob(f'artifacts/{arm}/round_*.json'), key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return [json.load(open(f)) for f in fs]


def pct(s):
    return f"{100*s['rate']:.1f}% [{100*s['ci'][0]:.1f},{100*s['ci'][1]:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    a = ap.parse_args()
    for arm in a.arms:
        R = rounds(arm)
        if not R:
            continue
        k = R[0]['k']
        print(f'\n### {arm}  (k={k}, T={R[0]["temperature"]})\n')
        print('| round | attempts/thm | targets cum | transfer cum | transfer this round | transfer greedy | held-out greedy | new proofs | transfer written ≥7 / ≥8 / ≥9 (cum) | frontier written / pruned (transfer) | frontier written / pruned (targets) |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in R:
            wh = r['transfer_cum']['written_hist']
            ge = lambda n: sum(v for kk, v in wh.items() if int(kk) >= n)
            print(f"| {r['round']} | {r['round']*k} | {pct(r['targets_cum'])} | {pct(r['transfer_cum'])} | {pct(r['transfer_round'])} | {pct(r['transfer_greedy'])} | {pct(r['heldout_greedy'])} | {r.get('new_proofs_this_round','')} | {ge(7)} / {ge(8)} / {ge(9)} | {r['transfer_cum']['frontier_written']} / {r['transfer_cum']['frontier_pruned']} | {r['targets_cum']['frontier_written']} / {r['targets_cum']['frontier_pruned']} |")
        last = R[-1]
        print('\nper-length transfer (cumulative, final round):')
        print('| gen length | ' + ' | '.join(sorted(last['transfer_cum']['by_len'], key=int)) + ' |')
        print('|---|' + '---|' * len(last['transfer_cum']['by_len']))
        print('| solved | ' + ' | '.join(f"{v['solved']}/{v['n']}" for kk, v in sorted(last['transfer_cum']['by_len'].items(), key=lambda x: int(x[0]))) + ' |')
        print('\nwritten hist (transfer cum, final):', last['transfer_cum']['written_hist'])
        print('pruned  hist (transfer cum, final):', last['transfer_cum']['pruned_hist'])
        print('written hist (targets cum, final):', last['targets_cum']['written_hist'])
        print('pruned  hist (targets cum, final):', last['targets_cum']['pruned_hist'])
        print('greedy transfer written hist (final):', last['transfer_greedy']['written_hist'])
        print('greedy reasons (final):', last['transfer_greedy']['reasons'])


if __name__ == '__main__':
    main()
