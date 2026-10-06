#!/usr/bin/env python3
"""figures/state_env.png for run `state-env`: transfer solved by L_true, arm S vs the whole-proof control C0,
at ladder rung T1 and frozen, plus the acquisition curve.  Reads artifacts/se/summary.json.

  python3 se_figures.py --summary artifacts/se/summary.json --out figures/state_env.png
"""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = '#1f6feb'      # arm S (state)
C = '#8b949e'      # C0 (whole proof)
SH = '#d29922'
SN = '#2da44e'


def bins(o, lo=7, hi=14):
    bb = o['derived']['by_bin']
    return [int(bb.get(str(L), {'solved': 0})['solved']) for L in range(lo, hi + 1)], \
           [int(bb.get(str(L), {'n': 0})['n']) for L in range(lo, hi + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='artifacts/se/summary.json')
    ap.add_argument('--out', default='figures/state_env.png')
    a = ap.parse_args()
    d = json.load(open(a.summary))
    d = d.get('ladders', d)          # artifacts/se/summary.json nests the ladder summary
    R, C0 = d['runs'], d['c0']
    Ls = list(range(7, 15))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for k, (rung, title) in enumerate((('T1', 'ladder T1 (8 rounds x k 32)'), ('frozen', 'frozen, equal attempts'))):
        A = ax[k]
        w = 0.16
        for j, (grp, pref, col, lab) in enumerate(((C0, f'la_{rung}_c0_s', C, 'C0 whole proof'),
                                                  (R, f'la_{rung}_S_s', S, 'S state'),
                                                  (R, f'la_{rung}_SH_s', SH, 'SH state+history'),
                                                  (R, f'la_{rung}_SN_s', SN, 'SN-v2 state, env names'))):
            for s in (0, 1):
                key = f'{pref}{s}'
                if key not in grp:
                    continue
                sol, tot = bins(grp[key])
                rate = [100.0 * x / max(1, n) for x, n in zip(sol, tot)]
                A.bar([x + (j - 1.5) * w * 1.25 + (s - 0.5) * w * 0.6 for x in range(len(Ls))], rate, w * 0.55,
                      color=col, alpha=1.0 if s == 0 else 0.55,
                      label=(lab if s == 0 else None), edgecolor='none')
        A.set_xticks(range(len(Ls)))
        A.set_xticklabels([f'{L}\nn={t}' for L, t in zip(Ls, bins(list(C0.values())[0], )[1])], fontsize=7)
        A.set_xlabel('$L_{true}$ (ND-derived upper bound under Lean)')
        A.set_ylabel('transfer theorems solved (%)')
        A.set_title(title, fontsize=10)
        A.axvline(5.5, color='#cf222e', ls=':', lw=1)
        A.text(5.6, A.get_ylim()[1] * 0.45, '$L_{true}\\geq13$\n0 in 11\nwhole-proof\nruns', fontsize=7, color='#cf222e')
        A.legend(fontsize=7, frameon=False, loc='upper right')
        A.spines[['top', 'right']].set_visible(False)
    A = ax[2]
    for grp, pref, col, lab in ((C0, 'la_T1_c0_s', C, 'C0 T1'), (R, 'la_T1_S_s', S, 'S T1'),
                                (R, 'la_T1_SH_s', SH, 'SH T1'), (R, 'la_T1_SN_s', SN, 'SN-v2 T1')):
        for s in (0, 1):
            key = f'{pref}{s}'
            if key not in grp:
                continue
            o = grp[key]
            rs = sorted(int(x) for x in o['rounds'])
            y = [o['rounds'][str(r)]['transfer_cum_reported']['solved'] for r in rs]
            A.plot(rs, y, color=col, alpha=1.0 if s == 0 else 0.55, marker='o', ms=3,
                   label=f'{lab} s{s}')
    A.set_xlabel('EI round'); A.set_ylabel('transfer theorems solved (cumulative, of 2,285)')
    A.set_title('acquisition (T1)', fontsize=10); A.legend(fontsize=7, frameon=False, ncol=2)
    A.spines[['top', 'right']].set_visible(False)
    fig.suptitle('state-env: transfer theorems solved by $L_{true}$ -- the step environment with the proof state (S, SH, SN-v2) vs the whole-proof control C0 (on file; Lean AND nd_verify). Two seeds each (s1 lighter). Lean alone decides.', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
