#!/usr/bin/env python3
"""Per-round cumulative RL / pretraining FLOPs and sigma capability for create-or-elicit's ladders.
Read-only. Reuses the rl-compute-share reviewer's FLOP functions (review_rcs/ourruns.py)."""
import glob, json, re, sys
sys.path.insert(0, '/home/dan/work/rl-compute-share/review_rcs')
import ourruns as O

W = '/home/dan/work'
COE = f'{W}/create-or-elicit/artifacts/coe/ladder'
ROUNDS = f'{W}/create-or-elicit/artifacts/coe/rounds'
regs = {r: O.load_reg(r) for r in ('create-or-elicit', 'trajectory', 'rl-continue')}
s1 = {**O.stage1_rows(regs['create-or-elicit']), **O.stage1_rows(regs['trajectory'])}

specs = []
for arm, seeds in (('ko', range(6)), ('full', range(3, 6)), ('dose1', range(3))):
    for sd in seeds:
        lad = f'la_{arm}_s{sd}'
        specs.append(dict(name=lad, arm={'ko': 'knockout', 'full': 'full', 'dose1': 'dose1'}[arm], seed=sd,
                          parts=[('create-or-elicit', [COE], None)], ladder=lad,
                          D=s1[f'ckpts/coe/stage1_{arm}_s{sd}.pt']['train_tokens'], rounds_file=f'{ROUNDS}/{lad}.json'))
for sd in range(3):
    lad = f'la_T1_best12_s{sd}'
    specs.append(dict(name=f'old_full_s{sd}', arm='full', seed=sd, ladder=lad,
                      parts=[('trajectory', [f'{W}/trajectory/artifacts/tj'], set(range(1, 9))),
                             ('rl-continue', [f'{W}/rl-continue/artifacts/rc'], set(range(9, 17)))],
                      D=s1[f'ckpts/tj/stage1_best12_s{sd}_b1200.pt']['train_tokens'], rounds_file=f'{ROUNDS}/old_full_s{sd}.json'))

def last(v):
    return (v[-1] if v else 0) or 0

out = []
for sp in specs:
    samp, ev, ft, pt, frl = {}, {}, {}, {}, {}
    for reg, dirs, rr in sp['parts']:
        per = O.ladder_rows(regs[reg], sp['ladder'])
        p, mix = O.round_files(dirs, sp['ladder'])
        for (ph, rd), mets in per.items():
            if rd is None or (rr is not None and rd not in rr):
                continue
            if ph == 'sample': samp[rd] = last(mets.get('gen_tokens'))
            elif ph == 'eval': ev[rd] = last(mets.get('gen_tokens'))
            elif ph == 'finetune': ft[rd] = last(mets.get('train_tokens'))
        for rd, v in p.items():
            if rr is None or rd in rr: pt[rd] = v or 0
        for rd, (n_rl, n_inj) in mix.items():
            if (rr is None or rd in rr) and n_rl is not None: frl[rd] = n_rl / (n_rl + 20000 + (n_inj or 0))
    R = json.load(open(sp['rounds_file']))
    rows, cum_tr, cum_only, missing = [], 0.0, 0.0, []
    D6 = 6 * sp['D']
    for r in sorted(int(x) for x in R['per_round']):
        g, e = samp.get(r), ev.get(r, 0)
        if g is None: missing.append(r); g = 0
        pre = pt.get(r, 0); pre_s = pre * g / (g + e) if (g + e) else 0
        cum_tr += 2 * (g + pre_s); cum_only += 2 * (g + pre_s)
        x_tr, x_only = cum_tr / D6, cum_only / D6            # after round r's sampling
        pr = R['per_round'][str(r)]
        held = sum(R['heldout_cum_by_round'].get(str(r), {}).values())
        rows.append(dict(round=r, x_train=x_tr, x_rl_only=x_only, sigma_targets_cum=pr.get('sigma_targets_cum'),
                         heldout_cum=held))
        f = ft.get(r, 0)                                      # fine-tune after round r feeds round r+1
        cum_tr += 6 * f; cum_only += 6 * f * frl.get(r, 1.0)
    out.append(dict(name=sp['name'], arm=sp['arm'], seed=sp['seed'], D=sp['D'], heldout_n=sum(R['heldout_sizes'].values()),
                    missing_sample_rounds=missing, rounds=rows, x_train_total_incl_last_ft=cum_tr / D6))
json.dump(out, open('rl_curve.json', 'w'), indent=1)
for o in out:
    a, b = o['rounds'][0], o['rounds'][-1]
    print(f"{o['name']:14s} {o['arm']:8s} D={o['D']:.2e} r{a['round']}-{b['round']} x_train r1={a['x_train']:.3f} last={b['x_train']:.2f} (+ft {o['x_train_total_incl_last_ft']:.2f}) "
          f"held {a['heldout_cum']}->{b['heldout_cum']}/{o['heldout_n']} targets {a['sigma_targets_cum']}->{b['sigma_targets_cum']} missing={o['missing_sample_rounds']}")
