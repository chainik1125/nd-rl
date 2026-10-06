#!/usr/bin/env python3
"""Headline stopping numbers: choose the operating point on DEV, apply it once to TEST.

For each adaptive policy (scorer, tau sweep) and each target mean-attempt budget m in {2, 4, 8}:
  dev:  over N_max in {8, 16} (the caps every dev pool supports) and the shared dev tau grid, take for each N the tau
        whose dev mean attempts (theorem-weighted over all dev pools) is nearest m, then keep the N with the higher dev
        accuracy.  The completion-stop policy ("stop at first finished") has no tau: choose N_max on dev the same way.
  test: apply exactly that (N_max, tau); report realised mean attempts, accuracy, mean tokens; compare (paired theorem
        bootstrap, B=1000) with the fixed-k mixture of the SAME selector (a) at the dev target m (weights frozen on dev)
        and (b) at the realised test mean attempts; and with fixed-k random pick at the realised mean.
Also reports the curve reading (nearest tau on the test curve itself, flagged as not dev-selected).

Usage: stopping_headline.py --dev <stopping dir prefix list> --test <stem> ... --out results/stopping_headline.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import fixed_at, RUNS

TARGETS = (2, 4, 8)
DEV_NS = (8, 16)


def load(prefix, scorer):
    p = f'{prefix}__{scorer}.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def boot(acc_t, att_t, fixed_same, fixed_rand, m_frozen, B=1000, seed=0):
    nT = len(acc_t)
    rng = np.random.default_rng(seed)
    def stats(ix):
        m = att_t[ix].mean()
        a = acc_t[ix].mean()
        return np.array([a - fixed_at(fixed_same[:, ix], m).mean(), a - fixed_at(fixed_same[:, ix], m_frozen).mean(),
                         a - fixed_at(fixed_rand[:, ix], m).mean(), m])
    d0 = stats(np.arange(nT))
    bs = np.array([stats(rng.integers(0, nT, nT)) for _ in range(B)])
    keys = ['vs_same_at_realised_m', 'vs_same_at_dev_target_m', 'vs_random_at_realised_m', 'realised_m']
    return {k: dict(est=float(d0[i]), lo=float(np.quantile(bs[:, i], 0.025)), hi=float(np.quantile(bs[:, i], 0.975)),
                    p_le0=float((bs[:, i] <= 0).mean())) for i, k in enumerate(keys)}


def boot_vs_ff(acc_t, att_t, ff, B=1000, seed=1):
    """paired theorem bootstrap of the adaptive policy vs stop-at-first-finished: (a) at first-finished's ceiling
    (N_max = K), (b) at the first-finished N_max whose mean attempts is nearest the adaptive policy's realised mean."""
    natt = ff['nsweep_att'].mean(1)
    g = int(np.argmin(np.abs(natt - att_t.mean())))
    out = {}
    rng = np.random.default_rng(seed)
    nT = len(acc_t)
    for lab, idx in (('vs_first_finished_ceiling', len(natt) - 1), ('vs_first_finished_nearest_att', g)):
        fa = ff['nsweep_acc'][idx]
        d = acc_t - fa
        bs = np.array([d[rng.integers(0, nT, nT)].mean() for _ in range(B)])
        out[lab] = dict(ff_N=idx + 1, ff_att=float(ff['nsweep_att'][idx].mean()), ff_acc=float(fa.mean()),
                        est=float(d.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                        p_le0=float((bs <= 0).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', nargs='+', required=True, help='npz prefixes, e.g. ~/runs/proof-selection/stopping/<stem>__dev')
    ap.add_argument('--test', nargs='+', required=True, help='npz prefixes of test pools')
    ap.add_argument('--scorers', required=True, help='comma list of tau-swept scorers (completion handled separately)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = dict(dev=a.dev, test=a.test, choices={}, test_results={})
    for s in ['completion'] + a.scorers.split(','):
        devs = [load(p, s) for p in a.dev]
        if any(d is None for d in devs):
            print('skip', s, 'missing dev'); continue
        out['choices'][s] = {}
        for m in TARGETS:
            best = None
            for N in DEV_NS:
                if s == 'completion':     # N sweep at tau = 1: choose N_max in 1..min K with dev mean attempts nearest m
                    Ks = min(d['nsweep_att'].shape[0] for d in devs)
                    atts = np.array([np.concatenate([d['nsweep_att'][n] for d in devs]).mean() for n in range(Ks)])
                    accs = np.array([np.concatenate([d['nsweep_acc'][n] for d in devs]).mean() for n in range(Ks)])
                    g = int(np.argmin(np.abs(atts - m)))
                    best = dict(N=g + 1, tau=0.5, dev_att=float(atts[g]), dev_acc=float(accs[g]))
                    break
                ni = [list(d['Ns']).index(N) for d in devs]
                taus = devs[0]['taus']
                assert all(np.array_equal(d['taus'], taus) for d in devs), 'dev pools must share the tau grid'
                atts = np.concatenate([d['adaptive_att'][i] for d, i in zip(devs, ni)], -1).mean(-1)
                accs = np.concatenate([d['adaptive_acc'][i] for d, i in zip(devs, ni)], -1).mean(-1)
                g = int(np.argmin(np.abs(atts - m)))
                c = dict(N=N, tau=float(taus[g]), dev_att=float(atts[g]), dev_acc=float(accs[g]))
                if best is None or abs(c['dev_att'] - m) < 0.25 * m and c['dev_acc'] > best['dev_acc'] or \
                        abs(best['dev_att'] - m) >= 0.25 * m and abs(c['dev_att'] - m) < abs(best['dev_att'] - m):
                    best = c
            out['choices'][s][str(m)] = best
        for p in a.test:
            t = load(p, s)
            tr = load(p, 'random')
            if t is None:
                continue
            res = out['test_results'].setdefault(os.path.basename(p), {}).setdefault(s, {})
            for m in TARGETS:
                c = out['choices'][s][str(m)]
                if s == 'completion':
                    acc_t, att_t = t['nsweep_acc'][c['N'] - 1], t['nsweep_att'][c['N'] - 1]
                    gen_t, tot_t = t['nsweep_gen'][c['N'] - 1], t['nsweep_tot'][c['N'] - 1]
                    disp_t = t['nsweep_disp'][c['N'] - 1]
                else:
                    ni = list(t['Ns']).index(c['N'])
                    g = int(np.where(t['taus'] == c['tau'])[0][0])
                    acc_t, att_t = t['adaptive_acc'][ni, g], t['adaptive_att'][ni, g]
                    gen_t, tot_t, disp_t = t['adaptive_gen'][ni, g], t['adaptive_tot'][ni, g], t['adaptive_disp'][ni, g]
                b = boot(acc_t, att_t, t['fixed_acc'], tr['fixed_acc'], m)
                b.update(boot_vs_ff(acc_t, att_t, load(p, 'completion')))
                mt = float(att_t.mean())
                res[str(m)] = dict(choice=c, test_att=mt, test_acc=float(acc_t.mean()), test_gen=float(gen_t.mean()),
                                   test_tot=float(tot_t.mean()), displaced=float(disp_t.mean()),
                                   fixed_same_at_realised=float(fixed_at(t['fixed_acc'], mt).mean()),
                                   fixed_same_at_target=float(fixed_at(t['fixed_acc'], m).mean()),
                                   fixed_random_at_realised=float(fixed_at(tr['fixed_acc'], mt).mean()),
                                   oracle_passk_at_realised=float(fixed_at(load(p, 'oracle')['fixed_acc'], mt).mean()),
                                   bootstrap=b)
    json.dump(out, open(a.out, 'w'), indent=1)
    for pn, d in out['test_results'].items():
        print(pn)
        for s, r in d.items():
            for m, x in r.items():
                b = x['bootstrap']['vs_same_at_realised_m']
                f = x['bootstrap']['vs_first_finished_nearest_att']; fc = x['bootstrap']['vs_first_finished_ceiling']
                print(f'  {s:20s} m={m}: N={x["choice"]["N"]:2d} att {x["test_att"]:5.2f} acc {x["test_acc"]:.3f} disp {x["displaced"]:.3f} '
                      f'| fixed-same {x["fixed_same_at_realised"]:.3f} oracle {x["oracle_passk_at_realised"]:.3f} '
                      f'| vs fixed-same {b["est"]:+.3f} [{b["lo"]:+.3f},{b["hi"]:+.3f}] '
                      f'| vs FF@{f["ff_att"]:.2f} {f["est"]:+.3f} [{f["lo"]:+.3f},{f["hi"]:+.3f}] '
                      f'| vs FF ceil@{fc["ff_att"]:.2f} {fc["est"]:+.3f} [{fc["lo"]:+.3f},{fc["hi"]:+.3f}]')


if __name__ == '__main__':
    main()
