#!/usr/bin/env python3
"""Phase 7 (pre-registered in logs/worker_B.md): Worker A's trained pairwise judge (judge_pair = mean masked loglik
under judge/pair_s0 and pair_s1) for count-stop selection and absolute-threshold stopping.
Usage: phase7.py dev | test"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase5 as P5
from stopping_phase2 import sig
from auroc import auroc

RUNS = P5.RUNS; RES = P5.RES; K = 16; R = 50
GENS = ('s0', 's1', 's2'); CS = (1, 2, 4, 6); M_T = (2, 3, 4, 8)


def rj(p):
    return {r['name']: r for r in map(json.loads, open(p))}


def load(stem, g, names=None, dev=False):
    pool = rj(f'{RUNS}/pools_v2/{stem}.jsonl'); lab = rj(f'{RUNS}/pools_v2/{stem}.labels.jsonl')
    sc = f'{RUNS}/scores/{stem}__'
    def jf(j):
        if j == 's0':
            return sc + ('loglik_mean_masked.jsonl' if '__s0_r16__k' in stem and not stem.endswith('k64b') else 'loglik_mean_masked_s0.jsonl')
        return sc + f'loglik_mean_masked_{j}.jsonl'
    J = {j: rj(jf(j)) for j in GENS}
    if dev:
        a, b = rj(sc + 'loglik_mean_masked_judge_pair_s0.jsonl'), rj(sc + 'loglik_mean_masked_judge_pair_s1.jsonl')
        JP = {n: (np.array(a[n]['scores']) + np.array(b[n]['scores'])) / 2 for n in a}
    else:
        jp = rj(sc + 'judge_pair.jsonl'); JP = {n: np.array(jp[n]['scores'], float) for n in jp}
    sib = [j for j in GENS if j != g]
    T = []
    for n in sorted(pool):
        if names is not None and n not in names:
            continue
        L = {j: np.array(J[j][n]['scores'], float) for j in GENS}
        T.append(dict(name=n, d=np.array([a['done'] for a in pool[n]['attempts']]), v=np.array(lab[n]['valid'], bool),
                      steps=np.array([a['steps'] for a in pool[n]['attempts']], float),
                      J={'sib': (L[sib[0]] + L[sib[1]]) / 2, 'jp': JP[n]}))
    return T


def dev_T():
    sp = json.load(open(f'{RUNS}/pools_v2/splits.json'))['dev']; names = set(n for v in sp.values() for n in v)
    return load('dev1108__s0_r16__k16', 's0', names, True) + load('rr600__s0_r16__k32', 's0', names, True)


def simulate(T, jtau, sib_tau):
    nT = len(T); res = {}
    def put(k, i, ti, acc, att, P):
        res.setdefault(k, np.zeros((P, 2, nT)))[i, :, ti] = (acc, att)
    for ti, t in enumerate(T):
        Pm = np.stack([np.random.default_rng([11, ti, r]).permutation(len(t['d'])) for r in range(R)])[:, :K]
        V = t['v'][Pm].astype(float); D = t['d'][Pm]; nf = np.cumsum(D, 1); nv = np.cumsum(V, 1)
        ev = np.maximum.accumulate(V, 1)
        first = lambda sg: np.where(sg.any(1), sg.argmax(1), K - 1)
        for N in range(1, K + 1):
            st = np.minimum(np.where((nv >= 1).any(1), (nv >= 1).argmax(1), K), N - 1)
            put('oracle', N - 1, ti, ev[np.arange(R), st].mean(), (st + 1).mean(), K)
            st = np.minimum(np.where((nf >= 1).any(1), (nf >= 1).argmax(1), K), N - 1)
            put('ff', N - 1, ti, V[np.arange(R), st].mean(), (st + 1).mean(), K)   # first finished = attempt st
        for j, taus in (('jp', jtau), ('sib', sib_tau)):
            S = np.where(t['d'], t['J'][j], -np.inf)[Pm]
            M = np.maximum.accumulate(S, 1)
            new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
            A = np.maximum.accumulate(np.where(new, np.arange(K)[None], 0), 1)
            Vs = np.take_along_axis(V, A, 1)
            g = lambda st: (Vs[np.arange(R), st].mean(), (st + 1).mean())
            for i, c in enumerate(range(1, 9)):
                put(f'count_{j}', i, ti, *g(first(nf >= c)), 8)
            tt = list(taus) + [np.inf]
            for i, tau in enumerate(tt):
                put(f'T_{j}', i, ti, *g(first(M >= tau)), len(tt))
    return res


def matched(acc_p, att_p, C):
    """count curve C (8, 2, nT) linearly mixed between adjacent c at the policy's mean attempts -> per-theorem acc."""
    catt = C[:, 1].mean(-1)
    if att_p <= catt[0]:
        return C[0, 0]
    if att_p >= catt[-1]:
        return C[-1, 0]
    i = int(np.searchsorted(catt, att_p)) - 1; w = (att_p - catt[i]) / (catt[i + 1] - catt[i])
    return (1 - w) * C[i, 0] + w * C[i + 1, 0]


def main():
    mode = sys.argv[1]
    fg5 = json.load(open(os.path.join(RUNS, 'stopping', 'phase5_fit_grids.json')))
    ch5 = json.load(open(os.path.join(RES, 'B5_dev_choice.json')))
    gf = os.path.join(RUNS, 'stopping', 'phase7_grid.json'); chf = os.path.join(RES, 'B7_dev_choice.json')
    if mode == 'dev':
        T = dev_T()
        x = np.concatenate([t['J']['jp'][t['d']] for t in T])
        jtau = sorted(set(np.quantile(x, np.linspace(0, 1, 101)).round(6).tolist()))
        json.dump(dict(jtau=jtau), open(gf, 'w'))
        res = simulate(T, jtau, fg5['sib']['tau'])
        ch = {}
        for k in ('T_jp', 'count_jp', 'count_sib', 'T_sib'):
            att = res[k][:, 1].mean(-1); acc = res[k][:, 0].mean(-1)
            ch[k] = {str(m): dict(idx=int(np.argmin(np.abs(att - m))), dev_att=float(att[np.argmin(np.abs(att - m))]),
                                  dev_acc=float(acc[np.argmin(np.abs(att - m))])) for m in M_T}
        ch['T_sib_phase5'] = {str(m): ch5['T_sib'][str(m)] for m in M_T if str(m) in ch5['T_sib']}
        json.dump(ch, open(chf, 'w'), indent=1)
        for k in ('T_jp', 'count_jp', 'count_sib', 'T_sib'):
            print(k, ' '.join(f'm{m}:{v["dev_att"]:.2f}/{v["dev_acc"]:.3f}' for m, v in ch[k].items()))
        a_jp = auroc(x, np.concatenate([t['v'][t['d']] for t in T]))
        a_sb = auroc(np.concatenate([t['J']['sib'][t['d']] for t in T]), np.concatenate([t['v'][t['d']] for t in T]))
        print('dev pooled finished AUROC judge', round(a_jp, 4), 'siblings', round(a_sb, 4))
        return
    jtau = json.load(open(gf))['jtau']; ch = json.load(open(chf))
    out = {}
    for set_ in ('holdout250', 'textbook72'):
        per = {}
        for g, stem in (('s0', f'{set_}__s0_r16__k64'), ('s1', f'{set_}__s1_r16__k64'), ('s2', f'{set_}__s2_r16__k64'),
                        ('s0b', f'{set_}__s0_r16__k64b')):
            T = load(stem, g[:2]); per[g] = (T, simulate(T, jtau, fg5['sib']['tau']))
            f = np.concatenate([t['d'] for t in T]); y = np.concatenate([t['v'] for t in T])[f]
            out[f'{set_}|{g}|pooled_auroc'] = dict(judge=float(auroc(np.concatenate([t['J']['jp'] for t in T])[f], y)),
                                                   siblings=float(auroc(np.concatenate([t['J']['sib'] for t in T])[f], y)))
            wj, ws = [], []
            for t in T:
                ff = t['d']
                if ff.sum() >= 2 and t['v'][ff].any() and (~t['v'][ff]).any():
                    wj.append(auroc(t['J']['jp'][ff], t['v'][ff])); ws.append(auroc(t['J']['sib'][ff], t['v'][ff]))
            out[f'{set_}|{g}|within_auroc'] = dict(judge=float(np.mean(wj)), siblings=float(np.mean(ws)), n=len(wj))
        nT = len(per['s0'][0]); rng = np.random.default_rng(0); BS = [rng.integers(0, nT, nT) for _ in range(2000)]
        boot = lambda d: (float(d.mean()), float(np.quantile([d[ix].mean() for ix in BS], .025)), float(np.quantile([d[ix].mean() for ix in BS], .975)))
        # (1) count-stop judge vs siblings
        for c in CS:
            row = {}
            dpool = np.zeros(nT)
            for g in per:
                r = per[g][1]; d = r['count_jp'][c - 1, 0] - r['count_sib'][c - 1, 0]
                row[g] = dict(att=float(r['count_jp'][c - 1, 1].mean()), acc_judge=float(r['count_jp'][c - 1, 0].mean()),
                              acc_sib=float(r['count_sib'][c - 1, 0].mean()), diff=boot(d))
                if g in GENS:
                    dpool += d / 3
            row['pooled_gens'] = dict(diff=boot(dpool), att=float(np.mean([row[g]['att'] for g in GENS])),
                                      acc_judge=float(np.mean([row[g]['acc_judge'] for g in GENS])),
                                      acc_sib=float(np.mean([row[g]['acc_sib'] for g in GENS])))
            out[f'{set_}|count|c{c}'] = row
        # (2) thresholds
        for m in M_T:
            m = str(m)
            ij = ch['T_jp'][m]['idx']; isb = ch5['T_sib'][m]['idx'] if m in ch5['T_sib'] else ch['T_sib'][m]['idx']
            row = {}
            dj_cj = np.zeros(nT); dj_cs = np.zeros(nT); dj_ts = np.zeros(nT)
            stats = {k: [] for k in ('att', 'acc', 'cj', 'cs', 'or', 'ts_att', 'ts_acc')}
            for g in per:
                r = per[g][1]
                a = r['T_jp'][ij]; ts = r['T_sib'][isb]; att = a[1].mean()
                cj = matched(a[0], att, r['count_jp']); cs = matched(a[0], att, r['count_sib'])
                O = r['oracle']; oatt = O[:, 1].mean(-1); oacc = O[:, 0].mean(-1)
                orv = float(np.interp(att, oatt, oacc)) if att <= oatt.max() else float(oacc.max())
                row[g] = dict(att=float(att), acc=float(a[0].mean()), count_judge_matched=float(cj.mean()),
                              count_sib_matched=float(cs.mean()), oracle_stop=orv, tsib_att=float(ts[1].mean()),
                              tsib_acc=float(ts[0].mean()), vs_count_judge=boot(a[0] - cj), vs_count_sib=boot(a[0] - cs),
                              vs_tsib=boot(a[0] - ts[0]))
                if g in GENS:
                    dj_cj += (a[0] - cj) / 3; dj_cs += (a[0] - cs) / 3; dj_ts += (a[0] - ts[0]) / 3
                    for k, v in (('att', att), ('acc', a[0].mean()), ('cj', cj.mean()), ('cs', cs.mean()), ('or', orv),
                                 ('ts_att', ts[1].mean()), ('ts_acc', ts[0].mean())):
                        stats[k].append(v)
            mean = {k: float(np.mean(v)) for k, v in stats.items()}
            row['pooled_gens'] = dict(**mean, vs_count_judge=boot(dj_cj), vs_count_sib=boot(dj_cs), vs_tsib=boot(dj_ts),
                                      gap_closed_vs_count_judge=(mean['acc'] - mean['cj']) / (mean['or'] - mean['cj']),
                                      gap_closed_vs_count_sib=(mean['acc'] - mean['cs']) / (mean['or'] - mean['cs']))
            out[f'{set_}|Tjp|m{m}'] = row
        out[f'{set_}|curves'] = {k: dict(per_gen={g: dict(att=per[g][1][k][:, 1].mean(-1).tolist(), acc=per[g][1][k][:, 0].mean(-1).tolist())
                                                  for g in GENS}) for k in ('count_jp', 'count_sib', 'T_jp', 'T_sib', 'oracle', 'ff')}
        out[f'{set_}|stars_Tjp'] = {m: out[f'{set_}|Tjp|m{m}']['pooled_gens'] for m in map(str, M_T)}
    json.dump(out, open(os.path.join(RES, 'B7_results.json'), 'w'))
    for k, v in out.items():
        if '|count|' in k:
            p = v['pooled_gens']; print(k, f'att {p["att"]:.2f} judge {p["acc_judge"]:.3f} sib {p["acc_sib"]:.3f} diff {p["diff"][0]:+.4f} [{p["diff"][1]:+.4f},{p["diff"][2]:+.4f}]',
                                        '| s0b', f'{v["s0b"]["diff"][0]:+.4f} [{v["s0b"]["diff"][1]:+.4f},{v["s0b"]["diff"][2]:+.4f}]')
        if '|Tjp|' in k:
            p = v['pooled_gens']
            print(k, f'att {p["att"]:.2f} acc {p["acc"]:.3f} | count-judge@att {p["cj"]:.3f} {p["vs_count_judge"][0]:+.4f} [{p["vs_count_judge"][1]:+.4f},{p["vs_count_judge"][2]:+.4f}] '
                     f'| count-sib@att {p["cs"]:.3f} {p["vs_count_sib"][0]:+.4f} [{p["vs_count_sib"][1]:+.4f},{p["vs_count_sib"][2]:+.4f}] | T-sib {p["ts_att"]:.2f}/{p["ts_acc"]:.3f} '
                     f'diff {p["vs_tsib"][0]:+.4f} [{p["vs_tsib"][1]:+.4f},{p["vs_tsib"][2]:+.4f}] | oracle {p["or"]:.3f} | gap closed vs count-judge {p["gap_closed_vs_count_judge"]:+.2f}, vs count-sib {p["gap_closed_vs_count_sib"]:+.2f}')
        if 'auroc' in k:
            print(k, v)


if __name__ == '__main__':
    main()
