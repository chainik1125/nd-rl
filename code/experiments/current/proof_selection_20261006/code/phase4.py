#!/usr/bin/env python3
"""Phase 4 (Worker B): (1) headline curves over the three generators, (2) mechanism probe, (3) ancestor judges.

(1) For each test set and generator pool (s0, s1, s2 r16; k=64, first 16 attempts of R=50 seeded orderings):
    fixed-k random pick (= pass rate), fixed-k completion pick, stop at first finished (N_max 1..16), count-stop c=1..8
    (N_max 16) selecting by self likelihood and by sibling likelihood (mean of the two other r16 seeds), oracle
    pass@k, oracle stop-at-first-valid.  Curves are interpolated onto a common log2 grid of mean attempts and
    summarised as mean and min-max over generators.  -> results/B4_headline_curves.json
(2) Mechanism: per (set, generator, judge in {self, each sibling}) mean masked loglik of finished-valid and finished-
    invalid attempts, raw and theorem-centred (attempt minus the mean over that theorem's finished attempts, theorems
    with both classes); self-minus-sibling shift split by validity; within-theorem AUROC by proof-length tercile
    (steps) and the self - sibling step-level mean-logp gap by step type.  -> results/B4_mechanism.json
(3) Ancestor judges on the s0_r16 pools: self (s0_r16), r8 ancestor, stage-1, s1, s2, siblings: within-theorem
    finished-only AUROC and k=8 completion_loglik selection (200 shared subsets), paired theorem bootstrap (B=2000)
    of judge - self.  -> results/B4_ancestors.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auroc import auroc

RUNS = os.path.expanduser('~/runs/proof-selection')
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
GENS = ('s0', 's1', 's2'); SETS = ('holdout250', 'textbook72')
GRID = 2 ** np.linspace(0, 4, 33)


def rj(p):
    return {r['name']: r for r in map(json.loads, open(p))}


def judge_file(stem, g, j):
    if g == 's0' and j == 's0':
        return f'{RUNS}/scores/{stem}__loglik_mean_masked.jsonl'
    return f'{RUNS}/scores/{stem}__loglik_mean_masked_{j}.jsonl'


def detail_file(stem, g, j):
    if g == 's0' and j == 's0':
        return f'{RUNS}/scores/aux/{stem}__loglik_detail.jsonl'
    return f'{RUNS}/scores/aux/{stem}__loglik_detail_{j}.jsonl'


def load(set_, g, judges):
    stem = f'{set_}__{g}_r16__k64'
    pool = rj(f'{RUNS}/pools_v2/{stem}.jsonl'); lab = rj(f'{RUNS}/pools_v2/{stem}.labels.jsonl')
    J = {j: rj(judge_file(stem, g, j)) for j in judges}
    T = []
    for n in sorted(pool):
        at = pool[n]['attempts']
        T.append(dict(name=n, d=np.array([a['done'] for a in at]), v=np.array(lab[n]['valid'], bool),
                      steps=np.array([a['steps'] for a in at]), L={j: np.array(J[j][n]['scores'], float) for j in judges},
                      actions=[a['actions'] for a in at]))
    return stem, T


# ------------------------------------------------------------------------------------------- (1) headline curves
def curves_for(T, g):
    sib = [j for j in GENS if j != g]
    K = 16; R = 50
    acc = {k: [] for k in ('random', 'fixed_completion', 'ff', 'count_self', 'count_sib', 'oracle_fixed', 'oracle_stop')}
    att = {k: [] for k in acc}
    per = []
    for ti, t in enumerate(T):
        P = np.stack([np.random.default_rng([7, ti, r]).permutation(len(t['d'])) for r in range(R)])[:, :K]
        D = t['d'][P]; V = t['v'][P].astype(float)
        s_self = np.where(t['d'], t['L'][g], -np.inf)[P]
        s_sib = np.where(t['d'], (t['L'][sib[0]] + t['L'][sib[1]]) / 2, -np.inf)[P]

        def incumbent(S):
            M = np.maximum.accumulate(S, 1)
            new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
            A = np.maximum.accumulate(np.where(new, np.arange(K)[None], 0), 1)
            return np.take_along_axis(V, A, 1)
        Vc = incumbent(D.astype(float)); Vself = incumbent(s_self); Vsib = incumbent(s_sib)
        ever = np.maximum.accumulate(V, 1)
        nf = np.cumsum(D, 1); nv = np.cumsum(V, 1)
        row = {}
        row['random'] = (np.arange(1, K + 1), np.full(K, t['v'].mean()))
        row['fixed_completion'] = (np.arange(1, K + 1), Vc.mean(0))
        row['oracle_fixed'] = (np.arange(1, K + 1), ever.mean(0))
        ff_first = np.where((nf >= 1).any(1), (nf >= 1).argmax(1), K)
        ov_first = np.where((nv >= 1).any(1), (nv >= 1).argmax(1), K)
        a_ff, y_ff, a_ov, y_ov = [], [], [], []
        for N in range(1, K + 1):
            st = np.minimum(ff_first, N - 1); a_ff.append((st + 1).mean()); y_ff.append(Vc[np.arange(R), st].mean())
            st = np.minimum(ov_first, N - 1); a_ov.append((st + 1).mean()); y_ov.append(ever[np.arange(R), st].mean())
        row['ff'] = (np.array(a_ff), np.array(y_ff)); row['oracle_stop'] = (np.array(a_ov), np.array(y_ov))
        a_c, y_s, y_b = [], [], []
        for c in range(1, 9):
            st = np.where((nf >= c).any(1), (nf >= c).argmax(1), K - 1)
            a_c.append((st + 1).mean()); y_s.append(Vself[np.arange(R), st].mean()); y_b.append(Vsib[np.arange(R), st].mean())
        row['count_self'] = (np.array(a_c), np.array(y_s)); row['count_sib'] = (np.array(a_c), np.array(y_b))
        per.append(row)
    out = {}
    for k in per[0]:
        x = np.mean([r[k][0] for r in per], 0); y = np.mean([r[k][1] for r in per], 0)
        out[k] = (x, y)
    return out


def headline():
    res = {}
    for set_ in SETS:
        cur = {}
        for g in GENS:
            _, T = load(set_, g, GENS)
            cur[g] = curves_for(T, g)
        S = {}
        for k in cur['s0']:
            ys = []
            for g in GENS:
                x, y = cur[g][k]
                o = np.argsort(x); x, y = x[o], y[o]
                yi = np.interp(GRID, x, y, left=np.nan, right=np.nan)
                ys.append(yi)
            ys = np.array(ys)
            S[k] = dict(per_generator={g: dict(att=cur[g][k][0].tolist(), acc=cur[g][k][1].tolist()) for g in GENS},
                        grid=GRID.tolist(), mean=np.nanmean(ys, 0).tolist() if True else None,
                        min=np.min(ys, 0).tolist(), max=np.max(ys, 0).tolist(),
                        all_defined=(~np.isnan(ys).any(0)).tolist())
        res[set_] = S
    json.dump(res, open(os.path.join(RES, 'B4_headline_curves.json'), 'w'))
    return res


# ------------------------------------------------------------------------------------------- (2) mechanism
def step_type(a_ids, itos):
    toks = [itos[x] for x in a_ids]
    if not toks or toks[0] == 'exact':
        return 'exact'
    if ':=' in toks:
        i = toks.index(':=') + 1
        if i < len(toks) and toks[i] == '(':
            return 'box_open'
        if i < len(toks) and toks[i] == 'Or.elim':
            return 'orelim_open'
        if i < len(toks) and toks[i].startswith('h') and toks[i][1:].isdigit():
            return 'premise'
    return 'have_atomic'


def mechanism():
    sys.path.insert(0, f'{RUNS}/nd-takehome')
    from tokenizer import make_tokenizer
    itos = make_tokenizer('lean_staten').itos
    out = {}
    for set_ in SETS:
        for g in GENS:
            stem, T = load(set_, g, GENS)
            sib = [j for j in GENS if j != g]
            rec = {}
            for j in list(GENS) + ['siblings']:
                Lget = (lambda t: (t['L'][sib[0]] + t['L'][sib[1]]) / 2) if j == 'siblings' else (lambda t, j=j: t['L'][j])
                v_raw, i_raw, v_c, i_c = [], [], [], []
                for t in T:
                    f = t['d']; x = Lget(t)
                    v_raw += list(x[f & t['v']]); i_raw += list(x[f & ~t['v']])
                    if (f & t['v']).any() and (f & ~t['v']).any():
                        m = x[f].mean(); v_c.append((x[f & t['v']] - m).mean()); i_c.append((x[f & ~t['v']] - m).mean())
                rec[j] = dict(valid_mean=float(np.mean(v_raw)), invalid_mean=float(np.mean(i_raw)), sep_raw=float(np.mean(v_raw) - np.mean(i_raw)),
                              valid_centred=float(np.mean(v_c)), invalid_centred=float(np.mean(i_c)),
                              sep_centred=float(np.mean(v_c) - np.mean(i_c)), n_valid=len(v_raw), n_invalid=len(i_raw), n_mixed=len(v_c))
            # self - siblings shift by validity (per attempt, finished only)
            sv, si = [], []
            for t in T:
                f = t['d']; dlt = t['L'][g] - (t['L'][sib[0]] + t['L'][sib[1]]) / 2
                sv += list(dlt[f & t['v']]); si += list(dlt[f & ~t['v']])
            rec['self_minus_sib_shift'] = dict(valid=float(np.mean(sv)), invalid=float(np.mean(si)))
            # within-theorem AUROC by proof-length tercile (steps among finished attempts of the set)
            allsteps = np.concatenate([t['steps'][t['d']] for t in T])
            q1, q2 = np.quantile(allsteps, [1 / 3, 2 / 3])
            byl = {}
            for lab, lo, hi in (('short', -1, q1), ('mid', q1, q2), ('long', q2, 1e9)):
                aself, asib = [], []
                for t in T:
                    m = t['d'] & (t['steps'] > lo) & (t['steps'] <= hi)
                    if m.sum() >= 2 and t['v'][m].any() and (~t['v'][m]).any():
                        aself.append(auroc(t['L'][g][m], t['v'][m]))
                        asib.append(auroc(((t['L'][sib[0]] + t['L'][sib[1]]) / 2)[m], t['v'][m]))
                byl[lab] = dict(steps=[float(lo), float(hi)], n=len(aself), self=float(np.mean(aself)) if aself else None,
                                siblings=float(np.mean(asib)) if asib else None,
                                gap=float(np.mean(np.array(asib) - np.array(aself))) if aself else None)
            rec['auroc_by_length'] = byl
            # step-type gaps: self - sibling-mean step mean logp, finished attempts, by validity
            det = {j: rj(detail_file(stem, g, j)) for j in GENS}
            gaps = {}
            for ti, t in enumerate(T):
                for ai in np.where(t['d'])[0]:
                    sl = {j: det[j][t['name']]['detail'][ai]['step_mean_lp'] for j in GENS}
                    for si_, a_ids in enumerate(t['actions'][ai]):
                        ty = step_type(a_ids, itos)
                        d_ = sl[g][si_] - (sl[sib[0]][si_] + sl[sib[1]][si_]) / 2
                        gaps.setdefault(ty, {'valid': [], 'invalid': []})['valid' if t['v'][ai] else 'invalid'].append(d_)
            rec['step_type_self_minus_sib'] = {ty: dict(valid=float(np.mean(v['valid'])) if v['valid'] else None,
                                                        invalid=float(np.mean(v['invalid'])) if v['invalid'] else None,
                                                        n_valid=len(v['valid']), n_invalid=len(v['invalid']))
                                               for ty, v in gaps.items()}
            out[f'{set_}|{g}'] = rec
    json.dump(out, open(os.path.join(RES, 'B4_mechanism.json'), 'w'), indent=1)
    return out


# ------------------------------------------------------------------------------------------- (3) ancestors
def ancestors(B=2000):
    judges = ('s0', 'r8', 'st1', 's1', 's2')
    out = {}
    for set_ in SETS:
        stem, T = load(set_, 's0', judges)
        for t in T:
            t['L']['siblings'] = (t['L']['s1'] + t['L']['s2']) / 2
        cols = list(judges) + ['siblings']
        nT = len(T)
        auc = {c: np.full(nT, np.nan) for c in cols}; sel = {c: np.zeros(nT) for c in cols}
        for ti, t in enumerate(T):
            f = t['d']; k = len(f)
            rng = np.random.default_rng([0, ti]); subs = np.stack([rng.permutation(k)[:8] for _ in range(200)])
            for c in cols:
                x = t['L'][c]
                if f.sum() >= 2:
                    auc[c][ti] = auroc(x[f], t['v'][f])
                s = np.where(f, x, -np.inf)
                sel[c][ti] = t['v'][subs[np.arange(200), np.argmax(s[subs], 1)]].mean()
        rng = np.random.default_rng(0); BS = [rng.integers(0, nT, nT) for _ in range(B)]
        r = {}
        for c in cols:
            r[c] = dict(auc=float(np.nanmean(auc[c])), sel8=float(sel[c].mean()))
            if c != 's0':
                for m, A in (('auc', auc), ('sel8', sel)):
                    d = A[c] - A['s0']; bs = np.array([np.nanmean(d[ix]) for ix in BS])
                    r[c][f'{m}_minus_self'] = dict(diff=float(np.nanmean(d)), lo=float(np.nanquantile(bs, .025)), hi=float(np.nanquantile(bs, .975)))
        d = auc['r8'] - auc['siblings']; bs = np.array([np.nanmean(d[ix]) for ix in BS])
        r['r8_minus_siblings_auc'] = dict(diff=float(np.nanmean(d)), lo=float(np.nanquantile(bs, .025)), hi=float(np.nanquantile(bs, .975)))
        out[set_] = r
    json.dump(out, open(os.path.join(RES, 'B4_ancestors.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    what = sys.argv[1:] or ['headline', 'mechanism', 'ancestors']
    if 'ancestors' in what:
        A = ancestors()
        for s, r in A.items():
            print('==', s)
            for c, v in r.items():
                print('  ', c, json.dumps(v))
    if 'mechanism' in what:
        M = mechanism()
        for k, rec in M.items():
            print('==', k)
            for j in list(GENS) + ['siblings']:
                x = rec[j]; print(f'   judge {j:8s} valid {x["valid_mean"]:+.4f} invalid {x["invalid_mean"]:+.4f} sep {x["sep_raw"]:+.4f} | centred valid {x["valid_centred"]:+.4f} invalid {x["invalid_centred"]:+.4f} sep {x["sep_centred"]:+.4f} (n mixed {x["n_mixed"]})')
            print('   self-sib shift', rec['self_minus_sib_shift'])
            print('   by length', {k2: (v['n'], v['self'] and round(v['self'], 3), v['siblings'] and round(v['siblings'], 3), v['gap'] and round(v['gap'], 3)) for k2, v in rec['auroc_by_length'].items()})
            print('   step types', {k2: (v['valid'] and round(v['valid'], 4), v['invalid'] and round(v['invalid'], 4), v['n_valid'], v['n_invalid']) for k2, v in rec['step_type_self_minus_sib'].items()})
    if 'headline' in what:
        H = headline()
        for s, S in H.items():
            print('==', s)
            for k, v in S.items():
                vals = []
                for m in (2, 4, 8):
                    i = int(np.argmin(np.abs(GRID - m)))
                    vals.append(f'm{m}: {v["mean"][i]:.3f} [{v["min"][i]:.3f},{v["max"][i]:.3f}]' if v['all_defined'][i] else f'm{m}: n/a')
                print('  ', k, ' | '.join(vals))
