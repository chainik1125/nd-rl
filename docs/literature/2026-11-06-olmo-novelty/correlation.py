#!/usr/bin/env python3
"""ProRL's "the weaker the start, the stronger the gain" correlation, computed on our stored IF results (run olmo-novelty).

ProRL (2505.24864v1, §4.1, Fig. 3) plots, per task, the base model's pass@128 against the change in pass@128 after RL.
We do the same per prompt (and per constraint type, ProRL's unit is a task) for Olmo-3-32B-Think-DPO (`dpo`, the RL
start) against Olmo-3.1-32B-Think RL checkpoints (`s0100`, `s0750`, `s2300`), from olmo-ifomega's fixed phase only:
`dpo` 16 draws per prompt, each RL checkpoint 8 (olmo_ifomega/results.csv; strict and loose grades). No sampling, no
pods. Pre-registered in preregistration/olmo-novelty.md (commit 664aaa93).

Primary (P1): Spearman rho across IFBench's 218 non-excluded prompts, strict, `s2300` vs `dpo`, between x = `dpo`
pass@1 from 8 of its 16 draws and y = `s2300` pass@1 minus `dpo` pass@1 from the other 8 (split-half: the gain does
not share the x estimate's noise). Draws are exchangeable, so the split is a random hypergeometric partition of each
prompt's 16 draws; rho is averaged over 2,000 partitions; the 95 % interval is a 2,000-resample prompt bootstrap with
a new partition each time. Secondary analyses as pre-registered (see ANALYSES below).

usage: python3 olmo_novelty/correlation.py   (repository root; writes olmo_novelty/correlation.csv and
       olmo_novelty/correlation_summary.csv, prints the summary table)
"""
import csv, json, os, sys, collections
import numpy as np
from scipy.stats import rankdata
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'olmo_ifomega', 'results.csv')
CAND = os.path.join(ROOT, 'olmo_ifomega', 'candidates.csv')
PROB = os.path.join(ROOT, 'olmo_ifomega', 'problems.jsonl')
OUT = os.path.join(ROOT, 'olmo_novelty', 'correlation.csv')
SUM = os.path.join(ROOT, 'olmo_novelty', 'correlation_summary.csv')
R_PART, B_BOOT, P_PERM = 2000, 2000, 2000
SEED = 20261107


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def pass_at_k(n, c, k):
    """Unbiased pass@k from n draws with c successes (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load():
    rows = list(csv.DictReader(open(RES)))
    probs = {}
    for line in open(PROB):
        p = json.loads(line)
        probs[p['pid']] = p
    data = collections.defaultdict(dict)  # (task, pid) -> model -> row
    for r in rows:
        if r['task'] not in ('ifbench', 'ifeval') or r['excluded']:
            continue
        data[(r['task'], r['pid'])][r['model']] = r
    # reach flags as olmo-ifomega defines them (dpo 0 in >= 256 draws, s2300 confirmed on fresh draws)
    cand = collections.defaultdict(dict)
    for r in csv.DictReader(open(CAND)):
        cand[(r['task'], r['pid'], r['kind'])][r['model']] = r
    reach = {}
    for (t, pid, kind), m in cand.items():
        if kind != 'reach' or 'dpo' not in m or 's2300' not in m:
            continue
        d, s = m['dpo'], m['s2300']
        big = int(d['draws']) >= 256
        reach[(t, pid)] = (big and d['succ_strict'] == '0' and s['confirmed_strict'] == 'True',
                           big and d['succ_hand'] == '0' and s['confirmed_hand'] == 'True')
    return data, probs, reach


def counts(data, task, model, grade):
    """Per prompt (pid, successes, n_fixed) for one model and grade; checks the fixed-phase sizes."""
    want = 16 if model == 'dpo' else 8
    out = {}
    for (t, pid), m in data.items():
        if t != task or model not in m:
            continue
        r = m[model]
        n = int(r['n_fixed'])
        assert n == want, (task, pid, model, n)
        out[pid] = int(r[f'succ_fixed_{grade}'])
    return out


def split_draw(rng, s16):
    """Successes in a random half (8 of 16 draws) of each prompt, given its 16-draw success count."""
    s16 = np.asarray(s16)
    return rng.hypergeometric(s16, 16 - s16, 8)


def xy(kind, a, b, r8):
    """x and gain y for one partition. kind 'p1': pass@1 rates; 'p8': pass@8 (binary at n = 8)."""
    if kind == 'p1':
        return a / 8.0, r8 / 8.0 - b / 8.0
    return (a > 0).astype(float), (r8 > 0).astype(float) - (b > 0).astype(float)


def prompt_analysis(rng, s16, r8, kind, mask=None):
    s16, r8 = np.asarray(s16), np.asarray(r8)
    if mask is not None:
        s16, r8 = s16[mask], r8[mask]
    n = len(s16)
    # mean rho over partitions
    rhos = []
    for _ in range(R_PART):
        a = split_draw(rng, s16)
        x, y = xy(kind, a, s16 - a, r8)
        rhos.append(spearman(x, y))
    rho = float(np.nanmean(rhos))
    # prompt bootstrap, new partition each time
    boots = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        a = split_draw(rng, s16[idx])
        x, y = xy(kind, a, s16[idx] - a, r8[idx])
        boots.append(spearman(x, y))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    # naive: same draws for x and the gain
    if kind == 'p1':
        xn, yn = s16 / 16.0, r8 / 8.0 - s16 / 16.0
    else:
        p8 = np.array([pass_at_k(16, c, 8) for c in s16])
        xn, yn = p8, (r8 > 0).astype(float) - p8
    naive = spearman(xn, yn)
    pear = float(np.corrcoef(xn, yn)[0, 1])
    # random-pairing null: RL's per-prompt counts permuted across prompts
    nulls = []
    for _ in range(P_PERM):
        rp = rng.permutation(r8)
        a = split_draw(rng, s16)
        x, y = xy(kind, a, s16 - a, rp)
        nulls.append(spearman(x, y))
    nm = float(np.nanmean(nulls))
    nlo, nhi = np.nanpercentile(nulls, [2.5, 97.5])
    # added (not pre-registered, descriptive): Spearman rho of x with RL's own rate, which is what makes the
    # observed rho weaker than the random-pairing null
    rx = []
    for _ in range(R_PART):
        a = split_draw(rng, s16)
        x, _y = xy(kind, a, s16 - a, r8)
        rx.append(spearman(x, r8 / 8.0 if kind == 'p1' else (r8 > 0).astype(float)))
    return dict(n_units=n, rho=rho, lo95=float(lo), hi95=float(hi), naive_rho=naive, naive_pearson=pear,
                null_mean=nm, null_lo95=float(nlo), null_hi95=float(nhi), rho_x_rl=float(np.nanmean(rx)))


def bins(rng, s16, r8):
    """Mean s2300 gain in pass@1 by dpo half-sample pass@1 bin (split-half), averaged over partitions."""
    edges = [(-1, 0, '0'), (0, 0.25, '(0,0.25]'), (0.25, 0.5, '(0.25,0.5]'), (0.5, 0.75, '(0.5,0.75]'),
             (0.75, 1.0, '(0.75,1]')]
    s16, r8 = np.asarray(s16), np.asarray(r8)
    acc = {lab: [0.0, 0.0] for _, _, lab in edges}
    for _ in range(R_PART):
        a = split_draw(rng, s16)
        x, y = xy('p1', a, s16 - a, r8)
        for lo, hi, lab in edges:
            m = (x > lo) & (x <= hi)
            acc[lab][0] += m.sum()
            acc[lab][1] += y[m].sum()
    return [(lab, acc[lab][0] / R_PART, acc[lab][1] / acc[lab][0] if acc[lab][0] else float('nan'))
            for _, _, lab in edges]


def type_analysis(rng, task, pids, s16, r8, probs, split):
    """ProRL's unit: per constraint type (types with >= 2 prompts), x = mean dpo pass@8, y = mean gain in pass@8."""
    types = collections.defaultdict(list)
    for i, pid in enumerate(pids):
        for t in probs[pid].get('instruction_id_list') or []:
            types[t].append(i)
    keep = [v for v in types.values() if len(v) >= 2]
    s16, r8 = np.asarray(s16), np.asarray(r8)

    def rho_once(idx_sets, a=None):
        xs, ys = [], []
        for v in idx_sets:
            v = np.asarray(v)
            if split:
                aa = a[v]
                x = (aa > 0).mean()
                y = (r8[v] > 0).mean() - ((s16[v] - aa) > 0).mean()
            else:
                p8 = np.array([pass_at_k(16, c, 8) for c in s16[v]])
                x = p8.mean()
                y = (r8[v] > 0).mean() - p8.mean()
            xs.append(x); ys.append(y)
        return spearman(xs, ys)

    if split:
        rho = float(np.nanmean([rho_once(keep, split_draw(rng, s16)) for _ in range(R_PART)]))
    else:
        rho = rho_once(keep)
    boots = []
    for _ in range(B_BOOT):
        pick = [keep[j] for j in rng.integers(0, len(keep), len(keep))]
        boots.append(rho_once(pick, split_draw(rng, s16) if split else None))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(n_units=len(keep), rho=rho, lo95=float(lo), hi95=float(hi))


def main():
    rng = np.random.default_rng(SEED)
    data, probs, reach = load()
    out_rows, summary = [], []
    for task in ('ifbench', 'ifeval'):
        models = ['s0100', 's0750', 's2300'] if task == 'ifbench' else ['s0750', 's2300']
        cs = {(m, g): counts(data, task, m, g) for m in ['dpo'] + models for g in ('strict', 'loose')}
        pids = sorted(cs[('dpo', 'strict')])
        for pid in pids:
            row = dict(task=task, pid=pid,
                       constraints=';'.join(probs[pid].get('instruction_id_list') or []),
                       reach_strict=int(reach.get((task, pid), (False, False))[0]),
                       reach_hand=int(reach.get((task, pid), (False, False))[1]))
            for g in ('strict', 'loose'):
                s = cs[('dpo', g)][pid]
                row[f'dpo_{g}_succ16'] = s
                row[f'dpo_{g}_pass1'] = round(s / 16, 4)
                row[f'dpo_{g}_pass8'] = round(pass_at_k(16, s, 8), 4)
                for m in models:
                    r = cs[(m, g)][pid]
                    row[f'{m}_{g}_succ8'] = r
                    row[f'gain_{m}_{g}_pass1_naive'] = round(r / 8 - s / 16, 4)
            out_rows.append(row)

        def add(aid, grade, model, k, unit, res, note=''):
            summary.append(dict(analysis=aid, task=task, grade=grade, model=model, k=k, unit=unit, **res,
                                note=note))

        for g in ('strict', 'loose'):
            s16 = np.array([cs[('dpo', g)][p] for p in pids])
            for m in models:
                r8 = np.array([cs[(m, g)][p] for p in pids])
                if g == 'loose' and m != 's2300':
                    continue
                aid = 'P1' if (task, g, m) == ('ifbench', 'strict', 's2300') else 'S'
                add(aid, g, m, 1, 'prompt', prompt_analysis(rng, s16, r8, 'p1'), 'split-half')
                if g == 'strict':
                    add('S', g, m, 8, 'prompt', prompt_analysis(rng, s16, r8, 'p8'), 'split-half; pass@8 binary')
                if g == 'strict' and m == 's2300':
                    mask = s16 >= 1
                    add('S-supported', g, m, 1, 'prompt', prompt_analysis(rng, s16, r8, 'p1', mask),
                        'only prompts with dpo >= 1 of 16')
                    for split in (True, False):
                        add('S-type', g, m, 8, 'constraint type',
                            type_analysis(rng, task, pids, s16, r8, probs, split),
                            'split-half' if split else 'naive (shared draws)')
                    for lab, nmean, gmean in bins(rng, s16, r8):
                        summary.append(dict(analysis='S-bin', task=task, grade=g, model=m, k=1, unit='prompt',
                                            n_units=round(nmean, 1), rho='', lo95='', hi95='', naive_rho='',
                                            naive_pearson='', null_mean='', null_lo95='', null_hi95='',
                                            rho_x_rl='', note=f'dpo half-sample pass@1 bin {lab}: mean gain {gmean:.3f}'))
                    # share of the total gain on prompts where dpo has 0 of 16, and on the reach prompts
                    gain = r8 / 8 - s16 / 16
                    z = s16 == 0
                    rs = np.array([reach.get((task, p), (False, False))[0] for p in pids])
                    summary.append(dict(analysis='S-share', task=task, grade=g, model=m, k=1, unit='prompt',
                                        n_units=int(z.sum()), rho='', lo95='', hi95='', naive_rho='',
                                        naive_pearson='', null_mean='', null_lo95='', null_hi95='', rho_x_rl='',
                                        note=(f'total pass@1 gain {gain.sum():.2f} over {len(pids)} prompts; '
                                              f'{gain[z].sum():.2f} ({gain[z].sum() / gain.sum():.0%}) on the '
                                              f'{int(z.sum())} with dpo 0/16; {gain[rs].sum():.2f} on the '
                                              f'{int(rs.sum())} strict-reach prompts')))

    keys = list(out_rows[0].keys())
    allkeys = keys + [k for r in out_rows for k in r if k not in keys]
    allkeys = list(dict.fromkeys(allkeys))
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=allkeys, restval='')
        w.writeheader(); w.writerows(out_rows)
    fields = ['analysis', 'task', 'grade', 'model', 'k', 'unit', 'n_units', 'rho', 'lo95', 'hi95', 'naive_rho',
              'naive_pearson', 'null_mean', 'null_lo95', 'null_hi95', 'rho_x_rl', 'note']
    with open(SUM, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval='')
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    for s in summary:
        f = lambda v: f'{v:+.3f}' if isinstance(v, float) else str(v)
        print(s['analysis'], s['task'], s['grade'], s['model'], f"k{s['k']}", s['unit'], s['n_units'],
              'rho', f(s.get('rho', '')), '[', f(s.get('lo95', '')), f(s.get('hi95', '')), ']',
              'naive', f(s.get('naive_rho', '')), 'null', f(s.get('null_mean', '')), 'x~RL', f(s.get('rho_x_rl', '')),
              '|', s['note'])


if __name__ == '__main__':
    main()
