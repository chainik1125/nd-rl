#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Goals variant vs the repr/ abs baseline (same 039 pretraining at 2,911 steps, same arm-C RL, same evaluation):
report/report.md, report/*.png, report/results.json.

    uv run --script analyse.py [--out report] [--abs-root /scratch/cowifornia/nd-rl/ab3/repr/eval] [--boot 200]

Groups (labels): goals = goals/eval/goals_s{0,1,2}; abs = repr/eval/abs_s{0,1,2} (read only); reference: arm C of
lengthexp (lengthexp/eval/C_s10{1,2,3}; frozen pretrain, a different pretraining recipe). Seeds are paired by the
pretraining seed (RL seed 101 / 102 / 103).
Metrics: pass@1 / pass@4 per length; the log-odds slope per extra line past 11 (grouped-binomial logistic over
L = 12..18, as lengthexp's analyse.py) per run, seed mean with a t-interval, the paired goals - abs difference and an
exact permutation test over the 6 runs; the share of attempts at least L_true lines long; the first-error position;
in-distribution checks (held-out greedy after pretraining and after RL, pass@1 at L6-11); annotation behaviour; the
training side (rounds, samples, compute, solve rates); the matched control.
"""
import argparse, collections, gzip, itertools, json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AB3 = '/scratch/cowifornia/nd-rl/ab3'
LENGTHS = tuple(range(6, 19))
TRAIN_MAX = 11
SEEDS = (0, 1, 2)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def pass_at_k(n, c, k):
    if k > n:
        return float('nan')
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def read_jsonl(fn):
    op = gzip.open if fn.endswith('.gz') else open
    with op(fn, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]


def current_unit_shas():
    """unit sha of every (set, L) of the current lengthexp test sets (as lengthexp/evaluate.py computes it)."""
    import hashlib
    out = {}
    for s in ('std', 'matched'):
        by = collections.defaultdict(list)
        for r in read_jsonl(os.path.join(HERE, 'testsets', f'{s}.jsonl')):
            by[r['L_true']].append(r)
        for L, rs in by.items():
            rs = sorted(rs, key=lambda r: r['name'])
            out[(s, L)] = hashlib.sha256('\n'.join(r['name'] + '\t' + r['prompt'] for r in rs).encode()).hexdigest()
    return out


UNIT_SHA = None
STALE = []


def load_eval(d):
    """{(set, L): rows} plus per-unit attempt lengths (written lines of every sample, from rows or samples). Units
    whose theorems differ from the current test sets (an earlier draw) are left out and listed in STALE."""
    global UNIT_SHA
    if not os.path.isdir(d):
        return None
    if UNIT_SHA is None:
        UNIT_SHA = current_unit_shas()
    units, lines, summ = {}, {}, {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.summary.json'):
            continue
        u = f[:-len('.summary.json')]
        s, L = u.rsplit('_L', 1)
        L = int(L)
        sm = json.load(open(os.path.join(d, f)))
        if sm.get('unit_sha') != UNIT_SHA.get((s, L)):
            STALE.append(f'{os.path.basename(d)}:{u}')
            continue
        rows = read_jsonl(os.path.join(d, u + '.jsonl'))
        units[(s, L)] = rows
        summ[(s, L)] = sm
        if rows and 'n_lines' in rows[0]:
            lines[(s, L)] = {r['name']: r['n_lines'] for r in rows}
        else:
            sp = os.path.join(d, 'samples', u + '.jsonl.gz')
            if os.path.isfile(sp):
                by = collections.defaultdict(list)
                for x in read_jsonl(sp):
                    by[x['name']].append(x['body'].split().count(';'))
                lines[(s, L)] = dict(by)
    return {'dir': d, 'units': units, 'lines': lines, 'summ': summ}


def passk(rows, k):
    return float(np.mean([pass_at_k(r['k'], r['n_ok'], k) for r in rows])) if rows else float('nan')


def fit_logistic(d, s, n, iters=100):
    """lengthexp/analyse.py's grouped-binomial logistic regression logit p = a + b d (Newton-Raphson)."""
    d, s, n = (np.asarray(x, float) for x in (d, s, n))
    m = n > 0
    d, s, n = d[m], s[m], n[m]
    out = dict(a=float('nan'), b=float('nan'), se_b=float('nan'), ok=False, why='')
    if len(d) < 2:
        out['why'] = 'fewer than 2 lengths'
        return out
    if (s > 0).sum() < 2:
        out['why'] = 'successes at fewer than 2 lengths'
        return out
    if (s < n).sum() == 0:
        out['why'] = 'no failures'
        return out
    p0 = np.clip(s.sum() / n.sum(), 1e-6, 1 - 1e-6)
    beta = np.array([math.log(p0 / (1 - p0)), 0.0])
    X = np.stack([np.ones_like(d), d], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (s - n * p)
        H = X.T @ (X * (n * p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            out['why'] = 'singular'
            return out
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    H = X.T @ (X * (n * p * (1 - p))[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    out.update(a=float(beta[0]), b=float(beta[1]), se_b=float(math.sqrt(max(cov[1, 1], 0))), ok=bool(abs(beta[1]) < 30))
    if not out['ok']:
        out['why'] = 'diverged'
    return out


def fit_run(e, lo, hi, boot, rng):
    pts = []
    for L in range(lo, hi + 1):
        rows = e['units'].get(('std', L), [])
        if rows:
            nok = np.array([r['n_ok'] for r in rows], float)
            kk = np.array([r['k'] for r in rows], float)
            pts.append((L, nok, kk))
    f = fit_logistic([L - TRAIN_MAX for L, _, _ in pts], [x.sum() for _, x, _ in pts], [k.sum() for _, _, k in pts])
    f['lengths'] = [L for L, _, _ in pts]
    f['boot_se'] = float('nan')
    if f['ok'] and boot:
        bs = []
        for _ in range(boot):
            ss, nn = [], []
            for (_, nok, kk) in pts:
                ix = rng.integers(0, len(nok), len(nok))
                ss.append(nok[ix].sum())
                nn.append(kk[ix].sum())
            g = fit_logistic([L - TRAIN_MAX for L, _, _ in pts], ss, nn)
            if g['ok']:
                bs.append(g['b'])
        if len(bs) > 10:
            f['boot_se'] = float(np.std(bs, ddof=1))
    return f


def tint(xs):
    xs = [x for x in xs if x is not None and x == x]
    if len(xs) < 2:
        return (float('nan'), float('nan'), float('nan'))
    m, s = float(np.mean(xs)), float(np.std(xs, ddof=1))
    h = T95[len(xs) - 1] * s / math.sqrt(len(xs))
    return m, m - h, m + h


def perm_test(a, b):
    """Exact permutation test of mean(a) - mean(b) over all relabelings of the pooled runs (two-sided)."""
    pool = list(a) + list(b)
    obs = abs(np.mean(a) - np.mean(b))
    cnt = tot = 0
    for ix in itertools.combinations(range(len(pool)), len(a)):
        x = [pool[i] for i in ix]
        y = [pool[i] for i in range(len(pool)) if i not in ix]
        tot += 1
        cnt += abs(np.mean(x) - np.mean(y)) >= obs - 1e-12
    return cnt / tot, tot


def fail_positions(rows):
    """lengthexp's failure position: bad_line -> its line, early_qed -> line + 1 (the QED slot); no_qed apart."""
    pos, kinds = [], collections.Counter()
    for r in rows:
        for kind, line, reason, start in r['fails']:
            kinds[kind] += 1
            if kind == 'bad_line':
                pos.append(line)
            elif kind == 'early_qed':
                pos.append(line + 1)
    return pos, kinds


def hazard_counts(rows):
    """lengthexp/analyse.py's per-line hazard: risk[p] = samples whose lines 1..p-1 were valid and that reached slot p
    (a correct proof is at risk through its QED slot n+1), fails[p] = first rejected line at p (early QED: the QED slot
    line+1); out-of-token samples are at risk through their last complete line and never fail."""
    fails, risk = collections.Counter(), collections.Counter()
    for r in rows:
        for n in r['ok_lines']:
            for p in range(1, n + 2):
                risk[p] += 1
        for kind, line, reason, start in r['fails']:
            if kind == 'bad_line':
                for p in range(1, line + 1):
                    risk[p] += 1
                fails[line] += 1
            elif kind == 'early_qed':
                for p in range(1, line + 2):
                    risk[p] += 1
                fails[line + 1] += 1
            elif kind == 'no_qed':
                for p in range(1, line + 1):
                    risk[p] += 1
    return fails, risk


def fmt(x, nd=3):
    return '-' if x is None or x != x else f'{x:.{nd}f}'


def pct(x, nd=1):
    return '-' if x is None or x != x else f'{100 * x:.{nd}f}%'


def mrange(xs, f=fmt):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return '-'
    return f'{f(np.mean(xs))} [{f(min(xs))}, {f(max(xs))}]' if len(xs) > 1 else f(xs[0])


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def jload(fn):
    try:
        return json.load(open(fn))
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'report'))
    ap.add_argument('--abs-root', default=f'{AB3}/repr/eval')
    ap.add_argument('--abs-pretrain-root', default=f'{AB3}/repr/pretrain')
    ap.add_argument('--abs-runs-root', default=f'{AB3}/repr/runs')
    ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    groups = {
        'goals': {s: load_eval(os.path.join(HERE, 'eval', f'goals_s{s}')) for s in SEEDS},
        'abs': {s: load_eval(os.path.join(a.abs_root, f'abs_s{s}')) for s in SEEDS},
        'arm C (frozen pretrain, reference)': {s: load_eval(f'{AB3}/lengthexp/eval/C_s{101 + s}') for s in SEEDS},
    }
    groups = {g: {s: e for s, e in v.items() if e is not None and len(e['units']) >= 1} for g, v in groups.items()}
    anchors = {'goals pretrain s0': load_eval(os.path.join(HERE, 'eval', 'pretrain_goals_s0'))}
    main_groups = [g for g in ('goals', 'abs') if groups.get(g)]
    res = {'groups': {g: sorted(v) for g, v in groups.items()}}
    md = ['---', 'written_on: 2026-09-25', '---', '', '# Goals variant vs abs baseline: report', '',
          f'Generated by `analyse.py`. goals: {sorted(groups["goals"])}; abs (repr/): {sorted(groups["abs"])}; arm C reference: '
          f'{sorted(groups["arm C (frozen pretrain, reference)"])}. Seeds are the pretraining seeds (RL seeds 101 / 102 / 103). '
          'k = 32 samples per theorem at T = 0.8; pass@k unbiased from n_ok of k, mean over theorems. Units evaluated on a '
          f'different theorem draw than goals/testsets (the lengthexp final draw) are left out: {len(STALE)} ({", ".join(STALE[:12])}'
          f'{" ..." if len(STALE) > 12 else ""}).', '']

    # ---- 1 pass@k
    tab = {}
    for g, runs in groups.items():
        for s, e in runs.items():
            for L in LENGTHS:
                rows = e['units'].get(('std', L), [])
                tab[(g, s, L)] = (passk(rows, 1), passk(rows, 4))
    res['pass'] = {f'{g}|{s}|{L}': v for (g, s, L), v in tab.items()}
    for k, ki in (('pass@1', 0), ('pass@4', 1)):
        md += [f'## {"1" if ki == 0 else "1b"}. {k} per length (mean [min, max] over seeds)', '']
        rows = []
        for g, runs in groups.items():
            rows.append([g] + [mrange([tab[(g, s, L)][ki] for s in runs]) for L in LENGTHS])
        md += [table(['group'] + [f'L{L}' for L in LENGTHS], rows), '']

    # ---- 2 slopes
    md += ['## 2. Log-odds drop per extra line past 11 (logistic fit over L = 12..18 per run)', '']
    fits = {}
    rows = []
    for g, runs in groups.items():
        for s, e in sorted(runs.items()):
            f = fit_run(e, 12, 18, a.boot, rng)
            fits[(g, s)] = f
        bs = [fits[(g, s)]['b'] for s in sorted(runs) if fits[(g, s)]['ok']]
        m, lo, hi = tint(bs)
        per = ', '.join(fmt(fits[(g, s)]['b'], 2) if fits[(g, s)]['ok'] else f"n/a ({fits[(g, s)]['why']})" for s in sorted(runs))
        rows.append([g, per, fmt(m, 2) if bs else '-', f'[{fmt(lo, 2)}, {fmt(hi, 2)}]',
                     fmt(np.mean([fits[(g, s)]['boot_se'] for s in runs if fits[(g, s)]['ok']]) if bs else float('nan'), 2)])
    md += [table(['group', 'slope per seed (s0, s1, s2)', 'mean', '95% t-interval', 'mean bootstrap SE (theorems)'], rows), '']
    res['fits'] = {f'{g}|{s}': f for (g, s), f in fits.items()}
    if len(main_groups) == 2:
        pairs = [s for s in SEEDS if s in groups['goals'] and s in groups['abs'] and fits[('goals', s)]['ok'] and fits[('abs', s)]['ok']]
        diffs = [fits[('goals', s)]['b'] - fits[('abs', s)]['b'] for s in pairs]
        m, lo, hi = tint(diffs)
        ga = [fits[('goals', s)]['b'] for s in pairs]
        ab = [fits[('abs', s)]['b'] for s in pairs]
        p, n_lab = perm_test(ga, ab) if len(pairs) >= 2 else (float('nan'), 0)
        res['slope_diff'] = {'pairs': pairs, 'diffs': diffs, 'mean': m, 'ci': [lo, hi], 'perm_p': p, 'relabelings': n_lab}
        md += [f'Paired difference goals - abs (per seed pair {pairs}): ' + ', '.join(fmt(x, 2) for x in diffs) +
               f'; mean {fmt(m, 2)}, 95% t-interval [{fmt(lo, 2)}, {fmt(hi, 2)}]; exact permutation p = {fmt(p, 3)} '
               f'({n_lab} relabelings; the smallest possible two-sided p with 3 + 3 runs is 0.1). A positive difference '
               'means goals loses less log-odds per extra line (flatter).', '']
    # pooled pass@1 over L >= 12 and at L12 / L13 with a theorem bootstrap of the paired difference
    md += ['### Pooled pass@1 over L = 12..18 and L = 12..14 (the lengths where abs is still above zero)', '']
    rows = []
    pooled = {}
    for g, runs in groups.items():
        for win, Ls in (('12-18', range(12, 19)), ('12-14', range(12, 15))):
            vals = []
            for s, e in runs.items():
                rr = [r for L in Ls for r in e['units'].get(('std', L), [])]
                vals.append(passk(rr, 1))
            pooled[(g, win)] = vals
        rows.append([g, mrange(pooled[(g, '12-18')], lambda x: fmt(x, 4)), mrange(pooled[(g, '12-14')], lambda x: fmt(x, 4))])
    md += [table(['group', 'pass@1 L12-18', 'pass@1 L12-14'], rows), '']
    res['pooled'] = {f'{g}|{w}': v for (g, w), v in pooled.items()}

    # ---- 3 attempt lengths
    md += ['## 3. Attempt length: share of attempts at least L_true lines long (all samples, stripped proofs)', '']
    rows = []
    share = {}
    for g, runs in groups.items():
        cells = []
        for L in LENGTHS:
            vals, meds = [], []
            for s, e in runs.items():
                ln = e['lines'].get(('std', L))
                if not ln:
                    continue
                allx = [x for v in ln.values() for x in v]
                vals.append(np.mean([x >= L for x in allx]))
                meds.append(float(np.median(allx)))
            share[(g, L)] = (vals, meds)
            cells.append(f'{mrange(vals, pct)} / {fmt(np.mean(meds), 1) if meds else "-"}' if vals else '-')
        rows.append([g] + cells)
    md += ['Cell: share of attempts with written lines >= L_true (mean [min, max] over seeds) / median written lines.', '',
           table(['group'] + [f'L{L}' for L in LENGTHS], rows), '']
    res['length_share'] = {f'{g}|{L}': v for (g, L), v in share.items()}

    # ---- 4 first-error position
    md += ['## 4. Where failed attempts go wrong', '',
           'Cell: mean failure position (first rejected line; the QED slot for an early QED) / share early QED / share '
           'out of tokens, pooled over seeds.', '']
    rows = []
    fpos = {}
    for g, runs in groups.items():
        cells = []
        for L in LENGTHS:
            pos, kinds = [], collections.Counter()
            for s, e in runs.items():
                p, k = fail_positions(e['units'].get(('std', L), []))
                pos += p
                kinds.update(k)
            n = sum(kinds.values())
            fpos[(g, L)] = {'mean_pos': float(np.mean(pos)) if pos else None, 'kinds': dict(kinds)}
            cells.append(f"{fmt(np.mean(pos), 1) if pos else '-'} / {pct(kinds['early_qed'] / n if n else float('nan'), 0)} / "
                         f"{pct(kinds['no_qed'] / n if n else float('nan'), 1)}")
        rows.append([g] + cells)
    md += [table(['group'] + [f'L{L}' for L in LENGTHS], rows), '']
    res['fail_positions'] = {f'{g}|{L}': v for (g, L), v in fpos.items()}

    # ---- 4b hazard on the long sets
    md += ['## 4b. Per-line failure hazard on the long sets', '',
           'P(first error at line p | lines 1..p-1 valid), seeds pooled, at positions with >= 100 samples at risk '
           '(lengthexp/analyse.py definition). The training maximum is 11 lines: a spike at p = 12 would mean attempts '
           'that survive die where training stopped. Cell: hazard / samples at risk.', '']
    haz = {}
    for L in (12, 13, 15, 16, 17, 18):
        rows = []
        for g, runs in groups.items():
            fails, risk = collections.Counter(), collections.Counter()
            for s, e in runs.items():
                f1, r1 = hazard_counts(e['units'].get(('std', L), []))
                fails.update(f1)
                risk.update(r1)
            haz[(g, L)] = (fails, risk)
            rows.append([g] + [f"{fails.get(p, 0) / risk[p]:.2f} / {risk[p]}" if risk.get(p, 0) >= 100 else '-' for p in range(1, 17)])
        md += [f'**L{L}**', '', table(['group'] + [f'p{p}' for p in range(1, 17)], rows), '']
    res['hazard'] = {f'{g}|{L}': {'fails': dict(f), 'risk': dict(r)} for (g, L), (f, r) in haz.items()}
    fig, axs = plt.subplots(1, 6, figsize=(19, 3.4), sharey=True)
    for j, L in enumerate((12, 13, 15, 16, 17, 18)):
        ax = axs[j]
        for g in groups:
            if not groups[g]:
                continue
            f, r = haz[(g, L)]
            ps = [p for p in sorted(r) if r[p] >= 100]
            ax.plot(ps, [f.get(p, 0) / r[p] for p in ps], '--' if g.startswith('arm C') else '-', marker='o', ms=3,
                    color={'goals': '#c0392b', 'abs': '#2c3e50'}.get(g, '#95a5a6'), label=g)
        ax.axvline(TRAIN_MAX + 1, color='k', lw=0.7, ls=':')
        ax.set_title(f'test L{L}', fontsize=9)
        ax.set_xlabel('line position p')
    axs[0].set_ylabel('P(fail at p | 1..p-1 valid)')
    axs[0].legend(fontsize=7)
    fig.suptitle('Per-line failure hazard (positions with >= 100 samples at risk; dotted = training max + 1)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(a.out, 'fig_hazard.png'), dpi=120)
    md += ['![hazard](fig_hazard.png)', '']

    # ---- 5 in-distribution
    md += ['## 5. In-distribution checks', '']
    rows = []
    indist = {}
    for g, pre_root, run_root, lab in (('goals', os.path.join(HERE, 'pretrain'), os.path.join(HERE, 'runs'), 'goals_s{}'),
                                        ('abs', a.abs_pretrain_root, a.abs_runs_root, 'abs_s{}')):
        pre = [jload(os.path.join(pre_root, lab.format(s), 'summary.json')) for s in SEEDS]
        rl = [jload(os.path.join(run_root, lab.format(s), 'summary.json')) for s in SEEDS]
        hp = [x.get('heldout_greedy') for x in pre if x]
        hp39 = [x.get('heldout_greedy_039') for x in pre if x]
        hr = [x.get('heldout_greedy') for x in rl if x]
        vl = [x.get('val_loss') for x in pre if x]
        p611 = []
        for s, e in groups.get(g, {}).items():
            rr = [r for L in range(6, 12) for r in e['units'].get(('std', L), [])]
            p611.append(passk(rr, 1))
        indist[g] = {'pretrain_val_loss': vl, 'pretrain_heldout_greedy': hp, 'pretrain_heldout_greedy_039': hp39, 'rl_heldout_greedy': hr, 'pass1_L6_11': p611}
        rows.append([g, mrange(vl, lambda x: fmt(x, 4)), mrange(hp), mrange(hp39), mrange(hr), mrange(p611, lambda x: fmt(x, 4))])
    md += [table(['group', 'pretrain val loss (039 slice: 1,024 two-line proofs, no annotations in either variant)',
                  'held-out greedy after pretraining (harness slice, 1,000)', '039 slice (2,000)',
                  'held-out greedy after RL', 'pass@1 pooled L6-11'], rows), '',
           'goals verifies the stripped proofs, with max_new 768 instead of 512 for the greedy checks (DESIGN.md 4).', '']
    res['indist'] = indist

    # ---- 6 annotation behaviour (goals)
    if groups.get('goals'):
        md += ['## 6. What the goals model writes (evaluation samples)', '',
               'Cell: annotation groups per sample / share of samples with a malformed group / share of correct samples whose '
               'annotation equals the canonical annotation of their own proof exactly / share of the canonical groups '
               'written exactly, pooled over seeds.', '']
        rows = []
        annres = {}
        for L in LENGTHS:
            c = collections.Counter()
            for s, e in groups['goals'].items():
                sm = e['summ'].get(('std', L))
                if sm and 'annotation' in sm:
                    c.update(sm['annotation']['counts'])
            annres[L] = dict(c)
            nok = c['ok']
            rows.append([f'L{L}', fmt(c['groups'] / max(1, c['samples']), 2), pct(c['with_malformed'] / max(1, c['samples']), 2),
                         pct(c['exact'] / nok) if nok else '-', pct(c['matched'] / c['n_canon']) if c['n_canon'] else '-',
                         fmt(c['extra'] / nok, 2) if nok else '-', nok])
        md += [table(['length', 'groups / sample', 'malformed', 'correct: exact canonical', 'correct: canonical groups matched',
                      'correct: extra groups / proof', 'correct samples'], rows), '']
        res['annotation_eval'] = annres
        # RL side
        rows = []
        for s in SEEDS:
            sm = jload(os.path.join(HERE, 'runs', f'goals_s{s}', 'summary.json'))
            if not sm:
                continue
            pr = sm.get('rl_stats', {}).get('per_round', [])
            if not pr:
                continue
            for lab, r in (('first', pr[0]), ('middle', pr[len(pr) // 2]), ('last', pr[-1])):
                an = r.get('annotation', {})
                rows.append([f's{s} round {r["round"]} ({lab})', fmt(an.get('groups_per_sample'), 2), pct(an.get('samples_with_malformed'), 2),
                             pct(an.get('exact_canonical')), pct(an.get('canon_groups_matched')), pct(an.get('written_groups_pending_goal'))])
        if rows:
            md += ['RL rounds (the samples the model trains on; consistency on <= 1,500 verified samples per round):', '',
                   table(['run / round', 'groups / sample', 'malformed', 'verified: exact canonical', 'canonical groups matched',
                          'written groups naming a pending goal'], rows), '']

    # ---- 7 training side
    md += ['## 7. Training side (RL runs)', '']
    rows = []
    train = {}
    for g, run_root, lab in (('goals', os.path.join(HERE, 'runs'), 'goals_s{}'), ('abs', a.abs_runs_root, 'abs_s{}')):
        for s in SEEDS:
            sm = jload(os.path.join(run_root, lab.format(s), 'summary.json'))
            if not sm:
                continue
            rs = sm.get('rl_stats', {})
            pr = rs.get('per_round', [])
            last = pr[-1] if pr else {}
            tb = rs.get('targets_by_len', {})
            sol = last.get('solved_cum_all_by_len', {})
            ft_rows = sum(x.get('ft_rows', 0) for x in pr)
            train[f'{g}|{s}'] = {'rounds': rs.get('rounds'), 'samples': rs.get('total_samples'), 'steps': rs.get('total_train_steps'),
                                 'frac': sm['compute']['frac'], 'rl_secs': sm['rl_secs'], 'ft_rows': ft_rows,
                                 'solved': {L: sol.get(L, 0) / tb[L] for L in tb if tb.get(L)}}
            rows.append([f'{g} s{s}', rs.get('rounds'), rs.get('total_samples'), rs.get('total_train_steps'), pct(sm['compute']['frac']),
                         f"{sm['rl_secs']:.0f}", f'{ft_rows:.3g}', fmt(sm.get('heldout_greedy')),
                         ', '.join(f'L{L} {sol.get(L, 0) / tb[L]:.2f}' for L in sorted(tb, key=int) if tb.get(L))])
    md += [table(['run', 'rounds', 'samples', 'optimizer steps', 'compute used', 'rl secs', 'fine-tune rows', 'held-out greedy',
                  'target subset solved at the end, by L_true'], rows), '']
    res['training'] = train

    # ---- 8 matched
    md += ['## 8. Matched-construction control (pass@1)', '']
    rows = []
    for g, runs in groups.items():
        rows.append([g] + [mrange([passk(e['units'].get(('matched', L), []), 1) for s, e in runs.items() if e['units'].get(('matched', L))])
                           for L in range(6, 15)])
    md += [table(['group'] + [f'L{L}' for L in range(6, 15)], rows), '']

    # ---- mechanism (analysis/mechanism.py output, CPU)
    mfn = os.path.join(HERE, 'analysis', 'mechanism.md')
    if os.path.isfile(mfn):
        md += ['## 8b. Mechanism: conjunction theorems (analysis/mechanism.py)', '', open(mfn).read(), '']

    # ---- anchor
    an = anchors['goals pretrain s0']
    if an:
        md += ['## 9. Pretrained checkpoint (goals s0, no RL) as an anchor: pass@1', '',
               table(['L'] + [f'L{L}' for L in LENGTHS], [['pass@1'] + [fmt(passk(an['units'].get(('std', L), []), 1)) for L in LENGTHS]]), '']

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {'goals': '#c0392b', 'abs': '#2c3e50', 'arm C (frozen pretrain, reference)': '#95a5a6'}
    for g, runs in groups.items():
        if not runs:
            continue
        ls = '--' if g.startswith('arm C') else '-'
        m1 = [np.mean([tab[(g, s, L)][0] for s in runs]) for L in LENGTHS]
        lo = [min(tab[(g, s, L)][0] for s in runs) for L in LENGTHS]
        hi = [max(tab[(g, s, L)][0] for s in runs) for L in LENGTHS]
        ax[0].plot(LENGTHS, m1, ls, color=cols[g], label=g, marker='o', ms=3)
        ax[0].fill_between(LENGTHS, lo, hi, color=cols[g], alpha=0.15)
        lg = [math.log(max(x, 1e-4) / (1 - min(x, 1 - 1e-4))) for x in m1]
        ax[1].plot(LENGTHS, lg, ls, color=cols[g], label=g, marker='o', ms=3)
        sh = [np.mean(share[(g, L)][0]) if share[(g, L)][0] else float('nan') for L in LENGTHS]
        ax[2].plot(LENGTHS, sh, ls, color=cols[g], label=g, marker='o', ms=3)
    for x in ax:
        x.axvline(TRAIN_MAX + 0.5, color='k', lw=0.6, ls=':')
        x.set_xlabel('test length L_true')
    ax[0].set_ylabel('pass@1')
    ax[1].set_ylabel('log-odds of pass@1 (floored at 1e-4)')
    ax[2].set_ylabel('share of attempts >= L_true lines')
    ax[0].legend(fontsize=8)
    ax[0].set_title('pass@1 (band: seed range)')
    ax[1].set_title('log-odds')
    ax[2].set_title('attempt length')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'fig_main.png'), dpi=130)
    md += ['## Figures', '', '![main](fig_main.png)', '']
    open(os.path.join(a.out, 'report.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1, default=float)
    print('wrote', os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    sys.exit(main())
