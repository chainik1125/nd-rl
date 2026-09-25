#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Length-generalisation analysis: report.md + figures from eval/<label>/ and runs/<label>/.

    uv run --script analyse.py [--eval-root eval] [--runs-root runs] [--out report]
                               [--train-max ARM=N ...] [--run-dir LABEL=PATH ...] [--boot 200]

Labels: 'pretrain' (training max 6) and '<arm>_s<seed>'; arms A, B, C have training max 7, 9, 11 (common.ARMS).
--train-max adds other arms (tests: --train-max n092=11). Sections (numbers as in the README):
  1 pass@1 / pass@4 / pass@32 per arm and length (mean and range over seeds)
  2 pass@1 against d = length - training max; logistic slope per run over d >= 1 (and over the common window
    L >= 12, and on generator theorems only); seed-level intervals; tests of slope differences between arms
  3 lengths of correct proofs vs L_true (label too high?)
  4 where failed samples go wrong (first rejected line; early QED; out of tokens; per-line hazard)
  5 training-side solve rates by length over rounds (runs/<label>/metrics.jsonl)
  6 matched-construction control (if eval has 'matched' units)
"""
import argparse, collections, datetime, itertools, json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ARMS, EVAL, K, LENGTHS, PRETRAIN_MAX, REPORT, RUNS, TESTSETS, pass_at_k, read_jsonl  # noqa: E402

COMMON_LO = max(ARMS.values()) + 1          # common window: L >= 12 (d >= 1 for every arm)
ARM_COLORS = {'pretrain': '#8a8984', 'A': '#86b6ef', 'B': '#2a78d6', 'C': '#0d366b', 'D': '#eb6834'}
EXTRA_COLORS = ['#eb6834', '#1baf7a', '#e87ba4', '#4a3aa7']
MARKERS = {'pretrain': 'o', 'A': 's', 'B': '^', 'C': 'D', 'D': 'v'}
KIND_COLORS = {'bad_line': '#2a78d6', 'index': '#eb6834', 'early_qed': '#1baf7a', 'no_qed': '#eda100'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'lines.linewidth': 2})


# ------------------------------------------------------------------ loading
def label_arm(label, extra):
    if label == 'pretrain':
        return 'pretrain', None
    a, _, s = label.rpartition('_s')
    if s.isdigit() and (a in ARMS or a in extra):
        return a, int(s)
    return None, None


STALE = []


def current_unit_shas():
    """sha of each unit's theorems in the current testsets/ (evaluate.py's unit_sha: sorted names + prompts)."""
    from common import sha256_str
    by = collections.defaultdict(list)
    for f in ('std.jsonl', 'matched.jsonl'):
        fn = os.path.join(TESTSETS, f)
        if os.path.isfile(fn):
            for r in read_jsonl(fn):
                by[(r['set'], r['L_true'])].append(r)
    return {k: sha256_str('\n'.join(r['name'] + '\t' + r['prompt'] for r in sorted(v, key=lambda r: r['name']))) for k, v in by.items()}


def load_evals(root, extra):
    ev = {}
    if not os.path.isdir(root):
        return ev
    cur = current_unit_shas()
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        arm, seed = label_arm(label, extra)
        if arm is None or not os.path.isdir(d):
            continue
        units, summ = {}, {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.summary.json'):
                continue
            u = fn[:-len('.summary.json')]
            s, _, L = u.rpartition('_L')
            rows_fn = os.path.join(d, u + '.jsonl')
            if not os.path.isfile(rows_fn):
                continue
            sm = json.load(open(os.path.join(d, fn)))
            if cur and sm.get('unit_sha') != cur.get((s, int(L))):
                STALE.append(f'{label}:{u}')          # evaluated on a different theorem draw: left out
                continue
            units[(s, int(L))] = read_jsonl(rows_fn)
            summ[(s, int(L))] = sm
        if units:
            meta = json.load(open(os.path.join(d, 'meta.json'))) if os.path.isfile(os.path.join(d, 'meta.json')) else {}
            ev[label] = {'arm': arm, 'seed': seed, 'units': units, 'summ': summ, 'meta': meta}
    return ev


def tmax(arm, extra):
    return PRETRAIN_MAX if arm == 'pretrain' else ARMS.get(arm, extra.get(arm))


def arm_order(ev, extra):
    arms = sorted({e['arm'] for e in ev.values()}, key=lambda a: (tmax(a, extra), a))
    return arms


def color(arm, arms):
    if arm in ARM_COLORS:
        return ARM_COLORS[arm]
    others = [a for a in arms if a not in ARM_COLORS]
    return EXTRA_COLORS[others.index(arm) % len(EXTRA_COLORS)]


def select(rows, gen_only=False):
    return [r for r in rows if r.get('source') != 'textbook'] if gen_only else rows


# ------------------------------------------------------------------ statistics
def passk(rows, k):
    if not rows:
        return float('nan')
    return float(np.mean([pass_at_k(r['k'], r['n_ok'], k) for r in rows]))


def fit_logistic(d, s, n, iters=100):
    """Grouped-binomial logistic regression logit p = a + b d by Newton-Raphson.
    Returns dict(a, b, se_b, ok, why). The MLE is finite only if >= 2 distinct d have successes."""
    d, s, n = (np.asarray(x, float) for x in (d, s, n))
    m = n > 0
    d, s, n = d[m], s[m], n[m]
    out = dict(a=float('nan'), b=float('nan'), se_b=float('nan'), ok=False, n_pts=int(len(d)),
               n_pos=int((s > 0).sum()), why='')
    if len(d) < 2:
        out['why'] = 'fewer than 2 lengths'
        return out
    if (s > 0).sum() < 2:
        out['why'] = 'successes at fewer than 2 lengths (slope not estimable)'
        return out
    if (s < n).sum() == 0:
        out['why'] = 'no failures'
        return out
    p0 = np.clip(s.sum() / n.sum(), 1e-6, 1 - 1e-6)
    beta = np.array([math.log(p0 / (1 - p0)), 0.0])
    X = np.stack([np.ones_like(d), d], 1)
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p)
        g = X.T @ (s - n * p)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            out['why'] = 'singular'
            return out
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (n * p * (1 - p))[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    out.update(a=float(beta[0]), b=float(beta[1]), se_b=float(math.sqrt(max(cov[1, 1], 0))), ok=bool(abs(beta[1]) < 30))
    if not out['ok']:
        out['why'] = 'diverged'
    return out


def run_points(e, Ls, gen_only=False, set_name='std'):
    """Per-length (L, successes, trials, per-theorem n_ok array, k) for one evaluated checkpoint."""
    pts = []
    for L in Ls:
        rows = select(e['units'].get((set_name, L), []), gen_only)
        if rows:
            nok = np.array([r['n_ok'] for r in rows], float)
            kk = np.array([r['k'] for r in rows], float)
            pts.append((L, nok.sum(), kk.sum(), nok, kk))
    return pts


def fit_run(e, lo, hi, tm, boot, rng, gen_only=False):
    pts = run_points(e, range(lo, hi + 1), gen_only)
    f = fit_logistic([p[0] - tm for p in pts], [p[1] for p in pts], [p[2] for p in pts])
    f['lengths'] = [p[0] for p in pts]
    f['boot_se'] = float('nan')
    if f['ok'] and boot:
        bs = []
        for _ in range(boot):
            ss, nn = [], []
            for (_, _, _, nok, kk) in pts:
                ix = rng.integers(0, len(nok), len(nok))
                ss.append(nok[ix].sum())
                nn.append(kk[ix].sum())
            g = fit_logistic([p[0] - tm for p in pts], ss, nn)
            if g['ok']:
                bs.append(g['b'])
        if len(bs) > 10:
            f['boot_se'] = float(np.std(bs, ddof=1))
    return f


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) (continued fraction, Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c if abs(c) > 1e-300 else 1 + aa / 1e-300
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c if abs(c) > 1e-300 else 1 + aa / 1e-300
            de = d * c
            h *= de
            if abs(de - 1) < 1e-12:
                break
        return h
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * cf(a, b, x) / a
    return 1 - math.exp(lbt) * cf(b, a, 1 - x) / b


def t_p_two_sided(t, df):
    if not np.isfinite(t) or df <= 0:
        return float('nan')
    return betainc(df / 2, 0.5, df / (df + t * t))


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return float('nan'), float('nan')
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    if vx + vy == 0:
        return float('nan'), float('nan')
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return t, t_p_two_sided(t, df)


def labelings(sizes):
    """All assignments of range(sum(sizes)) to groups of the given sizes (as lists of index lists)."""
    n = sum(sizes)
    def rec(rem, sizes):
        if not sizes:
            yield []
            return
        for c in itertools.combinations(rem, sizes[0]):
            rest = [i for i in rem if i not in c]
            for tail in rec(rest, sizes[1:]):
                yield [list(c)] + tail
    return rec(list(range(n)), list(sizes))


def slope_tests(by_arm, extra):
    """by_arm: {arm: [slopes]} for RL arms. Trend (slope-of-slopes vs training max) and omnibus F, both by
    exact permutation over all relabelings of the runs; pairwise Welch t and exact permutation."""
    arms = [a for a in by_arm if len(by_arm[a]) >= 1]
    vals = [v for a in arms for v in by_arm[a]]
    xs = [tmax(a, extra) for a in arms for _ in by_arm[a]]
    sizes = [len(by_arm[a]) for a in arms]
    res = {'arms': arms, 'n_runs': len(vals)}
    if len(arms) < 2 or len(vals) < 3:
        return res
    v = np.array(vals)
    xv = np.array(xs, float)

    def trend(vv):
        xc = xv - xv.mean()
        return float((xc * (vv - vv.mean())).sum() / (xc ** 2).sum())

    def fstat(groups):
        gm = [np.mean(g) for g in groups]
        grand = np.mean(np.concatenate(groups))
        ssb = sum(len(g) * (m - grand) ** 2 for g, m in zip(groups, gm))
        ssw = sum(((g - m) ** 2).sum() for g, m in zip(groups, gm))
        dfb, dfw = len(groups) - 1, sum(len(g) for g in groups) - len(groups)
        return (ssb / dfb) / (ssw / dfw) if ssw > 0 and dfw > 0 else float('inf')

    obs_t = trend(v)
    groups = [np.array(by_arm[a]) for a in arms]
    obs_f = fstat(groups)
    nt = nf = tot = 0
    for lab in labelings(sizes):
        perm = np.empty_like(v)
        order = [i for g in lab for i in g]
        perm = v[order]
        tot += 1
        if abs(trend(perm)) >= abs(obs_t) - 1e-12:
            nt += 1
        gs, i0 = [], 0
        for sz in sizes:
            gs.append(perm[i0:i0 + sz]); i0 += sz
        if fstat(gs) >= obs_f - 1e-12:
            nf += 1
    res.update(trend_per_line_of_train_max=obs_t, trend_p_perm=nt / tot, F=obs_f, F_p_perm=nf / tot, n_labelings=tot)
    pw = {}
    for a, b in itertools.combinations(arms, 2):
        t, p = welch(by_arm[a], by_arm[b])
        x, y = by_arm[a], by_arm[b]
        allv = np.array(x + y)
        obs = abs(np.mean(x) - np.mean(y))
        cnt = tt = 0
        for c in itertools.combinations(range(len(allv)), len(x)):
            rest = [i for i in range(len(allv)) if i not in c]
            tt += 1
            cnt += abs(allv[list(c)].mean() - allv[rest].mean()) >= obs - 1e-12
        pw[f'{a}-{b}'] = dict(diff=float(np.mean(x) - np.mean(y)), welch_t=t, welch_p=p, perm_p=cnt / tt, min_perm_p=2 / tt if len(x) == len(y) else 1 / tt)
    res['pairwise'] = pw
    return res


# ------------------------------------------------------------------ helpers for output
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '-'
    return f'{x:.{nd}f}'


def fmt_p(x):
    if x is None or not np.isfinite(x):
        return '-'
    return f'{x:.4f}' if x >= 1e-4 else f'{x:.1e}'


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def logit(p):
    return math.log(p / (1 - p))


def savefig(fig, out, name):
    fn = os.path.join(out, name)
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return name


# ------------------------------------------------------------------ sections
def section_pass(ev, arms, extra, lengths):
    """Section 1: pass@k per arm, length (mean and range over seeds)."""
    tab = {}
    for arm in arms:
        labs = [l for l, e in ev.items() if e['arm'] == arm]
        for L in lengths:
            vals = {k: [] for k in (1, 4, 32)}
            for l in labs:
                rows = ev[l]['units'].get(('std', L))
                if rows:
                    for k in vals:
                        vals[k].append(passk(rows, k))
            if vals[1]:
                tab[(arm, L)] = {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) for k, v in vals.items()}
    return tab


def pass_md(tab, arms, lengths, k):
    hdr = ['arm (train max)'] + [f'L{L}' for L in lengths]
    rows = []
    for arm in arms:
        r = [f'{arm} ({tmax(arm, EXTRA)})']
        for L in lengths:
            v = tab.get((arm, L))
            if not v:
                r.append('')
                continue
            m, lo, hi, n = v[k]
            r.append(f'{m:.3f}' + (f' [{lo:.3f}, {hi:.3f}]' if n > 1 else ''))
        rows.append(r)
    return md_table(hdr, rows)


def fig_main(ev, arms, extra, lengths, fits, out, gen_only=False):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for arm in arms:
        tm = tmax(arm, extra)
        labs = [l for l, e in ev.items() if e['arm'] == arm]
        c, mk = color(arm, arms), MARKERS.get(arm, 'o')
        xs, m, lo, hi, zero, ntot = [], [], [], [], [], []
        for L in lengths:
            per = []
            for l in labs:
                pts = run_points(ev[l], [L], gen_only)
                if pts:
                    per.append((pts[0][1], pts[0][2]))
            if not per:
                continue
            ps = [s / n for s, n in per]
            xs.append(L); m.append(np.mean(ps)); lo.append(min(ps)); hi.append(max(ps))
            zero.append(sum(s for s, _ in per) == 0); ntot.append(sum(n for _, n in per))
        if not xs:
            continue
        xs, m, lo, hi, z, ntot = map(np.array, (xs, m, lo, hi, zero, ntot))
        d = xs - tm
        lab = f'{arm} (train max {tm})' + (f', {len(labs)} seeds' if len(labs) > 1 else '')
        axs[0].errorbar(d, m, yerr=[m - lo, hi - m], color=c, marker=mk, ms=5, capsize=2, label=lab)
        floor = 0.5 / ntot                       # half a success over every sample pooled at that length
        fl = lambda v: np.array([logit(max(x, f)) for x, f in zip(v, floor)])
        lg, lgl, lgh = fl(m), fl(lo), fl(hi)
        nz = ~z
        axs[1].errorbar(d[nz], lg[nz], yerr=[lg[nz] - lgl[nz], lgh[nz] - lg[nz]], color=c, marker=mk, ms=5, capsize=2)
        axs[1].plot(d[z], lg[z], mk, color=c, ms=5, mfc='white', ls='none')
        f = fits.get(arm)
        if f and np.isfinite(f.get('b_mean', np.nan)):
            dd = np.array([1, max(d)])
            axs[1].plot(dd, f['a_mean'] + f['b_mean'] * dd, '--', color=c, lw=1)
        axs[2].plot(xs[nz], m[nz], color=c, marker=mk, ms=5)
        axs[2].plot(xs[z], floor[z], mk, color=c, ms=5, mfc='white', ls='none')
    axs[0].set_xlabel('d = test length - training max'); axs[0].set_ylabel('pass@1')
    axs[1].set_xlabel('d = test length - training max'); axs[1].set_ylabel('log-odds of pass@1')
    axs[2].set_xlabel('test length L_true'); axs[2].set_ylabel('pass@1 (log scale)'); axs[2].set_yscale('log')
    for ax in axs[:2]:
        ax.axvline(0.5, color=INK2, lw=0.8, ls=':')
    axs[0].legend(loc='upper right')
    t = 'generator theorems only' if gen_only else 'all test theorems'
    axs[1].set_title('hollow = no correct sample (drawn at half a success); dashed = mean fit over d >= 1', fontsize=8, color=INK2)
    for ax in axs:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle(f'pass@1 (k={K}, T=0.8) against distance beyond the training length ({t}); bars = range over seeds', fontsize=10)
    return savefig(fig, out, 'fig_main_gen.png' if gen_only else 'fig_main.png')


def section_fits(ev, arms, extra, lengths, boot, seed=0):
    """Section 2: per-run logistic fits for three windows."""
    rng = np.random.default_rng(seed)
    hi = max(lengths)
    windows = {'own (d >= 1)': (None, False), f'common (L >= {COMMON_LO})': (COMMON_LO, False),
               'own, generator theorems only': (None, True), f'common (L >= {COMMON_LO}), generator only': (COMMON_LO, True)}
    res = {}
    for wname, (lo_fixed, gen) in windows.items():
        per_run, arm_sum = {}, {}
        for arm in arms:
            tm = tmax(arm, extra)
            lo = lo_fixed if lo_fixed is not None else tm + 1
            for l, e in sorted(ev.items()):
                if e['arm'] != arm:
                    continue
                per_run[l] = dict(fit_run(e, lo, hi, tm, boot, rng, gen), arm=arm, seed=e['seed'])
            bs = [per_run[l]['b'] for l in per_run if per_run[l]['arm'] == arm and per_run[l]['ok']]
            as_ = [per_run[l]['a'] for l in per_run if per_run[l]['arm'] == arm and per_run[l]['ok']]
            n_runs = sum(1 for l in per_run if per_run[l]['arm'] == arm)
            s = dict(n=len(bs), n_runs=n_runs, b_mean=float(np.mean(bs)) if bs else float('nan'),
                     a_mean=float(np.mean(as_)) if as_ else float('nan'))
            if len(bs) >= 2:
                sd = float(np.std(bs, ddof=1))
                h = T975.get(len(bs) - 1, 1.96) * sd / math.sqrt(len(bs))
                s.update(sd=sd, ci=(s['b_mean'] - h, s['b_mean'] + h), range=(min(bs), max(bs)))
            arm_sum[arm] = s
        rl = {a: [per_run[l]['b'] for l in per_run if per_run[l]['arm'] == a and per_run[l]['ok']]
              for a in arms if a in ARMS}                      # the length-trend tests use A/B/C only
        res[wname] = dict(per_run=per_run, arms=arm_sum, tests=slope_tests({a: v for a, v in rl.items() if v}, extra))
        dc = {a: [per_run[l]['b'] for l in per_run if per_run[l]['arm'] == a and per_run[l]['ok']] for a in ('C', 'D')}
        if len(dc['C']) >= 2 and len(dc['D']) >= 2:
            t, p = welch(dc['D'], dc['C'])
            allv = np.array(dc['D'] + dc['C']); obs = abs(np.mean(dc['D']) - np.mean(dc['C'])); cnt = tt = 0
            for c in itertools.combinations(range(len(allv)), len(dc['D'])):
                rest = [i for i in range(len(allv)) if i not in c]
                tt += 1; cnt += abs(allv[list(c)].mean() - allv[rest].mean()) >= obs - 1e-12
            res[wname]['dc'] = dict(diff=float(np.mean(dc['D']) - np.mean(dc['C'])), welch_p=p, perm_p=cnt / tt, n_D=len(dc['D']), n_C=len(dc['C']))
    return res


def headline_md(ev, arms, extra, lengths, fits):
    own, com = fits['own (d >= 1)']['arms'], fits[f'common (L >= {COMMON_LO})']['arms']
    rows = []
    for arm in arms:
        tm = tmax(arm, extra)
        labs = [l for l, e in ev.items() if e['arm'] == arm]
        at0 = [p[1] / p[2] for l in labs for p in run_points(ev[l], [tm])]
        far = []
        for l in labs:
            pts = run_points(ev[l], [L for L in lengths if L >= COMMON_LO])
            if pts:
                far.append(sum(p[1] for p in pts) / sum(p[2] for p in pts))
        def ci(s):
            return f"{s['b_mean']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if s.get('ci') else fmt(s.get('b_mean'), 2)
        rng_ = lambda v: f'{np.mean(v):.4f}' + (f' [{min(v):.4f}, {max(v):.4f}]' if len(v) > 1 else '') if v else '-'
        rows.append([f'{arm} ({tm})', len(labs), rng_(at0), ci(own.get(arm, {})), ci(com.get(arm, {})), rng_(far)])
    return md_table(['arm (train max)', 'runs', 'pass@1 at its own training max (d = 0)', 'slope over d >= 1 [95% t-interval]',
                     f'slope over L >= {COMMON_LO} [95% t-interval]', f'pass@1 pooled over L >= {COMMON_LO} [seed range]'], rows)


def fits_md(res, arms, extra):
    parts = []
    for wname, r in res.items():
        rows = []
        for arm in arms:
            s = r['arms'].get(arm, {})
            runs = [f"{p['b']:.2f}" if p['ok'] else f"n/a ({p['why']})" for l, p in sorted(r['per_run'].items()) if p['arm'] == arm]
            bse = [p['boot_se'] for p in r['per_run'].values() if p['arm'] == arm and p['ok'] and np.isfinite(p['boot_se'])]
            ci = s.get('ci')
            Ls = sorted({L for p in r['per_run'].values() if p['arm'] == arm for L in p['lengths']})
            rows.append([f'{arm} ({tmax(arm, extra)})', f'{Ls[0]}-{Ls[-1]}' if Ls else '-', ', '.join(runs) or '-',
                         fmt(s.get('b_mean'), 2), f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '-',
                         fmt(float(np.mean(bse)), 2) if bse else '-'])
        t = r['tests']
        tl = []
        if 'trend_p_perm' in t:
            tl.append(f"trend of slope with training max: {t['trend_per_line_of_train_max']:+.3f} log-odds/line per extra "
                      f"training line, exact permutation p = {fmt_p(t['trend_p_perm'])} ({t['n_labelings']} relabelings of "
                      f"{t['n_runs']} runs); omnibus F = {t['F']:.2f}, permutation p = {fmt_p(t['F_p_perm'])}")
            for k, v in t['pairwise'].items():
                tl.append(f"{k}: difference {v['diff']:+.3f}, Welch p = {fmt_p(v['welch_p'])}, exact permutation p = "
                          f"{fmt_p(v['perm_p'])} (smallest possible {v['min_perm_p']:.2f})")
        if r.get('dc'):
            v = r['dc']
            tl.append(f"D vs C (not part of the A/B/C length tests): slope difference D - C {v['diff']:+.3f}, Welch p = "
                      f"{fmt_p(v['welch_p'])}, exact permutation p = {fmt_p(v['perm_p'])} ({v['n_D']} vs {v['n_C']} runs)")
        parts.append(f'**Window: {wname}.**\n\n' + md_table(
            ['arm (train max)', 'lengths used', 'slope per run (log-odds per line)', 'mean', 't-interval over seeds (95%)',
             'mean within-run bootstrap SE'], rows) + ('\n\n' + '\n'.join(f'- {x}' for x in tl) if tl else ''))
    return '\n\n'.join(parts)


def section_prooflen(ev, arms, extra, lengths):
    """Section 3: correct-proof lengths vs L_true."""
    shorter_any = collections.defaultdict(set)     # L -> theorem names with a correct proof shorter than L_true
    tab = {}
    for arm in arms:
        for L in lengths:
            ex, n_thm_short = [], 0
            for l, e in ev.items():
                if e['arm'] != arm:
                    continue
                for r in e['units'].get(('std', L), []):
                    xs = [x - r['L_true'] for x in r['ok_lines']]
                    ex += xs
                    if any(x < 0 for x in xs):
                        shorter_any[L].add(r['name'])
            if ex:
                ex = np.array(ex)
                tab[(arm, L)] = dict(n=len(ex), shorter=float((ex < 0).mean()), equal=float((ex == 0).mean()),
                                     median_excess=float(np.median(ex)), mean_excess=float(ex.mean()), min=int(ex.min()))
    return tab, shorter_any


def prooflen_md(tab, shorter_any, ev, arms, extra, lengths):
    hdr = ['arm'] + [f'L{L}' for L in lengths]
    rows = []
    for arm in arms:
        r = [arm]
        for L in lengths:
            v = tab.get((arm, L))
            r.append(f"{v['shorter']:.3f} / {v['median_excess']:+.0f} (n={v['n']})" if v else '')
        rows.append(r)
    n_thm = {}
    for L in lengths:
        for e in ev.values():
            if ('std', L) in e['units']:
                n_thm[L] = len(e['units'][('std', L)])
                break
    rows.append(['theorems with any shorter correct proof'] + [f'{len(shorter_any.get(L, ()))}/{n_thm.get(L, "-")}' for L in lengths])
    return md_table(hdr, rows)


def fig_prooflen(tab, arms, extra, lengths, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for arm in arms:
        xs = [L for L in lengths if (arm, L) in tab]
        if not xs:
            continue
        c, mk = color(arm, arms), MARKERS.get(arm, 'o')
        axs[0].plot(xs, [tab[(arm, L)]['shorter'] for L in xs], color=c, marker=mk, ms=5, label=arm)
        axs[1].plot(xs, [tab[(arm, L)]['mean_excess'] for L in xs], color=c, marker=mk, ms=5, label=arm)
    axs[0].set_xlabel('test length L_true'); axs[0].set_ylabel('fraction of correct proofs shorter than L_true')
    axs[1].set_xlabel('test length L_true'); axs[1].set_ylabel('mean (lines - L_true) of correct proofs')
    axs[1].axhline(0, color=INK2, lw=0.8)
    top = max([v['shorter'] for v in tab.values()] + [0])
    axs[0].set_ylim(0, max(0.05, 1.15 * top))
    axs[0].legend()
    fig.suptitle('Correct-proof lengths against the L_true label (pooled over seeds)', fontsize=10)
    return savefig(fig, out, 'fig_prooflen.png')


def fail_positions(rows):
    """Failure position counts: bad_line -> its line (index overflow separately), early_qed -> line+1 (the QED
    slot), no_qed counted apart. Also the per-position hazard (failures / samples still valid at that position)."""
    bad, idx, eq, nq = collections.Counter(), collections.Counter(), collections.Counter(), 0
    fails, risk = collections.Counter(), collections.Counter()
    for r in rows:
        for n in r['ok_lines']:
            for p in range(1, n + 2):
                risk[p] += 1
        for kind, line, reason, start in r['fails']:
            if kind == 'bad_line':
                (idx if reason == 'index_overflow' else bad)[line] += 1
                for p in range(1, line + 1):
                    risk[p] += 1
                fails[line] += 1
            elif kind == 'early_qed':
                eq[line + 1] += 1
                for p in range(1, line + 2):
                    risk[p] += 1
                fails[line + 1] += 1
            elif kind == 'no_qed':
                nq += 1
                for p in range(1, line + 1):
                    risk[p] += 1
            else:
                bad[line] += 1
    return bad, idx, eq, nq, fails, risk


def section_fail(ev, arms, extra, lengths):
    tab = {}
    for arm in arms:
        tm = tmax(arm, extra)
        for L in lengths:
            rows = [r for l, e in ev.items() if e['arm'] == arm for r in e['units'].get(('std', L), [])]
            if not rows:
                continue
            bad, idx, eq, nq, fails, risk = fail_positions(rows)
            nf = sum(bad.values()) + sum(idx.values()) + sum(eq.values()) + nq
            pos = [p for c in (bad, idx, eq) for p, n in c.items() for _ in range(n)]
            near = sum(n for c in (bad, idx, eq) for p, n in c.items() if tm <= p <= tm + 2)
            reasons = collections.Counter()
            for r in rows:
                for kind, line, reason, start in r['fails']:
                    reasons[f'{kind}:{reason}'] += 1
            tab[(arm, L)] = dict(n_fail=nf, bad=bad, idx=idx, eq=eq, nq=nq, fails=fails, risk=risk,
                                 mean_pos=float(np.mean(pos)) if pos else float('nan'),
                                 near_own=near / nf if nf else float('nan'),
                                 frac_eq=sum(eq.values()) / nf if nf else float('nan'),
                                 frac_nq=nq / nf if nf else float('nan'),
                                 frac_idx=(sum(idx.values()) + reasons.get('early_qed:index_limit', 0)) / nf if nf else float('nan'),
                                 reasons=reasons)
    return tab


def fail_md(tab, arms, extra, lengths):
    hdr = ['arm (train max)'] + [f'L{L}' for L in lengths]
    rows = []
    for arm in arms:
        r = [f'{arm} ({tmax(arm, extra)})']
        for L in lengths:
            v = tab.get((arm, L))
            r.append(f"{v['mean_pos']:.1f} / {v['near_own']:.2f} / {v['frac_eq']:.2f} / {v['frac_nq']:.2f} / {v['frac_idx']:.2f}" if v else '')
        rows.append(r)
    return md_table(hdr, rows)


def fig_fail(tab, arms, extra, lengths, out):
    Ls = [L for L in lengths if any((a, L) in tab for a in arms)]
    if not Ls:
        return None
    if len(Ls) > 7:
        Ls = [L for L in Ls if L % 2 == 0] or Ls
    fig, axs = plt.subplots(len(arms), len(Ls), figsize=(2.1 * len(Ls) + 1, 1.6 * len(arms) + 1), sharex=True, sharey=True, squeeze=False)
    xmax = max(Ls) + 6
    for i, arm in enumerate(arms):
        tm = tmax(arm, extra)
        for j, L in enumerate(Ls):
            ax = axs[i][j]
            v = tab.get((arm, L))
            if v and v['n_fail']:
                nf = v['n_fail']
                xs = np.arange(1, xmax + 1)
                b = np.array([v['bad'].get(x, 0) for x in xs]) / nf
                ix = np.array([v['idx'].get(x, 0) for x in xs]) / nf
                eq = np.array([v['eq'].get(x, 0) for x in xs]) / nf
                ax.bar(xs, b, width=0.8, color=KIND_COLORS['bad_line'], label='first rejected line')
                ax.bar(xs, ix, bottom=b, width=0.8, color=KIND_COLORS['index'], label='line index past N64')
                ax.bar(xs, eq, bottom=b + ix, width=0.8, color=KIND_COLORS['early_qed'], label='early QED (at the QED slot)')
                ax.bar([xmax + 1.5], [v['nq'] / nf], width=0.8, color=KIND_COLORS['no_qed'], label='no QED (out of tokens)')
            ax.axvline(tm + 0.5, color=INK, lw=0.9, ls='--')
            ax.axvline(L + 0.5, color=INK2, lw=0.8, ls=':')
            if i == 0:
                ax.set_title(f'test L{L}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{arm} (max {tm})')
            ax.grid(False)
    axs[-1][0].set_xlabel('line position of the failure')
    h, lab = axs[0][0].get_legend_handles_labels()
    for row in axs:
        for ax in row:
            hh, ll = ax.get_legend_handles_labels()
            if len(hh) > len(h):
                h, lab = hh, ll
    fig.legend(h, lab, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Where failed samples go wrong (fraction of failed samples; dashed = arm training max, dotted = test length)', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return savefig(fig, out, 'fig_first_error.png')


def fig_hazard(tab, arms, extra, lengths, out, min_risk=100):
    Ls = [L for L in lengths if L >= 8 and any((a, L) in tab for a in arms)]
    if not Ls:
        return None
    if len(Ls) > 6:
        Ls = [L for L in Ls if L % 2 == 0][:6]
    fig, axs = plt.subplots(1, len(Ls), figsize=(2.6 * len(Ls) + 1, 3.2), sharey=True, squeeze=False)
    for j, L in enumerate(Ls):
        ax = axs[0][j]
        for arm in arms:
            v = tab.get((arm, L))
            if not v:
                continue
            ps = [p for p in sorted(v['risk']) if v['risk'][p] >= min_risk]
            h = [v['fails'].get(p, 0) / v['risk'][p] for p in ps]
            c = color(arm, arms)
            ax.plot(ps, h, color=c, marker=MARKERS.get(arm, 'o'), ms=3, label=arm)
            ax.axvline(tmax(arm, extra) + 1, color=c, lw=0.9, ls='--')
        ax.set_title(f'test L{L}', fontsize=9)
        ax.set_xlabel('line position p')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[0][0].set_ylabel('P(fail at p | lines 1..p-1 valid)')
    axs[0][0].legend()
    fig.suptitle(f'Per-line failure hazard (positions with >= {min_risk} samples at risk; dashed = training max + 1, colored by arm)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return savefig(fig, out, 'fig_hazard.png')


def start_md(ev, arms, lengths):
    rows = []
    for arm in arms:
        st = collections.Counter()
        for e in ev.values():
            if e['arm'] != arm:
                continue
            for (s, L), rows_ in e['units'].items():
                if s != 'std':
                    continue
                for r in rows_:
                    for x in r['ok_starts']:
                        st[x] += 1
                    for f in r['fails']:
                        st[f[3]] += 1
        tot = sum(st.values())
        if not tot:
            continue
        vals = [x for x, n in st.items() if x is not None for _ in range(n)]
        q = np.percentile(vals, [50, 90, 99]) if vals else [np.nan] * 3
        room18 = sum(n for x, n in st.items() if x is not None and x + 17 > 64) / tot
        rows.append([arm, f'{q[0]:.0f}', f'{q[1]:.0f}', f'{q[2]:.0f}', f'{room18:.3f}'])
    return md_table(['arm', 'median first line index', '90th pct', '99th pct', 'fraction that cannot fit 18 lines (start > N47)'], rows)


def load_run(d):
    out = {'dir': d, 'sample': [], 'summary': {}, 'found': []}
    fn = os.path.join(d, 'metrics.jsonl')
    if os.path.isfile(fn):
        recs = read_jsonl(fn)
        out['sample'] = [r for r in recs if r.get('phase') == 'sample']
        out['summary'] = next((r for r in recs if r.get('phase') == 'summary'), {})
    ffn = os.path.join(d, 'found_targets.jsonl')
    if os.path.isfile(ffn):
        out['found'] = [{k: r.get(k) for k in ('name', 'L_true', 'written', 'pruned', 'round')} for r in read_jsonl(ffn)]
    sfn = os.path.join(d, 'summary.json')
    if os.path.isfile(sfn):
        out['harness'] = json.load(open(sfn))
    return out


def section_training(labels, run_dirs, extra):
    runs = {}
    for l in labels:
        if l == 'pretrain':
            continue
        d = run_dirs.get(l, os.path.join(RUNS_ROOT, l))
        if os.path.isdir(d):
            runs[l] = load_run(d)
    return runs


def cum_solved(r):
    """{L: [cumulative fraction of the run's target subset at length L solved by the end of round 1..R]}.
    From found_targets.jsonl (first round each target was solved) and the summary's targets_by_len."""
    tb = (r.get('summary') or {}).get('targets_by_len') or {}
    R = max([s.get('round', 0) for s in r.get('sample', [])] + [0])
    first = {}
    for f in r.get('found', []):
        if f.get('round') is not None:
            first[f['name']] = min(first.get(f['name'], 10 ** 9), f['round'])
    Ls = {}
    by = collections.defaultdict(list)
    lt = {f['name']: f['L_true'] for f in r.get('found', [])}
    for n, rd in first.items():
        by[str(lt[n])].append(rd)
    for L in sorted(set(tb) | set(by), key=int):
        den = tb.get(L)
        rds = np.array(by.get(L, []))
        Ls[L] = [(int((rds <= k).sum()) / den) if den else float('nan') for k in range(1, R + 1)]
    return Ls


def training_md(runs, extra):
    rows, rows2 = [], []
    for l, r in sorted(runs.items()):
        h = r.get('harness', {})
        sm = r.get('summary', {})
        tb = sm.get('targets_by_len') or {}
        comp = h.get('compute') or {}
        cs = cum_solved(r) if r.get('found') else {}
        solved = ', '.join(f'L{L} {cs[L][-1]:.2f} of {tb.get(L)}' for L in cs if cs[L]) if cs else '-'
        rows.append([l, h.get('status', '-'), str(sm.get('rounds', len(r.get('sample', [])))), str(sm.get('total_samples', '-')),
                     fmt(h.get('heldout_greedy'), 3),
                     fmt(comp.get('frac'), 2) if isinstance(comp, dict) and comp.get('frac') is not None else '-',
                     str(h.get('budget_exhausted', '-')), solved])
        if r.get('found'):
            w = collections.defaultdict(list)
            for f in r['found']:
                w[f['L_true']].append((f['written'], f['pruned']))
            arm, _ = label_arm(l, extra)
            m = tmax(arm, extra) if arm else None
            allw = np.array([x[0] for v in w.values() for x in v])
            rows2.append([l] + [f"{int(np.max([x[0] for x in w[L]]))} / {np.percentile([x[0] for x in w[L]], 90):.0f}" if w.get(L) else ''
                                for L in range(6, 12)] + [f'{(allw > m).mean():.3f}' if m else '-', str(int(allw.max()))])
    t1 = md_table(['run', 'status', 'rounds', 'samples', 'held-out greedy', 'compute used', 'budget exhausted',
                   'fraction of the target subset solved at the end, by L_true (subset size)'], rows)
    t2 = md_table(['run'] + [f'L_true {L}' for L in range(6, 12)] + ['share of found proofs longer than max_len_train', 'longest'], rows2) if rows2 else ''
    return t1 + ('\n\nLines of the proofs the run found (and so trained on), by the target\'s L_true: max / 90th percentile '
                 '(written length).\n\n' + t2 if t2 else '')


def fig_training(runs, arms, extra, out):
    arms_r = [a for a in arms if a != 'pretrain' and any(label_arm(l, extra)[0] == a and runs[l].get('found') for l in runs)]
    if not arms_r:
        return None
    fig, axs = plt.subplots(1, len(arms_r), figsize=(4 * len(arms_r), 3.3), squeeze=False, sharey=True)
    ramp = {'6': '#b7d3f6', '7': '#86b6ef', '8': '#5598e7', '9': '#2a78d6', '10': '#1c5cab', '11': '#0d366b'}
    for j, arm in enumerate(arms_r):
        ax = axs[0][j]
        labs = [l for l in runs if label_arm(l, extra)[0] == arm and runs[l].get('found')]
        cs = [cum_solved(runs[l]) for l in labs]
        lens = sorted({L for c in cs for L in c}, key=int)
        for L in lens:
            curves = [c[L] for c in cs if c.get(L)]
            n = min(len(c) for c in curves) if curves else 0
            if n == 0:
                continue
            m = np.mean([c[:n] for c in curves], 0)
            ax.plot(np.arange(1, n + 1), m, color=ramp.get(L, INK2), label=f'L{L}', lw=1.8)
        ax.set_title(f'arm {arm} (train max {tmax(arm, extra)}), {len(labs)} runs', fontsize=9)
        ax.set_xlabel('round')
    axs[0][0].set_ylabel('fraction of target subset solved (cumulative)')
    axs[0][-1].legend(fontsize=7, ncol=2, loc='upper left')
    fig.suptitle('Training-side solve rate by true length over rounds (mean over seeds; from found_targets.jsonl)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return savefig(fig, out, 'fig_training.png')


def matched_md(ev, arms, extra, lengths):
    have = any(s == 'matched' for e in ev.values() for (s, L) in e['units'])
    if not have:
        return None, None
    hdr = ['arm'] + [f'L{L}' for L in lengths]
    rows = []
    for arm in arms:
        r = [arm]
        for L in lengths:
            v = {}
            for s in ('std', 'matched'):
                ps = [passk(e['units'][(s, L)], 1) for e in ev.values() if e['arm'] == arm and (s, L) in e['units']]
                v[s] = float(np.mean(ps)) if ps else None
            r.append(f"{fmt(v['std'])} / {fmt(v['matched'])}" if v['matched'] is not None else '')
        rows.append(r)
    return md_table(hdr, rows), True


def fig_matched(ev, arms, extra, lengths, out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for arm in arms:
        c, mk = color(arm, arms), MARKERS.get(arm, 'o')
        for s, ls in (('std', '-'), ('matched', '--')):
            xs, ys = [], []
            for L in lengths:
                ps = [passk(e['units'][(s, L)], 1) for e in ev.values() if e['arm'] == arm and (s, L) in e['units']]
                if ps:
                    xs.append(L); ys.append(np.mean(ps))
            if xs:
                ax.plot(xs, ys, ls, color=c, marker=mk, ms=4, label=f'{arm} {s}')
    ax.set_xlabel('test length L_true'); ax.set_ylabel('pass@1')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Standard sets (solid) vs matched-construction control (dashed)', fontsize=10)
    return savefig(fig, out, 'fig_matched.png')


# ------------------------------------------------------------------ main
# ------------------------------------------------------------------ arm D vs C, construction families, attempt lengths
def test_families():
    fam = {}
    for f in ('std.jsonl', 'matched.jsonl'):
        fn = os.path.join(TESTSETS, f)
        if os.path.isfile(fn):
            for r in read_jsonl(fn):
                fam[r['name']] = r.get('family') or ('standard' if r.get('source') in ('gen', 'textbook') else r.get('source'))
    return fam


def seed_stats(ev, arm, unit, fn):
    vals = [fn(e['units'][unit]) for e in ev.values() if e['arm'] == arm and unit in e['units'] and e['units'][unit]]
    vals = [v for v in vals if v == v]
    return (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)), len(vals)) if vals else None


def cell(v, nd=3):
    if not v:
        return ''
    m, lo, hi, n = v
    return f'{m:.{nd}f}' + (f' [{lo:.{nd}f}, {hi:.{nd}f}]' if n > 1 else '')


def family_md(ev, arms, fam):
    """pass@1 by construction family on the constructed units: matched L6-14 and std L15-18 (mean over seeds)."""
    units = sorted({u for e in ev.values() for u in e['units'] if u[0] == 'matched' or (u[0] == 'std' and u[1] >= 15)},
                   key=lambda u: (u[1], u[0]))
    if not units:
        return None, {}
    fams = ['andi', 'chaink1', 'chaink2']
    out, rows = {}, []
    for arm in arms:
        for f in fams:
            r = [arm, f]
            for u in units:
                v = seed_stats(ev, arm, u, lambda rs, f=f: passk([x for x in rs if fam.get(x['name']) == f], 1) if any(fam.get(x['name']) == f for x in rs) else float('nan'))
                out[f'{arm}|{f}|{u[0]}_L{u[1]}'] = v
                r.append(f'{v[0]:.3f}' if v else '')
            rows.append(r)
    ns = [len([x for x in next(e['units'][u] for e in ev.values() if u in e['units'])]) for u in units]
    hdr = ['arm', 'family'] + [f"{'m' if u[0] == 'matched' else 's'}L{u[1]}" for u in units]
    return md_table(hdr, rows), out


def dc_md(ev, lengths, fits, fam):
    if not any(e['arm'] == 'D' for e in ev.values()) or not any(e['arm'] == 'C' for e in ev.values()):
        return None
    parts = []
    for k in (1, 4):
        rows = []
        for L in lengths:
            c = seed_stats(ev, 'C', ('std', L), lambda rs: passk(rs, k))
            d = seed_stats(ev, 'D', ('std', L), lambda rs: passk(rs, k))
            rows.append([f'L{L}', cell(c), cell(d), f'{100 * (d[0] - c[0]):+.1f} pts' if c and d else ''])
        parts.append(f'**Standard sets, pass@{k}** (mean [min, max] over seeds):\n\n' + md_table(['length', 'C', 'D', 'D - C'], rows))
    rows = []
    for w in ('own (d >= 1)', 'own, generator theorems only'):
        r = fits.get(w, {})
        for arm in ('C', 'D'):
            s_ = r.get('arms', {}).get(arm, {})
            runs = [f"{p['b']:.2f}" if p['ok'] else 'n/a' for l, p in sorted(r.get('per_run', {}).items()) if p['arm'] == arm]
            ci = s_.get('ci')
            rows.append([w, arm, ', '.join(runs), fmt(s_.get('b_mean'), 2), f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '-'])
        v = r.get('dc')
        if v:
            rows.append([w, 'D - C', '', f"{v['diff']:+.2f}", f"Welch p = {fmt_p(v['welch_p'])}, permutation p = {fmt_p(v['perm_p'])}"])
    parts.append('**Log-odds drop per line past 11** (grouped-binomial logistic fit per run over L >= 12, as in section 2):\n\n'
                 + md_table(['window', 'arm', 'slope per run', 'mean', '95% t-interval / test'], rows))
    cu = sorted({u for e in ev.values() for u in e['units'] if u[0] == 'matched' or (u[0] == 'std' and u[1] >= 15)}, key=lambda u: u[1])
    rows = []
    for u in cu:
        for f in ('andi', 'chaink1', 'chaink2', None):
            sel = (lambda rs, f=f: [x for x in rs if f is None or fam.get(x['name']) == f])
            c = seed_stats(ev, 'C', u, lambda rs: passk(sel(rs), 1) if sel(rs) else float('nan'))
            d = seed_stats(ev, 'D', u, lambda rs: passk(sel(rs), 1) if sel(rs) else float('nan'))
            n = len(sel(next(e['units'][u] for e in ev.values() if u in e['units'])))
            rows.append([f"{'matched' if u[0] == 'matched' else 'sealed'} L{u[1]}", f or 'all', n, cell(c), cell(d),
                         f'{100 * (d[0] - c[0]):+.1f} pts' if c and d else ''])
    parts.append('**Constructed sets, pass@1 by family** (matched L6-14, sealed L15-18; mean [min, max] over seeds):\n\n'
                 + md_table(['set', 'family', 'theorems', 'C', 'D', 'D - C'], rows))
    return '\n\n'.join(parts)


def attempt_lengths(eval_root, ev):
    """Written lines of every saved sample: per (label, set, L) median, p90, share >= L_true (the unit's length)."""
    import gzip
    out = {}
    for l, e in ev.items():
        for (s, L) in e['units']:
            fn = os.path.join(eval_root, l, 'samples', f'{s}_L{L}.jsonl.gz')
            if not os.path.isfile(fn):
                continue
            with gzip.open(fn, 'rt') as f:
                n = np.array([json.loads(x)['body'].count(';') for x in f], float)
            out[(l, s, L)] = n
    return out


def attempts_md(al, ev, arms, lengths, set_name):
    Ls = sorted({L for (l, s, L) in al if s == set_name})
    if not Ls:
        return None, {}
    rows, js = [], {}
    for arm in arms:
        r = [arm]
        for L in Ls:
            xs = [al[(l, s, x)] for (l, s, x) in al if s == set_name and x == L and ev[l]['arm'] == arm]
            if not xs:
                r.append(''); continue
            v = np.concatenate(xs)
            st = dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), share_ge_L=float((v >= L).mean()), n=int(len(v)))
            js[f'{arm}|{set_name}|{L}'] = st
            r.append(f"{st['median']:.0f} / {100 * st['share_ge_L']:.0f}%")
        rows.append(r)
    return md_table(['arm'] + [f'L{L}' for L in Ls], rows), js


def comp_training_md(runs):
    """Arm D: which families its solved training targets come from (pool: frozen_comp targets, 30% constructed)."""
    if not any(l.startswith('D_') and r.get('found') for l, r in runs.items()) or not os.path.isfile(COMP_TARGETS):
        return None
    fam = {}
    for r in read_jsonl(COMP_TARGETS):
        fam[r['name']] = r.get('family') or 'standard'
    pool = collections.Counter(fam.values())
    rows = []
    for l, r in sorted(runs.items()):
        if not l.startswith('D_') or not r.get('found'):
            continue
        solved = {f['name'] for f in r['found']}
        c = collections.Counter(fam.get(n, '?') for n in solved)
        tot = sum(c.values())
        cons = sum(v for k, v in c.items() if k not in ('standard', '?'))
        rows.append([l, tot, f'{100 * cons / tot:.1f}%' if tot else '-'] + [str(c.get(k, 0)) for k in ('standard', 'andi', 'chaink1', 'chaink2')])
    share = 100 * sum(v for k, v in pool.items() if k != 'standard') / sum(pool.values())
    Lt = {r['name']: r['L_true'] for r in read_jsonl(COMP_TARGETS)}
    poolL = collections.Counter((Lt[n], fam[n]) for n in fam)
    dl = [l for l, r in runs.items() if l.startswith('D_') and r.get('found')]
    rows2 = []
    for L_ in range(6, 12):
        tot = sum(poolL[(L_, f)] for f in ('standard', 'andi', 'chaink1', 'chaink2'))
        r2 = [f'L{L_}']
        for f in ('standard', 'andi', 'chaink1', 'chaink2'):
            exp = 1000 * poolL[(L_, f)] / tot if tot else 0
            m = np.mean([sum(1 for n in {x['name'] for x in runs[l]['found']} if Lt.get(n) == L_ and fam.get(n) == f) for l in dl])
            r2.append(f'{100 * m / exp:.0f}% ({m:.0f} of ~{exp:.0f})' if exp else '-')
        rows2.append(r2)
    extra_tab = ('\n\nShare of the training subset solved by the end, by L_true and family (mean over D runs). The subset has 1,000 '
                 'targets per length drawn uniformly within the length, so the per-family denominators are the pool shares x 1,000 '
                 '(estimated; the subset is not logged by family).\n\n' + md_table(['L_true', 'standard', 'andi', 'chaink1', 'chaink2'], rows2))
    return (f'Targets arm D solved during training (distinct targets with at least one found proof), by family. The merged pool is '
            f'{share:.0f}% constructed.\n\n' + md_table(['run', 'solved targets', 'constructed share', 'standard', 'andi', 'chaink1', 'chaink2'], rows) + extra_tab)



EXTRA = {}
RUNS_ROOT = RUNS
DEFAULT_EXTRA = {'D': 11}   # arm D: arm C's recipe on the composition-augmented pool (ab3/frozen_comp)
COMP_TARGETS = '/scratch/cowifornia/nd-rl/ab3/frozen_comp/data/targets_L6-11.jsonl'


def main():
    global EXTRA, RUNS_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-root', default=EVAL)
    ap.add_argument('--runs-root', default=RUNS)
    ap.add_argument('--out', default=REPORT)
    ap.add_argument('--train-max', action='append', default=[], help='ARM=N for arms outside A/B/C (tests)')
    ap.add_argument('--run-dir', action='append', default=[], help='LABEL=PATH (training run directory override)')
    ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args()
    EXTRA = dict(DEFAULT_EXTRA, **{k: int(v) for k, v in (x.split('=') for x in a.train_max)})
    RUNS_ROOT = a.runs_root
    run_dirs = dict(x.split('=', 1) for x in a.run_dir)
    os.makedirs(a.out, exist_ok=True)
    ev = load_evals(a.eval_root, EXTRA)
    if not ev:
        sys.exit(f'no evaluations under {a.eval_root}')
    arms = arm_order(ev, EXTRA)
    lengths = sorted({L for e in ev.values() for (s, L) in e['units'] if s == 'std'})
    man = json.load(open(os.path.join(TESTSETS, 'MANIFEST.json'))) if os.path.isfile(os.path.join(TESTSETS, 'MANIFEST.json')) else {}

    tab = section_pass(ev, arms, EXTRA, lengths)
    fits = section_fits(ev, arms, EXTRA, lengths, a.boot)
    own = fits['own (d >= 1)']['arms']
    f_main = fig_main(ev, arms, EXTRA, lengths, own, a.out)
    f_main_g = fig_main(ev, arms, EXTRA, lengths, fits['own, generator theorems only']['arms'], a.out, gen_only=True)
    pl_tab, shorter_any = section_prooflen(ev, arms, EXTRA, lengths)
    f_pl = fig_prooflen(pl_tab, arms, EXTRA, lengths, a.out)
    fl_tab = section_fail(ev, arms, EXTRA, lengths)
    f_fe = fig_fail(fl_tab, arms, EXTRA, lengths, a.out)
    f_hz = fig_hazard(fl_tab, arms, EXTRA, lengths, a.out)
    runs = section_training(list(ev), run_dirs, EXTRA)
    f_tr = fig_training(runs, arms, EXTRA, a.out) if runs else None
    m_md, has_m = matched_md(ev, arms, EXTRA, lengths)
    fam = test_families()
    fam_md, fam_js = family_md(ev, arms, fam)
    dc = dc_md(ev, lengths, fits, fam)
    al = attempt_lengths(a.eval_root, ev)
    at_std, at_js1 = attempts_md(al, ev, arms, lengths, 'std')
    at_m, at_js2 = attempts_md(al, ev, arms, lengths, 'matched')
    ctr = comp_training_md(runs) if runs else None
    f_m = fig_matched(ev, arms, EXTRA, lengths, a.out) if has_m else None

    # checks
    dis = collections.Counter()
    hit = n_samp = 0
    for e in ev.values():
        for s in e['summ'].values():
            dis.update(s.get('checker_disagreements', {}))
            hit += s.get('n_hit_max_new', 0)
            n_samp += s['n'] * s['k']
    comp = {}
    for L in lengths:
        for e in ev.values():
            if ('std', L) in e['units']:
                rows = e['units'][('std', L)]
                comp[L] = (len(rows), sum(r.get('source') == 'textbook' for r in rows),
                           len({r.get('schema') for r in rows if r.get('source') == 'textbook'}))
                break

    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L = []
    L.append('---\nwritten_on: ' + datetime.date.today().isoformat() + '\n---\n')
    L.append(f'# Length generalisation: report\n\nGenerated {now} by `analyse.py` from `{a.eval_root}`. '
             f'k = {K} samples per theorem at T = 0.8; pass@k are unbiased estimates from n_ok of k, averaged over theorems.\n')
    L.append('Evaluated checkpoints:\n\n' + md_table(['label', 'arm', 'seed', 'units', 'checkpoint sha256'],
             [[l, e['arm'], e['seed'] or '-', len(e['units']), (e['meta'].get('ckpt_sha256') or '')[:12]] for l, e in sorted(ev.items())]))
    allu = sorted({u for e in ev.values() for u in e['units']})
    miss = {l: [f'{s}_L{x}' for (s, x) in allu if (s, x) not in e['units']] for l, e in ev.items()}
    miss = {l: m for l, m in miss.items() if m}
    if STALE:
        L.append(f'\n**Left out (evaluated on an earlier theorem draw, awaiting re-evaluation):** {len(STALE)} units: ' + ', '.join(STALE[:40]) + (' ...' if len(STALE) > 40 else ''))
    if miss:
        L.append('\n**Incomplete evaluations** (units other labels have): ' + '; '.join(f'{l}: {", ".join(m)}' for l, m in sorted(miss.items())))
    L.append('\nTest sets (std): ' + ', '.join(f'L{x} n={c[0]} ({c[1]} textbook, {c[2]} schemata)' for x, c in comp.items())
             + ('. Missing source files: ' + ', '.join(os.path.basename(m) for m in man.get('missing', [])) if man.get('missing') else '') + '.')
    L.append('\n## Headline\n\n' + headline_md(ev, arms, EXTRA, lengths, fits))
    L.append('\n## 1. pass@k by arm and length\n\nMean over seeds, [min, max] over seeds in brackets.\n\n**pass@1**\n\n' + pass_md(tab, arms, lengths, 1)
             + '\n\n**pass@4**\n\n' + pass_md(tab, arms, lengths, 4) + '\n\n**pass@32**\n\n' + pass_md(tab, arms, lengths, 32))
    L.append(f'\n## 2. pass@1 against distance beyond the training length\n\n![main]({f_main})\n\n![main, generator only]({f_main_g})\n\n'
             'Reading: if the curves overlap in d, training on longer proofs only moved the edge; if arms with a larger training max '
             'have flatter curves (a smaller log-odds drop per extra line), they generalise more. Slopes are from a grouped-binomial '
             'logistic fit per run (log-odds of pass@1 per extra line); the t-interval is over seeds (3 seeds: t = 4.30), the bootstrap '
             'SE resamples theorems within a run. The "common" window uses the same absolute lengths for every arm, so differences in '
             'curvature over different d ranges cannot masquerade as slope differences.\n\n' + fits_md(fits, arms, EXTRA))
    L.append('\n## 3. Lengths of correct proofs vs the L_true label\n\nCell: fraction of correct proofs shorter than L_true / median '
             '(lines - L_true), pooled over seeds. A proof shorter than L_true shows that the label (from the restricted minlen search) '
             'is too high for that theorem. Last row: theorems (any arm) with such a proof.\n\n' + prooflen_md(pl_tab, shorter_any, ev, arms, EXTRA, lengths)
             + f'\n\n![proof lengths]({f_pl})')
    L.append('\n## 4. Where failed samples go wrong\n\nCell: mean failure position / fraction of failures at positions max..max+2 of the '
             "arm's own training max / fraction early QED / fraction out of tokens / fraction that ran past the N64 line-index "
             'vocabulary. Failure position = first line the incremental checker rejects, or the QED slot for an early QED.\n\n'
             + fail_md(fl_tab, arms, EXTRA, lengths) + f'\n\n![first error]({f_fe})\n\n![hazard]({f_hz})\n\n'
             'If failures pile up where each arm stops having training data (a length-specific circuit), the hazard curves spike at '
             "each arm's own dashed line; if they spread evenly, the hazard is flat in position.\n\n**First line index of samples** "
             '(the abs tokenizer has only N1..N64):\n\n' + start_md(ev, arms, lengths))
    if runs:
        L.append('\n## 5. Training-side diagnostics\n\n' + training_md(runs, EXTRA) + (f'\n\n![training]({f_tr})' if f_tr else '')
                 + (f'\n\n{ctr}' if ctr else ''))
    else:
        L.append('\n## 5. Training-side diagnostics\n\nNo run directories found.')
    L.append('\n## 6. Matched-construction control\n\n' + (f'Cell: pass@1 standard / matched.\n\n{m_md}\n\n![matched]({f_m})' if has_m
             else 'Not available (ab3/sealed/matched_L6-14.jsonl had not been evaluated).')
             + (f'\n\n**pass@1 by construction family** (mL = matched, sL = sealed; mean over seeds):\n\n{fam_md}' if fam_md else ''))
    L.append('\n## 7. Attempt lengths\n\nCell: median written lines of all samples / share of samples with at least L_true written lines '
             '(pooled over seeds). A proof cannot be correct with fewer lines than the theorem needs.\n\n'
             + (f'**Standard sets.**\n\n{at_std}\n\n' if at_std else '') + (f'**Matched constructed sets.**\n\n{at_m}' if at_m else ''))
    if dc:
        L.append('\n## 8. Arm D (composition-augmented RL pool) vs arm C\n\nArm D is arm C exactly (recipe, max_len_train 11, '
                 'max_proof_lines_train 11, compute budget, seeds 101-103) except the RL pool: `ab3/frozen_comp` holds '
                 '`data_comp/targets_L6-11.jsonl` (run 2\'s targets plus constructed ANDI/chain targets, 30% of each length bin). '
                 'D is not part of the A/B/C length-trend tests.\n\n'
                 + (open(os.path.join(HERE, 'conclusion_D.md')).read().strip() + '\n\n' if os.path.isfile(os.path.join(HERE, 'conclusion_D.md')) else '')
                 + dc)
    L.append(f'\n## Checks\n\n- Incremental checker vs nd_verify disagreements over {n_samp} samples: '
             + (', '.join(f'{k} {v}' for k, v in dis.items()) if dis else 'none') + f'.\n- Samples that hit max_new: {hit}.')
    with open(os.path.join(a.out, 'report.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    js = {'pass': {f'{k[0]}|{k[1]}': v for k, v in tab.items()},
          'fits': {w: {'arms': r['arms'], 'tests': r['tests'], 'per_run': r['per_run'], 'dc': r.get('dc')} for w, r in fits.items()},
          'prooflen': {f'{k[0]}|{k[1]}': v for k, v in pl_tab.items()},
          'families': fam_js, 'attempts': {**at_js1, **at_js2}}
    with open(os.path.join(a.out, 'results.json'), 'w') as f:
        json.dump(js, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    print('wrote', os.path.join(a.out, 'report.md'))


if __name__ == '__main__':
    main()
