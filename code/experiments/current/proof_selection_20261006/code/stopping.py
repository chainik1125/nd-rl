#!/usr/bin/env python3
"""Stopping simulator (Worker B, schema section 5) on a frozen-generator pool.

Policy: consume attempts in an order; after each attempt the policy knows every score so far; stop when the best score
so far >= tau or when the N_max-th attempt has been consumed; return the best-scoring attempt so far.
Frozen conventions (logs/worker_B.md): ties -> the earliest consumed attempt; all scores -inf -> the first consumed
attempt; attempts are i.i.d. fresh contexts, so we average over R random consumption orders per theorem (seeded), and
an `--orig_order` mode uses the generation order instead (exchangeability check).

Policies computed for every scorer:
  fixed-k            k = 1..K, select argmax score among the first k                (tau = +inf, N_max = k)
  adaptive(tau, N)   tau over the scorer's dev-quantile grid, N in {8, 16, 32, 64} (<= K)
  nsweep             tau fixed (scorer-specific, e.g. completion >= 1), N_max = 1..K  (the "stop at first finished" curve)
Built-in scorers: `random` (constant score: earliest consumed = uniform random pick), `completion` (done),
`oracle` (valid; stop at first valid; fixed-k = pass@k).  File scorers: scores/<stem>__<name>.jsonl.

Costs per attempt: attempts; generated tokens (pool gen_tokens); total tokens = prefill + generated + the scorer's own
forward (teacher-forced loglik family: one forward over every (state, action) pair, counted although it could be read
off the sampling logits for free; verifier_*: min(1024, same); completion/random/oracle: 0).  Prefill comes from the
pool's `prefill_tokens` if present, else from the replay (scores/aux/<stem>__shallow_features.jsonl).

Comparator at matched mean attempts: the random mixture of fixed-k at adjacent k with the same selector (linear
interpolation in k); paired theorem-level bootstrap (B=1000) recomputing the mixture weight in each resample.

Output: stopping/<stem>__<scorer>.json (curves + bootstrap) and stopping/<stem>__<scorer>.npz (per-theorem arrays).
"""
import argparse, json, math, os, sys
import numpy as np

RUNS = os.path.expanduser('~/runs/proof-selection')
LOGLIK_FAMILY = ('loglik_mean', 'loglik_mean_masked', 'loglik_mean_full', 'loglik_sum', 'loglik_min_step',
                 'entropy_mean', 'completion_loglik')
NSWEEP_TAU = {'completion': 0.5, 'oracle': 0.5}
NMAXES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
TARGETS = (1.5, 2, 3, 4, 6, 8, 12, 16)


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def load_pool(pool, scores_dir, labels=None, names=None, scorers=None):
    stem = os.path.basename(pool)[:-len('.jsonl')]
    rows = read_jsonl(pool)
    lab = {r['name']: r for r in read_jsonl(labels or pool[:-len('.jsonl')] + '.labels.jsonl')}
    if names is not None:
        rows = [r for r in rows if r['name'] in names]
    feats = {}
    fp = os.path.join(scores_dir, 'aux', f'{stem}__shallow_features.jsonl')
    if os.path.exists(fp):
        feats = {r['name']: r['features'] for r in read_jsonl(fp)}
    if scorers is None:
        scorers = sorted(f[len(stem) + 2:-6] for f in os.listdir(scores_dir)
                         if f.startswith(stem + '__') and f.endswith('.jsonl'))
        scorers = [x for x in scorers if x not in ('random', 'first')]   # built-in exact random; `first` is order-only
    sc = {}
    for s in scorers:
        p = os.path.join(scores_dir, f'{stem}__{s}.jsonl')
        if os.path.exists(p):
            sc[s] = {r['name']: r['scores'] for r in read_jsonl(p)}
    T = []
    for r in rows:
        at = r['attempts']
        k = len(at)
        v = np.array(lab[r['name']]['valid'], dtype=bool)
        assert len(v) == k
        d = np.array([a['done'] for a in at], dtype=bool)
        gen = np.array([a.get('gen_tokens', 0) for a in at], dtype=float)
        f = feats.get(r['name'])
        if at[0].get('prefill_tokens') is not None:
            pre = np.array([a['prefill_tokens'] for a in at], dtype=float)
        elif f is not None:
            pre = np.array([x['prefill_tokens_replay'] for x in f], dtype=float)
        else:
            pre = np.full(k, np.nan)
        fwd = np.array([x['tf_forward_tokens'] for x in f], dtype=float) if f is not None else np.full(k, np.nan)
        S = {'random': np.zeros(k), 'completion': d.astype(float), 'oracle': v.astype(float)}
        for s, m in sc.items():
            x = np.array(m[r['name']], dtype=float)
            assert len(x) == k, (s, r['name'])
            S[s] = np.where(np.isnan(x), -np.inf, x)
        T.append(dict(name=r['name'], k=k, v=v, d=d, gen=gen, pre=pre, fwd=fwd, S=S,
                      steps=np.array([a['steps'] for a in at]), base=np.array([a.get('base', -1) for a in at])))
    return stem, T, sorted(set(['random', 'completion', 'oracle']) | set(sc))


def scorer_cost(name, t):
    if name in LOGLIK_FAMILY:
        return t['fwd']
    if name.startswith('verifier'):
        return np.minimum(1024.0, t['fwd'])
    return np.zeros(t['k'])


def perms_for(T, R, seed, orig_order=False):
    out = []
    for ti, t in enumerate(T):
        if orig_order:
            out.append(np.arange(t['k'])[None])
        else:
            rng = np.random.default_rng([seed, ti])
            out.append(np.stack([rng.permutation(t['k']) for _ in range(R)]))
    return out


def simulate(T, perms, name, taus, nmaxes, nsweep_tau=None):
    """per-theorem arrays (R-averaged) for fixed-k, adaptive(tau, N), and the N sweep at a fixed tau."""
    K = min(t['k'] for t in T)
    nT = len(T)
    Ns = [n for n in nmaxes if n <= K]
    taus = np.asarray(taus, dtype=float)
    G = len(taus)
    fx = {q: np.zeros((K, nT)) for q in ('acc', 'gen', 'tot')}
    ad = {q: np.zeros((len(Ns), G, nT)) for q in ('acc', 'att', 'gen', 'tot', 'disp')}
    ns = {q: np.zeros((K, nT)) for q in ('acc', 'att', 'gen', 'tot', 'disp')} if nsweep_tau is not None else None
    for ti, (t, P) in enumerate(zip(T, perms)):
        P = P[:, :K]
        S = t['S'][name][P]
        V = t['v'][P]
        cg = np.cumsum(t['gen'][P], 1)
        ct = np.cumsum((t['gen'] + t['pre'] + scorer_cost(name, t))[P], 1)
        M = np.maximum.accumulate(S, 1)
        idx = np.broadcast_to(np.arange(K), S.shape)
        new = np.ones_like(S, dtype=bool)
        new[:, 1:] = S[:, 1:] > M[:, :-1]
        A = np.maximum.accumulate(np.where(new, idx, 0), 1)
        Vs = np.take_along_axis(V, A, 1).astype(float)
        if name == 'random':        # a uniform pick from a uniform random subset is marginally a uniform attempt: exact
            Vs = np.full(S.shape, t['v'].mean())
        ever = np.maximum.accumulate(Vs, 1)
        fx['acc'][:, ti] = Vs.mean(0)
        fx['gen'][:, ti] = cg.mean(0)
        fx['tot'][:, ti] = ct.mean(0)

        def at_stop(stop):          # stop: (..., R) indices
            g = lambda X: np.take_along_axis(np.broadcast_to(X, stop.shape[:-1] + X.shape), stop[..., None], -1)[..., 0]
            return dict(acc=g(Vs).mean(-1), att=(stop + 1).mean(-1), gen=g(cg).mean(-1), tot=g(ct).mean(-1),
                        disp=((g(ever) == 1) & (g(Vs) == 0)).mean(-1))
        for ni, N in enumerate(Ns):
            stop = np.minimum((M[None, :, :N] < taus[:, None, None]).sum(-1), N - 1)     # (G, R)
            o = at_stop(stop)
            for q in ad:
                ad[q][ni, :, ti] = o[q]
        if ns is not None:
            first = (M < nsweep_tau).sum(-1)                                            # (R,)
            stop = np.minimum(first[None, :], (np.arange(K))[:, None])                  # (K, R): N_max = 1..K
            o = at_stop(stop)
            for q in ns:
                ns[q][:, ti] = o[q]
    return dict(K=K, Ns=Ns, taus=taus, fixed=fx, adaptive=ad, nsweep=ns)


def mix_weights(m, K):
    """mixture of fixed k1 = floor(m), k1 + 1 with mean m (k in 1..K)."""
    m = min(max(m, 1.0), float(K))
    k1 = int(math.floor(m))
    if k1 >= K:
        return K - 1, 1.0
    return k1 - 1, m - k1            # index of k1 in the fixed arrays, weight on k1 + 1


def fixed_at(fixed_acc, m):
    i, w = mix_weights(m, fixed_acc.shape[0])
    return (1 - w) * fixed_acc[i] + w * fixed_acc[min(i + 1, fixed_acc.shape[0] - 1)]


def bootstrap_vs_fixed(acc_t, att_t, fixed_acc, fixed_ref=None, B=1000, seed=0):
    """paired theorem bootstrap of acc(policy) - acc(fixed-k mixture at the policy's mean attempts).  fixed_acc: (K, nT)
    same selector; fixed_ref: optional other fixed-k (e.g. random selector)."""
    nT = len(acc_t)
    rng = np.random.default_rng(seed)
    out = {}
    for lab, F in (('same_selector', fixed_acc), ('random_selector', fixed_ref)):
        if F is None:
            continue
        def diff(ix):
            m = att_t[ix].mean()
            return acc_t[ix].mean() - fixed_at(F[:, ix], m).mean()
        d0 = diff(np.arange(nT))
        bs = np.array([diff(rng.integers(0, nT, nT)) for _ in range(B)])
        out[lab] = dict(diff=float(d0), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                        p_le0=float((bs <= 0).mean()))
    return out


def summarize(stem, name, sim, sim_random, T, B=1000):
    K, Ns, taus = sim['K'], sim['Ns'], sim['taus']
    fx, ad, ns = sim['fixed'], sim['adaptive'], sim['nsweep']
    res = dict(pool=stem, scorer=name, n_theorems=len(T), K=K, Ns=Ns,
               taus=[None if not np.isfinite(x) else float(x) for x in taus],
               taus_inf=[('+inf' if x == np.inf else '-inf' if x == -np.inf else None) for x in taus],
               fixed=dict(k=list(range(1, K + 1)), acc=fx['acc'].mean(1).tolist(), gen=fx['gen'].mean(1).tolist(),
                          tot=fx['tot'].mean(1).tolist()),
               adaptive={}, nsweep=None, bootstrap={})
    for ni, N in enumerate(Ns):
        res['adaptive'][str(N)] = {q: ad[q][ni].mean(-1).tolist() for q in ad}
    if ns is not None:
        res['nsweep'] = {q: ns[q].mean(-1).tolist() for q in ns}
    # bootstrap at curve-matched mean attempts (curve reading on this pool, no selection across N)
    for ni, N in enumerate(Ns):
        att = ad['att'][ni].mean(-1)
        for m in TARGETS:
            if m > N:
                continue
            g = int(np.argmin(np.abs(att - m)))
            if abs(att[g] - m) > 0.15 * m:
                continue
            bt = bootstrap_vs_fixed(ad['acc'][ni, g], ad['att'][ni, g], fx['acc'], sim_random['fixed']['acc'], B=B)
            res['bootstrap'][f'N{N}_m{m}'] = dict(N=N, target=m, tau=res['taus'][g], tau_idx=g, att=float(att[g]),
                                                  acc=float(ad['acc'][ni, g].mean()),
                                                  fixed_same=float(fixed_at(fx['acc'], att[g]).mean()),
                                                  fixed_random=float(fixed_at(sim_random['fixed']['acc'], att[g]).mean()),
                                                  **bt)
    if ns is not None:
        for m in TARGETS:
            att = ns['att'].mean(-1)
            g = int(np.argmin(np.abs(att - m)))
            if abs(att[g] - m) > 0.15 * m:
                continue
            bt = bootstrap_vs_fixed(ns['acc'][g], ns['att'][g], fx['acc'], sim_random['fixed']['acc'], B=B)
            res['bootstrap'][f'nsweep_m{m}'] = dict(N=g + 1, target=m, att=float(att[g]), acc=float(ns['acc'][g].mean()),
                                                    fixed_same=float(fixed_at(fx['acc'], att[g]).mean()),
                                                    fixed_random=float(fixed_at(sim_random['fixed']['acc'], att[g]).mean()),
                                                    **bt)
    return res


def check_invariants(T, sims):
    """logged asserts (advisor list): fixed-k random = per-attempt pass rate; all selectors equal at k=1; tau=+inf ->
    fixed N; tau=-inf -> fixed 1; oracle fixed-k = pass@k (= any valid among first k)."""
    pr = np.array([t['v'].mean() for t in T])
    r = sims['random']
    out = {}
    out['random_fixed_vs_passrate_maxdiff'] = float(np.abs(r['fixed']['acc'] - pr[None]).max())
    ref = sims['completion']['fixed']['acc'][0]   # same orderings: every non-random selector picks the same first attempt
    out['k1_all_equal_maxdiff'] = float(max(np.abs(s['fixed']['acc'][0] - ref).max() for n, s in sims.items() if n != 'random'))
    out['k1_sampled_vs_passrate_meandiff'] = float(abs(ref.mean() - pr.mean()))
    worst_inf = worst_ninf = 0.0
    for s in sims.values():
        taus = s['taus']
        for ni, N in enumerate(s['Ns']):
            gi = np.where(taus == np.inf)[0]
            if len(gi):
                worst_inf = max(worst_inf, float(np.abs(s['adaptive']['acc'][ni, gi[0]] - s['fixed']['acc'][N - 1]).max()))
            gi = np.where(taus == -np.inf)[0]
            if len(gi):
                worst_ninf = max(worst_ninf, float(np.abs(s['adaptive']['acc'][ni, gi[0]] - s['fixed']['acc'][0]).max()))
    out['tau_+inf_equals_fixedN_maxdiff'] = worst_inf
    out['tau_-inf_equals_fixed1_maxdiff'] = worst_ninf
    # oracle pass@k vs the unbiased estimator 1 - C(n-c, k)/C(n, k), averaged (R-sampled so tolerance)
    o = sims['oracle']['fixed']['acc']
    est = []
    for t in T:
        n, c = t['k'], int(t['v'].sum())
        est.append([1 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0) for k in range(1, o.shape[0] + 1)])
    est = np.array(est).T
    out['oracle_passk_vs_unbiased_meandiff_max'] = float(np.abs(o.mean(1) - est.mean(1)).max())
    assert out['k1_all_equal_maxdiff'] < 1e-12 and worst_inf < 1e-12 and worst_ninf < 1e-12, out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--scores_dir', default=os.path.join(RUNS, 'scores'))
    ap.add_argument('--out_dir', default=os.path.join(RUNS, 'stopping'))
    ap.add_argument('--split', default=None, help='dev|train|test: restrict to these theorem names (pools_v2/splits.json)')
    ap.add_argument('--grid', default=None, help='tau grid json from dev (required for test pools)')
    ap.add_argument('--write_grid', default=None, help='dev only: write the tau grid (quantiles of these scores) here')
    ap.add_argument('--scorers', default=None)
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--orig_order', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    names = None
    if a.split:
        sp = json.load(open(os.path.join(RUNS, 'pools_v2', 'splits.json')))[a.split]
        names = set(n for v in sp.values() for n in v)
    scorers = a.scorers.split(',') if a.scorers else None
    stem, T, scs = load_pool(a.pool, a.scores_dir, a.labels, names, scorers)
    set_name = json.loads(open(a.pool).readline()).get('set')
    if set_name in ('holdout250', 'textbook72'):
        assert a.grid and not a.write_grid, 'test pools: tau grid must come from dev'
    grid = json.load(open(a.grid)) if a.grid else {}
    if a.write_grid:
        for s in scs:
            if s in ('random', 'oracle'):
                continue
            x = np.concatenate([t['S'][s] for t in T])
            x = x[np.isfinite(x)]
            q = np.unique(np.quantile(x, np.linspace(0, 1, 101))) if len(x) else np.array([])
            old = grid.get(s, [])
            grid[s] = sorted(set(float(v) for v in q) | set(old))
        json.dump(grid, open(a.write_grid, 'w'))
    perms = perms_for(T, a.R, a.seed, a.orig_order)
    sims = {}
    for s in scs:
        if s == 'oracle':
            taus = [0.5]
        elif s == 'random':
            taus = []
        else:
            taus = list(grid.get(s, []))
        taus = sorted(set(taus) | {np.inf, -np.inf})
        sims[s] = simulate(T, perms, s, taus, NMAXES, NSWEEP_TAU.get(s, 0.5 if s == 'oracle' else None))
    inv = check_invariants(T, sims)
    print(stem, a.split, 'theorems', len(T), 'invariants', json.dumps(inv))
    os.makedirs(a.out_dir, exist_ok=True)
    sfx = (f'__{a.split}' if a.split else '') + ('__origorder' if a.orig_order else '') + a.tag
    passrate = np.array([t['v'].mean() for t in T])
    for s in scs:
        res = summarize(stem, s, sims[s], sims['random'], T, B=a.B)
        res.update(split=a.split, orig_order=a.orig_order, R=1 if a.orig_order else a.R, invariants=inv,
                   names=[t['name'] for t in T], passrate=passrate.tolist(),
                   finished_rate=[float(t['d'].mean()) for t in T])
        with open(os.path.join(a.out_dir, f'{stem}{sfx}__{s}.json'), 'w') as f:
            json.dump(res, f)
        sm = sims[s]
        np.savez_compressed(os.path.join(a.out_dir, f'{stem}{sfx}__{s}.npz'), taus=sm['taus'], Ns=np.array(sm['Ns']),
                            passrate=passrate, **{f'fixed_{q}': v for q, v in sm['fixed'].items()},
                            **{f'adaptive_{q}': v for q, v in sm['adaptive'].items()},
                            **({f'nsweep_{q}': v for q, v in sm['nsweep'].items()} if sm['nsweep'] is not None else {}))


if __name__ == '__main__':
    main()
