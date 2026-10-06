#!/usr/bin/env python3
"""Reviewer's own spread / MDD / bimodality / variance-decomposition code."""
import json, math, itertools, statistics as st

def moments(xs):
    n = len(xs); m = sum(xs)/n
    m2 = sum((x-m)**2 for x in xs)/n
    m3 = sum((x-m)**3 for x in xs)/n
    m4 = sum((x-m)**4 for x in xs)/n
    return m, m2, m3, m4

def bimod_pop(xs):
    """b = (skew^2+1)/kurtosis, population moments (uniform -> 5/9)"""
    m, m2, m3, m4 = moments(xs)
    if m2 == 0: return float('nan')
    g1 = m3/m2**1.5; g2 = m4/m2**2
    return (g1*g1+1)/g2, g1, g2

def bimod_sas(xs):
    """SAS/Pfister: sample-corrected skewness g1 and kurtosis (excess+3)"""
    n = len(xs); m, m2, m3, m4 = moments(xs)
    if m2 == 0: return float('nan')
    s = math.sqrt(sum((x-m)**2 for x in xs)/(n-1))
    g1 = (n/((n-1)*(n-2))) * sum((x-m)**3 for x in xs) / s**3
    g2ex = ((n*(n+1))/((n-1)*(n-2)*(n-3))) * sum((x-m)**4 for x in xs)/s**4 - 3*(n-1)**2/((n-2)*(n-3))
    return (g1*g1+1)/(g2ex+3), g1, g2ex+3

# t quantiles by bisection on the CDF via incomplete beta (pure python)
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a+b, a+1, a-1
    c = 1.0; d = 1 - qab*x/qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1/d; h = d
    for m in range(1, MAXIT+1):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1 + aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1 + aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1 + aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1 + aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1/d; de = d*c; h *= de
        if abs(de-1) < EPS: break
    return h

def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = (math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
    bt = math.exp(lb)
    if x < (a+1)/(a+b+2):
        return bt*_betacf(a, b, x)/a
    return 1 - bt*_betacf(b, a, 1-x)/b

def t_cdf(t, df):
    x = df/(df+t*t)
    p = 0.5*betai(df/2, 0.5, x)
    return 1-p if t > 0 else p

def t_ppf(p, df):
    lo, hi = -300.0, 300.0
    for _ in range(200):
        mid = (lo+hi)/2
        if t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo+hi)/2

def mdd_const(n, alpha=0.05, power=0.80):
    df = 2*n-2
    return (t_ppf(1-alpha/2, df) + t_ppf(power, df)) * math.sqrt(2.0/n)

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)

def twoway(vals, pools, seeds):
    """vals: dict (pool,seed)->y, balanced, one obs per cell."""
    a, b = len(pools), len(seeds)
    y = {(p, s): vals[(p, s)] for p in pools for s in seeds}
    gm = sum(y.values())/(a*b)
    pm = {p: sum(y[(p, s)] for s in seeds)/b for p in pools}
    sm = {s: sum(y[(p, s)] for p in pools)/a for s in seeds}
    ss_p = b*sum((pm[p]-gm)**2 for p in pools)
    ss_s = a*sum((sm[s]-gm)**2 for s in seeds)
    ss_t = sum((v-gm)**2 for v in y.values())
    ss_e = ss_t - ss_p - ss_s
    ms_p, ms_s, ms_e = ss_p/(a-1), ss_s/(b-1), ss_e/((a-1)*(b-1))
    return {'grand_mean': gm, 'MS_pool': ms_p, 'MS_seed': ms_s, 'MS_resid': ms_e,
            'var_pool': (ms_p-ms_e)/b, 'var_seed': (ms_s-ms_e)/a, 'var_resid': ms_e,
            'df': [a-1, b-1, (a-1)*(b-1)], 'pool_means': pm, 'seed_means': sm}

def describe(xs):
    n = len(xs)
    s = st.stdev(xs)
    m = sum(xs)/n
    b, g1, g2 = bimod_pop(xs)
    bs = bimod_sas(xs)[0] if n > 3 else float('nan')
    return {'n': n, 'min': min(xs), 'max': max(xs), 'mean': m, 'sd': s,
            'ratio_maxmin': max(xs)/min(xs) if min(xs) else float('inf'),
            'range': max(xs)-min(xs),
            'mdd_n2': mdd_const(2)*s, 'mdd_n2_frac_mean': mdd_const(2)*s/m,
            'mdd_n6': mdd_const(6)*s, 'mdd_n13': mdd_const(13)*s,
            'bimod_pop': b, 'skew_pop': g1, 'kurt_pop': g2, 'bimod_sas': bs}
