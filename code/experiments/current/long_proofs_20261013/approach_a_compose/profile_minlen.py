"""Where does minlen spend its time? Run the search at a given body bound and histogram memo states by
|A - premises| (extra citable formulas: hypotheses + lemmas) and by goal. Usage:
    python profile_minlen.py BODY_BOUND SECS piece+piece ...
"""
import collections
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import certify  # noqa: E402,F401  (sets atom names)
from minlen import Search, Timeout, parse_thm  # noqa: E402
from compose import compose, fstr  # noqa: E402

bound, secs = int(sys.argv[1]), float(sys.argv[2])
rng = random.Random(0)
for chain in sys.argv[3:]:
    names = chain.split('+')
    text, meta = compose(names, rng)
    prompt = text.split(' PRF ', 1)[0] + ' PRF'
    prem, concl = parse_thm(prompt)
    s = Search(prem, concl, deadline=time.time() + secs)
    t0 = time.time()
    found = None
    try:
        for total in range(len(prem) + 1, len(prem) + bound + 1):
            r = s.best(concl, frozenset(prem), total - len(prem))
            if r:
                found = total - len(prem)
                break
    except Timeout:
        pass
    dt = time.time() - t0
    P = frozenset(prem)
    extra = collections.Counter(len(k[1] - P) for k in s.memo)
    goals = collections.Counter(fstr(k[0]) for k in s.memo)
    lem = collections.Counter()
    for k in s.memo:
        for f in k[1] - P:
            lem[fstr(f)] += 1
    print(f'{chain}: |S|={len(s.S)} prem={len(prem)} body_written={meta["n_body"]} found={found} '
          f'{dt:.1f}s calls={s.calls} memo={len(s.memo)}')
    print('  memo by |A-prem|:', dict(sorted(extra.items())))
    print('  top extra formulas in A:', lem.most_common(12))
    print('  top goals:', goals.most_common(8), flush=True)
