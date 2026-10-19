"""Probe: verify each piece and chains, and time minlen. Usage: python probe.py pieces|chains [secs]"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from certify import check, certify  # noqa: E402
from compose import PIECES, compose  # noqa: E402

mode = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
rng = random.Random(0)

if mode == 'pieces':
    for nm in sorted(PIECES):
        for names in ([nm], ['merge', nm]):
            text, meta = compose(names, rng)
            c = check(text)
            if c is None:
                print(nm, names, 'REJECTED', check.last_reason, text)
                continue
            r = certify(c['prompt'], c['n_prem'], floor=c['n_body'] + 1, secs=secs)
            print(f"{names!s:22s} written body {c['n_body']:2d} minlen exact {r['min_body']} cert_ge {r['cert_ge']} "
                  f"timeout {r['timeout']} {r['cert_secs']}s", flush=True)
else:
    chains = [a.split('+') for a in sys.argv[3:]]
    for names in chains:
        text, meta = compose(names, rng)
        c = check(text)
        if c is None:
            print(names, 'REJECTED', check.last_reason)
            continue
        t0 = time.time()
        r = certify(c['prompt'], c['n_prem'], floor=20, secs=secs)
        print(f"{'+'.join(names):40s} prem {c['n_prem']} body {c['n_body']:2d} atoms {c['n_atoms']} "
              f"min_body {r['min_body']} cert_ge {r['cert_ge']} timeout {r['timeout']} {r['cert_secs']}s", flush=True)
