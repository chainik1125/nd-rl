"""Cross-validation: sampler -> linearizer -> tokens -> independent verifier.

1. Positive: every sampled proof must verify.
2. Negative: random mutations of valid proofs should overwhelmingly fail.
"""
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.core import sample_tree, linearize, proof_to_tokens, theorem_id, rule_profile, DISCHARGING
from nd.verify import verify_text

def main(n=3000):
    rng = random.Random(0)
    n_ok = 0
    lengths = []
    n_discharge = 0
    fails = []
    for i in range(n):
        steps = rng.randint(2, 14)
        t = sample_tree(rng, n_steps=steps)
        if t is None:
            continue
        try:
            prem, concl, lines = linearize(t)
        except ValueError as e:
            fails.append(('linearize', str(e)))
            continue
        if concl in prem:
            continue  # trivial, skipped in datasets too
        toks = proof_to_tokens(prem, concl, lines)
        ok, reason, nl = verify_text(' '.join(toks))
        if ok:
            n_ok += 1
            lengths.append(nl)
            if rule_profile(t) & DISCHARGING:
                n_discharge += 1
        else:
            fails.append((theorem_id(prem, concl), reason))
            if len(fails) <= 5:
                print('FAIL:', reason)
                print(' '.join(toks)[:400])
    print(f'positive: {n_ok} verified, {len(fails)} failed')
    if lengths:
        import statistics
        print(f'proof lines: min {min(lengths)} med {statistics.median(lengths)} max {max(lengths)}')
        print(f'with discharging rules: {n_discharge}/{n_ok}')

    # negative tests: mutate token streams
    rng2 = random.Random(1)
    n_mut, n_caught = 0, 0
    silent = []
    for i in range(400):
        t = sample_tree(rng2, n_steps=rng2.randint(3, 10))
        if t is None:
            continue
        try:
            prem, concl, lines = linearize(t)
        except ValueError:
            continue
        toks = proof_to_tokens(prem, concl, lines)
        ok, _, _ = verify_text(' '.join(toks))
        if not ok:
            continue
        # mutate: swap a ref, change a rule, change an atom, drop a line marker
        toks2 = list(toks)
        idxs = [k for k, tk in enumerate(toks2)
                if (tk.startswith('N') and tk[1:].isdigit())
                or tk in ('P','Q','R','S')
                or tk in ('ANDI','ANDE1','ANDE2','IMPE','IMPI','ORI1','ORI2','ORE','NEGE','NEGI','BOTE','DN')]
        if not idxs:
            continue
        k = rng2.choice(idxs)
        tk = toks2[k]
        if tk.startswith('N') and tk[1:].isdigit():
            toks2[k] = f'N{max(1, int(tk[1:]) + rng2.choice([-2,-1,1,2]))}'
        elif tk in ('P','Q','R','S'):
            toks2[k] = rng2.choice([a for a in 'PQRS' if a != tk])
        else:
            toks2[k] = rng2.choice([r for r in ['ANDI','IMPE','NEGE','DN','BOTE'] if r != tk])
        if toks2 == list(toks):
            continue
        n_mut += 1
        ok2, _, _ = verify_text(' '.join(toks2))
        if not ok2:
            n_caught += 1
        else:
            silent.append(' '.join(toks2)[:200])
    print(f'negative: {n_caught}/{n_mut} mutations rejected')
    # a few mutations can legitimately still be valid proofs (e.g. atom swap in
    # an unused premise side); print a couple to eyeball
    for s in silent[:3]:
        print('still-valid mutation:', s)

    # offset re-indexing: shifted proofs must still verify; shifted-with-gap must not
    import re
    NTOK = re.compile(r'^N(\d+)$')
    rng3 = random.Random(2)
    n_off_ok, n_off = 0, 0
    for i in range(300):
        t = sample_tree(rng3, n_steps=rng3.randint(3, 10))
        if t is None:
            continue
        try:
            prem, concl, lines = linearize(t)
        except ValueError:
            continue
        toks = proof_to_tokens(prem, concl, lines)
        o = rng3.randint(1, 64 - len(lines)) if len(lines) < 64 else 0
        shifted = [f'N{int(m.group(1))+o}' if (m := NTOK.match(tk)) else tk
                   for tk in toks]
        ok, reason, _ = verify_text(' '.join(shifted))
        n_off += 1
        if ok:
            n_off_ok += 1
        elif n_off - n_off_ok <= 3:
            print('OFFSET FAIL:', reason)
    print(f'offset: {n_off_ok}/{n_off} shifted proofs verified')

if __name__ == '__main__':
    main()
