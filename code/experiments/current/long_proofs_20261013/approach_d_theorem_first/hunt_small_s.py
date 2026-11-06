"""Small-S hunt: random sequents whose formulas are all built from a tiny shared vocabulary (2-3 atoms, a few
compound subformulas), so minlen's S stays small (<= MAXS) and exhausting large bounds is cheap. Each valid,
consistent, all-premises-necessary sequent is run through the semantic-pruned minlen (minlen_sem) at body bound
MAXBODY to get its exact minimal body length or a lower bound. Looking for theorems with min_body >= 20.

    python hunt_small_s.py OUT.jsonl N SEED MAXS MAXBODY SECS
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import Search, parse_thm, Timeout
from minlen_sem import SemSearch
import sample_cnf as SC

out, n_want, seed, maxs, maxbody, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
rng = random.Random(seed)


def rand_formula(vocab, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(vocab)
    k = rng.random()
    if k < 0.25:
        return ('not', rand_formula(vocab, depth - 1))
    op = rng.choice(['and', 'or', 'imp'])
    return (op, rand_formula(vocab, depth - 1), rand_formula(vocab, depth - 1))


def sample():
    k = rng.choice([2, 2, 3])
    atoms = [('atom', a) for a in C.LETTERS[:k]]
    vocab = list(atoms)
    for _ in range(rng.randint(1, 3)):          # a few shared compound pieces
        vocab.append(rand_formula(vocab, 1))
    m = rng.randint(1, 4)
    prem = []
    while len(prem) < m:
        p = rand_formula(vocab, rng.choice([1, 1, 2]))
        if p not in prem and p != C.BOT:
            prem.append(p)
    concl = rand_formula(vocab, rng.choice([1, 1, 2]))
    if concl in prem or ('bot',) in prem:
        return None
    sat, valid = C.truth_table(prem, concl)
    if not (sat and valid):
        return None
    for i in range(m):
        if C.truth_table(prem[:i] + prem[i + 1:], concl)[1]:
            return None
    return prem, concl


seen, n, tries, t_start = set(), 0, 0, time.time()
with open(out, 'w') as f:
    while n < n_want:
        tries += 1
        s = sample()
        if s is None:
            continue
        prem, concl = s
        key = C.theorem_key(prem, concl)
        if key in seen:
            continue
        S = Search(prem, concl).S
        if len(S) > maxs:
            continue
        seen.add(key)
        prompt = SC.prompt_of(prem, concl)
        ss = SemSearch(prem, concl, deadline=time.time() + secs)
        t0 = time.time(); res = None; status = 'exhausted'
        try:
            for body in range(1, maxbody + 1):
                r = ss.best(concl, frozenset(prem), body)
                if r:
                    res = body; break
                if len(ss.memo) > 6_000_000:
                    status = 'memo cap'; break
        except Timeout:
            status = 'timeout'
        rec = {'prompt': prompt, 'n_prem': len(prem), 'S': len(S), 'min_body': res, 'status': status if res is None else 'found',
               'last_body': body, 'secs': round(time.time() - t0, 2), 'memo': len(ss.memo)}
        f.write(json.dumps(rec) + '\n'); f.flush()
        n += 1
        if res is None or res >= 14:
            print(json.dumps(rec), flush=True)
print(f'{n} sequents, {tries} draws, {time.time() - t_start:.0f}s')
