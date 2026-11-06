"""Yield test: how many long proofs per hour do the two generators make with 3-10 atoms?

    python yield_test.py {takehome,dan_forward,dan_goal} SECONDS OUT.jsonl

takehome    = Robbie's backward generator (~/repos/nd-takehome/gen.py), budget 20-100
dan_forward = Dan's forward generator (proof_gen.Generator.forward, strict/long settings), 20-150 steps
dan_goal    = Dan's goal mode (proof_gen.Generator.goal, strict/long settings), deeper goals/budgets

Per draw the atom count k is uniform on 3..10; atom names are P Q R S T U W X Y Z (no V: `v` is or).
Every kept record is verified by the take-home nd_verify with its atom set widened. Nothing in
either repo is modified; the 4-atom constants are patched in memory.
"""
import json
import random
import re
import sys
import time

sys.setrecursionlimit(20000)
TAKEHOME = '/Users/robbie/repos/nd-takehome'
DAN = '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/dan'
sys.path[:0] = [TAKEHOME, DAN]

import ndtok  # noqa: E402
import nd_verify  # noqa: E402
V = sys.modules["nd_verify.verify"]
verify_text = V.verify_text

LETTERS = ['P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
V.ATOM_NAMES.update(LETTERS)
ATOM_RE = re.compile(r'\b[' + ''.join(LETTERS) + r']\b')

mode, seconds, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rng = random.Random(0)


def renaming_key(thm):
    m = {}
    return ' '.join(m.setdefault(t, f'a{len(m)}') if t in LETTERS else t for t in thm.split())


def draw_takehome(g):
    k = rng.randint(3, 10)
    ndtok.ATOMS[:] = LETTERS[:k]  # gen.py imported this same list object
    budget = rng.randint(20, 100)
    rec = g.generate(budget=budget)
    return k, {'budget': budget}, rec['prompt'] + ' ' + rec['proof'], rec['thm']


def draw_dan(g, PG, goal):
    k = rng.randint(3, 10)
    g.atoms = LETTERS[:k]
    g.ore_steps, g.strict, g.bot_p = 3, True, 0.0  # sample_one's long settings
    n_prem = rng.choice([0, 1, 1, 2, 2, 3])
    if goal:
        gdepth, budget = rng.choice([3, 4, 5, 6]), rng.randint(5, 14)
        rec = g.goal(n_prem, gdepth, budget)
        knobs = {'gdepth': gdepth, 'budget': budget}
    else:
        n_steps = rng.randint(*STEPS)
        rec = g.forward(n_steps, n_prem)
        knobs = {'n_steps': n_steps}
    return k, knobs, rec['prompt'] + ' ' + rec['proof'], rec['thm']


if mode == 'takehome':
    import gen
    g = gen.Gen(seed=0, **json.loads(sys.argv[4] if len(sys.argv) > 4 else '{}'))
    draw = lambda: draw_takehome(g)  # noqa: E731
    fails = (gen.Detour, RecursionError)
else:
    import code.shared_code.proof_gen as PG
    STEPS = json.loads(sys.argv[4]) if len(sys.argv) > 4 else (20, 150)
    g = PG.Generator(random.Random(0), max_prem=3)
    draw = lambda: draw_dan(g, PG, mode == 'dan_goal')  # noqa: E731
    fails = (PG.GenFail, RecursionError)

stats = {'tries': 0, 'fail': 0, 'invalid': 0, 'dup': 0, 'kept': 0}
seen = set()
t0 = time.time()
with open(out_path, 'w') as out:
    while time.time() - t0 < seconds:
        stats['tries'] += 1
        try:
            k, knobs, text, thm = draw()
        except fails:
            stats['fail'] += 1
            continue
        ok, reason, n_lines = verify_text(text)
        if not ok:
            stats['invalid'] += 1
            if stats['invalid'] <= 3:
                print('INVALID', reason, text[:300], file=sys.stderr)
            continue
        key = renaming_key(thm)
        if key in seen:
            stats['dup'] += 1
            continue
        seen.add(key)
        stats['kept'] += 1
        n_prem = text.count(': PR')
        body = text.split(' PRF ', 1)[1]
        out.write(json.dumps({
            'mode': mode, 'k_allowed': k, 'n_atoms': len(set(ATOM_RE.findall(thm))), **knobs,
            'n_lines': n_lines, 'n_prem': n_prem, 'n_body': n_lines - n_prem,
            'max_depth': max([seg.count('|') for seg in body.split(';')] + [0]),
            'nd_tokens': len(body.split()), 'text': text}) + '\n')
stats['seconds'] = round(time.time() - t0, 1)
print(json.dumps(stats))
