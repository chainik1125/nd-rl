"""Loop-count trend over the k=16 runs: OK count, targets solved, failure class mix (all targets and never-solved)."""
import json, collections
B = '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe/B'


def cls(r):
    s = r['status']
    if s == 'OK': return 'OK'
    if s == 'PARSE no-eos': return 'ramble (no eos, restating)'
    if s == 'PARSE unbound':
        k = r.get('unbound_kind')
        if k == 'cites-enclosing-have': return 'circular: box closed by citing its own have'
        if k and 'leak' in k: return 'scope leak: cites a fact from a closed box'
        return 'hallucinated / forward name'
    if s == 'PARSE expected (': return 'one-armed Or.elim (second branch missing)'
    if s.startswith('PARSE'): return 'other grammar'
    ru = r.get('err_rule') or '?'
    if ru in ('IMPE/NEGE',): return 'ill-typed IMPE/NEGE (modus ponens on a wrong fact)'
    if ru == 'ORE': return 'ill-typed ORE (branch hyp != disjunct / split on non-disjunction)'
    if ru in ('ANDI', 'ORI', 'ANDE', 'R', 'BOTE', 'exact'): return 'ill-typed intro/elim/restate (ANDI ORI ANDE R BOTE)'
    if ru == 'DN': return 'ill-typed DN (byContradiction on wrong formula)'
    if ru == 'IMPI/NEGI': return 'ill-typed IMPI/NEGI (box hypothesis != antecedent)'
    return 'ill-typed other'


runs = {}
for L in (1, 2, 4, 8):
    runs[L] = [json.loads(l) for l in open(f'{B}/samples_L{L}_k16_s0.classified.jsonl')]
ok = {L: collections.defaultdict(int) for L in runs}
pool = {}
for L, rs in runs.items():
    for r in rs: ok[L][r['name']] += r['ok']; pool[r['name']] = r['pool']
names = list(pool)
never = [n for n in names if all(ok[L][n] == 0 for L in runs)]
print(f'targets {len(names)}, never solved at any loop count {len(never)}')
print('loops | ok samples /1040 | targets solved /65 | solved among unsolved-pool targets')
for L in runs:
    uns = sum(ok[L][n] > 0 for n in names if 'unsolved' in pool[n])
    print(f'{L:5d} | {sum(ok[L].values()):4d} | {sum(ok[L][n] > 0 for n in names):2d} | {uns}')
print('\nclass mix (all 1040 samples per loop count)')
cs = {L: collections.Counter(cls(r) for r in rs) for L, rs in runs.items()}
keys = sorted(set().union(*cs.values()), key=lambda k: -sum(c[k] for c in cs.values()))
print(f'{"class":62s} ' + ' '.join(f'L{L:<4d}' for L in runs))
for k in keys: print(f'{k:62s} ' + ' '.join(f'{cs[L][k]:5d}' for L in runs))
print(f'\nclass mix on the {len(never)} never-solved targets ({len(never) * 16} samples per loop count)')
cs = {L: collections.Counter(cls(r) for r in rs if r['name'] in never) for L, rs in runs.items()}
for k in keys: print(f'{k:62s} ' + ' '.join(f'{cs[L][k]:5d}' for L in runs))
print('\nper-target n_ok where any loop count differs by >= 3')
for n in names:
    v = [ok[L][n] for L in runs]
    if max(v) - min(v) >= 3: print(f'  {n:16s} {pool[n]:26s} ' + ' '.join(f'L{L}={ok[L][n]:2d}' for L in runs))
