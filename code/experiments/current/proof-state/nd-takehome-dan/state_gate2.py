#!/usr/bin/env python3
"""Gate 2 of run `state-env`: **your state is Lean's state**.

For each of N random (proof, step) pairs from the control set, cut the proof at that step boundary, ask Lean for the
tactic state there (`trace_state` at the cut, `sorry` for the goal and for every goal still open outside it), and
compare Lean's hypotheses (names and types) and goal with `state_env.Env.state_tokens()`'s.  Zero mismatches is the bar.

  python3 state_gate2.py --data data/p2/train_depth3_f0_a1.jsonl --n 1000 --out artifacts/se/gate2.json

Comparison is on *parsed* formulas, not strings: Lean pretty-prints with minimal parentheses and the environment
renders fully parenthesised, so both sides are parsed to the same formula tuples.  Lean groups hypotheses of equal
type onto one line (`h1 n1 : P ∧ Q`); the group is expanded.  `P Q R S : Prop` is dropped (it is the same in every
state and the environment does not render it).
"""
import argparse, bisect, collections, json, os, random, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lean_tok import LeanTokenizer, ParseFail
from state_env import Env, seq_tokens, split_actions, parse_ftoks, canonicalise, NDMAP

LEAN = os.path.expanduser('~/.elan/bin/lean')
PRELUDE = 'set_option linter.unusedVariables false\nset_option maxRecDepth 4000\nset_option format.width 100000\n'
LOC = re.compile(r'^[^\s:]*\.lean:(\d+):\d+: (?:warning|error|information)')

# ---------------------------------------------------------------- Lean's printed formulas
ATOMS = {'P': ('atom', 'P'), 'Q': ('atom', 'Q'), 'R': ('atom', 'R'), 'S': ('atom', 'S'), 'False': ('bot',)}


def ltok(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1; continue
        if c in '()¬∧∨→':
            out.append(c); i += 1; continue
        j = i
        while j < len(s) and (s[j].isalnum() or s[j] == '_'):
            j += 1
        if j == i:
            raise ValueError(f'lean token {s[i:]!r}')
        out.append(s[i:j]); i = j
    return out


def lparse(s):
    """Lean's pretty-printed formula -> formula tuple.  ¬ binds tightest, then ∧ (right), ∨ (right), → (right)."""
    t = ltok(s)
    p = [0]

    def peek():
        return t[p[0]] if p[0] < len(t) else None

    def atom():
        x = peek()
        if x == '(':
            p[0] += 1
            f = imp()
            if peek() != ')':
                raise ValueError('paren')
            p[0] += 1
            return f
        if x == '¬':
            p[0] += 1
            return ('not', atom())
        if x in ATOMS:
            p[0] += 1
            return ATOMS[x]
        raise ValueError(f'atom {x}')

    def conj():
        l = atom()
        if peek() == '∧':
            p[0] += 1
            return ('and', l, conj())
        return l

    def disj():
        l = conj()
        if peek() == '∨':
            p[0] += 1
            return ('or', l, disj())
        return l

    def imp():
        l = disj()
        if peek() == '→':
            p[0] += 1
            return ('imp', l, imp())
        return l

    f = imp()
    if p[0] != len(t):
        raise ValueError('trailing')
    return f


def parse_lean_state(block):
    """trace_state output -> (list of (name, formula), goal formula)."""
    hyps = []
    goal = None
    for line in block.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith('⊢'):
            goal = lparse(line[1:])
            continue
        if ' : ' not in line:
            raise ValueError(f'hyp line {line!r}')
        names, ty = line.split(' : ', 1)
        if ty.strip() == 'Prop':
            continue
        f = lparse(ty)
        for n in names.split():
            hyps.append((n, f))
    if goal is None:
        raise ValueError('no goal')
    return hyps, goal


# ---------------------------------------------------------------- our state
def env_state(env):
    hyps = []
    for fr in env.frames:
        for line in fr.htoks:
            f, used = parse_ftoks(line, 2)
            assert used == len(line) - 3, line
            hyps.append((line[0], f))
    return hyps, env.frames[-1].goal


def completion(env):
    """the Lean text that closes every open goal, with `trace_state` at the focused one."""
    s = ['trace_state', ';', 'sorry']
    for fr in reversed(env.frames[1:]):
        s += [')'] + (['sorry'] if fr.kind == 'or1' else []) + [';', 'sorry']
    return s


# ---------------------------------------------------------------- Lean runner
def run_batch(srcs, workdir, tag):
    """srcs: one-line Lean sources -> list of trace blocks (str) in order."""
    text = PRELUDE
    starts = []
    for k, s in enumerate(srcs):
        starts.append(text.count('\n') + 1)
        text += s.replace('theorem t ', f'theorem t{k} ', 1).rstrip('\n') + '\n'
    fn = os.path.join(workdir, f'g2_{tag}.lean')
    open(fn, 'w').write(text)
    try:
        p = subprocess.run([LEAN, '-DmaxErrors=100000000', fn], capture_output=True, text=True,
                           timeout=120 + 2 * text.count('\n'))
        o = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        o = ''
    os.remove(fn)
    blocks = [None] * len(srcs)
    errs = [False] * len(srcs)
    buf = []
    for line in o.splitlines():
        m = LOC.match(line)
        if m:
            k = bisect.bisect_right(starts, int(m.group(1))) - 1
            if 0 <= k < len(srcs):
                if blocks[k] is None and buf:
                    blocks[k] = '\n'.join(buf)
                if ': error' in line:
                    errs[k] = True
            buf = []
        else:
            buf.append(line)
    return blocks, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/p2/train_depth3_f0_a1.jsonl')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=200)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--out', default='artifacts/se/gate2.json')
    ap.add_argument('--dump', default='artifacts/se/gate2_cases.jsonl')
    ap.add_argument('--kinds', default='', help='comma-separated focused-frame kinds to keep (default: any)')
    ap.add_argument('--canon', action='store_true', help='arm SN: canonical (scope-determined) names')
    a = ap.parse_args()
    tk = LeanTokenizer('lean_staten' if a.canon else 'lean_state')
    rng = random.Random(a.seed)
    recs = [json.loads(l) for l in open(a.data) if l.strip()]
    want = set(x for x in a.kinds.split(',') if x)
    cases = []
    seen = set()
    tries = 0
    while len(cases) < a.n and tries < 4000 * a.n:
        tries += 1
        r = recs[rng.randrange(len(recs))]
        toks = canonicalise(r['prompt'], r['proof']) if a.canon else seq_tokens(r['proof'])
        acts = split_actions(toks)
        i = rng.randrange(len(acts))
        key = (r['name'], i)
        if key in seen:
            continue
        seen.add(key)
        env = Env(r['prompt'], canon=a.canon)
        for a2 in acts[:i]:
            ok, why = env.apply(a2)
            assert ok, why
        if want and env.frames[-1].kind not in want:
            seen.discard(key)
            continue
        src = tk.statement(r['prompt']) + ' ' + tk.text(env.text + completion(env))
        hyps, goal = env_state(env)
        cases.append({'name': r['name'], 'step': i, 'n_steps': len(acts), 'src': src,
                      'ours': [[n, tk.text(__import__('lean_tok').ftoks(f))] for n, f in hyps],
                      'ours_goal': tk.text(__import__('lean_tok').ftoks(goal)),
                      '_hyps': hyps, '_goal': goal, 'depth': len(env.frames) - 1,
                      'kind': env.frames[-1].kind})
    wd = tempfile.mkdtemp(prefix='seg2_')
    t0 = time.time()
    chunks = [(cases[i:i + a.chunk], i) for i in range(0, len(cases), a.chunk)]
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda c: run_batch([x['src'] for x in c[0]], wd, str(c[1])), chunks))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    blocks = [b for bs, _ in res for b in bs]
    errs = [e for _, es in res for e in es]
    mism = []
    n_ok = 0
    by_kind = collections.Counter()
    for c, b, e in zip(cases, blocks, errs):
        by_kind[c['kind']] += 1
        if e or b is None:
            mism.append({**{k: c[k] for k in ('name', 'step', 'kind', 'src', 'ours', 'ours_goal')},
                         'why': 'lean error' if e else 'no trace'})
            continue
        try:
            lh, lg = parse_lean_state(b)
        except Exception as ex2:
            mism.append({**{k: c[k] for k in ('name', 'step', 'kind', 'src', 'ours', 'ours_goal')},
                         'why': f'parse: {ex2}', 'lean': b})
            continue
        if lh == c['_hyps'] and lg == c['_goal']:
            n_ok += 1
        else:
            mism.append({**{k: c[k] for k in ('name', 'step', 'kind', 'src', 'ours', 'ours_goal')},
                         'why': 'mismatch', 'lean': b})
    out = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'data': a.data, 'n_cases': len(cases),
           'agree': n_ok, 'mismatches': len(mism), 'by_focused_frame_kind': dict(by_kind),
           'lean_secs': time.time() - t0, 'seed': a.seed}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    with open(a.dump, 'w') as f:
        for c in cases:
            f.write(json.dumps({k: c[k] for k in ('name', 'step', 'n_steps', 'depth', 'kind', 'src', 'ours', 'ours_goal')}, ensure_ascii=False) + '\n')
    if mism:
        with open(a.out.replace('.json', '') + '_mismatches.jsonl', 'w') as f:
            for m in mism:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(json.dumps(out, indent=1))
    for m in mism[:5]:
        print('MISMATCH', m['why'], m['name'], m['step'], '\n ours', m['ours'], m['ours_goal'], '\n lean', m.get('lean'))


if __name__ == '__main__':
    main()
