"""Lean as the ONLY checker (long_rl_20261014 overlay on the fork's lean_gate.py at 51604b3).

gate(tok, prompts, nd_proofs, texts) is called by sample.generate() for a LeanTokenizer. For every distinct
(prompt, literal Lean text) whose text parsed in the strict grammar it asks Lean (core) whether the literal text proves the
theorem, and returns the ND proofs with 'LEANREJ ' prefixed where Lean rejects. Unlike the fork's gate there is no
nd_verify: a sample counts iff Lean accepts it (Robbie, 2026-10-14). The ND proof returned is only the decoder's rendering,
used for line counts. lean_check() below is the fork's, unchanged.

Log: $LEAN_GATE_LOG (default artifacts/lean_gate.jsonl), one json line per generate() call.
"""
import os, re, sys, json, time, subprocess, tempfile, shutil, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEAN = os.path.expanduser('~/.elan/bin/lean')
CHUNK = int(os.environ.get('LEAN_GATE_CHUNK', '400'))
WORKERS = int(os.environ.get('LEAN_GATE_WORKERS', str(max(1, (os.cpu_count() or 2) // 2))))
ERR = re.compile(r'^[^\n]*?:(\d+):\d+: error', re.M)


def _run(lines, workdir, tag, depth=0):
    """lines: list of one-line theorem sources -> (list of bool, cpu seconds)"""
    fn = os.path.join(workdir, f'c_{tag}.lean')
    with open(fn, 'w') as f:
        f.write('set_option linter.unusedVariables false\n' + '\n'.join(lines) + '\n')
    t0 = time.time()
    try:
        p = subprocess.run([LEAN, '-DmaxErrors=100000000', fn], capture_output=True, text=True, timeout=60 + 2 * len(lines))
        o = p.stdout + p.stderr; rc = p.returncode
    except subprocess.TimeoutExpired:
        o = ''; rc = -9
    cpu = time.time() - t0
    os.remove(fn)
    bad = {int(m.group(1)) - 2 for m in ERR.finditer(o)}
    crashed = rc not in (0, 1) or (rc == 1 and not bad) or any(b < 0 or b >= len(lines) for b in bad)
    if crashed:
        if len(lines) == 1:
            return [False], cpu
        h = len(lines) // 2
        a, ca = _run(lines[:h], workdir, tag + 'a', depth + 1); b, cb = _run(lines[h:], workdir, tag + 'b', depth + 1)
        return a + b, cpu + ca + cb
    return [k not in bad for k in range(len(lines))], cpu


def lean_check(items):
    """items: list of (statement text, tactic text) -> (list of bool, wall seconds, summed process seconds)"""
    if not items:
        return [], 0.0, 0.0
    lines = [f'{s.replace("theorem t ", f"theorem t{k} ", 1)} {b}' for k, (s, b) in enumerate(items)]
    wd = tempfile.mkdtemp(prefix='leangate_')
    t0 = time.time()
    chunks = [(lines[i:i + CHUNK], i) for i in range(0, len(lines), CHUNK)]
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(lambda c: _run(c[0], wd, str(c[1])), chunks))
    shutil.rmtree(wd, ignore_errors=True)
    ok = [x for r, _ in res for x in r]
    return ok, time.time() - t0, sum(c for _, c in res)


def gate(tok, prompts, nd_proofs, texts):
    logfn = os.environ.get('LEAN_GATE_LOG', 'artifacts/lean_gate.jsonl')
    keys = {}
    for p, nd, tx in zip(prompts, nd_proofs, texts):
        if tx is not None and not nd.startswith('LEANPARSE'):
            keys.setdefault((p, tx), nd)
    items = list(keys)
    lean_ok, wall, cpu = lean_check([(tok.statement(p), tx) for p, tx in items])
    verdict = dict(zip(items, lean_ok))
    n_parse = sum(1 for nd in nd_proofs if nd.startswith('LEANPARSE'))
    parse_reasons = collections.Counter(nd[10:] for nd in nd_proofs if nd.startswith('LEANPARSE'))
    n_ok = sum(map(bool, lean_ok))
    rec = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'samples': len(prompts), 'parse_fail': n_parse,
           'distinct_checked': len(items), 'lean_ok': n_ok, 'lean_rej': len(items) - n_ok,
           'parse_reasons': dict(parse_reasons.most_common()), 'lean_wall_s': wall, 'lean_proc_s': cpu,
           'workers': WORKERS, 'chunk': CHUNK}
    os.makedirs(os.path.dirname(logfn) or '.', exist_ok=True)
    with open(logfn, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f'[lean_gate] {len(prompts)} samples, parse-fail {n_parse}, distinct checked {len(items)}: lean ok {n_ok}; '
          f'lean {wall:.1f}s wall ({cpu:.1f}s proc, {WORKERS} workers)', flush=True)
    return [nd if tx is None or nd.startswith('LEANPARSE') or verdict[(p, tx)] else 'LEANREJ ' + nd
            for p, nd, tx in zip(prompts, nd_proofs, texts)]
