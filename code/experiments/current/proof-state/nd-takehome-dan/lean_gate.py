"""Lean as the only in-loop checker for Lean-format models (runs lean-format, ds-*, lean-judge).

gate(tok, prompts, nd_proofs, texts) is called by sample.generate() for a LeanTokenizer.  For every distinct
(prompt, literal Lean text) whose text parsed in the strict grammar it asks Lean (core, no Mathlib) whether the literal
text proves the theorem, logs the outcome and the timings, and returns the ND proofs.

**Lean alone decides** (Dan, 2026-09-27; see the 2026-09-27 update of nd-rl's
`docs/project_strategy/2026-09-20-lean-default.md` and `LEAN_JUDGE.md`).  This file no longer calls `nd_verify`.
Accepted samples leave the gate as **clean ND strings** -- whatever the judge accepts becomes expert-iteration training
text, so a verdict prefix on an accepted sample would be trained on (pitfall 1).  Only rejects carry a marker:
`LEANREJ <nd>` when the literal text parsed the grammar but Lean rejected it.  Every verdict is also registered in
`lean_judge`'s in-process registry, so the judging pass that follows generate() is a dictionary lookup rather than a
second Lean run.

One theorem per chunk file line (or several lines, for `check_sources`), `lean -DmaxErrors=...` so every error is
reported; error line -> theorem.  A chunk whose lean process crashes / times out is split recursively; a single theorem
that crashes is rejected.
Log: $LEAN_GATE_LOG (default artifacts/lean_gate.jsonl), one json line per generate() call; Lean's rejects go to
<log>.leanrej.jsonl.  Set $LEAN_GATE_DUMP to append every distinct checked (prompt, literal text, ND string, verdict) to
a file -- the audit trail a reviewer re-derives counts from without re-running the model.
"""
import os, re, sys, json, time, bisect, subprocess, tempfile, shutil, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEAN = os.path.expanduser('~/.elan/bin/lean')
CHUNK = int(os.environ.get('LEAN_GATE_CHUNK', '400'))
WORKERS = int(os.environ.get('LEAN_GATE_WORKERS', str(max(1, (os.cpu_count() or 2) // 2))))
ERR = re.compile(r'^[^\n]*?:(\d+):\d+: error', re.M)
PRELUDE = 'set_option linter.unusedVariables false\nset_option maxRecDepth 4000\n'


def _run(srcs, workdir, tag, depth=0):
    """srcs: list of Lean theorem sources, each starting `theorem t ` and possibly spanning several lines
    -> (list of bool, cpu seconds)."""
    text = PRELUDE
    starts = []
    for k, s in enumerate(srcs):
        starts.append(text.count('\n') + 1)
        text += s.replace('theorem t ', f'theorem t{k} ', 1).rstrip('\n') + '\n'
    fn = os.path.join(workdir, f'c_{tag}.lean')
    with open(fn, 'w') as f:
        f.write(text)
    t0 = time.time()
    try:
        p = subprocess.run([LEAN, '-DmaxErrors=100000000', fn], capture_output=True, text=True,
                           timeout=60 + 2 * text.count('\n'))
        o = p.stdout + p.stderr; rc = p.returncode
    except subprocess.TimeoutExpired:
        o = ''; rc = -9
    cpu = time.time() - t0
    os.remove(fn)
    bad = set(); stray = False
    for m in ERR.finditer(o):
        k = bisect.bisect_right(starts, int(m.group(1))) - 1
        if 0 <= k < len(srcs):
            bad.add(k)
        else:
            stray = True
    crashed = rc not in (0, 1) or (rc == 1 and not bad) or stray
    if crashed:
        if len(srcs) == 1:
            return [False], cpu
        h = len(srcs) // 2
        a, ca = _run(srcs[:h], workdir, tag + 'a', depth + 1); b, cb = _run(srcs[h:], workdir, tag + 'b', depth + 1)
        return a + b, cpu + ca + cb
    return [k not in bad for k in range(len(srcs))], cpu


def check_sources(srcs):
    """srcs: list of Lean theorem sources (each starting `theorem t `, may span lines)
    -> (list of bool, wall seconds, summed process seconds).  Chunked and run in a thread pool."""
    if not srcs:
        return [], 0.0, 0.0
    wd = tempfile.mkdtemp(prefix='leangate_')
    t0 = time.time()
    chunks = [(srcs[i:i + CHUNK], i) for i in range(0, len(srcs), CHUNK)]
    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            res = list(ex.map(lambda c: _run(c[0], wd, str(c[1])), chunks))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    ok = [x for r, _ in res for x in r]
    return ok, time.time() - t0, sum(c for _, c in res)


def lean_check(items):
    """items: list of (one-line statement text, one-line tactic text) -> (list of bool, wall s, summed process s)."""
    return check_sources([f'{s} {b}' for s, b in items])


def gate(tok, prompts, nd_proofs, texts):
    """Lean on the literal text of every distinct (prompt, text); returns the ND proofs, clean where Lean accepted and
    prefixed `LEANREJ ` where Lean rejected.  Registers every verdict with lean_judge."""
    import lean_judge
    logfn = os.environ.get('LEAN_GATE_LOG', 'artifacts/lean_gate.jsonl')
    keys = {}
    for p, nd, tx in zip(prompts, nd_proofs, texts):
        if tx is not None and not nd.startswith('LEANPARSE'):
            keys.setdefault((p, tx), nd)
    items = list(keys.items())
    lean_ok, wall, cpu = lean_check([(tok.statement(p), tx) for (p, tx), _ in items])
    verdict = {k: bool(lo) for (k, _), lo in zip(items, lean_ok)}
    n_conflict0 = lean_judge.stats()['conflict']
    for ((p, tx), nd), lo in zip(items, lean_ok):
        lean_judge.register(p, nd, bool(lo))
    n_ok = sum(1 for v in verdict.values() if v)
    n_parse = sum(1 for nd in nd_proofs if nd.startswith('LEANPARSE'))
    parse_reasons = collections.Counter(nd[10:] for nd in nd_proofs if nd.startswith('LEANPARSE'))
    rec = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'samples': len(prompts), 'parse_fail': n_parse,
           'distinct_checked': len(items), 'lean_ok': n_ok, 'lean_rej': len(items) - n_ok,
           'registry_conflicts': lean_judge.stats()['conflict'] - n_conflict0,
           'parse_reasons': dict(parse_reasons.most_common()), 'lean_wall_s': wall, 'lean_proc_s': cpu,
           'workers': WORKERS, 'chunk': CHUNK, 'judge': 'lean-only'}
    os.makedirs(os.path.dirname(logfn) or '.', exist_ok=True)
    with open(logfn, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    dump = os.environ.get('LEAN_GATE_DUMP')
    if dump:      # every distinct checked sample with its Lean verdict: the audit trail an acceptance test or a reviewer
        with open(dump, 'a') as f:      # re-derives counts from, without re-running the model (run lean-judge, test 5)
            for (p, tx), nd in items:
                f.write(json.dumps({'prompt': p, 'lean_text': tx, 'nd': nd, 'lean_ok': verdict[(p, tx)]}, ensure_ascii=False) + '\n')
    rejs = [{'prompt': p, 'lean_text': tx, 'nd': nd} for ((p, tx), nd) in items if not verdict[(p, tx)]]
    if rejs:
        with open(logfn.replace('.jsonl', '') + '.leanrej.jsonl', 'a') as f:
            for d in rejs:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'[lean_gate] {len(prompts)} samples, parse-fail {n_parse}, distinct checked {len(items)}: lean ok {n_ok}, '
          f'lean rej {len(items) - n_ok}; lean {wall:.1f}s wall ({cpu:.1f}s proc, {WORKERS} workers)', flush=True)
    out = []
    for p, nd, tx in zip(prompts, nd_proofs, texts):
        if tx is None or nd.startswith('LEANPARSE') or verdict[(p, tx)]:
            out.append(nd)               # accepted: a CLEAN ND string (this is what gets trained on), or already marked
        else:
            out.append('LEANREJ ' + nd)  # the literal text parsed the grammar but Lean rejected it: never counted
    return out
