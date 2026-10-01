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


def cpu_quota():
    """CPUs this process may actually use: the cgroup quota (`cpu.max`, or cgroup v1's cfs quota; RunPod pods show 96 cores but grant ~7.6)
    intersected with the affinity mask."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    for fq, fp in (('/sys/fs/cgroup/cpu.max', None),                                            # cgroup v2
                   ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us')):  # v1 (RunPod A40s)
        try:
            if fp is None:
                q, per = open(fq).read().split()[:2]
            else:
                q, per = open(fq).read().strip(), open(fp).read().strip()
            if q not in ('max', '-1'):
                n = min(n, max(1, int(int(q) / int(per))))
            break
        except (OSError, ValueError):
            continue
    return n


# default: one Lean process per CPU of the quota (lean-prefilter, 2026-09-28; was cpu_count // 2, and job scripts set 3)
WORKERS = int(os.environ.get('LEAN_GATE_WORKERS', '0')) or cpu_quota()
THREADS = os.environ.get('LEAN_GATE_THREADS', '1')  # `lean -j N`; '' = Lean's default (a thread per hardware thread: 96 on a 7.6-CPU pod; measured 2-3x slower)
PREFILTER = os.environ.get('LEAN_PREFILTER', 'on')          # on | off | shadow  (see lean_prefilter.py, LEAN_GATE.md)
PIPELINE = os.environ.get('LEAN_GATE_PIPELINE', '1') == '1'  # sample.generate submits each decode chunk as it finishes
assert PREFILTER in ('on', 'off', 'shadow'), PREFILTER
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
        p = subprocess.run([LEAN] + (['-j', THREADS] if THREADS else []) + ['-DmaxErrors=100000000', fn], capture_output=True, text=True,
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
    cpu = sum(c for _, c in res)
    _count(len(srcs), cpu)
    return ok, time.time() - t0, cpu


def _count(n_texts, proc_s):
    """compute rows (REGISTRY.md): texts sent to a Lean process (registry / cache hits, parse failures and pre-filter
    rejects never reach here) and Lean process-seconds, added to the caller's open record.compute block."""
    import record
    record.count(lean_checks=n_texts, lean_s=proc_s)


def lean_check(items):
    """items: list of (one-line statement text, one-line tactic text) -> (list of bool, wall s, summed process s)."""
    return check_sources([f'{s} {b}' for s, b in items])




class Gate:
    """The gate over one generate() call.  submit() takes decode chunks as they finish: new distinct (prompt, text)
    pairs are pre-filtered (`lean_prefilter`, reject-only) and the survivors are handed to Lean worker threads in
    pieces of CHUNK while the GPU samples the next chunk; finish() waits for Lean and returns the ND proofs exactly
    as gate() always has.  LEAN_PREFILTER=off sends every text to Lean; =shadow sends every text to Lean, lets Lean
    decide, and logs each text the filter would have rejected but Lean accepted (a filter bug) to
    <log>.filterbug.jsonl."""

    def __init__(self, tok, pipeline=None):
        self.tok = tok
        self.pipeline = PIPELINE if pipeline is None else pipeline
        self.keys = {}            # (prompt, text) -> ND string, first-seen order
        self.filt = {}            # (prompt, text) -> filter reason or None
        self.lean_ok = {}         # (prompt, text) -> Lean verdict, for the texts sent to Lean
        self.pending = []
        self.futs = []
        self.stmt = {}
        self.n_piece = 0
        self.exposed_s = 0.0      # main-thread seconds spent inside submit()/finish()
        self.filter_s = 0.0
        self.t0 = time.time()
        self.wd = tempfile.mkdtemp(prefix='leangate_')
        self.ex = ThreadPoolExecutor(WORKERS)

    def _dispatch(self, keys):
        srcs = [f'{self.stmt[p]} {tx}' for p, tx in keys]
        tag = str(self.n_piece); self.n_piece += 1
        self.futs.append((keys, self.ex.submit(_run, srcs, self.wd, tag)))

    def submit(self, prompts, nd_proofs, texts):
        t0 = time.time()
        new = []
        for p, nd, tx in zip(prompts, nd_proofs, texts):
            if tx is not None and not nd.startswith('LEANPARSE') and (p, tx) not in self.keys:
                self.keys[(p, tx)] = nd; new.append((p, tx))
        tf = time.time()
        if PREFILTER != 'off':
            from lean_prefilter import reject_reason
        for p, tx in new:
            if p not in self.stmt:
                self.stmt[p] = self.tok.statement(p)
            r = reject_reason(self.stmt[p], tx) if PREFILTER != 'off' else None
            self.filt[(p, tx)] = r
            if r is None or PREFILTER == 'shadow':
                self.pending.append((p, tx))
        self.filter_s += time.time() - tf
        if self.pipeline:
            while len(self.pending) >= CHUNK:
                self._dispatch(self.pending[:CHUNK]); self.pending = self.pending[CHUNK:]
        self.exposed_s += time.time() - t0

    def finish(self, prompts, nd_proofs, texts):
        t0 = time.time()
        rest = self.pending; self.pending = []
        if self.pipeline:     # the tail: spread over the workers (>= 50 texts a process: a process costs ~2 s to start)
            n = max(1, min(WORKERS, len(rest) // 50))
            size = -(-len(rest) // n) if rest else 0
        else:
            size = CHUNK      # the pre-2026-09-28 gate: CHUNK pieces, all dispatched after sampling
        for i in range(0, len(rest), max(1, size)):
            self._dispatch(rest[i:i + size])
        cpu = 0.0
        for keys, fu in self.futs:
            ok, c = fu.result()
            cpu += c
            for k, v in zip(keys, ok):
                self.lean_ok[k] = bool(v)
        self.ex.shutdown()
        shutil.rmtree(self.wd, ignore_errors=True)
        _count(len(self.lean_ok), cpu)
        self.exposed_s += time.time() - t0
        return self._record(prompts, nd_proofs, texts, cpu, time.time() - t0)

    def _record(self, prompts, nd_proofs, texts, cpu, tail):
        import lean_judge
        logfn = os.environ.get('LEAN_GATE_LOG', 'artifacts/lean_gate.jsonl')
        items = list(self.keys.items())
        if PREFILTER == 'on':
            verdict = {k: self.filt[k] is None and self.lean_ok[k] for k, _ in items}
        else:
            verdict = {k: self.lean_ok[k] for k, _ in items}
        bugs = [k for k, _ in items if self.filt[k] is not None and self.lean_ok.get(k)] if PREFILTER == 'shadow' else []
        n_conflict0 = lean_judge.stats()['conflict']
        for (p, tx), nd in items:
            lean_judge.register(p, nd, verdict[(p, tx)])
        n_ok = sum(1 for v in verdict.values() if v)
        n_parse = sum(1 for nd in nd_proofs if nd.startswith('LEANPARSE'))
        parse_reasons = collections.Counter(nd[10:] for nd in nd_proofs if nd.startswith('LEANPARSE'))
        freasons = collections.Counter(r for r in self.filt.values() if r is not None)
        n_frej = sum(freasons.values())
        n_lean = len(self.lean_ok)
        n_trunc = parse_reasons.get('no-eos', 0)
        rec = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'samples': len(prompts), 'parse_fail': n_parse,
               'distinct_checked': len(items), 'lean_ok': n_ok, 'lean_rej': len(items) - n_ok,
               'registry_conflicts': lean_judge.stats()['conflict'] - n_conflict0,
               'parse_reasons': dict(parse_reasons.most_common()),
               'lean_wall_s': self.exposed_s, 'lean_tail_s': tail, 'gate_span_s': time.time() - self.t0,
               'lean_proc_s': cpu, 'lean_texts': n_lean, 'lean_procs': self.n_piece,
               'lean_proc_rej': sum(1 for v in self.lean_ok.values() if not v),
               'prefilter': PREFILTER, 'filter_rej': n_frej, 'filter_reasons': dict(freasons.most_common()),
               'filter_s': self.filter_s, 'filter_false_rej': len(bugs) if PREFILTER == 'shadow' else None,
               'trunc': n_trunc, 'trunc_rate': n_trunc / max(1, len(prompts)),
               'pipeline': self.pipeline, 'workers': WORKERS, 'chunk': CHUNK, 'judge': 'lean-only'}
        os.makedirs(os.path.dirname(logfn) or '.', exist_ok=True)
        with open(logfn, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        dump = os.environ.get('LEAN_GATE_DUMP')
        if dump:      # every distinct checked sample with its verdict: the audit trail an acceptance test or a reviewer
            with open(dump, 'a') as f:      # re-derives counts from, without re-running the model (run lean-judge, test 5)
                for (p, tx), nd in items:   # `lean`: Lean's own verdict, null if the filter kept the text from Lean
                    f.write(json.dumps({'prompt': p, 'lean_text': tx, 'nd': nd, 'lean_ok': verdict[(p, tx)],
                                        'filter': self.filt[(p, tx)], 'lean': self.lean_ok.get((p, tx))},
                                       ensure_ascii=False) + '\n')
        base = logfn.replace('.jsonl', '')
        rejs = [{'prompt': p, 'lean_text': tx, 'nd': nd, 'filter': self.filt[(p, tx)]} for (p, tx), nd in items
                if not verdict[(p, tx)]]
        if rejs:
            with open(base + '.leanrej.jsonl', 'a') as f:
                for d in rejs:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
        if bugs:
            with open(base + '.filterbug.jsonl', 'a') as f:
                for p, tx in bugs:
                    f.write(json.dumps({'prompt': p, 'lean_text': tx, 'filter': self.filt[(p, tx)]}, ensure_ascii=False) + '\n')
            print(f'[lean_gate] PREFILTER BUG: {len(bugs)} texts the filter rejects are accepted by Lean '
                  f'(see {base}.filterbug.jsonl)', flush=True)
        print(f'[lean_gate] {len(prompts)} samples, parse-fail {n_parse} (no-eos {n_trunc}, {100 * rec["trunc_rate"]:.3f} %), '
              f'distinct {len(items)}: ok {n_ok}, rej {len(items) - n_ok} (filter {n_frej}, prefilter={PREFILTER}); '
              f'lean on {n_lean} texts in {self.n_piece} procs, {cpu:.1f}s proc, exposed {self.exposed_s:.1f}s, '
              f'tail {tail:.1f}s, {WORKERS} workers', flush=True)
        if rec['trunc_rate'] > 0.001:
            print(f'[lean_gate] WARNING: {100 * rec["trunc_rate"]:.2f} % of samples hit max_new (policy: raise max_new '
                  f'above 0.1 %)', flush=True)
        out = []
        for p, nd, tx in zip(prompts, nd_proofs, texts):
            if tx is None or nd.startswith('LEANPARSE') or verdict[(p, tx)]:
                out.append(nd)               # accepted: a CLEAN ND string (this is what gets trained on), or already marked
            else:
                out.append('LEANREJ ' + nd)  # parsed the grammar, rejected by the pre-filter or by Lean: never counted
        return out


def gate(tok, prompts, nd_proofs, texts):
    """Lean on the literal text of every distinct (prompt, text); returns the ND proofs, clean where accepted and
    prefixed `LEANREJ ` where rejected.  Registers every verdict with lean_judge.  One-shot form of Gate."""
    g = Gate(tok, pipeline=False)
    g.submit(prompts, nd_proofs, texts)
    return g.finish(prompts, nd_proofs, texts)
