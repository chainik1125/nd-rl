#!/usr/bin/env python3
"""ND (spec.md) -> Lean 4 translation and checking (run 1, step 1).  Core Lean only (no Mathlib).

Translation (deterministic): atoms P Q R S are Prop variables; premises are hypotheses n1..nk (each PR line is re-stated
with its own formula, so a PR line that does not match the declared premise fails to type-check); every other line is a
`have nI : F := term`; a box (AS ... last line) becomes a lambda `fun nS => by <lines>; exact nE` emitted at the rule that
discharges it: IMPI (A → B), NEGI (¬A = A → False), ORE (Or.elim nJ (fun ..) (fun ..)); R = the cited name; ANDI ⟨a, b⟩;
ANDE1/2 .1/.2; IMPE application; ORI1/2 Or.inl/inr; NEGE `nb na` (¬A applied to A : False); BOTE `na.elim`;
DN `Classical.byContradiction (fun h => na h)`.  The verifier's structural rules are mirrored where Lean would not enforce
them: consecutive indices, depth transitions, boxes closed by a shallower line, a box cited only after it is closed and
with its exact last line, the final line at depth 0.  Violations raise TranslationError (reported as "structural").
Proofs whose citations are out of scope (a line inside a closed box) translate to Lean terms with unbound names, so Lean
rejects them itself.

  python nd2lean.py --check FILE.jsonl [--limit N] [--per_file 40] --out report.jsonl    # nd_verify vs Lean, per record
  (Lean stops reporting after ~100 errors per file, so chunks are small and re-split when the cap is hit)
  python nd2lean.py --one "THM ... PRF N1 ... QED"                                       # print the Lean source
Records: {prompt, proof} or {text}; a record with 'ok' (judged file) is used as-is, otherwise nd_verify is run.
"""
import argparse, json, os, sys, subprocess, tempfile, collections, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens, parse_formula, ParseError

LEAN = os.path.expanduser('~/.elan/bin/lean')


class TranslationError(Exception):
    pass


def lf(f):
    """formula tuple -> Lean 4 Prop expression (fully parenthesised)"""
    t = f[0]
    if t == 'atom': return f[1]
    if t == 'bot': return 'False'
    if t == 'not': return f'(¬{lf(f[1])})'
    op = {'and': '∧', 'or': '∨', 'imp': '→'}[t]
    return f'({lf(f[1])} {op} {lf(f[2])})'


def parse_prompt(prompt):
    toks = prompt.split()
    if toks[0] != 'THM' or 'SEQ' not in toks or toks[-1] != 'PRF':
        raise TranslationError('bad prompt')
    i = 1; prem = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i); prem.append(f)
            if toks[i] == ',': i += 1; continue
            break
    if toks[i] != 'SEQ': raise TranslationError('bad prompt')
    concl, i = parse_formula(toks, i + 1)
    if toks[i] != 'PRF' or i != len(toks) - 1: raise TranslationError('bad prompt')
    return prem, concl


class Box:
    def __init__(self, start, depth, hyp):
        self.start, self.depth, self.hyp = start, depth, hyp
        self.lines = []      # emitted Lean lines (strings, already indented relative to the box body)
        self.last = start    # idx of the last line inside the box (at the box's own depth)
        self.last_formula = hyp


def translate(prompt, proof, require_all_pr=True):
    """-> Lean source for one theorem named `t`, or raise TranslationError for structural violations.

    require_all_pr=False allows a proof that re-states only the first k of the declared premises (`lean_judge`, run
    lean-judge, 2026-09-27).  Lean does not require a premise to be re-stated; `lean_tok.inverse` accepts a literal text
    whose `h`-citations are a PREFIX h1..hk in order (`raise ParseFail('PR position')` otherwise), so the k-th PR line is
    still the k-th declared premise and the rendering is unchanged -- only this check is dropped.  The remaining premises
    stay declared in the statement and are simply unused, which cannot make a false theorem provable."""
    prem, concl = parse_prompt(prompt)
    toks = proof.split()
    if toks.count('QED') != 1 or toks[-1] != 'QED':
        raise TranslationError('QED')
    try:
        lines = parse_proof_tokens(toks)
    except ParseError as e:
        raise TranslationError(f'parse: {e}')
    if not lines:
        raise TranslationError('empty')
    # consecutive indices
    for a, b in zip(lines, lines[1:]):
        if b['idx'] != a['idx'] + 1:
            raise TranslationError('index')
    root = Box(None, 0, None)
    stack = [root]                 # open boxes; stack[-1] is the innermost
    closed = {}                    # start idx -> Box (closed, awaiting its discharging rule)
    formulas = {}                  # idx -> formula (for type annotations)
    ind = lambda: '  ' * len(stack)
    n_pr = 0
    for k, ln in enumerate(lines):
        i, d, f, rule, refs = ln['idx'], ln['depth'], ln['formula'], ln['rule'], ln['refs']
        cur = len(stack) - 1
        if rule == 'AS':
            if d < 1 or d > cur + 1:
                raise TranslationError('AS depth')
            while len(stack) - 1 >= d:           # close boxes with depth >= d
                b = stack.pop(); closed[b.start] = b
            b = Box(i, d, f); stack.append(b); formulas[i] = f
            continue
        if rule == 'PR':
            if d != 0 or k != n_pr:
                raise TranslationError('PR position')
            if n_pr >= len(prem):
                raise TranslationError('extra PR')
            n_pr += 1
            formulas[i] = f
            root.lines.append(f'have n{i} : {lf(f)} := h{n_pr}')     # k-th PR line vs the k-th declared premise (line indices may start anywhere)
            root.last = i; root.last_formula = f
            continue
        if d > cur:
            raise TranslationError('depth jump')
        while len(stack) - 1 > d:                # a line at depth d closes deeper boxes
            b = stack.pop(); closed[b.start] = b
        box = stack[-1]
        name = lambda j: f'n{j}'
        def cite(j):
            if j >= i: raise TranslationError('forward cite')
            return name(j)
        def box_term(s, e, want_hyp=None, want_false=False):
            b = closed.get(s)
            if b is None or b.last != e:
                raise TranslationError('bad box cite')
            if want_hyp is not None and b.hyp != want_hyp:
                raise TranslationError('box hyp')
            body = '\n'.join(('  ' * (len(stack) + 1)) + l for l in b.lines) if b.lines else ''
            ex = ('  ' * (len(stack) + 1)) + (f'exact (n{b.last} : False)' if want_false else f'exact n{b.last}')
            return f'(fun (n{s} : {lf(b.hyp)}) => by\n{body}\n{ex})' if body else f'(fun (n{s} : {lf(b.hyp)}) => by\n{ex})'
        F = lf(f)
        if rule == 'R': term = cite(refs[0]) if len(refs) == 1 else None
        elif rule == 'ANDI': term = f'⟨{cite(refs[0])}, {cite(refs[1])}⟩' if len(refs) == 2 else None
        elif rule == 'ANDE1': term = f'{cite(refs[0])}.1' if len(refs) == 1 else None
        elif rule == 'ANDE2': term = f'{cite(refs[0])}.2' if len(refs) == 1 else None
        elif rule == 'IMPE': term = f'{cite(refs[0])} {cite(refs[1])}' if len(refs) == 2 else None
        elif rule == 'ORI1': term = f'Or.inl {cite(refs[0])}' if len(refs) == 1 else None
        elif rule == 'ORI2': term = f'Or.inr {cite(refs[0])}' if len(refs) == 1 else None
        elif rule == 'NEGE': term = f'{cite(refs[1])} {cite(refs[0])}' if len(refs) == 2 else None
        elif rule == 'BOTE': term = f'{cite(refs[0])}.elim' if len(refs) == 1 else None
        elif rule == 'DN': term = f'Classical.byContradiction (fun hh => {cite(refs[0])} hh)' if len(refs) == 1 else None
        elif rule == 'IMPI':
            if len(refs) != 2: term = None
            else:
                s, e = refs
                if e >= i or s >= i: raise TranslationError('forward cite')
                term = box_term(s, e)
        elif rule == 'NEGI':
            if len(refs) != 2: term = None
            else:
                s, e = refs
                if e >= i: raise TranslationError('forward cite')
                term = box_term(s, e, want_false=True)     # the box must end in False; the have's type must be ¬hyp
        elif rule == 'ORE':
            if len(refs) != 5: term = None
            else:
                j, s1, e1, s2, e2 = refs
                if max(refs) >= i: raise TranslationError('forward cite')
                term = f'Or.elim {cite(j)} {box_term(s1, e1)} {box_term(s2, e2)}'
        else:
            raise TranslationError(f'rule {rule}')
        if term is None:
            raise TranslationError('arity')
        box.lines.append(f'have n{i} : {F} := {term}')
        box.last = i; box.last_formula = f; formulas[i] = f
    if len(stack) != 1:
        raise TranslationError('final line inside a box')
    if n_pr != len(prem) and require_all_pr:
        raise TranslationError('missing PR')
    last = lines[-1]
    if last['rule'] == 'AS' or (last['rule'] == 'PR' and last['formula'] != concl):
        raise TranslationError('final line')
    hyps = ' '.join(f'(h{j+1} : {lf(p)})' for j, p in enumerate(prem))
    body = '\n'.join('  ' + l for l in root.lines)
    src = f'theorem t (P Q R S : Prop) {hyps} : {lf(concl)} := by\n{body}\n  exact n{last["idx"]}\n'
    return src


def lean_check(srcs, per_file=40, workdir=None, _depth=0):
    """srcs: list of Lean theorem sources (each `theorem t ...`). Returns list of (ok, message)."""
    out = [None] * len(srcs)
    workdir = workdir or tempfile.mkdtemp(prefix='nd2lean_')
    for b in range(0, len(srcs), per_file):
        chunk = srcs[b:b + per_file]
        text = 'set_option maxRecDepth 4000\n'; starts = []
        for k, s in enumerate(chunk):
            starts.append(text.count('\n') + 1)
            text += s.replace('theorem t ', f'theorem t{k} ', 1) + '\n'
        fn = os.path.join(workdir, f'chunk_{b}.lean')
        open(fn, 'w').write(text)
        p = subprocess.run([LEAN, fn], capture_output=True, text=True)
        errs = collections.defaultdict(list)
        for m in re.finditer(r'^[^\n]*?:(\d+):(\d+): error(?:\([^)]*\))?: ([^\n]*)', p.stdout + p.stderr, re.M):
            line = int(m.group(1))
            k = max(j for j, s in enumerate(starts) if s <= line)
            errs[k].append(m.group(3)[:120])
        n_err = len(re.findall(r': error(?:\([^)]*\))?: ', p.stdout + p.stderr))
        if n_err >= 95 and len(chunk) > 1 and _depth < 3:      # Lean stops reporting after ~100 errors per file: split the chunk
            sub = lean_check(chunk, max(1, len(chunk) // 4), workdir, _depth + 1)
            for k in range(len(chunk)):
                out[b + k] = sub[k]
            continue
        if p.returncode != 0 and not errs:      # a crash or a parse error we could not attribute: mark the whole chunk
            for k in range(len(chunk)):
                errs[k].append('lean: ' + (p.stderr or p.stdout)[:120])
        for k in range(len(chunk)):
            out[b + k] = (k not in errs, '; '.join(errs.get(k, [])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check'); ap.add_argument('--out'); ap.add_argument('--limit', type=int); ap.add_argument('--per_file', type=int, default=40)
    ap.add_argument('--one'); ap.add_argument('--field', default='proof')
    a = ap.parse_args()
    if a.one:
        prompt, proof = a.one.split(' PRF ', 1); prompt += ' PRF'
        src = translate(prompt, proof); print(src); print(lean_check([src]))
        return
    recs = [json.loads(l) for l in open(a.check) if l.strip()]
    if a.limit: recs = recs[:a.limit]
    rows, srcs, idx = [], [], []
    t0 = time.time()
    for r in recs:
        if 'text' in r and 'prompt' not in r:
            prompt, proof = r['text'].split(' PRF ', 1); prompt += ' PRF'
        else:
            prompt = r['prompt']; proof = r.get(a.field) or r.get('proof') or r.get('gen_proof') or r.get('reference_proof')
        ok = r['ok'] if 'ok' in r else verify_text(prompt + ' ' + proof)[0]
        reason = r.get('reason') if 'ok' in r else verify_text(prompt + ' ' + proof)[1]
        row = {'name': r.get('name'), 'prompt': prompt, 'proof': proof, 'nd_ok': bool(ok), 'nd_reason': reason}
        try:
            src = translate(prompt, proof); row['lean_src'] = src; srcs.append(src); idx.append(len(rows))
        except TranslationError as e:
            row['lean_ok'] = False; row['lean_reason'] = f'structural: {e}'
        except Exception as e:
            row['lean_ok'] = False; row['lean_reason'] = f'translator crash: {type(e).__name__}: {e}'
        rows.append(row)
    res = lean_check(srcs, a.per_file)
    for k, (ok, msg) in zip(idx, res):
        rows[k]['lean_ok'] = ok; rows[k]['lean_reason'] = msg
    c = collections.Counter((r['nd_ok'], r['lean_ok']) for r in rows)
    print(f'{len(rows)} records in {time.time()-t0:.0f}s: nd_ok&lean_ok {c[(True, True)]}, nd_ok&lean_rej {c[(True, False)]}, nd_rej&lean_ok {c[(False, True)]}, both reject {c[(False, False)]}')
    dis = [r for r in rows if r['nd_ok'] != r['lean_ok']]
    for r in dis[:10]:
        print('DISAGREE', r['nd_ok'], r['lean_ok'], r['nd_reason'], '|', r['lean_reason'][:100], '|', r['proof'][:150])
    if a.out:
        with open(a.out, 'w') as f:
            for r in rows:
                r.pop('lean_src', None); f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
