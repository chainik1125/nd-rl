#!/usr/bin/env python3
"""lean_check — the free-form-Lean judge: Lean alone decides (proposal 9, run lean-only; allowlist widened in run lean-judge).

A candidate proof is ACCEPTED iff, in Lean 4 core (`import Lean`, no Mathlib),
  (a) the theorem elaborates without error,
  (b) its axioms are a subset of {propext, Classical.choice, Quot.sound}  (so no sorryAx),
  (c) every constant of the elaborated value is on the ALLOWLIST below (the constructors and eliminators of And, Or,
      Not, False, Iff, plus Classical.byContradiction; the type formers; `letFun`, which is what `have` elaborates to).
**2026-09-27 (run lean-judge, Dan's decision):** `Not.elim`, `Not.intro`, `And.elim` and `Iff.elim` are ON the allowlist.
`n.elim` on `n : ¬A` is a valid proof (Lean is sound); ND's `BOTE` rule forbids it only because it must cite a line that
is literally `F`, so the same inference takes an extra box, `NEGE` and `BOTE`.  The allowlist already admitted
`absurd : a → ¬a → b`, which is `Not.elim` with its arguments swapped.  `And.elim` and `Iff.elim` are the other
eliminators the `lean_seq` grammar can reach through `.elim`, so the free-form judge and the `lean_seq` judge accept the
same inferences.
The check inspects the elaborated term, never the source text: tactic sugar (intro, exact, constructor, cases, refine)
is fine when it elaborates to allowed constants; `simp`, `decide`, `omega`, `tauto`, `Classical.em`, `Decidable.*`,
`Or.resolve_*`, `propext` as a term, any library lemma, are rejected.

term_size = inference nodes of the elaborated value:
  1 per application headed by an allowed inference constant (type-former arguments contribute 0),
  1 per argument of an application headed by a local hypothesis (each is a modus ponens / ¬-elimination),
  1 per `fun` binder, 1 per structure projection (`.1`/`.2`);
  `letFun v (fun x => b)` (a `have`) counts size(v) + size(b); variables, types, mdata count 0.
So for nd2lean's rendering: IMPE/NEGE/ANDE/ANDI/ORI/BOTE = 1, IMPI/NEGI = 1 + body, ORE = 1 + 2 binders + bodies,
DN (`Classical.byContradiction (fun hh => n hh)`) = 3, R/PR = 0.

Interface
  check(sources, workers=, chunk=) -> (list of {'ok', 'size', 'reason'}, wall_s, summed process_s)
      sources: list of Lean theorem sources, each starting `theorem t ` (renamed t<k> internally; may span lines).
  python3 lean_check.py --selftest                                  hand-written cases -> artifacts/lj/lean_check_selftest.json
  python3 lean_check.py --check FILE.jsonl --out REPORT.jsonl       ND records {prompt, proof} (or {text}): nd2lean.translate + check
  ... --compare_nd                                                  ALSO run nd_verify and print the agreement table.  Off by
                                                                    default: nd_verify judges nothing (Dan, 2026-09-27); this
                                                                    flag exists only to reproduce pre-2026-09-27 numbers.
  python3 lean_check.py --texts FILE.jsonl --out REPORT.jsonl       literal Lean records {prompt, lean_text}: statement + text
Env: LEAN_CHECK_WORKERS (default cpu_count//2), LEAN_CHECK_CHUNK (default 300), LEAN (path to the lean binary).
"""
import os, re, sys, json, time, argparse, subprocess, tempfile, shutil, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEAN = os.environ.get('LEAN', os.path.expanduser('~/.elan/bin/lean'))
CHUNK = int(os.environ.get('LEAN_CHECK_CHUNK', '300'))
WORKERS = int(os.environ.get('LEAN_CHECK_WORKERS', str(max(1, (os.cpu_count() or 2) // 2))))
OK_AXIOMS = ('propext', 'Classical.choice', 'Quot.sound')
ALLOW_INF = ('And.intro', 'And.left', 'And.right', 'Or.inl', 'Or.inr', 'Or.elim', 'False.elim', 'absurd',
             'Iff.intro', 'Iff.mp', 'Iff.mpr', 'Classical.byContradiction',
             'And.rec', 'And.casesOn', 'Or.rec', 'Or.casesOn', 'False.rec', 'Iff.rec', 'Iff.casesOn',
             'Not.elim', 'Not.intro', 'And.elim', 'Iff.elim')   # 2026-09-27 (run lean-judge): `.elim` on ¬A / ∧ / ↔ is sound
ALLOW_TY = ('And', 'Or', 'Not', 'Iff', 'False')
ALLOW_SCAFFOLD = ('letFun',)

PRELUDE = r'''import Lean
open Lean Meta Elab Command
set_option linter.unusedVariables false
set_option maxRecDepth 4000
set_option Elab.async false
namespace NdCheck
def infList : List Name := [%(inf)s]
def tyList : List Name := [%(ty)s]
def okAxioms : List Name := [%(ax)s]
def allowed : List Name := infList ++ tyList ++ [``letFun]
partial def strip : Nat → Expr → Expr
  | 0, e => e
  | n+1, e => match e.consumeMData with
    | .lam _ _ b _ => strip n b
    | e' => e'
partial def size (e : Expr) : Nat :=
  match e with
  | .mdata _ b => size b
  | .lam _ _ b _ => 1 + size b
  | .letE _ _ v b _ => size v + size b
  | .proj _ _ b => 1 + size b
  | .app .. =>
    let f := e.getAppFn.consumeMData
    let args := e.getAppArgs
    let sumArgs (as : Array Expr) : Nat := as.foldl (fun s a => s + size a) 0
    match f with
    | .const c _ =>
      if c == ``letFun then
        let v := args.getD 2 (.sort .zero)
        let fb := args.getD 3 (.sort .zero)
        let inner := match fb.consumeMData with | .lam _ _ b _ => size b | fb' => size fb'
        size v + inner + sumArgs (args.extract 4 args.size)
      else if tyList.contains c then 0
      else 1 + sumArgs args
    | .lam .. => size f + sumArgs args
    | .proj _ _ b => 1 + size b + args.size + sumArgs args
    | _ => args.size + sumArgs args
  | .const c _ => if infList.contains c then 1 else 0
  | _ => 0
syntax (name := ndCheck) "nd_check " num num : command
@[command_elab ndCheck] def elabNdCheck : CommandElab := fun stx => do
  let k := stx[1].toNat
  let np := stx[2].toNat
  let n := Name.mkSimple s!"t{k}"
  match (← getEnv).find? n with
  | none => IO.println s!"NDCHECK {k} REJ nodecl"
  | some ci =>
    let v? : Option Expr := match ci with
      | .thmInfo t => some t.value
      | .defnInfo d => some d.value
      | _ => none
    match v? with
    | none => IO.println s!"NDCHECK {k} REJ novalue"
    | some v =>
      let axs ← collectAxioms n
      let badAx := axs.filter (fun a => !okAxioms.contains a)
      if badAx.size > 0 then
        IO.println s!"NDCHECK {k} REJ axiom:{badAx.toList}"
      else
        let cs := v.getUsedConstants
        let badC := cs.filter (fun c => !allowed.contains c)
        if badC.size > 0 then
          IO.println s!"NDCHECK {k} REJ const:{badC.toList}"
        else
          IO.println s!"NDCHECK {k} OK {size (strip np v)}"
end NdCheck
open NdCheck
'''
PRELUDE = PRELUDE % {'inf': ', '.join(f'``{n}' for n in ALLOW_INF), 'ty': ', '.join(f'``{n}' for n in ALLOW_TY), 'ax': ', '.join(f'``{n}' for n in OK_AXIOMS)}
N_PRELUDE = PRELUDE.count('\n')
RES = re.compile(r'^NDCHECK (\d+) (OK|REJ) ?(.*)$', re.M)
ERR = re.compile(r'^[^\n]*?:(\d+):\d+: error: ([^\n]*)', re.M)


def n_params(src):
    """number of declared binders of `theorem t <binders> : <type> := ...` (so that the value's leading lambdas for them are not
    counted as inference binders): each `(a b : T)` group counts its names, `[inst]` counts 1; stops at the first top-level `:`."""
    s = src[len('theorem t'):] if src.startswith('theorem t') else src
    s = s.lstrip(); i = 0; n = 0
    while i < len(s):
        c = s[i]
        if c in ' \t': i += 1; continue
        if c == ':': break
        if c in '([{':
            close = {'(': ')', '[': ']', '{': '}'}[c]; d = 0; j = i
            while j < len(s):
                if s[j] in '([{': d += 1
                elif s[j] in ')]}':
                    d -= 1
                    if d == 0: break
                j += 1
            grp = s[i + 1:j]
            if c == '[': n += 1
            else:
                names = grp.split(':', 1)[0].split()
                n += len(names) if ':' in grp else 1
            i = j + 1; continue
        # a bare identifier binder (rare): count it
        j = i
        while j < len(s) and s[j] not in ' \t:([{': j += 1
        n += 1; i = j
    return n


def _run(srcs, workdir, tag, depth=0):
    """srcs: list of theorem sources -> (list of result dicts, process seconds)"""
    fn = os.path.join(workdir, f'c_{tag}.lean')
    starts = []
    with open(fn, 'w') as f:
        f.write(PRELUDE)
        line = N_PRELUDE + 1
        for k, s in enumerate(srcs):
            s = s.rstrip('\n')
            starts.append(line)
            body = s.replace('theorem t ', f'theorem t{k} ', 1)
            f.write(body + '\n' + f'nd_check {k} {n_params(s)}\n')
            line += body.count('\n') + 2
    t0 = time.time()
    try:
        p = subprocess.run([LEAN, '-DmaxErrors=100000000', fn], capture_output=True, text=True, timeout=120 + 2 * len(srcs))
        o = p.stdout + p.stderr; rc = p.returncode
    except subprocess.TimeoutExpired:
        o = ''; rc = -9
    cpu = time.time() - t0
    os.remove(fn)
    res = {}
    for m in RES.finditer(o):
        k = int(m.group(1))
        if 0 <= k < len(srcs) and k not in res:
            res[k] = {'ok': m.group(2) == 'OK', 'size': int(m.group(3)) if m.group(2) == 'OK' else None, 'reason': '' if m.group(2) == 'OK' else m.group(3)[:200]}
    errs = collections.defaultdict(list)
    for m in ERR.finditer(o):
        ln = int(m.group(1))
        ks = [j for j, s in enumerate(starts) if s <= ln]
        if ks:
            errs[ks[-1]].append(m.group(2)[:120])
    for k in list(res) + [k for k in errs if k not in res]:
        if errs.get(k):          # ANY error attributed to the theorem's lines rejects it — Lean's parse-error recovery can still
            r = res.get(k)       # elaborate a truncated text (`( fun a => ( fun b => Or.inl` is closed and accepted as a term)
            reason = ('' if r is None or r['ok'] else r['reason'] + ' | ') + 'error: ' + '; '.join(errs[k])
            res[k] = {'ok': False, 'size': None, 'reason': reason[:300]}
    crashed = len(res) != len(srcs) or rc not in (0, 1)
    if crashed:
        if len(srcs) == 1:
            r = res.get(0)
            if r is None or r['ok']:      # a missing verdict, or an OK verdict from a process that did not exit cleanly: reject
                return [{'ok': False, 'size': None, 'reason': f'lean rc={rc}: ' + (o[-200:].replace('\n', ' ') if o else 'no output')}], cpu
            return [r], cpu
        h = len(srcs) // 2
        a, ca = _run(srcs[:h], workdir, tag + 'a', depth + 1); b, cb = _run(srcs[h:], workdir, tag + 'b', depth + 1)
        return a + b, cpu + ca + cb
    return [res[k] for k in range(len(srcs))], cpu


def check(sources, workers=None, chunk=None):
    """sources: list of Lean theorem sources (`theorem t ...`) -> (list of {'ok','size','reason'}, wall s, summed process s)"""
    if not sources:
        return [], 0.0, 0.0
    workers = workers or WORKERS; chunk = chunk or CHUNK
    wd = tempfile.mkdtemp(prefix='leancheck_')
    t0 = time.time()
    chunks = [(sources[i:i + chunk], i) for i in range(0, len(sources), chunk)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: _run(c[0], wd, str(c[1])), chunks))
    shutil.rmtree(wd, ignore_errors=True)
    out = [x for r, _ in res for x in r]
    assert len(out) == len(sources)
    return out, time.time() - t0, sum(c for _, c in res)


# ---------------------------------------------------------------- hand-written cases
SELFTEST = [
    # (name, source, expect_ok, expected size or None)
    ('and_intro_term', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := ⟨h.2, h.1⟩', True, 3),
    ('and_intro_explicit', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := And.intro (And.right h) (And.left h)', True, 3),
    ('fragment_have', 'theorem t (P Q : Prop) (h1 : (P ∧ Q)) : (Q ∧ P) := by have n1 : (P ∧ Q) := h1 ; have n2 : Q := n1.2 ; have n3 : P := n1.1 ; have n4 : (Q ∧ P) := ⟨n2, n3⟩ ; exact n4', True, 3),
    ('em_lemma', 'theorem t (P : Prop) : P ∨ ¬P := Classical.em P', False, None),
    ('simp_proof', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := by simp [h]', False, None),
    ('sorry', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := sorry', False, None),
    ('decide', 'theorem t : True ∨ False := by decide', False, None),
    ('cases_or', 'theorem t (P Q : Prop) (h : P ∨ Q) : Q ∨ P := by cases h with | inl a => exact Or.inr a | inr b => exact Or.inl b', False, None),   # `cases` introduces Eq / Eq.refl: off the allowlist (use Or.elim)
    ('or_elim_dot', 'theorem t (P Q : Prop) (h : P ∨ Q) : Q ∨ P := h.elim (fun a => Or.inr a) (fun b => Or.inl b)', True, 5),
    ('match_or', 'theorem t (P Q : Prop) (h : P ∨ Q) : Q ∨ P := match h with | Or.inl a => Or.inr a | Or.inr b => Or.inl b', False, None),   # match compiles to an auxiliary definition
    ('or_elim_term', 'theorem t (P Q : Prop) (h : P ∨ Q) : Q ∨ P := Or.elim h (fun a => Or.inr a) (fun b => Or.inl b)', True, 5),
    ('by_contradiction', 'theorem t (P : Prop) (h : ¬¬P) : P := Classical.byContradiction (fun hh => h hh)', True, 3),
    ('false_elim', 'theorem t (P : Prop) (h : False) : P := False.elim h', True, 1),
    ('old_bote_on_false', 'theorem t (P : Prop) (h : False) : P := h.elim', True, 1),
    ('old_bote_on_neg', 'theorem t (P Q : Prop) (h : ¬P) (a : P) : Q := h.elim a', True, 1),        # 2026-09-27: Not.elim is a valid proof
    ('not_elim_explicit', 'theorem t (P Q : Prop) (h : ¬P) (a : P) : Q := Not.elim h a', True, 1),
    ('not_intro', 'theorem t (P : Prop) (h : P → False) : ¬P := Not.intro h', True, 1),
    ('and_elim', 'theorem t (P Q R : Prop) (h : P ∧ Q) : R → (P ∧ Q) := And.elim (fun a => fun b => (fun _ => ⟨a, b⟩)) h', True, 5),
    ('iff_elim', 'theorem t (P Q : Prop) (h : P ↔ Q) (a : P) : Q := Iff.elim (fun mp => fun mpr => mp a) h', True, 4),
    # the canonical example of the 2026-09-27 decision note: `~P |- P > Q` as `N1 (~P) : PR ; N2 (P>Q) : BOTE N1`,
    # i.e. nd2lean's `n1.elim`.  nd_verify rejects it (BOTE must cite a line that is literally F); Lean accepts it.
    ('decision_note_bote', 'theorem t (P Q R S : Prop) (h1 : (¬ P)) : (P → Q) := by have n1 : (¬ P) := h1 ; have n2 : (P → Q) := n1.elim ; exact n2', True, 1),
    ('sorry_in_have', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := by have n1 : Q := sorry ; have n2 : P := h.1 ; exact ⟨n1, n2⟩', False, None),
    ('tauto_lemma_not_not', 'theorem t (P : Prop) (h : ¬¬P) : P := not_not.mp h', False, None),
    ('absurd_ok', 'theorem t (P Q : Prop) (h : ¬P) (a : P) : Q := absurd a h', True, 1),
    ('or_resolve', 'theorem t (P Q : Prop) (h : P ∨ Q) (n : ¬P) : Q := Or.resolve_left h n', False, None),
    ('mt_lemma', 'theorem t (P Q : Prop) (h : P → Q) (n : ¬Q) : ¬P := mt h n', False, None),
    ('tauto_like_decidable', 'theorem t (P : Prop) [Decidable P] : P ∨ ¬P := Decidable.em P', False, None),
    ('imp_intro_lambda', 'theorem t (P Q : Prop) (h : P → Q) : P → Q := fun a => h a', True, 2),
    ('neg_intro', 'theorem t (P : Prop) (h : ¬P) : ¬P := fun a => h a', True, 2),
    ('wrong_proof', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := ⟨h.1, h.2⟩', False, None),
    ('unbound_name', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := ⟨n3.2, n3.1⟩', False, None),
    ('iff_intro', 'theorem t (P : Prop) : P ↔ P := Iff.intro (fun a => a) (fun a => a)', True, 3),
    ('propext_term', 'theorem t (P Q : Prop) (h : P ↔ Q) : P = Q := propext h', False, None),
    ('have_unused_extra_premises', 'theorem t (P Q R S : Prop) (h1 : P) (h2 : Q) : Q := by have n1 : Q := h2 ; exact n1', True, 0),
    ('nested_depth3', 'theorem t (P Q R : Prop) : P → (Q → (R → (P ∧ Q))) := fun a => fun b => fun c => ⟨a, b⟩', True, 4),
    ('exfalso_tactic', 'theorem t (P : Prop) (h : False) : P := by exfalso ; exact h', True, 1),
    ('truncated_paren', 'theorem t (P Q R S : Prop) : (Q → (Q → (Q → (Q ∨ (Q → (R → S)))))) := ( fun n19 => ( fun n55 => Or.inl', False, None),   # parse error: Lean's recovery would close the parens
    ('truncated_anon_ctor', 'theorem t (P Q : Prop) (h : P ∧ Q) : Q ∧ P := ⟨h.2, h.1', False, None),
    ('unapplied_or_inl', 'theorem t (P Q R S : Prop) : (Q → (Q → (Q → (Q ∨ (Q → (R → S)))))) := ( fun n19 => ( fun n55 => Or.inl ) )', True, 3),   # well-formed: an unapplied constructor is a function
    ('contradiction_tactic', 'theorem t (P Q : Prop) (h : ¬P) (a : P) : Q := by contradiction', True, None),   # elaborates to allowed constants only (size 2)
]


def selftest(out='artifacts/lj/lean_check_selftest.json'):
    res, wall, cpu = check([s for _, s, _, _ in SELFTEST], workers=2, chunk=100)
    rows = []; bad = 0
    for (name, src, exp_ok, exp_size), r in zip(SELFTEST, res):
        good = (r['ok'] == exp_ok) and (exp_size is None or r['size'] == exp_size)
        bad += not good
        rows.append({'name': name, 'source': src, 'expect_ok': exp_ok, 'expect_size': exp_size, **r, 'pass': good})
        print(('PASS' if good else 'FAIL'), name, r)
    print(f'{len(rows) - bad}/{len(rows)} pass; {wall:.1f}s wall {cpu:.1f}s proc')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'pass': len(rows) - bad, 'n': len(rows), 'rows': rows, 'wall_s': wall, 'proc_s': cpu, 'lean': subprocess.run([LEAN, '--version'], capture_output=True, text=True).stdout.strip()}, open(out, 'w'), indent=1)
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--check', help='ND records jsonl')
    ap.add_argument('--texts', help='literal Lean records jsonl {prompt, lean_text}')
    ap.add_argument('--out'); ap.add_argument('--limit', type=int); ap.add_argument('--field', default='proof')
    ap.add_argument('--workers', type=int); ap.add_argument('--chunk', type=int)
    ap.add_argument('--compare_nd', action='store_true', help='also run nd_verify and print the agreement table; nd_verify judges '
                    'nothing (Dan, 2026-09-27) -- this flag exists only to reproduce pre-2026-09-27 numbers')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    verify_text = None
    if a.compare_nd:
        from nd_verify import verify_text
    recs = [json.loads(l) for l in open(a.check or a.texts) if l.strip()]
    if a.limit: recs = recs[:a.limit]
    t0 = time.time()
    rows, srcs, idx = [], [], []
    if a.check:
        from nd2lean import translate, TranslationError
        for r in recs:
            if 'text' in r and 'prompt' not in r:
                prompt, proof = r['text'].split(' PRF ', 1); prompt += ' PRF'
            else:
                prompt = r['prompt']; proof = r.get(a.field) or r.get('proof') or r.get('gen_proof') or r.get('reference_proof')
            ok, reason, nl = verify_text(prompt + ' ' + proof) if verify_text else (None, None, None)
            row = {'name': r.get('name'), 'prompt': prompt, 'proof': proof, 'nd_ok': None if ok is None else bool(ok), 'nd_reason': reason, 'nd_lines': nl}
            try:
                srcs.append(translate(prompt, proof)); idx.append(len(rows))
            except TranslationError as e:
                row.update({'lean_ok': False, 'size': None, 'lean_reason': f'structural: {e}'})
            except Exception as e:
                row.update({'lean_ok': False, 'size': None, 'lean_reason': f'translator crash: {type(e).__name__}: {e}'})
            rows.append(row)
    else:
        from lean_tok import LeanTokenizer
        tk = LeanTokenizer('lean_seq')
        for r in recs:
            row = {k: r.get(k) for k in ('prompt', 'lean_text', 'nd', 'kind', 'nd_ok')}
            if r.get('nd') and row.get('nd_ok') is None and verify_text:
                row['nd_ok'] = bool(verify_text(r['prompt'] + ' ' + r['nd'])[0])
            srcs.append(tk.statement(r['prompt']) + ' ' + r['lean_text']); idx.append(len(rows)); rows.append(row)
    t1 = time.time()
    res, wall, cpu = check(srcs, a.workers, a.chunk)
    for k, r in zip(idx, res):
        rows[k].update({'lean_ok': r['ok'], 'size': r['size'], 'lean_reason': r['reason']})
    c = collections.Counter((r.get('nd_ok'), r['lean_ok']) for r in rows)
    print(f'{len(rows)} records: translate {t1-t0:.0f}s, lean {wall:.0f}s wall / {cpu:.0f}s proc ({len(srcs)/max(wall,1e-9):.0f} per wall-s, {len(srcs)/max(cpu,1e-9):.0f} per proc-s, {WORKERS if a.workers is None else a.workers} workers): '
          f'nd_ok&lean_ok {c[(True, True)]}, nd_ok&lean_rej {c[(True, False)]}, nd_rej&lean_ok {c[(False, True)]}, both reject {c[(False, False)]}, nd_unknown {c[(None, True)] + c[(None, False)]}')
    sizes = [r['size'] for r in rows if r.get('size') is not None]
    if sizes:
        print('term_size min/median/max', min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes))
    for r in [r for r in rows if r.get('nd_ok') is not None and r['nd_ok'] != r['lean_ok']][:10]:
        print('DISAGREE nd', r['nd_ok'], 'lean', r['lean_ok'], r.get('nd_reason'), '|', r['lean_reason'][:100], '|', (r.get('proof') or r.get('lean_text'))[:150])
    if a.out:
        with open(a.out, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
