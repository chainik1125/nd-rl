#!/usr/bin/env python3
"""Acceptance test 1 and the pitfall unit tests of run lean-judge (`python3 tests/test_lean_only_judge.py`).

1. NO JUDGING PATH CALLS nd_verify.  The six loop files, the gate and the judge must not import or call `nd_verify`.
   Lean alone decides (Dan, 2026-09-27); `nd_verify` stays in the repository, unmodified, only so that pre-2026-09-27
   numbers can be reproduced.  Parser-only imports (`from nd_verify.verify import parse_proof_tokens`) are NOT judging and
   are allowed anywhere -- this test rejects them in the eight files below anyway, so that the files that decide are
   provably free of the old checker.
2. Pitfall 1: no verdict marker on an ACCEPTED sample leaving the gate (it would become expert-iteration training text).
3. Pitfall 2: a hindsight relabel against a rewritten theorem the gate never saw is judged, not silently accepted.
4. Pitfall 3: `coverage.py` judges the NORMALISED string; nd2lean's rendering of norm(nd) and of nd differ only by
   hypothesis renaming, and Lean's verdict is invariant to it.
5. Pitfall 4: no judging call is made one string at a time inside a loop -- `judge_many` is used by every loop file.
6. `lean_judge.verify_text` is a drop-in: same 3-tuple shape, same `n_lines` expression as `nd_verify.verify_text`.
"""
import os, re, sys, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

JUDGES = ['expert_iter.py', 'ladder_ei.py', 'coverage.py', 'eval_set.py', 'eval_targets.py', 'grpo.py',
          'lean_gate.py', 'lean_judge.py']
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + ((' -- ' + detail) if detail and not ok else ''))
    if not ok:
        fails.append(name)


# ---- 1. no nd_verify in any judging path (parsed, so that prose in a docstring is not a hit)
import ast


def nd_verify_uses(src):
    tree = ast.parse(src)
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            hits += [f'import {al.name}' for al in n.names if al.name.split('.')[0] == 'nd_verify']
        elif isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] == 'nd_verify':
            hits.append(f'from {n.module} import ' + ', '.join(al.name for al in n.names))
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'nd_verify':
            hits.append(f'nd_verify.{n.attr}')
    return hits


for fn in JUDGES:
    hits = nd_verify_uses(open(os.path.join(HERE, fn)).read())
    check(f'no nd_verify import or call in {fn}', not hits, '; '.join(hits))

# ---- 1b. each of the six loop files imports the Lean judge
for fn in JUDGES[:6]:
    src = open(os.path.join(HERE, fn)).read()
    check(f'{fn} imports lean_judge', 'from lean_judge import' in src or 'import lean_judge' in src)

# ---- 5. no per-string judging call inside a for/while body in the loop files
for fn in JUDGES[:6]:
    src = open(os.path.join(HERE, fn)).read()
    bad = [i + 1 for i, l in enumerate(src.split('\n'))
           if re.search(r'\bverify_text\(', l) and l.startswith(' ' * 8)]
    check(f'{fn} makes no deeply-indented one-string verify_text call', not bad, f'lines {bad}')

# ---- 6 + the judge's own behaviour
import lean_judge
from nd_verify import verify_text as ndv        # allowed HERE: this is the test, not a judging path

P = 'THM ( ~ P ) SEQ ( P > Q ) PRF'
ND = 'N1 ( ~ P ) : PR ; N2 ( P > Q ) : BOTE N1 ; QED'
ok, reason, nl = lean_judge.verify_text(P + ' ' + ND)
check('canonical decision-note proof accepted by the lean_seq judge', ok and nl == 2, f'{ok} {reason} {nl}')
check('nd_verify rejects it (this is the class Lean alone now counts)', not ndv(P + ' ' + ND)[0])

VALID = ('THM ( P & Q ) SEQ ( Q & P ) PRF',
         'N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; N4 ( Q & P ) : ANDI N2 N3 ; QED')
a = lean_judge.verify_text(VALID[0] + ' ' + VALID[1]); b = ndv(VALID[0] + ' ' + VALID[1])
check('shape and n_lines match nd_verify on a proof both accept', a[0] and b[0] and a[2] == b[2], f'{a} vs {b}')

INVALID = ('THM ( P & Q ) SEQ ( Q & P ) PRF', 'N1 ( P & Q ) : PR ; N2 ( Q & P ) : ANDI N1 N1 ; QED')
check('an invalid proof is rejected', not lean_judge.verify_text(INVALID[0] + ' ' + INVALID[1])[0])

# ---- 2. pitfall 1: the gate returns CLEAN ND strings for accepted samples
from lean_tok import LeanTokenizer
import lean_gate
tok = LeanTokenizer('lean_seq')
os.environ['LEAN_GATE_LOG'] = os.path.join(HERE, 'artifacts/lj/test_gate.jsonl')
good_text = 'have n1 : ( ¬ P ) := h1 ; have n2 : ( P → Q ) := n1.elim ; exact n2'
bad_text = 'have n1 : ( ¬ P ) := h1 ; have n2 : ( P → Q ) := n1 ; exact n2'      # type error: Lean rejects
bad_nd = 'N1 ( ~ P ) : PR ; N2 ( P > Q ) : R N1 ; QED'
out = lean_gate.gate(tok, [P, P, P], [ND, bad_nd, 'LEANPARSE no-eos'], [good_text, bad_text, None])
check('accepted sample leaves the gate as a CLEAN ND string (no LEAN* prefix)', out[0] == ND, out[0][:60])
check('Lean-rejected sample is marked', out[1].startswith('LEANREJ '), out[1][:60])
check('grammar-rejected sample keeps its LEANPARSE marker', out[2].startswith('LEANPARSE'), out[2][:40])
check('no LEAN marker anywhere in the accepted sample', 'LEAN' not in out[0])
v = lean_judge.judge_many([(P, out[0]), (P, out[1]), (P, out[2])])
check('the gate registered the accepted verdict (registry hit, no second Lean run)', v[0][0])
check('the marked samples are rejected by the judge', not v[1][0] and not v[2][0])

# ---- 3. pitfall 2: hindsight relabel against a rewritten theorem.
# expert_iter.py imports torch; these two functions are pure, so they are exec'd from its source so that the test runs
# on a CPU-only box as well.  (On the pod, test 5 runs the real thing end to end.)
_src = open(os.path.join(HERE, 'expert_iter.py')).read()
expert_iter = type(sys)('expert_iter_relabel_only')
expert_iter.__dict__['judge_many'] = lean_judge.judge_many
exec(_src[_src.index('def relabel_candidate'):_src.index('def main(')], expert_iter.__dict__)
pr = 'THM ( P & Q ) SEQ ( Q & P ) PRF'
proof = 'N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; QED'          # proves Q, not (Q & P)
cand = expert_iter.relabel_candidate(pr, proof)
check('relabel_candidate rewrites the theorem without judging', cand is not None and cand[0] == 'THM ( P & Q ) SEQ Q PRF', str(cand))
rb = expert_iter.relabel_batch([(pr, proof), (pr, 'N1 ( P & Q ) : PR ; N2 P : ANDE2 N1 ; QED')])
check('a true hindsight relabel is accepted', rb[0] is not None and rb[0][2] == 2, str(rb[0]))
check('a false hindsight relabel is rejected (the gate never saw this theorem)', rb[1] is None, str(rb[1]))

# ---- 4. pitfall 3: normalising before judging does not change the verdict
from normalize import norm
import nd2lean
from lean_gate import check_sources
SHIFTED = ('THM ( P & Q ) SEQ ( Q & P ) PRF',
           'N7 ( P & Q ) : PR ; N8 Q : ANDE2 N7 ; N9 P : ANDE1 N7 ; N10 ( Q & P ) : ANDI N8 N9 ; QED')
r1 = lean_judge.verify_text(SHIFTED[0] + ' ' + SHIFTED[1])
r2 = lean_judge.verify_text(SHIFTED[0] + ' ' + norm(SHIFTED[1]))
check('same verdict and n_lines before and after norm()', r1 == r2 and r1[0], f'{r1} vs {r2}')

print()
print('LEAN-ONLY JUDGE TESTS', 'PASS' if not fails else f'FAIL: {fails}')
sys.exit(0 if not fails else 1)
