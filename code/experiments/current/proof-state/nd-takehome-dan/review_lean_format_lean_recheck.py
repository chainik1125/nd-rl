#!/usr/bin/env python3
"""Reviewer (lean-format, phase 1): re-check EVERY counted proof (cumulative round-8 found files of all 16 arms, both pools, plus the
mechanism-test pass@16 proofs) with (i) nd_verify (unmodified) and (ii) Lean 4 through the unmodified nd2lean.translate, with my own
batching harness (not nd2lean.lean_check, not lean_gate).  Any theorem Lean flags is re-run alone.  Also renders a sample of 150 proofs per
arm in the training surface form (lean_tok) and checks that text in Lean, since the literal sampled text was not saved.
  python3 review_lean_format_lean_recheck.py > artifacts/review_lf/lean_recheck_stdout.txt"""
import json, os, re, sys, glob, subprocess, tempfile, collections, random, time
sys.path.insert(0, '.')
from nd_verify import verify_text
from nd2lean import translate
from review_lean_format_recount import jl

LEAN = os.path.expanduser('~/.elan/bin/lean')
ERR = re.compile(r'^[^\n]*?:(\d+):\d+: error', re.M)
WD = tempfile.mkdtemp(prefix='rv_lf_')


def run_lean(srcs, tag):
    """srcs: list of full theorem sources ('theorem t ...').  -> list of bool"""
    text = ''; starts = []
    for k, s in enumerate(srcs):
        starts.append(text.count('\n') + 1)
        text += s.replace('theorem t ', f'theorem t{k} ', 1).rstrip('\n') + '\n'
    fn = os.path.join(WD, f'{tag}.lean'); open(fn, 'w').write(text)
    p = subprocess.run([LEAN, '-DmaxErrors=1000000', fn], capture_output=True, text=True)
    o = p.stdout + p.stderr
    bad = set()
    for m in ERR.finditer(o):
        ln = int(m.group(1)); bad.add(max(j for j, s in enumerate(starts) if s <= ln))
    if p.returncode != 0 and not bad:
        if len(srcs) == 1: return [False]
        h = len(srcs) // 2
        return run_lean(srcs[:h], tag + 'a') + run_lean(srcs[h:], tag + 'b')
    os.remove(fn)
    return [k not in bad for k in range(len(srcs))]


def check(items, tag, chunk=250):
    """items: list of (prompt, proof).  -> (nd_ok list, lean_ok list, translation errors)"""
    nd = [verify_text(p + ' ' + q)[0] for p, q in items]
    srcs = []; terr = 0
    for p, q in items:
        try: srcs.append(translate(p, q))
        except Exception as e: srcs.append(None); terr += 1
    idx = [i for i, s in enumerate(srcs) if s is not None]
    ok = [False] * len(items)
    for b in range(0, len(idx), chunk):
        sub = idx[b:b + chunk]
        r = run_lean([srcs[i] for i in sub], f'{tag}_{b}')
        for i, v in zip(sub, r):
            ok[i] = v
            if not v:   # confirm alone
                ok[i] = run_lean([srcs[i]], f'{tag}_{b}_solo{i}')[0]
    return nd, ok, terr


def main():
    from lean_tok import LeanTokenizer, proof_tokens
    ltok = LeanTokenizer('lean_seq')
    rng = random.Random(0)
    tot = collections.Counter()
    arms = [f'{k}_d3_{s}_s{i}' for s in ('rand', 'seq') for k in ('ei', 'frozen') for i in (0, 1)] + [f'la_{k}_{s}_s{i}' for s in ('rand', 'seq') for k in ('T1', 'frozen') for i in (0, 1)]
    for arm in arms:
        for pool in ('found', 'found_transfer'):
            F = jl(f'artifacts/lf/{arm}/{pool}_8.jsonl')
            items = [(x['prompt'], x['proof']) for x in F]
            t0 = time.time()
            nd, lo, terr = check(items, f'{arm}_{pool}')
            c = collections.Counter(zip(nd, lo)); tot.update(c)
            # surface-form sample
            samp = rng.sample(items, min(150, len(items)))
            texts = []
            for p, q in samp:
                toks = [t if isinstance(t, str) else f'n{t[1]}' for t in proof_tokens(q)]
                texts.append('set_option linter.unusedVariables false in\n' + ltok.statement(p) + ' ' + ltok.text(toks))
            so = run_lean(texts, f'{arm}_{pool}_surface')
            print(f'{arm:20s} {pool:15s} n {len(items):>5} nd_verify ok {sum(nd):>5} lean(nd2lean) ok {sum(lo):>5} translate errors {terr} | 2x2 both {c[(True,True)]} nd_only {c[(True,False)]} lean_only {c[(False,True)]} neither {c[(False,False)]} | surface-form sample {sum(so)}/{len(so)} | {time.time()-t0:.0f}s', flush=True)
    for tag in ('rand', 'seq', 'seqfixed'):
        E = jl(f'artifacts/lf/stage1_full_{tag}_transfer2_k16.jsonl')
        items = [(x['prompt'], p) for x in E for p in x['proofs']]
        nd, lo, terr = check(items, f'mech_{tag}')
        c = collections.Counter(zip(nd, lo)); tot.update(c)
        print(f'mech stage1_full_{tag:9s} transfer pass@16  n {len(items):>5} nd_verify ok {sum(nd):>5} lean ok {sum(lo):>5} translate errors {terr} | both {c[(True,True)]} nd_only {c[(True,False)]} lean_only {c[(False,True)]} neither {c[(False,False)]}', flush=True)
        H = jl(f'artifacts/lf/stage1_full_{tag}_heldout_greedy.jsonl')
        items = [(x['prompt'], p) for x in H for p in x['proofs']]
        nd, lo, terr = check(items, f'held_{tag}')
        c = collections.Counter(zip(nd, lo)); tot.update(c)
        print(f'mech stage1_full_{tag:9s} heldout greedy     n {len(items):>5} nd_verify ok {sum(nd):>5} lean ok {sum(lo):>5} translate errors {terr} | both {c[(True,True)]} nd_only {c[(True,False)]} lean_only {c[(False,True)]} neither {c[(False,False)]}', flush=True)
    print('TOTAL 2x2 (nd_ok, lean_ok):', dict(tot))
    # negative control for my harness: swap theorems between consecutive proofs (should be rejected by both)
    F = jl('artifacts/lf/la_T1_seq_s0/found_transfer_8.jsonl')[:300]
    neg = [(F[i]['prompt'], F[(i + 1) % len(F)]['proof']) for i in range(len(F)) if F[i]['prompt'] != F[(i + 1) % len(F)]['prompt']]
    nd, lo, terr = check(neg, 'neg')
    print(f'negative control (theorem-swapped) n {len(neg)}: nd_verify accepts {sum(nd)}, lean accepts {sum(lo)} (translate errors {terr})')


if __name__ == '__main__':
    main()
