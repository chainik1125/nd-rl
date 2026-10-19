"""Check ND proofs in Lean (Robbie's rule, 2026-10-14: evidence proofs are checked in Lean, never nd_verify).

Uses nd2lean.py from ~/repos/nd-takehome at commit 51604b3 (extracted with `git show` into build/), unmodified except
that the generated theorem header binds every atom the theorem uses (the tool hardcodes `(P Q R S : Prop)`).
nd2lean.translate also rejects a box cited before it is closed (`forward cite`), which nd_verify accepts.

    from lean_check import lean_ok
    oks = lean_ok(['THM ... PRF N1 ... QED', ...])     # list of (ok, reason)
    python lean_check.py FILE.jsonl [...] [--limit N]   # Lean-checks 'text' and prompt+'min_proof' of each record
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigbound  # noqa: E402,F401  (puts vendor/fork on sys.path and widens nd_verify atoms)

BUILD = os.path.join(HERE, 'build', 'lean51604b3')
TAKEHOME = os.path.expanduser('~/repos/nd-takehome')
COMMIT = '51604b3'


def _load():
    os.makedirs(BUILD, exist_ok=True)
    for f in ('nd2lean.py',):
        p = os.path.join(BUILD, f)
        if not os.path.exists(p):
            src = subprocess.check_output(['git', '-C', TAKEHOME, 'show', f'{COMMIT}:{f}'])
            open(p, 'wb').write(src)
    sys.path.insert(0, BUILD)
    import nd2lean
    return nd2lean


def lean_ok(texts, per_file=40):
    L = _load()
    srcs, idx, out = [], [], [None] * len(texts)
    for k, t in enumerate(texts):
        prompt, proof = t.split(' PRF ', 1)
        prompt += ' PRF'
        try:
            src = L.translate(prompt, proof)
        except L.TranslationError as e:
            out[k] = (False, f'structural: {e}')
            continue
        atoms = sorted(set(re.findall(r'\b[A-Z]\b', prompt + ' ' + proof)) - {'N', 'F'})
        src = src.replace('(P Q R S : Prop)', '(' + ' '.join(sorted(set(atoms) | set('PQRS'))) + ' : Prop)', 1)
        srcs.append(src)
        idx.append(k)
    for k, r in zip(idx, L.lean_check(srcs, per_file)):
        out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    for p in a.paths:
        texts = []
        for line in open(p):
            r = json.loads(line)
            if 'text' in r:
                texts.append(r['text'])
            if r.get('min_proof'):
                texts.append(r['prompt'] + ' ' + r['min_proof'])
            if a.limit and len(texts) >= a.limit:
                break
        res = lean_ok(texts)
        bad = [(t, m) for t, (ok, m) in zip(texts, res) if not ok]
        print(f'{p}: {len(texts)} proofs, Lean accepts {len(texts) - len(bad)}, rejects {len(bad)}')
        for t, m in bad[:5]:
            print('  REJECT', m[:200], t[:200])


if __name__ == '__main__':
    main()
