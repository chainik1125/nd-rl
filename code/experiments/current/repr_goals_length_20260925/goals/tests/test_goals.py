#!/usr/bin/env python3
"""Tests of the goal annotation (lib/goals.py) and the tokenizer (lib/gtokenizer.py). CPU only.

    python tests/test_goals.py [--n 20000]      # --n: proofs sampled per source for the bulk checks (0 = all)

1. Hand-written cases: the canonical annotation of small proofs of each structure (IMPI chain, NEGI, ORE, ANDI split,
   lemma box, nested split).
2. Bulk, on verified proofs (cap-6 train and held-out, RL-found proofs): strip(annotate(p)) == p exactly, the
   verdict of the stripped text equals the plain verdict, annotate is deterministic and invariant to the start
   index, the tokenizer round-trips the annotated text, the index shift leaves annotation tokens alone, and
   consistency() of the canonical annotation is exact.
3. Bulk, on corrupted proofs (invalid): random well-formed GOAL groups inserted at random line boundaries strip back
   to the corrupted text exactly, so the verdict is unchanged (verify(strip(x)) == verify(plain)) for valid and
   invalid proofs alike.
4. Malformed annotations (GOAL inside a line, GOAL with an incomplete formula) are left in place and the proof is
   rejected.
"""
import argparse, glob, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import goals as G  # noqa: E402
from gtokenizer import GoalTokenizer  # noqa: E402

AB3 = '/scratch/cowifornia/nd-rl/ab3'
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        if len(FAILS) <= 20:
            print('FAIL', msg)


def ann(prompt, body, pol=G.DEFAULT_POLICY):
    return G.annotate(prompt, body, pol)


def hand_cases():
    # 1. nested IMPI chain, no premises: each AS line gets its box goal; nothing while unwinding
    p = 'THM SEQ ( P > ( Q > P ) ) PRF'
    b = 'N1 | P : AS ; N2 | | Q : AS ; N3 | | P : R N1 ; N4 | ( Q > P ) : IMPI N2 N3 ; N5 ( P > ( Q > P ) ) : IMPI N1 N4 ; QED'
    check(ann(p, b) == 'N1 | P : AS ; GOAL ( Q > P ) N2 | | Q : AS ; GOAL P N3 | | P : R N1 ; '
          'N4 | ( Q > P ) : IMPI N2 N3 ; N5 ( P > ( Q > P ) ) : IMPI N1 N4 ; QED', 'impi chain: ' + ann(p, b))
    # 'start' writes the conclusion after the (empty) premise block too
    check(ann(p, b, 'start').startswith('GOAL ( P > ( Q > P ) ) N1 | P : AS ; GOAL ( Q > P )'), 'event start: ' + ann(p, b, 'event'))
    # 2. NEGI (modus tollens): the box goal is F
    p = 'THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF'
    b = 'N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | P : AS ; N4 | Q : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; N6 ( ~ P ) : NEGI N3 N5 ; QED'
    check(ann(p, b) == 'N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | P : AS ; GOAL F N4 | Q : IMPE N1 N3 ; '
          'N5 | F : NEGE N4 N2 ; N6 ( ~ P ) : NEGI N3 N5 ; QED', 'negi: ' + ann(p, b))
    # 3. ORE: both case boxes get the case target; box ends are silent
    p = 'THM ( P v Q ) SEQ ( Q v P ) PRF'
    b = ('N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; N5 | ( Q v P ) : ORI1 N4 ; '
         'N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED')
    check(ann(p, b) == ('N1 ( P v Q ) : PR ; N2 | P : AS ; GOAL ( Q v P ) N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; '
                        'GOAL ( Q v P ) N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED'), 'ore: ' + ann(p, b))
    # 4. ANDI split of the conclusion: first conjunct at the start, the second when the first is written, nothing
    #    once both are written (the next line is the ANDI)
    p = 'THM ( P & Q ) , ( R & S ) SEQ ( ( Q & P ) & ( S & R ) ) PRF'
    b = ('N1 ( P & Q ) : PR ; N2 ( R & S ) : PR ; N3 Q : ANDE2 N1 ; N4 P : ANDE1 N1 ; N5 ( Q & P ) : ANDI N3 N4 ; '
         'N6 S : ANDE2 N2 ; N7 R : ANDE1 N2 ; N8 ( S & R ) : ANDI N6 N7 ; N9 ( ( Q & P ) & ( S & R ) ) : ANDI N5 N8 ; QED')
    check(ann(p, b) == ('N1 ( P & Q ) : PR ; N2 ( R & S ) : PR ; GOAL Q N3 Q : ANDE2 N1 ; GOAL P N4 P : ANDE1 N1 ; '
                        'N5 ( Q & P ) : ANDI N3 N4 ; GOAL S N6 S : ANDE2 N2 ; GOAL R N7 R : ANDE1 N2 ; '
                        'N8 ( S & R ) : ANDI N6 N7 ; N9 ( ( Q & P ) & ( S & R ) ) : ANDI N5 N8 ; QED'), 'split: ' + ann(p, b))
    # without splits ('boxes') a box-free proof has no annotation at all
    check(ann(p, b, 'boxes') == b, 'boxes no split: ' + ann(p, b, 'boxes'))
    # 5. a conjunct that is already written (a premise) is not owed
    p = 'THM P , ( P > Q ) SEQ ( P & Q ) PRF'
    b = 'N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; N4 ( P & Q ) : ANDI N1 N3 ; QED'
    check(ann(p, b) == 'N1 P : PR ; N2 ( P > Q ) : PR ; GOAL Q N3 Q : IMPE N2 N1 ; N4 ( P & Q ) : ANDI N1 N3 ; QED', 'avail: ' + ann(p, b))
    # 6. lemma box: after the lemma is discharged the proof returns to the conclusion, which is written again
    p = 'THM ( ( P > P ) > Q ) SEQ Q PRF'
    b = 'N1 ( ( P > P ) > Q ) : PR ; N2 | P : AS ; N3 | P : R N2 ; N4 ( P > P ) : IMPI N2 N3 ; N5 Q : IMPE N1 N4 ; QED'
    check(ann(p, b) == 'N1 ( ( P > P ) > Q ) : PR ; N2 | P : AS ; GOAL P N3 | P : R N2 ; N4 ( P > P ) : IMPI N2 N3 ; GOAL Q N5 Q : IMPE N1 N4 ; QED',
          'lemma: ' + ann(p, b))
    # 7. a split box goal: the box's first owed conjunct is written after its AS line
    p = 'THM P SEQ ( Q > ( P & Q ) ) PRF'
    b = 'N1 P : PR ; N2 | Q : AS ; N3 | ( P & Q ) : ANDI N1 N2 ; N4 ( Q > ( P & Q ) ) : IMPI N2 N3 ; QED'
    check(ann(p, b) == 'N1 P : PR ; N2 | Q : AS ; GOAL ( P & Q ) N3 | ( P & Q ) : ANDI N1 N2 ; N4 ( Q > ( P & Q ) ) : IMPI N2 N3 ; QED',
          'box split (both conjuncts available): ' + ann(p, b))
    # strip / read_annotations basics
    a = 'GOAL ( P & Q ) N1 P : PR ; GOAL Q GOAL ( ~ R ) N2 Q : R N1 ; GOAL P QED'
    plain, groups, bad = G.read_annotations(a)
    check(plain == 'N1 P : PR ; N2 Q : R N1 ; QED' and groups == {0: ['( P & Q )'], 1: ['Q', '( ~ R )'], 2: ['P']} and bad == 0,
          f'read_annotations: {plain!r} {groups} {bad}')
    for bad_text in ('N1 P GOAL Q : PR ; QED', 'N1 P : PR ; GOAL ( P & N2 Q : R N1 ; QED', 'N1 P : PR ; GOAL QED',
                     'N1 P : PR ; GOAL : N2 P : R N1 ; QED'):
        plain, groups, nb = G.read_annotations(bad_text)
        check(nb >= 1 and 'GOAL' in plain.split(), f'malformed kept: {bad_text!r} -> {plain!r}')
        check(not G.verify_text('THM P SEQ P PRF ' + plain)[0], f'malformed rejected: {bad_text!r}')


def sources(n, rng):
    out = {}
    for name, fns in (('train_cap6', [f'{AB3}/frozen/data/train_cap6.jsonl']),
                      ('heldout_cap6', [f'{AB3}/frozen/data/heldout_cap6.jsonl']),
                      ('found_rl', sorted(glob.glob(f'{AB3}/lengthexp/runs/*/found_targets.jsonl'))
                       + sorted(glob.glob(f'{AB3}/base039/runs/*/found_targets.jsonl')))):
        recs = []
        for fn in fns:
            for l in open(fn):
                if l.strip():
                    r = json.loads(l)
                    prompt = r.get('prompt') or ' '.join(('THM ' + r['thm'].replace('|-', 'SEQ') + ' PRF').split())
                    recs.append((prompt, r['proof']))
        if n and len(recs) > n:
            recs = rng.sample(recs, n)
        out[name] = recs
    return out


def shift_indices(body, s):
    return ' '.join(f'N{int(t[1:]) + s}' if re.fullmatch(r'N\d+', t) else t for t in body.split())


def corrupt(prompt, body, rng):
    """A random corruption of a proof body (may still verify)."""
    lines, _ = G.body_lines(body)
    lines = [list(x) for x in lines]
    kind = rng.randrange(6)
    i = rng.randrange(len(lines))
    ln = lines[i]
    if kind == 0 and len(lines) > 1:            # drop a line
        del lines[i]
    elif kind == 1:                             # change the rule
        c = ln.index(':')
        ln[c + 1] = rng.choice(['ANDI', 'ANDE1', 'IMPE', 'IMPI', 'ORI1', 'NEGE', 'NEGI', 'DN', 'R', 'AS'])
    elif kind == 2:                             # change a citation
        refs = [j for j, t in enumerate(ln) if re.fullmatch(r'N\d+', t) and j > 0]
        if refs:
            j = rng.choice(refs)
            ln[j] = f'N{max(1, int(ln[j][1:]) + rng.choice([-2, -1, 1, 2]))}'
    elif kind == 3:                             # change an atom in the formula
        atoms = [j for j, t in enumerate(ln[:ln.index(':')]) if t in 'PQRS']
        if atoms:
            ln[rng.choice(atoms)] = rng.choice('PQRS')
    elif kind == 4:                             # change the depth
        if '|' in ln:
            ln.remove('|')
        else:
            ln.insert(1, '|')
    else:                                       # truncate (no QED) or early QED
        lines = lines[:i]
    out = ' '.join(t for x in lines for t in x)
    if kind != 5 or rng.random() < 0.5:
        out += ' QED'
    return out.strip()


def random_formula(rng, d=0):
    r = rng.random()
    if d > 2 or r < 0.35:
        return rng.choice(['P', 'Q', 'R', 'S', 'F'])
    if r < 0.5:
        return f'( ~ {random_formula(rng, d + 1)} )'
    return f'( {random_formula(rng, d + 1)} {rng.choice("&v>")} {random_formula(rng, d + 1)} )'


def insert_random_goals(body, rng, p=0.4):
    """Random well-formed GOAL groups at random line boundaries (start of the body and after each ';')."""
    toks = body.split()
    out = []
    at = True
    for t in toks:
        if at and rng.random() < p:
            for _ in range(rng.choice([1, 1, 1, 2])):
                out += ['GOAL'] + random_formula(rng).split()
        out.append(t)
        at = t == ';'
    if at and rng.random() < p:
        out += ['GOAL'] + random_formula(rng).split()
    return ' '.join(out)


def bulk(n, seed=0):
    rng = random.Random(seed)
    tok = GoalTokenizer()
    srcs = sources(n, rng)
    for name, recs in srcs.items():
        c = {'proofs': 0, 'corrupted': 0, 'corrupted_valid': 0}
        for prompt, body in recs:
            body = ' '.join(body.split())
            ok = G.verify_text(prompt + ' ' + body)[0]
            if not ok:
                continue
            c['proofs'] += 1
            for pol in G.POLICIES:
                a = G.annotate(prompt, body, pol)
                check(G.strip(a) == body, f'{name} strip {pol}: {body} -> {a}')
                check(G.verify_text(prompt + ' ' + G.strip(a))[0], f'{name} verdict {pol}: {a}')
                check(G.annotate(prompt, body, pol) == a, f'{name} determinism {pol}')
                sh = shift_indices(body, 7)
                check(G.strip(G.annotate(prompt, sh, pol)) == sh and
                      G.read_annotations(G.annotate(prompt, sh, pol))[1] == G.read_annotations(a)[1], f'{name} shift {pol}')
            a = G.annotate(prompt, body)
            ids = tok.encode_proof(a)
            check(tok.decode(ids) == a, f'{name} tokenizer round trip: {a}')
            sh_ids = tok.shift_abs(ids, rng)
            dec = tok.decode(sh_ids)
            check(G.read_annotations(dec)[1] == G.read_annotations(a)[1] and G.verify_text(prompt + ' ' + G.strip(dec))[0],
                  f'{name} abs shift keeps annotations: {dec}')
            cs = G.consistency(prompt, a)
            check(cs is not None and cs['exact'] and cs['matched'] == cs['n_canon'] and cs['extra'] == 0,
                  f'{name} consistency of canonical: {cs}')
            # random annotations on the valid proof and on a corrupted copy
            for text in (body, corrupt(prompt, body, rng)):
                v0 = G.verify_text(prompt + ' ' + text)[0]
                if text != body:
                    c['corrupted'] += 1
                    c['corrupted_valid'] += v0
                x = insert_random_goals(text, rng)
                check(G.strip(x) == ' '.join(text.split()), f'{name} random strip: {text} -> {x} -> {G.strip(x)}')
                check(G.verify_text(prompt + ' ' + G.strip(x))[0] == v0, f'{name} random verdict: {x}')
        print(f'{name}: {c}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    a = ap.parse_args()
    hand_cases()
    print(f'hand cases: {len(FAILS)} failures', flush=True)
    bulk(a.n)
    print('ALL OK' if not FAILS else f'{len(FAILS)} FAILURES')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
