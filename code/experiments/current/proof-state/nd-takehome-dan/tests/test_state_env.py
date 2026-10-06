#!/usr/bin/env python3
"""Unit tests for the state environment of run `state-env` (`python3 tests/test_state_env.py`).  CPU only; needs Lean
at ~/.elan/bin/lean for test 6.

1. Every action shape round-trips: the rendered chunks of a decomposition reassemble the `lean_seq` token list byte for
   byte, for a hand-written proof of each shape (PR, R, ∧I/∧E, →E, ¬E, ∨I, .elim, DN, →I box, ¬I box, Or.elim).
2. The state is what it should be at each step: hypotheses in scope in introduction order, then the goal.  Names bound
   inside a closed box are gone; the finished `have` is in the parent.
3. The environment rejects what `lean_tok.inverse`'s grammar rejects: an out-of-scope name, `exact` on a hypothesis that
   is not the frame's last statement, `exact` whose declared formula is not the goal, a premise line out of position,
   a box binder whose formula is not the one the `have` declares, a malformed term.
4. A finished attempt's text always parses in the strict grammar (`lean_tok.inverse`) and denotes the ND proof it came
   from.
5. `lean_seq`'s vocabulary is unchanged by the new modes (107 tokens), and `lean_state` adds exactly four.
6. Lean agrees with the renderer on the state (a 30-case `state_gate2` run).
"""
import json, os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lean_tok import LeanTokenizer, ParseFail, inverse
from state_env import Env, decompose, seq_tokens, split_actions

FAIL = []


def ck(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


# hand-written ND proofs, one per rule shape (spec.md format)
CASES = [
    ('THM ( P & Q ) SEQ P PRF', 'N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; QED'),
    ('THM ( P & Q ) SEQ Q PRF', 'N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; QED'),
    ('THM P , Q SEQ ( P & Q ) PRF', 'N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED'),
    ('THM P SEQ ( P v Q ) PRF', 'N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; QED'),
    ('THM Q SEQ ( P v Q ) PRF', 'N1 Q : PR ; N2 ( P v Q ) : ORI2 N1 ; QED'),
    ('THM ( P > Q ) , P SEQ Q PRF', 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED'),
    ('THM ( ~ P ) , P SEQ F PRF', 'N1 ( ~ P ) : PR ; N2 P : PR ; N3 F : NEGE N2 N1 ; QED'),
    ('THM F SEQ P PRF', 'N1 F : PR ; N2 P : BOTE N1 ; QED'),
    ('THM ( ~ ( ~ P ) ) SEQ P PRF', 'N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; QED'),
    ('THM Q SEQ ( P > Q ) PRF', 'N1 Q : PR ; N2 | P : AS ; N3 | Q : R N1 ; N4 ( P > Q ) : IMPI N2 N3 ; QED'),
    ('THM ( ~ P ) SEQ ( ~ P ) PRF', 'N1 ( ~ P ) : PR ; N2 | P : AS ; N3 | F : NEGE N2 N1 ; N4 ( ~ P ) : NEGI N2 N3 ; QED'),
    ('THM ( P v Q ) SEQ ( Q v P ) PRF',
     'N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; N5 | ( Q v P ) : ORI1 N4 ; '
     'N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED'),
]


def t1_t4():
    tk = LeanTokenizer('lean_state')
    print('1/4. round trip, state, and the strict grammar on one proof per rule shape')
    for prompt, nd in CASES:
        toks = seq_tokens(nd)
        try:
            steps, toks2, env = decompose(prompt, nd)
        except Exception as e:
            ck(False, f'{nd[:40]}... decompose: {e}')
            continue
        ck(env.text == toks, f'byte-for-byte: {tk.text(toks)[:58]}')
        ck(inverse(env.text) == nd, f'inverse(text) == the ND proof: {nd[:40]}')
        ck(all(s[0][0] == '<st>' and s[0][-1] == '<act>' and '⊢' in s[0] for s in steps),
           'every state is <st> ... ⊢ goal <act>')


def t2():
    print('2. scope: a closed box\'s names are gone, the finished `have` is in the parent')
    prompt, nd = CASES[-1]                       # Or.elim
    steps, toks, env = decompose(prompt, nd)
    tk = LeanTokenizer('lean_state')
    names = [[t for i, t in enumerate(s[0]) if i + 1 < len(s[0]) and s[0][i + 1] == ':'] for s in steps]
    ck(names[0] == ['h1'], f'step 1 scope {names[0]}')
    ck(names[1] == ['h1', 'n1'], f'step 2 scope {names[1]}')
    ck(names[2] == ['h1', 'n1', 'n3'], f"branch 1 sees its binder n3 and not the have it is proving: {names[2]}")
    ck(names[4] == ['h1', 'n1', 'n5'], f'branch 2 sees only its own binder n5 (branch 1 is gone): {names[4]}')
    ck(names[6] == ['h1', 'n1', 'n2'], f'after both branches close, only the finished have n2: {names[6]}')
    ck([s[1] for s in steps][4][1] == 'n6' and steps[4][1][0] == 'have',
       'the environment named branch 2\'s binder n5, so the next have is n6, as in the original text')
    goals = [s[0][s[0].index('⊢') + 1:-1] for s in steps]
    ck(all(g == goals[0] for g in goals), 'every goal in this proof is the conclusion')


def t3():
    print('3. the environment rejects exactly what the grammar rejects')
    P = 'THM P , Q SEQ ( P & Q ) PRF'
    def fresh():
        return Env(P)
    bad = [
        (['have', 'n1', ':', 'P', ':=', 'h2', ';'], 'PR position', 'premise out of order'),
        (['have', 'n1', ':', 'P', ':=', 'n9', ';'], 'unbound', 'out-of-scope name'),
        (['have', 'n1', ':', 'P', ':=', 'h1'], 'PR form', 'missing semicolon'),
        (['exact', 'n1'], 'unbound', 'exact before anything is proved'),
        (['have', 'n1', ':', 'P', ':=', '⟨', 'n1', '⟩', ';'], 'term', 'malformed pair'),
        (['fun', 'n1'], 'action head', 'not an action'),
    ]
    for act, want, why in bad:
        ok, reason = fresh().apply(act)
        ck((not ok) and want in reason, f'{why}: rejected as {reason!r}')
    e = fresh()
    ck(e.apply(['have', 'n1', ':', 'P', ':=', 'h1', ';'])[0], 'premise 1 accepted')
    ck(e.apply(['have', 'n2', ':', 'Q', ':=', 'h2', ';'])[0], 'premise 2 accepted')
    ok, reason = e.apply(['exact', 'n1'])
    ck((not ok) and 'not last' in reason, f'exact on a non-last hypothesis: {reason!r}')
    ok, reason = e.apply(['exact', 'n2'])
    ck((not ok) and 'goal mismatch' in reason, f'exact whose formula is not the goal: {reason!r}')
    ck(e.apply(['have', 'n3', ':', '(', 'P', '∧', 'Q', ')', ':=', '⟨', 'n1', ',', 'n2', '⟩', ';'])[0], '∧-intro accepted')
    ck(e.apply(['exact', 'n3'])[0] and e.done, 'exact on the goal finishes the attempt')
    e2 = Env('THM Q SEQ ( P > Q ) PRF')
    e2.apply(['have', 'n1', ':', 'Q', ':=', 'h1', ';'])
    ok, reason = e2.apply(['have', 'n2', ':', '(', 'P', '→', 'Q', ')', ':=', '(', 'fun', '(', 'n3', ':', 'Q', ')', '=>', 'by'])
    ck((not ok) and 'binder mismatch' in reason, f'box binder that is not the antecedent: {reason!r}')


def t7():
    print('7. arm SN: the environment names what a step introduces (canonical, from a per-attempt base)')
    e = Env('THM P SEQ ( P v Q ) PRF', canon=True, base=5, assign=True)
    ck(e.apply(['have', 'n64', ':', 'P', ':=', 'h1', ';'])[0], 'a have with any name is accepted')
    ck(e.frames[0].htoks[-1][0] == 'n6', f'... and named n<base+1> = n6: {e.frames[0].htoks[-1][0]}')
    ck(e.apply(['have', 'n2', ':', '(', 'P', '∨', 'Q', ')', ':=', 'Or.inl', 'n6', ';'])[0], 'citations are the state\'s names')
    ck(e.apply(['exact', 'n7'])[0] and e.done and e.renamed == 2, f'renamed {e.renamed} of {e.defined}')
    prompt, nd = CASES[-1]
    steps, toks, env = decompose(prompt, nd, canon=True)
    e = Env(prompt, canon=True, base=0, assign=True)
    for s, a, h in steps:
        e.apply(a)
    ck(e.done and e.renamed == 0 and inverse(e.text) == nd, 'a canonical proof replays with 0 renames')


def t5():
    print('5. vocabularies')
    ck(LeanTokenizer('lean_seq').vocab_size == 107, 'lean_seq vocabulary unchanged at 107')
    ck(LeanTokenizer('lean_state').vocab_size == 111, 'lean_state adds exactly four tokens')
    ck(LeanTokenizer('lean_state').itos[LeanTokenizer('lean_state').ref0] == 'n1', 'ref0 still points at n1')
    ck(LeanTokenizer('lean_stateh').with_history and not LeanTokenizer('lean_state').with_history,
       'only lean_stateh prepends the history')


def t6():
    print('6. Lean agrees with the renderer on the state (30 cases)')
    if not os.path.exists(os.path.expanduser('~/.elan/bin/lean')):
        print('  skip (no lean)'); return
    if not os.path.exists('data/p2/train_depth3_f0_a1.jsonl'):
        print('  skip (no control set)'); return
    p = subprocess.run([sys.executable, 'state_gate2.py', '--n', '30', '--chunk', '15', '--seed', '3',
                        '--out', '/tmp/t6.json', '--dump', '/tmp/t6.jsonl'], capture_output=True, text=True)
    try:
        d = json.load(open('/tmp/t6.json'))
    except Exception:
        ck(False, f'gate2 did not run: {p.stdout[-300:]} {p.stderr[-300:]}'); return
    ck(d['mismatches'] == 0 and d['agree'] == 30, f"gate2 on 30 cases: agree {d['agree']}, mismatches {d['mismatches']}")


if __name__ == '__main__':
    t1_t4(); t2(); t3(); t5(); t7(); t6()
    print()
    if FAIL:
        print(f'{len(FAIL)} FAILURES'); [print(' -', f) for f in FAIL]; sys.exit(1)
    print('ALL TESTS PASS')
