#!/usr/bin/env python3
"""CPU tests of rl_goals.py's edits (judging stripped samples, found-proof records, relabelling, the mix).

    python tests/test_rl_goals.py
"""
import collections, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)
sys.path.insert(0, '/scratch/cowifornia/nd-rl/ab3/frozen/lib')
sys.path.insert(0, os.path.join(G, 'lib'))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location('rl_goals', os.path.join(G, 'rl_goals.py'))
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)
import goals as GL  # noqa: E402
from gtokenizer import GoalTokenizer  # noqa: E402

FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)
        print('FAIL', m)


def main():
    tok = GoalTokenizer()
    check(rl.CONFIG['max_new'] == 768 and rl.CONFIG['max_len_train'] == 11 and rl.CONFIG['max_proof_lines_train'] == 11,
          f"config {rl.CONFIG['max_new']} {rl.CONFIG['max_len_train']} {rl.CONFIG['max_proof_lines_train']}")
    t = {'name': 't0', 'prompt': 'THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF', 'L_true': 6, 'key': 'k0', 'thm': '( P > Q ) , ( ~ Q ) |- ( ~ P )'}
    plain = 'N3 ( P > Q ) : PR ; N4 ( ~ Q ) : PR ; N5 | P : AS ; N6 | Q : IMPE N3 N5 ; N7 | F : NEGE N6 N4 ; N8 ( ~ P ) : NEGI N5 N7 ; QED'
    canon = GL.annotate(t['prompt'], plain)
    wrong = canon.replace('GOAL F', 'GOAL ( P & Q )')                     # verified, but a non-canonical goal
    noann = plain                                                         # verified, no annotation at all
    shifted = GL.annotate(t['prompt'], ' '.join(f'N{int(x[1:]) + 2}' if x[0] == 'N' and x[1:].isdigit() else x for x in plain.split()))
    malformed = canon.replace('GOAL F', 'GOAL ( F')                        # malformed group: must fail
    midline = plain.replace('| P : AS', '| P GOAL F : AS')                 # GOAL inside a line: must fail
    samples = [(0, canon), (0, wrong), (0, noann), (0, shifted), (0, malformed), (0, midline)]
    found = collections.defaultdict(dict)
    relab = {}
    st = rl.judge_samples([t], samples, found, 1, relab, rl.CONFIG, set(), {})
    check(st['accept_rate'] == 4 / 6, f"accept {st['accept_rate']}")
    check(len(found['t0']) == 1, f'one distinct stripped proof: {list(found["t0"])}')
    rec = next(iter(found['t0'].values()))
    check(rec['proof'] == canon and rec['plain'] == plain and rec['written'] == 6, f'found record {rec}')
    a = st['annotation']
    check(abs(a['samples_with_malformed'] - 2 / 6) < 1e-9, f'malformed {a}')
    check(a['consistency_checked'] == 4 and abs(a['exact_canonical'] - 2 / 4) < 1e-9, f'consistency {a}')
    # relabelling: a proof of another conclusion from the same premises (ends in Q > Q) gets a canonical annotation
    t2 = {'name': 't1', 'prompt': 'THM P SEQ ( Q & P ) PRF', 'L_true': 6, 'key': 'k1', 'thm': 'P |- ( Q & P )'}
    other = 'N1 P : PR ; N2 | Q : AS ; N3 | | R : AS ; N4 | | Q : R N2 ; N5 | ( R > Q ) : IMPI N3 N4 ; N6 | Q : R N2 ; N7 ( Q > Q ) : IMPI N2 N6 ; QED'
    raw_other = 'N1 P : PR ; N2 | Q : AS ; GOAL S ' + other[len('N1 P : PR ; N2 | Q : AS ; '):]
    relab2 = {}
    rl.judge_samples([t2], [(0, raw_other)], collections.defaultdict(dict), 1, relab2, dict(rl.CONFIG, relabel_min_lines=7), set(), {})
    check(len(relab2) == 1, f'relabelled {relab2}')
    for p, q in relab2.values():
        check(p == 'THM P SEQ ( Q > Q ) PRF' and q == GL.annotate(p, other) and GL.strip(q) == other, f'relabel record {p} {q}')
    # the mix: the found record's raw text, and the retention records as given
    retained = [{'prompt': t['prompt'], 'proof': canon}]
    data, n_rl = rl.build_mix([t], found, retained, dict(rl.CONFIG, retain=1), random.Random(0), tok, {}, {}, {})
    check(n_rl >= 1 and tok.decode(data[0][1]) == canon, f'mix record {tok.decode(data[0][1])}')
    # max_proof_lines_train counts stripped lines
    check(rl.proof_lines(canon) == 6, 'proof_lines')
    print('ALL OK' if not FAILS else f'{len(FAILS)} FAILURES')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
