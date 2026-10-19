"""Agreement of ndcheck.classify (incremental checker) with nd_verify, plus the failure categories.

    python tests/test_ndcheck.py [--n 4000]        (also runs under pytest)

1. Every stored proof (minlen/gen proofs of the sealed and reserve records, cap-6 training proofs) must be
   'ok' for both.
2. Random corruptions of those proofs (token substitution / deletion / insertion / duplication, line
   swaps and deletions, index shifts, truncation, cut-offs without QED): kind == 'ok' must equal nd_verify's
   verdict, except for the documented forward-box-citation leniency of nd_verify.
3. Hand-made cases for each failure kind.
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ndcheck import classify, sequent  # noqa: E402
from nd_verify.verify import parse_proof_tokens  # noqa: E402

AB2 = '/scratch/cowifornia/nd-rl/ab2'
VOCAB = (['(', ')', '~', '&', 'v', '>', 'P', 'Q', 'R', 'S', 'F', '|', ':', ';', 'QED']
         + ['PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN']
         + [f'N{i}' for i in range(1, 30)])


def load_proofs(n, seed=0):
    rng = random.Random(seed)
    out = []
    for fn in (f'{AB2}/sealed/sealed_L12-14.jsonl', f'{AB2}/sealed/sealed_L15-16.jsonl'):
        for l in open(fn):
            r = json.loads(l)
            for f in ('minlen_proof', 'gen_proof'):
                if r.get(f):
                    out.append((r['prompt'], r[f]))
    res = []
    with open(f'{AB2}/data/reserve.jsonl') as f:
        for i, l in enumerate(f):
            if i % 97 == 0:
                r = json.loads(l)
                res += [(r['prompt'], r[k]) for k in ('minlen_proof', 'gen_proof') if r.get(k)]
            if len(res) >= n:
                break
    cap = []
    with open('/scratch/cowifornia/nd-rl/ab3/frozen/data/train_cap6.jsonl') as f:
        for i, l in enumerate(f):
            if i % 101 == 0:
                r = json.loads(l)
                cap.append((r['prompt'], r['proof']))
            if len(cap) >= n // 2:
                break
    out += res + cap
    rng.shuffle(out)
    return out


def forward_box_cite(prompt, body):
    """True if some IMPI/NEGI/ORE cites a box that starts after the citing line (nd_verify accepts it)."""
    try:
        lines = parse_proof_tokens(body.split())
    except Exception:
        return False
    for ln in lines:
        if ln['rule'] in ('IMPI', 'NEGI', 'ORE') and any(r >= ln['idx'] for r in ln['refs']):
            return True
    return False


def mutate(body, rng):
    toks = body.split()
    lines = ' '.join(toks[:-1]).split(' ; ')
    lines = [l + ' ;' for l in lines if l]
    op = rng.randrange(10)
    if op == 0:
        i = rng.randrange(len(toks)); toks[i] = rng.choice(VOCAB)
    elif op == 1:
        del toks[rng.randrange(len(toks))]
    elif op == 2:
        toks.insert(rng.randrange(len(toks) + 1), rng.choice(VOCAB))
    elif op == 3 and len(lines) > 1:
        i = rng.randrange(len(lines) - 1); lines[i], lines[i + 1] = lines[i + 1], lines[i]
        toks = ' '.join(lines + ['QED']).split()
    elif op == 4 and len(lines) > 1:
        del lines[rng.randrange(len(lines))]
        toks = ' '.join(lines + ['QED']).split()
    elif op == 5:   # renumber from a random start (valid if nothing else changes; overflow beyond 64 fails)
        s = rng.randint(0, 60)
        toks = [f'N{int(t[1:]) + s}' if t.startswith('N') and t[1:].isdigit() else t for t in toks]
    elif op == 6:   # truncate and put QED (early QED)
        k = rng.randrange(1, len(lines) + 1)
        toks = ' '.join(lines[:k] + ['QED']).split()
    elif op == 7:   # cut off without QED (ran out of tokens)
        toks = toks[:rng.randrange(1, len(toks))]
    elif op == 8:   # change a citation
        idx = [i for i, t in enumerate(toks) if t.startswith('N') and t[1:].isdigit() and i > 0 and toks[i - 1] != ';']
        if idx:
            i = rng.choice(idx); toks[i] = f'N{max(1, int(toks[i][1:]) + rng.choice([-2, -1, 1, 2]))}'
    else:           # duplicate a line (index mismatch) or change a depth marker
        if rng.random() < 0.5 and len(lines) > 1:
            i = rng.randrange(len(lines)); lines.insert(i, lines[i]); toks = ' '.join(lines + ['QED']).split()
        else:
            bars = [i for i, t in enumerate(toks) if t == '|']
            if bars and rng.random() < 0.5:
                del toks[rng.choice(bars)]
            else:
                idx = [i + 1 for i, t in enumerate(toks) if t.startswith('N') and (i == 0 or toks[i - 1] == ';')]
                toks.insert(rng.choice(idx), '|')
    return ' '.join(toks)


def run(n=4000, n_mut=6, seed=0, verbose=True):
    rng = random.Random(seed)
    proofs = load_proofs(n, seed)
    bad_clean = []
    for p, b in proofs:
        c = classify(p, b)
        if not (c['ok'] and c['kind'] == 'ok'):
            bad_clean.append((p, b, c))
    stats = {'clean': len(proofs), 'clean_fail': len(bad_clean), 'mut': 0, 'mut_ok': 0, 'agree': 0,
             'disagree_forward_box': 0, 'disagree_other': 0, 'kinds': {}}
    other = []
    for p, b in proofs:
        seq = sequent(p)
        for _ in range(n_mut):
            m = mutate(b, rng)
            c = classify(p, m, seq)
            stats['mut'] += 1
            stats['mut_ok'] += c['ok']
            key = c['kind'] + ':' + c['reason']
            stats['kinds'][key] = stats['kinds'].get(key, 0) + 1
            if c['ok'] == (c['kind'] == 'ok'):
                stats['agree'] += 1
            elif c['ok'] and forward_box_cite(p, m):
                stats['disagree_forward_box'] += 1
            else:
                stats['disagree_other'] += 1
                other.append((p, m, c))
    if verbose:
        print(json.dumps({k: v for k, v in stats.items() if k != 'kinds'}))
        for k, v in sorted(stats['kinds'].items(), key=lambda x: -x[1]):
            print(f'  {k:32s} {v}')
        for p, b, c in (bad_clean + other)[:5]:
            print('DISAGREE', c, '\n  ', p, '\n  ', b)
    return stats


def test_cases():
    p = 'THM ( P > Q ) , P SEQ Q PRF'
    good = 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED'
    c = classify(p, good)
    assert c['ok'] and c['kind'] == 'ok' and c['n_lines'] == 3 and c['start'] == 1, c
    c = classify(p, 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N2 N1 ; QED')
    assert not c['ok'] and c['kind'] == 'bad_line' and c['line'] == 3 and c['reason'] == 'rule', c
    c = classify(p, 'N1 ( P > Q ) : PR ; N2 P : PR ; QED')
    assert not c['ok'] and c['kind'] == 'early_qed' and c['line'] == 2 and c['reason'] == 'not_conclusion', c
    c = classify(p, 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1')
    assert not c['ok'] and c['kind'] == 'no_qed' and c['line'] == 2, c
    c = classify(p, 'N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; QED')
    assert not c['ok'] and c['kind'] == 'bad_line' and c['line'] == 1 and c['reason'] == 'premise', c
    c = classify(p, 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 | Q : AS ; QED')
    assert c['kind'] == 'early_qed' and c['reason'] == 'inside_subproof', c
    c = classify(p, 'N63 ( P > Q ) : PR ; N64 P : PR ; P : IMPE N63 N64 ; QED')
    assert c['kind'] == 'bad_line' and c['line'] == 3 and c['reason'] == 'index_overflow', c
    c = classify(p, 'N63 ( P > Q ) : PR ; N64 P : PR ; QED')
    assert c['kind'] == 'early_qed' and c['reason'] == 'index_limit', c
    c = classify(p, 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : QED')
    assert c['kind'] == 'bad_line' and c['line'] == 3 and c['reason'] == 'parse', c
    c = classify(p, 'N5 ( P > Q ) : PR ; N6 P : PR ; N7 Q : IMPE N5 N6 ; QED')
    assert c['ok'] and c['kind'] == 'ok' and c['start'] == 5, c


def test_agreement():
    st = run(n=600, n_mut=4, verbose=False)
    assert st['clean_fail'] == 0, st
    assert st['disagree_other'] == 0, st


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--mut', type=int, default=6)
    a = ap.parse_args()
    test_cases()
    print('hand-made cases: ok')
    st = run(a.n, a.mut)
    sys.exit(0 if st['clean_fail'] == 0 and st['disagree_other'] == 0 else 1)
