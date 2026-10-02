"""rel tokenizer: exact decode(encode(proof)) roundtrip on every cap-6 proof and pool proof, and the text-level
pipeline (verifier, incremental checker, normaliser, pruner) on decoded rel text.

    cd ab3/repr && $PY -m pytest -q -p no:cacheprovider tests/test_rel_tokenizer.py

Proof sources: frozen/data/train_cap6.jsonl + heldout_cap6.jsonl (all start at N1: exact text roundtrip) and the
"pool" proofs: arm C's found proofs (lengthexp/runs/C_s10*/found_targets.jsonl; abs samples start anywhere in
N1..N41, so the roundtrip is exact up to renumbering to N1) and the reference proofs of the test-set sources
(gen_proof / minlen_proof / constructed_proof). The RL pool (targets_L6-11) has no proofs.
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPR = os.path.dirname(HERE)
sys.path.insert(0, REPR)
from common_repr import DATA, LE, lib_path, read_jsonl  # noqa: E402
lib_path()
from tokenizer import Tokenizer  # noqa: E402
from nd_verify import verify_text  # noqa: E402

SOURCES = ['/scratch/cowifornia/nd-rl/ab2/data/reserve.jsonl', '/scratch/cowifornia/nd-rl/ab2/sealed/sealed_L12-14.jsonl',
           '/scratch/cowifornia/nd-rl/ab3/sealed/sealed_L15-16.jsonl', '/scratch/cowifornia/nd-rl/ab3/sealed/sealed_L17-18.jsonl',
           '/scratch/cowifornia/nd-rl/ab3/sealed/matched_L6-14.jsonl']


def renumber(body):
    """Proof text renumbered so that its first line is N1 (every N<k> shifted by the same amount)."""
    toks = body.split()
    s = int(toks[0][1:]) - 1
    return ' '.join(f'N{int(t[1:]) - s}' if re.fullmatch(r'N\d+', t) else t for t in toks)


def roundtrip(recs, tk, exact):
    bad, fail_enc, n = [], 0, 0
    for prompt, proof in recs:
        n += 1
        try:
            ids = tk.encode_proof(proof)
        except ValueError:          # a citation that points forward (nd_verify allows it for some box refs)
            fail_enc += 1
            continue
        d = tk.decode(ids)
        want = proof if exact else renumber(proof)
        if d != want:
            bad.append((proof, d))
    return n, fail_enc, bad


def cap6():
    return [(r['prompt'], r['proof']) for fn in ('train_cap6.jsonl', 'heldout_cap6.jsonl') for r in read_jsonl(os.path.join(DATA, fn))]


def pool():
    out = []
    for fn in sorted(glob.glob(os.path.join(LE, 'runs', 'C_s10*', 'found_targets.jsonl'))):
        tg = {}
        for r in read_jsonl(fn):
            out.append((None, r['proof']))
    for fn in SOURCES:
        if not os.path.isfile(fn):
            continue
        for r in read_jsonl(fn):
            for k in ('gen_proof', 'minlen_proof', 'constructed_proof'):
                if r.get(k):
                    out.append((r['prompt'], r[k]))
    return out


def test_rel_roundtrip_exact_on_all_cap6_proofs():
    tk = Tokenizer('rel')
    recs = cap6()
    assert all(p.split()[0] == 'N1' for _, p in recs)
    n, fe, bad = roundtrip(recs, tk, exact=True)
    assert n == 159990 and fe == 0 and not bad, (fe, bad[:3])


def test_rel_roundtrip_on_pool_proofs():
    tk = Tokenizer('rel')
    recs = pool()
    n, fe, bad = roundtrip(recs, tk, exact=False)
    print(f'pool proofs {n}, forward-citation encode failures {fe}, mismatches {len(bad)}')
    assert n > 10000 and not bad and fe == 0, (fe, bad[:3])


def test_decoded_rel_text_verifies_and_checks_like_the_original():
    """The verdict of nd_verify and of the evaluator's incremental checker are the same on the decoded rel text as on
    the original (renumbered) text; normalise() gives the same key."""
    sys.path.insert(0, LE)
    from ndcheck import classify
    from normalize import norm
    tk = Tokenizer('rel')
    recs = [(p, q) for p, q in pool() if p is not None][::7] + cap6()[::97]
    for prompt, proof in recs:
        d = tk.decode(tk.encode_proof(proof))
        a, b = verify_text(prompt + ' ' + proof), verify_text(prompt + ' ' + d)
        assert a[0] == b[0] and a[2] == b[2]
        ca, cb = classify(prompt, renumber(proof)), classify(prompt, d)
        assert (ca['kind'], ca['line'], ca['reason']) == (cb['kind'], cb['line'], cb['reason'])
        assert norm(proof) == norm(d)


def test_rel_malformed_citation_decodes_to_rejected_text():
    """A B<k> reaching before line 1 decodes to N0 / N-<j>, which nd_verify and the checker reject (never crash)."""
    sys.path.insert(0, LE)
    from ndcheck import classify
    tk = Tokenizer('rel')
    prompt = 'THM P SEQ ( P v Q ) PRF'
    ids = tk.encode_proof('N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; QED')
    b1, b5 = tk.stoi['B1'], tk.stoi['B5']
    bad = [b5 if x == b1 else x for x in ids]
    d = tk.decode(bad)
    assert 'N-3' in d
    assert not verify_text(prompt + ' ' + d)[0]
    c = classify(prompt, d)
    assert c['kind'] == 'bad_line' and c['line'] == 2
