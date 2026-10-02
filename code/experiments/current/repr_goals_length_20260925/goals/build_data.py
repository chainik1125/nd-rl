#!/usr/bin/env python3
"""Annotate the cap-6 pretraining data (and, as a check only, the held-out file and every RL-found proof) with the
canonical goal annotation, and validate: every proof verifies, strip(annotate(p)) == p, annotate is deterministic,
the stripped text verifies, the annotated text round-trips through the tokenizer and fits 1,024 tokens.

    python build_data.py            # -> data/train_cap6_goals.jsonl, data/heldout_cap6_goals.jsonl, data/VALIDATION.json

Records keep every original field; `proof` becomes the annotated proof, `plain` holds the original proof, `n_lines`
is unchanged (annotations add no lines). The order of train_cap6.jsonl is kept (039's val slice = the first 1,024).
"""
import glob, hashlib, json, multiprocessing as mp, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import goals as G  # noqa: E402
from gtokenizer import GoalTokenizer  # noqa: E402

AB3 = '/scratch/cowifornia/nd-rl/ab3'
SRC = {'train_cap6': f'{AB3}/frozen/data/train_cap6.jsonl', 'heldout_cap6': f'{AB3}/frozen/data/heldout_cap6.jsonl'}
FOUND = sorted(glob.glob(f'{AB3}/lengthexp/runs/*/found_targets.jsonl')) + sorted(glob.glob(f'{AB3}/base039/runs/*/found_targets.jsonl'))
OUT = os.path.join(HERE, 'data')
TOK = GoalTokenizer()


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def one(args):
    prompt, proof = args
    proof = ' '.join(proof.split())
    ok = G.verify_text(prompt + ' ' + proof)[0]
    if not ok:
        return None, {'not_verified': 1}
    a = G.annotate(prompt, proof, check=False)
    c = {'proofs': 1}
    if G.annotate(prompt, proof, check=False) != a:
        c['nondeterministic'] = 1
    if G.strip(a) != proof:
        c['strip_mismatch'] = 1
    if not G.verify_text(prompt + ' ' + G.strip(a))[0]:
        c['stripped_rejected'] = 1
    if TOK.decode(TOK.encode_proof(a)) != a:
        c['tokenizer_mismatch'] = 1
    n_seq = len(TOK.encode_prompt(prompt)) + len(TOK.encode_proof(a))
    if n_seq > 1024:
        c['over_1024'] = 1
    g = a.split().count('GOAL')
    c['groups'] = g
    c['with_groups'] = int(g > 0)
    c['tok_plain'] = len(proof.split())
    c['tok_annot'] = len(a.split())
    c['max_seq'] = n_seq
    return a, c


def merge(tot, c):
    for k, v in c.items():
        if k == 'max_seq':
            tot[k] = max(tot.get(k, 0), v)
        else:
            tot[k] = tot.get(k, 0) + v


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = mp.Pool(32)
    val = {'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'policy': G.DEFAULT_POLICY,
           'goals_py_sha256': sha256(os.path.join(HERE, 'lib', 'goals.py')), 'sources': {}}
    for name, fn in SRC.items():
        recs = [json.loads(l) for l in open(fn) if l.strip()]
        res = pool.map(one, [(r['prompt'], r['proof']) for r in recs], chunksize=512)
        tot = {}
        out = os.path.join(OUT, f'{name}_goals.jsonl')
        with open(out + '.tmp', 'w') as f:
            for r, (a, c) in zip(recs, res):
                merge(tot, c)
                if a is None:
                    raise SystemExit(f'{name}: record {r.get("name")} does not verify')
                f.write(json.dumps(dict(r, proof=a, plain=r['proof'])) + '\n')
        os.replace(out + '.tmp', out)
        tot['overhead'] = tot['tok_annot'] / tot['tok_plain'] - 1
        val['sources'][name] = dict(tot, file=out, sha256=sha256(out), src=fn, src_sha256=sha256(fn), n=len(recs))
        print(name, json.dumps(tot), flush=True)
    # RL-found proofs: check only (not written; not training data for this variant)
    pairs = []
    for fn in FOUND:
        for l in open(fn):
            if l.strip():
                r = json.loads(l)
                pairs.append((' '.join(('THM ' + r['thm'].replace('|-', 'SEQ') + ' PRF').split()), r['proof']))
    pairs = list(dict.fromkeys(pairs))
    tot = {}
    for a, c in pool.imap_unordered(one, pairs, chunksize=256):
        merge(tot, c)
    tot['overhead'] = tot['tok_annot'] / tot['tok_plain'] - 1
    val['sources']['found_rl'] = dict(tot, files=FOUND, n=len(pairs))
    print('found_rl', json.dumps(tot), flush=True)
    bad = {k: v for s in val['sources'].values() for k, v in s.items()
           if k in ('nondeterministic', 'strip_mismatch', 'stripped_rejected', 'tokenizer_mismatch', 'over_1024', 'not_verified') and v}
    val['ok'] = not bad
    json.dump(val, open(os.path.join(OUT, 'VALIDATION.json'), 'w'), indent=1)
    print('VALIDATION', 'ok' if not bad else f'FAILED {bad}')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
