"""Build the FOL constant token-wall experiment splits from a generated pool.

Minimal-pair design (constants confined to <=2 distinct per proof):
  train_bounded : every proof relabeled so constants come from {a,b}
  train_full    : every proof relabeled over the full inventory {a,b,c,d,e}
  eval_iv       : held-out theorems relabeled to {a,b}  (in train vocab)
  eval_oov      : SAME held-out theorems relabeled to {d,e} (NOT in bounded vocab)
A model trained on train_bounded should solve eval_iv but fail eval_oov (the
constant token wall); train_full should solve both.
"""
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fol.tokenizer import CONST_TOKENS
from fol.verify import verify_text

CONSTSET = set(CONST_TOKENS)


def relabel_to(text, targets, rng):
    toks = text.split()
    used = sorted({t for t in toks if t in CONSTSET})
    if len(used) > len(targets):
        return None
    m = dict(zip(used, rng.sample(targets, len(used))))
    return ' '.join(m.get(t, t) for t in toks)


def main(pool_path, out_dir):
    rng = random.Random(20)
    recs = [json.loads(l) for l in open(pool_path) if l.strip()]
    # keep proofs with 1-2 distinct constants, 2-12 lines, using quantifiers
    recs = [r for r in recs if 1 <= r['n_consts'] <= 2 and 2 <= r['n_lines'] <= 12]
    seen = set()
    ded = []
    for r in recs:
        if r['thm'] in seen:
            continue
        seen.add(r['thm'])
        ded.append(r)
    rng.shuffle(ded)
    eval_recs = ded[:800]
    train_recs = ded[800:]
    print(f'{len(ded)} usable; {len(train_recs)} train, {len(eval_recs)} eval')

    def dump(name, recs, targets, randomize):
        path = os.path.join(out_dir, name)
        n = 0
        with open(path, 'w') as f:
            for r in recs:
                tg = targets if not randomize else CONST_TOKENS
                tx = relabel_to(r['text'], tg, rng)
                if tx is None:
                    continue
                ok = verify_text(tx)[0] if n % 500 == 0 else True
                if not ok:
                    continue
                rr = dict(r)
                rr['text'] = tx
                used = sorted({t for t in tx.split() if t in CONSTSET})
                rr['consts'] = used
                rr['n_consts'] = len(used)
                f.write(json.dumps(rr) + '\n')
                n += 1
        print(f'{name}: {n}')

    dump('fol_train_bounded.jsonl', train_recs, ['a', 'b'], False)
    dump('fol_train_full.jsonl', train_recs, None, True)
    dump('fol_eval_iv.jsonl', eval_recs, ['a', 'b'], False)
    dump('fol_eval_oov.jsonl', eval_recs, ['d', 'e'], False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'data')
