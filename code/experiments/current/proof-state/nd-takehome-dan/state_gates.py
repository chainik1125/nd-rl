#!/usr/bin/env python3
"""Gates 1 and 3 of run `state-env` (CPU) plus the shape statistics the training and sampling settings come from.

  python3 state_gates.py --data data/p2/train_depth3_f0_a1.jsonl --out artifacts/se/gate13.json [--lean 5000]

Gate 1 (round trip): every proof of the set decomposes into actions and the actions' rendered chunks reassemble the
  `lean_seq` text byte for byte; a random `--lean N` of the reassembled texts are accepted by Lean through
  `lean_judge.judge_many` (the fallback path: nd2lean + Lean core).
Gate 3 (environment replay): the same walk IS the replay -- `state_env.decompose` drives a fresh `Env` with the
  proof's actions and every syntactic check of the environment must pass and the attempt must finish.
Statistics: actions per proof, action and state token lengths, and how often the next `have` name is NOT
  `max name in scope + 1` (the one place where the state does not determine the action's name token).
"""
import argparse, collections, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lean_tok import LeanTokenizer, ParseFail
from state_env import decompose, is_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/p2/train_depth3_f0_a1.jsonl')
    ap.add_argument('--out', default='artifacts/se/gate13.json')
    ap.add_argument('--lean', type=int, default=5000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--canon', action='store_true', help='arm SN: canonical (scope-determined) names')
    a = ap.parse_args()
    tk = LeanTokenizer('lean_staten' if a.canon else 'lean_state')
    rng = random.Random(a.seed)
    recs = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    nact = collections.Counter(); alen = collections.Counter(); slen = collections.Counter()
    akind = collections.Counter(); hist_len = collections.Counter()
    fails = []
    name_steps = 0; name_offbyN = 0
    texts = []
    t0 = time.time()
    for r in recs:
        try:
            steps, toks, env = decompose(r['prompt'], r['proof'], canon=a.canon)
        except (ParseFail, ValueError, AssertionError, KeyError, IndexError) as e:
            fails.append({'name': r.get('name'), 'reason': str(e)})
            continue
        if tk.text(env.text) != tk.text(toks):
            fails.append({'name': r.get('name'), 'reason': 'text mismatch'})
            continue
        texts.append((r['prompt'], env.nd()))
        nact[len(steps)] += 1
        for st, act, hs in steps:
            alen[len(act) + 1] += 1          # +1: the <eos> the model must emit
            slen[len(st)] += 1
            hist_len[len(hs)] += 1
            akind['exact' if act[0] == 'exact' else ('box' if act[-1] == 'by' else 'have')] += 1
            if act[0] == 'have':
                scope = set()
                for f in env.frames:
                    pass
                name_steps += 1
        # name predictability: recompute per step with a fresh replay of the scope
        from state_env import Env
        e2 = Env(r['prompt'], canon=a.canon)
        for st, act, hs in steps:
            if act[0] == 'have':
                if int(act[1][1:]) != e2.next_name():
                    name_offbyN += 1
            e2.apply(act)
    res = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'data': a.data, 'n_records': len(recs),
           'gate1_gate3_failures': len(fails), 'failure_examples': fails[:10],
           'actions_per_proof': dict(sorted(nact.items())),
           'mean_actions_per_proof': sum(k * v for k, v in nact.items()) / max(1, sum(nact.values())),
           'action_kinds': dict(akind),
           'action_len_tokens': dict(sorted(alen.items())),
           'action_len_max': max(alen) if alen else 0,
           'action_len_p999': _pct(alen, 0.999), 'action_len_p9999': _pct(alen, 0.9999),
           'state_len_tokens_max': max(slen) if slen else 0, 'state_len_p999': _pct(slen, 0.999),
           'state_len_mean': sum(k * v for k, v in slen.items()) / max(1, sum(slen.values())),
           'hist_len_max': max(hist_len) if hist_len else 0, 'hist_len_p999': _pct(hist_len, 0.999),
           'have_actions': name_steps, 'have_name_not_scopemax_plus1': name_offbyN,
           'decompose_secs': time.time() - t0}
    if a.lean:
        from lean_judge import judge_many, stats as jstats
        sel = rng.sample(texts, min(a.lean, len(texts)))
        t1 = time.time()
        out = judge_many(sel)
        bad = [(p, nd) for (p, nd), (ok, why, nl) in zip(sel, out) if not ok]
        res['lean_checked'] = len(sel); res['lean_rejected'] = len(bad)
        res['lean_reject_examples'] = [{'prompt': p, 'nd': nd} for p, nd in bad[:5]]
        res['lean_secs'] = time.time() - t1
        res['lean_judge_stats'] = jstats()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    for k, v in res.items():
        if k not in ('action_len_tokens', 'actions_per_proof', 'failure_examples', 'lean_reject_examples'):
            print(k, '=', v)
    print('actions_per_proof', res['actions_per_proof'])


def _pct(counter, q):
    tot = sum(counter.values())
    if not tot:
        return 0
    c = 0
    for k in sorted(counter):
        c += counter[k]
        if c >= q * tot:
            return k
    return max(counter)


if __name__ == '__main__':
    main()
