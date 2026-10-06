"""Shared helpers for the selection sprint (Worker A): paths, set files, pool/label IO, replay.

Selector *inputs* come from pool rows only (schema §1); Lean verdicts come only from `load_labels` (schema §2).
"""
import hashlib, json, os, sys

DAN = os.path.expanduser('~/runs/proof-selection/nd-takehome')
if DAN not in sys.path:
    sys.path.insert(0, DAN)
RUNS = os.path.expanduser('~/runs/proof-selection')
POOLS = os.environ.get('SEL_POOLS', os.path.join(RUNS, 'pools_v2'))      # overridable for the label-permutation test
SCORES = os.environ.get('SEL_SCORES', os.path.join(RUNS, 'scores'))
CKPT = os.path.join(RUNS, 'ckpts', 'la_T1_best12_s0_r16.pt')
HERE = os.path.dirname(os.path.abspath(__file__))
SPRINT = os.path.dirname(HERE)

SET_FILES = {
    'rl_targets': 'data/ladder/rl_targets.jsonl',
    'dev1108': 'data/bs/dev1108.jsonl',
    'rr600': 'data/ladder/transfer_long_rr600.jsonl',
    'holdout250': 'data/bs/holdout250.jsonl',
    'textbook72': 'data/bs/textbook72.jsonl',
}
SET_K = {'rl_targets': 16, 'dev1108': 16, 'rr600': 32, 'holdout250': 64, 'textbook72': 64}
GEN = 's0_r16'


def read_jsonl(fn):
    with open(fn) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(fn, rows):
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, fn)


def load_set(name):
    return read_jsonl(os.path.join(DAN, SET_FILES[name]))


def ref_lines(rec):
    for k in ('reference_lines', 'L_true', 'n_lines'):
        if rec.get(k) is not None:
            return int(rec[k])
    return None


def pool_stem(set_name, k=None, gen=GEN):
    """k may be an int or a string like '96b' (phase-2 extra pools)."""
    return f'{set_name}__{gen}__k{k or SET_K[set_name]}'


def pool_path(set_name, k=None, gen=GEN):
    return os.path.join(POOLS, pool_stem(set_name, k, gen) + '.jsonl')


def labels_path(set_name, k=None, gen=GEN):
    return os.path.join(POOLS, pool_stem(set_name, k, gen) + '.labels.jsonl')


def load_pool(set_name, k=None, gen=GEN):
    return read_jsonl(pool_path(set_name, k, gen))


def load_labels(set_name, k=None, gen=GEN):
    """name -> list[bool] (valid), aligned with attempts.  The ONLY sanctioned path to Lean verdicts."""
    return {r['name']: [bool(v) for v in r['valid']] for r in read_jsonl(labels_path(set_name, k, gen))}


def load_scores(set_name, scorer, k=None, gen=GEN):
    fn = os.path.join(SCORES, f'{pool_stem(set_name, k, gen)}__{scorer}.jsonl')
    return {r['name']: [float(x) for x in r['scores']] for r in read_jsonl(fn)}


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


_TOK = [None]


def get_tok():
    if _TOK[0] is None:
        from tokenizer import make_tokenizer
        _TOK[0] = make_tokenizer('lean_staten')
    return _TOK[0]


def replay(prompt, base, actions, tok=None, has_eos=None):
    """Replay stored model-emitted action ids through Env(canon, base, assign).

    -> (env, pairs) with pairs = [(state ids, effective action ids), ...] for every step the attempt took, where the
    effective action is the action after the environment's name assignment (what the SN model is trained on) and the
    last pair of a failed attempt holds the decoded emitted action that failed (the environment rejected it) or was
    truncated (has_eos False: never applied, exactly as in collection).  Also returns per-step state ids via env.
    """
    from state_env import Env
    tok = tok or get_tok()
    e = Env(prompt, canon=True, base=base, assign=True)
    pairs = []
    for t, a in enumerate(actions):
        st = tok.encode_toks(e.state_tokens())
        if has_eos is not None and not has_eos[t]:
            pairs.append((st, list(a)))
            break
        atoks = [tok.itos[x] for x in a]
        ok, _ = e.apply(atoks)
        eff = tok.encode_toks(e._eff) if ok else list(a)
        pairs.append((st, eff))
        if not ok or e.done:
            break
    return e, pairs


def state_sha1(ids):
    return hashlib.sha1(','.join(map(str, ids)).encode()).hexdigest()[:16]


def check_attempt(prompt, att, tok=None):
    """Replay one stored attempt; True iff every state's sha1 matches, steps match, and (done) nd matches byte for byte."""
    e, pairs = replay(prompt, att['base'], att['actions'], tok, att.get('has_eos'))
    if len(pairs) != len(att['state_sha1']) or e.steps != att['steps']:
        return False
    if any(state_sha1(st) != h for (st, _), h in zip(pairs, att['state_sha1'])):
        return False
    if bool(e.done) != bool(att['done']):
        return False
    if att['done'] and e.nd() != att['nd']:
        return False
    return True
