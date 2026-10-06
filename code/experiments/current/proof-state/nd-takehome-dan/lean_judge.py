#!/usr/bin/env python3
"""lean_judge -- Lean alone decides whether a proof counts (Dan, 2026-09-27).

`nd_verify` judges nothing any more: not the in-loop reward, not evaluation, not coverage, not counting.  It stays in
the repository unmodified only so that pre-2026-09-27 numbers (measured under Lean AND `nd_verify`) can be reproduced.
`verify_text` here is a drop-in replacement for `nd_verify.verify_text`, so each loop file changes only its import.

    verify_text(text) -> (ok, reason, n_lines)          text = '<prompt ending in PRF> <ND proof>'
    judge_many(pairs) -> [(ok, reason, n_lines), ...]   pairs = [(prompt, ND proof), ...]   <-- use this in a loop
    register(prompt, nd, ok)                            called by lean_gate.gate for every sample it checked
    stats() -> dict                                     marker / registry-hit / fallback counts and fallback Lean seconds

Decision order for one (prompt, ND proof):

  1. the ND string carries a marker -- `LEANPARSE <reason>` (the sampled tokens are outside the strict `lean_seq`
     grammar) or `LEANREJ <nd>` (the literal text parsed but Lean rejected it) -- => reject.  Accepted samples never
     carry a marker (pitfall 1: whatever the judge accepts becomes expert-iteration training text).
  2. `lean_gate.gate` registered a Lean verdict for this (prompt, ND string) => that verdict.  This is the hot path:
     the judging pass that follows `sample.generate()` is a dictionary lookup, no second Lean run.
  3. otherwise: translate the ND proof with `nd2lean.translate` and check it in Lean core, **batched** through
     `lean_gate.check_sources` (a Lean process costs ~1 s to start, so a one-string-at-a-time fallback inside a loop is
     catastrophic -- pitfall 4).  Strings that reach this path: hindsight relabels against a rewritten theorem
     (pitfall 2), dataset records, and samples from callers that use `generate_ids` directly and never see the gate
     (`coverage.py`, `grpo.py`).  The translation runs with `require_all_pr=False`: Lean does not require a premise to be
     re-stated, and omitting one is the largest single class of proofs Lean accepts and `nd_verify` rejects (41.7 % of the
     stored Lean-only class; see `LEAN_JUDGE.md`).

Why Lean on the grammar is a sound judge: the `lean_seq` vocabulary is 107 tokens (see `lean_tok.py`) and contains no
`sorry`, no tactic automation and no library lemma, so the only inferences reachable beyond the ND rules are the
eliminators `.elim` resolves to.  See `LEAN_JUDGE.md`.

`n_lines` is the `;`-count of the ND body -- the same expression `nd_verify.verify_text` returns, so line counts are
identical on every proof both accept.
"""
import os, sys, time, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nd2lean

MAX_REGISTRY = int(os.environ.get('LEAN_JUDGE_REGISTRY', '1000000'))

_registry = collections.OrderedDict()     # 16-byte digest of (prompt, nd) -> bool
_stats = collections.Counter()
_fallback_s = [0.0]


def _key(prompt, nd):
    return hashlib.blake2b(prompt.encode() + b'\x00' + nd.encode(), digest_size=16).digest()


def register(prompt, nd, ok):
    """Record a Lean verdict for (prompt, ND string).  Two literal texts denoting the same ND proof are alpha-variants
    (hypothesis names are labels), and Lean's verdict is invariant to renaming, so a conflict is a bug: it is counted,
    and acceptance wins (some literal text of this ND proof was accepted)."""
    k = _key(prompt, nd)
    if k in _registry:
        if _registry[k] != bool(ok):
            _stats['conflict'] += 1
            ok = True
        else:
            return
    _registry[k] = bool(ok)
    while len(_registry) > MAX_REGISTRY:
        _registry.popitem(last=False)


def stats():
    d = dict(_stats)
    d.setdefault('conflict', 0)
    d['registry'] = len(_registry)
    d['fallback_lean_s'] = _fallback_s[0]
    return d


def n_lines(nd):
    """lines of an ND proof body = number of ';' tokens, exactly as nd_verify.verify_text counts them."""
    return sum(1 for t in nd.split() if t == ';')


def split_text(text):
    """'THM ... SEQ ... PRF <body>' -> (prompt, body).  Returns (None, None) if there is no PRF."""
    i = text.find(' PRF ')
    if i < 0:
        return (None, None) if not text.endswith(' PRF') else (text, '')
    return text[:i + 4], text[i + 5:]


def judge_many(pairs):
    """pairs: list of (prompt, ND proof string) -> list of (ok, reason, n_lines), in order.
    Every string that needs Lean is translated and checked in one batched run."""
    res = [None] * len(pairs)
    todo = {}                             # (prompt, nd) -> list of indices
    for i, (prompt, nd) in enumerate(pairs):
        if prompt is None or nd is None:
            res[i] = (False, 'missing PRF', 0); _stats['malformed'] += 1
        elif nd.startswith('LEANPARSE'):
            res[i] = (False, 'lean_seq parse: ' + nd[10:], 0); _stats['marker_parse'] += 1
        elif nd.startswith('LEANREJ'):
            res[i] = (False, 'lean rejected', 0); _stats['marker_rej'] += 1
        else:
            k = _key(prompt, nd)
            if k in _registry:
                ok = _registry[k]
                res[i] = (ok, '' if ok else 'lean rejected', n_lines(nd) if ok else 0)
                _stats['hit'] += 1
            else:
                todo.setdefault((prompt, nd), []).append(i)
    if todo:
        keys = list(todo)
        srcs = []; idx = []
        for j, (prompt, nd) in enumerate(keys):
            try:
                srcs.append(nd2lean.translate(prompt, nd, require_all_pr=False)); idx.append(j)
            except Exception as e:                        # TranslationError, ParseError, ValueError from parse_prompt
                for i in todo[keys[j]]:
                    res[i] = (False, f'nd2lean: {e}', 0)
                _stats['untranslatable'] += 1
        t0 = time.time()
        ok_list, _, _ = lean_gate_check(srcs)
        _fallback_s[0] += time.time() - t0
        _stats['fallback'] += len(keys)
        for j, ok in zip(idx, ok_list):
            prompt, nd = keys[j]
            register(prompt, nd, ok)
            for i in todo[keys[j]]:
                res[i] = (bool(ok), '' if ok else 'lean rejected', n_lines(nd) if ok else 0)
    return res


def lean_gate_check(srcs):
    from lean_gate import check_sources           # imported lazily: lean_gate imports lean_judge inside gate()
    return check_sources(srcs)


def verify_text(text):
    """Drop-in for nd_verify.verify_text.  One string: prefer judge_many() inside a loop (pitfall 4)."""
    return judge_many([split_text(text)])[0]
