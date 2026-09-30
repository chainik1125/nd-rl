# imports the fork (dan_best-state), which is not resolvable here
# pyright: reportMissingImports=false
"""(state, action) pairs: the only new data path the recipes need for the state arms.

harness_state.py decomposes every record once, before pretrain() starts its clock (Dan's port
does the same: the decomposition is the format's cost, not the recipe's), and stores the pairs
on the record as r["pairs"]. A recipe then swaps two lines:

  data = pairs(train_recs, tok)            instead of [(encode_prompt, encode_proof) per record]
  batch(data, idxs, tok, rng, dev)         instead of the fork's train.batch (recipes 201/204)

and shifts names in both halves of a pair (Dan's tok.shift_pair: one lean_seq name offset per pair).
"""

from lean_tok import ParseFail  # fork
from state_env import decompose  # fork
from train import batch as proof_batch  # fork

STATE_MODES = ("lean_state", "lean_stateh", "lean_staten")


def encode(rec, tok):
    """record -> [(state ids, action ids + eos)], Dan's state_train.load for one record."""
    steps, _, _ = decompose(rec["prompt"], rec["proof"], canon=tok.canon)
    return [
        (tok.encode_toks(hs + st if tok.with_history else st), tok.encode_toks(act) + [tok.eos])
        for st, act, hs in steps
    ]


def attach(recs, tok):
    """r["pairs"] on every record that decomposes; returns the number skipped (as Dan's load)."""
    skipped = 0
    for r in recs:
        try:
            r["pairs"] = encode(r, tok)
        except (ParseFail, ValueError, AssertionError, IndexError):
            r["pairs"] = []
            skipped += 1
    return skipped


def pairs(recs, tok):
    return [pq for r in recs for pq in r["pairs"]]


def batch(data, idxs, tok, rng, dev):
    """The fork's train.batch, except a state-mode pair gets one name offset on both halves."""
    if tok.mode not in STATE_MODES:
        return proof_batch(data, idxs, tok, rng, dev)
    shifted = [tok.shift_pair(list(data[i][0]), list(data[i][1]), rng) for i in idxs]
    return proof_batch(shifted, range(len(shifted)), _NoShift(tok), rng, dev)


class _NoShift:
    """tok with shift_abs as the identity, so proof_batch pads and masks already-shifted pairs."""

    def __init__(self, tok):
        self.pad = tok.pad

    def shift_abs(self, ids, rng):
        return ids
