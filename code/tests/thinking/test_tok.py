"""`lean_thinkn`: vocabulary, round trips, and every path that treats ids >= ref0 as names."""

from __future__ import annotations

import random

import torch

from code.thinking.dan import trace_dsl
from code.thinking.data import trace_pairs
from code.thinking.tok import (
    END_THINK,
    NO,
    THINK,
    D,
    ThinkTokenizer,
    make_tokenizer,
    rebase_names,
    remap_ids,
)

from ._pipeline_helpers import pilot, pilot_rec

OLD = D.lean_tok.LeanTokenizer("lean_staten")


def test_vocabulary_layout() -> None:
    t = ThinkTokenizer()
    assert t.mode == "lean_thinkn" and t.vocab_size == 114 <= 256
    assert (t.stoi[THINK], t.stoi[END_THINK], t.stoi[NO]) == (39, 40, 41)
    assert t.ref0 == 50 == t.stoi["n1"] and OLD.ref0 == 47
    assert t.itos[:39] == OLD.itos[:39]
    assert t.itos[42:] == OLD.itos[39:]
    assert t.canon and t.state_mode and not t.with_history
    assert make_tokenizer("lean_thinkn").itos == t.itos
    assert make_tokenizer("lean_staten").itos == OLD.itos


def test_remap_moves_h_and_n_names_up_by_three() -> None:
    r = remap_ids(OLD, ThinkTokenizer())
    assert r[:39] == list(range(39)) and r[39:] == list(range(42, 114))


def test_ascii_round_trip_on_the_pilot() -> None:
    t = ThinkTokenizer()
    for rec in pilot():
        for st in rec["steps"]:
            for text in (st["think"], st["action"]):
                ids = t.encode_ascii(text)
                assert all(0 <= i < t.vocab_size for i in ids)
                assert [t.itos[i] for i in ids] == trace_dsl().tokens(text)


def test_decode_action_and_proof_decode_use_the_new_ids() -> None:
    t = ThinkTokenizer()
    act = t.encode_toks(["have", "n3", ":", "P", ":=", "h1", ";"]) + [t.eos, t.pad]
    assert t.decode_action(act) == (["have", "n3", ":", "P", ":=", "h1", ";"], True)
    prompt = "THM P SEQ P PRF"
    ids = t.encode_proof("N1 P : PR ; QED")
    assert min(i for i in ids if i not in (t.eos,)) >= 0 and t.ref0 in ids
    assert t.decode(ids).startswith("N1 P : PR")
    assert t.statement(prompt).startswith("theorem t")


def _names(ids: list[int], t: ThinkTokenizer) -> list[int]:
    return [i for i in ids if i >= t.ref0]


def test_shift_pair_shifts_names_only() -> None:
    t = ThinkTokenizer()
    pairs = trace_pairs(pilot_rec("k12_22530"), t)
    rng = random.Random(0)
    for p, q, _ in pairs:
        a, b = t.shift_pair(list(p), list(q), rng)
        for x, y in zip(list(p) + list(q), a + b, strict=True):
            if x < t.ref0:
                assert x == y  # think tokens (39-41) and h1..h8 (42-49) never move
            else:
                assert t.ref0 <= y < t.ref0 + 64


def test_training_make_batch_shifts_names_only() -> None:
    """state_train_best.make_batch (the best recipe's name augmentation) reads tok.ref0."""
    t = ThinkTokenizer()
    recs = [trace_pairs(r, t) for r in pilot()]
    data = [[(p, q) for p, q, _ in r] for r in recs]
    ids, plen, _, lens = D.state_train_best.to_device(data, "cpu")
    gen = torch.Generator().manual_seed(1)
    idxs = torch.arange(ids.shape[0])
    x0, _ = D.state_train_best.make_batch(ids, plen, lens, idxs, ids.shape[1], t, gen, shift=False)
    x1, m = D.state_train_best.make_batch(ids, plen, lens, idxs, ids.shape[1], t, gen, shift=True)
    low = x0 < t.ref0
    assert torch.equal(x0[low], x1[low])
    assert bool((x1[~low] >= t.ref0).all()) and bool((x1[~low] < t.ref0 + 64).all())
    assert bool((x1 - x0)[~low].ne(0).any())  # something was shifted
    assert int(x1.max()) < t.vocab_size


def test_fine_tune_batch_shifts_names_only() -> None:
    """state_train.batch (the expert-iteration fine-tune) shifts through tok.shift_pair."""
    t, t0 = ThinkTokenizer(), ThinkTokenizer()
    t0.shift = False
    recs = [[(p, q) for p, q, _ in trace_pairs(r, t)] for r in pilot()]
    idxs = list(range(len(recs)))
    x0, m0 = D.state_train.batch(recs, idxs, t0, random.Random(3), "cpu")
    x1, m1 = D.state_train.batch(recs, idxs, t, random.Random(3), "cpu")
    low = x0 < t.ref0
    assert torch.equal(m0, m1) and torch.equal(x0[low], x1[low])
    assert bool((x1[~low] >= t.ref0).all()) and int(x1.max()) < t.vocab_size
    assert bool((x1 != x0).any())


def test_sampler_prompt_ids_and_names_from_a_name_base() -> None:
    """The sampler's path: an Env with a random name base -> state ids; names stay >= ref0."""
    t = ThinkTokenizer()
    rec = pilot_rec("k12_37120")
    env = D.state_env.Env(rec["prompt"], canon=True, base=20, assign=True)
    T = trace_dsl()
    for st in rec["steps"][:3]:
        env.apply(T.tokens(st["action"]))
    ids = t.encode_toks(env.state_tokens())
    assert _names(ids, t) and min(_names(ids, t)) == t.stoi["n21"]
    assert [t.itos[i] for i in ids] == env.state_tokens()


def test_rebase_names() -> None:
    t = ThinkTokenizer()
    ids = [t.stoi["n21"], t.stoi[THINK], t.stoi["h1"], t.stoi["n25"]]
    assert rebase_names(ids, t, 20) == [t.stoi["n1"], t.stoi[THINK], t.stoi["h1"], t.stoi["n5"]]
    assert rebase_names([t.stoi["n3"]], t, 5) is None
