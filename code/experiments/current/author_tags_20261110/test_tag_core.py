"""Tests of the tag plumbing (tag_core.py) on every code path a state takes to the model. CPU only, tiny models.

    cd author_tags_20261110 && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
        $PY -m pytest --confcutdir=. -p no:cacheprovider -q test_tag_core.py

Paths covered: the tokenizer itself; Stage-1 pairs (tag_pretrain.tag_load, Dan's state_train); Dan's sampler
(state_sample.env_generate -> prompt_ids); the propensity read's samplers and scorer (factor_lib.run_env,
pc_lib.first_move_rows); GRPO's sampler (code.thinking.sample.think_generate, plain and logical arms) and its canary;
checkpoints saved by Dan's save_ckpt loading in both Dan's and Leon's loaders; name shifts.
"""

# ruff: noqa: E501
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

os.environ.update(ND_OFFLINE="1", ND_REGISTRY="0", ND_REGISTRY_SYNC="0", LEAN_GATE_WORKERS="8")
os.environ.pop(T.ENV, None)
PC_DIR = Path(
    "/scratch/cowifornia/nd-rl/worktrees/propensity-capability/code/experiments/current/propensity_capability_20261109"
)


@pytest.fixture(autouse=True)
def _no_process_tag():
    os.environ.pop(T.ENV, None)
    yield
    os.environ.pop(T.ENV, None)


@pytest.fixture(scope="module")
def dan():
    return T.install_dan()


@pytest.fixture(scope="module")
def records() -> list[dict]:
    out = []
    with gzip.open(T.CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= 120:
                break
    return out


def _dose_record() -> dict:
    with gzip.open(T.CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("coe_dose"):
                return r
    raise AssertionError("no dose record")


# ------------------------------------------------------------------------------------------------ tokenizer
def test_vocabulary_layout(dan) -> None:
    import lean_tok  # type: ignore[import-not-found]

    old = lean_tok.LeanTokenizer("lean_staten")
    new = dan()
    assert old.vocab_size == 111 and new.vocab_size == T.VOCAB == 113
    assert new.itos[:39] == old.itos[:39]
    assert new.itos[39:41] == ["<tagA>", "<tagB>"]
    assert new.itos[41:] == old.itos[39:]
    assert new.ref0 == T.REF0 == 49 and new.itos[new.ref0] == "n1" and new.itos[-1] == "n64"
    assert new.mode == T.MODE and new.canon and new.state_mode and not new.with_history


def test_states_get_the_tag_actions_do_not(dan, records) -> None:
    import lean_tok  # type: ignore[import-not-found]
    from state_env import decompose  # type: ignore[import-not-found]

    old, new = lean_tok.LeanTokenizer("lean_staten"), dan()
    r = records[0]
    steps, _, _ = decompose(r["prompt"], r["proof"], canon=True)
    for tag in T.READ_TAGS:
        new.author_tag = tag
        for st, act, _ in steps:
            want = [new.stoi[s] for s in T.tag_state(st, tag)]
            assert new.encode_toks(st) == want
            assert [new.itos[i] for i in new.encode_toks(act)] == [
                old.itos[i] for i in old.encode_toks(act)
            ]
            assert new.state_tag(new.encode_toks(st)) == tag
            if tag == "none":  # the untagged state is lean_staten's, token for token
                assert [new.itos[i] for i in new.encode_toks(st)] == [
                    old.itos[i] for i in old.encode_toks(st)
                ]


def test_errors_fail_closed(dan) -> None:
    tok = dan()
    st = ["<st>", "⊢", "P", "<act>"]
    with pytest.raises(T.TagError):
        tok.encode_toks(st)  # no tag declared anywhere
    tok.author_tag = "A"
    with pytest.raises(T.TagError):
        tok.encode_toks(["<st>", "<act>", "⊢", "P", "<act>"])  # <act> mid-list
    with pytest.raises(T.TagError):
        tok.encode_toks(["<st>", "⊢", "P", "<act>", "have"])  # state + action in one list
    with pytest.raises(T.TagError):
        tok.encode_toks(["<st>", "⊢", "P", "<tagB>", "<act>"])  # already tagged
    tok.author_tag = "C"
    with pytest.raises(T.TagError):
        tok.encode_toks(st)
    tok.author_tag = None
    os.environ[T.ENV] = "B"
    assert tok.state_tag(tok.encode_toks(st)) == "B"
    tok.author_tag = "A"  # the per-record tag wins over the process's
    assert tok.state_tag(tok.encode_toks(st)) == "A"


def test_name_shifts_never_move_a_tag(dan, records) -> None:
    import random

    import torch
    from state_env import decompose  # type: ignore[import-not-found]
    from state_train_best import make_batch  # type: ignore[import-not-found]

    tok = dan()
    tok.author_tag = "B"
    r = records[3]
    steps, _, _ = decompose(r["prompt"], r["proof"], canon=True)
    rng = random.Random(0)
    for st, act, _ in steps:
        p, q = tok.encode_toks(st), tok.encode_toks(act)
        for _ in range(20):
            p2, _ = tok.shift_pair(p, q, rng)
            assert p2[-2] == tok.tag_ids["B"] and p2[-1] == tok.act_id
    seqs = [tok.encode_toks(st) + tok.encode_toks(a) for st, a, _ in steps]
    Tm = max(map(len, seqs))
    ids = torch.zeros((len(seqs), Tm), dtype=torch.uint8)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    plen = torch.tensor([len(tok.encode_toks(st)) for st, _, _ in steps])
    lens = torch.tensor(list(map(len, seqs)))
    g = torch.Generator().manual_seed(0)
    x, _ = make_batch(ids, plen, lens, torch.arange(len(seqs)), Tm, tok, g)
    for i in range(len(seqs)):
        assert int(x[i, plen[i] - 2]) == tok.tag_ids["B"]


# ------------------------------------------------------------------------------------------------ installs
def test_install_dan_dispatch(dan) -> None:
    import lean_tok  # type: ignore[import-not-found]
    import state_train  # type: ignore[import-not-found]
    import tokenizer  # type: ignore[import-not-found]

    assert type(tokenizer.make_tokenizer(T.MODE)) is dan
    assert type(tokenizer.make_tokenizer("lean_staten")) is lean_tok.LeanTokenizer
    assert state_train.make_tokenizer is tokenizer.make_tokenizer
    assert T.install_dan() is dan  # idempotent


def test_install_thinking_dispatch() -> None:
    cls = T.install_thinking()
    from code.thinking import ckpt as tckpt  # type: ignore[import-not-found]
    from code.thinking import tok as ttok  # type: ignore[import-not-found]

    assert type(ttok.make_tokenizer(T.MODE)) is cls
    assert issubclass(cls, ttok.D.lean_tok.LeanTokenizer)
    assert type(ttok.make_tokenizer("lean_staten")) is ttok.D.lean_tok.LeanTokenizer
    assert type(ttok.make_tokenizer("lean_thinkn")) is ttok.ThinkTokenizer
    assert tckpt.make_tokenizer is ttok.make_tokenizer
    assert T.install_thinking() is cls


def test_dose1_steps_are_his() -> None:
    import torch

    for s, n in T.DOSE1_STEPS.items():
        ck = torch.load(
            str(T.KD / "bases" / f"stage1_dose1_s{s}.pt"), map_location="cpu", weights_only=False
        )
        assert ck["extra"]["steps"] == n and ck["tok_mode"] == "lean_staten"


# ------------------------------------------------------------------------------------------------ Stage-1 pairs
def test_stage1_pairs_carry_their_record_tag(dan, tmp_path) -> None:
    import state_train  # type: ignore[import-not-found]
    import tag_pretrain as TP

    tok = dan()
    src = tmp_path / "mini.jsonl.gz"
    with gzip.open(T.CORPUS, "rt", encoding="utf-8") as f:
        lines = [next(f) for _ in range(150)]
    lines.append(json.dumps(_dose_record()) + "\n")
    with gzip.open(src, "wt", encoding="utf-8") as g:
        g.writelines(lines)
    TP._cfg.update(data=str(src), limit=0)
    data, skipped = TP.tag_load(str(src), tok, 12)
    assert skipped == 0 and len(data) == len(lines)
    for line, pairs in zip(lines, data, strict=True):
        r = json.loads(line)
        for p, q in pairs:
            p, q = list(p), list(q)
            assert tok.state_tag(p) == r["author_tag"]
            assert not set(q) & set(tok.tag_ids.values()) and tok.act_id not in q
    assert tok.author_tag is None  # reset after loading
    assert TP._cfg["pairs"]["train"]["A"] > 0 and TP._cfg["pairs"]["train"]["B"] > 0
    # the same pairs as Dan's own (untagged) loader once the tag is dropped: decompose, canon, names all his
    assert state_train.load is not TP.tag_load  # tag_pretrain.main() was not run in this process
    old = state_train.make_tokenizer("lean_staten")
    plain = tmp_path / "plain.jsonl"
    plain.write_text("".join(lines))
    ref, _ = state_train.load(str(plain), old, 12)
    for rp, tp in zip(ref, data, strict=True):
        assert len(rp) == len(tp)
        for (p0, q0), (p1, q1) in zip(rp, tp, strict=True):
            assert [old.itos[i] for i in p0] == [
                tok.itos[i] for i in p1 if i not in tok.tag_ids.values()
            ]
            assert [old.itos[i] for i in q0] == [tok.itos[i] for i in q1]


# ------------------------------------------------------------------------------------------------ checkpoints
@pytest.fixture(scope="module")
def tiny_ckpt(dan, tmp_path_factory) -> Path:
    import best_model as BM  # type: ignore[import-not-found]
    import torch
    from model import save_ckpt  # type: ignore[import-not-found]

    torch.manual_seed(0)
    tok = dan()
    m = BM.ALiBiGPT(tok.vocab_size, 1, 32, 2, d_ff=64)
    p = tmp_path_factory.mktemp("ck") / "tiny_tag.pt"
    save_ckpt(str(p), m, tok.mode, extra={"test": True})
    return p


def test_ckpt_loads_in_both_loaders(tiny_ckpt, dan) -> None:
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]

    m1, t1, _ = load_ckpt(str(tiny_ckpt), "cpu")
    assert type(t1) is dan and m1.cfg["vocab"] == 113
    cls = T.install_thinking()
    from code.thinking.ckpt import load_ckpt as tload  # type: ignore[import-not-found]

    m2, t2, _ = tload(tiny_ckpt, "cpu")
    assert type(t2) is cls and t2.itos == t1.itos
    x = torch.tensor([[1, 2, 39, 40, 41, 49]])
    assert torch.allclose(m1(x), m2(x))
    assert T.ckpt_mode(tiny_ckpt) == T.MODE


# ------------------------------------------------------------------------------------------------ Dan's samplers
def test_dan_env_generate_carries_the_tag(tiny_ckpt, records) -> None:
    from model import load_ckpt  # type: ignore[import-not-found]
    from state_sample import env_generate  # type: ignore[import-not-found]

    model, tok, _ = load_ckpt(str(tiny_ckpt), "cpu")
    prompts = [r["prompt"] for r in records[:6]]
    for tag in T.READ_TAGS:
        T.set_process_tag(tag)
        T.spy_dan(tag)
        n0, b0 = T.SPY["states"], dict(T.SPY["by_tag"])
        env_generate(model, tok, prompts, greedy=False, temperature=0.8, max_action=8, max_steps=3, batch=4,
                     seed=1, stats={}, gate=False)  # fmt: skip
        assert T.SPY["states"] > n0 and T.SPY["by_tag"].get(tag, 0) > b0.get(tag, 0)
    T.set_process_tag("A")
    T.spy_dan("B")  # a run declared B whose states say A: the spy stops it
    with pytest.raises(T.TagError):
        env_generate(
            model, tok, prompts[:1], greedy=True, max_action=4, max_steps=1, batch=1, gate=False
        )


def test_propensity_read_paths_carry_the_tag(tiny_ckpt, records) -> None:
    if str(PC_DIR) not in sys.path:
        sys.path.append(str(PC_DIR))
    import pc_lib  # type: ignore[import-not-found]
    from model import load_ckpt  # type: ignore[import-not-found]

    model, tok, _ = load_ckpt(str(tiny_ckpt), "cpu")
    thm = next(r for r in records if r["n_prem"] >= 1)
    for tag in T.READ_TAGS:
        T.set_process_tag(tag)
        T.spy_dan(tag)
        n0 = T.SPY["states"]
        items = [
            {
                "name": "x",
                "prompt": thm["prompt"],
                "prefix": pc_lib.forced_prefix(thm["prompt"], "reductio"),
            }
        ]
        pc_lib.run_env(model, tok, items * 2, 0.8, 8, 4, 2, 3, gate=False)
        pc_lib.run_env(
            model, tok, [{"name": "x", "prompt": thm["prompt"]}] * 2, 0.8, 8, 4, 2, 3, gate=False
        )
        assert T.SPY["states"] > n0
        rows, _, labels = pc_lib.first_move_rows(tok, thm["prompt"], 5, ["reductio"])
        for st, _ in rows:
            assert tok.state_tag(st) == tag


# ------------------------------------------------------------------------------------------------ GRPO
@pytest.mark.parametrize("arm", ["plain", "logical"])
def test_grpo_sampler_and_traces_carry_the_tag(tiny_ckpt, records, arm) -> None:
    T.install_thinking()
    from code.thinking.ckpt import load_ckpt as tload  # type: ignore[import-not-found]
    from code.thinking.sample import think_generate  # type: ignore[import-not-found]

    model, tok, _ = tload(tiny_ckpt, "cpu")
    prompts = [r["prompt"] for r in records[:4]]
    for tag in T.READ_TAGS:
        T.set_process_tag(tag)
        T.spy_thinking(tag)
        n0 = T.SPY["states"]
        res = think_generate(model, tok, prompts, arm=arm, budgets=0, temperature=1.0, max_action=8, max_steps=3,
                             max_rej=0, batch=4, seed=2, stats={})  # fmt: skip
        assert T.SPY["states"] > n0
        n_tr = 0
        for r in res:
            for s in r["trace"]:
                assert tok.state_tag(list(s["prompt"])) == tag  # what the update trains on
                n_tr += 1
        assert n_tr > 0


def test_grpo_canary_carries_the_tag(tiny_ckpt) -> None:
    T.install_thinking()
    from code.thinking import grpo as tgrpo  # type: ignore[import-not-found]
    from code.thinking.ckpt import load_ckpt as tload  # type: ignore[import-not-found]

    _, tok, _ = tload(tiny_ckpt, "cpu")
    for tag in T.READ_TAGS:
        T.set_process_tag(tag)
        T.spy_thinking(tag)
        c0 = T.SPY["canary"]
        cd = tgrpo.load_canary(str(T.KD / "pools" / "canary_long.jsonl"), tok, False, 40)
        assert len(cd.canary) == 40 and T.SPY["canary"] > c0
        assert all(tok.state_tag(p) == tag for p, _ in cd.canary)
