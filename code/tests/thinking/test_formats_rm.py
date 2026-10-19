"""Formats R (root) and M (memo) end to end, line-boundary budget forcing, and the eval-time think
interventions: training batches, scripted sampling that replays a record, context carrying,
budget forcing, the think ban after ROOT, the context limit, checkpoints with a larger max_len, and
eval dispatch."""

from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import ckpt as C
from code.thinking import eval as evaluation
from code.thinking import rootplan as rp
from code.thinking import train
from code.thinking.data import FORMAT as PAIR_FORMAT
from code.thinking.data import trace_pairs
from code.thinking.sample import (
    LINE_CAP,
    decode_wave,
    intervene_thinks,
    judge,
    sample_format,
    split_lines,
    think_generate,
)
from code.thinking.tok import D, ThinkTokenizer

from ._pipeline_helpers import (
    HELDOUT,
    ScriptedModel,
    isolate,
    k12_head,
    needs_k12,
    needs_lean,
    pilot_rec,
    script_from_pairs,
    tiny_model,
)

T = ThinkTokenizer()
ACT, NL = T.stoi["<act>"], T.stoi["<nl>"]
pytestmark = needs_k12


@functools.cache
def _recs(fmt: str, n: int = 8) -> list[dict[str, Any]]:
    fn = rp.root_trace if fmt == "root" else rp.memo_trace
    return [fn(f"k12_{i}", r["prompt"], r["proof"])[0] for i, r in enumerate(k12_head(n))]


def _rec(fmt: str, min_steps: int = 4) -> dict[str, Any]:
    """A record with a multi-line think at step 0 and at least `min_steps` steps."""
    return next(
        r
        for r in _recs(fmt)
        if len(r["steps"]) >= min_steps and r["steps"][0]["think"].count("\n") >= 2
    )


def _script(fmt: str, enc: list[Any]) -> dict[bytes, list[int]]:
    """A ScriptedModel script that writes an encoded record: R keyed on the whole transcript up to
    each state's `<act>`, M on the state (its continuation: the prompt's tags + previous memo, then
    the target)."""
    script: dict[bytes, list[int]] = {}
    if fmt == "root":
        acc, j = b"", 0
        while j < len(enc):
            acc += enc[j][0]
            j += 1
            cont = b""
            while j < len(enc) and enc[j][1]:
                cont += enc[j][0]
                j += 1
            script[acc] = list(cont)
            acc += cont
        return script
    for p, q, _ in enc:
        state = p[: p.index(bytes([ACT])) + 1]
        script[state] = list(p[len(state) :] + q)
    return script


def _scripted(fmt: str, rec: dict[str, Any]) -> tuple[ScriptedModel, list[Any]]:
    enc = rp.encode(rec, T, fmt)
    return ScriptedModel(T, _script(fmt, enc)), enc


def _gen(model: Any, rec: dict[str, Any], fmt: str, **kw: Any) -> dict[str, Any]:
    kw = {"budgets": 1024, "max_action": 64, "batch": 1, "think_format": fmt, **kw}
    (x,) = think_generate(model, T, [rec["prompt"]], "plain", **kw)
    return x


@pytest.fixture(autouse=True)
def _base0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)  # the records' names (base 0)


# ------------------------------------------------------------------------------------- training
def test_root_batch_masks_match_the_sequence_layout() -> None:
    recs = _recs("root")
    seqs = [rp.encode(r, T, "root") for r in recs]
    data = train.SeqData(seqs, T, "cpu")
    idxs = torch.arange(len(seqs))
    x, m = data.batch(idxs, data.T, T, None, 0.5, shift=False)
    for r, segs in enumerate(seqs):
        ids, mask = rp.flatten(segs)
        think = rp.think_mask(ids, T)
        n = len(ids)
        assert x[r, :n].tolist() == ids and not x[r, n:].any()
        want = [0.5 if t else float(k) for k, t in zip(mask, think, strict=True)]
        assert m[r, :n].tolist() == want and not m[r, n:].any()
        toks = [T.itos[i] for i in ids]
        assert all(m[r, j] == 0 for j, s in enumerate(toks) if s in ("<st>", "<act>"))
    # one name offset per sequence: every name of a row moves by the same amount
    gen = torch.Generator().manual_seed(3)
    xs, _ = data.batch(idxs, data.T, T, gen, 1.0, shift=True)
    for r in range(len(seqs)):
        names = x[r] >= T.ref0
        d = (xs[r] - x[r])[names]
        assert torch.equal(xs[r][~names], x[r][~names]) and len(set(d.tolist())) <= 1
        assert int(xs[r].max()) < T.vocab_size


def test_memo_pairs_train_on_the_target_only() -> None:
    pairs = [rp.encode(r, T, "memo") for r in _recs("memo")]
    data = train.PairData(pairs, "cpu")
    idxs = torch.arange(len(data.lens_cpu))
    x, m = data.batch(idxs, data.T, T, None, 0.25, shift=False)
    flat = [pq for rec in pairs for pq in rec]
    for r, (p, q, tl) in enumerate(flat):
        n = len(p) + len(q)
        assert x[r, :n].tolist() == list(p + q)
        want = [0.0] * len(p) + [0.25] * tl + [1.0] * (len(q) - tl)
        assert m[r, :n].tolist() == want and not m[r, n:].any()
        assert T.itos[q[tl - 1]] == "</think>"  # the weighted part: MEMO_t </think>


def _pickle(tmp: Path, fmt: str) -> Path:
    recs = _recs(fmt)
    data = {
        "format": rp.SEQ_FORMAT if fmt == "root" else PAIR_FORMAT,
        "think_format": fmt,
        "mode": T.mode,
        "itos": list(T.itos),
        "no_trace_mode": "empty",
        "has_traces": True,
        "records": [rp.encode(r, T, fmt) for r in recs],
        "meta": [{"id": r["id"]} for r in recs],
        "config": {},
        "report": {},
    }
    out = tmp / f"T{fmt}.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)
    return out


def test_heldout_uses_the_formats_empty_layout() -> None:
    root = train.heldout_rm(HELDOUT, T, "root", limit=4)
    memo = train.heldout_rm(HELDOUT, T, "memo", limit=4)
    assert len(root) == len(memo) == 4
    for segs in root:
        assert [T.itos[i] for i in segs[1][0]] == ["<think>", "</think>"] and segs[1][1]
        ids, _ = rp.flatten(segs)
        assert ids.count(T.think) == 1
    for rec in memo:
        for p, q, tl in rec:
            assert list(p[-3:]) == [T.think, T.end_think, T.think] and tl == 1
            assert q[0] == T.end_think


@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_train_from_scratch_records_the_format(
    fmt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = isolate(tmp_path, monkeypatch)
    held = tmp / "held.jsonl"
    held.write_text("".join(HELDOUT.read_text().splitlines(keepends=True)[:6]))
    data = _pickle(tmp, fmt)
    out = tmp / "run"
    args = ["--data", str(data), "--heldout", str(held), "--out", str(out), "--best_steps", "3"]
    args += ["--best_dims", "2,32,4,64", "--tok_budget", "8192", "--device", "cpu"]
    args += ["--max_len", "4608"] if fmt == "root" else []
    assert train.main(args) == 0
    model, tok, extra = C.load_ckpt(out / "model.pt")
    assert extra["think_format"] == fmt and model.think_format == fmt and tok.mode == T.mode
    assert sample_format(model) == fmt
    assert model.cfg["max_len"] == (4608 if fmt == "root" else 1024) == extra["max_len"]
    assert extra["steps"] == 3 and extra["final_val"] > 0 and extra["final_val_action"] > 0
    assert extra["examples"] == ("sequences" if fmt == "root" else "pairs")
    if fmt == "root":
        assert extra["records"] == extra["pairs"] == len(_recs("root"))
        with pytest.raises(SystemExit):  # an example over --max_len is an error, not a truncation
            train.main([*args[:-2], "--max_len", "64", "--out", str(tmp / "r2")])


def test_ckpt_roundtrip_with_a_larger_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_len 4,608 survives save / load, and the ALiBi / NoPE model decodes at positions past
    1,024 exactly as its full forward (left-padded rows, KV cache, as `decode_wave` runs it)."""
    isolate(tmp_path, monkeypatch)
    torch.manual_seed(0)
    m = D.best_model.ALiBiGPT(T.vocab_size, 2, 32, 4, max_len=4608, d_ff=64).eval()
    C.save_ckpt(tmp_path / "m.pt", m, T.mode, extra={"think_format": "root"})
    m2, _, extra = C.load_ckpt(tmp_path / "m.pt")
    assert m2.cfg["max_len"] == 4608 and m2.think_format == "root" == extra["think_format"]
    L = 4608
    x = torch.randint(2, T.vocab_size, (1, L))
    P, pad = L - 6, 5
    with torch.no_grad():
        full = m2(x)
        idx = torch.zeros((2, P + pad), dtype=torch.long)
        keep = torch.zeros((2, P + pad), dtype=torch.bool)
        idx[0, pad:], keep[0, pad:] = x[0, :P], True
        idx[1, pad + 700 :], keep[1, pad + 700 :] = x[0, 700:P], True
        ref1 = m2(x[:, 700:])
        pos = (keep.cumsum(1) - 1).clamp(min=0)
        L0 = P + pad
        mask = torch.tril(torch.ones(L0, L0, dtype=torch.bool))[None, None] & keep[:, None, None]
        mask = mask | torch.eye(L0, dtype=torch.bool)[None, None]
        caches: list[dict[str, Any]] = [{"max": L0 + 6} for _ in m2.blocks]
        lg = m2(idx, pos=pos, mask=mask, caches=caches)[:, -1]
        keepb = torch.ones((2, L0 + 6), dtype=torch.bool)
        keepb[:, :L0] = keep
        cur = pos[:, -1]
        for t in range(6):
            assert torch.allclose(lg[0], full[0, P - 1 + t], atol=1e-5)
            assert torch.allclose(lg[1], ref1[0, P - 701 + t], atol=1e-5)
            cur = cur + 1
            nxt = x[0, P + t].repeat(2)[:, None]
            lg = m2(nxt, pos=cur[:, None], mask=keepb[:, None, None, : L0 + t + 1], caches=caches)
            lg = lg[:, -1]
    assert int(cur[0]) == L - 1


# ------------------------------------------------------------------------------------- sampling
@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_scripted_attempt_replays_the_record(fmt: str) -> None:
    for rec in _recs(fmt)[:5]:
        model, enc = _scripted(fmt, rec)
        st: dict[str, Any] = {}
        x = _gen(model, rec, fmt, stats=st)
        assert x["status"] == "done" and x["forced"] == 0 and x["steps"] == len(rec["steps"])
        tr = x["trace"]
        assert len(tr) == len(rec["steps"]) and not any(s.get("rejected") for s in tr)
        if fmt == "root":
            ids, _ = rp.flatten(enc)
            assert list(tr[-1]["prompt"]) + list(tr[-1]["gen"]) == ids  # the record's sequence
            root = list(enc[1][0][1:-1])
            assert list(tr[0]["think"]) == root and all(not s["think"] for s in tr[1:])
            assert x["think"] == len(root) and x["close"] == 1
        else:
            assert [(s["prompt"], s["gen"]) for s in tr] == [(p, q) for p, q, _ in enc]
            assert x["think"] == sum(tl - 1 for _, _, tl in enc) and x["close"] == len(enc)
        assert x["prefill"] == sum(len(s["prompt"]) for s in tr)
        assert x["tokens"] == x["think"] + x["close"] + x["action"] == st["sampled_tokens"]
        assert all(s["applied"] == s["action"] == s["gen"][-len(s["action"]) - 1 : -1] for s in tr)


def test_root_prompts_carry_the_whole_transcript() -> None:
    rec = _rec("root")
    model, _ = _scripted("root", rec)
    tr = _gen(model, rec, "root")["trace"]
    p0 = list(tr[0]["prompt"])
    assert p0 == list(tr[0]["state"]) + [T.think]
    for t in range(1, len(tr)):
        prev = list(tr[t - 1]["prompt"]) + list(tr[t - 1]["gen"])
        assert list(tr[t]["prompt"]) == prev + list(tr[t]["state"])
        for s in tr[:t]:  # every earlier state and action is in the prompt, in order
            assert bytes(s["state"]) in tr[t]["prompt"] and bytes(s["gen"]) in tr[t]["prompt"]


def test_memo_prompts_carry_the_previous_memo() -> None:
    rec = _rec("memo")
    model, _ = _scripted("memo", rec)
    tr = _gen(model, rec, "memo")["trace"]
    prev: list[int] = []
    for s in tr:
        assert list(s["prompt"]) == [*s["state"], T.think, *prev, T.end_think, T.think]
        prev = list(s["think"])
    assert any(s["think"] for s in tr[:-1])


def test_format_dispatches_on_the_model() -> None:
    rec = _rec("memo")
    scripted, enc = _scripted("memo", rec)
    model: Any = scripted
    model.think_format = "memo"  # as ckpt.load_ckpt sets it
    (x,) = think_generate(model, T, [rec["prompt"]], "plain", 1024, max_action=64, batch=1)
    assert [(s["prompt"], s["gen"]) for s in x["trace"]] == [(p, q) for p, q, _ in enc]


@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_budget_zero_is_the_empty_block(fmt: str) -> None:
    rec = _rec(fmt)
    model, _ = _scripted(fmt, rec)
    x = _gen(model, rec, fmt, budgets=0)
    p0 = list(x["trace"][0]["prompt"])
    tail = [T.think, T.end_think] if fmt == "root" else [T.think, T.end_think] * 2
    assert p0[-len(tail) :] == tail and x["think"] == 0 and x["close"] == 0


@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_budget_forcing_cuts_the_block(fmt: str) -> None:
    rec = _rec(fmt)
    model, _ = _scripted(fmt, rec)
    content = [list(rp.think_block(s["think"], rp.td())[1:-1]) for s in rec["steps"]]
    content = [T.encode_toks(c) for c in content]
    B = 3
    x = _gen(model, rec, fmt, budgets=B, force_mode="exact")
    last = x["trace"][-1]  # the cut block, then the script's next think token as the "action"
    t = len(x["trace"]) - 1
    assert x["status"].startswith("fail") and last["rejected"] and last["forced"]
    assert list(last["gen"][:B]) == content[t][:B] and last["gen"][B] == T.end_think
    # line forcing: a block ends at the end of the line that crosses B, never mid-line
    x = _gen(model, rec, fmt, budgets=B, force_mode="line")
    tr = x["trace"]
    assert x["status"].startswith("fail") and tr[-1]["forced"]
    for t, s in enumerate(tr):
        c, n = content[t], len(s["think"])
        assert list(s["think"]) == c[:n] and (n == len(c) or c[n] == NL)
        assert n <= len(c) and (n == len(c) or B <= n < B + LINE_CAP)
    if fmt == "root":
        c = content[0]
        cut = next(j for j in range(B, len(c)) if c[j] == NL)
        assert list(tr[0]["gen"][: cut + 1]) == c[:cut] + [T.end_think] and tr[0]["forced"]
        assert x["think"] == cut


def _line_script(lines: list[int], natural_close: bool = False) -> tuple[list[int], ScriptedModel]:
    """A model whose think block is the given line lengths ("P" tokens, `<nl>` between lines)."""
    content: list[int] = []
    for k, n in enumerate(lines):
        content += ([NL] if k else []) + [T.stoi["P"]] * n
    state = [T.stoi["<st>"], ACT]
    cont = [T.think, *content] + ([T.end_think] if natural_close else [NL]) + [T.eos]
    return content, ScriptedModel(T, {bytes(state): cont})


def test_line_forcing_closes_at_a_line_end_or_the_hard_cap() -> None:
    lines = [3, 5, 1, 7, 2, 90]
    content, model = _line_script(lines)
    prompt = [T.stoi["<st>"], ACT, T.think]
    ends = [j for j, x in enumerate(content) if x == NL]
    for B in range(1, 22):
        w = decode_wave(model, T, [prompt], [B], True, max_action=4, greedy=True)
        ids = w.ids[0]
        j = ids.index(T.end_think)
        assert w.forced[0]
        want = next((e for e in ends if e >= B), None)
        if want is not None and want < B + LINE_CAP:
            assert j == want and ids[:j] == content[:j]  # whole lines, at most one past B
        else:  # the line crossing B is longer than the cap: forced mid-line at B + 64
            assert j == B + LINE_CAP
        w = decode_wave(
            model, T, [prompt], [B], True, max_action=4, greedy=True, force_mode="exact"
        )
        assert w.ids[0].index(T.end_think) == B and w.forced[0]
    b = 30  # inside the 90-token line: the hard cap
    w = decode_wave(model, T, [prompt], [b], True, max_action=4, greedy=True)
    assert w.ids[0].index(T.end_think) == b + LINE_CAP and w.forced[0]
    content, model = _line_script([3, 5], natural_close=True)  # a natural close past B
    w = decode_wave(model, T, [prompt], [2], True, max_action=4, greedy=True)
    assert w.ids[0].index(T.end_think) == 3  # the first line's end: forced
    w = decode_wave(model, T, [prompt], [5], True, max_action=4, greedy=True)
    assert w.ids[0].index(T.end_think) == len(content) and not w.forced[0]  # its own close


class ThinkHungry(ScriptedModel):
    """The scripted model, but after ROOT's `</think>` it prefers every think token above the
    script (bias +60 on `<think> </think> <no>`)."""

    def forward(
        self, idx: torch.Tensor, pos: Any = None, mask: Any = None, caches: Any = None
    ) -> Any:
        out = super().forward(idx, pos, mask, caches)
        c = caches[0]
        for b in range(out.shape[0]):
            h = c["hist"][b, : int(c["len"][b])].tolist()
            if T.end_think in h:
                out[b, 0, [T.think, T.end_think, T.no]] = 60.0
        return out


def test_think_tokens_are_banned_after_root() -> None:
    rec = _rec("root")
    enc = rp.encode(rec, T, "root")
    x = _gen(ThinkHungry(T, _script("root", enc)), rec, "root")
    assert x["status"] == "done"
    ids, _ = rp.flatten(enc)
    assert list(x["trace"][-1]["prompt"]) + list(x["trace"][-1]["gen"]) == ids
    for s in x["trace"][1:]:
        assert not ({T.think, T.end_think, T.no} & set(s["gen"]))


def test_context_limit_ends_the_attempt() -> None:
    rec = _rec("root")
    scripted, enc = _scripted("root", rec)
    model: Any = scripted
    n = len(rp.flatten(enc)[0])
    model.cfg = {"max_len": n}
    assert _gen(model, rec, "root")["status"] == "done"
    model.cfg = {"max_len": n - 1}
    x = _gen(model, rec, "root")
    assert x["status"] == "fail:context" and x["trace"][-1]["rejected"]
    assert x["causes"] == ["context"]
    model.cfg = {"max_len": len(enc[0][0])}  # the first prompt does not fit: no draw at all
    x = _gen(model, rec, "root")
    assert x["status"] == "fail:context" and x["draws"] == 0 and x["trace"] == []


@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_guided_arms_run_in_both_formats(fmt: str) -> None:
    rec = _rec(fmt)
    m = tiny_model(T.vocab_size, seed=4)
    m.cfg["max_len"] = 4608
    st: dict[str, Any] = {}
    res = think_generate(
        m,
        T,
        [rec["prompt"]] * 3,
        "structural",
        budgets=[0, 6],
        max_action=12,
        max_steps=3,
        max_rej=2,
        batch=3,
        stats=st,
        think_format=fmt,
    )
    assert sum(r["tokens"] for r in res) == st["sampled_tokens"]
    for r in res:
        assert r["tokens"] == r["think"] + r["close"] + r["action"]
        assert r["draws"] == r["steps"] + len(r["causes"])


# ---------------------------------------------------------------------------------- interventions
class Echo(ScriptedModel):
    """Per state a scripted think block; after `</think>` the "action" echoes the think content
    the model sees (between the last `<think>` and `</think>`), then `<eos>`."""

    def forward(
        self, idx: torch.Tensor, pos: Any = None, mask: Any = None, caches: Any = None
    ) -> Any:
        out = super().forward(idx, pos, mask, caches)
        c = caches[0]
        for b in range(out.shape[0]):
            h = c["hist"][b, : int(c["len"][b])].tolist()
            j = len(h) - 1 - h[::-1].index(ACT)
            tail = h[j + 1 :]
            if T.end_think in tail:
                k = tail.index(T.end_think)
                seen = tail[1:k]
                done = len(tail) - k - 1
                out[b, 0] = -30.0
                out[b, 0, seen[done] if done < len(seen) else T.eos] = 30.0
        return out


def _echo_case() -> tuple[Echo, list[str], dict[str, list[int]]]:
    """Three theorems, think contents of 3, 7 and 12 tokens (lines of 'P' / 'Q' / 'R')."""
    prompts = [pilot_rec(r)["prompt"] for r in ("k12_37120", "k12_22530", "k12_16472")]
    thinks: dict[str, list[int]] = {}
    script: dict[bytes, list[int]] = {}
    for p, (sym, lines) in zip(prompts, [("P", [3]), ("Q", [2, 4]), ("R", [3, 3, 4])], strict=True):
        content: list[int] = []
        for k, n in enumerate(lines):
            content += ([NL] if k else []) + [T.stoi[sym]] * n
        thinks[p] = content
        state = T.encode_toks(D.state_env.Env(p, canon=True).state_tokens())
        script[bytes(state)] = [T.think, *content, T.end_think]
    return Echo(T, script), prompts, thinks


@pytest.mark.parametrize("kind", ["none", "empty", "filler", "corrupt", "swap"])
def test_interventions_replace_the_block_the_action_sees(kind: str) -> None:
    model, prompts, thinks = _echo_case()
    st: dict[str, Any] = {}
    res = think_generate(
        model,
        T,
        prompts,
        "plain",
        budgets=64,
        max_action=40,
        batch=3,
        stats=st,
        intervene=kind,
    )
    for p, x in zip(prompts, res, strict=True):
        (s,) = x["trace"]  # the echoed "action" is rejected: one draw
        own = thinks[p]
        seen = list(s["gen"][:-1]) if kind != "none" else list(s["gen"][len(own) + 1 : -1])
        assert s["gen"][-1] == T.eos
        if kind == "none":
            assert seen == own and "think_sampled" not in s
            continue
        assert list(s["think_sampled"]) == own and x["think"] == len(own) and x["close"] == 1
        assert x["tokens"] == len(own) + 1 + len(s["gen"])
        assert list(s["prompt"]) == list(s["state"]) + [T.think, *seen, T.end_think]
        assert x["prefill"] == len(s["state"]) + 1 + len(s["prompt"])
        if kind == "empty":
            assert seen == []
        elif kind == "filler":
            assert seen == [T.no] * len(own)
        elif kind == "corrupt":
            lines, got = split_lines(own, NL), split_lines(seen, NL)
            assert len(got) == len(lines) - len(lines) // 2
            assert all(ln in lines for ln in got)
        else:  # swap: the nearest-length block of another theorem
            others = sorted(
                (t for q, t in thinks.items() if q != p), key=lambda t: abs(len(t) - len(own))
            )
            assert seen == others[0] and seen != own
    assert st.get("intervene", {}).get("blocks", 0) == (0 if kind == "none" else 3)
    assert sum(r["tokens"] for r in res) == st["sampled_tokens"]


def test_swap_without_another_theorem_is_empty() -> None:
    st: dict[str, Any] = {}
    out = intervene_thinks("swap", [[5, 6], [7]], ["a", "a"], T, [0, 1], st)
    assert out == [[], []] and st["intervene"]["swap_no_donor"] == 2
    model, prompts, _ = _echo_case()
    with pytest.raises(ValueError):
        think_generate(model, T, prompts, "logical", 8, intervene="empty")


@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_filler_is_carried_in_the_rm_context(fmt: str) -> None:
    rec = _rec(fmt)
    model, enc = _scripted(fmt, rec)
    x = _gen(model, rec, fmt, intervene="filler")
    tr = x["trace"]
    if fmt == "root":  # ROOT replaced by <no> x len; the transcript carries the replaced ROOT
        n = len(enc[1][0]) - 2
        assert list(tr[0]["think"]) == [T.no] * n and list(tr[0]["think_sampled"]) == list(
            enc[1][0][1:-1]
        )
        assert list(tr[1]["prompt"][: len(enc[0][0]) + n + 2]) == [
            *enc[0][0],
            T.think,
            *[T.no] * n,
            T.end_think,
        ]
    else:  # the scripted model writes by position, so every step is accepted
        assert x["status"] == "done"
        prev = 0
        for s in tr:  # the action's input: the previous memo's filler, then this memo's
            n = len(s["think_sampled"])
            want = [*s["state"], T.think, *[T.no] * prev, T.end_think, T.think, *[T.no] * n]
            assert list(s["prompt"]) == [*want, T.end_think] and list(s["think"]) == [T.no] * n
            prev = n


# -------------------------------------------------------------------------------------- eval
@needs_lean
@pytest.mark.parametrize("fmt", ["root", "memo"])
def test_eval_dispatches_on_the_checkpoint_format(
    fmt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = isolate(tmp_path, monkeypatch)
    recs = [r for r in _recs(fmt) if len(r["steps"]) >= 3][:2]
    script: dict[bytes, list[int]] = {}
    for r in recs:
        script.update(_script(fmt, rp.encode(r, T, fmt)))
    model = ScriptedModel(T, script)
    monkeypatch.setattr(
        evaluation, "load_ckpt", lambda path, dev: (model, T, {"think_format": fmt})
    )
    sets = tmp / "set.jsonl"
    sets.write_text(
        "".join(json.dumps({"name": r["id"], "prompt": r["prompt"]}) + "\n" for r in recs)
    )
    out = tmp / "ev"
    a = evaluation.parser().parse_args(
        [
            "--ckpt",
            "x.pt",
            "--sets",
            str(sets),
            "--out",
            str(out),
            "--k",
            "2",
            "--budgets",
            "0,1024",
        ]
        + ["--arms", "plain", "--max_action", "64", "--batch", "4", "--device", "cpu"]
    )
    summ = evaluation.run(a)
    assert summ["layout"] == fmt and summ["think_format"] == fmt and summ["intervene"] == "none"
    assert set(summ["cells"]) == {"plain_B0", "plain_B1024"}
    assert summ["cells"]["plain_B1024"]["all"]["solved"] == 2  # the scripted records
    rows = [json.loads(x) for x in (out / "rows_plain_B1024.jsonl").read_text().splitlines()]
    assert all(r["n_ok"] == 2 and r["intervene"] == "none" for r in rows)
    assert set(rows[0]) >= {"ok", "status", "steps", "tokens", "think", "prefill", "budget"}
    a2 = evaluation.parser().parse_args(
        ["--ckpt", "x.pt", "--sets", str(sets), "--out", str(tmp / "ev2"), "--k", "1"]
        + [
            "--budgets",
            "1024",
            "--arms",
            "plain,logical",
            "--intervene",
            "empty",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(SystemExit):
        evaluation.run(a2)


@needs_lean
def test_eval_records_the_intervention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = isolate(tmp_path, monkeypatch)
    rec = pilot_rec("k12_37120")
    model = ScriptedModel(T, script_from_pairs(trace_pairs(rec, T)))
    monkeypatch.setattr(
        evaluation, "load_ckpt", lambda path, dev: (model, T, {"think_format": "think"})
    )
    sets = tmp / "set.jsonl"
    sets.write_text(json.dumps({"name": "a", "prompt": rec["prompt"]}) + "\n")
    a = evaluation.parser().parse_args(
        ["--ckpt", "x.pt", "--sets", str(sets), "--out", str(tmp / "ev"), "--k", "2"]
        + ["--budgets", "256", "--arms", "plain", "--intervene", "filler", "--device", "cpu"]
        + ["--max_action", "64"]
    )
    summ = evaluation.run(a)
    cell = summ["cells"]["plain_B256"]
    assert summ["intervene"] == "filler" == cell["intervene"]
    assert cell["stats"]["intervene"]["blocks"] >= 2
    rows = [json.loads(x) for x in (tmp / "ev" / "rows_plain_B256.jsonl").read_text().splitlines()]
    assert rows[0]["intervene"] == "filler"
    # the scripted model writes by position: a same-length filler keeps the actions, so it solves
    assert rows[0]["n_ok"] == 2
    res = judge(
        T,
        [rec["prompt"]],
        think_generate(
            model, T, [rec["prompt"]], "plain", 256, max_action=64, batch=1, intervene="empty"
        ),
    )
    assert not res[0]["ok"]  # an empty block shifts the scripted positions: the action breaks
