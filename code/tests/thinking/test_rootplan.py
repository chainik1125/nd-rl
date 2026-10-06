"""Whole-proof plans (`rootplan`): ROOT and MEMO validate strictly at their states, the actions are
the reference's, the plan covers every branch, the R sequence and M pair layouts, determinism, and
the gen / build CLIs."""

from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path
from typing import Any

from code.thinking import progtrace as pt
from code.thinking import rootplan as rp
from code.thinking.dan import load
from code.thinking.tok import ThinkTokenizer

from ._pipeline_helpers import k12_head, needs_k12

D = load()
N = 300
Rec = tuple[dict[str, Any], list[dict[str, Any]]]


@functools.cache
def _traces(fmt: str, n: int = N, seed: int = 0) -> list[Rec]:
    fn = rp.root_trace if fmt == "root" else rp.memo_trace
    return [fn(f"k12_{i}", r["prompt"], r["proof"], seed=seed) for i, r in enumerate(k12_head(n))]


def _env(prompt: str) -> Any:
    return D.state_env.Env(prompt, canon=True, base=0, assign=False)


def _lines(text: str) -> list[list[str]]:
    T = rp.td()
    return T.split_lines(["<think>", *T.tokens(text), "</think>"]) if text else []


@needs_k12
def test_root_and_memo_validate_strictly_with_reference_actions() -> None:
    T = rp.td()
    rows = k12_head(N)
    for fmt in rp.FORMATS:
        for (rec, infos), r in zip(_traces(fmt), rows, strict=True):
            assert rec["format"] == fmt
            v = T.validate(rec, lean=False, strict=True)
            assert v["ok"], (fmt, rec["id"], v["error"])
            steps, _, _ = T.decompose(r["prompt"], r["proof"], canon=True)
            assert [s["action"] for s in rec["steps"]] == [pt.ascii_line(a) for _, a, _ in steps]
            assert len(infos) == len(steps)
            env = _env(rec["prompt"])
            for t, st in enumerate(rec["steps"]):
                think = ["<think>", *T.tokens(st["think"]), "</think>"]
                act = T.tokens(st["action"])
                assert T.check_step(env, think, act, {}) == []  # strict: no soft finding
                body = len(think) - 2
                if fmt == "root":
                    assert body <= rp.PCfg.root_budget if t == 0 else body == 0
                else:
                    assert body <= rp.PCfg.memo_budget
                assert env.apply(act)[0]
            assert env.done


@needs_k12
def test_plans_are_anchored_at_the_states_goal_and_memo_names_the_next_statement() -> None:
    T = rp.td()
    n_atomic = n_named = 0
    for fmt in rp.FORMATS:
        for rec, _ in _traces(fmt):
            env = _env(rec["prompt"])
            for st in rec["steps"]:
                act = T.tokens(st["action"])
                goal = pt.ascii_line(T.ftoks(env.frames[-1].goal))
                lines = st["think"].split("\n") if st["think"] else []
                if lines:
                    assert lines[0].startswith(f"|- {goal} :="), (rec["id"], lines[0], goal)
                if fmt == "memo" and act[0] == "exact":
                    assert lines == [f"|- {goal} := {act[1]}"]
                if fmt == "memo" and pt.step_kind(act) == "decision":
                    _, f, j = pt.parse_have(act)
                    term = act[j:-1]
                    # a reiteration of a name in scope is that name in the plan, not a line
                    if act[-1] == ";" and term[0] != "Classical.byContradiction" and len(term) > 1:
                        n_atomic += 1
                        want = f"|- {pt.ascii_line(T.ftoks(f))} := {pt.ascii_line(term)}"
                        n_named += any(ln.endswith(want) for ln in lines)
                assert env.apply(act)[0]
    assert n_atomic > 300
    assert n_named / n_atomic > 0.97


def _reachable_formulas(term: Any) -> set[Any]:
    """Formulas of the reference statements a frame term uses that are not names in scope."""
    out: set[Any] = set()
    seen: set[int] = set()

    def go(n: Any) -> None:
        if isinstance(n, pt.Def):
            if id(n) in seen:
                return
            seen.add(id(n))
            u = pt.unreit(n)
            if u.inner[0] != "name":
                out.add(u.formula)
            go(u.inner)
            return
        for c in n[1:]:
            if isinstance(c, pt.Def) or (isinstance(c, tuple) and c and isinstance(c[0], str)):
                go(c)

    go(term)
    return out


@needs_k12
def test_root_covers_every_branch_of_the_reference_and_reuses_its_binder_names() -> None:
    T = rp.td()
    n_full = n_two_branch = 0
    used = total = 0
    for (rec, infos), r in zip(_traces("root"), k12_head(N), strict=True):
        inf = infos[0]
        used += inf["ref_names_used"]
        total += inf["ref_names_total"]
        if inf.get("truncated") or inf.get("omitted"):
            continue
        n_full += 1
        actions, _, pos, _ = rp._replay(r["prompt"], r["proof"])  # pyright: ignore[reportPrivateUsage]
        rec0, idx0 = pos[0]
        want = _reachable_formulas(pt.frame_term(rec0, idx0, {}))
        goals: set[Any] = set()
        for ln in _lines(rec["steps"][0]["think"]):
            if "⊢" in ln and ln[-1] != "<no>":
                goals.add(T.pform(ln, ln.index("⊢") + 1)[0])
        assert want <= goals, (rec["id"], rec["steps"][0]["think"])
        if any("Or.elim" in a for a in actions[1:]):
            n_two_branch += 1
    assert n_full > 0.95 * N
    assert n_two_branch > 20
    assert used / total > 0.6


@needs_k12
def test_memo_follows_the_frames_remaining_plan() -> None:
    """A memo that plans a box with the reference's binder name is followed, inside the box, by
    the box's own plan (anchored at the box's goal, the binder now a name in scope)."""
    T = rp.td()
    n_box = 0
    for rec, _ in _traces("memo", 120):
        env = _env(rec["prompt"])
        steps = rec["steps"]
        for t, st in enumerate(steps[:-1]):
            act = T.tokens(st["action"])
            depth = len(env.frames)
            assert env.apply(act)[0]
            if act[0] == "have" and len(env.frames) > depth and act[act.index(":=") + 1] == "(":
                x = act[act.index(":=") + 4]  # ( fun ( x : A ) => by
                if f"fun ( {x} :" in st["think"]:
                    n_box += 1
                    nxt = steps[t + 1]["think"].split("\n")[0]
                    goal = pt.ascii_line(T.ftoks(env.frames[-1].goal))
                    assert nxt.startswith(f"|- {goal} :="), (rec["id"], nxt)
    assert n_box > 50


def _decode(tok: Any, b: bytes) -> list[str]:
    return [tok.itos[x] for x in b]


@needs_k12
def test_root_sequence_layout() -> None:
    tok = ThinkTokenizer()
    T = rp.td()
    for rec, infos in _traces("root", 60):
        segs = rp.root_sequence(rec, tok)
        n = len(rec["steps"])
        assert len(segs) == 2 * n + 1
        env = _env(rec["prompt"])
        root = rp.think_block(rec["steps"][0]["think"], T)
        k = 0
        for t, st in enumerate(rec["steps"]):
            b, loss = segs[k]
            assert not loss and _decode(tok, b) == env.state_tokens()
            assert _decode(tok, b)[0] == "<st>" and _decode(tok, b)[-1] == "<act>"
            k += 1
            if t == 0:
                b, loss = segs[k]
                assert loss and _decode(tok, b) == root
                k += 1
            b, loss = segs[k]
            act = T.tokens(st["action"])
            assert loss and _decode(tok, b) == [*act, "<eos>"]
            k += 1
            assert env.apply(act)[0]
        ids, mask = rp.flatten(segs)
        assert len(ids) == len(mask) == rp.seq_len(infos)
        toks = [tok.itos[x] for x in ids]
        assert toks.count("<think>") == 1 and toks.count("</think>") == 1  # ROOT once, at step 0
        assert sum(mask) == sum(len(b) for b, x in segs if x)
        assert all(m == 0 for x, m in zip(toks, mask, strict=True) if x in ("<st>", "<act>"))
        tm = rp.think_mask(ids, tok)
        assert sum(tm) == len(root) and all(m for m, x in zip(mask, tm, strict=True) if x)
        emp = rp.root_sequence(rp.empty_record(rec), tok)
        assert _decode(tok, emp[1][0]) == ["<think>", "</think>"]


@needs_k12
def test_memo_pair_layout_chains_the_previous_memo() -> None:
    tok = ThinkTokenizer()
    T = rp.td()
    for rec, infos in _traces("memo", 60):
        pairs = rp.memo_pairs(rec, tok)
        assert [len(p) + len(q) for p, q, _ in pairs] == rp.memo_pair_lens(infos)
        env = _env(rec["prompt"])
        prev: list[str] = []
        for (p, q, tl), st in zip(pairs, rec["steps"], strict=True):
            ps, qs = _decode(tok, p), _decode(tok, q)
            state = env.state_tokens()
            assert ps == [*state, "<think>", *prev, "</think>", "<think>"]
            memo = rp.think_block(st["think"], T)[1:-1]
            act = T.tokens(st["action"])
            assert qs == [*memo, "</think>", *act, "<eos>"]
            assert tl == len(memo) + 1 and qs[tl - 1] == "</think>"
            assert env.apply(act)[0]
            prev = memo
        emp = rp.memo_pairs(rp.empty_record(rec), tok)
        assert all(_decode(tok, p)[-3:] == ["<think>", "</think>", "<think>"] for p, _, _ in emp)
        assert all(_decode(tok, q)[0] == "</think>" and t == 1 for _, q, t in emp)


@needs_k12
def test_generation_is_deterministic_per_seed() -> None:
    for fmt in rp.FORMATS:
        a = [rec for rec, _ in _traces(fmt, 60)]
        fn = rp.root_trace if fmt == "root" else rp.memo_trace
        b = [fn(f"k12_{i}", r["prompt"], r["proof"])[0] for i, r in enumerate(k12_head(60))]
        c = [rec for rec, _ in _traces(fmt, 60, seed=1)]
        assert a == b
        assert a != c


@needs_k12
def test_max_pair_and_sequence_lengths() -> None:
    m = max(max(rp.memo_pair_lens(infos)) for _, infos in _traces("memo"))
    assert m <= 1024
    s = [rp.seq_len(infos) for _, infos in _traces("root")]
    assert max(s) < 4096


def _k12_file(tmp: Path, n: int) -> Path:
    p = tmp / "k12.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in k12_head(n)))
    return p


@needs_k12
def test_gen_and_build_cli(tmp_path: Path) -> None:
    k12 = _k12_file(tmp_path, 50)
    tok = ThinkTokenizer()
    for fmt in rp.FORMATS:
        out = tmp_path / f"{fmt}.jsonl"
        args = ["gen", "--format", fmt, "--source", str(k12), "--limit", "40"]
        assert rp.main([*args, "--workers", "2", "--out", str(out), "--no-exclude"]) == 0
        recs = [json.loads(x) for x in out.read_text().splitlines()]
        stats = json.loads((tmp_path / f"{fmt}.stats.json").read_text())
        assert len(recs) == stats["counts"]["traces"] == 40
        assert stats["config"]["format"] == fmt
        if fmt == "root":
            assert stats["root_tokens_per_proof"]["max"] <= 640
            assert set(stats["sequences_over"]) == {"1024", "2048", "4096"}
        else:
            assert stats["max_pair_tokens"] <= 1024 and stats["pairs_over_1024"] == 0
        pkl = tmp_path / f"{fmt}.pkl"
        args = ["build", "--format", fmt, "--traces", str(out), "--plain", str(k12)]
        args += ["--empty-frac", "0.5", "--out", str(pkl), "--workers", "2"]
        args += ["--exclude", "--extra-exclude"]
        assert rp.main(args) == 0
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        assert data["think_format"] == fmt and data["itos"] == list(tok.itos)
        assert data["format"] == ("thinking-seq/1" if fmt == "root" else "thinking-pairs/1")
        rep = data["report"]
        assert rep["traces_valid"] == 40 and rep["plain_used"] == 10
        assert 5 <= rep["trace_records_emptied"] <= 35
        assert len(data["records"]) == len(data["meta"]) == 50
        srcs = [m["source"] for m in data["meta"]]
        assert srcs.count("plain") == 10 and srcs.count("trace_emptied") > 0
        for recd, m in zip(data["records"], data["meta"], strict=True):
            if fmt == "root":
                think = [tok.itos[x] for x in recd[1][0]]
                assert recd[1][1] and think[0] == "<think>" and think[-1] == "</think>"
                assert (len(think) == 2) == (m["source"] != "trace")
            else:
                empty = all(t == 1 for _, _, t in recd)
                assert empty == (m["source"] != "trace")
        assert (tmp_path / f"{fmt}.pkl.report.json").exists()
