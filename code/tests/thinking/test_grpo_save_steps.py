"""grpo.py --save_steps (2026-10-30, the high-RL low-RL arm): model-only checkpoints after listed
AdamW steps and a per-step compute log (steps.jsonl). It must not change training: with it, the
rollouts, metrics rows, printed lines and the regular checkpoints' weights equal
test_grpo_pilots.py's c4b8548 `plain` digests; only args.json (the new option) differs. The log's
totals must match the metrics rows, and the guarded config (KL anchor, rollbacks) must count anchor
tokens and keep a replayed step's checkpoint apart (_r1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from code.thinking import grpo

from ._grpo_digest import CONFIGS, run_digest, setup_go
from .test_grpo_pilots import EXTRA, GOLDEN


def _weights(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        st = torch.load(str(p), map_location="cpu", weights_only=False)["state"]
        for k in sorted(st):
            h.update(k.encode())
            h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_save_steps_changes_nothing_but_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        CONFIGS, "steps", lambda iso: [*EXTRA["plain"](iso), "--save_steps", "1,2,5"]
    )
    got = run_digest(tmp_path, monkeypatch, "steps")
    same = ("rollouts", "rows", "lines")
    assert {k: got[k] for k in same} == {k: GOLDEN["plain"][k] for k in same}
    assert got["args"] != GOLDEN["plain"]["args"]
    out = next(tmp_path.rglob("g/metrics.jsonl")).parent
    ck = out / "ckpts"
    regular = [p for p in ck.glob("*.pt") if "_st" not in p.name]
    assert _weights(regular) == GOLDEN["plain"]["weights"]
    rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    steps = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines()]
    assert [s["step"] for s in steps] == list(range(1, 1 + len(steps)))
    assert len(steps) == sum(r["steps"] for r in rows) >= 2
    want = sorted(f"g_st{n}.pt" for n in (1, 2, 5) if n <= len(steps))
    assert sorted(p.name for p in ck.glob("*_st*.pt")) == want
    assert [Path(s["ckpt"]).name for s in steps if s["ckpt"]] == want
    assert steps[-1]["trained_tokens"] == sum(r["trained_tokens"] for r in rows)
    upto = [r for r in rows if r["iter"] <= steps[-1]["iter"]]  # (the last may take no step)
    n_att = 2 * 4  # Go: 2 prompts x 4 attempts
    assert steps[-1]["attempts"] == n_att * len(upto)
    samp = sum(round(r["tokens_per_attempt"] * n_att) for r in upto)
    assert steps[-1]["sampled_tokens"] == samp
    assert steps[-1]["anchor_tokens"] == 0  # no anchor in this config
    # the step-2 checkpoint is the model after that step: the iteration's own checkpoint when the
    # step ends an iteration that saves one
    s2 = steps[1]
    if s2["minibatch"] == s2["updates"] and (ck / f"g_it{s2['iter']}.pt").exists():
        assert _weights([ck / "g_st2.pt"]) != ""  # loads
        a = torch.load(str(ck / "g_st2.pt"), map_location="cpu", weights_only=False)["state"]
        b = torch.load(str(ck / f"g_it{s2['iter']}.pt"), map_location="cpu", weights_only=False)
        assert all(torch.equal(a[k], b["state"][k]) for k in a)


def test_save_steps_guarded_counts_the_anchor_and_keeps_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guarded config (KL anchor, rollbacks with the restore check, rolling checkpoints):
    anchor tokens are counted, rolled-back compute stays in the totals, and a step number seen
    again after a rollback gets its own _r1 checkpoint."""
    go = setup_go(tmp_path, monkeypatch)
    argv = [*CONFIGS["guarded"](go.iso), "--save_steps", "1,2,3,4,5,6"]
    rows, out = go(argv)
    steps = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines()]
    assert steps and steps[-1]["anchor_tokens"] > 0
    assert steps[-1]["trained_tokens"] == sum(r["trained_tokens"] for r in rows)
    names = sorted(p.name for p in (out / "ckpts").glob("*_st*.pt"))
    seen = [s["step"] for s in steps]
    dup = {n for n in seen if seen.count(n) > 1}
    for n in set(seen) & set(range(1, 7)):
        assert f"g_st{n}.pt" in names
    for n in dup & set(range(1, 7)):
        assert f"g_st{n}_r1.pt" in names
    if any(r.get("rollbacks") for r in rows):
        assert dup, "a rollback restores the step count: some step number is seen again"


def test_parse_save_steps() -> None:
    assert grpo.parse_save_steps("1, 2,4,") == {1, 2, 4}
    with pytest.raises(AssertionError):
        grpo.parse_save_steps("0,1")
