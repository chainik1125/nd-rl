"""CPU smoke test of the whole pipeline: build -> train (and cold start) -> sample -> eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from code.thinking import ckpt as C
from code.thinking import data as dataset
from code.thinking import eval as evaluation
from code.thinking import train
from code.thinking.sample import judge, think_generate
from code.thinking.tok import D

from ._pipeline_helpers import HELDOUT, K12, PILOT, isolate, needs_k12


@pytest.mark.slow
@needs_k12
def test_cpu_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = isolate(tmp_path, monkeypatch)
    plain = tmp / "k12_500.jsonl"
    with open(K12) as f, open(plain, "w") as g:
        for i, line in enumerate(f):
            if i >= 500:
                break
            g.write(line)
    held = tmp / "held.jsonl"
    held.write_text("".join(HELDOUT.read_text().splitlines(keepends=True)[:50]))
    pairs = tmp / "pairs.pkl"
    assert (
        dataset.main(
            [
                "--traces",
                str(PILOT),
                "--plain",
                str(plain),
                "--out",
                str(pairs),
                "--exclude",
                str(held),
                "--no-lean",
            ]
        )
        == 0
    )
    rep = json.loads((tmp / "pairs.pkl.report.json").read_text())
    assert rep["traces_valid"] == 8 and rep["plain_used"] == 500

    out = tmp / "stage1"
    assert (
        train.main(
            [
                "--data",
                str(pairs),
                "--heldout",
                str(held),
                "--out",
                str(out),
                "--best_steps",
                "30",
                "--best_dims",
                "2,64,4,128",
                "--tok_budget",
                "4096",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    model, tok, extra = C.load_ckpt(out / "model.pt")
    assert tok.mode == "lean_thinkn" and extra["steps"] == 30 and extra["think_format"] == "think"
    assert model.cfg["n_layer"] == 2 and model.cfg["d"] == 64
    assert extra["final_val"] > 0 and extra["final_val_action"] > 0
    assert (out / "args.json").exists() and list((out / "registry").glob("*.jsonl"))
    curve = json.loads((out / "train.json").read_text())["curve"]
    assert curve[-1]["step"] == 30

    prompt = json.loads(held.read_text().splitlines()[0])["prompt"]
    st: dict[str, object] = {}
    res = think_generate(
        model,
        tok,
        [prompt],
        "plain",
        budgets=16,
        min_think=1,
        max_action=64,
        max_steps=12,
        batch=1,
        seed=0,
        stats=st,
    )
    res = judge(tok, [prompt], res)
    (r,) = res
    assert (
        r["status"] and r["draws"] >= 1 and r["think"] >= 1 and r["tokens"] == st["sampled_tokens"]
    )
    assert "ok" in r

    ev = tmp / "eval"
    summ = evaluation.run(
        evaluation.parser().parse_args(
            [
                "--ckpt",
                str(out / "model.pt"),
                "--sets",
                str(held),
                "--out",
                str(ev),
                "--k",
                "2",
                "--limit",
                "2",
                "--budgets",
                "0,8",
                "--arms",
                "plain,logical",
                "--max_action",
                "48",
                "--max_steps",
                "8",
                "--batch",
                "4",
                "--device",
                "cpu",
            ]
        )
    )
    assert set(summ["cells"]) == {"plain_B0", "plain_B8", "logical_B0", "logical_B8"}
    row = json.loads((ev / "rows_logical_B8.jsonl").read_text().splitlines()[0])
    assert len(row["status"]) == 2 and len(row["fail_step"]) == 2 and "think" in row["total"]

    # cold start from a (tiny) lean_staten checkpoint
    old = D.lean_tok.LeanTokenizer("lean_staten")
    torch.manual_seed(0)
    small = D.best_model.ALiBiGPT(old.vocab_size, 2, 64, 4, d_ff=128)
    C.save_ckpt(tmp / "staten.pt", small, "lean_staten", extra={"steps": 1})
    cs = tmp / "cold"
    assert (
        train.main(
            [
                "--data",
                str(pairs),
                "--out",
                str(cs),
                "--best_steps",
                "5",
                "--init",
                str(tmp / "staten.pt"),
                "--lr_scale",
                "0.3",
                "--tok_budget",
                "4096",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    m2, t2, e2 = C.load_ckpt(cs / "model.pt")
    assert t2.mode == "lean_thinkn" and e2["init_mode"] == "lean_staten" and m2.cfg["vocab"] == 114
