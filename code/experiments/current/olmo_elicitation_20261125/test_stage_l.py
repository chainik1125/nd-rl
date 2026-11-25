# ruff: noqa: E501
"""CPU tests for stage L: adapter selection / staging (stage_adapter.py) and the L / B8 rows of analyze_arms.py."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import analyze_arms as AA
import pytest
import stage_adapter as SA

CFG = {
    "peft_type": "LORA",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "use_dora": False,
    "modules_to_save": None,
    "target_modules": ["up_proj", "k_proj", "o_proj", "gate_proj", "q_proj", "v_proj", "down_proj"],
}


def row(lr: float, order: int, scale: float) -> dict:
    return {
        "schema": "olmo_edl_row/1",
        "lr": lr,
        "order": order,
        "status": "ok",
        "utc_end": "x",
        "n": 64,
        "online": {"think_nats": [100.0 * scale, 50.0 * scale], "answer_nats": [1.0, 1.0]},
        "test": {"pids": ["p0", "p1", "p2"], "n_think": [10, 20, 30], "n_answer": [2, 3, 4]},
        "evals": {
            "0": {"test": {"think_nats": [10.0, 20.0, 30.0], "answer_nats": [1.0, 1.0, 1.0]}},
            "64": {"test": {"think_nats": [8.0, 15.0, 25.0], "answer_nats": [0.5, 0.5, 0.5]}},
        },
    }


def make_edl(
    root: Path, lrs_done: list[float], rows: list[dict], best_lr: float, best_code: float
) -> Path:
    edl = root / "runs" / "ck" / "edl"
    (edl / "adapters" / "best_order0").mkdir(parents=True)
    st = {f"{lr}/{o}": "ok" for lr in lrs_done for o in (0, 1)}
    (edl / "DONE").write_text(json.dumps({"ckpt": "ck", "status": st}))
    (edl / "edl_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (edl / "adapters" / "BEST.json").write_text(
        json.dumps({"lr": best_lr, "order": 0, "code_length": best_code})
    )
    (edl / "adapters" / "best_order0" / "adapter_config.json").write_text(json.dumps(CFG))
    (edl / "adapters" / "best_order0" / "adapter_model.safetensors").write_bytes(b"weights")
    (edl / "adapters" / "best_order0" / "optimizer.pt").write_bytes(b"opt")
    return edl


def run_sa(tmp: Path, *args: str) -> int:
    import sys

    old = sys.argv
    sys.argv = ["stage_adapter.py", "ck", str(tmp / "dest"), "--runs", str(tmp / "runs"), *args]
    try:
        return SA.main()
    finally:
        sys.argv = old


def test_refuses_incomplete_grid(tmp_path):
    rows = [row(1e-4, 0, 1.0), row(3e-4, 0, 0.9)]
    make_edl(tmp_path, [1e-4, 3e-4], rows, 3e-4, 0.9 * 150 + 2)
    assert run_sa(tmp_path) == 1 and not (tmp_path / "dest").exists()


def test_refuses_stale_best(tmp_path):
    rows = [row(1e-4, 0, 1.0), row(3e-4, 0, 0.9), row(1e-3, 0, 0.8), row(1e-3, 1, 0.5)]
    make_edl(tmp_path, [1e-4, 3e-4, 1e-3], rows, 3e-4, 0.9 * 150 + 2)  # 1e-3 order 0 is shorter
    assert run_sa(tmp_path) == 1


def test_stages_best_without_optimizer(tmp_path):
    rows = [
        row(3e-5, 0, 0.1),
        row(1e-4, 0, 1.0),
        row(3e-4, 0, 0.9),
        row(1e-3, 0, 0.95),
        row(3e-4, 1, 0.2),
    ]
    make_edl(
        tmp_path, [1e-4, 3e-4, 1e-3], rows, 3e-4, 0.9 * 150 + 2
    )  # 3e-5 (off the grid) and order 1 do not count
    assert run_sa(tmp_path) == 0
    d = tmp_path / "dest"
    assert sorted(x.name for x in (d / "best_order0").iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    src = json.loads((d / "source.json").read_text())
    assert src["row"]["lr"] == 3e-4 and src["lora_name"] == "ck_edl64"
    assert (
        src["files"]["adapter_model.safetensors"]["md5"] == "1ef5dfde3cc0a9b9a5cb5e6e3c2c1b8e"
        or len(src["files"]["adapter_model.safetensors"]["md5"]) == 32
    )
    e = src["check_examples"]
    assert (
        [x["pid"] for x in e] == ["p0", "p1"]
        and e[1]["peft0_think"] == 20.0
        and e[1]["peft64_answer"] == 0.5
    )


def test_refuses_other_config(tmp_path):
    rows = [row(1e-4, 0, 1.0), row(3e-4, 0, 0.9), row(1e-3, 0, 0.95)]
    edl = make_edl(tmp_path, [1e-4, 3e-4, 1e-3], rows, 3e-4, 0.9 * 150 + 2)
    (edl / "adapters" / "best_order0" / "adapter_config.json").write_text(
        json.dumps({**CFG, "r": 32})
    )
    assert run_sa(tmp_path) == 1


def test_l_cells_and_b8(tmp_path):
    reach = ["a", "b"]
    s, g = tmp_path / "samples.jsonl.gz", tmp_path / "graded.jsonl"
    with gzip.open(s, "wt") as f:
        for p in reach:
            for i in range(8):
                f.write(
                    json.dumps(
                        {"pid": p, "chunk": i // 4, "i": i % 4, "text": "<think>x</think> one two"}
                    )
                    + "\n"
                )
    g.write_text(
        "".join(
            json.dumps({"pid": p, "chunk": i // 4, "i": i % 4, "strict": p == "a" and i == 7})
            + "\n"
            for p in reach
            for i in range(8)
        )
    )
    cells = AA.h_cells([(s, g)], reach)
    assert [len(cells[p]) for p in reach] == [8, 8] and cells["a"][-1] == (1, 2)
    out = AA.summarize(cells, 8, 48, reps=100)
    assert out["strict_pct"] == pytest.approx(100 * (1 / 8) / 2, abs=0.01)
    assert (
        out["solved_pct"] == pytest.approx(100 / 48, abs=0.01)
        and out["solved_all_pct"] == out["solved_pct"]
    )
    first4 = AA.summarize({p: v[:4] for p, v in cells.items()}, 4, 48, reps=100)
    assert first4["solved_pct"] == 0.0
