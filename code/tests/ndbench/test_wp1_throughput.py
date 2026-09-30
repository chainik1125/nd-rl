"""`tools.throughput` (plan §3 WP1): CPU-only, a tiny random checkpoint, small batches/samples so
the whole file stays well under the file's time budget."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from code.ndbench.adapters.absgpt import GPT, Tokenizer
from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.types import (
    JSON,
    LengthLabel,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.store import Store
from code.ndbench.tools.throughput import throughput


def _make_tiny_checkpoint(tmp_path: Path, name: str = "tiny.pt", seed: int = 0) -> Path:
    torch.manual_seed(seed)
    tok = Tokenizer("abs")
    model = GPT(vocab=tok.vocab_size, n_layer=1, d=32, n_head=4, max_len=128)
    path = tmp_path / name
    torch.save({"state": model.state_dict(), "tok_mode": "abs", "cfg": dict(model.cfg)}, path)
    return path


def test_throughput_synthetic_int_probe_set(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    result = throughput(
        models=[f"tiny={ckpt}"],
        sets=["3"],
        batches=[2, 4],
        device="cpu",
        n_samples=4,
        max_new_tokens=16,
        update_pairs=2,
        update_steps=2,
    )
    assert result["device"] == "cpu"
    rows = cast("list[dict[str, JSON]]", result["results"])
    assert len(rows) == 2  # one per batch
    for row in rows:
        assert row["model"] == "tiny"
        assert row["set"] == "3"
        assert row["n_samples"] == 4
        assert cast(float, row["samples_per_s"]) > 0
        assert cast(float, row["elapsed_s"]) > 0
        # a random 1-layer model will very often emit *some* decodable (if garbage) body, so the
        # learner-timing path should normally run; if it ever doesn't, it must degrade to None
        # rather than raising.
        steps_per_s = row["learner_steps_per_s"]
        assert steps_per_s is None or cast(float, steps_per_s) > 0


def test_throughput_file_probe_set(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    probe_path = tmp_path / "probe.jsonl"
    with open(probe_path, "w") as f:
        f.write(json.dumps({"prompt": "THM P SEQ P PRF"}) + "\n")
        f.write(json.dumps({"prompt": "THM P , Q SEQ P PRF"}) + "\n")

    result = throughput(
        models=[str(ckpt)],  # bare path -> label is the filename stem
        sets=[str(probe_path)],
        batches=[3],
        device="cpu",
        n_samples=2,
        max_new_tokens=8,
    )
    rows = cast("list[dict[str, JSON]]", result["results"])
    assert len(rows) == 1
    assert rows[0]["model"] == ckpt.stem
    assert rows[0]["n_samples"] == 2 * 2  # 2 prompts x 2 samples each
    assert rows[0]["max_new_tokens"] == 8


def test_throughput_two_models_and_two_sets(tmp_path: Path) -> None:
    ckpt_a = _make_tiny_checkpoint(tmp_path, name="a.pt", seed=0)
    ckpt_b = _make_tiny_checkpoint(tmp_path, name="b.pt", seed=1)
    result = throughput(
        models=[f"a={ckpt_a}", f"b={ckpt_b}"],
        sets=["2", "4"],
        batches=[2],
        device="cpu",
        n_samples=2,
        max_new_tokens=8,
    )
    rows = cast("list[dict[str, JSON]]", result["results"])
    assert len(rows) == 4  # 2 models x 2 sets x 1 batch
    labels = {(r["model"], r["set"]) for r in rows}
    assert labels == {("a", "2"), ("a", "4"), ("b", "2"), ("b", "4")}


def _toy_suite_manifest(n: int) -> SuiteManifest:
    return SuiteManifest(
        suite="toy_suite",
        version="v1",
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=n,
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )


def test_throughput_suite_name_probe_set(tmp_path: Path) -> None:
    """Fix list item 9: a `--sets` entry that is neither an existing file nor an integer resolves
    to a suite already built into `store`, sampling its own problems' real prompts and scaling
    `max_new_tokens` off their own certified lengths."""
    ckpt = _make_tiny_checkpoint(tmp_path)
    store = Store(tmp_path / "out")
    problems = [
        Problem(
            pid=f"toy:{i}",
            suite="toy_suite",
            version="v1",
            sequent=Sequent(premises=("P",), conclusion="P" if i == 0 else "( P & P )"),
            length=LengthLabel(L=5 + i, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
            ref=ReferenceProof(body="N1 P : PR ; QED", lines=1, pruned=1, source="constructed"),
            tags={},
        )
        for i in range(2)
    ]
    store.write_suite(problems, _toy_suite_manifest(len(problems)))

    result = throughput(
        models=[f"tiny={ckpt}"],
        sets=["toy_suite"],
        batches=[2],
        device="cpu",
        n_samples=2,
        max_new_tokens=64,
        store=store,
        suite_version="v1",
    )
    rows = cast("list[dict[str, JSON]]", result["results"])
    assert len(rows) == 1
    assert rows[0]["set"] == "toy_suite"
    assert rows[0]["n_samples"] == 2 * 2  # 2 problems x 2 samples each
    # scaled off max(L)=6 (5, 6 across the two problems) -> 6*6=36, within [16, 64]
    assert rows[0]["max_new_tokens"] == 36


def test_throughput_probe_set_neither_file_int_nor_known_suite_raises(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="no `store="):
        throughput(models=[f"tiny={ckpt}"], sets=["not_a_real_thing"], batches=[1], device="cpu")


def test_throughput_probe_set_unbuilt_suite_raises(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    store = Store(tmp_path / "out")  # creates the directory tree; no suite written
    with pytest.raises(ValueError, match="not a file, an integer length, or a suite built"):
        throughput(
            models=[f"tiny={ckpt}"],
            sets=["never_built"],
            batches=[1],
            device="cpu",
            store=store,
        )
