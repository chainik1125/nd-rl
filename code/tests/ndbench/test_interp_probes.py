"""`interp/probes.py`: dataset building (shapes/labels), the numpy/torch probes (ridge,
multinomial logistic, MDL online code) on synthetic data with a known ground truth, length
generalisation, and an end-to-end `run_probes` smoke test with a tiny CPU model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from code.ndbench.adapters.absgpt import AbsGptAdapter
from code.ndbench.adapters.base import ModelAdapter
from code.ndbench.core.types import Sequent
from code.ndbench.interp import probes
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs
from .test_wp5_fixtures import new_tiny_model, new_tokenizer


def _save_tiny_ckpt(path: Path, *, seed: int) -> None:
    # 4 layers, matching `probes.BLOCK_LAYERS = (0, 1, 2, 3)` (the real models are 4-layer too).
    tok = new_tokenizer()
    model = new_tiny_model(tok.vocab_size, seed=seed, n_layer=4)
    torch.save(
        {
            "state": model.state_dict(),
            "cfg": {"vocab": tok.vocab_size, "n_layer": 4, "d": 32, "n_head": 4},
            "tok_mode": tok.mode,
        },
        path,
    )


def test_rules_vocabulary_has_15_classes() -> None:
    assert len(probes.RULES) == 15
    assert {"PR", "AS", "R", "ANDI", "ORE"} <= set(probes.RULES)


# ------------------------------------------------------------------------------- build_dataset


def test_build_dataset_shapes_and_labels(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    _save_tiny_ckpt(ckpt, seed=0)
    adapter = AbsGptAdapter(ckpt, device="cpu")
    rows = load_valid_proofs()[:5]
    items = [(r["name"], Sequent.from_prompt(r["prompt"]), r["proof"], r["n_lines"]) for r in rows]
    examples = probes.build_dataset(adapter, items)
    assert len(examples) == sum(r["n_lines"] for r in rows)
    for pid, _seq, _body, L in items:
        ex_for_pid = [e for e in examples if e.pid == pid]
        assert len(ex_for_pid) == L
        # lines_to_go counts down to 0 at the last line, in written order
        assert [e.lines_to_go for e in ex_for_pid] == list(range(L - 1, -1, -1))
        assert [e.line_pos for e in ex_for_pid] == list(range(1, L + 1))
        for e in ex_for_pid:
            assert e.next_rule in probes.RULES
            assert set(e.layer_features) == set(probes.LAYER_KEYS)
            for feat in e.layer_features.values():
                assert feat.shape == (32,)
        # the first line of a proof is always PR or AS (never cites anything before it exists)
        assert ex_for_pid[0].next_rule in ("PR", "AS")


def test_build_dataset_cited_back_matches_hand_check(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = Sequent(premises=("P", "Q"), conclusion="( P & Q )")
    body = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED"

    class _StubAdapter:
        # build_dataset only calls token_count() directly; activations() is reached via the
        # monkeypatched full_layer_stack below, so this pure logic/label test needs no real model.
        def token_count(self, seq: Sequent, body: str | None = None) -> int:
            return len(seq.prompt().split())

    def fake_stack(
        adapter: ModelAdapter,
        pairs: list[tuple[Sequent, str]],
        block_layers: tuple[int, ...] = probes.BLOCK_LAYERS,
    ) -> list[Any]:
        del adapter, block_layers
        return [np.zeros((5, len(sq.prompt().split()) + len(b.split()), 4)) for sq, b in pairs]

    monkeypatch.setattr(probes, "full_layer_stack", fake_stack)
    examples = probes.build_dataset(cast(ModelAdapter, _StubAdapter()), [("t1", seq, body, 3)])

    by_pos = {e.line_pos: e for e in examples}
    assert by_pos[1].cited_back is None  # PR, no refs
    assert by_pos[2].cited_back is None  # PR, no refs
    assert by_pos[3].cited_back == 2  # ANDI N1 N2 at idx 3: idx - min(refs) = 3 - 1


# ------------------------------------------------------------------------------- split_by_pid


def test_split_by_pid_disjoint_and_covers() -> None:
    pids = [f"p{i}" for i in range(40)]
    splits = probes.split_by_pid(pids, val_frac=0.2, test_frac=0.2, seed=0)
    assert splits["train"] | splits["val"] | splits["test"] == set(pids)
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])
    assert len(splits["test"]) == 8
    assert len(splits["val"]) == 8


# ----------------------------------------------------------------------------------- ridge


def test_fit_ridge_recovers_linear_relationship() -> None:
    rng = np.random.default_rng(0)
    n, d = 500, 6
    x = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = x @ w_true + 3.0
    w, b = probes.fit_ridge(x, y, l2=0.0)
    pred = probes.predict_ridge(x, w, b)
    assert probes.r2_score(y, pred) > 0.999


def test_probe_regression_synthetic() -> None:
    rng = np.random.default_rng(1)
    d = 5

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d))
        w = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
        y = x @ w
        return x, y

    train, val, test = make(300), make(80), make(80)
    result = probes.probe_regression(train, val, test)
    assert cast(float, result["test_r2"]) > 0.95
    assert 0.0 <= cast(float, result["test_bin_acc"]) <= 1.0


# ---------------------------------------------------------------------- multinomial logistic


def test_fit_multinomial_and_probe_classification_synthetic() -> None:
    rng = np.random.default_rng(2)
    d, n_classes = 4, 3
    w = rng.normal(size=(d, n_classes)) * 3.0  # one shared ground-truth boundary for every split

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d))
        y = (x @ w).argmax(axis=-1)
        return x, y

    train, val, test = make(300), make(60), make(60)
    result = probes.probe_classification(train, val, test, n_classes)
    assert cast(float, result["test_acc"]) > 0.85


# --------------------------------------------------------------------------------------- MDL


def test_mdl_online_code_beats_uniform_on_learnable_data() -> None:
    rng = np.random.default_rng(3)
    d, n_classes, n = 4, 3, 400
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(d, n_classes)) * 4.0
    y = (x @ w).argmax(axis=-1)
    result = probes.mdl_online_code(x, y, n_classes, epochs=60)
    assert cast(float, result["code_length_bits"]) < cast(float, result["uniform_bits"])
    assert cast(float, result["compression"]) > 1.0


def test_mdl_online_code_empty_is_safe() -> None:
    result = probes.mdl_online_code(np.zeros((0, 3)), np.zeros((0,), dtype=np.int64), 3)
    assert result["code_length_bits"] == 0.0


# ------------------------------------------------------------------- length generalisation


def _fake_example(pid: str, feat: np.ndarray, lines_to_go: int) -> probes.ProbeExample:
    feats = {k: feat.astype(np.float32) for k in probes.LAYER_KEYS}
    return probes.ProbeExample(
        pid=pid,
        L=10,
        line_pos=1,
        layer_features=feats,
        next_rule="R",
        lines_to_go=lines_to_go,
        cited_back=None,
    )


def test_run_length_generalisation_reports_r2() -> None:
    rng = np.random.default_rng(4)
    w = rng.normal(size=3)

    def make_examples(prefix: str, n: int) -> list[probes.ProbeExample]:
        out = []
        for i in range(n):
            feat = rng.normal(size=3)
            out.append(_fake_example(f"{prefix}{i}", feat, int(round(float(feat @ w)))))
        return out

    train_ex = make_examples("tr", 60)
    ood_ex = make_examples("ood", 20)
    splits = probes.split_by_pid([e.pid for e in train_ex], seed=0)
    result = probes.run_length_generalisation(train_ex, ood_ex, splits, "0")
    assert result["empty"] is False
    assert cast(float, result["ood_r2"]) > 0.5


def test_run_length_generalisation_empty_when_no_ood() -> None:
    train_ex = [_fake_example("a", np.array([1.0, 0.0, -1.0]), 3)]
    splits = probes.split_by_pid(["a"], seed=0)
    result = probes.run_length_generalisation(train_ex, [], splits, "0")
    assert result["empty"] is True


# ------------------------------------------------------------------------------- run_probes


def test_run_probes_end_to_end_smoke(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    ckpt_base, ckpt_rl = tmp_path / "base.pt", tmp_path / "rl.pt"
    _save_tiny_ckpt(ckpt_base, seed=1)
    _save_tiny_ckpt(ckpt_rl, seed=2)
    base = AbsGptAdapter(ckpt_base, device="cpu")
    rl = AbsGptAdapter(ckpt_rl, device="cpu")

    cfg = {"seed": 0, "n_train": 4, "n_ood": 3, "scan_limit": 20_000}
    summary = probes.run_probes(store, cfg, base_adapter=base, rl_adapter=rl)
    assert summary["n_rows"] == 2 * len(probes.LAYER_KEYS)  # 2 models x 5 layers
    from code.ndbench.interp._store import load_interp_rows

    rows = load_interp_rows(store, "probes", summary["run_id"])
    assert len(rows) == summary["n_rows"]
    for row in rows:
        assert row["model"] in ("base", "rl")
        assert row["layer"] in probes.LAYER_KEYS
        assert "next_rule" in row and "lines_to_go" in row
        assert "length_generalisation" in row


# --------------------------------------------------------------------- success (optional target)


def test_build_success_dataset_and_probe_success(tmp_path: Path) -> None:
    """`success` (INTERP_SPEC.md §4, optional): not wired into `run_probes`' default pipeline (it
    needs a separate per-theorem `p_hat` label source -- see this coder's final report), but the
    two functions it is built from are exercised directly here."""
    ckpt = tmp_path / "m.pt"
    _save_tiny_ckpt(ckpt, seed=0)
    adapter = AbsGptAdapter(ckpt, device="cpu")
    rows = load_valid_proofs()[:20]
    items = [(r["name"], Sequent.from_prompt(r["prompt"]), r["proof"]) for r in rows]
    # alternate "mastered"/"not mastered" labels so both classes are present in every split
    labels = {pid: (1.0 if i % 2 == 0 else 0.0) for i, (pid, _seq, _body) in enumerate(items)}

    dataset = probes.build_success_dataset(adapter, items, labels)
    assert len(dataset) == len(items)
    pids = [d[0] for d in dataset]
    assert set(pids) == {pid for pid, _seq, _body in items}
    for pid, feats, y in dataset:
        assert set(feats) == set(probes.LAYER_KEYS)
        for f in feats.values():
            assert f.shape == (32,)
        assert y == int(labels[pid] >= 0.5)

    splits = probes.split_by_pid(pids, val_frac=0.2, test_frac=0.2, seed=0)
    result = probes.probe_success(dataset, splits, "0")
    assert result is not None
    assert 0.0 <= cast(float, result["test_acc"]) <= 1.0
    assert cast(int, result["n_train"]) > 0 and cast(int, result["n_test"]) > 0


def test_build_success_dataset_skips_unlabelled_pids(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    _save_tiny_ckpt(ckpt, seed=0)
    adapter = AbsGptAdapter(ckpt, device="cpu")
    rows = load_valid_proofs()[:3]
    items = [(r["name"], Sequent.from_prompt(r["prompt"]), r["proof"]) for r in rows]
    labels = {items[0][0]: 0.9}  # only the first pid has a label
    dataset = probes.build_success_dataset(adapter, items, labels)
    assert [d[0] for d in dataset] == [items[0][0]]


def test_probe_success_none_without_both_train_and_test() -> None:
    dataset = [("only_pid", {k: np.zeros(3) for k in probes.LAYER_KEYS}, 1)]
    splits = {"train": {"only_pid"}, "val": set(), "test": set()}
    assert probes.probe_success(dataset, splits, "0") is None


def test_position_baselines_lookup_tables() -> None:
    """Interp audit 2026-10-01: probes need a non-neural, position-only baseline. With every
    proof's rule sequence fixed by line index, the line-index table is perfect while the
    majority class is not."""
    from code.ndbench.core.types import Sequent
    from code.ndbench.interp.probes import position_baselines

    body = "N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; QED"
    items = [(f"p{i}", Sequent(premises=("P",), conclusion="( P v Q )"), body, 2) for i in range(6)]
    splits = {"train": {"p0", "p1", "p2", "p3"}, "val": set(), "test": {"p4", "p5"}}
    out = position_baselines(items, splits)
    assert out["n_test"] == 4
    assert out["next_rule_acc_majority"] == 0.5
    assert out["next_rule_acc_line_index"] == 1.0
    assert out["lines_to_go_r2_line_index"] == 1.0
