"""Separate frozen-model head: canonical joins and exact language-head preservation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

FOLDER = (
    Path(__file__).resolve().parents[1]
    / "experiments/current/learned_compute_allocation_20261006/selection"
)


def module(name):
    spec = importlib.util.spec_from_file_location("test_" + name, FOLDER / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_frozen_head_labels_join_by_ids_not_private_row_order(tmp_path):
    common = module("frozen_head_common")
    private = tmp_path / "private.jsonl"
    private.write_text(
        json.dumps(
            {
                "theorem_id": "t",
                "labels": [{"candidate_id": i, "valid": i == 3} for i in reversed(range(8))],
            }
        )
        + "\n"
    )
    cache = {"theorem_ids": np.asarray(["t"] * 8), "candidate_ids": np.arange(8)}
    assert common.aligned_labels(cache, private).tolist() == [0, 0, 0, 1, 0, 0, 0, 0]


def test_frozen_head_convex_fit_changes_only_new385_parameters():
    training = module("fit_frozen_head")
    torch.set_num_threads(2)
    features = torch.zeros((32, 384), dtype=torch.float64)
    features[:, 0] = torch.arange(32, dtype=torch.float64) - 15.5
    labels = (features[:, 0] > 0).double()
    before = features.clone()
    weight, bias, diagnostics = training.fit(features, labels)
    assert weight.numel() + bias.numel() == 385
    assert weight.dtype == bias.dtype == torch.float64
    assert torch.equal(features, before)
    assert ((features @ weight + bias > 0).double() == labels).all()
    assert diagnostics["optimizer_iterations"] <= 500
    assert diagnostics["objective"] < 0.6932


def test_original_lm_head_equals_frozen_final_ln_feature_readout():
    checkpoint = (
        Path(__file__).resolve().parents[2]
        / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
    )
    if not checkpoint.exists():
        pytest.skip("Research checkpoint is an untracked artifact")
    extraction = module("extract_frozen_head_features")
    model, tok, _ = extraction.load_ckpt(checkpoint, "cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = extraction.parameter_digest(model)
    ids = torch.tensor([[tok.stoi["<st>"], tok.stoi["<act>"]]])
    with torch.no_grad():
        original = model(ids)[:, -1]
        last_ln = model.ln_f(model.features(ids)[0])[:, -1]
        reconstructed = model.head(last_ln)
    torch.testing.assert_close(original, reconstructed, atol=1e-5, rtol=1e-5)
    assert before == extraction.parameter_digest(model)


def test_frozen_head_test_guard_precedes_model_and_private_labels(tmp_path, monkeypatch):
    evaluator = module("evaluate_frozen_head")
    root = FOLDER.parents[4]
    artifacts = root / "artifacts/learned_compute_allocation_20261006"
    public = tmp_path / "public.jsonl"
    public.write_text(json.dumps({"theorem_id": "guard", "split": "test", "candidates": []}) + "\n")
    missing_private = tmp_path / "must_not_open.jsonl"
    out = tmp_path / "must_not_create"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_frozen_head.py",
            "--head",
            str(artifacts / "selectors/frozen_head_v1/head.pt"),
            "--checkpoint",
            str(artifacts / "checkpoints/la_T1_best12_s0_r16.pt"),
            "--public",
            str(public),
            "--private",
            str(missing_private),
            "--out",
            str(out),
            "--split",
            "test",
        ],
    )
    with pytest.raises(ValueError, match="TEST requires frozen registration"):
        evaluator.main()
    assert not out.exists()
    assert not missing_private.exists()


def test_neutral_overflow_retains_candidates_and_charges_no_forward(tmp_path):
    from types import SimpleNamespace

    inference = module("frozen_head_inference")
    common = module("frozen_head_common")
    public = tmp_path / "public.jsonl"
    row = {
        "theorem_id": "overflow",
        "split": "test",
        "family": "forward",
        "candidates": [{"candidate_id": i} for i in range(8)],
    }
    public.write_text(json.dumps(row) + "\n")
    encoding = SimpleNamespace(
        read=lambda path: [row], model_inputs=lambda *args: [[1] * 8193 for _ in range(8)]
    )
    extraction = SimpleNamespace(encoding=encoding, digest=common.digest)
    metadata = inference.extract(extraction, None, None, str(public), "test", 8, 8192, tmp_path)
    cache, _ = common.read_cache(tmp_path, "test")
    predictions, summary = inference.predict(
        common,
        cache,
        np.asarray([1.0] + [0.0] * 7),
        torch.ones(384, dtype=torch.float64),
        torch.tensor(-3.0, dtype=torch.float64),
    )
    assert metadata["compute"]["trunk_feature_calls"] == 0
    assert summary["overflow_candidates"] == 8
    assert predictions[0]["candidate_validity_probabilities"] == [0.5] * 8
    assert predictions[0]["candidate_validity_logits"] == [0.0] * 8
    assert predictions[0]["candidate_input_tokens"] == [0] * 8
    assert predictions[0]["candidate_serialized_tokens"] == [8193] * 8
    assert predictions[0]["argmax_tie_expected_accuracy"] == 0.125


def test_partial_overflow_is_independent_of_other_candidate_labels():
    inference = module("frozen_head_inference")
    common = module("frozen_head_common")
    cache = {
        "features": np.zeros((8, 384), dtype=np.float32),
        "theorem_ids": np.asarray(["t"] * 8),
        "candidate_ids": np.arange(8),
        "families": np.asarray(["forward"] * 8),
        "input_tokens": np.asarray([9000, 9000] + [10] * 6),
        "overflow": np.asarray([True, True] + [False] * 6),
    }
    weight = torch.zeros(384, dtype=torch.float64)
    bias = torch.tensor(-3.0, dtype=torch.float64)
    left, _ = inference.predict(common, cache, np.asarray([1.0] + [0.0] * 7), weight, bias)
    right, _ = inference.predict(common, cache, np.ones(8), weight, bias)
    assert left[0]["candidate_validity_logits"] == right[0]["candidate_validity_logits"]
    assert left[0]["argmax_tied_candidate_ids"] == [0, 1]
    assert left[0]["argmax_tie_expected_accuracy"] == 0.5
