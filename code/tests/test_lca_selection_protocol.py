"""Regression checks for candidate visibility, transition rendering and checkpoint storage."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "code/experiments/current/learned_compute_allocation_20261006/selection"
UPSTREAM = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
pytestmark = pytest.mark.skipif(
    not UPSTREAM.exists(), reason="requires pinned sprint source artifact"
)


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"lca_{name}", SELECTION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_and_tokenizer(module, final="exact n1"):
    tokenizer = module.make_tokenizer("lean_staten")
    actions = ["have n7 : P := h1 ;", final]
    row = {
        "prompt": "THM P SEQ P PRF",
        "candidates": [
            {
                "canonical_name_base": 0,
                "raw_actions": [
                    tokenizer.encode_toks(action.split()) + [tokenizer.eos] for action in actions
                ],
            }
        ],
    }
    return row, tokenizer


def test_render_retains_implicit_case_branch_binder():
    module = load_module("render_transitions")
    from state_env import decompose

    tokenizer = module.make_tokenizer("lean_staten")
    prompt = "THM ( P v Q ) , ( P > R ) , ( Q > R ) SEQ R PRF"
    body = (
        "N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 ( Q > R ) : PR ; "
        "N4 | P : AS ; N5 | R : IMPE N2 N4 ; "
        "N6 | Q : AS ; N7 | R : IMPE N3 N6 ; "
        "N8 R : ORE N1 N4 N5 N6 N7 ; QED"
    )
    steps, expected, _ = decompose(prompt, body, canon=True)
    row = {
        "prompt": prompt,
        "candidates": [
            {
                "canonical_name_base": 0,
                "raw_actions": [
                    tokenizer.encode_toks(action) + [tokenizer.eos] for _, action, _ in steps
                ],
            }
        ],
    }
    module.render(row, tokenizer)
    candidate = row["candidates"][0]
    assert candidate["transition_text"] == tokenizer.text(expected)
    assert "=> by" in candidate["transition_text"]
    assert candidate["transition_tokens"][-1] == tokenizer.eos


def test_render_keeps_failed_action_without_verdict_information():
    module = load_module("render_transitions")
    row, tokenizer = row_and_tokenizer(module, "exact n2")
    alternate = copy.deepcopy(row)
    alternate["candidates"][0]["valid"] = True
    module.render(row, tokenizer)
    module.render(alternate, tokenizer)
    assert row["candidates"][0]["transition_text"].endswith("exact n2")
    assert (
        row["candidates"][0]["transition_tokens"] == alternate["candidates"][0]["transition_tokens"]
    )
    assert "LEANPARSE" not in row["candidates"][0]["transition_text"]


def test_render_rejects_trailing_actions_after_completion():
    module = load_module("render_transitions")
    row, tokenizer = row_and_tokenizer(module)
    row["candidates"][0]["raw_actions"].append(
        tokenizer.encode_toks(["exact", "n1"]) + [tokenizer.eos]
    )
    with pytest.raises(ValueError, match="extra action"):
        module.render(row, tokenizer)


def test_trim_preserves_sampled_padding_before_eos():
    module = load_module("collect")
    tokenizer = type("Tokens", (), {"pad": 0, "eos": 1})()
    assert module.trim([9, 0, 7, 1, 0, 0], tokenizer) == [9, 0, 7, 1]
    assert module.trim([9, 0, 7], tokenizer) == [9, 0, 7]


def test_local_checkpoint_metadata_supports_weights_only_reload(tmp_path):
    module = load_module("train")
    model = torch.nn.Linear(2, 2)
    model.cfg = {"test": True}
    path = tmp_path / "local.pt"
    module.save_ckpt(path, model, "test", {"torch_version": torch.__version__})
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["extra"] == json.loads(json.dumps({"torch_version": torch.__version__}))


@pytest.mark.parametrize("context", [1, 10000])
def test_selector_ties_and_overflow_are_uniform_and_labels_do_not_affect_scores(
    monkeypatch, context
):
    from types import SimpleNamespace

    module = load_module("train")
    from tokenizer import make_tokenizer

    tokenizer = make_tokenizer("lean_staten")
    rows = [
        {
            "theorem_id": str(i),
            "prompt": "THM P SEQ P PRF",
            "candidates": [{"effective_tokens": [tokenizer.eos]} for _ in range(8)],
        }
        for i in range(16)
    ]
    monkeypatch.setattr(
        module,
        "chooser_score",
        lambda model, tok, groups, architecture: torch.zeros(len(groups), 8),
    )
    model = SimpleNamespace(eval=lambda: None)
    args = SimpleNamespace(view="actions", architecture="pooled", context=context)
    labels = {row["theorem_id"]: [1, 0, 0, 0, 0, 0, 0, 0] for row in rows}
    prediction = module.evaluate(model, tokenizer, rows, labels, args, seed=77106)
    changed = module.evaluate(
        model, tokenizer, rows, {k: [1] * 8 for k in labels}, args, seed=77106
    )
    assert all(row["probabilities"] == [1 / 8] * 8 for row in prediction)
    assert all(row["argmax_tie_expected_accuracy"] == 1 / 8 for row in prediction)
    assert len({row["selected_candidate_id"] for row in prediction}) > 1
    assert [r["probabilities"] for r in prediction] == [r["probabilities"] for r in changed]
    assert [r["selected_candidate_id"] for r in prediction] == [
        r["selected_candidate_id"] for r in changed
    ]


def test_binary_critic_maps_raw_scores_and_overflow_independently(monkeypatch):
    from types import SimpleNamespace

    module = load_module("train_bce")
    from tokenizer import make_tokenizer

    tok = make_tokenizer("lean_staten")
    row = {
        "theorem_id": "binary",
        "prompt": "THM P SEQ P PRF",
        "candidates": [{"effective_tokens": [i]} for i in range(8)],
    }
    row["candidates"][7]["effective_tokens"] = [7] * 200
    monkeypatch.setattr(
        module,
        "score",
        lambda model, tok, seqs: torch.tensor([[float(s[-3])] + [0.0] * 7 for s in seqs]),
    )
    args = SimpleNamespace(view="actions", architecture="factorized", context=100)
    labels = {"binary": [0] * 7 + [1]}
    result = module.evaluate(SimpleNamespace(eval=lambda: None), tok, [row], labels, args, 77106)[0]
    assert result["candidate_validity_logits"] == [0, 1, 2, 3, 4, 5, 6, 0]
    assert result["candidate_overflow"] == [False] * 7 + [True]
    assert result["candidate_validity_probabilities"][7] == 0.5
    assert result["selected_candidate_id"] == 6
    changed = module.evaluate(
        SimpleNamespace(eval=lambda: None), tok, [row], {"binary": [1] * 8}, args, 77106
    )[0]
    assert result["candidate_validity_probabilities"] == changed["candidate_validity_probabilities"]
    assert result["binary_logloss"] != changed["binary_logloss"]


@pytest.mark.parametrize("status", ["draft", "frozen"])
def test_test_evaluator_rejects_unregistered_method_before_private_read(
    monkeypatch, tmp_path, status
):
    import hashlib
    import sys

    module = load_module("evaluate")
    public = tmp_path / "public.jsonl"
    public.write_text(
        json.dumps(
            {
                "theorem_id": "test",
                "split": "test",
                "prompt": "THM P SEQ P PRF",
                "candidates": [{"candidate_id": i} for i in range(8)],
            }
        )
        + "\n"
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not loaded when protocol is unregistered")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": status,
                "selection_split": "dev",
                "registered_public_pool_sha256": [hashlib.sha256(public.read_bytes()).hexdigest()],
                "registered_analyses": [],
            }
        )
    )
    private = tmp_path / "PRIVATE_MUST_NOT_BE_READ.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--public",
            str(public),
            "--private",
            str(private),
            "--split",
            "test",
            "--out",
            str(tmp_path / "out"),
            "--frozen-test-protocol",
            str(protocol),
        ],
    )
    with pytest.raises(ValueError, match="frozen|registered"):
        module.main()
    assert not (tmp_path / "out").exists()
