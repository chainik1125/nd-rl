"""Core symbolic scratchpad checks; no textbook or checkpoint payloads loaded."""
# ruff: noqa: E402

import json
import os
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
R1_COT = CODE / "experiments" / "current" / "r1_cot"
VENDOR = R1_COT / "vendor"
sys.path[:0] = [str(R1_COT.parent), str(VENDOR)]

from r1_cot.run import (
    ScratchTokenizer,
    backward_plan,
    ce_backward,
    encode_training,
    expand_vocabulary,
    group_advantages,
    guard_final_release,
    guard_keys,
    guard_training,
    judge_response,
    load_model,
    policy_update,
    sample,
)

# The pinned upstream modules intentionally use top-level imports. Resolve
# them after installing their vendored root rather than pretending that root
# is a normal project package in the static import search path.
collate_rollouts = import_module("nd_grpo").collate_rollouts
ModelConfig = import_module("nd_model").ModelConfig
ProofModel = import_module("nd_model").ProofModel
save_checkpoint = import_module("nd_model").save_checkpoint


def test_dependencies_are_vendored():
    for name in ("nd_grpo", "nd_model", "strict_verify"):
        module = import_module(name)
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(VENDOR.resolve())


PROMPT = "THM P SEQ P PRF"
BODY = "N1 P : PR ; QED"
ROW = {"prompt": PROMPT, "proof": BODY}


def tiny_model() -> Any:
    torch.manual_seed(11)
    return ProofModel(
        ModelConfig(vocab_size=43, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=128)
    ).eval()


def test_cold_start_and_immutable_target():
    tok = ScratchTokenizer()
    row = encode_training(ROW, tok, "cot", 128)
    result = judge_response(PROMPT, row["response_ids"], tok, "cot")
    assert result["reward"] == 1
    assert result["found_len"] == 1
    assert result["body"] == BODY
    assert result["scratch_tokens"] == 3
    assert backward_plan(BODY) == ["PR", "P", ";"]
    assert judge_response("THM P SEQ Q PRF", row["response_ids"], tok, "cot")["reward"] == 0


def test_dependency_style_omits_irrelevant_lines_but_keeps_literal_proof():
    tok = ScratchTokenizer()
    prompt = "THM P , Q SEQ ( ( P & Q ) v S ) PRF"
    body = (
        "N1 P : PR ; N2 Q : PR ; N3 ( P v R ) : ORI1 N1 ; "
        "N4 ( P & Q ) : ANDI N1 N2 ; N5 ( ( P & Q ) v S ) : ORI1 N4 ; QED"
    )
    row = {"prompt": prompt, "proof": body}
    reverse = encode_training(row, tok, "cot", 256)
    dependency = encode_training(row, tok, "cot", 256, scratch_style="dependency")
    end = dependency["response_ids"].index(tok.stoi["</THINK>"])
    words = [tok.itos[i] for i in dependency["response_ids"][1:end]]
    assert words == [
        "ORI1",
        "(",
        "(",
        "P",
        "&",
        "Q",
        ")",
        "v",
        "S",
        ")",
        ";",
        "ANDI",
        "(",
        "P",
        "&",
        "Q",
        ")",
        ";",
    ]
    assert len(dependency["response_ids"]) < len(reverse["response_ids"])
    assert judge_response(prompt, dependency["response_ids"], tok, "cot")["body"] == body
    assert judge_response(prompt, dependency["response_ids"], tok, "cot")["reward"] == 1
    with pytest.raises(ValueError, match="above cap"):
        encode_training(row, tok, "cot", 256, max_proof_lines=4, scratch_style="dependency")


def test_dependency_style_preserves_scope_and_direct_control_is_identical():
    tok = ScratchTokenizer()
    prompt = "THM SEQ ( P > ( P v Q ) ) PRF"
    body = "N1 | P : AS ; N2 | ( P v Q ) : ORI1 N1 ; N3 ( P > ( P v Q ) ) : IMPI N1 N2 ; QED"
    row = {"prompt": prompt, "proof": body}
    result = encode_training(row, tok, "cot", 256, scratch_style="dependency")
    end = result["response_ids"].index(tok.stoi["</THINK>"])
    scratch = [tok.itos[i] for i in result["response_ids"][1:end]]
    assert scratch.count("|") == 2
    assert "AS" in scratch
    assert judge_response(prompt, result["response_ids"], tok, "cot")["reward"] == 1
    assert encode_training(row, tok, "direct", 256, scratch_style="dependency") == encode_training(
        row, tok, "direct", 256, scratch_style="reverse"
    )


def test_final_boundaries_and_qed_are_never_repaired():
    tok = ScratchTokenizer()
    ids = encode_training(ROW, tok, "cot", 128)["response_ids"]
    for malformed in (
        ids[1:],
        ids[:-1],
        ids + [tok.stoi["P"]],
        ids[:2] + [tok.stoi["<FINAL>"]] + ids[2:],
        [tok.stoi["<THINK>"]] + ids,
    ):
        assert judge_response(PROMPT, malformed, tok, "cot")["reward"] == 0
    # QED in the scratch is permitted; only final proof QED terminates sampling.
    assert judge_response(PROMPT, ids[:1] + [tok.eos_id] + ids[1:], tok, "cot")["reward"] == 1


def test_sft_length_cap_and_context_fail_closed():
    tok = ScratchTokenizer()
    with pytest.raises(ValueError, match="above cap"):
        encode_training(ROW, tok, "cot", 128, max_proof_lines=0)
    with pytest.raises(ValueError, match="exceeds context"):
        encode_training(ROW, tok, "cot", 8)
    with pytest.raises(ValueError, match="Invalid SFT proof"):
        encode_training({"prompt": PROMPT, "proof": BODY + " P"}, tok, "cot", 128)


def test_future_and_self_references_get_zero():
    tok = ScratchTokenizer()
    prompt = "THM SEQ P PRF"
    # <REF><ENDREF> encodes zero distance, hence an invalid self citation N1.
    ids = [tok.stoi[t] for t in ("<LINE>", "P", ":", "R", "<REF>", "<ENDREF>", ";", "QED")]
    assert judge_response(prompt, ids, tok, "direct")["reward"] == 0


def test_vocabulary_expansion_and_context_preserve_old_weights(tmp_path: Path):
    torch.manual_seed(5)
    model = ProofModel(ModelConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=64))
    before = model.token_embedding.weight.detach().clone()
    expanded = expand_vocabulary(model)
    assert torch.equal(before, expanded.token_embedding.weight[:40])
    assert expanded.lm_head.weight is expanded.token_embedding.weight
    path = tmp_path / "model.pt"
    save_checkpoint(path, expanded, ScratchTokenizer())
    loaded, tok, _ = load_model(path, max_seq_len=256)
    assert isinstance(tok, ScratchTokenizer)
    assert loaded.config.max_seq_len == 256
    assert loaded.rope_cos.shape[0] == 256
    assert torch.equal(loaded.token_embedding.weight, expanded.token_embedding.weight)


def test_mask_credits_scratch_and_sampled_pad():
    tok = ScratchTokenizer()
    row = encode_training(ROW, tok, "cot", 128)
    row["response_ids"].insert(2, tok.pad_id)
    batch = collate_rollouts([row], tok.pad_id, "cpu")
    assert batch["mask"].sum() == len(row["response_ids"])
    assert batch["mask"][0, len(row["prompt_ids"]) + 1]
    model = tiny_model()
    loss = ce_backward(model, tok, [row], 1)
    assert loss > 0
    assert model.token_embedding.weight.grad[tok.pad_id].abs().sum() > 0


def test_cached_sampling_logprobs_match_teacher_mixed_prompt_lengths():
    model, tok = tiny_model(), ScratchTokenizer()
    prompts = [PROMPT, "THM ( P > Q ) , P SEQ Q PRF"]
    rows = sample(
        model, tok, prompts, mode="cot", temperature=0.65, seed=17, max_new_tokens=12, batch_size=2
    )
    batch = collate_rollouts(rows, tok.pad_id, "cpu")
    logp = (model(batch["inputs"]).float() / 0.65).log_softmax(-1)
    chosen = logp.gather(-1, batch["targets"][..., None]).squeeze(-1)
    for row, values, mask in zip(rows, chosen, batch["mask"], strict=True):
        assert len(row["response_ids"]) == len(row["rollout_logprobs"])
        assert torch.allclose(values[mask], torch.tensor(row["rollout_logprobs"]), atol=2e-6)


def test_advantages_zero_for_uniform_groups_and_loo_values():
    result = group_advantages([[0, 0], [1, 1], [0, 1]])
    assert result.tolist() == [[0, 0], [0, 0], [-1, 1]]
    with pytest.raises(ValueError, match="binary"):
        group_advantages([[0, 0.2]])


def test_policy_denominator_is_behavior_and_pad_is_not_filtered():
    model, tok = tiny_model(), ScratchTokenizer()
    rows = sample(
        model,
        tok,
        [PROMPT, PROMPT],
        mode="cot",
        temperature=0.8,
        seed=12,
        max_new_tokens=10,
        batch_size=2,
    )
    for index, row in enumerate(rows):
        row["advantage"] = (-1.0, 1.0)[index]
        # A slight perturbation distinguishes behavior denominator from teacher.
        row["rollout_logprobs"] = [value + 0.01 for value in row["rollout_logprobs"]]
    args = SimpleNamespace(
        microbatch=2,
        temperature=0.8,
        policy_epochs=1,
        clip=0.2,
        loss_normalization="fixed",
        max_new_tokens=10,
        max_logprob_error=0.5,
        retention_weight=0,
    )
    metrics = policy_update(model, tok, rows, torch.optim.AdamW(model.parameters(), lr=1e-4), args)
    for row in rows:
        assert row["old_logprobs"] == row["rollout_logprobs"]
    numerical_error = metrics["behavior_teacher_max_abs_error"]
    assert isinstance(numerical_error, (float, int))
    assert 0.009 < numerical_error < 0.011


def test_guard_checks_canonical_context_and_skeleton(tmp_path: Path):
    fields = (
        "canonical_theorem_sha256",
        "nonempty_premise_context_sha256",
        "closed_formula_skeleton_sha256",
    )
    path = tmp_path / "guards.json"
    for prompt, field_index in ((PROMPT, 0), (PROMPT, 1), ("THM SEQ ( P > P ) PRF", 2)):
        guards = {field: [] for field in fields}
        guards[fields[field_index]] = [guard_keys(prompt)[field_index]]
        path.write_text(json.dumps(guards))
        with pytest.raises(ValueError, match="Protected theorem"):
            guard_training([{"prompt": prompt}], tmp_path / "train.jsonl", path)
    with pytest.raises(ValueError, match="requires --guard-json"):
        guard_training([ROW], tmp_path / "train.jsonl", None)
    with pytest.raises(ValueError, match="sealed"):
        guard_training([ROW], tmp_path / "sealed_final" / "data.jsonl", path)


def test_scratch_qed_does_not_terminate_or_force_final():
    model, tok = tiny_model(), ScratchTokenizer()

    def always_qed(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        logits = torch.zeros((*tokens.shape, len(tok)))
        logits[..., tok.eos_id] = 100
        return logits

    model.forward = always_qed
    direct = sample(
        model, tok, [PROMPT], mode="direct", temperature=0, max_new_tokens=5, batch_size=1
    )[0]
    cot = sample(model, tok, [PROMPT], mode="cot", temperature=0, max_new_tokens=5, batch_size=1)[0]
    assert direct["response_ids"] == [tok.eos_id]
    assert cot["response_ids"] == [tok.eos_id] * 5
    assert judge_response(PROMPT, cot["response_ids"], tok, "cot")["reward"] == 0


def test_final_gate_refuses_before_opening_missing_payload(tmp_path: Path):
    path = tmp_path / "sealed_final" / "payload.jsonl"
    with pytest.raises(ValueError, match="final-release-json"):
        guard_final_release(path, "checkpoint_hash", None)
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "released": True,
                "no_further_training": True,
                "not_before_utc": "2099-01-01T00:00:00Z",
            }
        )
    )
    with pytest.raises(ValueError, match="deadline"):
        guard_final_release(path, "checkpoint_hash", release)


@pytest.mark.skipif(
    os.environ.get("R1_TEST_CUDA") != "1", reason="Requires explicitly allocated GPU"
)
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_precision_uses_matching_cache_and_teacher(precision: str):
    from r1_cot.run import amp

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU lacks BF16 support")
    model, tok = tiny_model().cuda(), ScratchTokenizer()
    model.r1_precision = precision
    observed = []
    original_allocate = model.allocate_cache

    def record_cache(*args: Any, **kwargs: Any) -> Any:
        cache = original_allocate(*args, **kwargs)
        observed.append(cache[0][0].dtype)
        return cache

    model.allocate_cache = record_cache
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        rows = sample(
            model,
            tok,
            [PROMPT, "THM ( P > Q ) , P SEQ Q PRF"],
            mode="cot",
            temperature=0.65,
            seed=17,
            max_new_tokens=12,
            batch_size=2,
        )
        batch = collate_rollouts(rows, tok.pad_id, "cuda")
        with amp(model):
            logp = (model(batch["inputs"]).float() / 0.65).log_softmax(-1)
        chosen = logp.gather(-1, batch["targets"][..., None]).squeeze(-1)
        tolerance = 0.05 if precision == "bf16" else 1e-5
        for row, values, mask in zip(rows, chosen, batch["mask"], strict=True):
            assert torch.allclose(
                values[mask].cpu(), torch.tensor(row["rollout_logprobs"]), atol=tolerance, rtol=0
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert observed == [torch.bfloat16 if precision == "bf16" else torch.float32]
