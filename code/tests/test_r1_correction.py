"""Rejected-draft cold starts use synthetic cap-limited fixtures only."""
# ruff: noqa: E402

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
R1_COT = CODE / "experiments" / "current" / "r1_cot"
sys.path[:0] = [str(R1_COT.parent), str(R1_COT / "vendor")]

from r1_cot.correction_plan import ATOMS, BINARY, RULES, corrupt_draft
from r1_cot.run import (
    ScratchRepairTokenizer,
    ScratchTokenizer,
    encode_training,
    expand_vocabulary,
    judge_response,
    load_model,
    record_cold_start_provenance,
    sample,
)

nd_model = import_module("nd_model")
strict_verify = import_module("strict_verify").safe_verify
FIXTURES = (
    ("THM P SEQ P PRF", "N1 P : PR ; QED"),
    ("THM F SEQ F PRF", "N1 F : PR ; QED"),
    (
        "THM SEQ ( P > ( P v Q ) ) PRF",
        "N1 | P : AS ; N2 | ( P v Q ) : ORI1 N1 ; N3 ( P > ( P v Q ) ) : IMPI N1 N2 ; QED",
    ),
    (
        "THM P , Q SEQ ( P & Q ) PRF",
        "N1 P : PR ; N2 Q : PR ; N3 ( P v R ) : ORI1 N1 ; N4 ( P & Q ) : ANDI N1 N2 ; QED",
    ),
)


def tiny_model(vocab_size: int) -> Any:
    return nd_model.ProofModel(
        nd_model.ModelConfig(
            vocab_size=vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=256
        )
    ).eval()


def test_seeded_single_token_drafts_are_rejected_and_correct_sources_unchanged():
    for prompt, body in FIXTURES:
        assert strict_verify(prompt, body)[0]
        seen = set()
        for seed in range(12):
            result = corrupt_draft(prompt, body, seed=seed)
            assert result == corrupt_draft(prompt, body, seed=seed)
            draft = result["draft"]
            provenance = result["provenance"]
            assert not strict_verify(prompt, draft)[0]
            assert len(body.split()) == len(draft.split())
            differences = [
                i
                for i, (old, new) in enumerate(zip(body.split(), draft.split(), strict=True))
                if old != new
            ]
            assert differences == [provenance["mutation"]["token_index"]]
            assert draft.split()[-1] == "QED"
            assert provenance["draft_strict_reward"] == 0
            assert provenance["correct_proof_lines"] <= provenance["original_proof_cap"] == 6
            assert strict_verify(prompt, body)[0]
            seen.add(draft)
        assert len(seen) > 1


def test_every_mutation_candidate_is_supported_in_frozen_base_vocabulary():
    tok = nd_model.ProofTokenizer()
    assert set((*ATOMS, *BINARY, *RULES)).issubset(tok.stoi)
    # The atom R and the reiteration rule deliberately share token id 9.
    assert tok.stoi["R"] == 9
    assert len(tok) == 40


def test_two_thousand_corruptions_of_one_thousand_synthetic_sources_roundtrip():
    tok = ScratchRepairTokenizer()
    atoms, operators = ("P", "Q", "R", "S"), ("&", "v", ">")
    distinct_prompts, mutation_tokens = set(), set()
    for index in range(1000):
        first = f"( {atoms[index % 4]} {operators[index // 4 % 3]} {atoms[index // 12 % 4]} )"
        second = (
            f"( {atoms[index // 48 % 4]} {operators[index // 192 % 3]} {atoms[index // 576 % 4]} )"
        )
        goal = f"( {first} & {second} )"
        prompt = f"THM {first} , {second} SEQ {goal} PRF"
        body = f"N1 {first} : PR ; N2 {second} : PR ; N3 {goal} : ANDI N1 N2 ; QED"
        distinct_prompts.add(prompt)
        assert strict_verify(prompt, body)[0]
        prompt_length = len(tok.encode_prompt(prompt))
        for seed in (index, index + 100003):
            result = corrupt_draft(prompt, body, seed=seed)
            replacement = result["provenance"]["mutation"]["after"]
            assert replacement in tok.stoi
            mutation_tokens.add(replacement)
            encoded = tok.encode_record(prompt + " " + result["draft"])
            decoded = tok.decode_body(encoded[prompt_length:])
            assert decoded == result["draft"]
            assert not strict_verify(prompt, decoded)[0]
    assert len(distinct_prompts) == 1000
    assert "R" in mutation_tokens


def test_correction_target_keeps_final_proof_and_has_explicit_draft_revision_markers():
    tok = ScratchRepairTokenizer()
    prompt, body = FIXTURES[2]
    encoded = encode_training(
        {"prompt": prompt, "proof": body},
        tok,
        "cot",
        256,
        scratch_style="correction",
        scratch_seed=71,
    )
    ids = encoded["response_ids"]
    assert ids[0] == tok.stoi["<THINK>"]
    assert ids[1] == tok.stoi["<DRAFT>"]
    error = ids.index(tok.stoi["<ERROR>"])
    assert ids[error + 1] == tok.stoi["<REVISE>"]
    draft = tok.decode_body(ids[2:error])
    assert not strict_verify(prompt, draft)[0]
    assert ids.count(tok.eos_id) == 2
    result = judge_response(prompt, ids, tok, "cot")
    assert result["reward"] == 1 and result["body"] == body
    final_at = ids.index(tok.stoi["<FINAL>"])
    bad_final = ids[: final_at + 1] + ids[2:error]
    assert judge_response(prompt, bad_final, tok, "cot")["reward"] == 0
    # Direct control shares all 46 vocabulary rows but receives no draft targets.
    direct = encode_training(
        {"prompt": prompt, "proof": body}, tok, "direct", 256, scratch_style="correction"
    )
    assert judge_response(prompt, direct["response_ids"], tok, "direct")["body"] == body
    assert tok.stoi["<DRAFT>"] not in direct["response_ids"]


def test_correction_cap_tokenizer_and_context_are_fail_closed():
    prompt, body = FIXTURES[2]
    row = {"prompt": prompt, "proof": body}
    with pytest.raises(ValueError, match="1..2"):
        corrupt_draft(prompt, body, max_proof_lines=2)
    with pytest.raises(ValueError, match="valid terminal proof"):
        corrupt_draft(prompt, "N1 P : PR ; QED")
    with pytest.raises(ValueError, match="repair tokenizer"):
        encode_training(row, ScratchTokenizer(), "cot", 256, scratch_style="correction")
    with pytest.raises(ValueError, match="exceeds context"):
        encode_training(row, ScratchRepairTokenizer(), "cot", 30, scratch_style="correction")


@pytest.mark.parametrize("original_size", [40, 43])
def test_repair_vocabulary_expansion_and_checkpoint_roundtrip(original_size: int, tmp_path: Path):
    model = tiny_model(original_size)
    original = model.token_embedding.weight.detach().clone()
    tok = ScratchRepairTokenizer()
    expanded = expand_vocabulary(model, tok)
    assert expanded.config.vocab_size == 46
    assert torch.equal(expanded.token_embedding.weight[:original_size], original)
    assert expanded.lm_head.weight is expanded.token_embedding.weight
    path = tmp_path / "repair.pt"
    nd_model.save_checkpoint(path, expanded, tok, scratch_style="correction")
    reloaded, loaded_tok, metadata = load_model(path)
    assert isinstance(loaded_tok, ScratchRepairTokenizer)
    assert loaded_tok.state_dict() == tok.state_dict()
    assert loaded_tok.from_state_dict(tok.state_dict()).state_dict() == tok.state_dict()
    assert torch.equal(reloaded.token_embedding.weight, expanded.token_embedding.weight)
    assert metadata["scratch_style"] == "correction"


def test_sampling_generates_draft_qed_revision_and_final_without_injected_tokens():
    tok, model = ScratchRepairTokenizer(), tiny_model(46)
    prompt, body = FIXTURES[0]
    target = encode_training(
        {"prompt": prompt, "proof": body}, tok, "cot", 256, scratch_style="correction"
    )["response_ids"]
    generated_at = 0

    def scripted(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        nonlocal generated_at
        logits = torch.full((*tokens.shape, len(tok)), -100.0)
        logits[..., target[generated_at]] = 100
        generated_at += 1
        return logits

    model.forward = scripted
    response = sample(
        model,
        tok,
        [prompt],
        mode="cot",
        temperature=0,
        max_new_tokens=len(target) + 5,
        batch_size=1,
    )[0]
    assert response["response_ids"] == target
    assert len(response["rollout_logprobs"]) == len(target)
    assert generated_at == len(target)
    assert judge_response(prompt, response["response_ids"], tok, "cot")["reward"] == 1


def test_provenance_records_source_cap_seed_and_rejected_edit(tmp_path: Path):
    tok = ScratchRepairTokenizer()
    data = [
        encode_training(
            {"prompt": p, "proof": b}, tok, "cot", 256, scratch_style="correction", scratch_seed=23
        )
        for p, b in FIXTURES
    ]
    record_cold_start_provenance(data, tmp_path, "cold_start_provenance.jsonl")
    records = [
        json.loads(line)
        for line in (tmp_path / "cold_start_provenance.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(FIXTURES)
    assert [row["source_row_index"] for row in records] == list(range(len(FIXTURES)))
    for row in records:
        assert row["seed"] == 23
        assert row["original_proof_cap"] == 6
        assert row["draft_strict_reward"] == 0
        assert row["mutation"]["before"] != row["mutation"]["after"]
