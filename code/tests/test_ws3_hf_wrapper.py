"""Plan T2 gate: the HF wrapper reproduces the fork model's logits and the tokenizer round-trips."""

import json
import os

import ndpath
import pytest
import torch

# isort: split
from hf_wrapper import (
    NDProofConfig,
    NDProofForCausalLM,
    build_tokenizer,
    decode_ids,
    wrap_fork_model,
)
from model import GPT  # fork_pin
from tokenizer import Tokenizer  # fork_pin

HELDOUT = os.path.join(ndpath.FORK_DATA, "heldout.jsonl")
have_data = os.path.exists(HELDOUT)


def _records(n):
    with open(HELDOUT) as f:
        rows = [json.loads(line) for line in f]
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def _random_pair(seed=0):
    torch.manual_seed(seed)
    gpt = GPT(99, 2, 64, 4, 256)  # small random model, same architecture family
    return gpt, wrap_fork_model(gpt)


def test_tokenizer_ids_match_fork():
    tok = build_tokenizer("abs")
    ft = Tokenizer("abs")
    assert len(tok) == ft.vocab_size == 99
    assert tok.pad_token_id == ft.pad == 0
    assert tok.eos_token_id == ft.eos
    text = "THM ( P > Q ) , P SEQ Q PRF"
    assert tok(text)["input_ids"] == ft.encode_prompt(text)


@pytest.mark.skipif(not have_data, reason="fork_pin data not fetched")
def test_tokenizer_roundtrip_exact():
    tok = build_tokenizer("abs")
    ft = Tokenizer("abs")
    for r in _records(300):
        ids = tok(r["prompt"] + " " + r["proof"])["input_ids"]
        assert ids == ft.encode_prompt(r["prompt"]) + ft.encode_proof(r["proof"])
        # decode_ids is the exact inverse of encode_proof (QED included)
        assert decode_ids("abs", ft.encode_proof(r["proof"])) == r["proof"]
        # the HF decode (used by TRL only for logging) keeps every token when specials are kept
        assert tok.decode(ids, skip_special_tokens=False) == r["prompt"] + " " + r["proof"]


def test_logits_match_no_padding():
    gpt, wrapped = _random_pair()
    x = torch.randint(1, 99, (3, 17))
    with torch.no_grad():
        ref = gpt(x)
        out = wrapped(input_ids=x).logits
    assert torch.allclose(ref, out, atol=1e-4)


def test_logits_match_left_padding():
    """Left-padded rows must equal the unpadded forward on the real tokens."""
    gpt, wrapped = _random_pair(1)
    tok = build_tokenizer("abs")
    a = "THM ( P > Q ) , P SEQ Q PRF"
    b = "THM SEQ ( P > P ) PRF"
    enc = tok([a, b], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = wrapped(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        for i, text in enumerate((a, b)):
            ids = torch.tensor([tok(text)["input_ids"]])
            ref = gpt(ids)[0]
            n = ids.shape[1]
            assert torch.allclose(ref, out[i, -n:], atol=1e-4), f"row {i}"


@pytest.mark.skipif(not have_data, reason="fork_pin data not fetched")
def test_logits_match_real_checkpoint_100_sequences():
    pytest.importorskip("safetensors")
    if not os.path.exists(ndpath.FORK_CKPT):
        pytest.skip("checkpoint not fetched")
    from hf_wrapper import load_pretrained_wrapper

    model, tok = load_pretrained_wrapper()
    ck = torch.load(ndpath.FORK_CKPT, map_location="cpu", weights_only=True)
    gpt = GPT(99, 4, 256, 8)
    gpt.load_state_dict(ck["state"])
    gpt.eval()
    ft = Tokenizer("abs")
    worst = 0.0
    with torch.no_grad():
        for r in _records(100):
            ids = torch.tensor([ft.encode_prompt(r["prompt"]) + ft.encode_proof(r["proof"])])
            diff = (gpt(ids) - model(input_ids=ids).logits).abs().max().item()
            worst = max(worst, diff)
    assert worst < 1e-4, worst


def test_generate_stops_at_qed_and_save_load_roundtrip(tmp_path):
    gpt, wrapped = _random_pair(2)
    tok = build_tokenizer("abs")
    enc = tok(["THM P SEQ P PRF", "THM SEQ ( P > P ) PRF"], return_tensors="pt", padding=True)
    out = wrapped.generate(
        **enc,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=False,
    )
    assert out.shape[1] <= enc["input_ids"].shape[1] + 12
    # save_pretrained / from_pretrained through the Auto registry (what TRL does for the ref model)
    from transformers import AutoModelForCausalLM

    d = tmp_path / "base"
    wrapped.save_pretrained(d)
    tok.save_pretrained(d)
    re = AutoModelForCausalLM.from_pretrained(d)
    assert isinstance(re, NDProofForCausalLM)
    x = torch.randint(1, 99, (2, 9))
    with torch.no_grad():
        assert torch.allclose(wrapped(input_ids=x).logits, re(input_ids=x).logits, atol=1e-5)
    assert isinstance(re.config, NDProofConfig)


# ---------------------------------------------------------------------------- KV cache


def _left_padded_batch(tok):
    a = "THM ( P > Q ) , P SEQ Q PRF"
    b = "THM SEQ ( P > P ) PRF"
    c = "THM ( P & Q ) , ( Q > R ) SEQ ( R v S ) PRF"
    return tok([a, b, c], return_tensors="pt", padding=True)


def test_cached_generate_matches_uncached_greedy():
    """Greedy ids and per-step scores are the same with and without the KV cache (left padding)."""
    _, wrapped = _random_pair(3)
    tok = build_tokenizer("abs")
    enc = _left_padded_batch(tok)
    outs = {}
    for use_cache in (False, True):
        outs[use_cache] = wrapped.generate(
            **enc,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=use_cache,
            output_scores=True,
            return_dict_in_generate=True,
        )
    assert torch.equal(outs[False].sequences, outs[True].sequences)
    assert len(outs[False].scores) == len(outs[True].scores) > 1
    for s0, s1 in zip(outs[False].scores, outs[True].scores, strict=True):
        finite = torch.isfinite(s0) & torch.isfinite(s1)
        assert torch.allclose(s0[finite], s1[finite], atol=1e-4)


def test_cache_decode_step_logits_match_full_forward():
    """prefill + one cached step == full forward on the extended sequence, per row, to 1e-4."""
    from hf_wrapper import NDCache

    _, wrapped = _random_pair(4)
    tok = build_tokenizer("abs")
    enc = _left_padded_batch(tok)
    ids, am = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        out = wrapped(input_ids=ids, attention_mask=am, use_cache=True)
        assert isinstance(out.past_key_values, NDCache)
        assert out.past_key_values.get_seq_length() == ids.shape[1]
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        am2 = torch.cat([am, torch.ones_like(nxt)], 1)
        pos2 = (am2.long().cumsum(-1) - 1).clamp(min=0)[:, -1:]
        step = wrapped(
            input_ids=nxt,
            attention_mask=am2,
            position_ids=pos2,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        full = wrapped(input_ids=torch.cat([ids, nxt], 1), attention_mask=am2, use_cache=False)
        assert torch.allclose(step.logits[:, -1], full.logits[:, -1], atol=1e-4)
        assert torch.allclose(out.logits, full.logits[:, :-1], atol=1e-4)


def test_cache_grows_past_initial_reserve():
    from hf_wrapper import NDCache

    gpt, wrapped = _random_pair(5)
    wrapped.cache_reserve = 2
    x = torch.randint(1, 99, (2, 5))
    with torch.no_grad():
        out = wrapped(input_ids=x, use_cache=True)
        cache = out.past_key_values
        assert cache.layers[0]["k"].shape[2] == 7
        seq = x
        for _ in range(6):
            nxt = out.logits[:, -1:].argmax(-1)
            seq = torch.cat([seq, nxt], 1)
            out = wrapped(input_ids=nxt, past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == 11 and cache.layers[0]["k"].shape[2] >= 11
        assert torch.allclose(out.logits[:, -1], gpt(seq)[:, -1], atol=1e-4)
    assert isinstance(cache, NDCache)


@pytest.mark.skipif(not have_data, reason="fork_pin data not fetched")
def test_trl_grpo_accepts_wrapper_as_policy_and_reference(tmp_path):
    """Gate T2: GRPOTrainer takes the wrapper for rollouts and reloads it as the reference model."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from eval.reward import ProofReward
    from hf_wrapper import load_pretrained_wrapper, save_base

    if not os.path.exists(ndpath.FORK_CKPT):
        pytest.skip("checkpoint not fetched")
    model, tok = load_pretrained_wrapper()
    save_base(model, tok, str(tmp_path / "hf_base"))
    ds = Dataset.from_list(
        [{"prompt": r["prompt"]} for r in _records(4)]
    )
    cfg = GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        num_generations=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_completion_length=24,
        max_steps=1,
        learning_rate=1e-5,
        beta=0.01,
        temperature=1.0,
        generation_kwargs={"use_cache": True},
        gradient_checkpointing=False,
        use_cpu=True,
        bf16=False,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_dropout=True,
        remove_unused_columns=False,
    )
    reward_fn = ProofReward("abs", dump_path=str(tmp_path / "completions.jsonl"))
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=cfg, train_dataset=ds, processing_class=tok
    )
    assert isinstance(trainer.ref_model, NDProofForCausalLM)
    assert trainer.ref_model is not trainer.model
    trainer.train()
    logs = [l for l in trainer.state.log_history if "reward" in l]
    assert logs and "kl" in logs[0] and "entropy" in logs[0]
    assert (tmp_path / "completions.jsonl").read_text().count("\n") == 4
