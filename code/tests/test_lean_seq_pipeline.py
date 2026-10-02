"""``LeanSeqTokenizer`` in the ND experiment pipeline (``r1_cot/vendor/nd_model.py``), end to end on
CPU: encode and collate, a few training steps, sampling, and a checkpoint round trip."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from nd_model import (
    EncodedRecords,
    ModelConfig,
    ProofModel,
    collate_records,
    encode_records,
    generate_bodies,
    load_checkpoint,
    save_checkpoint,
    train_steps,
)

from code.shared_code.lean_seq import LeanSeqTokenizer, parse_proof

FIXTURE = Path(__file__).parent / "fixtures" / "lean_seq_golden.jsonl"
GOLDEN: list[dict[str, Any]] = [
    json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()[1:]
]
# nd_model annotates its tokenizer as ProofTokenizer; LeanSeqTokenizer implements the same
# interface without inheriting from it.
TOK: Any = LeanSeqTokenizer()
RECORDS = [{"prompt": rec["prompt"], "proof": rec["proof"]} for rec in GOLDEN]


def tiny_model() -> Any:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=len(TOK), d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=1024
    )
    return ProofModel(config)


def test_encode_and_collate_mask_the_prompt() -> None:
    data: EncodedRecords = encode_records(TOK, RECORDS, 1024)
    assert data.prompt_lengths == [len(rec["prompt_ids"]) for rec in GOLDEN]
    inputs, targets = collate_records(data, range(len(GOLDEN)), TOK.pad_id)
    assert inputs.shape == targets.shape
    for row, rec in enumerate(GOLDEN):
        trained = [int(x) for x in targets[row] if int(x) != -100]
        assert trained == rec["proof_ids"]


def test_train_sample_and_checkpoint(tmp_path: Path) -> None:
    model = tiny_model()
    history = train_steps(
        model, TOK, RECORDS, steps=40, batch_size=16, lr=3e-3, amp=False, log_every=40
    )
    assert history[-1]["loss"] < history[0]["loss"]
    prompts = [rec["prompt"] for rec in GOLDEN[:6]]
    bodies = generate_bodies(model, TOK, prompts, max_new_tokens=96, amp=False)
    assert len(bodies) == len(prompts)
    for body in bodies:
        # Either outside the grammar (a marker) or a well-formed ND proof; never anything else.
        assert body.startswith("LEANPARSE") or parse_proof(body.split())
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, TOK, note="lean_seq")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = LeanSeqTokenizer.from_state_dict(checkpoint["tokenizer"])
    assert restored.itos == TOK.itos
    # nd_model's own loader knows only its token formats; load lean_seq checkpoints with
    # LeanSeqTokenizer.from_state_dict plus ProofModel (see shared_code/README.md).
    with pytest.raises(ValueError):
        load_checkpoint(path, device="cpu")
