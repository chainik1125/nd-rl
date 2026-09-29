"""Tests for ``code.shared_code.lean_seq``.

The golden fixture was produced by the original tokenizer (``dan-pandori/nd-takehome``,
``lean_tok.py``, mode ``lean_seq``) on 68 proofs from its pretraining set, chosen to cover every ND
rule. Matching it id for id is what makes checkpoints trained there usable here.
"""

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest

from code.shared_code.lean_seq import (
    RULES,
    LeanSeqTokenizer,
    NDParseError,
    parse_prompt,
    parse_proof,
    proof_tokens,
)


class Golden(TypedDict):
    prompt: str
    proof: str
    prompt_ids: list[int]
    proof_ids: list[int]
    statement: str
    text: str


FIXTURE = Path(__file__).parent / "fixtures" / "lean_seq_golden.jsonl"
_LINES = FIXTURE.read_text(encoding="utf-8").splitlines()
VOCAB: list[str] = json.loads(_LINES[0])["vocab"]
GOLDEN: list[Golden] = [json.loads(line) for line in _LINES[1:]]
TOK = LeanSeqTokenizer()


def ids_of(tokens: str) -> list[int]:
    return [TOK.stoi[t] for t in tokens.split()] + [TOK.eos_id]


def test_vocabulary_matches_the_original() -> None:
    assert TOK.vocab_size == 107
    assert TOK.itos == VOCAB


def test_fixture_covers_every_rule() -> None:
    seen = {ln.rule for rec in GOLDEN for ln in parse_proof(rec["proof"].split())}
    assert seen == RULES


@pytest.mark.parametrize("rec", GOLDEN, ids=[f"golden{i}" for i in range(len(GOLDEN))])
def test_golden_encoding(rec: Golden) -> None:
    assert TOK.encode_prompt(rec["prompt"]) == rec["prompt_ids"]
    assert TOK.encode_proof(rec["proof"]) == rec["proof_ids"]
    assert TOK.statement(rec["prompt"]) == rec["statement"]
    assert TOK.lean_text(rec["proof_ids"]) == rec["text"]


@pytest.mark.parametrize("rec", GOLDEN, ids=[f"golden{i}" for i in range(len(GOLDEN))])
def test_golden_round_trip(rec: Golden) -> None:
    assert TOK.decode(rec["proof_ids"]) == rec["proof"]


def test_name_shift_keeps_the_proof() -> None:
    rng = random.Random(0)
    for rec in GOLDEN:
        ids = rec["proof_ids"]
        shifted = TOK.shift_names(ids, rng)
        assert len(shifted) == len(ids)
        assert all(0 <= x < TOK.vocab_size for x in shifted)
        assert [x for x in shifted if x < TOK.first_name] == [x for x in ids if x < TOK.first_name]
        assert TOK.decode(shifted) == rec["proof"]


def test_premises_render_in_order() -> None:
    prompt = "THM ( P > Q ) , P SEQ Q PRF"
    proof = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"
    ids = TOK.encode_proof(proof)
    assert TOK.statement(prompt) == (
        "theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : P ) : Q := by"
    )
    assert TOK.lean_text(ids) == (
        "have n1 : ( P → Q ) := h1 ; have n2 : P := h2 ; have n3 : Q := n1 n2 ; exact n3"
    )
    assert TOK.decode(ids) == proof


@pytest.mark.parametrize(
    "tokens, reason",
    [
        ("have n1 : P := n2 ; exact n1", "unbound"),
        ("have n1 : P := h2 ; exact n1", "PR position"),
        ("have n1 : P := h1 ; have n2 : Q := h2 ; exact n1", "exact not last"),
        ("have n1 : P := h1 ; exact n1 ;", "trailing tokens"),
        ("have n1 : P := h1 ;", "expected exact"),
        ("have n1 : ( P ∧ := h1 ; exact n1", "formula"),
    ],
)
def test_decode_rejects_outside_the_grammar(tokens: str, reason: str) -> None:
    assert TOK.decode(ids_of(tokens)) == f"LEANPARSE {reason}"


def test_decode_needs_eos() -> None:
    assert TOK.decode(ids_of("have n1 : P := h1 ; exact n1")[:-1]) == "LEANPARSE no-eos"
    assert TOK.lean_text(ids_of("exact n1")[:-1]) is None


@pytest.mark.parametrize(
    "proof",
    [
        "N1 P : FOO ; QED",  # unknown rule
        "N1 P : PR ;",  # no QED
        "N1 P : PR ; N2 P : IMPI N1 N1 ; QED",  # cites something that is not a closed box
        "N1 | P : AS ; QED",  # ends inside a box
    ],
)
def test_bad_nd_proofs_raise(proof: str) -> None:
    with pytest.raises(NDParseError):
        proof_tokens(proof)


def test_bad_prompt_raises() -> None:
    with pytest.raises(NDParseError):
        parse_prompt("THM P Q")


# The experiment pipeline's tokenizer interface (``nd_model.ProofTokenizer``).


def test_pipeline_ids() -> None:
    assert len(TOK) == TOK.vocab_size == 107
    assert TOK.itos[TOK.prf_id] == "by"
    assert TOK.itos[TOK.pad_id] == "<pad>"
    assert TOK.itos[TOK.eos_id] == "<eos>"


@pytest.mark.parametrize("rec", GOLDEN, ids=[f"golden{i}" for i in range(len(GOLDEN))])
def test_encode_record_is_prompt_then_proof(rec: Golden) -> None:
    ids = rec["prompt_ids"] + rec["proof_ids"]
    text = rec["prompt"] + " " + rec["proof"]
    assert TOK.encode_record(text) == ids
    assert TOK.encode_record({"text": text}) == ids
    assert TOK.encode_record({"prompt": rec["prompt"], "proof": rec["proof"]}) == ids
    # The pipeline takes the prompt length as ids.index(prf_id) + 1.
    assert ids.index(TOK.prf_id) + 1 == len(rec["prompt_ids"])
    assert TOK.decode_body(rec["proof_ids"]) == rec["proof"]


def test_encode_record_needs_prf() -> None:
    with pytest.raises(NDParseError):
        TOK.encode_record("THM P SEQ P N1 P : PR ; QED")


def test_state_dict_round_trip() -> None:
    state = TOK.state_dict()
    assert state["version"] == "lean-seq-v1"
    assert LeanSeqTokenizer.from_state_dict(state).itos == TOK.itos
    with pytest.raises(ValueError):
        LeanSeqTokenizer.from_state_dict({**state, "version": "relative-unary-v1"})
    with pytest.raises(ValueError):
        LeanSeqTokenizer.from_state_dict({**state, "tokens": list(reversed(TOK.itos))})


def _lean() -> str | None:
    found = shutil.which("lean")
    if found:
        return found
    elan = Path(os.path.expanduser("~/.elan/bin/lean"))
    return str(elan) if elan.exists() else None


@pytest.mark.slow
def test_lean_accepts_the_golden_proofs(tmp_path: Path) -> None:
    lean = _lean()
    if lean is None:
        pytest.skip("Lean 4 is not installed")
    theorems = [
        rec["statement"].replace("theorem t ", f"theorem t{i} ", 1) + " " + rec["text"]
        for i, rec in enumerate(GOLDEN)
    ]
    src = tmp_path / "golden.lean"
    src.write_text("\n".join(theorems) + "\n", encoding="utf-8")
    run = subprocess.run([lean, str(src)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
