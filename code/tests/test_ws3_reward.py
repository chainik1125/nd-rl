"""Plan T3: the six required reward cases, plus the TRL calling convention."""

import ndpath  # noqa: F401

# isort: split
from eval.reward import ProofReward, reward, verify_body
from tokenizer import Tokenizer  # fork_pin

PROMPT = "THM ( P > Q ) , P SEQ Q PRF"
VALID = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"
VALID_SHIFTED = (
    "N7 ( P > Q ) : PR ; N8 P : PR ; N9 Q : IMPE N7 N8 ; QED"  # any start index is legal
)


def test_valid_proof():
    assert reward(PROMPT, VALID) == 1.0
    assert reward(PROMPT, VALID_SHIFTED) == 1.0


def test_valid_proof_of_wrong_goal():
    # A correct derivation of P, not of the stated conclusion Q.
    wrong = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 P : R N2 ; QED"
    assert reward(PROMPT, wrong) == 0.0


def test_truncated_proof():
    assert reward(PROMPT, "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE") == 0.0
    assert reward(PROMPT, "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ;") == 0.0  # no QED


def test_correct_proof_followed_by_junk():
    junk = VALID + " N4 R : PR ; ( ( ("
    ok, reason, n_lines = verify_body(PROMPT, junk)
    # The official verifier stops at the first QED; the strict wrapper keeps that decision.
    assert ok and n_lines == 3, (ok, reason, n_lines)
    assert reward(PROMPT, junk) == 1.0


def test_empty_output():
    assert reward(PROMPT, "") == 0.0
    assert reward(PROMPT, "QED") == 0.0


def test_unparseable_output():
    assert reward(PROMPT, ") ) ( : ; N N QED") == 0.0
    assert reward(PROMPT, "hello world") == 0.0


def test_forward_reference_rejected_by_strict_wrapper():
    # Cites a later line; the base verifier's soundness hole, closed by strict_verify.
    fwd = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N4 ; N4 P : R N2 ; QED"
    assert reward(PROMPT, fwd) == 0.0


def test_trl_calling_convention_uses_completion_ids(tmp_path):
    ft = Tokenizer("abs")
    dump = tmp_path / "completions.jsonl"
    fn = ProofReward("abs", dump_path=str(dump))
    ids_valid = ft.encode_proof(VALID)
    ids_trunc = ids_valid[:-1]  # missing QED
    # TRL decodes completions with skip_special_tokens=True, so QED (the EOS token) is stripped
    # from the text it hands the reward function; the ids are authoritative.
    completions = [
        " ".join(ft.itos[i] for i in ids_valid if i != ft.eos),
        " ".join(ft.itos[i] for i in ids_trunc if i != ft.eos),
    ]
    assert "QED" not in completions[0]
    out = fn(
        prompts=[PROMPT, PROMPT], completions=completions, completion_ids=[ids_valid, ids_trunc]
    )
    assert out == [1.0, 0.0]
    lines = dump.read_text().strip().split("\n")
    assert len(lines) == 2 and '"reward": 1.0' in lines[0]
    # the truncated body (no QED) is a parse failure for the verifier, so it counts as unparseable
    assert fn.last_stats == {"reward/unparseable_frac": 0.5, "reward/invalid_frac": 0.5}
