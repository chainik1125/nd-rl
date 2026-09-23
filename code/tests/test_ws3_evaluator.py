"""Plan T3 evaluator: Wilson CIs, pass rate by length, max length solved, textbook slices, T1 gate."""

import ndpath  # noqa: F401
import pytest
import torch

# isort: split
from eval.evaluator import aggregate, judge, next_token_accuracy, textbook_table, wilson
from tokenizer import Tokenizer  # fork_pin

PROMPT = "THM ( P > Q ) , P SEQ Q PRF"
VALID = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"
SHIFTED = "N40 ( P > Q ) : PR ; N41 P : PR ; N42 Q : IMPE N40 N41 ; QED"


def test_wilson():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.404) < 0.01 and abs(hi - 0.596) < 0.01


def test_judge_accepts_any_start_index():
    ft = Tokenizer("abs")
    rows = judge([PROMPT, PROMPT], [[ft.encode_proof(VALID)], [ft.encode_proof(SHIFTED)]])
    assert rows[0][0][1] and rows[1][0][1]
    assert rows[0][0][3] == 3 and rows[1][0][3] == 3


def _j(ok, reason="ok", n_tokens=10):
    return ("body", ok, reason, 3 if ok else 0, n_tokens)


def test_aggregate_by_length_and_max_length():
    records = [
        {"prompt": "a", "n_lines": 2},
        {"prompt": "b", "n_lines": 2},
        {"prompt": "c", "n_lines": 5},
        {"prompt": "d", "L_true": 7},
    ]
    judged = [
        [_j(True)],
        [_j(False, "parse error")],
        [_j(False), _j(True)],  # pass@2: solved if any sample verifies
        [_j(False, "verifier exception: X", n_tokens=50), _j(False, n_tokens=50)],
    ]
    res = aggregate(records, judged, k=2, temperature=1.0, max_new=50)
    assert res["by_length"][2]["solved"] == 1 and res["by_length"][2]["n"] == 2
    assert res["by_length"][5]["rate"] == 1.0
    assert res["by_length"][7]["solved"] == 0
    assert res["by_length"][2]["ci"] == list(wilson(1, 2))
    assert res["max_length_solved"] == 5
    assert res["overall"]["solved"] == 2 and res["overall"]["n"] == 4
    assert res["samples"] == 6
    assert res["unparseable_frac"] == pytest.approx(2 / 6)
    assert res["invalid_frac"] == pytest.approx(4 / 6)
    assert res["capped_frac"] == pytest.approx(2 / 6)
    assert res["completion_tokens_max"] == 50
    assert "by_slice" not in res


def test_aggregate_textbook_slices_and_unknown_length():
    records = [
        {"prompt": "a", "reference_lines": 7, "slice": "dev58"},
        {"prompt": "b", "reference_lines": 3, "slice": "dev58"},
        {"prompt": "c", "slice": "train14"},  # the 14 train problems carry no length
    ]
    judged = [[_j(True)], [_j(False)], [_j(True)]]
    res = aggregate(records, judged, k=1)
    assert res["by_slice"]["dev58"] == {"solved": 1, "n": 2, "rate": 0.5, "ci": list(wilson(1, 2))}
    assert res["by_slice"]["train14"]["solved"] == 1
    assert res["by_slice"]["all"] == {"solved": 2, "n": 3, "rate": 2 / 3, "ci": list(wilson(2, 3))}
    assert res["ref_len_ge7"]["solved"] == 1 and res["ref_len_ge7"]["n"] == 1
    assert res["by_length"][-1]["n"] == 1  # unknown length bucket
    assert res["max_length_solved"] == 7  # unknown-length solves do not count
    table = textbook_table({"control": res, "rl": res})
    assert "| dev58 | 1 / 2 | 1 / 2 |" in table
    assert "| reference length >= 7 | 1 / 1 | 1 / 1 |" in table


class _Oracle(torch.nn.Module):
    """Predicts the true next token with logit 10 everywhere it sees one; used to test masking."""

    def __init__(self, vocab=99):
        super().__init__()
        self.vocab = vocab
        self.config = type("C", (), {"tok_mode": "abs"})()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, use_cache=False):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab)
        nxt = input_ids[:, 1:]
        logits[:, :-1].scatter_(2, nxt.unsqueeze(-1), 10.0)
        return type("O", (), {"logits": logits})()


def test_next_token_accuracy_masks_prompt_and_pads():
    ft = Tokenizer("abs")
    recs = [{"prompt": PROMPT, "proof": VALID}, {"prompt": "THM P SEQ P PRF", "proof": "N1 P : PR ; QED"}]
    res = next_token_accuracy(_Oracle(), ft, recs, batch_size=2)
    assert res["next_token_accuracy"] == 1.0
    assert res["n_target_tokens"] == sum(len(ft.encode_proof(r["proof"])) for r in recs)
    assert 0 < res["loss"] < 0.01
