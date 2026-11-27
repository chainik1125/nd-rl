"""capacity.t64_from_runs: SPEC's step-64 lr rule (lowest dev loss at 64 among lrs that reached
64, including runs early-stopped later) and the difficulty-matched correction."""

import json
import math

import pytest
from capacity import t64_from_runs


def _row(lr, dev64, es, nats=(1.0, 2.0), toks=(10, 20)):
    return {
        "lr": lr,
        "early_stopped_at": es,
        "online": {
            "pids": ["a", "b"],
            "think_nats": list(nats),
            "answer_nats": [0.0, 0.0],
            "n_tok": list(toks),
        },
        "dev": {"0": {"loss_total": 0.5}, "64": {"loss_total": dev64}},
    }


def test_t64_rule_keeps_late_early_stops(tmp_path):
    p = tmp_path / "runs.jsonl"
    rows = [
        _row(0.1, 0.06, None),
        _row(0.3, 0.05, 512),  # early-stopped after 64: still counts
        _row(1.0, 0.01, 32),  # stopped before 64: excluded
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s0 = {
        "think_nats": [0.5, 1.0],
        "answer_nats": [0.0, 0.0],
        "n_tok": [10, 20],
        "L0_dev_total": 0.1,
    }
    raw, corr = t64_from_runs(p, s0, P=100)
    assert raw["lr"] == 0.3
    nats, toks = 3.0, 30
    assert raw["edl_tok_bits"] == pytest.approx((nats - toks * 0.05) / math.log(2))
    delta = 1.5 / 30 - 0.1
    assert corr["delta_nats_per_tok"] == pytest.approx(delta)
    assert corr["edl_tok_corr_bits"] == pytest.approx(
        (nats - toks * 0.05 - toks * delta) / math.log(2)
    )


def test_t64_uses_last_record_per_lr(tmp_path):
    p = tmp_path / "runs.jsonl"
    rows = [_row(0.3, 0.09, None), _row(0.3, 0.04, 512)]  # the extended record replaces the first
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s0 = {
        "think_nats": [0.5, 1.0],
        "answer_nats": [0.0, 0.0],
        "n_tok": [10, 20],
        "L0_dev_total": 0.1,
    }
    raw, _ = t64_from_runs(p, s0, P=100)
    assert raw["dev_nats_per_tok_total"] == 0.04
