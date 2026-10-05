"""Expert iteration keeps the sampled think tokens of accepted attempts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from code.thinking import ei
from code.thinking.data import trace_pairs
from code.thinking.sample import judge, think_generate
from code.thinking.tok import D, ThinkTokenizer

from ._pipeline_helpers import (
    ScriptedModel,
    isolate,
    needs_lean,
    pilot_rec,
    script_from_pairs,
)

T = ThinkTokenizer()


def _accepted(rec: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    model = ScriptedModel(T, script_from_pairs(trace_pairs(rec, T)))
    (x,) = think_generate(model, T, [rec["prompt"]], "plain", budgets=256, max_action=64, batch=1)
    assert x["status"] == "done"
    x["ok"] = True
    return x


def test_found_entry_keeps_the_think_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = pilot_rec("k12_22530")
    x = _accepted(rec, monkeypatch)
    e = ei.found_entry(x, T, {"name": "t0", "prompt": rec["prompt"]}, 1, think=True)
    pairs = trace_pairs(rec, T)
    assert e["think_tokens"] == sum(tl - 2 for _, _, tl in pairs) > 0
    assert [s["think"] for s in e["steps"]] == [
        " ".join(T.itos[i] for i in q[1 : tl - 1]) for _, q, tl in pairs
    ]
    assert ei.entry_pairs(e, T) == pairs  # the fine-tune pairs are the sampled sequence
    assert json.loads(json.dumps(e)) == e


def test_rebase_to_name_base_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = pilot_rec("k12_37120")
    x = _accepted(rec, monkeypatch)
    e0 = ei.found_entry(x, T, {"name": "t", "prompt": rec["prompt"]}, 1, think=True)

    def up(b: bytes) -> bytes:
        return bytes(i + 7 if i >= T.ref0 else i for i in b)

    shifted = dict(
        x,
        base=7,
        trace=[
            {k: (up(v) if isinstance(v, bytes) else v) for k, v in s.items()} for s in x["trace"]
        ],
    )
    e7 = ei.found_entry(shifted, T, {"name": "t", "prompt": rec["prompt"]}, 1, think=True)
    assert e7["steps"] == e0["steps"] and e7["base"] == 0 and e7["sample_base"] == 7


def test_validate_think_and_the_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = pilot_rec("k12_37120")
    e = ei.found_entry(
        _accepted(rec, monkeypatch), T, {"name": "t", "prompt": rec["prompt"]}, 1, True
    )
    assert ei.validate_entry(e) == [True] * len(e["steps"])
    bad = json.loads(json.dumps(e))
    bad["steps"][3]["think"] = "n1 : S"  # a false look
    flags = ei.validate_entry(bad)
    assert flags[3] is False and sum(flags) == len(flags) - 1
    bad["think_valid"] = flags
    full = ei.entry_pairs(bad, T)
    dropped = ei.entry_pairs(bad, T, "drop")
    emptied = ei.entry_pairs(bad, T, "empty")
    assert len(dropped) == len(full) - 1 and len(emptied) == len(full)
    assert emptied[3][2] == 2 and full[3][2] > 2  # empty block keeps the action


def test_build_mix_weights_found_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = pilot_rec("k12_37120")
    e = ei.found_entry(
        _accepted(rec, monkeypatch), T, {"name": "t", "prompt": rec["prompt"]}, 1, True
    )
    retained = [trace_pairs(pilot_rec("k12_16472"), T)] * 5
    mix, st = ei.build_mix(
        [{"name": "t"}], {"t": [e]}, retained, T, random.Random(0), 4, 4, 3, "off"
    )
    assert st == {"mix_rl_records": 4, "mix_retained": 3, "steps_dropped": 0}
    assert sum(r == ei.entry_pairs(e, T) for r in mix) == 4


@needs_lean
def test_ei_round_writes_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One frozen round (--no_train) with a model that writes the pilot trace."""
    iso_env = isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    recs = [pilot_rec("k12_37120"), pilot_rec("k12_16472")]
    script: dict[bytes, list[int]] = {}
    for r in recs:
        script.update(script_from_pairs(trace_pairs(r, T)))
    model = ScriptedModel(T, script)
    monkeypatch.setattr(ei, "load_ckpt", lambda path, dev: (model, T, {"think_format": "think"}))
    targets = iso_env / "targets.jsonl"
    targets.write_text(
        "".join(json.dumps({"name": r["id"], "prompt": r["prompt"]}) + "\n" for r in recs)
    )
    a = ei.parser().parse_args(
        [
            "--init",
            "x.pt",
            "--name",
            "s",
            "--out",
            str(iso_env / "ei"),
            "--targets",
            str(targets),
            "--train",
            "unused",
            "--rounds",
            "1",
            "--k",
            "2",
            "--budgets",
            "256",
            "--no_train",
            "--max_action",
            "64",
            "--batch",
            "2",
            "--validate_think",
            "drop",
            "--device",
            "cpu",
        ]
    )
    hist = ei.run(a)
    assert hist[0]["targets_round"]["solved"] == 2 and hist[0]["new_attempts"] == 2
    assert hist[0]["think_valid_frac"] == 1.0
    found = [
        json.loads(line) for line in (iso_env / "ei" / "found_1.jsonl").read_text().splitlines()
    ]
    assert len(found) == 2 and all(f["think_tokens"] > 0 for f in found)
    assert any(s["think"] for f in found for s in f["steps"])
    assert (iso_env / "ei" / "round_1.json").exists() and (iso_env / "ei" / "registry").is_dir()


def test_judge_marks_unfinished_attempts() -> None:
    res = [{"text": None, "nd": "LEANPARSE env: x"}]
    assert judge(T, ["THM P SEQ P PRF"], res, None)[0]["ok"] is False
