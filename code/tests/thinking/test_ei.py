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
    assert st == {
        "mix_rl_records": 4,
        "mix_retained": 3,
        "mix_targets": 1,
        "entries_forced_skipped": 0,
        "steps_dropped": 0,
    }
    assert sum(r == ei.entry_pairs(e, T) for r in mix) == 4


def test_build_mix_retain_zero_is_the_found_proofs_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """--retain 0 (line.sh's EI knob pilot ARM_kret0): no Stage-1 record, the same found records."""
    rec = pilot_rec("k12_37120")
    e = ei.found_entry(
        _accepted(rec, monkeypatch), T, {"name": "t", "prompt": rec["prompt"]}, 1, True
    )
    retained = [trace_pairs(pilot_rec("k12_16472"), T)] * 5
    mixes = [
        ei.build_mix([{"name": "t"}], {"t": [e]}, retained, T, random.Random(0), 4, 4, n, "off")
        for n in (0, 3)
    ]
    (mix0, st0), (mix3, st3) = mixes
    assert st0["mix_retained"] == 0 and st0["mix_rl_records"] == st3["mix_rl_records"] == 4
    assert mix0 == mix3[:4] == [ei.entry_pairs(e, T)] * 4


def _variant(e: dict[str, Any], think: str = "", action_step: int | None = None) -> dict[str, Any]:
    """The same entry with every think block replaced (a think variant of the same proof), or with
    one action changed (another proof)."""
    v = json.loads(json.dumps(e))
    for s in v["steps"]:
        s["think"] = think
    v["think_tokens"] = len(think.split()) * len(v["steps"])
    if action_step is not None:
        v["steps"][action_step]["action"] += " x"
    return v


def test_distinct_means_distinct_action_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = pilot_rec("k12_37120")
    e = ei.found_entry(
        _accepted(rec, monkeypatch), T, {"name": "t", "prompt": rec["prompt"]}, 1, True
    )
    assert e["forced"] == 0 and e["think_tokens"] > 0
    short = _variant(e)  # same actions, empty thinks
    assert ei.entry_key(short) == ei.entry_key(e) == tuple(s["action"] for s in e["steps"])
    other = _variant(e, action_step=2)
    # random: the first found variant stays
    have: dict[tuple[str, ...], int] = {}
    fs: list[dict[str, Any]] = []
    assert [ei.add_found(have, fs, x) for x in (e, short, other)] == ["new", "dup", "new"]
    assert fs == [e, other]
    # shortest: the variant with fewer think tokens replaces it, a longer one does not
    have, fs = {}, []
    got = [ei.add_found(have, fs, x, "shortest") for x in (e, short, _variant(e, "a b c d e f"))]
    assert got == ["new", "replaced", "dup"] and fs == [short]
    rep = ei.entries_report({"t": [e, other], "u": [short], "v": []})
    assert rep["all"]["targets"] == 2 and rep["all"]["mean"] == 1.5
    assert rep["all"]["hist"] == {"0": 0, "1": 1, "2": 1, "3": 0, ">=4": 0}


def test_forced_closes_are_counted_and_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = pilot_rec("k12_37120")
    x = _accepted(rec, monkeypatch)
    x["trace"][1]["forced"] = True
    x["trace"][4]["forced"] = True
    e = ei.found_entry(x, T, {"name": "t", "prompt": rec["prompt"]}, 1, True)
    assert e["forced"] == 2 and ei.n_forced(e) == 2
    retained = [trace_pairs(pilot_rec("k12_16472"), T)] * 5
    for forced, n_rl, skipped in (("drop", 0, 1), ("keep", 4, 0)):
        mix, st = ei.build_mix(
            [{"name": "t"}],
            {"t": [e]},
            retained,
            T,
            random.Random(0),
            4,
            4,
            3,
            "off",
            "random",
            forced,
        )
        assert st["mix_rl_records"] == n_rl and st["entries_forced_skipped"] == skipped
        assert st["mix_retained"] == 3
    # still solved; the entry stays in found but is not usable under drop
    rep = ei.entries_report({"t": [e]}, "drop")
    assert rep["all"]["targets"] == 1 and rep["usable"]["hist"]["0"] == 1
    assert rep["entries_forced"] == 1
    # a usable variant of the same proof replaces a forced one, even under `random`
    clean = dict(_variant(e, "a"), forced=0)
    have: dict[tuple[str, ...], int] = {}
    fs: list[dict[str, Any]] = []
    assert [ei.add_found(have, fs, y, "random", "drop") for y in (e, clean)] == ["new", "replaced"]
    have, fs = {}, []
    assert [ei.add_found(have, fs, y, "random", "keep") for y in (e, clean)] == ["new", "dup"]
    # records from before the `forced` field: a block of >= budget think tokens was forced
    legacy = {k: v for k, v in e.items() if k != "forced"}
    legacy["budget"] = 3
    legacy["steps"] = [
        dict(s, think=t)
        for s, t in zip(e["steps"][:4], ["a b c", "a b", "", "a b c d"], strict=True)
    ]
    assert ei.n_forced(legacy) == 2
    assert ei.n_forced(dict(legacy, budget=0)) == 0


def test_forced_policy_defaults() -> None:
    assert (
        ei.parser()
        .parse_args(["--init", "x", "--name", "n", "--out", "o", "--targets", "t", "--train", "p"])
        .forced
        is None
    )
    assert ei.forced_policy(None, effort=False) == "drop"
    assert ei.forced_policy(None, effort=True) == "keep"
    assert ei.forced_policy("drop", effort=True) == "drop"


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
    # k = 2 identical scripted proofs per target: one entry each, the second attempt a duplicate
    assert hist[0]["forced_policy"] == "drop"
    assert hist[0]["accepted_attempts"]["same_proof_kept_old"] == 2
    assert hist[0]["entries_per_target"]["all"]["hist"]["1"] == 2
    found = [
        json.loads(line) for line in (iso_env / "ei" / "found_1.jsonl").read_text().splitlines()
    ]
    assert len(found) == 2 and all(f["think_tokens"] > 0 for f in found)
    assert any(s["think"] for f in found for s in f["steps"])
    assert (iso_env / "ei" / "round_1.json").exists() and (iso_env / "ei" / "registry").is_dir()


def test_judge_marks_unfinished_attempts() -> None:
    res = [{"text": None, "nd": "LEANPARSE env: x"}]
    assert judge(T, ["THM P SEQ P PRF"], res, None)[0]["ok"] is False
