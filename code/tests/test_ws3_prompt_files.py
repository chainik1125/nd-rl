"""Prompt and eval files for run.py (train.prompt.file, eval.long.file, periodic long_file)."""

import json
import os

import ndpath
import pytest

# isort: split
from data import eval_records, key_of, load_pool, periodic_long, read_prompt_file, train_prompts

HAVE = os.path.exists(os.path.join(ndpath.FORK_DATA, "heldout.jsonl"))


def _write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


@pytest.mark.skipif(not HAVE, reason="fork_pin data not fetched")
def test_train_prompts_file_and_disjointness(tmp_path):
    rl = load_pool("rl_targets")
    a = [dict(r, length=int(r["L_true"])) for r in rl[:10]]
    b = [dict(r, length=int(r["L_true"])) for r in rl[10:20]]
    _write(tmp_path / "a.jsonl", a)
    _write(tmp_path / "b.jsonl", b)
    recs = train_prompts({"file": "a.jsonl", "disjoint_from": ["b.jsonl"]}, 0, str(tmp_path))
    assert [r["key"] for r in recs] == [key_of(r) for r in a]
    with pytest.raises(ValueError):  # a training file overlapping its held-out half is refused
        train_prompts({"file": "a.jsonl", "disjoint_from": ["a.jsonl"]}, 0, str(tmp_path))
    held = [dict(r, length=int(r.get("n_lines", 2))) for r in load_pool("heldout")[:3]]
    _write(tmp_path / "h.jsonl", held)
    with pytest.raises(ValueError):  # evaluation sets are always banned
        train_prompts({"file": "h.jsonl"}, 0, str(tmp_path))


@pytest.mark.skipif(not HAVE, reason="fork_pin data not fetched")
def test_eval_records_and_periodic(tmp_path):
    rl = load_pool("rl_targets")
    _write(tmp_path / "e.jsonl", [dict(r, length=int(r["L_true"])) for r in rl[:30]])
    assert len(eval_records({"file": "e.jsonl"}, "rl_targets", str(tmp_path))) == 30
    assert len(eval_records({"n_prompts": 7}, "rl_targets", str(tmp_path))) == 7
    assert len(periodic_long({"n_long": 5, "long_file": "e.jsonl"}, str(tmp_path))) == 5
    with pytest.raises(ValueError):
        _write(tmp_path / "bad.jsonl", [{"prompt": "x"}])
        read_prompt_file(str(tmp_path / "bad.jsonl"))
