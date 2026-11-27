"""dev_check.py: draw parsing, the per-prompt-mean rate and the bootstrap bounds."""

import json

import dev_check as dc


def test_cell_draws_and_rate(tmp_path):
    rows = [
        {"pid": "a", "strict": True},
        {"pid": "a", "strict": False},
        {"pid": "b", "strict": True},
    ]
    rows += [{"pid": "zz", "strict": True}]  # not a dev prompt: ignored
    (tmp_path / "graded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = dc.cell_draws(tmp_path, ["a", "b"])
    assert c == {"a": [1, 0], "b": [1]}
    assert dc.rate(c, ["a", "b"]) == 0.75  # mean over prompts of per-prompt rates


def test_boot_bounds_bracket_and_paired_zero():
    a = {p: [1, 0, 1, 0] for p in "abcdef"}
    lo, hi = dc.boot(a, None, list(a), 500, 1)
    assert lo <= 0.5 <= hi
    dlo, dhi = dc.boot(a, a, list(a), 500, 2)
    assert dlo < 0 < dhi  # same arm against itself: the interval straddles 0
