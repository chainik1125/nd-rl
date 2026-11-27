"""dev_check.py: draws, rates, bootstrap bounds, eligibility and the winner rule (audit T3)."""

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


PIDS = [f"p{i}" for i in range(32)]
PLAIN = {p: [1, 0] * 8 for p in PIDS}  # 16 plain draws per prompt, 50 %


def half():
    """4 draws per prompt, 2 passes each: exactly 50 %."""
    return {p: [1, 0, 1, 0] for p in PIDS}


def test_missing_draw_cannot_win_a_tie():
    # audit T3's case: the second candidate lost one failing draw, so its per-prompt mean is
    # 50.5 % against 50.0 %;
    # before the fix it won what is an exact tie the first candidate wins
    second = half()
    second["p0"] = [1, 0, 1]
    assert dc.rate(second, PIDS) > dc.rate(half(), PIDS)
    out = dc.evaluate([("first", half()), ("second", second)], PLAIN, PIDS, 4, 200)
    st = {r["cell"]: r["status"] for r in out["candidates"]}
    assert st == {"first": "ok", "second": "incomplete"}
    assert out["winner"] == "first"
    assert out["candidates"][1]["prompts_with_wrong_count"] == {"p0": 3}


def test_unsampled_candidate_is_skipped_and_tie_goes_to_earlier():
    empty = {p: [] for p in PIDS}
    out = dc.evaluate([("gate_failed", empty), ("a", half()), ("b", half())], PLAIN, PIDS, 4, 200)
    assert out["candidates"][0]["status"] == "not sampled"
    assert out["winner"] == "a"  # exact tie between a and b: the earlier-listed one
    better = half()
    better["p5"] = [1, 1, 1, 0]
    assert dc.evaluate([("a", half()), ("b", better)], PLAIN, PIDS, 4, 200)["winner"] == "b"


def test_no_eligible_candidate_gives_no_winner():
    empty = {p: [] for p in PIDS}
    out = dc.evaluate([("x", empty)], PLAIN, PIDS, 4, 200)
    assert out["winner"] is None


def test_unsampled_dir_reads_as_empty(tmp_path):
    assert dc.cell_draws(tmp_path / "missing", ["a"]) == {"a": []}
