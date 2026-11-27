import hashlib
import sys
from pathlib import Path

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import temp_repeat_flag as rf  # noqa: E402


def test_frozen_sha():
    want = (X / "temp" / "FROZEN_repeat_flag.sha256").read_text().split()[0]
    got = hashlib.sha256((X / "tools" / "temp_repeat_flag.py").read_bytes()).hexdigest()
    assert got == want, "the detector was edited after being frozen"


def test_flags_repeated_sentence_in_thinking():
    t = (
        "Let me think. "
        + "I should double check the constraint again and again. " * 12
        + "</think>Answer."
    )
    assert rf.flag_text(t)


def test_does_not_flag_repetition_in_the_answer():
    t = (
        "Short reasoning here.\n</think>"
        + "Repeat this exact sentence as requested by the prompt. " * 30
    )
    assert not rf.flag_text(t)


def test_ngram_rule():
    block = " ".join(f"w{i}" for i in range(10))
    t = (block + " ") * 60 + "</think>x"
    assert rf.flag_text(t)


def test_plain_text_not_flagged_and_no_think_uses_whole_text():
    assert not rf.flag_text("one two three. four five six seven eight nine ten eleven. " * 3)
    assert rf.flag_text(
        "This is a long repeated line number one.\n" * 11
    )  # no </think>: whole text is thinking


def test_short_units_ignored():
    assert not rf.flag_text(
        " ".join(f"Sure ok {i}." for i in range(100)) + "</think>"
    )  # short units, distinct 10-grams
    assert rf.flag_text(
        "ok. " * 100 + "</think>"
    )  # but a loop of short units is caught by the word 10-gram rule
