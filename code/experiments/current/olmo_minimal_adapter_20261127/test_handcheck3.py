"""handcheck3.py: kappa, majority labels, batch prompts without corner names, the bootstrap."""

import handcheck3 as h3


def test_kappa_and_agreement():
    pairs = [("genuine", "genuine")] * 6 + [("gamed", "gamed")] * 2 + [("genuine", "gamed")] * 2
    po, k = h3.kappa(pairs)
    assert po == 0.8
    assert 0.4 < k < 0.6  # pe = (8*6 + 2*4) / 100 = 0.56 -> k = 0.24 / 0.44


def test_majority_label_needs_two_agreeing_reads():
    assert h3.label({"a": "gamed", "b": "gamed"}) == "gamed"
    assert h3.label({"a": "gamed", "b": "genuine"}) == "unread"
    assert h3.label({"a": "gamed", "b": "genuine", "c": "genuine"}) == "genuine"
    assert h3.label({"a": "genuine"}) == "unread"


def test_batch_prompt_is_blind(tmp_path):
    (tmp_path / "items").mkdir()
    txt = h3.batch_prompt(tmp_path, "a00", ["h1234567", "habcdef0"])
    for word in ("Aprime", "corner", "OLD", "final25", "s523373504", "batch_a6s2", "key.csv"):
        assert word not in txt
    assert "h1234567.md" in txt and "reads/a00.json" in txt


def test_bootstrap_genuine_rate_brackets_point():
    strict = {"X": {f"p{i}": [1, 0] * 4 for i in range(20)}}
    labs = {"X": ["genuine"] * 3 + ["gamed"]}
    bs = h3.boot_genuine(strict, {"X": 0.0}, labs, reps=400, seed=1)
    lo, hi = h3.pct(bs["X"])
    assert lo < 0.5 * 0.75 < hi
