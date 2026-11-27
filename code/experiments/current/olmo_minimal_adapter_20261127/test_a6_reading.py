"""a6_reading.py: step-2 readings in the order 1, 2, 4, 3, 5 and the replicate check."""

import a6_reading as ar


def rows(rates: dict[str, float], half: float = 8.0) -> list[dict]:
    """Contrast rows from corner rates, each with a +-half interval (points)."""
    out = []
    for a, b in ar.PAIRS:
        d = rates[a] - rates[b]
        out.append(
            {"a": ar.CORNERS[a], "b": ar.CORNERS[b], "diff": d, "lo": d - half, "hi": d + half}
        )
    return out


def test_only_corner_a_high_reads_combination_not_both():
    # audit T4: A above B, C, D, which are alike: pool A-C and seed A-B are both above 0, yet the
    # reading must be 4 ("combination only"), because 4 is checked before 3
    r = ar.read_all(rows({"A": 65, "A'": 65, "B": 41, "C": 41, "D": 41}))
    assert r["primary"]["reading"] == "4 combination only"
    assert r["secondary_25nov_adapter_as_A"]["reading"] == "4 combination only"
    assert r["primary"]["seed_X"] is None


def test_pool_seed_both_unresolved():
    assert ar.read_all(rows({"A": 65, "A'": 65, "B": 65, "C": 41, "D": 41}))["primary"][
        "reading"
    ] == ("1 pool")
    assert ar.read_all(rows({"A": 65, "A'": 65, "B": 41, "C": 65, "D": 41}))["primary"][
        "reading"
    ] == ("2 seed")
    both = ar.read_all(rows({"A": 75, "A'": 75, "B": 55, "C": 55, "D": 35}))
    assert both["primary"]["reading"] == "3 both"
    assert both["primary"]["seed_X"] == 20
    assert ar.read_all(rows({"A": 45, "A'": 45, "B": 41, "C": 41, "D": 41}))["primary"][
        "reading"
    ] == ("5 unresolved")


def test_replicate_not_reproduced_reports_both_squares_and_sets_x():
    # the 25 Nov adapter is high, our replicate of it is not: primary (A') reads null-ish, secondary
    # (A) reads combination only, and |A'-A| becomes the run-to-run spread X
    r = ar.read_all(rows({"A": 65, "A'": 41, "B": 41, "C": 41, "D": 41}))
    assert r["replicate"]["A_reproduced"] is False
    assert r["primary"]["reading"] == "5 unresolved"
    assert r["secondary_25nov_adapter_as_A"]["reading"] == "4 combination only"
    assert r["readings_agree"] is False
    assert r["relabel_X"] == 24


def test_contrasts_arg_has_no_extra_hyphen():
    arg = ar.contrasts_arg()
    assert all(s.count("-") == 1 for s in arg.split(","))
    assert len(arg.split(",")) == 10
