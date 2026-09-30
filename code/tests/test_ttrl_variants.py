"""TTRL variant generator: truth tables, prompt round trip, and the variants' validity."""

import importlib.util
from pathlib import Path

VARIANTS = Path(__file__).parents[1] / "experiments/current/ttrl/variants.py"
spec = importlib.util.spec_from_file_location("ttrl_variants", VARIANTS)
assert spec is not None and spec.loader is not None
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_entails_is_classical():
    peirce = V.parse("( ( ( P > Q ) > P ) > P )")  # classical, not intuitionistic
    assert V.entails([], peirce)
    assert V.entails([V.parse("( P > Q )"), "P"], "Q")
    assert not V.entails([V.parse("( P > Q )"), "Q"], "P")
    assert V.entails(["P", V.parse("( ~ P )")], "F")


def test_prompt_round_trip():
    thm = "( P > Q ) , ( Q > R ) |- ( P > R )"
    prems, goal = V.parse_thm(thm)
    assert V.prompt(prems, goal) == "THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF"
    assert V.prompt([], goal) == "THM SEQ ( P > R ) PRF"


def test_variants_are_valid_new_and_nontrivial():
    prems, goal = V.parse_thm("( P > Q ) , ( Q > R ) , ( R > S ) |- ( P > S )")
    vs = V.variants(prems, goal, depth=2)
    keys = [V.key(v["prems"], v["goal"]) for v in vs]
    assert len(set(keys)) == len(keys)
    assert V.key(prems, goal) not in keys
    for v in vs:
        assert V.entails(v["prems"], v["goal"])
        assert v["goal"] not in v["prems"]
    by_kind = {(v["kind"], V.prompt(v["prems"], v["goal"])) for v in vs}
    assert ("dec_impi", "THM ( P > Q ) , ( Q > R ) , ( R > S ) , P SEQ S PRF") in by_kind
    assert ("spec", "THM Q , ( Q > R ) , ( R > S ) SEQ ( P > S ) PRF") in by_kind
    # Γ |= R is false (P could be false), so R is never proposed as a lemma of the target
    assert not any(v["kind"] == "cut_left" and v["goal"] == "R" for v in vs)
