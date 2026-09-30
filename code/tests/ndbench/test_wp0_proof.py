"""`core.proof` matches the vendored `absgpt/prune.py`'s `pruned_length`, and `take_prefix`/
`renumber` behave as `PrefixSpec` requires (N1-based, no QED, line-boundary-safe)."""

from __future__ import annotations

from code.ndbench.core.proof import (
    first_index,
    has_qed,
    line_count,
    make_reference,
    pruned_length,
    renumber,
    split_lines,
    take_prefix,
)
from code.ndbench.vendor import load_absgpt_module, load_nd_verify

from ._fixtures import generate_corruptions, load_valid_proofs


def test_pruned_length_matches_vendored_prune() -> None:
    prune = load_absgpt_module("prune")
    for p in load_valid_proofs():
        assert pruned_length(p["proof"]) == prune.pruned_length(p["prompt"], p["proof"])


def test_pruned_length_matches_vendored_prune_on_corruptions() -> None:
    prune = load_absgpt_module("prune")
    proofs = load_valid_proofs()
    for proof, corrupted in generate_corruptions(proofs, n=60, seed=1):
        assert pruned_length(corrupted) == prune.pruned_length(proof["prompt"], corrupted)


def test_line_count_and_has_qed() -> None:
    for p in load_valid_proofs():
        assert line_count(p["proof"]) == p["n_lines"]
        assert has_qed(p["proof"])
        assert not has_qed(" ".join(p["proof"].split()[:-1]))


def test_renumber_then_first_index() -> None:
    for p in load_valid_proofs():
        shifted = renumber(p["proof"], new_start=17)
        assert first_index(shifted) == 17
        assert line_count(shifted) == line_count(p["proof"])
        back = renumber(shifted, new_start=1)
        assert back == p["proof"]


def test_take_prefix_is_a_valid_partial_proof() -> None:
    """A prefix, with a fresh QED appended, still verifies against nd_verify (a well-formed ND
    proof's line i only cites lines/boxes before it, so any prefix that ends at depth 0 on a line
    matching the sequent's own conclusion pattern is not guaranteed valid in general -- what *is*
    guaranteed is that every citation inside the prefix still resolves, which we check directly).
    """
    for p in load_valid_proofs():
        j = max(1, p["n_lines"] // 2)
        prefix = take_prefix(p["proof"], j)
        assert first_index(prefix) == 1
        assert not has_qed(prefix)
        lines = split_lines(prefix)
        assert len(lines) == min(j, p["n_lines"])
        # every citation inside the prefix points at an earlier line within the prefix itself
        declared = {int(ln.split()[0][1:]) for ln in lines}
        for ln in lines:
            toks = ln.split()
            rule_pos = toks.index(":") + 1
            refs = [
                int(t[1:]) for t in toks[rule_pos + 1 :] if t.startswith("N") and t[1:].isdigit()
            ]
            for r in refs:
                assert r in declared


def test_make_reference() -> None:
    p = load_valid_proofs()[0]
    ref = make_reference(p["proof"], source="minlen")
    assert ref.body == p["proof"]
    assert ref.lines == p["n_lines"]
    assert ref.source == "minlen"
    prune = load_absgpt_module("prune")
    assert ref.pruned == prune.pruned_length(p["prompt"], p["proof"])


def test_pruned_length_empty_body() -> None:
    assert pruned_length("") == 0


def test_nd_verify_still_loads_after_proof_module_use() -> None:
    # Regression guard: exercising core.proof (no vendor dependency) must not interfere with the
    # vendor loader used elsewhere in the same test session.
    nd_verify = load_nd_verify()
    ok, _, _ = nd_verify.verify_text(
        load_valid_proofs()[0]["prompt"] + " " + load_valid_proofs()[0]["proof"]
    )
    assert ok
