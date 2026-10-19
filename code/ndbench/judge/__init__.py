"""`Judge` implementations: adjudicate sampled `Attempt`s against a `Sequent`.

`nd.py` (WP0) judges the token-format ND proofs via the vendored `nd_verify`/`ndcheck`. `lean.py`
(WP7) will judge Lean-rendered proofs.

`JUDGES` is a lazy `"module:attr"` registry (resolved by `cli.py`), mirroring
`adapters.base.ADAPTERS`, so `cli.py` never has to import `judge.lean` (and whatever it needs to
shell out to `lean`) just to run the (already-implemented) `nd` judge.
"""

JUDGES: dict[str, str] = {
    "nd": "code.ndbench.judge.nd:NDJudge",
    "lean": "code.ndbench.judge.lean:LeanJudge",
}
