"""`core.keys` matches the vendored `canon_key`/`pkey` implementations exactly (WP0 acceptance
test: "Sequent round-trip prompt<->thm and matches vendored canon_key")."""

from __future__ import annotations

from code.ndbench.core.keys import canon_key, pkey, prompt_to_thm, split_thm
from code.ndbench.vendor import ensure_takehome_on_path, load_absgpt_module

from ._fixtures import load_valid_proofs


def test_canon_key_matches_vendored_absgpt_keys() -> None:
    keys_mod = load_absgpt_module("keys")
    for p in load_valid_proofs():
        assert canon_key(p["thm"]) == keys_mod.canon_key(p["thm"])


def test_canon_key_matches_vendored_gen() -> None:
    ensure_takehome_on_path()
    import gen  # vendored takehome/gen.py

    for p in load_valid_proofs():
        assert canon_key(p["thm"]) == gen.canon_key(p["thm"])


def test_prompt_to_thm_matches_vendored_common() -> None:
    ensure_takehome_on_path()
    import common  # vendored lengthexp/common.py

    for p in load_valid_proofs():
        rec = {"prompt": p["prompt"]}
        assert prompt_to_thm(p["prompt"]) == common.thm_of(rec)


def test_pkey_matches_vendored_common() -> None:
    ensure_takehome_on_path()
    import common  # vendored lengthexp/common.py

    for p in load_valid_proofs():
        if (p.get("n_prem") or 0) > 5:
            continue  # pkey is O(n!); the plan's own dedup policy skips this range too
        assert pkey(p["thm"]) == common.pkey(p["thm"])


def test_split_thm_roundtrip() -> None:
    for p in load_valid_proofs():
        prem, concl = split_thm(p["thm"])
        rebuilt = (" , ".join(prem) + " |- " + concl) if prem else ("|- " + concl)
        assert rebuilt == p["thm"]
