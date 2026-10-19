"""`ExclusionIndex` (plan §2's dedup tiers, §3 WP2 row)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from code.ndbench.core.types import Sequent
from code.ndbench.logic.exclusion import EVAL_FILES, TRAIN_FILES, ExclusionIndex, default_tiers


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_status_matches_train_and_flags_eval(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(
        train_path,
        [{"name": "t0", "thm": "P , ( P > Q ) |- Q", "key": "P , ( P > Q ) |- Q"}],
    )
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(
        dev_path,
        [{"name": "d0", "thm": "( ~ ( ~ R ) ) |- R", "key": "( ~ ( ~ P ) ) |- P"}],
    )

    index = ExclusionIndex.build({"train": [train_path], "dev": [dev_path]})

    in_train_seq = Sequent(premises=("Q", "( Q > R )"), conclusion="R")  # same class as t0
    in_train, eval_sets = index.status(in_train_seq)
    assert in_train is True
    assert eval_sets == ()

    dev_seq = Sequent(premises=(), conclusion="( ~ ( ~ Q ) )")  # wrong shape, not a match
    assert index.status(dev_seq) == (False, ())

    matching_dev_seq = Sequent(premises=("( ~ ( ~ S ) )",), conclusion="S")  # same class as d0
    in_train2, eval_sets2 = index.status(matching_dev_seq)
    assert in_train2 is False
    assert eval_sets2 == ("dev",)

    unrelated = Sequent(premises=("P", "Q"), conclusion="( P & Q )")
    assert index.status(unrelated) == (False, ())


def test_pkey_matches_regardless_of_premise_order(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(
        train_path,
        [{"name": "t0", "thm": "P , Q |- ( P & Q )", "key": "P , Q |- ( P & Q )"}],
    )
    index = ExclusionIndex.build({"train": [train_path]})

    reordered = Sequent(premises=("Q", "P"), conclusion="( Q & P )")
    in_train, _ = index.status(reordered)
    assert in_train is True  # matches by pkey even though the premise order/key differ


def test_pkey_skipped_above_five_premises(tmp_path: Path) -> None:
    """A >5-premise training row is never used for pkey matching (plan §2: "no training theorem
    has more than 5"); a query sequent with >5 premises likewise never attempts a pkey lookup."""
    atoms6 = ["P", "Q", "R", "S", "T", "U"]
    thm = " , ".join(atoms6) + " |- P"
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, [{"name": "t0", "thm": thm, "key": thm}])
    index = ExclusionIndex.build({"train": [train_path]})
    assert index._train.pkeys == frozenset()  # no pkey was computed for the 6-premise row

    same_shape_reordered = Sequent(premises=tuple(reversed(atoms6)), conclusion="P")
    in_train, _ = index.status(same_shape_reordered)
    assert in_train is False  # key differs (order matters) and pkey lookup was skipped


def test_row_key_pkey_fallback_without_precomputed_fields(tmp_path: Path) -> None:
    """`train_cap6.jsonl`/`heldout_cap6.jsonl` predate the `pkey` field and only carry `thm`, not
    `key`/`pkey`; the index must still compute both correctly."""
    train_path = tmp_path / "legacy_train.jsonl"
    _write_jsonl(train_path, [{"name": "t0", "thm": "P , Q |- ( P & Q )"}])
    index = ExclusionIndex.build({"train": [train_path]})
    seq = Sequent(premises=("Q", "P"), conclusion="( Q & P )")
    in_train, _ = index.status(seq)
    assert in_train is True


def test_build_cache_roundtrip(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, [{"name": "t0", "thm": "P |- P", "key": "P |- P"}])
    cache_dir = tmp_path / "cache"

    index1 = ExclusionIndex.build({"train": [train_path]}, cache_dir=cache_dir)
    cache_files = list(cache_dir.glob("exclusion_*.json"))
    assert len(cache_files) == 1

    # The second build must not re-scan train_path at all: monkeypatch the only function that
    # would read it and confirm it is never called.
    import code.ndbench.logic.exclusion as exclusion_mod

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a cache hit must not re-read any tier file")

    original = exclusion_mod.read_jsonl
    exclusion_mod.read_jsonl = boom  # type: ignore[assignment]
    try:
        index2 = ExclusionIndex.build({"train": [train_path]}, cache_dir=cache_dir)
    finally:
        exclusion_mod.read_jsonl = original  # type: ignore[assignment]

    probe = Sequent(premises=("R",), conclusion="R")  # same class as "P |- P"
    assert index2.status(probe) == index1.status(probe) == (True, ())


def test_marks_a_known_train_cap6_theorem(tmp_path: Path) -> None:
    """Acceptance test: exclusion marks a known `train_cap6` theorem. Reads only the first 2,000
    rows of the real (large) `ab3/frozen/data/train_cap6.jsonl` into a temp slice, to keep this
    fast while still exercising the real file's actual schema."""
    real_path = TRAIN_FILES[0]
    assert real_path.name == "train_cap6.jsonl"
    if not real_path.exists():
        import pytest

        pytest.skip(f"{real_path} not available in this environment")

    rows = []
    with open(real_path) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) == 2000

    slice_path = tmp_path / "train_cap6_slice.jsonl"
    _write_jsonl(slice_path, rows)

    index = ExclusionIndex.build({"train": [slice_path]})
    known_row = rows[7]
    seq = Sequent.from_thm(known_row["thm"])
    in_train, _ = index.status(seq)
    assert in_train is True

    definitely_absent = Sequent(
        premises=("( P > ( Q > ( R > S ) ) )",), conclusion="( ( P & Q ) > ( R > S ) )"
    )
    if definitely_absent.key() not in index._train.keys:
        assert index.status(definitely_absent)[0] is False


def test_default_tiers_layout_and_paths_exist() -> None:
    tiers = default_tiers()
    assert "train" in tiers
    assert set(tiers) == {"train", *EVAL_FILES}
    for label, paths in tiers.items():
        for path in paths:
            assert path.exists(), f"{label}: {path} does not exist"


def test_default_tiers_arm_cd_adds_files() -> None:
    without = default_tiers(arm_cd=False)
    with_cd = default_tiers(arm_cd=True)
    assert len(with_cd["train"]) == len(without["train"]) + 2


def test_status_returns_eval_sets_sorted(tmp_path: Path) -> None:
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    row = {"name": "x", "thm": "P |- P", "key": "P |- P"}
    _write_jsonl(a_path, [row])
    _write_jsonl(b_path, [row])
    index = ExclusionIndex.build({"zeta": [a_path], "alpha": [b_path]})
    _, eval_sets = index.status(Sequent(premises=("Q",), conclusion="Q"))
    assert eval_sets == ("alpha", "zeta")  # sorted, not insertion order


def test_fingerprint_changes_when_a_tier_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"name": "t0", "thm": "P |- P", "key": "P |- P"}])
    cache_dir = tmp_path / "cache"
    ExclusionIndex.build({"train": [path]}, cache_dir=cache_dir)
    first_cache_files = set(cache_dir.glob("exclusion_*.json"))

    # A different file *size* changes the fingerprint deterministically (mtime alone has only
    # whole-second resolution in `_fingerprint`, too coarse for a fast test).
    _write_jsonl(
        path,
        [
            {
                "name": "t0_with_a_much_longer_name_to_change_the_byte_size",
                "thm": "Q |- Q",
                "key": "P |- P",
            }
        ],
    )
    ExclusionIndex.build({"train": [path]}, cache_dir=cache_dir)
    second_cache_files = set(cache_dir.glob("exclusion_*.json"))
    assert second_cache_files != first_cache_files  # a new fingerprint -> a new cache file


def test_no_permutation_explosion_within_bound(tmp_path: Path) -> None:
    """Sanity check that `pkey` matching remains usable (not accidentally O((n!)) per query in a
    way that would make a 5-premise lookup impractical): 5! = 120 permutations is fine."""
    atoms5 = ["P", "Q", "R", "S", "P"]
    thm = " , ".join(atoms5) + " |- P"
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"name": "t0", "thm": thm, "key": thm}])
    index = ExclusionIndex.build({"train": [path]})
    for perm in itertools.islice(itertools.permutations(atoms5), 10):
        seq = Sequent(premises=tuple(perm), conclusion="P")
        in_train, _ = index.status(seq)
        assert in_train is True
