"""`code.ndbench.edl.build_classes`: suite/tag assembly, the 64-vs-32 train/test split rule,
dedup-before-threshold, and "missing source -> skip with a warning" (plan §B7 / §3 WP5 row).

None of this needs a real WP2/WP3/WP4 suite (none is committed on this branch yet): every suite
here is a small, synthetic `Store` this file builds itself, and the two file-backed classes (`gen`
L10, heldout cap-6) read small temp JSONL fixtures rather than the real (multi-MB, external,
read-only) `ab2`/`ab3` files -- `code.ndbench.edl`'s module docstring documents the tag/suite
assumptions this exercises.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.types import (
    JSON,
    LengthLabel,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.edl import EDL_CLASSES, build_classes
from code.ndbench.logic.exclusion import default_tiers
from code.ndbench.store import Store

_MISSING = "/nonexistent/does-not-exist.jsonl"


def _manifest(suite: str, version: str, n: int) -> SuiteManifest:
    return SuiteManifest(
        suite=suite,
        version=version,
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=n,
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )


def _synthetic_problem(
    suite: str, i: int, tags: dict[str, JSON], *, length_l: int | None = 1
) -> Problem:
    seq = Sequent(premises=(), conclusion=f"SYN_{suite}_{i}")
    ref = ReferenceProof(body="N1 P : PR ; QED", lines=1, pruned=1, source="constructed")
    return Problem(
        pid=f"{suite}:{i}",
        suite=suite,
        version="v1",
        sequent=seq,
        length=LengthLabel(L=length_l, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
        ref=ref,
        tags=tags,
    )


def _write_suite(store: Store, suite: str, problems: list[Problem]) -> None:
    store.write_suite(problems, _manifest(suite, "v1", len(problems)))


def _class_counts(stats: dict[str, JSON]) -> dict[str, JSON]:
    """`stats["class_counts"]`, narrowed from `JSON` back to a dict (`build_classes` always puts a
    `dict[str, JSON]` there; `dict`/`list` are invariant, so pyright can't see that on its own --
    see `edl.py`'s `_j` helper for the same issue on the writing side)."""
    return cast(dict[str, JSON], stats["class_counts"])


def _warnings(stats: dict[str, JSON]) -> list[str]:
    return cast(list[str], stats["warnings"])


def test_missing_store_and_missing_files_skip_every_class_with_warnings(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    problems, stats = build_classes({}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING)
    assert problems == []
    assert stats["n_classes_built"] == 0
    assert stats["n_classes_total"] == len(EDL_CLASSES) == 12
    assert len(_warnings(stats)) == 12


def test_suite_class_below_128_uses_32_32_split(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    n8 = [_synthetic_problem("families", i, {"family": "and_tower", "n": 8}) for i in range(70)]
    n16 = [
        _synthetic_problem("families", 1000 + i, {"family": "and_tower", "n": 16}) for i in range(5)
    ]
    _write_suite(store, "families", n8 + n16)

    problems, stats = build_classes({}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING)

    assert _class_counts(stats)["and_tower_n8"] == {"n_train": 32, "n_test": 32}
    assert "and_tower_n16" not in _class_counts(stats)
    assert any("and_tower_n16" in w and "only 5" in w for w in _warnings(stats))

    class_problems = [p for p in problems if p.tags.get("class") == "and_tower_n8"]
    assert len(class_problems) == 64
    assert {cast(str, p.tags["split"]) for p in class_problems} == {"train", "test"}
    assert sum(1 for p in class_problems if p.tags["split"] == "train") == 32
    assert sum(1 for p in class_problems if p.tags["split"] == "test") == 32
    # every returned item still has its reference proof and the original family/n tags.
    assert all(p.ref is not None for p in class_problems)
    assert all(p.tags["family"] == "and_tower" and p.tags["n"] == 8 for p in class_problems)
    # no overlap between the two sides.
    train_pids = {p.pid for p in class_problems if p.tags["split"] == "train"}
    test_pids = {p.pid for p in class_problems if p.tags["split"] == "test"}
    assert train_pids.isdisjoint(test_pids)


def test_suite_class_at_or_above_128_uses_64_64_split(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    items = [
        _synthetic_problem("laws", i, {"schema": "contraposition", "s": 3}) for i in range(130)
    ]
    _write_suite(store, "laws", items)

    problems, stats = build_classes({}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING)

    assert _class_counts(stats)["contraposition_s3"] == {"n_train": 64, "n_test": 64}
    class_problems = [p for p in problems if p.tags.get("class") == "contraposition_s3"]
    assert len(class_problems) == 128


def test_dedup_happens_before_the_64_item_threshold(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    distinct = [
        _synthetic_problem("families", i, {"family": "reorder_and", "n": 5}) for i in range(60)
    ]
    # 10 exact duplicates (same sequent -> same Sequent.key()) of the first 10 distinct items.
    dupes = [
        _synthetic_problem("families", 500 + i, {"family": "reorder_and", "n": 5})
        for i in range(10)
    ]
    for i, d in enumerate(dupes):
        dupes[i] = Problem(
            pid=f"families:dupe:{i}",
            suite=d.suite,
            version=d.version,
            sequent=distinct[i].sequent,  # identical sequent -> identical key()
            length=d.length,
            ref=d.ref,
            tags=d.tags,
        )
    _write_suite(store, "families", distinct + dupes)

    problems, stats = build_classes({}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING)

    assert "reorder_and_n5" not in _class_counts(stats)
    assert any("reorder_and_n5" in w and "only 60" in w for w in _warnings(stats))
    assert not any(p.tags.get("class") == "reorder_and_n5" for p in problems)


def test_and_glue_matches_via_length_l_fallback_without_an_l_tag(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    items = [_synthetic_problem("glue", i, {"op": "AND"}, length_l=12) for i in range(64)]
    _write_suite(store, "glue", items)

    problems, stats = build_classes({}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING)

    assert _class_counts(stats)["and_glue_L12"] == {"n_train": 32, "n_test": 32}
    assert sum(1 for p in problems if p.tags.get("class") == "and_glue_L12") == 64


def test_and_glue_matches_via_explicit_l_tag(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    # length.L deliberately left at the default (1, not 12): only the "L" tag should match.
    items = [_synthetic_problem("glue", i, {"op": "AND", "L": 12}) for i in range(64)]
    _write_suite(store, "glue", items)

    problems, stats = build_classes({}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING)

    assert _class_counts(stats)["and_glue_L12"] == {"n_train": 32, "n_test": 32}
    assert sum(1 for p in problems if p.tags.get("class") == "and_glue_L12") == 64


def _write_jsonl(path: Path, rows: list[dict[str, JSON]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# gen_L10's seen-in-training check builds the default excluder, which reads every real tier file.
_MISSING_TIER_FILES = [str(p) for ps in default_tiers().values() for p in ps if not p.is_file()]


@pytest.mark.skipif(
    bool(_MISSING_TIER_FILES),
    reason=f"real exclusion tier files not available here: {', '.join(_MISSING_TIER_FILES)}",
)
def test_gen_l10_file_loader(tmp_path: Path) -> None:
    rows: list[dict[str, JSON]] = []
    for i in range(70):
        rows.append(
            {
                "name": f"gen_{i}",
                "prompt": f"THM SEQ SYN_gen_{i} PRF",
                "source": "gen",
                "L_true": 10,
                "n_lines": 10,
                "minlen_proof": f"N1 SYN_gen_{i} : PR ; QED",
            }
        )
    # excluded: textbook source, the wrong length, and a 'gen'/L==10 row with no minlen_proof
    # (never falls back to a non-minimal 'gen_proof' field -- module docstring note 5).
    rows.append(
        {
            "name": "textbook_1",
            "prompt": "THM SEQ SYN_tb PRF",
            "source": "textbook",
            "L_true": 10,
            "minlen_proof": "N1 SYN_tb : PR ; QED",
        }
    )
    rows.append(
        {
            "name": "wrong_len",
            "prompt": "THM SEQ SYN_wl PRF",
            "source": "gen",
            "L_true": 11,
            "minlen_proof": "N1 SYN_wl : PR ; QED",
        }
    )
    rows.append(
        {
            "name": "no_minlen_proof",
            "prompt": "THM SEQ SYN_nomp PRF",
            "source": "gen",
            "L_true": 10,
            "gen_proof": "N1 SYN_nomp : PR ; QED",
        }
    )
    reserve_path = tmp_path / "reserve.jsonl"
    _write_jsonl(reserve_path, rows)

    store = Store(tmp_path / "out")
    problems, stats = build_classes(
        {}, store=store, gen_path=reserve_path, heldout_cap6_path=_MISSING
    )

    assert _class_counts(stats)["gen_L10"] == {"n_train": 32, "n_test": 32}
    class_problems = [p for p in problems if p.tags.get("class") == "gen_L10"]
    assert len(class_problems) == 64
    assert all(p.length.method == "minlen_exact" for p in class_problems)
    assert all(p.length.L == 10 for p in class_problems)
    assert all(p.ref is not None and p.ref.body.startswith("N1 ") for p in class_problems)


class _StubExcluder:
    """A minimal `Excluder` (structural `Protocol`, see `core.types.Excluder`) marking exactly the
    sequents in `in_train_keys` as `seen_train`, for `test_gen_l10_drops_seen_train_rows` -- avoids
    pulling in the real, large `ab2`/`ab3` tier files `DefaultExcluder` would scan."""

    def __init__(self, in_train_keys: set[str]) -> None:
        self._keys = in_train_keys

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        return seq.key() in self._keys, ()


def test_gen_l10_drops_seen_train_rows(tmp_path: Path) -> None:
    rows: list[dict[str, JSON]] = []
    for i in range(70):
        rows.append(
            {
                "name": f"gen_{i}",
                "prompt": f"THM SEQ SYN_gentrain_{i} PRF",
                "source": "gen",
                "L_true": 10,
                "minlen_proof": f"N1 SYN_gentrain_{i} : PR ; QED",
            }
        )
    reserve_path = tmp_path / "reserve.jsonl"
    _write_jsonl(reserve_path, rows)

    # Mark the first 10 rows' sequents as seen_train: build_classes should drop them (not merely
    # flag them), leaving only 60 -- still >= 64? No: 60 < 64, so the class is skipped entirely,
    # which is itself the clearest proof the drop (not a flag) actually happened.
    in_train_keys = {Sequent.from_prompt(r["prompt"]).key() for r in rows[:10]}  # type: ignore[arg-type]
    store = Store(tmp_path / "out")
    problems, stats = build_classes(
        {},
        store=store,
        gen_path=reserve_path,
        heldout_cap6_path=_MISSING,
        excluder=_StubExcluder(in_train_keys),
    )

    assert "gen_L10" not in _class_counts(stats)
    assert any("gen_L10" in w and "only 60" in w for w in _warnings(stats))
    assert not any(p.tags.get("class") == "gen_L10" for p in problems)


def test_heldout_cap6_file_loader_and_insufficient_items_are_skipped(tmp_path: Path) -> None:
    rows: list[dict[str, JSON]] = [
        {
            "name": f"heldout_{i}",
            "prompt": f"THM SEQ SYN_ho_{i} PRF",
            "n_lines": 2,
            "proof": f"N1 SYN_ho_{i} : PR ; QED",
        }
        for i in range(30)
    ]
    heldout_path = tmp_path / "heldout_cap6.jsonl"
    _write_jsonl(heldout_path, rows)

    store = Store(tmp_path / "out")
    problems, stats = build_classes(
        {}, store=store, gen_path=_MISSING, heldout_cap6_path=heldout_path
    )

    assert "heldout_cap6" not in _class_counts(stats)
    assert any("heldout_cap6" in w and "only 30" in w for w in _warnings(stats))
    assert not any(p.tags.get("class") == "heldout_cap6" for p in problems)


def test_heldout_cap6_file_loader_succeeds_with_enough_rows(tmp_path: Path) -> None:
    rows: list[dict[str, JSON]] = [
        {
            "name": f"heldout_{i}",
            "prompt": f"THM SEQ SYN_ho_{i} PRF",
            "n_lines": 2,
            "proof": f"N1 SYN_ho_{i} : PR ; QED",
        }
        for i in range(64)
    ]
    heldout_path = tmp_path / "heldout_cap6.jsonl"
    _write_jsonl(heldout_path, rows)

    store = Store(tmp_path / "out")
    problems, stats = build_classes(
        {}, store=store, gen_path=_MISSING, heldout_cap6_path=heldout_path
    )

    assert _class_counts(stats)["heldout_cap6"] == {"n_train": 32, "n_test": 32}
    class_problems = [p for p in problems if p.tags.get("class") == "heldout_cap6"]
    assert len(class_problems) == 64
    assert all(p.ref is not None for p in class_problems)


def test_split_is_deterministic_given_rng_seed(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    items = [_synthetic_problem("families", i, {"family": "imp_nest", "n": 4}) for i in range(64)]
    _write_suite(store, "families", items)

    p1, _ = build_classes(
        {}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING, rng_seed=42
    )
    p2, _ = build_classes(
        {}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING, rng_seed=42
    )
    p3, _ = build_classes(
        {}, store=store, gen_path=_MISSING, heldout_cap6_path=_MISSING, rng_seed=7
    )

    key1 = [(p.pid, p.tags["split"]) for p in p1]
    key2 = [(p.pid, p.tags["split"]) for p in p2]
    key3 = [(p.pid, p.tags["split"]) for p in p3]
    assert key1 == key2
    assert key1 != key3
