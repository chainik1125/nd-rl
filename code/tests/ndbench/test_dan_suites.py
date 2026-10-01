"""`suites/danstate.py` (new suites for Dan's state-env models) and `logic/exclusion.py`'s Dan
"seen in training" tiers (plan `DANSTATE_PLAN.md` §3/§7 WP-D3).

Fast, CPU-only, small synthetic inputs throughout (see `_REFERENCES` below for the two hand
references the `materialize_profile`/`m_for_j` checks replay through the real vendored
`state_env`). Tests that need real, multi-GB Dan data or an already-built real `Store` skip if
that data is not present in this environment.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pytest

from code.ndbench.core.types import LengthLabel, PrefixSpec, Problem, ReferenceProof, Sequent
from code.ndbench.logic import exclusion as exc
from code.ndbench.store import Store
from code.ndbench.suites import danstate as ds
from code.ndbench.suites.glue import DEFAULT_STORE_ROOT as REAL_STORE_ROOT

from .test_wp6_helpers import manifest

# --------------------------------------------------------------------- hand-derived references
#
# Two real ND proofs (4 premises, 10 written lines each; one a plain IMPI/BOTE/ANDI proof, one
# containing an Or.elim/ORE), chosen so their `lean_seq` decomposition exercises every action
# kind `materialize_profile`'s docstring classifies: a plain `have`/atomic line, an `imp`-box
# open/close, and (the ORE one) an `Or.elim` open, its branch-1 close (which -- `Env._close`'s
# `or1` case -- injects the branch-2 `AS` line as a side effect), and its branch-2/`ORE` close.
# Verified by hand against `lean_tok.inverse`/`state_env.Env` directly (see the WP-D3 report).

_PROMPT_IMPI = (
    "THM ( P v ( ~ R ) ) , ( ~ P ) , ( Q & ( ~ S ) ) , ( Q v ( S v S ) ) SEQ "
    "( ( P > ( P > ( ~ P ) ) ) & ( ~ S ) ) PRF"
)
_BODY_IMPI = (
    "N1 ( P v ( ~ R ) ) : PR ; N2 ( ~ P ) : PR ; N3 ( Q & ( ~ S ) ) : PR ; "
    "N4 ( Q v ( S v S ) ) : PR ; N5 | P : AS ; N6 | F : NEGE N5 N2 ; "
    "N7 | ( P > ( ~ P ) ) : BOTE N6 ; N8 ( P > ( P > ( ~ P ) ) ) : IMPI N5 N7 ; "
    "N9 ( ~ S ) : ANDE2 N3 ; N10 ( ( P > ( P > ( ~ P ) ) ) & ( ~ S ) ) : ANDI N8 N9 ; QED"
)

_PROMPT_ORE = (
    "THM ( ( P & S ) v S ) , ( ~ ( P & S ) ) , Q , ( ( Q > ( ~ P ) ) v ( Q & P ) ) SEQ "
    "( ( ~ ( ~ P ) ) > S ) PRF"
)
_BODY_ORE = (
    "N1 ( ( P & S ) v S ) : PR ; N2 ( ~ ( P & S ) ) : PR ; N3 Q : PR ; "
    "N4 ( ( Q > ( ~ P ) ) v ( Q & P ) ) : PR ; N5 | ( ~ ( ~ P ) ) : AS ; "
    "N6 | | ( P & S ) : AS ; N7 | | S : ANDE2 N6 ; N8 | | S : AS ; "
    "N9 | S : ORE N1 N6 N7 N8 N8 ; N10 ( ( ~ ( ~ P ) ) > S ) : IMPI N5 N9 ; QED"
)

_REFERENCES = [(_PROMPT_IMPI, _BODY_IMPI, 10), (_PROMPT_ORE, _BODY_ORE, 10)]


# ----------------------------------------------------------------------- materialize_profile


@pytest.mark.parametrize("prompt,body,expected_l", _REFERENCES)
def test_materialize_profile_reaches_l_one_action_early(
    prompt: str, body: str, expected_l: int
) -> None:
    acts, counts, open_boxes = ds.materialize_profile(prompt, body)
    assert len(counts) == len(acts) + 1 == len(open_boxes)
    assert counts[0] == 0
    assert counts[-1] == expected_l
    # the final top-level `exact` cites the last line but creates none (module docstring).
    assert counts[-2] == expected_l
    assert open_boxes[0] == 0
    assert open_boxes[-1] == 0
    for lo, hi in zip(counts, counts[1:]):  # noqa: B905 (deliberately one shorter)
        assert hi - lo in (0, 1)  # each action materialises at most one new ND line


def test_materialize_profile_ore_open_boxes_peak_at_two() -> None:
    _, _, open_boxes = ds.materialize_profile(_PROMPT_ORE, _BODY_ORE)
    assert max(open_boxes) == 2  # the imp box containing the Or.elim


def test_materialize_profile_rejects_a_dangling_citation() -> None:
    bad_body = _BODY_IMPI.replace("NEGE N5 N2", "NEGE N5 N99")
    # `canonicalise` raises a bare `KeyError` for this particular corruption (an unbound name
    # lookup); `materialize_profile`'s own replay raises `ValueError` for others (module
    # docstring) -- either way, "rejected", not silently accepted.
    with pytest.raises((KeyError, ValueError)):
        ds.materialize_profile(_PROMPT_IMPI, bad_body)


# -------------------------------------------------------------------------------------- m_for_j


@pytest.mark.parametrize("prompt,body,expected_l", _REFERENCES)
def test_m_for_j_m0_mL_and_monotone(prompt: str, body: str, expected_l: int) -> None:
    acts, counts, _ = ds.materialize_profile(prompt, body)
    a_total = len(acts)
    assert ds.m_for_j(counts, 0, expected_l) == 0
    assert ds.m_for_j(counts, expected_l, expected_l) == a_total  # j >= L boundary convention
    ms = [ds.m_for_j(counts, j, expected_l) for j in range(expected_l + 1)]
    assert ms == sorted(ms)
    assert ms[0] == 0
    assert ms[-1] == a_total


def test_m_for_j_smallest_crossing_below_l() -> None:
    counts = (0, 2, 2, 5, 5, 5)  # A = 5, L = 5 (the last value)
    assert ds.m_for_j(counts, 0, 5) == 0
    assert ds.m_for_j(counts, 1, 5) == 1
    assert ds.m_for_j(counts, 2, 5) == 1
    assert ds.m_for_j(counts, 3, 5) == 3
    assert ds.m_for_j(counts, 5, 5) == 5  # boundary: j >= L returns len(counts) - 1


# --------------------------------------------------------------------------------- prefix_state


def _make_prefix_item(
    pid: str, prompt: str, body: str, j: int, length_l: int, src_pid: str
) -> Problem:
    ref = ReferenceProof(body=body, lines=length_l, pruned=length_l, source="minlen")
    label = LengthLabel(L=length_l, L_ub=length_l, L_lb=length_l, method="minlen_exact", bound=None)
    return Problem(
        pid=pid,
        suite="prefix",
        version="v1",
        sequent=Sequent.from_prompt(prompt),
        length=label,
        ref=ref,
        tags={"j": j, "L": length_l, "src_pid": src_pid, "bucket": length_l},
        prefix=PrefixSpec(j=j, body=""),  # the real prefix.py body is irrelevant to prefix_state
    )


def _build_prefix_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    problems: list[Problem] = []
    for prompt, body, length_l in _REFERENCES:
        src_pid = f"src_{abs(hash(prompt)) % 10_000}"
        for j in (0, 3, length_l - 1):
            pid = f"{src_pid}_j{j:03d}"
            problems.append(_make_prefix_item(pid, prompt, body, j, length_l, src_pid))
    store.write_suite(problems, manifest("prefix", "v1", len(problems)))
    return store


def test_prefix_state_statprefix_round_trips(tmp_path: Path) -> None:
    _build_prefix_store(tmp_path)
    cfg = {"store_root": str(tmp_path)}
    problems, stats = ds.prefix_state(cfg, certifier=None, excluder=None, rng_seed=0)
    assert stats["drops"] == {}
    assert len(problems) == stats["n_source"] == 6

    for p in problems:
        assert p.prefix is not None
        assert p.ref is not None
        m = cast(int, p.tags["m"])
        j = cast(int, p.tags["j"])
        length_l = cast(int, p.tags["L"])
        head, sep, tail_body = p.prefix.body.partition(" | ")
        assert sep == " | "
        assert head == f"STATEPREFIX m={m}"
        assert tail_body == p.ref.body  # the full reference, byte for byte, not truncated

        acts, counts, open_boxes = ds.materialize_profile(p.sequent.prompt(), tail_body)
        assert len(acts) == p.tags["A"]
        assert open_boxes[m] == p.tags["open_boxes"]
        assert cast(int, p.tags["A"]) - m == p.tags["actions_left"]
        if j == 0:
            assert m == 0
        assert counts[m] >= min(j, length_l)


def test_prefix_state_drops_items_with_no_ref(tmp_path: Path) -> None:
    store = Store(tmp_path)
    no_ref = Problem(
        pid="no_ref_j000",
        suite="prefix",
        version="v1",
        sequent=Sequent.from_prompt(_PROMPT_IMPI),
        length=LengthLabel(L=10, L_ub=10, L_lb=10, method="minlen_exact", bound=None),
        ref=None,
        tags={"j": 0, "L": 10, "src_pid": "no_ref", "bucket": 10},
        prefix=PrefixSpec(j=0, body=""),
    )
    store.write_suite([no_ref], manifest("prefix", "v1", 1))
    cfg = {"store_root": str(tmp_path)}
    problems, stats = ds.prefix_state(cfg, certifier=None, excluder=None, rng_seed=0)
    assert problems == []
    assert stats["drops"] == {"no_ref": 1}


# ----------------------------------------------------------------------------- skill_pool_sub


def _make_skill_item(pid: str, length_l: int, classical: bool, case_split: bool) -> Problem:
    return Problem(
        pid=pid,
        suite="skill_pool",
        version="v1",
        sequent=Sequent(premises=("P",), conclusion="P"),
        length=LengthLabel(L=length_l, L_ub=length_l, L_lb=None, method="reference", bound=None),
        ref=None,
        tags={"L": length_l, "classical": classical, "case_split": case_split},
    )


def _build_skill_pool_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    problems: list[Problem] = []
    i = 0
    for length_l in (6, 7, 8):
        for classical in (False, True):
            for case_split in (False, True):
                for _ in range(10):
                    problems.append(_make_skill_item(f"sp{i}", length_l, classical, case_split))
                    i += 1
    store.write_suite(problems, manifest("skill_pool", "v1", len(problems)))
    return store


def test_skill_pool_sub_deterministic_same_seed(tmp_path: Path) -> None:
    _build_skill_pool_store(tmp_path)
    cfg = {"store_root": str(tmp_path), "n_items": 24}
    p1, s1 = ds.skill_pool_sub(cfg, certifier=None, excluder=None, rng_seed=42)
    p2, s2 = ds.skill_pool_sub(cfg, certifier=None, excluder=None, rng_seed=42)
    assert [p.pid for p in p1] == [p.pid for p in p2]
    assert s1["by_cell"] == s2["by_cell"]
    assert all(p.tags["orig_suite"] == "skill_pool" for p in p1)
    assert all("skill_group" in p.tags for p in p1)

    p3, _ = ds.skill_pool_sub(cfg, certifier=None, excluder=None, rng_seed=43)
    assert [p.pid for p in p1] != [p.pid for p in p3]


# --------------------------------------------------------------------- dan_tiers / DanExcluder


def test_dan_tiers_structure_and_unknown_family() -> None:
    rl_targets = exc.DAN_LADDER_RL_TARGETS
    assert exc.dan_tiers("dan6")["train"] == [exc.DAN_STAGE1_FILES["dan6"], rl_targets]
    assert exc.dan_tiers("dan12")["train"] == [exc.DAN_STAGE1_FILES["dan12"], rl_targets]
    with pytest.raises(ValueError):
        exc.dan_tiers("dan6_but_wrong")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_dan_tiers_keys_agree_with_our_tier_keys(tmp_path: Path) -> None:
    """A theorem written once in Dan's row schema and once in our own tier-file schema (both
    `thm`/`key`) keys identically either way -- both go through the same `canon_key`/`pkey`
    (plan: "'seen' means the same thing for every model")."""
    thm = "P , ( P > Q ) |- Q"
    dan_file = tmp_path / "dan_stage1.jsonl"
    _write_jsonl(dan_file, [{"name": "d0", "thm": thm, "key": thm}])
    our_file = tmp_path / "our_train.jsonl"
    _write_jsonl(our_file, [{"name": "o0", "thm": thm, "key": thm}])

    dan_index = exc.ExclusionIndex.build({"train": [dan_file]})
    our_index = exc.ExclusionIndex.build({"train": [our_file]})
    assert dan_index._train.keys == our_index._train.keys
    assert dan_index._train.pkeys == our_index._train.pkeys

    renamed = Sequent(premises=("R", "( R > S )"), conclusion="S")  # same class, different atoms
    assert dan_index.status(renamed)[0] is True
    assert our_index.status(renamed)[0] is True


def test_dan_excluder_status_and_eval_sets_always_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = tmp_path / "stage1.jsonl"
    _write_jsonl(stage1, [{"name": "s0", "thm": "P |- P", "key": "P |- P"}])
    rl_targets = tmp_path / "rl_targets.jsonl"
    _write_jsonl(rl_targets, [{"name": "r0", "thm": "Q |- Q", "key": "Q |- Q"}])
    monkeypatch.setitem(exc.DAN_STAGE1_FILES, "dan6", stage1)
    monkeypatch.setattr(exc, "DAN_LADDER_RL_TARGETS", rl_targets)

    excluder = exc.DanExcluder("dan6", cache_dir=tmp_path / "cache")
    in_train, eval_sets = excluder.status(Sequent(premises=("P",), conclusion="P"))
    assert in_train is True
    assert eval_sets == ()
    in_train_rl, _ = excluder.status(Sequent(premises=("Q",), conclusion="Q"))
    assert in_train_rl is True  # seen via the ladder's rl_targets pool, not Stage-1
    in_train_no, _ = excluder.status(Sequent(premises=("R",), conclusion="S"))
    assert in_train_no is False


# ------------------------------------------------------------------------------ seen sidecars


def test_build_seen_sidecar_and_seen_for_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = tmp_path / "stage1.jsonl"
    _write_jsonl(stage1, [{"name": "s0", "thm": "P |- P", "key": "P |- P"}])
    rl_targets = tmp_path / "rl_targets.jsonl"
    _write_jsonl(rl_targets, [])
    monkeypatch.setitem(exc.DAN_STAGE1_FILES, "dan6", stage1)
    monkeypatch.setattr(exc, "DAN_LADDER_RL_TARGETS", rl_targets)

    store = Store(tmp_path / "out")
    problems = [
        Problem(
            pid="i0",
            suite="toy",
            version="v1",
            sequent=Sequent(premises=("P",), conclusion="P"),
            length=LengthLabel(L=1, L_ub=1, L_lb=1, method="reference", bound=None),
            ref=None,
            tags={},
        ),
        Problem(
            pid="i1",
            suite="toy",
            version="v1",
            sequent=Sequent(premises=("R",), conclusion="S"),
            length=LengthLabel(L=1, L_ub=1, L_lb=1, method="reference", bound=None),
            ref=None,
            tags={},
        ),
    ]
    store.write_suite(problems, manifest("toy", "v1", len(problems)))

    report = exc.build_seen_sidecars(
        store, [("toy", "v1")], families=("dan6",), cache_dir=tmp_path / "cache"
    )
    assert report["dan6"]["toy/v1"] == {"n_items": 2, "n_seen": 1}

    path = exc.seen_sidecar_path(store.root, "toy", "v1", "dan6")
    assert path.is_file()
    seen = exc.seen_for(store, "toy", "v1", "dan6")
    assert seen == {"i0": True, "i1": False}

    payload = json.loads(path.read_text())
    assert payload["family"] == "dan6"
    assert len(payload["sources"]) == 2
    assert payload["sources"][0]["md5"] is not None
    assert payload["sources"][1]["n_rows"] == 0


# ---------------------------------------------------------- real-data checks (skip if absent)

_REAL_DAN6_STAGE1 = exc.DAN_STAGE1_FILES["dan6"]
_REAL_RL_TARGETS = exc.DAN_LADDER_RL_TARGETS
_REAL_DAN_DATA_AVAILABLE = _REAL_DAN6_STAGE1.is_file() and _REAL_RL_TARGETS.is_file()


@pytest.mark.skipif(
    not _REAL_DAN_DATA_AVAILABLE, reason="Dan's real Stage-1/ladder data not available here"
)
def test_dan_excluder_real_dan6_recognises_its_own_stage1_row(tmp_path: Path) -> None:
    row = json.loads(_REAL_DAN6_STAGE1.open().readline())
    seq = Sequent.from_thm(row["thm"])
    excluder = exc.DanExcluder("dan6", cache_dir=tmp_path / "cache")
    in_train, eval_sets = excluder.status(seq)
    assert in_train is True
    assert eval_sets == ()


_REAL_PREFIX_ITEMS_PATH = Path(REAL_STORE_ROOT) / "suites" / "prefix" / "v1" / "items.jsonl"


@pytest.mark.skipif(
    not _REAL_PREFIX_ITEMS_PATH.is_file(), reason="real built prefix v1 suite not available here"
)
def test_prefix_state_real_prefix_suite_invariants() -> None:
    problems, stats = ds.prefix_state({}, certifier=None, excluder=None, rng_seed=0)
    assert stats["drops"] == {}
    assert len(problems) == stats["n_source"]

    by_src: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for p in problems:
        src_pid = cast(str, p.tags["src_pid"])
        j = cast(int, p.tags["j"])
        m = cast(int, p.tags["m"])
        by_src[src_pid].append((j, m))
        if j == 0:
            assert m == 0
        assert cast(int, p.tags["A"]) - m == p.tags["actions_left"]

    for items in by_src.values():
        items.sort()
        ms = [m for _, m in items]
        assert ms == sorted(ms)  # non-decreasing m as j increases, per source theorem
