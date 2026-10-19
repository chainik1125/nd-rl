"""`suites/existing.py` (plan §2, §3 WP2 row): `textbook_dev`, `textbook_law`, `gen_ctrl`,
`skill_pool`, `parts_pool`, `import_legacy`.

Every builder except `textbook_dev` (which certifies the real 58-item file -- that's its whole
job) and `import_legacy` (which imports the real, frozen legacy result directories -- fast, pure
JSON I/O, no minlen) is exercised here against small synthetic fixture files matching each source
file's real schema, so this file stays well under the plan's 60s-per-file budget."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from code.ndbench.core.keys import canon_key
from code.ndbench.core.types import JSON, Certifier, Excluder, Problem, Sequent
from code.ndbench.logic.certify import MinlenCertifier
from code.ndbench.store import Store
from code.ndbench.suites import existing

# `classical` runs the vendored `intuit.py`, which raises the recursion limit at import: run at that
# limit and restore it afterwards (conftest.py).
pytestmark = pytest.mark.usefixtures("intuit_recursion_limit")

#: `existing`'s builders return `dict[str, JSON]` stats (the plan's builder contract); tests dig
#: into nested stats freely, which the recursive `JSON` union makes painful for pyright to follow
#: through plain indexing. `_build`/`_import_legacy` narrow the result to `dict[str, Any]` once,
#: right at the call site, so the rest of each test reads naturally.
_BuildFn = Callable[..., tuple[list[Problem], dict[str, JSON]]]


def _build(
    fn: _BuildFn,
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None = None,
    excluder: Excluder | None = None,
    rng_seed: int = 0,
) -> tuple[list[Problem], dict[str, Any]]:
    problems, stats = fn(cfg, certifier=certifier, excluder=excluder, rng_seed=rng_seed)
    return problems, cast(dict[str, Any], stats)


def _import_legacy(store: Store) -> dict[str, Any]:
    return cast(dict[str, Any], existing.import_legacy(store))


def _s(v: object) -> str:
    """Narrow a `JSON`-typed dict value to `str` for pyright (runtime-checked)."""
    assert isinstance(v, str)
    return v


def _d(v: object) -> dict[str, Any]:
    """Narrow a `JSON`-typed dict value to `dict[str, Any]` for pyright (runtime-checked)."""
    assert isinstance(v, dict)
    return v


def _list_str(v: object) -> list[str]:
    """Narrow a `JSON`-typed dict value to `list[str]` for pyright (runtime-checked)."""
    assert isinstance(v, list)
    return v


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _manifest_for(suite: str, version: str, problems: list[Problem], stats: dict[str, Any]):
    from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
    from code.ndbench.core.types import SuiteManifest

    items_json = [p.to_json() for p in problems]
    return SuiteManifest(
        suite=suite,
        version=version,
        builder=f"code.ndbench.suites.existing:{suite}",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str(json.dumps(items_json, sort_keys=True)),
        excl_sources={},
        stats=stats,
    )


class _AlwaysExcluder:
    """A stub `Excluder` for tests that just need `seen_train`/`seen_eval` to be populated from
    something, without scanning any real tier file."""

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        return False, ()


# ------------------------------------------------------------------------------- textbook_dev


def test_textbook_dev_relabel_gives_12_for_the_plan_example(tmp_path: Path) -> None:
    """Acceptance test: the `textbook_dev` relabel gives 12 for `~(P>Q) |- P&~Q` (plan §0's
    example of a reference length -- 18 in the legacy file -- that minlen certifies shorter)."""
    path = tmp_path / "textbook_dev.jsonl"
    _write_jsonl(
        path,
        [
            {
                "name": "textbook_example",
                "thm": "( ~ ( P > Q ) ) |- ( P & ( ~ Q ) )",
                "key": "( ~ ( P > Q ) ) |- ( P & ( ~ Q ) )",
                "prompt": "THM ( ~ ( P > Q ) ) SEQ ( P & ( ~ Q ) ) PRF",
                "n_lines": 18,
                "L_true": 18,
                "source": "textbook_dev",
            }
        ],
    )
    certifier = MinlenCertifier(procs=1)
    problems, stats = _build(
        existing.textbook_dev, {"path": path}, certifier=certifier, excluder=_AlwaysExcluder()
    )
    assert len(problems) == 1
    (problem,) = problems
    assert problem.length.L == 12
    assert problem.length.method == "minlen_exact"
    assert problem.tags["reference_L"] == 18
    assert problem.suite == "textbook_dev"
    assert stats["n_items"] == 1


def test_textbook_dev_requires_a_certifier() -> None:
    with pytest.raises(ValueError):
        existing.textbook_dev({}, certifier=None, excluder=None, rng_seed=0)


@pytest.mark.slow
def test_textbook_dev_full_real_file_builds_and_writes_manifest(tmp_path: Path) -> None:
    """The full 58-item real file certifies within the plan's own bound/time (16, 120s) and
    writes a loadable `Store` suite. Marked slow: even though every individual case in this file
    resolves quickly (seconds, per manual timing), 58 real items together are safer run only with
    `--runslow` on a shared CPU budget."""
    if not existing.DEFAULT_TEXTBOOK_DEV_PATH.exists():
        pytest.skip("real ab2/sealed/textbook_dev.jsonl not available in this environment")
    certifier = MinlenCertifier(procs=8, cache_dir=tmp_path / "cache")
    problems, stats = _build(existing.textbook_dev, {"version": "v1"}, certifier=certifier)
    assert stats["n_items"] == 58
    assert len(problems) == 58
    store = Store(tmp_path / "out")
    store.write_suite(problems, _manifest_for("textbook_dev", "v1", problems, stats))
    loaded = store.load_suite("textbook_dev", "v1")
    assert len(loaded) == 58
    assert (tmp_path / "out" / "suites" / "textbook_dev" / "v1" / "manifest.json").exists()


# ------------------------------------------------------------------------------- textbook_law


def _law_row(name: str, l_true: int, schema: str, *, source: str = "textbook") -> dict[str, Any]:
    return {
        "name": name,
        "thm": "P |- P",
        "key": "P |- P",
        "prompt": "THM P SEQ P PRF",
        "n_lines": l_true,
        "L_true": l_true,
        "source": source,
        "schema": schema,
        "n_prem": 1,
    }


def test_textbook_law_caps_per_schema_and_per_length(tmp_path: Path) -> None:
    dev_path = tmp_path / "dev.jsonl"
    sealed_path = tmp_path / "sealed.jsonl"

    dev_rows: list[dict[str, Any]] = []
    # L8: 3 schemas x 10 rows each -> capped at 6/schema = 18 total (< 40, no further downsample).
    for schema_i in range(3):
        for i in range(10):
            dev_rows.append(_law_row(f"L8_s{schema_i}_{i}", 8, f"schema{schema_i}"))
    # L9: 10 schemas x 10 rows each -> capped at 6/schema = 60 total (> 40, downsampled to 40).
    for schema_i in range(10):
        for i in range(10):
            dev_rows.append(_law_row(f"L9_s{schema_i}_{i}", 9, f"schema{schema_i}"))
    # A decoy non-textbook row that must never be selected.
    dev_rows.append(_law_row("decoy_gen", 9, "schema0", source="gen"))
    _write_jsonl(dev_path, dev_rows)

    sealed_rows = [_law_row(f"L12_{i}", 12, "onlyschema") for i in range(3)]
    _write_jsonl(sealed_path, sealed_rows)

    problems, stats = _build(
        existing.textbook_law, {"dev_path": dev_path, "sealed_path": sealed_path}
    )

    by_l: dict[int, int] = {}
    for p in problems:
        assert isinstance(p.length.L, int)
        by_l[p.length.L] = by_l.get(p.length.L, 0) + 1
        assert p.length.method == "reference"
        assert "decoy" not in p.pid

    assert by_l == {8: 18, 9: 40, 12: 3}
    assert stats["n_items"] == 61
    assert stats["by_l"]["8"] == {"n_schemas": 3, "n_selected": 18}
    assert stats["by_l"]["9"] == {"n_schemas": 10, "n_selected": 40}
    assert stats["by_l"]["12"] == {"n_schemas": 1, "n_selected": 3}


def test_textbook_law_keeps_minlen_exact_when_proof_present(tmp_path: Path) -> None:
    dev_path = tmp_path / "dev.jsonl"
    row = _law_row("with_proof", 8, "schemaX")
    row["minlen_proof"] = "N1 P : PR ; QED"
    row["minlen_bound"] = 16
    _write_jsonl(dev_path, [row])
    sealed_path = tmp_path / "sealed_empty.jsonl"
    _write_jsonl(sealed_path, [])

    problems, _ = _build(existing.textbook_law, {"dev_path": dev_path, "sealed_path": sealed_path})
    assert len(problems) == 1
    assert problems[0].length.method == "minlen_exact"
    assert problems[0].length.bound == 16
    assert problems[0].ref is not None
    assert problems[0].ref.body == "N1 P : PR ; QED"


# ----------------------------------------------------------------------------------- gen_ctrl


def _dev_row(name: str, l_true: int, source: str) -> dict[str, Any]:
    return {
        "name": name,
        "thm": "P |- P",
        "key": "P |- P",
        "prompt": "THM P SEQ P PRF",
        "n_lines": l_true,
        "L_true": l_true,
        "source": source,
        "schema": None,
        "n_prem": 1,
    }


def test_gen_ctrl_selects_sixty_even_across_l8_11(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    rows = []
    for l_true in (8, 9, 10, 11):
        for i in range(30):
            rows.append(_dev_row(f"gen_L{l_true}_{i}", l_true, "gen"))
        rows.append(_dev_row(f"textbook_L{l_true}", l_true, "textbook"))  # must be excluded
    _write_jsonl(path, rows)

    problems, stats = _build(existing.gen_ctrl, {"path": path})
    assert len(problems) == 60
    assert all(p.tags["L"] in (8, 9, 10, 11) for p in problems)
    by_l: dict[int, int] = {}
    for p in problems:
        assert isinstance(p.length.L, int)
        by_l[p.length.L] = by_l.get(p.length.L, 0) + 1
    assert by_l == {8: 15, 9: 15, 10: 15, 11: 15}
    assert stats["n_target"] == 60
    for l_true in (8, 9, 10, 11):
        assert stats["by_l"][str(l_true)]["taken"] == 15


def test_gen_ctrl_takes_all_available_when_scarce(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    rows = [_dev_row("only_one_L8", 8, "gen")]
    _write_jsonl(path, rows)
    problems, stats = _build(existing.gen_ctrl, {"path": path})
    assert len(problems) == 1
    assert stats["by_l"]["8"] == {"available": 1, "taken": 1}


# --------------------------------------------------------------------------------- parts_pool


def _reserve_row(name: str, thm: str, l_true: int, pkey_value: str) -> dict[str, Any]:
    prem, concl = thm.split(" |- ")
    proof_lines = " ; ".join(f"N{i + 1} {p.strip()} : PR" for i, p in enumerate(prem.split(" , ")))
    return {
        "name": name,
        "thm": thm,
        "key": canon_key(thm),
        "pkey": pkey_value,
        "prompt": f"THM {prem} SEQ {concl} PRF",
        "n_lines": l_true,
        "L_true": l_true,
        "source": "gen",
        "schema": None,
        "n_prem": len(prem.split(" , ")),
        "minlen_proof": f"{proof_lines} ; QED",
    }


def test_parts_pool_stratifies_by_l_and_atom_count_one_per_pkey(tmp_path: Path) -> None:
    path = tmp_path / "reserve.jsonl"
    rows: list[dict[str, Any]] = []
    cells = [(2, 1), (2, 2), (3, 1), (8, 4)]
    for l_true, n_atoms in cells:
        atoms = ["P", "Q", "R", "S"][:n_atoms]
        for i in range(5):  # 5 distinct pkeys per cell
            thm = " , ".join(atoms) + f" |- {atoms[0]}"
            rows.append(
                _reserve_row(
                    f"L{l_true}_a{n_atoms}_{i}", thm, l_true, f"pkey_{l_true}_{n_atoms}_{i}"
                )
            )
        # A duplicate-pkey row that must be skipped (dedup).
        rows.append(
            _reserve_row(
                f"L{l_true}_a{n_atoms}_dup",
                " , ".join(atoms) + f" |- {atoms[0]}",
                l_true,
                f"pkey_{l_true}_{n_atoms}_0",
            )
        )
        # An out-of-range row (L=20) that must never be selected.
    rows.append(_reserve_row("out_of_range", "P |- P", 20, "pkey_oor"))
    _write_jsonl(path, rows)

    problems, stats = _build(existing.parts_pool, {"path": path, "n_items": 20})
    assert len(problems) == 20  # 4 cells x 5 target-per-cell (20 / 4 cells = 5 each)
    assert stats["n_candidates"] == 20  # 5 unique pkeys x 4 cells (dup + out-of-range excluded)
    assert stats["n_cells"] == 4
    for p in problems:
        assert p.length.method == "minlen_exact"
        assert p.ref is not None
    pids = {p.pid for p in problems}
    assert "out_of_range" not in pids
    assert not any(pid.endswith("_dup") for pid in pids)


def test_parts_pool_scan_limit_caps_rows_read(tmp_path: Path) -> None:
    path = tmp_path / "reserve.jsonl"
    rows = [_reserve_row(f"r{i}", "P |- P", 2, f"pkey_{i}") for i in range(30)]
    _write_jsonl(path, rows)
    _, stats = _build(existing.parts_pool, {"path": path, "n_items": 5, "scan_limit": 10})
    assert stats["n_scanned"] == 10
    assert stats["n_candidates"] == 10


# --------------------------------------------------------------------------------- skill_pool


def test_skill_pool_tags_classical_case_split_and_rule_uses(tmp_path: Path) -> None:
    path_a = tmp_path / "source_a.jsonl"
    path_b = tmp_path / "source_b.jsonl"
    peirce_row = {
        "name": "peirce_item",
        "thm": "|- ( ( ( P > Q ) > P ) > P )",
        "key": "|- ( ( ( P > Q ) > P ) > P )",
        "prompt": "THM SEQ ( ( ( P > Q ) > P ) > P ) PRF",
        "n_lines": 9,
        "L_true": 9,
        "source": "textbook",
        "schema": "peirce",
        "n_prem": 0,
    }
    dsj_row = {
        "name": "dsj_item",
        "thm": "( P v Q ) , ( ~ P ) |- Q",
        "key": "( P v Q ) , ( ~ P ) |- Q",
        "prompt": "THM ( P v Q ) , ( ~ P ) SEQ Q PRF",
        "n_lines": 7,
        "L_true": 7,
        "source": "textbook",
        "schema": "disjunctive_syllogism",
        "n_prem": 2,
        "minlen_proof": (
            "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; "
            "N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED"
        ),
        "minlen_bound": 10,
    }
    _write_jsonl(path_a, [peirce_row])
    _write_jsonl(path_b, [dsj_row])

    problems, stats = _build(existing.skill_pool, {"sources": {"a": path_a, "b": path_b}})
    assert len(problems) == 2
    by_pid = {p.pid: p for p in problems}
    peirce_problem = by_pid["a:peirce_item"]
    assert peirce_problem.tags["classical"] is True
    assert peirce_problem.tags["case_split"] is False
    assert "rule_uses" not in peirce_problem.tags
    assert peirce_problem.length.method == "reference"

    dsj_problem = by_pid["b:dsj_item"]
    assert dsj_problem.tags["classical"] is False
    assert dsj_problem.tags["case_split"] is True
    assert dsj_problem.tags["rule_uses"] == {"PR": 2, "AS": 2, "NEGE": 1, "BOTE": 1, "ORE": 1}
    assert dsj_problem.length.method == "minlen_exact"
    assert dsj_problem.ref is not None

    assert stats["by_source"]["a"] == {"n_items": 1, "n_classical": 1, "n_case_split": 0}
    assert stats["by_source"]["b"] == {"n_items": 1, "n_classical": 0, "n_case_split": 1}
    assert (
        "necessity" in stats["necessity_note"].lower() or "needs_depth" in stats["necessity_note"]
    )


def test_skill_pool_limit_per_source(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    _write_jsonl(path, [_dev_row(f"i{i}", 8, "gen") for i in range(10)])
    problems, stats = _build(existing.skill_pool, {"sources": {"a": path}, "limit_per_source": 3})
    assert len(problems) == 3
    assert stats["by_source"]["a"]["n_items"] == 3


# ------------------------------------------------------------------------------ manifests + store


@pytest.mark.parametrize("suite_name", ["textbook_law", "gen_ctrl", "parts_pool", "skill_pool"])
def test_builders_write_loadable_manifests_via_the_store(tmp_path: Path, suite_name: str) -> None:
    """Acceptance test: builders write manifests via the `Store`."""
    dev_path = tmp_path / "dev.jsonl"
    sealed_path = tmp_path / "sealed.jsonl"
    reserve_path = tmp_path / "reserve.jsonl"
    _write_jsonl(dev_path, [_law_row("d0", 8, "s0"), _dev_row("g0", 8, "gen")])
    _write_jsonl(sealed_path, [_law_row("s0", 12, "s0")])
    _write_jsonl(
        reserve_path,
        [
            _reserve_row("r0", "P , Q |- P", 3, "pk0"),
            _reserve_row("r1", "P , Q , R |- P", 4, "pk1"),
        ],
    )

    cfg: dict[str, Any] = {
        "textbook_law": {"dev_path": dev_path, "sealed_path": sealed_path},
        "gen_ctrl": {"path": dev_path, "n_items": 1},
        "parts_pool": {"path": reserve_path, "n_items": 2},
        "skill_pool": {"sources": {"dev": dev_path}},
    }[suite_name]

    certifier: Certifier | None = None
    excluder: Excluder | None = None
    build_fn = getattr(existing, suite_name)
    problems, stats = build_fn(cfg, certifier=certifier, excluder=excluder, rng_seed=0)
    assert len(problems) > 0

    store = Store(tmp_path / "out")
    manifest = _manifest_for(suite_name, "v1", problems, stats)
    store.write_suite(problems, manifest)

    loaded = store.load_suite(suite_name, "v1")
    assert len(loaded) == len(problems)
    manifest_path = tmp_path / "out" / "suites" / suite_name / "v1" / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["n_items"] == len(problems)


# ------------------------------------------------------------------------------ import_legacy


def test_import_legacy_runs_appear_in_the_store(tmp_path: Path) -> None:
    """Acceptance test: legacy runs appear in the store. Runs against the real (frozen)
    `ab2/monitor/final` and `ab3/lengthexp/eval` directories -- pure JSON I/O, no minlen search,
    empirically ~1.5s total (see this WP's final report)."""
    if not existing._LEGACY_MONITOR_MODELS["base"][0].exists():
        pytest.skip("real ab2/monitor/final/pretrain not available in this environment")

    store = Store(tmp_path)
    result = _import_legacy(store)
    assert result["n_runs"] > 0
    assert len(result["run_ids"]) == result["n_runs"]

    runs = store.runs()
    assert len(runs) == result["n_runs"]
    assert all(r["status"] == "legacy" for r in runs)
    assert all(r["judge_id"] == "legacy_import" for r in runs)

    base_runs = store.runs(model="base")
    champion_runs = store.runs(model="champion")
    armc_runs = store.runs(model="armC")
    pretrain_lengthexp_runs = [
        r for r in store.runs(model="pretrain") if "lengthexp" in _s(r["suite"])
    ]
    assert base_runs and champion_runs and armc_runs and pretrain_lengthexp_runs

    # plan §0: "Existing k=256 textbook results (max_new 512): base 9/58, RL 10/58".
    textbook_base = [r for r in base_runs if r["suite"] == "legacy_textbook"]
    textbook_champion = [r for r in champion_runs if r["suite"] == "legacy_textbook"]
    assert len(textbook_base) == 1
    assert len(textbook_champion) == 1
    base_items = store.items(_s(textbook_base[0]["run_id"]))
    champion_items = store.items(_s(textbook_champion[0]["run_id"]))
    assert sum(1 for it in base_items if it.c > 0) == 9
    assert sum(1 for it in champion_items if it.c > 0) == 10
    assert all(it.n == 256 for it in base_items)

    for r in base_runs:
        assert _d(r["identity"])["ckpt_sha256_prefix"] == "f4409e4f82d8922f"
    for r in champion_runs:
        assert _d(r["identity"])["ckpt_sha256_prefix"] == "548a9e343c71675f"


def test_import_legacy_is_idempotent_in_run_id(tmp_path: Path) -> None:
    if not existing._LEGACY_MONITOR_MODELS["base"][0].exists():
        pytest.skip("real ab2/monitor/final/pretrain not available in this environment")
    store = Store(tmp_path)
    first = _import_legacy(store)
    second = _import_legacy(store)
    assert sorted(_list_str(first["run_ids"])) == sorted(_list_str(second["run_ids"]))
