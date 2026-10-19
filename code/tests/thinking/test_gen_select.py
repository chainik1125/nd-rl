"""Source selection for LLM trace generation: eval-set exclusion by renaming class, strata,
determinism and provenance (code/thinking/gen/select.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from code.thinking.gen import select as sl

FORK = Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome"))
pytestmark = pytest.mark.skipif(
    not (FORK / "state_env.py").exists() or not sl.K12.exists(), reason="needs Dan's fork and K12"
)


def k12_rows(n: int = 400) -> list[dict[str, Any]]:
    out = []
    with open(sl.K12) as f:
        for i, line in enumerate(f):
            if i % 97 == 0:
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def rename(prompt: str, perm: dict[str, str]) -> str:
    return " ".join(perm.get(t, t) for t in prompt.split())


def swap_premises(prompt: str) -> str:
    body = prompt[len("THM ") : -len(" PRF")]
    lhs, rhs = body.split(" SEQ ")
    prem = lhs.split(" , ")
    return f"THM {' , '.join(reversed(prem))} SEQ {rhs} PRF" if len(prem) > 1 else prompt


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_excludes_renamings_and_premise_permutations(tmp_path: Path) -> None:
    rows = k12_rows(60)
    multi = [r for r in rows if r["n_prem"] >= 2][:3]
    single = [r for r in rows if r["n_prem"] < 2][:3]
    perm = {"P": "Q", "Q": "R", "R": "S", "S": "P"}
    evals = [{"prompt": rename(r["prompt"], perm)} for r in single]
    evals += [{"prompt": swap_premises(rename(r["prompt"], perm))} for r in multi]
    ex = sl.Exclusion({"evalset": write_jsonl(tmp_path / "eval.jsonl", evals)})
    picked, meta = sl.select(rows, n=len(rows), seed=1, exclusion=ex)
    got = {r["prompt"] for r in picked}
    for r in multi + single:
        assert r["prompt"] not in got
    assert meta["rows_excluded"] == len(multi) + len(single)
    assert meta["rows_excluded_by_set"] == {"evalset": len(multi) + len(single)}


def test_default_exclusions_cover_every_eval_set() -> None:
    ex = sl.default_excludes()
    labels = set(ex)
    assert {"textbook72_dev", "fork_p2_heldout", "pilot_examples"} <= labels
    assert {
        "ood_charles_long_reach",
        "ood_or_reorder",
        "ood_and_reorder",
        "ood_commute",
        "ood_case_product",
    } <= labels
    missing = [str(p) for p in ex.values() if not p.exists()]
    assert not missing, missing


def test_default_exclusions_drop_renamed_eval_items() -> None:
    """One item of every default evaluation set, atoms renamed: none may be selected."""
    ex_sets = sl.default_excludes()
    ex = sl.Exclusion(ex_sets)
    perm = {"P": "S", "S": "P", "Q": "R", "R": "Q"}
    for label, path in ex_sets.items():
        first = sl.read_jsonl(path)[0]
        assert label in ex.hits(*sl.keys_of(rename(first["prompt"], perm))), label


def test_balanced_strata_and_determinism() -> None:
    rows = k12_rows(400)
    a, meta = sl.select(rows, n=40, seed=7, exclusion=None)
    b, _ = sl.select(rows, n=40, seed=7, exclusion=None)
    c, _ = sl.select(rows, n=40, seed=8, exclusion=None)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]
    assert len(a) == 40 and len({r["id"] for r in a}) == 40
    cells = [c for c, v in meta["cells"].items() if v["available"]]
    picked = [c for c, v in meta["cells"].items() if v["selected"]]
    assert len(picked) == min(40, len(cells))
    for r in a:
        assert r["n_steps"] >= 1 and r["source"].startswith("K12 row")
        assert set(r["strata"]) == set(sl.STRATA)
        assert r["strata"]["ore"] == ("ORE" in r["rules"])
        assert r["strata"]["classical"] == ("DN" in r["rules"])


def test_proportional_allocation_matches_cell_sizes() -> None:
    rows = k12_rows(400)
    _, meta = sl.select(rows, n=100, seed=3, exclusion=None, dims=("prem",), alloc="proportional")
    tot = sum(v["available"] for v in meta["cells"].values())
    for v in meta["cells"].values():
        assert abs(v["selected"] - 100 * v["available"] / tot) <= 1.0


def test_run_writes_selection_with_provenance(tmp_path: Path) -> None:
    src = write_jsonl(tmp_path / "src.jsonl", k12_rows(80))
    ev = write_jsonl(tmp_path / "ev.jsonl", [{"prompt": sl.read_jsonl(src)[0]["prompt"]}])
    path, meta = sl.run(tmp_path / "run", src, n=10, seed=2, excludes={"ev": ev})
    sel = sl.read_jsonl(path)
    assert len(sel) == 10
    assert sl.read_jsonl(src)[0]["prompt"] not in {r["prompt"] for r in sel}
    on_disk = json.loads((tmp_path / "run" / "selection.meta.json").read_text())
    assert on_disk["source"]["sha256"] == sl.sha256(src)
    assert on_disk["exclusions"]["ev"]["n_items"] == 1
    assert on_disk["output_sha256"] == sl.sha256(path)
    assert on_disk["seed"] == 2 and "git" in on_disk and meta["n_selected"] == 10
