"""Tests of the tie-strength variants beyond the corpus: names, launcher plans, the held-out rule in pretraining, the
tag check's sets and corpus guard, and the gate summary. CPU only; no GPU, no launch.

    cd author_tags_20261110 && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
        $PY -m pytest --confcutdir=. -p no:cacheprovider -q test_tag_variants.py

The tests that need the built variants (tagcorpus.py --route-share) or the three default bases' check reads are
skipped when those files are missing.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

HAVE = {v: T.corpus_path(v).exists() and T.manifest_path(v).exists() for v in T.VARIANTS}
need_variants = pytest.mark.skipif(not all(HAVE.values()), reason="build the variants first")
CHECK = T.AT / "reads" / "check"
SMOKE_BASE = (
    T.AT / "smoke" / "pre" / "stage1_small_s0.pt"
)  # a toy base trained on the default corpus


# ------------------------------------------------------------------------------------------------ names
def test_default_names_unchanged() -> None:
    assert T.corpus_path() == T.CORPUS and T.manifest_path() == T.MANIFEST
    assert T.base_path(0) == T.AT / "bases" / "stage1_tagAB_s0.pt"
    assert T.base_name(2) == "tagAB_s2"


def test_variant_names_round_trip() -> None:
    for v in T.VARIANTS:
        for s in T.SEEDS:
            assert T.base_path(s, v) == T.AT / "bases" / f"stage1_tagAB_{v}_s{s}.pt"
            assert T.parse_base_name(T.base_name(s, v)) == (v, s)
            assert T.parse_base_name(T.base_path(s, v).name) == (v, s)
        assert T.corpus_path(v).name == f"train_k12dose1_tagAB_{v}.jsonl.gz"
    assert T.parse_base_name("tagAB_s1") == (None, 1)
    for bad in ("tagAB_p25_s0", "small_s0", "tagAB_p10", "x"):
        with pytest.raises(ValueError):
            T.parse_base_name(bad)
    assert (
        T.n_ordinary_a(0.10) == 13950 and T.n_ordinary_a(0.50) == 1550 and T.n_ordinary_a(1.0) == 0
    )


# ------------------------------------------------------------------------------------------------ launcher plans
def _ns(**kw: Any) -> argparse.Namespace:
    d = dict(gpu=0, seeds="0", tags="A,B,none", variant=None, bases=None, line=None, seg=1, resume=None,
             allow_unreadable=False, dry_run=True)  # fmt: skip
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture(scope="module")
def TL() -> Any:
    import tag_launch

    return tag_launch


@need_variants
def test_pretrain_plan_for_a_variant(TL: Any) -> None:
    for v in T.VARIANTS:
        if T.base_path(0, v).exists():
            continue  # already trained: the plan refuses it (below)
        job, steps = TL.plan(_ns(cmd="pretrain", variant=v))
        assert job == f"pretrain_{v}_s0" and [s["name"] for s in steps] == [
            f"pretrain_{v}_s0",
            f"sample_{v}_s0",
        ]
        argv = steps[0]["argv"]
        assert argv[argv.index("--out") + 1] == str(T.base_path(0, v))
        assert argv[argv.index("--variant") + 1] == v
        assert (
            "--steps" not in argv
        )  # tag_pretrain's default: DOSE1_STEPS[seed], as for the default bases
        assert steps[0]["timeout_s"] == 5400 and steps[1]["argv"][
            steps[1]["argv"].index("--ckpt") + 1
        ] == str(T.base_path(0, v))


def test_pretrain_plan_default_unchanged_and_never_overwrites(TL: Any) -> None:
    if not T.base_path(0).exists():
        pytest.skip("no default base")
    with pytest.raises(SystemExit, match="never overwritten"):
        TL.plan(_ns(cmd="pretrain"))


@need_variants
def test_check_plan(TL: Any) -> None:
    names = [T.base_name(0, v) for v in T.VARIANTS]
    if any((CHECK / f"{n}.json").exists() for n in names):
        pytest.skip("variant checks already written")
    job, steps = TL.plan(_ns(cmd="check", bases=",".join(names)))
    assert job == "check_p10_s0-p50_s0-p100_s0"
    for st, v in zip(steps, T.VARIANTS, strict=True):
        argv = st["argv"]
        assert argv[1].endswith("tag_check.py") and st["timeout_s"] == 2700
        assert argv[argv.index("--ckpt") + 1] == str(T.base_path(0, v))
        assert argv[argv.index("--corpus") + 1] == str(T.corpus_path(v))
        assert argv[argv.index("--out") + 1] == str(CHECK / f"tagAB_{v}_s0.json")
        assert argv[argv.index("--tags") + 1] == "A,B,none"
    with pytest.raises(SystemExit, match="named twice"):
        TL.plan(_ns(cmd="check", bases="tagAB_p10_s0,stage1_tagAB_p10_s0.pt"))
    with pytest.raises(ValueError):
        TL.plan(_ns(cmd="check", bases="small_s0"))


def test_check_plan_never_overwrites_a_check(TL: Any) -> None:
    if not (CHECK / "tagAB_s0.json").exists():
        pytest.skip("no default check read")
    with pytest.raises(SystemExit, match="never overwritten"):
        TL.plan(_ns(cmd="check", bases="tagAB_s0"))


def test_check_plan_needs_the_base_unless_dry_run(TL: Any) -> None:
    if T.base_path(0, "p50").exists() or (CHECK / "tagAB_p50_s0.json").exists():
        pytest.skip("p50 base or check exists")
    with pytest.raises(SystemExit, match="no base"):
        TL.plan(_ns(cmd="check", bases="tagAB_p50_s0", dry_run=False))


# ------------------------------------------------------------------------------------------------ pretraining
@need_variants
def test_heldout_rule_is_the_variants_rule() -> None:
    """tag_pretrain tags the untagged held-out records by the variant's pointwise rule, which reproduces the corpus."""
    import tag_pretrain as TP

    for v in T.VARIANTS:
        cfg = TP.variant_cfg(v)
        assert cfg["variant"] == v and cfg["manifest"] == T.manifest_path(v)
        with gzip.open(T.corpus_path(v), "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                assert cfg["heldout_tag"](r["key"], bool(r.get("coe_dose"))) == r[T.TAG_FIELD]
                if i >= 20000:
                    break
    hp100 = TP.variant_cfg("p100")["heldout_tag"]
    assert {hp100(f"unseen{i}", False) for i in range(5000)} == {"B"} and hp100("x", True) == "A"
    hp10 = TP.variant_cfg("p10")["heldout_tag"]
    share = sum(hp10(f"unseen{i}", False) == "A" for i in range(20000)) / 20000
    assert (
        abs(share - 13950 / 153450) < 0.01
    )  # an unseen ordinary record is A at the corpus's ordinary rate
    assert TP._cfg["heldout_tag"] is T.corpus_tag  # the default is untouched by variant_cfg


# ------------------------------------------------------------------------------------------------ the tag check
@need_variants
def test_check_sets_are_the_same_records_under_a_variant() -> None:
    """tag_check's four sets do not depend on the tags; only the per-record tag (own_A / own_B) follows the corpus."""
    import tag_check as TCK

    base = TCK.sets(40, T.CORPUS)
    for v in T.VARIANTS:
        var = TCK.sets(40, T.corpus_path(v))
        for name in base:
            assert [r["key"] for r in var[name]] == [r["key"] for r in base[name]], (v, name)
        with gzip.open(T.corpus_path(v), "rt", encoding="utf-8") as f:
            tags = {r["key"]: r[T.TAG_FIELD] for r in map(json.loads, f)}
        assert all(
            r[T.TAG_FIELD] == tags[r["key"]]
            for name in ("dose_seen", "ctrl_seen")
            for r in var[name]
        )
        assert all(r[T.TAG_FIELD] == "A" for r in var["dose_seen"])
        assert all(
            T.TAG_FIELD not in r for name in ("sigma_unseen", "ctrl_unseen") for r in var[name]
        )
    p100 = TCK.sets(40, T.corpus_path("p100"))
    assert {r[T.TAG_FIELD] for r in p100["ctrl_seen"]} == {"B"}


@need_variants
def test_check_refuses_a_corpus_the_base_was_not_trained_on() -> None:
    if not SMOKE_BASE.exists():
        pytest.skip("no smoke base")
    import tag_check as TCK

    with pytest.raises(SystemExit, match="not the corpus"):
        TCK.main(["--ckpt", str(SMOKE_BASE), "--out", "/dev/null", "--device", "cpu", "--n", "2",
                  "--corpus", str(T.corpus_path("p10"))])  # fmt: skip


# ------------------------------------------------------------------------------------------------ the gate
def _res(open_a: float, open_b: float, all_a: float, all_b: float) -> dict[str, Any]:
    def g(x: float) -> dict[str, Any]:
        return {"mean_logp_action": x, "actions": 10}

    return {"sets": {"sigma_unseen": {"A": {"open": g(open_a)}, "B": {"open": g(open_b)}, "none": {"open": g(-9.0)}},
                     "ctrl_unseen": {"A": {"all": g(all_a)}, "B": {"all": g(all_b)}}}}  # fmt: skip


def test_gate_row_arithmetic() -> None:
    import tag_gate as TG

    r = TG.gate_row(_res(-3.0, -4.0, -1.0, -1.25))
    assert r["d_open"] == pytest.approx(1.0) and r["d_ctrl"] == pytest.approx(0.25)
    assert r["d_open_minus_d_ctrl"] == pytest.approx(0.75) and r["passes"]
    assert not TG.gate_row(_res(-3.0, -3.5, -1.0, -1.0))["passes"]
    assert TG.gate_row(_res(-3.0, -3.0 - math.log(2), -1.0, -1.0))["passes"]  # the bar is inclusive


def test_gate_reproduces_the_default_bases() -> None:
    """GATE_RESULT.md: d_open - d_ctrl = +0.309, +0.090, +0.078; all three fail."""
    if not all((CHECK / f"tagAB_s{s}.json").exists() for s in T.SEEDS):
        pytest.skip("no check reads")
    import tag_gate as TG

    rows = TG.rows(CHECK)
    got = [round(rows[f"tagAB_s{s}"]["d_open_minus_d_ctrl"], 3) for s in T.SEEDS]
    assert got == [0.309, 0.090, 0.078]
    assert not any(rows[f"tagAB_s{s}"]["passes"] for s in T.SEEDS)
    assert list(rows)[:3] == ["tagAB_s0", "tagAB_s1", "tagAB_s2"]
