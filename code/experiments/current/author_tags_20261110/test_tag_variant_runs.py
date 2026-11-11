"""Tests of the tie-strength variants in the base read and the RL lines: names, read folders and ids, launcher plans,
refusals (missing bases, overwrites, wrong variant, wrong resume), default plans and outputs unchanged, the variant
line's D from its own base's sidecar, and the after-the-fact tag audit. CPU only; nothing touches a GPU or launches.

    cd author_tags_20261110 && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
        $PY -m pytest --confcutdir=. -p no:cacheprovider -q test_tag_variant_runs.py

Most tests run in a temporary AT (tag_core.AT monkeypatched; bases are placeholder files with a .tag.json). The ones
that compare with the real tree (the default read's ckpts.json and gate summary, the p10 base) skip when it is absent.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

REAL_AT = T.AT
H = str(T.HERE)


def _ns(**kw: Any) -> argparse.Namespace:
    d = dict(gpu=0, seeds="0,1,2", tags="A,B,none", variant=None, bases=None, line=None, seg=1, resume=None,
             allow_unreadable=False, dry_run=True)  # fmt: skip
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture()
def at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A temporary AT: every tag_core path (bases, reads, runs, checks) is read at call time from T.AT."""
    monkeypatch.setattr(T, "AT", tmp_path)
    (tmp_path / "bases").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


def fake_base(
    seed: int, variant: str | None = None, D: int = 470_000_000, side_variant: Any = "same"
) -> Path:
    p = T.base_path(seed, variant)
    p.write_bytes(b"placeholder")
    sv = variant if side_variant == "same" else side_variant
    at = {"mode": T.MODE, **({"variant": sv} if sv else {})}
    p.with_suffix(".tag.json").write_text(json.dumps({"train_tokens": D, "author_tags": at}))
    return p


@pytest.fixture(scope="module")
def TL() -> Any:
    import tag_launch

    return tag_launch


@pytest.fixture(scope="module")
def TR() -> Any:
    import tag_read

    return tag_read


@pytest.fixture(scope="module")
def TLN() -> Any:
    import tag_line

    return tag_line


@pytest.fixture(scope="module")
def TM() -> Any:
    import tag_measure

    return tag_measure


# ------------------------------------------------------------------------------------------------ names
def test_default_lines_unchanged_and_variant_lines_added() -> None:
    old = {f"t{t}_s{s}": {"tag": t, "seed": s, "base": f"stage1_tagAB_s{s}.pt", "ladder": f"la_dose1_s{s}"}
           for t in ("B", "A") for s in T.SEEDS}  # fmt: skip
    assert {k: T.LINES[k] for k in old} == old and list(T.LINES)[:6] == list(old)
    assert len(T.LINES) == 6 + 6 * len(T.VARIANTS)
    for name, L in T.LINES.items():
        tag, v, s = T.parse_line_name(name)
        assert (tag, v, s) == (L["tag"], L.get("variant"), L["seed"]) and T.line_name(
            tag, s, v
        ) == name
        assert L["base"] == T.base_path(s, v).name and L["ladder"] == f"la_dose1_s{s}"
        assert "_it" not in name  # tag_line parses snapshots as <line>_it<i>_snap.pt
    assert T.LINES["tB_p10_s0"] == {
        "tag": "B",
        "seed": 0,
        "base": "stage1_tagAB_p10_s0.pt",
        "ladder": "la_dose1_s0",
        "variant": "p10",
    }
    for bad in ("tC_s0", "tB_p25_s0", "tB_p10", "tB_s0_x", "B_s0", "tAB_s0"):
        with pytest.raises(ValueError):
            T.parse_line_name(bad)


def test_read_roots_ids_and_check_paths() -> None:
    import tag_read

    assert T.read_root() == REAL_AT / "reads" / "pc" == tag_read.ROOT
    assert T.read_root("p10") == REAL_AT / "reads" / "pc_p10"
    assert len({T.read_root(v) for v in (None, *T.VARIANTS)}) == 1 + len(T.VARIANTS)
    assert T.read_id(0, "A") == "tagAB_s0@A" and T.read_id(2, "none", "p10") == "tagAB_p10_s2@none"
    assert T.check_path("tagAB_p10_s1") == REAL_AT / "reads" / "check" / "tagAB_p10_s1.json"
    with pytest.raises(T.TagError):
        T.read_id(0, "C")


# ------------------------------------------------------------------------------------------------ tag_read
def test_ckpt_rows_default_unchanged_and_variant(TR: Any) -> None:
    rows = TR.ckpt_rows([0, 1, 2], ["A", "B", "none"])
    f = REAL_AT / "reads" / "pc" / "ckpts.json"
    if f.exists():  # the default read's own file: same entries, field for field
        assert list(rows.values()) == json.loads(f.read_text())
        assert TR.merge_ckpts(json.loads(f.read_text()), rows) == json.loads(f.read_text())
    var = TR.ckpt_rows([0, 1, 2], ["A", "B", "none"], variant="p10")
    assert list(var) == [T.read_id(s, t, "p10") for s in (0, 1, 2) for t in ("A", "B", "none")]
    for (cd, d), (cv, v) in zip(rows.items(), var.items(), strict=True):
        assert (
            v["path"] == str(T.base_path(d["seed"], "p10"))
            and v["variant"] == "p10"
            and "variant" not in d
        )
        assert {k: x for k, x in v.items() if k not in ("ckpt", "path", "variant", "exists")} == {
            k: x for k, x in d.items() if k not in ("ckpt", "path", "exists")
        }  # same noise seeds, tags, dose
        assert cv == cd.replace("tagAB_", "tagAB_p10_")


def test_merge_ckpts(TR: Any) -> None:
    a = TR.ckpt_rows([0], ["A", "B"], variant="p10")
    first = TR.merge_ckpts([], a)
    assert first == list(a.values())
    assert TR.merge_ckpts(first, a) == first  # a rerun (resume) changes nothing
    b = TR.ckpt_rows(
        [1], ["A", "B"], variant="p10"
    )  # a later run of another seed: appended, never replacing
    both = TR.merge_ckpts(first, b)
    assert [c["ckpt"] for c in both] == [*a, *b] and [c["priority"] for c in both] == [0, 1, 2, 3]
    bad = {k: {**c, "noise_seed": 7} for k, c in a.items()}
    with pytest.raises(SystemExit, match="not mixing"):
        TR.merge_ckpts(first, bad)
    other = TR.ckpt_rows([0], ["A"], base="/elsewhere.pt", variant="p10")
    with pytest.raises(SystemExit, match="path"):
        TR.merge_ckpts(first, other)


def test_read_gpu_refuses_missing_base_before_anything(
    at: Path, TR: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_base(0, "p10")

    def no_pc() -> None:
        raise AssertionError("reached pc_run (the gate) with a missing base")

    monkeypatch.setattr(TR, "_pc", no_pc)
    with pytest.raises(SystemExit, match="nothing taken") as e:
        TR.main(["gpu", "--variant", "p10", "--seeds", "0,1,2", "--gpu", "0"])
    assert "stage1_tagAB_p10_s1.pt" in str(e.value) and "stage1_tagAB_p10_s2.pt" in str(e.value)
    assert "--seeds 1 --variant p10" in str(e.value) and "stage1_tagAB_p10_s0.pt" not in str(
        e.value
    )
    assert not (at / "reads").exists()  # nothing written


def test_read_gpu_refuses_a_base_of_another_variant(
    at: Path, TR: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_base(0, "p10", side_variant="p50")
    monkeypatch.setattr(TR, "_pc", lambda: (_ for _ in ()).throw(AssertionError("reached pc_run")))
    with pytest.raises(SystemExit, match="variant 'p50', read as 'p10'"):
        TR.main(["gpu", "--variant", "p10", "--seeds", "0", "--gpu", "0"])
    fake_base(1, None, side_variant="p10")  # a default-named base whose sidecar says p10
    with pytest.raises(SystemExit, match="variant 'p10', read as None"):
        TR.main(["gpu", "--seeds", "1", "--gpu", "0"])


def test_read_root_guards(at: Path, TR: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TR, "_pc", lambda: (_ for _ in ()).throw(AssertionError("reached pc_run")))
    with pytest.raises(SystemExit, match="another read's folder"):
        TR.main(["rows", "--variant", "p10", "--seeds", "0", "--root", str(T.read_root())])
    with pytest.raises(SystemExit, match="another read's folder"):
        TR.main(["rows", "--variant", "p10", "--seeds", "0", "--root", str(T.read_root("p50"))])
    with pytest.raises(SystemExit, match="another read's folder"):
        TR.main(["rows", "--seeds", "0", "--root", str(T.read_root("p10"))])
    with pytest.raises(AssertionError, match="CPU runs"):
        TR.main(["gpu", "--variant", "p10", "--seeds", "0", "--device", "cpu"])


# ------------------------------------------------------------------------------------------------ launcher: read
def old_read_plan(seeds: str, tags: str, gpu: int) -> tuple[str, list[dict]]:
    """The read plan as it was before variants (b9b6c48), for the default bases."""
    py, est = T.PY, {"read_gpu_per_base_tag": 0.08, "rows": 0.1, "check": 0.1}
    ss = [int(s) for s in seeds.split(",")]
    n = len(ss) * len(tags.split(","))

    def step(name: str, argv: list[str], hours: float, e: float, env: dict | None = None) -> dict:
        return {
            "name": name,
            "argv": argv,
            "timeout_s": int(hours * 3600),
            "est_h": e,
            "env": env or {},
        }

    steps = [step("pc_gpu", [py, f"{H}/tag_read.py", "gpu", "--seeds", seeds, "--tags", tags, "--gpu", str(gpu)],
                  max(1.0, 2.5 * n * est["read_gpu_per_base_tag"]), n * est["read_gpu_per_base_tag"]),
             step("pc_rows", [py, f"{H}/tag_read.py", "rows", "--seeds", seeds, "--tags", tags], 1.0, est["rows"],
                  {"CUDA_VISIBLE_DEVICES": ""})]  # fmt: skip
    for s in ss:
        steps.append(step(f"check_s{s}", [py, f"{H}/tag_check.py", "--ckpt", str(T.base_path(s)), "--tags", tags,
                                          "--out", str(T.AT / "reads" / "check" / f"tagAB_s{s}.json")], 0.75, est["check"]))  # fmt: skip
    return f"read_s{'-'.join(map(str, ss))}", steps


def test_read_plan_default_identical_to_before(at: Path, TL: Any) -> None:
    for s in T.SEEDS:
        fake_base(s)
    for seeds in ("0,1,2", "1"):
        assert TL.plan(_ns(cmd="read", seeds=seeds, dry_run=False)) == old_read_plan(
            seeds, "A,B,none", 0
        )
    assert TL.plan(_ns(cmd="read", seeds="2", tags="A,B", gpu=3)) == old_read_plan("2", "A,B", 3)
    assert TL.preflight(_ns(cmd="read"))[0] == []


def test_read_plan_variant(at: Path, TL: Any) -> None:
    for s in T.SEEDS:
        fake_base(s, "p10")
    c0 = T.check_path("tagAB_p10_s0")
    c0.parent.mkdir(parents=True)
    c0.write_text("{}")  # seed 0's check exists: never re-run
    job, steps = TL.plan(_ns(cmd="read", variant="p10", dry_run=False))
    assert job == "read_p10_s0-1-2"
    assert [s["name"] for s in steps] == ["pc_gpu", "pc_rows", "check_p10_s1", "check_p10_s2"]
    _, dsteps = old_read_plan("0,1,2", "A,B,none", 0)
    for new, old in zip(steps[:2], dsteps[:2], strict=True):
        assert new["argv"] == [*old["argv"], "--variant", "p10"]
        assert (new["timeout_s"], new["est_h"], new["env"]) == (
            old["timeout_s"],
            old["est_h"],
            old["env"],
        )
    for st, s in zip(steps[2:], (1, 2), strict=True):
        assert st["argv"] == [T.PY, f"{H}/tag_check.py", "--ckpt", str(T.base_path(s, "p10")), "--tags", "A,B,none",
                              "--corpus", str(T.corpus_path("p10")), "--out", str(T.check_path(f"tagAB_p10_s{s}"))]  # fmt: skip
        assert st["timeout_s"] == 2700
    problems, notes = TL.preflight(_ns(cmd="read", variant="p10"))
    assert (
        problems == []
        and any("pc_p10" in n for n in notes)
        and any("tagAB_p10_s0.json exists" in n for n in notes)
    )


def test_read_plan_refuses_missing_bases_before_the_gate(
    at: Path, TL: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_base(0, "p10")

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("the gate or a launch was reached")

    monkeypatch.setattr(TL, "gate", boom)
    monkeypatch.setattr(TL.subprocess, "Popen", boom)
    a = _ns(cmd="read", variant="p10", dry_run=False)
    with pytest.raises(SystemExit, match="refused") as e:
        TL.launch(a)
    msg = str(e.value)
    assert (
        "stage1_tagAB_p10_s1.pt" in msg
        and "stage1_tagAB_p10_s2.pt" in msg
        and "pretrain --gpu N --seeds 2 --variant p10" in msg
    )
    assert not (at / "reads").exists() and not list((at / "logs").iterdir())
    job, steps = TL.plan(_ns(cmd="read", variant="p10"))  # a dry run still shows the plan ...
    assert job == "read_p10_s0-1-2" and len(steps) == 5
    assert len(TL.preflight(_ns(cmd="read", variant="p10"))[0]) == 2  # ... and the two refusals
    TL.plan(_ns(cmd="read", variant="p10", seeds="0", dry_run=False))  # seed 0 alone is fine


def test_read_plan_never_overwrites_a_read(at: Path, TL: Any) -> None:
    for s in T.SEEDS:
        fake_base(s)
        fake_base(s, "p10")
    r = T.read_root("p10") / "results"
    r.mkdir(parents=True)
    (r / "tagAB_p10_s1@B.jsonl").write_text("")
    with pytest.raises(SystemExit, match=r"never overwritten"):
        TL.plan(_ns(cmd="read", variant="p10", dry_run=False))
    TL.plan(
        _ns(cmd="read", variant="p10", seeds="0,2", dry_run=False)
    )  # other seeds of the variant: fine
    TL.plan(_ns(cmd="read", dry_run=False))  # the default read does not see the variant's folder
    d = T.read_root() / "results"
    d.mkdir(parents=True)
    (d / "tagAB_s0@A.jsonl").write_text("")
    with pytest.raises(SystemExit, match=r"tagAB_s0@A.*never overwritten"):
        TL.plan(_ns(cmd="read", dry_run=False))
    TL.plan(
        _ns(cmd="read", variant="p10", seeds="0", dry_run=False)
    )  # and the variant's does not see the default's


def test_read_plan_real_tree(TL: Any) -> None:
    if not (T.read_root() / "results" / "tagAB_s0@A.jsonl").exists():
        pytest.skip("no default read")
    with pytest.raises(SystemExit, match="never overwritten"):
        TL.plan(_ns(cmd="read", dry_run=False))
    _, steps = TL.plan(_ns(cmd="read", variant="p10"))
    names = [s["name"] for s in steps]
    assert names[:2] == ["pc_gpu", "pc_rows"]
    for s in T.SEEDS:
        assert (f"check_p10_s{s}" in names) != T.check_path(f"tagAB_p10_s{s}").exists()


def test_variant_flag_only_for_pretrain_and_read(TL: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for cmd in (["rl", "--line", "tB_s0"], ["check", "--bases", "tagAB_p10_s0"]):
        monkeypatch.setattr(
            sys, "argv", ["tag_launch.py", *cmd, "--gpu", "0", "--variant", "p10", "--dry-run"]
        )
        with pytest.raises(AssertionError, match="--variant is for pretrain and read"):
            TL.main()


# ------------------------------------------------------------------------------------------------ gate summary
GOLDEN_MD5 = "0c77a5b77c20d0878a4aa5a1181202d1"  # tag_gate_summary.py's output at b9b6c48 on the default read


def test_gate_summary_default_output_unchanged(capsys: pytest.CaptureFixture) -> None:
    if not all(
        (T.read_root() / "results" / f"{T.read_id(s, t)}.jsonl").exists()
        for s in T.SEEDS
        for t in ("A", "B", "none")
    ):
        pytest.skip("no default read")
    import tag_gate_summary as TGS

    TGS.main([])
    out = capsys.readouterr().out
    assert hashlib.md5(out.encode()).hexdigest() == GOLDEN_MD5
    assert (
        "uses red   s0 n=106  A    1.44  B    0.88  none    0.18  A-B  +0.56 [+0.06,+1.06]" in out
    )  # GATE_RESULT.md


def _summary_rows(seed: int, tag: str, variant: str) -> list[dict]:
    """Synthetic result rows (only the fields the summary reads): 2 theorems per set."""
    bump = {"A": 2, "B": 1, "none": 0}[tag]
    rows = []
    for st in ("strict106", "transfer_other"):
        for i in range(2):
            base = {
                "ckpt": T.read_id(seed, tag, variant),
                "set": st,
                "name": f"{st}_{i}",
                "k": 10,
                "applicable": True,
            }
            rows += [{**base, "arm": "U", "direction": "any", "solved": i + bump},
                     {**base, "arm": "U", "direction": "reductio", "solved": 0, "used": bump, "first": bump},
                     {**base, "arm": "E", "direction": "reductio", "logp_open": -1.0 * (3 - bump)},
                     {**base, "arm": "F", "direction": "reductio", "solved": 3 * bump}]  # fmt: skip
    return rows


def test_gate_summary_variant(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import tag_gate_summary as TGS

    res = tmp_path / "results"
    res.mkdir()
    for t in ("A", "B", "none"):
        (res / f"tagAB_p10_s0@{t}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in _summary_rows(0, t, "p10"))
        )
        (res / f"tagAB_s0@{t}.jsonl").write_text(
            ""
        )  # a default-named file is never read by a variant summary
    TGS.main(["--variant", "p10", "--root", str(tmp_path), "--seeds", "0,1"])
    out = capsys.readouterr().out.splitlines()
    assert (
        out[0].startswith("variant p10")
        and out[1] == "missing (no read under every tag of A, B, none): tagAB_p10_s1"
    )
    rows = [x for x in out if x.startswith(("uses red", "forced F", "logP E"))]
    assert len(rows) == 6 and all(" s0 n=2 " in x for x in rows)
    assert "A   20.00  B   10.00  none    0.00  A-B +10.00" in rows[0]  # uses red: 2/10, 1/10, 0/10
    assert (
        "A   -1.00  B   -2.00  none   -3.00  A-B  +1.00"
        in [x for x in rows if x.startswith("logP E")][0]
    )


# ------------------------------------------------------------------------------------------------ RL lines
def test_variant_line_recipe_is_the_default_recipe_on_the_variant_base(at: Path, TLN: Any) -> None:
    for v in T.VARIANTS:
        for t in ("A", "B"):
            for s in T.SEEDS:
                line, ref = T.line_name(t, s, v), T.line_name(t, s)
                for resume in (None, "/x/seg1/ckpts/r_it5_rolling.pt"):
                    seg = at / "runs" / line / "seg2"
                    got, want = (
                        TLN.grpo_argv(line, seg, 400, resume),
                        TLN.grpo_argv(ref, seg, 400, resume),
                    )
                    base = str(T.base_path(s, v))
                    for f, val in (("--init", base), ("--kl_anchor_ref", base), ("--name", line)):
                        assert got[got.index(f) + 1] == val
                        want[want.index(f) + 1] = val
                    assert (
                        got == want
                    )  # everything else is the default line's (the dd1 recipe of that seed)
    assert TLN.grpo_cmd("tB_s0", ["--x"]) == [
        T.PY,
        f"{H}/tag_grpo.py",
        "--tag",
        "B",
        "--",
        "--x",
    ]  # unchanged
    assert TLN.grpo_cmd("tA_p10_s2", ["--x"]) == [
        T.PY,
        f"{H}/tag_grpo.py",
        "--tag",
        "A",
        "--variant",
        "p10",
        "--",
        "--x",
    ]


def test_check_line_base(at: Path) -> None:
    with pytest.raises(
        SystemExit,
        match=r"train the base first: tag_launch.py pretrain --gpu N --seeds 1 --variant p10",
    ):
        T.check_line_base("tB_p10_s1")
    fake_base(1, "p10", D=123)
    assert T.check_line_base("tB_p10_s1")["train_tokens"] == 123
    fake_base(2, "p10", side_variant="p50")
    with pytest.raises(SystemExit, match="variant 'p50'; the line wants 'p10'"):
        T.check_line_base("tA_p10_s2")
    fake_base(0)
    assert T.check_line_base("tA_s0")["train_tokens"] == 470_000_000
    T.base_path(0, "p50").write_bytes(b"x")
    T.base_path(0, "p50").with_suffix(".tag.json").write_text(
        json.dumps({"author_tags": {"variant": "p50"}})
    )
    with pytest.raises(SystemExit, match="no train_tokens"):
        T.check_line_base("tB_p50_s0")


def test_line_D_and_view_use_the_variant_base_sidecar(at: Path, TM: Any) -> None:
    fake_base(0, None, D=1_000_000)
    fake_base(0, "p10", D=2_000_000)
    assert TM.line_D("tB_s0") == 1_000_000 and TM.line_D("tB_p10_s0") == 2_000_000
    rows = [{"iter": i, "sampled_gen_tokens": 1000, "sampled_prefill_tokens": 500, "item_prompt_tokens": 300,
             "item_gen_tokens": 200, "trained_tokens": 400} for i in (1, 2)]  # fmt: skip
    segs = [
        {"seg": 1, "start": "2026-11-11T00:00:00Z", "end": "2026-11-11T01:00:00Z", "resume": None}
    ]
    views = {}
    for line in ("tB_s0", "tB_p10_s0"):
        d = at / "runs" / line / "seg1"
        d.mkdir(parents=True)
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        views[line] = TM.line_view(line, segs=segs)
    d, v = views["tB_s0"], views["tB_p10_s0"]
    per_row = (2 * 1500 + 6 * 500) / 6  # Dan's rule: one row's compute in units of 6 N
    assert (
        d["D"] == 1_000_000
        and v["D"] == 2_000_000
        and v["variant"] == "p10"
        and v["base"] == "stage1_tagAB_p10_s0.pt"
    )
    assert d["x_dan"] == pytest.approx(2 * per_row / 1_000_000) and v["x_dan"] == pytest.approx(
        d["x_dan"] / 2
    )
    assert (
        v["gpuh"] == pytest.approx(1.0)
        and v["x_cap"] == T.X_CAP == 1.5
        and TM.stop_reason(v) is None
    )
    assert v["matched"] == d["matched"] == []  # the same ladder logic (no reads yet)


def test_segment_problems(at: Path, TM: Any) -> None:
    line = "tB_p10_s0"
    ld = at / "runs" / line
    assert TM.segment_problems(line, 1, None) == []
    (ld / "seg1" / "ckpts").mkdir(parents=True)
    assert any("never re-run" in x for x in TM.segment_problems(line, 1, None))
    assert any("needs --resume" in x for x in TM.segment_problems(line, 2, None))
    own = ld / "seg1" / "ckpts" / f"{line}_it25_rolling.pt"
    assert any(x == f"no {own}" for x in TM.segment_problems(line, 2, str(own)))
    own.write_bytes(b"x")
    assert TM.segment_problems(line, 2, str(own)) == []
    alarm = ld / "seg1" / "ckpts" / f"{line}_it30_alarm.pt"
    alarm.write_bytes(b"x")
    assert TM.segment_problems(line, 2, str(alarm)) == []
    other = (
        at / "runs" / "tB_s0" / "seg1" / "ckpts" / "tB_s0_it25_rolling.pt"
    )  # the default line's checkpoint
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")
    assert any("is not a" in x for x in TM.segment_problems(line, 2, str(other)))
    snap = (
        ld / "seg1" / "ckpts" / f"{line}_it30_snap.pt"
    )  # a model-only snapshot has no training state
    assert any("is not a" in x for x in TM.segment_problems(line, 2, str(snap)))
    assert any("starts from the base" in x for x in TM.segment_problems(line, 1, str(own)))
    (ld / "seg3").mkdir()
    assert any("segments [3] after 2" in x for x in TM.segment_problems(line, 2, str(own)))


def test_rl_plan(at: Path, TL: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_base(0, "p10", D=470_568_815)
    job, steps = TL.plan(_ns(cmd="rl", line="tB_p10_s0", dry_run=False))
    assert job == "rl_tB_p10_s0_seg1" and len(steps) == 1
    assert steps[0]["argv"] == [
        T.PY,
        f"{H}/tag_line.py",
        "--line",
        "tB_p10_s0",
        "--gpu",
        "0",
        "--seg",
        "1",
    ]
    assert steps[0]["timeout_s"] == 8 * 3600 and steps[0]["name"] == "line_tB_p10_s0"
    problems, notes = TL.preflight(_ns(cmd="rl", line="tB_p10_s0"))
    assert problems == [] and "D 470,568,815" in notes[0] and "stage1_tagAB_p10_s0.pt" in notes[0]
    fake_base(1)
    job, steps = TL.plan(
        _ns(cmd="rl", line="tA_s1", gpu=2, dry_run=False)
    )  # a default line: as before
    assert (job, steps[0]["argv"]) == (
        "rl_tA_s1_seg1",
        [T.PY, f"{H}/tag_line.py", "--line", "tA_s1", "--gpu", "2", "--seg", "1"],
    )
    monkeypatch.setattr(
        TL, "gate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gate reached"))
    )
    with pytest.raises(SystemExit, match=r"refused.*stage1_tagAB_p10_s1.pt"):
        TL.launch(
            _ns(cmd="rl", line="tA_p10_s1", dry_run=False)
        )  # no base: refused before the gate
    (at / "runs" / "tB_p10_s0" / "seg1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="never re-run"):
        TL.launch(_ns(cmd="rl", line="tB_p10_s0", dry_run=False))
    with pytest.raises(SystemExit, match="is not a"):
        TL.plan(
            _ns(
                cmd="rl",
                line="tB_p10_s0",
                seg=2,
                resume=str(at / "runs/tB_s0/seg1/ckpts/tB_s0_it5_rolling.pt"),
                dry_run=False,
            )
        )
    assert (
        TL.plan(_ns(cmd="rl", line="tA_p10_s1"))[0] == "rl_tA_p10_s1_seg1"
    )  # a dry run shows the plan anyway


def test_tag_line_dry_run_and_refusals(
    at: Path, TLN: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        TLN, "Seg", lambda a: (_ for _ in ()).throw(AssertionError("a segment was started"))
    )
    monkeypatch.setattr(
        sys, "argv", ["tag_line.py", "--line", "tA_p10_s1", "--gpu", "0", "--dry-run"]
    )
    TLN.main()
    j = json.loads(capsys.readouterr().out)
    assert (
        j["base_exists"] is False and j["D"] is None and "train the base first" in j["refuses"][0]
    )
    fake_base(1, "p10", D=321)
    TLN.main()
    j = json.loads(capsys.readouterr().out)
    assert (j["line"], j["tag"], j["variant"], j["D"], j["ladder"], j["refuses"]) == (
        "tA_p10_s1",
        "A",
        "p10",
        321,
        "la_dose1_s1",
        [],
    )
    assert j["base"] == str(T.base_path(1, "p10")) and j["seg_dir"] == str(
        at / "runs" / "tA_p10_s1" / "seg1"
    )
    assert j["cmd"][:7] == [T.PY, f"{H}/tag_grpo.py", "--tag", "A", "--variant", "p10", "--"]
    (at / "runs" / "tA_p10_s1" / "seg1").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["tag_line.py", "--line", "tA_p10_s1", "--gpu", "0"])
    with pytest.raises(SystemExit, match="never re-run"):
        TLN.main()


def test_segment_reads_the_variant_base_and_its_own_snapshots(
    at: Path, TLN: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = "tB_p10_s2"
    fake_base(2, "p10")
    ck = at / "runs" / line / "seg1" / "ckpts"
    ck.mkdir(parents=True)
    for n in (
        f"{line}_it10_snap.pt",
        f"{line}_it20_snap.pt",
        f"{line}_it15_rolling.pt",
        "tB_s2_it30_snap.pt",
    ):
        (ck / n).write_bytes(b"x")
    cmds: list[list[str]] = []

    class Done:
        returncode = 0

    def run(cmd: list[str], **kw: Any) -> Done:
        cmds.append(cmd)
        assert kw["env"][T.ENV] == "B"
        return Done()

    monkeypatch.setattr(TLN.subprocess, "run", run)
    a = argparse.Namespace(
        line=line, gpu=0, seg=1, resume=None, smoke=False, iters=400, rule_every=60.0
    )
    seg = TLN.Seg(a)
    (seg.dir / "reads").mkdir(parents=True)
    seg.done.set()  # one pass of the reader
    seg.reader()
    got = [
        (
            c[c.index("--ckpt") + 1],
            c[c.index("--round") + 1],
            c[c.index("--seed") + 1],
            c[c.index("--tag") + 1],
        )
        for c in cmds
    ]
    assert got == [(str(T.base_path(2, "p10")), "0", "2", "B"), (str(ck / f"{line}_it10_snap.pt"), "10", "2", "B"),
                   (str(ck / f"{line}_it20_snap.pt"), "20", "2", "B")]  # fmt: skip
    assert all(
        c[c.index("--out") + 1].startswith(str(at / "runs" / line / "seg1" / "reads")) for c in cmds
    )


def test_tag_grpo_refuses_an_init_of_another_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p10 = T.base_path(0, "p10")
    if not p10.exists() or not T.base_path(0).exists():
        pytest.skip("no p10 or default seed-0 base")
    import tag_grpo as TG

    class Reached(Exception):
        pass

    def stop() -> None:
        raise Reached

    monkeypatch.setattr(TG.T, "install_thinking", stop)
    monkeypatch.setattr(TG.T, "set_process_tag", lambda t: t)
    out = ["--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="variant 'p10'; this run is for 'p50'"):
        TG.main(["--tag", "B", "--variant", "p50", "--", "--init", str(p10), *out])
    with pytest.raises(SystemExit, match="variant 'p10'; this run is for None"):
        TG.main(
            ["--tag", "B", "--", "--init", str(p10), *out]
        )  # a default line never starts from a variant base
    with pytest.raises(SystemExit, match="variant None; this run is for 'p10'"):
        TG.main(["--tag", "B", "--variant", "p10", "--", "--init", str(T.base_path(0)), *out])
    with pytest.raises(Reached):  # the matching variant passes the check
        TG.main(["--tag", "B", "--variant", "p10", "--", "--init", str(p10), *out])
    assert T.ckpt_meta(p10)["variant"] == "p10" and T.ckpt_meta(T.base_path(0))["variant"] is None


def test_tag_audit(at: Path, TM: Any) -> None:
    line = "tB_p10_s0"
    d = at / "runs" / line / "seg1"
    (d / "reads").mkdir(parents=True)
    segs = [{"seg": 1}]
    assert TM.tag_audit(line, segs) == ["seg1: no author_tag.json"]
    good = {"tag": "B", "init": str(T.base_path(0, "p10")), "spy": {"by_tag": {"B": 900}}}
    (d / "author_tag.json").write_text(json.dumps(good))
    (d / "reads" / "it0000.json").write_text(
        json.dumps({"author_tag": "B", "tag_spy": {"by_tag": {"B": 400}}})
    )
    (d / "reads" / "it0010.json").write_text("{")  # a read being written is skipped
    assert TM.tag_audit(line, segs) == []
    (d / "author_tag.json").write_text(
        json.dumps({**good, "init": str(T.base_path(0)), "spy": {"by_tag": {"B": 9, "A": 1}}})
    )
    (d / "reads" / "it0020.json").write_text(
        json.dumps({"author_tag": "A", "tag_spy": {"by_tag": {"A": 4}}})
    )
    bad = TM.tag_audit(line, segs)
    assert (
        len(bad) == 3
        and "not the line's base" in bad[0]
        and "other tags {'A': 1}" in bad[1]
        and "it0020" in bad[2]
    )
