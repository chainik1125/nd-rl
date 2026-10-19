"""`code.thinking.figures`: bins, source loaders, the three figures' numbers and their files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from code.thinking import figures as F

FORK = Path(F.DEFAULT_FORK)
TB72 = FORK / "data" / "eval_only" / "textbook72"
DAN_EVAL = FORK / "artifacts" / "bs" / "eval"
needs_tb72 = pytest.mark.skipif(not TB72.exists(), reason="fork's textbook72 not available")
needs_dan = pytest.mark.skipif(not DAN_EVAL.exists(), reason="fork's best-state reads absent")

# a tiny textbook72 look-alike: five dev problems (one per bin, two in 6-10) and two train
DEV = {"d1": 3, "d2": 7, "d3": 12, "d4": 20, "d5": 10}
TRAIN = ["t1", "t2"]
#: per theorem: attempts ok out of 4
OK = {"d1": [1, 1, 0, 1], "d2": [0, 0, 0, 0], "d3": [0, 1, 0, 0], "d4": [0, 0, 0, 0],
      "d5": [1, 0, 0, 0], "t1": [0, 0, 1, 1], "t2": [0, 0, 0, 0]}  # fmt: skip


def write_tb72(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "textbook_dev.jsonl", "w") as f:
        for n, L in DEV.items():
            f.write(json.dumps({"name": n, "prompt": f"THM {n}", "reference_lines": L}) + "\n")
    with open(d / "textbook_train.jsonl", "w") as f:
        for n in TRAIN:
            f.write(json.dumps({"name": n, "prompt": f"THM {n}", "role": "rl_train"}) + "\n")
    return d


def eval_row(
    name: str, ok: list[int], file: str, tokens: list[int] | None = None
) -> dict[str, Any]:
    return {
        "file": file,
        "name": name,
        "k": len(ok),
        "n_ok": sum(ok),
        "ok": ok,
        "tokens": tokens if tokens is not None else [10] * len(ok),
    }


def file_of(name: str) -> str:
    return "textbook_dev" if name.startswith("d") else "textbook_train"


def write_eval_dir(d: Path, cells: dict[str, dict[str, list[int]]], tokens: int = 10) -> Path:
    """A code.thinking.eval output directory: rows_<cell>.jsonl + summary.json."""
    d.mkdir(parents=True, exist_ok=True)
    for cell, ok in cells.items():
        with open(d / f"rows_{cell}.jsonl", "w") as f:
            for n, o in ok.items():
                f.write(json.dumps(eval_row(n, o, file_of(n), [tokens] * len(o))) + "\n")
    (d / "summary.json").write_text(json.dumps({"cells": {c: {} for c in cells}}))
    return d


# ----------------------------------------------------------------------------------------- bins


def test_bin_of_matches_dans_inclusive_bins() -> None:
    assert [F.bin_of(x) for x in (1, 5, 6, 10, 11, 15, 16, 33)] == [
        "1–5", "1–5", "6–10", "6–10", "11–15", "11–15", "16+", "16+",
    ]  # fmt: skip
    assert F.bin_of(None) == "train14"
    with pytest.raises(ValueError):
        F.bin_of(0)


@needs_tb72
def test_real_textbook72_bins_are_5_24_16_13_plus_14() -> None:
    refs = F.reference_lengths(TB72)
    assert len(refs) == 72
    sizes: dict[str, int] = {}
    for r in refs.values():
        sizes[F.bin_of(r)] = sizes.get(F.bin_of(r), 0) + 1
    assert sizes == F.TB72_SIZES


def test_pass_at_k_and_ks_match_the_evaluator() -> None:
    from code.thinking import eval as E

    for n, c, k in [(4, 1, 2), (256, 3, 1), (256, 0, 256), (64, 64, 8), (8, 2, 7)]:
        assert F.pass_at_k(n, c, k) == pytest.approx(E.pass_at_k(n, c, k))
    assert F.ks_upto(64) == E.ks_upto(64) and F.ks_upto(48) == E.ks_upto(48)
    assert F.pass_at_k(4, 1, 2) == pytest.approx(0.5)


# -------------------------------------------------------------------------------------- loaders


def test_every_source_kind_gives_the_same_bins(tmp_path: Path) -> None:
    refs = F.reference_lengths(write_tb72(tmp_path / "tb"))
    expect = {"1–5": (1.0, 1), "6–10": (1.0, 2), "11–15": (1.0, 1), "16+": (0.0, 1),
              "train14": (1.0, 2)}  # fmt: skip
    # code.thinking.eval directory, and its rows file directly
    d = write_eval_dir(tmp_path / "ev", {"plain_B0": OK})
    srcs = [F.load_source(d), F.load_source(d / "rows_plain_B0.jsonl")]
    # fork guided_eval: <out>.json + <out>.rows.jsonl.gz; by prefix, by rows file, by json
    g = tmp_path / "ge" / "c12s0_logical"
    g.parent.mkdir()
    (Path(str(g) + ".json")).write_text(json.dumps({"arm": "logical", "k": 4}))
    with gzip.open(str(g) + ".rows.jsonl.gz", "wt") as f:
        for n, o in OK.items():
            r = eval_row(n, o, file_of(n))
            r["min_lines"] = DEV.get(n)
            f.write(json.dumps(r) + "\n")
    srcs += [F.load_source(g), F.load_source(str(g) + ".rows.jsonl.gz")]
    srcs.append(F.load_source(str(g) + ".json"))
    assert srcs[-1].kind == "guided_eval" and srcs[-1].cell == "logical"
    # per-theorem tables: Dan's state_eval rows (solved) and a csv with n_ok/k
    t = tmp_path / "state_eval_rows.jsonl"
    t.write_text(
        "".join(json.dumps({"name": n, "solved": sum(o) > 0}) + "\n" for n, o in OK.items())
    )
    c = tmp_path / "table.csv"
    with open(c, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "n_ok", "k"])
        for n, o in OK.items():
            w.writerow([n, sum(o), len(o)])
    srcs += [F.load_source(t), F.load_source(c)]
    # Dan's state_eval summary: solved/n per reference length
    by_len: dict[str, dict[str, int]] = {}
    for n, o in OK.items():
        key = str(DEV.get(n))
        cur = by_len.setdefault(key, {"solved": 0, "n": 0})
        cur["solved"] += int(sum(o) > 0)
        cur["n"] += 1
    s = tmp_path / "X__tb72.json"
    s.write_text(json.dumps({"k": 4, "by_len": by_len}))
    srcs.append(F.load_source(s))
    for src in srcs:
        assert F.bin_counts(src, refs, check_sizes=None) == expect, src.path


def test_pass_at_k_below_the_read_and_its_limits(tmp_path: Path) -> None:
    refs = F.reference_lengths(write_tb72(tmp_path / "tb"))
    src = F.load_source(write_eval_dir(tmp_path / "ev", {"plain_B0": OK}))
    got = F.bin_counts(src, refs, 1, check_sizes=None)
    # pass@1 = c / n per theorem
    assert got["1–5"] == (pytest.approx(0.75), 1)
    assert got["6–10"] == (pytest.approx(0.25), 2)
    assert got["train14"] == (pytest.approx(0.5), 2)
    with pytest.raises(ValueError, match="needs at least"):
        F.bin_counts(src, refs, 8, check_sizes=None)
    t = tmp_path / "solved.jsonl"
    t.write_text("".join(json.dumps({"name": n, "solved": True}) + "\n" for n in OK))
    with pytest.raises(ValueError, match="cannot give pass@2"):
        F.bin_counts(F.load_source(t), refs, 2, check_sizes=None)


def test_missing_theorems_and_wrong_bin_sizes_raise(tmp_path: Path) -> None:
    refs = F.reference_lengths(write_tb72(tmp_path / "tb"))
    part = {k: v for k, v in OK.items() if k != "d3"}
    src = F.load_source(write_eval_dir(tmp_path / "ev", {"plain_B0": part}))
    with pytest.raises(ValueError, match="1 of 7 theorems missing"):
        F.bin_counts(src, refs, check_sizes=None)
    full = F.load_source(write_eval_dir(tmp_path / "ev2", {"plain_B0": OK}))
    with pytest.raises(ValueError, match="bin sizes"):
        F.bin_counts(full, refs)  # textbook72's sizes, not the toy's


def test_ambiguous_eval_directory_needs_a_cell(tmp_path: Path) -> None:
    d = write_eval_dir(tmp_path / "ev", {"plain_B0": OK, "logical_B0": OK})
    with pytest.raises(ValueError, match="several cells"):
        F.load_source(d)
    assert F.load_source(d, "logical_B0").cell == "logical_B0"
    assert F.eval_cells(d) == ["logical_B0", "plain_B0"]
    assert F.cell_budget("logical_B64") == ("logical", 64)


def test_parse_group() -> None:
    g = F.parse_group("Round 8 = a.json, b.json,")
    assert g.label == "Round 8" and g.paths == ["a.json", "b.json"]
    with pytest.raises(argparse.ArgumentTypeError):
        F.parse_group("no-equals-sign")


# ------------------------------------------------------------------------------ length-bin figure


@needs_tb72
def test_lengthbin_cli_on_the_real_72(tmp_path: Path) -> None:
    refs = F.reference_lengths(TB72)
    names = sorted(refs)

    def seed_dir(i: int, frac: float) -> Path:
        # theorem j solved by seed i iff j / 72 < frac (deterministic patterns, 4 attempts each)
        ok = {n: [1, 0, 0, 0] if j / 72 < frac else [0, 0, 0, 0] for j, n in enumerate(names)}
        d = tmp_path / f"s{i}"
        d.mkdir()
        with open(d / "rows_plain_B0.jsonl", "w") as f:
            for n, o in ok.items():
                f.write(json.dumps(eval_row(n, o, "x")) + "\n")
        return d

    a = [seed_dir(i, f) for i, f in enumerate((0.25, 0.5, 0.75))]
    b = [seed_dir(10 + i, 1.0) for i in range(2)]
    out = tmp_path / "figs" / "fig_lengthbin"
    rc = F.main(
        [
            "lengthbin",
            "--out",
            str(out),
            "--tb72",
            str(TB72),
            "--group",
            "A=" + ",".join(map(str, a)),
            "--group",
            "B=" + ",".join(map(str, b)),
        ]
    )
    assert rc == 0
    for suf in (".pdf", ".svg", ".png", ".json", ".caption.md"):
        assert out.with_suffix(suf).stat().st_size > 0, suf
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["k_effective"] == 4
    ga, gb = data["groups"]
    assert ga["n"] == F.TB72_SIZES and len(ga["seeds"]) == 3
    for bname in F.BIN_ORDER:
        rates = [sd["bins"][bname]["rate"] for sd in ga["seeds"]]
        assert ga["mean"][bname] == pytest.approx(sum(rates) / 3)
        assert gb["mean"][bname] == 1.0
    # overall: seed i solves round(72 * frac) theorems
    tot = [sum(sd["bins"][x]["solved"] for x in F.BIN_ORDER) for sd in ga["seeds"]]
    assert tot == [18, 36, 54]
    cap = out.with_suffix(".caption.md").read_text()
    assert cap.startswith("**Solve rate by reference-length bin.**") and "pass@4" in cap
    assert "B (2 seeds) 100% / 100% / 100% / 100%, train14 100%" in cap


@needs_tb72
@needs_dan
def test_dans_best_state_summaries_match_numbers_md() -> None:
    """Dan's numbers.md 'textbook72 by reference_lines (per seed)' rows for these checkpoints."""
    refs = F.reference_lengths(TB72)
    want = {
        "Fz_best12_s0": [5, 16, 6, 0, 5],
        "Fz_best12_s1": [5, 15, 2, 0, 5],
        "Fz_best12_s2": [5, 13, 3, 0, 6],
        "T1_best12_s0": [5, 22, 10, 5, 10],
        "T1_best12_s1": [5, 23, 12, 4, 7],
        "T1_best12_s2": [5, 23, 11, 3, 10],
    }
    for lab, counts in want.items():
        src = F.load_source(DAN_EVAL / f"{lab}__tb72.json")
        assert src.kind == "state_eval summary" and src.k == 256
        got = F.bin_counts(src, refs)
        assert [got[b][0] for b in F.BIN_ORDER] == counts, lab


def test_label_slots() -> None:
    assert F._label_slots([0.5]) == ["above"]  # noqa: SLF001
    assert F._label_slots([0.2, 0.9]) == ["below", "above"]  # noqa: SLF001
    assert F._label_slots([0.2, 0.9, 0.5]) == ["below", "above", "right"]  # noqa: SLF001


# ---------------------------------------------------------------------------------- TTS figures

#: two seeds x budgets: seed s solves (B / 64 + s) / 4 of attempts on d1, nothing elsewhere
SETS_OK = {"d1": 0, "d2": 0, "t1": 0}


def tts_dirs(tmp_path: Path) -> list[Path]:
    dirs = []
    for s in range(2):
        cells: dict[str, dict[str, list[int]]] = {}
        for arm in ("plain", "logical"):
            for B in (0, 16, 64):
                c = min(4, B // 16 + s + (arm == "logical"))
                cells[f"{arm}_B{B}"] = {
                    "d1": [1] * c + [0] * (4 - c),
                    "d2": [0, 0, 0, 0],
                    "t1": [1, 0, 0, 0],
                }
        dirs.append(write_eval_dir(tmp_path / f"T2_s{s}", cells, tokens=20 + 10 * s))
    return dirs


def test_budget_data_values(tmp_path: Path) -> None:
    dirs = tts_dirs(tmp_path)
    g = F.Group("T2", [str(d) for d in dirs])
    data = F.budget_data([g], arms=["plain", "logical"], sets=["textbook_dev", "all"])
    plain = next(s for s in data["series"] if s["arm"] == "plain")
    # seed 0, B = 16: d1 1/4, d2 0 -> pass@1 on textbook_dev = 0.125
    assert plain["seeds"][0]["pass1"]["textbook_dev"]["16"] == pytest.approx(0.125)
    # mean over seeds at B = 64: s0 c = 4 -> 0.5; s1 c = min(4, 5) = 4 -> 0.5
    assert plain["mean"]["textbook_dev"]["64"] == pytest.approx(0.5)
    # 'all' includes t1 (1/4 each): seed 0, B = 0: (0 + 0 + 0.25) / 3
    assert plain["seeds"][0]["pass1"]["all"]["0"] == pytest.approx(0.25 / 3)
    assert plain["n"] == {"textbook_dev": 2, "all": 3}


def test_budget_x_puts_zero_one_octave_left() -> None:
    xs, brk = F._budget_x([0, 16, 64, 256])  # noqa: SLF001
    assert xs[16] == 4 and xs[64] == 6 and xs[256] == 8
    assert brk is not None and xs[0] < brk < xs[16]
    xs2, brk2 = F._budget_x([16, 64])  # noqa: SLF001
    assert brk2 is None and 0 not in xs2


def test_tokens_data_values(tmp_path: Path) -> None:
    dirs = tts_dirs(tmp_path)
    g = F.Group("T2", [str(d) for d in dirs])
    data = F.tokens_data([g], arms=["plain", "logical"], sets=["textbook_dev"], budgets=[16])
    assert [(s["arm"], s["budget"]) for s in data["series"]] == [("plain", 16), ("logical", 16)]
    p = data["series"][0]
    sd0 = p["seeds"][0]["sets"]["textbook_dev"]
    assert sd0["ks"] == [1, 2, 4] and sd0["tokens_per_attempt"] == 20
    # seed 0 plain B16: d1 c = 1 of 4, d2 0: pass@2 = mean(0.5, 0) = 0.25 at x = 2 * 20
    assert sd0["pts"][1] == [pytest.approx(40.0), pytest.approx(0.25)]
    # the mean over seeds averages x and y at each k (seed 1 spends 30 tokens an attempt)
    k4 = p["mean"]["textbook_dev"][2]
    assert k4[0] == pytest.approx((4 * 20 + 4 * 30) / 2)
    assert k4[1] == pytest.approx(0.5)


def test_tts_clis_write_their_files(tmp_path: Path) -> None:
    dirs = tts_dirs(tmp_path)
    grp = "T2=" + ",".join(map(str, dirs))
    t0 = write_eval_dir(tmp_path / "T0_s0", {"plain_B0": {"d1": [1, 0, 0, 0], "d2": [0, 0, 0, 0],
                                                         "t1": [0, 0, 0, 0]}})  # fmt: skip
    for cmd in ("budget", "tokens"):
        out = tmp_path / "figs" / f"fig_{cmd}"
        args = [cmd, "--out", str(out), "--group", grp, "--group", f"T0={t0}"]
        args += ["--sets", "textbook_dev,all"]
        assert F.main(args) == 0
        for suf in (".pdf", ".svg", ".png", ".json", ".caption.md"):
            assert out.with_suffix(suf).stat().st_size > 0, (cmd, suf)
        data = json.loads(out.with_suffix(".json").read_text())
        assert {s["group"] for s in data["series"]} == {"T2", "T0"}
        assert all(not math.isnan(v) for s in data["series"] for v in _flat(s["mean"]))


def _flat(x: Any) -> list[float]:
    if isinstance(x, dict):
        return [v for y in x.values() for v in _flat(y)]
    if isinstance(x, list):
        return [v for y in x for v in _flat(y)]
    return [float(x)]
