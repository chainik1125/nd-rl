# ruff: noqa: E501
"""handcheck5.py: round 5 keeps round 4's protocol (arrays, bootstrap, prompts, validation) and adds its own cells and
seeds, the drift items, the paired readings, the share of the final model's rate and the bounds for a cut cell."""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import a9
import handcheck as hc
import handcheck3 as h3
import handcheck4 as h4
import handcheck5 as h5
import numpy as np
import pytest

V2 = h3.V2
HAVE_DATA = (h5.ROUND4 / "labels.csv").exists() and (V2 / a9.REF_SAMPLES / "graded.jsonl").exists()
HAVE_INTERP = h5.INTERP_MANIFEST.exists()
CELL_WORDS = ("rank1", "full_n", "ref_again", "O1__", "A16__", "n16", "n64", "batch_a9", "r4:", "round4",
              "round5", "key.csv", "screen.csv", "batches.json", "cells.json", "s2300", "final25")  # fmt: skip


def test_round5_has_its_own_seeds_and_ids():
    s5 = {h5.SEED_ITEMS, h5.SEED_B, h5.SEED_C, h5.SEED_BOOT, h5.SEED_TA, h5.SEED_TB, h5.SEED_TC, h5.SEED_SPOT,
          h5.SEED_DRIFT}  # fmt: skip
    assert s5 == set(range(660051, 660060))
    s4 = {
        h4.SEED_ITEMS,
        h4.SEED_B,
        h4.SEED_C,
        h4.SEED_BOOT,
        h4.SEED_TA,
        h4.SEED_TB,
        h4.SEED_TC,
        h4.SEED_SPOT,
    }
    s3 = {h3.SEED_ITEMS, h3.SEED_B, h3.SEED_C, h3.SEED_BOOT}
    assert not s5 & (s4 | s3) and a9.SEED_SAMPLE not in s5 | s4 | s3
    kk = ("ifbench_0", 0, 3)
    assert h5.item_id("D", kk) != h4.item_id("D", kk) and re.fullmatch(
        r"h[0-9a-f]{7}", h5.item_id("r4:D", kk)
    )
    assert h5.item_id("r4:D", kk) != h5.item_id("ref", kk)
    assert list(h5.NEW.values()) == [a9.RANK1, a9.N16, a9.N4]
    assert (h5.N_DRIFT, h5.N_DRAWS, h5.REPS, h4.BATCH) == (100, 8, 10000, 16)
    assert (h4.SEED_ITEMS, h4.SEED_BOOT, h4.SEED_SPOT) == (
        660041,
        660044,
        660048,
    )  # round 4 untouched


def test_prereg_names_the_seeds():
    note = (Path(h5.__file__).parent / "PREREG.md").read_text()
    am = note[note.index("## Amendment 9 (") :]
    assert "sha1(`660051|cell|draw`)" in am and "(seed 660059)" in am
    assert (
        "660051, 660052, 660053; bootstrap 660054; task reads 660055, 660056, 660057; spot read 660058"
        in am
    )
    assert "100 items drawn uniformly without replacement from round 4's 1,340" in am


def _draws(rng, pids, n=8, p=0.5):
    out = {}
    for pid in pids:
        for j in range(n):
            ok = bool(rng.random() < p)
            words = " ".join(["w"] * int(rng.integers(3, 40))) + ("." if rng.random() < 0.6 else "")
            out[(pid, j // 4, j % 4)] = {
                "text": f"<think>t</think>{words}",
                "strict": ok,
                "per": None,
            }
    return out


def _labelled(rng, draws, tag):
    """Screen about a tenth of the passes; give the rest items with random labels, a few unread."""
    scr, item_of, plab, tlab = set(), {}, {}, {}
    for n, (k, d) in enumerate(sorted(draws.items())):
        if not d["strict"]:
            continue
        if rng.random() < 0.1:
            scr.add(k)
            continue
        item_of[k] = f"{tag}{n}"
        plab[item_of[k]] = str(rng.choice(["genuine", "gamed"], p=[0.7, 0.3]))
        tlab[item_of[k]] = str(rng.choice(["done", "not_done"], p=[0.8, 0.2]))
    return scr, item_of, plab, tlab


def test_arrays_and_bootstrap_equal_round4s_on_complete_cells():
    rng = np.random.default_rng(5)
    pids = [f"p{i:02d}" for i in range(12)]
    cells = {c: _draws(rng, pids, p=p) for c, p in (("x", 0.6), ("y", 0.3), ("z", 0.05))}
    lab = {c: _labelled(rng, d, c) for c, d in cells.items()}
    a5 = {c: h5.arrays(d, *lab[c], pids) for c, d in cells.items()}
    a4 = {c: h4.cell_arrays(d, *lab[c], pids) for c, d in cells.items()}
    for c in cells:
        for m in (*h4.METRICS, "words"):
            assert np.array_equal(a5[c][m], a4[c][m], equal_nan=True)
        p5, p4 = h5.point(a5[c]), h4.point(a4[c])
        assert p5.keys() == p4.keys() and all(np.isclose(p5[k], p4[k], equal_nan=True) for k in p4)
    b5, b4 = h5.boot(a5, reps=230, seed=7, block=100), h4.boot(a4, reps=230, seed=7, block=100)
    for c in cells:
        assert b5[c].keys() == b4[c].keys()
        for q in b4[c]:
            assert np.array_equal(b5[c][q], b4[c][q], equal_nan=True), (c, q)


def test_unfinished_draws_as_sampled_as_fail_and_as_accepted_pass():
    pids = ["p0", "p1"]
    draws = {
        ("p0", 0, i): {"text": "<think>t</think>A real answer.", "strict": True, "per": None}
        for i in range(4)
    }
    draws |= {
        ("p1", c, i): {"text": "<think>t</think>no", "strict": False, "per": None}
        for c in (0, 1)
        for i in range(4)
    }
    item_of = {k: f"h{n}" for n, k in enumerate(sorted(draws)) if draws[k]["strict"]}
    plab = dict.fromkeys(item_of.values(), "genuine") | {"h0": "gamed"}
    tlab = dict.fromkeys(item_of.values(), "done")
    a = h5.arrays(draws, set(), item_of, plab, tlab, pids)
    assert np.isnan(a["strict"][0, 4:]).all() and not np.isnan(a["strict"][1]).any()
    s = h5.sums(a)
    assert (int(s["n"]), s["strict"], s["genuine"], s["done"]) == (12, 4, 3, 4)
    p = h5.point(a)
    assert p["strict_rate"] == pytest.approx(4 / 12) and p["genuine_rate"] == pytest.approx(3 / 12)
    lo = h5.point(h5.arrays(draws, set(), item_of, plab, tlab, pids, fill="fail"))
    hi = h5.point(h5.arrays(draws, set(), item_of, plab, tlab, pids, fill="pass"))
    assert (lo["strict_rate"], lo["genuine_rate"], lo["task_done_rate"]) == (4 / 16, 3 / 16, 4 / 16)
    assert (hi["strict_rate"], hi["genuine_rate"], hi["task_done_rate"]) == (8 / 16, 7 / 16, 8 / 16)
    b = h5.boot({"c": a}, reps=50, seed=1)
    assert np.nanmin(b["c"]["strict_rate"]) >= 0 and np.nanmax(b["c"]["strict_rate"]) <= 1
    # an item left without two agreeing reads: not accepted by default, accepted in the "both ways" version
    plab["h1"], tlab["h1"] = "unread", "unread"
    assert h5.sums(h5.arrays(draws, set(), item_of, plab, tlab, pids))["genuine"] == 2
    both = h5.sums(h5.arrays(draws, set(), item_of, plab, tlab, pids, unread_accepted=True))
    assert (both["genuine"], both["done"]) == (3, 4)
    with pytest.raises(SystemExit):
        h5.arrays({("p0", 0, i): draws[("p0", 0, 0)] for i in range(9)}, set(), {}, {}, {}, pids)


def test_sign_test_and_drift(tmp_path):
    assert h5.sign_test(0, 0) == 1.0 and h5.sign_test(5, 5) == 1.0
    assert h5.sign_test(10, 0) == pytest.approx(2 / 1024) and h5.sign_test(1, 9) == pytest.approx(
        22 / 1024
    )
    assert h5.sign_test(3, 9) == pytest.approx(2 * sum((1, 12, 66, 220)) / 4096)
    r4 = tmp_path / "round4"
    r4.mkdir()
    with open(r4 / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "primary", "task"])
        w.writeheader()
        for n in range(20):
            w.writerow(
                {"item": f"o{n}", "primary": "genuine" if n < 14 else "gamed", "task": "done"}
            )
    key = {f"n{n}": {"r4_item": f"o{n}"} for n in range(20)} | {"own": {"r4_item": ""}}
    final = {"primary": {f"n{n}": "gamed" if n < 12 else "genuine" for n in range(20)} | {"own": "genuine"},
             "task": {f"n{n}": "done" for n in range(20)} | {"own": "done"}}  # fmt: skip
    d = h5.drift(key, final, r4)
    p = d["primary"]
    assert (p["n"], p["accepted_round4"], p["accepted_round5"]) == (20, 14, 8)
    assert (p["moved_to_accepted"], p["moved_to_not_accepted"], p["net_shift_points"]) == (
        6,
        12,
        -30.0,
    )
    assert p["sign_test_p"] == pytest.approx(h5.sign_test(6, 12)) and not p["shifted"]
    half = (
        196 * (18 / 20 - 0.3**2) ** 0.5 / 20**0.5
    )  # the interval says what "not shifted" leaves open
    assert (p["net_shift_lo"], p["net_shift_hi"]) == pytest.approx((-30 - half, -30 + half))
    only = h5.drift(key, final, r4, keep=lambda r: r["r4_item"] in ("o0", "o13", "o14"))
    assert (only["primary"]["n"], only["primary"]["moved_to_not_accepted"]) == (3, 1)
    assert d["task"]["raw_agreement"] == 1.0 and d["task"]["sign_test_p"] == 1.0
    assert d["reading"] == "no shift detected"
    final["primary"] = {k: "gamed" for k in final["primary"]}
    d = h5.drift(key, final, r4)
    assert d["primary"]["moved_to_not_accepted"] == 14 and d["primary"]["shifted"]
    assert d["reading"] == "readers shifted between rounds"


def test_contrast_list_names_the_reference():
    base = ["rank1_n64", "full_n16", "full_n4", "ref", "final_r4", "onepass_r4"]
    names = [n for _, _, n in h5.contrast_list(base)]
    assert names[:3] == ["rank1_n64-ref", "full_n16-ref", "full_n4-ref"]
    assert "ref-final_r4" in names and "full_n4-onepass_r4" in names and len(names) == 3 + 2 * 4
    r5 = [n for _, _, n in h5.contrast_list([*base, "final_r5", "onepass_r5"])]
    assert r5[:3] == names[:3] and len(r5) == 3 + 4 * 4
    assert r5[3] == "rank1_n64-final_r5" and "final_r5-final_r4" not in r5
    again = [n for _, _, n in h5.contrast_list([*base, "ref_again"])]
    assert again[:3] == ["rank1_n64-ref_again", "full_n16-ref_again", "full_n4-ref_again"]
    assert "ref_again-ref" in again and "rank1_n64-ref" in again and "ref_again-final_r4" in again
    two = [
        n for _, _, n in h5.contrast_list(["rank1_n64", "full_n4", "ref", "final_r4", "onepass_r4"])
    ]
    assert two[:2] == ["rank1_n64-ref", "full_n4-ref"] and not any("n16" in n for n in two)


def _reach():
    return sorted(ln.strip() for ln in hc.REACH.read_text().splitlines() if ln.strip())


def test_interp_cells_are_checked_against_their_md5_list(tmp_path):
    man = {}
    for a in h5.INTERP.values():
        d = tmp_path / a
        d.mkdir()
        (d / "samples.jsonl.gz").write_bytes(f"s {a}".encode())
        (d / "graded.jsonl").write_text(f"g {a}\n")
        man[a] = {"complete": True, "samples": str(d / "samples.jsonl.gz"), "graded": str(d / "graded.jsonl"),
                  "md5": {"samples": a9.file_digest(d / "samples.jsonl.gz", "md5"),
                          "graded": a9.file_digest(d / "graded.jsonl", "md5")}}  # fmt: skip
    mf = tmp_path / "cells_manifest.json"
    mf.write_text(json.dumps(man))
    cells = h5.interp_cells(mf)
    assert list(cells) == ["interp_a025", "interp_a050", "interp_a075", "interp_a100"]
    assert all(c["kind"] == "interp" for c in cells.values())
    assert h5.n_draws("interp_a050") == 4 and h5.n_draws("ref") == 8
    (tmp_path / "a075" / "graded.jsonl").write_text("changed\n")
    with pytest.raises(SystemExit, match="interp_a075"):
        h5.interp_cells(mf)
    man["a075"]["md5"]["graded"] = a9.file_digest(tmp_path / "a075" / "graded.jsonl", "md5")
    man["a025"]["complete"] = False
    mf.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="interp_a025"):
        h5.interp_cells(mf)


def test_arrays_with_four_draws_per_prompt():
    pids = ["p0", "p1"]
    draws = {(p, c, 0): {"text": "<think>t</think>An answer.", "strict": c < 2, "per": None}
             for p in pids for c in range(4)}  # fmt: skip
    item_of = {k: f"h{n}" for n, k in enumerate(sorted(draws)) if draws[k]["strict"]}
    lab = dict.fromkeys(item_of.values(), "genuine")
    done = dict.fromkeys(item_of.values(), "done")
    a = h5.arrays(draws, set(), item_of, lab, done, pids, n_draws=4)
    assert a["strict"].shape == (2, 4) and not np.isnan(a["strict"]).any()
    assert h5.point(a)["strict_rate"] == 0.5 == h5.point(a)["genuine_rate"]
    eight = h5.arrays({}, set(), {}, {}, {}, pids, fill="fail")
    b = h5.boot({"four": a, "eight": eight}, reps=20, seed=3)
    assert b["four"]["strict_rate"].shape == (20,) and (b["eight"]["strict_rate"] == 0).all()
    with pytest.raises(SystemExit):
        h5.arrays(draws, set(), item_of, lab, done, pids, n_draws=3)


def test_a_sampled_cell_without_grading_is_refused_not_dropped(tmp_path):
    batch = tmp_path / "batch_a9"
    (batch / a9.RANK1).mkdir(parents=True)
    with gzip.open(batch / a9.RANK1 / "samples.jsonl.gz", "wt") as f:
        f.write(json.dumps({"pid": "p", "chunk": 0, "i": 0, "text": "t"}) + "\n")
    with pytest.raises(SystemExit, match="no grading"):
        h5.find_cells(batch, tmp_path)
    (batch / a9.RANK1 / "graded.jsonl").write_text("")
    with pytest.raises(SystemExit, match="stored reference is missing"):
        h5.find_cells(batch, tmp_path)
    (tmp_path / "graded.jsonl").write_text("")
    assert list(h5.find_cells(batch, tmp_path)) == ["rank1_n64", "ref"]


def test_load_dir_reads_a_cut_file_and_keeps_complete_chunks_only(tmp_path):
    if not hc.REACH.exists():
        pytest.skip("reach prompt list not on this machine")
    pids = _reach()[:3]
    rows, graded = [], []
    for pid in pids:
        for ch in (0, 1):
            for i in range(4):
                rows.append(
                    {
                        "pid": pid,
                        "chunk": ch,
                        "i": i,
                        "text": f"<think>t</think>answer {pid} {ch} {i}",
                    }
                )
                graded.append(
                    {"pid": pid, "chunk": ch, "i": i, "strict": (i % 2 == 0), "per": "[True]"}
                )
    rows.append({"pid": "not_a_reach_prompt", "chunk": 0, "i": 0, "text": "x"})
    g = tmp_path / "graded.jsonl"
    g.write_text("".join(json.dumps(r) + "\n" for r in graded))
    full = tmp_path / "full.jsonl.gz"
    with gzip.open(full, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    d = h5.load_dir(full, g)
    assert len(d) == 24 and sum(v["strict"] for v in d.values()) == 12
    part = (
        tmp_path / "part.jsonl.gz"
    )  # the last chunk has 2 of its 4 rows, and the file has no gzip trailer
    with gzip.open(part, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows[:22])
    part.write_bytes(part.read_bytes()[:-9])
    d = h5.load_dir(part, g)
    assert len(d) % 4 == 0 and 0 < len(d) <= 20 and all(k[:2] != (pids[2], 1) for k in d)
    assert h5.read_rows(tmp_path / "absent.gz") == []


def test_salvage_keeps_whole_requests_and_leaves_the_fetched_batch_alone(tmp_path):
    batch = tmp_path / "batch_a9"
    rows = [
        {"pid": f"p{p}", "chunk": c, "i": i, "text": "t"}
        for p in range(3)
        for c in (0, 1)
        for i in range(4)
    ]
    done, cut = batch / "cell_done", batch / "cell_cut"
    for d in (done, cut, batch / "logs"):
        d.mkdir(parents=True)
    with gzip.open(done / "samples.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (done / "graded.jsonl").write_text(
        "".join(json.dumps({**r, "strict": True}) + "\n" for r in rows)
    )
    with gzip.open(cut / "samples.jsonl.gz", "wt") as f:
        f.writelines(
            json.dumps(r) + "\n" for r in rows[:18]
        )  # 4 whole requests and half of the fifth
    before = (cut / "samples.jsonl.gz").read_bytes()
    dest = tmp_path / "salvaged"
    info = h5.cmd_salvage(batch, dest)
    assert info["cell_done"] == {"rows_read": 24, "rows_kept": 24, "rows_of_cut_requests": 0,
                                 "graded_on_the_pod": True, "to_grade": False}  # fmt: skip
    assert info["cell_cut"] == {"rows_read": 18, "rows_kept": 16, "rows_of_cut_requests": 2,
                                "graded_on_the_pod": False, "to_grade": True}  # fmt: skip
    assert (dest / "cell_done").is_symlink() and (dest / "cell_done" / "graded.jsonl").exists()
    kept = h5.read_rows(dest / "cell_cut" / "samples.jsonl.gz")
    assert len(kept) == 16 and not (dest / "cell_cut" / "graded.jsonl").exists()
    assert (cut / "samples.jsonl.gz").read_bytes() == before and "logs" not in info
    assert json.loads((dest / "salvage.json").read_text())["cells"] == info
    with pytest.raises(SystemExit):
        h5.cmd_salvage(batch, dest)


@pytest.mark.skipif(
    not HAVE_DATA, reason="round-4 files and the stored reference are not on this machine"
)
def test_round4_cells_are_reproduced_from_the_stored_labels():
    pids = _reach()
    r4 = h5.round4_arrays(pids)
    want = json.loads((h5.ROUND4 / "round4_summary.json").read_text())["cells"]
    for name, cell in h5.R4_CELLS.items():
        p = h5.point(r4[name])
        for q in (
            "strict_rate",
            "genuine_rate",
            "task_done_rate",
            "unpunctuated_share",
            "median_words",
        ):
            assert p[q] == pytest.approx(want[cell][q]["value"]), (name, q)
    assert h5.point(r4["final_r4"])["genuine_rate"] == pytest.approx(
        182 / 384
    )  # the 47.4 % of Amendment 9(d)
    picked = h5.drift_pick()
    assert (
        len(picked) == 100 and len({r["item"] for r in picked}) == 100 and picked == h5.drift_pick()
    )
    assert {r["cell"] for r in picked} <= set(h4.CELLS)


@pytest.mark.skipif(
    not HAVE_DATA, reason="round-4 files and the stored reference are not on this machine"
)
@pytest.mark.parametrize("reread", [True, False])
def test_end_to_end_on_stand_in_cells(tmp_path, capsys, reread):
    """The whole round on stored cells standing in for the new ones, with reads taken from round 4's labels:
    rank-1 := round 4's cell D, 16 examples := C, 4 examples := A' with its file cut. Nothing is written under V2.
    With the re-read, every round-4 item of the final model (258) and of the one-pass adapter (149) is an item too."""
    batch = tmp_path / "batch_a9"
    batch.mkdir()
    (batch / a9.RANK1).symlink_to(h3.CORNERS["D"])
    (batch / a9.N16).symlink_to(h3.CORNERS["C"])
    cut = batch / a9.N4
    cut.mkdir()
    (cut / "graded.jsonl").write_text((h3.CORNERS["Aprime"] / "graded.jsonl").read_text())
    rows = h5.read_rows(h3.CORNERS["Aprime"] / "samples.jsonl.gz")
    assert len(rows) == 384
    with gzip.open(cut / "samples.jsonl.gz", "wt", compresslevel=1) as f:
        f.writelines(json.dumps(r) + "\n" for r in rows[:330])  # 82 whole requests and half of one
    out, items_dir = tmp_path / "round5", tmp_path / "hc5_read"
    interp = (
        reread and HAVE_INTERP
    )  # the four weight-interpolation cells, in one of the two variants
    args = (out, items_dir, batch, V2 / a9.REF_SAMPLES)
    info = h5.cmd_prep(*args, reread=reread, interp=interp)
    assert info["drift"] == 100 and info["not_sampled"] == []
    with pytest.raises(SystemExit):
        h5.cmd_prep(*args, reread=reread, interp=interp)  # never rebuilt
    key = {r["item"]: r for r in csv.DictReader((out / "key.csv").read_text().splitlines())}
    summ = {r["cell"]: r for r in json.loads((out / "screen_summary.json").read_text())}
    assert list(summ) == ["rank1_n64", "full_n16", "full_n4", "ref", *(h5.INTERP if interp else [])]
    if interp:  # the pass counts of the coordinator's note 9(k)
        assert [summ[n]["passes"] for n in h5.INTERP] == [2, 26, 102, 133]
        assert all(summ[n]["draws"] == 192 for n in h5.INTERP)
    assert (summ["ref"]["draws"], summ["ref"]["passes"]) == (384, 166)
    assert (summ["rank1_n64"]["passes"], summ["rank1_n64"]["screened_gamed"]) == (
        156,
        7,
    )  # round 4's D
    assert summ["full_n4"]["draws"] == 328
    from_r4 = [r for r in key.values() if r["r4_item"]]
    assert sum(r["uniform100"] == "1" for r in from_r4) == 100 == len(h5.drift_pick())
    assert {r["r4_item"] for r in from_r4 if r["uniform100"]} == {
        r["item"] for r in h5.drift_pick()
    }
    per_cell = Counter(r["cell"] for r in from_r4)
    if reread:
        assert (per_cell["r4:s2300"], per_cell["r4:D"]) == (258, 149)
        assert info["reread"] == len(from_r4) - 100 and 340 < info["reread"] < 407
    else:
        assert info["reread"] == 0 and len(from_r4) == 100
    assert len({r["r4_item"] for r in from_r4}) == len(from_r4)  # no round-4 item twice
    assert len(key) == sum(s["passes"] - s["screened_gamed"] for s in summ.values()) + len(from_r4)
    # blind: batch prompts and item files name no cell, and every batch is a mix
    man = json.loads((out / "batches.json").read_text())
    assert {t[:-2] for t in man} == {"a", "b", "ta", "tb"} and all(
        len(v) in (15, 16, 17) for v in man.values()
    )
    for tag in ("a00", "b03", "ta01", "tb02"):
        txt = (items_dir / "batches" / f"{tag}.prompt.md").read_text()
        assert not any(w in txt for w in CELL_WORDS), tag
        assert len({key[i]["cell"] for i in man[tag]}) >= 3
    assert (
        (items_dir / "batches" / "ta00.prompt.md")
        .read_text()
        .startswith(h4.TASK_INSTRUCTION + "\n")
    )
    assert (items_dir / "batches" / "a00.prompt.md").read_text() == h3.batch_prompt(
        items_dir, "a00", man["a00"]
    )
    assert sorted(p.name for p in items_dir.iterdir()) == [
        "batches",
        "items",
        "reads",
        "task_items",
    ]
    first = (items_dir / "items" / f"{man['a00'][0]}.md").read_text()
    assert first.startswith(f"# Item {man['a00'][0]}\n") and "## Verifiable constraints" in first
    # reads: round 4's final labels for the stand-ins and the drift items; the unread reference all accepted
    r4 = list(csv.DictReader((h5.ROUND4 / "labels.csv").read_text().splitlines()))
    by_draw = {(r["cell"], r["pid"], r["chunk"], r["i"]): r for r in r4}
    by_item = {r["item"]: r for r in r4}
    stand = {"rank1_n64": "D", "full_n16": "C", "full_n4": "Aprime"}

    def label(item, col):
        r = key[item]
        if r["r4_item"]:
            return by_item[r["r4_item"]][col]
        if r["cell"] == "ref" or r["cell"] in h5.INTERP:
            return "genuine" if col == "primary" else "done"
        return by_draw[(stand[r["cell"]], r["pid"], r["chunk"], r["i"])][col]

    for tag, items in man.items():
        if tag.startswith("t"):
            obj = {i: label(i, "task") for i in items}
        else:
            obj = [{"item": i, "label": label(i, "primary")} for i in items]
        (items_dir / "reads" / f"{tag}.json").write_text(json.dumps(obj))
    assert h4.cmd_status(out, items_dir, [], False) == {}
    assert (
        h5.cmd_third(out, items_dir, "primary") == [] and h5.cmd_third(out, items_dir, "task") == []
    )
    res = h5.cmd_table(out, items_dir, reps=300)
    want = json.loads((h5.ROUND4 / "round4_summary.json").read_text())["cells"]
    for name, cell in (
        ("rank1_n64", "D"),
        ("full_n16", "C"),
        ("final_r4", "s2300"),
        ("onepass_r4", "D"),
    ):
        for q in h5.RATES:
            assert res["cells"][name][q]["value"] == pytest.approx(want[cell][q]["value"]), (
                name,
                q,
            )
            assert (
                res["cells"][name][q]["lo"]
                <= res["cells"][name][q]["value"]
                <= res["cells"][name][q]["hi"]
            )
    ref = res["cells"]["ref"]
    assert ref["strict_rate"]["value"] == pytest.approx(166 / 384) == ref["genuine_rate"]["value"]
    assert res["contrasts"]["rank1_n64-onepass_r4"]["genuine_rate"]["diff"] == pytest.approx(0)
    c = res["contrasts"]["rank1_n64-ref"]["genuine_rate"]
    assert (
        c["diff"] == pytest.approx(want["D"]["genuine_rate"]["value"] - 166 / 384)
        and c["lo"] < c["diff"] < c["hi"]
    )
    yards = (
        ("final_r4", "onepass_r4", "final_r5", "onepass_r5")
        if reread
        else ("final_r4", "onepass_r4")
    )
    expect = {f"{n}-{r}" for n in ("rank1_n64", "full_n16", "full_n4", "ref") for r in yards}
    expect |= {f"{n}-ref" for n in ("rank1_n64", "full_n16", "full_n4")}
    expect |= {"interp_a100-final_r5", "interp_a100-final_r4"} if interp else set()
    assert set(res["contrasts"]) == expect
    sh = res["share_of_final"]["rank1_n64"]["genuine_rate"]
    assert (
        sh["value"] == pytest.approx(want["D"]["genuine_rate"]["value"] / (182 / 384))
        and sh["lo"] < sh["value"] < sh["hi"]
    )
    assert set(res["share_of_final"]) == {"rank1_n64", "full_n16", "full_n4", "ref"}
    if reread:  # the same draws with round 5's labels, which this test copied from round 4
        assert res["share_yardstick"] == "final_r5" and set(res["reread_cells"]) == {
            "final_r5",
            "onepass_r5",
        }
        for name, cell in (("final_r5", "s2300"), ("onepass_r5", "D")):
            for q in h5.RATES:
                assert res["cells"][name][q]["value"] == pytest.approx(want[cell][q]["value"]), (
                    name,
                    q,
                )
            assert res["cells"][name]["read"] == {"s2300": 258, "D": 149}[cell]
            assert (
                f"{name}-{name[:-1]}4" not in res["contrasts"]
            )  # same draws: compared item by item instead
            dd = res["drift_reread"][name]
            assert dd["primary"]["n"] == dd["task"]["n"] == res["cells"][name]["read"]
            assert (
                dd["primary"]["raw_agreement"] == 1.0 and dd["primary"]["net_shift_points"] == 0.0
            )
        assert res["share_of_final_round4_labels"]["rank1_n64"]["genuine_rate"][
            "value"
        ] == pytest.approx(sh["value"])
        assert res["items_from_round4"] == len(from_r4)
    else:
        assert (
            res["share_yardstick"] == "final_r4"
            and res["reread_cells"] == {}
            and res["drift_reread"] == {}
        )
        assert "share_of_final_round4_labels" not in res and "final_r5" not in res["cells"]
    if interp:  # 4 draws per prompt; a share of the same-engine final model; no reading
        top = res["cells"]["interp_a100"]
        for n in h5.INTERP:
            c = res["cells"][n]
            assert c["draws"] == 192 and c["strict_rate"]["value"] == pytest.approx(
                summ[n]["passes"] / 192
            )
            acc = (summ[n]["passes"] - summ[n]["screened_gamed"]) / 192
            assert c["genuine_rate"]["value"] == pytest.approx(acc) == c["task_done_rate"]["value"]
            assert c["genuine_rate"]["lo"] <= acc <= c["genuine_rate"]["hi"]
            sh = res["share_of_interp_a100"][n]["genuine_rate"]
            assert (
                sh["value"] == pytest.approx(acc / top["genuine_rate"]["value"])
                and n not in res["readings"]
            )
            assert n not in res["share_of_final"]
        assert res["share_of_interp_a100"]["interp_a100"]["genuine_rate"] == {
            "value": 1.0,
            "lo": 1.0,
            "hi": 1.0,
        }
        d = res["contrasts"]["interp_a100-final_r5"]["genuine_rate"]
        assert (
            d["diff"] == pytest.approx(top["genuine_rate"]["value"] - 182 / 384)
            and d["lo"] < d["diff"] < d["hi"]
        )
    else:
        assert "share_of_interp_a100" not in res and not set(h5.INTERP) & set(res["cells"])
    for n in ("rank1_n64", "full_n16", "full_n4"):
        r = res["readings"][n]
        assert (
            r["reference"] == "ref"
            and r["genuine_rate"] in a9.READINGS
            and r["task_done_rate"] in a9.READINGS
        )
        d = res["contrasts"][f"{n}-ref"]["genuine_rate"]
        assert r["genuine_rate"] == a9.reading(100 * d["diff"], 100 * d["lo"], 100 * d["hi"])
        assert r["interval_below_zero"]["genuine_rate"] == (d["hi"] < 0)
    dr = res["drift"]
    assert dr["primary"]["n"] == dr["task"]["n"] == 100 and dr["primary"]["raw_agreement"] == 1.0
    assert dr["primary"]["sign_test_p"] == 1.0 and dr["reading"] == "no shift detected"
    assert (
        dr["primary"]["net_shift_points"],
        dr["primary"]["net_shift_lo"],
        dr["primary"]["net_shift_hi"],
    ) == (0, 0, 0)
    assert (
        res["agreement"]["primary"]["raw_agreement"] == 1.0
        and res["agreement"]["primary"]["unread"] == 0
    )
    # the cut cell: 56 draws unfinished, three versions, and the fail / accepted-pass bounds bracket the as-sampled rate
    assert res["unfinished_draws"] == {"full_n4": 56} and res["cells"]["full_n4"]["draws"] == 328
    lo, hi = (
        res["bounds"][k]["cells"]["full_n4"]
        for k in ("unfinished_as_fail", "unfinished_as_accepted_pass")
    )
    assert lo["draws"] == hi["draws"] == 384
    for q in h5.RATES:
        assert lo[q]["value"] < res["cells"]["full_n4"][q]["value"] < hi[q]["value"]
        assert hi[q]["value"] - lo[q]["value"] == pytest.approx(56 / 384)
        assert (
            res["bounds"]["unfinished_as_fail"]["cells"]["rank1_n64"][q]["value"]
            == res["cells"]["rank1_n64"][q]["value"]
        )
    assert set(res["robust_to_the_cut"]) == {"rank1_n64", "full_n16", "full_n4"}
    assert res["robust_to_the_cut"]["rank1_n64"] == {"genuine_rate": True, "task_done_rate": True}
    assert "unread_as_accepted" not in res
    assert (out / "round5_summary.json").exists()
    lab = list(csv.DictReader((out / "labels.csv").read_text().splitlines()))
    assert len(lab) == len(key) and {r["primary"] for r in lab} <= {"genuine", "gamed"}
    assert h5.cmd_spot(out, items_dir) == 40
    md = h5.markdown(json.loads((out / "round5_summary.json").read_text()))
    assert (
        "| rank-1 adapter, 64 examples | 384 | 40.6 % [" in md
        and "| cell minus the reference |" in md
    )
    assert (
        md.count("| primary |") == (3 if reread else 1)
        and "Drift reading on the 100 items: no shift detected." in md
    )
    assert ("share of the rate of: the final model's own weights" in md) == bool(interp)
    assert "−" in md and "nan" not in md.split("| readers |")[0]
    h5.show(res)
    assert "share of the final model's rate" in capsys.readouterr().out
