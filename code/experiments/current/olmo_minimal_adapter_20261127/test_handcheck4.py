# ruff: noqa: E501
"""handcheck4.py: the census cells, both batch prompts, read validation, labels, per-draw rates, bootstrap, readings."""

import csv
import json
import re
from pathlib import Path

import handcheck3 as h3
import handcheck4 as h4
import numpy as np
import pytest

CELL_WORDS = ("Aprime", "corner", "OLD", "final25", "s523373504", "batch_a6s2", "s2300", "expo", "k16",
              "Xheld", "M2cond", "key.csv", "screen.csv", "batches.json")  # fmt: skip


def test_round3_constants_untouched():
    assert (h3.K, h3.REF_K, h3.BATCH, h3.SEED_ITEMS, h3.SEED_BOOT) == (24, 8, 16, 660021, 660024)
    assert list(h4.CELLS)[:5] == list(h3.CORNERS)
    assert h4.CELLS["s2300"][2] == 8 and h4.CELLS["expo"][3] == h4.CELLS["k16"][3] == 192


def test_first_n_takes_the_first_stored_draws_per_prompt():
    draws = {(p, c, i): {"n": (p, c, i)} for p in ("p1", "p2") for c in range(3) for i in range(2)}
    got = h4.first_n(draws, 3)
    assert sorted(got) == [(p, c, i) for p in ("p1", "p2") for c, i in ((0, 0), (0, 1), (1, 0))]
    assert h4.first_n(draws, None) == draws


def test_task_instruction_is_the_preregistered_text():
    note = (Path(h4.__file__).parent / "PREREG.md").read_text()
    quote = note[note.index("     > You are shown a request") :]
    quote = quote[: quote.index("not_done.") + len("not_done.")]
    assert " ".join(re.sub(r"^\s*> ", "", ln) for ln in quote.splitlines()) == h4.TASK_INSTRUCTION


def test_task_batch_prompt_is_blind_and_points_at_task_items(tmp_path):
    txt = h4.task_batch_prompt(tmp_path, "ta00", ["h1234567", "habcdef0"])
    assert txt.startswith(h4.TASK_INSTRUCTION + "\n")
    for word in CELL_WORDS:
        assert word not in txt
    assert f"{tmp_path}/task_items/h1234567.md" in txt and "reads/ta00.json" in txt
    assert "/items/" not in txt  # the primary item files carry the checker's verdicts


def test_primary_batches_are_round3s_prompt_and_split(tmp_path):
    (tmp_path / "batches").mkdir()
    items = [f"h{i:07x}" for i in range(40)]
    man = h4.write_batches(tmp_path, "primary", "a", items, 5)
    assert sorted(man) == ["a00", "a01", "a02"] and sorted(sum(man.values(), [])) == items
    assert sorted(len(v) for v in man.values()) == [13, 13, 14]
    for tag, chunk in man.items():
        txt = (tmp_path / "batches" / f"{tag}.prompt.md").read_text()
        assert txt == h3.batch_prompt(tmp_path, tag, chunk)
        for word in CELL_WORDS:
            assert word not in txt
    task = h4.write_batches(tmp_path, "task", "b", items, 6)
    assert sorted(task) == ["tb00", "tb01", "tb02"] and h4.kind_of("tb01") == "task"
    assert list(task.values()) != list(man.values())  # another seed, other neighbours


def test_validate_accepts_only_complete_files_with_allowed_labels(tmp_path):
    f = tmp_path / "r.json"
    items = ["h1", "h2"]
    assert h4.validate(f, "primary", items) == "missing"
    f.write_text("not json")
    assert h4.validate(f, "primary", items).startswith("not JSON")
    f.write_text(json.dumps([{"item": "h1", "label": "genuine"}, {"item": "h2", "label": "gamed"}]))
    assert h4.validate(f, "primary", items) is None
    assert h4.validate(f, "task", items) == "not a JSON object"
    f.write_text(json.dumps([{"item": "h1", "label": "genuine"}]))
    assert "1 missing" in h4.validate(f, "primary", items)
    f.write_text(json.dumps([{"item": "h1", "label": "genuine"}, {"item": "h1", "label": "gamed"}]))
    assert h4.validate(f, "primary", items) == "an item appears twice"
    f.write_text(
        json.dumps([{"item": "h1", "label": "genuine"}, {"item": "h2", "label": "unsure"}])
    )
    assert "unsure" in h4.validate(f, "primary", items)
    f.write_text(json.dumps({"h1": "done", "h2": "not_done"}))
    assert h4.validate(f, "task", items) is None
    f.write_text(json.dumps({"h1": "done", "h2": "not_done", "h3": "done"}))
    assert "1 extra" in h4.validate(f, "task", items)
    f.write_text(json.dumps({"h1": "done", "h2": "not done"}))
    assert "not done" in h4.validate(f, "task", items)


def _round(tmp_path):
    out, items_dir = tmp_path / "out", tmp_path / "read"
    for d in (out, items_dir / "batches", items_dir / "reads"):
        d.mkdir(parents=True)
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "cell", "pid", "chunk", "i", "dan_label"])
        w.writeheader()
        for n in range(4):
            w.writerow(
                {"item": f"h{n}", "cell": "X", "pid": "p", "chunk": 0, "i": n, "dan_label": ""}
            )
    man = {
        "a00": ["h0", "h1", "h2", "h3"],
        "b00": ["h0", "h1", "h2", "h3"],
        "ta00": ["h0", "h1", "h2", "h3"],
    }
    (out / "batches.json").write_text(json.dumps(man))
    return out, items_dir


def test_status_quarantines_invalid_files_and_reads_skip_them(tmp_path, capsys):
    out, items_dir = _round(tmp_path)
    rows = [{"item": f"h{n}", "label": "genuine"} for n in range(4)]
    (items_dir / "reads" / "a00.json").write_text(json.dumps(rows))
    (items_dir / "reads" / "b00.json").write_text(json.dumps(rows[:3]))
    assert h4.reads(out, items_dir, "primary") == {f"h{n}": {"a": "genuine"} for n in range(4)}
    fails = h4.cmd_status(out, items_dir, ["a00", "b00"], quarantine=True)
    assert list(fails) == ["b00"] and "1/2 valid" in capsys.readouterr().out
    assert not (items_dir / "reads" / "b00.json").exists()
    assert (out / "failed_reads" / "b00.0.json").exists()  # kept, but outside what readers see


def test_third_read_batches_hold_exactly_the_splits(tmp_path):
    out, items_dir = _round(tmp_path)
    a = [{"item": f"h{n}", "label": "genuine"} for n in range(4)]
    b = [{"item": f"h{n}", "label": "gamed" if n in (1, 3) else "genuine"} for n in range(4)]
    (items_dir / "reads" / "a00.json").write_text(json.dumps(a))
    (items_dir / "reads" / "b00.json").write_text(json.dumps(b))
    h4.cmd_third(out, items_dir, "primary")
    man = h4.manifest(out)
    assert sorted(man["c00"]) == ["h1", "h3"] and "tc00" not in man
    with pytest.raises(SystemExit):
        h4.cmd_third(out, items_dir, "primary")  # never redrawn


def test_majority_and_kappa_for_both_label_sets():
    g, t = h4.KINDS["primary"]["labels"], h4.KINDS["task"]["labels"]
    assert h4.majority({"a": "gamed", "b": "gamed"}, g) == "gamed"
    assert h4.majority({"a": "gamed", "b": "genuine"}, g) == "unread"
    assert h4.majority({"a": "gamed", "b": "genuine", "c": "genuine"}, g) == "genuine"
    assert h4.majority({"a": "not_done", "b": "done", "c": "not_done"}, t) == "not_done"
    assert h4.majority({"a": "done"}, t) == "unread"
    for r in (
        {"a": "gamed", "b": "gamed"},
        {"a": "gamed", "b": "genuine", "c": "genuine"},
        {"a": "genuine"},
    ):
        assert h4.majority(r, g) == h3.label(r)
    pairs = [("genuine", "genuine")] * 6 + [("gamed", "gamed")] * 2 + [("genuine", "gamed")] * 2
    assert h4.kappa(pairs, g) == h3.kappa(pairs)
    swap = {"genuine": "done", "gamed": "not_done"}
    assert h4.kappa([(swap[x], swap[y]) for x, y in pairs], t) == h3.kappa(pairs)


def _cell():
    """Two prompts x 4 draws: p1 has a screened pass, a gamed one, a genuine-but-not-done one, a fail; p2 one genuine+done pass."""
    txt = {
        "scr": "x</think>one two",
        "gam": "x</think>a b c",
        "gen": "x</think>A real answer. Yes.",
        "no": "x",
    }
    draws = {
        ("p1", 0, 0): {"text": txt["scr"], "strict": True},
        ("p1", 0, 1): {"text": txt["gam"], "strict": True},
        ("p1", 0, 2): {"text": txt["gen"], "strict": True},
        ("p1", 0, 3): {"text": txt["no"], "strict": False},
        **{("p2", 0, i): {"text": txt["no"], "strict": False} for i in range(3)},
        ("p2", 0, 3): {"text": txt["gen"], "strict": True},
    }
    item_of = {("p1", 0, 1): "h1", ("p1", 0, 2): "h2", ("p2", 0, 3): "h3"}
    plab = {"h1": "gamed", "h2": "genuine", "h3": "genuine"}
    tlab = {"h1": "not_done", "h2": "not_done", "h3": "done"}
    return h4.cell_arrays(draws, {("p1", 0, 0)}, item_of, plab, tlab, ["p1", "p2"])


def test_rates_count_accepted_passes_over_all_draws():
    q = h4.point(_cell())
    assert q["strict_rate"] == 4 / 8
    assert q["genuine_rate"] == 2 / 8  # passes neither screened nor read gamed, over all draws
    assert q["task_done_rate"] == 1 / 8  # passes neither screened nor labelled not_done
    assert q["screened_share"] == 1 / 4 and q["read_gamed_share"] == 1 / 3
    assert q["gamed_share_of_passes"] == 2 / 4 and q["read_not_done_share"] == 2 / 3
    assert q["unpunctuated_share"] == 2 / 4 and q["median_words"] == 3.5


def test_unread_items_are_not_counted_as_accepted():
    a = _cell()
    draws = {("p1", 0, i): {"text": "x</think>fine.", "strict": True} for i in range(2)}
    b = h4.cell_arrays(draws, set(), {k: f"h{k[2]}" for k in draws}, {"h0": "genuine"}, {}, ["p1"])
    assert h4.point(b)["genuine_rate"] == 0.5 and h4.point(b)["task_done_rate"] == 0.0
    assert a["read"].sum() - a["gamed"].sum() - a["genuine"].sum() == 0


def test_bootstrap_brackets_the_point_and_shares_prompts():
    rng = np.random.default_rng(0)
    base = {m: np.zeros((30, 8)) for m in h4.METRICS}
    base["strict"] = (rng.random((30, 8)) < np.linspace(0.1, 0.9, 30)[:, None]).astype(float)
    base["read"] = base["strict"].copy()
    base["genuine"] = base["strict"] * (rng.random((30, 8)) < 0.7)
    base["done"] = base["genuine"].copy()
    base["words"] = np.where(base["strict"] > 0, 12.0, np.nan)
    four = {m: v[:, :4] for m, v in base.items()}  # a cell with fewer draws per prompt
    bs = h4.boot({"X": base, "Y": base, "Z": four}, reps=600, seed=1, block=250)
    assert all(len(v) == 600 for v in bs["X"].values())
    for q in ("strict_rate", "genuine_rate"):
        lo, hi = h4.pct(bs["X"][q])
        assert lo < h4.point(base)[q] < hi
    assert set(np.unique(bs["Z"]["median_words"])) == {12.0}
    # the prompt resample is shared, so two copies of one cell differ less than independent prompt draws would
    paired = np.std(bs["X"]["strict_rate"] - bs["Y"]["strict_rate"])
    assert paired < 0.75 * np.sqrt(2) * np.std(bs["X"]["strict_rate"])
    con = h4.contrasts(
        {c: h4.point(base) for c in h4.CELLS}, {c: bs["X" if c < "C" else "Y"] for c in h4.CELLS}
    )
    assert (
        con["B-D"]["genuine_rate"]["diff"] == 0
        and con["B-D"]["genuine_rate"]["lo"] < 0 < con["B-D"]["genuine_rate"]["hi"]
    )
    assert set(con) == {n for _, _, n in h4.CONTRASTS} and len(con) == 14


def _con(**kw):
    z = {"diff": 0.0, "lo": -0.1, "hi": 0.1}
    return {n: {"strict_rate": {"diff": 0.2, "lo": 0.1, "hi": 0.3}, "genuine_rate": kw.get(n, z)}
            for n in ("B-D", "A'-C", "A-D", "A'-D")}  # fmt: skip


def test_readings_are_round3s_rules():
    # round 3's own numbers: A-D +8.4 [-4.7, +20.8] is "not shown" (8.4 > 24.22 / 3); A'-D +5.9 is "mostly"
    r = h4.readings(
        _con(**{"A-D": {"diff": 0.084, "lo": -0.047, "hi": 0.208}, "A'-D": {"diff": 0.059, "lo": -0.074, "hi": 0.187},
                "B-D": {"diff": 0.1, "lo": 0.01, "hi": 0.2}, "A'-C": {"diff": 0.1, "lo": 0.0, "hi": 0.2}}),
        "genuine_rate",
    )  # fmt: skip
    assert r["A-D"]["reading"] == "not shown" and r["A'-D"]["reading"] == "mostly"
    assert r["B-D"]["survives"] and not r["A'-C"]["survives"]  # the lower bound must be above 0
    assert r["B-D"]["ratio_to_raw"] == pytest.approx(0.5)
    # a small gap whose interval excludes 0 is not "mostly"
    r = h4.readings(_con(**{"A-D": {"diff": 0.05, "lo": 0.01, "hi": 0.09}}), "genuine_rate")
    assert r["A-D"]["reading"] == "not shown"
