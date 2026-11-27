"""Tests of lh_pairs: the anchor states (canary/1 rows of untagged records), the split by theorem, the balancing by
sampled side, and the builder of pairs, fine-tune files and reward-model probe twins on real c24 proofs with the corpus
agent's twins (Lean replaced by a stub here; the CPU smoke runs real Lean).

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_pairs.py
"""

# ruff: noqa: E501, I001

from __future__ import annotations

import gzip
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core as C  # noqa: E402
import lh_pairs as LP  # noqa: E402

DATA = C.LH / "data"
needs = pytest.mark.skipif(
    not (DATA / "c24.jsonl.gz").exists() or not C.HABIT_TABLE.exists(),
    reason="c24 or the habit table not built",
)


def records(n: int) -> list[dict]:
    with gzip.open(DATA / "c24.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in itertools.islice(f, n)]


def test_split_and_balance():
    pairs = [
        {"theorem": f"t{i // 3}", "k": i, "sampled_side": "A" if i % 3 == 0 else "B"}
        for i in range(60)
    ]  # 20 x 3
    train, held, names = LP.split(pairs, 8, "s")
    assert (
        sum(p["sampled_side"] == "A" for p in held) >= 4 and len(names) == 4
    )  # whole theorems until 4 A-sampled
    assert not {p["theorem"] for p in train} & names and len(train) + len(held) == 60
    assert LP.split(pairs, 8, "s")[1] == held  # seeded
    b = LP.balance(train, "x")
    assert (
        sum(p["sampled_side"] == "A" for p in b) == sum(p["sampled_side"] == "B" for p in b) == 16
    )
    pos = [train.index(p) for p in b]
    assert pos == sorted(pos) and LP.balance(train, "x") == b  # the input order kept, seeded
    hb = LP.balance(held, "y", cap=6)
    assert len(hb) == 6 and sum(p["sampled_side"] == "A" for p in hb) == 3


def test_extra_probe_rows():
    def x(t, k, form, lines):
        return {
            "theorem": t,
            "key": f"{t}#{k}",
            "prompt": "P",
            "hab": {},
            "reit_form": form,
            "lines": lines,
        }

    res = [x("t1", 0, None, 5), x("t1", 1, "A", 7), x("t1", 2, "A", 9), x("t1", 3, None, 4),
           x("t2", 0, "B", 6), x("t2", 1, "B", 6), x("t3", 0, None, 3), x("t3", 1, "A", 8)]  # fmt: skip
    proofs = {r["key"]: f"proof {r['key']}" for r in res}
    rows, n = LP.extra_probe_rows(res, proofs, {"t1", "t2"}, "s", 10)  # t3 is not held out
    ns = [r for r in rows if r["habit"] == "probe:nosite"]
    ln = [r for r in rows if r["habit"] == "probe:length"]
    assert len(ns) == 2 and all(
        r["reit_form"][0] is not None and r["reit_form"][1] is None for r in ns
    )
    assert (
        len(ln) == 1 and ln[0]["lines"] == [9, 7] and ln[0]["preferred"] == "proof t1#2"
    )  # t2's pair: equal lengths
    assert not [r for r in rows if r["key"].startswith("t3")] and n["probe:nosite"] == 2
    assert LP.extra_probe_rows(res, proofs, {"t1", "t2"}, "s", 10)[0] == rows  # seeded
    assert (
        len(LP.extra_probe_rows(res, proofs, {"t1", "t2"}, "s", 1)[0]) == 2
    )  # at most 1 of each kind


def test_form_of():
    assert (
        LP.form_of({"sites": 2, "a": 2, "b": 0}) == "A"
        and LP.form_of({"sites": 2, "a": 0, "b": 2}) == "B"
    )
    assert LP.form_of({"sites": 2, "a": 1, "b": 1}) == "mixed" and LP.form_of({"sites": 0}) is None
    assert (
        LP.form_of({"sites": 2, "a": 1, "b": 0}) == "mixed"
    )  # a site without a form is not a pure member


@needs
def test_anchor_rows(tmp_path, monkeypatch):
    out = tmp_path / "anchor.jsonl"
    meta = LP.anchor_main(["--corpus", "c24", "--n", "40", "--out", str(out)])
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert (
        meta["rows"] == len(rows) == 40
        and meta["steps"] == sum(len(r["steps"]) for r in rows)
        and meta["min_hyp"] == 0
    )
    assert all(r["format"] == "canary/1" and r["steps"] and r["steps"][0]["i"] == 0 for r in rows)
    ks = tmp_path / "anchor.klstates.jsonl"
    assert ks.exists() and len(ks.read_text().splitlines()) == 40
    with pytest.raises(SystemExit, match="never overwritten"):
        LP.anchor_main(["--corpus", "c24", "--n", "40", "--out", str(out)])
    import tag_core as T

    T.install_thinking()
    T.install_dan()
    from code.thinking.canary import load_canary  # type: ignore[import-not-found]

    tok = T.install_dan()()
    T.set_process_tag(
        "none"
    )  # the GRPO process's tag (tag_grpo --tag none): the anchor's states are read untagged
    cd = load_canary(out, tok, False, 0)
    assert len(cd.anchor) == meta["steps"] and not cd.canary


@needs
def test_build_on_corpus_proofs(tmp_path, monkeypatch):
    import lh_twin as TW

    recs = records(600)
    d = tmp_path / "pairs" / "c24_s0"
    monkeypatch.setattr(C, "pairs_dir", lambda c, r=None, seed=0: d if r is None else d / r)
    rows = [{"theorem": f"th{i // 2}", "key": r["key"], "prompt": r["prompt"], "attempt": i % 2, "proof": r["proof"]}
            for i, r in enumerate(recs)]  # fmt: skip
    LP.write_jsonl(d / "sampled.jsonl.gz", rows)
    calls = []

    def fake_lean(pairs, workers=8, cache=None):
        calls.append(len(pairs))
        return [i % 10 != 0 for i in range(len(pairs))]  # every tenth twin "rejected by Lean"

    monkeypatch.setattr(TW, "lean_check_twins", fake_lean)
    monkeypatch.setattr(
        C, "TRAIN_PAIR_CAP", 100
    )  # small caps here (4,000 on the pod): the seeded draws are exercised
    monkeypatch.setattr(C, "SFT_PROOF_CAP", 150)
    monkeypatch.setattr(C, "PROBE_EXTRA_ROWS", 7)
    doc = LP.build_main(["--corpus", "c24", "--rater", "reit", "--heldout", "20", "--workers", "2"])
    rd = {k: [json.loads(x) for x in (d / "reit" / f"{k}.jsonl").read_text().splitlines()]
          for k in ("train", "heldout", "sft_pref", "sft_sampled", "probe_twins")}  # fmt: skip
    tr, he = rd["train"], rd["heldout"]
    bal = doc["balance"]
    assert (
        doc["why"].get("twin_lean_rejected", 0) == calls[0] - doc["pairs"] > 0 and len(calls) == 2
    )
    assert sum(doc["why"].values()) == doc["distinct"] == 600
    # balanced train and held-out (design audit F5), counts before and after
    for k, xs in (("train", tr), ("heldout", he)):
        assert (
            bal[f"{k}_after"]["A"]
            == bal[f"{k}_after"]["B"]
            == sum(x["sampled_side"] == "A" for x in xs)
            == len(xs) / 2
        )
    cap = doc["cap"]
    assert (
        len(he) <= 20
        and len(tr) == cap["train_capped"] == 100 < cap["train_balanced"]
        and not cap["train_below_cap"]
    )
    assert (
        sum(bal["train_before"].values())
        == cap["sft_before_cap"]
        > 150
        == cap["sft_capped"]
        == len(rd["sft_pref"])
        == len(rd["sft_sampled"])
    )
    assert [x["key"] for x in rd["sft_pref"]] == [
        x["key"] for x in rd["sft_sampled"]
    ]  # the same pairs in both files
    nt = doc[
        "norm_tokens"
    ]  # Amendment 7 point 4: batch x mean action tokens per proof of sft_sampled
    assert (
        nt["batch"] == C.OFFLINE_BATCH
        and nt["proofs"] == 150
        and nt["N"] == pytest.approx(C.OFFLINE_BATCH * nt["action_tokens"] / 150)
    )
    # the fine-tune files: every train pair before the balancing; preferred = all reit A, sampled = as the base wrote it
    import lh_profile as PR

    tab = C.profile_tables()
    for x, y in zip(rd["sft_pref"][:60], rd["sft_sampled"][:60], strict=True):
        assert x["key"] == y["key"] and set(x) == {"key", "prompt", "proof"}
        pa, pb = PR.profile(x["prompt"], x["proof"], tab), PR.profile(y["prompt"], y["proof"], tab)
        assert LP.form_of(pa["habits"]["reit"]) == "A" and LP.reading(pa, "reit") == LP.reading(
            pb, "reit"
        )
    comp = doc["a_share"]
    assert comp["sft_pref"]["reit"] == 100.0 and set(comp) == {
        "base_sampled",
        "sft_pref",
        "sft_sampled",
    }
    assert (
        comp["sft_pref"]["order"] == comp["sft_sampled"]["order"]
    )  # only the rated habit differs between the files
    for x in tr[:40] + he:
        assert (
            set(x) == {"key", "prompt", "preferred", "other", "habit", "sampled_side"}
            and x["habit"] == "reit"
        )
        pa, pb = (
            PR.profile(x["prompt"], x["preferred"], tab),
            PR.profile(x["prompt"], x["other"], tab),
        )
        assert LP.form_of(pa["habits"]["reit"]) == "A" and LP.form_of(pb["habits"]["reit"]) == "B"
        assert LP.reading(pa, "reit") == LP.reading(pb, "reit")
    assert not {x["key"].split("#")[0] for x in tr} & {x["key"].split("#")[0] for x in he}
    # probe twins: held-out theorems only, every other flippable habit, A member preferred, nothing else changed
    held_th = {x["key"].split("#")[0] for x in he}
    pt = [x for x in rd["probe_twins"] if not x["habit"].startswith("probe:")]
    ex = [x for x in rd["probe_twins"] if x["habit"].startswith("probe:")]
    assert (
        pt
        and {x["habit"] for x in pt} <= set(TW.FLIPPABLE) - {"reit"}
        and sum(doc["probe_twins"].values()) == len(pt)
    )
    # Amendment 7 point 6: probe:nosite (preferred has a reiteration site) and probe:length (preferred is longer)
    assert {x["habit"] for x in ex} == {"probe:nosite", "probe:length"} and len(ex) <= 14
    assert doc["probe_extra"]["probe:nosite"] == sum(x["habit"] == "probe:nosite" for x in ex) <= 7
    for x in ex:
        assert x["key"].split("#")[0] in held_th | {r["theorem"] for r in rows} - {
            y["key"].split("#")[0] for y in tr
        }
        pa, pb = (
            PR.profile(x["prompt"], x["preferred"], tab),
            PR.profile(x["prompt"], x["other"], tab),
        )
        sa, sb = pa["habits"]["reit"]["sites"], pb["habits"]["reit"]["sites"]
        if x["habit"] == "probe:nosite":
            assert sa > 0 and sb == 0
        else:
            assert (
                sa > 0
                and sb > 0
                and LP.form_of(pa["habits"]["reit"]) == LP.form_of(pb["habits"]["reit"])
                and pa["lines"] > pb["lines"]
            )
    assert {x["key"].split("#")[0] for x in pt} <= {r["theorem"] for r in rows} - {
        x["key"].split("#")[0] for x in tr
    }
    assert held_th <= {r["theorem"] for r in rows}
    # a rater without a GRPO line: no probe twins (--no_probe; PREREG Amendment 4)
    doc2 = LP.build_main(
        [
            "--corpus",
            "c24",
            "--rater",
            "atom",
            "--heldout",
            "20",
            "--workers",
            "2",
            "--limit",
            "200",
            "--no_probe",
        ]
    )
    assert (
        not (d / "atom" / "probe_twins.jsonl").read_text().strip()
        and doc2["probe_twins"] == {}
        and doc2["probe_extra"] == {}
    )
    for x in pt[:40]:
        h = x["habit"]
        pa, pb = (
            PR.profile(x["prompt"], x["preferred"], tab),
            PR.profile(x["prompt"], x["other"], tab),
        )
        assert (
            LP.form_of(pa["habits"][h]) == "A"
            and LP.form_of(pb["habits"][h]) == "B"
            and LP.reading(pa, h) == LP.reading(pb, h)
        )
