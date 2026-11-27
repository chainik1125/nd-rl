"""Tests of lh_measure (version 2): the 24-habit counts of a row against an independent count from the detector's own
reading, the corpus round trip, the step limit, the totals (pass@1, pass@8, the share at the limit), the patched row,
the sets and the run defaults.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_measure.py
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
import lh_measure as LM  # noqa: E402

DATA = C.LH / "data"
needs = pytest.mark.skipif(
    not (DATA / "c24.jsonl.gz").exists() or not C.HABIT_TABLE.exists(),
    reason="c24 or the habit table not built",
)


def records(name: str, n: int) -> list[dict]:
    with gzip.open(DATA / f"{name}.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in itertools.islice(f, n)]


def brute(prof: dict) -> dict:
    """first / later / link / linko / pair / cond from the detector's `seq`, counted a second way (by position lists)."""
    seq = [
        (h, s)
        for _, h, s in prof["seq"]
        if h not in ("route", "split", "exfalso") and s in ("A", "B")
    ]
    out: dict = {
        "first": {},
        "later": {},
        "link": {},
        "linko": {},
        "pair": {},
        "cond": {},
        "condn": {},
    }
    if not seq:
        return out
    out["first"][seq[0][0]] = [int(seq[0][1] == "A"), int(seq[0][1] == "B")]
    for h, s in seq[1:]:
        v = out["later"].setdefault(h, [0, 0])
        v[s == "B"] += 1
    grp = {"notes": [i for i, (h, _) in enumerate(seq) if h.startswith("note_")],
           "natural": [i for i, (h, _) in enumerate(seq) if not h.startswith("note_")], "all": list(range(len(seq)))}  # fmt: skip
    for x, ix in grp.items():
        if not ix:
            continue
        i = ix[0]
        for y, iy in grp.items():
            v, vo = [0, 0, 0, 0], [0, 0, 0, 0]
            for j in iy:
                if j > i:
                    v[2 * (seq[i][1] == "B") + (seq[j][1] == "B")] += 1
                    if seq[j][0] != seq[i][0]:
                        vo[2 * (seq[i][1] == "B") + (seq[j][1] == "B")] += 1
            out["link"][f"{x}|{y}"], out["linko"][f"{x}|{y}"] = v, vo
    for h1 in {h for h, _ in seq}:
        i = next(k for k, (h, _) in enumerate(seq) if h == h1)
        for j in range(i + 1, len(seq)):
            v = out["pair"].setdefault(f"{h1}|{seq[j][0]}", [0, 0, 0, 0])
            v[2 * (seq[i][1] == "B") + (seq[j][1] == "B")] += 1
    for r in ("reit", "note_atom"):
        i = next((k for k, (h, _) in enumerate(seq) if h == r), None)
        if i is None:
            continue
        out["condn"][r] = [int(seq[i][1] == "A"), int(seq[i][1] == "B")]
        for h, s in seq[:i]:
            v = out["cond"].setdefault(r, {}).setdefault(h, [0, 0, 0, 0])
            v[2 * (seq[i][1] == "B") + (s == "B")] += 1
    return out


@needs
def test_counts_match_the_detector():
    P, tab = LM.profiler()
    for r in records("c24", 300):
        c = LM.empty_lh()
        LM.add_proof(c, r["prompt"], r["proof"])
        p = P.profile(r["prompt"], r["proof"], tab)
        assert c["proofs"] == 1 and c["lines"] == p["lines"]
        for h, v in p["habits"].items():
            assert c["hab"][h] == [v["sites"], v["a"], v["b"], v.get("other", 0), v.get("none", 0)]
        b = brute(p)
        assert c["first"] == b["first"] and c["later"] == b["later"], r["key"]
        for fld in ("link", "linko"):
            assert {k: v for k, v in c[fld].items() if any(v)} == {
                k: v for k, v in b[fld].items() if any(v)
            }, (fld, r["key"])
        assert c["pair"] == b["pair"], r["key"]
        assert {k: v for k, v in c["cond"].items() if v} == b["cond"], r["key"]
        assert {k: v for k, v in c["condn"].items() if any(v)} == b["condn"], r["key"]
        assert not any(
            k.split("|")[0] in ("split", "exfalso") or k.split("|")[1] in ("split", "exfalso")
            for k in c["pair"]
        )


@needs
def test_corpus_round_trip_by_author():
    """In c24 every linked habit is in its author's form: the A-author's notes read 100 % A, the B-author's 0 %."""
    by = {"A": LM.empty_lh(), "B": LM.empty_lh()}
    for r in records("c24", 400):
        LM.add_proof(by[r["author"]], r["prompt"], r["proof"])
    for au, want in (("A", 100.0), ("B", 0.0)):
        hab = by[au]["hab"]
        a = sum(v[1] for h, v in hab.items() if h.startswith("note_"))
        b = sum(v[2] for h, v in hab.items() if h.startswith("note_"))
        assert a + b > 50 and 100 * a / (a + b) == want
        assert by[au]["unread"] == 0


def test_near_misses_are_summed(monkeypatch):
    """lh_profile's `near` counters are summed per row; a profile without them counts 0 (design audit F9)."""
    profs = iter([{"ok": True, "lines": 5, "habits": {}, "seq": [], "near": {"ande_apart": 1, "late_note": 2}},
                  {"ok": True, "lines": 4, "habits": {}, "seq": []}])  # fmt: skip

    class P:
        @staticmethod
        def profile(prompt, proof, tab):
            return next(profs)

    monkeypatch.setattr(LM, "profiler", lambda: (P, None))
    c = LM.lh_counts("p", ["x", "y"])
    assert c["near"] == {
        "ande_apart": 1,
        "dead_r_in_box": 0,
        "cited_header_copy": 0,
        "double_note": 0,
        "late_note": 2,
    }
    t = LM.totals([{"attempts": 8, "accepted": 2, "lh": c}])
    assert t["near"]["late_note"] == 2 and t["near_per_proof"]["ande_apart"] == 0.5


def test_probe_main(tmp_path, monkeypatch):
    """`probe`: one `run` per snapshot (untagged, marker_plain, k 8) in one process, in order, stopping after the first
    snapshot above the threshold and then probing the last (PREREG Amendment 4, point 1); DONE per snapshot."""
    calls = []

    def fake_main(argv):
        calls.append(argv[argv.index("--ckpt_id") + 1])
        d = Path(argv[argv.index("--out_dir") + 1])
        assert (
            argv[argv.index("--tags") + 1] == "none"
            and argv[argv.index("--sets") + 1] == "marker_plain"
            and argv[argv.index("--k") + 1] == "8"
        )
        sh = {"reit": 40.0 + 10 * len(calls)}
        (d / "marker_plain__none.summary.json").write_text(
            json.dumps({"lh_totals": {"a_share": sh, "accepted_pct": 60.0}})
        )

    monkeypatch.setattr(LM.AM, "main", fake_main)
    monkeypatch.setattr(LM, "install", lambda: None)
    ids = ["step25", "step50", "step100", "step150", "final"]
    argv = ["--ckpts", ",".join(f"{i}.pt" for i in ids), "--ids", ",".join(ids), "--out_root", str(tmp_path / "p"),
            "--raw_root", str(tmp_path / "r"), "--id_prefix", "arm@", "--stop_habit", "reit", "--stop_above", "65"]  # fmt: skip
    out = LM.probe_main(argv)
    assert calls == [
        "arm@step25",
        "arm@step50",
        "arm@step100",
        "arm@final",
    ]  # 50, 60, 70 > 65: stop; then the last
    assert (
        out["stopped_at"] == "step100"
        and out["skipped"] == ["step150"]
        and out["snapshots"]["final"]["a_share"]["reit"] == 80.0
    )
    assert json.loads((tmp_path / "p" / "probe.json").read_text())["acceptance"].startswith(
        "the Lean gate"
    )
    LM.probe_main(argv)  # all DONE: no new run
    assert len(calls) == 4
    calls.clear()
    full = LM.probe_main(
        [*argv[:6], "--out_root", str(tmp_path / "q"), "--raw_root", str(tmp_path / "r2")]
    )  # no stop rule
    assert len(calls) == 5 and full["stopped_at"] is None and not full["skipped"]


def test_unreadable_is_counted():
    c = LM.empty_lh()
    LM.add_proof(c, "THM P SEQ P PRF", "this is not a proof")
    assert c["unread"] == 1 and c["proofs"] == 0 and not c["hab"]


def test_step_limit(monkeypatch):
    LM.T.install_dan()
    import state_sample as SS  # type: ignore[import-not-found]  (Dan's tree)

    seen = {}

    def orig(model, tok, prompts, greedy=True, temperature=1.0, max_action=256, max_steps=48, **kw):
        seen["max_steps"] = max_steps
        return ["LEANPARSE env: step cap", "N1 P : PR ; QED"] * (len(prompts) // 2)

    monkeypatch.setattr(SS, "env_generate", orig)
    LM.install_step_limit()
    assert LM.AM.MAX_STEPS == 51
    nds = SS.env_generate(None, None, ["p1", "p1", "p2", "p2"], greedy=False, max_steps=48)
    assert seen["max_steps"] == 51 and len(nds) == 4
    SS.env_generate(None, None, ["p1", "p1"], False, 0.8, 256, 48)  # positional too
    assert seen["max_steps"] == 51
    c = LM.lh_counts("p1", [])
    assert c["seen"] == 2 and c["step_cap"] == 1 and c["max_steps"] == 51
    assert LM.at_limit("LEANPARSE env: step cap") and not LM.at_limit(
        "LEANPARSE env: action truncated"
    )


def test_totals():
    rows = [{"attempts": 8, "accepted": 0, "lh": {"seen": 8, "step_cap": 2, "proofs": 0, "lines": 0, "hab": {}}},
            {"attempts": 8, "accepted": 4, "lh": {"seen": 8, "step_cap": 0, "proofs": 4, "lines": 40,
                                                   "hab": {"reit": [10, 6, 2, 0, 0]}}}]  # fmt: skip
    t = LM.totals(rows)
    assert t["accepted_pct"] == 25.0 and t["pass1"] == 25.0 and t["pass8"] == 50.0
    assert (
        t["step_cap_pct"] == 12.5
        and t["mean_lines"] == 10.0
        and t["a_share"]["reit"] == 75.0
        and t["step_limit"] == 51
    )
    assert LM.pass_at(16, 1, 8) == pytest.approx(0.5) and LM.pass_at(4, 1, 8) is None


@needs
def test_patched_row_has_no_proof_text():
    r = records("c24", 1)[0]
    LM._LAST.clear()
    row = LM.theorem_row({"name": "x", "prompt": r["prompt"], "key": r["key"], "n_lines": r["n_lines"]}, [r["proof"]], 8,
                         {"ckpt": "c", "tag": "none", "set": "marker_plain", "k": 8, "noise_seed": 0})  # fmt: skip
    assert row["lh"]["proofs"] == 1 and row["lh"]["step_cap"] is None and "habits" in row
    assert r["proof"] not in json.dumps(row)


def test_strict106_logs_in_its_folder(tmp_path, monkeypatch):
    """Dan's lean_gate writes artifacts/lean_gate.jsonl under its cwd (Dan's tree) by default: the read must not."""
    LM.T.install_dan()
    import tag_danread as TD  # type: ignore[import-not-found]

    seen = {}

    def fake_main(argv):
        seen["log"] = LM.os.environ["LEAN_GATE_LOG"]
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(
            json.dumps(
                {
                    "n_prompts": 106,
                    "k": 2,
                    "strict_n": 0,
                    "env_stats": {"env_end": {"step_cap": 53}},
                }
            )
        )

    monkeypatch.setattr(TD, "main", fake_main)
    monkeypatch.setattr(LM, "install", lambda: None)
    monkeypatch.setattr(LM.AM, "danread", lambda ns: Path(ns.out).write_text("{}\n"))
    monkeypatch.delenv("LEAN_GATE_LOG", raising=False)
    LM.strict106_main(
        ["--ckpt", "x.pt", "--ckpt_id", "x", "--out_dir", str(tmp_path / "m"), "--k", "2", "--cpu"]
    )
    assert seen["log"] == str(tmp_path / "m" / "strict106_lean_gate.jsonl")
    r = json.loads((tmp_path / "m" / "strict106_read.json").read_text())
    assert r["max_steps"] == 51 and r["step_cap_share"] == 0.25


def test_sets_report():
    rep = LM.sets_main(["--dry-run"])
    mp = rep["sets"]["marker_plain"]
    assert mp["theorems"] == 500 and mp["two_plus"] == 397 and mp["two_plus_pct"] == 79.4
    assert (
        rep["note_plain_rule"]["build_note_plain"] is False
        and rep["sets"]["route_unseen"]["theorems"] == 300
    )


def test_run_defaults_dry_run(capsys):
    if not (C.LH / "sets" / "marker_plain.jsonl").exists():
        pytest.skip("LH/sets not installed")
    LM.main(["run", "--ckpt", "/nonexistent.pt", "--ckpt_id", "x", "--dry-run"])
    info = json.loads(capsys.readouterr().out)
    assert info["k"] == 8 and info["theorems"] == {"marker_plain": 500, "route_unseen": 300}
    assert info["attempts"] == 800 * 8 * 3 and info["out_dir"] == str(C.LH / "measure" / "x")
