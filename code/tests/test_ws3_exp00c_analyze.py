"""exp-00c analyze.py on tiny synthetic run folders: pairing, sign test, counts, bootstrap
determinism, selection rule with its exclusions, and skipping runs without evaluate dumps."""

import gzip
import importlib.util
import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "experiments", "ws3", "exp-00c-lr-ei-cpu", "analyze.py")


def _load():
    spec = importlib.util.spec_from_file_location("exp00c_analyze", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load()


def _rows(prompt, length, n, c):
    return [{"label": "x", "name": prompt, "prompt": prompt, "length": length, "slice": "s",
             "completion": "c", "ok": i < c, "reason": "", "n_lines_written": 1, "n_tokens": 5,
             "k": n, "temperature": 1.0} for i in range(n)]


def _write_dump(d, part, problems, gz=True):
    """problems: list of (prompt, length, n, c)."""
    os.makedirs(d, exist_ok=True)
    rows = [r for p in problems for r in _rows(*p)]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if gz:
        with gzip.open(os.path.join(d, part + ".jsonl.gz"), "wt") as f:
            f.write(text)
    else:
        with open(os.path.join(d, part + ".jsonl"), "w") as f:
            f.write(text)


# Frozen long_pool: length 7 has four solved problems and two unsolved; length 8 two unsolved.
FROZEN_LONG = [("a7", 7, 4, 1), ("b7", 7, 4, 2), ("c7", 7, 4, 2), ("d7", 7, 4, 3),
               ("e7", 7, 4, 0), ("f7", 7, 4, 0), ("a8", 8, 4, 0), ("b8", 8, 4, 0)]
FROZEN_GREEDY = [(f"g{i}", 2 + i % 5, 1, 1 if i < 8 else 0) for i in range(10)]
FROZEN_TB = [(f"t{i}", -1, 2, i % 2) for i in range(4)]


def _trained_long(delta_solved):
    """Shift the frozen-solved length-7 counts by delta_solved (list of 4 ints)."""
    out = []
    for (p, L, n, c), d in zip(FROZEN_LONG[:4], delta_solved, strict=True):
        out.append((p, L, n, c + d))
    out += [("e7", 7, 4, 2), ("f7", 7, 4, 1), ("a8", 8, 4, 3), ("b8", 8, 4, 0)]
    return out


def _metrics(path, kl_last, grpo=True):
    with open(path, "w") as f:
        if grpo:
            for s in range(10):
                f.write(json.dumps({"step": s, "reward": s / 10, "kl": kl_last * s / 9,
                                    "entropy": 2.0 - s * 0.1, "groups/frac_mixed": 0.5,
                                    "rollout/total_completions": 64 * (s + 1)}) + "\n")
        else:
            for r in range(3):
                f.write(json.dumps({"round": r, "reward": 0.1 * r,
                                    "rollout/total_completions": 100 * (r + 1)}) + "\n")
                f.write(json.dumps({"round": r, "sft/loss_last": 1.0 - 0.2 * r}) + "\n")


def _make_run(exp, folder, delta_solved, greedy_lost=0, kl_last=0.1, grpo=True, gz=True):
    run = exp / folder
    base, ev = str(run / "baseline" / "dumps"), str(run / "evaluate" / "dumps")
    _write_dump(base, "long_pool", FROZEN_LONG, gz)
    _write_dump(base, "greedy_heldout", FROZEN_GREEDY, gz)
    _write_dump(base, "textbook72", FROZEN_TB, gz)
    _write_dump(ev, "long_pool", _trained_long(delta_solved), gz)
    g = [(p, L, n, 0 if i < greedy_lost else c) for i, (p, L, n, c) in enumerate(FROZEN_GREEDY)]
    _write_dump(ev, "greedy_heldout", g, gz)
    _write_dump(ev, "textbook72", FROZEN_TB, gz)
    _metrics(run / "metrics.jsonl", kl_last, grpo)


# ------------------------------------------------------------------ statistics


def test_sign_test():
    assert A.sign_test(0, 0) == 1.0
    assert A.sign_test(3, 3) == 1.0
    assert A.sign_test(5, 0) == pytest.approx(2 / 32)
    assert A.sign_test(0, 5) == pytest.approx(2 / 32)
    # 8 up 2 down: 2 * P(X <= 2), X ~ Bin(10, .5) = 2 * 56 / 1024
    assert A.sign_test(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert A.sign_test(1, 1) == 1.0


def test_bootstrap_deterministic():
    xs = [0.1, -0.2, 0.3, 0.0, 0.25, -0.05]
    a, b = A.bootstrap_ci(xs), A.bootstrap_ci(xs)
    assert a == b
    assert a[0] <= sum(xs) / len(xs) <= a[1]
    assert A.bootstrap_ci([]) is None
    assert A.bootstrap_ci([0.5, 0.5]) == [0.5, 0.5]


def test_pair_asserts_same_prompts():
    b = {"x": {"n": 1, "c": 1, "p": 1.0, "length": 2}}
    t = {"y": {"n": 1, "c": 1, "p": 1.0, "length": 2}}
    with pytest.raises(AssertionError):
        A.pair(b, t)


# ------------------------------------------------------------------ per-run numbers


def test_headline_counts_and_new_capability(tmp_path):
    _make_run(tmp_path, "run_lr1e-4", [+1, +1, 0, -1], gz=False)  # uncompressed dumps
    r = A.analyze_run(str(tmp_path), "run_lr1e-4", "grpo", 1e-4)
    h = r["headline"]
    assert (h["n_problems"], h["n_frozen_solved"]) == (6, 4)
    assert (h["up"], h["down"], h["unchanged"]) == (2, 1, 1)
    assert h["sign_p"] == pytest.approx(1.0)
    assert h["mean_diff_solved"] == pytest.approx((0.25 + 0.25 + 0 - 0.25) / 4)
    assert h["mean_diff_all"] == pytest.approx((0.25 + 0.25 - 0.25 + 0.5 + 0.25) / 6)
    nc = r["new_capability"]
    assert nc["7"] == {"n_problems": 6, "n_frozen_zero": 2, "new_capability": 1, "new_any": 2}
    assert nc["8"] == {"n_problems": 2, "n_frozen_zero": 2, "new_capability": 1, "new_any": 1}
    s = r["secondary"]
    assert s["greedy_lost"] == 0 and s["greedy_gained"] == 0
    assert s["greedy_overall"]["delta"] == pytest.approx(0.0)
    assert s["textbook72"]["frozen"] == pytest.approx(0.25)
    assert set(s["long_pool_by_length"]) == {"7", "8"}
    d = r["dynamics"]
    assert d["final_kl"] == pytest.approx(0.1) and d["max_kl"] == pytest.approx(0.1)
    assert d["entropy_first5"] == pytest.approx(1.8)
    assert d["total_completions"] == 640
    assert d["reward_last5"] == pytest.approx(0.7)


def test_ei_dynamics(tmp_path):
    _make_run(tmp_path, "run_ei", [0, 0, 0, 0], grpo=False)
    r = A.analyze_run(str(tmp_path), "run_ei", "ei", None)
    d = r["dynamics"]
    assert d["total_completions"] == 300
    assert [x["round"] for x in d["rounds"]] == [0, 1, 2]
    assert d["rounds"][2]["sft/loss_last"] == pytest.approx(0.6)
    assert d["rounds"][1]["reward"] == pytest.approx(0.1)


# ------------------------------------------------------------------ selection and skipping


def test_selection_and_skip(tmp_path, capsys):
    # lr1e-4: best mean diff but final KL 1.5 -> excluded
    _make_run(tmp_path, "run_lr1e-4", [+2, +2, +1, +1], kl_last=1.5)
    # lr3e-4: good diff but greedy drops 3/10 = 0.3 -> excluded
    _make_run(tmp_path, "run_lr3e-4", [+2, +1, +1, +1], greedy_lost=3)
    # lr3e-5: modest diff, qualifies
    _make_run(tmp_path, "run_lr3e-5", [+1, 0, 0, 0])
    # run_ei has no evaluate dumps -> skipped
    os.makedirs(tmp_path / "run_ei" / "baseline" / "dumps")
    out = A.run_all(str(tmp_path))
    assert out["skipped"] == ["run_ei"]
    assert "skip run_ei" in capsys.readouterr().out
    sel = out["selection"]
    assert sel["chosen_lr"] == pytest.approx(3e-5)
    reasons = {e["folder"]: " ".join(e["reasons"]) for e in sel["excluded"]}
    assert "KL" in reasons["run_lr1e-4"] and "greedy" in reasons["run_lr3e-4"]
    for f in ("analysis.json", "analysis.md", "analysis_len7.png"):
        assert (tmp_path / f).exists()
    assert "chosen lr = 3e-05" in (tmp_path / "analysis.md").read_text()
    # determinism of the bootstrap across runs of the whole pipeline
    again = A.run_all(str(tmp_path), verbose=False)
    ci = [r["headline"]["mean_diff_solved_ci"] for r in out["runs"]]
    assert ci == [r["headline"]["mean_diff_solved_ci"] for r in again["runs"]]


def test_selection_none_qualifies_and_ignores_textbook(tmp_path):
    _make_run(tmp_path, "run_lr1e-4", [+1, 0, 0, 0], kl_last=2.0)
    _make_run(tmp_path, "run_lr3e-4", [+1, 0, 0, 0], greedy_lost=1)  # drop 0.1 > 0.02
    out = A.run_all(str(tmp_path), verbose=False)
    assert out["skipped"] == ["run_lr3e-5", "run_ei"]
    assert out["selection"]["chosen_lr"] is None
    assert A.selection_text(out["selection"]) == "Selection rule: none qualifies"
    # textbook72 changes must not affect selection
    res = out["runs"]
    for r in res:
        r["secondary"]["textbook72"] = {"delta": 1.0}
    assert A.select_lr(res)["chosen_lr"] is None


def test_selection_boundary_greedy_drop_exactly_002():
    def run(lr, md, gd, kl):
        return {"kind": "grpo", "lr": lr, "folder": f"run_lr{lr}",
                "headline": {"mean_diff_solved": md},
                "secondary": {"greedy_overall": {"delta": gd}}, "dynamics": {"final_kl": kl}}

    res = [run(1e-4, 0.05, -0.02, 1.0), run(3e-4, 0.10, -0.0201, 0.5), run(1e-3, 0.2, 0, 1.01)]
    sel = A.select_lr(res)
    assert sel["chosen_lr"] == 1e-4  # -0.02 is not "more than 0.02"; KL 1.0 does not exceed 1.0
    assert not math.isnan(sel["considered"][0]["mean_diff_solved"])
