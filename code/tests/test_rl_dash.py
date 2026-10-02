"""Tests for the static RL dashboard renderer (code/tools/rl_dash.py)."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "rl_dash", Path(__file__).resolve().parents[1] / "tools" / "rl_dash.py"
)
assert SPEC and SPEC.loader
rl_dash = importlib.util.module_from_spec(SPEC)
sys.modules["rl_dash"] = rl_dash  # dataclasses resolve annotations through sys.modules
SPEC.loader.exec_module(rl_dash)


def _write(path: Path, recs: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _old_run(d: Path, n_rounds: int = 4) -> Path:
    """A run in the seed recipe's schema (no dashboard fields)."""
    d.mkdir(parents=True)
    recs: list[dict[str, Any]] = []
    for r in range(1, n_rounds + 1):
        recs.append(
            {
                "phase": "sample",
                "step": 10 * r,
                "round": r,
                "n_prompts": 400,
                "k": 32,
                "samples": 12800,
                "accept_rate": 0.1 * r,
                "qed_rate": 0.99,
                "targets_solved_round": 40 * r,
                "targets_solved_cum": 50 * r,
                "new_proofs": 30 + r,
                "mean_written_len": 6.0 + r / 10,
                "mean_pruned_len": 5.8,
                "sample_secs": 14.0,
                "judge_secs": 1.0,
                "sample_flops": 1.5e13,
                "fail_reasons": [["rule check failed: ANDI", 900], ["bad box cite", 100]],
                "window_by_len": {"6": 200, "7": 200},
                "samples_by_len": {"6": 6400, "7": 6400},
                "accepted_by_len": {"6": 1000, "7": 280 * r},
                "solved_by_len_round": {"6": 30, "7": 10},
                "solved_cum_all_by_len": {"6": 40 * r, "7": 10 * r},
            }
        )
        recs.append(
            {"phase": "train", "step": 10 * r + 5, "round": r, "loss": 0.05, "grad_norm": 0.1 * r}
        )
        recs.append(
            {
                "phase": "train",
                "step": 10 * r + 9,
                "round": r,
                "round_end": True,
                "loss": 0.05 - r / 1000,
                "ft_flops": 3.7e13,
                "mix_rl_records": 100 * r,
                "n_records": 20000 + 100 * r,
            }
        )
    recs.append({"phase": "eval", "step": 99, "heldout_greedy": 0.95, "source": "harness"})
    recs.append(
        {
            "phase": "summary",
            "step": 99,
            "rounds": n_rounds,
            "n_targets": 800,
            "targets_by_len": {"6": 400, "7": 400},
        }
    )
    _write(d / "metrics.jsonl", recs)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "heldout_greedy": 0.95,
                "compute": {"used_flops": 3.8e15},
                "rl_stats": {
                    "n_targets": 800,
                    "per_round": [
                        {"round": r, "secs": 30.0, "ft_secs": 9.0} for r in range(1, n_rounds + 1)
                    ],
                },
            }
        )
    )
    return d


def _new_run(d: Path, train: bool = True) -> Path:
    """A run in the seed_dash schema (every dashboard field)."""
    _old_run(d)
    recs = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines()]
    for rec in recs:
        if rec["phase"] == "sample":
            r = rec["round"]
            rec.update(
                {
                    "frac_all_fail": 0.8,
                    "frac_all_pass": 0.05,
                    "frac_mixed": 0.15,
                    "pass_at": {"1": 0.1, "4": 0.13, "32": 0.2},
                    "pass_at_by_len": {"6": {"1": 0.3, "4": 0.4, "32": 0.5}, "7": {"1": 0.05}},
                    "distinct_per_solved": 1.3,
                    "entropy": 0.05,
                    "kl_to_init_sampled": 0.001 * r,
                    "len_accepted": 6.1,
                    "len_rejected": 6.0,
                    "trunc_rate": 0.01,
                    "long_enough": 0.2,
                    "long_enough_by_len": {"6": 0.9, "7": 0.1},
                    "ends_on_conclusion": 0.9,
                    "ends_on_conclusion_by_len": {"6": 0.95, "7": 0.85},
                    "first_error_line_hist": {"3": 200, "4": 300, "5": 250},
                    "lines_at_risk_hist": {"1": 1000, "2": 1000, "3": 1000, "4": 800, "5": 500},
                    "cite_dist_hist": {"1": 500, "2": 200, "5": 3},
                    "cite_dist_max": 5,
                }
            )
        if rec.get("round_end"):
            rec.update(
                {
                    "n_rl_records": 100,
                    "n_retained": 20000,
                    "ft_secs": 9.0,
                    "grad_norm_mean": 0.1,
                    "grad_norm_max": 0.3,
                    "gpu_mem_gb": 4.2,
                    "phase_secs": {"sample": 14, "judge": 1, "finetune": 9, "mix": 0.4},
                    "phase_flops": {"sample": 1.5e13, "finetune": 3.7e13, "dash": 1e12, "eval": 0},
                }
            )
    for r in range(0, 5):
        recs.append(
            {
                "phase": "eval",
                "step": 10 * r,
                "round": r,
                "source": "dash",
                "heldout_greedy_dash": 0.96 - r / 100,
                "heldout_greedy_by_len": {str(L): 1 - L / 100 for L in range(2, 7)},
                "kl_to_init": 0.002 * r,
                "ref_logp_unsolved": -0.5 + r / 100,
                "ref_tok_acc": 0.93,
                "ref_logp_by_len": {"9": -0.4, "10": -0.6},
            }
        )
    _write(d / "metrics.jsonl", recs)
    _write(
        d / "examples.jsonl",
        [
            {
                "round": 1,
                "accepted": [
                    {
                        "name": "t1",
                        "L_true": 6,
                        "prompt": "THM P SEQ P PRF",
                        "proof": "N1 P : PR ; QED",
                        "written": 1,
                        "reason": "ok",
                    }
                ],
                "rejected": [],
            }
        ],
    )
    return d


def test_derived_fields_are_exact(tmp_path: Path) -> None:
    run = rl_dash.load_run(str(_old_run(tmp_path / "old")))
    row = run.rows[2]
    assert row["pass1"] == pytest.approx(0.2)  # = accept_rate
    assert row["frac_all_fail"] == pytest.approx(1 - 80 / 400)
    assert row["pass32"] == pytest.approx(80 / 400)
    assert row["trunc_rate"] == pytest.approx(0.01)
    assert row["n_retained"] == 20000
    assert run.source["pass1"].startswith("derived")
    assert "pass4" not in row and "pass4" not in run.source  # missing, not zero
    assert "entropy" not in row and "heldout" not in row


def test_render_marks_missing_not_zero(tmp_path: Path) -> None:
    old = rl_dash.load_run("old=" + str(_old_run(tmp_path / "runs" / "old_s1")))
    new = rl_dash.load_run("new=" + str(_new_run(tmp_path / "runs" / "new_s1")))
    ctl = rl_dash.load_run("ctl=" + str(_new_run(tmp_path / "runs" / "ctl")))
    files = rl_dash.render([old, new], ctl, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in files}
    for run in ("old", "new", "ctl"):
        for key in ("1_limiting", "2_diversity", "3_integrity", "4_retention", "6_length_wall"):
            assert f"{run}/{key}.html" in names
    assert "5_comparison.html" in names and "index.html" in names
    div_old = (tmp_path / "out" / "old" / "2_diversity.html").read_text()
    assert "pass@4 (missing)" in div_old  # a series the run lacks is named, not drawn as 0
    assert "old (missing)" in div_old  # entropy: only the control has it; the run is named missing
    integ_old = (tmp_path / "out" / "old" / "3_integrity.html").read_text()
    assert "missing: not logged" in integ_old  # no examples.jsonl
    ret_new = (tmp_path / "out" / "new" / "4_retention.html").read_text()
    assert "missing: not logged" not in ret_new
    wall = (tmp_path / "out" / "new" / "6_length_wall.html").read_text()
    assert "Per-line failure hazard" in wall and "<svg" in wall
    assert "http://" not in wall.replace("http://www.w3.org", "") and "https://" not in wall


def test_eval_dir_join(tmp_path: Path) -> None:
    run_dir = _old_run(tmp_path / "exp" / "runs" / "A_s1")
    ev = tmp_path / "exp" / "eval" / "A_s1"
    (ev / "samples").mkdir(parents=True)
    (ev / "meta.json").write_text(json.dumps({"ckpt": str(run_dir / "final.pt")}))
    (ev / "std_L10.summary.json").write_text(
        json.dumps({"pass@1": 0.1, "pass@4": 0.2, "pass@32": 0.3})
    )
    rows = [{"n_ok": 1, "ok_lines": [10], "fails": [["bad_line", 4, "rule", 1]] * 150}]
    _write(ev / "std_L10.jsonl", rows)
    with gzip.open(ev / "samples" / "std_L10.jsonl.gz", "wt") as f:
        f.write(json.dumps({"body": " ; ".join(["x"] * 10) + " ; QED"}) + "\n")
        f.write(json.dumps({"body": "x ; QED"}) + "\n")
    run = rl_dash.load_run(str(run_dir))
    assert run.eval_dir == ev
    e = rl_dash.load_eval(ev)
    assert e["pass1"][10] == 0.1 and e["long_enough"][10] == 0.5
    assert e["hazard"][10][4] == pytest.approx(150 / 151)  # 150 fail at line 4, 151 reach it


def test_flatten_record() -> None:
    rec = {
        "phase": "sample",
        "step": 3,
        "t": 1.0,
        "round": 2,
        "pass_at": {"1": 0.1, "4": 0.2},
        "fail_reasons": [["bad box cite", 7]],
        "stopped_early": False,
        "name": "x",
    }
    flat = rl_dash.flatten_record(rec)
    assert flat == {
        "sample/round": 2,
        "sample/pass_at.1": 0.1,
        "sample/pass_at.4": 0.2,
        "sample/fail/bad box cite": 7,
        "sample/stopped_early": 0,
    }


def test_wandb_dry_run_uploads_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = rl_dash.load_run(str(_new_run(tmp_path / "r")))
    rl_dash.wandb_sync([run], "nd-rl-hillclimb", None, "dry-run")
    assert "nothing uploaded" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rl_dash.main(["wandb", str(tmp_path / "r"), "--mode", "online"])


def test_ticks() -> None:
    assert rl_dash.nice_ticks(0, 1) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(t == int(t) for t in rl_dash.nice_ticks(1, 3, 7, integer=True))
    assert rl_dash.tick_fmt(0.25, 0.25, "pct") == "25%"
    assert rl_dash.fmt(None) == "missing"
