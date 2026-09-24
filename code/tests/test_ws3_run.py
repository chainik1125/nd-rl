"""run.py end to end on the smoke config: every stage runs and writes what the plan requires."""

import json
import os
import shutil

import ndpath
import pytest

# isort: split
import run as ws3_run
from metrics_callback import REQUIRED_METRICS

HAVE = os.path.exists(ndpath.FORK_CKPT) and os.path.exists(os.path.join(ndpath.FORK_DATA, "heldout.jsonl"))


@pytest.mark.skipif(not HAVE, reason="fork_pin data or checkpoint not fetched")
def test_smoke_all_stages(tmp_path):
    src = os.path.join(ndpath.WS3, "configs", "smoke_cpu.yaml")
    cfg_path = tmp_path / "config.yaml"
    text = open(src).read().replace("out_dir: ../../artifacts/ws3/smoke_cpu", "out_dir: run")
    cfg_path.write_text(text)
    ws3_run.main(["--config", str(cfg_path)])
    out = tmp_path / "run"
    # baseline: T1 gate numbers and the three evaluations, each with a dump
    b = json.load(open(out / "baseline" / "summary.json"))
    assert 0 <= b["nta"]["next_token_accuracy"] <= 1 and b["nta"]["n_records"] == 200
    for part in ("greedy_heldout", "textbook", "long_pool"):
        assert (out / "baseline" / f"{part}.json").exists()
    assert b["textbook"]["by_slice"]["all"]["n"] == 72
    assert b["textbook"]["by_slice"]["dev58"]["n"] == 58 and b["textbook"]["by_slice"]["train14"]["n"] == 14
    assert sum(1 for _ in open(out / "baseline" / "dumps" / "textbook72.jsonl")) == 144
    # train: metrics.jsonl has every plan section 4 metric, one line per step, utc + step
    rows = [json.loads(l) for l in open(out / "metrics.jsonl") if l.strip()]
    cov = ws3_run.metric_coverage(rows)
    assert all(cov.values()), {k: v for k, v in cov.items() if not v}
    roll = [r for r in rows if "reward" in r]
    assert [r["step"] for r in roll] == [1, 2, 3]
    assert all("utc" in r and "rollout/completions_per_sec" in r for r in roll)
    assert sum(1 for _ in open(out / "completions.jsonl")) == 3 * 8
    assert (out / "prompts.jsonl").exists() and (out / "run_manifest.json").exists()
    assert (out / "final" / "config.json").exists()
    # evaluate: same evaluations on the trained model plus the comparison
    comp = json.load(open(out / "evaluate" / "comparison.json"))
    assert set(comp) >= {"greedy_heldout", "textbook", "long_pool"}
    assert comp["textbook"]["by_slice"]["control"]["all"]["n"] == 72
    # report: graphs and tables
    assert (out / "report" / "curves.png").exists()
    assert (out / "report" / "greedy_heldout.png").exists()
    md = (out / "report" / "report.md").read_text()
    assert "MISSING" not in md and "textbook 72" in md
    for k in REQUIRED_METRICS:
        assert k in md
