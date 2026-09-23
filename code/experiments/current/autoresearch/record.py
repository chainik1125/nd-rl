#!/usr/bin/env python3
"""Post-run record: results/<NNN>_<slug>/results.json from
artifacts/autoresearch/<NNN>_<slug>/summary.json.

  python3 record.py <NNN>_<slug> keep|discard|crash|timeout "surprise text"

Validates the pre-registered params.json first, then checks that the harness sha, pretrain sha
and budget that actually ran equal what was registered (otherwise the run is not the experiment
that was pre-registered). Copies the loss curve, per-round RL numbers and the eval histograms
into git. Applies keep/discard, refreshes RESULTS.md, and commits/pushes only this outcome.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import table
from params import HERE, check, commit_and_push, git, registration_files, sha
from schema import EvalStats, PretrainStats, Results, RlRound, Status

ART = HERE.parents[3] / "artifacts" / "autoresearch"


def results_from_summary(s: dict[str, Any], status: Status, surprise: str | None) -> Results:
    pre, rl, ev = s.get("pretrain", {}), s.get("rl", {}), s.get("eval", {})
    rounds: list[dict[str, Any]] = rl.get("rounds", [])
    r = Results(
        commit=git("rev-parse", "--short", "HEAD").strip(),
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        status=status,
        metric=s["metric"],
        metric_rl=s.get("metric_rl") or (rounds[-1]["targets_solved"] if rounds else None),
        harness_sha=s["harness_sha"],
        pretrain_sha=s["pretrain_sha"],
        rules=s["rules"],
        gpu=s.get("gpu"),
        total_secs=round(s["total_secs"]),
        reason=s.get("reason"),
        pretrain=PretrainStats(**{**pre, "secs": round(pre["secs"], 1) if "secs" in pre else None}),
        rl_rounds=[
            RlRound(
                **{k: v for k, v in rd.items() if k not in ("sampling", "finetune", "secs")},
                samples_per_sec=round(rd["sampling"]["samples_per_sec"]),
                secs=round(rd["secs"]),
            )
            for rd in rounds
        ],
        eval=EvalStats(
            **{
                k: ev[k]
                for k in ("solved", "n", "ge", "lstar", "heldout_greedy", "max_pruned", "by_bin")
                if k in ev
            }
        ),
        holdout_metric=None,
        surprise=surprise,
        seed_metrics=s.get("seed_metrics"),  # present when run_idea.py merged several seeds
        seed_metric_rl=s.get("seed_metric_rl"),
        seed_status=s.get("seed_status"),
    )
    if status in ("keep", "discard") and r.pretrain.curve is None:
        print(
            "WARNING: pretrain stats have no `curve`; pretrain.py must return one",
            file=sys.stderr,
        )
    return r


def main() -> None:
    name, status_arg = sys.argv[1], sys.argv[2]
    surprise = sys.argv[3] if len(sys.argv) > 3 else None
    assert status_arg in ("keep", "discard", "crash", "timeout"), status_arg
    status: Status = status_arg
    p = check(name)
    registration_files(name)
    s: dict[str, Any] = json.loads((ART / name / "summary.json").read_text())
    if s["status"] != "ok":
        assert status in ("crash", "timeout"), f"harness status {s['status']}"
    for k in ("harness_sha", "pretrain_sha"):
        assert s[k] == getattr(p, k), f"{k}: ran {s[k]} but registered {getattr(p, k)}"
    b, ran = p.protocol.budget, s["rules"]
    assert (ran["TOTAL_SECS"], ran["PRETRAIN_SECS"], ran["PARAM_CAP"]) == (
        b.total_secs,
        b.pretrain_secs,
        b.param_cap,
    ), f"budget ran {ran} but registered {b}"
    pretrain = HERE / "pretrain.py"
    baseline = git("show", f"{p.base_commit}:./pretrain.py")
    current = pretrain.read_text()
    if sha(pretrain) != p.pretrain_sha and not (status != "keep" and current == baseline):
        raise SystemExit(
            "pretrain.py changed since registration; refusing to commit or overwrite it"
        )
    committed_pretrain = git("show", "HEAD:./pretrain.py")
    # The registered candidate itself may have been committed after registration (e.g. a manual
    # commit before a resume); keep/discard then safely applies to it.
    committed_candidate = sha(pretrain) == p.pretrain_sha and committed_pretrain == current
    if committed_pretrain != baseline and not committed_candidate:
        raise SystemExit(
            "committed pretrain.py changed since registration; refusing to overwrite it"
        )
    r = results_from_summary(s, status, surprise)
    result_path = HERE / "results" / name / "results.json"
    if result_path.exists():
        recorded = Results.model_validate_json(result_path.read_text())
        fields = {"commit", "recorded_at"}
        if recorded.model_dump(exclude=fields) != r.model_dump(exclude=fields):
            raise SystemExit(
                "results.json already records a different outcome; refusing to overwrite"
            )
        r = recorded  # Retrying a failed commit/push must not rewrite the evidence/timestamp.
    else:
        if git("status", "--porcelain", "--", "RESULTS.md").strip():
            raise SystemExit("RESULTS.md has pending changes; refusing to overwrite them")
        result_path.write_text(r.model_dump_json(indent=2) + "\n")
    table.main()
    if status != "keep":
        git("restore", f"--source={p.base_commit}", "--staged", "--worktree", "--", "pretrain.py")
    commit = commit_and_push(
        [f"results/{name}/results.json", "RESULTS.md", "pretrain.py"],
        f"autoresearch {name.replace('_', ' ', 1)}: {status}, {r.metric}",
    )
    if session := os.environ.get("ND_AUTORESEARCH_SESSION"):
        receipt = Path(session) / "completed.json"
        pending = receipt.with_suffix(".tmp")
        pending.write_text(json.dumps({"name": name, "commit": commit}))
        pending.replace(receipt)  # The loop advances only after both Git operations succeed.
    print(
        json.dumps(
            {
                "status": r.status,
                "metric": r.metric,
                "metric_rl": r.metric_rl,
                "total_secs": r.total_secs,
                "pretrain_secs": r.pretrain.secs,
                "greedy": r.pretrain.heldout_greedy,
                "r1_solved": r.rl_rounds[0].targets_solved if r.rl_rounds else None,
            }
        )
    )


if __name__ == "__main__":
    main()
