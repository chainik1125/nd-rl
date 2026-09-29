"""Materialize the pre-outcome primary training and evaluation contract once."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    files = [ROOT / "code" / name for name in
        ("modeling.py", "train.py", "protocol.py", "evaluate.py", "analyze.py", "pipeline.py")]
    files += [ROOT / "data" / name for name in
        ("manifest.json", "shared_cap6.jsonl", "rl_short_prompts.jsonl", "training_exclusions.json",
         "eval_short_panel.jsonl", "eval_long_panel.jsonl")]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    save(ROOT / "plans/primary_source_freeze.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashes, "no_held_out_outcomes_observed": True})
    stages = [
        ("sft", ["sft", "--out", "runs/sft", "--steps", "80", "--accumulation", "4", "--learning-rate", "2e-5"]),
        ("pairs", ["pairs", "--adapter", "runs/sft/final", "--out", "runs/pairs", "--pair-batch-size", "16", "--max-pair-attempts", "4"]),
        ("dpo", ["dpo", "--adapter", "runs/sft/final", "--pairs", "runs/pairs/pairs.jsonl", "--out", "runs/dpo", "--steps", "40", "--accumulation", "4", "--learning-rate", "5e-6"]),
        ("grpo", ["grpo", "--adapter", "runs/dpo/final", "--out", "runs/grpo", "--steps", "40", "--accumulation", "16", "--group-size", "4", "--learning-rate", "1e-5"]),
    ]
    steps = []
    for name, args in stages:
        args += ["--rank", "16", "--max-length", "4096", "--max-completion-length", "2048", "--seed", "202609092014", "--deadline-utc", "2026-09-10T05:14:00+00:00"]
        requirement = {"shared_pool": 32} if name == "pairs" else {"status": "complete", "mechanical_fixture": False}
        steps.append({"name": name, "entrypoint": "train.py", "args": args,
            "required_records": [{"path": str(ROOT / "runs" / name / "summary.json"), "equals": requirement}]})
    save(ROOT / "plans/primary_training.json", {
        "source_freeze": str(ROOT / "plans/primary_source_freeze.json"),
        "out": str(ROOT / "runs/training_pipeline"), "not_after_utc": "2026-09-10T05:14:00+00:00", "steps": steps,
        "selection_rule": "Fixed final optimizer step; no held-out outcome checkpoint selection",
        "pair_coverage_rule": "At most four attempts per shared theorem; use every available first verifier-invalid sample. Report absent pairs and family coverage; do not fabricate negatives.",
        "rl_signal_rule": "Complete the fixed40updates even if groups starve; report real nonzero gradient/weight-change counts and no-gain honestly. Stop on nonfinite gradients or failed likelihood audit.",
    })
    save(ROOT / "plans/primary_training_job.json", {
        "job_dir": str(ROOT / "jobs/primary_training"), "entrypoint": "pipeline.py", "candidate_gpus": [3, 4],
        "queue_timeout_seconds": 10800, "timeout_seconds": 18000, "not_after_utc": "2026-09-10T05:14:00+00:00",
        "wait_for_success": [
            {"path": str(ROOT / "jobs/mechanical_smoke32b_v2/completed.json"), "equals": {"exit_code": 0}},
            {"path": str(ROOT / "runs/mechanical_smoke32b_v2/summary.json"), "equals": {"passed": True, "mechanical_only": True}},
        ],
        "args": ["--plan", str(ROOT / "plans/primary_training.json")],
    })
    save(ROOT / "plans/primary_evaluation_contract.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "x": 6, "cohorts": {"short_panel": 56, "long_panel": 64},
        "arms": ["base_zero", "base_icl32", "sft", "dpo", "rl"],
        "modes": ["greedy", "sampled"], "sampled_k": 4, "max_new_tokens": 2048,
        "temperature_sampled": 1.0, "top_p": 1.0, "top_k": 0, "seed": 202609092015,
        "batch_size": {"base_zero": 16, "base_icl32": 4, "sft": 16, "dpo": 16, "rl": 16},
        "stop": "Actually generated full closing marker, native EOS, or cap; literal tokens, no repair",
        "selection_basis": "Training-only capacity: batch4 8.2k-token ICL fits, batch16 shortprompt gives69.49native tokens/sec. No held-out outcomes read.",
        "priority": "Complete matched greedy120targets for all5arms before sampled4; report incomplete secondary sampling rather than replace targets or alter budgets based on outcomes.",
        "textbook": "Development only after completed training; final only after global freeze and05:14UTC. Fixed greedy-only matched panel has priority over additional sampling.",
    })
    print(json.dumps({"frozen_files": len(hashes), "training_stages": [name for name, _ in stages], "status": "frozen"}))


if __name__ == "__main__":
    main()
