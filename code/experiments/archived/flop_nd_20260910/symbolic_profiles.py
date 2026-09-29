"""Frozen development profiles, endpoint release, and exact LM token audits.

This wrapper does not alter the frozen proposer, features, or optimizer. It
launches the already frozen evaluator and adds audit-only labels/token counts.
"""
import argparse
from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import common

ROOT = common.ROOT
PLAN = ROOT / "reports/symbolic_development_profiles.json"
RELEASE = ROOT / "reports/symbolic_development_release.json"
LEDGER = ROOT / "results/symbolic_development_ledger.jsonl"


def weight_paths():
    weights = {}
    for seed in (4301, 4302):
        for mode in ("full", "reset", "bc"):
            tag = f"{mode}_s{seed}"
            weights[tag] = str(ROOT / f"results/symbolic_general_{tag}/symbolic_general_checkpoint.pt")
        weights[f"initial_s{seed}"] = str(ROOT / f"results/symbolic_general_full_s{seed}/symbolic_general_initial.pt")
    return weights


def prepare():
    cohorts = {"general_development": ("data/v1/development.jsonl", "data/v1/audit/development.jsonl"),
               "structured_development": ("data/structured_lm_v1/development.jsonl", "data/structured_lm_v1/audit/development.jsonl")}
    cohort_rows = {}
    for tag, (data, audit) in cohorts.items():
        cohort_rows[tag] = {"data": str(ROOT / data), "audit": str(ROOT / audit),
                            "data_sha256": common.sha(ROOT / data), "audit_sha256": common.sha(ROOT / audit)}
    proposal = {"cohorts": cohort_rows, "weights": weight_paths(), "seed": 202609101935,
        "profiles": [{"name": "greedy", "attempts": 1, "sample": False, "temperature": 0},
                     {"name": "sample4", "attempts": 4, "sample": True, "temperature": 1}],
        "uniform": {"attempts": 4, "first_attempt_profile_derived": True},
        "max_lines": 24, "max_scope": 4, "lm_response_token_cap": 1024,
        "lm_token_convention": "Exact frozen relative body codec including QED, plus one FINAL token; raw certificate must fit",
        "seed_convention": "Same declared seed for every evaluator process; native sequential RNG, not matched LM token random draws",
        "sample_temperature_note": "Symbolic native temperature1 differs from LM0.8; distributions and action spaces are different",
        "candidate_support": "Unchanged frozen general proposer; no structured training for any policy",
        "policy_code_sha256": common.sha(ROOT / "code/symbolic_general_eval.py"),
        "wrapper_code_sha256": common.sha(__file__), "final_access": False}
    if PLAN.exists():
        previous = json.loads(PLAN.read_text())
        for key, value in proposal.items():
            if previous[key] != value:
                raise ValueError(f"Development profile freeze changed: {key}")
    else:
        common.write_json(PLAN, {**proposal, "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return json.loads(PLAN.read_text())


def release_if_ready(plan):
    for name, path in plan["weights"].items():
        if not Path(path).exists():
            return False
        if not name.startswith("initial"):
            summary = json.loads(Path(path).with_name("symbolic_general_train_summary.json").read_text())
            if summary["status"] != "complete":
                return False
    import torch
    for seed in (4301, 4302):
        initials = [torch.load(ROOT / f"results/symbolic_general_{mode}_s{seed}/symbolic_general_initial.pt",
                     weights_only=False, map_location="cpu")["state_dict"] for mode in ("full", "reset", "bc")]
        assert all(torch.equal(initials[0][key], state[key]) for state in initials[1:] for key in initials[0])
    payload = {"status": "released_development_only", "weights": {k: common.sha(v) for k, v in plan["weights"].items()},
        "profile_sha256": common.sha(PLAN), "initial_tensor_identity": "Full/reset/BC initial tensors match exactly per seed",
        "release_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "final_released": False}
    if RELEASE.exists():
        old = json.loads(RELEASE.read_text())
        assert old["weights"] == payload["weights"] and old["profile_sha256"] == payload["profile_sha256"]
    else:
        common.write_json(RELEASE, payload)
    return True


def annotate(out, cohort, cap):
    from nd_environment import Tokenizer
    from nd_verify import parse_proof_tokens
    tokenizer = Tokenizer()
    audit = {x["source_id"]: x for x in common.read_jsonl(cohort["audit"])}
    rows = common.read_jsonl(out / "symbolic_general_attempts.jsonl")
    grouped = defaultdict(lambda: defaultdict(int))
    success_ids, compatible_ids = set(), set()
    enriched = []
    for row in rows:
        metadata = audit[row["source_id"]]
        assert row["prompt"] == metadata["prompt"]
        row["family"] = metadata["family"]
        row["reference_lines"] = metadata["n_lines"]
        row["reference_length_bin"] = metadata.get("length_bin", str(metadata["n_lines"]))
        prefix_len = len(tokenizer.encode_prompt(row["prompt"]))
        row["raw_relative_response_tokens"] = 1 + len(tokenizer.encode_record(row["prompt"] + " " + row["body"])) - prefix_len
        row["pruned_relative_response_tokens"] = (1 + len(tokenizer.encode_record(row["prompt"] + " " + row["pruned_body"])) - prefix_len) if row["success"] else None
        row["raw_fits_lm_token_cap"] = row["raw_relative_response_tokens"] <= cap
        row["lm_compatible_success"] = row["success"] and row["raw_fits_lm_token_cap"]
        if row["success"]:
            proof = parse_proof_tokens(row["pruned_body"].split())
            row["pruned_rules"] = [line["rule"] for line in proof]
            row["pruned_max_scope"] = max(line["depth"] for line in proof)
            success_ids.add(row["source_id"])
            if row["lm_compatible_success"]:
                compatible_ids.add(row["source_id"])
        else:
            row["pruned_rules"], row["pruned_max_scope"] = [], None
        for key in ["all", f"family/{row['family']}", f"reference_lines/{row['reference_lines']}"]:
            values = grouped[key]
            values["attempts"] += 1
            values["successes"] += int(row["success"])
            values["lm_compatible_successes"] += int(row["lm_compatible_success"])
            values["raw_relative_response_tokens"] += row["raw_relative_response_tokens"]
            values["proofs_using_ore"] += int("ORE" in row["pruned_rules"])
            values["proofs_using_impi"] += int("IMPI" in row["pruned_rules"])
            values["pruned_over_six"] += int(row["success"] and row["pruned_verdict"]["n_lines"] > 6)
        enriched.append(row)
    common.write_jsonl(out / "symbolic_profile_attempts.jsonl", enriched)
    common.write_json(out / "symbolic_profile_summary.json", {"groups": {k: dict(v) for k, v in grouped.items()},
       "theorems": len({r["source_id"] for r in rows}), "theorems_with_any_success": len(success_ids),
       "theorems_with_any_lm_compatible_success": len(compatible_ids), "lm_token_cap": cap,
       "audit_sha256": cohort["audit_sha256"], "raw_attempts_sha256": common.sha(out / "symbolic_general_attempts.jsonl"),
       "annotated_attempts_sha256": common.sha(out / "symbolic_profile_attempts.jsonl")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--launch", action="store_true")
    p.add_argument("--wait-endpoints", action="store_true")
    args = p.parse_args()
    plan = prepare()
    print("Development profiles frozen:", PLAN, flush=True)
    if not args.launch:
        return
    deadline = datetime.datetime.fromisoformat("2026-09-11T03:40:00+00:00").timestamp()
    while not release_if_ready(plan):
        if not args.wait_endpoints:
            raise RuntimeError("All declared endpoints must exist before development evaluation")
        if time.time() >= deadline:
            raise RuntimeError("Development launch deadline reached")
        time.sleep(30)
    process_env = os.environ.copy()
    process_env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    jobs = []
    for cohort_tag, cohort in plan["cohorts"].items():
        for tag, checkpoint in plan["weights"].items():
            for profile in plan["profiles"]:
                jobs.append((cohort_tag, cohort, tag, checkpoint, profile))
        jobs.append((cohort_tag, cohort, "uniform", None, {"name": "sample4", "attempts": 4, "sample": True}))
    for cohort_tag, cohort, tag, checkpoint, profile in jobs:
        job = f"{cohort_tag}_{tag}_{profile['name']}"
        out = ROOT / f"results/symbolic_profile_{job}"
        if (out / "symbolic_profile_summary.json").exists():
            continue
        if common.sha(cohort["data"]) != cohort["data_sha256"] or common.sha(cohort["audit"]) != cohort["audit_sha256"]:
            raise RuntimeError("Frozen development cohort changed")
        command = [sys.executable, str(ROOT / "code/symbolic_general_eval.py"), "--population", cohort["data"],
            "--out", str(out), "--policy", "model" if checkpoint else "random", "--attempts", str(profile["attempts"]),
            "--seed", str(plan["seed"]), "--max-lines", str(plan["max_lines"]), "--max-scope", str(plan["max_scope"]),
            "--threads", "4", "--max-seconds", "1800"]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        if profile["sample"]:
            command += ["--sample"]
        out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with (out / "symbolic_profile.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=process_env)
            with LEDGER.open("a") as f:
                f.write(json.dumps({"event": "start", "job": job, "pid": proc.pid, "driver_pid": os.getpid(),
                    "command": command, "start_unix": started, "profile_sha256": common.sha(PLAN), "release_sha256": common.sha(RELEASE)}) + "\n")
            print("Started", job, "pid", proc.pid, flush=True)
            try:
                code = proc.wait(timeout=min(1860, deadline - started))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                raise
        with LEDGER.open("a") as f:
            f.write(json.dumps({"event": "finish", "job": job, "exit_code": code, "elapsed_seconds": time.time() - started}) + "\n")
        if code:
            raise RuntimeError(f"Development evaluation failed: {job}")
        annotate(out, cohort, plan["lm_response_token_cap"])
        print("Finished", job, "seconds", round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    main()
