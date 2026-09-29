"""Prepare the outcome-blind v4 continuation; never launch or open held-out data.

Root-owned operation, intended only after capacity and mechanical gates finish.
No existing artifact is rewritten. Publication preflights every output and input,
then writes the new source freeze last as the bundle's commit marker. An I/O or
concurrent-change failure can leave new uncommitted plans, never a fabricated
successful old pipeline. Those files require explicit root review, not overwrite.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from continuation_prefix import Receipts, require, validate_prefix

SPRINT = Path(__file__).resolve().parents[1]
DEADLINE = "2026-09-10T05:14:00+00:00"
APPROVED = {
    "code/modeling.py": "a8da6e3fc1d8b4223fb4f4ece8285430c18101124476434b983b55afbca37056",
    "code/train.py": "70622fa9bd7ab0b56beaa1f8fc823a955ab64ca1952225d5a28c412d9446322b",
    "code/launch.py": "f3008ad72910df217cd1ff475e8bceb71becf69a5853b416fcc73f415ee9b053",
}
NUMERICS_SHA = "1f8fb573de162acb84eebe8b82723fb1acd2dc19a474a20d7d16668635a6ea37"
PREFIX_VALIDATOR_SHA = "bd258f9b488c229b0bfd4ca85670854fdade522a045967f46cf343adaa0aaa98"
POLICY_ID = "nf4_fp32_compute_expanded_gqa_efficient_v4"
ARMS = ["base_zero", "base_icl32", "sft", "dpo", "rl"]
OUTPUT_NAMES = (
    "continuation_prefix_v4.json", "primary_training_v4.json", "primary_training_job_v4.json",
    "eval_greedy_v4.json", "eval_greedy_job_v4.json", "eval_sampled_v4.json", "eval_sampled_job_v4.json",
    "primary_source_freeze_v4.json",
)
DIAGNOSTICS = (
    "likelihood_replay_simplex2_v3", "likelihood_shapes_v3", "precision_probe_v3",
    "precision_probe_full_v3", "all_fp32_efficient_probe_v3",
)
GATES = ("capacity32b_v4", "mechanical_smoke32b_v4")
GRPO_ARGS = ["grpo", "--adapter", "runs/dpo/final", "--out", "runs/grpo", "--steps", "40",
             "--accumulation", "16", "--group-size", "4", "--learning-rate", "1e-5",
             "--rank", "16", "--max-length", "4096", "--max-completion-length", "2048",
             "--seed", "202609092014", "--deadline-utc", DEADLINE]


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fresh_paths(sprint):
    paths = [sprint / "plans" / name for name in OUTPUT_NAMES]
    paths += [sprint / "runs" / name for name in
              ("grpo_v4", "training_continuation_v4", "eval_greedy_v4", "eval_sampled_v4",
               "eval_greedy_pipeline_v4", "eval_sampled_pipeline_v4")]
    paths += [sprint / "jobs" / name for name in ("primary_training_v4", "eval_greedy_v4", "eval_sampled_v4")]
    return paths


def require_fresh(sprint):
    require((sprint / "plans").is_dir(), "Existing sprint plans directory required")
    for path in fresh_paths(sprint):
        require(not os.path.lexists(path), f"New v4 target already exists: {path}")


def option(args, flag):
    require(args.count(flag) == 1, f"Expected unique option {flag}")
    return args[args.index(flag) + 1]


def replace_option(args, flag, old, new):
    require(option(args, flag) == old, f"Unexpected original value of {flag}")
    args[args.index(flag) + 1] = new


def finite_below(value, ceiling):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= ceiling


def check_job(sprint, receipts, name):
    plan = receipts.json(sprint / "plans" / (name + ".json"))
    job = sprint / "jobs" / name
    require(receipts.json(job / "spec.json") == plan, f"{name} launched a different spec")
    completed = receipts.json(job / "completed.json")
    require(type(completed.get("exit_code")) is int and completed["exit_code"] == 0
            and completed.get("gpu") == 3, f"{name} did not exit successfully on GPU3")
    require(plan["candidate_gpus"] == [3] and plan["execution_host"] == "g374"
            and plan["not_after_utc"] == DEADLINE, f"{name} has different host/GPU/deadline")
    return plan


def check_gates(sprint, receipts):
    capacity_plan = check_job(sprint, receipts, "capacity32b_v4")
    args = capacity_plan["args"]
    require(capacity_plan["entrypoint"] == "capacity.py" and
            option(args, "--new-tokens") == "2048" and option(args, "--batch-size") == "4" and
            option(args, "--icl-demonstrations") == "32" and "--skip-update" in args and
            option(args, "--data") == str(sprint / "data/shared_cap6.jsonl"), "Capacity gate changed scope")
    capacity = receipts.json(sprint / "runs/capacity32b_v4/completed.json")
    require(capacity.get("status") == "completed", "Full v4 ICL capacity gate remains incomplete")
    loaded = [row for row in capacity["events"] if row["event"] == "loaded"]
    generated = [row for row in capacity["events"] if row["event"] == "icl_generation"]
    require(len(loaded) == len(generated) == 1 and generated[0]["samples"] == 4
            and generated[0]["max_new_tokens"] == 2048, "Capacity generation evidence is incomplete")
    policy = loaded[0]["identity"]["numerical_policy"]
    require(policy["policy_id"] == POLICY_ID and policy["source_sha256"] == NUMERICS_SHA,
            "Capacity used a different numerical runtime")
    smoke_plan = check_job(sprint, receipts, "mechanical_smoke32b_v4")
    require(smoke_plan["entrypoint"] == "train.py" and smoke_plan["args"][0] == "smoke", "Wrong mechanical entrypoint")
    smoke_dir = sprint / "runs/mechanical_smoke32b_v4"
    smoke = receipts.json(smoke_dir / "summary.json")
    run = receipts.json(smoke_dir / "run.json")
    require(run["mechanical_fixture"] is True and run["args"]["stage"] == "smoke"
            and run["source_sha256"] == {Path(name).name: digest for name, digest in APPROVED.items()
                                         if name in {"code/modeling.py", "code/train.py"}}, "Smoke source identity mismatch")
    require(smoke.get("passed") is True and smoke.get("mechanical_only") is True
            and smoke.get("not_a_research_checkpoint") is True, "Fresh v4 mechanical gate did not pass")
    require(finite_below(smoke["reload_max_abs_error"], .1) and finite_below(smoke["behavior_max_abs_error"], .1),
            "Mechanical likelihood/reload audit exceeds unchanged tolerance")
    for stage in ("sft", "dpo", "grpo"):
        result = smoke[stage]
        require(result == receipts.json(smoke_dir / stage / "summary.json"), "Mechanical nested summary differs")
        require(result["mechanical_fixture"] is True and result["status"] == "complete"
                and result["optimizer_steps"] == result["requested_steps"] == 1
                and result["nonzero_gradient_steps"] > 0 and result["observed_parameter_changes"] > 0,
                "Mechanical stage lacks its real optimizer/change evidence")
        weight = smoke_dir / stage / "final/adapter_model.safetensors"
        audit = result["saved_tensor_audit"]
        require(audit["all_tensors_exact"] is True and audit["dtype"] == "float32"
                and audit["adapter_name"] == "default" and Path(audit["checkpoint"]).resolve() == weight
                and audit["sha256"] == receipts.digest(weight), "Mechanical saved checkpoint audit mismatch")
    audit_events = receipts.rows(smoke_dir / "grpo/events.jsonl")
    for kind in ("actual_grpo_likelihood_audit", "actual_grpo_gradient_likelihood_audit"):
        audits = [row for row in audit_events if row.get("kind") == kind]
        require(audits and all(row["tolerance"] == .1 and finite_below(row["max_abs_error"], .1)
                              and row["denominator"] == "actual_streamed_sampling_logprobs"
                              and (kind != "actual_grpo_gradient_likelihood_audit" or row["scorer_requires_grad"] is True)
                              for row in audits),
                f"Missing or failed fresh mechanical guard: {kind}")
    return {"capacity": {"path": str(sprint / "runs/capacity32b_v4/completed.json"), "status": "completed"},
            "mechanical": {"path": str(smoke_dir / "summary.json"), "passed": True, "mechanical_only": True}}


def check_diagnostic(sprint, receipts):
    name = "all_fp32_efficient_probe_v3"
    check_job(sprint, receipts, name)
    record = receipts.json(sprint / "runs" / name / "summary.json")
    require(record["status"] == "complete" and record["optimizer_steps"] == 0
            and record["adapter_before"] == record["adapter_after"]
            and record["passes_existing_guard_on_recorded_actions"] is True
            and record["frozen_tolerance_changed"] is False, "Recorded-action diagnostic did not pass unchanged guard")
    errors, gradient = record["same_candidate_cached_vs_teacher"], record["gradient_audit"]
    require(errors["frozen_tolerance"] == .1 and finite_below(errors["max_abs_error"], .1)
            and errors["tokens"] == 1805 and errors["tokens_above_frozen_tolerance"] == 0,
            "Recorded-action likelihood evidence does not cover the failed prefix")
    require(gradient["backward_exercised"] is True and gradient["gradient_enabled"] is True
            and gradient["optimizer_steps"] == 0 and gradient["tensors_with_gradients"] == 896
            and finite_below(gradient["gradient_norm"], 1e30) and gradient["gradient_norm"] > 0,
            "Recorded-action diagnostic lacks finite differentiable policy evidence")


def inventory(sprint):
    """Only known code/plan/diagnostic roots; never discover evaluation payloads."""
    paths = set()
    for directory in (sprint / "code", sprint / "review_tests", sprint / "code/archive_v2", sprint / "code/archive_v3"):
        paths.update(path for path in directory.glob("*.py")
                     if path.name != "make_textbook_release_v4.py"
                     and not path.name.startswith("test_textbook_release_v4"))
    paths.update(path for path in (sprint / "plans").glob("*.json") if path.name not in OUTPUT_NAMES)
    for name in (*DIAGNOSTICS, *GATES):
        for parent in (sprint / "runs" / name, sprint / "jobs" / name):
            require(parent.is_dir(), f"Missing diagnostic/gate directory: {parent}")
            paths.update(path for path in parent.glob("*") if path.is_file() and path.suffix in {".json", ".jsonl", ".log"})
    for stage in ("sft", "dpo", "grpo"):
        parent = sprint / "runs/mechanical_smoke32b_v4" / stage
        paths.update(path for path in parent.glob("*") if path.is_file() and path.suffix in {".json", ".jsonl"})
        paths.update(path for path in (parent / "final").glob("adapter*") if path.is_file())
    for name in ("runtime_v4_amendment.md", "precision_probe_design.md", "efficient_precision_probe.md"):
        paths.add(sprint / "reports" / name)
    # Optional protected-release tooling is independently developed and is not a
    # prerequisite for the primary synthetic pipeline. Its eventual release must
    # bind its own final implementation/tests plus this completed training chain.
    return {path.resolve() for path in paths}


def job(sprint, name, plan_name, dependencies):
    return {"job_dir": str(sprint / "jobs" / name), "entrypoint": "pipeline.py",
            "args": ["--plan", str(sprint / "plans" / plan_name)], "candidate_gpus": [3],
            "execution_host": "g374", "execution_alias": "simplex2", "storage_host": "g178", "paid_spend": 0,
            "queue_timeout_seconds": 28800, "timeout_seconds": 18000, "not_after_utc": DEADLINE,
            "wait_for_success": dependencies,
            "launch_contract": "Root must execute this spec on simplex2 (g374); launcher enforces one GPU and the absolute deadline."}


def dependency(sprint, relative, **equals):
    return {"path": str(sprint / relative), "equals": equals}


def construct_plans(sprint, prefix, old_training, old_evals, created):
    """Pure metadata transform; output bytes can be inspected/tested before writes."""
    prefix = dict(prefix, created_utc=created)
    prefix_name = "continuation_prefix_v4.json"
    prefix_ref = {"path": str(sprint / "plans" / prefix_name), "sha256": digest_bytes(encoded(prefix))}
    freeze_path = str(sprint / "plans/primary_source_freeze_v4.json")
    old_step = old_training["steps"][-1]
    require(old_step["name"] == "grpo" and old_step["entrypoint"] == "train.py" and old_step["args"] == GRPO_ARGS,
            "Original GRPO schedule differs from the approved forty-step continuation")
    step = deepcopy(old_step)
    replace_option(step["args"], "--out", "runs/grpo", "runs/grpo_v4")
    step["required_records"] = [dependency(sprint, "runs/grpo_v4/summary.json", status="complete",
        mechanical_fixture=False, stage="grpo", optimizer_steps=40, requested_steps=40, deadline_reached=False)]
    training = deepcopy(old_training)
    training.update(source_freeze=freeze_path, out=str(sprint / "runs/training_continuation_v4"), steps=[step],
        not_after_utc=DEADLINE, prefix_receipt=prefix_ref, approved_source_changes=APPROVED,
        numerical_helper={"path": str(sprint / "code/numerics.py"), "sha256": NUMERICS_SHA},
        continuation_kind="restart_grpo_from_exact_successful_v3_dpo_endpoint_no_failed_optimizer_state",
        original_failed_pipeline=str(sprint / "runs/training_pipeline_v3"),
        starting_checkpoint_files=prefix["checkpoint_files"],
        successful_prefix_stages=["sft", "pairs", "dpo"], old_failed_grpo_optimizer_steps=0,
        evaluation_runtime_rule="Apply the same approved v4 numerical preparation to every subsequent evaluation arm",
        no_held_out_outcomes_observed=True)
    require(training["not_after_utc"] == old_training["not_after_utc"] == DEADLINE, "Training deadline drift")
    gates = [dependency(sprint, "runs/capacity32b_v4/completed.json", status="completed"),
             dependency(sprint, "jobs/capacity32b_v4/completed.json", exit_code=0),
             dependency(sprint, "runs/mechanical_smoke32b_v4/summary.json", passed=True, mechanical_only=True),
             dependency(sprint, "jobs/mechanical_smoke32b_v4/completed.json", exit_code=0)]
    docs = {prefix_name: prefix, "primary_training_v4.json": training,
            "primary_training_job_v4.json": job(sprint, "primary_training_v4", "primary_training_v4.json", gates)}
    training_done = [dependency(sprint, "jobs/primary_training_v4/completed.json", exit_code=0),
                     dependency(sprint, "runs/training_continuation_v4/completed.json", status="complete", stages=["grpo"]),
                     deepcopy(step["required_records"][0])]
    for mode in ("greedy", "sampled"):
        plan = deepcopy(old_evals[mode])
        require(plan["mode"] == mode and plan["not_after_utc"] == DEADLINE
                and [row["name"] for row in plan["steps"]] == ARMS, "Original evaluation phase identity changed")
        plan.update(source_freeze=freeze_path, out=str(sprint / "runs" / f"eval_{mode}_pipeline_v4"),
                    prefix_receipt=prefix_ref, common_numerical_policy=POLICY_ID)
        for value in plan["steps"]:
            arm, args = value["name"], value["args"]
            # Exact scientific argument order and values: only the run output and
            # final RL adapter differ from the preserved v3 phase specification.
            expected = ["--arm", arm, "--out", str(sprint / "runs" / f"eval_{mode}" / arm),
                        "--cohorts", "long_panel", "short_panel", "--modes", mode,
                        "--max-new-tokens", "2048", "--batch-size", "4" if arm == "base_icl32" else "16",
                        "--seed", "202609092015"]
            if not arm.startswith("base_"):
                expected += ["--adapter-path", str(sprint / "runs" / ("grpo" if arm == "rl" else arm) / "final")]
            require(args == expected and value["entrypoint"] == "evaluate.py", "Original evaluation settings/order changed")
            replace_option(args, "--out", str(sprint / "runs" / f"eval_{mode}" / arm),
                           str(sprint / "runs" / f"eval_{mode}_v4" / arm))
            if arm == "rl":
                replace_option(args, "--adapter-path", str(sprint / "runs/grpo/final"), str(sprint / "runs/grpo_v4/final"))
            value["required_records"] = [dependency(sprint, f"runs/eval_{mode}_v4/{arm}/evaluation_status.json",
                                                   status="complete", attempts=120 if mode == "greedy" else 480)]
        docs[f"eval_{mode}_v4.json"] = plan
        dependencies = deepcopy(training_done)
        if mode == "sampled":
            dependencies += [dependency(sprint, "jobs/eval_greedy_v4/completed.json", exit_code=0),
                dependency(sprint, "runs/eval_greedy_pipeline_v4/completed.json", status="complete", stages=ARMS)]
            dependencies += [dependency(sprint, f"runs/eval_greedy_v4/{arm}/evaluation_status.json", status="complete", attempts=120)
                             for arm in ARMS]
        docs[f"eval_{mode}_job_v4.json"] = job(sprint, f"eval_{mode}_v4", f"eval_{mode}_v4.json", dependencies)
    return docs


def build_bundle(sprint=SPRINT):
    """Read-only until publish_bundle is called; full validation is mandatory."""
    sprint = Path(sprint).resolve()
    require_fresh(sprint)
    require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), "Writing block already started")
    receipts = Receipts()
    for relative, expected in {**APPROVED, "code/numerics.py": NUMERICS_SHA,
                               "code/continuation_prefix.py": PREFIX_VALIDATOR_SHA}.items():
        require(receipts.digest(sprint / relative) == expected, f"Approved implementation changed: {relative}")
    prefix = validate_prefix(sprint, approved_source_changes=APPROVED)
    require(prefix["status"] == "validated" and prefix["v3_pipeline_success"] is False
            and prefix["stage_optimizer_steps"] == {"sft": 80, "dpo": 40, "failed_grpo": 0}
            and len(prefix["sha256"]) == 73, "Unexpected validated prefix or receipt count")
    for path, value in prefix["sha256"].items():
        receipts.remember(path, value)
    gates = check_gates(sprint, receipts)
    check_diagnostic(sprint, receipts)
    old_training = receipts.json(sprint / "plans/primary_training_v3.json")
    old_evals = {mode: receipts.json(sprint / "plans" / f"eval_{mode}_v3.json") for mode in ("greedy", "sampled")}
    observed_inventory = inventory(sprint)
    for path in sorted(observed_inventory):
        receipts.digest(path)
    created = datetime.now(timezone.utc).isoformat()
    docs = construct_plans(sprint, prefix, old_training, old_evals, created)
    payloads = {sprint / "plans" / name: encoded(value) for name, value in docs.items()}
    hashes = dict(receipts.sha256)
    hashes.update({str(path): digest_bytes(raw) for path, raw in payloads.items()})
    freeze_path = sprint / "plans/primary_source_freeze_v4.json"
    require(str(freeze_path) not in hashes, "Source freeze must not contain its own hash")
    freeze = {"version": 4, "utc": created, "sha256": dict(sorted(hashes.items())),
              "supersedes": str(sprint / "plans/primary_source_freeze_v3.json"),
              "prefix_receipt": docs["primary_training_v4.json"]["prefix_receipt"],
              "prefix_receipt_count": 73, "approved_source_changes": APPROVED,
              "numerical_helper_sha256": NUMERICS_SHA, "numerical_policy_id": POLICY_ID,
              "successful_gate_metadata": gates, "no_held_out_outcomes_observed": True,
              "read_protected_textbook_payloads": False, "old_failed_pipeline_preserved": True,
              "publication_rule": "This freeze is written last, after all new plans and stable input checks; never self-hashed"}
    payloads[freeze_path] = encoded(freeze)
    require(set(path.name for path in payloads) == set(OUTPUT_NAMES), "Wrong output bundle membership")
    receipts.verify_stable()
    require(inventory(sprint) == observed_inventory, "Input inventory changed during validation")
    require_fresh(sprint)
    return {"sprint": sprint, "payloads": payloads, "input_sha256": dict(receipts.sha256),
            "inventory": observed_inventory, "freeze_path": freeze_path}


def publish_bundle(bundle):
    """Exclusive-create only; no launching, release creation, or cleanup on failure."""
    sprint, payloads, freeze_path = bundle["sprint"], bundle["payloads"], bundle["freeze_path"]
    require_fresh(sprint)
    check = Receipts()
    for name, expected in bundle["input_sha256"].items():
        require(check.digest(name) == expected, f"Validated input changed before publication: {name}")
    require(inventory(sprint) == bundle["inventory"], "Input inventory changed before publication")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), "Writing block already started")
    written = []
    for path in [*sorted(set(payloads) - {freeze_path}), freeze_path]:
        if path == freeze_path:
            check.verify_stable()
            require(inventory(sprint) == bundle["inventory"], "Input inventory changed before freeze commit")
            require(all(check.digest(value) == digest_bytes(payloads[value]) for value in written), "A newly written plan changed")
        require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), "Writing block began before bundle commit")
        with path.open("xb") as stream:
            stream.write(payloads[path])
            stream.flush()
            os.fsync(stream.fileno())
        written.append(path)
    return {"status": "frozen", "version": 4, "created_files": [str(path) for path in written],
            "source_freeze_sha256": digest_bytes(payloads[freeze_path]),
            "frozen_files": len(json.loads(payloads[freeze_path])["sha256"]), "launched_jobs": 0,
            "old_failed_pipeline_preserved": True, "read_protected_textbook_payloads": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--write", action="store_true", help="Create the fresh plans and commit freeze after every gate passes")
    args = parser.parse_args()
    bundle = build_bundle()
    result = publish_bundle(bundle) if args.write else {
        "status": "ready", "write_performed": False, "output_files": [str(path) for path in bundle["payloads"]],
        "frozen_input_count": len(bundle["input_sha256"]), "launched_jobs": 0}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
