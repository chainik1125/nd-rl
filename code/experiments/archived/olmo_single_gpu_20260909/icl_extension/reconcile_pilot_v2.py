"""Root-owned CPU reconciliation of five complete endpoints across three executions.

This does not change or impersonate budget_pilot_v1.analyze: that historical
wrapper correctly rejects its cancelled owner. Every gate below precedes the
first raw-outcome read by the unchanged native-token/verifier analyzer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from budget_pilot_v1 import ARMS, COUNTS, guard_metadata, verify_sources

FREEZE_SHA = "14f6f58860ce609cae17d1a2e917b8f6489c94910479ddc5e31230d8464e0fb9"
ANALYZER_SHA = "37f02d136a9a215fabdc1f3c3de738433f389349036eeaec4bf59884ca10017e"
GPU_UUID = "GPU-64be1f00-efff-44f5-0340-f73ed3c3274c"
HARD_CUTOFF = "2026-09-10T18:54:00+00:00"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path):
    return {"path": str(path), "sha256": digest(path)}


def complete_status(value):
    require(value.get("status") == "complete"
            and value.get("attempts") == value.get("planned_attempts") == 24,
            "All five whole 24-target endpoint datasets must be complete")


def prefix_check(raw, receipt):
    n = receipt["bytes"]
    require(len(raw) >= n and hashlib.sha256(raw[:n]).hexdigest() == receipt["sha256"],
            "Original committed ICL bytes changed")
    if receipt.get("must_remain_entire_file_identical"):
        require(len(raw) == n, "Immutable ICL manifest/targets changed")
    require(raw.endswith(b"\n"), "Uncommitted trailing bytes remain")
    return {"original_bytes": n, "current_bytes": len(raw), "prefix_identical": True}


def expected_stage_command(step):
    return [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "code/evaluate.py"), *step["args"]]


def check_stage(events, step):
    starts = [e for e in events if e["event"] == "stage_start" and e["name"] == step["name"]]
    exits = [e for e in events if e["event"] == "stage_exit" and e["name"] == step["name"]]
    require(len(starts) == len(exits) == 1 and starts[0]["command"] == expected_stage_command(step)
            and exits[0]["exit_code"] == 0, "Exact successful evaluator stage missing")
    require(datetime.fromisoformat(starts[0]["utc"]) < datetime.fromisoformat(exits[0]["utc"]),
            "Stage timing is inconsistent")


def check_lease(events, spec, completion, stage_events, failed=False):
    acquisitions = [e for e in events if e["event"] == "acquired"]
    terminals = [e for e in events if e["event"] in ("completed", "failed")]
    require(len(acquisitions) == len(terminals) == 1, "Expected one acquired and terminal lease")
    acquired, terminal = acquisitions[0], terminals[0]
    require(acquired["command"] == [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "code/pipeline.py"), *spec["args"]]
            and acquired["entry_sha256"] == digest(ROOT / "code/pipeline.py"), "Acquired pipeline differs")
    require(acquired["gpu"]["index"] == completion["gpu"] == 3, "Lease and completion GPU differ")
    if failed:
        require(terminal["event"] == "failed" and terminal["error"] == completion["error"] == "KeyboardInterrupt"
                and completion["exit_code"] == 1 and completion["status"] == "failed", "Wrong cancelled lease")
        require(abs((datetime.fromisoformat(terminal["utc"]) - datetime.fromisoformat(completion["utc"])).total_seconds()) < 2,
                "Cancellation terminal time differs from receipt")
    else:
        require(terminal["event"] == "completed" and {k: v for k, v in terminal.items() if k not in ("event", "utc")} == completion
                and completion["exit_code"] == 0, "Terminal event differs from successful completion")
    begin, end = (datetime.fromisoformat(e["utc"]) for e in (acquired, terminal))
    require(begin < end, "Invalid lease timing")
    require(all(begin <= datetime.fromisoformat(e["utc"]) <= end for e in stage_events),
            "Pipeline event lies outside its owning lease")
    return acquired, terminal


def validate_provenance():
    checked = {}
    def read(name, lines=False):
        path = ROOT / name
        checked[str(path)] = digest(path)
        raw = path.read_text()
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    freeze_path = ROOT / "plans/icl_resume_source_freeze_v2.json"
    require(digest(freeze_path) == FREEZE_SHA, "Authorized v2 source freeze changed")
    frozen = read("plans/icl_resume_source_freeze_v2.json")
    require(len(frozen["sha256"]) == 276, "Wrong v2 source count")
    verify_sources(frozen["sha256"])
    checked.update(frozen["sha256"])
    trained_identity = read("reports/trained_contingency_analysis_v1/CONTINGENCY_IDENTITY.json")
    verify_sources(trained_identity["sha256"])
    checked.update(trained_identity["sha256"])
    require(digest(ROOT / "code/analyze.py") == ANALYZER_SHA, "Native analyzer changed")

    old_plan = read("plans/eval_budget_pilot_v1.json")
    old_spec = read("plans/eval_budget_pilot_job_v1.json")
    require(read("jobs/eval_budget_pilot_v1/spec.json") == old_spec, "Old owner specification changed")
    require(read("runs/eval_budget_pilot_pipeline_v1/plan.json") == old_plan, "Old pipeline plan changed")
    old_owner = read("jobs/eval_budget_pilot_v1/completed.json")
    require(old_owner.get("exit_code") == 1 and old_owner.get("status") == "failed"
            and old_owner.get("error") == "KeyboardInterrupt" and old_owner.get("gpu") == 3,
            "Original controlled cancellation must remain a failure")
    require(not (ROOT / "runs/eval_budget_pilot_pipeline_v1/completed.json").exists(),
            "Do not fabricate original pipeline completion")
    old_events = read("runs/eval_budget_pilot_pipeline_v1/events.jsonl", lines=True)
    old_owner_events = read("jobs/eval_budget_pilot_v1/events.jsonl", lines=True)
    old_acquired, _ = check_lease(old_owner_events, old_spec, old_owner, old_events, failed=True)
    require(old_acquired["spec_sha256"] == digest(ROOT / "plans/eval_budget_pilot_job_v1.json")
            and old_acquired["gpu"]["uuid"] == "GPU-539b2ed7-e2bc-5d36-b909-e6f25c26f875", "Wrong original acquisition")
    require([e["name"] for e in old_events if e["event"] == "stage_exit"] == ["base_zero"],
            "Unexpected original completed-stage sequence")
    old_starts = [e for e in old_events if e["event"] == "stage_start"]
    require([e["name"] for e in old_starts] == ["base_zero", "base_icl32"]
            and old_starts[1]["command"] == expected_stage_command(old_plan["steps"][1]),
            "Missing original ICL execution provenance")
    check_stage(old_events, old_plan["steps"][0])

    for suffix, plan_name, freeze_name, stages in [
        ("trained_contingency_v1", "eval_trained_contingency_v1", "trained_contingency_source_freeze_v1", ["sft", "rl", "dpo"]),
        ("icl_resume_v2", "eval_icl_resume_v2", "icl_resume_source_freeze_v2", ["base_icl32"]),
    ]:
        plan_path = ROOT / f"plans/{plan_name}.json"
        spec_path = ROOT / f"plans/{plan_name.replace('_v1', '_job_v1').replace('_v2', '_job_v2')}.json"
        plan = read(str(plan_path.relative_to(ROOT)))
        spec = read(str(spec_path.relative_to(ROOT)))
        pipeline = f"runs/eval_{suffix.replace('_v1', '_pipeline_v1').replace('_v2', '_pipeline_v2')}"
        job = f"jobs/eval_{suffix}"
        require(read(f"{pipeline}/plan.json") == plan and read(f"{job}/spec.json") == spec,
                "Owning or pipeline specification changed")
        owner_completion = read(f"{job}/completed.json")
        require(owner_completion.get("exit_code") == 0, "Reconciled owner failed or unfinished")
        completion = read(f"{pipeline}/completed.json")
        require(completion == {"status": "complete", "stages": stages,
                "plan_sha256": digest(plan_path), "source_freeze_sha256": digest(ROOT / f"plans/{freeze_name}.json")},
                "Pipeline did not complete its exact frozen plan")
        events = read(f"{pipeline}/events.jsonl", lines=True)
        require([e["name"] for e in events if e["event"] == "stage_exit"] == stages,
                "Unexpected completed-stage sequence")
        for step in plan["steps"]:
            check_stage(events, step)
        owner_events = read(f"{job}/events.jsonl", lines=True)
        acquired, terminal = check_lease(owner_events, spec, owner_completion, events)
        require(acquired["spec_sha256"] == digest(spec_path), "Acquired job spec mismatch")
        if suffix == "icl_resume_v2":
            require(acquired["gpu"]["index"] == 3 and acquired["gpu"]["uuid"] == GPU_UUID,
                    "Resume acquired a different physical GPU")
            require(datetime.fromisoformat(terminal["utc"]) <= datetime.fromisoformat(HARD_CUTOFF),
                    "Resume exceeded authorized GPU cutoff")
            require(plan["steps"] == [old_plan["steps"][1]], "Original ICL evaluator step changed")

    prefixes = read("reports/icl_pre_resume_prefixes_v2.json")
    prefix_results = {name: prefix_check((ROOT / name).read_bytes(), record)
                      for name, record in prefixes["files"].items()}
    require(prefixes["existing_attempts"] == 12 and not (ROOT / "jobs/eval_icl_resume_v1").exists(),
            "Wrong original ICL prefix or expired resume was launched")
    binding = read("runs/budget_pilot_endpoints_v1.json")
    require(binding["status"] == "complete" and binding["optimizer_steps"] == 40, "Wrong training endpoint")
    verify_sources(binding["checkpoint_files"])
    directories = [ROOT / "runs/eval_budget_pilot_v1" / arm for arm in ARMS[:2]]
    directories += [ROOT / "runs/eval_trained_contingency_v1" / arm for arm in ARMS[2:]]
    plans = [read(str((d / "evaluation_plan.json").relative_to(ROOT))) for d in directories]
    manifest = read("data_budget_pilot_v1/manifest.json")
    targets = guard_metadata(plans, binding, manifest, ROOT)
    identities = []
    for arm, directory in zip(ARMS, directories, strict=True):
        status = read(str((directory / "evaluation_status.json").relative_to(ROOT)))
        complete_status(status)
        model = status["model_identity"]
        require(model["numerical_policy"]["policy_id"] == "nf4_fp32_compute_expanded_gqa_efficient_v4",
                "Wrong actual inference numerical operator")
        require(model["adapter_path"] == (None if arm.startswith("base_") else binding["checkpoint_map"][arm]),
                "Actual loaded adapter differs from bound endpoint")
        for audit in model["adapter_load_audits"]:
            checkpoint = str(Path(binding["checkpoint_map"][arm]) / "adapter_model.safetensors")
            require(audit["adapter_name"] == "default" and audit["checkpoint"] == checkpoint
                    and audit["sha256"] == binding["checkpoint_files"][checkpoint], "Load audit is for another checkpoint")
            require(audit["all_tensors_exact"] and audit["tensors"] == 896 and audit["parameters"] == 134217728
                    and audit["dtype"] == "float32" and digest(audit["checkpoint"]) == audit["sha256"],
                    "Adapter load audit failed")
        require(len(model["adapter_load_audits"]) == (0 if arm.startswith("base_") else 1),
                "Missing or unexpected adapter audit")
        identities.append(model)
        for name in ("targets.jsonl", "attempts.jsonl", "compute_batches.jsonl"):
            checked[str(directory / name)] = digest(directory / name)
    for model in identities[1:]:
        for key in ("model_id", "revision", "config_sha256", "download_manifest_sha256", "local_model",
                    "quantization", "attention", "nonquantized_dtype", "numerical_policy", "versions"):
            require(model[key] == identities[0][key], f"Actual common runtime/operator differs: {key}")
    checked[str(Path(__file__).resolve())] = digest(__file__)
    return directories, {"kind": "reconciled_five_arm_provenance_gate_v2", "status": "passed_before_native_outcome_reads",
        "checked_utc": datetime.now(timezone.utc).isoformat(), "arms": ARMS, "cohort_counts": COUNTS,
        "source_inputs": 276, "target_count": len(targets), "old_owning_job_still_failed": True,
        "original_owning_pipeline_complete": False, "icl_prefixes": prefix_results,
        "known_trained_outcomes_at_extension": True, "zero_or_icl_outcomes_read_by_gate": False,
        "actual_common_model_runtime_operator_equal": True, "physical_host_switch_disclosed": True,
        "icl_model_identity_scope": "Final status describes the resumed invocation; the original twelve attempts retain original manifest, frozen source/input and owning-stage provenance, not newly measured historical runtime or cross-device equivalence.",
        "cross_host_bitwise_equivalence_claim": False, "sha256": checked}


def run(out, check_only=False):
    require(not out.exists() and out.resolve().is_relative_to(ROOT / "reports"), "Need a fresh sprint report output")
    directories, gate = validate_provenance()
    if check_only:
        return gate
    # Persist the completed provenance gate before any native-outcome analysis.
    gate_path = ROOT / "icl-extension-20260910-1754/reports/reconciliation_gate_v2.json"
    require(not gate_path.exists(), "Do not replace an existing gate")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_path.open("x") as stream:
        json.dump(gate, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"pre_outcome_provenance_gate": "pass", "gate": str(gate_path)}), flush=True)
    from analyze import run as native_analyze
    report = native_analyze(SimpleNamespace(runs=directories, allow_partial=False, out=out,
                                          bootstrap_draws=10000, no_plots=True))
    require(report["status"] == "complete", "Native replay did not complete")
    verify_sources(gate["sha256"])
    rows = [json.loads(line) for line in (out / "per_theorem.jsonl").read_text().splitlines()]
    require(len(rows) == 120, "Native analyzer did not replay all five whole endpoints")
    identity = {"kind": "separate_budget_contingency_pilot_report", "status": "complete",
        "completion_scope": "Five complete 24-target endpoint datasets reconciled across three executions; not original-job success",
        "arms": ARMS, "cohort_counts": COUNTS, "strictly_replayed_attempts": 120,
        "strictly_accepted_attempts": sum(row["solved"] for row in rows),
        "primary120_completed_by_this_report": False, "sampled_k4_complete": False, "pooled24_headline": False,
        "original_owning_job_complete": False, "original_owning_pipeline_complete": False,
        "original_failure_preserved": True, "endpoint_binding": ref(ROOT / "runs/budget_pilot_endpoints_v1.json"),
        "pilot_source_freeze": ref(ROOT / "plans/budget_pilot_source_freeze_v1.json"),
        "analysis_summary": ref(out / "summary.json"), "analysis_per_theorem": ref(out / "per_theorem.jsonl"),
        "analysis_producer": {"gate": ref(gate_path), "script": ref(Path(__file__).resolve()),
            "entrypoint": "unchanged code/analyze.py::run", "analyzer_sha256": ANALYZER_SHA,
            "original_budget_pilot_analyze_wrapper_invoked": False, "allow_partial": False, "bootstrap_draws": 10000},
        "reconciliation": {"kind": "user_authorized_icl_completion_v2", "status": "complete_after_provenance_gate",
            "original_icl_attempts": 12, "new_icl_attempts": 12, "total_icl_attempts": 24,
            "original_prefixes_preserved": True, "original_icl_manifest_unchanged": True,
            "original_host": "simplex2/g374", "resumed_host": "simplex1/g178", "resumed_gpu_uuid": GPU_UUID,
            "known_trained_outcomes_at_extension": True, "cross_host_bitwise_equivalence_claim": False},
        "analysis_performs_independent_likelihood_recomputation": False,
        "compute_caveat": "Original interrupted ICL call has unknown additional output bounded by8192 native tokens; closed-call token totals are lower bounds on total generation compute. Its elapsed GPU lease remains charged.",
        "sha256": {**gate["sha256"], str(gate_path): digest(gate_path), **{
            str(out / name): digest(out / name) for name in
            ("summary.json", "per_theorem.jsonl", "certificates_analysis_only.jsonl", "report.md")}}}
    with (out / "PILOT_IDENTITY.json").open("x") as stream:
        json.dump(identity, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve(), args.check_only), sort_keys=True))
