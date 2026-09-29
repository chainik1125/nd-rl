"""Separate outcome-blind 24-target budget pilot; root-owned CPU operations.

Never change a primary file, inspect predictions while selecting, or issue a
textbook release. Preparation, endpoint binding, and analysis are separate acts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SPRINT = Path(__file__).resolve().parents[1]
SALT = "olmo-20260909-budget-pilot-24-v1"
PRIMARY_SHA = "7f01ae7e6d5938c661a45f3f60c8a7c5f8e14e4a8a553b9ca6147f258bcf4292"
VALIDATOR_SHA = "3be875f9d2a965464ead9f9fd20f377a7906f53cc12d90e2b785e36d86386d60"
VALIDATOR_TEST_SHA = "997747883586376e6029504a7d2fc3c597e909862a133e18ad888329bcb344e5"
DEADLINE = "2026-09-10T05:14:00+00:00"
ARMS = ["base_zero", "base_icl32", "sft", "dpo", "rl"]
COUNTS = {"long_panel": 16, "short_panel": 8}
DATA = "data_budget_pilot_v1"
FREEZE = "plans/budget_pilot_source_freeze_v1.json"
RUNTIME_FREEZE = "plans/budget_pilot_runtime_freeze_v1.json"
PLAN = "plans/eval_budget_pilot_v1.json"
JOB = "plans/eval_budget_pilot_job_v1.json"
RUN = "runs/eval_budget_pilot_v1"
PIPELINE = "runs/eval_budget_pilot_pipeline_v1"
BINDING = "runs/budget_pilot_endpoints_v1.json"
NEW_CODE = ["code/budget_pilot_v1.py", "code/test_budget_pilot_v1.py",
            "code/make_textbook_release_v4.py", "code/test_textbook_release_v4.py"]
COMMITTED_REPORTS = ["reports/budget_pilot_v1_design.md"]
REQUIRED_TRAINING_RECEIPTS = [
    *["runs/grpo_v4/" + name for name in ("run.json", "summary.json", "events.jsonl", "rollouts.jsonl", "trainer_config.json", "trainer_state.json")],
    *["runs/training_continuation_v4/" + name for name in ("plan.json", "events.jsonl", "completed.json")],
    *["jobs/primary_training_v4/" + name for name in ("spec.json", "completed.json")]]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {"path": str(path), "sha256": digest(path)}


def rank(row):
    canonical = row["canonical_hash"]
    return hashlib.sha256((SALT + "\n" + canonical).encode()).hexdigest(), canonical


def select_panels(raw_panels):
    """Selection uses canonical identity and pre-existing long-length strata only."""
    output, identities = {}, {}
    for name, expected in (("eval_short_panel.jsonl", 56), ("eval_long_panel.jsonl", 64)):
        raw = raw_panels[name]
        require(raw.endswith(b"\n"), "Original panel must end in a newline")
        lines = raw.splitlines(keepends=True)
        rows = [json.loads(line) for line in lines]
        require(len(rows) == len({row["canonical_hash"] for row in rows}) == expected, "Original panel denominator/uniqueness changed")
        require(all(not {"proof", "text", "reference_proof", "answer"}.intersection(row) for row in rows), "Pilot input contains a proof label")
        if expected == 56:
            require(all(2 <= row["reference_lines"] <= 6 for row in rows), "Short reference-length range changed")
            chosen = sorted(rows, key=rank)[:8]
        else:
            require({row["reference_lines"] for row in rows} == set(range(7, 15)), "Long strata changed")
            chosen = []
            for length in range(7, 15):
                stratum = sorted((row for row in rows if row["reference_lines"] == length), key=rank)
                require(len(stratum) >= 2, "Long stratum has fewer than two targets")
                chosen.extend(stratum[:2])
        selected = {row["canonical_hash"] for row in chosen}
        # Literal original lines, original relative order: no prompt rewrite.
        output[name] = b"".join(line for line, row in zip(lines, rows, strict=True) if row["canonical_hash"] in selected)
        retained = [row for row in rows if row["canonical_hash"] in selected]
        identities[name] = {"canonical_hashes": [row["canonical_hash"] for row in retained],
            "reference_length_counts": dict(sorted(Counter(row["reference_lines"] for row in retained).items())),
            "family_counts": dict(sorted(Counter(row["family"] for row in retained).items())), "rows": len(retained)}
    return output, identities


def replacement(args, option, value):
    require(args.count(option) == 1, f"Expected exactly one {option}")
    args[args.index(option) + 1] = value


def dependency(path, **equals):
    return {"path": str(path), "equals": equals}


def construct_plans(root, original, greedy_job, sampled_job):
    plan = deepcopy(original)
    require([step["name"] for step in plan["steps"]] == ARMS and plan["mode"] == "greedy", "Original five-arm greedy plan changed")
    plan.update(source_freeze=str(root / RUNTIME_FREEZE), out=str(root / PIPELINE),
        scope="SEPARATE outcome-blind budget-contingency pilot: short8/long16, not completion of the original120-target primary.",
        pilot_manifest=str(root / DATA / "manifest.json"), endpoint_binding=str(root / BINDING),
        parent_primary_plan=ref(root / "plans/eval_greedy_v4.json"),
        priority_amendment="Pilot greedy24 after exact successful RL40, then original greedy120, then original sampled K4; absolute cutoff unchanged.")
    for step in plan["steps"]:
        arm, args = step["name"], step["args"]
        require(step["entrypoint"] == "evaluate.py" and args[args.index("--arm") + 1] == arm, "Unexpected evaluation entrypoint/arm")
        require("--data-dir" not in args, "Original evaluation data override changed")
        replacement(args, "--out", str(root / RUN / arm))
        args.extend(["--data-dir", str(root / DATA)])
        step["required_records"] = [dependency(root / RUN / arm / "evaluation_status.json", status="complete", attempts=24, planned_attempts=24)]
    pilot_job = deepcopy(greedy_job)
    pilot_job.update(args=["--plan", str(root / PLAN)], job_dir=str(root / "jobs/eval_budget_pilot_v1"))
    pilot_job["wait_for_success"].append(dependency(root / BINDING, status="complete", kind="synthetic_pilot_endpoint_binding"))
    pilot_job["wait_for_success"].append(dependency(root / RUNTIME_FREEZE, status="complete", kind="synthetic_pilot_runtime_freeze"))
    full_job = deepcopy(greedy_job)
    full_job["job_dir"] = str(root / "jobs/eval_greedy_after_pilot_v1")
    full_job["wait_for_success"].extend([
        dependency(root / "jobs/eval_budget_pilot_v1/completed.json", exit_code=0),
        dependency(root / PIPELINE / "completed.json", status="complete", stages=ARMS),
        *[dependency(root / RUN / arm / "evaluation_status.json", status="complete", attempts=24, planned_attempts=24) for arm in ARMS]])
    later_sampled = deepcopy(sampled_job)
    later_sampled["job_dir"] = str(root / "jobs/eval_sampled_after_pilot_v1")
    for gate in later_sampled["wait_for_success"]:
        if gate["path"] == str(root / "jobs/eval_greedy_v4/completed.json"):
            gate["path"] = str(root / "jobs/eval_greedy_after_pilot_v1/completed.json")
    docs = {PLAN: plan, JOB: pilot_job, "plans/eval_greedy_after_pilot_job_v1.json": full_job,
            "plans/eval_sampled_after_pilot_job_v1.json": later_sampled}
    require(all(value["not_after_utc"] == DEADLINE for value in docs.values()), "Absolute cutoff changed")
    return docs


def primary_sources(root):
    path = root / "plans/primary_source_freeze_v4.json"
    require(digest(path) == PRIMARY_SHA, "Primary v4 freeze changed")
    original = read(path)
    require(len(original["sha256"]) == 232, "Original frozen input count changed")
    sources = {**original["sha256"], str(path): PRIMARY_SHA}
    for name in NEW_CODE + COMMITTED_REPORTS:
        sources[str(root / name)] = digest(root / name)
    require(sources[str(root / NEW_CODE[2])] == VALIDATOR_SHA and sources[str(root / NEW_CODE[3])] == VALIDATOR_TEST_SHA,
        "Completed-chain validator or its tests changed")
    verify_sources(sources)
    return sources


def verify_sources(sources):
    for path, expected in sources.items():
        require(digest(path) == expected, f"Pinned source/input changed: {path}")


def prepare(root=SPRINT):
    sources = primary_sources(root)
    require(not (root / DATA).exists() and not (root / FREEZE).exists(), "Pilot preparation output already exists")
    raw = {name: (root / "data" / name).read_bytes() for name in ("eval_short_panel.jsonl", "eval_long_panel.jsonl")}
    selected, identities = select_panels(raw)
    selected["shared_cap6.jsonl"] = (root / "data/shared_cap6.jsonl").read_bytes()
    bank = [json.loads(line) for line in selected["shared_cap6.jsonl"].splitlines()]
    require(len(bank) == 32 and len({row["canonical_hash"] for row in bank}) == 32, "Shared bank changed")
    chosen = {key for value in identities.values() for key in value["canonical_hashes"]}
    require(chosen.isdisjoint(row["canonical_hash"] for row in bank), "Pilot target overlaps training bank")
    manifest = {"kind": "separate_outcome_blind_budget_pilot", "created_utc": datetime.now(timezone.utc).isoformat(),
        "salt": SALT, "selection_rule": "SHA256(salt + newline + canonical_hash); smallest8 short, smallest2 within each long reference length7..14; retain original relative order.",
        "selection_uses_model_outcomes_or_token_lengths": False, "parent_source_freeze": ref(root / "plans/primary_source_freeze_v4.json"),
        "parent_inputs": {name: ref(root / "data" / name) for name in selected},
        "builder": ref(root / "code/budget_pilot_v1.py"), "cohort_order": ["long_panel", "short_panel"],
        "cohort_counts": COUNTS, "selection": identities,
        "output_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in selected.items()},
        "reporting_rule": "Report short8 and long16 separately; per-length n2 descriptive; not the primary120; overlapping later full panel is not independent replication.",
        "priority_amendment": "Pilot greedy24 after exact successful RL40, then original greedy120, then originalK4; all stop at05:14UTC.",
        "primary_completion_claim": False, "protected_payloads_read": False}
    docs = construct_plans(root, read(root / "plans/eval_greedy_v4.json"), read(root / "plans/eval_greedy_job_v4.json"), read(root / "plans/eval_sampled_job_v4.json"))
    payloads = {root / DATA / name: value for name, value in selected.items()}
    payloads[root / DATA / "manifest.json"] = encoded(manifest)
    payloads.update({root / name: encoded(value) for name, value in docs.items()})
    require(not any(path.exists() for path in payloads), "Fresh pilot output path required")
    extended = {**sources, **{str(path): hashlib.sha256(value).hexdigest() for path, value in payloads.items()}}
    freeze = {"kind": "separate_budget_pilot_source_freeze", "created_utc": manifest["created_utc"],
        "parent": ref(root / "plans/primary_source_freeze_v4.json"), "parent_receipt_count": 232,
        "salt": SALT, "primary_membership_unchanged": True, "priority_amendment": manifest["priority_amendment"],
        "future_endpoint_receipt": str(root / BINDING), "sha256": dict(sorted(extended.items())),
        "publication_rule": "Written last after verifying all original and new inputs; future exact RL40 endpoint bound separately before inference."}
    verify_sources(sources)
    for path, value in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
    verify_sources(extended)
    with (root / FREEZE).open("xb") as stream:
        stream.write(encoded(freeze))
    return {"status": "prepared_not_launched", "freeze": ref(root / FREEZE), "cohort_counts": COUNTS, "protected_payloads_read": False}


def check_pilot_freeze(root=SPRINT):
    frozen = read(root / FREEZE)
    require(frozen["parent"] == ref(root / "plans/primary_source_freeze_v4.json")
        and frozen["parent"]["sha256"] == PRIMARY_SHA and frozen["salt"] == SALT, "Wrong pilot parent or selector")
    inherited = primary_sources(root)
    require(all(frozen["sha256"].get(path) == value for path, value in inherited.items()), "Pilot freeze omitted or replaced inherited sources")
    for name in (PLAN, JOB, DATA + "/manifest.json", DATA + "/shared_cap6.jsonl", DATA + "/eval_short_panel.jsonl", DATA + "/eval_long_panel.jsonl"):
        require(str(root / name) in frozen["sha256"], "Pilot freeze omitted a required input")
    verify_sources(frozen["sha256"])
    selected, identities = select_panels({name: (root / "data" / name).read_bytes()
        for name in ("eval_short_panel.jsonl", "eval_long_panel.jsonl")})
    manifest = read(root / DATA / "manifest.json")
    require(manifest["salt"] == SALT and manifest["cohort_counts"] == COUNTS
        and manifest["selection"] == json.loads(json.dumps(identities)), "Pilot selection metadata changed")
    for name, expected in selected.items():
        require((root / DATA / name).read_bytes() == expected, "Pilot differs from the deterministic original-panel subset")
    require((root / DATA / "shared_cap6.jsonl").read_bytes() == (root / "data/shared_cap6.jsonl").read_bytes(),
        "Pilot demonstration bank differs from original32")
    return frozen


def binding_from_validated_chain(validated, root=SPRINT):
    """Whitelist projection: never carry textbook authorization or payload fields."""
    require(validated["completed_lineage"] == ["v3_sft", "v3_generated_pairs", "v3_dpo", "v4_grpo"]
        and validated["v3_pipeline_success"] is False, "Wrong validated training lineage")
    allowed = [root / name for name in ("code", "data", "plans", "models/olmo32b")]
    allowed += [root / "runs" / name for name in ("sft", "pairs", "dpo", "grpo", "grpo_v4", "training_pipeline_v3", "training_continuation_v4")]
    allowed += [root / "jobs" / name for name in ("primary_training_v3", "primary_training_v4")]
    allowed.append(root.parent / "r1-sprint-20260909-1625/code/vendor")
    training_receipts = {name: value for name, value in validated["validated_receipts_sha256"].items()
        if any(Path(name).resolve().is_relative_to(directory) for directory in allowed)
        and Path(name).name not in {"textbook_dev.jsonl", "textbook_final.jsonl"}}
    require(all(str(root / name) in training_receipts for name in REQUIRED_TRAINING_RECEIPTS),
        "Completed-chain validator omitted required GRPO numerical/training evidence")
    return {"kind": "synthetic_pilot_endpoint_binding", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_source_freeze": ref(root / FREEZE), "checkpoint_files": validated["checkpoint_files"],
        "checkpoint_map": validated["checkpoint_map"], "completed_training": validated["completed_training"],
        "completed_lineage": validated["completed_lineage"], "v3_pipeline_success": False, "optimizer_steps": 40,
        "validated_training_receipts_sha256": dict(sorted(training_receipts.items())),
        "rl_signal": validated["rl_signal"], "numerical_policy": "nf4_fp32_compute_expanded_gqa_efficient_v4",
        "validator": ref(root / "code/make_textbook_release_v4.py"), "binding_implementation": ref(root / "code/budget_pilot_v1.py"),
        "validation_reuse": "Completed-chain validator invoked in memory; all textbook cohort, payload, and authorization fields discarded. No textbook release issued.",
        "protected_payloads_read": False, "textbook_release_issued": False}


def merge_receipts(*maps):
    result = {}
    for values in maps:
        for path, value in values.items():
            require(path not in result or result[path] == value, f"Conflicting receipt would replace a frozen identity: {path}")
            result[path] = value
    return result


def runtime_from_binding(binding, frozen, root=SPRINT):
    return {"kind": "synthetic_pilot_runtime_freeze", "status": "complete", "static_pilot_freeze": ref(root / FREEZE),
        "endpoint_binding": ref(root / BINDING), "sha256": merge_receipts(frozen["sha256"],
            {str(root / FREEZE): digest(root / FREEZE), str(root / BINDING): digest(root / BINDING)},
            binding["validated_training_receipts_sha256"], binding["checkpoint_files"],
            {receipt["path"]: receipt["sha256"] for receipt in binding["completed_training"]})}


def bind_endpoints(root=SPRINT):
    check_pilot_freeze(root)
    require(not (root / BINDING).exists() and not (root / RUNTIME_FREEZE).exists(), "Endpoint binding/runtime freeze already exists")
    # This exact already-tested function validates complete historical receipts,
    # 40 fresh/640 differentiable audits and exact saved FP32 checkpoint identity.
    # It reads opaque textbook manifest metadata only. Do not save its return as
    # a release: whitelist only nonprotected synthetic endpoint-binding fields.
    from make_textbook_release_v4 import build_release
    validated = build_release("dev", sprint=root)
    binding = binding_from_validated_chain(validated, root)
    frozen = check_pilot_freeze(root)
    verify_sources(binding["checkpoint_files"])
    with (root / BINDING).open("xb") as stream:
        stream.write(encoded(binding))
    runtime = runtime_from_binding(binding, frozen, root)
    verify_sources(runtime["sha256"])
    with (root / RUNTIME_FREEZE).open("xb") as stream:
        stream.write(encoded(runtime))
    return {"status": "complete", "binding": ref(root / BINDING), "runtime_freeze": ref(root / RUNTIME_FREEZE), "textbook_release_issued": False}


def guard_metadata(plans, binding, manifest, root=SPRINT):
    """All five identical pilot targets/endpoints required before prediction reads."""
    from evaluate import MODEL_ID, MODEL_REVISION, stable_hash
    require(len(plans) == 5 and [plan["arm"] for plan in plans] == ARMS, "Every pilot report requires all five declared arms")
    targets = []
    for cohort, filename in (("long_panel", "eval_long_panel.jsonl"), ("short_panel", "eval_short_panel.jsonl")):
        targets.extend(dict(json.loads(line), cohort=cohort) for line in (root / DATA / filename).read_text().splitlines())
    target_identity = stable_hash(targets)
    require(dict(Counter(row["cohort"] for row in targets)) == COUNTS and manifest["cohort_counts"] == COUNTS, "Pilot counts changed")
    expected_inputs = {str(root / DATA / name): digest(root / DATA / name) for name in
        ("manifest.json", "shared_cap6.jsonl", "eval_short_panel.jsonl", "eval_long_panel.jsonl")}
    for plan in plans:
        arm = plan["arm"]
        require(plan["target_count"] == 24 and plan["target_identity"] == target_identity
            and plan["cohorts"] == ["long_panel", "short_panel"] and plan["input_files"] == expected_inputs,
            "Pilot/full membership or source inputs mixed")
        require(plan["model_id"] == MODEL_ID and plan["model_revision"] == MODEL_REVISION
            and Path(plan["model_path"]).resolve() == root / "models/olmo32b"
            and plan["modeling_sha256"] == digest(root / "code/modeling.py")
            and plan["evaluator_sha256"] == digest(root / "code/evaluate.py"), "Pilot runtime changed")
        settings = plan["settings"]
        expected = {"modes": ["greedy"], "max_new_tokens": 2048, "sampled_k": 4, "sampled_temperature": 1.0,
            "top_p": 1.0, "top_k": 0, "seed": 202609092015, "batch_size": 4 if arm == "base_icl32" else 16,
            "stop_on_actually_generated_final_marker": True, "native_eos_fallback": True, "inserted_eos": False}
        require(all(settings.get(key) == value for key, value in expected.items())
            and plan["demonstrations_per_query"] == (32 if arm == "base_icl32" else 0), "Pilot decoding or demonstrations changed")
        expected_adapter = {} if arm.startswith("base_") else {path: value for path, value in binding["checkpoint_files"].items()
            if Path(path).parent == Path(binding["checkpoint_map"][arm])}
        require(plan["adapter_files"] == expected_adapter, "Pilot inference used a different bound endpoint")
    return targets


def analyze_pilot(out, root=SPRINT):
    frozen = check_pilot_freeze(root)
    binding = read(root / BINDING)
    runtime = read(root / RUNTIME_FREEZE)
    require(runtime == runtime_from_binding(binding, frozen, root),
        "Runtime freeze does not bind the exact pilot and endpoints")
    verify_sources(runtime["sha256"])
    require(binding["status"] == "complete" and binding["kind"] == "synthetic_pilot_endpoint_binding"
        and binding["pilot_source_freeze"] == ref(root / FREEZE) and binding["optimizer_steps"] == 40,
        "Missing exact completed RL40 endpoint binding")
    verify_sources(binding["checkpoint_files"])
    for receipt in binding["completed_training"]:
        require(ref(Path(receipt["path"])) == receipt, "Bound completed-training metadata changed")
    require(read(root / "jobs/eval_budget_pilot_v1/completed.json").get("exit_code") == 0, "Pilot owning job incomplete")
    completion = read(root / PIPELINE / "completed.json")
    require(completion == {"status": "complete", "stages": ARMS, "plan_sha256": digest(root / PLAN),
        "source_freeze_sha256": digest(root / RUNTIME_FREEZE)}, "Pilot pipeline did not finish its exact frozen plan")
    directories = [root / RUN / arm for arm in ARMS]
    plans = [read(directory / "evaluation_plan.json") for directory in directories]
    for directory in directories:
        state = read(directory / "evaluation_status.json")
        require(state.get("status") == "complete" and state.get("attempts") == state.get("planned_attempts") == 24,
            "All five complete pilot endpoints required")
    guard_metadata(plans, binding, read(root / DATA / "manifest.json"), root)
    # No changed combiner: it intentionally accepts only original56/64 panels.
    # The unchanged analyzer supports matched same-mode cohorts of arbitrary N.
    from analyze import run
    report = run(SimpleNamespace(runs=directories, allow_partial=False, out=out, bootstrap_draws=10000, no_plots=True))
    label = {"kind": "separate_budget_contingency_pilot_report", "status": "complete", "cohort_counts": COUNTS,
        "primary120_completed_by_this_report": False, "endpoint_binding": ref(root / BINDING),
        "pilot_source_freeze": ref(root / FREEZE), "analysis_summary": ref(out / "summary.json"),
        "inference": "Short8 and long16 separately; n2 per long length is descriptive only. Overlap with any later primary120 is not independent replication.",
        "pooled24_headline": False, "analysis_performs_independent_likelihood_recomputation": False}
    with (out / "PILOT_IDENTITY.json").open("xb") as stream:
        stream.write(encoded(label))
    with (out / "PILOT_README.md").open("x") as stream:
        stream.write("# Separate outcome-blind budget pilot (8 short, 16 long)\n\nNot completion of the original 120-target primary. Read short and long results separately; each long reference-length bin has only two targets. The later full panel overlaps and is not independent replication. Numerical guards are historical validated receipts, not an independent likelihood rerun.\n")
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "bind", "analyze"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "analyze":
        if args.out is None or not args.out.resolve().is_relative_to(SPRINT / "reports"):
            parser.error("Analysis needs a fresh specific --out within sprint reports")
        result = analyze_pilot(args.out.resolve())
    else:
        result = prepare() if args.action == "prepare" else bind_endpoints()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
