"""Metadata-only release after the validated v3 prefix and completed v4 RL.

Root-owned operation. Never open, hash, or parse protected theorem payloads.
The existing evaluator's generic release checks are deliberately unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import continuation_prefix as prefix

SPRINT = Path(__file__).resolve().parents[1]
FINAL_FLOOR = datetime.fromisoformat("2026-09-10T05:14:00+00:00")
APPROVED = {
    "code/modeling.py": "a8da6e3fc1d8b4223fb4f4ece8285430c18101124476434b983b55afbca37056",
    "code/train.py": "70622fa9bd7ab0b56beaa1f8fc823a955ab64ca1952225d5a28c412d9446322b",
    "code/launch.py": "f3008ad72910df217cd1ff475e8bceb71becf69a5853b416fcc73f415ee9b053",
}
NUMERICS_SHA = "1f8fb573de162acb84eebe8b82723fb1acd2dc19a474a20d7d16668635a6ea37"
PREFIX_SHA = "bd258f9b488c229b0bfd4ca85670854fdade522a045967f46cf343adaa0aaa98"
PRIMARY_FREEZE_SHA = "7f01ae7e6d5938c661a45f3f60c8a7c5f8e14e4a8a553b9ca6147f258bcf4292"
DATA_SHA = {"dev": "1b04192d8a2d91b488a0884ccb096004003dbfcf40bb7aea55caa81c3f3cf53c",
            "final": "3ca20ac49db6e3fc4d090c1424f48ff0b89de4686110bb00e194e4b51e4ae6aa"}
SETTINGS = {"max_new_tokens": 2048, "sampled_k": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0}
require = prefix.require


class SafeReceipts(prefix.Receipts):
    def __init__(self, sprint):
        super().__init__()
        self.sprint = Path(sprint).resolve()

    def guard(self, path):
        path = Path(path).resolve()
        textbook = self.sprint.parent / "r1-cot-20260909/data"
        require(not path.is_relative_to(textbook) or path == textbook / "manifest.json",
                "Release validator must never inspect protected theorem payloads")
        require(path.is_relative_to(self.sprint) or path.is_relative_to(self.sprint.parent / "r1-sprint-20260909-1625/code/vendor")
                or path == textbook / "manifest.json", "Release receipt path is outside approved metadata scope")
        return path

    def digest(self, path):
        return super().digest(self.guard(path))

    def json(self, path):
        return super().json(self.guard(path))

    def rows(self, path):
        return super().rows(self.guard(path))


def _path(sprint, value):
    return prefix._path(sprint, value)


def _ref(receipts, path):
    return {"path": str(path), "sha256": receipts.digest(path)}


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_likelihood_receipt(row, *, phase, tokens=None):
    require(row.get("phase") == phase and row.get("tolerance") == 0.1
        and row.get("denominator") == "actual_streamed_sampling_logprobs"
        and row.get("scorer_model_training") is True
        and row.get("scorer_requires_grad") is (phase == "actual_gradient_loss"), "Wrong likelihood audit policy/scope")
    require(type(row.get("tokens")) is int and row["tokens"] > 0
        and (tokens is None or row["tokens"] == tokens), "Likelihood audit omitted or miscounted sampled tokens")
    maximum, mean, p99 = (row.get(key) for key in ("max_abs_error", "mean_abs_error", "p99_abs_error"))
    require(all(_finite_number(value) for value in (maximum, mean, p99))
        and 0 <= mean <= maximum <= 0.1 and 0 <= p99 <= maximum + 1e-7,
        "Missing, nonfinite, or failed likelihood audit")


def check_rollouts(sprint, receipts, events, summary):
    protocol, modeling, tokenizer = prefix._load_tools(sprint)
    vocabulary_size = len(tokenizer)
    bank = receipts.rows(sprint / "data/shared_cap6.jsonl")
    pool = {row["canonical_hash"]: row for row in bank}
    require(len(bank) == len(pool) == 32, "Shared training bank changed")
    rows = receipts.rows(sprint / "runs/grpo_v4/rollouts.jsonl")
    require(len(rows) == 640, "Completed 40-step continuation must have 640 sampled responses")
    by_kind = {kind: [row for row in events if row.get("kind") == kind] for kind in
        ("generation_batch", "reward_batch", "actual_grpo_likelihood_audit", "actual_grpo_gradient_likelihood_audit", "gradient", "parameter_update")}
    for kind in ("generation_batch", "actual_grpo_likelihood_audit", "gradient"):
        require([row.get("step") for row in by_kind[kind]] == list(range(40)), f"Missing/repeated/reordered {kind} receipt")
    require([row.get("batch") for row in by_kind["reward_batch"]] == list(range(40)), "Reward batch sequence changed")
    gradient_audits = by_kind["actual_grpo_gradient_likelihood_audit"]
    require([row.get("step") for row in gradient_audits] == [step for step in range(40) for _ in range(16)],
            "Missing or repeated actual-gradient microbatch audits")
    valid_total = mixed_total = token_total = 0
    for batch in range(40):
        samples = rows[16 * batch:16 * (batch + 1)]
        lengths, finishes, rewards, prompt_tokens = [], Counter(), [], 0
        for index, row in enumerate(samples):
            require(row.get("batch") == batch and row.get("index") == index
                and row.get("mechanical_reward_fixture") is False, "Mechanical/misordered RL action")
            source = pool.get(row.get("canonical_hash"))
            require(source is not None and row.get("theorem") == source["prompt"], "RL action is outside the frozen training bank")
            ids, logps = row.get("completion_ids"), row.get("behavior_logprobs")
            require(isinstance(ids, list) and 0 < len(ids) <= 2048
                and all(type(token) is int and 0 <= token < vocabulary_size for token in ids), "Invalid native RL action IDs")
            require(isinstance(logps, list) and len(logps) == len(ids)
                and all(_finite_number(value) and value <= 1e-6 for value in logps), "Invalid recorded behavior likelihoods")
            text = modeling.decode_completion(tokenizer, ids)
            trimmed, finish = modeling.trim_completion(tokenizer, ids)
            require(trimmed == ids and text == row.get("completion"), "Literal native action replay differs")
            require(finish != "length" or len(ids) == 2048, "Unexplained short length-truncated action")
            verdict = protocol.judge(source["prompt"], text)
            require(verdict == row.get("verdict") and row.get("reward") == verdict["reward"], "RL verifier reward replay differs")
            count = len(tokenizer.encode(protocol.render_prompt(source["prompt"]), add_special_tokens=False))
            require(count + len(ids) <= 4096, "RL action exceeds frozen context budget")
            prompt_tokens += count
            lengths.append(len(ids)); finishes[finish] += 1; rewards.append(verdict["reward"])
        for offset in range(0, 16, 4):
            require(len({row["canonical_hash"] for row in samples[offset:offset + 4]}) == 1, "Relative-reward group mixes theorems")
        mixed = sum(min(rewards[i:i + 4]) != max(rewards[i:i + 4]) for i in range(0, 16, 4))
        generated, reward, fresh = (by_kind[kind][batch] for kind in
            ("generation_batch", "reward_batch", "actual_grpo_likelihood_audit"))
        require(generated.get("finish_counts") == dict(finishes) and generated.get("prompt_tokens") == prompt_tokens
            and generated.get("completion_tokens") == sum(lengths), "Generation accounting differs from native actions")
        require(reward.get("examples") == 16 and reward.get("groups") == 4 and reward.get("mixed_groups") == mixed
            and reward.get("mixed_group_fraction") == mixed / 4 and reward.get("success_fraction") == sum(rewards) / 16
            and reward.get("reward_mean") == sum(rewards) / 16 and reward.get("completion_tokens") == sum(lengths)
            and reward.get("mechanical_reward_fixture") is False, "RL signal receipt differs from actual rewards")
        check_likelihood_receipt(fresh, phase="fresh_rollout", tokens=sum(lengths))
        require(fresh.get("finish_counts") == dict(finishes), "Fresh likelihood audit uses different actions")
        active = gradient_audits[16 * batch:16 * (batch + 1)]
        require(Counter(row.get("tokens") for row in active) == Counter(lengths), "Differentiable audits do not cover every sampled response")
        for audit in active:
            check_likelihood_receipt(audit, phase="actual_gradient_loss")
            require(audit.get("finish_counts") == dict(finishes), "Gradient audit references a different rollout")
        # All sixteen microbatch guards must occur after generation and before
        # its optimizer gradient receipt, not after a parameter update.
        positions = [events.index(row) for row in active]
        require(events.index(generated) < events.index(reward) < events.index(fresh) < min(positions)
            and max(positions) < events.index(by_kind["gradient"][batch]), "Likelihood guards are outside the pre-update window")
        valid_total += int(sum(rewards)); mixed_total += mixed; token_total += sum(lengths)
    norms = [row.get("norm") for row in by_kind["gradient"]]
    require(all(_finite_number(norm) and norm >= 0 for norm in norms), "Nonfinite/missing optimizer gradients")
    changes = by_kind["parameter_update"]
    require(all(type(row.get("step")) is int and 0 <= row["step"] < 40
        and _finite_number(row.get("max_abs_change")) and row["max_abs_change"] >= 0 for row in changes), "Invalid parameter-change receipt")
    require(summary.get("nonzero_gradient_steps") == sum(norm > 0 for norm in norms)
        and summary.get("observed_parameter_changes") == sum(row["max_abs_change"] > 0 for row in changes), "Summary misreports actual gradient/update counts")
    return {"responses": 640, "completion_tokens": token_total, "verifier_valid_responses": valid_total,
        "groups": 160, "mixed_groups": mixed_total, "fresh_likelihood_audits": 40,
        "actual_gradient_likelihood_audits": 640, "nonzero_gradient_steps": summary["nonzero_gradient_steps"],
        "observed_parameter_changes": summary["observed_parameter_changes"], "strict_native_reward_replay": True}


def build_release(cohort, *, sprint=SPRINT, now=None):
    sprint = Path(sprint).resolve()
    now = now or datetime.now(timezone.utc)
    require(cohort in DATA_SHA and now.tzinfo is not None, "Invalid cohort or naive release time")
    require(cohort != "final" or now >= FINAL_FLOOR, "Final textbook release is not yet permitted")
    receipts = SafeReceipts(sprint)
    for name, digest in {**APPROVED, "code/numerics.py": NUMERICS_SHA, "code/continuation_prefix.py": PREFIX_SHA}.items():
        require(receipts.digest(sprint / name) == digest, f"Unapproved v4 source: {name}")
    freeze_path, prefix_path = (sprint / "plans" / name for name in ("primary_source_freeze_v4.json", "continuation_prefix_v4.json"))
    plan_path, job_plan = (sprint / "plans" / name for name in ("primary_training_v4.json", "primary_training_job_v4.json"))
    freeze = receipts.json(freeze_path)
    require(receipts.digest(freeze_path) == PRIMARY_FREEZE_SHA, "Unapproved primary v4 freeze identity")
    require(freeze.get("version") == 4 and _path(sprint, freeze.get("supersedes", "")) == sprint / "plans/primary_source_freeze_v3.json"
        and isinstance(freeze.get("sha256"), dict), "Missing v4 source freeze")
    for path, digest in freeze["sha256"].items():
        require(receipts.digest(path) == digest, f"Frozen v4 input changed: {path}")
    required_frozen = {str(sprint / name) for name in (*APPROVED, "code/numerics.py", "code/continuation_prefix.py")}
    required_frozen.update(map(str, (prefix_path, plan_path, job_plan)))
    require(required_frozen <= set(freeze["sha256"]), "V4 freeze omits required sources or continuation plans")
    live_prefix = prefix.validate_prefix(sprint, approved_source_changes=APPROVED)
    saved_prefix = receipts.json(prefix_path)
    require({key: value for key, value in saved_prefix.items() if key != "created_utc"} == live_prefix,
            "Saved v3 prefix does not match complete live validation")
    require(live_prefix.get("status") == "validated" and live_prefix.get("v3_pipeline_success") is False,
            "Continuation must preserve the failed original v3 pipeline identity")
    require(all(freeze["sha256"].get(path) == digest for path, digest in live_prefix["sha256"].items()),
            "V4 freeze does not commit the entire validated v3 prefix")
    # Optional protected-release tooling is intentionally not in the earlier
    # primary training freeze. Bind the finalized builder/tests in this release
    # itself and verify their bytes remain stable throughout validation.
    release_implementation = _ref(receipts, sprint / "code/make_textbook_release_v4.py")
    release_tests = _ref(receipts, sprint / "code/test_textbook_release_v4.py")
    plan = receipts.json(plan_path)
    require(plan.get("approved_source_changes") == APPROVED
        and plan.get("prefix_receipt") == _ref(receipts, prefix_path)
        and plan.get("numerical_helper") == _ref(receipts, sprint / "code/numerics.py")
        and _path(sprint, plan["source_freeze"]) == freeze_path
        and _path(sprint, plan["out"]) == sprint / "runs/training_continuation_v4"
        and datetime.fromisoformat(plan["not_after_utc"]) == FINAL_FLOOR, "Continuation plan lineage or deadline changed")
    old_plan = receipts.json(sprint / "plans/primary_training_v3.json")
    expected_step = json.loads(json.dumps(old_plan["steps"][-1]))
    expected_step["args"][expected_step["args"].index("--out") + 1] = "runs/grpo_v4"
    expected_step["required_records"] = [{"path": str(sprint / "runs/grpo_v4/summary.json"),
        "equals": {"status": "complete", "mechanical_fixture": False, "stage": "grpo", "optimizer_steps": 40,
                   "requested_steps": 40, "deadline_reached": False}}]
    require(plan.get("steps") == [expected_step], "Continuation changes frozen RL command, budget, or endpoint")
    pipeline_dir = sprint / "runs/training_continuation_v4"
    pipeline_path, job_path = pipeline_dir / "completed.json", sprint / "jobs/primary_training_v4/completed.json"
    pipeline = receipts.json(pipeline_path)
    require(pipeline == {"status": "complete", "stages": ["grpo"], "plan_sha256": receipts.digest(plan_path),
        "source_freeze_sha256": receipts.digest(freeze_path)}, "Successful complete v4 pipeline identity required")
    require(receipts.digest(pipeline_dir / "plan.json") == receipts.digest(plan_path), "Pipeline executed a different plan")
    timeline = receipts.rows(pipeline_dir / "events.jsonl")
    require([(row.get("event"), row.get("name")) for row in timeline] ==
        [("stage_start", "grpo"), ("stage_pid", "grpo"), ("stage_exit", "grpo"), ("pipeline_completed", None)], "Continuation events incomplete/reordered")
    require(timeline[0]["command"] == [str(sprint / ".venv/bin/python"), "-u", str(sprint / "code/train.py"), *expected_step["args"]]
        and type(timeline[1].get("pid")) is int and timeline[1]["pid"] > 0
        and type(timeline[2].get("exit_code")) is int and timeline[2]["exit_code"] == 0
        and timeline[3].get("stages") == ["grpo"], "Continuation pipeline did not execute the required successful command")
    job = receipts.json(job_path)
    require(type(job.get("exit_code")) is int and job["exit_code"] == 0, "Owning v4 training job has not exited successfully")
    spec = receipts.json(job_plan)
    require(spec == receipts.json(sprint / "jobs/primary_training_v4/spec.json")
        and spec.get("entrypoint") == "pipeline.py" and spec.get("args") == ["--plan", str(plan_path)]
        and _path(sprint, spec["job_dir"]) == sprint / "jobs/primary_training_v4"
        and datetime.fromisoformat(spec["not_after_utc"]) == FINAL_FLOOR, "Owning v4 job differs from its frozen specification")
    for dependency in spec.get("wait_for_success", []):
        metadata = receipts.json(dependency["path"])
        require(all(metadata.get(key) == value for key, value in dependency["equals"].items()), "Frozen prerequisite did not succeed")
    stage_dir = sprint / "runs/grpo_v4"
    require(not any(stage_dir.glob("*resume*")), "Release requires the original complete v4 endpoint, not an unplanned resume")
    run, summary, events = receipts.json(stage_dir / "run.json"), receipts.json(stage_dir / "summary.json"), receipts.rows(stage_dir / "events.jsonl")
    old_run = receipts.json(sprint / "runs/grpo/run.json")
    expected_args = {**old_run["args"], "out": str(stage_dir)}
    require(run.get("args") == expected_args and run.get("data") == old_run["data"]
        and run.get("versions") == old_run["versions"] and run.get("mechanical_fixture") is False
        and run.get("source_sha256") == {name: APPROVED["code/" + name] for name in ("modeling.py", "train.py")}, "V4 run differs from fixed RL provenance")
    old_config = receipts.json(sprint / "runs/grpo/trainer_config.json")
    current_config = receipts.json(stage_dir / "trainer_config.json")
    generated_paths = {"output_dir", "logging_dir"}
    require({key: value for key, value in current_config.items() if key not in generated_paths}
        == {key: value for key, value in old_config.items() if key not in generated_paths}
        and current_config.get("output_dir") == str(stage_dir)
        and _path(sprint, current_config.get("logging_dir", "")).is_relative_to(stage_dir / "runs"),
        "RL trainer/loss/sampling settings changed")
    require(summary.get("stage") == "grpo" and summary.get("status") == "complete" and summary.get("mechanical_fixture") is False
        and summary.get("optimizer_steps") == summary.get("requested_steps") == 40 and summary.get("deadline_reached") is False,
        "Incomplete or mechanical v4 GRPO endpoint")
    prefix._same_event(summary, events, "stage_complete")
    identity = prefix._one(events, "model_loaded")["identity"]
    dpo_dir = sprint / "runs/dpo/final"
    require(_path(sprint, identity["adapter_path"]) == dpo_dir
        and identity.get("adapter_sha256") == {name: receipts.digest(dpo_dir / name) for name in ("adapter_model.safetensors", "adapter_config.json")}
        and identity.get("model_id") == prefix.MODEL_ID and identity.get("revision") == prefix.MODEL_REVISION,
        "V4 GRPO did not load the exact frozen DPO/base checkpoint")
    old_identity = prefix._one(receipts.rows(sprint / "runs/grpo/events.jsonl"), "model_loaded")["identity"]
    require(all(identity.get(key) == old_identity.get(key) for key in
        ("config_sha256", "download_manifest_sha256", "rank", "versions", "trainable_parameters")),
        "V4 base/tokenizer/runtime identity differs from the validated prefix")
    prefix._check_audit(identity["adapter_load_audits"][-1], dpo_dir / "adapter_model.safetensors", prefix.PINNED["runs/dpo/final/adapter_model.safetensors"])
    numeric = identity.get("numerical_policy", {})
    require(numeric.get("source_sha256") == NUMERICS_SHA and numeric.get("policy_id") == "nf4_fp32_compute_expanded_gqa_efficient_v4"
        and numeric.get("bnb_linear_count") == 448 and numeric.get("bnb_compute_dtype") == "float32"
        and numeric.get("attention_kernel") == "EFFICIENT_ATTENTION" and numeric.get("native_enable_gqa") is False,
        "Wrong v4 numerical policy")
    require(summary.get("model_identity") == identity and summary.get("protocol") == run["data"]["protocol"], "Endpoint model/protocol identity differs")
    start = prefix._one(events, "stage_start")
    require(start.get("stage") == "grpo" and start.get("rows") == 32 and start.get("steps") == 40
        and start.get("mechanical") is False and start.get("model_identity") == identity, "Wrong RL stage-start identity")
    preserved = prefix._one(events, "trainer_preserved_adapter_values")
    dpo_summary = receipts.json(sprint / "runs/dpo/summary.json")
    require(preserved.get("before") == preserved.get("after") == dpo_summary["adapter_digest"], "TRL initialization perturbed the DPO checkpoint")
    signal = check_rollouts(sprint, receipts, events, summary)
    require(receipts.json(stage_dir / "trainer_state.json").get("global_step") == 40, "Final trainer state is not step40")
    adapter = stage_dir / "final"
    require(_path(sprint, summary["adapter"]) == adapter, "RL endpoint adapter path changed")
    weight = adapter / "adapter_model.safetensors"
    prefix._check_audit(summary["saved_tensor_audit"], weight, receipts.digest(weight))
    prefix._check_tensor_header(weight, summary["saved_tensor_audit"])
    saved_config = receipts.json(adapter / "adapter_config.json")
    starting_config = receipts.json(dpo_dir / "adapter_config.json")
    # PEFT serializes the target-module set in process-dependent order.
    require(set(saved_config.get("target_modules", [])) == set(starting_config.get("target_modules", []))
        and {key: value for key, value in saved_config.items() if key != "target_modules"}
        == {key: value for key, value in starting_config.items() if key != "target_modules"},
        "Saved RL adapter configuration differs from the exact DPO start")
    checkpoint_files = dict(live_prefix["checkpoint_files"])
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        checkpoint_files[str(adapter / name)] = receipts.digest(adapter / name)
    for name, digest in checkpoint_files.items():
        require(receipts.digest(name) == digest, "Released checkpoint changed during validation")
    manifest_path = sprint.parent / "r1-cot-20260909/data/manifest.json"
    manifest = receipts.json(manifest_path)
    relative = "textbook_dev.jsonl" if cohort == "dev" else "sealed_final/textbook_final.jsonl"
    require(manifest["files"][relative] == DATA_SHA[cohort], "Original opaque textbook split identity changed")
    # Construct a name only. Protected files are never accessed here.
    data_path = manifest_path.parent / relative
    completed_paths = [sprint / "runs/sft/summary.json", sprint / "runs/dpo/summary.json", stage_dir / "summary.json", pipeline_path]
    release = {"kind": "textbook_development_release" if cohort == "dev" else "textbook_final_release",
        "utc": now.isoformat(), "allowed_arms": ["base_zero", "base_icl32", "sft", "dpo", "rl"], "settings": dict(SETTINGS),
        "checkpoint_files": checkpoint_files, "checkpoint_map": {"sft": str(sprint / "runs/sft/final"), "dpo": str(dpo_dir), "rl": str(adapter)},
        "completed_training": [_ref(receipts, path) for path in completed_paths],
        "data_path": str(data_path), "data_sha256": DATA_SHA[cohort],
        "opaque_manifest_path": str(manifest_path), "opaque_manifest_sha256": receipts.digest(manifest_path),
        "source_freeze_path": str(freeze_path), "source_freeze_sha256": receipts.digest(freeze_path),
        "training_plan": _ref(receipts, plan_path), "training_job": _ref(receipts, job_path),
        "prefix_receipt": _ref(receipts, prefix_path), "v3_pipeline_success": False,
        "release_implementation": release_implementation, "release_tests": release_tests,
        "completed_lineage": ["v3_sft", "v3_generated_pairs", "v3_dpo", "v4_grpo"], "rl_signal": signal,
        "all_checkpoints_and_evaluation_settings_frozen": True, "no_training_or_setting_selection_after_release": True,
        "protected_payload_read_by_release_builder": False}
    if cohort == "final":
        release["not_before_utc"] = FINAL_FLOOR.isoformat()
    from evaluate import check_textbook_release
    for arm in release["allowed_arms"]:
        endpoint = None if arm.startswith("base_") else Path(release["checkpoint_map"][arm])
        check_textbook_release(release, arm, "textbook_" + cohort, endpoint, now=now)
    receipts.verify_stable()
    release["validated_receipts_sha256"] = dict(sorted(receipts.sha256.items()))
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("dev", "final"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.out is None:
        parser.error("--out is required unless --check-only")
    if args.out is not None and not args.out.resolve().is_relative_to(SPRINT / "plans"):
        parser.error("Release output must stay inside the sprint plans directory")
    release = build_release(args.cohort)
    if not args.check_only:
        with args.out.open("x") as stream:
            json.dump(release, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"cohort": args.cohort, "ready": True, "created": None if args.check_only else str(args.out),
        "checkpoint_files": len(release["checkpoint_files"]), "protected_payload_read": False}))


if __name__ == "__main__":
    main()
