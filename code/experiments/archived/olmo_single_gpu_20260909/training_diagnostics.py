"""Read-only snapshots of the four research training stages; CPU only.

This helper never opens evaluation predictions, textbook payloads, or adapter
weights. It writes a new standalone analysis artifact and changes no trainer,
frozen protocol, data membership, checkpoint, or active training log.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from protocol import judge


SPRINT = Path(__file__).resolve().parents[1]
STAGES = ("sft", "pairs", "dpo", "grpo")
INSPECTED_TRAINER_SHA256 = "c3eac8fa28bda6330fd4dee131e7fe7d9bdc0fa35d9de944fce52cb97b379bab"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


class Snapshot:
    """Each permitted file is read once; partially appended JSONL stays pending."""
    def __init__(self):
        self.inputs = []
        self.warnings = []

    def read(self, path: Path, jsonl=False):
        if not path.is_file():
            return [] if jsonl else None
        with path.open("rb") as stream:
            data = stream.read()
        info = {"path": str(path.resolve()), "sha256_of_observed_bytes": sha_bytes(data), "observed_bytes": len(data), "read_utc": datetime.now(timezone.utc).isoformat()}
        self.inputs.append(info)
        if not jsonl:
            try:
                return json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.warnings.append({"path": str(path), "kind": "json_incomplete_or_malformed_at_snapshot"})
                return None
        result = []
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                kind = "incomplete_trailing_append_pending" if index == len(lines) - 1 and not line.endswith(b"\n") else "malformed_committed_jsonl_record"
                self.warnings.append({"path": str(path), "kind": kind, "line": index + 1})
        info["parsed_records"] = len(result)
        return result


def is_mechanical(value):
    return any(value.get(field) is True for field in ("mechanical_fixture", "mechanical_reward_fixture", "mechanical_only", "mechanical"))


def optimizer_summary(events, summaries, trainer_state):
    gradients = [event for event in events if event.get("kind") == "gradient"]
    valid = [event for event in gradients if isinstance(event.get("norm"), (float, int)) and math.isfinite(event["norm"])]
    positive = [event for event in valid if event["norm"] > 0]
    changes = [event for event in events if event.get("kind") == "parameter_update"]
    latest = summaries[-1] if summaries else {}
    curves = defaultdict(list)
    session = -1
    for index, event in enumerate(events):
        if event.get("kind") == "train_begin":
            session += 1
        if event.get("kind") == "gradient" and isinstance(event.get("norm"), (float, int)):
            curves["callback_gradient_norm"].append({"event_index": index, "session": max(session, 0), "step": event.get("step"), "value": event["norm"]})
        if event.get("kind") == "trainer_log":
            for metric, value in event.get("metrics", {}).items():
                if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                    curves[metric].append({"event_index": index, "session": max(session, 0), "step": event.get("step"), "value": value})
    return {
        "latest_summary_optimizer_global_step": latest.get("optimizer_steps"),
        "latest_trainer_state_global_step": (trainer_state or {}).get("global_step"),
        "requested_steps_latest_summary": latest.get("requested_steps"),
        "pre_optimizer_callback_observations": len(gradients),
        "finite_gradient_observations": len(valid),
        "nonzero_gradient_observations": len(positive),
        "zero_gradient_observations": sum(event["norm"] == 0 for event in valid),
        "nonfinite_or_malformed_gradient_observations": len(gradients) - len(valid),
        "distinct_logged_pre_optimizer_step_indices": len({event.get("step") for event in gradients}),
        "distinct_nonzero_pre_optimizer_step_indices": len({event.get("step") for event in positive}),
        "training_invocations_observed": sum(event.get("kind") == "train_begin" for event in events),
        "summary_nonzero_gradient_counts_by_invocation": [summary.get("nonzero_gradient_steps") for summary in summaries],
        "parameter_change_probe_events": len(changes),
        "positive_parameter_change_probe_events": sum(event.get("max_abs_change", 0) > 0 for event in changes),
        "summary_parameter_change_witnesses_by_invocation": [summary.get("observed_parameter_changes") for summary in summaries],
        "final_summary_train_loss": latest.get("metrics", {}).get("train_loss"),
        "curves": dict(curves),
        "limits": [
            "Gradient events occur before optimizer.step; a logged callback alone does not prove that optimizer.step subsequently completed.",
            "The gradient norm is the callback's observed parameter-gradient norm; the helper does not reinterpret its clipping/scaling location.",
            "The callback probes one strongest-gradient tensor until the first positive parameter change, then stops probing. observed_parameter_changes is an existence witness, not the number of all parameter-changing steps.",
            "Nonzero gradients do not by themselves establish nonzero parameter changes; zero current gradients need not prevent optimizer momentum/weight-decay updates.",
            "Global optimizer steps are cumulative trainer indices; resumed invocations reset callback counters. Repeated step indices count as separate actual observations and are not silently deduplicated.",
            "Trainer loss/reward logs use library aggregation windows. A final train_loss scalar is not the final instantaneous loss, and cumulative num_tokens values must not be summed across log records.",
        ],
    }


def epoch_exposure_summary(stage, events, config, summaries, trainer_state, run_count, source_verified, dataset_rows):
    """Exact scalar exposure for this unresumed, single-GPU batch-one recipe."""
    if stage not in ("sft", "dpo"):
        return {"status": "not_applicable_to_this_stage"}
    starts = [event for event in events if event.get("kind") == "stage_start"]
    if not starts or config is None:
        return {"status": "pending_dataset_size_or_trainer_config"}
    size = starts[-1].get("rows")
    latest = summaries[-1] if summaries else {}
    logs = [event for event in events if event.get("kind") == "trainer_log" and isinstance(event.get("metrics", {}).get("epoch"), (int, float))]
    steps = latest.get("optimizer_steps", (trainer_state or {}).get("global_step"))
    if steps is None and logs:
        steps = logs[-1]["step"]
    if steps is None:
        return {"status": "pending_completed_optimizer_step_metadata"}
    batch = config.get("per_device_train_batch_size")
    accumulation = config.get("gradient_accumulation_steps")
    result = {"dataset_rows": size, "reported_optimizer_global_step": steps, "requested_gradient_accumulation": accumulation,
        "configured_per_device_batch_size": batch, "nominal_steps_times_batch_times_accumulation": steps * batch * accumulation if batch and accumulation else None}
    if not source_verified:
        return dict(result, status="unknown_installed_trainer_semantics_not_pinned")
    if run_count != 1 or sum(event.get("kind") == "train_begin" for event in events) != 1:
        return dict(result, status="unknown_actual_compute_exposure_after_resume_or_missing_invocation_markers")
    if batch != 1 or config.get("dataloader_drop_last") is not False or not isinstance(size, int) or size < 1 or not isinstance(accumulation, int) or accumulation < 1:
        return dict(result, status="unknown_outside_inspected_single_gpu_batch_one_configuration")
    if (trainer_state or {}).get("train_batch_size", 1) != 1:
        return dict(result, status="unknown_effective_training_batch_differs_from_one")
    pattern = [min(accumulation, size - offset) for offset in range(0, size, accumulation)]
    full_epochs, partial_updates = divmod(int(steps), len(pattern))
    presentations = full_epochs * size + sum(pattern[:partial_updates])
    expected_epoch = full_epochs + sum(pattern[:partial_updates]) / size
    corroboration = []
    if latest.get("metrics", {}).get("epoch") is not None:
        corroboration.append({"source": "summary", "epoch": latest["metrics"]["epoch"], "step": steps})
    if trainer_state and trainer_state.get("global_step") == steps and trainer_state.get("epoch") is not None:
        corroboration.append({"source": "trainer_state", "epoch": trainer_state["epoch"], "step": steps})
    mismatches = []
    for event in logs:
        epoch_index, remainder = divmod(event["step"], len(pattern))
        implied_epoch = epoch_index + sum(pattern[:remainder]) / size
        # Trainer-facing progress output can round epoch to two decimals.
        if abs(event["metrics"]["epoch"] - implied_epoch) > .00501:
            mismatches.append({"step": event["step"], "reported_epoch": event["metrics"]["epoch"], "expected_epoch": implied_epoch})
    if any(abs(value["epoch"] - expected_epoch) > .00501 for value in corroboration) or mismatches:
        return dict(result, status="unknown_logged_epoch_does_not_match_remainder_schedule", log_epoch_mismatches=mismatches)
    cardinalities = pattern * full_epochs + pattern[:partial_updates]
    family_counts = None
    if dataset_rows is not None and len(dataset_rows) == size and partial_updates == 0:
        family_counts = {family: count * full_epochs for family, count in Counter(row["family"] for row in dataset_rows).items()}
    return {
        **result, "status": "derived_and_epoch_corroborated" if corroboration or logs else "derived_from_pinned_schedule_only",
        "actual_row_presentations": presentations, "actual_optimizer_update_cardinality_histogram": dict(Counter(cardinalities)),
        "optimizer_cardinality_pattern_per_epoch": pattern, "full_epochs": full_epochs, "remaining_optimizer_updates_in_next_epoch": partial_updates,
        "expected_epoch": expected_epoch, "epoch_corroboration": corroboration, "logged_step_epoch_checks": len(logs),
        "dataset_row_presentations_each": full_epochs if partial_updates == 0 else None,
        "family_row_presentations_for_complete_epochs": family_counts,
        "response_presentations": presentations * 2 if stage == "dpo" else presentations,
        "unit": "preference_pairs (each has chosen and rejected responses)" if stage == "dpo" else "supervised proof sequences",
        "limits": [
            "Derivation uses the inspected single-GPU, batch-one, no-drop-last, unresumed dataloader and its exact epoch-end accumulation remainder.",
            "Per-update theorem/family order is not logged. Complete-epoch per-family totals are derivable only when input membership is hash-verified.",
            "DPO chosen/rejected response presentations are two per pair presentation. Frozen-reference precomputation is additional work and is not included in optimizer data presentations.",
            "Trainer train_samples_per_second uses nominal max_steps times total_train_batch_size; it can overstate actual row presentations when accumulation remainders occur.",
        ],
    }


def replay_training_sample(theorem, text, ids, saved, tokenizer=None):
    verdict = judge(theorem, text)
    checks = {"strict_text_replayed": True, "saved_verdict_matches": verdict == saved, "native_decode_replayed": False, "native_text_matches": None}
    if tokenizer is not None and isinstance(ids, list):
        from modeling import decode_completion
        checks["native_decode_replayed"] = True
        checks["native_text_matches"] = decode_completion(tokenizer, ids) == text
    return verdict, checks


def sample_tokens(sample, verdict, tokenizer=None):
    ids = sample.get("response_ids")
    if not isinstance(ids, list):
        return {"missing_native_ids": 1}
    eos = getattr(tokenizer, "eos_token_id", None)
    return {
        "native_generated_tokens": len(ids),
        "failed_sample_native_tokens": 0 if verdict["accepted"] else len(ids),
        "native_prefill_tokens": len(sample["prompt_ids"]) if isinstance(sample.get("prompt_ids"), list) else sample.get("prompt_tokens", 0),
        "eos_native_tokens": sum(value == eos for value in ids) if eos is not None else 0,
        "eos_count_available": int(eos is not None),
        "post_stop_padding_positions": sample.get("post_stop_padding_positions", 0),
        "prompt_padding_positions": sample.get("prompt_padding_positions", 0),
        "logged_completion_count_mismatches": int(sample.get("completion_tokens", len(ids)) != len(ids)),
    }


def pair_summary(samples, pairs, gold, tokenizer=None):
    counts, tokens, finishes, sample_families = Counter(), Counter(), Counter(), Counter()
    rounds = defaultdict(Counter)
    seen_attempts = Counter()
    for row in samples:
        key = row.get("canonical_hash")
        counts["sample_records"] += 1
        seen_attempts[(key, row.get("attempt"))] += 1
        if key not in gold:
            counts["samples_outside_shared_pool"] += 1
            continue
        sample = row["sample"]
        verdict, checks = replay_training_sample(gold[key]["prompt"], sample["text"], sample.get("response_ids"), row.get("verdict"), tokenizer)
        counts["strictly_valid_samples"] += verdict["accepted"]
        counts["strictly_invalid_samples"] += not verdict["accepted"]
        rounds[row.get("attempt")]["samples"] += 1
        rounds[row.get("attempt")]["strictly_valid"] += verdict["accepted"]
        rounds[row.get("attempt")]["strictly_invalid"] += not verdict["accepted"]
        counts["saved_verdict_mismatches"] += not checks["saved_verdict_matches"]
        counts["native_decode_replayed_samples"] += checks["native_decode_replayed"]
        counts["native_text_mismatches"] += checks["native_text_matches"] is False
        finishes[sample.get("finish_reason", "unknown")] += 1
        sample_families[gold[key]["family"]] += 1
        tokens.update(sample_tokens(sample, verdict, tokenizer))
    pair_families, sources = Counter(), Counter()
    pair_ids = Counter()
    rejected_valid = 0
    for row in pairs:
        key = row.get("canonical_hash")
        pair_ids[key] += 1
        if key not in gold:
            counts["pairs_outside_shared_pool"] += 1
            continue
        pair_families[gold[key]["family"]] += 1
        source = (row.get("generation_policy", "unknown"), row.get("source_adapter", "unknown"), row.get("source_adapter_sha256", "unknown"))
        sources[source] += 1
        rejected_valid += judge(gold[key]["prompt"], row["rejected"])["accepted"]
    covered = set(pair_ids) & set(gold)
    coverage_families = Counter(gold[key]["family"] for key in covered)
    pool_families = Counter(row["family"] for row in gold.values())
    return {
        **dict(counts), "pair_records": len(pairs), "covered_shared_theorems": len(covered), "shared_pool": len(gold),
        "coverage_fraction": len(covered) / len(gold) if gold else None,
        "duplicate_pair_records": sum(max(0, count - 1) for count in pair_ids.values()),
        "duplicate_sample_attempt_records": sum(max(0, count - 1) for count in seen_attempts.values()),
        "rejected_responses_that_strictly_verify": rejected_valid,
        "uncovered_shared_hashes": sorted(set(gold) - covered),
        "token_counts": dict(tokens), "finish_counts": dict(finishes), "sample_family_counts": dict(sample_families),
        "sample_attempt_round_counts": {str(attempt): dict(values) for attempt, values in rounds.items()},
        "pair_family_counts": dict(pair_families),
        "family_coverage": {family: {"covered": coverage_families[family], "pool": count} for family, count in sorted(pool_families.items())},
        "pair_sources": [{"generation_policy": policy, "adapter": adapter, "adapter_sha256": digest, "pairs": count} for (policy, adapter, digest), count in sorted(sources.items())],
        "limits": [
            "Pair acquisition adaptively stops sampling a theorem after its first invalid output; aggregate sample validity is not a fixed-budget training-accuracy estimate.",
            "Pair family counts describe the selected preference dataset; source-family draw counts during subsequent DPO optimization are not recorded by the trainer.",
            "The helper replays rejected text but does not replace the trainer's complete chosen-label/token-boundary/provenance validation.",
            "Native counts describe recorded samples only. A batch event can precede per-sample writes; unfinished/unflushed compute cannot be reconstructed completely from these logs.",
        ],
    }


def rollout_summary(rows, events, group_size, gold, tokenizer=None):
    counts, tokens, families, group_types, group_families = Counter(), Counter(), Counter(), Counter(), defaultdict(Counter)
    actual_rows = []
    mechanical_rows = 0
    for row in rows:
        if is_mechanical(row):
            mechanical_rows += 1
            continue
        actual_rows.append(row)
        key = row.get("canonical_hash")
        if key not in gold or row.get("theorem") != gold[key]["prompt"]:
            counts["rollouts_outside_shared_pool_or_prompt_mismatch"] += 1
            continue
        ids = row.get("completion_ids")
        verdict, checks = replay_training_sample(gold[key]["prompt"], row["completion"], ids, row.get("verdict"), tokenizer)
        counts["strictly_valid_rollouts"] += verdict["accepted"]
        counts["strictly_invalid_rollouts"] += not verdict["accepted"]
        counts["saved_verdict_mismatches"] += not checks["saved_verdict_matches"]
        counts["reward_verifier_mismatches"] += row.get("reward") != verdict["reward"]
        counts["native_decode_replayed_rollouts"] += checks["native_decode_replayed"]
        counts["native_text_mismatches"] += checks["native_text_matches"] is False
        families[gold[key]["family"]] += 1
        if isinstance(ids, list):
            tokens["native_generated_tokens"] += len(ids)
            tokens["failed_rollout_native_tokens"] += 0 if verdict["accepted"] else len(ids)
            if tokenizer is not None:
                tokens["eos_native_tokens"] += sum(value == tokenizer.eos_token_id for value in ids)
        else:
            counts["rollouts_missing_native_ids"] += 1
        logprobs = row.get("behavior_logprobs")
        if isinstance(logprobs, list) and isinstance(ids, list):
            counts["behavior_logprob_length_mismatches"] += len(logprobs) != len(ids)
            counts["nonfinite_behavior_logprob_rows"] += any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in logprobs)
    # Batch counters reset across resumed trainer invocations. Group only
    # contiguous call chunks, never merge equal batch numbers across resumes.
    chunks = []
    for row in actual_rows:
        if not chunks or row.get("index") == 0 or row.get("batch") != chunks[-1][0].get("batch"):
            chunks.append([])
        chunks[-1].append(row)
    group_records = []
    for occurrence, chunk in enumerate(chunks):
        if [row.get("index") for row in chunk] != list(range(len(chunk))):
            counts["malformed_indexed_reward_batch_chunks"] += 1
            counts["unclassifiable_rollout_rows"] += len(chunk)
            continue
        complete = len(chunk) // group_size
        counts["trailing_incomplete_group_rows"] += len(chunk) % group_size
        for number in range(complete):
            group = chunk[number * group_size:(number + 1) * group_size]
            keys = {row.get("canonical_hash") for row in group}
            rewards = [row.get("reward") for row in group]
            if len(keys) != 1 or next(iter(keys)) not in gold:
                kind = "invalid_group_membership"
                family = "unknown"
            elif not all(value in (0.0, 1.0) for value in rewards):
                kind = "nonbinary_rewards"
                family = gold[next(iter(keys))]["family"]
            else:
                positives = sum(rewards)
                kind = "all_zero" if positives == 0 else "all_one" if positives == group_size else "mixed"
                family = gold[next(iter(keys))]["family"]
            group_types[kind] += 1
            group_families[family][kind] += 1
            group_records.append({"call_occurrence": occurrence, "logged_batch": chunk[0].get("batch"), "within_call_group": number, "kind": kind, "family": family, "positive_rewards": sum(value == 1 for value in rewards), "size": group_size})
    finish_counts = Counter()
    generation_events = [event for event in events if event.get("kind") == "generation_batch" and not is_mechanical(event)]
    for event in generation_events:
        finish_counts.update(event.get("finish_counts", {}))
    reward_events = [event for event in events if event.get("kind") == "reward_batch" and not is_mechanical(event)]
    classified = sum(group_types[kind] for kind in ("all_zero", "all_one", "mixed"))
    return {
        **dict(counts), "actual_verifier_rollout_records": len(actual_rows), "mechanical_rollout_records_excluded": mechanical_rows,
        "group_size": group_size, "complete_group_counts": {**{kind: group_types[kind] for kind in ("all_zero", "mixed", "all_one")}, **dict(group_types)}, "binary_classified_groups": classified,
        "positive_reward_groups": group_types["mixed"] + group_types["all_one"],
        "mixed_group_fraction": group_types["mixed"] / classified if classified else None,
        "group_family_counts": {family: dict(values) for family, values in sorted(group_families.items())},
        "rollout_family_counts": dict(families), "token_counts_from_recorded_native_ids": dict(tokens),
        "finish_counts_from_generation_events": dict(finish_counts), "generation_event_batches": len(generation_events),
        "generation_event_prompt_tokens": sum(event.get("prompt_tokens", 0) for event in generation_events),
        "generation_event_completion_tokens": sum(event.get("completion_tokens", 0) for event in generation_events),
        "reward_event_examples": sum(event.get("examples", 0) for event in reward_events),
        "reward_event_mixed_groups": sum(event.get("mixed_groups", 0) for event in reward_events),
        "groups_in_record_order": group_records,
        "limits": [
            "A mixed binary-reward group supplies nonconstant group-relative advantages; it does not prove a nonzero gradient or a parameter update.",
            "All-zero/all-one groups give zero centered outcome advantages under this recipe. Optimizer state can still move parameters; these counts alone do not identify all parameter-changing steps.",
            "Reward-call batch numbers restart on trainer resume. Contiguous index-zero chunks identify occurrences, but rollout rows lack optimizer-step and invocation IDs, so no exact rollout-to-optimizer-step join is claimed.",
            "Finish reasons and prefill totals for GRPO are logged at generation-batch level. Individual rollout finish reasons and post-stop padding positions are not saved in rollouts.jsonl.",
            "Events and rollouts are read at different instants. Their totals can temporarily differ while a batch is being appended; mismatches are not silently filled with invented rows.",
            "Recorded behavioral log-probability lengths/finite values are checked; this helper does not rerun the GPU likelihood or gradient audit.",
            "These are training acquisition rollouts, not a fixed-budget evaluation of the shared pool or held-out theorems. A pre-update batch from a few sampled theorems does not estimate overall DPO accuracy.",
        ],
    }


def stage_snapshot(root, stage, snapshot, override=None):
    directory = Path(override).resolve() if override is not None else (root / "runs" / stage).resolve()
    if not directory.is_relative_to((root / "runs").resolve()) or directory == (root / "runs").resolve():
        raise ValueError("Stage directories must be specific runs below this sprint's runs directory")
    if not directory.is_dir():
        return {"status": "pending_no_stage_directory", "directory": str(directory), "events": [], "summaries": [], "runs": [], "trainer_state": None, "trainer_configs": [], "samples": [], "pairs": [], "rollouts": []}
    run_files = [directory / "run.json", *sorted(directory.glob("run_resume_*.json"))]
    summary_files = [directory / "summary.json", *sorted(directory.glob("summary_resume_*.json"))]
    runs = [value for path in run_files if (value := snapshot.read(path)) is not None]
    summaries = [value for path in summary_files if (value := snapshot.read(path)) is not None]
    configs = [value for path in [directory / "trainer_config.json", *sorted(directory.glob("trainer_config_resume_*.json"))] if (value := snapshot.read(path)) is not None]
    events = snapshot.read(directory / "events.jsonl", True)
    mechanical = any(is_mechanical(value) for value in runs + summaries) or any(event.get("kind") == "stage_start" and is_mechanical(event) for event in events)
    status = "excluded_mechanical_stage" if mechanical else summaries[-1].get("status", "summary_present") if summaries else "pending_or_in_progress_records_present"
    numerical_failures = [event for event in events if event.get("kind") == "actual_grpo_likelihood_audit" and (not math.isfinite(event.get("max_abs_error", float("inf"))) or event.get("max_abs_error", float("inf")) > event.get("tolerance", 0))]
    if numerical_failures and not summaries and not mechanical:
        status = "failed_numerical_audit_before_recorded_optimizer_step" if not any(event.get("kind") == "gradient" for event in events) else "failed_numerical_audit_with_prior_optimizer_callback_records"
    if stage == "pairs" and summaries and summaries[-1].get("status") == "partial" and not mechanical:
        status = "completed_acquisition_partial_coverage"
    return {"status": status, "directory": str(directory), "events": events, "summaries": summaries, "runs": runs, "trainer_configs": configs,
        "trainer_state": snapshot.read(directory / "trainer_state.json"),
        "samples": snapshot.read(directory / "pair_samples.jsonl", True) if stage == "pairs" and not mechanical else [],
        "pairs": snapshot.read(directory / "pairs.jsonl", True) if stage == "pairs" and not mechanical else [],
        "rollouts": snapshot.read(directory / "rollouts.jsonl", True) if stage == "grpo" and not mechanical else []}


def build_report(root: Path, native_replay=True, tokenizer=None, stage_directories=None):
    snapshot = Snapshot()
    shared = snapshot.read(root / "data/shared_cap6.jsonl", True)
    gold = {row["canonical_hash"]: row for row in shared}
    stages = {stage: stage_snapshot(root, stage, snapshot, (stage_directories or {}).get(stage)) for stage in STAGES}
    trainer_source = {"status": "unavailable", "matches_inspected_semantics": False}
    package = importlib.util.find_spec("transformers")
    if package and package.origin:
        path = Path(package.origin).parent / "trainer.py"
        content = path.read_bytes()
        identity = sha_bytes(content)
        trainer_source = {"path": str(path), "sha256": identity, "matches_inspected_semantics": identity == INSPECTED_TRAINER_SHA256,
            "epoch_remainder_lines": [2606, 2621], "epoch_progress_line": 2754, "ceil_updates_and_nominal_sample_count_lines": [5682, 5700]}
        snapshot.inputs.append({"path": str(path), "sha256_of_observed_bytes": identity, "observed_bytes": len(content), "purpose": "Verify installed Trainer epoch-remainder and nominal-speed-counter semantics"})
    observed_hashes = {item["path"]: item["sha256_of_observed_bytes"] for item in snapshot.inputs}
    has_generations = bool(stages["pairs"]["samples"] or stages["grpo"]["rollouts"])
    tokenizer_status = "not_needed_no_recorded_generations"
    if has_generations and tokenizer is None and native_replay:
        try:
            from modeling import load_tokenizer
            tokenizer = load_tokenizer(root / "models/olmo32b")
            tokenizer_status = "local_native_tokenizer_loaded_cpu_only"
        except Exception as exc:
            tokenizer_status = "unavailable_" + type(exc).__name__
            snapshot.warnings.append({"kind": "native_replay_unavailable", "exception": type(exc).__name__})
    elif tokenizer is not None:
        tokenizer_status = "provided_cpu_tokenizer"
    elif has_generations:
        tokenizer_status = "disabled_counts_and_strict_text_replay_only"
    report = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "stages": {}, "native_replay": tokenizer_status,
        "shared_training_theorems": len(gold), "shared_training_family_counts": dict(Counter(row["family"] for row in shared)), "installed_trainer_semantics": trainer_source}
    for stage, data in stages.items():
        if data["status"] == "excluded_mechanical_stage":
            report["stages"][stage] = {"status": data["status"], "run_directory": data["directory"], "mechanical_event_records": len(data["events"]), "included_in_research_totals": False}
            continue
        runs, summaries, events = data["runs"], data["summaries"], data["events"]
        entry = {"status": data["status"], "run_directory": data["directory"], "included_in_research_totals": True, "run_invocation_records": len(runs), "summary_invocation_records": len(summaries), "completion_record_present": bool(summaries),
            "event_kind_counts": dict(Counter(event.get("kind", "unknown") for event in events)),
            "optimizer": optimizer_summary(events, summaries, data["trainer_state"]),
            "latest_run_args": runs[-1].get("args", {}) if runs else {},
            "trainer_source_sha256_by_invocation": [run.get("source_sha256", {}).get("train.py") for run in runs]}
        membership = None
        if stage == "sft" and runs and runs[-1].get("data", {}).get("input_sha256", {}).get("shared_cap6.jsonl") == observed_hashes.get(str((root / "data/shared_cap6.jsonl").resolve())):
            membership = shared
        if stage == "dpo" and runs and stages["pairs"]["summaries"]:
            pair_path = (root / (runs[-1].get("args", {}).get("pairs") or "unknown")).resolve()
            expected_pair_path = Path(stages["pairs"]["directory"]) / "pairs.jsonl"
            expected_pair_hash = stages["pairs"]["summaries"][-1].get("pairs_sha256")
            if pair_path == expected_pair_path and expected_pair_hash == observed_hashes.get(str(expected_pair_path)) and all(row["canonical_hash"] in gold for row in stages["pairs"]["pairs"]):
                membership = [gold[row["canonical_hash"]] for row in stages["pairs"]["pairs"]]
        entry["optimizer_data_exposure"] = epoch_exposure_summary(stage, events, data["trainer_configs"][-1] if data["trainer_configs"] else None, summaries, data["trainer_state"], len(runs), trainer_source["matches_inspected_semantics"], membership)
        if stage == "pairs":
            entry["acquisition"] = pair_summary(data["samples"], data["pairs"], gold, tokenizer)
        if stage == "grpo":
            sizes = {int(run.get("args", {}).get("group_size", 4)) for run in runs}
            if len(sizes) > 1:
                entry["rollouts"] = {"status": "unknown_group_size_changed_across_invocations"}
            else:
                entry["rollouts"] = rollout_summary(data["rollouts"], events, next(iter(sizes), 4), gold, tokenizer)
            entry["numerical_audits"] = [{key: value for key, value in event.items() if key not in ("utc", "kind")} for event in events if event.get("kind") == "actual_grpo_likelihood_audit"]
        report["stages"][stage] = entry
    meaningful = any(entry.get("optimizer", {}).get("pre_optimizer_callback_observations", 0) or entry.get("acquisition", {}).get("sample_records", 0) or entry.get("rollouts", {}).get("actual_verifier_rollout_records", 0) for entry in report["stages"].values())
    report["status"] = "training_records_available" if meaningful else "pending_no_research_training_records"
    report["all_research_stages_have_complete_summaries"] = all(entry["status"] == "complete" for entry in report["stages"].values())
    report["all_stage_completion_records_present"] = all(entry.get("completion_record_present", False) for entry in report["stages"].values())
    report["inputs"] = snapshot.inputs
    report["warnings"] = snapshot.warnings
    report["scope"] = {"stages_allowlist": list(STAGES), "heldout_or_textbook_payloads_read": False, "evaluation_predictions_read": False, "adapter_weights_read": False, "gpu_calls": 0, "mechanical_directories_traversed": False,
        "snapshot_limit": "Read-only non-atomic prefix snapshot. Appending logs may advance between reads. No claim of complete total compute for unfinished/unflushed calls.",
        "source_family_limit": "Family labels come only from the frozen shared training pool. SFT/DPO per-update theorem/family identities are not logged. Complete-epoch family totals can be derived when dataset membership is hash-verified and the inspected epoch schedule is corroborated; partial-epoch family membership remains unknown.",
        "nonzero_gradient_limit": "Gradient callback observations and first-change parameter probes are different measurements; observed_parameter_changes must not be interpreted as a count of all successful updates."}
    return report


def plot_report(report, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for stage in ("sft", "dpo", "grpo"):
        curves = report["stages"][stage].get("optimizer", {}).get("curves", {})
        for axis, field in ((axes[0, 0], "loss"), (axes[0, 1], "callback_gradient_norm")):
            values = curves.get(field, [])
            if values:
                axis.plot([value["step"] for value in values], [value["value"] for value in values], label=stage)
    for axis, title in ((axes[0, 0], "Stage-specific losses (different objectives)"), (axes[0, 1], "Observed pre-optimizer gradient norm")):
        axis.set_title(title)
        axis.set_xlabel("Trainer global step (may repeat after resume)")
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend()
        else:
            axis.text(.5, .5, "Pending actual training records", ha="center", transform=axis.transAxes)
    groups = report["stages"]["grpo"].get("rollouts", {}).get("groups_in_record_order", [])
    if groups:
        totals = Counter()
        series = defaultdict(list)
        for group in groups:
            totals[group["kind"]] += 1
            for kind in ("all_zero", "mixed", "all_one"):
                series[kind].append(totals[kind])
        axes[1, 0].stackplot(range(1, len(groups) + 1), *[series[kind] for kind in ("all_zero", "mixed", "all_one")], labels=["all zero", "mixed", "all one"], alpha=.8)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(.5, .5, "Pending actual verifier-RL groups", ha="center", transform=axes[1, 0].transAxes)
    failed_before_update = report["stages"]["grpo"]["status"] == "failed_numerical_audit_before_recorded_optimizer_step"
    axes[1, 0].set_title("Verifier groups before failed audit; 0 optimizer steps" if failed_before_update else "Cumulative verifier reward groups")
    axes[1, 0].set_xlabel("Recorded complete group (not optimizer step)")
    axes[1, 0].set_ylabel("Cumulative number of groups")
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    coverage = report["stages"]["pairs"].get("acquisition", {}).get("family_coverage", {})
    if coverage:
        names = list(coverage)
        axes[1, 1].bar(range(len(names)), [coverage[name]["covered"] for name in names], label="paired")
        axes[1, 1].scatter(range(len(names)), [coverage[name]["pool"] for name in names], marker="_", color="black", label="shared pool")
        axes[1, 1].set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        axes[1, 1].legend()
    axes[1, 1].set_title("Preference-pair coverage by source family")
    axes[1, 1].set_ylabel("Distinct shared training theorems")
    fig.suptitle("Training-only diagnostic snapshot; mechanical runs excluded", fontsize=12)
    names = []
    for extension in ("png", "pdf", "svg"):
        name = "training_diagnostics." + extension
        fig.savefig(out / name, dpi=180, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def markdown_report(report):
    lines = ["# Training-only diagnostic snapshot", "", f"Snapshot: {report['snapshot_utc']}", "", "| Stage | State | Completed global steps | Nonzero gradient observations | First-change witnesses | Actual optimizer row presentations |", "|---|---|---:|---:|---:|---:|"]
    for stage, entry in report["stages"].items():
        optimizer = entry.get("optimizer", {})
        exposure = entry.get("optimizer_data_exposure", {})
        values = [optimizer.get("latest_summary_optimizer_global_step"), optimizer.get("nonzero_gradient_observations"), optimizer.get("positive_parameter_change_probe_events"), exposure.get("actual_row_presentations")]
        lines.append(f"| {stage} | {entry['status']} | " + " | ".join("unknown" if value is None else str(value) for value in values) + " |")
    dpo = report["stages"]["dpo"].get("optimizer_data_exposure", {})
    if "actual_row_presentations" in dpo:
        lines += ["", f"DPO uses {dpo['dataset_rows']} pairs with requested accumulation {dpo['requested_gradient_accumulation']}. Its actual epoch update pattern is {dpo['optimizer_cardinality_pattern_per_epoch']}, producing {dpo['actual_row_presentations']} pair presentations and {dpo['response_presentations']} chosen/rejected response presentations. The nominal steps × batch × accumulation count is {dpo['nominal_steps_times_batch_times_accumulation']} and must not be used as the actual exposure. The installed Trainer's epoch-remainder code and logged epoch values corroborate this reconstruction."]
    pairs = report["stages"]["pairs"].get("acquisition", {})
    if pairs.get("sample_records"):
        lines += ["", f"Adaptive preference acquisition recorded {pairs['sample_records']} samples: {pairs.get('strictly_valid_samples', 0)} strictly valid and {pairs.get('strictly_invalid_samples', 0)} invalid. It produced {pairs['pair_records']} pairs covering {pairs['covered_shared_theorems']}/{pairs['shared_pool']} shared theorems. Round counts: {pairs['sample_attempt_round_counts']}. Sampling stops after each theorem's first invalid response, so aggregate validity is not a fixed-budget accuracy estimate."]
    grpo = report["stages"]["grpo"].get("rollouts", {})
    if grpo.get("actual_verifier_rollout_records"):
        lines += ["", f"The selected GRPO run recorded {grpo['actual_verifier_rollout_records']} rollout rows and {grpo.get('strictly_valid_rollouts', 0)} strictly valid proofs. Complete group counts are {grpo['complete_group_counts']}. Stage state: {report['stages']['grpo']['status']}. These acquisition groups do not establish completed RL training or overall starting-checkpoint accuracy."]
    lines += ["", "The parameter-change callback stops probing after its first positive change. A value of one proves that at least one probed tensor changed; it is not the total number of parameter-changing optimizer steps. Gradient callbacks are separate pre-optimizer observations. Training losses from SFT, DPO and GRPO have different objectives and are not directly comparable on a common scale.", "", "This is a read-only, non-atomic snapshot of explicit training directories. Mechanical runs are excluded. No evaluation predictions, textbook payloads, adapter weights, or GPU execution were used. Exact input-prefix hashes, source semantics, numeric checks, family coverage, native-token counts and pending/failed-stage limits are in training_diagnostics.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SPRINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-native-replay", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--stage-directory", action="append", default=[], metavar="STAGE=PATH", help="Explicitly select a training run directory, e.g. grpo=runs/grpo_v4; never traverses other runs")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh standalone output directory; no artifact is overwritten")
    stage_directories = {}
    for item in args.stage_directory:
        if "=" not in item:
            parser.error("Stage directory overrides must be STAGE=PATH")
        stage, directory = item.split("=", 1)
        if stage not in STAGES or stage in stage_directories:
            parser.error("Unknown or duplicated stage directory override")
        stage_directories[stage] = (args.root.resolve() / directory).resolve()
    report = build_report(args.root.resolve(), native_replay=not args.no_native_replay, stage_directories=stage_directories)
    args.out.mkdir(parents=True)
    report["figures"] = [] if args.no_plots or report["status"].startswith("pending") else plot_report(report, args.out)
    report["helper_sha256"] = sha_bytes(Path(__file__).read_bytes())
    (args.out / "training_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out / "training_diagnostics.md").write_text(markdown_report(report))
    print(json.dumps({"status": report["status"], "out": str(args.out.resolve()), "stage_statuses": {stage: value["status"] for stage, value in report["stages"].items()}, "figures": report["figures"]}, sort_keys=True))


if __name__ == "__main__":
    main()
