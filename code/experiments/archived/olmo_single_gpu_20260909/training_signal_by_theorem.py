"""Descriptive, CPU-only native replay of repeated theorem visits in GRPO v4.

No release gate, learning-curve claim, checkpoint choice, GPU use, or evaluation
input access. Each invocation writes a fresh snapshot; live files are non-atomic.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from modeling import decode_completion, load_tokenizer, trim_completion
from protocol import SPRINT, judge, protocol_identity, render_prompt

V4_FREEZE_SHA256 = "7f01ae7e6d5938c661a45f3f60c8a7c5f8e14e4a8a553b9ca6147f258bcf4292"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Snapshot:
    def __init__(self):
        self.files = []

    def read(self, path, jsonl=False, expected_sha256=None):
        raw = path.read_bytes()
        if expected_sha256 is not None and digest(raw) != expected_sha256:
            raise ValueError(f"Pinned source digest mismatch: {path}")
        committed = raw[:raw.rfind(b"\n") + 1] if jsonl else raw
        self.files.append({"path": str(path), "read_utc": datetime.now(timezone.utc).isoformat(),
            "observed_bytes": len(raw), "observed_sha256": digest(raw),
            "parsed_prefix_bytes": len(committed), "parsed_prefix_sha256": digest(committed),
            "pending_trailing_bytes": len(raw) - len(committed)})
        if jsonl:
            return [json.loads(line) for line in committed.splitlines() if line.strip()]
        return json.loads(raw)


def event_links(events):
    """Join only the explicit unresumed generation→reward event chain."""
    links, pending, generations = {}, None, []
    last_generation_step = -1
    logs, gradients, changes = [], [], []
    if sum(row.get("kind") == "train_begin" for row in events) > 1:
        raise ValueError("This descriptive helper does not join resumed invocations")
    for index, row in enumerate(events):
        kind = row.get("kind")
        if kind == "generation_batch":
            if pending is not None:
                raise ValueError("Generation receipt has no intervening reward batch")
            step = row["step"]
            if type(step) is not int or step < 0 or step <= last_generation_step:
                raise ValueError("Generation optimizer steps must be strictly increasing nonnegative integers")
            last_generation_step = step
            pending = dict(row, event_index=index)
            generations.append(pending)
        elif kind == "reward_batch":
            if pending is None or row["batch"] in links:
                raise ValueError("Reward receipt has no unique preceding generation receipt")
            links[row["batch"]] = {"generation": pending, "reward": row}
            pending = None
        elif kind == "trainer_log":
            logs.append(row["step"])
        elif kind == "gradient":
            gradients.append(row)
        elif kind == "parameter_update":
            changes.append(row)
    completed = max(logs, default=0)
    return links, {"confirmed_completed_optimizer_steps": completed,
        "completion_evidence": "largest trainer_log global_step; gradient callbacks alone are pre-optimizer",
        "generation_events": len(generations), "reward_events": len(links),
        "generation_without_reward_event": pending,
        "pre_optimizer_gradient_observations": len(gradients),
        "nonzero_gradient_observations": sum(row["norm"] > 0 for row in gradients),
        "parameter_change_witnesses": sum(row["max_abs_change"] > 0 for row in changes),
        "parameter_probe_limit": "First-positive-change-only probe, not a count of every parameter-changing update"}


def replay(rows, events, bank, tokenizer, cap=2048, max_length=4096):
    pool = {row["canonical_hash"]: row for row in bank}
    if len(pool) != len(bank):
        raise ValueError("Duplicate shared theorem identity")
    # GPT2TokenizerFast.__len__ is surprisingly expensive for this vocabulary;
    # compute it once, not once per sampled action. This changes no validation.
    vocabulary_size = len(tokenizer)
    prompt_lengths = {key: len(tokenizer.encode(render_prompt(row["prompt"]), add_special_tokens=False))
                      for key, row in pool.items()}
    links, progress = event_links(events)
    by_batch = defaultdict(list)
    last_batch = -1
    for row in rows:
        batch, index = row["batch"], row["index"]
        if type(batch) is not int or type(index) is not int or batch < 0 or index != len(by_batch[batch]):
            raise ValueError("Rollout rows have repeated or noncontiguous batch indices")
        if batch != last_batch:
            if batch != last_batch + 1:
                raise ValueError("Rollout batch order is not the recorded contiguous invocation order")
            last_batch = batch
        if row["mechanical_reward_fixture"] is not False or row["canonical_hash"] not in pool:
            raise ValueError("Mechanical or out-of-bank rollout")
        source = pool[row["canonical_hash"]]
        if row["theorem"] != source["prompt"]:
            raise ValueError("Theorem text differs from the shared identity")
        ids = row["completion_ids"]
        if not ids or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
            raise ValueError("Invalid native token IDs")
        text = decode_completion(tokenizer, ids)
        trimmed, finish = trim_completion(tokenizer, ids)
        if text != row["completion"] or trimmed != ids:
            raise ValueError("Completion differs from literal native-token replay")
        prompt_tokens = prompt_lengths[row["canonical_hash"]]
        if prompt_tokens + cap > max_length:
            raise ValueError("Unaccounted batch context clipping; this helper expects the frozen cap")
        if len(ids) > cap or (finish == "length" and len(ids) != cap):
            raise ValueError("Native stopping does not respect the frozen completion cap")
        verdict = judge(source["prompt"], text)
        if verdict != row["verdict"] or row["reward"] != verdict["reward"]:
            raise ValueError("Saved verdict/reward differs from strict replay")
        behavior = row["behavior_logprobs"]
        if len(behavior) != len(ids) or not all(math.isfinite(value) for value in behavior):
            raise ValueError("Behavior log-probability metadata mismatch")
        by_batch[batch].append({"canonical_hash": row["canonical_hash"], "index": index,
            "accepted": verdict["accepted"], "proof_lines": verdict["n_lines"] if verdict["accepted"] else None,
            "tokens": len(ids), "prompt_tokens": prompt_tokens, "finish": finish,
            "failure_reason": None if verdict["accepted"] else verdict["reason"]})
    visits, seen = [], Counter()
    incomplete = []
    for batch, values in sorted(by_batch.items()):
        if len(values) > 16:
            raise ValueError("More than sixteen rollouts in a frozen generation batch")
        link = links.get(batch)
        step = link["generation"]["step"] if link else None
        completed = step is not None and step < progress["confirmed_completed_optimizer_steps"]
        lifecycle = "optimizer_completed" if completed else ("generated_not_confirmed_optimized" if link else "rollouts_without_event_join")
        if len(values) == 16 and link:
            generation, reward = link["generation"], link["reward"]
            counts = Counter(value["finish"] for value in values)
            tokens = sum(value["tokens"] for value in values)
            if (generation["completion_tokens"] != tokens or generation["finish_counts"] != dict(counts)
                    or generation["prompt_tokens"] != sum(value["prompt_tokens"] for value in values)
                    or reward["examples"] != 16 or reward["groups"] != 4 or reward["completion_tokens"] != tokens
                    or reward["success_fraction"] != sum(value["accepted"] for value in values) / 16):
                raise ValueError("Native rollout batch differs from its generation/reward receipts")
        batch_mixed = 0
        for offset in range(0, len(values), 4):
            group = values[offset:offset + 4]
            if len(group) < 4:
                incomplete.append({"batch": batch, "group_offset": offset, "observed_responses": len(group)})
                continue
            keys = {value["canonical_hash"] for value in group}
            if len(keys) != 1:
                raise ValueError("Relative-reward group contains different theorems")
            key = group[0]["canonical_hash"]
            seen[key] += 1
            successes = sum(value["accepted"] for value in group)
            classification = "all_zero" if successes == 0 else "all_one" if successes == 4 else "mixed"
            batch_mixed += classification == "mixed"
            visits.append({"canonical_hash": key, "name": pool[key]["name"], "family": pool[key]["family"],
                "gold_lines": pool[key]["n_lines"], "presentation": seen[key], "recorded_group_index": len(visits) + 1,
                "batch": batch, "group_in_batch": offset // 4, "generation_optimizer_step": step,
                "lifecycle": lifecycle, "strict_successes": successes, "responses": 4, "group_class": classification,
                "capped_responses": sum(value["finish"] == "length" for value in group),
                "finish_counts": dict(Counter(value["finish"] for value in group)),
                "response_tokens": [value["tokens"] for value in group],
                "mean_response_tokens": statistics.mean(value["tokens"] for value in group),
                "verified_proof_lines": [value["proof_lines"] for value in group],
                "failure_reasons": dict(Counter(value["failure_reason"] for value in group if not value["accepted"]))})
        if len(values) == 16 and link and link["reward"]["mixed_groups"] != batch_mixed:
            raise ValueError("Mixed-group count differs from reward receipt")
    sweep_blocks = []
    for start in range(0, len(visits), len(bank)):
        group = visits[start:start + len(bank)]
        keys = [value["canonical_hash"] for value in group]
        sweep_blocks.append({"block": start // len(bank) + 1, "complete_group_count": len(group),
            "unique_theorems": len(set(keys)), "repeated_theorems_within_block": len(keys) - len(set(keys)),
            "covers_full_bank_exactly_once": len(keys) == len(bank) and set(keys) == set(pool),
            "canonical_order": keys,
            "definition": "Consecutive blocks of 32 observed groups; coverage checked from identities, not assumed from step count"})
    totals = {}
    for scope in ("all_generated_groups", "optimizer_completed", "generated_not_confirmed_optimized", "rollouts_without_event_join"):
        selected = visits if scope == "all_generated_groups" else [value for value in visits if value["lifecycle"] == scope]
        totals[scope] = {"groups": len(selected), "group_counts": dict(Counter(value["group_class"] for value in selected)),
            "responses": 4 * len(selected), "strict_successes": sum(value["strict_successes"] for value in selected),
            "capped_responses": sum(value["capped_responses"] for value in selected),
            "native_tokens": sum(sum(value["response_tokens"]) for value in selected)}
    return {"progress": progress, "aggregate": totals, "visits": visits, "observed_sweep_blocks": sweep_blocks,
        "native_replayed_responses": len(rows), "incomplete_groups_pending": incomplete,
        "event_batches_without_rollout_rows_in_snapshot": sorted(set(links) - set(by_batch)),
        "theorems_presented_more_than_once": sum(count > 1 for count in seen.values()),
        "presentation_count_histogram": dict(Counter(seen.get(key, 0) for key in pool))}


def markdown(report):
    progress, aggregate = report["progress"], report["aggregate"]
    lines = ["# Repeated presentations of the same training theorems", "",
        f"Snapshot: {report['snapshot_utc']}. Training-only, non-atomic prefix; no held-out measurement.", "",
        f"Confirmed completed optimizer steps: {progress['confirmed_completed_optimizer_steps']}/40, from trainer-log global-step evidence. Replayed native responses: {report['native_replayed_responses']}. Theorems with repeated presentations: {report['theorems_presented_more_than_once']}/32.", "",
        "| Lifecycle | Groups | All zero | Mixed | All one | Valid responses / responses | Capped responses |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in aggregate.items():
        counts = values["group_counts"]
        lines.append(f"| {name} | {values['groups']} | {counts.get('all_zero', 0)} | {counts.get('mixed', 0)} | {counts.get('all_one', 0)} | {values['strict_successes']}/{values['responses']} | {values['capped_responses']} |")
    lines += ["", "## Same-theorem visits", "",
        "Entries within each cell follow that theorem's observed presentation order. Success is strict valid responses out of four; mean output length includes failures and scratch. Verified proof lines list only valid outputs. A trailing `*` on a visit's success count means its optimizer update is not confirmed by the event snapshot.", "",
        "| Training theorem (family, identity) | Successes by visit | Group class by visit | Capped by visit | Mean output tokens by visit | Valid proof-line ranges by visit |",
        "|---|---|---|---|---|---|"]
    for source in report["theorem_inventory"]:
        values = [row for row in report["visits"] if row["canonical_hash"] == source["canonical_hash"]]
        if not values:
            lines.append(f"| {source['family']} / {source['canonical_hash'][:8]} | pending | — | — | — | — |")
            continue
        success = [f"{row['strict_successes']}/4" + ("" if row["lifecycle"] == "optimizer_completed" else "*") for row in values]
        ranges = []
        for row in values:
            lengths = [value for value in row["verified_proof_lines"] if value is not None]
            ranges.append((f"{min(lengths)}–{max(lengths)}" if min(lengths) != max(lengths) else str(lengths[0])) if lengths else "none")
        lines.append(f"| {source['family']} / {source['canonical_hash'][:8]} | {' → '.join(success)} | {' → '.join(row['group_class'] for row in values)} | {' → '.join(str(row['capped_responses']) for row in values)} | {' → '.join(f'{row['mean_response_tokens']:.1f}' for row in values)} | {' → '.join(ranges)} |")
    lines += ["", "## Identity/order and interpretation", ""]
    for block in report["observed_sweep_blocks"]:
        lines.append(f"- Consecutive group block {block['block']}: {block['complete_group_count']} groups, {block['unique_theorems']} distinct training theorems, {block['repeated_theorems_within_block']} within-block repeats; covers all 32 exactly once: {block['covers_full_bank_exactly_once']}.")
    lines += ["", "A mixed binary-reward group provides nonconstant relative rewards, but does not establish a useful gradient, parameter change, or causal learning gain. This is one stochastic training trajectory on repeated prompts, not a held-out learning curve. Changes between visits can reflect sampling variation, batch context, and intervening updates. All-zero and all-one groups supply no within-group binary reward contrast.", "",
        "The explicit generation→reward event chain associates rollout batch IDs with optimizer-step indices. A trainer_log at global step N confirms completed progress through N updates; pre-optimizer gradient callbacks alone do not. Missing cross-file rows/events remain pending because logs are read at different instants. Saved behavior log-probabilities are checked for native length and finiteness; no GPU likelihood or gradient calculation is rerun.", "",
        "No plot is added: the same-theorem table exposes the observed visits, lengths and caps directly without implying a smooth learning curve. Full visit metadata, exact byte-prefix hashes and source identities are in training_signal_by_theorem.json. No checkpoint selection, training change, GPU operation, or evaluation access occurred.", ""]
    return "\n".join(lines)


def run(out):
    out = out.resolve()
    if out.exists() or not out.is_relative_to(SPRINT / "reports") or out == SPRINT / "reports":
        raise ValueError("Use a fresh report directory inside this sprint")
    snapshot = Snapshot()
    helper_hash = digest(Path(__file__).read_bytes())
    freeze_path = SPRINT / "plans/primary_source_freeze_v4.json"
    freeze = snapshot.read(freeze_path, expected_sha256=V4_FREEZE_SHA256)
    # Include the original map itself in the end-of-replay stability recheck.
    static = {str(freeze_path): V4_FREEZE_SHA256}
    def frozen(path):
        raw = path.read_bytes()
        value = digest(raw)
        if freeze["sha256"].get(str(path)) != value:
            raise ValueError(f"Frozen replay input changed: {path}")
        static[str(path)] = value
        return raw
    for name in ("modeling.py", "protocol.py", "train.py", "numerics.py"):
        frozen(SPRINT / "code" / name)
    for relative in protocol_identity()["files"]:
        frozen(SPRINT.parent / relative)
    model_path = SPRINT / "models/olmo32b"
    for name in ("config.json", "download_completed.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt"):
        frozen(model_path / name)
    bank = [json.loads(line) for line in frozen(SPRINT / "data/shared_cap6.jsonl").splitlines() if line.strip()]
    if len(bank) != 32:
        raise ValueError("The fixed training bank must have 32 theorems")
    directory = SPRINT / "runs/grpo_v4"
    run_record = snapshot.read(directory / "run.json")
    config = snapshot.read(directory / "trainer_config.json")
    if run_record["mechanical_fixture"] or run_record["args"]["resume"] is not None or list(directory.glob("run_resume*.json")):
        raise ValueError("Only the actual unresumed v4 training run is supported")
    if run_record["source_sha256"] != {name: static[str(SPRINT / "code" / name)] for name in ("modeling.py", "train.py")}:
        raise ValueError("Training provenance differs from frozen v4 code")
    required = {"num_generations": 4, "generation_batch_size": 16, "gradient_accumulation_steps": 16,
                "steps_per_generation": 16, "num_iterations": 1, "max_steps": 40}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected training schedule for descriptive batch join")
    rows = snapshot.read(directory / "rollouts.jsonl", jsonl=True)
    events = snapshot.read(directory / "events.jsonl", jsonl=True)
    tokenizer = load_tokenizer(model_path)
    report = replay(rows, events, bank, tokenizer)
    for name, value in static.items():
        if digest(Path(name).read_bytes()) != value:
            raise ValueError("Static replay source changed during snapshot")
    if digest(Path(__file__).read_bytes()) != helper_hash:
        raise ValueError("Analysis helper changed during snapshot")
    report.update(snapshot_utc=datetime.now(timezone.utc).isoformat(),
        status="descriptive_non_atomic_training_prefix", static_input_sha256=static, input_prefixes=snapshot.files,
        analysis_helper_sha256=helper_hash,
        theorem_inventory=[{key: row[key] for key in ("canonical_hash", "name", "family", "n_lines")} for row in bank],
        native_token_and_strict_verdict_replay=True, causal_learning_curve=False,
        frozen_files_modified=False, heldout_or_protected_data_read=False)
    out.mkdir(parents=True)
    with (out / "training_signal_by_theorem.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (out / "training_signal_by_theorem.md").open("x") as stream:
        stream.write(markdown(report))
    return {"status": report["status"], "out": str(out), "confirmed_completed_optimizer_steps": report["progress"]["confirmed_completed_optimizer_steps"],
            "native_replayed_responses": len(rows), "repeated_theorems": report["theorems_presented_more_than_once"], "aggregate": report["aggregate"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out), sort_keys=True))
