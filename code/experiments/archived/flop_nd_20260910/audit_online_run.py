"""Reconstruct online admissions, reset schedules, rewards and GAE from raw logs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import common
import learning as L
import online_bank as OB
from analyze_lm import jsonl, training_analysis
from nd_environment import Tokenizer, episode, judge, strict_complete
from proof_diagnostics import compress_proof


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_ema(stats, positions, rewards, config):
    """Independent scalar recurrence; ignore old-cut tail after advancement."""
    index = positions.index(stats["cut"])
    for i, reward in enumerate(rewards):
        stats["n"] += 1
        stats["success"] += reward
        alpha = config["curriculum_ema_alpha"]
        stats["ema"] = alpha * reward + (1.0 - alpha) * stats.get("ema", 0.0)
        passed = stats["ema"] > config["backward_threshold"]
        forced = config["curriculum_force_episodes"] > 0 and stats["n"] > config["curriculum_force_episodes"]
        if index and (passed or forced):
            event = {"mode": "flop_ema", "old": stats["cut"], "new": positions[index - 1],
                     "n": stats["n"], "success": stats["success"], "ema": stats["ema"],
                     "trigger": "ema" if passed else "forced_episode_count",
                     "old_cut_group_outcomes_unused_after_advance": len(rewards) - i - 1}
            stats.update(cut=positions[index - 1], n=0, success=0, ema=0.0)
            return event
    return None


def audit(directory):
    directory = Path(directory)
    started = time.monotonic()
    summary = training_analysis(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    require(config["objective"] == "ppo" and config["curriculum_mode"] == "flop_ema", "This audit expects the declared EMA/PPO continuation")
    require(common.sha(directory / "final.pt") == summary["result"]["final_checkpoint_sha256"], "Final checkpoint changed")
    require(common.digest(manifest["identity"]) == manifest["identity_sha256"], "Training identity hash mismatch")
    sources = common.read_jsonl(config["data"])
    targets = common.read_jsonl(config["targets"])
    retention = common.read_jsonl(config["retention"])
    for row in sources + retention:
        verdict = strict_complete(row["prompt"], row["body"])
        require(verdict["ok"] and verdict["n_lines"] <= 6, "External supervision violates the cap-six contract")
    require(not any(set(r) - {"source_id", "name", "prompt"} for r in targets), "Target file carries proof labels")
    tok = Tokenizer()
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    max_seq_len = checkpoint["config"]["max_seq_len"]
    del checkpoint
    bank = OB.OnlineProofBank(targets, tok, max_seq_len=max_seq_len, max_new_tokens=config["max_new_tokens"])
    imported = OB.import_main_discoveries(bank, config["initial_bank"], config["checkpoint"])
    for key in ("imported", "admissions", "manifest_sha256", "later_than_checkpoint_not_admitted", "outside_active_targets_not_admitted"):
        require(imported[key] == manifest["online_import"][key], "Initial proof-bank import differs from recorded provenance")
    expected_admissions = [{"step": 0, **r} for r in imported["admissions"]]
    initial_count = len(bank.records)
    cuts = {r["source_id"]: L.admissible_cuts(r) for r in sources}
    source_stats = {sid: {"cut": max(positions), "n": 0, "success": 0} for sid, positions in cuts.items()}
    online_stats = {sid: {"cut": max(bank.cuts[sid]), "n": 0, "success": 0} for sid in bank.records}
    source_episodes = {(r["source_id"], cut): episode(r, cut, tok) for r in sources for cut in cuts[r["source_id"]]}
    rngs = {key: random.Random(config["seed"] + offset) for key, offset in
            (("source", 0), ("target", 200003), ("mix", 300007), ("reset", 1000033), ("retention", 1000003))}
    group_stream = iter(jsonl(directory / "groups.jsonl"))
    raw_stream = iter(jsonl(directory / "attempts.jsonl"))
    counts, cache, discoveries, witnesses, events = Counter(), {}, {}, {}, []
    schedule_digest = retention_digest = ""
    max_gae_error = 0.0
    for update in jsonl(directory / "updates.jsonl"):
        step = update["step"]
        selected = [("source", sources[rngs["source"].randrange(len(sources))]) for _ in range(manifest["sources_per_update"])]
        selected += [("target", targets[rngs["target"].randrange(len(targets))]) for _ in range(manifest["targets_per_update"])]
        rngs["mix"].shuffle(selected)
        groups, rows_by_group = [], []
        # All group starts are selected before any proof from this batch is admitted.
        for index, (kind, problem) in enumerate(selected):
            group = next(group_stream)
            sid = problem["source_id"]
            draw = rngs["reset"].random()
            can_reset = config["start"] == "reset" and draw >= config["empty_fraction"]
            available = sid in bank.records
            if kind == "source":
                cut = source_stats[sid]["cut"] if can_reset else 0
                expected = source_episodes[sid, cut]
                curriculum_kind = "external_source"
            else:
                cut = online_stats[sid]["cut"] if can_reset and config["online_bank"] and available else 0
                expected = bank.episodes[sid][cut] if cut else {"prompt_ids": tok.encode_prompt(problem["prompt"]) + [tok.prf_id], "prefix_ids": []}
                curriculum_kind = "online_target" if config["online_bank"] and available else "unassisted_target"
            assisted = kind == "target" and cut > 0
            expected_fields = {"step": step, "group": index, "source_id": sid, "training_kind": kind,
                               "reset_draw": draw, "reset_index": cut, "prefix_tokens": len(expected["prefix_ids"]),
                               "bank_available_at_selection": available, "online_assisted": assisted,
                               "curriculum_kind": curriculum_kind, "curriculum_id": f"{curriculum_kind}:{sid}"}
            require(all(group[k] == v for k, v in expected_fields.items()), "Group selection, RNG, reset or bank-availability mismatch")
            rows = [next(raw_stream) for _ in range(config["group_size"])]
            for attempt, raw in enumerate(rows):
                require(raw["step"] == step and raw["group"] == index and raw["attempt_index"] == attempt, "Raw attempt order mismatch")
                require(raw["source_id"] == sid and raw["prompt"] == problem["prompt"] and raw["training_kind"] == kind, "Raw theorem identity mismatch")
                require(raw["prompt_ids"] == expected["prompt_ids"] and raw["prefix_ids"] == expected["prefix_ids"], "Raw context differs from verified selected prefix")
                require(raw["reset_index"] == cut and raw["online_assisted"] == assisted and raw["curriculum_kind"] == curriculum_kind, "Raw assistance metadata mismatch")
                length = len(raw["response_ids"])
                require(0 < length <= config["max_new_tokens"], "Raw response budget exceeded")
                for field in ("rollout_logprobs", "old_values", "token_rewards", "returns", "advantages_raw", "advantages"):
                    require(len(raw[field]) == length and all(math.isfinite(x) for x in raw[field]), f"Invalid action-aligned array: {field}")
                require(raw["terminated"] is True and raw["truncated"] is False and raw["bootstrap_value"] == 0, "Incorrect terminal bootstrapping")
                require(raw["token_rewards"] == [0.0] * (length - 1) + [float(raw["reward"])], "Unexpected shaped/intermediate reward")
                key = common.digest([raw["prompt"], raw["prefix_ids"], raw["response_ids"]])
                if key not in cache:
                    cache[key] = judge(raw, raw["response_ids"], tok)
                verdict = cache[key]
                require(all(verdict[k] == raw[k] for k in ("reward", "body", "n_lines")), "Raw reward/certificate differs from sampled tokens")
                next_value = next_advantage = 0.0
                for t in range(length - 1, -1, -1):
                    delta = raw["token_rewards"][t] + config["gamma"] * next_value - raw["old_values"][t]
                    value = delta + config["gamma"] * config["gae_lambda"] * next_advantage
                    error = max(abs(value - raw["advantages_raw"][t]), abs(value + raw["old_values"][t] - raw["returns"][t]))
                    max_gae_error = max(max_gae_error, error)
                    require(error < 1e-4, "Stored GAE/return differs from independent recurrence")
                    next_value, next_advantage = raw["old_values"][t], value
                counts[kind + "_attempts"] += 1
                counts[kind + "_successes"] += raw["reward"]
                counts["sampled_tokens"] += length
                if kind == "target":
                    prefix = "assisted_target" if assisted else "full_start_target"
                    counts[prefix + "_attempts"] += 1
                    counts[prefix + "_successes"] += raw["reward"]
            require(sum(r["reward"] for r in rows) == group["successes"] and sum(len(r["response_ids"]) for r in rows) == group["generated_tokens"], "Group/raw accounting mismatch")
            groups.append(group)
            rows_by_group.append(rows)
        for group, rows in zip(groups, rows_by_group):
            sid, kind = group["source_id"], group["training_kind"]
            for raw in rows:
                if not raw["reward"]:
                    continue
                require(raw["finish_reason"] == "qed", "Nonterminal response counted as success")
                witness_id = common.digest([raw["prompt"], raw["body"]])
                if witness_id not in witnesses:
                    comp = compress_proof(raw["prompt"], raw["body"])
                    require(strict_complete(raw["prompt"], comp["compressed_proof"])["ok"], "Compressed witness invalid")
                    witnesses[witness_id] = {"source_id": sid, "training_kind": kind, "first_step": step,
                        "written": raw["n_lines"], "pruned": comp["dependency_pruned_length"], "compressed": comp["compressed_length"]}
                if kind != "target":
                    continue
                if witness_id not in discoveries:
                    discoveries[witness_id] = {"step": step, "witness_id": witness_id, "source_id": sid,
                        "prompt": raw["prompt"], "body": raw["body"], "n_lines": raw["n_lines"],
                        "independent_ok": raw.get("independent_ok"), "admitted_for_training": config["online_bank"] and sid not in bank.records,
                        "first_proof_for_target": sid not in bank.records, "external_supervision": False, "online_assisted": group["online_assisted"]}
                if sid not in bank.records:
                    provenance = {"kind": "own_online_rollout", "run_identity_sha256": manifest["identity_sha256"],
                        "starting_checkpoint_sha256": manifest["identity"]["inputs"]["checkpoint"], "attempt_step": step,
                        "attempt_group": group["group"], "attempt_index": raw["attempt_index"], "behavior_completed_local_updates": step - 1}
                    admission = bank.admit(raw, provenance)
                    require(admission is not None, "First target proof was not admitted")
                    expected_admissions.append({"step": step, **admission})
                    online_stats[sid] = {"cut": max(bank.cuts[sid]), "n": 0, "success": 0}
            field, decision = None, None
            if config["start"] == "reset" and kind == "source":
                field = "curriculum_update"
                if group["reset_index"] == source_stats[sid]["cut"]:
                    decision = independent_ema(source_stats[sid], cuts[sid], [r["reward"] for r in rows], config)
            if config["start"] == "reset" and kind == "target" and config["online_bank"] and group["bank_available_at_selection"]:
                field = "online_curriculum_update"
                if group["reset_index"] == online_stats[sid]["cut"]:
                    decision = independent_ema(online_stats[sid], bank.cuts[sid], [r["reward"] for r in rows], config)
            require(group.get("curriculum_update") == (decision if field == "curriculum_update" else None), "Source EMA decision mismatch")
            require(group.get("online_curriculum_update") == (decision if field == "online_curriculum_update" else None), "Target EMA decision mismatch")
            if decision:
                events.append({"step": step, "kind": kind, "source_id": sid, **decision})
        flattened = np.array([a for rows in rows_by_group for row in rows for a in row["advantages_raw"]], dtype=np.float64)
        mean, std = float(flattened.mean()), float(flattened.std())
        denominator = std if std > 1e-8 else 1.0
        scale = denominator if config["advantage_normalization"] != "none" else 1.0
        center = mean if config["advantage_normalization"] == "center_scale" else 0.0
        for rows in rows_by_group:
            for row in rows:
                require(all(math.isclose((a - center) / scale, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(row["advantages_raw"], row["advantages"])), "Advantage normalization mismatch")
        schedule_digest = common.digest([schedule_digest, [(g["training_kind"], g["source_id"], g["reset_index"]) for g in groups]])
        chosen = [retention[rngs["retention"].randrange(len(retention))]["source_id"] for _ in range(config["retention_batch"])] if retention and config["retention_weight"] else []
        retention_digest = common.digest([retention_digest, chosen])
        require(update["schedule_digest"] == schedule_digest and update["retention_schedule_digest"] == retention_digest, "Task or retention RNG schedule digest mismatch")
        require(update["online_bank_size"] == len(bank.records), "Online bank size accounting mismatch")
        require(update["new_first_proof_admissions"] == sum(r["step"] == step for r in expected_admissions), "Admission update accounting mismatch")
    require(next(group_stream, None) is None and next(raw_stream, None) is None, "Uncommitted extra group or attempt")
    stored_bank = json.loads((directory / "bank.json").read_text())
    require(stored_bank["records"] == bank.records, "Final first-proof bank differs from replayed sampled provenance")
    require(stored_bank["online_curriculum"] == online_stats == json.loads((directory / "online_curriculum.json").read_text()), "Final target EMA state mismatch")
    require(source_stats == json.loads((directory / "curriculum.json").read_text()), "Final source EMA state mismatch")
    require(common.read_jsonl(directory / "admissions.jsonl") == expected_admissions, "Admission archive differs from reconstructed first proofs")
    recorded = list(jsonl(directory / "discoveries.jsonl")) if (directory / "discoveries.jsonl").exists() else []
    require(recorded == list(discoveries.values()), "Discovery archive differs from actual successful target responses")
    for kind in ("source", "target"):
        require(counts[kind + "_attempts"] == summary["result"]["attempts"][kind] and counts[kind + "_successes"] == summary["result"]["successes"][kind], "Final raw outcome ledger mismatch")
    require(counts["sampled_tokens"] == summary["result"]["generated_tokens"], "Final token ledger mismatch")
    require(config["online_bank"] or counts["assisted_target_attempts"] == 0, "Disabled online arm received target prefixes")
    return {"status": "passed", "directory": str(directory.resolve()), "counts": dict(counts), "training_summary": summary,
        "initial_bank_count": initial_count, "final_bank_count": len(bank.records), "new_first_proofs": len(bank.records) - initial_count,
        "bank_raw_length_histogram": dict(Counter(r["n_lines"] for r in bank.records.values())),
        "bank_compressed_length_histogram": dict(Counter(compress_proof(r["prompt"], r["body"])["compressed_length"] for r in bank.records.values())),
        "source_curriculum_advances": sum(e["kind"] == "source" for e in events), "target_curriculum_advances": sum(e["kind"] == "target" for e in events),
        "source_final_cuts": dict(Counter(s["cut"] for s in source_stats.values())), "target_final_cuts": dict(Counter(s["cut"] for s in online_stats.values())),
        "independent_gae_max_abs_error": max_gae_error, "unique_raw_verdicts_recomputed": len(cache), "witnesses": list(witnesses.values()),
        "elapsed_seconds": time.monotonic() - started, "code_sha256": common.sha(__file__),
        "input_sha256": {str(p.resolve()): common.sha(p) for p in [directory / f for f in ("manifest.json", "result.json", "attempts.jsonl", "groups.jsonl", "updates.jsonl", "bank.json", "admissions.jsonl", "curriculum.json", "online_curriculum.json")]},
        "scope": "Root audit reconstructs source/target/reset/retention RNGs, all supplied prefixes, every sampled reward, independent GAE recurrence and sign-preserving normalization, first-proof admission provenance, and every EMA advance. Imported proof provenance is rechecked against the completed primary raw archives."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    common.write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("training_summary", "witnesses", "input_sha256")}))
