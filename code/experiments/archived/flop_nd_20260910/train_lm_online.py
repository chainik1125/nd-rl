#!/usr/bin/env python3
"""Matched continuation with optional replay of first online target proofs.

Run `python code/train_lm_online.py --config configs/example.json`. JSON keys match
snake_case CLI flags. Raw attempts contain old behavior probabilities, values,
rewards, advantages and lambda-return targets. final.pt and step_*.pt preserve
the standard policy keys, with critic and complete resume state kept separate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from dataclasses import asdict

import torch

import common
import learning as L
import ppo_core as P
import online_bank as OB
from nd_environment import FORMAT, Tokenizer, episode, judge, split_body, strict_complete

VERSION = "flop-nd-token-ppo-online-v1"


def append_jsonl(path, records):
    with Path(path).open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def training_sources(path):
    records = common.read_jsonl(path)
    if not records:
        raise ValueError(f"No training sources in {path}")
    ids = set()
    for record in records:
        if any(word in str(record.get("role", "")).lower() for word in ("eval", "final", "confirmation", "development", "heldout")):
            raise ValueError("Evaluation-role records cannot enter training")
        prompt = record["prompt"]
        body = record.get("body") or record["text"].split("PRF", 1)[1].strip()
        verdict = strict_complete(prompt, body)
        if not verdict["ok"]:
            raise ValueError(f"Unverified external source: {verdict['reason']}")
        n_lines = len(split_body(body))
        if n_lines > 6:
            raise ValueError("All external source/retention proofs, including premises, must satisfy cap six")
        if "n_lines" in record and record["n_lines"] != n_lines:
            raise ValueError("External proof length metadata disagrees with certificate")
        record.update(body=body, n_lines=n_lines,
                      source_id=record.get("source_id", common.digest([prompt, body])))
        if record["source_id"] in ids:
            raise ValueError("Duplicate external source_id")
        ids.add(record["source_id"])
    return records


def training_targets(path):
    if not path:
        return []
    if any(part.startswith(("final", "development", "confirmation", "heldout"))
           for part in Path(path).name.lower().replace("-", "_").split("_")):
        raise ValueError("Evaluation-named files cannot supply training targets")
    records = common.read_jsonl(path)
    if not records:
        raise ValueError("Targets file is empty")
    forbidden = {"body", "text", "proof", "certificate", "response_ids", "prefix_ids",
                 "n_lines", "source_lines", "reference_length", "reference_lines", "default_cut",
                 "reference", "label", "labels", "answer", "solution"}
    ids = set()
    for row in records:
        if forbidden.intersection(row) or any(k.startswith("reference_") for k in row):
            raise ValueError("Training targets must contain theorem-only inputs, without proof/reference labels")
        if any(word in str(row.get("role", "")).lower() for word in ("eval", "final", "confirmation", "development", "heldout")):
            raise ValueError("Evaluation-role targets cannot enter training")
        if str(row.get("name", "")).lower().startswith(("final_", "development_", "confirmation_", "heldout_")):
            raise ValueError("Evaluation-named targets cannot enter training")
        row.setdefault("source_id", row.get("target_id", common.digest(row["prompt"])))
        if row["source_id"] in ids:
            raise ValueError("Duplicate target source_id")
        ids.add(row["source_id"])
    return records


def target_episode(record, tok):
    original = tok.encode_prompt(record["prompt"])
    # No reference proof is needed or accepted to make a target episode.
    return {"prompt": record["prompt"], "prompt_ids": original + [tok.prf_id],
            "original_prompt_ids": original, "prefix_ids": [], "response_ids": [],
            "reset_index": 0, "source_id": record["source_id"],
            "family": record.get("family", "unknown"), "format": FORMAT,
            "training_kind": "target", "prefix_state": {"goal_already_available": False}}


def build_optimizer(model, critic, args):
    parameters = list(model.parameters()) + (list(critic.parameters()) if critic is not None else [])
    return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay,
                             betas=(0.9, 0.95), fused=next(model.parameters()).is_cuda)


def code_identity():
    files = [Path(__file__), Path(P.__file__), Path(L.__file__), Path(common.__file__),
             Path(__file__).with_name("nd_environment.py"), Path(OB.__file__)]
    return {str(path.resolve()): common.sha(path) for path in files} | common.dependency_hashes()


def run_identity(args):
    # Resource ceilings may be extended explicitly on resume; the model/data and
    # learning protocol cannot change within a resumed run.
    excluded = {"resume", "max_updates", "max_generated_tokens", "max_seconds", "max_flop_proxy",
                "device", "cpu_threads", "log_every", "checkpoint_every"}
    config = {k: v for k, v in vars(args).items() if k not in excluded}
    inputs = {key: common.sha(getattr(args, key))
              for key in ("checkpoint", "data", "retention", "targets", "initial_bank") if getattr(args, key)}
    return {"version": VERSION, "config": config, "inputs": inputs, "code": code_identity()}


def atomic_checkpoint(path, model, critic, tok, optimizer, state, metadata):
    payload = {"format_version": 1, "config": asdict(model.config), "tokenizer": tok.state_dict(),
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "critic": ({k: v.detach().cpu() for k, v in critic.state_dict().items()}
                          if critic is not None else None),
               "metadata": {"format": FORMAT, "implementation": VERSION, **metadata},
               "training_state": {**state, "optimizer": optimizer.state_dict(),
                                  "torch_rng_state": torch.get_rng_state(),
                                  "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def recover_logs(output, checkpoint_step):
    """Remove only uncommitted log tails after the explicitly selected checkpoint."""
    recovered = {}
    for name in ("attempts.jsonl", "groups.jsonl", "updates.jsonl", "discoveries.jsonl", "admissions.jsonl"):
        path = output / name
        if not path.exists():
            continue
        keep = []
        discarded = 0
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                discarded += 1
                continue
            if row["step"] <= checkpoint_step:
                keep.append(row)
            else:
                discarded += 1
        if discarded:
            temporary = path.with_suffix(".recover.tmp")
            common.write_jsonl(temporary, keep)
            temporary.replace(path)
            recovered[name] = discarded
    return recovered


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but it is unavailable")
    if args.device.startswith("cuda"):
        # Use a fixed arithmetic setting across arms. BF16 remains selectable.
        torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ("manifest.json", "attempts.jsonl", "final.pt")):
        raise ValueError("Output already contains a run; use a fresh directory or explicit --resume checkpoint")
    identity = run_identity(args)
    identity_sha = common.digest(identity)
    sources = training_sources(args.data)
    retention_records = training_sources(args.retention) if args.retention else []
    targets = training_targets(args.targets)
    if args.target_fraction > 0 and not targets:
        raise ValueError("Positive target_fraction requires theorem-only targets")
    if args.retention_weight > 0 and (not retention_records or args.retention_batch == 0):
        raise ValueError("Positive retention_weight requires retention records and batch size")
    n_targets = round(args.prompts_per_step * args.target_fraction)
    n_sources = args.prompts_per_step - n_targets
    if 0 < args.target_fraction < 1 and (n_targets == 0 or n_sources == 0):
        raise ValueError("Rounded target fraction removes a population; increase prompts_per_step")
    model, tok, origin_metadata = L.load_model(args.resume or args.checkpoint, args.device)
    critic = P.ValueHead(model.config.d_model).to(args.device) if args.objective == "ppo" else None
    critic_initialization = "zero"
    if critic is not None and not args.resume:
        starting_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if starting_payload.get("critic") is not None:
            critic.load_state_dict(starting_payload["critic"], strict=True)
            critic_initialization = "starting_checkpoint"
        del starting_payload
    optimizer = build_optimizer(model, critic, args)
    # Pre-verify all admissible reset states once. All supplied certificates have
    # passed cap six above, regardless of whether this arm starts full or reset.
    cuts = {r["source_id"]: L.admissible_cuts(r) for r in sources}
    episodes = {(r["source_id"], cut): {**episode(r, cut, tok), "training_kind": "source"}
                for r in sources for cut in cuts[r["source_id"]]}
    target_episodes = {r["source_id"]: target_episode(r, tok) for r in targets}
    retention = [episode(r, 0, tok) for r in retention_records]
    longest_prompt = max(len(e["prompt_ids"]) for e in list(episodes.values()) + list(target_episodes.values()))
    if longest_prompt + args.max_new_tokens > model.config.max_seq_len:
        raise ValueError("Fixed response budget must fit every source reset and target context; no batch-dependent truncation")
    if retention and max(len(e["prompt_ids"]) + len(e["response_ids"]) for e in retention) > model.config.max_seq_len:
        raise ValueError("Retention certificate exceeds model context")
    rngs = {key: random.Random(args.seed + offset) for key, offset in
            (("source", 0), ("target", 200003), ("mix", 300007), ("reset", 1000033), ("retention", 1000003))}
    state = {"step": 0, "optimizer_steps": 0, "generated_tokens": 0, "retention_examples": 0,
             "retention_response_tokens": 0, "elapsed_seconds": 0.0, "ledger": {},
             "identity_sha256": identity_sha, "identity": identity,
             "curriculum": {r["source_id"]: {"cut": max(cuts[r["source_id"]]), "n": 0, "success": 0}
                            for r in sources},
             "schedule_digest": "", "retention_schedule_digest": "", "discoveries": [],
             "online_bank": {}, "online_curriculum": {}, "online_import": None,
             "successes": {"source": 0, "target": 0}, "attempts": {"source": 0, "target": 0},
             "last_budget_config": {k: getattr(args, k) for k in ("max_updates", "max_generated_tokens", "max_seconds", "max_flop_proxy")}}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = payload["training_state"]
        if previous["identity_sha256"] != identity_sha:
            raise ValueError("Resume learning protocol, source identities or code differ from checkpoint")
        optimizer.load_state_dict(previous.pop("optimizer"))
        if critic is not None:
            critic.load_state_dict(payload["critic"], strict=True)
        torch.set_rng_state(previous.pop("torch_rng_state"))
        cuda_states = previous.pop("cuda_rng_states")
        if cuda_states and args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(cuda_states)
        for name, rng in rngs.items():
            rng.setstate(previous["rngs"][name])
        state = previous
        recovery = recover_logs(output, state["step"])
        append_jsonl(output / "resume_events.jsonl", [{"checkpoint": str(Path(args.resume).resolve()),
                       "checkpoint_sha256": common.sha(args.resume), "step": state["step"],
                       "discarded_uncommitted_log_rows": recovery, "budget_config": vars(args),
                       "time_unix": time.time()}])
    bank = OB.OnlineProofBank(targets, tok, max_seq_len=model.config.max_seq_len,
                             max_new_tokens=args.max_new_tokens, records=state["online_bank"])
    if args.initial_bank and not args.resume:
        imported = OB.import_main_discoveries(bank, args.initial_bank, args.checkpoint)
        state["online_import"] = imported
        state["online_bank"] = bank.records
        state["online_curriculum"] = {source_id: {"cut": max(bank.cuts[source_id]), "n": 0, "success": 0}
                                      for source_id in bank.records}
        append_jsonl(output / "admissions.jsonl", [{"step": 0, **row} for row in imported["admissions"]])
    if set(state["online_curriculum"]) != set(bank.records):
        raise ValueError("Online bank and curriculum checkpoint identities differ")
    if any(stats["cut"] not in bank.cuts[source_id] for source_id, stats in state["online_curriculum"].items()):
        raise ValueError("Restored online reset is not admissible within the fixed context budget")
    if not args.resume:
        manifest = {"identity": identity, "identity_sha256": identity_sha, "config": vars(args),
                    "source_count": len(sources), "retention_count": len(retention), "target_count": len(targets),
                    "source_cap": 6, "target_reference_labels_loaded": False,
                    "online_discovery_admission": "First strictly verified full-start target proof, archived in both on/off arms",
                    "online_replay_enabled": args.online_bank, "initial_online_bank_count": len(bank.records),
                    "online_import": state["online_import"],
                    "curriculum_mode": args.curriculum_mode,
                    "curriculum_ema_alpha": args.curriculum_ema_alpha,
                    "curriculum_force_episodes": args.curriculum_force_episodes,
                    "curriculum_deviation": "Batched outcomes stop crediting the old cut after an EMA advance; configured empty_fraction remains a shared intervention.",
                    "continuation_initialization": "Starting policy and saved critic, fresh matched Adam and seeded RNG, external-source curriculum restarted at latest admissible cut; explicit resume restores all states",
                    "critic_initialization": critic_initialization, "sources_per_update": n_sources,
                    "targets_per_update": n_targets, "effective_target_fraction": n_targets / args.prompts_per_step,
                    "policy_parameters": sum(p.numel() for p in model.parameters()),
                    "critic_parameters": sum(p.numel() for p in critic.parameters()) if critic is not None else 0,
                    "origin_metadata": origin_metadata, "longest_prompt_tokens": longest_prompt,
                    "terminal_semantics": "QED and response-budget exhaustion both terminate the proof task; bootstrap zero.",
                    "gae": {"gamma": args.gamma, "lambda": args.gae_lambda},
                    "critic_state_alignment": "final_norm at prompt_length - 1 predicts first sampled token; no future token visible",
                    "normalization": args.loss_normalization,
                    "critic_trunk_gradients": args.objective == "ppo",
                    "retention_objective": "Mean per-example response token NLL, temperature 1, shared across arms",
                    "time_unix": time.time()}
        common.write_json(output / "manifest.json", manifest)
    ledger = state["ledger"]
    stop_requested = {"signal": None}
    original_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        original_handlers[sig] = signal.signal(sig, lambda signum, frame: stop_requested.update(signal=signum))
    start_time = time.monotonic()
    elapsed_before = state["elapsed_seconds"]
    total_rows = args.prompts_per_step * args.group_size
    max_step_tokens = total_rows * args.max_new_tokens
    stop_reason = "update_ceiling"
    discovery_ids = set(state["discoveries"])

    def elapsed():
        return elapsed_before + time.monotonic() - start_time

    def save(path):
        state["elapsed_seconds"] = elapsed()
        state["rngs"] = {key: rng.getstate() for key, rng in rngs.items()}
        state["discoveries"] = sorted(discovery_ids)
        state["online_bank"] = bank.records
        atomic_checkpoint(path, model, critic, tok, optimizer, state,
                          {"step": state["step"], "objective": args.objective, "start": args.start,
                           "source_cap": 6, "identity_sha256": identity_sha,
                           "generated_tokens": state["generated_tokens"]})
        common.write_json(output / "bank.json", {"online_replay_enabled": args.online_bank,
                          "records": bank.records, "online_curriculum": state["online_curriculum"]})
        common.write_json(output / "latest_checkpoint.json", {"path": str(path.resolve()), "step": state["step"],
                          "generated_tokens": state["generated_tokens"], "elapsed_seconds": elapsed()})

    try:
        for step in range(state["step"] + 1, args.max_updates + 1):
            if stop_requested["signal"] is not None:
                stop_reason = f"signal_{stop_requested['signal']}"
                break
            if elapsed() >= args.max_seconds:
                stop_reason = "wall_time_ceiling"
                break
            if state["generated_tokens"] + max_step_tokens > args.max_generated_tokens:
                stop_reason = "generated_token_ceiling_conservative_reservation"
                break
            if L.total_flop_proxy(ledger) >= args.max_flop_proxy:
                stop_reason = "flop_proxy_ceiling"
                break
            step_start = time.monotonic()
            flops_before = L.total_flop_proxy(ledger)
            selected = [("source", sources[rngs["source"].randrange(len(sources))]) for _ in range(n_sources)]
            selected += [("target", targets[rngs["target"].randrange(len(targets))]) for _ in range(n_targets)]
            rngs["mix"].shuffle(selected)
            sampled_episodes = []
            group_info = []
            for group_index, (kind, record) in enumerate(selected):
                source_id = record["source_id"]
                # Draw for every selected task in both on/off arms. Admission is
                # deterministic and cannot perturb source/target/reset RNGs.
                reset_draw = rngs["reset"].random()
                can_reset = args.start == "reset" and reset_draw >= args.empty_fraction
                if kind == "source":
                    cut = state["curriculum"][source_id]["cut"] if can_reset else 0
                    item = episodes[source_id, cut]
                    curriculum_kind = "external_source"
                else:
                    cut = state["online_curriculum"][source_id]["cut"] if can_reset and args.online_bank and source_id in bank.records else 0
                    item = bank.episodes[source_id][cut] if cut else target_episodes[source_id]
                    curriculum_kind = "online_target" if args.online_bank and source_id in bank.records else "unassisted_target"
                sampled_episodes.extend([item] * args.group_size)
                group_info.append({"step": step, "group": group_index, "training_kind": kind,
                                   "source_id": source_id, "family": record.get("family", "unknown"),
                                   "reset_index": cut, "prefix_tokens": len(item["prefix_ids"]),
                                   "curriculum_kind": curriculum_kind, "curriculum_id": f"{curriculum_kind}:{source_id}",
                                   "bank_available_at_selection": source_id in bank.records,
                                   "online_assisted": kind == "target" and cut > 0,
                                   "reset_draw": reset_draw})
            state["schedule_digest"] = common.digest([state["schedule_digest"],
                    [(g["training_kind"], g["source_id"], g["reset_index"]) for g in group_info]])
            rows = L.sample(model, tok, sampled_episodes, temperature=args.temperature,
                            seed=args.seed + step * 997, max_new_tokens=args.max_new_tokens,
                            batch_size=args.batch_size, precision=args.precision, ledger=ledger)
            behavior = P.audit_and_attach_old_values(model, critic, tok, rows, temperature=args.temperature,
                     microbatch=args.microbatch, precision=args.precision,
                     tolerance=args.max_logprob_error, ledger=ledger)
            new_discoveries = []
            new_admissions = []
            for group, (kind, record) in zip(group_info, selected):
                offset = group["group"] * args.group_size
                batch_rows = rows[offset:offset + args.group_size]
                for index, row in enumerate(batch_rows):
                    row.update(judge(row, row["response_ids"], tok))
                    if row["finish_reason"] == "token_limit" and row["reward"]:
                        raise AssertionError("A token-limited response received terminal success")
                    row.update(step=step, group=group["group"], attempt_index=index)
                    row.update(curriculum_kind=group["curriculum_kind"], curriculum_id=group["curriculum_id"],
                               online_assisted=group["online_assisted"])
                    state["attempts"][kind] += 1
                    state["successes"][kind] += row["reward"]
                    if row["reward"] and kind == "target":
                        witness_id = common.digest([row["prompt"], row["body"]])
                        if witness_id not in discovery_ids:
                            discovery_ids.add(witness_id)
                            new_discoveries.append({"step": step, "witness_id": witness_id,
                               "source_id": row["source_id"], "prompt": row["prompt"], "body": row["body"],
                               "n_lines": row["n_lines"], "independent_ok": row.get("independent_ok"),
                               "admitted_for_training": args.online_bank and row["source_id"] not in bank.records,
                               "first_proof_for_target": row["source_id"] not in bank.records,
                               "external_supervision": False, "online_assisted": group["online_assisted"]})
                        if row["source_id"] not in bank.records:
                            provenance = {"kind": "own_online_rollout", "run_identity_sha256": identity_sha,
                                          "starting_checkpoint_sha256": identity["inputs"]["checkpoint"],
                                          "attempt_step": step, "attempt_group": group["group"],
                                          "attempt_index": index, "behavior_completed_local_updates": state["step"]}
                            admission = bank.admit(row, provenance)
                            if admission:
                                new_admissions.append({"step": step, **admission})
                                state["online_curriculum"][row["source_id"]] = {
                                    "cut": max(bank.cuts[row["source_id"]]), "n": 0, "success": 0}
                successes = sum(r["reward"] for r in batch_rows)
                group.update(successes=successes, group_size=args.group_size,
                             mixed=0 < successes < args.group_size,
                             generated_tokens=sum(len(r["response_ids"]) for r in batch_rows))
                if kind == "source" and args.start == "reset":
                    stats = state["curriculum"][record["source_id"]]
                    if group["reset_index"] == stats["cut"]:
                        decision = OB.advance_curriculum(stats, cuts[record["source_id"]],
                            [r["reward"] for r in batch_rows], mode=args.curriculum_mode,
                            window=args.curriculum_rollouts, backward_threshold=args.backward_threshold,
                            forward_threshold=args.forward_threshold, ema_alpha=args.curriculum_ema_alpha,
                            force_episodes=args.curriculum_force_episodes)
                        if decision:
                            group["curriculum_update"] = decision
                if kind == "target" and args.start == "reset" and args.online_bank and group["bank_available_at_selection"]:
                    stats = state["online_curriculum"][record["source_id"]]
                    if group["reset_index"] == stats["cut"]:
                        decision = OB.advance_curriculum(stats, bank.cuts[record["source_id"]],
                            [r["reward"] for r in batch_rows], mode=args.curriculum_mode,
                            window=args.curriculum_rollouts, backward_threshold=args.backward_threshold,
                            forward_threshold=args.forward_threshold, ema_alpha=args.curriculum_ema_alpha,
                            force_episodes=args.curriculum_force_episodes)
                        if decision:
                            group["online_curriculum_update"] = decision
            advantage_stats = P.attach_advantages(rows, objective=args.objective, group_size=args.group_size,
                       gamma=args.gamma, gae_lambda=args.gae_lambda, normalization=args.advantage_normalization)
            retained = [retention[rngs["retention"].randrange(len(retention))] for _ in range(args.retention_batch)] if retention and args.retention_weight else []
            state["retention_schedule_digest"] = common.digest([state["retention_schedule_digest"], [r["source_id"] for r in retained]])
            # Persist actual sampled actions and pre-update learning targets before
            # the update; a resume discards any tail newer than its checkpoint.
            append_jsonl(output / "attempts.jsonl", [{k: v for k, v in r.items() if k != "prefix_state"} for r in rows])
            append_jsonl(output / "groups.jsonl", group_info)
            if new_discoveries:
                append_jsonl(output / "discoveries.jsonl", new_discoveries)
            if new_admissions:
                append_jsonl(output / "admissions.jsonl", new_admissions)
            update = P.update_policy(model, critic, tok, rows, optimizer, retained,
                       objective=args.objective, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                       microbatch=args.microbatch, retention_weight=args.retention_weight,
                       precision=args.precision, ledger=ledger, clip=args.clip,
                       value_clip=args.value_clip, value_coef=args.value_coef,
                       entropy_coef=args.entropy_coef, epochs=args.epochs,
                       loss_normalization=args.loss_normalization, grad_clip=args.grad_clip)
            state["step"] = step
            state["optimizer_steps"] += update["optimizer_steps"]
            state["generated_tokens"] += sum(len(r["response_ids"]) for r in rows)
            state["retention_examples"] += len(retained) * args.epochs
            state["retention_response_tokens"] += sum(len(r["response_ids"]) for r in retained) * args.epochs
            group_kinds = {kind: [g for g in group_info if g["training_kind"] == kind] for kind in ("source", "target")}
            update.update(step=step, generated_tokens_total=state["generated_tokens"],
                optimizer_steps_total=state["optimizer_steps"], elapsed_seconds=elapsed(),
                step_seconds=time.monotonic() - step_start, behavior_audit=behavior,
                advantages=advantage_stats,
                outcome={kind: {"successes": sum(g["successes"] for g in gs),
                                "attempts": len(gs) * args.group_size,
                                "mixed_groups": sum(g["mixed"] for g in gs)} for kind, gs in group_kinds.items()},
                critic_nonzero_target_deltas=sum(v != old for r in rows for v, old in zip(r["returns"], r["old_values"])) if args.objective == "ppo" else 0,
                mean_old_value=sum(sum(r["old_values"]) for r in rows) / sum(len(r["response_ids"]) for r in rows),
                retention_examples=len(retained) * args.epochs,
                retention_response_tokens=sum(len(r["response_ids"]) for r in retained) * args.epochs,
                cumulative_flop_proxy=L.total_flop_proxy(ledger),
                step_flop_proxy=L.total_flop_proxy(ledger) - flops_before,
                schedule_digest=state["schedule_digest"], retention_schedule_digest=state["retention_schedule_digest"])
            update.update(online_bank_enabled=args.online_bank, online_bank_size=len(bank.records),
                          new_first_proof_admissions=len(new_admissions),
                          online_assisted_attempts=sum(g["online_assisted"] for g in group_info) * args.group_size,
                          online_assisted_successes=sum(g["successes"] for g in group_info if g["online_assisted"]),
                          target_full_start_attempts=sum(g["training_kind"] == "target" and not g["online_assisted"] for g in group_info) * args.group_size,
                          target_full_start_successes=sum(g["successes"] for g in group_info if g["training_kind"] == "target" and not g["online_assisted"]))
            append_jsonl(output / "updates.jsonl", [update])
            common.write_json(output / "progress.json", update)
            common.write_json(output / "curriculum.json", state["curriculum"])
            common.write_json(output / "online_curriculum.json", state["online_curriculum"])
            common.write_json(output / "ledger.json", ledger)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(update), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save(output / f"step_{step:06d}.pt")
        save(output / "final.pt")
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
    result = {"objective": args.objective, "start": args.start, "step": state["step"],
              "optimizer_steps": state["optimizer_steps"], "generated_tokens": state["generated_tokens"],
              "retention_examples": state["retention_examples"], "retention_response_tokens": state["retention_response_tokens"],
              "successes": state["successes"], "attempts": state["attempts"], "unique_target_discoveries": len(discovery_ids),
              "online_bank_enabled": args.online_bank, "online_bank_size": len(bank.records),
              "online_import": state["online_import"],
              "elapsed_seconds": elapsed(), "stop_reason": stop_reason,
              "ledger": ledger, "cumulative_flop_proxy": L.total_flop_proxy(ledger),
              "max_flop_proxy": args.max_flop_proxy,
              "flop_proxy_overshoot": max(0, L.total_flop_proxy(ledger) - args.max_flop_proxy),
              "wall_time_overshoot": max(0, elapsed() - args.max_seconds),
              "budget_stop_policy": "Token ceiling reserves one worst-case batch; FLOP/time ceilings finish an in-progress update then stop.",
              "schedule_digest": state["schedule_digest"], "retention_schedule_digest": state["retention_schedule_digest"],
              "identity_sha256": identity_sha, "final_checkpoint_sha256": common.sha(output / "final.pt")}
    common.write_json(output / "result.json", result)
    print(json.dumps({"finished": True, **{k: v for k, v in result.items() if k != "ledger"}}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="JSON settings; command line flags override these values")
    for key in ("checkpoint", "data", "retention", "targets", "output", "resume", "initial_bank"):
        p.add_argument("--" + key.replace("_", "-"), default=None)
    p.add_argument("--online-bank", action="store_true", default=False)
    p.add_argument("--curriculum-mode", choices=("window", "flop_ema"), default="window")
    p.add_argument("--curriculum-ema-alpha", type=float, default=.4)
    p.add_argument("--curriculum-force-episodes", type=int, default=20000)
    p.add_argument("--objective", choices=("relative", "ppo"), default="ppo")
    p.add_argument("--start", choices=("full", "reset"), default="reset")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--max-updates", "--steps", dest="max_updates", type=int, default=100)
    p.add_argument("--max-generated-tokens", type=int, default=10**12)
    p.add_argument("--max-seconds", type=float, default=3600)
    p.add_argument("--max-flop-proxy", type=float, default=1e30)
    for key, default in (("prompts_per_step", 4), ("group_size", 4), ("batch_size", 16),
                         ("microbatch", 8), ("max_new_tokens", 256), ("retention_batch", 8),
                         ("epochs", 1), ("curriculum_rollouts", 16), ("checkpoint_every", 25), ("log_every", 1)):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    for key, default in (("temperature", .8), ("lr", 1e-5), ("weight_decay", 0.),
                         ("retention_weight", .1), ("gamma", 1.), ("gae_lambda", 1.),
                         ("clip", .2), ("value_coef", .5), ("entropy_coef", 0.),
                         ("grad_clip", 1.), ("empty_fraction", .25), ("backward_threshold", .9),
                         ("forward_threshold", 0.), ("target_fraction", 0.), ("max_logprob_error", .001)):
        p.add_argument("--" + key.replace("_", "-"), type=float, default=default)
    p.add_argument("--value-clip", type=float, default=None)
    p.add_argument("--advantage-normalization", choices=("none", "std", "center_scale"), default="none")
    p.add_argument("--loss-normalization", choices=("fixed_budget", "action_tokens"), default="fixed_budget")
    return p


def arguments(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text())
        known = {action.dest for action in p._actions}
        unknown = set(config) - known
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**config)
    args = p.parse_args(argv)
    del args.config
    for name in ("checkpoint", "data", "output"):
        if not getattr(args, name):
            p.error(f"--{name} is required")
    for key in ("checkpoint", "data", "retention", "targets", "output", "resume", "initial_bank"):
        if getattr(args, key):
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
    for key in ("max_updates", "max_generated_tokens", "max_seconds", "max_flop_proxy", "cpu_threads",
                "prompts_per_step", "group_size", "batch_size", "microbatch", "max_new_tokens", "epochs",
                "curriculum_rollouts", "log_every", "lr", "temperature", "grad_clip"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    for key in ("retention_batch", "checkpoint_every", "retention_weight", "value_coef", "entropy_coef", "weight_decay"):
        if getattr(args, key) < 0:
            p.error(f"{key} must be nonnegative")
    for key in ("target_fraction", "empty_fraction", "gamma", "gae_lambda", "backward_threshold", "forward_threshold"):
        if not 0 <= getattr(args, key) <= 1:
            p.error(f"{key} must lie in [0, 1]")
    if args.objective == "relative" and args.group_size < 2:
        p.error("Relative objective requires group_size >= 2")
    if args.forward_threshold > args.backward_threshold:
        p.error("Forward threshold must not exceed backward threshold")
    if not 0 < args.curriculum_ema_alpha <= 1 or args.curriculum_force_episodes < 0:
        p.error("Curriculum EMA alpha must lie in (0,1] and force episodes must be nonnegative")
    return args


if __name__ == "__main__":
    run(arguments())
