"""Group-relative clipped policy gradients on raw natural-deduction tokens.

Only generated RL targets and optional verified cap-six replay receive gradients.
Public validation and generated held-out/transfer sets are evaluation-only.
The sampling policy and policy likelihoods use the same explicit temperature.
The KL penalty is the exact categorical KL(current || frozen Stage-1 anchor),
averaged over response positions and then sequences, rather than a sampled KL
estimator. Old log probabilities are recomputed in float32 before any update;
rollout-time log probabilities are also retained to expose numerical differences.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F

from nd_data import canonical_theorem
from nd_metrics import aggregate_attempts, evaluate_records, safe_verify
from nd_model import (collate_records, encode_records, generate_bodies,
                      load_checkpoint, save_checkpoint)

ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def event(out, kind, **values):
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **values}
    with (Path(out) / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def choose_targets(pool, seed, round_index, count):
    """Shared finite-target schedule: sample without replacement, rounds start at 1."""
    if not 1 <= count <= len(pool):
        raise ValueError("targets-per-round must fit the finite RL pool")
    return random.Random(seed + round_index).sample(pool, count)


def centered_advantages(rewards, normalize=False, baseline="group_mean"):
    """Default group centering, or a batch-wide leave-one-sample-out baseline."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Rewards must have shape [targets, group_size>=2]")
    if baseline == "batch_loo":
        if normalize:
            raise ValueError("batch_loo does not permit standard-deviation normalization")
        return rewards - (rewards.sum() - rewards) / (rewards.numel() - 1)
    if baseline != "group_mean":
        raise ValueError("Unknown advantage baseline")
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize:
        advantages = advantages / rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
    return advantages


@torch.inference_mode()
def rollout_tokens(model, tokenizer, prompts, *, temperature=0.8, seed=0,
                   max_new_tokens=768, batch_size=128):
    """One raw encoded continuation per prompt, retaining malformed model actions."""
    if temperature <= 0 or batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Positive rollout temperature, token limit and batch size required")
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    encoded = [tokenizer.encode_prompt(prompt) for prompt in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    results = [None] * len(prompts)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    autocast = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
    with autocast:
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            lengths = torch.tensor([len(encoded[index]) for index in indices], device=device)
            prompt_length = max(len(encoded[index]) for index in indices)
            limit = min(max_new_tokens, model.config.max_seq_len - prompt_length)
            if limit < 1:
                raise ValueError("Prompt exceeds model context window")
            batch = len(indices)
            tokens = torch.full((batch, prompt_length), tokenizer.pad_id, dtype=torch.long, device=device)
            for row, index in enumerate(indices):
                tokens[row, -len(encoded[index]):] = torch.tensor(encoded[index], device=device)
            columns = torch.arange(prompt_length, device=device)
            valid = columns[None] >= prompt_length - lengths[:, None]
            positions = (columns[None] - prompt_length + lengths[:, None]).clamp_min(0)
            causal = torch.ones((prompt_length, prompt_length), device=device, dtype=torch.bool).tril()
            cache = model.allocate_cache(batch, prompt_length + limit, dtype)
            logits = model(tokens, positions=positions,
                           attention_mask=valid[:, None, None] & causal[None, None], cache=cache)[:, -1]
            valid_keys = torch.ones((batch, prompt_length + limit), device=device, dtype=torch.bool)
            valid_keys[:, :prompt_length] = valid
            done = torch.zeros(batch, device=device, dtype=torch.bool)
            responses = torch.empty((batch, limit), dtype=torch.long, device=device)
            logps = torch.empty((batch, limit), dtype=torch.float32, device=device)
            for step in range(limit):
                scaled_logits = logits.float() / temperature
                distribution = scaled_logits.log_softmax(-1)
                sampled = torch.multinomial(scaled_logits.softmax(-1), 1, generator=rng).squeeze(1)
                selected_logp = distribution.gather(1, sampled[:, None]).squeeze(1)
                sampled = torch.where(done, tokenizer.eos_id, sampled)
                responses[:, step], logps[:, step] = sampled, selected_logp
                done |= sampled == tokenizer.eos_id
                if bool(done.all()) or step + 1 == limit:
                    break
                cache_start = prompt_length + step
                logits = model(sampled[:, None], positions=(lengths + step)[:, None],
                               attention_mask=valid_keys[:, None, None, :cache_start + 1],
                               cache=cache, cache_start=cache_start)[:, -1]
            response_lists = responses[:, :step + 1].cpu().tolist()
            probability_lists = logps[:, :step + 1].cpu().tolist()
            for row, index in enumerate(indices):
                response = response_lists[row]
                if tokenizer.eos_id in response:
                    response = response[:response.index(tokenizer.eos_id) + 1]
                results[index] = {"prompt_ids": encoded[index], "response_ids": response,
                                  "rollout_logprobs": probability_lists[row][:len(response)]}
    return results


def collate_rollouts(rows, pad_id, device):
    """Mask by sequence lengths, never token value: a sampled PAD remains an action."""
    sequences = [row["prompt_ids"] + row["response_ids"] for row in rows]
    if not rows or any(not row["response_ids"] for row in rows):
        raise ValueError("Rollouts must contain at least one sampled response token")
    length = max(map(len, sequences)) - 1
    inputs = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    targets = inputs.clone()
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    old = torch.zeros_like(inputs, dtype=torch.float32)
    for index, (row, sequence) in enumerate(zip(rows, sequences)):
        size = len(sequence) - 1
        inputs[index, :size] = torch.tensor(sequence[:-1], device=device)
        targets[index, :size] = torch.tensor(sequence[1:], device=device)
        start = len(row["prompt_ids"]) - 1
        mask[index, start:size] = True
        if "old_logprobs" in row:
            if len(row["old_logprobs"]) != len(row["response_ids"]):
                raise ValueError("Old probabilities must cover every sampled action")
            old[index, start:size] = torch.tensor(row["old_logprobs"], device=device)
    advantages = torch.tensor([row.get("advantage", 0.0) for row in rows], device=device)
    return {"inputs": inputs, "targets": targets, "mask": mask,
            "old_logprobs": old, "advantages": advantages}


def sequence_mean(values, mask):
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def grpo_objective(log_probs, anchor_log_probs, batch, clip=0.2, kl=0.01):
    chosen = log_probs.gather(-1, batch["targets"][..., None]).squeeze(-1)
    ratio = (chosen - batch["old_logprobs"]).exp()
    advantage = batch["advantages"][:, None]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    pg_loss = -sequence_mean(surrogate, batch["mask"]).mean()
    categorical_kl = (log_probs.exp() * (log_probs - anchor_log_probs)).sum(-1)
    kl_loss = sequence_mean(categorical_kl, batch["mask"]).mean()
    loss = pg_loss + kl * kl_loss
    mask = batch["mask"]
    return loss, {"policy_loss": pg_loss, "anchor_kl": kl_loss,
                  "ratio_mean": sequence_mean(ratio, mask).mean(),
                  "max_abs_ratio_minus_one": (ratio[mask] - 1).abs().max(),
                  "clip_fraction": sequence_mean(((ratio < 1 - clip) | (ratio > 1 + clip)).float(), mask).mean(),
                  "response_tokens": mask.sum()}


@torch.inference_mode()
def score_old_logprobs(model, tokenizer, rows, temperature, batch_size):
    model.eval()
    device = next(model.parameters()).device
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        batch = collate_rollouts(chunk, tokenizer.pad_id, device)
        logp = (model(batch["inputs"]).float() / temperature).log_softmax(-1)
        chosen = logp.gather(-1, batch["targets"][..., None]).squeeze(-1)
        for row, values, mask in zip(chunk, chosen, batch["mask"]):
            row["old_logprobs"] = values[mask].cpu().tolist()


def sample_group(model, tokenizer, targets, args, round_index, role):
    rows, verdicts = [], []
    rewards = torch.zeros(len(targets), args.group_size)
    for attempt in range(args.group_size):
        seed = args.seed * 100000 + round_index * 1000 + attempt
        sampled = rollout_tokens(model, tokenizer, [target["prompt"] for target in targets],
                                  temperature=args.temperature, seed=seed,
                                  max_new_tokens=args.max_new_tokens, batch_size=args.decode_batch)
        bodies = [tokenizer.decode_body(row["response_ids"]) for row in sampled]
        _, judged = evaluate_records(targets, bodies)
        for index, (target, sample, body, verdict) in enumerate(zip(targets, sampled, bodies, judged)):
            reward = float(verdict["solved"])
            rewards[index, attempt] = reward
            rows.append({"name": target["name"], "prompt": target["prompt"],
                         "target_index": index, "attempt": attempt, "sampling_seed": seed,
                         "round": round_index, "role": role, "proof": body, "reward": reward,
                         "reason": verdict["reason"], "found_len": verdict["found_len"], **sample})
            verdicts.append({**verdict, "round": round_index, "attempt": attempt,
                             "sampling_seed": seed, "role": role})
    advantages = centered_advantages(rewards, args.normalize_advantages, args.advantage_baseline)
    for row in rows:
        row["advantage"] = float(advantages[row["target_index"], row["attempt"]])
    score_old_logprobs(model, tokenizer, rows, args.temperature, args.train_batch)
    logprob_errors = [abs(a - b) for row in rows for a, b in zip(row["old_logprobs"], row["rollout_logprobs"])]
    return rows, verdicts, {"targets": len(targets), "groups_with_mixed_rewards": int(((rewards.min(1).values == 0) & (rewards.max(1).values == 1)).sum()),
                            "all_zero_groups": int((rewards.sum(1) == 0).sum()),
                            "all_one_groups": int((rewards.sum(1) == args.group_size).sum()),
                            "reward_mean": float(rewards.mean()), "advantage_mean": float(advantages.mean()),
                            "advantage_rms": float(advantages.square().mean().sqrt()),
                            "advantage_normalization": "none" if args.advantage_baseline == "batch_loo" else "population_std" if args.normalize_advantages else "center_only",
                            "advantage_baseline": args.advantage_baseline,
                            "nonzero_advantage_samples": int((advantages != 0).sum()),
                            "rollout_vs_recomputed_logprob_mean_abs_error": sum(logprob_errors) / len(logprob_errors),
                            "rollout_vs_recomputed_logprob_max_abs_error": max(logprob_errors),
                            "response_tokens": sum(len(row["response_ids"]) for row in rows)}


def serialize_rollouts(directory, role, rows, verdicts):
    write_jsonl(directory / f"{role}.tokens.jsonl", rows)
    write_jsonl(directory / f"{role}.proofs.jsonl", [{key: row[key] for key in
                  ("name", "prompt", "proof", "round", "attempt", "sampling_seed", "reward")} for row in rows])
    write_jsonl(directory / f"{role}.verdicts.jsonl", verdicts)


def update_policy(model, anchor, tokenizer, rows, short_data, optimizer, args, round_index, directory):
    device = next(model.parameters()).device
    model.eval()  # Deterministic dropout setting while keeping gradients enabled.
    anchor.eval()
    rng = random.Random(args.seed + round_index)
    history = []
    for epoch in range(args.update_epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for offset in range(0, len(order), args.train_batch):
            chunk = [rows[index] for index in order[offset:offset + args.train_batch]]
            batch = collate_rollouts(chunk, tokenizer.pad_id, device)
            with torch.no_grad():
                anchor_logp = (anchor(batch["inputs"]).float() / args.temperature).log_softmax(-1)
            logp = (model(batch["inputs"]).float() / args.temperature).log_softmax(-1)
            loss, metrics = grpo_objective(logp, anchor_logp, batch, args.clip, args.kl)
            short_loss = torch.zeros((), device=device)
            short_tokens = 0
            if args.short_coef:
                short_indices = [rng.randrange(len(short_data)) for _ in chunk]
                x, y = collate_records(short_data, short_indices, tokenizer.pad_id, device)
                short_logits = model(x).float()
                short_loss = F.cross_entropy(short_logits.flatten(0, 1), y.reshape(-1))
                short_tokens = int((y != -100).sum())
                loss = loss + args.short_coef * short_loss
            optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite GRPO objective")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("Non-finite GRPO gradient")
            row = {"epoch": epoch + 1, "minibatch": offset // args.train_batch,
                   "loss_before_update": float(loss.detach()), "short_ce_before_update": float(short_loss.detach()),
                   "short_tokens": short_tokens,
                   "objective_tokens": int(metrics["response_tokens"]) + short_tokens,
                   "grad_norm": float(grad_norm), **{key: float(value.detach()) for key, value in metrics.items()}}
            history.append(row)
            with (directory / "optimization.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            optimizer.step()
    return history


def optimization_summary(history):
    return {"updates": len(history),
            "policy_response_tokens": sum(int(row["response_tokens"]) for row in history),
            "cap6_ce_tokens": sum(row["short_tokens"] for row in history),
            "objective_tokens": sum(row["objective_tokens"] for row in history),
            "token_count_definition": "Unweighted objective positions, including repeated epochs; excludes prompt/padding and anchor/old-likelihood forward overhead. This is not a FLOP count.",
            "first_pre_update": history[0], "last_pre_update": history[-1]}


def evaluate_sets(model, tokenizer, data, directory, args):
    result = {}
    for label, path in (("heldout", data / "heldout.jsonl"), ("transfer", data / "transfer.jsonl"),
                        ("validation", ROOT / "targets/validation_36.jsonl")):
        targets = read_jsonl(path)
        bodies = generate_bodies(model, tokenizer, [row["prompt"] for row in targets],
                                 temperature=0, seed=args.seed, max_new_tokens=args.max_new_tokens,
                                 batch_size=args.decode_batch)
        metrics, verdicts = evaluate_records(targets, bodies)
        write_json(directory / "eval" / f"{label}.metrics.json", metrics)
        write_jsonl(directory / "eval" / f"{label}.verdicts.jsonl", verdicts)
        write_jsonl(directory / "eval" / f"{label}.proofs.jsonl", [
            {"name": row["name"], "prompt": row["prompt"], "proof": body} for row, body in zip(targets, bodies)])
        result[label] = metrics
    write_json(directory / "eval/summary.json", result)
    return result


def load_gradient_data(data, short_coef, extra_exclusions=(), exclusion_hash_paths=()):
    """Canonical exclusion includes validation and each generated evaluation role."""
    blocked = set()
    exclusion_paths = [data / "heldout.jsonl", data / "transfer.jsonl", ROOT / "targets/validation_36.jsonl"]
    exclusion_paths.extend(Path(path) for path in extra_exclusions)
    for path in exclusion_paths:
        blocked.update(canonical_theorem(row) for row in read_jsonl(path))
    blocked_hashes = set()
    for path in exclusion_hash_paths:
        values = json.loads(Path(path).read_text())["canonical_theorem_sha256"]
        if not isinstance(values, list) or any(not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value) for value in values):
            raise ValueError("Opaque exclusions must be lowercase SHA256 strings")
        blocked_hashes.update(values)
    def excluded(key):
        return key in blocked or hashlib.sha256(key.encode()).hexdigest() in blocked_hashes
    targets = read_jsonl(data / "rl_targets.jsonl")
    keys = [canonical_theorem(row) for row in targets]
    rl_keys = set(keys)
    if len(rl_keys) != len(keys) or any(excluded(key) for key in keys):
        raise ValueError("RL targets overlap evaluation canonical classes or contain duplicate classes")
    targets = [{key: row[key] for key in ("name", "prompt", "thm", "n_lines", "bin") if key in row} for row in targets]
    short = read_jsonl(data / "train.jsonl") if short_coef else []
    for row in short:
        canonical = canonical_theorem(row)
        if not 2 <= row["n_lines"] <= 6 or excluded(canonical) or canonical in rl_keys:
            raise ValueError("Replay must contain cap-six proofs disjoint from RL targets and evaluation roles")
        text_prompt, body = row["text"].split("PRF", 1)
        if " ".join((text_prompt + " PRF").split()) != " ".join(row["prompt"].split()):
            raise ValueError("Replay text must use the declared prompt")
        ok, reason, length = safe_verify(row["prompt"], body)
        if not ok or length != row["n_lines"]:
            raise ValueError(f"Invalid cap-six replay proof: {reason}")
    return targets, short


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--targets-per-round", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--short-coef", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--decode-batch", type=int, default=128)
    parser.add_argument("--train-batch", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--advantage-baseline", choices=["group_mean", "batch_loo"], default="group_mean")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Additional prompt JSONL used only for canonical membership exclusion; repeatable")
    parser.add_argument("--exclude-hashes", action="append", default=[],
                        help="JSON canonical_theorem_sha256 list for opaque membership exclusion; repeatable")
    args = parser.parse_args()
    run_started = time.monotonic()
    if min(args.rounds, args.targets_per_round, args.update_epochs, args.train_batch,
           args.decode_batch, args.max_new_tokens) < 1 or args.group_size < 2:
        parser.error("Positive round/update/batch counts and group-size >=2 required")
    if args.lr <= 0 or args.temperature <= 0 or args.kl < 0 or args.short_coef < 0 or not 0 < args.clip < 1:
        parser.error("Invalid optimizer, temperature or clipping configuration")
    if args.advantage_baseline == "batch_loo" and args.normalize_advantages:
        parser.error("batch_loo requires no standard-deviation normalization")
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    targets, short_rows = load_gradient_data(data, args.short_coef, args.exclude, args.exclude_hashes)
    if args.targets_per_round > len(targets):
        raise ValueError("Requested target count exceeds RL pool")
    model, tokenizer, metadata = load_checkpoint(args.ckpt)
    anchor, _, _ = load_checkpoint(args.ckpt)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    short_data = encode_records(tokenizer, short_rows, model.config.max_seq_len) if short_rows else None
    source_names = ("nd_grpo.py", "nd_model.py", "prove.py", "nd_data.py", "nd_metrics.py", "nd_verify/verify.py")
    config = {"arguments": vars(args), "source_sha256": {name: sha256(ROOT / name) for name in source_names},
              "data_sha256": {name: sha256(data / name) for name in ("train.jsonl", "heldout.jsonl", "rl_targets.jsonl", "transfer.jsonl")},
              "validation_sha256": sha256(ROOT / "targets/validation_36.jsonl"),
              "additional_exclusion_sha256": {path: sha256(path) for path in args.exclude},
              "opaque_exclusion_sha256": {path: sha256(path) for path in args.exclude_hashes},
              "checkpoint_sha256": sha256(args.ckpt), "checkpoint_metadata": metadata,
              "model_config": asdict(model.config), "torch_version": str(torch.__version__),
              "device": str(next(model.parameters()).device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
              "rollout_precision": "CUDA BF16/FP16 autocast, CPU float32; old denominator recomputed float32 before updating; rollout and recomputed probabilities both retained",
              "algorithm": "Configured reward baseline; clipped per-token surrogate; sequence-mean reduction; exact categorical KL(current||fixed anchor); optional cap6 replay CE; persistent AdamW across rounds; float32 likelihoods/updates.",
              "advantage_definition": "r_i - (sum_all_rewards-r_i)/(N-1) over the complete rollout batch, no std scaling" if args.advantage_baseline == "batch_loo" else "r_i - mean reward within the same theorem group; optional population std scaling",
              "algorithm_reference_inspected": "/data/users/dmitry/nd_sprint/code/train/grpo.py (code only; no external proof data/weights read)",
              "validation_policy": "Evaluation/canonical exclusion only; no validation/reference/test gradients; no test reads."}
    write_json(out / "config.json", config)
    for name in source_names:
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0, fused=device.type == "cuda")
    initial_evaluations = evaluate_sets(model, tokenizer, data, out / "initial", args)
    write_json(out / "initial_eval.json", initial_evaluations)
    event(out, "initial_evaluation_complete", transfer=initial_evaluations["transfer"]["overall"],
          validation=initial_evaluations["validation"]["overall"])
    all_adaptive, all_frozen, trajectory = [], [], []
    for round_index in range(1, args.rounds + 1):
        round_started = time.monotonic()
        directory = out / f"round_{round_index:02d}"
        directory.mkdir()
        selected = choose_targets(targets, args.seed, round_index, args.targets_per_round)
        write_jsonl(directory / "targets.jsonl", selected)
        event(out, "round_start", round=round_index, targets=len(selected), group_size=args.group_size)
        rows, adaptive_verdicts, advantage_summary = sample_group(model, tokenizer, selected, args, round_index, "adaptive")
        serialize_rollouts(directory, "adaptive", rows, adaptive_verdicts)
        if round_index == 1:
            frozen_rows = [{**row, "role": "frozen"} for row in rows]
            frozen_verdicts = [{**row, "role": "frozen"} for row in adaptive_verdicts]
            write_json(directory / "frozen_reuse.json", {"identical_to": "adaptive", "reason": "Same initial weights, prompts, precision and RNG seeds"})
        else:
            frozen_rows, frozen_verdicts, _ = sample_group(anchor, tokenizer, selected, args, round_index, "frozen")
        serialize_rollouts(directory, "frozen", frozen_rows, frozen_verdicts)
        all_adaptive.extend(adaptive_verdicts)
        all_frozen.extend(frozen_verdicts)
        write_json(directory / "advantage_summary.json", advantage_summary)
        event(out, "rollouts_complete", round=round_index, **advantage_summary)
        update_started = time.monotonic()
        history = update_policy(model, anchor, tokenizer, rows, short_data, optimizer, args, round_index, directory)
        update_seconds = time.monotonic() - update_started
        optimization = optimization_summary(history)
        write_json(directory / "optimization_summary.json", optimization)
        save_checkpoint(directory / "model.pt", model, tokenizer, stage="grpo", round=round_index,
                        seed=args.seed, source_checkpoint_sha256=config["checkpoint_sha256"])
        evaluation_started = time.monotonic()
        evaluations = evaluate_sets(model, tokenizer, data, directory, args)
        discovery = {"adaptive": aggregate_attempts(all_adaptive), "frozen": aggregate_attempts(all_frozen)}
        write_json(directory / "discovery.json", discovery)
        trajectory.append({"round": round_index, "greedy": evaluations,
                           "advantage_summary": advantage_summary, "discovery": discovery,
                           "optimization": optimization,
                           "timing_seconds": {"rollout_and_serialization": update_started - round_started,
                                              "update": update_seconds,
                                              "evaluation": time.monotonic() - evaluation_started,
                                              "total": time.monotonic() - round_started}})
        write_json(out / "trajectory.json", trajectory)
        event(out, "round_complete", round=round_index,
              transfer=evaluations["transfer"]["overall"], validation=evaluations["validation"]["overall"])
    shutil.copy2(out / f"round_{args.rounds:02d}/model.pt", out / "final.pt")
    write_jsonl(out / "adaptive_attempts.verdicts.jsonl", all_adaptive)
    write_jsonl(out / "frozen_attempts.verdicts.jsonl", all_frozen)
    write_json(out / "performance.json", {
        "elapsed_seconds": time.monotonic() - run_started,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda})
    write_json(out / "objective_budget.json", {
        "updates": sum(row["optimization"]["updates"] for row in trajectory),
        **{key: sum(row["optimization"][key] for row in trajectory)
           for key in ("policy_response_tokens", "cap6_ce_tokens", "objective_tokens")},
        "token_count_definition": trajectory[0]["optimization"]["token_count_definition"]})
    event(out, "complete")


if __name__ == "__main__":
    main()
