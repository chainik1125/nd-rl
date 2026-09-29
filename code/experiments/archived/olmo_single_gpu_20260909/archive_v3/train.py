"""Actual TRL SFT/DPO/GRPO, confined to the frozen 32-proof training pool.

The smoke stage uses deliberately artificial rewards solely to exercise the
optimizer. Its adapters are mechanical fixtures, never research checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

from modeling import (adapter_digest, assert_adapter_matches_checkpoint, completion_logprobs,
                      decode_completion, file_sha, load_adapter_exact, load_model,
                      load_tokenizer, memory_snapshot, sample_completions,
                      seed_everything, versions)
from protocol import judge, protocol_identity, render_completion, render_prompt

SPRINT = Path(__file__).resolve().parents[1]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def event(out, kind, **values):
    record = {"utc": utcnow(), "kind": kind, **values}
    with (Path(out) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_training(data_dir):
    """Check release hashes and every gold proof before any GPU work."""
    data_dir = Path(data_dir).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    paths = ("shared_cap6.jsonl", "rl_short_prompts.jsonl", "training_exclusions.json")
    for name in paths:
        if file_sha(data_dir / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Frozen training input changed: {name}")
    gold = read_jsonl(data_dir / paths[0])
    rl = read_jsonl(data_dir / paths[1])
    if len(gold) != 32 or len(rl) != 32 or manifest["x"] != 6:
        raise ValueError("Primary sprint requires exactly 32 matched cap-six examples")
    by_id = {row["canonical_hash"]: row for row in gold}
    if len(by_id) != 32 or set(by_id) != {row["canonical_hash"] for row in rl}:
        raise ValueError("RL and supervised theorem identities differ")
    for row in gold:
        verdict = judge(row["prompt"], render_completion(row["proof"]))
        if not verdict["accepted"] or verdict["n_lines"] != row["n_lines"] or verdict["n_lines"] > 6:
            raise ValueError("Gold proof failed strict replay/cap")
    for row in rl:
        if row["prompt"] != by_id[row["canonical_hash"]]["prompt"]:
            raise ValueError("RL theorem text differs from shared pool")
        if set(row) & {"proof", "text", "answer", "n_lines", "reference_lines"}:
            raise ValueError("RL dataset contains supervised proof/length fields")
    return gold, rl, {"manifest_sha256": file_sha(data_dir / "manifest.json"),
        "input_sha256": {name: file_sha(data_dir / name) for name in paths},
        "rows": 32, "proof_cap": 6, "protocol": protocol_identity()}


def encode_supervision(tokenizer, rows, max_length):
    encoded = []
    for row in rows:
        prefix = render_prompt(row["prompt"])
        answer = render_completion(row["proof"])
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        # Separate tokenization is also used by DPO. Guard the boundary here.
        if tokenizer.encode(prefix + answer, add_special_tokens=False) != prompt_ids + answer_ids[:-1]:
            raise ValueError("Prompt/completion token boundary is not prefix-preserving")
        if len(prompt_ids) + len(answer_ids) > max_length:
            raise ValueError("Supervised example exceeds context; truncation is forbidden")
        encoded.append({"input_ids": prompt_ids + answer_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
            "labels": [-100] * len(prompt_ids) + answer_ids})
    return encoded


class SupervisedCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        width = max(len(row["input_ids"]) for row in rows)
        # Right padding keeps each unpadded sequence's position indices intact.
        return {field: torch.tensor([row[field] + [pad] * (width - len(row[field])) for row in rows])
                for field, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def callbacks(out, deadline=None, reference_name=None, resume_adapter=None):
    import torch
    from transformers import TrainerCallback

    class Audit(TrainerCallback):
        def __init__(self):
            self.nonzero_gradient_steps = 0
            self.weight_changes = 0
            self.before = None
            self.reference_before = None
            self.deadline_reached = False

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if resume_adapter:
                event(out, "exact_resumed_adapter", **assert_adapter_matches_checkpoint(model, resume_adapter))
            if reference_name:
                reference_loads = [item for item in getattr(model, "_sprint_adapter_load_audits", [])
                                   if item["adapter_name"] == reference_name]
                if reference_loads:
                    # Trainer resume must not silently replace the intended
                    # SFT reference with a different saved adapter.
                    assert_adapter_matches_checkpoint(model, Path(reference_loads[-1]["checkpoint"]).parent,
                                                      reference_name)
                self.reference_before = adapter_digest(model, reference_name)
                if not self.reference_before["parameters"]:
                    raise ValueError("Missing frozen reference adapter")
            event(out, "train_begin", step=state.global_step, memory=memory_snapshot())

        def on_step_begin(self, args, state, control, **kwargs):
            if deadline and datetime.now(timezone.utc) >= deadline:
                self.deadline_reached = True
                control.should_training_stop = True
                control.should_save = True
                event(out, "deadline_reached", step=state.global_step)
            return control

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            best = None
            squared_norm = 0.0
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                norm = float(parameter.grad.detach().float().norm())
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite adapter gradient")
                squared_norm += norm * norm
                if best is None or norm > best[0]:
                    best = (norm, name, parameter)
            if squared_norm > 0:
                self.nonzero_gradient_steps += 1
            # One strongest-gradient tensor establishes an actual parameter update.
            if best and best[0] > 0 and self.weight_changes == 0:
                self.before = (best[1], best[2], best[2].detach().clone())
            event(out, "gradient", step=state.global_step, norm=squared_norm**0.5,
                  memory=memory_snapshot())

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.before is not None:
                name, parameter, before = self.before
                maximum = float((parameter.detach().float() - before.float()).abs().max())
                if maximum > 0:
                    self.weight_changes += 1
                event(out, "parameter_update", step=state.global_step, parameter=name, max_abs_change=maximum)
                self.before = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            event(out, "trainer_log", step=state.global_step, metrics=logs or {})

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if reference_name:
                after = adapter_digest(model, reference_name)
                if self.reference_before != after:
                    raise RuntimeError("DPO reference adapter changed during optimization")
                event(out, "reference_frozen", before=self.reference_before, after=after)

    return Audit()


def config_common(args, out):
    return dict(output_dir=str(out), per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        max_steps=args.steps, bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch",
        lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0,
        max_grad_norm=1.0, logging_steps=1, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=None, save_only_model=False, report_to="none", push_to_hub=False,
        # Transformers forwards this seed to NumPy's legacy RNG, which has a
        # narrower seed range than the frozen protocol seed.
        seed=args.seed % (2**32), data_seed=args.seed % (2**32), dataloader_num_workers=0,
        remove_unused_columns=False, disable_tqdm=True)


def reward_function(out, group_size, synthetic=False):
    call_index = 0

    def verifier_reward(prompts, completions, theorem, canonical_hash=None, completion_ids=None, **kwargs):
        nonlocal call_index
        if len(completions) % group_size:
            raise ValueError("Reward batch is not made of complete groups")
        for offset in range(0, len(completions), group_size):
            if len(set(theorem[offset:offset + group_size])) != 1 or (canonical_hash is not None and
                    len(set(canonical_hash[offset:offset + group_size])) != 1):
                raise ValueError("A relative-reward group contains different theorems")
        verdicts = [judge(query, completion) for query, completion in zip(theorem, completions, strict=True)]
        rewards = [float(index % group_size == 0) for index in range(len(completions))] if synthetic else [v["reward"] for v in verdicts]
        groups = [rewards[index:index + group_size] for index in range(0, len(rewards), group_size)]
        mixed = sum(min(group) != max(group) for group in groups)
        with (Path(out) / "rollouts.jsonl").open("a") as stream:
            for index, (query, completion, verdict, reward) in enumerate(zip(theorem, completions, verdicts, rewards, strict=True)):
                stream.write(json.dumps({"batch": call_index, "index": index,
                    "canonical_hash": canonical_hash[index] if canonical_hash else None,
                    "theorem": query, "completion": completion, "reward": reward,
                    "verdict": verdict, "completion_ids": completion_ids[index] if completion_ids is not None else None,
                    "behavior_logprobs": kwargs.get("behavior_logprobs", [None] * len(completions))[index],
                    "mechanical_reward_fixture": synthetic}, sort_keys=True) + "\n")
        event(out, "reward_batch", batch=call_index, examples=len(rewards), groups=len(groups),
            mixed_groups=mixed, mixed_group_fraction=mixed / len(groups),
            success_fraction=sum(v["accepted"] for v in verdicts) / len(verdicts),
            reward_mean=sum(rewards) / len(rewards), mechanical_reward_fixture=synthetic,
            completion_tokens=sum(len(ids) for ids in completion_ids) if completion_ids is not None else None)
        call_index += 1
        return rewards

    return verifier_reward


def marker_grpo_class():
    """Keep TRL's optimizer/loss, replace only raw generation and stop metrics."""
    from trl import GRPOTrainer

    class MarkerGRPOTrainer(GRPOTrainer):
        def _generate_single_turn(self, prompts):
            if self.tools or self.use_vllm or not all(isinstance(p, str) for p in prompts):
                raise ValueError("Sprint sampler supports one GPU and raw single-turn completions only")
            step = self.state.global_step
            if getattr(self, "_sprint_sampling_step", None) != step:
                self._sprint_sampling_step, self._sprint_sampling_count = step, 0
            seed = self.args.seed + 1000003 * step + self._sprint_sampling_count
            self._sprint_sampling_count += 1
            self._sprint_samples = sample_completions(self.model, self.processing_class, prompts,
                max_new_tokens=self.max_completion_length, temperature=self.temperature,
                batch_size=len(prompts), seed=seed, max_seq_len=self._sprint_max_length,
                return_logprobs=True)
            return ([row["prompt_ids"] for row in self._sprint_samples],
                [row["response_ids"] for row in self._sprint_samples], None,
                {"behavior_logprobs": [row["rollout_logprobs"] for row in self._sprint_samples]})

        def _generate(self, prompts):
            result = list(super()._generate(prompts))
            # Parent decoding strips arbitrary special tokens. Literal scoring
            # must retain them, and only an actual terminal EOS is omitted.
            result[3] = [row["text"] for row in self._sprint_samples]
            mode = "train" if self.model.training else "eval"
            metrics = self._metrics[mode]
            finished = [row for row in self._sprint_samples if row["finish_reason"] != "length"]
            lengths = [row["completion_tokens"] for row in finished] or [0]
            metrics["completions/clipped_ratio"][-1] = 1 - len(finished) / len(self._sprint_samples)
            for key, value in (("mean", sum(lengths) / len(lengths)), ("min", min(lengths)), ("max", max(lengths))):
                metrics[f"completions/{key}_terminated_length"][-1] = value
            event(self._sprint_out, "generation_batch", step=self.state.global_step,
                finish_counts=dict(Counter(row["finish_reason"] for row in self._sprint_samples)),
                prompt_tokens=sum(row["prompt_tokens"] for row in self._sprint_samples),
                completion_tokens=sum(row["completion_tokens"] for row in self._sprint_samples))
            return tuple(result)

        def _generate_and_score_completions(self, inputs):
            import torch
            output = super()._generate_and_score_completions(inputs)
            behavior = torch.zeros_like(output["completion_ids"], dtype=torch.float32)
            for index, row in enumerate(self._sprint_samples):
                count = len(row["response_ids"])
                if len(row["rollout_logprobs"]) != count or int(output["completion_mask"][index].sum()) != count:
                    raise ValueError("Behavior likelihood and TRL action mask disagree")
                if not all(math.isfinite(value) for value in row["rollout_logprobs"]):
                    raise FloatingPointError("Nonfinite sampled behavior likelihood")
                behavior[index, :count] = torch.tensor(row["rollout_logprobs"], device=behavior.device)
            # Actual sampling probabilities, rather than recomputed teacher
            # probabilities, define the PPO denominator on every rollout.
            output["old_per_token_logps"] = behavior
            if not getattr(self, "_sprint_likelihood_audited", False):
                input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
                attention = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
                with torch.no_grad(), self.compute_loss_context_manager():
                    teacher, _ = self._get_per_token_logps_and_entropies(self.model, input_ids, attention,
                        output["completion_ids"].shape[1], batch_size=self.args.per_device_train_batch_size)
                delta = (teacher.float() - behavior).abs()[output["completion_mask"].bool()]
                maximum = float(delta.max())
                event(self._sprint_out, "actual_grpo_likelihood_audit", step=self.state.global_step,
                    tokens=delta.numel(), max_abs_error=maximum, mean_abs_error=float(delta.mean()),
                    p99_abs_error=float(torch.quantile(delta, 0.99)),
                    tolerance=self._sprint_max_logprob_error, scorer_model_training=self.model.training,
                    finish_counts=dict(Counter(row["finish_reason"] for row in self._sprint_samples)),
                    denominator="actual_streamed_sampling_logprobs")
                if not math.isfinite(maximum) or maximum > self._sprint_max_logprob_error:
                    raise RuntimeError("Actual GRPO behavior/train-scorer likelihood audit failed")
                self._sprint_likelihood_audited = True
            return output

    return MarkerGRPOTrainer


def snapshot_adapters(model):
    """Keep exact incoming FP32 values across TRL's lossy constructor cast."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if "lora_" in name}


def restore_adapters(model, snapshot):
    import torch
    parameters = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
    if set(parameters) != set(snapshot):
        raise RuntimeError("TRL constructor changed the set of adapter parameters")
    for name, parameter in parameters.items():
        # Preserve Parameter object identity; the optimizer is created in train().
        parameter.data = snapshot[name].to(device=parameter.device, dtype=torch.float32)


def train_stage(args, model, tokenizer, rows, out, stage, mechanical=False):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit = callbacks(out, args.deadline, "reference" if stage == "dpo" else None, args.resume)
    common = config_common(args, out)
    model.train()
    incoming_digest = adapter_digest(model)
    exact_adapters = snapshot_adapters(model)
    if stage == "sft":
        examples = encode_supervision(tokenizer, rows, args.max_length)
        config = SFTConfig(**common, max_length=None, packing=False,
            dataset_kwargs={"skip_prepare_dataset": True}, completion_only_loss=False)
        trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer,
            train_dataset=Dataset.from_list(examples), data_collator=SupervisedCollator(tokenizer.pad_token_id),
            callbacks=[audit])
    elif stage == "dpo":
        config = DPOConfig(**common, max_length=None, max_prompt_length=None,
            model_adapter_name="default", ref_adapter_name="reference", beta=args.dpo_beta,
            loss_type="sigmoid", disable_dropout=True, precompute_ref_log_probs=True,
            precompute_ref_batch_size=1, use_logits_to_keep=True)
        trainer = DPOTrainer(model=model, args=config, processing_class=tokenizer,
            train_dataset=Dataset.from_list(rows), callbacks=[audit])
    elif stage == "grpo":
        if args.accumulation % args.group_size:
            raise ValueError("Accumulation must be divisible by group size on one GPU")
        for row in rows:
            if len(tokenizer.encode(render_prompt(row["prompt"]), add_special_tokens=False)) + args.max_completion_length > args.max_length:
                raise ValueError("RL prompt plus completion budget exceeds context; no truncation")
        config = GRPOConfig(**common, max_prompt_length=None, max_completion_length=args.max_completion_length,
            num_generations=args.group_size, steps_per_generation=args.accumulation,
            num_iterations=1, beta=0.0, loss_type="dr_grpo", scale_rewards=False,
            temperature=1.0, top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0,
            disable_dropout=True, use_vllm=False, mask_truncated_completions=False,
            generation_kwargs={"use_cache": True, "logits_to_keep": 1},
            epsilon=0.2, importance_sampling_level="token")
        examples = [{"prompt": render_prompt(row["prompt"]), "theorem": row["prompt"],
                     "canonical_hash": row["canonical_hash"]} for row in rows]
        trainer = marker_grpo_class()(model=model, args=config, processing_class=tokenizer,
            reward_funcs=reward_function(out, args.group_size, mechanical),
            train_dataset=Dataset.from_list(examples), callbacks=[audit])
        trainer._sprint_max_length = args.max_length
        trainer._sprint_out = out
        trainer._sprint_max_logprob_error = args.max_logprob_error
    else:
        raise ValueError(stage)
    # Casting back to FP32 alone cannot recover values rounded to BF16. Restore
    # the exact incoming tensors, ensuring DPO→RL has no initialization change.
    restore_adapters(model, exact_adapters)
    del exact_adapters
    restored_digest = adapter_digest(model)
    if incoming_digest != restored_digest:
        raise RuntimeError("Trainer initialization perturbed the starting checkpoint")
    event(out, "trainer_preserved_adapter_values", before=incoming_digest, after=restored_digest)
    write_once(out / ("trainer_config_resume_" + str(time.time_ns()) + ".json" if args.resume else "trainer_config.json"), config.to_dict())
    event(out, "stage_start", stage=stage, rows=len(rows), steps=args.steps,
          model_identity=model._sprint_identity, mechanical=mechanical)
    start = time.monotonic()
    outcome = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    final = out / ("final_resume_" + str(time.time_ns()) if args.resume else "final")
    if final.exists():
        raise ValueError("Refusing to overwrite final adapter")
    # Save only the trained default adapter; frozen reference is an input artifact.
    model.set_adapter("default")
    model.save_pretrained(final, selected_adapters=["default"], safe_serialization=True)
    tokenizer.save_pretrained(final)
    saved_tensor_audit = assert_adapter_matches_checkpoint(model, final)
    trainer.save_state()
    summary = {"stage": stage, "mechanical_fixture": mechanical,
        "status": "complete" if trainer.state.global_step >= args.steps else "partial",
        "optimizer_steps": trainer.state.global_step, "requested_steps": args.steps,
        "deadline_reached": audit.deadline_reached, "wall_seconds": time.monotonic() - start,
        "nonzero_gradient_steps": audit.nonzero_gradient_steps,
        "observed_parameter_changes": audit.weight_changes, "metrics": outcome.metrics,
        "adapter": str(final), "adapter_digest": adapter_digest(model, "default"),
        "saved_tensor_audit": saved_tensor_audit,
        "memory": memory_snapshot(), "model_identity": model._sprint_identity,
        "protocol": protocol_identity()}
    write_once(out / ("summary_resume_" + str(time.time_ns()) + ".json" if args.resume else "summary.json"), summary)
    event(out, "stage_complete", **summary)
    del trainer
    gc.collect()
    return summary


def validate_pairs(path, gold, tokenizer, max_length, expected_adapter=None):
    pairs = read_jsonl(path)
    by_id = {row["canonical_hash"]: row for row in gold}
    if not pairs:
        raise ValueError("No verifier-invalid model samples available for DPO")
    result = []
    adapter_hash = file_sha(Path(expected_adapter) / "adapter_model.safetensors") if expected_adapter else None
    for pair in pairs:
        source = by_id.get(pair["canonical_hash"])
        if source is None or pair["prompt"] != render_prompt(source["prompt"]):
            raise ValueError("Preference example is outside frozen training pool")
        if pair["chosen"] != render_completion(source["proof"]):
            raise ValueError("Preference chosen response differs from shared gold proof")
        if judge(source["prompt"], pair["rejected"])["accepted"]:
            raise ValueError("Rejected response is verifier-valid")
        if not pair.get("sampled_response_ids") or pair.get("generation_policy") != "sft_adapter":
            raise ValueError("Rejected response lacks generated-token provenance")
        if expected_adapter and (pair.get("source_adapter") != str(Path(expected_adapter).resolve()) or
                                 pair.get("source_adapter_sha256") != adapter_hash):
            raise ValueError("Preference sampling policy differs from requested SFT checkpoint")
        if decode_completion(tokenizer, pair["sampled_response_ids"]) != pair["rejected"]:
            raise ValueError("Rejected text differs from generated tokens")
        for response in (pair["chosen"], pair["rejected"]):
            if len(tokenizer.encode(pair["prompt"], add_special_tokens=False)) + len(tokenizer.encode(response, add_special_tokens=False)) + 1 > max_length:
                raise ValueError("DPO pair exceeds context; no truncation")
        result.append({key: pair[key] for key in ("prompt", "chosen", "rejected")})
    return result


def make_pairs(args, model, tokenizer, gold):
    pairs_path, samples_path = args.out / "pairs.jsonl", args.out / "pair_samples.jsonl"
    previous_pairs = read_jsonl(pairs_path) if args.resume_pairs and pairs_path.exists() else []
    previous_samples = read_jsonl(samples_path) if args.resume_pairs and samples_path.exists() else []
    if previous_pairs:
        validate_pairs(pairs_path, gold, tokenizer, args.max_length, args.adapter)
    accepted = {row["canonical_hash"] for row in previous_pairs}
    attempted = {(row["canonical_hash"], row["attempt"]) for row in previous_samples}
    reasons = Counter(row["verdict"]["reason"] for row in previous_samples)
    count = len(accepted)
    start = time.monotonic()
    mode = "a" if args.resume_pairs else "x"
    with pairs_path.open(mode) as pairs, samples_path.open(mode) as samples:
        for attempt in range(args.max_pair_attempts):
            pending = [row for row in gold if row["canonical_hash"] not in accepted and
                       (row["canonical_hash"], attempt) not in attempted]
            for offset in range(0, len(pending), args.pair_batch_size):
                if args.deadline and datetime.now(timezone.utc) >= args.deadline:
                    break
                batch = pending[offset:offset + args.pair_batch_size]
                batch_key = f"{args.seed}:{attempt}:" + ":".join(row["canonical_hash"] for row in batch)
                batch_seed = int(hashlib.sha256(batch_key.encode()).hexdigest()[:15], 16)
                outputs = sample_completions(model, tokenizer, [render_prompt(row["prompt"]) for row in batch],
                    max_new_tokens=args.max_completion_length, seed=batch_seed,
                    batch_size=len(batch), max_seq_len=args.max_length)
                event(args.out, "preference_generation_batch", attempt=attempt,
                    canonical_hashes=[row["canonical_hash"] for row in batch], seed=batch_seed,
                    prompt_tokens=sum(row["prompt_tokens"] for row in outputs),
                    completion_tokens=sum(row["completion_tokens"] for row in outputs),
                    post_stop_padding_positions=sum(row["post_stop_padding_positions"] for row in outputs))
                for row, output in zip(batch, outputs, strict=True):
                    verdict = judge(row["prompt"], output["text"])
                    samples.write(json.dumps({"canonical_hash": row["canonical_hash"], "attempt": attempt,
                        "sample": output, "verdict": verdict}, sort_keys=True) + "\n")
                    samples.flush()
                    reasons[verdict["reason"]] += 1
                    if not verdict["accepted"]:
                        pair = {"canonical_hash": row["canonical_hash"], "prompt": render_prompt(row["prompt"]),
                            "chosen": render_completion(row["proof"]), "rejected": output["text"],
                            "sampled_response_ids": output["response_ids"], "generation_policy": "sft_adapter",
                            "source_adapter": str(args.adapter.resolve()), "sample_seed": output["seed"],
                            "source_adapter_sha256": model._sprint_identity["adapter_sha256"]["adapter_model.safetensors"],
                            "sampling": output["sampling"], "verdict": verdict}
                        pairs.write(json.dumps(pair, sort_keys=True) + "\n")
                        pairs.flush()
                        accepted.add(row["canonical_hash"])
                        count += 1
                    event(args.out, "preference_sample", canonical_hash=row["canonical_hash"],
                        attempt=attempt, verifier_valid=verdict["accepted"], completion_tokens=output["completion_tokens"],
                        finish_reason=output["finish_reason"], pairs=count)
            if len(accepted) == len(gold) or (args.deadline and datetime.now(timezone.utc) >= args.deadline):
                break
    summary = {"pairs": count, "shared_pool": len(gold), "covered_theorems": sorted(accepted),
        "status": "complete" if len(accepted) == len(gold) else "partial",
        "reasons": dict(reasons), "wall_seconds": time.monotonic() - start,
        "pairs_sha256": file_sha(args.out / "pairs.jsonl"), "memory": memory_snapshot()}
    write_once(args.out / ("summary_resume_" + str(time.time_ns()) + ".json" if args.resume_pairs else "summary.json"), summary)
    event(args.out, "pairs_complete", **summary)


def smoke(args, model, tokenizer):
    """Exercise true library gradients, without treating parity rewards as proofs."""
    import torch
    fixture = {"prompt": "THM P SEQ P PRF", "proof": "N1 P : PR ;\nQED",
               "canonical_hash": hashlib.sha256(b"mechanical_only_P").hexdigest()}
    if not judge(fixture["prompt"], render_completion(fixture["proof"]))["accepted"]:
        raise AssertionError("Mechanical fixture is not verifier-valid")
    args.steps = 1
    args.accumulation = args.group_size
    args.max_completion_length = min(args.max_completion_length, 16)
    first = train_stage(args, model, tokenizer, [fixture] * 4, args.out / "sft", "sft", True)
    if first["nonzero_gradient_steps"] == 0 or first["observed_parameter_changes"] == 0:
        raise RuntimeError("Mechanical SFT did not change adapter parameters")
    prompt_ids = tokenizer.encode(render_prompt(fixture["prompt"]), add_special_tokens=False)
    response_ids = tokenizer.encode(render_completion(fixture["proof"]), add_special_tokens=False)
    scores = completion_logprobs(model, prompt_ids, response_ids)
    identity = model._sprint_identity
    base = model.unload()
    model = load_adapter_exact(base, first["adapter"], trainable=True)
    model._sprint_identity = identity
    event(args.out, "exact_adapter_reload", **model._sprint_adapter_load_audits[-1])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    restored = completion_logprobs(model, prompt_ids, response_ids)
    reload_error = max(abs(a - b) for a, b in zip(scores, restored, strict=True))
    if reload_error > args.max_logprob_error:
        raise RuntimeError(f"Save/reload likelihood mismatch: {reload_error}")
    model = load_adapter_exact(model, first["adapter"], adapter_name="reference", trainable=False)
    event(args.out, "exact_reference_load", **model._sprint_adapter_load_audits[-1])
    model.set_adapter("default")
    mechanical_pair = {"prompt": render_prompt(fixture["prompt"]),
        "chosen": render_completion(fixture["proof"]),
        "rejected": render_completion("N1 Q : PR ;\nQED")}
    dpo = train_stage(args, model, tokenizer, [mechanical_pair] * 4, args.out / "dpo", "dpo", True)
    if dpo["nonzero_gradient_steps"] == 0 or dpo["observed_parameter_changes"] == 0:
        raise RuntimeError("Mechanical DPO did not change adapter parameters")
    model.delete_adapter("reference")
    sample = sample_completions(model, tokenizer, [render_prompt(fixture["prompt"])],
        max_new_tokens=16, seed=args.seed, return_logprobs=True)[0]
    teacher = completion_logprobs(model, sample["prompt_ids"], sample["response_ids"])
    errors = [abs(a - b) for a, b in zip(teacher, sample["rollout_logprobs"], strict=True)]
    event(args.out, "likelihood_audit", reload_max_abs_error=reload_error,
        behavior_max_abs_error=max(errors), behavior_mean_abs_error=sum(errors) / len(errors),
        tolerance=args.max_logprob_error)
    if max(errors) > args.max_logprob_error:
        raise RuntimeError("Behavior and teacher likelihood audit failed")
    second = train_stage(args, model, tokenizer, [fixture] * 4, args.out / "grpo", "grpo", True)
    if second["nonzero_gradient_steps"] == 0 or second["observed_parameter_changes"] == 0:
        raise RuntimeError("Mechanical mixed-reward RL did not change adapter parameters")
    write_once(args.out / "summary.json", {"passed": True, "mechanical_only": True,
        "not_a_research_checkpoint": True, "sft": first, "dpo": dpo, "grpo": second,
        "reload_max_abs_error": reload_error, "behavior_max_abs_error": max(errors)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "sft", "pairs", "dpo", "grpo"))
    parser.add_argument("--model", type=Path, default=SPRINT / "models/olmo32b")
    parser.add_argument("--data", type=Path, default=SPRINT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-pairs", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-pair-attempts", type=int, default=4)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=202609092014)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--max-logprob-error", type=float, default=0.1)
    parser.add_argument("--deadline-utc", default="2026-09-10T05:14:00+00:00")
    args = parser.parse_args()
    args.deadline = datetime.fromisoformat(args.deadline_utc).astimezone(timezone.utc) if args.deadline_utc else None
    args.out = args.out.resolve()
    if args.stage in ("pairs", "dpo", "grpo") and args.adapter is None:
        parser.error("This stage requires the preceding checkpoint via --adapter")
    if args.stage == "dpo" and args.pairs is None:
        parser.error("DPO requires generated --pairs")
    if args.resume and args.stage not in ("sft", "dpo", "grpo"):
        parser.error("Resume is supported only for trainer stages")
    if args.resume_pairs and args.stage != "pairs":
        parser.error("--resume-pairs applies only to pair generation")
    if min(args.steps, args.accumulation, args.max_length, args.max_completion_length, args.rank, args.save_steps) < 1 or args.group_size < 2:
        parser.error("Training sizes must be positive; group size must be at least two")
    if args.deadline and datetime.now(timezone.utc) >= args.deadline:
        parser.error("Training deadline has elapsed")
    if args.out.exists() and not (args.resume or args.resume_pairs):
        parser.error("Output directory already exists; use a fresh directory or explicit trainer --resume")
    args.out.mkdir(parents=True, exist_ok=bool(args.resume or args.resume_pairs))
    gold, rl, data_audit = load_training(args.data)
    provenance = {"args": vars(args), "data": data_audit, "versions": versions(),
        "source_sha256": {p.name: file_sha(p) for p in (Path(__file__), Path(__file__).with_name("modeling.py"))},
        "created_utc": utcnow(), "mechanical_fixture": args.stage == "smoke"}
    if args.resume or args.resume_pairs:
        original = json.loads((args.out / "run.json").read_text())
        stable = ("stage", "model", "data", "adapter", "pairs", "rank", "seed", "group_size",
                  "accumulation", "learning_rate", "max_length", "max_completion_length", "pair_batch_size")
        if original["data"] != data_audit or any(str(original["args"].get(key)) != str(vars(args).get(key)) for key in stable):
            raise ValueError("Resume changes frozen data, protocol, model, or experiment settings")
    write_once(args.out / ("run_resume_" + str(time.time_ns()) + ".json" if args.resume or args.resume_pairs else "run.json"), provenance)
    seed_everything(args.seed)
    tokenizer = load_tokenizer(args.model)
    # Verify the default TRL tokenizer call matches our raw completion policy.
    for row in gold:
        prompt = render_prompt(row["prompt"])
        if tokenizer.encode(prompt) != tokenizer.encode(prompt, add_special_tokens=False):
            raise ValueError("TRL default prompt tokenization differs from the frozen raw template")
    if args.stage == "sft":
        encode_supervision(tokenizer, gold, args.max_length)
    pairs = validate_pairs(args.pairs, gold, tokenizer, args.max_length, args.adapter) if args.stage == "dpo" else None
    model = load_model(args.model, args.adapter, trainable=args.stage != "pairs", rank=args.rank)
    event(args.out, "model_loaded", identity=model._sprint_identity, memory=memory_snapshot())
    if args.stage == "pairs":
        make_pairs(args, model, tokenizer, gold)
    elif args.stage == "smoke":
        smoke(args, model, tokenizer)
    else:
        if args.stage == "dpo":
            import torch
            model = load_adapter_exact(model, args.adapter, adapter_name="reference", trainable=False)
            event(args.out, "exact_reference_load", **model._sprint_adapter_load_audits[-1])
            model.set_adapter("default")
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.to(torch.float32)
                    if ".reference." in name:
                        parameter.requires_grad_(False)
        train_stage(args, model, tokenizer, gold if args.stage == "sft" else pairs if args.stage == "dpo" else rl,
                    args.out, args.stage)


if __name__ == "__main__":
    main()
