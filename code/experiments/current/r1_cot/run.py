"""R1-inspired symbolic scratchpad SFT and group-relative policy gradients.

Only explicit JSONL paths are opened. RL consumes prompts, never reference proofs.
The sampler never masks actions or inserts thought/final/QED boundaries.
Dependencies: nd_model, nd_grpo, nd_metrics, nd_verify, strict_verify on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from nd_data import canonical_theorem, sequent_parts
from nd_grpo import collate_rollouts
from nd_metrics import target_prompt
from nd_model import ModelConfig, ProofModel, ProofTokenizer, save_checkpoint
from strict_verify import safe_verify
from torch.nn import functional as F

if __package__:
    from .correction_plan import VERSION as CORRECTION_PLAN_VERSION
    from .correction_plan import corrupt_draft
    from .scratch_variants import VERSION as DEPENDENCY_PLAN_VERSION
    from .scratch_variants import plan as dependency_plan
else:
    from correction_plan import VERSION as CORRECTION_PLAN_VERSION
    from correction_plan import corrupt_draft
    from scratch_variants import VERSION as DEPENDENCY_PLAN_VERSION
    from scratch_variants import plan as dependency_plan


def scratch_version(style):
    return {
        "reverse": "reverse-line-goals-v1",
        "dependency": DEPENDENCY_PLAN_VERSION,
        "correction": CORRECTION_PLAN_VERSION,
    }[style]


class ScratchTokenizer(ProofTokenizer):
    VERSION = "relative-unary-r1-symbolic-v1"
    TOKENS = ProofTokenizer.TOKENS + ("<THINK>", "</THINK>", "<FINAL>")


class ScratchRepairTokenizer(ScratchTokenizer):
    VERSION = "relative-unary-r1-repair-v1"
    TOKENS = ScratchTokenizer.TOKENS + ("<DRAFT>", "<ERROR>", "<REVISE>")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def guard_keys(prompt):
    premises, goal = sequent_parts(prompt)

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    canonical = digest(canonical_theorem(prompt))
    context = (
        digest(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF"))
        if premises
        else None
    )
    skeleton = digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None
    return canonical, context, skeleton


def validate_training_path(data_path):
    parts = Path(data_path).resolve().parts
    if any("sealed" in part.lower() or part.lower() in ("final", "test") for part in parts):
        raise ValueError("Training may not open sealed/final/test paths")
    if any(tag in Path(data_path).name.lower() for tag in ("textbook_dev", "textbook_final")):
        raise ValueError("Evaluation data cannot be training data")


def guard_training(rows, data_path, guard_path, allow_unguarded_smoke=False):
    validate_training_path(data_path)
    if not guard_path:
        if not allow_unguarded_smoke or len(rows) > 16:
            raise ValueError("Training requires --guard-json; unguarded smoke limited to 16 rows")
        return
    fields = (
        "canonical_theorem_sha256",
        "nonempty_premise_context_sha256",
        "closed_formula_skeleton_sha256",
    )
    obj = json.loads(Path(guard_path).read_text())
    if any(not isinstance(obj.get(field), list) for field in fields):
        raise ValueError("Missing guard hash lists")
    for field in fields:
        if not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in obj[field]
        ):
            raise ValueError("Malformed guard hash")
    sets = [set(obj[field]) for field in fields]
    for row in rows:
        if any(tag in str(row.get("role", "")).lower() for tag in ("dev", "final", "test", "eval")):
            raise ValueError("Evaluation-role row cannot enter training")
        if any(
            key is not None and key in protected
            for key, protected in zip(guard_keys(target_prompt(row)), sets, strict=True)
        ):
            raise ValueError("Protected theorem/context/skeleton in training data")


def guard_final_release(data_path, checkpoint_sha, release_path):
    protected = any(
        "sealed" in part.lower() or part.lower() in ("final", "test")
        for part in Path(data_path).resolve().parts
    )
    if not protected:
        return
    if not release_path:
        raise ValueError("Final evaluation requires --final-release-json from sprint-wide freeze")
    release = json.loads(Path(release_path).read_text())
    if release.get("released") is not True or release.get("no_further_training") is not True:
        raise ValueError("Final release must close further training")
    not_before = datetime.fromisoformat(release["not_before_utc"].replace("Z", "+00:00"))
    earliest = datetime(2026, 9, 9, 13, 25, tzinfo=UTC)
    if not_before.tzinfo is None or not_before < earliest or datetime.now(UTC) < not_before:
        raise ValueError("Final evaluation release is earlier than sprint deadline")
    if checkpoint_sha not in release.get("checkpoints", []):
        raise ValueError("Checkpoint is not in final candidate release")
    if sha256(data_path) not in release.get("data_sha256", []):
        raise ValueError("Data is not in final evaluation release")


def guard_development_release(data_path, checkpoint_sha, release_path):
    if "textbook_dev" not in Path(data_path).name.lower():
        return
    if not release_path:
        raise ValueError(
            "Textbook development evaluation needs --evaluation-release-json after an RL run"
        )
    release = json.loads(Path(release_path).read_text())
    if checkpoint_sha not in release.get("checkpoints", []):
        raise ValueError("Checkpoint is not in paired development release")
    if sha256(data_path) not in release.get("data_sha256", []):
        raise ValueError("Data is not in paired development release")
    rl_checkpoint = Path(release["completed_rl_checkpoint"])
    rl_sha = release["completed_rl_checkpoint_sha256"]
    completed = json.loads((rl_checkpoint.parent / "completed.json").read_text())
    if sha256(rl_checkpoint) != rl_sha or completed.get("final_sha256") != rl_sha:
        raise ValueError("Paired RL completion marker or checkpoint hash differs")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def emit(output, kind, **fields):
    row = {"kind": kind, "unix_time": time.time(), **fields}
    with (Path(output) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def load_model(path, device="cpu", max_seq_len=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    state = checkpoint["tokenizer"]
    if state == ScratchRepairTokenizer().state_dict():
        tok = ScratchRepairTokenizer()
    elif state == ScratchTokenizer().state_dict():
        tok = ScratchTokenizer()
    elif state == ProofTokenizer().state_dict():
        tok = ProofTokenizer()
    else:
        raise ValueError("Only relative-unary checkpoints are supported")
    config = dict(checkpoint["config"])
    if max_seq_len is not None:
        config["max_seq_len"] = max_seq_len
    model = ProofModel(ModelConfig(**config))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), tok, checkpoint.get("metadata", {})


def expand_vocabulary(model, tokenizer=None):
    target = tokenizer or ScratchTokenizer()
    old_size = model.config.vocab_size
    if old_size == len(target):
        return model
    if old_size not in (len(ProofTokenizer()), len(ScratchTokenizer())) or old_size > len(target):
        raise ValueError("Unexpected original vocabulary size")
    config = asdict(model.config)
    config["vocab_size"] = len(target)
    expanded = ProofModel(ModelConfig(**config)).to(next(model.parameters()).device)
    state = model.state_dict()
    new_state = expanded.state_dict()
    for name, values in state.items():
        if name in ("token_embedding.weight", "lm_head.weight"):
            new_state[name][:old_size].copy_(values)
        else:
            new_state[name].copy_(values)
    expanded.load_state_dict(new_state, strict=True)
    return expanded.eval()


def split_record(row):
    prompt = target_prompt(row)
    body = row.get("body", row.get("proof"))
    if body is None and "text" in row:
        words = row["text"].split()
        body = " ".join(words[words.index("PRF") + 1 :])
    if not isinstance(body, str):
        raise ValueError("SFT needs text, body, or proof in every row")
    return prompt, " ".join(body.split())


def backward_plan(body):
    """Reverse line order, retaining each rule and local goal, omitting citations.

    This cold-start trace uses only the supplied short proof. It supplies no
    search oracle, long proof, textbook content, or verifier during generation.
    """
    steps = []
    for line in body.split(";"):
        words = line.split()
        if not words or words == ["QED"]:
            continue
        colon = words.index(":")
        formula = [word for word in words[1:colon] if word != "|"]
        rule = words[colon + 1]
        steps.append([rule, *formula, ";"])
    return [word for step in reversed(steps) for word in step]


def encode_training(
    row, tok, mode, max_seq_len, max_proof_lines=6, scratch_style="reverse", scratch_seed=0
) -> dict[str, Any]:
    scratch_version(scratch_style)  # Reject an unrecognized cold-start configuration.
    prompt, body = split_record(row)
    ok, reason, lines = safe_verify(prompt, body)
    if not ok or body.split().count("QED") != 1 or body.split()[-1] != "QED":
        raise ValueError("Invalid SFT proof: " + reason)
    if lines > max_proof_lines:
        raise ValueError(f"SFT proof has {lines} lines, above cap {max_proof_lines}")
    prompt_ids = tok.encode_prompt(prompt)
    final_ids = tok.encode_record(prompt + " " + body)[len(prompt_ids) :]
    provenance = None
    if mode == "cot":
        if not isinstance(tok, ScratchTokenizer):
            raise ValueError("CoT requires expanded tokenizer; run prepare")
        plan_words = (
            backward_plan(body)
            if scratch_style == "reverse"
            else dependency_plan(body, max_proof_lines=max_proof_lines)
        )
        plan_ids = [tok.stoi[word] for word in plan_words]
        if scratch_style == "correction":
            if not isinstance(tok, ScratchRepairTokenizer):
                raise ValueError("Correction cold starts require a prepared repair tokenizer")
            corrupted = corrupt_draft(
                prompt, body, seed=scratch_seed, max_proof_lines=max_proof_lines
            )
            if corrupted["provenance"]["mutation"]["after"] not in tok.stoi:
                raise ValueError("Draft mutation is outside the checkpoint tokenizer vocabulary")
            draft_ids = tok.encode_record(prompt + " " + corrupted["draft"])[len(prompt_ids) :]
            plan_ids = [
                tok.stoi["<DRAFT>"],
                *draft_ids,
                tok.stoi["<ERROR>"],
                tok.stoi["<REVISE>"],
                *plan_ids,
            ]
            provenance = corrupted["provenance"]
        response = [
            tok.stoi["<THINK>"],
            *plan_ids,
            tok.stoi["</THINK>"],
            tok.stoi["<FINAL>"],
            *final_ids,
        ]
    else:
        response = final_ids
    if len(prompt_ids) + len(response) > max_seq_len:
        raise ValueError("SFT record exceeds context; refusing truncation")
    result = {"prompt_ids": prompt_ids, "response_ids": response}
    if provenance is not None:
        result["cold_start_provenance"] = provenance
    return result


def record_cold_start_provenance(data, output, filename):
    rows = [
        dict(source_row_index=index, **row["cold_start_provenance"])
        for index, row in enumerate(data)
        if "cold_start_provenance" in row
    ]
    if not rows:
        return
    path = Path(output) / filename
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    emit(output, "cold_start_provenance", path=str(path), rows=len(rows), sha256=sha256(path))


def judge_response(prompt, response, tok, mode):
    """Verify the exact final suffix against the immutable original prompt."""
    ids = list(response)
    scratch_count = 0
    if mode == "cot":
        start, end, final = (tok.stoi[t] for t in ("<THINK>", "</THINK>", "<FINAL>"))
        if not ids or ids[0] != start or any(ids.count(t) != 1 for t in (start, end, final)):
            return {
                "reward": 0,
                "reason": "invalid thought boundaries",
                "found_len": None,
                "scratch_tokens": 0,
                "body": "",
            }
        end_at, final_at = ids.index(end), ids.index(final)
        if end_at < 1 or final_at != end_at + 1:
            return {
                "reward": 0,
                "reason": "invalid final boundary",
                "found_len": None,
                "scratch_tokens": 0,
                "body": "",
            }
        scratch_count = end_at - 1
        ids = ids[final_at + 1 :]
    if not ids or ids[-1] != tok.eos_id or ids.count(tok.eos_id) != 1:
        return {
            "reward": 0,
            "reason": "missing or nonterminal final QED",
            "found_len": None,
            "scratch_tokens": scratch_count,
            "body": "",
        }
    # The base decoder is reversible through QED; never decode scratch as proof.
    body = tok.decode_body(ids)
    ok, reason, lines = safe_verify(prompt, body)
    return {
        "reward": int(ok),
        "reason": reason,
        "found_len": lines if ok else None,
        "scratch_tokens": scratch_count,
        "body": body,
    }


def amp(model):
    if (
        next(model.parameters()).device.type == "cuda"
        and getattr(model, "r1_precision", "bf16") == "bf16"
    ):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA training currently requires BF16 support")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def sample(
    model, tok, prompts, *, mode, temperature=0.8, seed=0, max_new_tokens=768, batch_size=32
) -> list[dict[str, Any]]:
    if temperature < 0 or batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Invalid sampler configuration")
    model.eval()
    device = next(model.parameters()).device
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and getattr(model, "r1_precision", "bf16") == "bf16"
        else torch.float32
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    encoded = [tok.encode_prompt(prompt) for prompt in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    results = [None] * len(prompts)
    with amp(model):
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            lengths = torch.tensor([len(encoded[i]) for i in indices], device=device)
            width = max(len(encoded[i]) for i in indices)
            limit = min(max_new_tokens, model.config.max_seq_len - width)
            if limit < 1:
                raise ValueError("Prompt exceeds context")
            n = len(indices)
            tokens = torch.full((n, width), tok.pad_id, dtype=torch.long, device=device)
            for row, i in enumerate(indices):
                tokens[row, -len(encoded[i]) :] = torch.tensor(encoded[i], device=device)
            columns = torch.arange(width, device=device)
            valid = columns[None] >= width - lengths[:, None]
            positions = (columns[None] - width + lengths[:, None]).clamp_min(0)
            causal = torch.ones((width, width), device=device, dtype=torch.bool).tril()
            cache = model.allocate_cache(n, width + limit, dtype)
            logits = model(
                tokens,
                positions=positions,
                attention_mask=valid[:, None, None] & causal[None, None],
                cache=cache,
            )[:, -1]
            valid_keys = torch.ones((n, width + limit), device=device, dtype=torch.bool)
            valid_keys[:, :width] = valid
            done = torch.zeros(n, device=device, dtype=torch.bool)
            final_seen = torch.full((n,), mode == "direct", device=device, dtype=torch.bool)
            stop_lengths = torch.full((n,), limit, device=device, dtype=torch.long)
            generated = torch.empty((n, limit), device=device, dtype=torch.long)
            logps = torch.empty((n, limit), device=device, dtype=torch.float32)
            for step in range(limit):
                distribution = (logits.float() / (temperature or 1.0)).log_softmax(-1)
                if temperature:
                    chosen = torch.multinomial(distribution.exp(), 1, generator=generator).squeeze(
                        1
                    )
                    chosen_logps = distribution.gather(1, chosen[:, None]).squeeze(1)
                else:
                    chosen = logits.argmax(-1)
                    # Greedy policy is a point mass; it is never used for RL.
                    chosen_logps = torch.zeros(n, device=device)
                generated[:, step] = chosen
                logps[:, step] = chosen_logps
                if mode == "cot":
                    final_seen |= chosen == tok.stoi["<FINAL>"]
                newly_done = (~done) & final_seen & (chosen == tok.eos_id)
                stop_lengths[newly_done] = step + 1
                done |= newly_done
                if bool(done.all()) or step + 1 == limit:
                    break
                cache_start = width + step
                logits = model(
                    chosen[:, None],
                    positions=(lengths + step)[:, None],
                    attention_mask=valid_keys[:, None, None, : cache_start + 1],
                    cache=cache,
                    cache_start=cache_start,
                )[:, -1]
            all_ids = generated[:, : step + 1].cpu().tolist()
            all_logps = logps[:, : step + 1].cpu().tolist()
            for row, i in enumerate(indices):
                length = min(int(stop_lengths[row]), step + 1)
                results[i] = {
                    "prompt_ids": encoded[i],
                    "response_ids": all_ids[row][:length],
                    "rollout_logprobs": all_logps[row][:length],
                }
            del cache
    if any(row is None for row in results):
        raise RuntimeError("Sampler did not return every requested response")
    return [row for row in results if row is not None]


def ce_backward(model, tok, rows, microbatch, coefficient=1.0):
    denominator = sum(len(row["response_ids"]) for row in rows)
    loss_value = 0.0
    for offset in range(0, len(rows), microbatch):
        batch = collate_rollouts(
            rows[offset : offset + microbatch], tok.pad_id, next(model.parameters()).device
        )
        with amp(model):
            logits = model(batch["inputs"]).float()
            losses = F.cross_entropy(
                logits.flatten(0, 1), batch["targets"].flatten(), reduction="none"
            ).view_as(batch["targets"])
            loss = (losses * batch["mask"]).sum() / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite SFT loss")
        (coefficient * loss).backward()
        loss_value += float(loss.detach())
    return loss_value


def train_sft(model, tok, rows, args):
    data = [
        encode_training(
            row,
            tok,
            args.mode,
            model.config.max_seq_len,
            args.max_proof_lines,
            args.scratch_style,
            args.seed,
        )
        for row in rows
    ]
    if not data:
        raise ValueError("Empty SFT data")
    record_cold_start_provenance(data, args.output, "cold_start_provenance.jsonl")
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    model.train()
    started = time.monotonic()
    actual_response_tokens = 0
    for step in range(1, args.steps + 1):
        batch = [rng.choice(data) for _ in range(args.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss = ce_backward(model, tok, batch, args.microbatch)
        actual_response_tokens += sum(len(row["response_ids"]) for row in batch)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite SFT gradient")
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            emit(
                args.output,
                "sft",
                step=step,
                loss=loss,
                grad_norm=float(norm),
                actual_response_tokens=actual_response_tokens,
                elapsed_seconds=time.monotonic() - started,
            )
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                Path(args.output) / f"step_{step:06d}.pt",
                model,
                tok,
                mode=args.mode,
                stage="sft",
                step=step,
                scratch_style=args.scratch_style,
                scratch_plan_version=scratch_version(args.scratch_style),
                args=vars(args),
            )


def group_advantages(rewards, baseline="loo"):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Require group_size >= 2")
    if not bool(((rewards == 0) | (rewards == 1)).all()):
        raise ValueError("Rewards must be binary")
    if baseline == "loo":
        return rewards - (rewards.sum(1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
    if baseline == "standardized":
        centered = rewards - rewards.mean(1, keepdim=True)
        return centered / rewards.std(1, unbiased=False, keepdim=True).clamp_min(1e-8)
    raise ValueError("Unknown advantage baseline")


def policy_update(model, tok, rows, optimizer, args, retention=()):
    model.eval()  # Dropout must not change the behavior policy's likelihoods.
    device = next(model.parameters()).device
    max_error = 0.0
    # Teacher forcing audits numerical cache drift only. The PPO denominator
    # is the actual behavior-policy log probability captured during sampling.
    with torch.no_grad():
        for offset in range(0, len(rows), args.microbatch):
            chunk = rows[offset : offset + args.microbatch]
            batch = collate_rollouts(chunk, tok.pad_id, device)
            with amp(model):
                logps = (model(batch["inputs"]).float() / args.temperature).log_softmax(-1)
            chosen = logps.gather(-1, batch["targets"][..., None]).squeeze(-1)
            for row, values, mask in zip(chunk, chosen, batch["mask"], strict=True):
                teacher_values = values[mask].cpu().tolist()
                if len(teacher_values) != len(row["rollout_logprobs"]):
                    raise ValueError("Behavior likelihood does not cover every action")
                row["old_logprobs"] = list(row["rollout_logprobs"])
                error = max(
                    abs(a - b) for a, b in zip(teacher_values, row["rollout_logprobs"], strict=True)
                )
                max_error = max(max_error, error)
    if max_error > args.max_logprob_error:
        raise RuntimeError(f"Behavior/teacher numerical drift {max_error} exceeds configured bound")
    losses = []
    for _ in range(args.policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for offset in range(0, len(rows), args.microbatch):
            chunk = rows[offset : offset + args.microbatch]
            batch = collate_rollouts(chunk, tok.pad_id, device)
            with amp(model):
                logps = (model(batch["inputs"]).float() / args.temperature).log_softmax(-1)
                chosen = logps.gather(-1, batch["targets"][..., None]).squeeze(-1)
                ratio = (chosen - batch["old_logprobs"]).exp()
                advantage = batch["advantages"][:, None]
                surrogate = torch.minimum(
                    ratio * advantage, ratio.clamp(1 - args.clip, 1 + args.clip) * advantage
                )
                # Every sampled token, including scratch/PAD/boundaries, earns
                # its group's advantage. No invalid-token or length filtering.
                denominator = (
                    batch["mask"].sum(-1)
                    if args.loss_normalization == "sequence"
                    else args.max_new_tokens
                )
                sequence_loss = -(surrogate * batch["mask"]).sum(-1) / denominator
                loss = sequence_loss.sum() / len(rows)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite GRPO loss")
            loss.backward()
            loss_value += float(loss.detach())
        retention_loss = None
        if retention and args.retention_weight:
            retention_loss = ce_backward(
                model, tok, retention, args.microbatch, args.retention_weight
            )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite GRPO gradient")
        optimizer.step()
        losses.append(loss_value)
    return {
        "policy_losses": losses,
        "retention_loss": retention_loss,
        "grad_norm": float(norm),
        "behavior_teacher_max_abs_error": max_error,
        "retention_response_tokens": args.policy_epochs
        * sum(len(r["response_ids"]) for r in retention)
        if args.retention_weight
        else 0,
        "policy_response_tokens": args.policy_epochs * sum(len(r["response_ids"]) for r in rows),
    }


def train_rl(model, tok, data, args):
    if args.temperature <= 0 or args.group_size < 2 or args.policy_epochs < 1:
        raise ValueError("RL needs positive temperature and >=2 samples per group")
    # Deliberately discard all body/reference fields before scheduling RL.
    prompts = [target_prompt(row) for row in data]
    if not prompts:
        raise ValueError("Empty RL prompt pool")
    retention = []
    if args.retention_data:
        validate_training_path(args.retention_data)
        retention_rows = read_jsonl(args.retention_data)
        guard_training(
            retention_rows, args.retention_data, args.guard_json, args.allow_unguarded_smoke
        )
        retention = [
            encode_training(
                row,
                tok,
                args.mode,
                model.config.max_seq_len,
                args.max_proof_lines,
                args.scratch_style,
                args.seed,
            )
            for row in retention_rows
        ]
        record_cold_start_provenance(
            retention, args.output, "retention_cold_start_provenance.jsonl"
        )
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    started = time.monotonic()
    total_tokens = 0
    with (Path(args.output) / "rollouts.jsonl").open("a") as trace:
        for step in range(1, args.steps + 1):
            selected = [rng.choice(prompts) for _ in range(args.prompts_per_step)]
            expanded = [prompt for prompt in selected for _ in range(args.group_size)]
            rows = sample(
                model,
                tok,
                expanded,
                mode=args.mode,
                temperature=args.temperature,
                seed=args.seed * 1000000 + step,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.decode_batch,
            )
            for i, (row, prompt) in enumerate(zip(rows, expanded, strict=True)):
                row.update(judge_response(prompt, row["response_ids"], tok, args.mode))
                row.update(
                    prompt=prompt,
                    group=i // args.group_size,
                    attempt=i % args.group_size,
                    step=step,
                )
            rewards = torch.tensor([row["reward"] for row in rows]).reshape(-1, args.group_size)
            advantages = group_advantages(rewards, args.advantage).flatten().tolist()
            for row, advantage in zip(rows, advantages, strict=True):
                row["advantage"] = advantage
            retained = (
                [rng.choice(retention) for _ in range(args.retention_batch)] if retention else []
            )
            metrics = policy_update(model, tok, rows, optimizer, args, retained)
            for row in rows:
                trace.write(json.dumps(row) + "\n")
            trace.flush()
            total_tokens += sum(len(row["response_ids"]) for row in rows)
            emit(
                args.output,
                "rl",
                step=step,
                reward_mean=float(rewards.float().mean()),
                groups_mixed=int(((rewards.sum(1) > 0) & (rewards.sum(1) < args.group_size)).sum()),
                max_found_length=max((row["found_len"] or 0 for row in rows), default=0),
                mean_scratch_tokens=sum(row["scratch_tokens"] for row in rows) / len(rows),
                actual_response_tokens=total_tokens,
                elapsed_seconds=time.monotonic() - started,
                **metrics,
            )
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    Path(args.output) / f"step_{step:06d}.pt",
                    model,
                    tok,
                    mode=args.mode,
                    stage="rl",
                    step=step,
                    scratch_style=args.scratch_style,
                    scratch_plan_version=scratch_version(args.scratch_style),
                    args=vars(args),
                )


def evaluate(model, tok, data, args, checkpoint_sha):
    # A completed checkpoint hash is recorded before the first target is sampled.
    write_json(
        Path(args.output) / "evaluation_freeze.json",
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "sha256": checkpoint_sha,
            "data_sha256": sha256(args.data),
            "args": vars(args),
            "unix_time": time.time(),
        },
    )
    prompts = [target_prompt(row) for row in data]
    if not prompts:
        raise ValueError("Empty evaluation set")
    attempts = []
    with (Path(args.output) / "attempts.jsonl").open("w") as stream:
        for attempt in range(args.samples):
            rows = sample(
                model,
                tok,
                prompts,
                mode=args.mode,
                temperature=args.temperature,
                seed=args.seed + attempt,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.decode_batch,
            )
            for index, (target, prompt, row) in enumerate(zip(data, prompts, rows, strict=True)):
                result = {
                    "index": index,
                    "name": target.get("name", target.get("id", str(index))),
                    "prompt": prompt,
                    "generated_tokens": len(row["response_ids"]),
                    "attempt": attempt,
                    **row,
                    **judge_response(prompt, row["response_ids"], tok, args.mode),
                }
                if args.mode == "direct":
                    result["final_tokens"] = len(row["response_ids"])
                else:
                    final_id = tok.stoi["<FINAL>"]
                    result["final_tokens"] = (
                        len(row["response_ids"]) - row["response_ids"].index(final_id) - 1
                        if final_id in row["response_ids"]
                        else 0
                    )
                for field in ("reference_lines", "n_lines", "proof_length", "split", "source"):
                    if field in target:
                        result[field] = target[field]
                stream.write(json.dumps(result) + "\n")
                attempts.append(result)
            stream.flush()
            emit(args.output, "eval_progress", attempt=attempt + 1, samples=args.samples)
    shortest = {}
    for row in attempts:
        if row["reward"]:
            shortest[row["index"]] = min(shortest.get(row["index"], float("inf")), row["found_len"])
    # A generated length is not a lower bound on theorem difficulty.
    reference_groups = {}
    for index, target in enumerate(data):
        length = target.get("reference_lines", target.get("n_lines", target.get("proof_length")))
        if length is not None:
            group = reference_groups.setdefault(str(length), {"total": 0, "solved": 0})
            group["total"] += 1
            group["solved"] += int(index in shortest)
    summary = {
        "checkpoint_sha256": checkpoint_sha,
        "data_sha256": sha256(args.data),
        "problems": len(data),
        "samples_per_problem": args.samples,
        "solved": len(shortest),
        "solve_rate": len(shortest) / len(data),
        "max_shortest_found_length": max(shortest.values(), default=None),
        "shortest_found_length_histogram": dict(Counter(shortest.values())),
        "solved_above_initial_cap": sum(n > args.max_proof_lines for n in shortest.values()),
        "by_reference_length": reference_groups,
        "actual_response_tokens": sum(len(row["response_ids"]) for row in attempts),
        "initial_proof_cap": args.max_proof_lines,
        "length_caveat": "Written proof length, not minimum proof complexity.",
    }
    if sha256(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during evaluation")
    write_json(Path(args.output) / "summary.json", summary)
    write_json(
        Path(args.output) / "completed.json",
        {"stage": "eval", "checkpoint_sha256": checkpoint_sha, "unix_time": time.time()},
    )
    emit(args.output, "eval_complete", **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "sft", "rl", "eval"))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--fresh-config", help="JSON ModelConfig path; prepare only, no inherited weights"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--mode", choices=("direct", "cot"), default="cot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-proof-lines", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--decode-batch", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--retention-data")
    parser.add_argument("--retention-weight", type=float, default=0.1)
    parser.add_argument("--retention-batch", type=int, default=16)
    parser.add_argument("--guard-json")
    parser.add_argument("--allow-unguarded-smoke", action="store_true")
    parser.add_argument("--advantage", choices=("loo", "standardized"), default="loo")
    parser.add_argument("--loss-normalization", choices=("fixed", "sequence"), default="fixed")
    parser.add_argument("--max-logprob-error", type=float, default=0.5)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--scratch-style", choices=("reverse", "dependency", "correction"), default="reverse"
    )
    parser.add_argument("--final-release-json")
    parser.add_argument("--evaluation-release-json")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.microbatch < 1 or args.samples < 1:
        parser.error("steps, microbatch, and samples must be positive")
    if args.command != "prepare" and not args.data:
        parser.error("--data is required for sft, rl, and eval")
    if bool(args.checkpoint) == bool(args.fresh_config):
        parser.error("Provide exactly one of --checkpoint or --fresh-config")
    if args.fresh_config and args.command != "prepare":
        parser.error("--fresh-config is for prepare only")
    if args.command in ("sft", "rl"):
        validate_training_path(args.data)
        if args.retention_data:
            validate_training_path(args.retention_data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("--output must be a fresh empty directory to avoid overwriting artifacts")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    checkpoint_sha = sha256(args.checkpoint) if args.checkpoint else None
    if args.fresh_config:
        config = json.loads(Path(args.fresh_config).read_text())
        config["vocab_size"] = len(ProofTokenizer())
        if args.max_seq_len:
            config["max_seq_len"] = args.max_seq_len
        model = ProofModel(ModelConfig(**config)).to(args.device)
        tok, source_metadata = ProofTokenizer(), {"fresh_config_sha256": sha256(args.fresh_config)}
    else:
        model, tok, source_metadata = load_model(args.checkpoint, args.device, args.max_seq_len)
    model.r1_precision = args.precision
    if args.command == "prepare":
        if args.scratch_style == "correction":
            tok = ScratchRepairTokenizer()
            model = expand_vocabulary(model, tok)
        elif args.mode == "cot":
            tok = (
                ScratchRepairTokenizer()
                if isinstance(tok, ScratchRepairTokenizer)
                else ScratchTokenizer()
            )
            model = expand_vocabulary(model, tok)
        save_checkpoint(
            output / "final.pt",
            model,
            tok,
            stage="prepared",
            mode=args.mode,
            source_checkpoint_sha256=checkpoint_sha,
            scratch_plan_version=scratch_version(args.scratch_style),
            args=vars(args),
        )
        write_json(
            output / "manifest.json",
            {
                "args": vars(args),
                "source_sha256": checkpoint_sha,
                "final_sha256": sha256(output / "final.pt"),
            },
        )
        return
    if args.mode == "cot" and not isinstance(tok, ScratchTokenizer):
        parser.error("CoT needs a prepared expanded-vocabulary checkpoint")
    if args.scratch_style == "correction" and not isinstance(tok, ScratchRepairTokenizer):
        parser.error("Correction style needs a prepared 46-token checkpoint for both modes")
    if args.command == "eval":
        guard_final_release(args.data, checkpoint_sha, args.final_release_json)
        guard_development_release(args.data, checkpoint_sha, args.evaluation_release_json)
    manifest = {
        "args": vars(args),
        "source_sha256": checkpoint_sha,
        "data_sha256": sha256(args.data),
        "source_metadata": source_metadata,
        "config": asdict(model.config),
        "tokenizer": tok.state_dict(),
        "scratch_plan_version": scratch_version(args.scratch_style),
    }
    if args.retention_data:
        manifest["retention_data_sha256"] = sha256(args.retention_data)
    if args.guard_json:
        manifest["guard_sha256"] = sha256(args.guard_json)
    if args.final_release_json:
        manifest["final_release_sha256"] = sha256(args.final_release_json)
    if args.evaluation_release_json:
        manifest["evaluation_release_sha256"] = sha256(args.evaluation_release_json)
    write_json(output / "manifest.json", manifest)
    if args.command in ("sft", "rl"):
        validate_training_path(args.data)
    data = read_jsonl(args.data)
    if args.command in ("sft", "rl"):
        guard_training(data, args.data, args.guard_json, args.allow_unguarded_smoke)
    if args.command == "sft":
        train_sft(model, tok, data, args)
    elif args.command == "rl":
        train_rl(model, tok, data, args)
    else:
        evaluate(model, tok, data, args, checkpoint_sha)
        return
    save_checkpoint(
        output / "final.pt",
        model,
        tok,
        mode=args.mode,
        stage=args.command,
        source_checkpoint_sha256=checkpoint_sha,
        data_sha256=sha256(args.data),
        scratch_plan_version=scratch_version(args.scratch_style),
        args=vars(args),
    )
    write_json(
        output / "completed.json",
        {
            "final_sha256": sha256(output / "final.pt"),
            "steps": args.steps,
            "unix_time": time.time(),
        },
    )


if __name__ == "__main__":
    main()
