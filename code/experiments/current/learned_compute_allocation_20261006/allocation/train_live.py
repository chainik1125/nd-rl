"""Joint same-model generation, acceptance and selection with final-outcome RL.

All token and decision likelihoods are summed per complete episode. Stop labels
never come from correctness. Env provides only grammar/scope transitions; the
original theorem is the authority for a delayed selected-proof Lean judgment.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"
import torch
from controller import Controller, encode_history
from live import PROTOCOL_VERSION, generate_attempt, token_logprobs
from train_controller import batch_states, read_jsonl


def episode(controller, tok, row, rng, args):
    candidates, private, generation, decisions = [], [], [], []
    controller_seconds = 0.0
    for n in range(1, args.max_attempts + 1):
        candidate, measurement, actions = generate_attempt(
            controller,
            tok,
            row["prompt"],
            candidates,
            rng,
            context_limit=args.context,
            attempt_bound=args.attempt_bound,
            max_action=args.max_action,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )
        candidates.append(candidate)
        private.append(measurement)
        generation.extend(actions)
        ids = encode_history(
            tok,
            tok.encode_prompt(row["prompt"]),
            [c["transition_tokens"] for c in candidates],
            args.context,
        )
        # The reserve is fixed and independent of a future attempt's actual output.
        # State reserve covers a next authoritative state in addition to output.
        forced = (
            n == args.max_attempts
            or len(ids) + args.attempt_bound + args.state_reserve + 3 > args.context
        )
        controller_started = time.monotonic()
        with torch.no_grad():
            accept, index = batch_states(controller, tok, [(ids, n)], "cuda")
            accepted = accept.sample().bool()
            if getattr(args, "fixed_count", None) is not None:
                accepted[:] = n >= args.fixed_count
            if forced:
                accepted[:] = True
            selected = index.sample()
            lp = controller.action_logprob(
                accept, index, accepted.float(), selected, torch.tensor([forced], device="cuda")
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decision_seconds = time.monotonic() - controller_started
        controller_seconds += decision_seconds
        if getattr(args, "fixed_count", None) is not None:
            lp = index.log_prob(selected) if accepted.item() else torch.zeros(1, device="cuda")
        decisions.append(
            {
                "ids": ids,
                "count": n,
                "accepted": bool(accepted.item()),
                "index": int(selected.item()),
                "forced": forced,
                "stopping_trainable": getattr(args, "fixed_count", None) is None and not forced,
                "logprob": float(lp.item()),
                "accept_probability": float(accept.probs.item()),
                "selection_probabilities": index.probs[0, :n].tolist(),
                "controller_wall_seconds": decision_seconds,
            }
        )
        if accepted.item():
            return {
                "theorem_id": row["theorem_id"],
                "prompt": row["prompt"],
                "attempts": n,
                "selected_index": int(selected.item()),
                "forced": forced,
                "termination_reason": (
                    "attempt_limit" if n == args.max_attempts else "context_limit"
                )
                if forced
                else "voluntary",
                "candidates": candidates,
                "private_attempts": private,
                "generation": generation,
                "decisions": decisions,
                "generated_tokens": sum(c["generated_tokens"] for c in candidates),
                "generator_prefill_tokens": sum(c["prompt_tokens"] for c in candidates),
                "generator_forward_calls": sum(c["forward_calls"] for c in candidates),
                "generator_forward_tokens": sum(c["forward_tokens"] for c in candidates),
                "generator_attention_score_cells_per_layer": sum(
                    c["attention_score_cells_per_layer"] for c in candidates
                ),
                "diagnostic_replay_forward_calls": sum(
                    c["likelihood_replay_forward_calls"] for c in candidates
                ),
                "diagnostic_replay_forward_tokens": sum(
                    c["likelihood_replay_forward_tokens"] for c in candidates
                ),
                "controller_prefill_tokens": sum(len(d["ids"]) for d in decisions),
                "controller_forward_calls": len(decisions),
                "controller_attention_score_cells_per_layer": sum(
                    controller.base.cfg["n_head"] * len(d["ids"]) ** 2 for d in decisions
                ),
                "sampling_wall_seconds": sum(c["sampling_wall_seconds"] for c in candidates),
                "controller_wall_seconds": controller_seconds,
                "policy_execution_wall_seconds": sum(c["sampling_wall_seconds"] for c in candidates)
                + controller_seconds,
                "diagnostic_replay_wall_seconds": sum(
                    c["likelihood_replay_wall_seconds"] for c in candidates
                ),
            }
    raise RuntimeError("complete episode did not terminate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cost", type=float, default=0)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--attempt-bound", type=int, default=1024)
    parser.add_argument("--state-reserve", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-action", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--fixed-count", type=int)
    parser.add_argument("--teacher-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.episodes < 2 or args.temperature <= 0:
        raise ValueError("invalid episodic baseline or sampling temperature")
    sys.path.insert(0, args.source)
    from lean_gate import lean_check
    from model import load_ckpt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for filename in ["train_live.py", "live.py", "controller.py", "train_controller.py"]:
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    source_files = [
        "model.py",
        "best_model.py",
        "sample.py",
        "state_env.py",
        "state_sample.py",
        "lean_tok.py",
        "tokenizer.py",
        "lean_gate.py",
        "nd2lean.py",
    ]
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    theorem_rng = random.Random(args.seed)
    model, tok, _ = load_ckpt(args.base, "cuda")
    controller = Controller(model, args.max_attempts).cuda()
    teacher = None
    if args.teacher_diagnostics:
        teacher, _, _ = load_ckpt(args.base, "cuda")
        teacher.eval()
        teacher.requires_grad_(False)
    rows = read_jsonl(args.data)
    if any(r.get("split") != "train" for r in rows):
        raise ValueError("joint policy updates require train split only")
    config = vars(args) | {
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "command": sys.argv,
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "runner_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
        },
        "source_hashes": {
            f: hashlib.sha256((Path(args.source) / f).read_bytes()).hexdigest()
            for f in source_files
        },
        "loss": "episode_sum_all_raw_generation_and_decision_logprobs_mean_episodes_LOO_baseline",
        "entropy_bonus": 0,
        "kl_penalty": 0,
        "discount": 1,
        "initial_heads": "zeros_uniform",
        "theorem_schedule": "independent_seeded_RNG_unaffected_by_rollout_length",
        "episode_seed_rule": "seed_plus_update_times_1000003_plus_slot_times_104729",
        "checkpoint_naming": "completed_update_count",
        "controller_candidate_view": "full_public_Env_text_transitions_plus_failed_attempt",
        "generator_history_view": "effective_action_trace_with_each_action_EOS",
        "live_protocol_version": PROTOCOL_VERSION,
        "sampling_precision": "fp32",
        "replay_precision": "fp32",
        "torch_version": str(torch.__version__),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "generator": "same_trainable_base_authoritative_state_prefixed_by_effective_prior_attempts",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    head_parameters = list(controller.accept_head.parameters()) + list(
        controller.select_head.parameters()
    )
    head_ids = {id(parameter) for parameter in head_parameters}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in controller.parameters() if id(p) not in head_ids],
                "lr": args.lr,
            },
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=0,
    )
    with (out / "episodes.jsonl").open("w") as log, (out / "metrics.jsonl").open("w") as metrics:
        for update in range(args.updates):
            start = time.monotonic()
            controller.eval()
            episodes = []
            for slot in range(args.episodes):
                episode_seed = args.seed + update * 1000003 + slot * 104729
                torch.manual_seed(episode_seed)
                ep = episode(
                    controller,
                    tok,
                    theorem_rng.choice(rows),
                    random.Random(episode_seed),
                    args,
                )
                ep["episode_seed"] = episode_seed
                ep["episode_slot"] = slot
                episodes.append(ep)
            pending, indices = [], []
            for i, ep in enumerate(episodes):
                chosen = ep["private_attempts"][ep["selected_index"]]
                ep["correct"] = False
                if chosen["parse_complete"]:
                    pending.append((tok.statement(ep["prompt"]), chosen["lean_text"]))
                    indices.append(i)
            verdicts, lean_wall, lean_process = lean_check(pending)
            for i, verdict in zip(indices, verdicts, strict=True):
                episodes[i]["correct"] = bool(verdict)
            # These additional judgments are diagnostics only: reward remains
            # the selected proof's outcome and never enters the next observation.
            clean_pending, clean_indices = [], []
            for i, ep in enumerate(episodes):
                first = ep["private_attempts"][0]
                ep["first_attempt_correct_diagnostic"] = False
                if first["parse_complete"]:
                    clean_pending.append((tok.statement(ep["prompt"]), first["lean_text"]))
                    clean_indices.append(i)
            clean_verdicts, clean_lean_wall, clean_lean_process = lean_check(clean_pending)
            for i, verdict in zip(clean_indices, clean_verdicts, strict=True):
                episodes[i]["first_attempt_correct_diagnostic"] = bool(verdict)
            teacher_sum, sampled_sum, diagnostic_tokens = 0.0, 0.0, 0
            teacher_started = time.monotonic()
            if teacher is not None:
                with torch.no_grad():
                    for ep in episodes:
                        for action in ep["generation"]:
                            teacher_lp, _ = token_logprobs(
                                teacher,
                                action["prompt_ids"],
                                action["raw_ids"],
                                "cuda",
                                args.temperature,
                            )
                            action["teacher_logprobs_diagnostic"] = teacher_lp.tolist()
                            teacher_sum += float(teacher_lp.sum())
                            sampled_sum += sum(action["logprobs"])
                            diagnostic_tokens += len(action["raw_ids"])
            teacher_diagnostic_seconds = time.monotonic() - teacher_started
            for ep in episodes:
                ep["reward"] = float(ep["correct"]) - args.cost * (ep["attempts"] - 1)
                ep["update"] = update
                log.write(json.dumps(ep) + "\n")
            log.flush()
            rewards = torch.tensor([ep["reward"] for ep in episodes], device="cuda")
            advantages = rewards - (rewards.sum() - rewards) / (args.episodes - 1)
            optimizer.zero_grad()
            replay_error = 0.0
            for i, ep in enumerate(episodes):
                coefficient = -advantages[i] / args.episodes
                for action in ep["generation"]:
                    lp, _ = token_logprobs(
                        controller.base,
                        action["prompt_ids"],
                        action["raw_ids"],
                        "cuda",
                        args.temperature,
                    )
                    old = torch.tensor(action["logprobs"], device="cuda")
                    error = float((lp.detach() - old).abs().max())
                    replay_error = max(replay_error, error)
                    if not torch.allclose(lp.detach(), old, atol=1e-3, rtol=1e-4):
                        raise RuntimeError("raw generated token likelihood replay mismatch")
                    (coefficient * lp.sum()).backward()
                trace = ep["decisions"]
                accept, index = batch_states(
                    controller, tok, [(d["ids"], d["count"]) for d in trace], "cuda"
                )
                accepted = torch.tensor([d["accepted"] for d in trace], device="cuda").float()
                selected = torch.tensor([d["index"] for d in trace], device="cuda")
                forced = torch.tensor([d["forced"] for d in trace], device="cuda")
                if args.fixed_count is None:
                    lp = controller.action_logprob(accept, index, accepted, selected, forced)
                else:
                    lp = torch.where(
                        accepted.bool(), index.log_prob(selected), torch.zeros_like(accepted)
                    )
                old = torch.tensor([d["logprob"] for d in trace], device="cuda")
                if not torch.allclose(lp.detach(), old, atol=1e-3, rtol=1e-4):
                    raise RuntimeError("decision likelihood replay mismatch")
                (coefficient * lp.sum()).backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), 1))
            stop_grad = (
                float(controller.accept_head.weight.grad.norm())
                if controller.accept_head.weight.grad is not None
                else 0.0
            )
            generation_grad = float(controller.base.head.weight.grad.norm())
            optimizer.step()
            metric = {
                "update": update,
                "completed_updates": update + 1,
                "seconds": time.monotonic() - start,
                "accuracy": sum(ep["correct"] for ep in episodes) / args.episodes,
                "first_attempt_accuracy_diagnostic": sum(
                    ep["first_attempt_correct_diagnostic"] for ep in episodes
                )
                / args.episodes,
                "sampled_teacher_mean_logprob_diagnostic": teacher_sum / diagnostic_tokens
                if diagnostic_tokens
                else None,
                "sampled_policy_minus_teacher_mean_logprob_diagnostic": (sampled_sum - teacher_sum)
                / diagnostic_tokens
                if diagnostic_tokens
                else None,
                "reward": float(rewards.mean()),
                "attempts": sum(ep["attempts"] for ep in episodes) / args.episodes,
                "forced": sum(ep["forced"] for ep in episodes),
                "gradient_norm_before_clip": grad_norm,
                "stop_gradient_after_clip": stop_grad,
                "generation_gradient_after_clip": generation_grad,
                "max_replay_logprob_error": replay_error,
                "lean_wall_seconds": lean_wall,
                "lean_process_seconds": lean_process,
                "first_attempt_diagnostic_lean_wall_seconds": clean_lean_wall,
                "first_attempt_diagnostic_lean_process_seconds": clean_lean_process,
                "teacher_diagnostic_wall_seconds": teacher_diagnostic_seconds,
            }
            metrics.write(json.dumps(metric) + "\n")
            metrics.flush()
            print(json.dumps(metric), flush=True)
            if (update + 1) % args.save_every == 0 or update + 1 == args.updates:
                torch.save(
                    {
                        "state": controller.state_dict(),
                        "config": config,
                        "update": update,
                        "completed_updates": update + 1,
                    },
                    out / f"checkpoint_{update + 1:05d}.pt",
                )


if __name__ == "__main__":
    main()
