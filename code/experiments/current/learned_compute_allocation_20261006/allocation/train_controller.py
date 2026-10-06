"""Complete-episode outcome REINFORCE on frozen raw proof pools.

This is a controller-only preliminary reduction: proof generation is frozen and
independent, so its policy likelihoods do not belong in the controller loss.
No correctness-to-stop labels, supervised selector, or cost-trained start.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"
import torch
from controller import Controller, encode_history


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def batch_states(model, tok, states, device):
    lengths = torch.tensor([len(s[0]) for s in states], device=device)
    counts = torch.tensor([s[1] for s in states], device=device)
    ids = torch.full((len(states), int(lengths.max())), tok.pad, device=device, dtype=torch.long)
    for row, (tokens, _) in enumerate(states):
        ids[row, : len(tokens)] = torch.tensor(tokens, device=device)
    return model.distributions(ids, lengths, counts)


def collect(
    model, tok, row, labels, rng, device, cap, fixed_count=None, greedy=False, attempt_bound=512
):
    candidates = list(row["candidates"])
    rng.shuffle(candidates)
    theorem = tok.encode_prompt(row["prompt"])
    traces = []
    sampled = []
    forced_reason = None
    label_by_id = {v["candidate_id"]: v for v in labels}
    for n, candidate in enumerate(candidates[: model.max_attempts], 1):
        sampled.append(candidate)
        raw = [v["transition_tokens"] for v in sampled]
        ids = encode_history(tok, theorem, raw, cap)
        next_size = attempt_bound + 3
        forced = n == model.max_attempts or n == len(candidates) or len(ids) + next_size > cap
        if forced:
            forced_reason = "attempt_limit" if n == model.max_attempts else "context_limit"
        with torch.no_grad():
            accept_dist, index_dist = batch_states(model, tok, [(ids, n)], device)
            accept = accept_dist.probs >= 0.5 if greedy else accept_dist.sample().bool()
            if fixed_count is not None:
                accept[:] = n >= fixed_count
            if forced:
                accept[:] = True
            index = index_dist.probs.argmax(-1) if greedy else index_dist.sample()
            lp = model.action_logprob(
                accept_dist,
                index_dist,
                accept.float(),
                index,
                torch.tensor([forced], device=device),
            )
        if fixed_count is not None:
            lp = index_dist.log_prob(index) if accept.item() else torch.zeros(1, device=device)
        traces.append(
            {
                "ids": ids,
                "count": n,
                "accepted": bool(accept.item()),
                "index": int(index.item()),
                "forced": forced,
                "stopping_trainable": fixed_count is None and not forced,
                "logprob": float(lp.item()),
                "accept_probability": float(accept_dist.probs.item()),
                "selection_probabilities": index_dist.probs[0, :n].tolist(),
            }
        )
        if accept.item():
            chosen = sampled[int(index.item())]
            correct = bool(label_by_id[chosen["candidate_id"]]["valid"])
            return traces, {
                "theorem_id": row["theorem_id"],
                "attempts": n,
                "selected_candidate_id": chosen["candidate_id"],
                "correct": correct,
                "forced": forced,
                "termination_reason": forced_reason if forced else "voluntary",
                "generated_tokens": sum(v["generated_tokens"] for v in sampled),
                "generator_prefill_tokens": sum(v["prompt_tokens"] for v in sampled),
                "generator_forward_calls": sum(v["forward_calls"] for v in sampled),
                "controller_forward_calls": len(traces),
                "requested_fixed_count": fixed_count,
                "controller_prefill_tokens": sum(len(t["ids"]) for t in traces),
                "candidate_order": [v["candidate_id"] for v in candidates],
                "decisions": [{k: v for k, v in t.items() if k != "ids"} for t in traces],
            }
    raise RuntimeError("episode failed to terminate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cost", type=float, default=0)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--attempt-bound", type=int, default=512)
    parser.add_argument("--fixed-count", type=int)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()
    if args.episodes < 2:
        raise ValueError("leave-one-out baseline requires at least two episodes")
    sys.path.insert(0, args.source)
    from model import load_ckpt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for filename in ["train_controller.py", "controller.py"]:
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, tok, extra = load_ckpt(args.base, "cuda")
    controller = Controller(model, args.max_attempts).cuda()
    rows = read_jsonl(args.public)
    private = {v["theorem_id"]: v["labels"] for v in read_jsonl(args.private)}
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("controller updates are restricted to train theorem pools")
    # Admit every theorem. Oversize history is an explicit protocol failure,
    # never silent label- or candidate-length-dependent theorem filtering.
    eligible = rows
    for row in rows:
        longest = max(len(c["transition_tokens"]) for c in row["candidates"])
        if longest > args.attempt_bound:
            raise ValueError("observed candidate exceeds declared per-attempt bound")
        if len(tok.encode_prompt(row["prompt"])) + longest + 4 > args.context:
            raise ValueError("mandatory first attempt cannot fit declared context")
    config = vars(args) | {
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "command": sys.argv,
        "runner_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
        },
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "base_extra": extra,
        "training_rows": len(rows),
        "eligible_rows": len(eligible),
        "loss": "complete_episode_mean_REINFORCE_leave_one_out_baseline",
        "generator": "frozen_independent_raw_pool",
        "controller_candidate_view": "full_public_transition_rendering",
        "input_sha256": hashlib.sha256(Path(args.public).read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(Path(args.private).read_bytes()).hexdigest(),
        "entropy_bonus": 0,
        "kl_penalty": 0,
        "discount": 1,
        "initial_heads": "zeros_uniform",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0)
    with (out / "episodes.jsonl").open("w") as log:
        for update in range(args.updates):
            controller.eval()
            episodes = []
            for _ in range(args.episodes):
                row = rng.choice(eligible)
                trace, result = collect(
                    controller,
                    tok,
                    row,
                    private[row["theorem_id"]],
                    rng,
                    "cuda",
                    args.context,
                    attempt_bound=args.attempt_bound,
                    fixed_count=args.fixed_count,
                )
                result["reward"] = float(result["correct"]) - args.cost * (result["attempts"] - 1)
                result["update"] = update
                log.write(json.dumps(result) + "\n")
                episodes.append((trace, result["reward"]))
            log.flush()
            rewards = torch.tensor([e[1] for e in episodes], device="cuda")
            advantage = rewards - (rewards.sum() - rewards) / max(1, len(rewards) - 1)
            optimizer.zero_grad()
            # Accumulate each episode's sum-logprob loss, then mean over episodes.
            # No normalization by number of attempts/tokens (which would change objective).
            losses = []
            for i, (trace, _) in enumerate(episodes):
                states = [(t["ids"], t["count"]) for t in trace]
                accept, index = batch_states(controller, tok, states, "cuda")
                accepted = torch.tensor([t["accepted"] for t in trace], device="cuda").float()
                indices = torch.tensor([t["index"] for t in trace], device="cuda")
                forced = torch.tensor([t["forced"] for t in trace], device="cuda")
                if args.fixed_count is None:
                    lp = controller.action_logprob(accept, index, accepted, indices, forced)
                else:
                    lp = torch.where(
                        accepted.bool(), index.log_prob(indices), torch.zeros_like(accepted)
                    )
                old_lp = torch.tensor([t["logprob"] for t in trace], device="cuda")
                if not torch.allclose(lp.detach(), old_lp, atol=1e-3, rtol=1e-4):
                    raise RuntimeError("behavior and replay log probabilities disagree")
                loss = -advantage[i] * lp.sum() / args.episodes
                loss.backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1)
            optimizer.step()
            print(
                json.dumps(
                    {"update": update, "reward": float(rewards.mean()), "loss": sum(losses)}
                ),
                flush=True,
            )
            if update % args.save_every == 0 or update + 1 == args.updates:
                torch.save(
                    {"state": controller.state_dict(), "config": config, "update": update},
                    out / f"checkpoint_{update:05d}.pt",
                )


if __name__ == "__main__":
    main()
