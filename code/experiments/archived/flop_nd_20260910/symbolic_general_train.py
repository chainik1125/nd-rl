"""Scratch PPO/critic for the finite-action ND diagnostic."""
from collections import Counter, deque
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.distributions import Categorical

import common
from nd_environment import strict_complete
from symbolic_general_env import Environment
from symbolic_general_model import Policy, features, pack, FEATURE_VERSION


def advantages(episode, bootstrap, gamma=.95, lam=.95):
    gae = 0.
    for transition in reversed(episode):
        live = float(not transition["done"])
        delta = transition["reward"] + gamma * live * bootstrap - transition["value"]
        gae = delta + gamma * lam * live * gae
        transition["advantage"] = gae
        transition["return"] = gae + transition["value"]
        bootstrap = transition["value"]


def setup(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    sources = common.read_jsonl(args.sources)
    targets = common.read_jsonl(args.targets)
    for source in sources:
        verdict = strict_complete(source["prompt"], source["body"])
        if not verdict["ok"] or verdict["n_lines"] > 6:
            raise ValueError("Unverified or over-cap source")
    if any(set(target) - {"source_id", "name", "prompt"} for target in targets):
        raise ValueError("Targets must be theorem-only records")
    model = Policy(args.width, args.layers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "symbolic_general_config.json").exists() and not args.resume:
        raise ValueError("Refusing to overwrite existing run")
    config = {k: v for k, v in vars(args).items() if k != "fn"}
    config.update({"source_sha256": common.sha(args.sources), "target_sha256": common.sha(args.targets),
      "code_hashes": {p.name: common.sha(p) for p in sorted(Path(__file__).parent.glob("symbolic_general_*.py"))},
      "checker_sha256": common.sha(Path(__file__).with_name("nd_environment.py")),
      "vendor_hashes": common.dependency_hashes(), "feature_version": FEATURE_VERSION,
      "pid": os.getpid(), "start_unix": time.time(), "parameter_count": sum(p.numel() for p in model.parameters())})
    if args.resume:
        previous = json.loads((out / "symbolic_general_config.json").read_text())
        for key in ("sources", "targets", "mode", "seed", "steps", "width", "layers", "batch", "minibatch",
                    "epochs", "lr", "clip", "entropy", "gamma", "gae_lambda", "source_portion", "curriculum_window",
                    "curriculum_threshold", "max_lines", "max_scope", "source_sha256", "target_sha256", "code_hashes"):
            if config[key] != previous[key]:
                raise ValueError(f"Resume changes frozen field {key}")
        with (out / "symbolic_general_resumes.jsonl").open("a") as f:
            f.write(json.dumps({"pid": os.getpid(), "start_unix": time.time(), "max_seconds": args.max_seconds,
                               "checkpoint_sha256": common.sha(out / "symbolic_general_recovery.pt")}) + "\n")
        config = previous
    else:
        common.write_json(out / "symbolic_general_config.json", config)
        torch.save({"state_dict": model.state_dict(), "config": config}, out / "symbolic_general_initial.pt")
    return rng, sources, targets, model, out, config


def ppo(args):
    rng, sources, targets, model, out, config = setup(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    source_lengths = [len(Environment(s["prompt"]).source_actions(s)) for s in sources]
    cuts = [n - 1 if args.mode == "reset" else 0 for n in source_lengths]
    histories = [deque(maxlen=args.curriculum_window) for _ in sources]
    accounting, metrics = Counter(), []
    total, update, optimizer_steps = 0, 0, 0
    elapsed_offset = 0.
    if args.resume:
        recovery = torch.load(out / "symbolic_general_recovery.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(recovery["state_dict"])
        optimizer.load_state_dict(recovery["optimizer"])
        cuts = recovery["cuts"]
        histories = [deque(values, maxlen=args.curriculum_window) for values in recovery["histories"]]
        accounting.update(recovery["accounting"])
        metrics = recovery["metrics"]
        total, update, optimizer_steps = recovery["sampled_actions"], recovery["updates"], recovery["optimizer_steps"]
        elapsed_offset = recovery["elapsed_seconds"]
        rng.setstate(recovery["random_state"])
        torch.set_rng_state(recovery["torch_rng_state"])
        np.random.set_state(recovery["numpy_rng_state"])
        success_path = out / "symbolic_general_successes.jsonl"
        if success_path.exists():
            retained = [row for row in common.read_jsonl(success_path) if row["actions_so_far"] <= total]
            common.write_jsonl(success_path, retained)
    started = time.monotonic()
    def elapsed():
        return elapsed_offset + time.monotonic() - started
    def save_recovery():
        snapshot = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "cuts": cuts,
            "histories": [list(h) for h in histories], "accounting": dict(accounting), "metrics": metrics,
            "sampled_actions": total, "updates": update, "optimizer_steps": optimizer_steps,
            "elapsed_seconds": elapsed(), "random_state": rng.getstate(), "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(), "config": config}
        temporary = out / "symbolic_general_recovery.pt.tmp"
        torch.save(snapshot, temporary)
        temporary.replace(out / "symbolic_general_recovery.pt")
    status = "complete"
    while total < args.steps:
        if elapsed() >= args.max_seconds:
            status = "time_limit"
            break
        transitions, episodes = [], []
        while len(transitions) < min(args.batch, args.steps - total):
            is_source = rng.random() < args.source_portion
            index = rng.randrange(len(sources) if is_source else len(targets))
            row = sources[index] if is_source else targets[index]
            env = Environment(row["prompt"], max_scope=args.max_scope, max_lines=args.max_lines)
            cut = cuts[index] if is_source else 0
            if is_source:
                env.replay_source(row, cut)
            accounting["source_episodes_started" if is_source else "target_episodes_started"] += 1
            accounting["premise_lines_registered"] += len(env.premises)
            accounting["prefix_inferences_replayed"] += cut
            trajectory, done, reward = [], env.success(), float(env.success())
            while not done and len(transitions) + len(trajectory) < args.steps - total:
                aa = env.actions()
                if not aa:
                    done = True
                    break
                af, sf = features(env, aa)
                with torch.no_grad():
                    logits, value = model(torch.from_numpy(af), torch.from_numpy(sf))
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action).item()
                reward, done = env.step(aa[action.item()], check_membership=False)
                trajectory.append({"actions": af, "state": sf, "action": action.item(), "logp": logp,
                                   "value": value.item(), "reward": reward, "done": done})
                accounting["candidate_actions_scored"] += len(aa)
                accounting["max_candidate_actions"] = max(accounting["max_candidate_actions"], len(aa))
            bootstrap = 0.
            if not done:
                af, sf = features(env, env.actions())
                with torch.no_grad():
                    bootstrap = model.critic(torch.from_numpy(sf)).item()
            advantages(trajectory, bootstrap, args.gamma, args.gae_lambda)
            transitions.extend(trajectory)
            if done:
                label = "source" if is_source else "target"
                accounting[f"{label}_episodes_completed"] += 1
                accounting[f"{label}_episodes_successful"] += int(reward)
                episodes.append({"source": is_source, "success": reward, "cut": cut})
                if reward:
                    with (out / "symbolic_general_successes.jsonl").open("a") as f:
                        f.write(json.dumps({"source_id": row["source_id"], "is_source": is_source, "cut": cut,
                            "actions_so_far": total + len(transitions), "prompt": row["prompt"], "body": env.body(),
                            "verdict": strict_complete(env.prompt, env.body())}) + "\n")
                if is_source:
                    histories[index].append(reward)
                    if (args.mode == "reset" and cuts[index] > 0 and len(histories[index]) == args.curriculum_window
                            and np.mean(histories[index]) >= args.curriculum_threshold):
                        cuts[index] -= 1
                        histories[index].clear()
                        accounting["curriculum_advances"] += 1
        if not transitions:
            raise RuntimeError("No sampled actions in rollout batch")
        adv = np.asarray([t["advantage"] for t in transitions], np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for t, a in zip(transitions, adv):
            t["normalized_advantage"] = float(a)
        losses, entropies, kl_values = [], [], []
        for _ in range(args.epochs):
            order = list(range(len(transitions)))
            rng.shuffle(order)
            for start in range(0, len(order), args.minibatch):
                batch = [transitions[i] for i in order[start:start + args.minibatch]]
                aa, ss, mm = pack(batch)
                chosen = torch.tensor([t["action"] for t in batch])
                old_logp = torch.tensor([t["logp"] for t in batch])
                advantage = torch.tensor([t["normalized_advantage"] for t in batch])
                returns = torch.tensor([t["return"] for t in batch])
                logits, values = model(aa, ss, mm)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(chosen)
                ratio = (logp - old_logp).exp()
                policy_loss = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - args.clip, 1 + args.clip) * advantage).mean()
                value_loss = .5 * (values - returns).square().mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + value_loss - args.entropy * entropy
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
                entropies.append(float(entropy.detach()))
                kl_values.append(float(((ratio - 1) - (logp - old_logp)).mean().detach()))
        total += len(transitions)
        update += 1
        metric = {"update": update, "sampled_actions": total, "optimizer_steps": optimizer_steps,
             "mean_loss": float(np.mean(losses)), "mean_entropy": float(np.mean(entropies)),
             "mean_approx_kl": float(np.mean(kl_values)), "mean_cut": float(np.mean(cuts)),
             "cuts": cuts.copy(), "success_fraction": float(np.mean([e["success"] for e in episodes])) if episodes else None,
             "elapsed_seconds": elapsed(), "accounting": dict(accounting)}
        metrics.append(metric)
        common.write_jsonl(out / "symbolic_general_metrics.jsonl", metrics)
        if update % 16 == 0:
            save_recovery()
        if update % 4 == 0 or total == args.steps:
            print(json.dumps({k: v for k, v in metric.items() if k != "cuts"}), flush=True)
    summary = {"status": status, "sampled_actions": total, "updates": update, "optimizer_steps": optimizer_steps,
               "elapsed_seconds": elapsed(), "accounting": dict(accounting), "cuts": cuts,
               "config": config}
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), **summary},
               out / "symbolic_general_checkpoint.pt")
    common.write_json(out / "symbolic_general_train_summary.json", summary)
    save_recovery()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=str(common.ROOT / "data/v1/sources.jsonl"))
    p.add_argument("--targets", default=str(common.ROOT / "data/v1/targets.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["full", "reset"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--width", type=int, default=500)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--minibatch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--entropy", type=float, default=.01)
    p.add_argument("--gamma", type=float, default=.95)
    p.add_argument("--gae-lambda", type=float, default=.95)
    p.add_argument("--source-portion", type=float, default=.8)
    p.add_argument("--curriculum-window", type=int, default=64)
    p.add_argument("--curriculum-threshold", type=float, default=.9)
    p.add_argument("--max-lines", type=int, default=24)
    p.add_argument("--max-scope", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    ppo(args)


if __name__ == "__main__":
    main()
