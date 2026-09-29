"""Unaided endpoint evaluation; only theorem prompts enter the environment."""
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import random
import time
import torch
from torch.distributions import Categorical

import common
from nd_environment import strict_complete
from symbolic_general_env import Environment
from symbolic_general_model import Policy, features, FEATURE_VERSION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy", choices=["model", "random"], required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--max-lines", type=int, default=24)
    p.add_argument("--max-scope", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=1800)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = common.read_jsonl(args.population)
    model = None
    if args.policy == "model":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config["feature_version"] != FEATURE_VERSION:
            raise ValueError("Checkpoint feature schema mismatch")
        model = Policy(config["width"], config["layers"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "symbolic_general_eval_config.json").exists():
        raise ValueError("Refusing to overwrite prior evaluation")
    config = vars(args).copy()
    config.update({"population_sha256": common.sha(args.population),
        "checkpoint_sha256": common.sha(args.checkpoint) if args.checkpoint else None,
        "code_hashes": {p.name: common.sha(p) for p in sorted(Path(__file__).parent.glob("symbolic_general_*.py"))},
        "checker_sha256": common.sha(Path(__file__).with_name("nd_environment.py")),
        "feature_version": FEATURE_VERSION, "pid": os.getpid(), "start_unix": time.time()})
    common.write_json(out / "symbolic_general_eval_config.json", config)
    started, completed = time.monotonic(), 0
    groups, successes_by_theorem = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    trace_path = out / "symbolic_general_attempts.jsonl"
    for row in rows:
        if time.monotonic() - started > args.max_seconds:
            break
        prompt = row["prompt"]
        source_id = row.get("source_id") or row.get("theorem_id") or common.digest(prompt)
        family = row.get("family", "unlabeled")
        for attempt in range(args.attempts):
            env = Environment(prompt, max_scope=args.max_scope, max_lines=args.max_lines)
            trace, reward, done, scored = [], float(env.success()), env.success(), 0
            while not done:
                aa = env.actions()
                if not aa:
                    break
                if model is None:
                    idx = rng.randrange(len(aa))
                else:
                    af, sf = features(env, aa)
                    with torch.no_grad():
                        logits, value = model(torch.from_numpy(af), torch.from_numpy(sf))
                        idx = (Categorical(logits=logits).sample() if args.sample else logits.argmax()).item()
                action = aa[idx]
                scored += len(aa)
                trace.append({"rule": action.rule, "formula": action.formula, "refs": action.refs,
                              "depth": action.depth, "candidate_count": len(aa), "selected_index": idx})
                reward, done = env.step(action, check_membership=False)
            full = strict_complete(prompt, env.body())
            if bool(reward) != full["ok"]:
                raise AssertionError("Terminal reward disagrees with strict check")
            pruned_body = env.pruned_body() if reward else None
            pruned = strict_complete(prompt, pruned_body) if reward else None
            if pruned and not pruned["ok"]:
                raise AssertionError("Scope-sensitive dependency pruning produced invalid certificate")
            result = {"source_id": source_id, "family": family, "attempt": attempt, "prompt": prompt,
               "success": bool(reward), "supplied_prefix_lines": 0, "generated_actions": env.generated,
               "registered_premise_lines": len(env.premises), "candidates_scored": scored,
               "body": env.body(), "verdict": full, "pruned_body": pruned_body, "pruned_verdict": pruned,
               "action_trace": trace}
            with trace_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
            for key in [family, "all"]:
                groups[key]["attempts"] += 1
                groups[key]["successes"] += int(reward)
                groups[key]["generated_actions"] += env.generated
                groups[key]["candidate_actions_scored"] += scored
                groups[key]["successful_pruned_lines_total"] += pruned["n_lines"] if pruned else 0
            successes_by_theorem[source_id] += int(reward)
            completed += 1
        if completed % 32 == 0:
            print(json.dumps({"attempts": completed, "successes": groups["all"]["successes"],
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    summary = {"config": config, "groups": dict(groups), "attempts": completed,
               "theorems_attempted": len(successes_by_theorem),
               "theorems_with_any_success": sum(v > 0 for v in successes_by_theorem.values()),
               "elapsed_seconds": time.monotonic() - started,
               "status": "complete" if completed == len(rows) * args.attempts else "time_limit",
               "attempts_sha256": common.sha(trace_path) if trace_path.exists() else None}
    common.write_json(out / "symbolic_general_eval_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()
