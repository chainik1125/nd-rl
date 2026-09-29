"""Behavior cloning control using the exact same cap-six source actions only."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=str(common.ROOT / "data/v1/sources.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--presentations", type=int)
    p.add_argument("--full-batch-updates", type=int)
    p.add_argument("--width", type=int, default=500)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=120)
    args = p.parse_args()
    if (args.presentations is None) == (args.full_batch_updates is None):
        p.error("Specify exactly one of --presentations or --full-batch-updates")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sources = common.read_jsonl(args.sources)
    states = []
    for source in sources:
        verdict = strict_complete(source["prompt"], source["body"])
        assert verdict["ok"] and verdict["n_lines"] <= 6
        env = Environment(source["prompt"])
        for desired in env.source_actions(source):
            aa = env.actions()
            af, sf = features(env, aa)
            states.append({"actions": af, "state": sf, "action": aa.index(desired)})
            env.step(desired)
    model = Policy(args.width, args.layers)
    # The critic remains at its matched random initialization in BC.
    optimizer = torch.optim.Adam(model.scorer.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "symbolic_general_config.json").exists():
        raise ValueError("Refusing to overwrite existing BC run")
    config = vars(args).copy()
    config.update({"mode": "bc", "source_sha256": common.sha(args.sources), "feature_version": FEATURE_VERSION,
        "code_hashes": {p.name: common.sha(p) for p in sorted(Path(__file__).parent.glob("symbolic_general_*.py"))},
        "checker_sha256": common.sha(Path(__file__).with_name("nd_environment.py")), "pid": os.getpid(),
        "start_unix": time.time(), "unique_source_states": len(states), "external_sources": len(sources),
        "targets_used": 0, "critic_trained": False, "parameter_count": sum(p.numel() for p in model.parameters())})
    common.write_json(out / "symbolic_general_config.json", config)
    torch.save({"state_dict": model.state_dict(), "config": config}, out / "symbolic_general_initial.pt")
    metrics, total, update = [], 0, 0
    goal = args.presentations if args.presentations is not None else args.full_batch_updates * len(states)
    started = time.monotonic()
    while total < goal and time.monotonic() - started < args.max_seconds:
        batch = states if args.full_batch_updates is not None else [rng.choice(states) for _ in range(min(args.minibatch, goal - total))]
        aa, ss, mm = pack(batch)
        chosen = torch.tensor([t["action"] for t in batch])
        logits, _ = model(aa, ss, mm)
        loss = -Categorical(logits=logits).log_prob(chosen).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.scorer.parameters(), .5)
        optimizer.step()
        total += len(batch)
        update += 1
        if update % 32 == 0 or total == goal:
            row = {"presentations": total, "optimizer_steps": update, "last_batch_cross_entropy": float(loss.detach()),
                   "last_batch_accuracy": float((logits.argmax(-1) == chosen).float().mean()),
                   "elapsed_seconds": time.monotonic() - started}
            metrics.append(row)
            common.write_jsonl(out / "symbolic_general_metrics.jsonl", metrics)
            print(json.dumps(row), flush=True)
    summary = {"config": config, "presentations": total, "optimizer_steps": update,
               "elapsed_seconds": time.monotonic() - started,
               "status": "complete" if total == goal else "time_limit"}
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), **summary},
               out / "symbolic_general_checkpoint.pt")
    common.write_json(out / "symbolic_general_train_summary.json", summary)


if __name__ == "__main__":
    main()
