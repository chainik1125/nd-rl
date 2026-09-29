"""Replay archived successful episodes in completed seed-only symbolic runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import common
from nd_environment import strict_complete
from nd_verify import parse_proof_tokens
from symbolic_general_env import Environment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory, trainer_directory):
    started = time.monotonic()
    directory, trainer_directory = Path(directory), Path(trainer_directory)
    config_path = directory / "symbolic_general_config.json"
    summary_path = directory / "symbolic_general_train_summary.json"
    config, summary = (json.loads(p.read_text()) for p in (config_path, summary_path))
    require(summary["status"] == "complete" and summary["sampled_actions"] == config["steps"], "Training budget incomplete")
    require(config["source_portion"] == 1, "This audit is for seed-only runs")
    require(common.sha(config["sources"]) == config["source_sha256"], "Training sources changed")
    require(common.sha(config["targets"]) == config["target_sha256"], "Target file changed")
    require(not common.read_jsonl(config["targets"]), "Seed-only training received target statements")
    for name, digest in config["code_hashes"].items():
        require(common.sha(trainer_directory / name) == digest, f"Trainer dependency changed: {name}")
    require(common.sha(trainer_directory / "nd_environment.py") == config["checker_sha256"], "Training checker changed")
    require(common.sha(trainer_directory / "minimal_curriculum.py") == config["curriculum_helper_sha256"], "Curriculum changed")
    for path, digest in config["vendor_hashes"].items():
        require(common.sha(path) == digest, "Training vendor dependency changed")
    sources = {r["source_id"]: r for r in common.read_jsonl(config["sources"])}
    for source in sources.values():
        verdict = strict_complete(source["prompt"], source["body"])
        require(verdict["ok"] and verdict["n_lines"] <= 6, "Invalid or over-cap seed certificate")
    archive = directory / "symbolic_general_successes.jsonl"
    counts, histogram, cache, witnesses = Counter(), {}, {}, {}
    previous_action = 0
    rows = common.read_jsonl(archive) if archive.exists() else []
    for row in rows:
        require(row["is_source"] and row["source_id"] in sources, "Unexpected nonseed success")
        source = sources[row["source_id"]]
        require(row["prompt"] == source["prompt"], "Archived theorem differs from source")
        require(previous_action <= row["actions_so_far"] <= summary["sampled_actions"], "Invalid sampled-action counter")
        previous_action = row["actions_so_far"]
        key = (row["source_id"], row["body"], row["cut"])
        if key not in cache:
            env = Environment(row["prompt"], max_lines=config["max_lines"], max_scope=config["max_scope"])
            expected = env.source_actions(source)
            actual = env.source_actions(row)
            require(0 <= row["cut"] < len(expected), "Invalid source reset cut")
            require(actual[:row["cut"]] == expected[:row["cut"]], "Archived body did not retain supplied source prefix")
            if config["mode"] == "full":
                require(row["cut"] == 0, "Full-start training used an assisted prefix")
            for action in actual:
                env.step(action)
            require(env.success() and env.body() == row["body"], "Archived episode is not a legal completed trajectory")
            verdict = strict_complete(row["prompt"], row["body"])
            require(verdict["ok"] and row["verdict"]["ok"], "Archived reward disagrees with verification")
            pruned = env.pruned_body()
            prune_verdict = strict_complete(row["prompt"], pruned)
            require(prune_verdict["ok"], "Dependency-pruned certificate failed verification")
            rules = Counter(x["rule"] for x in parse_proof_tokens(row["body"].split()))
            cache[key] = {"raw_lines": verdict["n_lines"], "pruned_lines": prune_verdict["n_lines"],
                          "raw_reiterations": rules["R"], "pruned_body": pruned}
        item = cache[key]
        counts["successful_episodes_replayed"] += 1
        category = "full_start" if row["cut"] == 0 else "assisted_start"
        counts[category] += 1
        counts[category + "_without_reiteration"] += int(item["raw_reiterations"] == 0)
        histogram.setdefault(category, Counter())[str(item["raw_lines"])] += 1
        if row["cut"] == 0:
            witnesses.setdefault(common.digest([row["prompt"], row["body"]]), {**row, **item})
    require(counts["successful_episodes_replayed"] == summary["accounting"].get("source_episodes_successful", 0),
            "Success archive does not cover the training success ledger")
    require(summary["accounting"].get("target_episodes_started", 0) == 0, "Theorem-only targets were sampled")
    paths = [config_path, summary_path, Path(config["sources"]), Path(config["targets"]),
             directory / "symbolic_general_checkpoint.pt", Path(__file__), trainer_directory / "symbolic_general_train.py"]
    if archive.exists():
        paths.append(archive)
    return {"status": "passed", "run": str(directory.resolve()), "counts": dict(counts),
            "raw_line_histogram": {k: dict(v) for k, v in histogram.items()}, "unique_trajectories_replayed": len(cache),
            "unique_full_start_certificates": len(witnesses), "full_start_witnesses": list(witnesses.values())[:16],
            "sampled_actions": summary["sampled_actions"], "optimizer_steps": summary["optimizer_steps"],
            "final_cuts": summary["cuts"], "input_sha256": {str(p.resolve()): common.sha(p) for p in paths},
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Every archived successful episode is accounted for; identical body/cut pairs reuse a verified trajectory. Failed training episodes are not archived, so their trajectories are not independently replayed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trainer-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.trainer_directory)
    common.write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("full_start_witnesses", "input_sha256")}))
