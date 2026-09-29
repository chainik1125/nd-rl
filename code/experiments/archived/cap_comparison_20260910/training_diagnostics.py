"""Describe observed RL reward information using training logs only."""
from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path

from analyze import read, sha, write


def summarize_groups(rows, group_size):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    totals = Counter()
    for group in by_group.values():
        if len(group) != group_size or {r["attempt"] for r in group} != set(range(group_size)):
            raise ValueError("Incomplete or duplicate training group")
        if len({r["prompt"] for r in group}) != 1:
            raise ValueError("Training group contains different theorems")
        correct = sum(r["reward"] for r in group)
        p = correct / group_size
        totals["groups"] += 1
        totals["mixed_groups"] += int(0 < correct < group_size)
        totals["all_success_groups"] += int(correct == group_size)
        totals["all_failure_groups"] += int(correct == 0)
        totals["responses"] += len(group)
        totals["correct_responses"] += correct
        totals["generated_tokens"] += sum(len(r["response_ids"]) for r in group)
        totals["nonzero_advantage_tokens"] += sum(len(r["response_ids"]) for r in group if r["advantage"])
        totals["empirical_reward_entropy_bits_sum"] += -p * math.log2(p) - (1-p) * math.log2(1-p) if 0 < p < 1 else 0
    return totals


def summarize(root, arm):
    base = root / "runs" / arm / "rl"
    completion = read(base / "completed.json")
    if completion["budget_reached"] is not True:
        raise ValueError("Diagnostics require completed RL")
    manifest = read(base / "manifest.json")
    data = root / "data/rl_common.jsonl"
    if sha(data) != manifest["data_sha256"]:
        raise ValueError("Training data identity changed")
    targets = {r["prompt"]: r for l in data.read_text().splitlines() if (r := json.loads(l))}
    events = {r["step"]: r for l in (base / "events.jsonl").read_text().splitlines()
              if (r := json.loads(l)).get("kind") == "rl"}
    buckets, trajectory, support = defaultdict(Counter), [], defaultdict(set)
    families, family_support = defaultdict(Counter), defaultdict(set)
    current, rows = None, []

    def consume(step, batch):
        all_totals = summarize_groups(batch, manifest["args"]["group_size"])
        event = events[step]
        if (all_totals["mixed_groups"] != event["groups_mixed"]
                or all_totals["nonzero_advantage_tokens"] != event["nonzero_advantage_response_tokens"]
                or all_totals["generated_tokens"] != event["policy_response_tokens"]):
            raise ValueError("Rollout information differs from training event")
        trajectory.append({"step": step, "total_flop_proxy": event["total_flop_proxy"], **all_totals})
        subset, family_subset = defaultdict(list), defaultdict(list)
        for row in batch:
            if row["prompt"] not in targets:
                raise ValueError("Training rollout theorem absent from frozen RL pool")
            bucket = targets[row["prompt"]]["sampling_bucket"]
            subset[bucket].append(row)
            support[bucket].add(row["prompt"])
            family = bucket + "/" + targets[row["prompt"]].get("family", "unlabelled")
            family_subset[family].append(row)
            family_support[family].add(row["prompt"])
        for bucket, group in subset.items():
            buckets[bucket].update(summarize_groups(group, manifest["args"]["group_size"]))
        for family, group in family_subset.items():
            families[family].update(summarize_groups(group, manifest["args"]["group_size"]))

    with (base / "rollouts.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if current is not None and row["step"] != current:
                if row["step"] != current + 1:
                    raise ValueError("Non-contiguous rollout steps")
                consume(current, rows)
                rows = []
            current = row["step"]
            rows.append(row)
        if rows:
            consume(current, rows)
    if len(trajectory) != completion["actual_steps"]:
        raise ValueError("Training diagnostic trajectory is incomplete")
    for collection, seen in ((buckets, support), (families, family_support)):
        for key, totals in collection.items():
            totals["unique_prompts_seen"] = len(seen[key])
            totals["mixed_group_fraction"] = totals["mixed_groups"] / totals["groups"]
            totals["signal_token_fraction"] = totals["nonzero_advantage_tokens"] / totals["generated_tokens"]
            totals["mean_empirical_reward_entropy_bits"] = totals["empirical_reward_entropy_bits_sum"] / totals["groups"]
    for field in ("groups", "responses", "correct_responses", "generated_tokens", "nonzero_advantage_tokens"):
        if sum(v[field] for v in families.values()) != sum(v[field] for v in buckets.values()):
            raise ValueError("Family totals differ from verified source totals")
    return {"arm": arm, "rl_checkpoint_sha256": completion["final_sha256"],
            "events_sha256": sha(base / "events.jsonl"), "rollouts_sha256": sha(base / "rollouts.jsonl"),
            "training_only": True,
            "interpretation": "Observed within-group reward variation at eight samples; empirical entropy is downward biased and is not epistemic uncertainty or a causal estimate of learning value.",
            "buckets": dict(buckets), "families": dict(families), "trajectory": trajectory}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    result = summarize(args.root.resolve(), args.arm)
    write(args.root / "reports" / f"{args.arm}_training_information.json", result)
    print(json.dumps({"arm": args.arm, "buckets": result["buckets"]}), flush=True)


if __name__ == "__main__":
    main()
