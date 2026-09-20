"""Audit finite proposer coverage and soundness on authorized sources/targets."""
from collections import Counter
import json
from pathlib import Path
import random
import time
import os

import common
from nd_environment import independent_replay, strict_complete
from symbolic_general_env import Environment, Line, render_line, subformulas, formula_size

ROOT = common.ROOT


def check_proposal(env, action):
    idx = len(env.lines) + 1
    ctx = (env.stack[:action.depth - 1] + (idx,)) if action.rule == "AS" else env.stack[:action.depth]
    added = Line(action.formula, action.rule, action.refs, ctx)
    independent_replay(env.prompt, env.prefix() + " " + render_line(idx, added))


def main():
    sources_path, targets_path = ROOT / "data/v1/sources.jsonl", ROOT / "data/v1/targets.jsonl"
    sources, targets = common.read_jsonl(sources_path), common.read_jsonl(targets_path)
    started = time.monotonic()
    report = {"pid": os.getpid(), "start_unix": time.time(), "input_hashes": {
        "sources": common.sha(sources_path), "targets": common.sha(targets_path)},
        "code_sha256": common.sha(Path(__file__).with_name("symbolic_general_env.py")),
        "checker_sha256": common.sha(Path(__file__).with_name("nd_environment.py")),
        "sources": [], "pilot": {}}
    totals, action_counts, all_rules = Counter(), [], Counter()
    for row in sources:
        env = Environment(row["prompt"], max_scope=4, max_lines=24)
        desired = env.source_actions(row)
        supported, unsupported = True, None
        for step, expected in enumerate(desired):
            aa = env.actions()
            action_counts.append(len(aa))
            for a in aa:
                check_proposal(env, a)
                totals["source_prefix_proposals_independently_checked"] += 1
                all_rules[a.rule] += 1
            if expected not in aa:
                supported, unsupported = False, {"generated_step": step, "rule": expected.rule,
                    "depth": expected.depth, "formula": expected.formula,
                    "formula_allowed": env.formula_allowed(expected.formula), "refs": expected.refs,
                    "prefix_already_terminal": env.success()}
                totals["unsupported_sources"] += 1
                break
            env.step(expected)
            totals["source_reference_actions_supported"] += 1
        if supported:
            verdict = strict_complete(env.prompt, env.body())
            assert verdict["ok"] and verdict["n_lines"] == row["n_lines"]
            totals["fully_supported_sources"] += 1
        report["sources"].append({"source_id": row["source_id"], "family": row["family"],
             "base_pool_size": len(env.base), "base_pool_max_formula_nodes": max(map(formula_size, env.base)),
             "generated_reference_actions": len(desired), "fully_supported": supported,
             "unsupported": unsupported})

    # Fixed uniform-policy throughput/soundness pilot; training-target prompts only.
    rng = random.Random(59100)
    selected = rng.sample(targets, 32)
    pilot_start = time.monotonic()
    pilot_actions, pilot_successes, pilot_proposals = 0, 0, 0
    pilot_rows = []
    for row in selected:
        env = Environment(row["prompt"], max_scope=4, max_lines=24)
        largest = 0
        for _ in range(24):
            aa = env.actions()
            if not aa:
                break
            largest = max(largest, len(aa))
            # Audit a deterministic random sample of candidates, plus the chosen action.
            for a in rng.sample(aa, min(8, len(aa))):
                check_proposal(env, a)
                pilot_proposals += 1
            action = rng.choice(aa)
            check_proposal(env, action)
            pilot_proposals += 1
            reward, done = env.step(action, check_membership=False)
            pilot_actions += 1
            if done:
                pilot_successes += int(reward)
                break
        pilot_rows.append({"source_id": row["source_id"], "prompt": row["prompt"], "success": env.success(),
                           "generated_actions": env.generated, "max_candidates": largest,
                           "base_pool_size": len(env.base), "body": env.body(),
                           "verdict": strict_complete(env.prompt, env.body())})
    report["totals"] = dict(totals)
    report["candidate_rule_counts"] = dict(all_rules)
    report["source_action_counts"] = {"min": min(action_counts), "max": max(action_counts),
              "mean": sum(action_counts) / len(action_counts), "n_states": len(action_counts)}
    report["pilot"] = {"target_count": len(selected), "generated_actions": pilot_actions,
             "successes": pilot_successes, "proposals_independently_checked": pilot_proposals,
             "elapsed_seconds": time.monotonic() - pilot_start, "max_candidates": max(x["max_candidates"] for x in pilot_rows),
             "seed": 59100, "max_scope": 4, "max_lines": 24}
    report["elapsed_seconds"] = time.monotonic() - started
    common.write_json(ROOT / "reports/symbolic_general_coverage.json", report)
    common.write_jsonl(ROOT / "results/symbolic_general_pilot_attempts.jsonl", pilot_rows)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
