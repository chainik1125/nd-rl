"""Recheck frozen raw outputs and summarize mechanism/failure diagnostics."""
from collections import Counter
import json
from pathlib import Path
import random
import torch

import common
from nd_environment import strict_complete
from symbolic_nd import Environment, Policy, fsize

ROOT = common.ROOT
torch.set_num_threads(4)
population = {r["source_id"]: r for r in common.read_jsonl(ROOT / "data/symbolic_final.jsonl")}
summaries, failures, verified, raw = {}, {}, Counter(), {}
for folder in sorted((ROOT / "results").glob("symbolic_eval_*")):
    summary_path = folder / "symbolic_eval_summary.json"
    if not summary_path.exists():
        continue
    tag = folder.name.removeprefix("symbolic_eval_")
    summary = json.loads(summary_path.read_text())
    attempts = common.read_jsonl(folder / "symbolic_attempts.jsonl")
    raw[tag] = attempts
    groups = {}
    for family, depth in sorted({(a["family"], a["depth"]) for a in attempts}):
        group = [a for a in attempts if a["family"] == family and a["depth"] == depth]
        successes = [a for a in group if a["success"]]
        key = f"{family}/d{depth}"
        assert len(successes) == summary["groups"][key]["successes"]
        groups[key] = {"successes": len(successes), "attempts": len(group),
                       "mean_analytic_random": summary["groups"][key]["mean_random_analytic"],
                       "mean_actions": summary["groups"][key]["mean_generated_actions"]}
    for a in attempts:
        verified["raw_attempts"] += 1
        row = population[a["source_id"]]
        assert a["prompt"] == row["prompt"]
        env = Environment(row, budget=a["budget"])
        reward = 0.
        for source, side in a["action_trace"]:
            available = env.actions()
            matching = [candidate for candidate in available if candidate.source == source - 1 and candidate.side == side]
            assert len(matching) == 1
            reward, done = env.step(matching[0])
        assert done and bool(reward) == a["success"] and env.body() == a["body"]
        if a["success"]:
            verified["successful_certificates_reverified"] += 1
            full = strict_complete(a["prompt"], a["body"])
            pruned = strict_complete(a["prompt"], a["pruned_body"])
            assert full["ok"] and pruned["ok"]
            assert pruned["n_lines"] == row["depth"] + 1
            assert env.pruned_body() == a["pruned_body"]
    summaries[tag] = {"groups": groups, "attempts_sha256": common.sha(folder / "symbolic_attempts.jsonl"),
                      "summary_sha256": common.sha(summary_path)}

    if tag.startswith(("full_", "reset_")):
        model = Policy()
        checkpoint = ROOT / f"results/symbolic_{tag}/symbolic_checkpoint.pt"
        model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"])
        model.eval()
        first_divergences = Counter()
        examples = []
        for a in attempts:
            if a["success"]:
                continue
            row = population[a["source_id"]]
            env = Environment(row)
            for step, ((source, side), true_side) in enumerate(zip(a["action_trace"], row["directions"])):
                aa = env.actions()
                chosen = next(x for x in aa if x.source == source - 1 and x.side == side)
                expected_result = env.formulas[-1][true_side + 1]
                expected = next(x for x in aa if x.result == expected_result)
                if chosen.result != expected_result:
                    first_divergences[(row["family"], row["depth"], step)] += 1
                    if len([x for x in examples if x["family"] == row["family"]]) < 2:
                        af, sf = env.features(aa)
                        with torch.no_grad():
                            logits, _ = model(torch.from_numpy(af), torch.from_numpy(sf))
                        examples.append({"source_id": a["source_id"], "family": row["family"], "depth": row["depth"],
                          "first_wrong_step_zero_based": step, "chosen_source": chosen.source + 1,
                          "correct_source": expected.source + 1, "chosen_side": chosen.side, "correct_side": expected.side,
                          "chosen_formula_size": fsize(chosen.result), "correct_formula_size": fsize(expected.result),
                          "chosen_equals_goal": chosen.result == env.goal, "correct_equals_goal": expected.result == env.goal,
                          "chosen_logit": logits[aa.index(chosen)].item(), "correct_logit": logits[aa.index(expected)].item(),
                          "prompt": row["prompt"], "attempt_body": a["body"]})
                    break
                env.step(chosen)
        failures[tag] = {"first_divergence_counts": [{"family": k[0], "depth": k[1], "step": k[2], "count": v}
                        for k, v in sorted(first_divergences.items())], "examples": examples}

initial_checks = {}
gains = {}
for seed in (4101, 4102):
    initials = [torch.load(ROOT / f"results/symbolic_{mode}_s{seed}/symbolic_initial.pt",
                 map_location="cpu", weights_only=False)["state_dict"] for mode in ("full", "reset")]
    assert all(torch.equal(initials[0][key], initials[1][key]) for key in initials[0])
    initial_checks[str(seed)] = "All state_dict tensors identical between full and reset arms"
    initial_success = {x["source_id"] for x in raw[f"initial_s{seed}"] if x["success"]}
    for mode in ("full", "reset"):
        rows = raw[f"{mode}_s{seed}"]
        groups = {}
        for family, depth in sorted({(x["family"], x["depth"]) for x in rows}):
            group = [x for x in rows if x["family"] == family and x["depth"] == depth]
            groups[f"{family}/d{depth}"] = {
                "new_successes_over_initial": sum(x["success"] and x["source_id"] not in initial_success for x in group),
                "lost_initial_successes": sum(not x["success"] and x["source_id"] in initial_success for x in group)}
        gains[f"{mode}_s{seed}"] = groups

example = next(x for x in raw["reset_s4101"] if x["family"] == "spine" and x["depth"] == 63 and x["success"])
common.write_json(ROOT / "reports/symbolic_64line_witness.json", example)
common.write_json(ROOT / "reports/symbolic_failures.json", failures)
report = {"audited_counts": dict(verified), "evaluations": summaries, "initial_tensor_equality": initial_checks,
          "gains_vs_exact_initial": gains, "successful_witness_sha256": common.sha(ROOT / "reports/symbolic_64line_witness.json"),
          "population_sha256": common.sha(ROOT / "data/symbolic_final.jsonl"),
          "training": {f"{mode}_s{seed}": json.loads((ROOT / f"results/symbolic_{mode}_s{seed}/symbolic_train_summary.json").read_text())
                       for mode in ("full", "reset") for seed in (4101, 4102)}}
common.write_json(ROOT / "reports/symbolic_audit.json", report)
print(json.dumps({"audited_counts": dict(verified), "initial_tensor_equality": initial_checks,
                  "failures": {tag: d["first_divergence_counts"] for tag, d in failures.items()}}, indent=2))
