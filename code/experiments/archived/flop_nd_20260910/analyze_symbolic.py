"""Replay every legal-action evaluation and summarize actual certificates.

Labels are joined after evaluation. Both raw and pruned certificates are checked
again. Token eligibility means exact serialization in the LM response codec,
including FINAL/QED; it is not a claim of matched inference compute.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import common
from nd_environment import strict_complete, Tokenizer
from nd_verify import parse_proof_tokens
from proof_diagnostics import compress_proof
from symbolic_general_env import Environment, Action

BUDGETS = (512, 1024, 2048)


def tup(value):
    return tuple(tup(x) for x in value) if isinstance(value, list) else value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def response_tokens(tok, prompt, body):
    return 1 + len(tok.encode_record(prompt + " " + body)) - len(tok.encode_prompt(prompt))


def strata(audit):
    result = {"all": sorted(audit), "reference_over6": sorted(k for k, r in audit.items() if r["n_lines"] > 6)}
    for key in ("family", "length_bin", "target_atom"):
        values = sorted({r[key] for r in audit.values() if key in r})
        for value in values:
            result[f"{key}:{value}"] = sorted(k for k, r in audit.items() if r.get(key) == value)
    if any("target_atom" in r for r in audit.values()):
        result["target_atom:not_P"] = sorted(k for k, r in audit.items() if r.get("target_atom") != "P")
        for family in sorted({r["family"] for r in audit.values()}):
            for length in sorted({r["n_lines"] for r in audit.values()}):
                result[f"family_length:{family}:{length}"] = sorted(k for k, r in audit.items() if r["family"] == family and r["n_lines"] == length)
    return {k: v for k, v in result.items() if v}


def summarize(rows, ids):
    selected = [rows[sid] for sid in ids]
    metrics = ("first", "any", "attempts", "successful_attempts", "actions", "candidates_scored",
               "any_pruned_over6", "shortest_found_pruned_over6", "any_pruned_at_least13", "any_pruned_at_least25")
    result = {"theorems": len(ids), **{key: sum(r[key] for r in selected) for key in metrics}}
    for budget in BUDGETS:
        for kind in ("raw", "pruned"):
            for profile in ("first", "any"):
                key = f"{kind}_eligible_{budget}_{profile}"
                result[key] = sum(r[key] for r in selected)
    result["first_rate"] = result["first"] / len(ids)
    result["any_rate"] = result["any"] / len(ids)
    result["shortest_pruned_histogram"] = dict(sorted(Counter(r["shortest_pruned"] for r in selected if r["any"]).items()))
    result["maximum_pruned"] = max((r["maximum_pruned"] for r in selected), default=0)
    for rule in ("IMPI", "ORE", "NEGI", "DN", "BOTE"):
        result[f"any_pruned_using_{rule}"] = sum(any(s["pruned_rules"].get(rule, 0) for s in r["successes"]) for r in selected)
    return result


def analyze(directory, audit_path, evaluator_directory):
    directory, audit_path, evaluator_directory = map(Path, (directory, audit_path, evaluator_directory))
    result_path = directory / "symbolic_general_eval_summary.json"
    result = json.loads(result_path.read_text())
    config_path = directory / "symbolic_general_eval_config.json"
    config = json.loads(config_path.read_text())
    require(result.get("status") == "complete", "Partial evaluation cannot enter final analysis")
    require(result["config"] == config, "Config/result identity mismatch")
    require(common.sha(config["population"]) == config["population_sha256"], "Population changed")
    if config["checkpoint"]:
        require(common.sha(config["checkpoint"]) == config["checkpoint_sha256"], "Evaluated checkpoint changed")
    for name, digest in config["code_hashes"].items():
        require(common.sha(evaluator_directory / name) == digest, "Evaluation program changed")
    require(common.sha(evaluator_directory / "nd_environment.py") == config["checker_sha256"], "Evaluation checker changed")
    require(common.sha(Path(__file__).with_name("symbolic_general_env.py")) == common.sha(evaluator_directory / "symbolic_general_env.py"), "Audit/evaluation environment mismatch")
    attempts_path = directory / "symbolic_general_attempts.jsonl"
    require(common.sha(attempts_path) == result["attempts_sha256"], "Raw attempt archive changed")
    population = common.read_jsonl(config["population"])
    reference = common.read_jsonl(audit_path)
    audit = {r["source_id"]: r for r in reference}
    require(len(audit) == len(reference), "Duplicate audit identities")
    require({r["source_id"]: r["prompt"] for r in population} == {k: r["prompt"] for k, r in audit.items()}, "Audit/population mismatch")
    require(len(population) == len(audit), "Duplicate theorem-only identities")
    tok, grouped, counts, witnesses = Tokenizer(), defaultdict(list), Counter(), []
    with attempts_path.open() as stream:
        for text in stream:
            raw = json.loads(text)
            sid = raw["source_id"]
            require(sid in audit and raw["prompt"] == audit[sid]["prompt"], "Unknown theorem in raw attempts")
            require(raw["supplied_prefix_lines"] == 0, "Evaluation received a proof prefix")
            env = Environment(raw["prompt"], max_lines=config["max_lines"], max_scope=config["max_scope"])
            require(len(env.premises) == raw["registered_premise_lines"], "Premise accounting mismatch")
            scored = 0
            for index, trace in enumerate(raw["action_trace"]):
                require(not env.success(), "Action after successful termination")
                actions = env.actions()
                action = Action(trace["rule"], tup(trace["formula"]), tuple(trace["refs"]), trace["depth"])
                require(len(actions) == trace["candidate_count"], "Candidate count mismatch")
                require(0 <= trace["selected_index"] < len(actions) and actions[trace["selected_index"]] == action, "Selected action absent from legal candidates")
                scored += len(actions)
                env.step(action, check_membership=False)
                counts["inferences_replayed"] += 1
            require(env.generated == raw["generated_actions"] == len(raw["action_trace"]), "Generated action accounting mismatch")
            require(scored == raw["candidates_scored"], "Candidate scoring accounting mismatch")
            require(env.body() == raw["body"], "Replayed certificate differs from raw output")
            verdict = strict_complete(raw["prompt"], raw["body"])
            require(verdict["ok"] == bool(raw["success"]) == env.success(), "Strict checker/reward mismatch")
            require(raw["verdict"] == verdict, "Stored verifier result differs")
            record = {"attempt": raw["attempt"], "success": int(raw["success"]), "actions": env.generated, "candidates_scored": scored}
            counts["attempts_replayed"] += 1
            if raw["success"]:
                body = env.pruned_body()
                pruned = strict_complete(raw["prompt"], body)
                require(pruned["ok"] and body == raw["pruned_body"] and raw["pruned_verdict"] == pruned, "Pruned certificate mismatch")
                compressed = compress_proof(raw["prompt"], raw["body"])
                require(compressed["raw_valid"] and strict_complete(raw["prompt"], compressed["compressed_proof"])["ok"], "Independent compression failed")
                rules = dict(Counter(line["rule"] for line in parse_proof_tokens(body.split())))
                record.update(raw_lines=verdict["n_lines"], pruned_lines=pruned["n_lines"],
                              compressed_lines=compressed["compressed_length"], pruned_rules=rules,
                              raw_tokens=response_tokens(tok, raw["prompt"], raw["body"]),
                              pruned_tokens=response_tokens(tok, raw["prompt"], body))
                counts["successful_certificates_reverified"] += 1
                if pruned["n_lines"] > 6 or rules.get("ORE") or rules.get("NEGI"):
                    witnesses.append({"source_id": sid, "family": audit[sid]["family"],
                        "reference_lines": audit[sid]["n_lines"], "target_atom": audit[sid].get("target_atom"),
                        "prompt": raw["prompt"], "body": raw["body"], "pruned_body": body, **record})
            grouped[sid].append(record)
    require(set(grouped) == set(audit), "Incomplete population coverage")
    rows = {}
    for sid, group in grouped.items():
        group.sort(key=lambda x: x["attempt"])
        require([r["attempt"] for r in group] == list(range(config["attempts"])), "Missing or duplicated attempts")
        successes = [r for r in group if r["success"]]
        pruned = [r["pruned_lines"] for r in successes]
        row = {"first": group[0]["success"], "any": int(bool(successes)), "attempts": len(group),
               "successful_attempts": len(successes), "actions": sum(r["actions"] for r in group),
               "candidates_scored": sum(r["candidates_scored"] for r in group),
               "any_pruned_over6": int(any(n > 6 for n in pruned)),
               "shortest_found_pruned_over6": int(bool(pruned) and min(pruned) > 6),
               "any_pruned_at_least13": int(any(n >= 13 for n in pruned)),
               "any_pruned_at_least25": int(any(n >= 25 for n in pruned)),
               "shortest_pruned": min(pruned, default=None), "maximum_pruned": max(pruned, default=0),
               "successes": successes}
        for budget in BUDGETS:
            for kind in ("raw", "pruned"):
                row[f"{kind}_eligible_{budget}_first"] = int(bool(group[0]["success"]) and group[0][f"{kind}_tokens"] <= budget)
                row[f"{kind}_eligible_{budget}_any"] = int(any(r[f"{kind}_tokens"] <= budget for r in successes))
        rows[sid] = row
    require(counts["attempts_replayed"] == result["attempts"] == len(audit) * config["attempts"], "Summary attempt accounting mismatch")
    require(sum(r["any"] for r in rows.values()) == result["theorems_with_any_success"], "Summary theorem success accounting mismatch")
    require(counts["successful_certificates_reverified"] == result["groups"]["all"]["successes"], "Summary attempt success accounting mismatch")
    witnesses.sort(key=lambda x: (-x["pruned_lines"], x["source_id"], x["attempt"]))
    unique = list({common.digest([w["prompt"], w["pruned_body"]]): w for w in witnesses}.values())
    paths = [audit_path, config_path, result_path, attempts_path, Path(__file__), Path(__file__).with_name("symbolic_general_env.py"),
             Path(__file__).with_name("nd_environment.py"), *common.VENDOR.rglob("*.py")]
    return {"status": "complete", "run": str(directory.resolve()), "audit": str(audit_path.resolve()),
        "profile": {k: config[k] for k in ("policy", "attempts", "sample", "seed", "max_lines", "max_scope")},
        "checkpoint_sha256": config["checkpoint_sha256"], "root_replay_counts": dict(counts),
        "input_sha256": {str(p.resolve()): common.sha(p) for p in paths},
        "summaries": {key: summarize(rows, ids) for key, ids in strata(audit).items()},
        "rows": rows, "witnesses": unique[:64],
        "interpretation": "Unaided theorem inputs in a symbolic legal-action interface; premises and serialization are supplied by the environment. Exact response token eligibility is not neural compute matching. Pruned length is a certificate property, not an unrestricted proof lower bound."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--evaluator-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.audit, args.evaluator_directory)
    common.write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "counts": result["root_replay_counts"], "all": result["summaries"]["all"]}))


if __name__ == "__main__":
    main()
