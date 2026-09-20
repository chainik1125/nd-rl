#!/usr/bin/env python3
"""Audit structured populations and optionally recheck model attempts.

This script never loads a model. With no --attempts it checks reference proofs,
canonical splits/exclusions, restricted lower bounds and token-budget coverage.
With attempt JSONL files it re-verifies successes and reports written/pruned
lengths separately from known reference lengths.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import common
from nd_environment import Tokenizer, parse_prompt, strict_complete, judge
from nd_verify import parse_proof_tokens
from proof_diagnostics import compress_proof
from prepare_data import identities, prompt_of, OLDER
from structured_generate import summary, BUDGETS


def atom_paths(formula, target, path=()):
    if formula == target:
        return [path]
    if formula[0] == "and":
        return atom_paths(formula[1], target, path + (0,)) + atom_paths(formula[2], target, path + (1,))
    return []


def conjunction_only(formula):
    return formula[0] == "atom" or (formula[0] == "and" and all(conjunction_only(f) for f in formula[1:]))


def derive_restricted_bound(prompt, family):
    """Re-derive the row's syntactic fragment bound from its theorem alone."""
    premises, goal = parse_prompt(prompt)
    if family.startswith("projection_"):
        if len(premises) != 1 or goal[0] != "atom":
            raise ValueError("Projection family has unexpected theorem shape")
        context, target, overhead = premises[0], goal, 1
    elif family == "implication_projection":
        implications = [f for f in premises if f[0] == "imp"]
        conjunctions = [f for f in premises if f[0] == "and"]
        if len(premises) != 2 or len(implications) != 1 or len(conjunctions) != 1 or goal[0] != "atom":
            raise ValueError("Implication family has unexpected theorem shape")
        implication, context = implications[0], conjunctions[0]
        if implication[2] != goal or implication[1][0] != "atom":
            raise ValueError("Implication does not bridge target and goal")
        target, overhead = implication[1], 3
        if atom_paths(context, goal):
            raise ValueError("Goal leaks into a conjunction decoy")
    elif family == "scope_projection":
        if premises or goal[0] != "imp" or goal[2][0] != "atom":
            raise ValueError("Scope family has unexpected theorem shape")
        context, target, overhead = goal[1], goal[2], 2
    else:
        raise ValueError("Unknown structured family")
    if not conjunction_only(context):
        raise ValueError("Context is not a positive conjunction tree")
    paths = atom_paths(context, target)
    if len(paths) != 1:
        raise ValueError("Projection target must occur exactly once")
    path = paths[0]
    if abs(path.count(0) - path.count(1)) > 1:
        raise ValueError("Projection path is not balanced left/right")
    return {"depth": len(path), "directions": list(path), "lines": len(path) + overhead,
            "target_atom": target[1]}


def audit_dataset(data):
    manifest = json.loads((data / "manifest.json").read_text())
    for relative, expected in manifest["population_sha256"].items():
        if common.sha(data / relative) != expected:
            raise ValueError(f"Population hash differs from freeze: {relative}")
    excluded_t, excluded_c = set(), set()
    guard_path = OLDER / "data/all_textbook_guards.json"
    if common.sha(guard_path) != manifest["guard_sha256"]:
        raise ValueError("Textbook guard identity changed after freeze")
    guard_rows = json.loads(guard_path.read_text())
    guards = [set(guard_rows[key]) for key in ("canonical_theorem_sha256", "nonempty_premise_context_sha256", "closed_formula_skeleton_sha256")]
    for name, expected in manifest["exposure_sha256"].items():
        path = Path(name)
        if common.sha(path) != expected:
            raise ValueError("An exclusion source changed after freeze")
        for row in common.read_jsonl(path):
            t, c, _ = identities(prompt_of(row))
            excluded_t.add(t)
            excluded_c.add(c)
    tok = Tokenizer()
    populations = {}
    all_theorems = set()
    contexts = {}
    bounds_checked = certificates_checked = 0
    for role in ("targets", "development", "final"):
        rows = common.read_jsonl(data / "audit" / f"{role}.jsonl")
        inputs = common.read_jsonl(data / f"{role}.jsonl")
        if len(rows) != len(inputs):
            raise ValueError("Input/audit population size mismatch")
        for row, theorem_input in zip(rows, inputs):
            if set(theorem_input) != {"name", "source_id", "prompt"}:
                raise ValueError("Theorem-only input carries labels or extra fields")
            if any(theorem_input[k] != row[k] for k in theorem_input):
                raise ValueError("Input/audit identity mismatch")
            t, c, guard = identities(row["prompt"])
            if t != row["theorem_id"] or c != row["context_id"]:
                raise ValueError("Canonical identities do not recompute")
            if t in excluded_t or c in excluded_c or t in all_theorems:
                raise ValueError("Exposure or canonical duplicate survived exclusion")
            if any(key is not None and key in values for key, values in zip(guard, guards)):
                raise ValueError("Textbook guard survived exclusion")
            all_theorems.add(t)
            if c in contexts and contexts[c] != role:
                raise ValueError("Cross-role canonical context overlap")
            contexts[c] = role
            expected_role = "targets" if int(c[:12], 16) % 100 < 50 else "development" if int(c[:12], 16) % 100 < 75 else "final"
            if role != expected_role:
                raise ValueError("Split does not follow frozen context hash")
            bound = derive_restricted_bound(row["prompt"], row["family"])
            if bound["lines"] != row["n_lines"] or bound["lines"] != row["restricted_lower_bound_lines"]:
                raise ValueError("Restricted bound differs from reference length")
            if bound["directions"] != row["directions"] or bound["depth"] != row["projection_depth"]:
                raise ValueError("Stored projection path differs from theorem")
            bounds_checked += 1
            for body in (row["body"], row["compressed_body"]):
                verdict = strict_complete(row["prompt"], body)
                if not verdict["ok"] or verdict["n_lines"] != bound["lines"]:
                    raise ValueError("Reference or compressed certificate failed strict verification")
                certificates_checked += 1
            compressed = compress_proof(row["prompt"], row["body"])
            if compressed["dependency_pruned_length"] != bound["lines"]:
                raise ValueError("Reference has removable dependencies")
            prompt_n = len(tok.encode_prompt(row["prompt"]))
            response_n = 1 + len(tok.encode_record(row["prompt"] + " " + row["body"])) - prompt_n
            if response_n != row["reference_tokens"] or prompt_n + 1 != row["prompt_tokens"]:
                raise ValueError("Stored reference token accounting is inconsistent")
        populations[role] = rows
    return populations, {"status": "passed", "theorems": len(all_theorems), "canonical_contexts": len(contexts),
                         "restricted_bounds_checked": bounds_checked, "strict_certificates_checked": certificates_checked,
                         "excluded_theorems": len(excluded_t), "excluded_contexts": len(excluded_c),
                         "cross_role_context_overlap": 0, "theorem_only_input_fields": ["name", "source_id", "prompt"],
                         "population_manifest_sha256": common.sha(data / "manifest.json"),
                         "summary_by_role": {role: summary(rows) for role, rows in populations.items()}}


def recheck_attempts(path, reference):
    tok = Tokenizer()
    out = []
    for row in common.read_jsonl(path):
        if row["source_id"] not in reference:
            raise ValueError("Attempt theorem is outside selected structured population")
        target = reference[row["source_id"]]
        if row["prompt"] != target["prompt"] or row.get("reset_index", 0) != 0 or row.get("prefix_ids", []):
            raise ValueError("Attempt prompt mismatch or supplied proof prefix")
        if "response_ids" in row:
            verdict = judge({"prompt": row["prompt"], "prefix_ids": []}, row["response_ids"], tok)
            generated_tokens = len(row["response_ids"])
        else:
            verdict = strict_complete(row["prompt"], row.get("body", ""))
            verdict["reward"] = int(verdict["ok"])
            verdict["body"] = row.get("body", "")
            generated_tokens = row.get("generated_tokens")
        if int(row.get("reward", verdict["reward"])) != verdict["reward"]:
            raise ValueError("Reported and independently rechecked reward differ")
        result = {"source_id": row["source_id"], "attempt": row.get("attempt", row.get("attempt_index", 0)),
                  "family": target["family"], "reference_lines": target["n_lines"],
                  "reference_tokens": target["reference_tokens"], "generated_tokens": generated_tokens,
                  "reward": verdict["reward"], "reason": verdict["reason"]}
        if verdict["reward"]:
            compression = compress_proof(row["prompt"], verdict["body"])
            if not strict_complete(row["prompt"], compression["compressed_proof"])["ok"]:
                raise AssertionError("Pruned model certificate failed strict checker")
            result.update(written_lines=verdict["n_lines"], pruned_lines=compression["dependency_pruned_length"],
                          shorter_than_reference=compression["dependency_pruned_length"] < target["n_lines"])
            rules = {line["rule"] for line in parse_proof_tokens(compression["compressed_proof"].split())}
            allowed = {"PR", "ANDE1", "ANDE2"} if target["family"].startswith("projection_") else (
                      {"PR", "ANDE1", "ANDE2", "IMPE"} if target["family"] == "implication_projection" else
                      {"AS", "ANDE1", "ANDE2", "IMPI"})
            result["rules"] = sorted(rules)
            result["uses_rules_outside_bound_fragment"] = bool(rules - allowed)
            if result["shorter_than_reference"] and not (rules - allowed) and target["family"] != "scope_projection":
                raise AssertionError("A strictly valid proof violates the claimed projection/IMPE-fragment bound")
        out.append(result)
    return out


def aggregate(attempts, references):
    by_theorem = defaultdict(list)
    for row in attempts:
        by_theorem[row["source_id"]].append(row)
    for rows in by_theorem.values():
        indices = [r["attempt"] for r in rows]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate attempt index for a theorem")
    strata = {"all": list(references)}
    for key, row in references.items():
        strata.setdefault(f"{row['family']}/lines{row['n_lines']}", []).append(key)
    result = {}
    for name, ids in sorted(strata.items()):
        tested = [i for i in ids if i in by_theorem]
        subset = [row for i in tested for row in by_theorem[i]]
        good = [row for row in subset if row["reward"]]
        first = [min(by_theorem[i], key=lambda r: r["attempt"]) for i in tested]
        result[name] = {"planned_theorems": len(ids), "tested_theorems": len(tested), "missing_theorems": len(ids) - len(tested),
                        "attempts": len(subset), "attempt_successes": len(good),
                        "first_successes": sum(r["reward"] for r in first),
                        "any_successes": sum(any(r["reward"] for r in by_theorem[i]) for i in tested),
                        "successful_written_lengths": dict(Counter(r["written_lines"] for r in good)),
                        "successful_pruned_lengths": dict(Counter(r["pruned_lines"] for r in good)),
                        "successful_pruned_above6": sum(r["pruned_lines"] > 6 for r in good),
                        "shorter_than_reference": sum(r["shorter_than_reference"] for r in good),
                        "outside_restricted_fragment_successes": sum(r["uses_rules_outside_bound_fragment"] for r in good),
                        "generated_tokens": sum(r["generated_tokens"] or 0 for r in subset),
                        "unknown_token_counts": sum(r["generated_tokens"] is None for r in subset),
                        "failure_reasons": dict(Counter(r["reason"] for r in subset if not r["reward"])),
                        "known_reference_fits": {str(b): sum(references[i]["reference_tokens"] <= b for i in tested) for b in BUDGETS}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=common.ROOT / "data/structured_v1")
    p.add_argument("--role", choices=("all", "targets", "development", "final"), default="all")
    p.add_argument("--attempts", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, default=common.ROOT / "reports/structured_audit.json")
    args = p.parse_args()
    populations, audit = audit_dataset(args.data)
    references = {r["source_id"]: r for role, rows in populations.items() if args.role == "all" or args.role == role for r in rows}
    result = {"dataset_audit": audit, "analysis_code_sha256": common.sha(__file__),
              "role": args.role, "runs": {}, "no_model_evaluation_performed": not args.attempts}
    for path in args.attempts:
        attempts = recheck_attempts(path, references)
        result["runs"][str(path.resolve())] = {"attempts_sha256": common.sha(path), "metrics": aggregate(attempts, references)}
    common.write_json(args.output, result)
    print(json.dumps({"audit": audit["status"], "theorems": audit["theorems"], "runs": len(result["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
