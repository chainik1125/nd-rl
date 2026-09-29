#!/usr/bin/env python3
"""Frozen theorem-only repeated-procedure challenges, with separate audit proofs.

There are no model calls. Every audited reference follows a simple explicit
schema, is checked by both strict checkers, and survives dependency pruning.
Reference length is not asserted to be the unrestricted ND minimum.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import time

import common
from nd_data import formula_text
from nd_environment import Tokenizer, strict_complete, split_body, parse_prompt
from proof_diagnostics import compress_proof
from prepare_data import identities, prompt_of, PRIOR, OLDER

FAMILIES = ("projection_atomic", "projection_compound", "implication_projection", "scope_projection")
LENGTHS = (7, 9, 13, 17, 25)
BUDGETS = (512, 1024, 2048)


def atom(name):
    return ("atom", name)


def line(index, formula, rule, refs=(), depth=0):
    return " ".join([f"N{index}"] + ["|"] * depth + [formula_text(formula), ":", rule]
                    + [f"N{r}" for r in refs] + [";"])


def balanced_directions(rng, depth):
    n_left = depth // 2 + (rng.randrange(2) if depth % 2 else 0)
    values = [0] * n_left + [1] * (depth - n_left)
    rng.shuffle(values)
    return values


def make_decoy(rng, atoms, mode):
    if mode == "atomic":
        return atom(rng.choice(atoms))
    # A compound decoy has exactly two leaves. This introduces distractor
    # branches without the enormous references of depth-three decoy trees.
    return ("and", atom(rng.choice(atoms)), atom(rng.choice(atoms)))


def spine(rng, depth, directions, decoy_atoms, mode):
    current = atom("P")
    for direction in reversed(directions):
        selected_mode = ("compound" if rng.random() < .25 else "atomic") if mode == "mixed" else mode
        decoy = make_decoy(rng, decoy_atoms, selected_mode)
        current = ("and", current, decoy) if direction == 0 else ("and", decoy, current)
    return current


def rename(formula, mapping):
    if formula[0] == "atom":
        return atom(mapping[formula[1]])
    return (formula[0], *(rename(child, mapping) for child in formula[1:]))


def make_task(rng, n_lines, family):
    overhead = 1 if family.startswith("projection_") else 3 if family == "implication_projection" else 2
    depth = n_lines - overhead
    directions = balanced_directions(rng, depth)
    mode = "compound" if family == "projection_compound" else "mixed" if family in ("scope_projection", "implication_projection") else "atomic"
    decoy_atoms = "RS" if family == "implication_projection" else "QRS"
    context = spine(rng, depth, directions, decoy_atoms, mode)
    names = list("PQRS")
    rng.shuffle(names)
    mapping = dict(zip("PQRS", names))
    context, target, implication_goal = rename(context, mapping), atom(mapping["P"]), atom(mapping["Q"])
    implication = ("imp", target, implication_goal)
    body = []
    if family == "scope_projection":
        prompt = f"THM SEQ {formula_text(('imp', context, target))} PRF"
        body.append(line(1, context, "AS", depth=1))
        active_depth, ancestor_line = 1, 1
    elif family == "implication_projection":
        premises = [context, implication]
        if rng.randrange(2):
            premises.reverse()
        prompt = "THM " + " , ".join(formula_text(f) for f in premises) + f" SEQ {formula_text(implication_goal)} PRF"
        body.extend(line(i + 1, f, "PR") for i, f in enumerate(premises))
        active_depth, ancestor_line = 0, premises.index(context) + 1
        implication_line = premises.index(implication) + 1
    else:
        prompt = f"THM {formula_text(context)} SEQ {formula_text(target)} PRF"
        body.append(line(1, context, "PR"))
        active_depth, ancestor_line = 0, 1
    current = context
    for direction in directions:
        current = current[direction + 1]
        index = len(body) + 1
        body.append(line(index, current, "ANDE1" if direction == 0 else "ANDE2", (ancestor_line,), active_depth))
        ancestor_line = index
    if family == "scope_projection":
        body.append(line(len(body) + 1, ("imp", context, target), "IMPI", (1, ancestor_line)))
    elif family == "implication_projection":
        body.append(line(len(body) + 1, implication_goal, "IMPE", (implication_line, ancestor_line)))
    certificate = " ".join(body) + " QED"
    if len(body) != n_lines:
        raise AssertionError("Length schema mismatch")
    restrictions = {
        "projection_atomic": "PR plus ANDE1/ANDE2 only",
        "projection_compound": "PR plus ANDE1/ANDE2 only",
        "implication_projection": "PR plus ANDE1/ANDE2 and IMPE only",
        "scope_projection": "one AS of C, ANDE1/ANDE2 only inside that box, final IMPI discharging C",
    }
    explanation = (
        "The only target-atom occurrence in C is at depth d. Each ANDE traverses one edge, so d eliminations plus one mandatory PR are necessary."
        if family.startswith("projection_") else
        "Both premises must be written. The only route to the goal in this fragment is IMPE on P>Q; P occurs only at depth d in C, requiring d eliminations. Total d+3."
        if family == "implication_projection" else
        "Within the declared single-box projection schema, one AS and one IMPI surround the d necessary eliminations. Total d+2."
    )
    return {"prompt": prompt, "body": certificate, "n_lines": n_lines, "family": family,
            "projection_depth": depth, "directions": directions, "target_atom": target[1],
            "decoy_mode": mode, "restricted_lower_bound_lines": n_lines,
            "restricted_rules": restrictions[family], "restricted_bound_argument": explanation,
            "unrestricted_nd_minimum_known": False, "premise_order_randomized": family == "implication_projection"}


def exclusion_paths():
    prior = [OLDER / "data/sft_cap6.jsonl"] + [PRIOR / "data/released_v1" / filename for filename in
            ("candidate.jsonl", "diagnostic_ore.jsonl", "adaptation.jsonl", "retention.jsonl")]
    current = [common.ROOT / "data/v1" / filename for filename in
               ("sources.jsonl", "retention.jsonl", "targets.jsonl", "development.jsonl", "final.jsonl")]
    symbolic = [common.ROOT / "data" / filename for filename in
                ("symbolic_seeds.jsonl", "symbolic_dev.jsonl", "symbolic_final.jsonl")]
    return prior + current + [path for path in symbolic if path.exists()]


def summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"], row["n_lines"]].append(row)
    result = []
    for (family, lines), members in sorted(grouped.items()):
        tokens = [r["reference_tokens"] for r in members]
        result.append({"family": family, "reference_lines": lines, "n": len(members),
                       "projection_depth": members[0]["projection_depth"],
                       "reference_tokens": {"min": min(tokens), "median": statistics.median(tokens), "max": max(tokens)},
                       "prompt_tokens_max": max(r["prompt_tokens"] for r in members),
                       "reference_fits": {str(b): sum(r["reference_tokens"] <= b for r in members) for b in BUDGETS},
                       "left_edges": sum(r["directions"].count(0) for r in members),
                       "right_edges": sum(r["directions"].count(1) for r in members),
                       "max_direction_count_imbalance_per_task": max(abs(r["directions"].count(0) - r["directions"].count(1)) for r in members),
                       "dependency_pruned_lengths": dict(Counter(r["dependency_pruned_lines"] for r in members))})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=common.ROOT / "data/structured_v1")
    p.add_argument("--seed", type=int, default=202609101953)
    p.add_argument("--targets-per-cell", type=int, default=32)
    p.add_argument("--development-per-cell", type=int, default=16)
    p.add_argument("--final-per-cell", type=int, default=32)
    p.add_argument("--max-attempts", type=int, default=100000)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "audit").mkdir()
    rng = random.Random(args.seed)
    paths = exclusion_paths()
    start = time.monotonic()
    excluded_t, excluded_c = set(), set()
    for path in paths:
        for record in common.read_jsonl(path):
            theorem, context, _ = identities(prompt_of(record))
            excluded_t.add(theorem)
            excluded_c.add(context)
        print(json.dumps({"phase": "exclude", "file": str(path), "theorems": len(excluded_t), "contexts": len(excluded_c)}), flush=True)
    guard_path = OLDER / "data/all_textbook_guards.json"
    guard_source = json.loads(guard_path.read_text())
    guards = [set(guard_source[key]) for key in ("canonical_theorem_sha256", "nonempty_premise_context_sha256", "closed_formula_skeleton_sha256")]
    quotas = {"targets": args.targets_per_cell, "development": args.development_per_cell, "final": args.final_per_cell}
    populations = {role: [] for role in quotas}
    counts = {role: Counter() for role in quotas}
    seen = set()
    rejected = Counter()
    tok = Tokenizer()
    attempt = 0
    cells = [(family, length) for family in FAMILIES for length in LENGTHS]
    for family, length in cells:
        local_attempts = 0
        while any(counts[role][family, length] < quota for role, quota in quotas.items()):
            attempt += 1
            local_attempts += 1
            if local_attempts > args.max_attempts:
                raise RuntimeError(f"Incomplete quota for {family}/{length}: {dict((r, counts[r][family,length]) for r in quotas)}")
            row = make_task(rng, length, family)
            theorem, context, guard = identities(row["prompt"])
            if theorem in seen:
                rejected["duplicate"] += 1
                continue
            if theorem in excluded_t or context in excluded_c:
                rejected["exposed"] += 1
                continue
            if any(key is not None and key in values for key, values in zip(guard, guards)):
                rejected["textbook_guard"] += 1
                continue
            bucket = int(context[:12], 16) % 100
            role = "targets" if bucket < 50 else "development" if bucket < 75 else "final"
            if counts[role][family, length] >= quotas[role]:
                rejected["filled_role_cell"] += 1
                continue
            verdict = strict_complete(row["prompt"], row["body"])
            if not verdict["ok"]:
                raise AssertionError(verdict)
            compression = compress_proof(row["prompt"], row["body"])
            if compression["dependency_pruned_length"] != length or compression["compressed_length"] != length:
                raise AssertionError("Reference unexpectedly loses lines under compression")
            if not strict_complete(row["prompt"], compression["compressed_proof"])["ok"]:
                raise AssertionError("Compressed reference fails strict check")
            original = tok.encode_prompt(row["prompt"])
            reference_tokens = 1 + len(tok.encode_record(row["prompt"] + " " + row["body"])) - len(original)
            row.update(source_id=common.digest([row["prompt"], "structured-v1"]), theorem_id=theorem,
                       context_id=context, name=f"{role}_structured_{theorem[:16]}", role=role,
                       length_bin=str(length), reference_tokens=reference_tokens,
                       prompt_tokens=len(original) + 1, dependency_pruned_lines=compression["dependency_pruned_length"],
                       compressed_body=compression["compressed_proof"], independent_ok=verdict["independent_ok"],
                       reference_fits_budget={str(b): reference_tokens <= b for b in BUDGETS},
                       context_fits_4096={str(b): len(original) + 1 + b <= 4096 for b in BUDGETS},
                       provenance={"generator": "structured_generate.py", "seed": args.seed, "attempt": attempt,
                                   "external_training_label": False})
            populations[role].append(row)
            counts[role][family, length] += 1
            seen.add(theorem)
        print(json.dumps({"phase": "cell_complete", "family": family, "lines": length,
                          "attempts": local_attempts, "elapsed_seconds": time.monotonic() - start}), flush=True)
    all_rows = [row for rows in populations.values() for row in rows]
    context_sets = {role: {r["context_id"] for r in rows} for role, rows in populations.items()}
    theorem_sets = {role: {r["theorem_id"] for r in rows} for role, rows in populations.items()}
    overlaps = {f"{a}:{b}": {"contexts": len(context_sets[a] & context_sets[b]), "theorems": len(theorem_sets[a] & theorem_sets[b])}
                for a in populations for b in populations if a < b}
    if any(x["contexts"] or x["theorems"] for x in overlaps.values()):
        raise AssertionError("Cross-role leakage")
    for role, rows in populations.items():
        common.write_jsonl(out / "audit" / f"{role}.jsonl", rows)
        common.write_jsonl(out / f"{role}.jsonl", [{key: row[key] for key in ("name", "source_id", "prompt")} for row in rows])
    identity_files = [Path(__file__), Path(__file__).with_name("prepare_data.py"), Path(__file__).with_name("nd_environment.py"),
                      common.VENDOR / "nd_data.py", common.VENDOR / "proof_diagnostics.py", common.VENDOR / "nd_verify/verify.py"]
    manifest = {"status": "complete", "schema": "structured-proof-procedures-v1", "seed": args.seed,
                "families": list(FAMILIES), "reference_lengths": list(LENGTHS), "budgets": list(BUDGETS),
                "counts": {role: len(rows) for role, rows in populations.items()},
                "quotas_per_family_length_cell": quotas, "attempts": attempt,
                "elapsed_seconds": time.monotonic() - start, "rejected": dict(rejected),
                "excluded_theorems": len(excluded_t), "excluded_contexts": len(excluded_c),
                "cross_role_overlaps": overlaps, "exposure_sha256": {str(path): common.sha(path) for path in paths},
                "guard_sha256": common.sha(guard_path), "generator_sha256": common.sha(__file__),
                "code_sha256": {str(path.resolve()): common.sha(path) for path in identity_files},
                "population_sha256": {str(path.relative_to(out)): common.sha(path) for path in sorted(out.rglob("*.jsonl"))},
                "summary": summary(all_rows), "summary_by_role": {role: summary(rows) for role, rows in populations.items()},
                "split": "Canonical context hash modulo100: targets<50, development<75, final>=75; globally canonical-theorem deduplicated.",
                "canonicalization": "All24 bijections of PQRS and premise permutation; closed contexts additionally collapse atom identities to the same skeleton, exactly as primary prepare_data.py.",
                "data_separation": "Theorem-only role files contain exactly name/source_id/prompt. All certificates, directions, bounds and lengths stay under audit/.",
                "lower_bounds": "Exact only within each row's explicit restricted_rules fragment/schema; unrestricted natural-deduction minimum is unknown.",
                "shortcut_notes": ["The target atom never appears in off-path decoys.",
                    "Implication tasks exclude both P and goalQ from all decoys; only P>Q can produce Q in the restricted fragment.",
                    "All contexts are positive and satisfiable; no contradiction premise supplies a direct explosion shortcut.",
                    "Compound decoys can expose local size/type shortcuts. Success should also be stratified by family.",
                    "Dependency pruning removes no reference lines but is not a global shortest-proof search.",
                    "Repeated textual formulas make reference token cost grow quadratically with projection depth."]}
    common.write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "counts": manifest["counts"], "manifest": str(out / "manifest.json"),
                      "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
