"""Post-hoc proof compression diagnostics. Never use outputs as training data.

Accepted raw model proofs remain the primary written-length metric. This
analysis preserves every prompted premise and re-verifies every reduction
against the same prompt. It only removes unused dependencies and reiterations;
it is not a shortest-proof solver and supplies no theorem-difficulty bound.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from nd_data import formula_text
from nd_metrics import safe_verify
from nd_verify import parse_proof_tokens


def _contexts(lines):
    stack, result = [], {}
    for line in lines:
        depth, idx = line["depth"], line["idx"]
        if line["rule"] == "AS":
            stack = stack[:depth - 1] + [idx]
        else:
            stack = stack[:depth]
        result[idx] = tuple(stack)
    return result


def _dependency_lines(lines):
    """Retain the final proof dependency graph AND every declared PR line."""
    if not lines:
        return []
    by_id = {line["idx"]: line for line in lines}
    contexts = _contexts(lines)
    keep = set()

    def visit(idx):
        if idx in keep:
            return
        keep.add(idx)
        line = by_id[idx]
        for ref in tuple(line["refs"]) + contexts[idx]:
            visit(ref)

    visit(lines[-1]["idx"])
    for line in lines:
        if line["rule"] == "PR":
            visit(line["idx"])
    return [line for line in lines if line["idx"] in keep]


def _render(lines):
    renumber = {line["idx"]: i + 1 for i, line in enumerate(lines)}
    body = []
    for line in lines:
        tokens = [f"N{renumber[line['idx']]}"] + ["|"] * line["depth"]
        tokens += [formula_text(line["formula"]), ":", line["rule"]]
        tokens += [f"N{renumber[ref]}" for ref in line["refs"]] + [";"]
        body.append(" ".join(tokens))
    return "\n".join(body) + "\nQED"


def compress_proof(prompt, body):
    """Return independently verified dependency/reiteration compression stats.

    The returned proof is explicitly analysis-only. Failed raw attempts cannot
    become successes here. Raw lengths exclude text after the first QED, as in
    the primary metrics; that ignored suffix is separately counted.
    """
    ok, reason, raw_length = safe_verify(prompt, body)
    if not ok:
        return {"analysis_only": True, "raw_valid": False, "reason": reason}
    tokens = body.split()
    qed_position = tokens.index("QED")
    suffix = tokens[qed_position + 1:]
    lines = parse_proof_tokens(tokens[:qed_position + 1])
    original_premises = [line["formula"] for line in lines if line["rule"] == "PR"]
    pruned = _dependency_lines(lines)
    pruned_body = _render(pruned)
    valid, diagnostic, dependency_length = safe_verify(prompt, pruned_body)
    if not valid:
        raise AssertionError(f"Dependency compression failed verification: {diagnostic}")

    current = pruned
    removed_reiterations = 0
    while True:
        improved = False
        for line in current:
            if line["rule"] != "R":
                continue
            old_idx, replacement = line["idx"], line["refs"][0]
            candidate = [dict(other, refs=[replacement if ref == old_idx else ref for ref in other["refs"]])
                         for other in current if other["idx"] != old_idx]
            if not candidate:
                continue
            candidate = _dependency_lines(candidate)
            candidate_body = _render(candidate)
            accepted, _, _ = safe_verify(prompt, candidate_body)
            if accepted:
                current = candidate
                removed_reiterations += 1
                improved = True
                break
        if not improved:
            break
    compressed_body = _render(current)
    accepted, diagnostic, compressed_length = safe_verify(prompt, compressed_body)
    if not accepted:
        raise AssertionError(f"Reiteration compression failed verification: {diagnostic}")
    assert [line["formula"] for line in current if line["rule"] == "PR"] == original_premises
    return {"analysis_only": True, "raw_valid": True, "raw_written_length": raw_length,
            "dependency_pruned_length": dependency_length,
            "compressed_length": compressed_length,
            "unused_lines_removed": raw_length - dependency_length,
            "additional_lines_removed_with_reiteration": dependency_length - compressed_length,
            "reiteration_rewrites": removed_reiterations,
            "original_reiteration_lines": sum(line["rule"] == "R" for line in lines),
            "preserved_premise_lines": len(original_premises),
            "ignored_trailing_semicolons": suffix.count(";"),
            "compressed_proof": compressed_body}


def analyze_records(records, min_distinct=5, save_compressed=False):
    rows, failed = [], Counter()
    grouped = defaultdict(list)
    for i, record in enumerate(records):
        prompt, body = record.get("prompt"), record.get("proof")
        result = compress_proof(prompt, body)
        result["name"] = record.get("name", str(i))
        key = hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest() if isinstance(prompt, str) else str(i)
        result["theorem_key"] = key
        result["proof_hash"] = hashlib.sha256(body.encode()).hexdigest() if isinstance(body, str) else None
        if result["raw_valid"]:
            grouped[key].append(result)
        else:
            failed[result["reason"]] += 1
        if not save_compressed:
            result.pop("compressed_proof", None)
        rows.append(result)

    accepted = [row for row in rows if row["raw_valid"]]

    def distribution(field, shortest):
        histogram = Counter()
        for group in grouped.values():
            values = [min(row[field] for row in group)] if shortest else {row[field] for row in group}
            histogram.update(values)
        return {"histogram": dict(sorted(histogram.items())),
                "robust_frontier": max((n for n, count in histogram.items() if count >= min_distinct), default=0)}

    summary = {"analysis_only": True, "n_attempts": len(rows), "n_valid_attempts": len(accepted),
               "n_solved_theorems": len(grouped), "failed_reasons": dict(failed),
               "compressible_valid_attempts": sum(r["compressed_length"] < r["raw_written_length"] for r in accepted),
               "attempts_with_unused_steps": sum(r["unused_lines_removed"] > 0 for r in accepted),
               "attempts_with_reiteration_reduction": sum(r["additional_lines_removed_with_reiteration"] > 0 for r in accepted),
               "unused_lines_removed_total": sum(r["unused_lines_removed"] for r in accepted),
               "reiteration_reduction_lines_total": sum(r["additional_lines_removed_with_reiteration"] for r in accepted),
               "raw_shortest_per_theorem": distribution("raw_written_length", True),
               "dependency_shortest_per_theorem": distribution("dependency_pruned_length", True),
               "compressed_shortest_per_theorem": distribution("compressed_length", True),
               "raw_all_found_lengths": distribution("raw_written_length", False),
               "compressed_all_found_lengths": distribution("compressed_length", False),
               "theorems_raw_over6_compressed_at_most6": sum(min(r["raw_written_length"] for r in group) > 6 and
                                                             min(r["compressed_length"] for r in group) <= 6
                                                             for group in grouped.values()),
               "limitations": ["Compression is post-hoc analysis; primary metrics retain actual raw written lengths.",
                                "Every prompted PR line is retained in its original order.",
                                "Compression removes only unused dependencies and reiterations, not all redundant reasoning.",
                                "Compressed length is still an upper bound, not a shortest-proof estimate.",
                                "No compressed proof is used for training or RL rewards."]}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Non-test proof/verdict JSONLs containing prompt and proof")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-compressed-proofs", action="store_true")
    args = parser.parse_args()
    for path in args.inputs:
        if re.search(r"(?:^|[/_.-])tests?(?:[/_.-]|$)", path.lower()):
            parser.error("Test files are aggregate-score-only; proof diagnostics are forbidden there.")
    records = []
    for path in args.inputs:
        records.extend(json.loads(line) for line in Path(path).open() if line.strip())
    summary, rows = analyze_records(records, save_compressed=args.save_compressed_proofs)
    summary["inputs"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.inputs}
    summary["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "records": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
