"""Auditable verifier metrics for generated and validation targets.

Never use this module to inspect the held-out test set: use score_test.py there.
All solve denominators and frontier counts use distinct prompted sequents.
For repeated attempts, the headline length is the shortest accepted proof found
for each theorem. ``all_found_length_histogram`` additionally counts each
theorem once at every accepted length, for the literal written-length metric.
Neither length measure establishes a lower bound on a theorem's difficulty.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from nd_verify import verify_text
from nd_verify.verify import parse_formula


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson binomial interval (95% with the default z)."""
    if not 0 <= k <= n:
        raise ValueError("Wilson counts must satisfy 0 <= k <= n")
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"solved": k, "n": n, "rate": k / n if n else 0.0,
            "ci95": list(wilson(k, n))}


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _accepted_tokens(body: str) -> list[str]:
    tokens = body.split()
    if "QED" in tokens:
        return tokens[:tokens.index("QED") + 1]
    return tokens


def safe_verify(prompt: str, body: str) -> tuple[bool, str, int]:
    """Judge a body only against the supplied prompt; malformed outputs fail.

    The immutable verifier can raise exceptions for malformed text. These earn
    zero reward, including deeply nested malformed formulas. The official
    verifier stops parsing at the first QED but counts trailing semicolons in
    its length. We retain its acceptance decision while reporting only lines
    in the accepted proof, so trailing garbage cannot increase the frontier.
    """
    if not isinstance(prompt, str) or not isinstance(body, str):
        return False, "malformed input: prompt and proof must be strings", 0
    prompt_tokens = prompt.split()
    if (not prompt_tokens or prompt_tokens[0] != "THM"
            or prompt_tokens[-1] != "PRF" or prompt_tokens.count("PRF") != 1
            or "QED" in prompt_tokens):
        return False, "malformed prompt: expected THM ... SEQ ... PRF", 0
    try:
        ok, reason, reported_length = verify_text(prompt + " " + body)
        if ok:
            return True, reason, _accepted_tokens(body).count(";")
        return False, reason, reported_length
    except Exception as exc:
        # Catch malformed model text, not process-control BaseExceptions.
        return False, f"verifier exception: {type(exc).__name__}", 0


def target_prompt(target: Mapping[str, Any]) -> str:
    """Extract the immutable target prompt, never an output-provided sequent."""
    if isinstance(target.get("prompt"), str):
        return _normalized(target["prompt"])
    text = target.get("text")
    if isinstance(text, str) and "PRF" in text.split():
        toks = text.split()
        return " ".join(toks[:toks.index("PRF") + 1])
    raise ValueError("Every target needs a prompt or a full text containing PRF")


def _parse_prompt(prompt: str) -> tuple[list[tuple], tuple]:
    tokens = prompt.split()
    if not tokens or tokens[0] != "THM":
        raise ValueError("missing THM")
    premises, i = [], 1
    while tokens[i] != "SEQ":
        formula, i = parse_formula(tokens, i)
        premises.append(formula)
        if tokens[i] == ",":
            i += 1
        elif tokens[i] != "SEQ":
            raise ValueError("expected comma or SEQ")
    conclusion, i = parse_formula(tokens, i + 1)
    if tokens[i:] != ["PRF"]:
        raise ValueError("expected terminal PRF")
    return premises, conclusion


def triviality_flags(prompt: str) -> dict[str, Any]:
    """Conservative syntactic easy-case tags, not a shortest-proof oracle.

    ``known_short_bound`` includes every premise line required by the format.
    A false tag means undetected, not that a theorem is difficult.
    """
    try:
        premises, conclusion = _parse_prompt(prompt)
    except Exception:
        return {"parseable": False, "detected_easy": False, "known_short_bound": None}
    premise_set = set(premises)
    flags = {
        "conclusion_is_premise": conclusion in premise_set,
        "falsum_is_premise": ("bot",) in premise_set,
        "contradictory_premises": any(("not", p) in premise_set for p in premises),
        "one_step_projection": any(p[0] == "and" and conclusion in p[1:] for p in premises),
        "one_step_disjunction": conclusion[0] == "or" and any(p in conclusion[1:] for p in premises),
        "reflexive_implication": conclusion[0] == "imp" and conclusion[1] == conclusion[2],
    }
    bounds = []
    n = len(premises)
    if flags["conclusion_is_premise"]:
        bounds.append(n if premises[-1] == conclusion else n + 1)
    if any(flags[k] for k in ("falsum_is_premise", "one_step_projection", "one_step_disjunction")):
        bounds.append(n + 1)
    if flags["contradictory_premises"] or flags["reflexive_implication"]:
        bounds.append(n + 2)
    return {"parseable": True, **flags, "detected_easy": any(flags.values()),
            "known_short_bound": min(bounds) if bounds else None}


def _length(target: Mapping[str, Any], fields: Sequence[str]) -> int | None:
    for field in fields:
        value = target.get(field)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ValueError(f"Target has invalid {field}: {value!r}") from None
    return None


def evaluate_records(
    targets: Iterable[Mapping[str, Any]],
    bodies: Iterable[str | Mapping[str, Any]],
    *, min_distinct: int = 5, cap: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one positional body per target, preserving missing-output failures.

    ``bodies`` may contain strings or output records with a ``proof`` string.
    An output record's prompt never controls the reward. Missing tail records
    are explicit failures; excess records raise to catch alignment mistakes.
    """
    targets, bodies = list(targets), list(bodies)
    if len(bodies) > len(targets):
        raise ValueError("More proof bodies than targets")
    rows = []
    for i, target in enumerate(targets):
        prompt = target_prompt(target)
        output = bodies[i] if i < len(bodies) else None
        body = output.get("proof") if isinstance(output, Mapping) else output
        mismatch = (isinstance(output, Mapping) and "prompt" in output
                    and (not isinstance(output["prompt"], str)
                         or _normalized(output["prompt"]) != prompt))
        if i >= len(bodies):
            ok, reason, n_lines = False, "missing output", 0
        else:
            ok, reason, n_lines = safe_verify(prompt, body)
        rows.append({
            "name": str(target.get("name", target.get("thm", i))),
            "thm": target.get("thm"), "prompt": prompt,
            "theorem_key": _hash(prompt), "proof": body,
            "solved": bool(ok), "reason": reason,
            "found_len": n_lines if ok else None,
            "generating_len": _length(target, ("n_lines",)),
            "ref_len": _length(target, ("reference_lines", "min_lines_ub", "n_lines")),
            "bin": target.get("bin", "all"),
            "proof_hash": _hash(" ".join(_accepted_tokens(body))) if ok else None,
            "output_prompt_mismatch": bool(mismatch),
            "triviality": triviality_flags(prompt),
        })
    return aggregate_attempts(rows, min_distinct=min_distinct, cap=cap), rows


def _reason_group(reason: str) -> str:
    reason = re.sub(r" \(line \d+\)$", "", reason)
    return re.sub(r" at (?:line|position) \d+$", "", reason)


def aggregate_attempts(
    rows: Iterable[Mapping[str, Any]], *, min_distinct: int = 5, cap: int = 6,
) -> dict[str, Any]:
    """Aggregate scored attempts, deduplicating exact prompted theorems.

    Supply at least one row per target (including failed/missing attempts) so
    cumulative solve-rate denominators include the full target pool. Success
    intervals describe coverage of this pool; repeated attempts are correlated
    and are not treated as independent theorems. This function trusts scored
    verdicts from ``evaluate_records`` rather than rechecking proof strings.
    """
    if min_distinct < 1:
        raise ValueError("min_distinct must be positive")
    rows = list(rows)
    grouped = defaultdict(list)
    for row in rows:
        key = row.get("theorem_key")
        if not key:
            if not isinstance(row.get("prompt"), str):
                raise ValueError("Scored rows require theorem_key or prompt")
            key = _hash(_normalized(row["prompt"]))
        grouped[key].append(row)

    shortest_hist = Counter()
    all_lengths = defaultdict(set)
    unique_proofs = set()
    theorem_rows = []
    duplicate_label_conflicts = 0
    for key, attempts in grouped.items():
        solved = [r for r in attempts if r["solved"]]
        lengths = {int(r["found_len"]) for r in solved}
        shortest = min(lengths) if lengths else None
        if shortest is not None:
            shortest_hist[shortest] += 1
        for length in lengths:
            all_lengths[length].add(key)
        for row in solved:
            proof_hash = row.get("proof_hash")
            if not proof_hash and isinstance(row.get("proof"), str):
                proof_hash = _hash(" ".join(_accepted_tokens(row["proof"])))
            if not proof_hash:
                raise ValueError("Accepted verdict needs proof_hash or proof")
            unique_proofs.add((key, proof_hash))
        # Duplicate names or attempts never create extra theorem votes. If
        # conflicting upper-bound labels exist, use the smallest known bound.
        item = dict(attempts[0])
        for label in ("generating_len", "ref_len"):
            values = {r.get(label) for r in attempts if r.get(label) is not None}
            duplicate_label_conflicts += int(len(values) > 1)
            item[label] = min(values) if values else None
        item.update(solved=bool(solved), found_len=shortest)
        theorem_rows.append(item)

    def grouped_rates(field: str) -> dict[str, Any]:
        groups = defaultdict(list)
        for row in theorem_rows:
            groups[row.get(field)].append(row)
        return {str(key) if key is not None else "unknown":
                _rate(sum(bool(r["solved"]) for r in values), len(values))
                for key, values in sorted(groups.items(), key=lambda kv: str(kv[0]))}

    def length_comparison(field: str) -> dict[str, Any]:
        comparable = [r for r in theorem_rows if r["solved"] and r.get(field) is not None]
        pairs = Counter((r[field], r["found_len"]) for r in comparable)
        n = len(comparable)
        return {
            "n": n,
            "shorter": sum(r["found_len"] < r[field] for r in comparable),
            "equal": sum(r["found_len"] == r[field] for r in comparable),
            "longer": sum(r["found_len"] > r[field] for r in comparable),
            "mean_found": sum(r["found_len"] for r in comparable) / n if n else None,
            "mean_label": sum(r[field] for r in comparable) / n if n else None,
            "pairs": [{"label_len": a, "found_len": b, "n": c}
                      for (a, b), c in sorted(pairs.items())],
        }

    triviality = {}
    tags = sorted({k for r in theorem_rows for k, v in r.get("triviality", {}).items()
                   if isinstance(v, bool) and k != "parseable"})
    for tag in tags:
        tagged = [r for r in theorem_rows if r.get("triviality", {}).get(tag)]
        triviality[tag] = _rate(sum(bool(r["solved"]) for r in tagged), len(tagged))
    known_within_cap = [r for r in theorem_rows
                        if r.get("triviality", {}).get("known_short_bound") is not None
                        and r["triviality"]["known_short_bound"] <= cap]
    triviality["known_bound_within_cap"] = _rate(
        sum(bool(r["solved"]) for r in known_within_cap), len(known_within_cap))
    undetected = [r for r in theorem_rows if not r.get("triviality", {}).get("detected_easy", False)]
    triviality["no_detected_easy_pattern"] = _rate(
        sum(bool(r["solved"]) for r in undetected), len(undetected))

    all_hist = {k: len(v) for k, v in all_lengths.items()}
    frontier = max((length for length, n in shortest_hist.items() if n >= min_distinct), default=None)
    all_frontier = max((length for length, n in all_hist.items() if n >= min_distinct), default=None)
    solved_count = sum(bool(r["solved"]) for r in theorem_rows)
    reasons = Counter(_reason_group(str(r.get("reason", "unknown"))) for r in rows if not r["solved"])
    return {
        "schema_version": 1, "overall": _rate(solved_count, len(theorem_rows)),
        "by_n_lines": grouped_rates("generating_len"), "by_ref_lines": grouped_rates("ref_len"),
        "by_bin": grouped_rates("bin"), "n_attempts": len(rows),
        "n_valid_attempts": sum(bool(r["solved"]) for r in rows),
        "n_unique_proofs": len(unique_proofs),
        "n_unique_solved_theorems": solved_count,
        "duplicate_theorem_rows": len(rows) - len(theorem_rows),
        "duplicate_length_label_conflicts": duplicate_label_conflicts,
        "failure_reasons": dict(reasons.most_common()),
        "output_prompt_mismatches": sum(bool(r.get("output_prompt_mismatch")) for r in rows),
        "found_length_histogram": {str(k): v for k, v in sorted(shortest_hist.items())},
        "all_found_length_histogram": {str(k): v for k, v in sorted(all_hist.items())},
        "found_length_policy": "shortest accepted proof per distinct prompted theorem",
        "robust_frontier": frontier, "all_found_robust_frontier": all_frontier,
        "frontier_min_distinct": min_distinct,
        "max_found_length": max(all_hist, default=None),
        "solved_with_shortest_found_above_cap": sum(r["solved"] and r["found_len"] > cap for r in theorem_rows),
        "cap": cap, "found_vs_generating": length_comparison("generating_len"),
        "found_vs_reference": length_comparison("ref_len"), "triviality": triviality,
    }
