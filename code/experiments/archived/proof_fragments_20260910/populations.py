"""Prospective synthetic populations; no evaluation corpus is opened."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time
import common
from nd_data import generate_candidate, canonical_theorem, sequent_parts
from nd_environment import independent_replay, strict_complete, split_body, episode

FAMILIES = ("ANDI", "ANDE1", "ANDE2", "IMPE", "IMPI_flat", "IMPI_nested", "ORI1", "ORI2", "NEGE", "NEGI", "BOTE", "DN", "ORE")

def family_of(raw):
    rule = raw["rules"][-1]
    return ("IMPI_nested" if raw["max_depth"] > 1 else "IMPI_flat") if rule == "IMPI" else rule

def identities(prompt):
    premises, goal = sequent_parts(prompt)
    theorem = common.digest(canonical_theorem(prompt))
    context = common.digest(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF")) if premises else None
    closed = common.digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None
    # Existing guards hash raw strings, whereas digest() JSON-serializes. Keep
    # exact existing hashes in a second field to use historical blind guards.
    import hashlib
    old = lambda s: hashlib.sha256(s.encode()).hexdigest()
    guard = (old(canonical_theorem(prompt)),
             old(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF")) if premises else None,
             old(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None)
    return {"theorem_id": theorem, "context_id": context or closed, "guard_ids": guard}

def load_guards(path):
    if not path:
        return [set(), set(), set()]
    x = json.loads(Path(path).read_text())
    fields = ("canonical_theorem_sha256", "nonempty_premise_context_sha256", "closed_formula_skeleton_sha256")
    return [set(x[k]) for k in fields]

def existing_exposure(paths):
    theorem, contexts = set(), set()
    for path in paths:
        for row in common.read_jsonl(path):
            prompt = row.get("prompt") or row["text"].split("PRF", 1)[0] + "PRF"
            ids = identities(prompt)
            theorem.add(ids["theorem_id"])
            contexts.add(ids["context_id"])
    return theorem, contexts

def enrich(raw, seed, attempt, role):
    body = raw["text"].split("PRF", 1)[1].strip()
    verdict = strict_complete(raw["prompt"], body)
    if not verdict["ok"]:
        raise AssertionError(verdict)
    lines = split_body(body)
    ids = identities(raw["prompt"])
    row = {**raw, **ids, "body": body, "family": family_of(raw),
           "source_id": common.digest([raw["prompt"], body]), "role": role,
           "provenance": {"kind": "fresh_synthetic_forward", "generation_seed": seed,
                          "generation_attempt": attempt, "full_source_lines": len(lines),
                          "source_body_sha256": common.digest(body),
                          "source_external_supervision_cap": len(lines)}}
    # The default family intervention is a verified final window of up to 3
    # complete inferences. Supplied context retains its complete source cap.
    cut = max(0, len(lines) - 3)
    while cut and independent_replay(row["prompt"], " ".join(lines[:cut]))["goal_already_available"]:
        cut -= 1
    row["default_cut"] = cut
    state = independent_replay(row["prompt"], " ".join(lines[:cut]))
    row["fragment_descriptor"] = {
        "current_assumption_depth": len(state["active_assumptions"]),
        "continuation_lines": len(lines) - cut,
        "discharge": any(r in ("IMPI", "NEGI", "ORE") for r in raw["rules"][cut:]),
        "rule_transition": raw["rules"][max(0, cut - 1):],
        "used_prior_lines": sorted({r for ln in independent_replay(row["prompt"], body, True)["lines"][cut:]
                                    for r in ln["refs"] if r <= cut})}
    return row

def audit(populations, exposure_paths=()):
    seen_t, seen_c = {}, {}
    report = {}
    exposed_t, exposed_c = existing_exposure(exposure_paths)
    for role, rows in populations.items():
        local_t, local_c = set(), set()
        for row in rows:
            ids = identities(row["prompt"])
            verdict = strict_complete(row["prompt"], row["body"])
            if not verdict["ok"]:
                raise AssertionError("Unverified population record")
            if not row["n_lines"] == row["provenance"]["full_source_lines"] == verdict["n_lines"]:
                raise AssertionError("Declared source lengths differ from independently checked body")
            for name, key, previous, local in (("theorem", ids["theorem_id"], seen_t, local_t),
                                              ("context", ids["context_id"], seen_c, local_c)):
                if key in previous and previous[key] != role:
                    raise AssertionError(f"Cross-role {name} overlap: {role}/{previous[key]}")
                previous[key] = role
                local.add(key)
            if role in ("selection", "confirmation", "final_confirmation") and (ids["theorem_id"] in exposed_t or ids["context_id"] in exposed_c):
                raise AssertionError("Probe overlaps baseline exposure")
            if role in ("candidate", "adaptation", "retention") and row["n_lines"] > 6:
                raise AssertionError("Cap6 source supervision violated")
        report[role] = {"n": len(rows), "theorems": len(local_t), "contexts": len(local_c),
                        "families": dict(Counter(r["family"] for r in rows)),
                        "lengths": dict(Counter(r["n_lines"] for r in rows))}
    return {"ok": True, "role_statistics": report,
            "baseline_exposed_theorems": len(exposed_t), "baseline_exposed_contexts": len(exposed_c)}

def generate(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; populations are immutable")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    guards = load_guards(args.guard_json)
    exposed_t, exposed_c = existing_exposure(args.exclude_training)
    populations = {k: [] for k in ("candidate", "adaptation", "retention", "selection", "confirmation", "final_confirmation")}
    counts = {k: Counter() for k in populations}
    quotas = {"candidate": args.per_family, "adaptation": args.adaptation,
              "retention": args.retention, "selection": args.probes_per_bin,
              "confirmation": args.probes_per_bin, "final_confirmation": args.final_probes_per_bin}
    # Candidate/adaptation/retention disjoint context allocation; selection and
    # confirmation each have three length strata. Assignment is by context
    # hash, independent of model outcomes and accepted proof length.
    def role_for(context):
        bucket = int(context[:12], 16) % 20
        return ("candidate" if bucket < 6 else "adaptation" if bucket < 10 else "retention" if bucket < 12
                else "selection" if bucket < 15 else "confirmation" if bucket < 17 else "final_confirmation")
    def needed(role, key):
        if role == "candidate": return counts[role][key] < quotas[role]
        if role in ("selection", "confirmation", "final_confirmation"): return counts[role][key] < quotas[role]
        return len(populations[role]) < quotas[role]
    def done():
        return (all(counts["candidate"][f] >= args.per_family for f in args.families)
                and len(populations["adaptation"]) >= args.adaptation
                and len(populations["retention"]) >= args.retention
                and all(counts[r][b] >= quotas[r] for r in ("selection", "confirmation", "final_confirmation") for b in ("2-6", "7-9", "10-12")))
    seen = set()
    start = time.monotonic()
    attempt = 0
    observed_terminal_counts = Counter()
    while not done() and attempt < args.max_attempts:
        attempt += 1
        if attempt % 50000 == 0:
            progress = {"attempt": attempt, "counts": {k: dict(v) for k,v in counts.items()},
                        "accepted": {k: len(v) for k,v in populations.items()}, "elapsed_s": time.monotonic()-start}
            print(json.dumps(progress), flush=True)
            common.write_json(out / "progress.json", progress)
        # Balance generation effort toward unfinished cap6 or long quotas.
        short_done = (all(counts["candidate"][f] >= args.per_family for f in args.families)
                      and len(populations["adaptation"]) >= args.adaptation and len(populations["retention"]) >= args.retention)
        max_lines = rng.choice((9, 10, 12, 14, 16)) if short_done or rng.random() < 0.4 else 6
        raw = generate_candidate(rng, max_lines=max_lines, formula_limit=args.formula_limit)
        if raw is not None:
            observed_terminal_counts[f"{family_of(raw)}:cap6" if raw["n_lines"] <= 6 else f"{family_of(raw)}:long"] += 1
        if raw is None or not 2 <= raw["n_lines"] <= 12:
            continue
        ids = identities(raw["prompt"])
        if any(k in protected for k, protected in zip(ids["guard_ids"], guards) if k is not None):
            continue
        if ids["theorem_id"] in seen:
            continue
        role = role_for(ids["context_id"])
        family = family_of(raw)
        if role in ("candidate", "adaptation", "retention"):
            if raw["n_lines"] > 6 or (role == "candidate" and family not in args.families):
                continue
            key = family
        else:
            if ids["theorem_id"] in exposed_t or ids["context_id"] in exposed_c:
                continue
            key = "2-6" if raw["n_lines"] <= 6 else "7-9" if raw["n_lines"] <= 9 else "10-12"
        if not needed(role, key):
            continue
        row = enrich(raw, args.seed, attempt, role)
        row["length_bin"] = "2-6" if raw["n_lines"] <= 6 else "7-9" if raw["n_lines"] <= 9 else "10-12"
        populations[role].append(row)
        counts[role][key] += 1
        seen.add(ids["theorem_id"])
        if sum(map(len, populations.values())) % 250 == 0:
            print(json.dumps({"attempt": attempt, "accepted": {k: len(v) for k,v in populations.items()},
                              "candidate_families": dict(counts["candidate"]), "elapsed_s": time.monotonic()-start}), flush=True)
    for role, rows in populations.items():
        common.write_jsonl(out / f"{role}.jsonl", rows)
    result = audit(populations, args.exclude_training)
    result.update({"args": vars(args), "status": "complete" if done() else "incomplete_quotas_do_not_train", "attempts": attempt, "elapsed_seconds": time.monotonic() - start,
                   "observed_terminal_counts_before_guards": dict(observed_terminal_counts),
                   "input_sha256": {str(p): common.sha(p) for p in args.exclude_training + ([args.guard_json] if args.guard_json else [])},
                   "population_sha256": {r: common.sha(out / f"{r}.jsonl") for r in populations},
                   "dependency_sha256": common.dependency_hashes()})
    common.write_json(out / "manifest.json", result)
    print(json.dumps(result), flush=True)
    if not done():
        raise RuntimeError(f"Incomplete populations saved but not approved: exhausted {attempt} attempts")

def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--per-family", type=int, default=64)
    p.add_argument("--adaptation", type=int, default=2048)
    p.add_argument("--retention", type=int, default=512)
    p.add_argument("--probes-per-bin", type=int, default=128)
    p.add_argument("--final-probes-per-bin", type=int, default=256)
    p.add_argument("--max-attempts", type=int, default=3000000)
    p.add_argument("--formula-limit", type=int, default=23)
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
    p.add_argument("--guard-json")
    p.add_argument("--exclude-training", nargs="*", default=[])
    return p

if __name__ == "__main__":
    generate(parser().parse_args())
