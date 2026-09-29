"""Independently recompute release identities, separation, caps and schemas.

Does not import prepare_data or its identity function. No model is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

import common
from nd_environment import strict_complete, Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def identities(prompt):
    prefix, conclusion = prompt.removeprefix("THM").strip().split("SEQ", 1)
    conclusion = conclusion.removesuffix("PRF").strip()
    premises = [x.strip() for x in prefix.split(",") if x.strip()]
    candidates, contexts = [], []
    for perm in itertools.permutations("PQRS"):
        table = str.maketrans("PQRS", "".join(perm))
        pre = " , ".join(sorted(x.translate(table) for x in premises))
        candidates.append(pre + " |- " + conclusion.translate(table))
        contexts.append(pre + " |- F")
    canonical = min(candidates)
    context = min(contexts) if premises else re.sub(r"\b[PQRS]\b", "A", conclusion)
    digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    raw_sha = lambda x: hashlib.sha256(x.encode()).hexdigest()
    return digest(canonical), digest(context), (raw_sha(canonical), raw_sha(context) if premises else None,
                                               raw_sha(context) if not premises else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/review_data.json")
    args = parser.parse_args()
    data = args.data
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["status"] == "complete", "Incomplete release cannot pass readiness"
    checked_hashes = {}
    for path, expected in manifest["exposure_sha256"].items():
        actual = common.sha(path)
        assert actual == expected, f"Exposure drift: {path}"
        checked_hashes[path] = actual
    for path, expected in manifest["population_sha256"].items():
        actual = common.sha(data / path)
        assert actual == expected, f"Population drift: {path}"
        checked_hashes[str(data / path)] = actual
    seen_t, seen_c = set(), set()
    exposure_stats = {}
    for path in manifest["exposure_sha256"]:
        records = common.read_jsonl(path)
        for row in records:
            prompt = row.get("prompt") or row["text"].split("PRF", 1)[0] + "PRF"
            t, c, _ = identities(prompt)
            seen_t.add(t)
            seen_c.add(c)
            # Every external proof in the imported lineage must retain cap6.
            body = row.get("body") or row["text"].split("PRF", 1)[1].strip()
            assert body.count(";") <= 6
        exposure_stats[path] = len(records)
    assert len(seen_t) == manifest["excluded_theorems"]
    assert len(seen_c) == manifest["excluded_contexts"]
    sources = common.read_jsonl(data / "sources.jsonl")
    retention = common.read_jsonl(data / "retention.jsonl")
    for row in sources + retention:
        verdict = strict_complete(row["prompt"], row["body"])
        assert verdict["ok"] and verdict["n_lines"] <= 6
        assert verdict["n_lines"] == row["n_lines"]
    assert len({r["source_id"] for r in sources}) == len(sources)
    # Recreate source selection from the declared seed without training outcomes.
    paths = [Path(x) for x in manifest["exposure_sha256"]]
    candidate_paths = [p for p in paths if p.name in ("candidate.jsonl", "diagnostic_ore.jsonl")]
    candidates = [row for path in candidate_paths for row in common.read_jsonl(path)]
    family_names = sorted({r["family"] for r in candidates})
    rng = random.Random(manifest["args"]["seed"])
    chosen = []
    for f in family_names:
        rows = sorted([r for r in candidates if r["family"] == f], key=lambda r: r["source_id"])
        chosen.extend(rng.sample(rows, min(8, len(rows))))
    assert [r["source_id"] for r in chosen] == [r["source_id"] for r in sources]
    guards_path = ROOT.parent / "r1-sprint-20260909-1625/data/all_textbook_guards.json"
    assert common.sha(guards_path) == manifest["guard_sha256"]
    guard_json = json.loads(guards_path.read_text())
    guards = [set(guard_json[k]) for k in ("canonical_theorem_sha256", "nonempty_premise_context_sha256", "closed_formula_skeleton_sha256")]
    by_role, context_sets = {}, {}
    tok = Tokenizer()
    all_ids = set()
    for role in ("targets", "development", "final"):
        prompts = common.read_jsonl(data / (role + ".jsonl"))
        audit = common.read_jsonl(data / "audit" / (role + ".jsonl"))
        assert len(prompts) == len(audit)
        assert all(set(row) == {"name", "source_id", "prompt"} for row in prompts)
        assert len({r["source_id"] for r in prompts}) == len(prompts)
        contexts, counts, tokens = set(), Counter(), []
        for r, a in zip(prompts, audit):
            assert all(r[k] == a[k] for k in r)
            t, c, old_guard = identities(r["prompt"])
            assert t == a["theorem_id"] and c == a["context_id"]
            assert t not in seen_t and c not in seen_c
            assert t not in all_ids
            all_ids.add(t)
            assert not any(k is not None and k in g for k, g in zip(old_guard, guards))
            verdict = strict_complete(r["prompt"], a["body"])
            assert verdict["ok"] and verdict["n_lines"] == a["n_lines"]
            counts[a["length_bin"]] += 1
            contexts.add(c)
            n = 1 + len(tok.encode_record(a["text"])) - len(tok.encode_prompt(r["prompt"]))
            assert n == a["reference_tokens"]
            tokens.append(n)
        assert dict(counts) == manifest["quotas"][role]
        context_sets[role] = contexts
        by_role[role] = {"theorems": len(prompts), "contexts": len(contexts), "length_counts": dict(counts),
                         "reference_response_tokens_min": min(tokens), "reference_response_tokens_max": max(tokens),
                         "reference_response_tokens_above_512": sum(x > 512 for x in tokens),
                         "reference_response_tokens_above_1024": sum(x > 1024 for x in tokens)}
    intersections = {f"{a}:{b}": len(context_sets[a] & context_sets[b]) for a, b in itertools.combinations(context_sets, 2)}
    assert not any(intersections.values())
    result = {"ok": True, "exposure_records": exposure_stats, "excluded_theorems": len(seen_t),
              "excluded_contexts": len(seen_c), "sources": len(sources), "retention": len(retention),
              "source_families": dict(Counter(r["family"] for r in sources)), "prospective_source_selection_reproduced": True,
              "roles": by_role, "context_intersections": intersections, "checked_input_hashes": checked_hashes,
              "data_manifest_sha256": common.sha(data / "manifest.json"), "review_code_sha256": common.sha(__file__),
              "audit_method": "Independent 24-bijection canonicalization, all exposure cap counts, strict source/retention/audit certificates, exact prompt-only schemas, full prompt/audit bijection; CPU only, no model loaded"}
    common.write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "checked_input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
