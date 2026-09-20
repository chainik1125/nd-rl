"""Derive an audited primary release after the immutable collector has ended.

This is population integrity work, not a model evaluation release. No model
outcomes or checkpoints are inputs. The raw collection is never rewritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import common
from populations import audit, identities, load_guards
from nd_environment import strict_complete

PROBE_ROLES = ("selection", "confirmation", "final_confirmation")
ROLES = ("candidate", "adaptation", "retention") + PROBE_ROLES
BINS = ("2-6", "7-9", "10-12")

def check_record(row, expected_role=None, cap=None):
    if expected_role is not None and row.get("role") != expected_role:
        raise ValueError(f"Expected role {expected_role}, found {row.get('role')}")
    verdict = strict_complete(row["prompt"], row["body"])
    n = verdict["n_lines"]
    if not verdict["ok"]:
        raise ValueError(f"Unverified source: {verdict['reason']}")
    if not n == row["n_lines"] == row["provenance"]["full_source_lines"]:
        raise ValueError("Source line counts disagree")
    if cap is not None and n > cap:
        raise ValueError("Source supervision cap exceeded")
    ids = identities(row["prompt"])
    if any(row.get(k) != ids[k] for k in ("theorem_id", "context_id")):
        raise ValueError("Canonical source/context identity differs")
    if list(row.get("guard_ids", [])) != list(ids["guard_ids"]):
        raise ValueError("Blind guard identities differ")
    if row["source_id"] != common.digest([row["prompt"], row["body"]]):
        raise ValueError("Source certificate identity differs")
    if row["provenance"]["source_body_sha256"] != common.digest(row["body"]):
        raise ValueError("Body provenance identity differs")
    if " ".join(row["text"].split()) != " ".join((row["prompt"] + " " + row["body"]).split()):
        raise ValueError("Text/body certificate differs")
    actual_bin = "2-6" if n <= 6 else "7-9" if n <= 9 else "10-12"
    if row["length_bin"] != actual_bin or not 2 <= n <= 12:
        raise ValueError("Reference length stratum differs")
    return ids

def validate_roles(populations, families, per_family, role_totals, probe_per_bin, guards):
    if set(populations) != set(ROLES):
        raise ValueError("Missing or unexpected population role")
    for role, rows in populations.items():
        seen_source, seen_theorem = set(), set()
        for row in rows:
            ids = check_record(row, role, 6 if role in ("candidate", "adaptation", "retention") else None)
            if row["source_id"] in seen_source or row["theorem_id"] in seen_theorem:
                raise ValueError(f"Duplicate source/theorem inside {role}")
            seen_source.add(row["source_id"]); seen_theorem.add(row["theorem_id"])
            if any(k in protected for k, protected in zip(ids["guard_ids"], guards) if k is not None):
                raise ValueError(f"Blind textbook guard collision in {role}")
        if role == "candidate":
            expected = Counter({f: per_family for f in families})
            if Counter(r["family"] for r in rows) != expected:
                raise ValueError("Primary candidate family quotas differ")
        elif role in PROBE_ROLES:
            if Counter(r["length_bin"] for r in rows) != Counter({b:probe_per_bin[role] for b in BINS}):
                raise ValueError(f"Probe quotas differ for {role}")
        elif len(rows) != role_totals[role]:
            raise ValueError(f"Population quota differs for {role}")

def run(args):
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if output == source or (output.exists() and any(output.iterdir())):
        raise ValueError("Release needs a distinct empty output directory")
    raw_manifest_path = source / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    status = raw_manifest.get("status")
    if status not in ("complete", "incomplete_quotas_do_not_train"):
        raise ValueError("Collector has not produced a final complete/incomplete manifest")
    amendment_path = Path(args.amendment).resolve()
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get("status") != "prospective_before_any_fragment_model_outcomes"
            or amendment.get("no_model_confirmation_outcomes_opened") is not True):
        raise ValueError("Expected prospective collection-only amendment")
    families = amendment["primary_candidate_families"]
    if len(families) != 12 or len(set(families)) != 12 or "ORE" in families:
        raise ValueError("Amendment must specify twelve distinct non-ORE primary families")
    if amendment["primary_sources_per_family"] != 64 or amendment["final_confirmation_per_length_bin"] != 256:
        raise ValueError("Amendment differs from authorized source/probe quotas")
    if amendment.get("main_bank_size_per_condition") != 192:
        raise ValueError("Amendment differs from authorized main bank size")
    input_hashes = {str(raw_manifest_path): common.sha(raw_manifest_path), str(amendment_path):common.sha(amendment_path)}
    for path, digest in raw_manifest["input_sha256"].items():
        if common.sha(path) != digest:
            raise ValueError(f"Original collector input changed: {path}")
        input_hashes[str(Path(path).resolve())] = digest
    for path, digest in raw_manifest["dependency_sha256"].items():
        if common.sha(path) != digest:
            raise ValueError(f"Original collector dependency changed: {path}")
    for role in ROLES:
        path = source / f"{role}.jsonl"
        actual = common.sha(path)
        if actual != raw_manifest["population_sha256"][role]:
            raise ValueError(f"Raw population identity changed: {role}")
        input_hashes[str(path)] = actual
    # The original collector writes manifest only after leaving its proposal
    # loop and saving all roles. Refuse any truncation relative to that receipt.
    raw = {role:common.read_jsonl(source/f"{role}.jsonl") for role in ROLES}
    for role, rows in raw.items():
        if len(rows) != raw_manifest["role_statistics"][role]["n"]:
            raise ValueError("Raw manifest population count changed")
    diagnostic = [r for r in raw["candidate"] if r["family"] == "ORE"]
    if any(r["family"] not in set(families) | {"ORE"} for r in raw["candidate"]):
        raise ValueError("Unspecified collected candidate family")
    primary = {**raw, "candidate":[r for r in raw["candidate"] if r["family"] in families]}
    exposure_paths = raw_manifest["args"]["exclude_training"]
    guard_path = raw_manifest["args"]["guard_json"]
    if not exposure_paths or not guard_path:
        raise ValueError("Release requires bound baseline exposure and textbook guard inputs")
    if any(str(p) not in raw_manifest["input_sha256"] for p in exposure_paths + [guard_path]):
        raise ValueError("Baseline exposure/guard input was not bound by the collector")
    guards = load_guards(guard_path)
    validate_roles(primary, families, 64, {"adaptation":2048,"retention":512},
                   {"selection":128,"confirmation":128,"final_confirmation":256}, guards)
    for row in diagnostic:
        ids = check_record(row, "candidate", 6)
        if any(k in protected for k, protected in zip(ids["guard_ids"], guards) if k is not None):
            raise ValueError("ORE diagnostic guard collision")
    # ORE diagnostics remain in their original candidate context allocation;
    # they are not a new held-out population and may share candidate contexts.
    collection_audit = audit(raw, exposure_paths)
    primary_audit = audit(primary, exposure_paths)
    diagnostic_rows = [{**r,"role":"diagnostic_ore","diagnostic_only":True,
                        "collection_role":"candidate","primary_training_admissible":False} for r in diagnostic]
    output.mkdir(parents=True, exist_ok=True)
    for role, rows in primary.items():
        common.write_jsonl(output/f"{role}.jsonl", rows)
    common.write_jsonl(output/"diagnostic_ore.jsonl", diagnostic_rows)
    manifest = {"status":"released_complete_primary", "released_unix":time.time(),
                "source_generation_status":status,"source_generation_attempts":raw_manifest["attempts"],
                "source_generation_manifest_sha256":common.sha(raw_manifest_path),
                "amendment_sha256":common.sha(amendment_path),"amendment":amendment,
                "input_sha256":input_hashes,
                "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
                "dependency_sha256":common.dependency_hashes(),
                "population_sha256":{role:common.sha(output/f"{role}.jsonl") for role in ROLES+("diagnostic_ore",)},
                "primary_audit":primary_audit,"collection_audit_including_ore":collection_audit,
                "role_statistics":primary_audit["role_statistics"],
                "diagnostic_ore":{"n":len(diagnostic_rows),"unique_sources":len({r["source_id"] for r in diagnostic_rows}),
                    "unique_contexts":len({r["context_id"] for r in diagnostic_rows}),
                    "contexts_shared_with_primary_candidates":len({r["context_id"] for r in diagnostic_rows}&{r["context_id"] for r in primary["candidate"]}),
                    "not_a_separate_heldout_population":True},
                "model_outcomes_used":False,"model_evaluation_release":False}
    common.write_json(output/"manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"source_generation_status":status,
                      "primary_sources":len(primary["candidate"]),"diagnostic_ore_sources":len(diagnostic_rows),
                      "manifest_sha256":common.sha(output/"manifest.json")}),flush=True)

def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--amendment",required=True)
    return p

if __name__=="__main__":run(parser().parse_args())
