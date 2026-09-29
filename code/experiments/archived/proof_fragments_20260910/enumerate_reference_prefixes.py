"""Candidate-reference acquisition control for failed-rollout mining."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import common
import nd_environment as E
from nd_data import PERMUTATIONS,formula_text
from populations import identities,existing_exposure,load_guards
from mine_failed_prefixes import local_prefixes,role_for_context
from release_populations import check_record


def canonical_prefix_key(prompt,body):
    """Exact proof structure modulo atom renaming; ordered premises retained."""
    premises,_=E.parse_prompt(prompt);lines=E.independent_replay(prompt,body,True)["lines"]
    value=min(json.dumps([
        [formula_text(p).translate(mapping) for p in premises],
        [[line["depth"],formula_text(line["formula"]).translate(mapping),line["rule"],line["refs"]] for line in lines]
    ],separators=(",",":")) for mapping in PERMUTATIONS)
    return common.digest(value)


def compare(mined,occurrences,representatives):
    endpoints=defaultdict(list);prefixes=defaultdict(list)
    for row in occurrences:
        link={"parent_source_id":row["provenance"]["parent_source_id"],"source_id":row["source_id"],
              "parent_full_source_lines":row["provenance"]["parent_full_source_lines"],"prefix_lines":row["n_lines"]}
        endpoints[row["theorem_id"]].append(link);prefixes[row["canonical_prefix_id"]].append(link)
    representative_prefixes={r["canonical_prefix_id"] for r in representatives}
    rows=[]
    for row in mined:
        key=canonical_prefix_key(row["prompt"],row["body"])
        rows.append({"mined_source_id":row["source_id"],"theorem_id":row["theorem_id"],"context_id":row["context_id"],
                     "family":row["family"],"n_lines":row["n_lines"],"canonical_prefix_id":key,
                     "endpoint_in_candidate_reference_prefixes":row["theorem_id"] in endpoints,
                     "exact_canonical_prefix_in_candidate_reference_prefixes":key in prefixes,
                     "exact_canonical_prefix_in_deduplicated_representatives":key in representative_prefixes,
                     "endpoint_matches":endpoints.get(row["theorem_id"],[]),"exact_prefix_matches":prefixes.get(key,[])})
    return rows


def describe(rows):
    return {"n":len(rows),"unique_contexts":len({r["context_id"] for r in rows}),
            "canonical_endpoints":len({r["theorem_id"] for r in rows}),"canonical_prefixes":len({r["canonical_prefix_id"] for r in rows}),
            "certificate_length_histogram":dict(Counter(r["n_lines"] for r in rows)),
            "parent_source_length_histogram":dict(Counter(r["provenance"]["parent_full_source_lines"] for r in rows)),
            "local_family_histogram":dict(Counter(r["family"] for r in rows)),
            "parent_family_histogram":dict(Counter(r["provenance"]["parent_family"] for r in rows)),
            "unique_parent_sources":len({r["provenance"]["parent_source_id"] for r in rows})}


def run(args):
    started=time.time();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Reference-prefix output must be empty")
    mining=Path(args.mining_directory).resolve();miner_path=mining/"manifest.json"
    miner=json.loads(miner_path.read_text());completion=json.loads((mining/"completed.json").read_text())
    if not completion.get("ok") or completion["result_sha256"]!=common.sha(mining/"result.json"):
        raise ValueError("Mining diagnostic is incomplete or changed")
    mined_path=mining/"diagnostic_certificates.jsonl"
    if completion["certificates_sha256"]!=common.sha(mined_path):raise ValueError("Mined certificates changed")
    if miner.get("command")!="mine_failed_prefixes" or miner.get("status")!="diagnostic_only_not_training_ready":
        raise ValueError("Expected the bounded diagnostic-only miner")
    for path,digest in miner["input_sha256"].items():
        if common.sha(path)!=digest:raise ValueError("A frozen mining input changed")
    candidates_path=Path(args.candidates).resolve();candidate_hash=common.sha(candidates_path);population=candidates_path.parent
    pop_manifest_path=population/"manifest.json";pop=json.loads(pop_manifest_path.read_text())
    if (pop.get("status")!="released_complete_primary" or pop["population_sha256"]["candidate"]!=candidate_hash
            or miner["input_sha256"].get(str(candidates_path))!=candidate_hash):
        raise ValueError("Candidates differ from the complete primary release used in mining")
    baseline=Path(miner["args"]["baseline_data"]).resolve();guard_path=Path(miner["args"]["guard_json"]).resolve()
    known_paths=[baseline]+[population/f"{role}.jsonl" for role in ("candidate","adaptation","retention")]
    for path in known_paths:
        if miner["input_sha256"].get(str(path))!=common.sha(path):raise ValueError("Known-endpoint exclusion differs from miner")
    known,_=existing_exposure(known_paths);guards=load_guards(guard_path)
    candidates=common.read_jsonl(candidates_path)
    if len(candidates)!=768:raise ValueError("This acquisition control enumerates all768 released candidate sources")
    for row in candidates:check_record(row,"candidate",6)
    mined=common.read_jsonl(mined_path)
    for row in mined:
        if row.get("role")!="diagnostic_failed_rollout" or row.get("training_ready") is not False:
            raise ValueError("Unexpected mined role/status")
        if not E.strict_complete(row["prompt"],row["body"])["ok"]:raise ValueError("Invalid mined certificate")
    occurrences=[];representatives=[];seen=set();counts=Counter();reasons=Counter()
    for index,parent in enumerate(candidates):
        # Scan every original reference, including closed-source proofs whose
        # new local goal changes the closed-formula context hash and role bucket.
        local,diagnostic=local_prefixes(parent["prompt"],parent["body"])
        if diagnostic["first_invalid_reason"] is not None:raise AssertionError("Frozen candidate has an invalid prefix")
        counts.update(diagnostic["prefix_checks"]);counts["strictly_verified_local_prefix_occurrences"]+=len(local)
        for candidate in local:
            if role_for_context(candidate["context_id"])!="candidate":reasons["context_outside_candidate_partition"]+=1;continue
            if any(k is not None and k in group for k,group in zip(candidate["guard_ids"],guards)):
                reasons["blind_textbook_guard_collision"]+=1;continue
            if candidate["theorem_id"] in known:reasons["canonical_complete_endpoint_already_known"]+=1;continue
            record={**candidate,"source_id":common.digest([candidate["prompt"],candidate["body"]]),
                "canonical_prefix_id":canonical_prefix_key(candidate["prompt"],candidate["body"]),
                "role":"diagnostic_reference_prefix","diagnostic_only":True,"training_ready":False,"length_bin":"2-6",
                "provenance":{"kind":"candidate_reference_closed_local_prefix","parent_source_id":parent["source_id"],
                    "parent_theorem_id":parent["theorem_id"],"parent_context_id":parent["context_id"],"parent_family":parent["family"],
                    "parent_full_source_lines":parent["n_lines"],"parent_prompt":parent["prompt"],"parent_body":parent["body"],
                    "parent_original_provenance":parent["provenance"],"candidate_file_sha256":candidate_hash,
                    "candidate_row_index":index,"full_source_lines":candidate["n_lines"],"source_body_sha256":common.digest(candidate["body"]),
                    "original_premises_preserved":True,"closed_top_level_prefix":True,"no_model_generation":True}}
            occurrences.append(record)
            if record["theorem_id"] in seen:reasons["canonical_duplicate_within_reference_enumeration"]+=1
            else:seen.add(record["theorem_id"]);representatives.append(record)
    comparisons=compare(mined,occurrences,representatives)
    endpoint_matches=[r for r in comparisons if r["endpoint_in_candidate_reference_prefixes"]]
    exact_matches=[r for r in comparisons if r["exact_canonical_prefix_in_candidate_reference_prefixes"]]
    output.mkdir(parents=True,exist_ok=True)
    common.write_jsonl(output/"diagnostic_certificates.jsonl",representatives)
    common.write_jsonl(output/"admissible_occurrences.jsonl",occurrences)
    common.write_jsonl(output/"mined_overlap.jsonl",comparisons)
    result={"status":"diagnostic_only_not_training_ready","candidate_sources_enumerated":len(candidates),"known_complete_endpoints":len(known),
            "counts":dict(counts),"rejection_counts":dict(reasons),"all_admissible_occurrences":describe(occurrences),
            "canonical_endpoint_representatives":describe(representatives),"mined_certificates_compared":len(mined),
            "mined_endpoint_overlap":len(endpoint_matches),"mined_exact_canonical_prefix_overlap":len(exact_matches),
            "mined_exact_overlap_with_only_retained_representatives":sum(r["exact_canonical_prefix_in_deduplicated_representatives"] for r in comparisons),
            "overlap_unique_contexts":{"endpoint":len({r["context_id"] for r in endpoint_matches}),"exact_prefix":len({r["context_id"] for r in exact_matches})},
            "overlap_family_histograms":{"endpoint":dict(Counter(r["family"] for r in endpoint_matches)),"exact_prefix":dict(Counter(r["family"] for r in exact_matches))},
            "overlap_certificate_length_histograms":{"endpoint":dict(Counter(r["n_lines"] for r in endpoint_matches)),"exact_prefix":dict(Counter(r["n_lines"] for r in exact_matches))},
            "new_model_rollouts":0,"optimizer_updates":0,"model_flop_proxy":0,"elapsed_seconds":time.time()-started,
            "selection_or_confirmation_sources_enumerated":False,"textbook_payloads_opened":False,
            "comparison_scope":"Exact-prefix overlap uses every admissible pre-dedup reference prefix, so representative choice cannot hide alternative known proofs."}
    manifest={"command":"enumerate_reference_prefixes","args":vars(args),"status":result["status"],
              "endpoint_equivalence":"Existing canonical theorem identity: atom renaming and premise permutation invariant",
              "exact_prefix_equivalence":"Atom renaming invariant; exact ordered premises, line order, formulae, scopes, rule labels and references preserved",
              "same_role_guard_and_known_endpoint_exclusions_as_miner":True,
              "input_sha256":{str(miner_path):common.sha(miner_path),str(mined_path):common.sha(mined_path),str(pop_manifest_path):common.sha(pop_manifest_path),
                              str(guard_path):common.sha(guard_path),**{str(p):common.sha(p) for p in known_paths}},
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},"dependency_sha256":common.dependency_hashes()}
    common.write_json(output/"manifest.json",manifest);common.write_json(output/"result.json",result)
    common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json"),"certificates_sha256":common.sha(output/"diagnostic_certificates.jsonl"),
                                              "occurrences_sha256":common.sha(output/"admissible_occurrences.jsonl"),"overlap_sha256":common.sha(output/"mined_overlap.jsonl")})
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("candidates","mining-directory","output"):p.add_argument("--"+flag,required=True)
    run(p.parse_args())
