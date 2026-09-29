"""Propose, but never release, matched mined/reference full-certificate cohorts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import common
import learning as L
import nd_environment as E
from release_populations import check_record
from oracle_sft import bind_population


def full_encoding(row,tok):
    check_record(row,row["role"],6)
    encoded=E.episode(row,0,tok)
    if encoded["prefix_ids"] or encoded["reset_index"]:raise AssertionError("Full-certificate mode supplied proof tokens")
    if not E.judge(encoded,encoded["response_ids"],tok)["reward"]:raise AssertionError("Full local certificate did not round-trip")
    if E.split_body(tok.decode_body(encoded["response_ids"][1:]))!=E.split_body(row["body"]):
        raise AssertionError("Tokenization changed the complete local certificate")
    return encoded


def make_pairs(mined,references,tok,seed):
    mine_ids={r["theorem_id"] for r in mined}
    if len(mine_ids)!=len(mined) or len({r["theorem_id"] for r in references})!=len(references):
        raise ValueError("Cohort inventories must contain distinct canonical endpoints")
    encoded={r["source_id"]:full_encoding(r,tok) for r in mined+references}
    eligible=[r for r in mined if 2<=r["n_lines"]<=5]
    excluded=[r for r in mined if not 2<=r["n_lines"]<=5]
    controls=[r for r in references if r["theorem_id"] not in mine_ids and 2<=r["n_lines"]<=5]
    need=Counter(r["n_lines"] for r in eligible);available=Counter(r["n_lines"] for r in controls)
    if any(available[n]<count for n,count in need.items()):raise ValueError("Insufficient distinct reference-only endpoints in an exact line-length stratum")
    order=list(range(len(controls)));random.Random(seed).shuffle(order)
    tie_rank={controls[i]["source_id"]:rank for rank,i in enumerate(order)}
    unused={r["source_id"]:r for r in controls};pairs=[]
    for mine in eligible:
        mine_encoded=encoded[mine["source_id"]]
        def distance(row):
            other=encoded[row["source_id"]]
            return (abs(len(mine_encoded["response_ids"])-len(other["response_ids"])),
                    abs(len(mine_encoded["prompt_ids"])-len(other["prompt_ids"])),tie_rank[row["source_id"]])
        match=min((r for r in unused.values() if r["n_lines"]==mine["n_lines"]),key=distance)
        unused.pop(match["source_id"]);pairs.append((mine,match))
    return pairs,excluded,encoded,{"eligible_mined_by_length":dict(need),"reference_only_available_by_length":dict(available),
                                  "reference_endpoints_excluded_for_any_mined_overlap":len(references)-len(controls)}


def ancestry_flags(row,review):
    theorem=row["theorem_id"]
    endpoints=review.get("intermediate_exposure_hits",{}).get(theorem,[])
    proofs=review.get("proof_exposure_hits",{}).get(theorem,[])
    roles=lambda hits:sorted({h["role"] for h in hits})
    ancestral={"baseline","adaptation"};acquired={"candidate","selection"}
    return {"endpoint_prefix_exposure_roles":roles(endpoints),"exact_proof_prefix_exposure_roles":roles(proofs),
            "in_parent_supervised_ancestry_endpoint_inventory":any(h["role"] in ancestral for h in endpoints),
            "in_parent_supervised_ancestry_exact_proof_inventory":any(h["role"] in ancestral for h in proofs),
            "in_acquired_candidate_or_selection_endpoint_inventory":any(h["role"] in acquired for h in endpoints),
            "in_acquired_candidate_or_selection_exact_proof_inventory":any(h["role"] in acquired for h in proofs),
            "retention_inventory_endpoint_match":any(h["role"]=="retention" for h in endpoints),
            "retention_inventory_exact_proof_match":any(h["role"]=="retention" for h in proofs),
            "endpoint_exposure_hits":endpoints,"exact_proof_exposure_hits":proofs,
            "ancestry_classification":"Membership in baseline/adaptation training-source inventories; individual historical sample draws are not newly reconstructed here."}


def statistics(rows,encoded):
    response=[len(encoded[r["source_id"]]["response_ids"]) for r in rows]
    context=[len(encoded[r["source_id"]]["prompt_ids"]) for r in rows]
    counts=Counter(r["context_id"] for r in rows)
    return {"n":len(rows),"line_histogram":dict(Counter(r["n_lines"] for r in rows)),"family_histogram":dict(Counter(r["family"] for r in rows)),
            "unique_contexts":len(counts),"max_context_multiplicity":max(counts.values()),
            "context_effective_count":len(rows)**2/sum(n*n for n in counts.values()),
            "full_response_tokens":{"total":sum(response),"mean":sum(response)/len(response),"min":min(response),"max":max(response)},
            "prompt_tokens":{"total":sum(context),"mean":sum(context)/len(context),"min":min(context),"max":max(context)},
            "max_encoded_length":max(a+b for a,b in zip(response,context))}


def bind_diagnostic(directory,role):
    directory=Path(directory).resolve();completion=json.loads((directory/"completed.json").read_text())
    data=directory/"diagnostic_certificates.jsonl"
    if not completion.get("ok") or completion["certificates_sha256"]!=common.sha(data):raise ValueError("Diagnostic certificate inventory changed")
    if completion["result_sha256"]!=common.sha(directory/"result.json"):raise ValueError("Diagnostic result changed")
    rows=common.read_jsonl(data)
    if any(r.get("role")!=role or r.get("training_ready") is not False for r in rows):raise ValueError("Unexpected diagnostic role or status")
    return rows,data


def run(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Proposal output must be empty")
    mined,mined_path=bind_diagnostic(args.mined_directory,"diagnostic_failed_rollout")
    references,reference_path=bind_diagnostic(args.reference_directory,"diagnostic_reference_prefix")
    overlap_path=reference_path.parent/"mined_overlap.jsonl";overlap={r["mined_source_id"]:r for r in common.read_jsonl(overlap_path)}
    if len(mined)!=85 or len(overlap)!=85:raise ValueError("This bounded proposal uses the current85 mined certificates")
    exposure_path=Path(args.exposure_review).resolve();review=json.loads(exposure_path.read_text())
    if review.get("status")!="independent_cpu_review_passed" or review.get("certificate_sha256")!=common.sha(mined_path):
        raise ValueError("Ancestry review does not bind the mined inventory")
    reference_manifest=json.loads((reference_path.parent/"manifest.json").read_text())
    if reference_manifest["input_sha256"].get(str(mined_path))!=common.sha(mined_path):raise ValueError("Reference overlap report concerns a different mined inventory")
    reference_completion=json.loads((reference_path.parent/"completed.json").read_text())
    if reference_completion["overlap_sha256"]!=common.sha(overlap_path):raise ValueError("Known overlap flags changed")
    retention=bind_population(args.retention,"retention")
    checkpoint_hash=common.sha(args.checkpoint)
    if any(r["provenance"]["initial_checkpoint_sha256"]!=checkpoint_hash for r in mined):
        raise ValueError("Proposal checkpoint differs from the mined parent policy's common adapted initialization")
    retention_records=common.read_jsonl(retention[0])
    if len(retention_records)!=512:raise ValueError("Expected the common512-source retention population")
    for row in retention_records:check_record(row,"retention",6)
    model,tok,metadata=L.load_model(args.checkpoint,"cpu")
    if metadata.get("format")!=E.FORMAT:raise ValueError("Proposal must bind the common adapted input format")
    pairs,excluded,encoded,eligibility=make_pairs(mined,references,tok,args.seed)
    if len(pairs)!=84 or len(excluded)!=1 or excluded[0]["n_lines"]!=6:raise ValueError("Expected84eligible mined rows and exactly one length6 exclusion")
    mine_rows=[];reference_rows=[];matching=[]
    for index,(mine,reference) in enumerate(pairs):
        flags=ancestry_flags(mine,review)
        mine_rows.append({**mine,"comparison_arm":"mined","paired_index":index,"paired_source_id":reference["source_id"],
                          "known_reference_overlap":overlap[mine["source_id"]],"supervised_ancestry_and_acquired_reference_exposure":flags})
        reference_rows.append({**reference,"comparison_arm":"reference","paired_index":index,"paired_source_id":mine["source_id"],
                               "endpoint_disjoint_from_all85_mined":True,"acquired_candidate_reference_prefix":True})
        a,b=encoded[mine["source_id"]],encoded[reference["source_id"]]
        matching.append({"paired_index":index,"mined_source_id":mine["source_id"],"reference_source_id":reference["source_id"],
                         "certificate_lines":mine["n_lines"],"mined_full_response_tokens":len(a["response_ids"]),"reference_full_response_tokens":len(b["response_ids"]),
                         "response_token_distance":abs(len(a["response_ids"])-len(b["response_ids"])),
                         "prompt_token_distance":abs(len(a["prompt_ids"])-len(b["prompt_ids"])),
                         "same_context":mine["context_id"]==reference["context_id"],"same_family":mine["family"]==reference["family"]})
    stats={"mined":statistics(mine_rows,encoded),"reference":statistics(reference_rows,encoded)}
    retention_encoded=L.encode_population(retention_records,tok,"empty")
    max_retention_length=max(len(r["prompt_ids"])+len(r["response_ids"]) for r in retention_encoded)
    if max(max(x["max_encoded_length"] for x in stats.values()),max_retention_length)>min(args.fixed_pad,model.config.max_seq_len):
        raise ValueError("Declared full-certificate padding cannot fit every proposed example")
    family_mine=Counter(r["family"] for r in mine_rows);family_ref=Counter(r["family"] for r in reference_rows)
    residual={"exact_line_histograms_match":stats["mined"]["line_histogram"]==stats["reference"]["line_histogram"],
              "mean_absolute_response_token_distance":sum(r["response_token_distance"] for r in matching)/84,
              "max_absolute_response_token_distance":max(r["response_token_distance"] for r in matching),
              "exact_response_token_pairs":sum(r["response_token_distance"]==0 for r in matching),
              "same_context_pairs":sum(r["same_context"] for r in matching),"same_family_pairs":sum(r["same_family"] for r in matching),
              "shared_contexts_between_arms":len({r["context_id"] for r in mine_rows}&{r["context_id"] for r in reference_rows}),
              "family_total_variation":sum(abs(family_mine[f]-family_ref[f]) for f in set(family_mine)|set(family_ref))/(2*84)}
    exposure_counts={key:sum(r["supervised_ancestry_and_acquired_reference_exposure"][key] for r in mine_rows)
                     for key in ("in_parent_supervised_ancestry_endpoint_inventory","in_parent_supervised_ancestry_exact_proof_inventory",
                                 "in_acquired_candidate_or_selection_endpoint_inventory","in_acquired_candidate_or_selection_exact_proof_inventory",
                                 "retention_inventory_endpoint_match","retention_inventory_exact_proof_match")}
    output.mkdir(parents=True,exist_ok=True)
    common.write_jsonl(output/"mined.jsonl",mine_rows);common.write_jsonl(output/"reference.jsonl",reference_rows)
    common.write_jsonl(output/"excluded_mined.jsonl",[{**r,"proposal_exclusion_reason":"length6 has no eligible reference-control support"} for r in excluded])
    common.write_jsonl(output/"matching.jsonl",matching)
    manifest={"status":"proposed_not_released","training_authorized":False,"n_per_arm":84,"role_preservation":True,"supervision":"full-certificate",
              "args":vars(args),"eligibility":eligibility,"statistics":stats,"residual_imbalance":residual,"mined_exposure_classes":exposure_counts,
              "match_rule":"Exact complete local-certificate line length; greedy response-token distance, prompt-token distance, then seeded random tie rank. No model outcome is used.",
              "control_endpoint_disjoint_from_all85_mined":True,"paired_order_preserves_per_draw_line_length_matching":True,
              "cohort_sha256":{"mined":common.sha(output/"mined.jsonl"),"reference":common.sha(output/"reference.jsonl")},
              "input_sha256":{str(mined_path):common.sha(mined_path),str(reference_path):common.sha(reference_path),str(overlap_path):common.sha(overlap_path),
                              str(exposure_path):common.sha(exposure_path),str(Path(args.checkpoint).resolve()):common.sha(args.checkpoint),
                              str(retention[0]):retention[1],str(retention[2]):retention[3]},
              "full_tokenization_roundtrip_passed":True,"maximum_full_source_cap":6,
              "limitations":["Online cohort is conditioned on failed rollouts and is not a random sample of all learnable local goals.",
                             "Line lengths and padded compute can be matched; context multiplicity, families and remaining token lengths differ.",
                             "Some mined endpoints/proofs occur in parent supervised-source inventories, others only in acquired candidate references; exposure flags are preserved.",
                             "Control endpoints are disjoint from mined endpoints; this comparison cannot isolate acquisition method from theorem distribution.",
                             "Per-example historical ancestor sampling draws are not reconstructed by the exposure inventory review.",
                             "The one six-line mined certificate is excluded for lack of control support.",
                             "Reference-control ancestry exposure beyond its acquired-candidate origin is not measured in this proposal."],
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},"dependency_sha256":common.dependency_hashes()}
    common.write_json(output/"manifest.json",manifest)
    template={"frozen":False,"approved_for_diagnostic_full_certificate_sft":False,"proposal_manifest_sha256":common.sha(output/"manifest.json"),
              "cohort_sha256":list(manifest["cohort_sha256"].values()),"checkpoints":[common.sha(args.checkpoint)],"retention_sha256":retention[1],
              "supervision":"full-certificate","full_source_supervision_cap":6,
              "training_config":{"steps":None,"seed":None,"lr":5e-5,"batch_size":32,"fixed_pad":args.fixed_pad,"retention_fraction":.5,"precision":"fp32"}}
    common.write_json(output/"release_template.json",template)
    common.write_json(output/"proposal_result.json",{"status":manifest["status"],"eligibility":eligibility,"statistics":stats,
        "residual_imbalance":residual,"mined_exposure_classes":exposure_counts,"proposal_manifest_sha256":common.sha(output/"manifest.json")})
    print(json.dumps({"status":manifest["status"],"eligibility":eligibility,"statistics":stats,"residual_imbalance":residual,"mined_exposure_classes":exposure_counts}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("mined-directory","reference-directory","exposure-review","checkpoint","retention","output"):p.add_argument("--"+flag,required=True)
    p.add_argument("--seed",type=int,default=20260910);p.add_argument("--fixed-pad",type=int,default=512);run(p.parse_args())
