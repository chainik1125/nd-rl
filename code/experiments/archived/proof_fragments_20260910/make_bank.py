"""Freeze distinct-source RL banks without consulting model outcomes.

Selected: exactly all sources in a declared family shortlist. Uniform: sample
without replacement inside each full-source-length stratum to match a frozen
selected reference bank's histogram. Uniform may overlap the selected bank;
that overlap is measured, not prohibited by a changed sampling distribution.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import time
import common
from nd_environment import Tokenizer, episode
from release_populations import check_record

def validated_pool(rows):
    seen_sources, seen_theorems = set(), set()
    for row in rows:
        check_record(row,"candidate",6)
        if row.get("diagnostic_only") or row.get("primary_training_admissible") is False:
            raise ValueError("Diagnostic records cannot enter a primary RL bank")
        if row["source_id"] in seen_sources or row["theorem_id"] in seen_theorems:
            raise ValueError("Candidate pool contains duplicate source/theorem")
        seen_sources.add(row["source_id"]);seen_theorems.add(row["theorem_id"])
    return rows

def select_families(rows,families,per_family=64,size=192,seed=0):
    if len(set(families))!=len(families):raise ValueError("Repeated selected family")
    if len(families)*per_family!=size:raise ValueError("Selected family count does not match bank size")
    selected=[r for r in rows if r["family"] in set(families)]
    if Counter(r["family"] for r in selected)!=Counter({f:per_family for f in families}):
        raise ValueError("Selected family bank lacks exact distinct-source quotas")
    selected.sort(key=lambda r:r["source_id"])
    random.Random(seed).shuffle(selected)
    return selected

def uniform_match(rows,reference,size=192,seed=0):
    if len(reference)!=size or len({r["source_id"] for r in reference})!=size:
        raise ValueError("Reference bank size or source uniqueness differs")
    by_source={r["source_id"]:r for r in rows}
    for row in reference:
        if row["source_id"] not in by_source or common.digest(row)!=common.digest(by_source[row["source_id"]]):
            raise ValueError("Reference bank is not an exact subset of the bound candidate pool")
    quota=Counter(r["n_lines"] for r in reference)
    rng=random.Random(seed)
    selected=[]
    for length,n in sorted(quota.items()):
        pool=sorted((r for r in rows if r["n_lines"]==length),key=lambda r:r["source_id"])
        if len(pool)<n:raise ValueError(f"Cannot match length {length} without replacement")
        selected.extend(rng.sample(pool,n))
    rng.shuffle(selected)
    if len({r["source_id"] for r in selected})!=size:raise AssertionError("Uniform sampling reused a source")
    if Counter(r["n_lines"] for r in selected)!=quota:raise AssertionError("Uniform source-length histogram differs")
    return selected

def describe(rows):
    tok=Tokenizer()
    token_lengths=[];suffix_lengths=[];prefix_lengths=[];depths=[]
    for r in rows:
        e=episode(r,r["default_cut"],tok)
        token_lengths.append(len(e["prompt_ids"])+len(e["response_ids"]))
        suffix_lengths.append(len(e["response_ids"]));prefix_lengths.append(len(e["prefix_ids"]))
        depths.append(len(e["prefix_state"]["active_assumptions"]))
    context_counts=Counter(r["context_id"] for r in rows)
    def distribution(values):
        ordered=sorted(values)
        return {"min":min(values),"mean":statistics.mean(values),"median":statistics.median(values),"max":max(values),
                "p10":ordered[int(.1*(len(ordered)-1))],"p90":ordered[int(.9*(len(ordered)-1))]}
    return {"n":len(rows),"unique_sources":len({r["source_id"] for r in rows}),
            "unique_theorems":len({r["theorem_id"] for r in rows}),"unique_contexts":len(context_counts),
            "max_context_multiplicity":max(context_counts.values()),
            "context_effective_count":len(rows)**2/sum(n*n for n in context_counts.values()),
            "family_histogram":dict(Counter(r["family"] for r in rows)),
            "full_source_length_histogram":dict(Counter(r["n_lines"] for r in rows)),
            "source_max_depth_histogram":dict(Counter(r["max_depth"] for r in rows)),
            "default_cut_depth_histogram":dict(Counter(depths)),
            "total_encoded_tokens":distribution(token_lengths),"default_suffix_tokens":distribution(suffix_lengths),
            "supplied_prefix_tokens":distribution(prefix_lengths)}

def run(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Bank output must be empty")
    data=Path(args.data).resolve()
    metadata_path=data.parent/"manifest.json"
    metadata=json.loads(metadata_path.read_text())
    if metadata.get("status")!="released_complete_primary":raise ValueError("Banks require root's audited primary population release")
    data_hash=common.sha(data)
    if metadata["population_sha256"].get("candidate")!=data_hash:
        raise ValueError("Candidate pool differs from the released population")
    rows=validated_pool(common.read_jsonl(data))
    if len(rows)!=12*64 or len(set(r["family"] for r in rows))!=12:
        raise ValueError("Expected 768-source twelve-family primary pool")
    input_hashes={str(data):data_hash,str(metadata_path):common.sha(metadata_path)}
    reference=None
    if args.mode=="selected":
        if not args.families or len(args.families)!=3 or args.reference_bank:
            raise ValueError("Selected bank needs exactly three explicit families and no reference bank")
        selected=select_families(rows,args.families,64,192,args.seed)
        rule="All 64 distinct sources from each of three prospectively frozen families"
    else:
        if args.families or not args.reference_bank:
            raise ValueError("Uniform matching needs a reference bank and no family shortlist")
        reference_path=Path(args.reference_bank).resolve()
        input_hashes[str(reference_path)]=common.sha(reference_path)
        reference_metadata=reference_path.parent/"manifest.json"
        if not reference_metadata.exists():raise ValueError("Reference bank lacks immutable manifest")
        refmeta=json.loads(reference_metadata.read_text())
        if refmeta.get("status")!="frozen_bank" or refmeta.get("mode")!="selected" or refmeta["bank_sha256"]!=input_hashes[str(reference_path)]:
            raise ValueError("Reference bank differs from its selected-bank manifest")
        if refmeta["input_sha256"].get(str(data))!=data_hash:
            raise ValueError("Reference bank was built from a different candidate pool")
        input_hashes[str(reference_metadata)]=common.sha(reference_metadata)
        reference=common.read_jsonl(reference_path)
        selected=uniform_match(rows,reference,192,args.seed)
        rule="Uniform without replacement within each exact full-source-length stratum of the selected reference bank"
    receipts=describe(selected)
    if reference is not None:
        reference_receipts=describe(reference)
        receipts.update({"reference_statistics":reference_receipts,
                         "exact_full_source_length_histogram_match":receipts["full_source_length_histogram"]==reference_receipts["full_source_length_histogram"],
                         "sources_shared_with_reference":len({r["source_id"] for r in selected}&{r["source_id"] for r in reference}),
                         "contexts_shared_with_reference":len({r["context_id"] for r in selected}&{r["context_id"] for r in reference})})
    output.mkdir(parents=True,exist_ok=True)
    common.write_jsonl(output/"bank.jsonl",selected)
    manifest={"status":"frozen_bank","mode":args.mode,"rule":rule,"seed":args.seed,
              "created_unix":time.time(),"families":args.families,"size":192,"per_selected_family":64,
              "input_sha256":input_hashes,"bank_sha256":common.sha(output/"bank.jsonl"),
              "source_ids_in_bank_order":[r["source_id"] for r in selected],
              "source_set_sha256":common.digest(sorted(r["source_id"] for r in selected)),
              "statistics":receipts,"model_outcomes_used_by_builder":False,
              "family_choices_supplied_by_root":args.mode=="selected",
              "token_length_matching":"none; token lengths and scopes are descriptive mechanisms",
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes()}
    common.write_json(output/"manifest.json",manifest)
    print(json.dumps({"status":"frozen_bank","mode":args.mode,"bank_sha256":manifest["bank_sha256"],"statistics":receipts}),flush=True)

def parser():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=("selected","uniform"),required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--families",nargs="*");p.add_argument("--reference-bank")
    return p

if __name__=="__main__":run(parser().parse_args())
