"""Reconstruct primary raw rollouts and independently verify reported rewards."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import time
import common
from nd_environment import Tokenizer,episode,judge,strict_complete
from proof_diagnostics import compress_proof
from analyze_lm import jsonl,training_analysis


def audit(directory):
    directory=Path(directory);started=time.monotonic()
    summary=training_analysis(directory)
    config=summary["config"]
    sources={r["source_id"]:r for r in common.read_jsonl(config["data"])}
    targets={r["source_id"]:r for r in common.read_jsonl(config["targets"])}
    if set(sources)&set(targets):raise AssertionError("Source/target identities overlap")
    tok=Tokenizer();prefixes={};verdict_cache={};witnesses={};counts=Counter();groups={}
    for row in jsonl(directory/"attempts.jsonl"):
        step,g,attempt=row["step"],row["group"],row["attempt_index"]
        if not 1<=step<=summary["updates"]:raise AssertionError("Uncommitted raw attempt")
        group_key=(step,g)
        if group_key not in groups:groups[group_key]={"indices":set(),"successes":0,"tokens":0}
        group=groups[group_key]
        if attempt in group["indices"]:raise AssertionError("Duplicate raw attempt")
        group["indices"].add(attempt)
        kind=row["training_kind"];sid=row["source_id"]
        bank=sources if kind=="source" else targets if kind=="target" else None
        if bank is None or sid not in bank or row["prompt"]!=bank[sid]["prompt"]:
            raise AssertionError("Raw rollout task identity mismatch")
        if kind=="source":
            key=(sid,row["reset_index"])
            if key not in prefixes:prefixes[key]=episode(sources[sid],row["reset_index"],tok)
            expected=prefixes[key]
            if row["prompt_ids"]!=expected["prompt_ids"] or row["prefix_ids"]!=expected["prefix_ids"]:
                raise AssertionError("Source prefix differs from cap-six input")
        else:
            if row["reset_index"] or row["prefix_ids"] or row["prompt_ids"]!=tok.encode_prompt(row["prompt"])+[tok.prf_id]:
                raise AssertionError("Theorem-only target has hidden prefix assistance")
        n=len(row["response_ids"])
        if not 1<=n<=config["max_new_tokens"]:raise AssertionError("Response budget violated")
        for field in ("rollout_logprobs","old_values","token_rewards","returns","advantages_raw","advantages"):
            if len(row[field])!=n or not all(math.isfinite(x) for x in row[field]):
                raise AssertionError(f"Invalid sampled-token array: {field}")
        if row["token_rewards"]!=[0.]*(n-1)+[float(row["reward"])]:
            raise AssertionError("Nonterminal/shaped token reward in primary")
        cache_key=common.digest([row["prompt"],row["prefix_ids"],row["response_ids"]])
        if cache_key not in verdict_cache:verdict_cache[cache_key]=judge(row,row["response_ids"],tok)
        verdict=verdict_cache[cache_key]
        if verdict["reward"]!=row["reward"]:raise AssertionError("Raw reward disagrees with reconstructed certificate")
        if row["reward"]:
            if verdict["body"]!=row["body"] or verdict["n_lines"]!=row["n_lines"]:
                raise AssertionError("Successful raw body/length differs from sampled tokens")
            if row["finish_reason"]!="qed":raise AssertionError("Nonterminal raw proof counted as success")
            key=common.digest([row["prompt"],row["body"]])
            if key not in witnesses:
                comp=compress_proof(row["prompt"],row["body"])
                if not strict_complete(row["prompt"],comp["compressed_proof"])["ok"]:
                    raise AssertionError("Analysis-reduced certificate invalid")
                witnesses[key]={"source_id":sid,"training_kind":kind,"first_step":step,"written":row["n_lines"],
                                "pruned":comp["dependency_pruned_length"],"compressed":comp["compressed_length"],
                                "prompt":row["prompt"],"body":row["body"],"compressed_body":comp["compressed_proof"]}
        group["successes"]+=row["reward"];group["tokens"]+=n
        counts[kind+"_attempts"]+=1;counts[kind+"_successes"]+=row["reward"];counts["sampled_tokens"]+=n
    declared_groups=list(jsonl(directory/"groups.jsonl"))
    if len(groups)!=len(declared_groups):raise AssertionError("Raw/group ledger cardinality mismatch")
    for declared in declared_groups:
        raw=groups[declared["step"],declared["group"]]
        if raw["indices"]!=set(range(config["group_size"])) or raw["successes"]!=declared["successes"] or raw["tokens"]!=declared["generated_tokens"]:
            raise AssertionError("Raw attempts differ from committed group ledger")
    for kind in ("source","target"):
        if counts[kind+"_attempts"]!=summary["result"]["attempts"][kind] or counts[kind+"_successes"]!=summary["result"]["successes"][kind]:
            raise AssertionError("Raw counts differ from final training endpoint")
    if counts["sampled_tokens"]!=summary["result"]["generated_tokens"]:raise AssertionError("Raw token total mismatch")
    target_witnesses=[w for w in witnesses.values() if w["training_kind"]=="target"]
    by_target=defaultdict(list)
    for witness in target_witnesses:by_target[witness["source_id"]].append(witness)
    discoveries={r["witness_id"]:r for r in jsonl(directory/"discoveries.jsonl")}
    if set(discoveries)!={k for k,w in witnesses.items() if w["training_kind"]=="target"}:
        raise AssertionError("Discovery archive differs from actual successful target responses")
    return {"status":"passed","directory":str(directory.resolve()),"analysis_code_sha256":common.sha(__file__),
        "elapsed_seconds":time.monotonic()-started,"raw_counts":dict(counts),"groups":len(groups),
        "unique_encoded_rollouts_rechecked":len(verdict_cache),"unique_successful_certificates":len(witnesses),
        "unique_discovered_targets":len(by_target),
        "target_any_pruned_over6":sum(any(w["pruned"]>6 for w in ws) for ws in by_target.values()),
        "target_shortest_found_pruned_over6":sum(min(w["pruned"] for w in ws)>6 for ws in by_target.values()),
        "target_shortest_pruned_histogram":dict(Counter(min(w["pruned"] for w in ws) for ws in by_target.values())),
        "target_witnesses":target_witnesses,"training_summary":summary,
        "input_sha256":{str(directory/name):common.sha(directory/name) for name in
                        ("attempts.jsonl","groups.jsonl","updates.jsonl","discoveries.jsonl","manifest.json","result.json")},
        "review_scope":"Root read-only audit. Exact duplicate encoded rollouts reuse a verified verdict. Every reported count reconciles to raw attempts."}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();result=audit(args.directory);common.write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("training_summary","target_witnesses","input_sha256")}))
