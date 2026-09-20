"""Candidate-only all-cut fitting and separate fresh-draw controller audit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import torch
import common
import learning as L
import nd_environment as E
from fragment_experiment import training_rows
from initial_cuts import RULE,choose_cut,state_hash,validate_map

def candidate_identity(path,digest):
    metadata=json.loads((Path(path).parent/"manifest.json").read_text())
    return ((metadata.get("status")=="released_complete_primary" and metadata.get("population_sha256",{}).get("candidate")==digest)
            or (metadata.get("status")=="frozen_bank" and metadata.get("bank_sha256")==digest))

def contract(args,checkpoint_hash):
    return {"checkpoint_sha256":checkpoint_hash,"format":E.FORMAT,"temperature":args.temperature,
            "max_new_tokens":args.max_new_tokens,"precision":args.precision,"group_size":8,
            "sampler_sha256":common.sha(L.__file__),"environment_sha256":common.sha(E.__file__)}

def metadata_compatibility(manifest,current):
    args=manifest.get("args",{})
    for field in ("temperature","max_new_tokens","precision"):
        if args.get(field)!=current[field]:return f"different {field}"
    if args.get("samples")!=8:return "producer did not draw exactly eight attempts per state"
    if manifest.get("input_sha256",{}).get("checkpoint")!=current["checkpoint_sha256"]:return "different checkpoint"
    if manifest.get("format")!=current["format"]:return "different state format"
    code=manifest.get("code_sha256",{})
    for filename,key in (("learning.py","sampler_sha256"),("nd_environment.py","environment_sha256")):
        observed=[v for p,v in code.items() if Path(p).name==filename]
        if observed!=[current[key]]:return f"different or unbound {filename}"
    if manifest.get("dependency_sha256")!=common.dependency_hashes():return "different pinned model/parser/verifier dependencies"
    return None

def reuse_endpoints(directories,expected,tok,current):
    groups={};receipts=[];seeds=[];charged=0
    for directory in directories:
        directory=Path(directory).resolve()
        if any(tag in directory.name.lower() for tag in ("confirmation","textbook","final")):
            raise ValueError("Pilot reuse is restricted to candidate diagnostics")
        manifest_path=directory/"manifest.json";manifest=json.loads(manifest_path.read_text())
        problem=metadata_compatibility(manifest,current)
        receipt={"directory":str(directory),"manifest_sha256":common.sha(manifest_path),"reused_states":0}
        if problem:
            receipts.append({**receipt,"status":"incompatible_resample","reason":problem});continue
        producer_data=manifest["args"]["data"];producer_data_hash=manifest["input_sha256"]["data"]
        if not candidate_identity(producer_data,producer_data_hash):
            raise ValueError("Producer input is not a frozen candidate bank/population")
        completion=json.loads((directory/"completed.json").read_text())
        result_path=directory/"result.json"
        if not completion.get("ok") or completion["result_sha256"]!=common.sha(result_path):
            raise ValueError("Reused producer is incomplete or its result changed")
        result=json.loads(result_path.read_text())
        attempts_path=directory/"attempts.jsonl";attempts_hash=common.sha(attempts_path)
        raw=common.read_jsonl(attempts_path);local=defaultdict(list)
        for i,row in enumerate(raw):
            key=(row.get("source_id"),row.get("reset_index"))
            if key not in expected:continue
            ep=expected[key]
            if any(row.get(k)!=ep[k] for k in ("prompt","prompt_ids","original_prompt_ids","prefix_ids","reset_index","source_id","source_lines","format")):
                raise ValueError("Reused attempt differs from the exact source/prefix state")
            if row.get("effective_max_new_tokens")!=current["max_new_tokens"]:
                raise ValueError("Reused attempt had a clipped/different response budget")
            verdict=E.judge(ep,row["response_ids"],tok)
            if verdict["reward"]!=row.get("reward"):raise ValueError("Reused reward differs from current strict verification")
            local[key].append({**ep,"response_ids":row["response_ids"],"rollout_logprobs":row["rollout_logprobs"],
                               "reward":verdict["reward"],"finish_reason":row.get("finish_reason"),
                               "effective_max_new_tokens":row["effective_max_new_tokens"],
                               "pilot_origin":{"kind":"reused_endpoint","directory":str(directory),"attempts_sha256":attempts_hash,
                                               "attempt_row_index":i,"attempt_index":row.get("attempt_index")}})
        for key,rows in local.items():
            if len(rows)!=8 or sorted(r["pilot_origin"]["attempt_index"] for r in rows)!=list(range(8)):
                raise ValueError("Reused state does not contain one distinct set of eight attempts")
            if key not in groups:
                groups[key]=rows;receipt["reused_states"]+=1
        proxy=result.get("total_flop_proxy",L.total_flop_proxy(result.get("ledger",{})))
        charged+=proxy;seeds.append(manifest["args"]["seed"])
        receipts.append({**receipt,"status":"compatible","attempts_sha256":attempts_hash,"result_sha256":common.sha(result_path),
                         "original_full_run_flop_proxy_charged":proxy,"producer_seed":manifest["args"]["seed"]})
    return groups,receipts,seeds,charged

def setup(args):
    started=time.time()
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Pilot output must be empty")
    data=Path(args.data).resolve();data_hash=common.sha(data)
    if not candidate_identity(data,data_hash):raise ValueError("All-cut pilot requires a frozen candidate population/bank")
    records=training_rows(data)
    if any(r["role"]!="candidate" for r in records):raise ValueError("All-cut pilot uses candidate sources only")
    if len({r["source_id"] for r in records})!=len(records):raise ValueError("Duplicate candidate sources")
    if args.temperature<=0:raise ValueError("Pilot needs a stochastic categorical policy")
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tok,metadata=L.load_model(args.checkpoint,"cpu" if args.plan_only else args.device)
    if metadata.get("format")!=E.FORMAT:raise ValueError("Pilot must use the common adapted policy")
    current=contract(args,common.sha(args.checkpoint))
    expected={(r["source_id"],cut):E.episode(r,cut,tok) for r in records for cut in L.admissible_cuts(r)}
    if any(len(e["prompt_ids"])+args.max_new_tokens>model.config.max_seq_len for e in expected.values()):
        raise ValueError("Full fixed pilot response budget does not fit every admissible state")
    output.mkdir(parents=True,exist_ok=True)
    manifest={"command":"all_cut_pilot_"+args.command,"args":vars(args),"contract":current,
              "input_sha256":{"checkpoint":current["checkpoint_sha256"],"data":data_hash},
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes(),"started_unix":started}
    common.write_json(output/"manifest.json",manifest)
    return output,records,model,tok,current,expected,manifest

def serialize_attempt(row):
    return {k:v for k,v in row.items() if k!="prefix_state"}

def group_summary(rows):
    k=sum(r["reward"] for r in rows);n=len(rows)
    return {"successes":k,"attempts":n,"mixed":0<k<n,"advantage_energy":n*k*(n-k)/(n-1)**2,
            "generated_tokens":sum(len(r["response_ids"]) for r in rows),
            "active_advantage_tokens":sum(len(r["response_ids"]) for r in rows) if 0<k<n else 0,
            "supplied_prefix_tokens_per_attempt":len(rows[0]["prefix_ids"])}

def fit(args):
    output,records,model,tok,current,expected,manifest=setup(args)
    groups,receipts,reused_seeds,reused_proxy=reuse_endpoints(args.reuse,expected,tok,current)
    missing=[key for key in expected if key not in groups]
    plan={"n_sources":len(records),"n_admissible_states":len(expected),"reused_states":len(groups),
          "new_states":len(missing),"new_attempts":8*len(missing),"reuse_receipts":receipts,
          "reused_original_full_run_flop_proxy":reused_proxy,"fit_seed":args.seed}
    common.write_json(output/"plan.json",plan)
    if args.plan_only:
        print(json.dumps(plan),flush=True);return
    ledger={};verification_start=time.monotonic()
    if groups:
        L.audit_behavior(model,tok,[r for group in groups.values() for r in group],temperature=args.temperature,
                         batch_size=args.microbatch,precision=args.precision,tolerance=args.max_logprob_error,ledger=ledger)
    expanded=[expected[key] for key in missing for _ in range(8)]
    print(json.dumps({"stage":"sampling_missing_cuts",**{k:plan[k] for k in ("n_sources","new_states","new_attempts","reused_states")}}),flush=True)
    generated=L.sample(model,tok,expanded,temperature=args.temperature,seed=args.seed,max_new_tokens=args.max_new_tokens,
                       batch_size=args.batch_size,precision=args.precision,ledger=ledger) if expanded else []
    if generated:
        L.audit_behavior(model,tok,generated,temperature=args.temperature,batch_size=args.microbatch,precision=args.precision,
                         tolerance=args.max_logprob_error,ledger=ledger)
    for index,key in enumerate(missing):
        rows=generated[index*8:(index+1)*8]
        for attempt,row in enumerate(rows):
            row.update(E.judge(row,row["response_ids"],tok));row["pilot_origin"]={"kind":"fresh_fit","seed":args.seed,"attempt_index":attempt}
        groups[key]=rows
    sources={};group_rows=[]
    for record in records:
        sid=record["source_id"];cuts=L.admissible_cuts(record)
        successes={cut:sum(row["reward"] for row in groups[(sid,cut)]) for cut in cuts}
        decision=choose_cut(successes)
        sources[sid]={**decision,"source_lines":record["n_lines"],"family":record["family"],"context_id":record["context_id"],
                      "admissible_cuts":cuts,"chosen_state_sha256":state_hash(expected[(sid,decision["cut"])])}
        for cut in cuts:group_rows.append({"source_id":sid,"context_id":record["context_id"],"family":record["family"],"reset_index":cut,**group_summary(groups[(sid,cut)])})
    ledger["runtime_measurement"]={"fit_sampling_and_checks_seconds":time.monotonic()-verification_start}
    ledger["verification"]={"attempt_reward_checks":8*len(expected),"admissible_states_reconstructed":len(expected)}
    frozen={"status":"frozen_initial_cuts","rule":RULE,**current,"sources":sources,"fit_seed":args.seed,
            "reused_producer_seeds":reused_seeds,"counters_initialized_to_zero":True,
            "candidate_data_sha256":manifest["input_sha256"]["data"],"reuse_receipts":receipts,
            "fit_compute_ledger":ledger,"reused_original_full_run_flop_proxy":reused_proxy,
            "new_pilot_flop_proxy":L.total_flop_proxy(ledger),"total_pilot_flop_proxy":L.total_flop_proxy(ledger)+reused_proxy,
            "frozen_before_fresh_audit":True,"confirmation_data_used":False}
    common.write_json(output/"initial_cuts.json",frozen)
    common.write_jsonl(output/"fit_groups.jsonl",group_rows)
    common.write_jsonl(output/"fit_attempts.jsonl",[serialize_attempt(row) for key in expected for row in groups[key]])
    summary={"initial_cuts_sha256":common.sha(output/"initial_cuts.json"),"n_sources":len(sources),
             "cut_histogram":dict(Counter(r["cut"] for r in sources.values())),"decision_reasons":dict(Counter(r["reason"] for r in sources.values())),
             "total_pilot_flop_proxy":frozen["total_pilot_flop_proxy"],"elapsed_seconds":time.time()-manifest["started_unix"]}
    common.write_json(output/"result.json",summary);common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json"),"initial_cuts_sha256":summary["initial_cuts_sha256"]})
    print(json.dumps(summary),flush=True)

def audit(args):
    output,records,model,tok,current,expected,manifest=setup(args)
    map_path=Path(args.initial_cuts).resolve();value=json.loads(map_path.read_text())
    cuts=validate_map(value,records,tok,current["checkpoint_sha256"],args.temperature,args.max_new_tokens,args.precision,8)
    if value.get("sampler_sha256")!=current["sampler_sha256"] or value.get("environment_sha256")!=current["environment_sha256"]:
        raise ValueError("Audit sampler/state code differs from fit")
    audit_seeds=[args.seed+i*1000003 for i in range(3)]
    forbidden={value["fit_seed"],*value.get("reused_producer_seeds",[])}
    if any(seed in forbidden for seed in audit_seeds):raise ValueError("Audit draws must use fresh seeds independent of all fit/reused draws")
    common.write_json(output/"map_binding.json",{"path":str(map_path),"sha256":common.sha(map_path),"audit_seeds":audit_seeds})
    if args.plan_only:
        common.write_json(output/"plan.json",{"n_sources":len(records),"attempts":3*8*len(records),"audit_seeds":audit_seeds});return
    results={};all_ledger={};all_groups=[]
    for condition,seed in zip(("chosen","full","latest"),audit_seeds):
        episodes=[]
        for r in records:
            cut=cuts[r["source_id"]] if condition=="chosen" else 0 if condition=="full" else max(L.admissible_cuts(r))
            episodes.extend([expected[(r["source_id"],cut)]]*8)
        ledger={};rows=L.sample(model,tok,episodes,temperature=args.temperature,seed=seed,max_new_tokens=args.max_new_tokens,
                              batch_size=args.batch_size,precision=args.precision,ledger=ledger)
        likelihood=L.audit_behavior(model,tok,rows,temperature=args.temperature,batch_size=args.microbatch,precision=args.precision,
                                    tolerance=args.max_logprob_error,ledger=ledger)
        condition_groups=[]
        for index,record in enumerate(records):
            group=rows[index*8:(index+1)*8]
            for attempt,row in enumerate(group):
                row.update(E.judge(row,row["response_ids"],tok));row.update(condition=condition,attempt_index=attempt,sampling_seed=seed)
            condition_groups.append({"condition":condition,"source_id":record["source_id"],"context_id":record["context_id"],
                                     "family":record["family"],"reset_index":group[0]["reset_index"],**group_summary(group)})
        common.write_jsonl(output/f"{condition}_attempts.jsonl",[serialize_attempt(r) for r in rows])
        all_groups.extend(condition_groups);all_ledger[condition]=ledger
        results[condition]={"n_sources":len(records),"mixed_groups":sum(g["mixed"] for g in condition_groups),
                            "successes":sum(g["successes"] for g in condition_groups),"attempts":len(rows),
                            "advantage_energy":sum(g["advantage_energy"] for g in condition_groups),
                            "active_advantage_tokens":sum(g["active_advantage_tokens"] for g in condition_groups),
                            "total_flop_proxy":L.total_flop_proxy(ledger),"likelihood_audit":likelihood,"seed":seed}
        ledger["verification"]={"attempt_reward_checks":len(rows)}
        print(json.dumps({"condition":condition,**results[condition]}),flush=True)
    common.write_jsonl(output/"audit_groups.jsonl",all_groups)
    result={"initial_cuts_sha256":common.sha(map_path),"conditions":results,"ledger":all_ledger,
            "fresh_audit_flop_proxy":L.total_flop_proxy(all_ledger),"fit_pilot_flop_proxy":value["total_pilot_flop_proxy"],
            "total_fit_plus_audit_flop_proxy":L.total_flop_proxy(all_ledger)+value["total_pilot_flop_proxy"],
            "policy_updates":0,"independent_of_fit_draws":True,"elapsed_seconds":time.time()-manifest["started_unix"]}
    common.write_json(output/"result.json",result);common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json")})

def parser():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    for name in ("fit","audit"):
        s=sub.add_parser(name)
        for flag in ("checkpoint","data","output"):s.add_argument("--"+flag,required=True)
        s.add_argument("--seed",type=int,required=True);s.add_argument("--temperature",type=float,default=.8)
        s.add_argument("--max-new-tokens",type=int,default=512);s.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
        s.add_argument("--device",default="cpu");s.add_argument("--batch-size",type=int,default=32);s.add_argument("--microbatch",type=int,default=16)
        s.add_argument("--max-logprob-error",type=float,default=.001);s.add_argument("--cpu-threads",type=int,default=1)
        s.add_argument("--plan-only",action="store_true")
        if name=="fit":s.add_argument("--reuse",nargs="*",default=[])
        else:s.add_argument("--initial-cuts",required=True)
    return p

if __name__=="__main__":
    args=parser().parse_args();globals()[args.command](args)
