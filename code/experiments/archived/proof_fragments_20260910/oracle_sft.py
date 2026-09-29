"""Explicit cap12 oracle-information SFT control; never a cap6 training arm.

Uses the candidate and selection reference certificates already acquired by
the selector, retaining their original population roles and identities. It
never evaluates selection loss as held-out evidence and cannot open either
confirmation role as a training input. Existing cap6 gates are unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
import common
import learning as L
from nd_environment import FORMAT, episode
from nd_model import save_checkpoint
from release_populations import check_record

CONTROL = "cap12_oracle_information_full_proof_sft_v1"

def budget_integer(value):
    d=Decimal(str(value))
    if not d.is_finite() or d<=0 or d!=d.to_integral_value():
        raise ValueError("FLOP proxy budget must be a positive integer")
    return int(d)

def step_proxy(model,batch_size,fixed_pad):
    p=sum(x.numel() for x in model.parameters()); c=model.config; s=fixed_pad-1
    return 6*p*batch_size*s+12*c.n_layers*batch_size*s*s*c.d_model

def budget_plan(budget,per_step,rounding):
    budget=budget_integer(budget)
    steps,remainder=divmod(budget,per_step)
    if not steps:raise ValueError("Budget is below one complete padded training update")
    if rounding=="exact" and remainder:
        raise ValueError(f"Exact budget must be divisible by one update ({per_step}); remainder={remainder}. Use an explicitly declared floor policy or a divisible budget.")
    if rounding not in ("exact","floor"):raise ValueError("Unknown budget rounding policy")
    return {"requested_flop_proxy":budget,"one_step_flop_proxy":per_step,"derived_steps":steps,
            "scheduled_training_flop_proxy":steps*per_step,"unused_flop_proxy":remainder,
            "budget_rounding":rounding,"overshoot":0}

def bind_population(path,expected_role):
    path=Path(path).resolve()
    if any(any(tag in part.lower() for tag in ("confirmation","textbook","sealed")) for part in path.parts):
        raise ValueError("Oracle SFT cannot use confirmation, sealed, or textbook data")
    manifest_path=path.parent/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("status")!="released_complete_primary":
        raise ValueError("Oracle inputs must come from the audited primary release")
    digest=common.sha(path)
    if manifest["population_sha256"].get(expected_role)!=digest:
        raise ValueError(f"Input is not the released {expected_role} population")
    return path,digest,manifest_path,common.sha(manifest_path)

def validate_sources(candidates,selection,retention):
    populations={"candidate":candidates,"selection":selection,"retention":retention}
    seen_source,seen_theorem,seen_context={},{},{}
    for role,rows in populations.items():
        if not rows:raise ValueError(f"Empty oracle input role {role}")
        for row in rows:
            check_record(row,role,12 if role=="selection" else 6)
            if row.get("diagnostic_only"):
                raise ValueError("Diagnostic-only certificate cannot enter this oracle union")
            if row["source_id"] in seen_source or row["theorem_id"] in seen_theorem:
                raise ValueError("Duplicate certificate/theorem across oracle information inputs")
            context=row["context_id"]
            if context in seen_context and seen_context[context]!=role:
                raise ValueError("Oracle input roles share a canonical context")
            seen_source[row["source_id"]]=role;seen_theorem[row["theorem_id"]]=role;seen_context[context]=role
    return {role:{"n":len(rows),"source_length_histogram":dict(Counter(r["n_lines"] for r in rows)),
                  "unique_contexts":len({r["context_id"] for r in rows}),"source_cap":12 if role=="selection" else 6}
            for role,rows in populations.items()}

def encode_with_origin(rows,tok):
    return [{**episode(r,0,tok),"oracle_original_role":r["role"],"oracle_length_bin":r["length_bin"]} for r in rows]

def train(model,tok,primary,retention,*,steps,batch_size,fixed_pad,lr,seed,retention_fraction,
          precision,output,ledger,log_every=25,save_every=0):
    primary_rng=random.Random(seed);retention_rng=random.Random(seed+1000003)
    nr=round(batch_size*retention_fraction);np=batch_size-nr
    if not 0<nr<batch_size:raise ValueError("Oracle control requires nonempty primary and fixed retention shares")
    optimizer=L.optimizer_for(model,lr)
    model.eval();start=time.monotonic()
    role_counts=Counter();role_tokens=Counter();role_padded_slots=Counter();role_unique=defaultdict(set)
    role_length_counts=defaultdict(Counter);role_length_tokens=defaultdict(Counter)
    exposure_counts=Counter();primary_schedule=hashlib.sha256();retention_schedule=hashlib.sha256()
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    for step in range(1,steps+1):
        p_rows=[primary[primary_rng.randrange(len(primary))] for _ in range(np)]
        r_rows=[retention[retention_rng.randrange(len(retention))] for _ in range(nr)]
        rows=p_rows+r_rows
        primary_ids=[r["source_id"] for r in p_rows];retention_ids=[r["source_id"] for r in r_rows]
        primary_schedule.update(json.dumps(primary_ids).encode());retention_schedule.update(json.dumps(retention_ids).encode())
        for row in rows:
            role=row["oracle_original_role"];length=str(row["source_lines"]);tokens=len(row["response_ids"])
            role_counts[role]+=1;role_tokens[role]+=tokens;role_padded_slots[role]+=fixed_pad-1
            role_length_counts[role][length]+=1;role_length_tokens[role][length]+=tokens
            role_unique[role].add(row["source_id"]);exposure_counts[row["source_id"]]+=1
        batch=L.collate(rows,tok.pad_id,next(model.parameters()).device,fixed_pad)
        optimizer.zero_grad(set_to_none=True)
        chosen=L.logprobs(model,batch,precision=precision)
        loss=(-(chosen*batch["mask"]).sum(-1)/batch["mask"].sum(-1)).mean()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError("Nonfinite oracle-SFT gradient")
        optimizer.step();L.account(ledger,model,batch,"oracle_sft_training",True)
        with (output/"training_exposures.jsonl").open("a") as stream:
            stream.write(json.dumps({"step":step,"primary_source_ids":primary_ids,"retention_source_ids":retention_ids,
                                      "primary_original_roles":[r["oracle_original_role"] for r in p_rows]})+"\n")
        if step==1 or step%log_every==0 or step==steps:
            progress={"step":step,"loss_nats_per_token_proof_mean":float(loss.detach()),"grad_norm":float(norm),
                      "training_flop_proxy":L.total_flop_proxy(ledger),"elapsed_seconds":time.monotonic()-start}
            common.write_json(output/"progress.json",progress);print(json.dumps(progress),flush=True)
        if save_every and step%save_every==0:
            save_checkpoint(output/f"step_{step:06d}.pt",model,tok,format=FORMAT,control=CONTROL,
                            full_source_supervision_cap=12,step=step,selection_loss_is_in_sample=True)
    displacement=math.sqrt(sum(float(((p.detach()-initial[n]).float()**2).sum()) for n,p in model.named_parameters()))
    return {"steps":steps,"parameter_displacement_l2":displacement,"optimizer":"AdamW with common empty initial state",
            "sampling":"Uniform IID over all distinct candidate+selection union certificates; separate fixed retention share",
            "loss":"Mean of proof-level mean response-token NLL; zero direct context/padding loss",
            "retention_fraction":nr/batch_size,"primary_schedule_sha256":primary_schedule.hexdigest(),
            "retention_schedule_sha256":retention_schedule.hexdigest(),
            "role_exposure_counts":dict(role_counts),"role_response_tokens":dict(role_tokens),
            "role_padded_token_slots":dict(role_padded_slots),"role_unique_exposed_sources":{r:len(v) for r,v in role_unique.items()},
            "role_reference_length_exposures":{r:dict(v) for r,v in role_length_counts.items()},
            "role_reference_length_response_tokens":{r:dict(v) for r,v in role_length_tokens.items()},
            "source_exposure_counts":dict(exposure_counts),"training_elapsed_seconds":time.monotonic()-start}

def run(args):
    started=time.time()
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Oracle-control output must be empty")
    bound={role:bind_population(path,role) for role,path in
           (("candidate",args.candidates),("selection",args.selection),("retention",args.retention))}
    if len({str(v[2]) for v in bound.values()})!=1:raise ValueError("Oracle inputs must share the same frozen population release")
    selector_path=Path(args.selector_manifest).resolve()
    selector=json.loads(selector_path.read_text())
    if selector.get("args",{}).get("command")!="calibrate":raise ValueError("Expected the actual calibration input manifest")
    checkpoint_hash=common.sha(args.checkpoint)
    expected={"data":bound["candidate"][1],"selection":bound["selection"][1],"retention":bound["retention"][1],"checkpoint":checkpoint_hash}
    if any(selector.get("input_sha256",{}).get(k)!=v for k,v in expected.items()):
        raise ValueError("Oracle certificates, retention or base weights differ from the selector's acquired inputs")
    if args.retention_fraction!=selector["args"].get("retention_fraction",.5):
        raise ValueError("Oracle retention fraction differs from the common calibration retention share")
    populations={role:common.read_jsonl(values[0]) for role,values in bound.items()}
    source_statistics=validate_sources(populations["candidate"],populations["selection"],populations["retention"])
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tok,checkpoint_metadata=L.load_model(args.checkpoint,"cpu" if args.plan_only else args.device)
    if checkpoint_metadata.get("format")!=FORMAT:
        raise ValueError("Oracle control must start from the common adapted prefix-format checkpoint")
    primary=encode_with_origin(populations["candidate"]+populations["selection"],tok)
    retention=encode_with_origin(populations["retention"],tok)
    for role,rows in populations.items():
        source_statistics[role].update({
            "acquired_full_certificate_literal_tokens":sum(len(r["text"].split()) for r in rows),
            "acquired_full_certificate_relative_tokens":sum(len(tok.encode_record(r["text"])) for r in rows),
            "acquired_distinct_certificates":len(rows)})
    largest=max(len(r["prompt_ids"])+len(r["response_ids"]) for r in primary+retention)
    if not largest<=args.fixed_pad<=model.config.max_seq_len:
        raise ValueError(f"Fixed padding/context cannot fit all oracle certificates: required={largest}, configured={args.fixed_pad}")
    plan=budget_plan(args.flop_budget,step_proxy(model,args.batch_size,args.fixed_pad),args.budget_rounding)
    input_hashes={str(selector_path):common.sha(selector_path),str(Path(args.checkpoint).resolve()):checkpoint_hash}
    for path,digest,meta,meta_digest in bound.values():input_hashes[str(path)]=digest;input_hashes[str(meta)]=meta_digest
    manifest={"control":CONTROL,"status":"planned" if args.plan_only else "running","args":vars(args),
              "format":FORMAT,"full_source_supervision_cap":12,"training_start":"empty prefix only",
              "source_statistics":source_statistics,"max_encoded_length":largest,"budget_plan":plan,
              "input_sha256":input_hashes,"selector_input_binding":expected,
              "original_population_roles_preserved":True,"selection_loss_is_in_sample":True,
              "confirmation_payloads_opened":False,"textbook_payloads_opened":False,
              "uniform_union_role_probability":{r:len(populations[r])/len(primary) for r in ("candidate","selection")},
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes(),"started_unix":started,
              "eligible_evaluation_roles":["confirmation","final_confirmation"]}
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/"manifest.json",manifest)
    common.write_jsonl(output/"oracle_source_index.jsonl",[
        {"source_id":r["source_id"],"original_role":r["role"],"full_source_lines":r["n_lines"],
         "context_id":r["context_id"],"theorem_id":r["theorem_id"],"provenance":r["provenance"],
         "source_file_sha256":bound[r["role"]][1],"usage":"retention" if r["role"]=="retention" else "explicit_oracle_full_proof_supervision"}
        for rows in populations.values() for r in rows])
    if args.plan_only:
        common.write_json(output/"planned.json",plan);print(json.dumps(plan),flush=True);return
    ledger={"preparation":{"wall_seconds":time.time()-started,"verified_source_certificates":sum(map(len,populations.values()))}}
    result=train(model,tok,primary,retention,steps=plan["derived_steps"],batch_size=args.batch_size,fixed_pad=args.fixed_pad,
        lr=args.lr,seed=args.seed,retention_fraction=args.retention_fraction,precision=args.precision,output=output,
        ledger=ledger,log_every=args.log_every,save_every=args.save_every)
    if L.total_flop_proxy(ledger)!=plan["scheduled_training_flop_proxy"]:raise AssertionError("Actual fixed padded training proxy differs from plan")
    save_checkpoint(output/"final.pt",model,tok,format=FORMAT,control=CONTROL,full_source_supervision_cap=12,
                    source_checkpoint_sha256=checkpoint_hash,oracle_source_file_sha256={r:v[1] for r,v in bound.items()},
                    selection_loss_is_in_sample=True,eligible_evaluation_roles=["confirmation","final_confirmation"],
                    optimizer_initialization="empty common AdamW state")
    result.update({"control":CONTROL,"full_source_supervision_cap":12,"budget_plan":plan,"ledger":ledger,
                   "total_flop_proxy":L.total_flop_proxy(ledger),"final_sha256":common.sha(output/"final.pt"),
                   "selection_loss_is_in_sample":True,"selection_reference_losses_reported_as_heldout":False,
                   "evaluation_executed":False,"elapsed_seconds":time.time()-manifest["started_unix"]})
    common.write_json(output/"result.json",result)
    common.write_json(output/"completed.json",{"ok":True,"final_sha256":result["final_sha256"],"result_sha256":common.sha(output/"result.json"),"finished_unix":time.time()})
    print(json.dumps({"control":CONTROL,"steps":result["steps"],"total_flop_proxy":result["total_flop_proxy"],
                      "role_exposure_counts":result["role_exposure_counts"],"final_sha256":result["final_sha256"]}),flush=True)

def parser():
    p=argparse.ArgumentParser()
    for flag in ("checkpoint","candidates","selection","retention","selector-manifest","output","flop-budget"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--budget-rounding",choices=("exact","floor"),default="exact")
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--fixed-pad",type=int,default=1024)
    p.add_argument("--retention-fraction",type=float,default=.5);p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--seed",type=int,default=20260910);p.add_argument("--device",default="cpu")
    p.add_argument("--precision",choices=("fp32","bf16"),default="fp32");p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--log-every",type=int,default=25);p.add_argument("--save-every",type=int,default=0)
    p.add_argument("--plan-only",action="store_true")
    return p

if __name__=="__main__":run(parser().parse_args())
