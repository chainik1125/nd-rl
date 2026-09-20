"""Retention-only trajectory with separate exposure and compute endpoints."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import torch
import common
import learning as L
from nd_environment import FORMAT
from nd_model import save_checkpoint
from oracle_sft import bind_population,budget_integer
from release_populations import check_record

CONTROL="cap6_retention_only_v1"


def validate_records(records,expected_count=512):
    if len(records)!=expected_count:raise ValueError(f"Expected exactly {expected_count} frozen retention sources")
    for row in records:check_record(row,"retention",6)
    for key in ("source_id","theorem_id"):
        if len({r[key] for r in records})!=len(records):raise ValueError("Duplicate retention certificate/theorem")


def validate_main_binding(args,main,checkpoint_hash,retention_hash):
    original=main.get("args",{})
    if original.get("command")!="rl":raise ValueError("Retention control needs a main RL input manifest")
    if main.get("input_sha256",{}).get("checkpoint")!=checkpoint_hash or main.get("input_sha256",{}).get("retention")!=retention_hash:
        raise ValueError("Control must share the main adapted checkpoint and retention population")
    expected={"seed":args.seed,"lr":args.lr,"retention_batch":args.batch_size,"retention_weight":args.retention_weight,"precision":args.precision}
    if any(original.get(k)!=v for k,v in expected.items()):raise ValueError("Retention schedule/optimizer settings differ from bound main RL run")
    for name,path in (("learning.py",L.__file__),):
        matches=[v for p,v in main.get("code_sha256",{}).items() if Path(p).name==name]
        if matches!=[common.sha(path)]:raise ValueError("Policy-update implementation differs from main RL")
    if main.get("dependency_sha256")!=common.dependency_hashes():raise ValueError("Pinned optimizer/model dependencies differ from main RL")


def draw_batch(rows,rng,batch_size):
    return [rows[rng.randrange(len(rows))] for _ in range(batch_size)]


def step_proxy(model,rows):
    p=sum(x.numel() for x in model.parameters());c=model.config;b=len(rows)
    s=max(len(r["prompt_ids"])+len(r["response_ids"])-1 for r in rows)
    return 6*p*b*s+12*c.n_layers*b*s*s*c.d_model


def retention_update(model,tok,rows,optimizer,*,retention_weight=.1,precision="fp32",ledger=None):
    # The unchanged main update function skips its policy loop for an empty
    # rollout list, then applies exactly its retained-proof loss and optimizer.
    return L.policy_update(model,tok,[],optimizer,rows,temperature=.8,max_new_tokens=512,
                           retention_weight=retention_weight,precision=precision,ledger=ledger)


def schedule_plan(model,rows,*,seed,batch_size,steps,max_flop_proxy,save_at_steps):
    if steps<1 or batch_size<1:raise ValueError("Positive step and batch limits are required")
    if any(type(s) is not int or s<1 for s in save_at_steps):raise ValueError("Exposure checkpoints need positive integer update counts")
    requested=set(save_at_steps);rng=random.Random(seed+1000003);schedule=hashlib.sha256()
    cumulative=0;tokens=0;endpoints={};step=0
    while step<steps and cumulative<max_flop_proxy:
        step+=1;retained=draw_batch(rows,rng,batch_size)
        schedule.update(json.dumps([r["source_id"] for r in retained]).encode())
        cumulative+=step_proxy(model,retained);tokens+=sum(len(r["response_ids"]) for r in retained)
        if step in requested:endpoints[str(step)]={"step":step,"retention_examples":step*batch_size,"retention_response_tokens":tokens,
            "retention_schedule_sha256":schedule.hexdigest(),"actual_flop_proxy":cumulative,"endpoint_kind":"retention_exposure_matched_only",
            "compute_matched_to_main":False}
    if requested-set(map(int,endpoints)):raise ValueError("Requested exposure checkpoint exceeds the first stopping ceiling")
    bound=step_proxy(model,[max(rows,key=lambda r:len(r["prompt_ids"])+len(r["response_ids"]))]*batch_size)
    stop="flop_proxy_ceiling" if cumulative>=max_flop_proxy else "step_ceiling"
    return {"steps":step,"stop_reason":stop,"actual_flop_proxy":cumulative,"max_flop_proxy":max_flop_proxy,
            "flop_proxy_overshoot":max(0,cumulative-max_flop_proxy),"worst_case_one_step_flop_proxy":bound,
            "retention_schedule_sha256":schedule.hexdigest(),"retention_examples":step*batch_size,
            "retention_response_tokens":tokens,"exposure_endpoints":endpoints,
            "final_endpoint_kind":"declared_total_compute_budget_endpoint" if stop=="flop_proxy_ceiling" else "step_ceiling_only"}


def train(model,tok,rows,*,plan,seed,batch_size,lr,retention_weight,precision,output,ledger,checkpoint_metadata,log_every=100):
    optimizer=L.optimizer_for(model,lr);model.eval()
    rng=random.Random(seed+1000003);schedule=hashlib.sha256();tokens=0;counts=Counter();saved=[];start=time.monotonic()
    for step in range(1,plan["steps"]+1):
        if L.total_flop_proxy(ledger)>=plan["max_flop_proxy"]:raise AssertionError("Training passed the completed-step compute stop")
        retained=draw_batch(rows,rng,batch_size);ids=[r["source_id"] for r in retained]
        schedule.update(json.dumps(ids).encode());tokens+=sum(len(r["response_ids"]) for r in retained);counts.update(ids)
        before=L.total_flop_proxy(ledger)
        update=retention_update(model,tok,retained,optimizer,retention_weight=retention_weight,precision=precision,ledger=ledger)
        cumulative=L.total_flop_proxy(ledger);actual_step=cumulative-before
        if actual_step!=step_proxy(model,retained) or actual_step>plan["worst_case_one_step_flop_proxy"]:
            raise AssertionError("Actual retention dense proxy differs from schedule or exceeds its bound")
        event={**update,"step":step,"retention_source_ids":ids,"retention_examples":len(retained),
               "retention_response_tokens":sum(len(r["response_ids"]) for r in retained),"step_flop_proxy":actual_step,
               "cumulative_flop_proxy":cumulative,"weighted_retention_loss":retention_weight*update["retention_loss"]}
        with (output/"updates.jsonl").open("a") as stream:stream.write(json.dumps(event)+"\n")
        if step==1 or step%log_every==0 or step==plan["steps"]:
            progress={k:v for k,v in event.items() if k!="retention_source_ids"}
            progress["elapsed_seconds"]=time.monotonic()-start
            common.write_json(output/"progress.json",progress);print(json.dumps(progress),flush=True)
        if str(step) in plan["exposure_endpoints"]:
            endpoint=plan["exposure_endpoints"][str(step)]
            if endpoint["retention_schedule_sha256"]!=schedule.hexdigest():raise AssertionError("Exposure schedule differs from plan")
            path=output/f"step_{step:06d}.pt"
            save_checkpoint(path,model,tok,**checkpoint_metadata,**endpoint)
            receipt={**endpoint,"checkpoint":str(path),"checkpoint_sha256":common.sha(path)}
            common.write_json(output/f"step_{step:06d}.json",receipt);saved.append(receipt)
    if any((L.total_flop_proxy(ledger)!=plan["actual_flop_proxy"],schedule.hexdigest()!=plan["retention_schedule_sha256"],tokens!=plan["retention_response_tokens"])):
        raise AssertionError("Actual retention trajectory exposure/compute differs from the presampled schedule")
    return {"steps":plan["steps"],"retention_schedule_sha256":schedule.hexdigest(),"retention_response_tokens":tokens,
            "retention_examples":plan["steps"]*batch_size,"unique_exposed_sources":len(counts),"source_exposure_counts":dict(counts),
            "exposure_checkpoints":saved,"training_elapsed_seconds":time.monotonic()-start}


def run(args):
    started=time.time();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Retention-only output must be empty")
    if args.retention_weight<=0:raise ValueError("Retention-only loss weight must be positive")
    bound=bind_population(args.retention,"retention")
    main_path=Path(args.main_manifest).resolve();main=json.loads(main_path.read_text());checkpoint_hash=common.sha(args.checkpoint)
    validate_main_binding(args,main,checkpoint_hash,bound[1])
    records=common.read_jsonl(bound[0]);validate_records(records)
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tok,metadata=L.load_model(args.checkpoint,"cpu" if args.plan_only else args.device)
    if metadata.get("format")!=FORMAT:raise ValueError("Control must begin from the common adapted checkpoint")
    rows=L.encode_population(records,tok,"empty")
    if max(len(r["prompt_ids"])+len(r["response_ids"]) for r in rows)>model.config.max_seq_len:
        raise ValueError("Retention reference exceeds model context; no truncation allowed")
    plan=schedule_plan(model,rows,seed=args.seed,batch_size=args.batch_size,steps=args.steps,
                       max_flop_proxy=budget_integer(args.max_flop_proxy),save_at_steps=args.save_at_steps)
    main_ceiling=main["args"]["max_flop_proxy"]
    final_compute_matched=plan["stop_reason"]=="flop_proxy_ceiling" and plan["max_flop_proxy"]==main_ceiling
    manifest={"control":CONTROL,"args":vars(args),"format":FORMAT,"full_source_supervision_cap":6,
              "training_roles":["retention"],"n_sources":len(records),"policy_rollouts":0,"source_policy_loss":0,
              "loss":"retention_weight times mean of per-proof mean response-token NLL; zero direct supplied/padding loss",
              "retention_rng_seed":args.seed+1000003,"optimizer":"Identical empty-state main AdamW with norm1 clipping",
              "input_sha256":{"checkpoint":checkpoint_hash,"retention":bound[1],"population_manifest":bound[3],"main_manifest":common.sha(main_path)},
              "source_checkpoint_metadata":metadata,"budget_plan":plan,
              "main_training_flop_proxy_ceiling":main_ceiling,"final_compute_ceiling_matched_to_main":final_compute_matched,
              "compute_proxy_note":"Same declared dense projection/MLP and attention proxy as main RL; hardware FLOPs are not measured.",
              "matching_note":"Requested update checkpoints match retained examples and schedule, not main total compute. Only a compute-ceiling endpoint matches the declared total proxy budget.",
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes(),"started_unix":started}
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/"manifest.json",manifest)
    common.write_jsonl(output/"retention_source_index.jsonl",[{k:r[k] for k in ("source_id","theorem_id","context_id","role","n_lines","provenance")} for r in records])
    common.write_json(output/"plan.json",plan)
    if args.plan_only:print(json.dumps(plan),flush=True);return
    checkpoint_metadata={"format":FORMAT,"control":CONTROL,"full_source_supervision_cap":6,"training_roles":["retention"],
                         "source_checkpoint_sha256":checkpoint_hash,"retention_sha256":bound[1],"policy_rollouts":0,
                         "eligible_evaluation_roles":["confirmation","final_confirmation"]}
    ledger={"preparation":{"verified_source_certificates":len(records),"elapsed_seconds":time.time()-started}}
    result=train(model,tok,rows,plan=plan,seed=args.seed,batch_size=args.batch_size,lr=args.lr,retention_weight=args.retention_weight,
                 precision=args.precision,output=output,ledger=ledger,checkpoint_metadata=checkpoint_metadata,log_every=args.log_every)
    save_checkpoint(output/"final.pt",model,tok,**checkpoint_metadata,step=plan["steps"],endpoint_kind=plan["final_endpoint_kind"],
                    actual_flop_proxy=plan["actual_flop_proxy"],retention_schedule_sha256=result["retention_schedule_sha256"],
                    retention_examples=result["retention_examples"],retention_response_tokens=result["retention_response_tokens"],
                    compute_ceiling_matched_to_main=final_compute_matched,requested_flop_proxy=plan["max_flop_proxy"])
    result.update({"control":CONTROL,"policy_rollouts":0,"policy_updates_from_rollouts":0,"budget_plan":plan,
                   "stop_reason":plan["stop_reason"],"final_endpoint_kind":plan["final_endpoint_kind"],
                   "compute_ceiling_matched_to_main":final_compute_matched,"main_training_flop_proxy_ceiling":main_ceiling,
                   "ledger":ledger,"total_flop_proxy":L.total_flop_proxy(ledger),"final_sha256":common.sha(output/"final.pt"),
                   "flop_proxy_overshoot":plan["flop_proxy_overshoot"],"worst_case_one_step_flop_proxy":plan["worst_case_one_step_flop_proxy"],
                   "elapsed_seconds":time.time()-started})
    common.write_json(output/"result.json",result);common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json"),"final_sha256":result["final_sha256"]})


def parser():
    p=argparse.ArgumentParser()
    for flag in ("checkpoint","retention","main-manifest","output"):p.add_argument("--"+flag,required=True)
    p.add_argument("--save-at-steps",type=int,nargs="*",default=[]);p.add_argument("--steps",type=int,default=100000)
    p.add_argument("--max-flop-proxy",default="4e15");p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--seed",type=int,default=20260910);p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--retention-weight",type=float,default=.1);p.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--device",default="cpu");p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--log-every",type=int,default=100);p.add_argument("--plan-only",action="store_true")
    return p


if __name__=="__main__":run(parser().parse_args())
