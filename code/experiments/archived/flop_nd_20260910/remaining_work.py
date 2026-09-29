"""Run predeclared controls, online continuations, and sealed final evaluation.

Waits for the entire frozen primary pipeline. All choices depend on declared
conditions, endpoint identities, and update counts, never evaluation outcomes.
Each generated pipeline is frozen before launch through the existing GPU lease
manager. No cloud resources or external publication are used.
"""
from __future__ import annotations
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import common

ROOT=common.ROOT
PYTHON="/data/users/dmitry/simplex-research/.venv/bin/python"
SNAPSHOT=Path(__file__).resolve().parent
COORDINATOR=ROOT/"jobs"/SNAPSHOT.name
CONDITIONS=("relative_full","relative_reset","ppo_full","ppo_reset")
DEADLINE="2026-09-11T04:25:00Z"


def utc():return datetime.now(timezone.utc).isoformat()


def frozen_hashes(extra=()):
    paths=[*SNAPSHOT.glob("*.py"),ROOT/"configs/secondary_online_protocol_v1.json",
           *[ROOT/"data/v1"/name for name in ("sources.jsonl","targets.jsonl","retention.jsonl")],*extra]
    return {str(p.resolve()):common.sha(p) for p in paths}


def evaluate_stage(name,checkpoint,data,audit,*,tokens=512,sampled=False,release=None):
    command=[PYTHON,str(SNAPSHOT/"evaluate_lm.py"),"--checkpoint",str(checkpoint),"--data",str(data),
             "--audit",str(audit),"--output",str(ROOT/"results"/name),"--temperature","0.8" if sampled else "0",
             "--attempts","4" if sampled else "1","--max-new-tokens",str(tokens),"--batch-size","32",
             "--seed","202609101935","--device","cuda"]
    if release:command.extend(["--release",str(release)])
    return {"name":name,"command":command,"timeout_seconds":900,
            "input_sha256":frozen_hashes([data,audit]+([release] if release else [])+([checkpoint] if checkpoint.exists() else []))}


def launch_pipeline(name,stages,extra=()):
    pipeline=ROOT/"configs"/(name+"_pipeline.json")
    job=ROOT/"configs"/("job_"+name+".json")
    common.write_json(pipeline,{"output":str(ROOT/"jobs"/(name+"_pipeline")),"deadline_utc":DEADLINE,"stages":stages})
    spec={"name":name,"command":[PYTHON,str(SNAPSHOT/"run_pipeline.py"),"--config",str(pipeline)],
          "candidate_gpus":[3],"deadline_utc":DEADLINE,"queue_timeout_seconds":600,
          "timeout_seconds":min(9*3600,max(1,int(datetime.fromisoformat(DEADLINE.replace("Z","+00:00")).timestamp()-time.time()))),
          "input_sha256":frozen_hashes([pipeline,*extra])}
    common.write_json(job,spec)
    with (COORDINATOR/(name+"_launcher.log")).open("w",buffering=1) as log:
        child=subprocess.Popen([PYTHON,str(ROOT/"tools/launch.py"),"--spec",str(job)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        common.write_json(COORDINATOR/"active_child.json",{"pid":child.pid,"name":name,"started_utc":utc(),"job_spec_sha256":common.sha(job)})
        code=child.wait()
    if code:raise RuntimeError(f"Frozen follow-up pipeline {name} exited {code}")


def main_endpoints():
    points={"baseline":ROOT/"models/adapt_final.pt"};results={}
    for seed in (1,2):
        for condition in CONDITIONS:
            name=f"main_{condition}_seed{seed}_v1";directory=ROOT/"results"/name
            result=json.loads((directory/"result.json").read_text())
            checkpoint=directory/"final.pt"
            if common.sha(checkpoint)!=result["final_checkpoint_sha256"]:raise ValueError("Primary endpoint changed")
            if result["stop_reason"] not in ("flop_proxy_ceiling","flop_ceiling"):
                raise ValueError(f"Primary arm did not reach declared compute endpoint: {name}/{result['stop_reason']}")
            points[name]=checkpoint;results[name]=result
    return points,results


def controls(primary_results):
    stages=[];points={};configs=[];mapping={}
    for seed in (1,2):
        endpoints={condition:primary_results[f"main_{condition}_seed{seed}_v1"]["step"] for condition in CONDITIONS}
        maximum=max(endpoints.values());name=f"retention_seed{seed}_v1";output=ROOT/"results"/name
        config={"checkpoint":str(ROOT/"models/adapt_final.pt"),"retention":str(ROOT/"data/v1/retention.jsonl"),
                "output":str(output),"seed":20260919+seed,"device":"cuda","precision":"fp32","cpu_threads":4,
                "max_updates":maximum,"max_seconds":1800,"retention_batch":16,"epochs":4,"microbatch":16,
                "lr":5e-5,"weight_decay":0.,"retention_weight":.1,"grad_clip":1.,"checkpoint_every":500,
                "checkpoint_steps":sorted(set(endpoints.values())),"log_every":100}
        path=ROOT/"configs"/(name+".json");common.write_json(path,config);configs.append(path)
        stages.append({"name":name,"command":[PYTHON,str(SNAPSHOT/"review_retention_control.py"),"--config",str(path)],
                       "timeout_seconds":1900,"input_sha256":frozen_hashes([path,ROOT/"models/adapt_final.pt",ROOT/"data/v1/retention.jsonl"])})
        for condition,step in endpoints.items():
            label=f"retention_seed{seed}_step{step:06d}"
            points[label]=output/f"step_{step:06d}.pt"
            mapping[f"main_{condition}_seed{seed}_v1"]=label
    common.write_json(ROOT/"configs/retention_endpoint_mapping_v1.json",{"status":"frozen_from_main_update_counts","mapping":mapping,
        "note":"Exact main retention exposure schedules; controls are not compute matched."})
    launch_pipeline("retention_v1",stages,configs)
    for seed in (1,2):
        result=json.loads((ROOT/f"results/retention_seed{seed}_v1/result.json").read_text())
        expected=max(primary_results[f"main_{c}_seed{seed}_v1"]["step"] for c in CONDITIONS)
        if result["step"]!=expected:raise ValueError("Retention control failed to reach exact exposure endpoint")
    for label,path in points.items():
        if not path.exists():raise ValueError(f"Missing exact retention checkpoint: {label}")
    return points


def online_continuations(primary_points):
    protocol=json.loads((ROOT/"configs/secondary_online_protocol_v1.json").read_text())
    stages=[];points={};inputs=[]
    for seed in (1,2):
        base_name=f"main_ppo_reset_seed{seed}_v1";checkpoint=primary_points[base_name]
        bank=ROOT/"configs"/f"initial_bank_primary_ppo_reset_seed{seed}_v1.json"
        with (COORDINATOR/f"initial_bank_seed{seed}.log").open("w") as log:
            subprocess.run([PYTHON,str(SNAPSHOT/"online_bank.py"),"--checkpoint",str(checkpoint),"--output",str(bank)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
        inputs.append(bank)
        order=(True,False) if seed==1 else (False,True)
        for enabled in order:
            name=f"online_{'on' if enabled else 'off'}_seed{seed}_v1"
            config=json.loads((ROOT/"configs"/(base_name+".json")).read_text())
            config.update(protocol["shared_settings"])
            config.update(checkpoint=str(checkpoint),output=str(ROOT/"results"/name),seed=protocol["continuation_seeds"][seed-1],
                          max_flop_proxy=protocol["budget"]["per_arm_flop_proxy"],max_seconds=protocol["budget"]["max_seconds"],
                          max_updates=protocol["budget"]["max_updates"],max_generated_tokens=protocol["budget"]["max_generated_tokens"],
                          online_bank=enabled,initial_bank=str(bank),resume=None)
            path=ROOT/"configs"/(name+".json");common.write_json(path,config);inputs.append(path)
            stages.append({"name":name,"command":[PYTHON,str(SNAPSHOT/"train_lm_online.py"),"--config",str(path)],
                           "timeout_seconds":1500,"input_sha256":frozen_hashes([path,checkpoint,bank])})
            points[name]=ROOT/"results"/name/"final.pt"
            for sampled in (False,True):
                suffix="sample4" if sampled else "greedy"
                stages.append(evaluate_stage(name+"_dev_"+suffix,points[name],ROOT/"data/v1/development.jsonl",
                                             ROOT/"data/v1/audit/development.jsonl",sampled=sampled))
    launch_pipeline("online_v1",stages,inputs)
    for name,path in points.items():
        result=json.loads((path.parent/"result.json").read_text())
        if result["stop_reason"] not in ("flop_proxy_ceiling","flop_ceiling"):
            raise ValueError(f"Online run did not reach declared compute endpoint: {name}")
    return points


def final_evaluation(points):
    stages=[];release_paths=[]
    for population,tokens in (("v1",512),("structured_lm_v1",1024)):
        data=ROOT/"data"/population/"final.jsonl";audit=data.parent/"audit/final.jsonl"
        release=ROOT/"configs"/("final_release_"+population+".json")
        common.write_json(release,{"status":"released","frozen_utc":utc(),
            "checkpoint_sha256":{name:common.sha(path) for name,path in points.items()},
            "checkpoint_paths":{name:str(path) for name,path in points.items()},
            "data_sha256":common.sha(data),"audit_sha256":common.sha(audit),
            "inference_profiles":[{"temperature":temp,"attempts":attempts,"max_new_tokens":tokens,"batch_size":32,
                                   "seed":202609101935,"precision":"fp32"} for temp,attempts in ((0,1),(.8,4))],
            "selection":"All predeclared primary, exact retention-exposure, and online continuation endpoints; no performance selection."})
        release_paths.append(release)
        for name,checkpoint in points.items():
            for sampled in (False,True):
                suffix="sample4" if sampled else "greedy"
                tag=name+("_final_" if population=="v1" else "_structured_final_")+suffix
                stages.append(evaluate_stage(tag,checkpoint,data,audit,tokens=tokens,sampled=sampled,release=release))
    common.write_json(ROOT/"configs/all_lm_endpoints_v1.json",{"status":"frozen","checkpoint_paths":{k:str(v) for k,v in points.items()},
                      "checkpoint_sha256":{k:common.sha(v) for k,v in points.items()},"frozen_utc":utc()})
    launch_pipeline("final_lm_v1",stages,release_paths)


def main():
    out=COORDINATOR;out.mkdir(parents=True,exist_ok=False)
    event={"status":"waiting_for_primary","pid":os.getpid(),"started_utc":utc(),
           "code_sha256":common.sha(__file__),"protocol_sha256":common.sha(ROOT/"configs/secondary_online_protocol_v1.json")}
    common.write_json(out/"progress.json",event)
    end=datetime.fromisoformat(DEADLINE.replace("Z","+00:00")).timestamp()
    try:
        marker=ROOT/"jobs/main_v1/completed.json"
        while not marker.exists():
            if time.time()>end-3600:raise TimeoutError("Insufficient time for declared remaining stages")
            time.sleep(10)
        primary_status=json.loads(marker.read_text())
        if primary_status["status"]!="completed":raise RuntimeError("Primary pipeline did not complete normally")
        primary,results=main_endpoints()
        event.update(status="retention_controls",primary_completed_utc=utc());common.write_json(out/"progress.json",event)
        failures=[];retention={};online={}
        try:
            if end-time.time()<7200:raise TimeoutError("Control launch omitted to preserve final-evaluation reserve")
            retention=controls(results)
        except Exception as error:
            failures.append({"stage":"retention_controls","error":type(error).__name__,"message":str(error),"utc":utc()})
            common.write_json(out/"followup_failures.json",failures)
        event.update(status="online_continuations",retention_completed_utc=utc());common.write_json(out/"progress.json",event)
        try:
            if end-time.time()<5400:raise TimeoutError("Online launch omitted to preserve final-evaluation reserve")
            online=online_continuations(primary)
        except Exception as error:
            failures.append({"stage":"online_continuations","error":type(error).__name__,"message":str(error),"utc":utc()})
            common.write_json(out/"followup_failures.json",failures)
        event.update(status="final_evaluation",online_completed_utc=utc());common.write_json(out/"progress.json",event)
        final_evaluation({**primary,**retention,**online})
        event.update(status="completed_with_followup_failures" if failures else "completed",followup_failures=failures,finished_utc=utc())
    except BaseException as error:
        event.update(status="failed",error=type(error).__name__,message=str(error),failed_utc=utc())
        raise
    finally:
        common.write_json(out/"result.json",event)
        common.write_json(out/"progress.json",event)


if __name__=="__main__":main()
