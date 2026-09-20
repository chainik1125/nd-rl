"""Frozen 48-endpoint RL calibration scoring on released selection only."""
import argparse
import json
from pathlib import Path
import time
import torch
import common
import learning as L
import reference_scoring as R
from nd_environment import FORMAT
from release_populations import check_record
from score_only import summarize


def value(command,flag):
    try:return command[command.index("--"+flag)+1]
    except (ValueError,IndexError):raise ValueError(f"Missing declared training argument --{flag}")


def values(command, flag):
    start = command.index("--" + flag) + 1
    end = next((i for i in range(start, len(command)) if command[i].startswith("--")), len(command))
    return command[start:end]


def validate_pipeline(pipeline,design):
    if pipeline.get("status")!="frozen_ready" or design.get("status")!="frozen_before_main_heldout_predictions":
        raise ValueError("A prospectively frozen RL-calibration pipeline and design are required")
    short=design["short_proxy_ceiling"];long=design["long_proxy_ceiling"]
    if pipeline["short_training_proxy_ceiling"]!=short or pipeline["long_training_proxy_ceiling"]!=long:
        raise ValueError("Pipeline horizons differ from the frozen design")
    order=pipeline["arm_order"];commands=pipeline["commands"]
    if len(order)!=24 or len(commands)!=24 or len(set(tuple(x) for x in order))!=24:
        raise ValueError("Expected 24 distinct family/control arms")
    families={x[0] for x in order}
    if len(families)!=12 or any({kind for f,kind in order if f==family}!={"family","control"} for family in families):
        raise ValueError("Every one of 12 families needs both family and control arms")
    endpoints=[]
    for (family,arm),command in zip(order,commands):
        if "rl" not in command or value(command,"start")!="full":raise ValueError("Scoring design requires full-start RL interventions")
        if Path(command[1]).resolve() != (Path(design["training_snapshot"])/"fragment_experiment.py").resolve():
            raise ValueError("Training program differs from the frozen design snapshot")
        if Path(value(command,"data")).name != f"{family}_{arm}.jsonl":
            raise ValueError("Declared family/control identity differs from its source bank")
        if int(value(command,"seed"))!=design["training_seed"] or value(command,"checkpoint")!=design["checkpoint"]:
            raise ValueError("Training seed or common checkpoint differs from design")
        if float(value(command,"max-flop-proxy"))!=long:raise ValueError("Long RL ceiling differs")
        if [float(x) for x in values(command,"flop-milestones")] != [short,long]:
            raise ValueError("Both declared compute milestones must be saved")
        output=Path(value(command,"output")).resolve()
        for horizon,ceiling,path in (("short",short,output/f"flops_{short:.0f}.pt"),("long",long,output/"final.pt")):
            endpoints.append({"family":family,"arm":arm,"horizon":horizon,"training_proxy_ceiling":ceiling,
                              "checkpoint":str(path),"producer_directory":str(output),"command":command})
    declared=[str(Path(p).resolve()) for p in pipeline["checkpoints_for_scoring"]]
    if declared!=[r["checkpoint"] for r in endpoints] or len(set(declared))!=48:
        raise ValueError("The 48 scored paths must exactly equal the declared ordered short/long endpoints")
    return endpoints


def selection_metadata(path,digest,pipeline_inputs=None):
    metadata_path=Path(path).parent/"manifest.json";metadata=json.loads(metadata_path.read_text())
    if metadata.get("status")!="released_complete_primary" or metadata.get("population_sha256",{}).get("selection")!=digest:
        raise ValueError("Scorer accepts only the released selection population hash")
    if pipeline_inputs is not None and metadata_path.resolve() not in {Path(p).resolve() for p in pipeline_inputs}:
        raise ValueError("Released selection manifest is not a declared pipeline input")
    return metadata_path


def bind_producer(endpoint,base_hash):
    directory=Path(endpoint["producer_directory"]);manifest_path=directory/"manifest.json"
    manifest=json.loads(manifest_path.read_text());args=manifest["args"]
    completion=json.loads((directory/"completed.json").read_text());result_path=directory/"result.json";result=json.loads(result_path.read_text())
    if not completion.get("ok") or completion["result_sha256"]!=common.sha(result_path):raise ValueError("RL producer is incomplete or changed")
    if completion["final_sha256"]!=common.sha(directory/"final.pt") or result["final_sha256"]!=completion["final_sha256"]:
        raise ValueError("Completed producer checkpoint changed")
    if manifest["input_sha256"]["checkpoint"]!=base_hash:raise ValueError("RL endpoint did not begin from frozen adapted policy")
    for key in ("checkpoint","data","retention"):
        if common.sha(args[key])!=manifest["input_sha256"][key]:raise ValueError("Producer input identity changed")
    for path,digest in {**manifest["code_sha256"],**manifest["dependency_sha256"]}.items():
        if common.sha(path)!=digest:raise ValueError("Producer source/dependency changed")
    for flag in ("checkpoint","data","retention","output","steps","start","temperature","max-new-tokens","batch-size","microbatch",
                 "prompts-per-step","group-size","lr","retention-batch","retention-weight","empty-fraction","max-generated-tokens",
                 "max-flop-proxy","curriculum-rollouts","backward-threshold","forward-threshold","precision","seed","cpu-threads","device"):
        actual=args[flag.replace("-","_")];raw=value(endpoint["command"],flag)
        expected=type(actual)(raw)
        if actual!=expected:raise ValueError(f"Producer --{flag} differs from declared command")
    if args["flop_milestones"] != [float(x) for x in values(endpoint["command"],"flop-milestones")]:
        raise ValueError("Producer saved milestones differ from the declared command")
    updates=common.read_jsonl(directory/"updates.jsonl")
    if len(updates)!=result["steps"] or [u["step"] for u in updates]!=list(range(1,len(updates)+1)):
        raise ValueError("Producer update history incomplete")
    declared_budget = float(value(endpoint["command"], "max-flop-proxy"))
    step_bound = result["worst_case_one_step_flop_proxy"]
    running_total = 0
    for update in updates:
        cost = update["step_flop_proxy"]
        if not 0 < cost <= step_bound:
            raise ValueError("Producer step cost is nonpositive or exceeds its bound")
        running_total += cost
        if update["cumulative_flop_proxy"] != running_total:
            raise ValueError("Producer cumulative proxy differs from the sum of step costs")
    if not updates or running_total != result["total_flop_proxy"] or L.total_flop_proxy(result["ledger"]) != running_total:
        raise ValueError("Producer total proxy differs from its completed history or ledger")
    if (result["max_flop_proxy"] != declared_budget or result["stop_reason"] != "flop_proxy_ceiling"
            or running_total < declared_budget or running_total - updates[-1]["step_flop_proxy"] >= declared_budget
            or result["flop_proxy_overshoot"] != running_total - declared_budget):
        raise ValueError("Completed producer does not reflect the declared budget and first crossing")
    ceiling=endpoint["training_proxy_ceiling"]
    first=next((u for u in updates if u["cumulative_flop_proxy"]>=ceiling),None)
    if first is None:raise ValueError("Declared horizon was not reached")
    if endpoint["horizon"]=="long" and first["step"]!=result["steps"]:raise ValueError("Final checkpoint is not the first completed long-budget crossing")
    if first["cumulative_flop_proxy"]-ceiling>result["worst_case_one_step_flop_proxy"]:
        raise ValueError("Training overshoot exceeds bound")
    return {"producer_manifest_sha256":common.sha(manifest_path),"producer_result_sha256":common.sha(result_path),
            "producer_updates_sha256":common.sha(directory/"updates.jsonl"),"source_bank_sha256":manifest["input_sha256"]["data"],
            "training_steps":first["step"],"actual_training_flop_proxy":first["cumulative_flop_proxy"],
            "training_proxy_history_reconciled":True,
            "training_flop_proxy_overshoot":first["cumulative_flop_proxy"]-ceiling,"training_one_step_proxy_bound":result["worst_case_one_step_flop_proxy"],
            "retention_examples_through_checkpoint":sum(u["retention_examples"] for u in updates[:first["step"]]),
            "retention_response_tokens_through_checkpoint":sum(u["retention_response_tokens"] for u in updates[:first["step"]]),
            "mixed_groups_through_checkpoint":sum(u["mixed_groups"] for u in updates[:first["step"]])}


def validate_checkpoint_metadata(endpoint,metadata,producer,base_hash):
    if metadata.get("format")!=FORMAT:raise ValueError("Endpoint has the wrong state format")
    if endpoint["horizon"]=="short":
        if (metadata.get("step")!=producer["training_steps"] or metadata.get("requested_flop_proxy")!=endpoint["training_proxy_ceiling"]
                or metadata.get("actual_flop_proxy")!=producer["actual_training_flop_proxy"]):
            raise ValueError("Short milestone metadata differs from actual frozen training history")
    elif metadata.get("source_checkpoint_sha256")!=base_hash or metadata.get("stage")!="rl":
        raise ValueError("Long endpoint is not the declared RL final checkpoint")


def enrich_scores(scores,records,checkpoint_hash):
    source={r["source_id"]:r for r in records}
    for row in scores:
        original=source[row["source_id"]]
        row.update({"theorem_id":original["theorem_id"],"context_id":original["context_id"],"length_bin":original["length_bin"],
                    "role":"selection","start":"full","checkpoint_sha256":checkpoint_hash,"score_temperature":1.})
    return scores


def run(args):
    started=time.time();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Selection scoring output must be empty")
    pipeline_path=Path(args.pipeline).resolve();design_path=Path(args.design).resolve()
    pipeline=json.loads(pipeline_path.read_text());design=json.loads(design_path.read_text())
    endpoints=validate_pipeline(pipeline,design)
    data=Path(args.selection).resolve();data_hash=common.sha(data);population_metadata=selection_metadata(data,data_hash,pipeline["inputs"])
    base_path=Path(args.base_checkpoint).resolve();base_hash=common.sha(base_path)
    if str(base_path)!=str(Path(design["checkpoint"]).resolve()):raise ValueError("Selection gradient/scoring base differs from frozen design")
    # Require all 48 endpoints and bind every checkpoint before proof payloads.
    checkpoint_hashes={e["checkpoint"]:common.sha(e["checkpoint"]) for e in endpoints}
    producers={}
    for endpoint in endpoints:
        key=(endpoint["producer_directory"],endpoint["horizon"])
        producers[key]=bind_producer(endpoint,base_hash)
    pipeline_inputs={str(Path(p).resolve()):common.sha(p) for p in pipeline["inputs"]}
    records=common.read_jsonl(data)
    if not records:raise ValueError("Empty selection population")
    for row in records:check_record(row,"selection",12)
    if len({r["source_id"] for r in records})!=len(records) or len({r["theorem_id"] for r in records})!=len(records):
        raise ValueError("Duplicate selection source/endpoint")
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,tok,base_metadata=L.load_model(base_path,args.device)
    if base_metadata.get("format")!=FORMAT:raise ValueError("Common adapted policy format is required")
    query=L.encode_population(records,tok,"empty")
    if any(len(r["prompt_ids"])+len(r["response_ids"])>base.config.max_seq_len for r in query):raise ValueError("Complete selection reference exceeds context budget")
    manifest={"command":"score_rl_selection","args":vars(args),"role":"selection","start":"full","score_temperature":1.,
              "n_declared_endpoints":48,"policy_updates":0,"generation_executed":False,
              "gradient_target":"Equal-length-stratum mean complete-reference bits at common adapted policy, T=1",
              "gradient_used_for_parameter_updates":False,"input_sha256":{str(pipeline_path):common.sha(pipeline_path),str(design_path):common.sha(design_path),
                  str(data):data_hash,str(population_metadata):common.sha(population_metadata),str(base_path):base_hash,**checkpoint_hashes},
              "pipeline_input_sha256":pipeline_inputs,"producer_bindings":[{k:v for k,v in e.items() if k!="command"}|producers[(e["producer_directory"],e["horizon"])] for e in endpoints],
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},"dependency_sha256":common.dependency_hashes(),"started_unix":started}
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/"manifest.json",manifest)
    ledger={};base_ledger={};base_scores=enrich_scores(R.score(base,tok,query,args.batch_size,base_ledger,args.precision,1.),records,base_hash)
    common.write_jsonl(output/"base_selection_nll.jsonl",base_scores);ledger["base_scoring"]=base_ledger
    base_result={"checkpoint_sha256":base_hash,**summarize(base_scores),"ledger":base_ledger,"total_flop_proxy":L.total_flop_proxy(base_ledger)}
    gradient=None;gradient_hash=None
    if args.actual_displacement:
        gradient_ledger={};gradient=L.query_gradient(base,tok,query,batch_size=args.gradient_batch_size,ledger=gradient_ledger,precision=args.precision)
        if any(p.grad is not None for p in base.parameters()):raise AssertionError("Query gradient left trainable model gradients attached")
        frozen={"gradient":{n:g.detach().cpu() for n,g in gradient.items()},"base_checkpoint_sha256":base_hash,"selection_sha256":data_hash,
                "score_temperature":1.,"weighting":"equal reference-length strata, whole-proof bits","ledger":gradient_ledger,
                "learning_sha256":common.sha(L.__file__),"gradient_used_for_updates":False}
        torch.save(frozen,output/"selection_gradient.pt");gradient_hash=common.sha(output/"selection_gradient.pt")
        common.write_json(output/"selection_gradient.json",{k:v for k,v in frozen.items() if k!="gradient"}|{"gradient_artifact_sha256":gradient_hash})
        ledger["single_frozen_query_gradient"]=gradient_ledger
    results=[]
    for index,endpoint in enumerate(endpoints):
        model,endpoint_tok,metadata=L.load_model(endpoint["checkpoint"],args.device)
        producer=producers[(endpoint["producer_directory"],endpoint["horizon"])]
        validate_checkpoint_metadata(endpoint,metadata,producer,base_hash)
        if endpoint_tok.state_dict()!=tok.state_dict() or model.config!=base.config:raise ValueError("Endpoint tokenizer/model architecture differs from base")
        arm_ledger={};scores=enrich_scores(R.score(model,tok,query,args.batch_size,arm_ledger,args.precision,1.),records,checkpoint_hashes[endpoint["checkpoint"]])
        name=f"{index:02d}_{endpoint['family']}_{endpoint['arm']}_{endpoint['horizon']}";directory=output/name;directory.mkdir()
        common.write_jsonl(directory/"selection_nll.jsonl",scores)
        result={k:v for k,v in endpoint.items() if k!="command"}|producer|summarize(scores)
        result.update(checkpoint_sha256=checkpoint_hashes[endpoint["checkpoint"]],score_temperature=1.,scores_directory=name,
                      absolute_bits_gain_from_base=base_result["length_balanced_mean_bits"]-result["length_balanced_mean_bits"])
        if gradient is not None:
            result["selection_gradient_dot_actual_displacement"]=L.displacement_dot(model,base,gradient)
            original=dict(base.named_parameters())
            result["parameter_displacement_l2"]=sum(float(((p.detach()-original[n].detach()).float()**2).sum()) for n,p in model.named_parameters())**.5
            parameters=sum(p.numel() for p in model.parameters())
            arm_ledger["displacement_parameter_arithmetic"]={"parameters":parameters,"flop_proxy":6*parameters,
                "definition":"Approximate subtract/multiply/reduce work for dot and displacement squared norm; no model forward or optimizer update"}
            result["frozen_gradient_artifact_sha256"]=gradient_hash
        result.update(ledger=arm_ledger,total_flop_proxy=L.total_flop_proxy(arm_ledger))
        common.write_json(directory/"result.json",result);results.append(result);ledger[name]=arm_ledger
        common.write_json(output/"progress.json",{"completed_endpoints":len(results),"declared_endpoints":48,"latest":name})
        print(json.dumps({"completed":len(results),"total":48,"endpoint":name,"length_balanced_mean_bits":result["length_balanced_mean_bits"]}),flush=True)
        del model
    result={"role":"selection","score_temperature":1.,"base":base_result,"checkpoints":results,"n_endpoints":len(results),
            "ledger":ledger,"total_flop_proxy":L.total_flop_proxy(ledger),"frozen_gradient_sha256":gradient_hash,
            "policy_updates":0,"generation_executed":False,"ranking_performed":False,"final_data_accessed":False,"elapsed_seconds":time.time()-started}
    common.write_json(output/"result.json",result);common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json")})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("pipeline","design","selection","base-checkpoint","output"):p.add_argument("--"+flag,required=True)
    p.add_argument("--actual-displacement",action="store_true");p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--gradient-batch-size",type=int,default=16);p.add_argument("--device",default="cpu")
    p.add_argument("--precision",choices=("fp32","bf16"),default="fp32");p.add_argument("--cpu-threads",type=int,default=1)
    run(p.parse_args())
