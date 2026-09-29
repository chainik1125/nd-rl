"""Released confirmation reference-loss scoring; no sampling or updates."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time
import torch
import common
import learning as L
import reference_scoring as R
from nd_environment import FORMAT

ALLOWED_ROLES=("confirmation","final_confirmation")

def validate_release(release,role,data_hash,checkpoint_hashes):
    if role not in ALLOWED_ROLES:raise ValueError("Score-only confirmation evaluator rejects other roles")
    if release.get("frozen") is not True or release.get("no_further_selector_changes") is not True:
        raise ValueError("A frozen release closing selector changes is required")
    if role=="final_confirmation" and release.get("no_further_training") is not True:
        raise ValueError("Final confirmation requires closed training endpoints")
    if data_hash not in release.get("data_sha256",[]):raise ValueError("Confirmation data hash is not released")
    if not checkpoint_hashes or any(h not in release.get("checkpoints",[]) for h in checkpoint_hashes):
        raise ValueError("Every checkpoint hash must be in root's release")

def data_role(path,data_hash):
    metadata_path=Path(path).parent/"manifest.json"
    metadata=json.loads(metadata_path.read_text())
    matches=[r for r in ALLOWED_ROLES if metadata.get("population_sha256",{}).get(r)==data_hash]
    if len(matches)!=1:raise ValueError("Data must match one confirmation role in frozen population metadata")
    return matches[0],metadata_path

def summarize(scores):
    by_length=defaultdict(list);by_family=defaultdict(list);by_family_length=defaultdict(list)
    for row in scores:
        by_length[row["length_bin"]].append(row["bits"])
        by_family[row["family"]].append(row["bits"])
        by_family_length[(row["family"],row["length_bin"])].append(row["bits"])
    def mean(values):return sum(values)/len(values)
    lengths={k:{"n":len(v),"mean_bits":mean(v)} for k,v in by_length.items()}
    return {"n_theorems":len(scores),"mean_bits":mean([r["bits"] for r in scores]),
            "length_balanced_mean_bits":mean([x["mean_bits"] for x in lengths.values()]),
            "aggregate_bits_per_token":sum(r["bits"] for r in scores)/sum(r["tokens"] for r in scores),
            "by_length":lengths,"by_family":{k:{"n":len(v),"mean_bits":mean(v)} for k,v in by_family.items()},
            "by_family_length":[{"family":f,"length_bin":b,"n":len(v),"mean_bits":mean(v)} for (f,b),v in sorted(by_family_length.items())]}

def run(args):
    started=time.time();output=Path(args.output).resolve()
    R.validate_temperature(args.score_temperature)
    if output.exists() and any(output.iterdir()):raise ValueError("Score-only output must be empty")
    release_path=Path(args.release).resolve();release=json.loads(release_path.read_text())
    data=Path(args.data).resolve();data_hash=common.sha(data)
    role,population_metadata=data_role(data,data_hash)
    checkpoints=[Path(p).resolve() for p in args.checkpoints]
    if len(checkpoints)!=len(set(checkpoints)):raise ValueError("Repeated checkpoint path")
    hashes={str(p):common.sha(p) for p in checkpoints}
    validate_release(release,role,data_hash,list(hashes.values()))
    # No proof payload has been parsed before all role/data/checkpoint gates.
    records=common.read_jsonl(data)
    if not records or any(r.get("role")!=role for r in records):raise ValueError("Released payload's role differs")
    if len({r["source_id"] for r in records})!=len(records):raise ValueError("Duplicate confirmation source")
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest={"command":"score_only_confirmation","args":vars(args),"role":role,"start":"full","score_temperature":args.score_temperature,
              "generation_executed":False,"optimizer_updates":0,"input_sha256":{
                  str(data):data_hash,str(release_path):common.sha(release_path),str(population_metadata):common.sha(population_metadata),**hashes},
              "format":FORMAT,"started_unix":started,
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes()}
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/"manifest.json",manifest)
    by_source={r["source_id"]:r for r in records};results=[];total_ledger={}
    for index,path in enumerate(checkpoints):
        model,tok,metadata=L.load_model(path,args.device)
        if metadata.get("format")!=FORMAT:raise ValueError("Checkpoint lacks the common adapted input format")
        if metadata.get("eligible_evaluation_roles") and role not in metadata["eligible_evaluation_roles"]:
            raise ValueError("Checkpoint's source-oracle regime excludes this evaluation role")
        encoded=L.encode_population(records,tok,"empty")
        if any(len(r["prompt_ids"])+len(r["response_ids"])>model.config.max_seq_len for r in encoded):
            raise ValueError("A whole reference does not fit the model context")
        ledger={};t0=time.time()
        scores=R.score(model,tok,encoded,args.batch_size,ledger,args.precision,args.score_temperature)
        for row in scores:
            source=by_source[row["source_id"]]
            row.update({"context_id":source["context_id"],"theorem_id":source["theorem_id"],
                        "length_bin":source["length_bin"],"role":role,"checkpoint_sha256":hashes[str(path)]})
        name=f"{index:03d}_"+re.sub(r"[^A-Za-z0-9_.-]","_",path.parent.name)
        directory=output/name;directory.mkdir()
        common.write_jsonl(directory/"whole_proof_nll.jsonl",scores)
        result={"checkpoint":str(path),"checkpoint_sha256":hashes[str(path)],"scores_directory":name,"score_temperature":args.score_temperature,
                **summarize(scores),"ledger":ledger,"total_flop_proxy":L.total_flop_proxy(ledger),
                "elapsed_seconds":time.time()-t0,"full_source_supervision_cap":metadata.get("full_source_supervision_cap")}
        common.write_json(directory/"result.json",result);results.append(result);total_ledger[name]=ledger
        common.write_json(output/"progress.json",{"completed":len(results),"total":len(checkpoints),"latest":result["checkpoint"]})
        print(json.dumps({"completed":len(results),"total":len(checkpoints),"checkpoint":str(path),
                          "length_balanced_mean_bits":result["length_balanced_mean_bits"]}),flush=True)
        del model
    summary={"role":role,"start":"full","generation_executed":False,"optimizer_updates":0,"score_temperature":args.score_temperature,
             "checkpoints":results,"total_flop_proxy":L.total_flop_proxy(total_ledger),"ledger":total_ledger,
             "elapsed_seconds":time.time()-started}
    common.write_json(output/"result.json",summary)
    common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json"),"finished_unix":time.time()})

def parser():
    p=argparse.ArgumentParser();p.add_argument("--checkpoints",nargs="+",required=True)
    p.add_argument("--data",required=True);p.add_argument("--release",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--score-temperature",type=float,default=1.)
    return p

if __name__=="__main__":run(parser().parse_args())
