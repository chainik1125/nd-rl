"""Unaided theorem-only evaluation; reference proofs supply audit strata only."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import torch
import common
import learning as L
from nd_environment import judge, Tokenizer, strict_complete
from nd_verify import parse_proof_tokens
from proof_diagnostics import compress_proof, _dependency_lines, _render

def empty_episode(row,tok):
    original=tok.encode_prompt(row["prompt"])
    return {"name":row["name"],"source_id":row["source_id"],"prompt":row["prompt"],
            "prompt_ids":original+[tok.prf_id],"original_prompt_ids":original,"prefix_ids":[],
            "reset_index":0,"format":"nd-prefix-double-prf-final-v1"}

def aggregate(rows,audit):
    grouped=defaultdict(list)
    for r in rows:grouped[r["source_id"]].append(r)
    result={}
    strata={"all":set(grouped)}
    for key in ("length_bin","family"):
        for i in grouped:strata.setdefault(f"{key}:{audit[i][key]}",set()).add(i)
    for name,ids in sorted(strata.items()):
        subset=[r for i in ids for r in grouped[i]]
        first=[next(r for r in grouped[i] if r["attempt"]==0) for i in ids]
        solved=[r for r in subset if r["reward"]]
        result[name]={"theorems":len(ids),"attempts":len(subset),"first_successes":sum(r["reward"] for r in first),
                      "any_successes":sum(any(r["reward"] for r in grouped[i]) for i in ids),
                      "attempt_successes":len(solved),"generated_tokens":sum(r["generated_tokens"] for r in subset),
                      "success_written_lengths":dict(Counter(r["n_lines"] for r in solved)),
                      "success_pruned_lengths":dict(Counter(r["compression"]["dependency_pruned_length"] for r in solved)),
                      "failure_reasons":dict(Counter(r["reason"] for r in subset if not r["reward"]))}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--release",type=Path)
    p.add_argument("--temperature",type=float,default=0)
    p.add_argument("--attempts",type=int,default=1)
    p.add_argument("--max-new-tokens",type=int,default=1024)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--seed",type=int,default=202609101935)
    p.add_argument("--precision",choices=["fp32","bf16"],default="fp32")
    p.add_argument("--device",default="cuda")
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4)
    if args.attempts<1 or args.temperature<0:raise ValueError("Invalid sampling specification")
    if args.temperature==0 and args.attempts!=1:raise ValueError("Greedy evaluation must use one attempt")
    if args.data.stem=="final":
        if args.limit:raise ValueError("Final evaluation must include the whole frozen population")
        if args.release is None:raise ValueError("Final evaluation requires a frozen release manifest")
        release=json.loads(args.release.read_text())
        if release.get("status")!="released" or common.sha(args.checkpoint) not in release["checkpoint_sha256"].values():
            raise ValueError("Checkpoint absent from frozen final release")
        if common.sha(args.data)!=release["data_sha256"] or common.sha(args.audit)!=release["audit_sha256"]:
            raise ValueError("Final population identity mismatch")
        profile={k:getattr(args,k) for k in ("temperature","attempts","max_new_tokens","batch_size","seed","precision")}
        if profile not in release["inference_profiles"]:
            raise ValueError("Inference settings absent from frozen final release")
    args.output.mkdir(parents=True,exist_ok=False)
    inputs={str(x.resolve()):common.sha(x) for x in (args.checkpoint,args.data,args.audit)}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    common.write_json(args.output/"config.json",{"args":config,"input_sha256":inputs,"code_sha256":common.sha(__file__),
                      "inference":"Empty prefix; original theorem plus format delimiter only. No proof search, verification feedback, or reference scoring during generation."})
    data=common.read_jsonl(args.data)
    if args.limit:data=data[:args.limit]
    if any(set(r)-{"name","source_id","prompt"} for r in data):raise ValueError("Evaluation prompts carry non-prompt fields")
    model,tok,_=L.load_model(args.checkpoint,args.device)
    audit={r["source_id"]:r for r in common.read_jsonl(args.audit)}
    if any(r["source_id"] not in audit or r["prompt"]!=audit[r["source_id"]]["prompt"] for r in data):
        raise ValueError("Prompt/audit identity mismatch")
    ledger={};all_rows=[];start=time.monotonic()
    # Whole groups remain adjacent; batching never changes the per-theorem cap.
    block=max(1,args.batch_size//args.attempts)
    with (args.output/"attempts.jsonl").open("w",buffering=1) as stream:
        for offset in range(0,len(data),block):
            records=data[offset:offset+block]
            episodes=[{**empty_episode(r,tok),"attempt":a} for r in records for a in range(args.attempts)]
            if any(len(r["prompt_ids"])+args.max_new_tokens>model.config.max_seq_len for r in episodes):
                raise ValueError("Declared generation cap does not fit context")
            samples=L.sample(model,tok,episodes,temperature=args.temperature,seed=args.seed+offset*1009,
                             max_new_tokens=args.max_new_tokens,batch_size=args.batch_size,precision=args.precision,ledger=ledger)
            for sample in samples:
                verdict=judge(sample,sample["response_ids"],tok)
                row={k:sample[k] for k in ("name","source_id","prompt","attempt","response_ids","finish_reason")}
                row.update(verdict,generated_tokens=len(sample["response_ids"]),reset_index=0,
                           reference_length=audit[sample["source_id"]]["n_lines"],length_bin=audit[sample["source_id"]]["length_bin"],
                           family=audit[sample["source_id"]]["family"])
                if row["reward"]:
                    row["compression"]=compress_proof(row["prompt"],row["body"])
                    dependency_body=_render(_dependency_lines(parse_proof_tokens(row["body"].split())))
                    for reduced in (dependency_body,row["compression"]["compressed_proof"]):
                        if not strict_complete(row["prompt"],reduced)["ok"]:
                            raise AssertionError("Analysis compression failed independent strict verification")
                stream.write(json.dumps(row)+"\n");all_rows.append(row)
            event={"theorems_done":offset+len(records),"theorems_total":len(data),"attempt_successes":sum(r["reward"] for r in all_rows),
                   "attempts":len(all_rows),"elapsed_seconds":time.monotonic()-start}
            common.write_json(args.output/"progress.json",event);print(json.dumps(event),flush=True)
    metrics=aggregate(all_rows,audit)
    result={"status":"completed","metrics":metrics,"elapsed_seconds":time.monotonic()-start,
            "ledger":ledger,"input_sha256":inputs,"attempts_sha256":common.sha(args.output/"attempts.jsonl"),
            "per_theorem_token_cap":args.attempts*args.max_new_tokens,"sampled_attempts":args.attempts,
            "temperature":args.temperature,"notes":"Reference lengths are known-witness upper bounds; compression is analysis-only."}
    common.write_json(args.output/"result.json",result);print(json.dumps(result),flush=True)

if __name__=="__main__":main()
