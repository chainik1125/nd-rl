"""Read-only, teacher-forced reference diagnostics on development populations.

Nonunique verified references are conditioning sequences, not validity labels.
No optimizer, gradients, generation, curriculum, or checkpoint selection.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
import common
import learning as L
from nd_environment import episode,strict_complete

ROOT=Path(__file__).resolve().parents[1]
RULES={"PR","AS","R","ANDI","ANDE1","ANDE2","IMPE","IMPI","ORI1","ORI2","ORE","NEGE","NEGI","BOTE","DN"}

def parse_labels(response,tok):
    """Parse codec position, so token R is an atom before ':' and a rule after."""
    tokens=[tok.itos[x] for x in response]
    labels=[];lines=[];line=None;state="start"
    for index,t in enumerate(tokens):
        fine=coarse=None
        if state=="start":
            if t!="<FINAL>":raise ValueError("Expected FINAL before proof")
            coarse,fine="structure","format_delimiter";state="between_lines"
        elif state=="between_lines":
            if t=="<LINE>":
                line={"line":len(lines)+1,"formula_tokens":0,"formula_positions":[],"rule":None,"positions":[]}
                lines.append(line);coarse,fine="structure","line_marker";state="formula"
            elif t=="QED":coarse,fine="structure","terminal";state="done"
            else:raise ValueError("Expected next line or QED")
        elif state=="formula":
            if t=="|" and not line["formula_tokens"]:coarse,fine="structure","scope"
            elif t==":":
                if not line["formula_tokens"]:raise ValueError("Missing formula")
                coarse,fine="structure","separator";state="rule"
            else:
                if t in "PQRS" and len(t)==1:fine="formula_atom"
                elif t=="F":fine="formula_constant"
                elif t in ("&","v",">","~"):fine="formula_operator"
                elif t in ("(",")"):fine="formula_parenthesis"
                else:raise ValueError("Unexpected formula token: "+t)
                coarse="formula";line["formula_tokens"]+=1;line["formula_positions"].append(index)
        elif state=="rule":
            if t not in RULES:raise ValueError("Unknown proof rule")
            coarse,fine="rule","rule";line["rule"]=t;state="references"
        elif state=="references":
            if t=="<REF>":coarse,fine="citation","reference_start";state="distance"
            elif t==";":coarse,fine="structure","separator";state="between_lines"
            else:raise ValueError("Unexpected rule tail")
        elif state=="distance":
            if t=="<BACK>":coarse,fine="citation","reference_distance"
            elif t=="<ENDREF>":coarse,fine="citation","reference_end";state="references"
            else:raise ValueError("Unexpected citation token")
        else:raise ValueError("Tokens after terminal QED")
        label={"coarse":coarse,"fine":fine,"line":line["line"] if line and fine!="terminal" else None}
        labels.append(label)
        if line and fine!="terminal":line["positions"].append(index)
    if state!="done":raise ValueError("No terminal QED")
    for line in lines:
        role="premise" if line["rule"]=="PR" else "assumption" if line["rule"]=="AS" else "inference"
        width=line["formula_tokens"]
        bucket="1-4" if width<=4 else "5-12" if width<=12 else "13-24" if width<=24 else "25+"
        for index in line["positions"]:
            labels[index].update(line_rule=line["rule"],line_role=role,formula_width=width,formula_width_bin=bucket)
    return labels,lines

def metrics(nll,correct):
    n=len(nll)
    return {"tokens":n,"nll_nats":sum(nll),"top1_correct":sum(correct)}

def finish(s):
    s=dict(s);n=s["tokens"]
    s["nll_nats_per_token"]=s["nll_nats"]/n if n else None
    s["bits_per_token"]=s["nll_nats_per_token"]/math.log(2) if n else None
    s["top1_accuracy"]=s["top1_correct"]/n if n else None
    return s

def summarize_reference(record,ep,labels,nll,correct):
    result={k:record[k] for k in ("source_id","family","length_bin","n_lines","context_id")}
    result.update(prompt_tokens=len(ep["prompt_ids"]),response_tokens=len(ep["response_ids"]),
                  total=finish(metrics(nll,correct)),teacher_forced_all_top1=all(correct))
    groups=defaultdict(list)
    for i,label in enumerate(labels):
        for kind in ("coarse","fine"):
            groups[kind+":"+label[kind]].append(i)
        if label["coarse"]=="formula":
            groups["formula_line_role:"+label["line_role"]].append(i)
            groups["formula_width:"+label["formula_width_bin"]].append(i)
        if label["coarse"]=="rule":groups["rule_name:"+label["line_rule"]].append(i)
    result["categories"]={k:finish(metrics([nll[i] for i in ids],[correct[i] for i in ids])) for k,ids in groups.items()}
    return result

def aggregate(records):
    ids=defaultdict(list)
    for row in records:
        ids["all"].append(row)
        for key in ("family","length_bin"):
            ids[key+":"+str(row[key])].append(row)
        ids["family_length:"+row["family"]+":"+str(row["length_bin"])].append(row)
    output={}
    def add(xs):return finish({k:sum(x[k] for x in xs) for k in ("tokens","nll_nats","top1_correct")})
    for name,rows in ids.items():
        cats=sorted({k for row in rows for k in row["categories"]})
        output[name]={"references":len(rows),"contexts":len({r["context_id"] for r in rows}),
                      "total":add([r["total"] for r in rows]),
                      "proof_mean_nll_nats_per_token":sum(r["total"]["nll_nats_per_token"] for r in rows)/len(rows),
                      "proof_mean_top1_accuracy":sum(r["total"]["top1_accuracy"] for r in rows)/len(rows),
                      "whole_reference_all_top1":sum(r["teacher_forced_all_top1"] for r in rows),
                      "categories":{k:add([r["categories"][k] for r in rows if k in r["categories"]]) for k in cats}}
    return output

def model_state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def population(path,tok):
    path=Path(path)
    if path.name!="development.jsonl":raise ValueError("Only development diagnostics are authorized")
    audit_path=path.parent/"audit"/path.name
    prompts=common.read_jsonl(path);audits=common.read_jsonl(audit_path)
    byid={r["source_id"]:r for r in audits}
    if len(byid)!=len(audits) or len({r["source_id"] for r in prompts})!=len(prompts):raise ValueError("Duplicate identities")
    if set(byid)!={r["source_id"] for r in prompts}:raise ValueError("Population/audit identity mismatch")
    result=[]
    for prompt in prompts:
        if set(prompt)-{"source_id","name","prompt"}:raise ValueError("Non-theorem fields in input population")
        row=byid[prompt["source_id"]]
        if row["prompt"]!=prompt["prompt"] or row.get("role")!="development":raise ValueError("Development audit mismatch")
        if not strict_complete(row["prompt"],row["body"])["ok"]:raise ValueError("Invalid reference proof")
        ep=episode(row,0,tok)
        if ep["prefix_ids"] or ep["prompt_ids"][-2:]!=[tok.prf_id,tok.prf_id]:raise ValueError("Wrong empty-prefix format")
        labels,lines=parse_labels(ep["response_ids"],tok)
        if len(lines)!=row["n_lines"]:raise ValueError("Reference line metadata mismatch")
        result.append((row,ep,labels))
    return result,{str(path.resolve()):common.sha(path),str(audit_path.resolve()):common.sha(audit_path)}

@torch.inference_mode()
def run(args):
    torch.set_num_threads(4);torch.manual_seed(0)
    start=time.monotonic()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=common.sha(args.checkpoint)
    model,tok,metadata=L.load_model(args.checkpoint,"cpu")
    for p in model.parameters():p.requires_grad_(False)
    before=model_state_hash(model)
    populations={};input_hashes={str(args.checkpoint.resolve()):checkpoint_hash}
    for name,path in (("primary",ROOT/"data/v1/development.jsonl"),("structured",ROOT/"data/structured_lm_v1/development.jsonl")):
        populations[name],hashes=population(path,tok);input_hashes.update(hashes)
    for records in populations.values():
        if any(len(ep["prompt_ids"])+len(ep["response_ids"])>model.config.max_seq_len for _,ep,_ in records):
            raise ValueError("Reference does not fit context: no truncation")
    manifest={"status":"running","pid":os.getpid(),"checkpoint":str(args.checkpoint.resolve()),
              "checkpoint_metadata":metadata,"input_sha256":input_hashes,"model_state_before_sha256":before,
              "code_sha256":{str(Path(p).resolve()):common.sha(p) for p in (__file__,L.__file__,ROOT/"code/nd_environment.py")}|common.dependency_hashes(),
              "device":"cpu","cpu_threads":4,"precision":"fp32","temperature":1.,"batch_size":args.batch_size,
              "max_seconds":args.max_seconds,"parameter_count":sum(p.numel() for p in model.parameters()),
              "format":"Original theorem + PRF delimiter (double PRF); reference response starts FINAL; no supplied proof prefix",
              "sampling":"None. Teacher-forced conditional likelihood of fixed verified references."}
    progress_path=args.output.with_name(args.output.stem+"_progress.json")
    common.write_json(progress_path,manifest)
    results={};ledger={};completed=0
    for name,records in populations.items():
        scored=[]
        ordered=sorted(records,key=lambda r:len(r[1]["prompt_ids"])+len(r[1]["response_ids"]))
        for offset in range(0,len(ordered),args.batch_size):
            if time.monotonic()-start>args.max_seconds:raise TimeoutError("CPU diagnostic time ceiling reached")
            chunk=ordered[offset:offset+args.batch_size]
            batch=L.collate([r[1] for r in chunk],tok.pad_id,"cpu")
            logits=model(batch["inputs"]).float()
            chosen=logits.log_softmax(-1).gather(-1,batch["targets"][...,None]).squeeze(-1)
            correct=logits.argmax(-1)==batch["targets"]
            for i,(record,ep,labels) in enumerate(chunk):
                nll=(-chosen[i][batch["mask"][i]]).double().tolist()
                hits=correct[i][batch["mask"][i]].tolist()
                scored.append(summarize_reference(record,ep,labels,nll,hits))
            L.account(ledger,model,batch,"teacher_forced_reference")
            completed+=len(chunk)
            event={"population":name,"completed_population":len(scored),"population_size":len(records),
                   "completed_total":completed,"elapsed_seconds":time.monotonic()-start}
            common.write_json(progress_path,{**manifest,"progress":event})
            if offset==0 or (offset//args.batch_size+1)%8==0:print(json.dumps(event),flush=True)
        results[name]={"summary":aggregate(scored),"per_reference":sorted(scored,key=lambda r:r["source_id"])}
    after=model_state_hash(model)
    if after!=before or any(p.grad is not None for p in model.parameters()):raise AssertionError("Read-only model audit failed")
    if common.sha(args.checkpoint)!=checkpoint_hash:raise ValueError("Checkpoint changed during audit")
    report={**manifest,"status":"completed","model_state_after_sha256":after,"weights_unchanged":True,
            "elapsed_seconds":time.monotonic()-start,"ledger":ledger,"populations":results,
            "notes":["References are nonunique witnesses. Their likelihood and token accuracy do not measure theorem validity or optimal proof strategy.",
                     "Every prediction is conditioned on the complete correct reference prefix; local token accuracy does not imply free-running proof completion.",
                     "Formula/rule R is disambiguated by parse position. Formula parentheses and operator tokens are included in coarse formula.",
                     "Category and total means are token-weighted; separate proof-mean metrics weight references equally.",
                     "Formula/premise burden is diagnostic association, not evidence establishing copying as the causal bottleneck.",
                     "Whole-reference all-top1 is conditional agreement with this particular reference, not a theorem-level validity statistic."]}
    common.write_json(args.output,report)
    common.write_json(progress_path,{**manifest,"status":"completed","elapsed_seconds":report["elapsed_seconds"]})
    print(json.dumps({"status":"completed","output":str(args.output),"elapsed_seconds":report["elapsed_seconds"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=ROOT/"models/adapt_final.pt")
    p.add_argument("--output",type=Path,default=ROOT/"reports/review_representation_baseline.json")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-seconds",type=int,default=900)
    args=p.parse_args()
    if args.batch_size<1 or args.max_seconds<1:p.error("Positive batch size/time ceiling required")
    run(args)
