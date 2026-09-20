"""Candidate-only pre/post-adaptation reference and conditional coding costs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import torch
import common
import learning as L
import reference_scoring as R
from fragment_experiment import training_rows
from nd_environment import FORMAT


def candidate_rows(path):
    path=Path(path).resolve();digest=common.sha(path)
    metadata_path=path.parent/"manifest.json";metadata=json.loads(metadata_path.read_text())
    if not ((metadata.get("status")=="released_complete_primary" and metadata.get("population_sha256",{}).get("candidate")==digest)
            or (metadata.get("status")=="frozen_bank" and metadata.get("bank_sha256")==digest)):
        raise ValueError("Characterization requires a frozen candidate population/bank identity")
    records=training_rows(path)
    if not records or any(r["role"]!="candidate" for r in records):
        raise ValueError("Characterization only consumes cap6 candidate certificates")
    if len({r["source_id"] for r in records})!=len(records):raise ValueError("Duplicate candidate source")
    return records,metadata_path


def summary(scores):
    values=defaultdict(list)
    for row in scores:values[row["family"]].append(row)
    def aggregate(rows):
        return {"n_sources":len(rows),"mean_total_bits":sum(r["bits"] for r in rows)/len(rows),
                "mean_bits_per_token":sum(r["bits_per_token"] for r in rows)/len(rows),
                "aggregate_bits_per_token":sum(r["bits"] for r in rows)/sum(r["tokens"] for r in rows),
                "response_tokens":sum(r["tokens"] for r in rows)}
    return {**aggregate(scores),"by_family":{family:aggregate(rows) for family,rows in sorted(values.items())}}


def run(args):
    started=time.time();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Characterization output must be empty")
    R.validate_temperature(args.score_temperature)
    data=Path(args.data).resolve();records,population_metadata=candidate_rows(data)
    paths={"base_sft":Path(args.base_checkpoint).resolve(),"adapted":Path(args.adapted_checkpoint).resolve()}
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest={"command":"characterize_candidates","args":vars(args),"role":"candidate","source_cap":6,
              "generation_executed":False,"optimizer_updates":0,"score_temperature":args.score_temperature,
              "input_format_for_both_models":FORMAT,"initial_means":"pre-shared-adaptation SFT weights, not random initialization",
              "format_shift_note":"Both checkpoints are scored with the common double-PRF format, even when base SFT did not train on that format.",
              "input_sha256":{"data":common.sha(data),"population_metadata":common.sha(population_metadata),
                              **{name:common.sha(path) for name,path in paths.items()}},
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes(),"started_unix":started}
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/"manifest.json",manifest)
    common.write_jsonl(output/"source_index.jsonl",[
        {k:r[k] for k in ("source_id","theorem_id","context_id","role","family","n_lines","default_cut","fragment_descriptor","provenance")}
        for r in records])
    lookup={r["source_id"]:r for r in records};all_scores={};ledgers={};results={}
    for label,path in paths.items():
        model,tok,metadata=L.load_model(path,args.device)
        if label=="adapted" and metadata.get("format")!=FORMAT:
            raise ValueError("Adapted endpoint does not declare the common input format")
        results[label]={"checkpoint":str(path),"checkpoint_sha256":manifest["input_sha256"][label],
                        "checkpoint_metadata":metadata,"common_format_used":FORMAT,"modes":{}}
        for mode,encoding in (("full_start","empty"),("fragment","fragment")):
            encoded=L.encode_population(records,tok,encoding)
            if any(len(r["prompt_ids"])+len(r["response_ids"])>model.config.max_seq_len for r in encoded):
                raise ValueError("Whole reference continuation does not fit; no truncation allowed")
            ledger={};scores=R.score(model,tok,encoded,args.batch_size,ledger,args.precision,args.score_temperature)
            for row,ep in zip(scores,encoded):
                source=lookup[row["source_id"]]
                row.update({"model_label":label,"mode":mode,"role":"candidate","theorem_id":source["theorem_id"],
                            "context_id":source["context_id"],"checkpoint_sha256":manifest["input_sha256"][label],
                            "format":FORMAT,"supplied_context_tokens":len(ep["prompt_ids"]),"prefix_tokens":len(ep["prefix_ids"]),
                            "fragment_descriptor":source["fragment_descriptor"]})
            common.write_jsonl(output/f"{label}_{mode}_bits.jsonl",scores)
            all_scores[(label,mode)]={r["source_id"]:r for r in scores};ledgers[f"{label}_{mode}"]=ledger
            results[label]["modes"][mode]={**summary(scores),"ledger":ledger,"total_flop_proxy":L.total_flop_proxy(ledger)}
        print(json.dumps({"completed_model":label,"n_candidates":len(records),"score_temperature":args.score_temperature}),flush=True)
        del model
    paired=[]
    for record in records:
        sid=record["source_id"]
        row={k:record[k] for k in ("source_id","theorem_id","context_id","family","n_lines","default_cut")}
        for mode in ("full_start","fragment"):
            initial=all_scores[("base_sft",mode)][sid];current=all_scores[("adapted",mode)][sid]
            assert initial["tokens"]==current["tokens"] and initial["reset_index"]==current["reset_index"]
            row[mode]={"base_sft_bits":initial["bits"],"adapted_bits":current["bits"],
                       "base_sft_bits_per_token":initial["bits_per_token"],"adapted_bits_per_token":current["bits_per_token"],
                       "bit_reduction":initial["bits"]-current["bits"],"tokens":initial["tokens"],"reset_index":initial["reset_index"]}
        paired.append(row)
    common.write_jsonl(output/"paired_characterization.jsonl",paired)
    result={"role":"candidate","source_cap":6,"n_sources":len(records),"families":dict(Counter(r["family"] for r in records)),
            "format":FORMAT,"score_temperature":args.score_temperature,"models":results,
            "generation_executed":False,"optimizer_updates":0,"total_flop_proxy":L.total_flop_proxy(ledgers),
            "ledger":ledgers,"elapsed_seconds":time.time()-started,
            "interpretation":"Operational conditional coding costs on already released candidate certificates; not intrinsic proof complexity or held-out transfer performance."}
    common.write_json(output/"result.json",result)
    common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json")})


def parser():
    p=argparse.ArgumentParser()
    for flag in ("base-checkpoint","adapted-checkpoint","data","output"):p.add_argument("--"+flag,required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--score-temperature",type=float,default=1.)
    return p


if __name__=="__main__":run(parser().parse_args())
