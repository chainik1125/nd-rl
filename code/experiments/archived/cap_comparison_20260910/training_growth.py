"""Reverify and compress observed training certificates above their initial cap."""
import argparse
from collections import Counter
import json
from pathlib import Path

import analyze as a


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--arm",required=True)
    args=parser.parse_args();root=args.root.resolve()
    analysis=a.read(root/"reports/analysis_results.json");arm=analysis["arms"][args.arm]
    spec,_,_=a.validate_arm(root,args.arm,analysis["training_protocol_sha256"])
    information=a.read(root/"reports"/f"{args.arm}_training_information.json")
    if information["rl_checkpoint_sha256"]!=arm["training"]["rl"]["checkpoint_sha256"]:
        raise ValueError("Growth audit checkpoint differs from training diagnostics")
    source=root/spec["run_base"]/"rl/rollouts.jsonl"
    if a.sha(source)!=information["rollouts_sha256"]:
        raise ValueError("Growth audit rollouts differ from completed training diagnostics")
    manifest=a.read(root/spec["run_base"]/"rl/manifest.json")
    data=root/"data/rl_common.jsonl"
    if a.sha(data)!=manifest["data_sha256"]:
        raise ValueError("Training prompt identity differs")
    targets={r["prompt"]:r for line in data.read_text().splitlines() if (r:=json.loads(line))}
    tok=a.trainer.ScratchTokenizer();cache={};counts=Counter();histogram=Counter();theorems=set();certificates=set()
    largest=None;maximum_row=None;largest_by_bucket={}

    def verify(row):
        key=(row["prompt"],tuple(row["response_ids"]))
        if key not in cache:
            if row["prompt"] not in targets:
                raise ValueError("Training growth theorem absent from frozen pool")
            judged=a.trainer.judge_response(row["prompt"],row["response_ids"],tok,spec["mode"])
            if judged["reward"]!=1:
                raise ValueError("Training growth candidate fails exact replay")
            compressed=a.compress_proof(row["prompt"],judged["body"])
            valid,reason,length=a.safe_verify(row["prompt"],compressed["compressed_proof"])
            if not valid or length!=compressed["compressed_length"]:
                raise ValueError("Training growth compression is invalid: "+reason)
            cache[key]=(judged,compressed)
        judged,compressed=cache[key]
        if any(judged[k]!=row[k] for k in ("reward","reason","body","found_len")):
            raise ValueError("Training growth candidate metadata differs from exact replay")
        return compressed

    with source.open() as stream:
        for line in stream:
            row=json.loads(line)
            if not row["reward"]:continue
            counts["accepted_rollouts"]+=1
            if maximum_row is None or row["found_len"]>maximum_row["found_len"]:
                maximum_row=row
            if row["found_len"]<=spec["cap"]:continue
            counts["raw_above_cap_rollouts"]+=1
            compressed=verify(row);length=compressed["compressed_length"];histogram[length]+=1
            if length<=spec["cap"]:continue
            counts["compressed_above_cap_rollouts"]+=1
            theorems.add(row["prompt"]);certificates.add((row["prompt"],compressed["compressed_proof"]))
            target=targets[row["prompt"]];bucket=target["sampling_bucket"]
            if (largest is None or length>largest["compressed_length"]
                    or bucket not in largest_by_bucket or length>largest_by_bucket[bucket]["compressed_length"]):
                example={"step":row["step"],"group":row["group"],"attempt":row["attempt"],
                         "sampling_bucket":target["sampling_bucket"],"family":target.get("family"),
                         "prompt":row["prompt"],"original_proof":row["body"],"raw_length":row["found_len"],
                         "compressed_proof":compressed["compressed_proof"],"compressed_length":length}
                if largest is None or length>largest["compressed_length"]:largest=example
                if bucket not in largest_by_bucket or length>largest_by_bucket[bucket]["compressed_length"]:
                    largest_by_bucket[bucket]=example
    if maximum_row is not None:verify(maximum_row)
    if counts["accepted_rollouts"]!=sum(v["correct_responses"] for v in information["buckets"].values()):
        raise ValueError("Growth audit acceptance total differs from full training diagnostics")
    result={"kind":"posthoc_training_growth_audit","arm":args.arm,"cap":spec["cap"],
            "rl_checkpoint_sha256":arm["training"]["rl"]["checkpoint_sha256"],"rollouts_sha256":a.sha(source),
            "data_sha256":a.sha(data),"counts":dict(counts),
            "maximum_raw_training_witness":maximum_row["found_len"] if maximum_row else None,
            "compressed_length_histogram_for_raw_above_cap_candidates":dict(sorted(histogram.items())),
            "unique_theorems_with_compressed_witness_above_cap":len(theorems),
            "unique_theorems_by_sampling_bucket":dict(Counter(targets[p]["sampling_bucket"] for p in theorems)),
            "unique_compressed_certificates_above_cap":len(certificates),"largest_compressed_above_cap_example":largest,
            "largest_compressed_above_cap_example_by_bucket":largest_by_bucket,
            "scope":"Training-only generated certificates, not held-out transfer. Every raw-above-cap candidate was replayed and compressed; the recorded raw maximum was also replayed. Repeated certificates are not independent discoveries. Dependency/reiteration compression does not prove minimality."}
    destination=root/"reports"/f"{args.arm}_training_growth.json";a.write(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("largest_compressed_above_cap_example","largest_compressed_above_cap_example_by_bucket","rollouts_sha256","data_sha256","rl_checkpoint_sha256")}))


if __name__=="__main__":
    main()
