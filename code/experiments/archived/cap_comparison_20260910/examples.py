"""Show source-bound completed-run examples; never open unreleased book data."""
import argparse
import json
from pathlib import Path

import analyze as a


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--arm",required=True)
    args=parser.parse_args();root=args.root.resolve()
    analysis=a.read(root/"reports/analysis_results.json")
    arm=analysis["arms"][args.arm]
    spec,_,_=a.validate_arm(root,args.arm,analysis["training_protocol_sha256"])
    data_path=root/spec["development_data"]["path"]
    if a.sha(data_path)!=spec["development_data"]["sha256"]:
        raise ValueError("Example reference metadata differs from original dataset")
    references=[a.trainer.reference_length(json.loads(line)) for line in data_path.read_text().splitlines()]
    path=root/spec["run_base"]/"evaluations/rl_dev_greedy/attempts.jsonl"
    if a.sha(path)!=arm["endpoints"]["rl_greedy"]["attempts_sha256"]:
        raise ValueError("Example attempts changed since strict replay")
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    chosen=[]
    gains=[r for r in rows if r["reward"] and not arm["endpoints"]["sft_greedy"]["outcomes"][r["index"]]]
    if gains:chosen.append(("First newly solved greedy problem after RL",gains[0]))
    for rule in ("IMPE","ANDE1"):
        candidates=[r for r in rows if not r["reward"] and references[r["index"]]>6
                    and r["reason"].startswith(f"rule check failed: {rule} ")]
        if candidates:chosen.append((f"First longer-reference greedy failure at {rule}",candidates[0]))
    tok=a.trainer.ScratchTokenizer()
    lines=[f"# Completed-run examples: {arm['label']}","",
           "Illustrative, posthoc examples from the 58-problem adaptive textbook development set. "
           "Selection is the first newly solved greedy problem, then the first longer-reference greedy failures at IMPE and ANDE1. "
           "These examples do not alter the frozen experiment or substitute for aggregate results.","",
           f"Checkpoint SHA256: `{arm['endpoints']['rl_greedy']['checkpoint_sha256']}`. Attempts SHA256: `{a.sha(path)}`. Dataset SHA256: `{a.sha(data_path)}`."]
    identities=[]
    for title,row in chosen:
        judged=a.trainer.judge_response(row["prompt"],row["response_ids"],tok,spec["mode"])
        if any(judged[k]!=row[k] for k in ("reward","reason","body","found_len")):
            raise ValueError("Example verification disagrees with saved verdict")
        lines += ["",f"## {title}","",f"Dataset index {row['index']}; reference witness length {references[row['index']]}. "
                  f"Verifier verdict: **{'accepted' if row['reward'] else 'rejected'}**.","",f"```text\n{row['prompt']}\n```"]
        if spec["mode"]=="cot" and row["response_ids"] and row["response_ids"][0]==tok.stoi["<THINK>"] and tok.stoi["</THINK>"] in row["response_ids"]:
            end=row["response_ids"].index(tok.stoi["</THINK>"])
            plan=" ".join(tok.itos[i] for i in row["response_ids"][1:end]).replace(" ; "," ;\n")
            lines += ["","Generated symbolic scratchpad:","",f"```text\n{plan}\n```"]
        proof=row["body"].replace(" ; "," ;\n")
        lines += ["","Generated final proof:","",f"```text\n{proof}\n```","",f"Verifier message: `{row['reason']}`."]
        identities.append({"selection":title,"index":row["index"],"reward":row["reward"],"reason":row["reason"]})
    output=root/"reports"/f"{args.arm}_examples.md"
    output.write_text("\n".join(lines)+"\n")
    print(json.dumps({"output":str(output),"examples":identities}),flush=True)


if __name__=="__main__":
    main()
