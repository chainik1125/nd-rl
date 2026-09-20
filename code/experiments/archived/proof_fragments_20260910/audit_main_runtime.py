"""Read-only reconstruction of a completed prefix of an active RL run."""
import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random
import time
import torch
import common
import learning as L
import nd_environment as E
from fragment_experiment import training_rows


def log_prefix(path, steps):
    records=[];digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for line in handle:
            row=json.loads(line)
            if row["step"]>steps:break
            records.append(row);digest.update(line)
    return records,digest.hexdigest()


def stream_prefix(path, steps, digest):
    """Keep complete endpoint audits bounded in memory without changing scope."""
    with Path(path).open("rb") as handle:
        for line in handle:
            row=json.loads(line)
            if row["step"]>steps:break
            digest.update(line)
            yield row


def run(directory, steps):
    started=time.monotonic();directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text());args=manifest["args"]
    assert args["command"]=="rl" and not args.get("initial_cuts"),"This audit reconstructs the original controller"
    for paths in (manifest["code_sha256"],manifest["dependency_sha256"]):
        for path,digest in paths.items():assert common.sha(path)==digest,("source identity",path)
    for key,digest in manifest["input_sha256"].items():assert common.sha(args[key])==digest,("input identity",key)
    for filename,path in (("learning.py",L.__file__),("nd_environment.py",E.__file__)):
        assert common.sha(path)==next(v for p,v in manifest["code_sha256"].items() if Path(p).name==filename)
    torch.set_num_threads(1)
    model,tok,_=L.load_model(args["checkpoint"])
    records=training_rows(args["data"])
    if args["families"]:records=[r for r in records if r["family"] in args["families"]]
    retention=L.encode_population(training_rows(args["retention"]),tok,"empty")
    indices={r["source_id"]:r for r in records}
    admissible={sid:L.admissible_cuts(r) for sid,r in indices.items()}
    curriculum={sid:{"cut":max(cuts),"n":0,"success":0} for sid,cuts in admissible.items()}
    updates,update_hash=log_prefix(directory/"updates.jsonl",steps)
    assert len(updates)==steps
    stream_hashes={name:hashlib.sha256() for name in ("groups","rollouts")}
    streams={name:stream_prefix(directory/f"{name}.jsonl",steps,stream_hashes[name]) for name in stream_hashes}
    source_rng=random.Random(args["seed"]);reset_rng=random.Random(args["seed"]+1000033)
    retention_rng=random.Random(args["seed"]+1000003)
    retention_hash=hashlib.sha256();source_hash=hashlib.sha256()
    p=sum(x.numel() for x in model.parameters());c=model.config
    def dense(rows,backward):
        b=len(rows);s=max(len(r["prompt_ids"])+len(r["response_ids"])-1 for r in rows)
        return (6 if backward else 2)*p*b*s+(12 if backward else 4)*c.n_layers*b*s*s*c.d_model
    total_flops=0;total_tokens=0;active_tokens=0;max_error=0.;mixed=0;moves=0
    phase_flops=Counter();outcome_counts=Counter();cut_counts=Counter();source_counts=Counter()
    move_directions=Counter();first_move_step=None
    nrows=args["prompts_per_step"]*args["group_size"]
    for step in range(1,steps+1):
        update=updates[step-1];assert update["step"]==step
        rows=list(islice(streams["rollouts"],nrows))
        events=list(islice(streams["groups"],args["prompts_per_step"]))
        assert len(rows)==nrows and len(events)==args["prompts_per_step"]
        selected=[records[source_rng.randrange(len(records))] for _ in range(args["prompts_per_step"])]
        selected_cuts=[curriculum[r["source_id"]]["cut"] if args["start"]=="reset" and reset_rng.random()>=args["empty_fraction"] else 0 for r in selected]
        source_hash.update(json.dumps([r["source_id"] for r in selected]).encode())
        step_mixed=0;step_active=0
        for g,(record,cut,event) in enumerate(zip(selected,selected_cuts,events)):
            assert event["step"]==step and event["group"]==g
            assert event["source_id"]==record["source_id"] and event["reset_index"]==cut
            assert cut in admissible[record["source_id"]]
            group=rows[g*args["group_size"]:(g+1)*args["group_size"]]
            ep=E.episode(record,cut,tok);rewards=[]
            for row in group:
                assert row["step"]==step
                for key in ("prompt_ids","original_prompt_ids","prefix_ids","source_id","source_lines","reset_index","format"):
                    assert row[key]==ep[key],("state mismatch",step,key)
                assert row["effective_max_new_tokens"]==args["max_new_tokens"]
                assert len(row["rollout_logprobs"])==len(row["response_ids"])
                assert all(math.isfinite(x) and x<=0.000001 for x in row["rollout_logprobs"])
                reward=E.judge(ep,row["response_ids"],tok)["reward"]
                assert reward==row["reward"],("reward mismatch",step,g)
                rewards.append(reward)
            k=sum(rewards);n=len(group)
            assert n==args["group_size"] and event["successes"]==k
            assert event["group_size"]==n and event["mixed"]==(0<k<n)
            assert event["generated_tokens"]==sum(len(r["response_ids"]) for r in group)
            assert event["prefix_tokens"]==len(ep["prefix_ids"])
            for row,reward in zip(group,rewards):
                expected=reward-(k-reward)/(n-1)
                assert row["advantage"]==expected
                if expected:step_active+=len(row["response_ids"])
            assert abs(sum(r["advantage"] for r in group))<1e-12
            step_mixed+=0<k<n;outcome_counts[k]+=1;cut_counts[cut]+=1;source_counts[record["source_id"]]+=1
            stats=curriculum[record["source_id"]];expected_move=None
            if args["start"]=="reset" and cut==stats["cut"]:
                stats["n"]+=n;stats["success"]+=k
                if stats["n"]>=args["curriculum_rollouts"]:
                    rate=stats["success"]/stats["n"]
                    cuts=admissible[record["source_id"]];pos=cuts.index(stats["cut"])
                    nextpos=max(0,pos-1) if rate>args["backward_threshold"] else min(len(cuts)-1,pos+1) if rate<args["forward_threshold"] else pos
                    if nextpos!=pos:
                        expected_move={"old":stats["cut"],"new":cuts[nextpos],"n":stats["n"],"success":stats["success"]}
                        move_directions["backward" if nextpos<pos else "forward"]+=1
                        if first_move_step is None:first_move_step=step
                        stats.update(cut=cuts[nextpos],n=0,success=0);moves+=1
            assert event.get("curriculum_move")==expected_move
        assert step_mixed==update["mixed_groups"] and step_active==update["active_advantage_tokens"]
        mixed+=step_mixed;active_tokens+=step_active
        total_tokens+=sum(len(r["response_ids"]) for r in rows)
        assert total_tokens==update["generated_tokens_total"]
        assert update["likelihood"]["n_action_tokens"]==sum(len(r["response_ids"]) for r in rows)
        assert update["likelihood"]["max_abs_error"]<=args["max_logprob_error"]
        max_error=max(max_error,update["likelihood"]["max_abs_error"])
        if not step_mixed:assert update["policy_loss"]==0
        phase=Counter()
        ordered=sorted(rows,key=lambda row:len(row["prompt_ids"]))
        for offset in range(0,len(ordered),args["batch_size"]):
            chunk=ordered[offset:offset+args["batch_size"]];n=len(chunk)
            width=max(len(r["prompt_ids"]) for r in chunk);decode=max(len(r["response_ids"]) for r in chunk)-1
            pairs=n*(width*width+sum(width+k for k in range(1,decode+1)))
            phase["generation"]+=2*p*n*(width+decode)+4*c.n_layers*c.d_model*pairs
        for offset in range(0,len(rows),args["microbatch"]):
            chunk=rows[offset:offset+args["microbatch"]]
            batch=L.collate(chunk,tok.pad_id)
            for row,mask in zip(chunk,batch["mask"]):
                start=len(row["prompt_ids"])-1;end=start+len(row["response_ids"])
                assert not bool(mask[:start].any()) and bool(mask[start:end].all()) and not bool(mask[end:].any())
            phase["behavior_audit"]+=dense(chunk,False);phase["rl_training"]+=dense(chunk,True)
        retained=[retention[retention_rng.randrange(len(retention))] for _ in range(args["retention_batch"])]
        retention_hash.update(json.dumps([r["source_id"] for r in retained]).encode())
        assert update["retention_examples"]==len(retained)
        assert update["retention_response_tokens"]==sum(len(r["response_ids"]) for r in retained)
        if retained and args["retention_weight"]:phase["rl_retention"]+=dense(retained,True)
        assert sum(phase.values())==update["step_flop_proxy"],("compute mismatch",step,dict(phase),update["step_flop_proxy"])
        phase_flops.update(phase);total_flops+=sum(phase.values())
        assert total_flops==update["cumulative_flop_proxy"]
    for stream in streams.values():assert next(stream,None) is None,"Extra row inside audited step prefix"
    hashes={"updates":update_hash,**{name:digest.hexdigest() for name,digest in stream_hashes.items()}}
    endpoint=None
    if (directory/"completed.json").exists():
        completion=json.loads((directory/"completed.json").read_text())
        result=json.loads((directory/"result.json").read_text())
        if result["steps"]==steps:
            assert completion["ok"] and completion["result_sha256"]==common.sha(directory/"result.json")
            assert completion["final_sha256"]==result["final_sha256"]==common.sha(directory/"final.pt")
            assert result["total_flop_proxy"]==total_flops==L.total_flop_proxy(result["ledger"])
            assert result["max_flop_proxy"]==args["max_flop_proxy"]
            assert result["stop_reason"]=="flop_proxy_ceiling"
            assert all(0<u["step_flop_proxy"]<=result["worst_case_one_step_flop_proxy"] for u in updates)
            assert total_flops>=args["max_flop_proxy"] and total_flops-updates[-1]["step_flop_proxy"]<args["max_flop_proxy"]
            assert result["flop_proxy_overshoot"]==total_flops-args["max_flop_proxy"]
            assert result["flop_proxy_overshoot"]<=result["worst_case_one_step_flop_proxy"]
            assert dict(phase_flops)=={name:result["ledger"][name]["flop_proxy"] for name in phase_flops}
            assert result["curriculum"]==curriculum==json.loads((directory/"curriculum.json").read_text())
            assert result["mixed_groups"]==mixed and result["n_groups"]==steps*args["prompts_per_step"]
            assert result["generated_tokens"]==total_tokens and result["active_advantage_tokens"]==active_tokens
            assert result["retention_total_examples"]==steps*args["retention_batch"]
            endpoint={"completed_result_sha256":completion["result_sha256"],"final_checkpoint_sha256":result["final_sha256"],
                      "exact_final_curriculum_agreement":True,"all_phase_and_step_compute_reconciled":True,
                      "first_budget_crossing_is_final_step":True,"flop_proxy_overshoot":result["flop_proxy_overshoot"],
                      "worst_case_one_step_flop_proxy":result["worst_case_one_step_flop_proxy"],
                      "zero_advantage_updates":sum(u["mixed_groups"]==0 for u in updates)}
    return {"status":"passed","directory":str(directory.resolve()),"steps_audited":steps,
            "manifest_sha256":common.sha(directory/"manifest.json"),"log_prefix_sha256":hashes,
            "input_sha256":manifest["input_sha256"],"n_sources":len(records),"families":dict(Counter(r["family"] for r in records)),
            "n_rollouts":steps*nrows,"n_groups":steps*args["prompts_per_step"],"mixed_groups":mixed,
            "outcome_success_count_histogram":dict(outcome_counts),"cut_histogram":dict(cut_counts),"curriculum_moves":moves,
            "controller_move_directions":dict(move_directions),"first_controller_move_step":first_move_step,
            "final_controller_cut_histogram":dict(Counter(v["cut"] for v in curriculum.values())),
            "final_controller_state_sha256":common.digest(curriculum),"endpoint_accounting":endpoint,
            "generated_tokens":total_tokens,"active_advantage_tokens":active_tokens,"max_logged_behavior_teacher_error":max_error,
            "phase_flop_proxy":dict(phase_flops),"total_flop_proxy":total_flops,
            "source_selection_schedule_sha256":source_hash.hexdigest(),"retention_schedule_sha256":retention_hash.hexdigest(),
            "retention_examples":steps*args["retention_batch"],"distinct_sampled_sources":len(source_counts),
            "checks":["frozen code/dependency/input hashes","actual cap6 certificates and source provenance","exact seeded source/reset/retention schedule",
                      "every saved state and strict reward","sourcewise LOO advantages and mixed groups","all action/context/padding masks",
                      "curriculum counters and threshold moves","unclipped fixed response budget","dense generation/audit/backward/retention proxy per update"],
            "limitation":"Per-update policies were not saved, so teacher probabilities are bounded by the saved runtime audits; this read-only audit does not recompute changing-policy logits.",
            "elapsed_seconds":time.monotonic()-started}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run",required=True);parser.add_argument("--steps",type=int,default=300)
    parser.add_argument("--output",required=True);args=parser.parse_args()
    result=run(args.run,args.steps);common.write_json(Path(args.output),result);print(json.dumps(result),flush=True)
