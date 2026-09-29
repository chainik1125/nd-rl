"""Runnable, artifact-first fragment adaptation/calibration/evaluation/RL CLI."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time
import torch
import common
import learning as L
from nd_environment import FORMAT, episode, judge, strict_complete, independent_replay
from nd_model import save_checkpoint
from proof_diagnostics import compress_proof

def initialize(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError("Immutable run output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    inputs = {k: common.sha(v) for k,v in vars(args).items() if k in ("checkpoint","data","retention","selection","release","initial_cuts") and v}
    manifest = {"args":vars(args),"input_sha256":inputs,"format":FORMAT,
                "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},
                "dependency_sha256":common.dependency_hashes(),"started_unix":time.time(),
                "compute_proxy_note":"Dense projection/MLP and quadratic attention approximation; hardware FLOPs not measured."}
    common.write_json(out/"manifest.json",manifest)
    model,tok,_ = L.load_model(args.checkpoint,args.device)
    return out,model,tok,{},manifest

def training_rows(path):
    if any(w in Path(path).name.lower() for w in ("confirmation","selection","textbook_dev","textbook_final")):
        raise ValueError("Protected probe payload cannot be optimized")
    rows = common.read_jsonl(path)
    if any(r.get("role") not in ("candidate","adaptation","retention") for r in rows):
        raise ValueError("Only generated candidate/adaptation/retention roles may be optimized")
    for r in rows:
        verdict=strict_complete(r["prompt"],r["body"])
        if not verdict["ok"] or not r["n_lines"]==r["provenance"]["full_source_lines"]==verdict["n_lines"]<=6:
            raise ValueError("Clean training actual source length/provenance cap6 violated")
    return rows

def finish(out,model,tok,result,ledger,manifest,save=True):
    result.update({"ledger":ledger,"total_flop_proxy":L.total_flop_proxy(ledger),"elapsed_seconds":time.time()-manifest["started_unix"]})
    if save:
        save_checkpoint(out/"final.pt",model,tok,format=FORMAT,source_checkpoint_sha256=manifest["input_sha256"]["checkpoint"], stage=manifest["args"]["command"])
        result["final_sha256"]=common.sha(out/"final.pt")
    common.write_json(out/"result.json",result)
    common.write_json(out/"completed.json",{"ok":True,"finished_unix":time.time(),"result_sha256":common.sha(out/"result.json"),"final_sha256":result.get("final_sha256")})
    print(json.dumps(result),flush=True)

def adapt(args):
    out,model,tok,ledger,manifest=initialize(args)
    records=training_rows(args.data)
    # Exactly half of the shared adaptation pool uses empty starts; remainder
    # samples permissible cuts. Both downstream arms inherit this one model.
    rows=L.encode_population(records,tok,"empty")+L.encode_population(records,tok,"random",args.seed)
    common.write_json(out/"encoding_audit.json",{"records":len(records),"examples":len(rows),
        "max_sequence_length":max(len(r["prompt_ids"])+len(r["response_ids"]) for r in rows),
        "max_source_lines":max(r["source_lines"] for r in rows),"supplied_tokens_direct_loss":0,
        "empty_starts":sum(r["reset_index"]==0 for r in rows)})
    result=L.train_sft(model,tok,rows,[],steps=args.steps,batch_size=args.batch_size,fixed_pad=args.fixed_pad,
        lr=args.lr,seed=args.seed,ledger=ledger,output=out,retention_fraction=0,precision=args.precision,checkpoint_every=args.save_every)
    finish(out,model,tok,result,ledger,manifest)

def mean_bits(scores): return sum(r["bits"] for r in scores)/len(scores)

def length_bits(scores):
    groups=defaultdict(list)
    for r in scores:
        groups["2-6" if r["source_lines"]<=6 else "7-9" if r["source_lines"]<=9 else "10-12"].append(r["bits"])
    return {k:sum(v)/len(v) for k,v in groups.items()}

def balanced_bits(scores):
    values=length_bits(scores)
    return sum(values.values())/len(values)

def matched_controls(primary,ordinary):
    if not ordinary: raise ValueError("No ordinary control rows")
    if len(ordinary)<len(primary): raise ValueError("Cannot match ordinary sources without replacement")
    controls=[]; receipts=[]
    unused=list(ordinary)
    for r in primary:
        def distance(c):
            return (abs(c["source_lines"]-r["source_lines"])*50
                    +abs(len(c["prefix_state"]["active_assumptions"])-len(r["prefix_state"]["active_assumptions"]))*50
                    +abs(len(c["response_ids"])-len(r["response_ids"]))
                    +abs(len(c["prompt_ids"])-len(r["prompt_ids"]))*.1)
        candidate=min(unused,key=lambda c:(distance(c),c["source_id"]))
        unused.remove(candidate)
        controls.append(candidate)
        receipts.append({"family_source":r["source_id"],"ordinary_source":candidate["source_id"],
                         "family_context":r.get("context_id"),"ordinary_context":candidate.get("context_id"),
                         "distance":distance(candidate),"family_tokens":len(r["response_ids"]),
                         "control_tokens":len(candidate["response_ids"]),"family_source_lines":r["source_lines"],
                         "control_source_lines":candidate["source_lines"],
                         "exact_source_length_match":candidate["source_lines"]==r["source_lines"],
                         "exact_suffix_length_match":len(candidate["response_ids"])==len(r["response_ids"]),
                         "exact_depth_match":len(candidate["prefix_state"]["active_assumptions"])==len(r["prefix_state"]["active_assumptions"])})
    return controls,receipts

def intervene(args):
    out,model,tok,ledger,manifest=initialize(args)
    candidates=training_rows(args.data)
    selected=set(args.families or [r["family"] for r in candidates])
    candidates=[r for r in candidates if r["family"] in selected]
    counts=Counter(r["family"] for r in candidates)
    if set(counts)!=selected: raise ValueError("Requested family missing from source bank")
    if args.sampling=="uniform-sources":
        if args.families:raise ValueError("Uniform-source intervention consumes the whole frozen bank; do not filter families")
        bank_manifest=json.loads((Path(args.data).parent/"manifest.json").read_text())
        if bank_manifest.get("status")!="frozen_bank" or bank_manifest.get("bank_sha256")!=common.sha(args.data):
            raise ValueError("Uniform-source intervention requires an immutable bank identity")
        if len({r["source_id"] for r in candidates})!=len(candidates):
            raise ValueError("Uniform-source bank contains duplicate certificates")
    elif len(set(counts.values()))!=1:
        raise ValueError("Family-balanced mixture requires equal distinct-source quotas; no implicit duplication")
    rows=L.encode_population(candidates,tok,"fragment")
    retention=L.encode_population(training_rows(args.retention),tok,"empty")
    common.write_json(out/"mixture.json",{"families":sorted(selected),"source_counts":dict(counts),
                                         "source_ids":[r["source_id"] for r in candidates],"source_cap":6,
                                         "sampling":args.sampling,"unique_contexts":len({r["context_id"] for r in candidates}),
                                         "full_source_length_histogram":dict(Counter(r["n_lines"] for r in candidates))})
    result=L.train_sft(model,tok,rows,retention,steps=args.steps,batch_size=args.batch_size,
        fixed_pad=args.fixed_pad,lr=args.lr,seed=args.seed,ledger=ledger,output=out,
        retention_fraction=args.retention_fraction,precision=args.precision)
    finish(out,model,tok,result,ledger,manifest)

def calibrate(args):
    out,base,tok,ledger,manifest=initialize(args)
    candidates=training_rows(args.data)
    retained=training_rows(args.retention)
    probes=common.read_jsonl(args.selection)
    if any(r["role"]!="selection" for r in probes): raise ValueError("Calibration may open selection only")
    candidate_rows=L.encode_population(candidates,tok,"fragment")
    retention=L.encode_population(retained,tok,"empty")
    query=L.encode_population(probes,tok,"empty")
    base_scores=L.score(base,tok,query,args.batch_size,ledger,args.precision)
    surprise=L.score(base,tok,candidate_rows,args.batch_size,ledger,args.precision)
    common.write_jsonl(out/"baseline_selection_nll.jsonl",base_scores)
    common.write_jsonl(out/"candidate_conditional_nll.jsonl",surprise)
    query_gradient=L.query_gradient(base,tok,query,args.microbatch,ledger,args.precision) if args.actual_displacement else None
    families=args.families or sorted({r["family"] for r in candidates})
    arms=[]
    for family in families:
        primary=[r for r in candidate_rows if r["family"]==family]
        ordinary=[r for r in candidate_rows if r["family"]!=family]
        controls,matches=matched_controls(primary,ordinary)
        common.write_jsonl(out/f"{family}_matching.jsonl",matches)
        arm_results={}
        for kind,rows in (("control",controls),("family",primary)):
            armout=out/f"{family}_{kind}"; armout.mkdir()
            model,_,_=L.load_model(args.checkpoint,args.device)
            armledger={}
            first_displacement={}
            def first_step(step):
                if query_gradient is not None:
                    first_displacement["probe_gradient_dot_displacement"]=L.displacement_dot(model,base,query_gradient)
                    common.write_json(armout/"step1_displacement.json",first_displacement)
                if args.score_first_step:
                    values=L.score(model,tok,query,args.batch_size,armledger,args.precision)
                    common.write_jsonl(armout/"step1_selection_nll.jsonl",values)
            result=L.train_sft(model,tok,rows,retention,steps=args.steps,batch_size=args.batch_size,
                fixed_pad=args.fixed_pad,lr=args.lr,seed=args.seed,ledger=armledger,output=armout,
                retention_fraction=args.retention_fraction,precision=args.precision,callback=first_step)
            scores=L.score(model,tok,query,args.batch_size,armledger,args.precision)
            common.write_jsonl(armout/"selection_nll.jsonl",scores)
            save_checkpoint(armout/"final.pt",model,tok,format=FORMAT,family=family,arm=kind,
                            source_checkpoint_sha256=manifest["input_sha256"]["checkpoint"])
            result.update({"selection_mean_bits":balanced_bits(scores),"selection_by_length_bits":length_bits(scores),"ledger":armledger,
                           "checkpoint_sha256":common.sha(armout/"final.pt")})
            if query_gradient is not None:
                result.update({"probe_gradient_dot_displacement":L.displacement_dot(model,base,query_gradient),
                               "step1_probe_gradient_dot_displacement":first_displacement["probe_gradient_dot_displacement"]})
            common.write_json(armout/"result.json",result)
            arm_results[kind]=result
            del model
        training_budget=arm_results["family"]["ledger"]["sft_training"]["flop_proxy"]
        control_budget=arm_results["control"]["ledger"]["sft_training"]["flop_proxy"]
        if training_budget!=control_budget: raise AssertionError("Matched intervention compute differs")
        if arm_results["family"]["retention_schedule_sha256"]!=arm_results["control"]["retention_schedule_sha256"]:
            raise AssertionError("Paired intervention retention schedule differs")
        improvement=arm_results["control"]["selection_mean_bits"]-arm_results["family"]["selection_mean_bits"]
        absolute=balanced_bits(base_scores)-arm_results["family"]["selection_mean_bits"]
        surprise_values=[r for r in surprise if r["family"]==family]
        arms.append({"family":family,"utility_bits":improvement,"utility_bits_per_flop_proxy":improvement/training_budget,
                     "absolute_gain_bits":absolute,"conditional_bits_per_token":sum(r["bits"] for r in surprise_values)/sum(r["tokens"] for r in surprise_values),
                     "conditional_mean_bits":mean_bits(surprise_values),"training_flop_proxy":training_budget,
                     "matching_mean_distance":sum(r["distance"] for r in matches)/len(matches),
                     "control_unique_sources":len({r["ordinary_source"] for r in matches}),
                     "control_unique_contexts":len({r["ordinary_context"] for r in matches}),
                     "family_unique_contexts":len({r["family_context"] for r in matches}),
                     "control_max_source_multiplicity":max(Counter(r["ordinary_source"] for r in matches).values()),
                     "unmatched_source_length":sum(not r["exact_source_length_match"] for r in matches),
                     "unmatched_suffix_length":sum(not r["exact_suffix_length_match"] for r in matches),
                     "unmatched_depth":sum(not r["exact_depth_match"] for r in matches),
                     "utility_by_length":{b:arm_results["control"]["selection_by_length_bits"][b]-arm_results["family"]["selection_by_length_bits"][b] for b in length_bits(base_scores)},
                     "family_displacement_l2":arm_results["family"]["parameter_displacement_l2"],
                     "control_displacement_l2":arm_results["control"]["parameter_displacement_l2"]})
        if query_gradient is not None:
            arms[-1].update({"first_order_actual_displacement_prediction_bits":arm_results["control"]["probe_gradient_dot_displacement"]-arm_results["family"]["probe_gradient_dot_displacement"],
                             "step1_first_order_actual_displacement_prediction_bits":arm_results["control"]["step1_probe_gradient_dot_displacement"]-arm_results["family"]["step1_probe_gradient_dot_displacement"]})
        common.write_json(out/"calibration_progress.json",arms)
    uniform=list(families); random.Random(args.seed).shuffle(uniform)
    ranking={"frozen":True,"selection_sha256":manifest["input_sha256"]["selection"],
             "transfer":sorted(arms,key=lambda x:(-x["utility_bits"],x["family"])),
             "surprise":sorted(arms,key=lambda x:(-x["conditional_bits_per_token"],x["family"])),
             "uniform":uniform,"confirmation_opened":False,"top_k":args.top_k}
    common.write_json(out/"ranking.json",ranking)
    ledger["calibration_arms"]={f"{f}_{kind}":json.loads((out/f"{f}_{kind}"/"result.json").read_text())["ledger"] for f in families for kind in ("control","family")}
    finish(out,base,tok,{"ranking":ranking,"baseline_selection_mean_bits":balanced_bits(base_scores),"baseline_by_length_bits":length_bits(base_scores)},ledger,manifest,save=False)

def release_check(args,records=None):
    if records is None:
        # Inspect only public identity metadata before parsing proof payloads.
        roles=set()
        if "final_confirmation" in Path(args.data).name:roles.add("final_confirmation")
        elif "confirmation" in Path(args.data).name:roles.add("confirmation")
        manifest_path=Path(args.data).parent/"manifest.json"
        if manifest_path.exists():
            metadata=json.loads(manifest_path.read_text())
            data_hash=common.sha(args.data)
            roles.update(k for k,v in metadata.get("population_sha256",{}).items() if v==data_hash)
    else:roles={r["role"] for r in records}
    if roles & {"confirmation","final_confirmation"}:
        if not args.release: raise ValueError("Confirmation requires root's frozen release")
        release=json.loads(Path(args.release).read_text())
        if not release.get("frozen") or not release.get("no_further_selector_changes"):
            raise ValueError("Confirmation selection/configuration is not frozen")
        if common.sha(args.data) not in release["data_sha256"] or common.sha(args.checkpoint) not in release["checkpoints"]:
            raise ValueError("Confirmation checkpoint/data differs from release")
        if args.start!="full": raise ValueError("Confirmation decisive evaluation must be unaided")
        if "final_confirmation" in roles and not release.get("no_further_training"):
            raise ValueError("Final confirmation requires frozen training endpoints")

def legal_prefix_diagnostic(prompt,body):
    parts=body.split(";")
    accepted=0; reason="no complete lines"
    for i in range(len(parts)-1):
        try:
            independent_replay(prompt,";".join(parts[:i+1])+" ;")
            accepted=i+1
        except Exception as e:
            reason=str(e); break
    return {"legal_prefix_lines":accepted,"first_invalid_reason":reason}

def wilson(k,n):
    if not n:return [0.,1.]
    z=1.959963984540054; den=1+z*z/n
    center=(k/n+z*z/(2*n))/den
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def evaluate(args):
    out,model,tok,ledger,manifest=initialize(args)
    release_check(args)
    records=common.read_jsonl(args.data)
    release_check(args,records)
    episodes=[episode(r,0 if args.start=="full" else r["default_cut"] if args.start=="fragment" else max(L.admissible_cuts(r)),tok) for r in records]
    if any(len(e["prompt_ids"])+args.max_new_tokens>model.config.max_seq_len for e in episodes):
        raise ValueError("Fixed evaluation response budget exceeds context; reduce budget prospectively")
    nll=L.score(model,tok,L.encode_population(records,tok,"empty"),args.batch_size,ledger,args.precision)
    common.write_jsonl(out/"whole_proof_nll.jsonl",nll)
    if args.start!="full":
        common.write_jsonl(out/"conditional_nll.jsonl",L.score(model,tok,episodes,args.batch_size,ledger,args.precision))
    expanded=[e for e in episodes for _ in range(args.samples)]
    generated=L.sample(model,tok,expanded,temperature=args.temperature,seed=args.seed,max_new_tokens=args.max_new_tokens,
                       batch_size=args.batch_size,precision=args.precision,ledger=ledger)
    if args.temperature>0 and args.audit_likelihood:
        likelihood=L.audit_behavior(model,tok,generated,temperature=args.temperature,batch_size=args.microbatch,
                                    precision=args.precision,tolerance=args.max_logprob_error,ledger=ledger)
    else: likelihood=None
    attempts=[]; solved=set(); by_source={r["source_id"]:r for r in records}
    for i,row in enumerate(generated):
        verdict=judge(row,row["response_ids"],tok)
        attempt={k:v for k,v in row.items() if k!="prefix_state"}
        attempt.update({"attempt_index":i%args.samples,**verdict,
                        "length_bin":by_source[row["source_id"]]["length_bin"],
                        "context_id":by_source[row["source_id"]]["context_id"]})
        attempt.update(legal_prefix_diagnostic(row["prompt"],verdict.get("body","")))
        if verdict["reward"]:
            solved.add(row["source_id"])
            compression=compress_proof(row["prompt"],verdict["body"])
            if not strict_complete(row["prompt"],compression["compressed_proof"])["ok"]:
                raise AssertionError("Compressed witness fails independent strict check")
            attempt["compression"]=compression
        attempts.append(attempt)
    common.write_jsonl(out/"attempts.jsonl",attempts)
    bins={}
    for b in sorted({r["length_bin"] for r in records}):
        group=[r for r in records if r["length_bin"]==b]
        count=sum(r["source_id"] in solved for r in group)
        bins[b]={"solved":count,"n":len(group),"pass_at_k":count/len(group),"wilson95":wilson(count,len(group))}
    result={"strict_solved":len(solved),"n_theorems":len(records),"pass_at_k":len(solved)/len(records),
            "samples":args.samples,"start":args.start,"by_length":bins,"mean_whole_proof_bits":mean_bits(nll),
            "max_solved_reference_length":max((r["n_lines"] for r in records if r["source_id"] in solved),default=0),
            "solved_above_cap6":sum(r["n_lines"]>6 and r["source_id"] in solved for r in records),
            "likelihood_audit":likelihood,"failure_reasons":dict(Counter(a["reason"] for a in attempts if not a["reward"]))}
    finish(out,model,tok,result,ledger,manifest,save=False)

def rl(args):
    out,model,tok,ledger,manifest=initialize(args)
    records=training_rows(args.data)
    if args.families: records=[r for r in records if r["family"] in args.families]
    if not records: raise ValueError("No selected training sources")
    retention=L.encode_population(training_rows(args.retention),tok,"empty")
    admissible={r["source_id"]:L.admissible_cuts(r) for r in records}
    curriculum={r["source_id"]:{"cut":max(admissible[r["source_id"]]),"n":0,"success":0} for r in records}
    initial_map=None
    if args.initial_cuts:
        from initial_cuts import validate_map
        initial_map=json.loads(Path(args.initial_cuts).read_text())
        cuts=validate_map(initial_map,records,tok,manifest["input_sha256"]["checkpoint"],args.temperature,
                          args.max_new_tokens,args.precision,args.group_size)
        for source_id,cut in cuts.items():curriculum[source_id]["cut"]=cut
    rng=random.Random(args.seed)
    reset_rng=random.Random(args.seed+1000033)
    retention_rng=random.Random(args.seed+1000003)
    optimizer=L.optimizer_for(model,args.lr)
    generated_tokens=0; all_groups=[]; updates=[]
    largest_context=max(len(episode(r,max(admissible[r["source_id"]]),tok)["prompt_ids"]) for r in records)
    if largest_context+args.max_new_tokens>model.config.max_seq_len:
        raise ValueError("Fixed RL response budget does not fit every admissible context")
    largest_retention=max(len(r["prompt_ids"])+len(r["response_ids"]) for r in retention)
    step_bound=L.worst_case_rl_step_flops(model,args.prompts_per_step*args.group_size,largest_context,args.max_new_tokens,
        args.retention_batch if args.retention_weight else 0,largest_retention)
    milestones=sorted(args.flop_milestones or [])
    stop_reason="step_ceiling"
    for step in range(1,args.steps+1):
        if L.total_flop_proxy(ledger)>=args.max_flop_proxy:
            stop_reason="flop_proxy_ceiling";break
        if generated_tokens+args.prompts_per_step*args.group_size*args.max_new_tokens>args.max_generated_tokens:
            stop_reason="generated_token_ceiling";break
        before_flops=L.total_flop_proxy(ledger)
        selected=[records[rng.randrange(len(records))] for _ in range(args.prompts_per_step)]
        group_episodes=[]; group_cuts=[]
        for r in selected:
            cut=curriculum[r["source_id"]]["cut"] if args.start=="reset" and reset_rng.random()>=args.empty_fraction else 0
            group_cuts.append(cut)
            group_episodes.extend([episode(r,cut,tok)]*args.group_size)
        rows=L.sample(model,tok,group_episodes,temperature=args.temperature,seed=args.seed+step*997,
                       max_new_tokens=args.max_new_tokens,batch_size=args.batch_size,precision=args.precision,ledger=ledger)
        likelihood=L.audit_behavior(model,tok,rows,temperature=args.temperature,batch_size=args.microbatch,
                                    precision=args.precision,tolerance=args.max_logprob_error,ledger=ledger)
        generated_tokens+=sum(len(r["response_ids"]) for r in rows)
        stepgroups=[]
        for g,(record,cut) in enumerate(zip(selected,group_cuts)):
            group=rows[g*args.group_size:(g+1)*args.group_size]
            rewards=[]
            for r in group:
                verdict=judge(r,r["response_ids"],tok); r.update(verdict); rewards.append(verdict["reward"])
            total=sum(rewards)
            for r,reward in zip(group,rewards):
                r["advantage"]=reward-(total-reward)/(args.group_size-1)
            stats=curriculum[record["source_id"]]
            event={"step":step,"group":g,"source_id":record["source_id"],"family":record["family"],
                   "reset_index":cut,"source_lines":record["n_lines"],"successes":total,"group_size":args.group_size,
                   "mixed":0<total<args.group_size,"generated_tokens":sum(len(r["response_ids"]) for r in group),
                   "prefix_tokens":len(group[0]["prefix_ids"])}
            if args.start=="reset" and cut==stats["cut"]:
                stats["n"]+=args.group_size; stats["success"]+=total
                if stats["n"]>=args.curriculum_rollouts:
                    rate=stats["success"]/stats["n"]
                    positions=admissible[record["source_id"]]; pos=positions.index(stats["cut"])
                    newpos=max(0,pos-1) if rate>args.backward_threshold else min(len(positions)-1,pos+1) if rate<args.forward_threshold else pos
                    if newpos!=pos:
                        event["curriculum_move"]={"old":stats["cut"],"new":positions[newpos],"n":stats["n"],"success":stats["success"]}
                        stats.update(cut=positions[newpos],n=0,success=0)
            stepgroups.append(event)
        retained=[retention[retention_rng.randrange(len(retention))] for _ in range(args.retention_batch)]
        update=L.policy_update(model,tok,rows,optimizer,retained,temperature=args.temperature,max_new_tokens=args.max_new_tokens,
            microbatch=args.microbatch,retention_weight=args.retention_weight,precision=args.precision,ledger=ledger)
        update.update({"step":step,"generated_tokens_total":generated_tokens,"mixed_groups":sum(g["mixed"] for g in stepgroups),
                       "active_advantage_tokens":sum(len(r["response_ids"]) for r in rows if r["advantage"]),"likelihood":likelihood,
                       "cumulative_flop_proxy":L.total_flop_proxy(ledger),"step_flop_proxy":L.total_flop_proxy(ledger)-before_flops,
                       "retention_examples":len(retained),"retention_response_tokens":sum(len(r["response_ids"]) for r in retained)})
        if update["step_flop_proxy"]>step_bound:
            raise AssertionError("Observed step proxy exceeds declared worst-case bound")
        updates.append(update); all_groups.extend(stepgroups)
        with (out/"rollouts.jsonl").open("a") as f:
            for r in rows: f.write(json.dumps({"step":step,**{k:v for k,v in r.items() if k!="prefix_state"}})+"\n")
        with (out/"groups.jsonl").open("a") as f:
            for g in stepgroups:f.write(json.dumps(g)+"\n")
        with (out/"updates.jsonl").open("a") as f:f.write(json.dumps(update)+"\n")
        common.write_json(out/"progress.json",update)
        common.write_json(out/"curriculum.json",curriculum)
        if step==1 or step%10==0: print(json.dumps(update),flush=True)
        if args.save_every and step%args.save_every==0: save_checkpoint(out/f"step_{step:06d}.pt",model,tok,step=step,format=FORMAT)
        while milestones and L.total_flop_proxy(ledger)>=milestones[0]:
            milestone=milestones.pop(0)
            save_checkpoint(out/f"flops_{milestone:.0f}.pt",model,tok,step=step,format=FORMAT,
                            requested_flop_proxy=milestone,actual_flop_proxy=L.total_flop_proxy(ledger))
    result={"steps":len(updates),"generated_tokens":generated_tokens,"mixed_groups":sum(g["mixed"] for g in all_groups),
            "n_groups":len(all_groups),"active_advantage_tokens":sum(u["active_advantage_tokens"] for u in updates),
            "start":args.start,"families":sorted({r["family"] for r in records}),"n_training_sources":len(records),
            "curriculum":curriculum,"normalization":"fixed max_new_tokens shared by full and reset arms"}
    result.update({"stop_reason":stop_reason,"max_flop_proxy":args.max_flop_proxy,
                   "flop_proxy_overshoot":max(0,L.total_flop_proxy(ledger)-args.max_flop_proxy),
                   "worst_case_one_step_flop_proxy":step_bound,
                   "overshoot_policy":"Complete the current on-policy update; stop before the next update after crossing the ceiling.",
                   "retention_total_examples":sum(u["retention_examples"] for u in updates)})
    result.update({"controller_initialization":"difficulty_pilot" if initial_map else "original_latest_admissible",
                   "initial_cuts_sha256":manifest["input_sha256"].get("initial_cuts"),
                   "initialization_pilot_flop_proxy":initial_map.get("total_pilot_flop_proxy",0) if initial_map else 0,
                   "initialization_pilot_not_in_direct_training_ceiling":True})
    finish(out,model,tok,result,ledger,manifest)

def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    for name in ("adapt","calibrate","intervene","evaluate","rl"):
        s=sub.add_parser(name)
        s.add_argument("--checkpoint",required=True);s.add_argument("--data",required=True);s.add_argument("--output",required=True)
        s.add_argument("--device",default="cpu");s.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
        s.add_argument("--seed",type=int,default=20260910);s.add_argument("--cpu-threads",type=int,default=1)
        s.add_argument("--batch-size",type=int,default=32);s.add_argument("--microbatch",type=int,default=16)
        if name in ("adapt","calibrate","intervene","rl"):
            s.add_argument("--steps",type=int,default=100);s.add_argument("--lr",type=float,default=5e-5)
            s.add_argument("--save-every",type=int,default=0)
        if name in ("adapt","calibrate","intervene"):s.add_argument("--fixed-pad",type=int,default=512)
        if name in ("calibrate","intervene","rl"):
            s.add_argument("--retention",required=True);s.add_argument("--families",nargs="*")
        if name in ("calibrate","intervene"):
            s.add_argument("--retention-fraction",type=float,default=.5)
        if name=="intervene":s.add_argument("--sampling",choices=("balanced-families","uniform-sources"),default="balanced-families")
        if name=="calibrate":
            s.add_argument("--selection",required=True)
            s.add_argument("--score-first-step",action="store_true");s.add_argument("--top-k",type=int,default=3)
            s.add_argument("--actual-displacement",action="store_true")
        if name in ("evaluate","rl"):
            s.add_argument("--start",choices=("full","reset","fragment") if name=="evaluate" else ("full","reset"),default="full");s.add_argument("--temperature",type=float,default=.8)
            s.add_argument("--max-new-tokens",type=int,default=512);s.add_argument("--max-logprob-error",type=float,default=.001)
        if name=="evaluate":
            s.add_argument("--samples",type=int,default=8);s.add_argument("--release")
            s.add_argument("--audit-likelihood",action="store_true")
        if name=="rl":
            s.add_argument("--initial-cuts")
            s.add_argument("--prompts-per-step",type=int,default=4);s.add_argument("--group-size",type=int,default=8)
            s.add_argument("--retention-batch",type=int,default=16);s.add_argument("--retention-weight",type=float,default=.1)
            s.add_argument("--empty-fraction",type=float,default=.2);s.add_argument("--max-generated-tokens",type=int,default=2000000)
            s.add_argument("--max-flop-proxy",type=float,default=1e30)
            s.add_argument("--flop-milestones",type=float,nargs="*",default=[])
            s.add_argument("--curriculum-rollouts",type=int,default=64)
            s.add_argument("--backward-threshold",type=float,default=.8);s.add_argument("--forward-threshold",type=float,default=.1)
    return p

if __name__=="__main__":
    args=parser().parse_args()
    globals()[args.command](args)
