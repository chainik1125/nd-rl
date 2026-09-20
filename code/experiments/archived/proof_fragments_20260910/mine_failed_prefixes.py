"""CPU-only mining of closed top-level local certificates from failed attempts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import common
import nd_environment as E
from nd_data import formula_text
from populations import identities,load_guards,existing_exposure,family_of
from fragment_experiment import training_rows
from audit_main_runtime import log_prefix


def role_for_context(context):
    bucket=int(context[:12],16)%20
    return ("candidate" if bucket<6 else "adaptation" if bucket<10 else "retention" if bucket<12
            else "selection" if bucket<15 else "confirmation" if bucket<17 else "final_confirmation")


def local_prefixes(prompt,decoded_body):
    """Stop at the first invalid inference; never repair, renumber or rescope."""
    premises,original_goal=E.parse_prompt(prompt)
    words=decoded_body.split();prefix=[];results=[];counts=Counter();accepted=0;first_invalid=None
    for word in words:
        if word=="QED":break
        prefix.append(word)
        if word!=";":continue
        if accepted>=6:break
        try:state=E.independent_replay(prompt," ".join(prefix))
        except Exception as exc:
            first_invalid=str(exc);counts["first_invalid_inference"]+=1;break
        accepted+=1;last=state["lines"][-1]
        if state["active_assumptions"] or last["depth"]!=0:
            counts["open_scope_prefix"]+=1;continue
        if last["rule"]=="PR" or last["formula"] in premises:
            counts["merely_premise_conclusion"]+=1;continue
        if last["formula"]==original_goal:
            counts["original_target_conclusion"]+=1;continue
        new_prompt="THM "+" , ".join(formula_text(f) for f in premises)+" SEQ "+formula_text(last["formula"])+" PRF"
        if E.parse_prompt(new_prompt)[0]!=premises:raise AssertionError("Mining changed the original premises")
        body=" ".join(prefix)+" QED";verdict=E.strict_complete(new_prompt,body)
        if not verdict["ok"] or not verdict["independent_ok"] or not verdict["n_lines"]==accepted<=6:
            raise AssertionError("Closed valid prefix failed either full-certificate verifier")
        raw={"prompt":new_prompt,"body":body,"text":new_prompt+"\n"+body,"n_lines":accepted,
             "rules":[line["rule"] for line in state["lines"]],"max_depth":max(line["depth"] for line in state["lines"]),
             "n_premises":len(premises),"original_target_was_replaced_by_local_conclusion":True}
        results.append({**raw,**identities(new_prompt),"family":family_of(raw),"verdict":verdict})
    return results,{"longest_checked_legal_prefix_at_most6":accepted,"first_invalid_reason":first_invalid,
                    "prefix_checks":dict(counts),"decoded_complete_inference_boundaries":decoded_body.count(";"),
                    "decoded_literal_tokens":len(words)}


def decode_failed_attempt(row,tok):
    response=row["response_ids"];final=tok.stoi["<FINAL>"]
    if response.count(final)!=1:return None,"ambiguous_or_missing_FINAL"
    k=response.index(final);pre=response[:k]
    if pre and (pre[0]!=tok.stoi["<THINK>"] or pre[-1]!=tok.stoi["</THINK>"]):return None,"malformed_scratch_boundary"
    return tok.decode_body(response[k+1:]),None


def run(args):
    started=time.time();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Diagnostic output must be empty")
    if not 1<=args.through_step<=1000:raise ValueError("This bounded diagnostic permits only the first1000 main updates")
    run_dir=Path(args.run).resolve();manifest_path=run_dir/"manifest.json";main=json.loads(manifest_path.read_text());main_args=main["args"]
    if main_args.get("command")!="rl" or main_args.get("start")!="full":raise ValueError("Only full-start main RL rollouts are eligible")
    for path,digest in main["dependency_sha256"].items():
        if common.sha(path)!=digest:raise ValueError("Producer dependency changed")
    for path,digest in main["code_sha256"].items():
        if common.sha(path)!=digest:raise ValueError("Producer snapshot changed")
    if [h for p,h in main["code_sha256"].items() if Path(p).name=="nd_environment.py"]!=[common.sha(E.__file__)]:
        raise ValueError("Current checker/serialization differs from the producer")
    if main["input_sha256"]["data"]!=common.sha(main_args["data"]):raise ValueError("Producer bank changed")
    parent_records=training_rows(main_args["data"]);parents={r["source_id"]:r for r in parent_records}
    if any(r["role"]!="candidate" for r in parent_records):raise ValueError("Producer bank must contain only candidate sources")
    population=Path(args.populations).resolve();pop_manifest_path=population/"manifest.json"
    pop_manifest=json.loads(pop_manifest_path.read_text())
    if pop_manifest.get("status")!="released_complete_primary":raise ValueError("Known populations must be the frozen primary release")
    guard_path=Path(args.guard_json).resolve();baseline_path=Path(args.baseline_data).resolve()
    for path in (guard_path,baseline_path):
        if pop_manifest["input_sha256"].get(str(path))!=common.sha(path):raise ValueError("Baseline/guard input is not bound by the frozen release")
    known_paths=[]
    for role in ("candidate","adaptation","retention"):
        path=population/f"{role}.jsonl"
        if common.sha(path)!=pop_manifest["population_sha256"][role]:raise ValueError("Known training-role population changed")
        known_paths.append(path)
    known_theorems,_=existing_exposure(known_paths+[baseline_path]);guards=load_guards(guard_path)
    updates,updates_hash=log_prefix(run_dir/"updates.jsonl",args.through_step)
    if len(updates)!=args.through_step or [u["step"] for u in updates]!=list(range(1,args.through_step+1)):
        raise ValueError("Requested main update prefix is not completed")
    attempts,attempts_hash=log_prefix(run_dir/"rollouts.jsonl",args.through_step)
    expected=args.through_step*main_args["prompts_per_step"]*main_args["group_size"]
    if len(attempts)!=expected:raise ValueError("Incomplete rollout prefix")
    tok=E.Tokenizer();counts=Counter();reasons=Counter();records=[];attempt_audits=[];seen=set();novel_occurrences=0
    for index,row in enumerate(attempts):
        counts["total_rollouts"]+=1
        if row["reward"]:counts["successful_original_attempts_skipped"]+=1;continue
        counts["failed_original_attempts"]+=1
        parent=parents[row["source_id"]];ep=E.episode(parent,0,tok)
        for key in ("prompt_ids","prefix_ids","original_prompt_ids","prompt","source_id","reset_index","source_lines","format"):
            if row[key]!=ep[key]:raise ValueError("Failed attempt differs from its exact original empty-start state")
        original=E.judge(ep,row["response_ids"],tok)
        if original["reward"]!=0:raise ValueError("Original target is not actually a failed attempt")
        body,decode_error=decode_failed_attempt(row,tok)
        audit={"rollout_row_index":index,"parent_source_id":row["source_id"],"step":row["step"],
               "action_sha256":common.digest(row["response_ids"]),"original_failure_reason":original.get("reason"),
               "original_sampled_tokens":len(row["response_ids"]),"original_relative_line_markers":row["response_ids"].count(tok.stoi["<LINE>"])}
        if decode_error:
            reasons[decode_error]+=1;attempt_audits.append({**audit,"extraction_rejection":decode_error});continue
        local,diagnostic=local_prefixes(parent["prompt"],body);audit.update(diagnostic);audit["eligible_local_prefixes"]=len(local)
        counts.update(diagnostic["prefix_checks"]);counts["strictly_verified_local_prefix_occurrences"]+=len(local)
        for candidate in local:
            if role_for_context(candidate["context_id"])!="candidate":reasons["context_outside_candidate_partition"]+=1;continue
            if any(k is not None and k in group for k,group in zip(candidate["guard_ids"],guards)):
                reasons["blind_textbook_guard_collision"]+=1;continue
            theorem=candidate["theorem_id"]
            if theorem in known_theorems:reasons["canonical_theorem_already_known"]+=1;continue
            novel_occurrences+=1
            if theorem in seen:reasons["canonical_duplicate_within_mining"]+=1;continue
            seen.add(theorem);source_id=common.digest([candidate["prompt"],candidate["body"]])
            records.append({**candidate,"source_id":source_id,"role":"diagnostic_failed_rollout","diagnostic_only":True,"training_ready":False,
                "length_bin":"2-6","provenance":{"kind":"online_failed_rollout_closed_local_prefix","parent_source_id":row["source_id"],
                    "parent_source_context_id":parent["context_id"],"parent_full_source_lines":parent["n_lines"],"original_target_prompt":parent["prompt"],
                    "original_target_reward":0,"main_run_manifest_sha256":common.sha(manifest_path),"initial_checkpoint_sha256":main["input_sha256"]["checkpoint"],
                    "completed_policy_updates_before_attempt":row["step"]-1,"step":row["step"],"rollout_row_index":index,"rollout_prefix_sha256":attempts_hash,
                    "sampling_seed":main_args["seed"]+row["step"]*997,"sampling_temperature":main_args["temperature"],
                    "sampling_max_new_tokens":main_args["max_new_tokens"],
                    "action_sha256":audit["action_sha256"],"original_action_ids":row["response_ids"],"original_decoded_body":body,
                    "original_failed_full_attempt_complete_inference_boundaries":body.count(";"),
                    "original_failed_full_attempt_sampled_tokens":len(row["response_ids"]),"original_failure_reason":original.get("reason"),
                    "full_source_lines":candidate["n_lines"],"source_body_sha256":common.digest(candidate["body"]),
                    "external_reset_source_cap":6,"mined_complete_certificate_cap":6,"original_premises_preserved":True,"closed_top_level_prefix":True}})
        attempt_audits.append(audit)
    output.mkdir(parents=True,exist_ok=True)
    common.write_jsonl(output/"diagnostic_certificates.jsonl",records);common.write_jsonl(output/"failed_attempt_audit.jsonl",attempt_audits)
    result={"status":"diagnostic_only_not_training_ready","through_step":args.through_step,"counts":dict(counts),"rejection_counts":dict(reasons),
            "novel_prefix_occurrences_before_canonical_dedup":novel_occurrences,"novel_canonical_certificates":len(records),
            "novel_contexts":len({r["context_id"] for r in records}),"novel_families":dict(Counter(r["family"] for r in records)),
            "novel_lengths":dict(Counter(r["n_lines"] for r in records)),"known_training_canonical_theorems":len(known_theorems),
            "policy_updates":0,"new_model_rollouts":0,"model_flop_proxy":0,"elapsed_seconds":time.time()-started,
            "parent_main_training_flop_proxy_through_step":updates[-1]["cumulative_flop_proxy"],
            "parent_rollout_cost_already_charged_in_main_ledger":True,
            "provenance_note":"Only retained complete certificates are capped at6; complete original failed-attempt lengths and raw actions are retained separately.",
            "no_open_scope_rewriting":True,"original_failed_target_success_not_changed":True,"confirmation_payloads_opened":False,"textbook_payloads_opened":False}
    manifest={"command":"mine_failed_prefixes","args":vars(args),"status":result["status"],
              "role_partition":"int(context_sha256[:12],16)%20; candidate iff bucket<6, identical to frozen collector",
              "novelty_reference":"union of baseline direct-SFT and released candidate/adaptation/retention canonical theorems",
              "input_sha256":{str(manifest_path):common.sha(manifest_path),str(pop_manifest_path):common.sha(pop_manifest_path),
                              str(guard_path):common.sha(guard_path),str(baseline_path):common.sha(baseline_path),
                              **{str(p):common.sha(p) for p in known_paths}},"log_prefix_sha256":{"updates":updates_hash,"rollouts":attempts_hash},
              "code_sha256":{str(p):common.sha(p) for p in Path(__file__).parent.glob("*.py")},"dependency_sha256":common.dependency_hashes()}
    common.write_json(output/"manifest.json",manifest);common.write_json(output/"result.json",result)
    common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json"),"certificates_sha256":common.sha(output/"diagnostic_certificates.jsonl")})
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("run","populations","baseline-data","guard-json","output"):p.add_argument("--"+flag,required=True)
    p.add_argument("--through-step",type=int,default=1000);run(p.parse_args())
