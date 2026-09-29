"""Prespecified outcome-variance initialization; original controller unchanged."""
import common
from nd_environment import FORMAT,episode

RULE="max_k8_minus_k_earliest_tie_v1"

def choose_cut(successes,group_size=8):
    if group_size!=8 or not successes:raise ValueError("Initial-cut pilot needs exactly eight attempts per admissible cut")
    values={int(k):int(v) for k,v in successes.items()}
    if any(not 0<=v<=8 for v in values.values()):raise ValueError("Invalid pilot success count")
    scores={k:v*(8-v) for k,v in values.items()}
    maximum=max(scores.values())
    if maximum:
        cut=min(k for k,v in scores.items() if v==maximum);reason="observed_mixed_frontier"
    else:
        solved=[k for k,v in values.items() if v==8]
        cut=min(solved) if solved else max(values)
        reason="fallback_earliest_all_success" if solved else "fallback_latest_all_failed"
    return {"cut":cut,"score":scores[cut],"reason":reason,"observed_mixed_frontier":maximum>0,
            "successes":values,"variance_estimate":scores[cut]/56}

def state_hash(row):
    return common.digest({k:row[k] for k in ("prompt_ids","original_prompt_ids","prefix_ids","source_id","reset_index","source_lines","format")})

def validate_map(value,records,tok,checkpoint_hash,temperature,max_new_tokens,precision,group_size=8):
    import learning as L
    import nd_environment as E
    expected={"status":"frozen_initial_cuts","rule":RULE,"format":FORMAT,"checkpoint_sha256":checkpoint_hash,
              "temperature":temperature,"max_new_tokens":max_new_tokens,"precision":precision,"group_size":group_size,
              "sampler_sha256":common.sha(L.__file__),"environment_sha256":common.sha(E.__file__)}
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError("Initial-cut map differs from the frozen policy/sampling contract")
    if value.get("counters_initialized_to_zero") is not True:raise ValueError("Pilot outcomes must not preload curriculum counters")
    result={}
    for record in records:
        entry=value["sources"].get(record["source_id"])
        if entry is None:raise ValueError("Initial-cut map does not cover a bank source")
        cut=entry["cut"]
        if type(cut) is not int:raise ValueError("Noninteger initial cut")
        ep=episode(record,cut,tok)
        if entry["chosen_state_sha256"]!=state_hash(ep):raise ValueError("Mapped source/prefix state differs")
        result[record["source_id"]]=cut
    return result
