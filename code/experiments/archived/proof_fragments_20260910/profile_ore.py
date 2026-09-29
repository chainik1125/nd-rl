"""Bounded CPU-only support diagnostic; outputs are never training sources."""
from collections import Counter
import random
import time
import common
from nd_data import generate_candidate
from populations import identities, load_guards, family_of

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--guard-json")
    p.add_argument("--attempts-per-cap",type=int,default=50000);args=p.parse_args()
    guards=load_guards(args.guard_json);results=[];examples=[]
    for cap in (6,12):
        rng=random.Random(928714+cap);counts=Counter();unique=set();contexts=set();start=time.monotonic()
        for attempt in range(args.attempts_per_cap):
            raw=generate_candidate(rng,max_lines=cap,formula_limit=23)
            if not raw:continue
            counts["accepted_all_rules"]+=1
            if raw["rules"][-1]!="ORE":continue
            counts["ORE_any_pruned_length"]+=1
            if raw["n_lines"]>6:continue
            counts["ORE_pruned_cap6"]+=1
            ids=identities(raw["prompt"])
            if any(k in protected for k,protected in zip(ids["guard_ids"],guards) if k is not None):
                counts["ORE_cap6_guard_rejected"]+=1;continue
            counts["ORE_cap6_guard_passed"]+=1
            if int(ids["context_id"][:12],16)%20>=6:
                counts["ORE_cap6_other_role_bucket"]+=1;continue
            counts["ORE_cap6_candidate_bucket"]+=1
            unique.add(ids["theorem_id"]);contexts.add(ids["context_id"])
            if len(examples)<20:examples.append({"diagnostic_only_never_train":True,"raw":raw,"identities":ids,"generation_cap":cap})
        results.append({"generation_cap":cap,"attempts":args.attempts_per_cap,"counts":dict(counts),
                        "unique_candidate_theorems":len(unique),"unique_candidate_contexts":len(contexts),
                        "elapsed_seconds":time.monotonic()-start})
        print(common.json.dumps(results[-1]),flush=True)
    common.write_json(args.output,{"diagnostic_only_never_train":True,"results":results,"examples":examples})

if __name__=="__main__":main()
