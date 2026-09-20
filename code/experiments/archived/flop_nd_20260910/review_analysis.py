"""Independent bounded analysis audit; development and completed pilot only."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import analyze_lm as A
import common
from nd_environment import strict_complete

ROOT = Path(__file__).resolve().parents[1]

def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def main():
    base = ROOT / "results/baseline_development_greedy_v1"
    audit_path = ROOT / "data/v1/audit/development.jsonl"
    audit = {r["source_id"]: r for r in lines(audit_path)}
    raw = lines(base / "attempts.jsonl")
    solved = [r for r in raw if r["reward"]]
    assert len(raw) == len(audit) == 384
    for r in raw:
        assert r["generated_tokens"] == len(r["response_ids"]) <= 512
        assert r["reset_index"] == 0
        if r["reward"]:
            assert strict_complete(r["prompt"], r["body"])["ok"]
            assert strict_complete(r["prompt"], r["compression"]["compressed_proof"])["ok"]
    expected = {"theorems":384,"successes":len(solved),"tokens":sum(len(r["response_ids"]) for r in raw),
                "failed_attempts":len(raw)-len(solved),
                "successes_by_reference_bin":dict(Counter(audit[r["source_id"]]["length_bin"] for r in solved)),
                "pruned_histogram":dict(Counter(r["compression"]["dependency_pruned_length"] for r in solved)),
                "compressed_histogram":dict(Counter(r["compression"]["compressed_length"] for r in solved)),
                "reference_over6_successes":sum(audit[r["source_id"]]["n_lines"]>6 for r in solved),
                "reference_over6_theorems":sum(r["n_lines"]>6 for r in audit.values()),
                "pruned_over6_successes":sum(r["compression"]["dependency_pruned_length"]>6 for r in solved)}
    parsed = A.read_evaluation(base,audit,common.sha(audit_path))
    summary = A.summary_rows(parsed["rows"],list(audit))
    assert summary["first"] == summary["any"] == expected["successes"] == 81
    assert summary["tokens"] == expected["tokens"] == 41116
    assert summary["shortest_pruned_histogram"] == expected["pruned_histogram"]
    assert summary["any_pruned_over6"] == expected["pruned_over6_successes"] == 2
    assert expected["reference_over6_successes"] == 24 and expected["reference_over6_theorems"] == 288
    # Independent whole-context draw, expanded to theorem indices each time.
    values = np.array([1.,-1.,.5,0.,1.,0.])
    contexts = ["A","A","A","B","C","C"]
    members = [[0,1,2],[3],[4,5]]
    rng = np.random.default_rng(191)
    boot=[]
    for _ in range(4097):
        selected=[i for c in rng.integers(0,3,size=3) for i in members[c]]
        boot.append(float(np.mean(values[selected])))
    reported=A.paired_context_bootstrap(values,contexts,4097,191)
    assert reported["estimate"] == float(values.mean())
    assert reported["ci95_context_bootstrap"] == np.quantile(boot,[.025,.975]).tolist()
    # Assert mean-before-bootstrap by exercising full contrast assembly with
    # independently specified policy outcomes; no checkpoint or model needed.
    vectors={"a0":[1,1,0],"b0":[0,1,1],"a1":[1,0,1],"b1":[0,0,0]}
    fakeaudit=[{"source_id":str(i),"prompt":"p","context_id":c,"n_lines":7,"length_bin":"7-9","family":"f"} for i,c in enumerate(["A","A","B"])]
    def fake_read(path,audit,*args,**kwargs):
        rows={}
        for i,x in enumerate(vectors[path]):
            row={k:x for k in ("first","any","attempt_successes","any_pruned_over6","shortest_found_pruned_over6","any_compressed_over6","shortest_found_compressed_over6")}
            row.update(attempts=1,tokens=10,shortest_pruned=7 if x else None,shortest_compressed=7 if x else None,max_pruned=7 if x else None)
            rows[str(i)]=row
        return {"rows":rows,"profile":{"same":True}}
    with tempfile.TemporaryDirectory() as tmp:
        ap=Path(tmp)/"audit.jsonl";common.write_jsonl(ap,fakeaudit)
        cfg={"audit":str(ap),"evaluations":{k:k for k in vectors},"contrasts":{"effect":{"seed0":{"a0":1,"b0":-1},"seed1":{"a1":1,"b1":-1}}},"bootstrap_replicates":300,"bootstrap_seed":9}
        with patch.object(A,"read_evaluation",fake_read):
            contrasts=A.evaluation_analysis(cfg)["contrasts"]["effect"]["all"]["first"]
        assert contrasts["seed0"]["estimate"] == 0
        assert contrasts["seed1"]["estimate"] == 2/3
        assert contrasts["mean_across_seeds"] == A.paired_context_bootstrap([1,0,0],["A","A","B"],300,9)
    pilot=ROOT/"results/pilot_ppo_reset_v1"
    attempts=lines(pilot/"attempts.jsonl"); groups=lines(pilot/"groups.jsonl"); updates=lines(pilot/"updates.jsonl")
    result=json.loads((pilot/"result.json").read_text())
    group_raw=defaultdict(list)
    for r in attempts:group_raw[r["step"],r["group"]].append(r)
    assert len(group_raw)==len(groups)==400 and len(updates)==result["step"]==100
    counts=Counter(); successes=Counter(); mixed=Counter()
    for g in groups:
        rows=group_raw[g["step"],g["group"]]
        assert len(rows)==g["group_size"]==8
        assert {r["attempt_index"] for r in rows}==set(range(8))
        assert len({(r["source_id"],r["reset_index"],r["training_kind"]) for r in rows})==1
        n=sum(r["reward"] for r in rows)
        assert n==g["successes"] and bool(0<n<8)==g["mixed"]
        assert sum(len(r["response_ids"]) for r in rows)==g["generated_tokens"]
        kind=g["training_kind"]+("_assisted" if g["reset_index"] else "_empty")
        counts[kind]+=1;successes[kind]+=n;mixed[kind]+=bool(0<n<8)
    for u in updates:
        rows=[r for key,rs in group_raw.items() if key[0]==u["step"] for r in rs]
        assert sum(len(r["response_ids"]) for r in rows)==u["advantages"]["action_tokens"]
        for k in ("source","target"):
            selected=[r for r in rows if r["training_kind"]==k]
            assert len(selected)==u["outcome"][k]["attempts"]
            assert sum(r["reward"] for r in selected)==u["outcome"][k]["successes"]
    token_count=sum(len(r["response_ids"]) for r in attempts)
    assert token_count==result["generated_tokens"]==221381
    assert sum(u["step_flop_proxy"] for u in updates)==result["cumulative_flop_proxy"]
    analyzed=A.training_analysis(pilot)
    b=analyzed["curves"][0]
    assert b["groups"]==counts and b["group_successes"]==successes and b["mixed_groups"]==mixed
    assert b["target_successes"]==130 and b["source_successes"]==1290
    assert b["generated_tokens"]==token_count
    p={"updates":100,"groups":dict(counts),"group_successes":dict(successes),"mixed_groups":dict(mixed),
       "generated_tokens":token_count,"source_successes":1290,"target_successes":130,
       "unique_witnesses":analyzed["unique_witnesses"],"unique_targets":analyzed["unique_discovered_targets"],
       "curriculum_decisions":sum("curriculum_window" in g for g in groups),
       "flop_proxy":result["cumulative_flop_proxy"]}
    assert p["unique_witnesses"]==27 and p["unique_targets"]==23 and p["curriculum_decisions"]==0
    malformed={}
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        for name in ("manifest.json","curriculum.json","discoveries.jsonl"):
            if (pilot/name).exists():(d/name).symlink_to(pilot/name)
        def reject(label, changed_result=None, changed_groups=None, changed_updates=None):
            common.write_json(d/"result.json",changed_result or result)
            common.write_jsonl(d/"groups.jsonl",groups if changed_groups is None else changed_groups)
            common.write_jsonl(d/"updates.jsonl",updates if changed_updates is None else changed_updates)
            try:A.training_analysis(d)
            except ValueError as exc:malformed[label]=str(exc)
            else:raise AssertionError("Malformed ledger accepted: "+label)
        reject("missing_group",changed_groups=groups[1:])
        reject("duplicate_group",changed_groups=groups+[groups[0]])
        reject("truncated_updates",changed_updates=updates[:-1])
        wrong=copy.deepcopy(result);wrong["generated_tokens"]+=1
        reject("endpoint_tokens",changed_result=wrong)
        wrongg=copy.deepcopy(groups);wrongg[0]["generated_tokens"]+=1
        reject("group_tokens",changed_groups=wrongg)
    try:A.read_evaluation(base,audit,"altered-audit-digest")
    except ValueError as exc:malformed["audit_digest"]=str(exc)
    else:raise AssertionError("Altered audit digest accepted")
    report={"ok":True,"analysis_sha256":common.sha(A.__file__),"baseline":expected,"pilot":p,
            "independent_cluster_bootstrap":reported,"paired_seed_contrasts":contrasts,
            "malformed_artifacts_rejected":malformed,
            "provenance":{"audit":common.sha(audit_path),"baseline_attempts":common.sha(base/"attempts.jsonl"),"pilot_attempts":common.sha(pilot/"attempts.jsonl")},
            "scope":"CPU only; baseline development and completed 100-update pilot only; no final outcomes or checkpoints loaded."}
    common.write_json(ROOT/"reports/review_analysis.json",report)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
