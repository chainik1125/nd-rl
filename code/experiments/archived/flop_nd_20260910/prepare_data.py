"""Freeze cap-six sources and fresh theorem-only training/evaluation pools.

Long references are emitted only under audit/. They never enter trainer inputs.
Splits are assigned by canonical context before inspecting model performance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import time
import common
from nd_data import canonical_theorem, sequent_parts, generate_candidate
from nd_environment import strict_complete, Tokenizer

PRIOR = common.ROOT.parent / "proof-fragments-20260910-1800"
OLDER = common.ROOT.parent / "r1-sprint-20260909-1625"

def identities(prompt):
    premises, goal = sequent_parts(prompt)
    canonical = canonical_theorem(prompt)
    context = canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF") if premises else re.sub(r"\b[PQRS]\b", "A", goal)
    return common.digest(canonical), common.digest(context), (
        hashlib.sha256(canonical.encode()).hexdigest(),
        hashlib.sha256(context.encode()).hexdigest() if premises else None,
        hashlib.sha256(context.encode()).hexdigest() if not premises else None)

def prompt_of(row):
    return row.get("prompt") or row["text"].split("PRF", 1)[0] + "PRF"

def family(raw):
    r = raw["rules"][-1]
    return ("IMPI_nested" if raw["max_depth"] > 1 else "IMPI_flat") if r == "IMPI" else r

def length_bin(n):
    return "2-6" if n <= 6 else "7-9" if n <= 9 else "10-12" if n <= 12 else "13-20"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=202609101934)
    p.add_argument("--max-attempts", type=int, default=1500000)
    p.add_argument("--targets-per-bin", type=int, default=256)
    p.add_argument("--dev-per-bin", type=int, default=96)
    p.add_argument("--final-per-bin", type=int, default=192)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "audit").mkdir()
    rng = random.Random(args.seed)
    source_paths = [PRIOR / "data/released_v1" / f for f in ("candidate.jsonl", "diagnostic_ore.jsonl", "adaptation.jsonl", "retention.jsonl")]
    exposure_paths = [OLDER / "data/sft_cap6.jsonl"] + source_paths
    exposure_t, exposure_c = set(), set()
    start = time.monotonic()
    for path in exposure_paths:
        for row in common.read_jsonl(path):
            t, c, _ = identities(prompt_of(row)); exposure_t.add(t); exposure_c.add(c)
        print(json.dumps({"phase":"exclusions","file":str(path),"theorems":len(exposure_t),"contexts":len(exposure_c)}),flush=True)
    guards_path = OLDER / "data/all_textbook_guards.json"
    g = json.loads(guards_path.read_text())
    guards = [set(g[k]) for k in ("canonical_theorem_sha256", "nonempty_premise_context_sha256", "closed_formula_skeleton_sha256")]
    candidates = common.read_jsonl(source_paths[0]) + common.read_jsonl(source_paths[1])
    by_family = defaultdict(list)
    for row in candidates: by_family[row["family"]].append(row)
    sources = []
    for f, rows in sorted(by_family.items()):
        rows = sorted(rows,key=lambda r:r["source_id"])
        chosen = rng.sample(rows,min(8,len(rows)))
        for r in chosen:
            verdict = strict_complete(r["prompt"],r["body"])
            assert verdict["ok"] and verdict["n_lines"] <= 6
            sources.append({**r,"role":"training_source","selection":"uniform_within_terminal_rule_family_seed202609101934"})
    retention = common.read_jsonl(source_paths[-1])
    for r in retention:
        v = strict_complete(r["prompt"],r["body"]); assert v["ok"] and v["n_lines"]<=6
    common.write_jsonl(out/"sources.jsonl",sources)
    common.write_jsonl(out/"retention.jsonl",retention)
    quotas = {"targets":{"7-9":args.targets_per_bin,"10-12":args.targets_per_bin},
              "development":{b:args.dev_per_bin for b in ("2-6","7-9","10-12","13-20")},
              "final":{b:args.final_per_bin for b in ("2-6","7-9","10-12","13-20")}}
    populations={k:[] for k in quotas};counts={k:Counter() for k in quotas}
    seen=set();rejected=Counter();tok=Tokenizer()
    def done():return all(counts[r][b]>=q for r,qs in quotas.items() for b,q in qs.items())
    attempt=0
    for attempt in range(1,args.max_attempts+1):
        if done():break
        raw=generate_candidate(rng,max_lines=rng.choice((6,9,12,16,20,24)),formula_limit=31)
        if raw is None or not 2<=raw["n_lines"]<=20:continue
        t,c,guard=identities(raw["prompt"])
        if t in seen:rejected["duplicate"]+=1;continue
        if t in exposure_t or c in exposure_c:rejected["exposed"]+=1;continue
        if any(key is not None and key in values for key,values in zip(guard,guards)):rejected["textbook_guard"]+=1;continue
        bucket=int(c[:12],16)%100
        role="targets" if bucket<50 else "development" if bucket<75 else "final"
        b=length_bin(raw["n_lines"])
        if b not in quotas[role] or counts[role][b]>=quotas[role][b]:continue
        body=raw["text"].split("PRF",1)[1].strip()
        verdict=strict_complete(raw["prompt"],body)
        if not verdict["ok"]:raise AssertionError(verdict)
        ids=tok.encode_record(raw["text"])
        prompt_len=len(tok.encode_prompt(raw["prompt"]))+1
        reference_tokens=1+len(ids)-len(tok.encode_prompt(raw["prompt"]))
        if prompt_len+max(1024,reference_tokens)>4096:rejected["context_limit"]+=1;continue
        row={**raw,"body":body,"source_id":common.digest([raw["prompt"],body]),"theorem_id":t,"context_id":c,
             "name":f"{role}_{t[:16]}","role":role,"family":family(raw),"length_bin":b,"reference_tokens":reference_tokens,
             "provenance":{"kind":"synthetic_forward_audit_reference","seed":args.seed,"attempt":attempt,"external_training_label":False}}
        populations[role].append(row);counts[role][b]+=1;seen.add(t)
        if sum(len(x) for x in populations.values())%100==0:
            event={"phase":"generate","attempt":attempt,"counts":{r:dict(x) for r,x in counts.items()},"elapsed_seconds":time.monotonic()-start}
            common.write_json(out/"progress.json",event);print(json.dumps(event),flush=True)
    for role,rows in populations.items():
        common.write_jsonl(out/"audit"/(role+".jsonl"),rows)
        common.write_jsonl(out/(role+".jsonl"),[{k:r[k] for k in ("name","source_id","prompt")} for r in rows])
    role_contexts={r:{x["context_id"] for x in rows} for r,rows in populations.items()}
    overlaps={f"{a}:{b}":len(role_contexts[a]&role_contexts[b]) for a in populations for b in populations if a<b}
    assert not any(overlaps.values())
    manifest={"status":"complete" if done() else "incomplete","args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "attempts":attempt,"elapsed_seconds":time.monotonic()-start,"quotas":quotas,"counts":{r:dict(x) for r,x in counts.items()},
              "sources":len(sources),"source_families":dict(Counter(r["family"] for r in sources)),"retention":len(retention),
              "excluded_theorems":len(exposure_t),"excluded_contexts":len(exposure_c),"cross_role_context_overlaps":overlaps,
              "rejected":dict(rejected),"exposure_sha256":{str(p):common.sha(p) for p in exposure_paths},"guard_sha256":common.sha(guards_path),
              "population_sha256":{str(p.relative_to(out)):common.sha(p) for p in out.rglob("*.jsonl")},
              "schema":"Trainer receives theorem-only longer targets. Audit references and strata are excluded from training inputs.",
              "matching":"Global atom bijections and premise permutation; nonempty premise context or closed formula skeleton. Semantic equivalence not excluded."}
    common.write_json(out/"manifest.json",manifest);print(json.dumps(manifest),flush=True)
    if not done():raise RuntimeError("Incomplete quotas; explicit review required before release")

if __name__=="__main__":main()
