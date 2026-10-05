import gzip
import json
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "/scratch/cowifornia/nd-rl/think/nd-takehome")
from fdepth import thm_depth

S = "/scratch/cowifornia/nd-rl/bench/out/store/samples/"
items = {
    json.loads(l)["pid"]: json.loads(l)
    for l in open("/scratch/cowifornia/nd-rl/bench/out/suites/textbook_dev/v1/items.jsonl")
}
for run in [
    "textbook_dev-v1-dan12-t1-s0-be17b9a943a81b5d",
    "textbook_dev-v1-dan12-fz-s0-2e3d9e626f5a039f",
]:
    per = defaultdict(list)
    for l in gzip.open(S + run + ".jsonl.gz", "rt"):
        r = json.loads(l)
        per[r["pid"]].append(r)
    print("##", run, "items", len(per), "draws/item", st.median(len(v) for v in per.values()))
    unsolved = []
    bands = Counter()
    for pid, lst in per.items():
        c = sum(x["ok"] for x in lst)
        p1 = c / len(lst)
        it = items[pid]
        bands["0" if c == 0 else ("<5%" if p1 < 0.05 else ("5-30%" if p1 < 0.3 else ">=30%"))] += 1
        if c == 0:
            reasons = Counter()
            steps = []
            m = len(it["sequent"]["premises"])
            for x in lst:
                nat = json.loads(x["native"]) if x.get("native") else {}
                end = nat.get("end", "")
                reasons["lean_rejected" if end == "done" else end.replace("syntax:", "")] += 1
                if nat.get("steps") is not None:
                    steps.append(nat["steps"] - m)
            prompt = (
                "THM "
                + " , ".join(it["sequent"]["premises"])
                + " SEQ "
                + it["sequent"]["conclusion"]
                + " PRF"
                if it["sequent"]["premises"]
                else "THM SEQ " + it["sequent"]["conclusion"] + " PRF"
            )
            unsolved.append(
                (
                    pid,
                    it["length"].get("L"),
                    m,
                    thm_depth(prompt),
                    st.median(steps),
                    reasons.most_common(3),
                    it["sequent"]["conclusion"][:60],
                )
            )
    print("per-item pass@1 bands", dict(bands))
    for u in sorted(unsolved, key=lambda x: x[1] or 0):
        print("  unsolved", u)
