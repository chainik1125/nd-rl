import json, re, collections, sys  # noqa: E401, I001

sys.path.insert(0, "/scratch/cowifornia/nd-rl/worktrees/knockout-grpo")
from code.thinking.data import draw_level

K = "/scratch/cowifornia/nd-rl/knockout_grpo"
T = "/scratch/cowifornia/nd-rl/think/traces/search/full"
specs = [(0, None, 0.2), (64, "e64", 0.4), (256, "e256", 0.4)]
PLAN = re.compile(r"\|- \( ~ \( ~ (.+?) \) \) := fun \( n\d+ : \( ~ ")
DNCITE = re.compile(r"Classical\.byContradiction (n\d+)\b")
NEGIACT = re.compile(r"have (n\d+) : \( ~ \( ~ (.+) \) \) := \( fun \( n\d+ : \( ~ ")


def files(arm):
    if arm == "full":
        return {64: f"{T}/e64.jsonl", 256: f"{T}/e256.jsonl"}
    return {64: f"{K}/corpora/{arm}/e64.jsonl", 256: f"{K}/corpora/{arm}/e256.jsonl"}


res = {}
for arm in ["full", "ko", "d01", "d1"]:
    R = {lv: {json.loads(l)["id"]: json.loads(l) for l in open(p)} for lv, p in files(arm).items()}  # noqa: E741, SIM115
    c = collections.Counter()
    for i, r256 in R[256].items():
        lv = draw_level(i, specs)
        prem = set()
        for st in r256["steps"]:
            m = re.match(r"have (n\d+) : .* := h\d+ ;", st["action"])
            if m:
                prem.add(m.group(1))
        # proof-level (all levels train actions): NEGI-built ~~G line in the proof
        if any(NEGIACT.search(st["action"]) for st in r256["steps"]):
            c["proof_negi_dneg_line"] += 1
        if lv == 0:
            continue
        r = R[lv][i]
        th = "\n".join(st.get("think", "") for st in r["steps"])
        f = collections.Counter()
        for line in th.split("\n"):
            if "Classical.byContradiction" in line:
                f["detour_any"] = 1
                if "<no>" in line:
                    f["detour_dead"] = 1
                for nm in DNCITE.findall(line):
                    f["dn_cites_named_line"] = 1
                    if nm not in prem:
                        f["dn_cites_nonpremise_name"] = 1
            if "( ~ ( ~" in line:
                f["dneg_literal"] = 1
        if "Classical.byContradiction |- ( ~ ( ~" in th and PLAN.search(th):
            f["reductio_plan"] = 1
        c.update(f)
    res[arm] = dict(c)
    print(arm, dict(c), flush=True)
json.dump(res, open(f"{K}/trace_leak2.json", "w"), indent=1)  # noqa: SIM115
