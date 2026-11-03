import json, re, collections, sys  # noqa: E401, I001

sys.path.insert(0, "/scratch/cowifornia/nd-rl/worktrees/knockout-grpo")
from code.thinking.data import draw_level

K = "/scratch/cowifornia/nd-rl/knockout_grpo"
T = "/scratch/cowifornia/nd-rl/think/traces/search/full"
specs = [(0, None, 0.2), (64, "e64", 0.4), (256, "e256", 0.4)]
PLAN = re.compile(r"\|- \( ~ \( ~ (.+?) \) \) := fun \( n\d+ : \( ~ ")


def files(arm):
    if arm == "full":
        return {64: f"{T}/e64.jsonl", 256: f"{T}/e256.jsonl"}
    return {64: f"{K}/corpora/{arm}/e64.jsonl", 256: f"{K}/corpora/{arm}/e256.jsonl"}


out = {}
for arm in ["full", "ko", "d01", "d1"]:
    R = {lv: {json.loads(l)["id"]: json.loads(l) for l in open(p)} for lv, p in files(arm).items()}  # noqa: E741, SIM115
    c = collections.Counter()
    plan_ids = set()
    for i in R[256]:
        lv = draw_level(i, specs)
        for L in (64, 256):
            th = "\n".join(st.get("think", "") for st in R[L][i]["steps"])
            goal = "Classical.byContradiction |- ( ~ ( ~" in th
            plan = goal and bool(PLAN.search(th))
            if goal:
                c[f"detour_goal_e{L}"] += 1
            if plan:
                c[f"reductio_plan_e{L}"] += 1
                plan_ids.add(i)
            if L == lv:  # noqa: SIM300
                if goal:
                    c["detour_goal_trained"] += 1
                if plan:
                    c["reductio_plan_trained"] += 1
    c["reductio_plan_any_level_file"] = len(plan_ids)
    c["plan_records_refill"] = sum(1 for i in plan_ids if i.startswith("ko_"))
    out[arm] = dict(c)
    print(arm, dict(c), flush=True)
json.dump(out, open(f"{K}/trace_leak4.json", "w"), indent=1)  # noqa: SIM115
