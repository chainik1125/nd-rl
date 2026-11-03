import json, re, collections, sys  # noqa: E401, I001

sys.path.insert(0, "/scratch/cowifornia/nd-rl/worktrees/knockout-grpo")
from code.thinking.data import draw_level

K = "/scratch/cowifornia/nd-rl/knockout_grpo"
T = "/scratch/cowifornia/nd-rl/think/traces/search/full"
specs = [(0, None, 0.2), (64, "e64", 0.4), (256, "e256", 0.4)]
DNCITE = re.compile(r"Classical\.byContradiction (n\d+)\b")
HAVE = re.compile(r"have (n\d+) : (.*?) := (.*)$")
FUN = re.compile(r"fun \( (n\d+) : ")


def files(arm):
    if arm == "full":
        return {64: f"{T}/e64.jsonl", 256: f"{T}/e256.jsonl"}
    return {64: f"{K}/corpora/{arm}/e64.jsonl", 256: f"{K}/corpora/{arm}/e256.jsonl"}


ex = {}
for arm in ["ko"]:
    R = {lv: {json.loads(l)["id"]: json.loads(l) for l in open(p)} for lv, p in files(arm).items()}  # noqa: E741, SIM115
    c = collections.Counter()
    for i, r256 in R[256].items():  # noqa: B007
        lv = draw_level(i, specs)
        if lv == 0:
            continue
        r = R[lv][i]
        intro = {}  # name -> kind of its latest introduction before the step
        f = collections.Counter()
        for st in r["steps"]:
            for line in st.get("think", "").split("\n"):
                for nm in DNCITE.findall(line):
                    k = intro.get(nm, "unknown")
                    f["dn_cite_" + k] = 1
                    if k == "unknown" and arm == "ko" and len(ex) < 4:
                        ex[i] = line
            a = st["action"]
            m = HAVE.match(a)
            if m:
                nm, form, rhs = m.groups()
                if rhs.startswith("h"):
                    intro[nm] = "premise"
                elif form.startswith("( ~ ( ~") and rhs.startswith("( fun ( n") and ": ( ~" in rhs:
                    intro[nm] = "negi_dneg"
                else:
                    intro[nm] = "derived_other"
            for nm in FUN.findall(a):
                intro[nm] = "hypothesis"
        c.update(f)
    print(arm, dict(c), flush=True)
print(ex)
