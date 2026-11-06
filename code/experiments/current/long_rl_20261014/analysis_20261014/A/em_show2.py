import os, sys, json
os.environ.setdefault("ND_ARCH", "untied")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe"); sys.path.insert(0, ".")
import ndmodel as M, torch
torch.set_num_threads(3)
from feat import parse_prompt, show
E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
d = [json.loads(l) for l in open("feats2.jsonl")]
r = [r for r in d if r["pool"] == "old" and r.get("schema") == "excluded_middle"][0]
_, goal = parse_prompt(r["prompt"])
G = show(goal); A = show(goal[1]); nG, nA = f"( ~ {G} )", f"( ~ {A} )"
p = (f"N1 | {nG} : AS ; N2 | | {A} : AS ; N3 | | {G} : ORI1 N2 ; N4 | | F : NEGE N3 N1 ; N5 | {nA} : NEGI N2 N4 ; "
     f"N6 | {G} : ORI2 N5 ; N7 | F : NEGE N6 N1 ; N8 ( ~ {nG} ) : NEGI N1 N7 ; N9 {G} : DN N8 ; QED")
model, tok = M.load(f"{E}/resume/untied-ei_s1/final.pt")
rows = M.token_logp(model, tok, r["prompt"], p)
print(r["prompt"])
print(" ".join(f"{t}[{l:.1f}|{a}]" if l < -1 else t for t, l, a in rows))
