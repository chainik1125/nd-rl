"""usage: samp.py <ckpt: final|pretrain> <k> <out.json> <prompt1> [<prompt2> ...]  -- samples k per prompt, Lean-judged."""
import os, sys, json, time
os.environ.setdefault("ND_ARCH", "untied")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe")
import ndmodel as M, torch
torch.set_num_threads(3)
E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
ck, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompts = sys.argv[4:]
if len(prompts) == 1 and prompts[0].endswith(".json"):
    prompts = json.load(open(prompts[0]))
model, tok = M.load(f"{E}/resume/untied-ei_s1/{ck}.pt")
t = time.time()
recs = [{"prompt": p, "body": 20, "name": f"q{i}"} for i, p in enumerate(prompts)]
res = []
for i in range(0, len(recs), max(1, 240 // k)):
    chunk = recs[i:i + max(1, 240 // k)]
    res += M.sample(model, tok, chunk, k=k)
json.dump([{"prompt": p, "samples": [{"nd": nd, "ok": ok, "raw": raw} for nd, ok, raw in r]} for p, r in zip(prompts, res)], open(out, "w"))
for p, r in zip(prompts, res):
    print(sum(ok for _, ok, _ in r), "/", len(r), p)
print("secs", time.time() - t)
