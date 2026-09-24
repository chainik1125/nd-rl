"""Follow-up on the recount's leakage numbers: are the overlaps inherited from the fixed
pre-training set train.jsonl? Prints no test-file prompt text (plan: never inspect)."""
import os, sys, json, gzip
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recount import read_jsonl, canon_key, thm_of_prompt, FORK_DATA, RUN, TB, canonical_theorem

prompts = read_jsonl(os.path.join(RUN, "prompts.jsonl"))
train = read_jsonl(os.path.join(FORK_DATA, "train.jsonl.gz"))
held = read_jsonl(os.path.join(FORK_DATA, "heldout.jsonl"))
ts = read_jsonl(os.path.join(FORK_DATA, "test_short_prompts.jsonl"))
tl = read_jsonl(os.path.join(FORK_DATA, "test_long_prompts.jsonl"))
v36 = read_jsonl(os.path.join(FORK_DATA, "validation_36.jsonl"))
tb = read_jsonl(os.path.join(TB, "textbook_dev.jsonl")) + read_jsonl(os.path.join(TB, "textbook_train.jsonl"))
rl = read_jsonl(os.path.join(FORK_DATA, "rl_targets.jsonl"))
tr_ = read_jsonl(os.path.join(FORK_DATA, "transfer.jsonl"))
ck = lambda recs: {canon_key(thm_of_prompt(r["prompt"])) for r in recs}
ct = lambda recs: {canonical_theorem(r["prompt"]) for r in recs}
out = {}
K = {"train": ck(train), "heldout": ck(held), "test_short": ck(ts), "test_long": ck(tl), "validation_36": ck(v36),
     "textbook72": ck(tb), "rl_targets": ck(rl), "transfer": ck(tr_), "rl_prompts_1600": ck(prompts)}
C = {"train": ct(train), "heldout": ct(held), "test_short": ct(ts), "test_long": ct(tl), "validation_36": ct(v36),
     "textbook72": ct(tb), "rl_targets": ct(rl), "transfer": ct(tr_), "rl_prompts_1600": ct(prompts)}
names = ["heldout", "test_short", "test_long", "validation_36", "textbook72", "rl_targets", "transfer"]
out["fixed_train_jsonl_vs"] = {n: {"canon_key": len(K["train"] & K[n]), "canonical_theorem": len(C["train"] & C[n]),
                                   "n_classes": len(K[n])} for n in names}
out["rl_prompts_1600_vs"] = {n: {"canon_key": len(K["rl_prompts_1600"] & K[n]), "canonical_theorem": len(C["rl_prompts_1600"] & C[n])} for n in names}
out["heldout_vs_test"] = {n: {"canon_key": len(K["heldout"] & K[n])} for n in ("test_short", "test_long")}
out["rl_targets_vs_test"] = {n: {"canon_key": len(K["rl_targets"] & K[n])} for n in ("test_short", "test_long")}
# the specific RL prompts involved (indices and lengths only; no test text)
hits_ts = [(i, r["length"]) for i, r in enumerate(prompts) if canon_key(thm_of_prompt(r["prompt"])) in K["test_short"]]
hits_held = [(i, r["length"]) for i, r in enumerate(prompts) if canonical_theorem(r["prompt"]) in C["heldout"]]
out["rl_prompt_rows_in_test_short_class"] = hits_ts
out["rl_prompt_rows_in_heldout_canonical_theorem_class"] = hits_held
# how many completions/steps touched those prompts
comp = read_jsonl(os.path.join(RUN, "completions.jsonl.gz"))
hp = {prompts[i]["prompt"] for i, _ in hits_ts}
out["completions_on_test_short_class_prompts"] = sum(1 for r in comp if r["prompt"] in hp)
out["steps_touching_them"] = sorted({r["step"] for r in comp if r["prompt"] in hp})
# heldout canonical_theorem hits: show the premise-permutation pairs (heldout is not a test file)
for i, L in hits_held:
    p = prompts[i]["prompt"]
    mates = [h["prompt"] for h in held if canonical_theorem(h["prompt"]) == canonical_theorem(p)]
    out.setdefault("heldout_pairs", []).append({"rl_prompt": p, "heldout_prompts": mates,
                                                "same_canon_key": any(canon_key(thm_of_prompt(m)) == canon_key(thm_of_prompt(p)) for m in mates)})
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "leak_detail.json"), "w"), indent=1)
