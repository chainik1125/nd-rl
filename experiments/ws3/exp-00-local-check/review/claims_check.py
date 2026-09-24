"""Phase 2: recompute the executor's remaining claims (numbers.md, results.md) from the dumps."""
import os, sys, json
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from recount import read_jsonl, RUN, EXP, safe_verify, DUMPS
sys.path.insert(0, os.path.join(EXP, "..", "..", "..", "code", "ws3")); import ndpath
from tokenizer import Tokenizer
tok = Tokenizer("abs")
out = {}
rows = {k: read_jsonl(os.path.join(RUN, v)) for k, v in DUMPS.items()}
# (1) completions that differ between frozen and RL, per eval
for part in ("greedy_heldout", "textbook72", "long_pool"):
    b, e = rows["baseline/" + part], rows["evaluate/" + part]
    assert [r["name"] for r in b] == [r["name"] for r in e]
    out[f"differing_completions_{part}"] = sum(1 for x, y in zip(b, e) if x["completion"] != y["completion"])
# (2) per-problem textbook hits (my verification)
hits = {}
for lab in ("baseline/textbook72", "evaluate/textbook72"):
    c = Counter()
    for r in rows[lab]:
        c[r["name"]] += int(safe_verify(r["prompt"], r["completion"])[0])
    hits[lab] = {k: v for k, v in c.items() if v}
out["textbook_hits_per_solved_problem"] = hits
# (3) executor's 20 rows (pick_samples.py seed 0)
comp = read_jsonl(os.path.join(RUN, "completions.jsonl.gz"))
ex_idx = [663, 1553, 2281, 2289, 3578, 4242, 4617, 4969, 5866, 6311, 6634, 6890, 7808, 7961, 8268, 8376, 9558, 10131, 12383, 12418]
v = [(i, safe_verify(comp[i]["prompt"], comp[i]["completion"])[0], comp[i]["reward"], comp[i]["step"]) for i in ex_idx]
out["executor_20_rows"] = {"accepted_mine": sum(1 for _, ok, _, _ in v if ok), "rejected_mine": sum(1 for _, ok, _, _ in v if not ok),
                           "mismatch_vs_reward": [i for i, ok, rw, _ in v if ok != (rw >= 0.5)],
                           "rejected_rows": [(i, safe_verify(comp[i]["prompt"], comp[i]["completion"])[1]) for i, ok, _, _ in v if not ok]}
# (4) reason distribution, unparseable fraction, token counts per dump
def is_unparseable(reason):
    return reason.startswith("verifier exception") or reason.startswith("strict parser exception") or "parse" in reason.lower() or "malformed" in reason.lower()
stats = {}
for lab in DUMPS:
    rs = rows[lab]
    reasons = Counter()
    for r in rs:
        ok, reason, _ = safe_verify(r["prompt"], r["completion"])
        reasons[reason.split("(")[0].split(":")[0].strip() if not ok else "ok"] += 1
    unp = sum(1 for r in rs if is_unparseable(safe_verify(r["prompt"], r["completion"])[1]))
    ntok_field = [r["n_tokens"] for r in rs]
    ntok_mine = [len(tok.encode_proof(r["completion"])) if "QED" in r["completion"] else len(r["completion"].split()) for r in rs]
    stats[lab] = {"reasons_top": dict(reasons.most_common(8)), "unparseable_frac_mine": unp / len(rs),
                  "n_tokens_field_mean": sum(ntok_field) / len(rs), "n_tokens_field_max": max(ntok_field),
                  "whitespace_tokens_mean": sum(len(r["completion"].split()) for r in rs) / len(rs),
                  "whitespace_tokens_max": max(len(r["completion"].split()) for r in rs),
                  "n_tokens_field_eq_whitespace": sum(1 for r in rs if r["n_tokens"] == len(r["completion"].split())),
                  "capped_frac_mine(n_tokens>=400)": sum(1 for r in rs if r["n_tokens"] >= 400) / len(rs)}
out["dump_stats"] = stats
# (5) training metrics odds and ends
m = read_jsonl(os.path.join(RUN, "metrics.jsonl")); roll = [x for x in m if "reward" in x]
out["train"] = {"entropy_first": roll[0]["entropy"], "entropy_last": roll[-1]["entropy"],
                "mean_len_first": roll[0]["completions/mean_length"], "mean_len_last": roll[-1]["completions/mean_length"],
                "grad_norm_mean": sum(x["grad_norm"] for x in roll) / len(roll), "grad_norm_min": min(x["grad_norm"] for x in roll),
                "grad_norm_max": max(x["grad_norm"] for x in roll),
                "secs_per_step_mean": sum(x["rollout/secs_per_step"] for x in roll) / len(roll),
                "kl_first": roll[0]["kl"], "kl_last": roll[-1]["kl"]}
ts = json.load(open(os.path.join(RUN, "train_summary.json")))
out["train_summary_json"] = {k: ts[k] for k in ("secs", "steps", "completions_per_step", "mean_reward_first5", "mean_reward_last5", "max_kl")}
st = read_jsonl(os.path.join(RUN, "stages.jsonl")); out["stages"] = {s["stage"]: round(s["secs"]) for s in st}; out["stages_total"] = round(sum(s["secs"] for s in st))
# sampler throughput values in the summaries (not independently checkable, listed for the wording check)
thr = {}
for d in ("baseline", "evaluate"):
    s = json.load(open(os.path.join(RUN, d, "summary.json")))
    for part in ("greedy_heldout", "textbook", "long_pool"):
        thr[f"{d}/{part}"] = round(s[part]["samples_per_sec"], 1)
out["summary_samples_per_sec"] = thr
# (6) files cited in numbers.md / results.md
cited = ["stages.jsonl", "baseline/environment.json", "baseline/nta.json", "baseline/greedy_heldout.json", "baseline/dumps/greedy_heldout.jsonl",
         "evaluate/greedy_heldout.json", "evaluate/dumps/greedy_heldout.jsonl", "baseline/textbook.json", "baseline/dumps/textbook72.jsonl",
         "evaluate/textbook.json", "evaluate/dumps/textbook72.jsonl", "baseline/long_pool.json", "baseline/dumps/long_pool.jsonl",
         "evaluate/long_pool.json", "evaluate/dumps/long_pool.jsonl", "train_summary.json", "run_manifest.json", "metrics.jsonl",
         "completions.jsonl.gz", "prompts.jsonl", "eval_periodic.jsonl", "report/metric_coverage.json",
         "report/curves.png", "report/periodic_eval.png", "report/greedy_heldout.png", "report/long_pool.png", "evaluate/comparison.json"]
out["cited_files"] = {c: (os.path.exists(os.path.join(RUN, c)) or os.path.exists(os.path.join(RUN, c + ".gz"))) for c in cited}
out["cited_files_missing"] = [c for c, ok in out["cited_files"].items() if not ok]
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "claims_check.json"), "w"), indent=1)
