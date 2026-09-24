"""Write numbers.md for exp-00 from the run folder. Every number cites the file it came from.

    uv run python ../experiments/ws3/exp-00-local-check/make_numbers.py
"""
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")


def J(*p):
    with open(os.path.join(RUN, *p)) as f:
        return json.load(f)


def rows(*p):
    with open(os.path.join(RUN, *p)) as f:
        return [json.loads(l) for l in f if l.strip()]


def ci(v):
    return f"{v['rate']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"


out = [f"# exp-00 local check: numbers", "",
       f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by `make_numbers.py`. "
       "Paths are relative to `experiments/ws3/exp-00-local-check/run/`; the `.jsonl` dumps are committed gzipped (`.jsonl.gz`). Executor numbers only; "
       "the reviewer recounts from the dumps (`review.md`).", ""]

st = rows("stages.jsonl")
out += ["## Stage wall-clock (`stages.jsonl`)", "", "| stage | seconds | finished (UTC) |", "| --- | --- | --- |"]
out += [f"| {s['stage']} | {s['secs']:.0f} | {s['utc']} |" for s in st]
out += [f"| total | {sum(s['secs'] for s in st):.0f} | |", ""]

env = J("baseline", "environment.json")
out += ["## Environment (`baseline/environment.json`)", "",
        f"- git commit `{env['git_commit']}`; checkpoint sha256 `{env['checkpoint_sha256']}`",
        f"- versions: {env['versions']}; platform {env['platform']}; torch threads {env['cpu_threads']}", ""]

n = J("baseline", "nta.json")
out += ["## (a) T1 gate (`baseline/nta.json`)", "",
        f"- next-token accuracy **{n['next_token_accuracy']:.4f}** on {n['n_records']} heldout records "
        f"({n['n_target_tokens']} target tokens); mean loss {n['loss']:.4f}; gate {n['gate_threshold']}: "
        f"**{'PASS' if n['gate_pass'] else 'FAIL'}**", ""]

def length_block(title, part, k_note):
    b = J("baseline", f"{part}.json")
    have_e = os.path.exists(os.path.join(RUN, "evaluate", f"{part}.json"))
    e = J("evaluate", f"{part}.json") if have_e else None
    o = [f"## {title}", "", f"k={b['k']}, temperature {b['temperature']}, {b['n_records']} records, {b['samples']} samples per model. {k_note}", "",
         "| length | frozen solved/n | frozen rate [Wilson 95%] | RL solved/n | RL rate [Wilson 95%] |", "| --- | --- | --- | --- | --- |"]
    Ls = sorted(set(b["by_length"]) | set(e["by_length"] if e else {}), key=int)
    for L in Ls:
        bv = b["by_length"].get(L); ev = e["by_length"].get(L) if e else None
        o.append(f"| {L} | {bv['solved']}/{bv['n']} | {ci(bv)} | " + (f"{ev['solved']}/{ev['n']} | {ci(ev)} |" if ev else "- | - |"))
    bo = b["overall"]; eo = e["overall"] if e else None
    o.append(f"| all | {bo['solved']}/{bo['n']} | {ci(bo)} | " + (f"{eo['solved']}/{eo['n']} | {ci(eo)} |" if eo else "- | - |"))
    o += ["", f"- max length solved: frozen {b['max_length_solved']}" + (f", RL {e['max_length_solved']}" if e else ""),
          f"- unparseable fraction of samples: frozen {b['unparseable_frac']:.4f}" + (f", RL {e['unparseable_frac']:.4f}" if e else ""),
          f"- completion tokens mean/max: frozen {b['completion_tokens_mean']:.0f}/{b['completion_tokens_max']}, capped fraction {b['capped_frac']:.4f}"
          + (f"; RL {e['completion_tokens_mean']:.0f}/{e['completion_tokens_max']}, capped {e['capped_frac']:.4f}" if e else ""),
          f"- sampler throughput: frozen {b['samples_per_sec']:.1f} samples/s" + (f", RL {e['samples_per_sec']:.1f}" if e else ""),
          f"- sources: `baseline/{part}.json`, `baseline/dumps/{part if part != 'textbook' else 'textbook72'}.jsonl`"
          + (f", `evaluate/{part}.json`, `evaluate/dumps/{part if part != 'textbook' else 'textbook72'}.jsonl`" if e else ""), ""]
    return o, b, e

o, _, _ = length_block("(b) greedy solve rate by length, heldout stride", "greedy_heldout", "Length = the record's reference `n_lines`.")
out += o
o, tb, te = length_block("(c) textbook 72 by reference length", "textbook", "Length -1 = the 14 `textbook_train` problems, which carry no reference length.")
out += o
out += ["### (c) textbook 72 by slice", "", "| slice | frozen solved/n [Wilson 95%] | RL solved/n [Wilson 95%] |", "| --- | --- | --- |"]
for s in ("dev58", "train14", "all"):
    bv = tb["by_slice"][s]; ev = te["by_slice"][s] if te else None
    out.append(f"| {s} | {bv['solved']}/{bv['n']} [{bv['ci'][0]:.3f}, {bv['ci'][1]:.3f}] | " + (f"{ev['solved']}/{ev['n']} [{ev['ci'][0]:.3f}, {ev['ci'][1]:.3f}] |" if ev else "- |"))
bv = tb["ref_len_ge7"]; ev = te["ref_len_ge7"] if te else None
out.append(f"| reference length >= 7 | {bv['solved']}/{bv['n']} [{bv['ci'][0]:.3f}, {bv['ci'][1]:.3f}] | " + (f"{ev['solved']}/{ev['n']} [{ev['ci'][0]:.3f}, {ev['ci'][1]:.3f}] |" if ev else "- |"))
out.append("")
o, _, _ = length_block("(d) long pool (`rl_targets` stride) pass@32 by true length", "long_pool", "Length = `L_true`.")
out += o

if os.path.exists(os.path.join(RUN, "train_summary.json")):
    ts = J("train_summary.json"); man = J("run_manifest.json")
    m = rows("metrics.jsonl"); roll = [r for r in m if "reward" in r]; ev = [r for r in m if "eval/pass_rate" in r]
    out += ["## (e) GRPO (`train_summary.json`, `run_manifest.json`, `metrics.jsonl`)", "",
            f"- {ts['steps']} optimizer steps, {ts['completions_per_step']} completions per step, {ts['secs']:.0f} s wall-clock (includes periodic evals)",
            f"- prompt set: {man['n_prompts']} prompts, lengths {man['prompt_lengths']}, all from `train.jsonl` (`prompts.jsonl`); leak check passed at start",
            f"- trainer: lr {man['trainer_config']['learning_rate']}, beta {man['trainer_config']['beta']}, epsilon {man['trainer_config']['epsilon']}, "
            f"temperature {man['trainer_config']['temperature']}, group {man['trainer_config']['num_generations']}, cap {man['trainer_config']['max_completion_length']} tokens, loss_type {man['trainer_config']['loss_type']}",
            f"- mean reward, first 5 steps: **{ts['mean_reward_first5']:.3f}**; last 5 steps: **{ts['mean_reward_last5']:.3f}**",
            f"- max KL over the run: **{ts['max_kl']:.5f}**",
            f"- within-group reward std: min {min(r['reward_std'] for r in roll):.3f}, mean {sum(r['reward_std'] for r in roll)/len(roll):.3f}; steps with std == 0: {sum(1 for r in roll if r['reward_std'] == 0)}",
            f"- frac_reward_zero_std (groups with no signal), mean over steps: {sum(r['frac_reward_zero_std'] for r in roll)/len(roll):.3f}",
            f"- entropy: first {roll[0]['entropy']:.4f}, last {roll[-1]['entropy']:.4f}",
            f"- completion length mean: first {roll[0]['completions/mean_length']:.0f}, last {roll[-1]['completions/mean_length']:.0f}; max over run {max(r['completions/max_length'] for r in roll):.0f}; fraction at cap, max over steps {max(r['completions/clipped_ratio'] for r in roll):.4f}",
            f"- unparseable fraction, mean over steps {sum(r['reward/unparseable_frac'] for r in roll)/len(roll):.4f}; invalid fraction mean {sum(r['reward/invalid_frac'] for r in roll)/len(roll):.4f}",
            f"- grad norm: mean {sum(r['grad_norm'] for r in roll)/len(roll):.3f}, max {max(r['grad_norm'] for r in roll):.3f}; learning rate constant {roll[0]['learning_rate']}",
            f"- rollout throughput: mean {sum(r['rollout/completions_per_sec'] for r in roll)/len(roll):.1f} completions/s (generation + optimizer step), {sum(r['rollout/secs_per_step'] for r in roll)/len(roll):.1f} s/step",
            f"- completions dumped: {sum(1 for _ in __import__('gzip').open(os.path.join(RUN, 'completions.jsonl.gz'), 'rt'))} rows in `completions.jsonl.gz`", "",
            "### (e) mean reward per step (`metrics.jsonl`)", "", "| step | reward | reward_std | kl | entropy | mean len | grad_norm |", "| --- | --- | --- | --- | --- | --- | --- |"]
    out += [f"| {r['step']} | {r['reward']:.3f} | {r['reward_std']:.3f} | {r['kl']:.5f} | {r['entropy']:.4f} | {r['completions/mean_length']:.0f} | {r['grad_norm']:.3f} |" for r in roll]
    out += ["", "### (e) periodic greedy eval (`metrics.jsonl`, `eval_periodic.jsonl`; 50 heldout + 20 rl_targets records)", ""]
    Ls = sorted({k[len("eval/pass_rate_len"):] for r in ev for k in r if k.startswith("eval/pass_rate_len")}, key=int)
    out += ["| step | " + " | ".join(f"len {L}" for L in Ls) + " | all |", "| --- | " + " | ".join("---" for _ in Ls) + " | --- |"]
    for r in ev:
        out.append(f"| {r['step']} | " + " | ".join(f"{r.get(f'eval/solved_len{L}', 0)}/{r.get(f'eval/n_len{L}', 0)}" for L in Ls) + f" | {r['eval/pass_rate']:.3f} |")
    out.append("")
    cov = J("report", "metric_coverage.json") if os.path.exists(os.path.join(RUN, "report", "metric_coverage.json")) else None
    if cov:
        out += ["### Gate B metric coverage (`report/metric_coverage.json`)", "", f"- all present: **{all(cov.values())}**; missing: {[k for k, v in cov.items() if not v] or 'none'}", ""]

with open(os.path.join(HERE, "numbers.md"), "w") as f:
    f.write("\n".join(out))
print("\n".join(out))
