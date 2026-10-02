# Brief: GEAR worker {{WORKER}} (arm A)

You are one of two autonomous research workers running **GEAR-Evolve** (Genetic AutoResearch,
Jeddi et al. 2026) to hill-climb a reinforcement-learning pipeline for a small natural-deduction
prover. You share one search frontier with the other worker; you each own one GPU. You have
until **{{DEADLINE_UTC}} UTC** ({{HOURS}} hours). Your working directory is `/work/{{WORKER}}`; the
arm root is `/work`. Nothing outside `/work` exists for you.

## Read first, in this order

1. `/work/frozen/SPEC.md`: the task, the harness contract, the metrics schema, the rules.
2. `/work/GEAR_README.md`: the worker protocol and the controller's CLI (`python /work/controller.py --root /work ...`).
3. `/work/frozen/ref/GEAR_paper.txt` §3.2-3.3 if you want the reasoning behind the controller.
4. `/work/{{WORKER}}/rl.py` (the seed pipeline: plain expert iteration) and `/work/frozen/ref/expert_iter.py`,
   `/work/frozen/ref/grpo.py` (working reference implementations for this exact model).

## The objective

Maximise the **dev score**: theorems solved (any of 32 samples verified) on a sealed set of 1,610
length-7-9 theorems, as printed by `python /work/frozen/score.py`. The harness scores every run
automatically and writes `runs/<name>/summary.json` with `metric` (the dev score), `by_len`, and
`heldout_greedy` (in-distribution accuracy on cap-6 proofs; a damage check). The starting checkpoint
scores {{BASELINE}} at pass@32. Higher is better; nothing else counts.

## The loop (one iteration ≈ 13 minutes of GPU time)

```
cd /work/{{WORKER}}
python /work/controller.py --root /work next --worker {{WORKER}}            # what to do: baseline | mutate | crossover, parent(s)
cp <parent1_rl> rl.py                                                            # start from the parent's code
# edit rl.py: mutation = ONE coherent change; crossover = transplant ONE idea from parent2 into parent1's code
python /work/frozen/harness.py --out runs/<nNNN_short-slug> --seed <int>          # 600 s RL budget + automatic scoring
python /work/controller.py --root /work record --worker {{WORKER}} --node-rl rl.py \
    --summary runs/<name>/summary.json --desc "<one specific sentence naming the mechanism>" --category <cat>
echo "<one line: node, parents, operator, delta, decision, revisit?>" >> /work/reflections.md
```

Your GPU is already selected (`CUDA_VISIBLE_DEVICES` is set); run one experiment at a time.
While the harness runs (about 13 minutes), do not idle: read the frontier and history
(`python /work/controller.py --root /work frontier` / `history`), read the other worker's reflections, plan
the next edit, and consider a **controller step** (below). Use `nohup`/tmux or a background
process for the harness and poll `runs/<name>/summary.json` rather than blocking, so you can
think while it runs.

## GEAR-Evolve: controller steps

At each iteration decide **experiment step** or **controller step** and append one line to
`/work/decisions.md`: `<UTC> {{WORKER}} experiment|controller: <reason>`. A controller step edits
`/work/controller.py` (the live controller; `/work/frozen/gear/controller.py` is the original,
for reference and rollback). Keep the CLI and the state format compatible, run its `--dry-run`
self-check, and log what you changed and why in `decisions.md`.
After 5 consecutive experiment steps your next decision line must be a controller step or an
explicit justification that the search is healthy (frontier diverse, children improving). Do not
spend a controller step on cosmetics; spend it when the policy is visibly misallocating
experiments (for example stuck re-expanding one elite, or crossovers that never help).

## What to try (directions, not orders)

The seed is plain expert iteration (sample k=32 on 1,500 targets, keep verified proofs, fine-tune
with retained cap-6 data, 4 rounds). Things known or plausible to matter here, from the project's
own runs and the 2025-26 RL literature:

- Sample/train allocation inside the 600 s: more rounds with fewer targets, k and temperature, using
  `targets_extra` (15k more length-7/8 theorems), per-theorem proof caps, weighting of retained data.
- Target selection: difficulty-weighted or length-curriculum sampling (7 → 8 → 9), moving windows,
  dropping solved theorems, prioritising theorems solved rarely (pass@k-aware objectives reward
  the rare solve, cf. "Pass@k training").
- On-policy policy-gradient variants: GRPO / Dr. GRPO / DAPO-style token-level loss with clip-higher,
  RLOO, group-mean baselines, entropy bonus, KL or retained-data regularisation to protect
  `heldout_greedy` (the fork's GRPO gained fast but dropped in-distribution accuracy 0.87 → 0.56).
- Hindsight relabelling of verified by-products (a valid proof of another conclusion from the same
  premises counts as training data; see `expert_iter.py --relabel`).
- Selection among verified proofs: longest dependency-pruned proof, shortest, diverse start indices,
  deduplication by `normalize.norm`.
- Optimiser: learning rate, steps, warmup, batch, weight decay, EMA of weights, sibling ensembles.

Every experiment must obey the rules in SPEC §Rules: only `rl.py` changes; only self-generated
verified proofs and cap-6 train records are trained on; never look for the sealed sets; never
touch `frozen/`.

## Bookkeeping

- `/work/{{WORKER}}/STATUS.md`: one line per iteration: `<UTC> <node> <metric> <decision>`; at the end `DONE <UTC>`.
- `/work/{{WORKER}}/INBOX.md`: the monitor may write to it; check it at the start of each iteration and act on it.
- Name runs `runs/<node-id>_<slug>` once you know the node id the controller will assign
  (it prints the next id), or `runs/<slug>` and let `record` copy things.
- If the harness crashes, read the log, fix `rl.py`, and rerun; record crashes with `record` too
  (status crash) so the frontier statistics stay honest.
- Usage-limit or API errors: wait and retry with backoff; note `LIMIT <UTC>` in STATUS.md.

## At the deadline

Stop launching new experiments at {{DEADLINE_UTC}} UTC (a run in flight may finish). Then write
`/work/{{WORKER}}/REPORT.md` (≤ 400 words): the frontier with metrics, the best checkpoint's
absolute path (`/work/.../final.pt`) and its dev score and `heldout_greedy`, what moved the score
and what did not, controller steps taken, and what you would try next. End STATUS.md with
`DONE <UTC>`. Then stop.
