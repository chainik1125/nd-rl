# GEAR controller: worker and auditor protocol (run 2, ab2)

This directory has the GEAR-Evolve search controller (Jeddi et al. 2026, "GEAR: Genetic
AutoResearch for Agentic Code Evolution", sections 3.2 and 3.3), adapted to the RL hill-climb in
`../SPEC.md`. Seven hill-climbing workers share one frontier of up to four `rl.py` variants, and
an eighth agent, the **auditor**, re-runs every candidate BEST with a second seed before it may
take the BEST role. The controller decides which parent(s) each worker starts from, whether to
mutate or cross over, and whether a finished child joins the frontier. It never runs experiments
and never touches a GPU. You run the experiments, on a GPU the harness locks for you.

Run 1's workers evolved the controller from version 1 to 8 (see the changelog in the header of
`controller.py`). This is version 9: all eight run-1 changes are kept, with their fixed bands
rescaled to the new score, plus the run-2 rules below.

Files (`ROOT=/scratch/cowifornia/nd-rl/ab2`; host paths, no containers):

| file | role |
|---|---|
| `$ROOT/controller.py` | the editable controller (starts as a copy of `frozen/gear/controller.py`). Stdlib only; every command prints one JSON object |
| `$ROOT/test_controller.py` | simulated seven-worker + auditor test, plus the two-seed / floor / source / auditor tests (`python3 test_controller.py`) |
| `$ROOT/gear_state.json` | the authoritative state. Only the controller writes it |
| `$ROOT/nodes/<id>/` | `rl.py` (that node's code), `meta.json` (a copy of the node record), `summary.json` (its run), `replicate_summary.json` (the auditor's second seed, if any) |
| `$ROOT/reflections.md` | one line per experiment, appended by workers |
| `$ROOT/decisions.md` | one line per iteration, experiment step or controller step (GEAR-Evolve) |
| `$ROOT/ideas.md` | the research log: one row per idea tried (format below) |

```bash
ROOT=/scratch/cowifornia/nd-rl/ab2
CTRL="python3 $ROOT/controller.py --root $ROOT"
export AB_WORKER=W        # your worker name from the brief (w1 ... w7, or auditor); stored in the GPU lock
```

## The worker loop

Do these steps in order on every iteration. `W` is your worker name. Always use the same name.

0. **Decide** whether this iteration is an experiment step or a controller step (see
   [GEAR-Evolve](#gear-evolve-controller-steps)). Append the decision line to
   `$ROOT/decisions.md` before doing anything else.
1. **Ask**: run `$CTRL next --worker W`. Read `action`, `parent1`, `parent1_rl`, `parent1_desc`,
   `parent2`, `parent2_rl`, `parent2_desc`, `margins`, `pending_best` and `notes`. If you already
   had an assignment, `next` gives you the same one again (the notes say "already assigned").
2. **Start from the parent**: `cp <parent1_rl> ./rl.py`. Always start from `parent1_rl`, never
   from whatever was in your workspace before. Read the `KNOWN_FAILURES` block at the top of the
   seed `rl.py` and `$ROOT/ideas.md` before choosing an idea: do not re-run a known failure unless
   you state why this version differs.
3. **Edit** according to `action`:
   - `baseline`: change nothing. Run the seed exactly as it is.
   - `mutate`: make one coherent change to `./rl.py` (see [Operators](#operators-in-code-terms)).
   - `crossover`: keep parent1's code as the base, read `parent2_rl`, and transplant exactly one
     coherent idea from parent2 into `./rl.py`.
4. **Run** the frozen harness in your workspace:
   `python frozen/harness.py --out runs/<name> --seed 0 --gpu auto`. Use a new `<name>` for every
   run. `--gpu auto` (the default) waits for a free GPU, holds it for the whole run including
   scoring, and releases it at exit. Always let the run finish, including a crash or timeout. It
   writes `runs/<name>/summary.json` with `metric` (dev L8-11), `frontier` (L12-14) and
   `heldout_greedy`.
5. **Record**:
   ```bash
   $CTRL record --worker W --node-rl ./rl.py --summary runs/<name>/summary.json \
       --desc "<one sentence>" --category <category> --source "<URL or own idea>"
   ```
   `--source` is required: the URL of the paper, repository, issue or post the idea came from, or
   `own idea`. For a crossover, give the source of the transplanted idea. The controller takes the
   parent(s) and the operator from your in-flight assignment. It copies `rl.py` and `summary.json`
   into `nodes/<new id>/`, applies the rules below and prints `decision`, `role`, `evicted`,
   `delta` (the child's metric minus parent1's) and `frontier_count`. For a `baseline`, `--desc`,
   `--category` and `--source` are optional. A crashed or timed-out run is recorded like any
   other: pass its summary, or a missing path if the harness died before writing one. It becomes
   `CRASH` or `TIMEOUT`, counts as a completed experiment, and never enters the frontier.
   If the decision is `PENDING_BEST`, your child beat the BEST by more than the margin. It stays
   out of the frontier (nobody can use it as a parent yet) until the auditor's second seed
   confirms or demotes it. Carry on with your next iteration; do not replicate it yourself.
6. **Reflect**: append one line to `$ROOT/reflections.md`, and add or update the idea's row in
   `$ROOT/ideas.md` (formats below).
7. Go back to step 0.

If you are interrupted between `next` and `record` and the run is lost, run
`$CTRL release --worker W`. This drops the assignment and undoes its bookkeeping (parent use
counts, co-use counts, and a pending fresh-elite crossover). Never `release` a run that finished.
Record it instead, even if it crashed. At most seven workers can hold assignments at once; an
eighth name is refused.

Other commands: `$CTRL frontier` shows the current elites and the PENDING_BEST queue (JSON with a
`table` field of human-readable lines, including each elite's frontier count L12-14, number of
seeds and source). `$CTRL history` shows every node with its parents, operator, metric, seeds,
frontier count, decision and source (`table` field; a `*` marks nodes that went through the
two-seed rule). `$CTRL next --worker W --dry-run` shows what you would get without reserving
anything. If you need to record a run under different parents than the ones assigned, use
`--parents n003,n007 --operator crossover`. That is an exception; give the reason in the
reflection.

## The auditor loop

The auditor is the pseudo-worker `auditor`. It never records children and never gets parent
assignments.

1. `$CTRL next --worker auditor`. The answer is either `{"action": "idle"}` or
   `{"action": "replicate", "node", "node_rl", "seed", "command", "first_metric", ...}` for the
   oldest PENDING_BEST node. Asking again returns the same job until it is recorded or released.
2. On `replicate`, run exactly the printed command in the auditor workspace (it runs
   `nodes/<id>/rl.py` unmodified with a new seed, `--gpu auto` by default):
   `python frozen/harness.py --out runs/rep_<id>_s<seed> --seed <seed> --rl <node_rl>`.
3. `$CTRL replicate --node <id> --summary runs/rep_<id>_s<seed>/summary.json`. The controller
   checks that the run used the node's `rl.py` (its SHA-256 in `summary.json`) and a seed not used
   before, then resolves the node (see [two-seed rule](#promotion-v9)). A crashed or timed-out
   replicate leaves the node pending; `next --worker auditor` gives it again with the next seed.
   After two failed replicates it becomes `REPLICATE_FAILED` and is not promoted.
4. Append one line to `$ROOT/reflections.md`:
   `- rep n012 | seeds 0,1 | 431, 419 -> mean 425 | BEST (vs n007 409, margin 12) | ...`.
5. When idle, check recent nodes for rule compliance (SPEC.md "Rules"): read the diff of each new
   node's `rl.py` against its parent1, and log any violation (training on anything other than
   verified self-generated proofs, verified by-products or cap-6 records; reading the sealed or
   scorer-only files; editing `frozen/`) in `$ROOT/audit.md` with the node id. Then ask again.
   `release --worker auditor` drops a replicate job that cannot be run.

### Reflections (lock-free append rule)

Each experiment gets exactly one line, appended with a single write. Never edit or reorder
earlier lines. A single `>>` append of one short line is atomic on Linux, so the workers can
append without a lock:

```bash
echo "- n007 | parents n004 | mutate | 612 (+14 vs 598) front 41 | REPLACE_WEAKEST -> DIVERSE | worth revisiting: the kl penalty helped L11 but hurt heldout; try it on the LEAN line" >> $ROOT/reflections.md
```

The fields are: node | parent(s) | operator | metric (delta vs parent1) and frontier count |
decision -> role | one sentence on whether the idea merits revisiting, and under what
conditions. Do not spread a reflection over several lines.

### The research log (`ideas.md`)

`ideas.md` is a Markdown table with one row per idea (not per run):

```
| date | worker | idea | source | nodes | verdict |
|---|---|---|---|---|---|
| 2026-09-25 | w3 | GRPO advantage replaces SFT on L10-11 targets | https://arxiv.org/abs/2402.03300 | n014, n021 | -35, -28: does not help at 1,200 s |
```

Add a row the first time you test an idea, appending one line. Record later tests of the same
idea by appending a new row that names the earlier one ("see row of 2026-09-25 w3"), never by
editing another worker's row. `source` is the same URL (or `own idea`) you gave `--source`. The
verdict gives the numbers (delta vs parent1, and the two-seed mean if replicated) and a
one-clause reading. It stays `pending` until the node is recorded.

## Operators in code terms

- **Mutation** (one parent): one coherent change to parent1's `rl.py`. A coherent change is one
  idea that you can state in one sentence and could revert as a unit. For example, add a KL
  penalty to the initial policy, or replace the sampling temperature schedule. Several edits that
  all serve that one idea are fine. Two unrelated ideas are not: split them into two
  experiments, otherwise neither result can be attributed.
- **Crossover** (two parents): start from parent1's `rl.py`, which is the base and stays the
  base. Read parent2's `rl.py`, find the one idea in parent2 that parent1 lacks (diff them;
  `parent2_desc` says what parent2 tried), and transplant exactly that idea into the base. Adapt
  it as needed so it fits parent1's code. Do not merge the two files wholesale, and do not add a
  new idea of your own. The description must name the transplanted idea and its source node, for
  example "grpo group mean advantage on n002 plus 30 percent cap6 retention records transplanted
  from n001". If the parents turn out to share the idea (for example, parent2 descends from
  parent1), transplant the most distinct remaining difference and say so in the reflection. This
  is also a candidate for a controller step.

## Descriptions and categories

`--desc` is one specific sentence that names the mechanism and its key values: "grpo group mean
advantage replaces the sft step, no kl", not "try grpo" or "improve training". The controller
measures novelty (Eq. 2) and DIVERSE eligibility with the Jaccard distance between lowercased
alphanumeric token sets. Two different ideas described with the same generic words look
identical to it, and one idea described in different words looks novel. Use the concrete nouns of
the mechanism (grpo, kl, retention, cap6, temperature, curriculum, lr, cosine, hindsight...) and
the numbers. Do not describe the outcome ("better", "fixes") and do not repeat the parent's
description. The description is about what this child changed.

`--category` is the broad kind of change. The coverage term (Eq. 1) rewards role x category
combinations that are rare in the frontier.

| category | examples |
|---|---|
| `optimizer` | lr, Adam betas, weight decay, grad clipping, optimizer choice |
| `sampling` | k per target, temperature, rollout length, which targets get sampled, batch sizing of generation |
| `data` | cap-6 retention mix, hindsight relabelling, dedup/normalisation of proofs |
| `curriculum` | ordering or weighting of targets by length or difficulty, focus on unsolved |
| `objective` | SFT vs GRPO vs other losses, baselines/advantages, KL or entropy terms |
| `schedule` | how the 1,200 s are split: rounds, train steps per round, lr schedule over the run |
| `other` | anything else, such as efficiency work that frees time for more samples |

## What the controller does (version 9)

- **Baseline first.** `n000` is the seed. The first `next` returns `baseline`. While the baseline
  is running, other workers are told to `mutate` the unevaluated seed so no GPU is idle.
- **Filling** (frontier smaller than P = 4): mutate the best available evaluated elite; workers
  are kept off each other's p1 while there is a choice.
- **Parent selection** (Eq. 1): p1 is the frontier elite with the highest
  `mean_child_improvement/20 + 0.5*sqrt(log(N+2)/(n_e+1)) + lambda*nov + gamma*cov + rho`.
  - `n_e` counts the times the node served as p1 or p2, incremented the moment `next` assigns it.
  - `nov` is the minimum Jaccard distance to the other p1s of the last 4 assignments.
  - `cov` is `1/sqrt(k+1)`, where `k` counts the other elites with the same role and category.
  - `rho` is -0.2 if the node was p1 at the previous assignment step and -0.05 two steps ago.
  - Lambda and gamma are 0.3, and fall to 0.15 after 8 completed experiments.
  - An elite that is currently p1 for another worker is excluded while another candidate remains,
    unless every other candidate is more than the share band below it (v2). With seven workers
    and four elites, sharing is common once all four are held.
  - An elite more than the p1-floor band below the BEST is excluded if another remains (v3).
  - An elite that was p1 in each of the last 2 steps is skipped if another candidate remains.
  - `next` prints the full per-candidate breakdown in `p1_scores`.
- **Crossover** is forced when the frontier is full and fewer than 1 of the last 4 experiments
  (completed plus other workers' in-flight) was a crossover, and right after a mutation child was
  promoted into a full frontier (a "fresh elite"). Refinements from run 1:
  - v4: a fresh elite that scored below its own parent1 does not force a crossover.
  - v5: p2 is never p1's ancestor or descendant (if another candidate remains).
  - v6: if every other elite is p1's lineage, the forced crossover becomes a mutation of p1.
  - v7: an elite whose own change lost to its parent1 by more than the loss band is not a p2.
  - v8: a clear leader (ahead of every other elite by more than the share band) is mutated
    instead of a due crossover. v9 caps this: once 4 other workers hold the leader as p1, the
    normal policy applies.
  - Otherwise the action is `mutate`. p2 maximises Eq. 3:
    `jaccard(p1,e) + 0.2*[role differs] - 0.1*(prior co-uses of the pair)` (`p2_scores`).

### Promotion (v9)

The margins scale with the score. With `best` = the current BEST's metric (dev L8-11 pass@32):

| quantity | value | at best = 400 | at best = 1,000 |
|---|---|---|---|
| improvement margin | `max(8, round(0.6 * sqrt(best)))` | 12 | 19 |
| tie margin (LEAN band, DIVERSE slack) | 2 x improvement | 24 | 38 |
| loss band (v7) | 1 x tie | 24 | 38 |
| share band (v2, v8) | 2 x tie | 48 | 76 |
| p1-floor band (v3) | 3.5 x tie | 84 | 133 |

`next`, `frontier` and `record` print the current values under `margins`. For a child with a
metric, the checks run in this order:

1. **Held-out floor.** `heldout_greedy < 0.85` (or missing): decision `FLOOR`. The node is
   evaluated and kept in `nodes/` and `history`, but never promoted.
2. **Two-seed rule for BEST.** If a BEST exists and `metric > best + improvement margin`: decision
   `PENDING_BEST`. The node is not in the frontier and cannot be a parent. It waits in the
   auditor's queue. When the auditor records the replicate, the node's `metric`,
   `heldout_greedy`, `frontier` and `by_len` become the **two-seed means**, and the node goes
   through checks 1 and 3 again with those means: it becomes BEST only if the mean still beats
   the BEST at that moment by the margin; otherwise it is demoted to an ordinary candidate and
   gets whatever the normal rules give (`REPLACE_WEAKEST`, `LEAN`, `DIVERSE` or `DISCARD`).
3. **Normal promotion** (single seed). With an empty slot, the child fills it (FILL). Otherwise:
   1. `metric > best + improvement`: BEST (only reachable through check 2 or an empty frontier).
      The weakest non-best elite is evicted.
   2. `metric > weakest + improvement`: REPLACE_WEAKEST.
   3. `metric >= best - tie` and `heldout_greedy >= LEAN's + 0.01` (the BEST's if there is no
      LEAN): LEAN. The child replaces the current LEAN.
   4. `metric >= weakest - tie` and Jaccard distance `>= 0.65` from every elite's description:
      DIVERSE. The weakest elite that is neither BEST nor LEAN is evicted.
   5. Otherwise DISCARD.

**BEST is sticky.** The BEST role moves only to a node that beat the BEST by the improvement
margin, which after the first elite means a confirmed two-seed mean. A single-seed child that
lands above the BEST but within the margin enters as an ordinary elite and does not take the
role. LEAN is the elite within the tie margin of the BEST with the highest `heldout_greedy`, at
least 0.01 above the BEST's. Every other elite is DIVERSE.

**Node statistics**: `n_children` counts every child, including crashes.
`mean_child_improvement` is the mean of `child.metric - node.metric` over successful children
(two-seed means where they exist). Both parents of a crossover are credited.

Noise: one seed of the dev score moves by roughly the tie margin between runs. The improvement
margin is below that on purpose: a slightly lucky child may enter the frontier as an ordinary
elite and must then keep producing good children. The BEST role, which the final evaluation uses,
needs two seeds.

## GEAR-Evolve: controller steps

At every iteration, before `next`, you decide:

- **Experiment step**: run one child with the current controller (the loop above), or
- **Controller step**: edit the search policy in `controller.py` because the current policy is
  producing bad suggestions.

Write the decision first, as one line appended to `<root>/decisions.md`:

```
- 2026-09-23T14:05Z | W=A | experiment | step 12 | search healthy: 3 distinct p1 in last 4, last crossover promoted
- 2026-09-23T14:31Z | W=B | controller | after exp 14 | crossover paired n009 with its own parent n006 twice; add ancestor exclusion to pick_p2
```

The format is: time | worker | experiment/controller | position | reason. The reason for a
controller step names the concrete bad suggestion you saw and the fix.

**The five-step rule.** Count your own consecutive experiment lines. After 5 consecutive
experiment steps, your next decision line must be either a controller step or an experiment step
with an explicit justification that the search is healthy. The justification must cite evidence
from `frontier`/`history`: distinct p1s recently, crossover pairs not repeating, recent
promotions, elites not all minor variants of one line. "Continuing" is not a justification.

Symptoms that warrant a controller step:
- Crossover pairs a node with its own ancestor.
- The same pair repeats.
- A weak old ancestor keeps winning p1 on inflated productivity. Children of a weak node improve
  on it easily, so its `mean_child_improvement` is large.
- A known-bad elite keeps being chosen.
- DIVERSE slots are filled by near-duplicates.
- One line dominates every p1.
- Several workers get redundant assignments (with seven workers and four elites some sharing is
  unavoidable; the same p1 for four or more workers at once is not).
- The auditor's replicate queue grows faster than it drains (more than two PENDING_BEST nodes).

**How to make a controller step** (keep the CLI contract and the state format compatible):

1. Take the edit lock so two workers do not edit at the same time:
   `exec 9>/scratch/cowifornia/nd-rl/ab2/controller_edit.lock; flock -n 9 || { echo "another worker is editing"; }`.
   If the lock is busy, do an experiment step instead.
2. `cp $ROOT/controller.py $ROOT/controller.new.py` and edit the copy. Keep:
   - the commands and their flags;
   - the JSON keys that workers read (`action`, `parent1`, `parent1_rl`, `parent2`, `parent2_rl`,
     `notes`, `decision`, `role`, ...);
   - the state file layout. You may add keys; give them defaults when they are missing so the
     old state still loads. Bump `STATE_VERSION` only if you also add a migration in
     `Store.load`.

   Bump `CONTROLLER_VERSION`. Make one policy change per controller step.
3. Self-check against the live state: `python3 $ROOT/controller.new.py --root $ROOT --dry-run`.
   It must print `"ok": true` and a frontier. The command exits non-zero if the state fails
   validation. Also try `python3 $ROOT/controller.new.py --root $ROOT next --worker W --dry-run`
   and check that the answer now avoids the failure you saw. `--dry-run` writes nothing. If
   `test_controller.py` sits next to it, copy it and run it against the new file.
4. Install atomically: `mv $ROOT/controller.new.py $ROOT/controller.py`. The other workers' next
   calls pick it up, and the state is untouched.
5. Log the result in `decisions.md` (a second line: "controller step done: <what changed>, dry-run
   ok, version N").

Never hand-edit `gear_state.json` or `nodes/`.

The two-seed rule, the held-out floor, `--source`, the seven-worker cap and the auditor commands
are part of the run's protocol (SPEC.md), not search policy: a controller step may not remove or
weaken them. Policy changes (parent selection, crossover triggers, the Eq. 1 / Eq. 3 weights, the
DIVERSE rule, the band multipliers) are fair game.

## Worked example: the two-seed rule

The BEST is n007 (dev 409; margin `max(8, round(0.6 * sqrt(409)))` = 12). Worker w4 records n012
with 431:

```
$ $CTRL record --worker w4 --node-rl ./rl.py --summary runs/x12/summary.json \
    --desc "k 48 samples per target on L10-11 targets only, k 16 on L6-9" --category sampling \
    --source "own idea"
{"ok": true, "node": "n012", "decision": "PENDING_BEST", "role": null, "metric": 431,
 "frontier_count": 57, "pending_best": ["n012"],
 "notes": ["beats BEST n007 (409) by more than 12: PENDING_BEST until the auditor's second seed; not in the frontier yet"], ...}

$ $CTRL next --worker auditor
{"action": "replicate", "node": "n012", "seed": 1,
 "command": "python frozen/harness.py --out runs/rep_n012_s1 --seed 1 --rl /scratch/cowifornia/nd-rl/ab2/nodes/n012/rl.py", ...}

$ python frozen/harness.py --out runs/rep_n012_s1 --seed 1 --rl /scratch/cowifornia/nd-rl/ab2/nodes/n012/rl.py
$ $CTRL replicate --node n012 --summary runs/rep_n012_s1/summary.json
{"ok": true, "node": "n012", "decision": "BEST", "role": "BEST", "metric": 425.0, "metric_seeds": [431, 419],
 "notes": ["two-seed mean 425 (seeds [0, 1]: [431, 419]) vs BEST n007 409, margin 12"], ...}
```

Had the second seed scored 395, the mean (413) would not beat 409 + 12, and n012 would have been
demoted to the ordinary rules with 413: with the weakest elite at, say, 390, that is more than 12
above it, so `REPLACE_WEAKEST`, and n007 would have stayed BEST.
