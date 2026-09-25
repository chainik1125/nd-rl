# GEAR controller: worker protocol

This directory has the GEAR-Fixed search controller (Jeddi et al. 2026, "GEAR: Genetic
AutoResearch for Agentic Code Evolution", sections 3.2 and 3.3), adapted to the RL hill-climb in
`../SPEC.md`. Two workers, each with its own GPU and workspace, share one frontier of up to four
`rl.py` variants. The controller decides which parent(s) each worker starts from, whether to
mutate or cross over, and whether a finished child joins the frontier. The controller never runs
experiments and never touches a GPU. You run the experiments.

Files:

| file | role |
|---|---|
| `controller.py` | the controller: one stdlib-only file. Every command prints one JSON object |
| `test_controller.py` | simulated two-worker test (`python3 test_controller.py`) |
| `<root>/gear_state.json` | the authoritative state. Only the controller writes it |
| `<root>/nodes/<id>/` | `rl.py` (that node's code), `meta.json` (a copy of the node record), `summary.json` (a copy of the run's summary) |
| `<root>/reflections.md` | one line per experiment, appended by workers |
| `<root>/decisions.md` | one line per iteration, experiment step or controller step (GEAR-Evolve) |

`<root>` defaults to `/work` (the arm directory inside the container). Override it with `--root`
or the `GEAR_ROOT` environment variable. Below, `CTRL` is the path your brief gives for the
editable controller. If the brief gives none, use `/work/controller.py`:

```bash
CTRL="python3 /work/controller.py --root /work"
```

## The loop

Do these steps in order on every iteration. `W` is your worker name from the brief (for example
`A` or `B`). Always use the same name.

0. **Decide** whether this iteration is an experiment step or a controller step (see
   [GEAR-Evolve](#gear-evolve-controller-steps)). Append the decision line to
   `<root>/decisions.md` before doing anything else.
1. **Ask**: run `$CTRL next --worker W`. Read `action`, `parent1`, `parent1_rl`, `parent1_desc`,
   `parent2`, `parent2_rl`, `parent2_desc` and `notes`. If you already had an assignment, `next`
   gives you the same one again (the notes say "already assigned").
2. **Start from the parent**: run `cp <parent1_rl> ./rl.py`. Always start from `parent1_rl`, never
   from whatever was in your workspace before.
3. **Edit** according to `action`:
   - `baseline`: change nothing. Run the seed exactly as it is.
   - `mutate`: make one coherent change to `./rl.py` (see [Operators](#operators-in-code-terms)).
   - `crossover`: keep parent1's code as the base, read `parent2_rl`, and transplant exactly one
     coherent idea from parent2 into `./rl.py`.
4. **Run** the frozen harness in your workspace:
   `python frozen/harness.py --out runs/<name> --seed 0`. Use a new `<name>` for every run.
   Always let it finish, including a crash or timeout. It writes `runs/<name>/summary.json`.
5. **Record**:
   ```bash
   $CTRL record --worker W --node-rl ./rl.py --summary runs/<name>/summary.json \
       --desc "<one sentence>" --category <category>
   ```
   The controller takes the parent(s) and the operator from your in-flight assignment. It copies
   `rl.py` and `summary.json` into `nodes/<new id>/`, applies the promotion rules and prints
   `decision`, `role`, `evicted` and `delta` (the child's metric minus parent1's). For a
   `baseline`, `--desc` and `--category` are optional. A crashed or timed-out run is recorded
   like any other: pass its summary, or a missing path if the harness died before writing one.
   It becomes `CRASH` or `TIMEOUT`, counts as a completed experiment, and never enters the
   frontier.
6. **Reflect**: append one line to `<root>/reflections.md` (format below).
7. Go back to step 0.

If you are interrupted between `next` and `record` and the run is lost, run
`$CTRL release --worker W`. This drops the assignment and undoes its bookkeeping (parent use
counts, co-use counts, and a pending fresh-elite crossover). Never `release` a run that finished.
Record it instead, even if it crashed.

Other commands: `$CTRL frontier` shows the current elites (JSON with a `table` field of
human-readable lines). `$CTRL history` shows every node with its parents, operator, metric and
decision. `$CTRL next --worker W --dry-run` shows what you would get without reserving anything.
If you need to record a run under different parents than the ones assigned, use
`--parents n003,n007 --operator crossover`. That is an exception, and you should give the reason
in the reflection.

### Reflections (lock-free append rule)

Each experiment gets exactly one line, appended with a single write. Never edit or reorder
earlier lines. A single `>>` append of one short line is atomic on Linux, so the two workers can
append without a lock:

```bash
echo "- n007 | parents n004 | mutate | 142 (+14 vs 128) | REPLACE_WEAKEST -> DIVERSE | worth revisiting: the kl penalty helped L9 but hurt heldout; try it on the LEAN line" >> /work/reflections.md
```

The fields are: node | parent(s) | operator | metric (delta vs parent1) | decision -> role |
one sentence on whether the idea merits revisiting, and under what conditions. Do not spread a
reflection over several lines.

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
| `data` | cap-6 retention mix, extra targets pool, hindsight relabelling, dedup/normalisation of proofs |
| `curriculum` | ordering or weighting of targets by length or difficulty, focus on unsolved |
| `objective` | SFT vs GRPO vs other losses, baselines/advantages, KL or entropy terms |
| `schedule` | how the 600 s are split: rounds, train steps per round, lr schedule over the run |
| `other` | anything else, such as efficiency work that frees time for more samples |

## What the controller does (GEAR-Fixed, as implemented)

- **Baseline first.** `n000` is the seed. The first `next` returns `baseline`. While the baseline
  is running, the other worker is told to `mutate` the unevaluated seed so the GPU is not idle.
- **Filling** (frontier smaller than P = 4): mutate the best evaluated elite. The second worker
  gets the next best, so the two workers never share p1 when there is a choice. Every successful
  child fills an empty slot.
- **Parent selection** (Eq. 1): p1 is the frontier elite with the highest
  `mean_child_improvement/20 + 0.5*sqrt(log(N+2)/(n_e+1)) + lambda*nov + gamma*cov + rho`.
  - `n_e` counts the times the node served as p1 or p2. It is incremented the moment `next`
    assigns it.
  - `nov` is the minimum Jaccard distance to the other p1s of the last 4 assignments.
  - `cov` is `1/sqrt(k+1)`, where `k` counts the other elites with the same role and category.
  - `rho` is -0.2 if the node was p1 at the previous assignment step and -0.05 if it was p1 two
    steps ago.
  - Lambda and gamma are 0.3, and fall to 0.15 after 8 completed experiments.
  - An elite that is currently p1 for the other worker is excluded, and so is an elite that was
    p1 in each of the last 2 steps, provided another candidate remains.
  - `next` prints the full per-candidate breakdown in `p1_scores`.
- **Crossover** is forced in two cases:
  - When the frontier is full and fewer than 1 of the last 4 experiments was a crossover. That
    count covers completed experiments plus the other worker's in-flight one.
  - Right after a mutation child was promoted into a full frontier (a "fresh elite"). The next
    `next` crosses the fresh elite with the pool. If the fresh elite is BEST, it is p1. Otherwise
    it is p2, transplanted onto the highest-scoring other elite.
  - Otherwise the action is `mutate`. p2 maximises Eq. 3:
    `jaccard(p1,e) + 0.2*[role differs] - 0.1*(prior co-uses of the pair)`. `next` prints it in
    `p2_scores`.
- **Promotion** (checks in order). With an empty slot, the child fills it (FILL). Otherwise:
  1. `metric > best + 5`: BEST. The weakest non-best elite is evicted.
  2. `metric > weakest + 5`: REPLACE_WEAKEST.
  3. `metric >= best - 10` and `heldout_greedy >= LEAN's + 0.01` (the BEST's, if there is no
     LEAN): LEAN. The child replaces the current LEAN.
  4. `metric >= weakest - 10` and Jaccard distance `>= 0.65` from every elite's description:
     DIVERSE. The weakest elite that is neither BEST nor LEAN is evicted.
  5. Otherwise DISCARD. The node stays in `nodes/` and `history`, but not in the frontier.

  After every change the roles are relabelled:
  - BEST is the highest metric.
  - LEAN is the elite within 10 of the best whose `heldout_greedy` is highest, provided it is at
    least 0.01 above the BEST's. In this setting LEAN means "same score, least in-distribution
    damage".
  - Every other elite is DIVERSE.
- **Node statistics**: `n_children` counts every child, including crashes.
  `mean_child_improvement` is the mean of `child.metric - node.metric` over successful children.
  Both parents of a crossover are credited.

Noise: the SPEC says run-to-run noise is about 15 theorems. The margins (5 and 10) are below
that. This is deliberate for population search: a slightly lucky child enters the frontier and
must then keep producing good children. Do not treat a +6 promotion as evidence. Re-run a
promising node with another seed if a claim depends on it.

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
- The two workers get redundant assignments.

**How to make a controller step** (keep the CLI contract and the state format compatible):

1. Take the edit lock so the two workers do not edit at the same time:
   `exec 9>/work/controller_edit.lock; flock -n 9 || { echo "other worker is editing"; }`.
   If the lock is busy, do an experiment step instead.
2. `cp /work/controller.py /work/controller.new.py` and edit the copy. Keep:
   - the commands and their flags;
   - the JSON keys that workers read (`action`, `parent1`, `parent1_rl`, `parent2`, `parent2_rl`,
     `notes`, `decision`, `role`, ...);
   - the state file layout. You may add keys; give them defaults when they are missing so the
     old state still loads. Bump `STATE_VERSION` only if you also add a migration in
     `Store.load`.

   Bump `CONTROLLER_VERSION`. Make one policy change per controller step.
3. Self-check against the live state: `python3 /work/controller.new.py --root /work --dry-run`.
   It must print `"ok": true` and a frontier. The command exits non-zero if the state fails
   validation. Also try `python3 /work/controller.new.py --root /work next --worker W --dry-run`
   and check that the answer now avoids the failure you saw. `--dry-run` writes nothing. If
   `test_controller.py` sits next to it, copy it and run it against the new file.
4. Install atomically: `mv /work/controller.new.py /work/controller.py`. The other worker's next
   call picks it up, and the state is untouched.
5. Log the result in `decisions.md` (a second line: "controller step done: <what changed>, dry-run
   ok, version N").

Never hand-edit `gear_state.json` or `nodes/`.

## Worked example

After a baseline (41) and three filling mutations (63, 70, 58), the frontier is full. None of the
last 4 experiments was a crossover, so worker A is forced to cross over (output shortened):

```
$ $CTRL next --worker A
{
 "action": "crossover", "worker": "A", "step": 5,
 "parent1": "n000", "parent1_rl": "/work/nodes/n000/rl.py",
 "parent1_desc": "seed baseline rl.py unmodified", "parent1_role": "DIVERSE", "parent1_metric": 41,
 "parent2": "n002", "parent2_rl": "/work/nodes/n002/rl.py",
 "parent2_desc": "grpo group mean advantage replaces the sft step, no kl",
 "parent2_category": "objective", "parent2_role": "BEST", "parent2_metric": 70,
 "completed_experiments": 4, "lambda": 0.3, "gamma": 0.3,
 "frontier": [ ... 4 rows ... ],
 "notes": ["crossover forced: 0 crossover(s) in the last 4 experiments ['baseline', 'mutate', 'mutate', 'mutate']"],
 "p1_scores": {
  "n000": {"total": 2.1485, "productivity": 1.1, "ucb": 0.4485, "novelty": 1.0, "coverage": 1.0, "recency": 0.0, "n_uses": 1, "N": 3},
  "n001": {"total": 1.3355, "productivity": 0.35, "ucb": 0.4485, "novelty": 0.9565, "coverage": 1.0, "recency": -0.05, ...},
  "n002": {"total": 0.2355, "productivity": -0.6, "ucb": 0.4485, "novelty": 0.9565, "coverage": 1.0, "recency": -0.2, ...},
  "n003": {"total": 1.2228, "productivity": 0.0, "ucb": 0.6343, "novelty": 0.9615, "coverage": 1.0, "recency": 0.0, ...}},
 "p2_scores": {"n001": {"total": 1.2, "jaccard": 1.0, "role_mismatch": true, "co_use": 0},
               "n002": {"total": 1.2, "jaccard": 1.0, "role_mismatch": true, "co_use": 0},
               "n003": {"total": 1.0, "jaccard": 1.0, "role_mismatch": false, "co_use": 0}}
}
```

Meanwhile worker B asks and is kept off A's p1:

```
$ $CTRL next --worker B
{"action": "mutate", "worker": "B", "step": 6, "parent1": "n001", "parent2": null,
 "notes": ["excluded ['n000']: in flight as p1 for another worker", "mutate the highest Eq. 1 score"], ...}
```

A copies `nodes/n000/rl.py`, transplants the GRPO objective from `nodes/n002/rl.py`, runs the
harness and records:

```
$ $CTRL record --worker A --node-rl ./rl.py --summary runs/x05/summary.json \
    --desc "grpo group mean advantage from n002 transplanted onto the seed sampling loop" --category objective
{"ok": true, "node": "n004", "decision": "BEST", "role": "BEST", "evicted": "n000",
 "metric": 77, "parent1_metric": 41, "delta": 36, "status": "evaluated",
 "operator": "crossover", "parents": ["n000", "n002"], "completed_experiments": 5, ...}

$ echo "- n004 | parents n000+n002 | crossover | 77 (+36 vs 41; +7 vs n002) | BEST -> BEST | revisit: gain over n002 is within noise, re-run before building on it" >> /work/reflections.md

$ $CTRL frontier | python3 -c "import json,sys; print('\n'.join(json.load(sys.stdin)['table']))"
id    role     metric by_len           heldout kids uses meanimp last category   description
n004  BEST         77 62/10/5            0.800    0    0       -    - objective  grpo group mean advantage from n002 transplanted onto the seed sampling loop
n002  DIVERSE      70 60/6/4             0.780    2    2    -2.5    4 objective  grpo group mean advantage replaces the sft step, no kl
n001  DIVERSE      63 53/6/4             0.810    1    2    +7.0    6 data       expert iteration sft on verified samples with 30 percent cap6 retention records per batch
n003  DIVERSE      58 48/6/4             0.850    0    0       -    - sampling   sample 64 per target at temperature 1.0 restricted to still unsolved targets
```

This example also shows a known weakness of the fixed policy. The seed `n000` won p1 on
productivity 1.1 because all its filling-phase children improved on it, and its p2 `n002` is its
own grandchild. That is exactly the kind of degenerate crossover a controller step should fix
(compare the paper's Table 3, edits 2 and 4).
