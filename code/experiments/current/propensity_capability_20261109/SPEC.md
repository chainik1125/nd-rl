---
written_on: 2026-11-09
written_by: agent:claude (orchestrator), for Leon
---

# Propensity versus capability by proof direction: build spec

## The idea (Leon, 2026-11-06 and 2026-11-09)

A theorem can often be proved in more than one direction (directly, by contradiction, by cases).
Which direction a model takes when left alone is its **propensity**. Whether it can finish when it
is made to take a given direction is its **capability**. Forcing the direction separates the two:

    success per attempt via direction d  ≈  P(takes d unprompted) × P(finishes | takes d)

Read at a base model and after RL, the pattern says what RL did:

| what changes under RL | reading |
|---|---|
| takes d more often, finishing when forced stays flat | elicitation: RL selected something the base could already do |
| finishing when forced rises | a candidate capability gain |
| both rise | a mix, sized by the two factors |

The factor read of 2026-11-06 (`../factor_read_20261106/`) did this for one direction (the reductio
route, forced from a reference proof) on the 106 theorems that require it. This build generalises
it: several directions, theorems where the direction is optional, any checkpoint in Dan's state
format. "Prompting" here means forcing the model's first proof action; these models take no
natural-language prompt.

## Directions

Each direction is defined by an opening action in Dan's state environment that can be forced as the
first action of an attempt. Agent A confirms the exact action form of each from Dan's `state_env`
and `factor_lib.py`, and that the environment accepts it.

| id | opening | applicable when |
|---|---|---|
| `reductio` | open a box assuming ¬G for the goal G (to reach ⊥, close by negation introduction, then double-negation elimination) | always |
| `negation` | goal is ¬A: open a box assuming A | goal is a negation |
| `conditional` | goal is A → B: open a box assuming A | goal is an implication |
| `cases` | a premise is A ∨ B: start the case split on it (the first disjunctive premise) | some premise is a disjunction |
| `direct` | first action opens no box | classification only; it is not forced |

Opening one of these never makes a provable theorem unprovable, but the step cap can bind; record
attempts that end by the cap separately from refusals and Lean rejections.

## Arms, per checkpoint and theorem

- **U (unforced):** k attempts sampled as our existing reads do (temperature 0.8, Dan's
  `env_generate` settings). For each attempt record: Lean verdict, the direction of its first
  action, and which directions it uses anywhere (a reductio box on the goal, a case split on a
  premise, and so on), judged with the verifier and parser, never by string matching alone.
- **F_d (forced):** for each applicable d except `direct`, k_f attempts whose first action is
  forced to d's opening. Record the Lean verdict and how the attempt ended.
- **E (exact):** for each applicable d except `direct`, the exact log-probability of d's opening
  as the first action under softmax(logits / 0.8), with variable names marginalised or scored
  name-free as `factor_lib.score_rows` does. No sampling. This is what resolves propensities near
  zero.

Defaults to size against the GPU budget: k = 32, k_f = 16.

## Sets and checkpoints

- Theorems: the 106 strict theorems, plus a fixed subsample of the other 726 in
  `transfer_track832.jsonl` (default 240, stratified by `n_lines` and `source`, seed 0, the same
  theorems for every checkpoint). These are evaluation sets: read-only, nothing trained on them.
- Checkpoints, in priority order if the budget binds: Dan's full-corpus base `stage1_full_s3.pt`;
  the 1% bases and their RL ends (3 seeds); the 0% bases and RL ends; the 0.1% bases and RL ends.
  Paths for the dose lines are in `../factor_read_20261106/ckpts.json`. Check each path exists:
  intermediate checkpoints were deleted on 2026-11-09, bases and ends were kept.

## Result files (the contract between the runner and the analysis)

One file per checkpoint, `/scratch/cowifornia/nd-rl/propensity_capability/results/<ckpt_id>.jsonl`,
one JSON object per line, no proof text:

    {"ckpt": str, "dose": "full"|"0"|"0.1"|"1", "seed": int, "stage": "base"|"rl_end"|"matched",
     "x_dan": float|null, "set": "strict106"|"transfer_other", "name": str, "n_lines": int,
     "schema": str|null, "source": str,
     "arm": "U"|"F"|"E", "direction": "any"|"reductio"|"negation"|"conditional"|"cases"|"direct",
     "applicable": bool, "k": int,
     "dir_class": "required"|"helpful"|"idle"|null,  # added 2026-11-09, see "Router or default shift"
     "solved": int,          # U/any: attempts Lean accepted. U/<d>: accepted attempts that use d.
                             # F/<d>: accepted attempts.
     "used": int|null,       # U/<d>: attempts that use d anywhere. Otherwise null.
     "first": int|null,      # U/<d>: attempts whose first action is d's opening. Otherwise null.
     "solved_first": int|null,  # U/<d>: accepted attempts whose first action is d's opening. Otherwise
                             # null. (added by the runner 2026-11-09; analyze.py reads it as optional)
     "capped": int|null,     # attempts ended by the step cap
     "refused": int|null,    # attempts ended by an illegal action or a refusal
     "logp_open": float|null,  # E/<d>: natural log of P(first action is d's opening). Otherwise null.
     "tokens": int|null}     # tokens generated in this row's attempts

Raw attempts (which hold proof text) go to
`/scratch/cowifornia/nd-rl/propensity_capability/raw/<ckpt_id>/` and are never committed.

If the runner has to change this contract, it changes this section of the file in the same commit
and tells the orchestrator, who tells the analysis side.

## What the analysis reports

Per checkpoint and direction, over applicable theorems, as % of attempts with intervals from a
bootstrap over theorems:

- propensity: `used / k` from U, and the exact first-move probability from E;
- capability: `solved / k` from F;
- unforced success, and the share of it that goes through each direction;
- finishing given the model chose d unprompted (`solved / used` from U), next to finishing when
  forced, since their gap is the selection effect.

Then base → RL end for each dose and seed: the change in propensity and in capability, the reading
from the table above under thresholds fixed in `PREREG.md` before any result is seen, and one
figure with propensity on one axis and capability on the other, an arrow from base to RL end per
seed, one panel per direction. Figures go through `code/ndbench/figures/style.py` (5.5 in wide,
serif 7–8 pt, PDF, SVG and PNG, colour-blind-safe palette).

## Decisions after the pre-registration draft (2026-11-09 evening PT, before any result)

- **"First action" means the first action after the premise lines**, since the environment only
  accepts premise lines first. E and F work from that point.
- **"Uses reductio anywhere" counts a reductio on any formula**, not only on the goal. The factor
  read showed that many required reductios are on a subgoal. The forced opening and the first-move
  propensity are on the goal. These are two different propensities and are reported separately.
- **`solved_first` is added to U rows**, so the identity P(solve and open with d) = propensity ×
  capability can be checked against the forced arm.
- **`stage: "matched"` is allowed.** The 0% and 0.1% checkpoints at the RL compute where the 1%
  lines end (listed in `../factor_read_20261106/ckpts_matched.json`, files under
  `/scratch/cowifornia/nd-rl/factor_read/matched/ckpts/`) are read as a secondary set after the
  bases and RL ends, if the GPU budget allows. They are the only way to compare doses at equal
  compute.

## Router or default shift (added 2026-11-09 evening PT, before any result)

When RL raises a direction's propensity, it can rise only where the direction is needed (a router:
the disposition stays tied to context) or everywhere (a shift of the default, the kind of change
that shows up off-distribution). To tell them apart, each theorem and direction gets a class,
fixed before the read:

- `required`: no proof within the step cap avoids the direction. For `reductio` the 106 strict
  theorems are required by Dan's definition; for other directions, from the shortest-proof search
  if it can decide this, otherwise null.
- `helpful`: a proof avoiding the direction exists, but the shortest proof using it is shorter.
- `idle`: the direction does not shorten the proof.

If the shortest-proof search cannot separate `helpful` from `idle` cheaply on CPU, the runner sets
both to null for the optional theorems and says so; the analysis then compares `required` (the 106)
with the optional set as a whole. The analysis reports the base → RL-end change in propensity in
each class and the ratio between classes, as a pre-registered secondary analysis.

## Known traps

- On a theorem where reductio is optional, a forced reductio can be finished by proving the goal
  directly inside the box and then contradicting the assumption. That is a real proof, but it means
  forced-reductio capability there is partly direct-proof ability. Report the 106 (where the route
  is required) and the optional set separately, and say this.
- Forced first actions put the model in states it rarely visits; low finishing when forced can
  reflect that unfamiliar state and not a missing skill.
- "Uses d anywhere" and "opens with d" are different propensities. Keep both; do not merge them.
- A propensity of zero in k attempts is not zero. Use E for anything below about 1/k.

## Rules for this build

- No GPU jobs from the build agents: unit tests and CPU smoke runs only. The orchestrator launches
  the read on one lab GPU when it is free. Hand over one gated command (`gpu.py` check, lock,
  `timeout`, progress log), a time estimate, and a `--dry-run` that lists what it would do.
- `textbook72` and the long test sets stay unread. Nothing goes to Runpod.
- Do not commit or push. Do not edit `../factor_read_20261106/`; import from it or copy what you
  need into this folder with a note saying where it came from.
