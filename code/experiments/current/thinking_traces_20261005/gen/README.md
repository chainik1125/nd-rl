---
written_on: 2026-10-05
written_by: agent:claude
---

# Calibration of LLM trace generation (proposed, not run)

Status: **nothing here has run beyond free dry runs**. No API call, no GPU. The paid steps and the
distractor sampling wait for Leon's approval ([PROPOSAL.md §8](../PROPOSAL.md), decision 3(i)).
Code: [`code/thinking/gen/`](../../../../thinking/gen/README.md), branch `leon/thinking`,
uncommitted.

## What the calibration decides

About 50 stratified K12 proofs × {Opus 5.5, Sonnet 5.5, Haiku 4.5}, reference-guided and
error-seeded, at effort `low`. It measures, per model:
- acceptance under the strict validator, first pass and after up to 2 feedback retries;
- **billed tokens per accepted trace** (input, cache read, cache write, output, retries and
  rejected proofs included), the figure for Leon's rule: use Sonnet 5.5 only where it uses under 2×
  Opus 5.5's tokens;
- error categories, which drive the next prompt revision;
- think tokens per trace;
- the deviation rate.

Config: [`calibration.yaml`](calibration.yaml). Run directory:
`/scratch/cowifornia/nd-rl/think/gen_runs/calib_20261006/` (not created yet).

Selection (dry run, seed 20261006): 50 proofs, 451 states, 226 decision states.
- Lines 2–4 / 5–7 / 8–10 / 11–12: 7 / 14 / 16 / 13.
- ORE: 23. DN: 24. Premises 0 / 1 / 2 / 3: 12 / 15 / 13 / 10.
- 30 K12 rows are excluded by renaming class:
  - p2 heldout: 22, all by premise order (`pkey`);
  - pilot rows: 6;
  - textbook72 dev: 1;
  - Charles candidate_v0: 1;
  - every OOD tier: 0.

## Commands (repository root `/scratch/cowifornia/nd-rl/think/nd-rl`)

```bash
G="uv run --project code --no-sync python -m code.thinking.gen"
C=code/experiments/current/thinking_traces_20261005/gen/calibration.yaml
RUN=/scratch/cowifornia/nd-rl/think/gen_runs/calib_20261006

# 1. select (free, CPU, about 6 s)
$G select --config $C
# 2. distractors: NEEDS APPROVAL. CPU, 226 states x 16 samples (estimate 2-5 min from the
#    measured 48 samples in 1.5 s); nvidia-smi first if --device cuda is chosen instead
$G distractors --config $C
# 3. build the requests; prints the dry-run estimate (free)
$G build --config $C
# 4. dry run of a submission (the default; writes requests/<tag>.r1.estimate.json, sends nothing)
$G submit --config $C --requests $RUN/requests/opus55.r1.jsonl
# 5. submit: PAID, NEEDS APPROVAL. Set each --max-usd from the worst case that step 4 prints
#    (about 9.64 / 4.82 / 2.41 without distractors)
export ANTHROPIC_API_KEY=...   # in the shell only; never in the config
$G submit --config $C --requests $RUN/requests/opus55.r1.jsonl   --submit --max-usd 10
$G submit --config $C --requests $RUN/requests/sonnet55.r1.jsonl --submit --max-usd 5
$G submit --config $C --requests $RUN/requests/haiku45.r1.jsonl  --submit --max-usd 2.5
# 6. poll (free); check every 30 min; most batches end within 1 h, at most 24 h
$G poll --config $C
# 7. collect (free): validate, write stats and report, write retry requests requests/<tag>.r2.jsonl
$G collect --config $C
# 8. retries: PAID. Dry-run, then submit each new file and repeat 6-7, once for r2 and once for r3
$G submit --config $C --requests $RUN/requests/opus55.r2.jsonl
$G submit --config $C --requests $RUN/requests/opus55.r2.jsonl --submit --max-usd <its worst case>
# 9. report
$G report --config $C        # also $RUN/collect/stats.json, $RUN/ledger.jsonl
```

Add `--lean` to `collect` to check every accepted trace in Lean 4 as well. It is not batched:
measured at 1.35 s per trace, one Lean call each (8 pilot traces). That is about 1 min for 50 and
about 45 min for 2,000.

The ledger records every submission: batch id, model, request count, estimated cost and
`--max-usd`. After each `collect`, a `usage` row adds the billed tokens and cost read from the
results.

## Cost estimate (verified prices, batch, 1 h cache; arithmetic from the dry run)

Per request:
- S = 5,733 tokens of system prompt (production prompt + 2 worked examples), cached;
- U = 1,665 tokens of user turn (states, reference actions and an assumed 3 distractors per
  decision state);
- O = 664 tokens of visible JSON (3 × the action tokens + 10 per step);
- T = thinking tokens.

Assumptions: cache hit rate h = 0.7, since batch cache hits are best effort (30–98%); r = 1.3
attempts per proof. The real distractor file replaces the assumed 3 per state, and most states
will have fewer.

Opus 5.5, T = 2,000, per attempt ($/MTok batch):

    5,733 × (0.7 × 0.10 + 0.3 × 4.00) + 1,665 × 2.00 + (664 + 2,000) × 10.00
      = 7,281 + 3,330 + 26,640 µ$ = $0.0373;  × 1.3 × 50 = $2.42

Sonnet 5.5 halves every term except the cache read, giving $1.22. Haiku 4.5 gives $0.61: its
thinking is capped by `budget_tokens` 4096, and caching applies only if the true prefix reaches
its 4,096-token minimum. If it does not, S is billed at $0.50/MTok, which adds about $0.06.

| calibration, 50 proofs | T = 500 | **T = 2,000** | T = 5,000 | worst case, first round |
|---|---|---|---|---|
| Opus 5.5 | $1.45 | **$2.42** | $4.37 | $9.64 |
| Sonnet 5.5 | $0.74 | **$1.22** | $2.20 | $4.82 |
| Haiku 4.5 | $0.37 | **$0.61** | $0.95 | $2.41 |
| **total** | $2.56 | **$4.25** | $7.52 | $16.87 |

How the worst case is built:
- 1.25 × the estimated input;
- the system prompt priced as a 1 h cache write on every request;
- all 16,000 `max_tokens` spent on every request.

Retry batches are separate submissions, each with its own dry run and `--max-usd`. The expected
column already includes them through r = 1.3. The proposal's earlier calibration figure, about $7
on the standard API, was 2× higher because it priced without the batch discount.

## The 2,000-proof step (after calibration picks the model)

```bash
RUN2=/scratch/cowifornia/nd-rl/think/gen_runs/k2_<date>
$G select --run-dir $RUN2 --n 2000 --seed 20261007 --exclude calibration=$RUN/selection.jsonl
$G distractors --run-dir $RUN2 --device cuda --k 16     # needs approval and a free GPU (nvidia-smi)
$G build --run-dir $RUN2 --models claude-opus-5-5 --effort low --max-tokens 16000 \
    --distractors $RUN2/distractors.jsonl
$G submit --run-dir $RUN2 --requests $RUN2/requests/opus55.r1.jsonl                 # dry run
$G submit --run-dir $RUN2 --requests $RUN2/requests/opus55.r1.jsonl --submit --max-usd 400
# then poll / collect / retry rounds as in steps 6-9
```

The dry run (seed 20261007) selects 2,000 proofs with 18,066 states, 8,803 of them decision
states. Calibration proofs are excluded. Per request: S 5,733, U 1,674, O 646 + T. The request file
is 36 MB, under the 256 MB limit.

| 2,000 proofs | T = 500 | **T = 2,000** | T = 5,000 | worst case, first round |
|---|---|---|---|---|
| Opus 5.5 | $57 | **$96** | $174 | $386 |
| Sonnet 5.5 | $29 | **$49** | $88 | $193 |

Opus at T = 2,000: 2,000 × 1.3 × $0.0371 = $96. That is above the proposal's $76, because the two
worked examples nearly double the cached prefix and the estimate assumes 3 distractors per decision
state. After calibration, replace these numbers with traces wanted × measured $ per accepted
trace. That figure already includes retries and rejected proofs.

## Open choices for Leon

1. Approve the CPU distractor run for the 50 proofs.
2. Approve the three first-round batches and up to two retry rounds. The suggested cap is $20
   total, against an expected $2.6–7.5.
3. Choose effort for calibration: `low` (the proposal) for all three models. Sonnet 5.5's effort
   levels are recalibrated against Sonnet 5's, so equal names do not mean equal thinking.
4. Deviations stay `allow` (PROPOSAL decision 2); `--deviations forbid` switches them off.

## Not yet known (each needs the paid calibration)

- Acceptance rates, thinking-token use, and whether Opus 5.5's `reasoning_extraction` classifier
  refuses this prompt. Refusals are recorded and not retried.
- Whether structured outputs together with Haiku 4.5's extended thinking are accepted. The
  structured-outputs page does not say.
- Distractor yield. In the 3 CPU test states, stage-1 never erred in 16 samples. On its own
  training distribution, many decision states may get none; `--k 64` or `--with-dead` would add
  more.
