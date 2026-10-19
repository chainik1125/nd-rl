---
written_on: 2026-10-07
written_by: agent:claude
---

# Graded-length programmatic traces with effort tokens (T1v1g4)

**Status (2026-10-07):** traces and training pickle built, CPU smoke test passed, GPU run not launched.
Leon launches it with the command [below](#stage-1-training-command). Branch `leon/thinking`,
worktree `/scratch/cowifornia/nd-rl/think/nd-rl`, base commit `38b7c4b`, changes uncommitted.

## Why

Today's Stage-1 thinking model T1v1 is pretrained on the v1 programmatic traces at their natural length,
plus 20% empty blocks. It never sees the effort tokens `<e0> <e16> <e64> <e256>` during pretraining:
`effort.expand_to_effort` adds them afterwards. At sampling time the budget caps each think block
(one block per proof step) by force-closing it at a line boundary. Under expert iteration, a 64-token
budget did worse than no thinking: the model starts a long plan and gets cut off.

T1v1g4 pretrains on the same proof steps with each think block written at one of four lengths, tagged with
its effort token. Each shorter block is a compression of the full block that fits its budget, not a
truncation of it.

## Hard rule

Every think line is written by code, by `code/thinking/progtrace.py`. No language model wrote, paraphrased,
compressed or rewrote any trace content. The compressor only selects, drops and re-marks lines that the
generator itself wrote. The worker agents wrote generator, data and training code and tests only.

## Measured before choosing the rules

These are think-body tokens: the tokens between `<think>` and `</think>`, `<nl>` included. That is what
the sampler's budget counts. The figures come from v1 (`prog_traces.jsonl`), measured on the first 20,000
records unless stated otherwise.

| | value |
|---|---|
| Non-empty blocks, body tokens p10 / p25 / p50 / p75 / p90 / p99 | 20 / 40 / 78 / 126 / 180 / 234 |
| Non-empty blocks that already fit 16 / 64 / 128 tokens | 7.7% / 41.5% / 75.9% |
| Median line: goal line / candidate / look | 22 / 16 / 10 tokens |
| Decision steps whose whole plan fits 16 / 64 tokens (5,000 proofs) | 27% / 77% |
| Decision steps whose first plan line alone is over 16 / over 64 tokens | 69% / 10% |

So 64 tokens holds the plan for most steps, but rarely the dead ends and candidates too. 16 tokens holds one
short claim. Formulas are long in this language (a goal line has a median of 22 tokens), and the validator
needs every goal line to refine a goal already shown, starting from the frame goal. So at 16 tokens many
decisions cannot be stated at all.

## Design

The generator gains `Cfg.level` and `--level {0,16,64,256}`. The profile stays v1.

- **e256 is today's trace.** The v1 generator is unchanged, with a 240-token block cap and the same seed and
  rng stream. The batch output is byte-identical to `prog_traces.jsonl`.
- **e16 and e64:** for each step the generator first writes the e256 block exactly as today, consuming the
  same random numbers. While writing, it records which lines form which segment:
  - a plan line, at a given depth;
  - a dead end, before or after the reference reduction, with its form and alternative kind;
  - a viable alternative;
  - a refuted candidate, marked if it is a base-model distractor;
  - a dead box candidate;
  - the reference candidate;
  - a check.

  A deterministic compressor, which draws no random numbers, then takes the first variant on a ladder that
  passes both tests. The variant must fit the level's budget (body tokens ≤ level), and `trace_dsl.check_step`
  must accept it with no soft findings, which is today's strict standard.
- **The final rung is the empty block.** It always passes, so no trace is ever dropped.
- **e0** is the empty block `<think> </think>`.

Every line of a compressed block is one of five things:
- a line of the e256 block for that step;
- such a line with `<no>` appended (a dead end shortened to its first reduction, now flagged);
- a bare `<no>` verdict;
- a look `n : A` at a name the action cites;
- the action itself as a candidate.

Lines keep their original order.

**e64 ladder:** the main candidates and the choice, with dead ends shortened or pruned.

1. `full`: the whole e256 block, if it fits. A level is a cap, so a short search is kept whole.
2. `nocheck`: drop the checks after the passing candidate, and the viable alternatives that are dropped
   without a stated reason.
3. `short`: shorten every multi-line dead end (one- or two-level backtracking, box lookahead) to its first
   reduction plus `<no>`. Strip the looks from refuted candidates (candidate, `<no>`).
4. `prune`: drop segments one at a time, in this order:
   - ex-falso dead ends (TRACE_AUDIT: 72% of v1 dead ends are this ritual move);
   - the reference candidate, when the plan already names the action. The line only repeats the action.
     On 1,500 proofs, dropping it this early keeps 58% more refuted candidates at e64 for the same tokens;
   - deeper dead ends, deepest first;
   - top-level dead ends;
   - refuted candidates, non-distractors first and the base model's own distractors last.
5. `plan_cut`: a prefix of the plan, top levels first, then the action as a candidate. v1 already does
   this when a plan exceeds its budget. If the action does not fit, the prefix alone.
6. Fallbacks:
   - `looks`: greedy looks at the names the action cites;
   - `ref`: the action as a candidate;
   - `empty`.

**e16 ladder:** the essential decision only.

1. `full`, if the whole block fits.
2. `plan`: the plan lines alone, from the goal down to the action's reduction.
3. `plan_cut`: a plan prefix.
4. The same fallbacks: looks at the cited facts, the action, empty.

## Traces

All K12 proofs at the three non-empty levels: `--profile v1 --level L --workers 32`, CPU only, 80–95 s
each. Output is in `/scratch/cowifornia/nd-rl/think/traces/graded/e{256,64,16}.jsonl`, with `.stats.json`
beside each.

Every level has 154,977 traces with 0 dropped, and all three files share the same ids, order and actions.
The CLI validated every trace strictly (`trace_dsl.validate(strict=True)`). Re-validation of random samples
also passed: 2,000 per level (worker), plus 5,000 per level at e16 and e64 with another seed (auditor; its e64
file was the one from before the final prune-order change). Every sample passed 100%.

Over all 546,592 decision steps (10.1% are already empty at e256):

| | e256 | e64 | e16 |
|---|---|---|---|
| Body tokens of non-empty decision blocks: mean / p50 / p90 / max | 94.0 / 83 / 185 / 239 | 43.4 / 48 / 62 / 64 | 10.0 / 11 / 15 / 16 |
| Empty decision blocks | 10.1% | 10.8% | 31.1% |
| Decision blocks with 1–16 / 17–64 / 65–256 tokens | 5.5% / 28.2% / 56.3% | 9.8% / 79.4% / 0 | 68.9% / 0 / 0 |
| Max body tokens, all steps | 239 | 64 | 16 |

Which rung produced the block, as a share of the 491,506 decision steps with a non-empty e256 block:

| rung | e64 | e16 |
|---|---|---|
| `full` (the whole block fits) | 37.4% | 6.1% |
| `nocheck` / `short` / `prune` | 4.5% / 4.8% / 30.5% | – |
| `plan` | – | 21.1% |
| `plan_cut` | 12.4% | 4.0% |
| `looks` | 7.2% | 44.4% |
| `ref` | 2.4% | 0.9% |
| `empty` | 0.8% | 23.4% |

Checks over the whole files:
- Every compressed line is a line of that step's e256 block, such a line with `<no>` appended, a look at a
  name the action cites, or the action. That is 0 lines of any other kind.
- e64: 1,386,756 e256 lines, 14,444 shortened dead ends, 33,600 looks and 12,926 action lines.
- e16: 301,404 e256 lines, 185,950 looks and 1,632 action lines.
- No look cites a name outside the action.
- No step's e16 block is longer than its e64 block, and no e64 block is longer than its e256 block.
- The e256 run reproduces `prog_traces.jsonl` byte for byte (`cmp`).

**Sample by eye.** The auditor looked at 16 random steps, on the e64 file from before the final prune-order
change, and I looked at 6 more on the final files. e64 reads as a shorter version of the same search in
every case:
- the plan is intact, or cut to a clean prefix;
- dead ends and refuted candidates keep their `<no>`;
- nothing refers to a dropped line.

e16 is the plan, a plan prefix, or the facts the action cites. When it is a look, it names the action's
premises before the action. That is a decision preview, not a search, just as the v1 plan lines already
preview the action.

## Three example steps

These are generator output, quoted verbatim in the ASCII surface (`|-` ⊢, `~` ¬, `&` ∧, `v` ∨, `>` →,
`F` False, `<| |>` ⟨ ⟩). e0 is `<think> </think>` for every step.

1. Plan intact at all levels. e64 drops a viable alternative that was given up without a stated reason, and the looks after each refuted candidate.
2. A pruned search. e64 shortens the two-level dead end to one flagged line. It drops the ex-falso ritual and the reference candidate, which only repeats the plan's last line. e16 keeps the one fact the action cites.
3. A long box opener. e64 keeps the first plan line, which is the box's own reduction, and e16 is empty because nothing that states this decision fits in 16 tokens.

**k12_17080, step 6**

```
STATE
<st> h1 : ( ( R v ( S v S ) ) > ( ( ~ Q ) & R ) )
h2 : ( ~ ( ( ( ~ Q ) & R ) & ( R v ( S v S ) ) ) )
h3 : ( F > R )
n1 : ( ( R v ( S v S ) ) > ( ( ~ Q ) & R ) )
n2 : ( ~ ( ( ( ~ Q ) & R ) & ( R v ( S v S ) ) ) )
n3 : ( F > R )
n5 : ( R v ( S v S ) )
n6 : ( ( ~ Q ) & R )
n7 : ( ( ( ~ Q ) & R ) & ( R v ( S v S ) ) )
|- R <act>
ACTION have n8 : F := n2 n7 ;
--- e256 (93 tokens)
|- R := Classical.byContradiction |- ( ~ ( ~ R ) )
|- R := n3 |- F
|- F := n2 n7
have n8 : F := n7 n2 ;
n7 : ( ( ( ~ Q ) & R ) & ( R v ( S v S ) ) )
<no>
have n8 : F := n1 n7 ;
n7 : ( ( ( ~ Q ) & R ) & ( R v ( S v S ) ) )
<no>
--- e64 (34 tokens)
|- R := n3 |- F
|- F := n2 n7
have n8 : F := n7 n2 ;
<no>
have n8 : F := n1 n7 ;
<no>
--- e16 (12 tokens)
|- R := n3 |- F
|- F := n2 n7
--- e0 (0 tokens)
(empty)
```

**k12_104571, step 5**

```
STATE
<st> h1 : ( ( ( Q > Q ) v ( S v S ) ) v ( ( Q > S ) & R ) )
n1 : ( ( ( Q > Q ) v ( S v S ) ) v ( ( Q > S ) & R ) )
n3 : ( ( Q > S ) & R )
n4 : ( Q > Q )
|- ( ( Q > Q ) v ( S v S ) ) <act>
ACTION have n5 : ( ( Q > Q ) v ( S v S ) ) := Or.inl n4 ;
--- e256 (127 tokens)
|- ( ( Q > Q ) v ( S v S ) ) := Or.inr |- ( S v S )
|- ( S v S ) := Or.inl |- S
|- S <no>
|- ( S v S ) <no>
|- ( ( Q > Q ) v ( S v S ) ) := Or.inl n4
|- ( ( Q > Q ) v ( S v S ) ) := |- F .elim <no>
have n5 : ( ( Q > Q ) v ( S v S ) ) := Or.inr n4 ;
<no>
have n5 : ( ( Q > Q ) v ( S v S ) ) := Or.inl n4 ;
--- e64 (64 tokens)
|- ( ( Q > Q ) v ( S v S ) ) := Or.inr |- ( S v S ) <no>
|- ( ( Q > Q ) v ( S v S ) ) := Or.inl n4
have n5 : ( ( Q > Q ) v ( S v S ) ) := Or.inr n4 ;
<no>
--- e16 (7 tokens)
n4 : ( Q > Q )
--- e0 (0 tokens)
(empty)
```

**k12_114511, step 2**

```
STATE
<st> h1 : ( ( S v P ) > R )
h2 : ( S v P )
n1 : ( ( S v P ) > R )
n2 : ( S v P )
|- ( ( ~ ( S & S ) ) > ( P v ( ( S v P ) > R ) ) ) <act>
ACTION have n3 : ( ( ~ ( S & S ) ) > ( P v ( ( S v P ) > R ) ) ) := ( fun ( n4 : ( ~ ( S & S ) ) ) => by
--- e256 (162 tokens)
|- ( ( ~ ( S & S ) ) > ( P v ( ( S v P ) > R ) ) ) := |- F .elim
|- F <no>
|- ( ( ~ ( S & S ) ) > ( P v ( ( S v P ) > R ) ) ) := fun ( n4 : ( ~ ( S & S ) ) ) => |- ( P v ( ( S v P ) > R ) )
n4 : ( ~ ( S & S ) ) |- ( P v ( ( S v P ) > R ) ) := Or.elim n2 ( fun ( n5 : S ) => |- ( P v ( ( S v P ) > R ) ) ) ( fun ( n6 : P ) => |- ( P v ( ( S v P ) > R ) ) )
--- e64 (54 tokens)
|- ( ( ~ ( S & S ) ) > ( P v ( ( S v P ) > R ) ) ) := fun ( n4 : ( ~ ( S & S ) ) ) => |- ( P v ( ( S v P ) > R ) )
--- e16 (0 tokens)
(empty)
--- e0 (0 tokens)
(empty)
```

## Training pickle

`/scratch/cowifornia/nd-rl/think/runs/data/T1v1g4.pkl`, mode `lean_thinke`, built 23:47 UTC from the final
level files. Its report is in `T1v1g4.pkl.report.json`.

```
cd /scratch/cowifornia/nd-rl/think/nd-rl
for L in 256 64 16; do ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m code.thinking.progtrace \
  --profile v1 --level $L --workers 32 --out /scratch/cowifornia/nd-rl/think/traces/graded/e$L.jsonl; done
G=/scratch/cowifornia/nd-rl/think/traces/graded
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m code.thinking.data \
  --graded 0:0.2 16:$G/e16.jsonl:0.2 64:$G/e64.jsonl:0.3 256:$G/e256.jsonl:0.3 \
  --extra-exclude /scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72/textbook_train.jsonl \
  --no-lean --out /scratch/cowifornia/nd-rl/think/runs/data/T1v1g4.pkl
```

- **Levels.** Each proof draws one level from `unit_hash(id/effort)`, so the draw is deterministic. Every
  pair of that proof uses that level's blocks, with `<eB>` ending its state:
  `state <eB> | <think> ... </think> action <eos>`. This is the layout of `effort.relabel_records`.
- **Same proofs.** The pickle has the same 154,977 proofs as T1v1, with identical ids in the same order, and
  1,239,820 pairs.
- **Validity.** Exclusions, eval-class hits, missing records and invalid traces are all 0.
- **Spot check.** I checked 3,000 random records (24,062 pairs). Each pair's think block equals its drawn
  level's file block, and each state ends in the matching effort token.

| | T1v1g4 | T1v1 |
|---|---|---|
| Records at e0 / e16 / e64 / e256 | 31,026 / 31,068 / 46,422 / 46,461 (20.0 / 20.0 / 30.0 / 30.0%) | 30,925 emptied (20.0%), 124,052 full |
| Non-empty think body: e16 / e64 / e256, mean (max) | 9.8 (16) / 42.0 (64) / 89.0 (239) | 88.8 (239) |
| Non-empty think blocks | 411,256 | 437,114 |
| Pair tokens: mean / max (pairs over 1,024) | 117.8 / 595 (0) | 130.2 / 594 (0) |
| Tokens total / loss tokens total | 146.0M / 44.1M | 161.4M / 60.7M |

The pickle holds 10% fewer tokens and 27% fewer loss tokens than T1v1. The run matches T1v1's compute by
optimizer steps (24,000 steps at the same 18,432-token batch budget), so it sees about 3.0 epochs where
T1v1 saw 2.7.

**Smoke tests (CPU, no GPU).**
- `train.py` on the final pickle, tiny dims (`--best_dims 2,64,2,128 --best_steps 5 --device cpu
  --no_compile`): `mode lean_thinke, 154977 records, 1239820 pairs (longest 595 tokens, max_len 1024),
  format think`, 5 steps, final val 4.75. The checkpoint reloads with `ckpt.load_ckpt` as `lean_thinke`
  (EffortTokenizer, vocab 118).
  - Output: `runs/smoke_T1v1g4/tiny_final`.
- The real recipe dims (6 layers, d 384) for 2 steps on the previous build of the pickle (only e64's prune
  order differed): exit 0, final val 4.78.
  - Output: `runs/smoke_T1v1g4/full`.
- With default flags a `lean_thinke` pickle still goes through the fork's `state_train_best.train`
  unchanged (max_len 1024). That function reads `tok.vocab_size` (118) and `tok.ref0` (54). The effort ids
  (42–45) sit below `ref0`, so the name shift leaves them alone.

## Stage-1 training command

T1v1's recipe: the `queue.sh` / `launch_next.sh` form and T1v1_s0's `args.json`. That is 24,000 steps,
seed 0, and the defaults lr 1e-3, wd 0.1, tok_budget 18,432 and mtp 0.3. It runs on GPU 2, and the first
line waits until the GPU is free (< 500 MiB). T1v1_s0 took 52 min.

```
cd /scratch/cowifornia/nd-rl/think/nd-rl
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 500 ]; do sleep 30; done
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2 nohup timeout 7200 code/.venv/bin/python -m code.thinking.train \
  --data /scratch/cowifornia/nd-rl/think/runs/data/T1v1g4.pkl \
  --heldout /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl \
  --out /scratch/cowifornia/nd-rl/think/runs/stage1/T1v1g4_s0 \
  --seed 0 --best_steps 24000 > /scratch/cowifornia/nd-rl/think/runs/logs/train_T1v1g4_s0.log 2>&1 &
```

The held-out loss is not quite T1v1's: held-out states now end in `<e0>`. At sampling time
`sample.step_prompt` already appends `<eB>` for a `lean_thinke` checkpoint, so evals at budget B condition
on B with no further change.

## Code and tests

- `code/thinking/progtrace.py`: `Cfg.level`, `--level`, `Seg`, `Block.mark`, `tile`, `short_form`,
  `prune_order`, `decision_fallbacks`, `compress`, and stats (`rungs`, `max_think_tokens`,
  `think_tokens_e256_per_step`, `other_segments`).
- `code/thinking/data.py`: `draw_level`, `build_graded`, and `--graded LEVEL[:PATH]:PROB ...`, which writes
  a `lean_thinke` pickle.
- `code/thinking/train.py`: picks the tokenizer from the pickle's mode. For `lean_thinke`, held-out states
  get `<e0>` appended (the held-out layout is the empty block). `--init` and the root/memo formats exit for
  `lean_thinke`.
- Tests: `code/tests/thinking/test_progtrace_graded.py`, `test_data_graded.py` and `test_train_effort.py`.

Checks run on the final code:
- pytest on `test_progtrace.py`, `test_progtrace_graded.py`, `test_data_graded.py`,
  `test_train_effort.py`, `test_data.py`, `test_effort.py` and `test_train.py`: **56 passed** (CPU, 34 s).
- ruff check, ruff format --check and pyright report 0 errors on the six changed or new files. Every line
  is at most 100 characters.

The graded generator tests check four things:
- each level respects its budget;
- each level's traces pass strict validation with unchanged actions;
- e256 equals `Cfg.v1()` output, and the stored `prog_traces.jsonl` for the first 100 records;
- e0 is empty, and compressed lines come only from the allowed kinds.

## Open questions

- At e16, 23% of non-empty decision blocks become empty, because no claim that carries the decision fits
  in 16 tokens. Most of these are box openers with long formulas. The model therefore sees `<e16>` steps
  that do not think. Whether e16 helps at all is an empirical question for the eval.
- e16 and e64 are always compressions of the v1 block. They inherit its weaknesses (TRACE_AUDIT: plan
  copying, predictable candidate verdicts). The e64 pruning removes the ex-falso ritual first.
- Mix: e0 20%, e16 20%, e64 30%, e256 30% per proof. T1v1 had 80% full traces, so T1v1g4 has fewer think
  tokens. Total compute is matched by steps (24k at the same token budget), not by think tokens.
