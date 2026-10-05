---
written_on: 2026-10-05
written_by: agent:claude
---

# Thinking-trace pipeline: commands, configs and GPU estimates

Status: code and CPU tests only. **Nothing here has been trained or evaluated**; every GPU number
below is an estimate with its arithmetic. Branch `leon/thinking`, uncommitted.

The design is [PROPOSAL.md](../PROPOSAL.md) §1, §2, §4 and §6: a per-step Markov think block,
`state <think> … </think> action <eos>`, in tokenizer mode `lean_thinkn` (the 111 `lean_staten`
tokens plus `<think> </think> <no>`, inserted before `h1`, so ref0 = 50 and 114 tokens).

## Where the code is

`code/thinking/` (package; module docstrings hold the details):

| module | what |
| --- | --- |
| `vendor/danstate/` | Dan's fork at `4dcd0e4f`, 19 files, unmodified (`PROVENANCE.md`, `SHA256SUMS`); written by `vendor_danstate.py` |
| `dan.py` | loads the vendored modules under private names; `bare()` for calls with lazy imports; `trace_dsl()` |
| `tok.py` | `ThinkTokenizer` (`lean_thinkn`), id remap, name rebasing |
| `data.py` | the training-pair builder (CLI) |
| `ckpt.py` | checkpoint load/save (forced offline), run environment, cold-start vocabulary expansion |
| `train.py` | Stage-1 entry point (Dan's best recipe), cold start, think-token loss weight; EI fine-tune loop |
| `sample.py` | think-aware batched sampler: plain, structural and logical redraws |
| `lean.py` | Lean gate with a configurable binary |
| `eval.py` | pass@k over a budget grid, plain and guided, with token accounting (CLI) |
| `ei.py` | expert iteration that keeps the sampled think tokens (CLI) |

Tests: `code/tests/thinking/test_{vendor,tok,data,train,sample,ei,smoke}.py`. Configs here:
[`env.sh`](env.sh) (paths, settings) and [`arms.yaml`](arms.yaml) (the arms as flags).
[`trace_dsl.py`](../trace_dsl.py) now defaults to the vendored copy when `ND_TAKEHOME` is unset;
only its import block changed.

## Design choices

- **Vendoring.** Dan's files are byte-identical copies, loaded through `dan.py`. Changes are wrappers
  and a `LeanTokenizer` subclass. Four pieces are copied, each with an upstream `file:line@4dcd0e4f`
  comment:
  - `state_train_best.train`, as `train.train_best`, for the cold start and think weights. With
    neither, the entry point calls the fork's function. A test shows the copy matches it bit for bit.
  - `model.load_ckpt`, as `ckpt.load_ckpt`, so that `lean_thinkn` checkpoints load.
  - the plain AdamW loop of `state_train.main`, as `train.finetune` (the EI fine-tune).
  - the decode loops of `sample.generate_ids_fast` and `guided_sample.decode`, merged in
    `sample.decode_wave` with the think-block masks.
- **Data.** Each pair's target is the think block plus the action plus `<eos>`, with mask 1; the
  state has mask 0.
  - Think lines are joined by `<nl>`, with no trailing `<nl>`; this is trace_dsl's own tokenisation.
  - `--no-trace-mode none` writes the bare action, which is the T0 format; `empty` writes
    `<think> </think>`.
  - Eval exclusion matches the atom-renaming key (`canon_key`, as in the fork's `gen.py` and
    ndbench) or the premise-order-invariant `pkey` (ndbench's dedup rule).
  - Full K12 in `empty` mode, measured on CPU: 1,239,820 pairs, mean 98.9 tokens, max 394, none
    over 1,024. 23 records were dropped: 22 share a class with heldout, 1 with textbook_dev.
- **Loss weight.** `--think_weight w` weights every token of the think block, tags included, by w.
  The action and `<eos>` weigh 1. With w = 0 only the action is trained.
- **Cold start.** A `lean_staten` checkpoint has exactly two vocabulary-sized tensors, the embedding
  and the head (no tying, no head bias; the MTP head is not saved). Old rows move to their new ids.
  - New embedding rows are drawn from the per-dimension Gaussian fitted to the old rows.
  - New head rows are the mean of the old head rows (Hewitt 2021's mean initialisation).
  - Old-token logits are exactly unchanged; this is tested on the real Stage-1 checkpoint.
  - The MTP head starts fresh. `--lr_scale` scales both the AdamW and the Muon peak.
- **Sampler.** The state is prefilled with `<think>`. Inside the block, `<eos>`, `<think>` and
  `<pad>` are banned; `</think>` is banned before m tokens and forced after B. B = 0 prefills
  `<think> </think>`. Only the action reaches `Env.apply`.
  - Per-attempt budgets are allowed, and the worklist is filled in budget order so that a wave is
    not stretched by its longest-budget rows.
  - Inside actions nothing is masked by default; this matches the fork. `--mask_action_think` bans
    think tokens there.
- **Redraws.** The whole think + action is redrawn from the same state, without replacement over
  that whole decoded segment (`_swor_shift` on the masked, tempered log-probs; a forced token has
  log-prob 0).
  - An identical (think, action) is never redrawn. The same rejected action can come back after a
    different think; `stats['repeat_action_redraws']` counts this.
  - Every decoded token is counted, including those of rejected draws.
- **EI.** An accepted attempt stores, per step, its state, its sampled think content and the action
  the environment applied. Names are rebased from the attempt's name base to base 0.
  - The fine-tune pairs are these sequences.
  - `--validate_think drop|empty` removes, or empties, the steps whose think block fails strict
    `trace_dsl.check_step`.
  - Distinct means a distinct (think, action) sequence.
- **Safety.** Every entry point forces `ND_OFFLINE=1`; a test makes the upload functions raise and
  shows they are never called. Registry rows go to `<out>/registry` and the Lean log to
  `<out>/lean_gate.jsonl`.

## Commands

From the repository root, after `nvidia-smi` to pick a free GPU `G`. Everything writes under
`$RUNS` (default `/scratch/cowifornia/nd-rl/think/runs`).

```bash
source code/experiments/current/thinking_traces_20261005/pipeline/env.sh

# 1. Build data (CPU, about 1.5 min each for full K12; trace validation ~13 ms a trace with Lean)
$PY -m code.thinking.data --plain $K12 --no-trace-mode none  --extra-exclude $TB_TRAIN --out $RUNS/data/T0.pkl
$PY -m code.thinking.data --plain $K12 --no-trace-mode empty --extra-exclude $TB_TRAIN --out $RUNS/data/T0e.pkl
$PY -m code.thinking.data --traces $LLM_TRACES:4 $PROG_TRACES --plain $K12 --no-trace-mode empty \
    --extra-exclude $TB_TRAIN --out $RUNS/data/T2.pkl          # T1: --traces $PROG_TRACES only
#    each writes <out>.report.json: counts, invalid traces by reason, length distribution, pairs > max_len

# 2. Pretrain from scratch (one RTX 3090; seeds 0 1 2)
CUDA_VISIBLE_DEVICES=$G $PY -m code.thinking.train --data $RUNS/data/T2.pkl --heldout $HELDOUT \
    --out $RUNS/stage1/T2_s0 --seed 0 --best_steps 24000

# 3. Cold start from state_env_october's Stage 1 (lean_staten -> lean_thinkn)
CUDA_VISIBLE_DEVICES=$G $PY -m code.thinking.train --data $RUNS/data/T2.pkl --heldout $HELDOUT \
    --out $RUNS/cold/T2_s0 --init $STAGE1 --lr_scale 0.3 --best_steps 3000 --seed 0

# 4. Evaluate: smoke first (about a minute), then the first read on Tier A + textbook_dev58
CUDA_VISIBLE_DEVICES=$G $PY -m code.thinking.eval --ckpt $RUNS/stage1/T2_s0/model.pt --sets $SETS_FIRST \
    --out $RUNS/eval/smoke --k 4 --limit 8 --budgets 0,64 --arms plain --batch 512
CUDA_VISIBLE_DEVICES=$G $PY -m code.thinking.eval --ckpt $RUNS/stage1/T2_s0/model.pt --sets $SETS_FIRST \
    --out $RUNS/eval/T2_s0 --k 16 --budgets 0,64,256 --arms plain,logical --batch 1024 --seed 1
#    T0 checkpoints are read once (B = 0, no think block): the format is detected from the checkpoint

# 5. Expert iteration (per-attempt budgets; keep only steps whose think validates)
CUDA_VISIBLE_DEVICES=$G $PY -m code.thinking.ei --init $RUNS/stage1/T2_s0/model.pt --name T2_s0 \
    --out $RUNS/ei/T2_s0 --targets $TARGETS --train $RUNS/data/T2.pkl --budgets 16,64,256 \
    --validate_think drop --batch 1024 --seed 0 --rounds 8 --k 32
#    frozen control: add --no_train; transfer reads: --transfer $TRANSFER; held-out greedy: --heldout $HELDOUT
```

Outputs:

- train: `model.pt` (the fork's checkpoint format; `extra` holds the curve, `think_format`, the data
  path and, for a cold start, `init`), `train.json`, `args.json`, `registry/`.
- eval: `rows_<arm>_B<B>.jsonl` (per attempt: status, accepted steps, `fail_step`, token counters)
  and `summary.json` (pass@k in % of each set; tokens per theorem and in total).
- EI: `found_<r>.jsonl` (the token strings of the accepted sequences), `round_<r>.json`, `ckpts/`.

Batch size on a 24 GB card: the KV cache is batch × 6 layers × 384 × (state + B + 1 + max_action) ×
4 bytes. At batch 1,024, a 400-token state, B = 256 and max_action 512, that is 1,024 × 6 × 384 ×
1,169 × 4 = 11.0 GB. Batch 2,048 fits only with max_action 256 and B ≤ 64 (13.6 GB).

## GPU estimates (one RTX 3090; not measured)

**Measured on Dan's run (A40, `stage1_best12_s0_b1200.pt` extra).** 23,951 steps in 1,200.7 s,
which is 50.1 ms a step. It saw 4,546,684 pairs, 3.67 epochs of 1,239,995 pairs, at 96.9 tokens a
pair: 440 M tokens, or 18,392 real tokens a step against a padded budget of 18,432. The
CONTEXT_SURVEY's "about 21.5 A40-min" includes setup.

**Per-step cost is flat in sequence length.** Batches have a fixed token budget, so a step always
processes about 18.4 k tokens. Attention is about 3 % of the FLOPs at T ≈ 100:

- dense: 6 × 11.4 M parameters (trunk plus MTP) ≈ 68 MFLOP a token;
- attention: QKᵀ and AV, forward and backward, 7 blocks (6 plus MTP): 84 · d · T̄_ctx ≈
  84 × 384 × 60 ≈ 1.9 MFLOP a token at a mean causal context of 60.

So longer pairs cost more steps, not slower steps.

**3090 against A40.** No measured number exists for this recipe. Both cards are GA102. The A40 runs
at about 25 TFLOP/s (1.26 TFLOP a step / 50 ms), below the 3090's 71 TFLOP/s bf16 peak. I assume
the 3090 is 1.0–1.5 × the A40's step time, i.e. 50–75 ms a step. The proposal's prior of 30–40 min
would mean 75–100 ms. The first run's `train.json` curve (secs, step) settles it.

| job | steps | arithmetic | 3090 estimate |
| --- | --- | --- | --- |
| T0 / T0e Stage 1 | 24,000 | 24,000 × 50–75 ms; T0e has +2 tokens a pair (98.9 / 96.9 = 1.02), so 3.59 epochs | 20–30 min + ~2 min setup |
| T1 / T2 Stage 1, equal tokens | 24,000 | as above, about 2.8–3.0 epochs | 20–30 min |
| T1 / T2 Stage 1, equal epochs (3.67) | 29,000–31,700 | tokens a proof (8.0 pairs): 775 (T0) → 791 + 150–230 think = 941–1,021, ×1.21–1.32 | 24–40 min |
| cold start | 3,000 | 3,000 × 50–75 ms | 3–4 min + setup |
| 4 arms × 3 seeds, equal tokens | 12 × 24,000 | 12 × 22–32 min | 4.4–6.4 GPU-h |

The 150–230 think tokens a proof is the proposal's assumption for K12 (the pilot has 232, but it
over-represents decision-rich proofs). The real number comes from the report of the trace build.

**Evaluation.** A wave decodes until its longest row ends, and at this model size a decode step
costs about the same whatever fraction of rows is live. So a wave's wall time grows with
B + 1 + (longest action).

- Taking the longest action in a wave as about 80 tokens, a wave costs about 1.2× the no-think wave
  at B = 16, 1.8× at B = 64 and 4.2× at B = 256 (example: (256 + 1 + 80) / 80 ≈ 4.2).
- The grid {0, 64, 256} therefore costs about 7× one no-think read.
- `ood_check/EVAL_SPEC.md` estimates (also unmeasured) 27 min for a plain read and 58 min for a
  logical read of 840 theorems × k 64 on a 3090.

| read | workload | estimate |
| --- | --- | --- |
| first read: SETS_FIRST (138 theorems), k 16, B ∈ {0, 64, 256}, plain + logical | 2,208 attempts a cell = 4.1 % of 53,760 | plain 1.1 min × 7 + logical 2.4 min × 7 ≈ 25 min a checkpoint |
| full SETS (898 theorems), k 64, same grid | 57,472 attempts a cell = 1.07× | (29 + 62) min × 7 ≈ 10.6 h a checkpoint: do not run first |

**Expert iteration.** Dan's T1 ladder costs 7.2–8.5 A40-h a seed (CONTEXT_SURVEY: 8 rounds, k 32,
4,495 targets, transfer and held-out reads, 600-step fine-tunes).

- Sampling dominates. With budgets {16, 64, 256} drawn per attempt and budget-ordered waves, the
  mean wave multiplier is about (1.2 + 1.8 + 4.2) / 3 ≈ 2.4.
- That gives about 17–20 A40-h, or 17–31 GPU-h on a 3090, per seed.
- A first EI pass with 1,000 targets and 4 rounds, without transfer, is about
  (1,000 / 4,495) × (4 / 8) ≈ 0.11 of that: **2–3.5 GPU-h**.

## Decisions and known gaps

- **textbook train14.** The textbook72 manifest says "never train or tune on these problems or on a
  renaming of them". The default exclusion follows the brief (OOD tiers, dev58, heldout). The
  commands add `--extra-exclude $TB_TRAIN`, which drops the pilot's 2 textbook traces. Leon to
  confirm.
- **Not built here:** the trace sources (LLM traces come from `code/thinking/gen`, by another agent;
  the programmatic traces of PROPOSAL §3.1 (d) do not exist yet). Also not built: GRPO and the
  causal-corruption read-outs of §4.
- **EI:**
  - Allocation is uniform only (no T2–T6 rungs); relabelling and injection are not ported.
  - The fine-tune runs in-process, not as a `state_train.py` child.
  - Steps whose `</think>` was forced at the budget are stored, flagged `forced` in the sampler
    trace, and trained like the others.
- **Redraw scope** is the whole segment (above). If `repeat_action_redraws` turns out to be high,
  forbidding the rejected action across thinks is a possible variant; it is not implemented.
- **Not measured:** step time on a 3090, think lengths of a trained model, and eval and EI wall
  times. Every number in the estimate tables is arithmetic on the numbers cited.
- Lean here is 4.34.0; pods run 4.34.1.

## Checks that ran

All on CPU, 2026-10-05:

- `code/tests/thinking/test_{vendor,tok,data,train,sample,ei,smoke}.py`: 50 passed with `--runslow`
  (the smoke test is the one slow test).
- `make -C code test` (whole repository, fast tests): 1,192 passed, 15 skipped.
- `test_trace_dsl.py`: 26 passed both with `ND_TAKEHOME` unset (vendored copy) and pointing at the
  clone; the clone's `git status --ignored` stays empty.
- `python -m code.thinking.vendor_danstate --check`: identical to the clone at 4dcd0e4f.
- `make -C code check`: pyright 0 errors, ruff format clean; ruff check fails only on the sibling
  notes' `ood_check/*.py` scripts (222 findings, pre-existing, not touched here).
