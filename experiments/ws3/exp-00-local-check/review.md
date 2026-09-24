# exp-00 local check: review

Reviewer session, separate from the executor (plan section 6). Phase 1 (this section) was written
from the raw artefacts under `run/`, `proposal.md`, `config.yaml`, `status.md` and the code in
`code/ws3/` only; `numbers.md`, `results.md`, `run/report/report.md` and `make_numbers.py` were
not opened before the Recount was committed. All numbers below come from my own scripts in
`review/` (`recount.py`, `leak_detail.py`, `weights_check.py`, `nta.py`; outputs `recount.json`,
`leak_detail.json`, `weights_check.json`, `nta.json`, `samples.txt`), run as
`uv run python ../experiments/ws3/exp-00-local-check/review/<script>.py` from `code/`. The only
executor-adjacent code imported is `ndpath` (sys.path only), `strict_verify.safe_verify`
(checker of record) and the vendored `nd_data.canonical_theorem` (second-opinion key). Nothing
from `code/ws3/eval/` or `code/ws3/data.py` was used.

## Recount

### 0. Gates A and B, fixed components

| check | result |
| --- | --- |
| Gate A: `proposal.md` commit `a0b2ec6` at 2026-09-23T14:58:38Z (committer and author date) | precedes first `utc` in `run/baseline/environment.json` (14:59:37Z) and first `utc` in `metrics.jsonl` (15:03:15Z). **Pass.** (The status note that an earlier launch was killed and its partial `run/` deleted before the proposal commit cannot be verified from the artefacts; the surviving artefacts are all after the commit.) |
| `status.md` last line | `DONE 2026-09-23T15:16:22Z`. Pass. |
| `bash code/ws3/fork_pin/fetch.sh --verify` | 13/13 hashes match (`model.py`, `tokenizer.py`, `nd_verify/*`, 8 data files, `pretrain.pt`). Pass. |
| `git diff charles-pre-ws3 -- code/experiments/current/r1_cot/vendor/` | empty; `strict_verify.py` unchanged. Pass. |
| `git diff a0b2ec6 -- code/ws3/fork_pin/` (proposal commit to HEAD) | empty. Pass. |
| `run/hf_base/model.safetensors` vs `fork_pin/ckpt/pretrain.pt` (`weights_check.py`) | 52/52 tensors, max abs diff 0.0: the control model is bit-identical to the pinned checkpoint. `run/final` differs from it in all 52 tensors, max abs diff 3.8e-4, relative Frobenius 0.0022. |
| `run_manifest.json` / `environment.json` | `git_commit` a0b2ec6, checkpoint sha256 `f4409e4f…` (matches MANIFEST), torch 2.11.0, transformers 5.17.0, trl 1.13.0, CPU. |
| Post-run code edit `7fd042a` (`run.py`) | 2 lines in `plot_pass_by_length` only (`max(0.0, …)` clamp on error bars). Plotting only; no training or evaluation code touched. `stages.jsonl` shows the report stage rerun at 15:14:04Z. |
| Textbook72 MANIFEST | `textbook_dev.jsonl` sha256 `1b04192d…` (58 records) and `textbook_train.jsonl` `e6a219e0…` (14 records) both match. 43 dev problems have `reference_lines >= 7`. |
| Eval-file grep in `run.py`, `data.py`, `metrics_callback.py`, `stdout.log` | `heldout`: `run.py` lines 178, 242, 310 read `heldout.jsonl` for the greedy eval, the T1 next-token gate, and the periodic-eval callback (evaluation only; the callback never feeds the trainer's dataset); `data.py` lines 21 and 39 name it as an `EVAL_ONLY` pool. `transfer`, `validation_36`: `data.py` line 21 only, as banned keys. `textbook_dev`/`textbook_train`: `data.py` line 83, loaded for the textbook eval and the ban list. `test_short`/`test_long`: no hits (not in `EVAL_ONLY`; see section 3). `stdout.log`: two `greedy heldout by length` result lines only. **No training code reads an evaluation file.** |
| Metric coverage (plan section 4) | `metrics.jsonl` has 57 lines: 50 rollout lines (steps 1 to 50), 6 periodic-eval lines (steps 0, 10, 20, 30, 40, 50), 1 TRL end-of-training summary. Every one of the 57 lines has `utc` and `step`. Keys on all 50 rollout lines: mean reward `reward`; within-group std `reward_std` (plus `frac_reward_zero_std`); KL `kl`; entropy `entropy`; unparseable `reward/unparseable_frac`; invalid `reward/invalid_frac`; completion length `completions/mean_length`, `completions/max_length`; cap fraction `completions/clipped_ratio`; grad norm `grad_norm`; learning rate `learning_rate`; throughput `rollout/completions_per_sec`. Pass rate by length: `eval/pass_rate_len<L>`, `eval/solved_len<L>`, `eval/n_len<L>` on the 6 eval lines. **All present.** |

### 1. Re-verification of every dumped proof

Every row of every dump was re-run through `strict_verify.safe_verify(prompt, completion)` and
compared to the row's `ok` (dumps) or `reward >= 0.5` (`completions.jsonl.gz`), and the
accepted rows' line count to `n_lines_written` / `n_lines`.

| dump | rows | accepted (mine) | accepted (executor field) | verdict disagreements | line-count disagreements |
| --- | --- | --- | --- | --- | --- |
| baseline/greedy_heldout | 1,000 | 907 | 907 | 0 | 0 |
| baseline/textbook72 | 4,608 | 240 | 240 | 0 | 0 |
| baseline/long_pool | 6,400 | 149 | 149 | 0 | 0 |
| evaluate/greedy_heldout | 1,000 | 911 | 911 | 0 | 0 |
| evaluate/textbook72 | 4,608 | 264 | 264 | 0 | 0 |
| evaluate/long_pool | 6,400 | 112 | 112 | 0 | 0 |
| eval_periodic | 420 | 268 | 268 | 0 | 0 |
| completions (training rollouts) | 12,800 | 11,308 | 11,308 | 0 | 0 |

36,836 rows, zero disagreements. Rewards in `completions.jsonl.gz` take only the values 0.0
and 1.0 (binary reward confirmed).

### 2. Pass rate by length (own pass@k and Wilson 95% intervals)

A record is solved if any of its k samples verifies. Both dumps have exactly the same prompt
sets (`same_prompt_set_baseline_vs_evaluate`: greedy True, long_pool True; textbook 72/72 names
present in both, no extras, dump `prompt`, `slice` and `length` fields match the textbook files
and `rl_targets.jsonl` `L_true` with 0 mismatches). Samples per record: greedy 1 (T=0.0),
textbook 64 (T=1.0), long pool 32 (T=1.0), in both dumps. `config.yaml` seed 0 for both
stages (`run.seed`, passed to both `common_evals` calls); the TRL config also carries `seed: 0`.
The equal-attempts control condition holds.

Greedy, heldout stride, k=1, T=0 (control = frozen checkpoint, RL = `run/final`):

| length | control | RL |
| --- | --- | --- |
| 2 | 200/200 = 1.000 [0.981, 1.000] | 200/200 = 1.000 [0.981, 1.000] |
| 3 | 198/200 = 0.990 [0.964, 0.997] | 199/200 = 0.995 [0.972, 0.999] |
| 4 | 190/200 = 0.950 [0.910, 0.973] | 187/200 = 0.935 [0.892, 0.962] |
| 5 | 174/200 = 0.870 [0.816, 0.910] | 179/200 = 0.895 [0.845, 0.930] |
| 6 | 145/200 = 0.725 [0.659, 0.782] | 146/200 = 0.730 [0.665, 0.787] |
| all | 907/1000 = 0.907 [0.887, 0.923] | 911/1000 = 0.911 [0.892, 0.927] |

Long pool (`rl_targets` stride of 200, by `L_true`), pass@32, T=1.0:

| L_true | control | RL |
| --- | --- | --- |
| 7 | 14/82 = 0.171 [0.105, 0.266] | 11/82 = 0.134 [0.077, 0.224] |
| 8 | 0/44 [0, 0.080] | 0/44 [0, 0.080] |
| 9 | 0/40 [0, 0.088] | 0/40 [0, 0.088] |
| 10 | 0/20 [0, 0.161] | 0/20 [0, 0.161] |
| 11 | 0/5 | 0/5 |
| 12 | 0/8 | 0/8 |
| 14 | 0/1 | 0/1 |
| all | 14/200 = 0.070 [0.042, 0.114] | 11/200 = 0.055 [0.031, 0.096] |

Max length solved: 6 greedy, 7 in the long pool, for both models. The 11 RL-solved length-7
records are a strict subset of the 14 control-solved ones (control-only: `la_rl_targets_990`,
`_2464`, `_2882`).

Periodic greedy eval during training (50 heldout + 20 `rl_targets` prompts, k=1), recounted
from `eval_periodic.jsonl.gz` and equal to the `metrics.jsonl` eval lines at every step:
45/70 at steps 0, 10, 20, 30 and 44/70 at steps 40, 50 (one length-4 prompt lost). Lengths
7 to 12: 0/20 at every step.

### 3. Textbook solves on the 72, equal attempts (k=64, T=1.0, seed 0, both dumps)

| slice | control (frozen checkpoint) | RL (`run/final`) |
| --- | --- | --- |
| dev58 | 8/58 = 0.138 [0.072, 0.249] | 8/58 = 0.138 [0.072, 0.249] |
| train14 | 2/14 = 0.143 [0.040, 0.399] | 2/14 = 0.143 [0.040, 0.399] |
| all 72 | 10/72 = 0.139 [0.077, 0.237] | 10/72 = 0.139 [0.077, 0.237] |
| dev, reference_lines >= 7 (43) | **0/43** [0, 0.082] | **0/43** [0, 0.082] |

Solved sets: 9 problems in common; control-only `textbook_eec1f557…` (ref 6,
`((P&Q)>P)&(P>(P&Q)) |- P>Q`), RL-only `textbook_b3af4553…` (ref 6, a NEGI proof). Every solve
is at reference length 6 or below (ref lengths of the solved dev problems: 1, 2, 2, 3, 5, 6, 6,
6, and 6 for the RL-only one; the two train14 solves, `P |- ~~P` and `Q |- P>Q`, carry no
reference length). One solved problem is trivial (`textbook_34f93da8…`, the identity
`((P>Q)&(Q>P)) |- ((P>Q)&(Q>P))`). Slice history reminder for any write-up: the 14 were RL
training prompts in the cap-comparison sprint, the 58 were its development benchmark.

### 4. Leakage (renaming classes)

My `canon_key` (relabel P,Q,R,S by first appearance in `premises |- conclusion`) reproduces the
fork's `key` field on all 154,990 `train.jsonl.gz`, 5,000 `heldout`, 4,495 `rl_targets` and
2,285 `transfer` records with 0 mismatches, from both the `thm` and the `prompt` field. The
1,600 `prompts.jsonl` rows: all `source == "train"`, 1,600 distinct prompts and 1,600 distinct
classes, lengths 3/4/5 = 513/549/538, every stored `key` equals my recomputed key, and all 1,600
keys, prompt texts and `canonical_theorem` values are in `train.jsonl.gz` (1600/1600 each).
All 12,800 training completions are on these 1,600 prompts (8 per prompt, 256 per step).

Intersections of the 1,600 RL prompt classes with the evaluation pools:

| pool | `canon_key` (fork's class) | `canonical_theorem` (atom bijection + premise permutation) |
| --- | --- | --- |
| (a) heldout.jsonl (5,000) | **0** | 2 |
| (b) transfer.jsonl (2,285) | **0** | 0 |
| (c) textbook72 | **0** | 0 |
| (d) validation_36.jsonl | **0** | 0 |
| rl_targets.jsonl (4,495) | 0 | 0 |
| test_short_prompts.jsonl (267 prompts, 256 classes) | **2** | 2 |
| test_long_prompts.jsonl (532 prompts, 531 classes) | 0 | 0 |

The two `canonical_theorem` hits on heldout are premise-permutation variants, not renamings
(e.g. RL prompt `((P&Q)>P) , (P&Q) |- P` vs heldout `(P&Q) , ((P&Q)>P) |- P`). The fork's split
is by `canon_key`, which keeps premise order, so this is inherited from the fixed data: the
whole `train.jsonl.gz` shares 95 `canonical_theorem` classes with `heldout.jsonl` and 0
`canon_key` classes.

**The two `test_short` hits are also inherited from the fixed data and are a finding about the
fixed components, not about this run:** `train.jsonl.gz` (the set the checkpoint was pre-trained
on for about five epochs) shares **110 of the 256** `test_short` renaming classes and 6 of the
531 `test_long` classes (`heldout.jsonl` shares 6 with `test_short`; `rl_targets` and
`transfer` share 0). The executor's `data.banned_keys()` covers heldout, transfer,
validation_36 and the 72 but not the test files, so `build_prompt_set` cannot exclude them.
The RL prompt rows involved are `prompts.jsonl` rows 119 (length 4) and 575 (length 3), 16
completions in total at steps 9 and 45. No test file was scored in this run and no reported
number depends on the test files; no test prompt text is quoted here. Whether the strict
wording of the quarantine rule applies is discussed in the Verdict.

### 5. Reward sanity

- `reward_std` (TRL's within-group std, averaged over the 32 groups of 8) is nonzero on 50/50
  steps, min 0.174, mean 0.315.
- `frac_reward_zero_std` per step: mean **0.678**, range 0.469 to 0.844; recomputed from the
  completions dump (1,600 groups of 8 consecutive rows, all 8 sharing one prompt, 0 malformed
  groups) it is exactly 0.6775. So about two thirds of groups carried no learning signal, but
  every step had some.
- Mean reward per step from `metrics.jsonl`: first 5 steps 0.876, last 5 steps 0.923, first 10
  0.871, last 10 0.906, min 0.754 (step 19), max 0.969 (step 46). Recomputed from the
  completions dump by re-verification: first 5 0.876, last 5 0.923, overall 0.883 (identical).
  Accept rate by prompt length in the rollouts: 3: 3945/4104, 4: 4020/4392, 5: 3343/4304.
  Note `completions.jsonl.gz` numbers steps 0 to 49 while `metrics.jsonl` numbers them 1 to 50.
- KL max 0.00172, last 0.00050. Entropy 0.055 to 0.078, first 0.062, last 0.062 (flat).
  grad_norm max 0.293, learning rate constant 1e-5, `completions/clipped_ratio` 0 on every step
  (max completion length 226 of the 400 cap), mean length 60 to 83 tokens, unparseable
  0.4 percent mean, invalid 11.7 percent mean, throughput mean 31.6 completions/s.
- Trivial instances (conclusion is a premise, or `F` among premises) among accepted proofs:
  greedy 0/907 and 0/911; long pool 0/149 and 0/112; training rollouts 0/11,308 (0 of the 1,600
  prompts is trivial); textbook 1/240 and 2/264 accepted rows, which are 1 of the 10 solved
  problems in each dump (the identity above). The `eval_periodic` dump: 0/268.

### 6. Raw samples read

Twenty random rows of `completions.jsonl.gz`, seed 20260923, indices
[446, 537, 923, 1367, 3337, 3432, 4027, 4040, 4799, 5659, 6065, 6903, 7947, 8087, 8450, 8499,
9528, 10539, 10779, 11228] (full text in `review/samples.txt`): 16 accepted, 4 rejected, my
verdicts equal the stored rewards. The four rejections are legitimate (ORI1 producing the wrong
disjunct twice, IMPE citing a conjunction `(~Q)&R` instead of the antecedent `~Q`, NEGE on two
non-contradictory lines `Q` and `~R`). The accepted ones are ordinary 3 to 5 line derivations,
e.g. row 8087:

```
THM ( ~ ( ~ ( ~ ( S > ( Q v Q ) ) ) ) ) SEQ ( ~ ( S > ( Q v Q ) ) ) PRF
N1 ( ~ ( ~ ( ~ ( S > ( Q v Q ) ) ) ) ) : PR ; N2 | ( ~ ( ~ ( S > ( Q v Q ) ) ) ) : AS ; N3 | F : NEGE N2 N1 ; N4 ( ~ ( ~ ( ~ ( S > ( Q v Q ) ) ) ) ) : NEGI N2 N3 ; N5 ( ~ ( S > ( Q v Q ) ) ) : DN N4 ; QED
```

Every accepted proof at prompt length >= 7 was inspected: 149 rows (14 records) in the control
long pool and 112 rows (11 records) in the RL long pool, all with exactly 7 lines by the strict
verifier's count; **none at length 8 or above** in any dump, and **no textbook problem with
`reference_lines >= 7` was accepted** by either model (0 rows). All 25 distinct records have the
same shape: premises, two or three nested assumptions, one inner rule (ANDI, ANDE, ORI), closed
by nested IMPI. They are genuine 7-line derivations of the stated sequents, but they are the
easiest kind of length-7 target: the model is not doing anything it does not do at length 5.
Two examples (control dump):

```
THM ( S & R ) , ( R v Q ) SEQ ( ( ~ ( ~ P ) ) > ( ( ~ P ) > R ) ) PRF
N17 ( S & R ) : PR ; N18 ( R v Q ) : PR ; N19 | ( ~ ( ~ P ) ) : AS ; N20 | | ( ~ P ) : AS ; N21 | | R : ANDE2 N17 ; N22 | ( ( ~ P ) > R ) : IMPI N20 N21 ; N23 ( ( ~ ( ~ P ) ) > ( ( ~ P ) > R ) ) : IMPI N19 N22 ; QED
```

```
THM ( P > ( P v Q ) ) SEQ ( P > ( ( ~ P ) > ( ( ( ~ P ) & P ) v ( ( ~ P ) & ( P > ( P v Q ) ) ) ) ) ) PRF
N36 ( P > ( P v Q ) ) : PR ; N37 | P : AS ; N38 | | ( ~ P ) : AS ; N39 | | ( ( ~ P ) & ( P > ( P v Q ) ) ) : ANDI N38 N36 ; N40 | | ( ( ( ~ P ) & P ) v ( ( ~ P ) & ( P > ( P v Q ) ) ) ) : ORI2 N39 ; N41 | ( ( ~ P ) > ( ( ( ~ P ) & P ) v ( ( ~ P ) & ( P > ( P v Q ) ) ) ) ) : IMPI N38 N40 ; N42 ( P > ( ( ~ P ) > ( ( ( ~ P ) & P ) v ( ( ~ P ) & ( P > ( P v Q ) ) ) ) ) ) : IMPI N37 N41 ; QED
```

Textbook solves quoted (control; the RL dump's accepted samples for the same problems are
mostly token-identical, as expected at KL 0.0005 with the same seed):

```
THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF        (ref 6)
N20 ( P > Q ) : PR ; N21 ( Q > R ) : PR ; N22 | P : AS ; N23 | Q : IMPE N20 N22 ; N24 | R : IMPE N21 N23 ; N25 ( P > R ) : IMPI N22 N24 ; QED

THM ( ~ ( P v ( Q v R ) ) ) SEQ ( ~ R ) PRF        (ref 6)
N14 ( ~ ( P v ( Q v R ) ) ) : PR ; N15 | R : AS ; N16 | ( Q v R ) : ORI2 N15 ; N17 | ( P v ( Q v R ) ) : ORI2 N16 ; N18 | F : NEGE N17 N14 ; N19 ( ~ R ) : NEGI N15 N18 ; QED
```

### 7. T1 gate, recounted

`nta.py` (own forward pass through the fork's `GPT`, prompt-masked, all 5,000 heldout records,
right-padded batches): next-token accuracy **0.98346**, teacher-forced loss 0.06686, 363,090
target tokens. Gate threshold 0.95: pass.

### 8. Wording against n (what the numbers support)

- One seed, 50 steps, KL 0.0005: control and RL are the same model to within noise on every
  evaluation. Greedy 907 vs 911 of 1,000, long pool 14 vs 11 of 200, textbook 10 vs 10 of 72
  (9 shared), 0 vs 0 of 43. None of these differences is a result; the plan needs 3 seeds for
  any control-vs-RL difference and treats 0 to 1 textbook solves as noise.
- The reward rise (first 5 steps 0.876 to last 5 0.923) is a within-run trend on the training
  prompts at one seed with per-step noise of about 0.05 (min 0.754, max 0.969); it is consistent
  with "reward moves up without the KL blowing up" as a wiring check, not with any capability claim.
- The long-pool length-7 solves are the frozen checkpoint's (RL solves a subset), and all are the
  nested-IMPI schema; "max length solved 7" should not be read as length-7 capability.
