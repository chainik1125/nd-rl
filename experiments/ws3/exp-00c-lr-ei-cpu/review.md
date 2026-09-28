# exp-00c learning-rate sweep and expert iteration on CPU: review

Reviewer session, separate from the executor (plan section 6). Phase 1 (this section) was written
from `proposal.md`, `configs/`, `launch.sh`, `status.md`, `make_split.py`, `make_pool.py`, the
raw files (`split/`, `eval_set.jsonl`, `candidates.jsonl.gz`, `pool.jsonl`,
`filter_summary.json`, `filter_dump.jsonl.gz`, and under each `run_<arm>/`: `baseline/`,
`evaluate/` dumps, `metrics.jsonl`, `completions.jsonl.gz`, `prompts.jsonl`, `run_manifest.json`,
`stages.jsonl`, `eval_periodic`, `sft_data/`, `hf_base/`, `final/`), the code under `code/ws3/`,
and git history. `results.md`, `analysis.md`, `analysis.json`, `analysis_len7.png`,
`analyze.py`, any `run_<arm>/report/` file and the `verify_lr3e-5/` and `verify_ei/` folders were
not opened before this section was committed. (From `stdout.log` I read only the two
`=== make_pool` timestamp lines.)

All numbers below come from my own scripts in `review/`, run from `code/` as
`uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/review/<script>.py`:
`recount.py` (outputs `recount.json`, `recount_stdout.txt`), `leak.py` (`leak.json`),
`samples.py` (`samples_random.txt`, `long_proofs.txt`, `structure.json`), `weights_check.py`
(`weights_check.json`), helpers in `recount_lib.py`. The only project code imported is `ndpath`
(sys.path), `strict_verify.safe_verify` (checker of record) and the vendored
`nd_data.canonical_theorem` (second-opinion key). Nothing from `code/ws3/eval/` or
`code/ws3/data.py` is imported.

Notation. "avg" = mean over problems of the per-problem verified fraction (k=32, T=1.0). CI =
95% normal interval over problems (a 10,000-resample bootstrap, seed 0, agrees to two decimals;
both are in `recount.json`). "any" = problems with at least one verified sample.
"reliable" = per-problem rate >= 0.5. Per-problem categories on the eval set, base = frozen
checkpoint: **new capability** base 0/32 and trained >= 0.5; **new-any** base 0/32 and trained
>= 1/32; **sharpened** base > 0 and trained rate higher; **worse** base > 0 and trained rate
lower; **lost** base > 0 and trained 0/32.

## Recount

### 0. Gates A and B, fixed components

| check | result |
| --- | --- |
| Gate A: `proposal.md` commit `487f43d` at 2026-09-27T23:36:25Z (author and committer) | precedes `=== make_pool start 23:36:36Z` (stdout.log line 1), the `filter_dump.jsonl.gz` / `filter_summary.json` / `pool.jsonl` mtimes (23:40:42Z), the baseline `environment.json` (23:40:45Z), and the first `metrics.jsonl` `utc` of every arm: lr3e-5 23:44:12Z, lr1e-4 23:57:22Z, lr3e-4 2026-09-28T00:09:45Z, ei 00:23:44Z. **Pass.** `split/`, `eval_set.jsonl`, `candidates.jsonl.gz` (mtimes 23:31Z) predate the commit, as the proposal says; they are model-free, and the committed copies equal the working files (clean tree, `git diff 487f43d HEAD` empty for them). The pre-proposal plumbing smoke the proposal describes (one GRPO step at lr 1e-3 in a scratch folder) cannot be verified from the artefacts; it is disclosed and no artefact here depends on it. |
| `status.md` last line | `DONE 2026-09-28T00:36:16Z`. Pass. |
| `bash code/ws3/fork_pin/fetch.sh --verify` | all 13 hashes match (model.py, tokenizer.py, nd_verify/*, 8 data files, pretrain.pt). Pass. |
| `git diff charles-pre-ws3 -- code/experiments/current/r1_cot/vendor/` | empty (0 lines). Pass. |
| code change under `code/` between the proposal commit and the run | `git diff 487f43d HEAD -- code/` and `git diff 487f43d -- code/` (working tree): empty. The only commit between the proposal and the run is `b01be6f` (`.gitignore`). `run_manifest.json` `git_commit` = `b01be6f`. `analyze.py`, `make_*.py`, `configs/`, `launch.sh`, `proposal.md` unchanged since `487f43d`. Pass. |
| control model | `run_<arm>/hf_base/model.safetensors` vs `fork_pin/ckpt/pretrain.pt`: 52/52 tensors, max abs diff 0.0 in all four arms. `final/` differs in 52/52 tensors; relative Frobenius distance lr3e-5 0.0034, lr1e-4 0.0096, lr3e-4 0.0249, ei 0.0089. Checkpoint sha256 in `run_manifest.json` = `418af3f9…` (MANIFEST). |
| shared baseline | the three copied `run_<arm>/baseline/dumps/*` are content-identical to `run_lr3e-5/baseline/dumps/*` (decompressed; the `.gz` bytes differ only by gzip header time). |
| equal evaluation conditions | every dump: long pool k=32 T=1.0 (6,400 rows), textbook k=32 T=1.0 (2,304 rows), greedy k=1 T=0 (1,000 rows), for the frozen model and all four arms. The long-pool dump prompt set = `eval_set.jsonl` (200 names, lengths match `L_true` with 0 mismatches); textbook dump names = the 72; greedy sets identical across models. |
| Plan section 4 metrics, GRPO arms | `metrics.jsonl` 54 lines each (48 rollout lines, 5 periodic-eval lines at steps 0/12/24/36/48, 1 end summary). On all 48 rollout lines: `reward`, `reward_std`, `kl`, `entropy`, `reward/unparseable_frac`, `reward/invalid_frac`, `completions/mean_length`, `completions/max_length`, `completions/clipped_ratio`, `grad_norm`, `learning_rate`, `rollout/completions_per_sec`, `utc`, `step` (0 missing); `eval/pass_rate_len<L>` on the 5 eval lines; `utc` and `step` on every line. **Complete.** |
| EI metric set | 7 lines (4 periodic-eval at rounds 0 to 3, 3 round lines). Each round line has `reward`, unparseable, invalid, `ei/prompts_solved_frac`, `ei/kept_this_round`, `ei/train_records`, `sft/loss_first`/`sft/loss_last`, `rollout/total_completions` (4,158 / 8,316 / 12,474), `rollout/completions_per_sec`, `utc`, `step`. **Complete.** |
| training code vs evaluation files | `data.train_prompts` reads `pool.jsonl` and reads heldout, transfer, validation_36, the 72 textbook problems, the test prompt files and `split/rl_targets_eval.jsonl.gz` only to build a banned-key set (`assert_no_leak`). The periodic-eval callback (GRPO `run.py` 400-413, EI `rl/ei.py` 191-205) samples greedily on 50 heldout + 40 `eval_set` records and only writes `metrics.jsonl`/`eval_periodic`; `final/` is the last step (`trainer.save_model` after `train()`, `save_strategy: no`; EI saves after round 3), never a periodic-eval-selected checkpoint. No training batch contains an evaluation prompt (section 3). **Pass**, same reading as the exp-00 review. |

### 1. Re-verification of every dumped sample

Every row re-run through `safe_verify(prompt, completion)` and compared with the stored `ok`
(dumps) or `reward >= 0.5` (completions), and accepted rows' line counts with
`n_lines_written` / `n_lines`.

| dump | rows | accepted (mine) | verdict disagreements | line-count disagreements |
| --- | --- | --- | --- | --- |
| frozen long_pool / greedy / textbook72 | 6,400 / 1,000 / 2,304 | 627 / 978 / 233 | 0 | 0 |
| lr3e-5 long_pool / greedy / textbook72 | 6,400 / 1,000 / 2,304 | 2,207 / 982 / 260 | 0 | 0 |
| lr1e-4 long_pool / greedy / textbook72 | 6,400 / 1,000 / 2,304 | 2,017 / 978 / 275 | 0 | 0 |
| lr3e-4 long_pool / greedy / textbook72 | 6,400 / 1,000 / 2,304 | 1,720 / 881 / 194 | 0 | 0 |
| ei long_pool / greedy / textbook72 | 6,400 / 1,000 / 2,304 | 3,130 / 978 / 248 | 0 | 0 |
| filter_dump (frozen, 8 per candidate) | 13,296 | 3,158 | 0 | - |
| training completions lr3e-5 / lr1e-4 / lr3e-4 / ei | 12,288 / 12,288 / 12,288 / 12,474 | 10,337 / 10,417 / 7,490 / 9,819 | 0 | 0 |
| eval_periodic lr3e-5 / lr1e-4 / lr3e-4 / ei | 450 / 450 / 450 / 360 | - | 0 | - |
| ei sft_data rounds 1 / 2 / 3 | 696 / 781 / 786 | 696 / 781 / 786 | 0 | - |

115,127 rows, zero disagreements. Rewards take only 0.0 and 1.0.

**1b. Split, filter and pool.** `make_split.py`'s rule (sha256 of the fork key, even = train)
reproduces both halves; train half 2,223 (864 at L7, 498 at L8), eval half 2,272 (936 at L7,
502 at L8); halves share 0 `canon_key` and 0 `canonical_theorem` classes; together they are
exactly the 4,495 `rl_targets` names, with every field equal except `source` (overwritten to
`rl_targets`) and the added `key`/`length`. `eval_set.jsonl`: 200 records (160 L7, 40 L8), all in
the eval half. Candidates 1,662 = 864 L7 + 498 L8 train-half + 300 L6 from `train.jsonl`.
Filter recount (8 samples per candidate): L6 237 all / 63 mixed / 0 zero; L7 46 all / 131 mixed /
687 zero; L8 0 all / 4 mixed / 494 zero. Pool = every mixed L7 and L8 prompt plus all 63 mixed L6
prompts (the cap of 131 // 2 = 65 did not bind): **198 = 63 L6 + 131 L7 + 4 L8**, each record's
`frozen_filter_ok` equals my count. EI k = round(12,288 / (3 x 198)) = 21, so EI drew
3 x 21 x 198 = **12,474** samples, 186 (1.5%) more than a GRPO arm's 12,288, as the
pre-registered rounding rule implies.

**1c. Eval set (held-out half), avg solve rate by length, k=32, T=1.0.**

| model | L7 avg [95% CI] (n=160) | L7 any | L7 reliable (>=0.5 / >=0.9) | L8 avg [95% CI] (n=40) | L8 any | L8 reliable |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 0.1225 [0.079, 0.166] | 39 | 19 / 9 | 0.0000 | 0 | 0 |
| lr3e-5 | 0.4299 [0.365, 0.495] | 97 | 74 / 45 | 0.0047 [0, 0.010] | 3 | 0 |
| lr1e-4 | 0.3908 [0.327, 0.455] | 97 | 65 / 37 | 0.0125 [0, 0.027] | 5 | 0 |
| lr3e-4 | 0.3332 [0.270, 0.397] | 85 | 53 / 33 | 0.0109 [0, 0.031] | 2 | 0 |
| ei | 0.5449 [0.479, 0.611] | 118 | 93 / 65 | 0.2656 [0.151, 0.381] | 19 | 12 |

**1d. Per problem, frozen vs trained, eval set.**

| arm | L7 new capability | L7 new-any | L7 sharpened | L7 worse | L7 lost | L7 lost reliable (>=.5 to <.5) | L8 new capability | L8 new-any |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr3e-5 | **42** | 58 | 31 | 4 | 0 | 0 | 0 | 3 |
| lr1e-4 | **30** | 58 | 30 | 3 | 0 | 0 | 0 | 5 |
| lr3e-4 | **26** | 49 | 22 | 17 | 3 | 5 | 0 | 2 |
| ei | **62** | 79 | 29 | 8 | 0 | 1 | **12** | 19 |

Paired comparison on the 39 L7 problems the frozen model solves at least once (headline of the
proposal), exact two-sided sign test on up vs down (ties dropped), mean paired difference with a
bootstrap 95% CI (seed 0), and the same mean over all 160 L7 problems:

| arm | up / down / same | sign test p | mean diff, 39 base-solved [CI] | mean diff, all 160 [CI] |
| --- | --- | --- | --- | --- |
| lr3e-5 | 31 / 4 / 4 | 3.5e-6 | +0.273 [0.184, 0.372] | +0.307 [0.251, 0.367] |
| lr1e-4 | 30 / 3 / 6 | 1.4e-6 | **+0.325** [0.230, 0.429] | +0.268 [0.215, 0.323] |
| lr3e-4 | 22 / 17 / 0 | 0.52 | +0.163 [0.020, 0.306] | +0.211 [0.153, 0.271] |
| ei | 29 / 8 / 2 | 7.5e-4 | +0.245 [0.135, 0.364] | +0.423 [0.357, 0.488] |

**1e. Greedy heldout stride (lengths 2 to 6, 1,000 problems, k=1, T=0).**

| model | avg (solved/1000) | L2 / L3 / L4 / L5 / L6 | lost / gained vs frozen | drop vs frozen |
| --- | --- | --- | --- | --- |
| frozen | 0.978 (978) | 1.000 / 1.000 / 0.980 / 0.985 / 0.925 | - | - |
| lr3e-5 | 0.982 (982) | 1.000 / 1.000 / 0.985 / 0.985 / 0.940 | 2 / 6 | -0.004 |
| lr1e-4 | 0.978 (978) | 0.990 / 1.000 / 0.980 / 0.980 / 0.940 | 9 / 9 | 0.000 |
| lr3e-4 | **0.881** (881) | 0.820 / 0.945 / 0.870 / 0.925 / 0.845 | **106** / 9 | **0.097** |
| ei | 0.978 (978) | 0.995 / 0.995 / 0.985 / 0.985 / 0.930 | 10 / 10 | 0.000 |

### 2. Textbook 72 (k=32, T=1.0, seed 0, same for every model)

| model | dev58 avg / any / reliable | train14 avg / any / reliable | all 72 avg / any / reliable / expected solves | dev ref >= 7 (43) avg / any / reliable |
| --- | --- | --- | --- | --- |
| frozen | 0.0927 / 7 / 5 | 0.1362 / 2 / 2 | 0.1011 / 9 / 7 / 7.28 | 0 / **0** / 0 |
| lr3e-5 | 0.1024 / 9 / 6 | 0.1562 / 3 / 2 | 0.1128 / 12 / 8 / 8.13 | 0.0015 / **1** / 0 |
| lr1e-4 | 0.1105 / 9 / 7 | 0.1562 / 3 / 2 | 0.1194 / 12 / 9 / 8.59 | 0.0029 / **1** / 0 |
| lr3e-4 | 0.0781 / 7 / 5 | 0.1094 / 3 / 1 | 0.0842 / 10 / 6 / 6.06 | 0 / **0** / 0 |
| ei | 0.0959 / 9 / 6 | 0.1562 / 3 / 2 | 0.1076 / 12 / 8 / 7.75 | 0.0022 / **1** / 0 |

Problems gained (frozen 0/32, trained >= 1/32): lr3e-5, lr1e-4, ei: `textbook_902dbc…`,
`textbook_c18068…`, `textbook_f8e226…`; lr3e-4: `902dbc…`, `f8e226…`. Lost: only lr3e-4 loses
`textbook_dd80d9…` (`~~P |- P`). Textbook new capability (>= 0.5): lr1e-4 `902dbc…` only. The one
reference-length >= 7 solve is `textbook_c18068…`, `|- (P v Q) > (Q v P)` (reference 7), at
2/32 (lr3e-5), 4/32 (lr1e-4), 3/32 (ei), always the same genuine 7-line proof (assume `P v Q`,
two ORI cases, ORE, IMPI). Per-slice CIs are in `recount.json`; every textbook difference
between models is inside them. Slice history: the 14 were RL prompts in the cap-comparison
sprint, the 58 its development benchmark.

### 3. Leakage

My `canon_key` reproduces the fork's `key` on all `rl_targets`, heldout and transfer records
(0 mismatches). Trained-on sets: `pool.jsonl` (198), each arm's `prompts.jsonl` (198, identical
to the pool), the distinct prompts of each `completions.jsonl.gz` (198 each, all in the pool),
EI `sft_data` round 1/2/3 prompts (187/197/197), and for completeness the 1,662 filter
candidates. Evaluation sets: `eval_set.jsonl`, the whole eval half (2,272), heldout, transfer,
validation_36 (and its reference-proof file), textbook 72, `test_short_prompts`,
`test_long_prompts` (prompts only).

**Every intersection is 0** under `canon_key`, under `canonical_theorem`, and as exact prompt
text, for every trained-on set against every evaluation set (`leak.json`). The fixed
pre-training set `train.jsonl.gz` also shares 0 classes (either key) with `eval_set`.
As an extra, stricter check I erased atom identity entirely (all atoms -> X, premises sorted):
0 of the 160 L7 and 0 of the 40 L8 eval problems have a skeleton that appears in the pool. The
two split halves are disjoint and their union is exactly `rl_targets` (section 1b). The test-file
classes are banned from RL prompts in this run (`make_split.py`, `data.train_prompts`), which
closes the exp-00 review's test-file flag for this experiment.

### 4. Reward sanity

| arm | training reward re-check | groups with nonzero std (of 768) | first 5 / last 5 steps | reward first 5 / last 5 steps | steps with no mixed group | KL max / last | entropy first / last | clipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr3e-5 | 0 / 12,288 disagree | 0.645 | 0.80 / 0.55 | 0.707 / 0.908 | 0 | 0.268 (one step, step 10; all others <= 0.018) / 0.0096 | 0.042 / 0.039 | 0 |
| lr1e-4 | 0 / 12,288 | 0.542 | 0.69 / 0.40 | 0.723 / 0.913 | 0 | 0.045 / 0.028 | 0.042 / 0.038 | 0 |
| lr3e-4 | 0 / 12,288 | 0.608 | 0.69 / 0.51 | 0.515 / 0.798 | 0 | 0.164 / 0.019 | 0.042 / 0.043 | max 0.023 |

Groups are 16 consecutive rows sharing one prompt and step (768 per arm, 0 malformed; 16 distinct
prompts per step). The reward means recomputed from the completions equal the `metrics.jsonl`
values. EI: verified fraction by round 0.603 / 0.845 / 0.913 (4,158 rows per round); SFT sets
696 / 781 / 786 records, every one a verified completion sampled that round for a pool prompt,
at most 4 per prompt.

Trivial instances among accepted rows (conclusion among the premises, or `F` a premise): 0 in
every long-pool, greedy and training-completion set (0/627, 0/2,207, 0/2,017, 0/1,720, 0/3,130;
0 of 10,337 / 10,417 / 7,490 / 9,819 training accepts); textbook 4/233, 3/260, 1/275, 6/194,
2/248 accepted rows, all on the identity-type textbook problems already noted in the exp-00
review. **One accepted proof is shorter than its `L_true`:** lr1e-4 on `la_rl_targets_3983`
(`(Q v Q) |- P > (S > Q)`, `L_true` 8), 7 lines: its ORE cites the same one-line subproof
`Q : AS` for both disjuncts of `Q v Q` (`ORE N1 N4 N4 N4 N4`). This is logically sound (both
cases are the same formula), accepted by the fork's `nd_verify` and the strict wrapper, and
shows `L_true` is minimal under the generator's proof shapes rather than under the checker; 1
sample of 6,400, not a reward bug. No textbook accept is shorter than its `reference_lines`.

### 5. Samples read

`samples_random.txt`: 20 random rows per model (frozen and four arms) from the concatenated
evaluation dumps, seed 20260927 (100 rows). My verdict equals the stored one on all 100. The
rejections are ordinary errors (ANDE/ORI on the wrong conjunct, IMPE with the wrong antecedent,
ORE with an unrelated assumption, parse errors like a stray `N64` as a rule name, a duplicated
line label). The frozen model's dominant length-7 failure is visible in these rows: in nested
implications it closes the innermost IMPI by citing the assumption itself as the consequent
(`IMPI N34 N34` for `~~~~S > ~S`, `IMPI N33 N33` for `S > Q`) instead of deriving the consequent
with one rule (DN, ORI, ANDI, ANDE). The trained samples insert that step, e.g. lr1e-4 row
1173: `… N29 | | | ~~~~S : AS ; N30 | | | ~S : DN N27 ; N31 | | (~~~~S > ~S) : IMPI N29 N30 …`.

`long_proofs.txt`: every distinct accepted proof (after renumbering lines) whose problem has
length >= 8 or whose proof has >= 8 lines, from every evaluation dump and every training
completion: **167 distinct proofs on 105 problems** (63 from evaluation dumps, 104 from training
rollouts). All read. On eval-set L8 problems: 26 distinct proofs on 22 problems (EI 19, lr1e-4 5, lr3e-5 3, lr3e-4 2
problems); 15 of the 22 are solved only by EI. Every one is a genuine derivation of the stated
sequent. Most share one shape: premises, three nested assumptions (often vacuous: `~~S`, `R`,
`R`), one inner rule (ANDI, ANDE, ORI, DN, IMPE), then three IMPI closings; the others have two
nested assumptions plus one or two extra steps (a top-level ORI1, IMPE or ANDE2, or a second
ANDI), and one is the 7-line ORE shortcut above. They are the length-7 nested-IMPI schema with
one more assumption or step, not new proof techniques. The 9- and 10-line training accepts
(lr3e-4 mostly) add redundant lines (repeated `BOTE`, repeated `ANDI`), which the checker allows;
they are correct. EI's greedy heldout has three 8-line proofs of length-4/6 problems with padding
lines (`R N1`, `R N3`), also correct.

Where the L7 lift comes from (`structure.json`, grouping eval L7 problems by the rule set of the
generator's own proof): the gains are concentrated in the nested-IMPI families where the frozen
model's rate is 0, e.g. `DN+IMPI` (13 problems) frozen 0.00 -> lr3e-5 0.80 / lr1e-4 0.53 /
lr3e-4 0.75 / ei 0.93; `IMPI+ORI1` (10) 0.00 -> 0.61 / 0.19 / 0.38 / 0.81; `ANDE2+IMPI` (5)
0.00 -> 0.86 / 0.60 / 0.67 / 0.91. Families that need a real case split or a contradiction
(`schema:import`, `schema:disjunctive_syllogism`, `BOTE+IMPI+NEGE+ORE`) stay at 0 to 0.02 in
every arm. Many of the 42 / 30 / 26 / 62 "new capability" L7 problems are therefore one repaired
failure mode (derive the consequent before the innermost IMPI) applied across a common template
of the `rl_targets` generator, which the pool also contains (pool rule sets include `ANDI+IMPI`
10, `IMPE+IMPI` 7, `DN+IMPI` 3). That is genuine held-out transfer by renaming class and by
skeleton, but within one generator template, not broad length-7 competence.

### 6. Pre-registered selection rule, recomputed

Among the GRPO arms, highest mean paired difference on L7 problems the frozen model solves at
least once, excluding greedy heldout drop > 0.02 or final KL > 1.0:

| arm | mean diff (39 base-solved) | greedy drop | final KL | excluded |
| --- | --- | --- | --- | --- |
| lr3e-5 | +0.273 | -0.004 | 0.0096 | no |
| lr1e-4 | **+0.325** | 0.000 | 0.028 | no |
| lr3e-4 | +0.163 | **0.097** | 0.019 | yes (greedy drop) |

**Selected: lr 1e-4.** Note the rule selects on the evaluation set itself (the held-out-half
stride), which plan T5 allows ("pick by avg solve rate on the `rl_targets` stride"); the
selected arm's eval-set numbers therefore carry a small winner's-curse bias. The ranking is not
robust across metrics on one seed: lr3e-5 is higher than lr1e-4 on L7 avg (0.430 vs 0.391), L7
reliable (74 vs 65), L7 new capability (42 vs 30) and the all-160 mean difference (+0.307 vs
+0.268), and the two lr3e-5/lr1e-4 CIs overlap heavily on every metric.

### 7. Against the proposal's expectations (for the record)

Pool 131 L7 (expected 90 to 170, met), 4 L8 (expected 5 to 25, below), L6 63 (expected at the
cap of 65; the cap did not bind); frozen L7 avg 0.1225 and 39/160 solved once (expected 0.08 to
0.14 and 30 to 40, met), L8 0/40 (met). lr3e-5 max KL 0.268 (expected 0.001 to 0.01; one-step
spike, otherwise <= 0.018) and eval L7 strongly up (expected roughly neutral). lr1e-4 max KL
0.045 (expected 0.01 to 0.2, met), greedy drop 0.000 (met), mean paired difference +0.325
(expected 0 to +0.08, far above). lr3e-4 degraded greedy by 0.097 (one of the two predicted
outcomes). EI verified fraction rose round 1 to 3 (met); its L7 mean paired difference +0.245 is
within 0.08 of the best GRPO arm's +0.325, not within 0.03. **New capability** 42 / 30 / 26 / 62
at L7 and 12 at L8 for EI (expected 0 to 1, and 0 to 2): this is the proposal's own "surprisingly
good" row (new capability >= 3 or L8 avg > 0.05), which requires a reward-bug and leak check.
Both checks came back clean here: 0 verdict disagreements in 115,127 re-verified rows, 0
overlap under every key including the atom-erased skeleton, and the long proofs are genuine.
The surprise is explained by the template concentration in section 5, not by a bug or a leak.
