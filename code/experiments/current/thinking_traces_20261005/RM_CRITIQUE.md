---
written_on: 2026-10-06
written_by: agent:claude
papers:
  - ren2025deepseekproverv2advancingformalmathematical
---

# Adversarial review of thinking formats R (root plan) and M (memo)

Scope:
- the two formats in [`rootplan.py`](../../../thinking/rootplan.py) and their sampling and training plumbing in
  [`sample.py`](../../../thinking/sample.py) and [`train.py`](../../../thinking/train.py);
- the generated data `/scratch/cowifornia/nd-rl/think/traces/{root,memo}_traces.jsonl` (+ `.stats.json`) and
  the built pickles `runs/data/{TR,TM}.pkl`;
- the tests [`test_rootplan.py`](../../../tests/thinking/test_rootplan.py) and
  [`test_formats_rm.py`](../../../tests/thinking/test_formats_rm.py);
- the Stage-1 reads ([STAGE1_STATS.md](STAGE1_STATS.md)), the GRPO lr-check runs, and the first review
  ([LIT_CRITIQUE.md](LIT_CRITIQUE.md), C8).

Branch `leon/thinking`, HEAD `b3c78cd`. `rootplan.py`, `grpo.py` and the R/M tests are uncommitted, and
`sample.py` / `train.py` have uncommitted edits; this review reads the working tree as of 2026-10-06 ~20:30 UTC.

This is a deliberately hostile reading, as Leon asked. No GPU was used. Nothing existing was edited. No R or M model
exists yet: the R/M smoke logs in `runs/smoke_rm/` are empty. **So every statement about model behaviour below is a
prediction, not an observation.**

Labels:
- **Observed:** a number computed in this review from stored files, or a fact read from code (with its location).
- **Reported:** what a cited paper states, cited by arXiv id.
- **Interpretation:** my reading, which may be wrong.

## 1. Verdicts

### R ("think once, full context")

**No, not for the question being asked.** Stage-1 R will show that the model "uses its thinking", but trivially,
because the actions are a transcription of ROOT. It will not show test-time scaling. And it cannot be evaluated
where it is supposed to help, on long proofs.

- **The ROOT is the proof.** Every decision action that is not a plain restatement of a name (92.4% of decision
  actions) has a ROOT line with the *identical goal formula and term skeleton*: 12,504 of 12,504 in a 1-in-40
  sample. Premise and `exact` steps (56% of all steps) are fixed by the state. Generating ROOT *is* proving the
  theorem; the actions add no decisions.
- **The one real hope.** Writing the proof top-down as goal reductions may be easier for a 9.5M model than writing
  it as forward steps, because it makes each subgoal formula explicit before it is needed. That is a claim about
  proof *order*. It is not test-time scaling:
  - ROOT length is set by proof length;
  - exploration is 9.2% of ROOT tokens;
  - the B cap only cuts the plan.
- **Single biggest threat: length.** On the long proofs that are the point, R is outside its training distribution
  in three ways at once (§3, R2).

  | | 21–30-line references | K12 training |
  |---|---|---|
  | ROOT cut at the 640-token budget | 54–66% | 8 of 154,977 |
  | transcript tokens, mean | 3.0k–5.0k | 882 (p99 2,445) |
  | transcripts over the model's 4,608-token context | 8–49% | 0 |

  Late actions must also retrieve ROOT from 3–5k tokens back. Of the 8 heads, 4 are NoPE, and the 4 ALiBi heads
  are penalised by −12 to −750 logits at a distance of 3,000 tokens.

### M ("memo rewritten every step")

**No, as designed.** M will also "use" its memo, and almost all of that use is copying.

- **Where the copying is.**
  - 96.6% of memo tokens at ordinary decision steps are lines already in the previous memo.
  - Premise-step memos are 100% copies.
  - `exact`-step memos are the one line `|- G := nK`.
  - A ~40-line rule reproduces 87.7% of decision actions from the memo, at skeleton level.
- **Where the planning is.** The real planning happens at step 0 and right after a box closes. There the memo
  re-derives, from the state, a plan that it had already written and then dropped on entering the box: 92.4% of
  those tokens appeared in an earlier memo (§4, M2).
- **The memo drops the wrong part.** It keeps what the Markov state already implies (the current frame's goal) and
  discards what the state does not show (the outer frames' remaining plans). That is the inverse of a useful
  carried state.
- **Single biggest threat: copy-dominated cost.** M writes about 2× T1's think tokens on K12: 479 against 238 per
  proof, rising to 995 against 442 at 12 lines. At 21–30 lines it writes 2,746 memo tokens per proof, 5.5× the
  action tokens. Its only "more thinking" lever, exploration, is 1.1% of memo tokens. At matched sampled tokens it
  must beat T0's per-attempt success on 21–30-line problems by roughly 6.6× just to break even. Stage 1 already
  showed per-step thinking (T1) losing to T0 below about 3k tokens per theorem.
- **Its strength.** M is Markov, so it does fit long proofs. Its largest pair at 21–30 lines is 816 tokens,
  against 840 in training. That is the one reason to redesign M rather than drop it.

### A pre-flight bug shared by R and M (fix before any GPU)

**Observed (code).**
- **Training.** Names are shifted by one random offset per sequence or pair, U[0, 64 − max name], over every name
  token, the think block included (`state_train_best.make_batch`, called from `train.SeqData.batch`).
- **Production sampling.** Each attempt draws an independent name base U[0, 32]
  (`sample.py:597–598`, `NAME_BASE_MAX = 32`). The environment then *renames* every name an action introduces
  (`assign=True`, `state_env._assign_names`).
- **What R and M carry forward.**
  - R's transcript keeps the model's own tokens, ROOT and action_0 with the model's names
    (`a.ctx = list(p) + list(ids)`, `sample.py:752–753`).
  - M carries MEMO_0 forward.
- **The tests don't cover it.** All R/M tests run with `NAME_BASE_MAX = 0` (autouse fixture,
  `test_formats_rm.py:100–102`).

**Interpretation.**
- ROOT and MEMO_0 are written at step 0, before any `n`-name is visible. 74% of them contain `n`-names (binders).
- From step 1 on, the R transcript and the M prompt therefore hold plan names at the model's sampled offset and
  state names at the environment's base. Whenever the two differ, which is at least 97% of attempts, the input is
  one never seen in training.
- T1 is immune, because a per-step block is written after the state it refers to.
- **Fix.** For R/M, set the environment's base from the model's first emitted name. Otherwise report R/M at base 0
  and say so. Without this, an R/M loss on any read is uninterpretable.

## 2. What the traces contain (observed, from real records)

1-in-40 sample of the 154,977 K12 traces of each format (3,875 proofs, 30,766 steps), scripts in §7. Step kinds:
- premise restatements: 22.9%;
- decisions: 44.0%;
- `exact` closes: 33.1%.

**Predictability of actions from the plan**

| measure | R (ROOT) | M (MEMO_t) |
|---|---|---|
| live plan lines per decision step | 0.99 | – |
| decision actions whose goal formula is a plan line's goal | 95.9% | 94.9% |
| decision actions with an identical goal + term-skeleton line (names abstracted) | 92.4% | 94.8% |
| the same, excluding reiterations `have nK : X := nJ` (7.6% of decisions; the plan writes the name, never the step) | **100%** (12,504/12,504) | – |
| decision action picked by a fixed emission-order rule on the memo (~40 lines of Python) | – | 87.7% |
| `exact` actions: memo is literally `\|- G := nK` | – | 100% |
| names cited in decision actions: copied verbatim from the plan line / premise `hK` → its restated `nJ` / hole → the name of the statement proving it / binder alias | 29.7% / 45.1% / 22.2% / 3.1% | – |

All name lookups that are not copies are lookups by formula in the state. The emission-rule misses are the
reference's ordering conventions: statements hoisted out of a box before it opens, and right-before-left pairs.

**Memo carry-over** (M, t ≥ 1; a line counts as "old" if its goal and abstracted right-hand side were in the
previous memo)

| step | memo tokens in new lines |
|---|---|
| premise | 0% (3,360/3,360 memos have no new line) |
| decision, not right after a box close | 3.4% (about 60% of it is exploration) |
| decision right after a box close (22.9% of decisions) | 97.2%, of which **92.4% were written in an earlier memo and dropped** |
| all memos at t ≥ 1 with no new line at all | 55.3% |

- Step 0 holds 24% of all memo tokens.
- Exploration is 1.1% of memo tokens; 2.2% of memos contain a dead end.
- Consecutive memos that contradict each other's plan for the same goal: 0.03%. This is negligible, but it does
  exist: the K12 references contain redundant statements.

**Length on long reference proofs.** CPU generation with `rootplan.root_trace` / `memo_trace` on the reference
proofs of the five OOD sets (840 + 80 theorems, binned by ND lines). K12 training values are in brackets.

| ND lines | n | ROOT cut at 640 | R transcript mean (max) | over 4,608 | R re-prefill / Markov prefill per attempt | M step-0 memo cut | M memo tokens per proof | action tokens per proof |
|---|---|---|---|---|---|---|---|---|
| K12 2–12 | 154,977 | [0.005%] | [882 (4,458)] | [0] | [4,571 / 650] | [6.9%; 26% at 12 lines] | [479; 995 at 12 lines] | [124] |
| ≤12 (OOD) | 201 | 0% | 1,206 (2,020) | 0% | 7,868 / 816 | 25% | 772 | 159 |
| 13–20 | 274 | 5% | 2,980 (5,183) | 15% | 28,413 / 2,266 | 89% | 2,058 | 294 |
| 21–30 | 234 | 66% | 4,991 (8,211) | 49% | 67,208 / 3,898 | 100% | 2,746 | 497 |
| 31+ | 131 | 100% | 8,392 (12,434) | 100% | 186,035 / 6,809 | 100% | 3,450 | 977 |
| Charles 21–30 only | 13 | 54% | 3,001 | 8% | – | 100% | 1,663 | – |

The generated families have large formulas, so their states are long. Charles's textbook problems are the better
guide to the sealed 21–30 set, and on them the context overflow is 8%, not 49%. ROOT truncation and step-0 memo
truncation are high on both.

## 3. Ranked critiques: R

### R1. ROOT is a re-rendering of the proof; the actions are a transcription. The "uses its thinking" test is pre-passed

**Issue.** LIT_CRITIQUE C8 said R "re-renders the reference proof". The numbers in §2 make that exact:
- every non-reiteration decision action has a ROOT line with the same formula and term skeleton;
- 70% of the cited names are lookups by formula in the state, which needs no decision;
- the 0.99 live ROOT lines per decision step are the decision steps, reordered top-down.

So the work splits cleanly:
- all decisions are made while writing ROOT;
- the actions (48% of R's loss tokens; ROOT is about 52%) teach copying plus name lookup.

**Reported.**
- Bachmann and Nagarajan (arXiv 2403.06963): under teacher forcing, later tokens can be fitted by a "Clever Hans
  cheat" that reads the revealed prefix. The hard first decision is then learned at ~1/d, chance on a degree-d
  path-star graph. Training on *reversed* paths reaches 100%.
- Prover-V1.5 (arXiv 2408.08152) has a plan-first CoT format ("propose complete mathematical reasoning at the
  beginning of the proof block and detailed step planning before each tactic"). On miniF2F pass@32 it scores:

  | model | non-CoT | CoT |
  |---|---|---|
  | SFT | 49.8 ± 0.3% | 50.4 ± 0.4% |
  | RL | 50.5 ± 0.6% | 51.6 ± 0.5% |

  "the advantage of CoT over the non-CoT mode is amplified as the sample budget increases."
- Draft-Sketch-Prove (arXiv 2210.12283) and Prover-V2 (arXiv 2504.21801) plan with a *sketch with gaps*. The gaps
  are filled by a separate search: an automated prover in DSP (20.9% → 39.3% on competition problems), a 7B
  prover solving the `have … := by sorry` subgoals in Prover-V2.
- Gandhi et al. (arXiv 2503.01307): "RL selectively amplifies empirically useful behaviors while suppressing
  others". Backtracking and verification were retained and strengthened; backward chaining and subgoal setting
  diminished.
- The project's own precedent: r1_cot, a think block that re-rendered the proof, lost at matched tokens (10/58 vs
  11/58 at k 16; PROPOSAL §0).
- **On whether models use their CoT:**
  - Lanham et al. (arXiv 2307.13702): models vary across tasks in how strongly they condition on the CoT, "sometimes
    relying heavily on the CoT and other times primarily ignoring it"; larger models are less faithful.
  - Turpin et al. (arXiv 2305.04388): CoT explanations rationalise answers driven by biasing features, with
    accuracy drops of up to 36%.
  - Stechly et al. (arXiv 2505.13775), ~380M models from scratch on A* traces:

    | traces | Wilson mazes | OOD Drunkard mazes |
    |---|---|---|
    | correct | 79.9% | 0.0% |
    | swapped between problems | 83.3% | 11.7% |

    Under GRPO, accuracy rose while trace validity fell from the mid-80s to the mid-50s %.
- Small transformers trained from scratch do benefit from intermediate-step data (arXiv 2307.03381: CoT-style data
  "significantly and simultaneously improves accuracy, sample complexity, and convergence speed"). This is the
  steelman for writing *something* before the answer.

**Interpretation.**
- **Where R could gain.** The actions after ROOT are the Clever-Hans part. The only place R can gain is in ROOT
  generation itself. The best argument *for* R is the path-star reversal result: a top-down goal reduction is
  the "reverse" order of forward ND, and each line's subgoal is given by its parent's hole.
- **What R is not.** Unlike DSP and Prover-V2, ROOT has no gaps. There is no division of labour, only the same
  proof written twice.
- **What RL will do to it.** ROOT consists almost entirely of the two behaviours that Gandhi et al. saw RL
  suppress, backward chaining and subgoal setting.
- **The dependence test will mislead.** The within-model test "B 0 vs free B" will show large dependence, and so
  will swap and corrupt interventions. That shows plan-following, which is guaranteed by construction. It does
  not show reasoning.
- **The faithfulness question is inverted.** Lanham's and Stechly's worry is a model that ignores its trace. R
  cannot ignore its trace: its actions *are* the trace. So R will "fail" a Stechly swap test (a foreign plan is
  followed), where per-step T1 might pass it. Neither outcome says the plan computed anything.
- **ROOT is a rationalisation.** Every ROOT is written from the known answer, Turpin's concern turned around. At
  test time the model must produce the rationalisation before it has an answer, which means solving the problem
  in the plan.

**Fix or control.**
- **Oracle-ROOT read.** Insert the reference ROOT at test time on held-out K12 or the 13–20-line set, which has
  references. If it reaches ≥ 90%, transcription is solved and the whole question is ROOT generation.
- **R-verbatim.** ROOT = the action list itself, in forward order, inside the think block. If R ≈ R-verbatim, the
  top-down order is not what helps, and any gain is "write the proof twice".
- **R-noplan** (§3, R8).

### R2. Length: the long-proof regime is outside R's trained context, ROOT is cut where it matters, and late actions must retrieve ROOT through NoPE heads

**Observed.**
- **R transcripts, K12:** mean 882 tokens, p99 2,445, max 4,458; 8 of 154,977 over 4,096. The model context is
  4,608 (`train.py` docstring), and the sampler ends an attempt that would exceed it with `fail:context`.
- **R transcripts, 21–30-line references:** mean 3,001 (Charles) to 4,991 (generated families); 8% and 49% over
  4,608. At 13–20 lines, the RL pool's range, 15% of the family references are over 4,608.
- **ROOT budget.** The 640-token budget cuts ROOT in 54–66% of 21–30-line references and 100% of 31+, against
  8 ROOTs (0.005%) in training. A cut ROOT covers the path to the first statement, then a depth-first prefix
  (`PlanWriter.select`). Actions beyond its coverage never occur in training: K12 ROOTs cover 100% of lines.
- **Attention.** `ALiBiGPT` has 6 layers and 8 heads per layer: 4 ALiBi heads with slopes 1/4, 1/16, 1/64 and
  1/256, and 4 NoPE heads (`best_model.alibi_slopes`). At a distance of 3,000 tokens the ALiBi biases are −750,
  −188, −47 and −12 logits.
- **Re-prefill.** The sampler re-prefills the whole transcript every step: 4,571 tokens per K12 attempt against 650
  for a Markov prompt (6.1×), and 67,208 against 3,898 at 21–30 lines (17×).

**Reported.**
- ALiBi (arXiv 2108.12409) extrapolates in *perplexity*, 1024 → 2048 at 1.3B, by a linear distance penalty: a
  recency bias.
- Kazemnejad et al. (arXiv 2305.19466), small models trained from scratch: "the most commonly used positional
  encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream
  tasks", with NoPE best. Also, "scratchpad is not always helpful … its format highly impacts the model's
  performance".
- Wang et al. (arXiv 2404.12224): NoPE "still has a limited context length". Its failure is tied to "the
  distraction of attention distributions" and is fixed by per-head temperature.
- PENCIL (arXiv 2503.14337), 10.6M-parameter 6-layer models from scratch: on SAT with n = 10, plain CoT reaches 50%
  with ~13.8k-token contexts, while PENCIL's erased traces reach 100% with ~2.5k.

**Interpretation.**
- On 21–30-line problems, R's late actions need a copy from ROOT 2–5k tokens back. Only the 4 NoPE heads, plus one
  weak ALiBi head, can reach it, at a distance R never trained on.
- **Expected failure mode.** Rising per-step error with distance from ROOT. The model may also fall back to the
  no-plan behaviour it learned from the 20% emptied records.
- R as specified cannot answer "does thinking help on long proofs". At least 8% of the sealed set's references
  do not fit in context, and half or more need a ROOT longer than any it was trained to write.

**Fix or control.**
- Report per-step error by step index (or by distance to ROOT) against T1.
- Raising the context does not fix this, because the training lengths stay at K12's. Either:
  - add 13–20-line traced proofs to Stage-1 data (generator families, decontaminated), or
  - make R Markov: ROOT + current state, no history. That is the M-frozen arm of §6, and its context is bounded by
    640 + state.
- Use KV-cached sampling for R, or report forward tokens (§3, R6).

### R3. Exposure bias with no way to revise: plan-following is unconditional, and observations add only rejections

**Observed.**
- In the 80% traced records, the actions follow ROOT by construction. No training sequence has an action that
  departs from its plan.
- After ROOT, think tokens are banned (`mask_action_think = True` for R).
- Guided redraws at t > 0 resample the action only, so a wrong ROOT line cannot be redrawn mid-attempt.
- The environment is deterministic: each `state_t` is a function of the prompt and `action_0..t-1`. In R's
  transcript every observation is therefore redundant with the history. Only a rejection, in the guided arms,
  carries new information.

**Reported.**
- Bengio et al. (arXiv 1506.03099): "This discrepancy between training and inference can yield errors that can
  accumulate quickly along the generated sequence."
- Arora et al. (arXiv 2204.01171): exposure bias causes error accumulation that perplexity does not capture.
- ReAct (arXiv 2210.03629): reasoning interleaved with observations helps to "induce, track, and update action
  plans". ALFWorld best-of-6: ReAct 71% against Act-only 45%.
- Prover-V1.5 *truncates at the first Lean error* and resumes from the verified prefix with the Lean state as a
  comment.
- Prover-V2: under RL, "the structure of the generated proofs frequently diverges from the lemma decomposition
  provided by the chain-of-thought guidance". It needed a consistency reward.
- **Whole-proof provers that work at scale do not plan once and transcribe.**
  - Kimina-Prover (arXiv 2504.11354) interleaves informal reasoning with Lean snippets in a long CoT, with RL from
    Qwen2.5-72B: 80.7% on miniF2F at pass@8192.
  - Goedel-Prover (arXiv 2502.07640) has no long CoT and reaches 57.6% pass@32 by whole-proof generation and expert
    iteration.
  - Both start from pretrained 7B–72B language models.

**Interpretation.**
- R is a plan-then-execute agent, ReWOO rather than ReAct (arXiv 2305.18323; plan-first reported 5× token
  efficiency on HotpotQA). It works when plans are right.
- At test time ROOT is sampled at T 0.8. A wrong subgoal formula, wrong term, or wrong `<no>` is copied into the
  actions. The model has never seen a plan overruled, and has no way to emit a corrected plan.
- Prover-V2's divergence problem may not appear: R's actions were trained only as transcription. The opposite
  problem will: faithful execution of wrong plans.

**Fix or control.**
- Train re-plan blocks: allow `<think>` after a rejection or a box close (ReAct-style).
- Corrupt 10–20% of ROOT lines in training, and have a later re-plan block repair them.
- Then add DAgger-style rounds on model-sampled ROOTs.
- Measure the share of failed R attempts whose first wrong action matches a wrong ROOT line, against those that
  depart from a correct ROOT.

### R4. Names: the base mismatch (§1, the shared bug) and binder aliases

**Observed.**
- **Base mismatch:** §1, the shared bug.
- **Aliases.** 29.9% of ROOT binder occurrences use a name the proof never uses: an alias forced because the
  environment reuses names across sibling boxes. 44% of proofs have at least one alias.
- **Clashes,** where a plan name is used by the proof for a different formula: 0.5% (R), ~0% (M).

**Interpretation.**
- The aliases force the model to cite binders by formula lookup in the state, never by copying from ROOT.
- That is probably benign, and may even help, because it pushes the model to read the state.
- But it means ROOT's contexts (`n9 : A |- …`) name objects that do not exist in the environment.

**Fix.** Write binders in plans by formula, or as placeholders resolved at the box-opening step. That also removes
the base problem for binders.

### R5. The empty-frac design makes the budget knob a plan truncator and hides a no-plan policy inside R

**Observed.**
- 30,925 of 154,977 records (19.95%) are emptied: whole ROOT removed, by a hash of the record id
  (`rootplan.build`). This is fixed per record at build time.
- Partial ROOTs: 8 in all of training.

**Reported.** Dualformer (arXiv 2410.09918) on mazes keeps the full trace with p 0.45. Partial drops (close
clauses, cost tokens, 30% of create clauses) take 1/6 each, and the *whole* trace is dropped with p 0.05, resampled
every batch.

**Interpretation.**
1. **No-plan mass at free budget.** The emptying is independent of the prompt, so p(`</think>` first | state_0)
   should be learned as ≈ 0.2. At T 0.8 that is about 15% of "free-budget" R attempts with no plan. The arithmetic:
   0.2^1.25 / (0.2^1.25 + 0.8^1.25) = 0.15. This still has to be checked on a model.
2. **B < ROOT length truncates the plan.** That case is out of distribution. Any "more budget, more success" curve
   for R is then partly manufactured by damaging the low budgets: LIT_CRITIQUE C2, in a new form.
3. **R at B 0 is not a control.** It is a no-plan, history-conditioned policy trained on 20% of the data inside
   plan-dominated weights, and GRPO can collapse onto it (R7).

The project's design is Dualformer's Level 4 only, at 4× its maze rate, with no partial levels.

**Fix.**
- Use min-think 1 (or ban an immediate `</think>`) for "free" reads, or report the planless share.
- Replace record-level emptying with per-batch line-prefix truncation at line boundaries plus a small whole-drop
  rate.
- Run R-noplan as a separate model.

### R6. Compute and token fairness

**Observed.**

| | think tokens per proof (K12 / 12 lines) | prefill per attempt |
|---|---|---|
| T0 | 0 (actions 124) | 650 (K12) |
| R | ROOT 132 / 225 | re-prefill 4,571 (K12); with a KV cache it would be the transcript, 882 |
| T1 v2 | 238 / 442 | – |

**Interpretation.**
- On sampled tokens, R is cheap: about 2× T0 per attempt on K12, cheaper than T1. On forward tokens with the
  current sampler it is 6–17× T0.
- The matched-token headline must state which cost it uses, and should show both. The README's "sampled tokens"
  convention flatters R.
- Training is roughly fair on tokens: the recipe's token budget, with the same states and actions plus ROOT. But
  R's attention cost is quadratic in ~900-token sequences, against ~150-token pairs.

### R7. GRPO on R

**Observed** (lr-check runs, init T1v1_s0 with effort tokens, budgets {0, 16, 64, 256}, 8 iterations;
`runs/grpo/lrcheck_*/metrics.jsonl`).

| lr | pass@1 at e0, iteration 8 | at e256 | think tokens per attempt |
|---|---|---|---|
| 1.5e-4 | 75.2% | 71.0% | 138 → 32 |
| 5e-5 | 72.9% | 73.9% | 138 → 80 |

The think tokens per attempt are averaged over a curriculum-weighted budget mix, so part of the drop may be mix
rather than policy. Still, the no-think mode caught up with or passed the thinking mode on the RL pool within
8 iterations.

**Interpretation.**
- **Credit is clean for R.** Actions are near-deterministic given ROOT, so within a group the outcome variance sits
  mostly on ROOT. GRPO on R is close to whole-proof RL, as in Prover-V1.5's RLPAF. That is the strongest point
  in R's favour.
- **Three ways it goes wrong:**
  1. **Empty-ROOT collapse.** If the 15–20% planless mode does as well as planning on the ≤ 20-line pool, as e0
     did above, GRPO will amplify it.
  2. **Nothing to scale.** ROOT length is pinned to proof length, so RL cannot make it "think longer". The only
     elastic part is exploration (9.2% of ROOT tokens), which never changes the action, so RL should prune it.
  3. **The longest targets give no gradient.** 15% of 13–20-line references overflow the context; such attempts
     are masked as overlong. The longest RL targets are exactly those.
- RL on R will therefore improve in-distribution ROOT accuracy at best. It will not produce test-time scaling.

### R8. Missing controls for R

| control | isolates | priority |
|---|---|---|
| R-noplan: same data, 100% empty ROOT, full history | the plan, against the history format (states are redundant given the history) | must |
| oracle ROOT at test | ROOT generation against transcription | must (CPU-cheap read) |
| swap ROOT from another theorem with the same proof length; shuffled ROOT lines; filler ROOT (`intervene`) | plan-following, which should be strong and is expected | should |
| R-verbatim: the forward actions as ROOT | top-down order against "write it twice" | should |
| R-Markov: ROOT + current state, no history (= M-frozen) | history and length | must if long proofs are the target |
| per-step error by distance to ROOT | the position or length mechanism (R2) | must |
| base 0 against base U[0, 32] | the name bug | must |

## 4. Ranked critiques: M

### M1. The memo is a copy channel; the actions are a transcription of its head line

**Observed (§2).**
- At `exact` steps (33% of steps) the memo is `|- G := nK` and the action is `exact nK`: 100%.
- Premise-step memos (23%) never contain a line absent from the previous memo.
- At decision steps:
  - 94.8% of actions have a memo line with the identical formula and skeleton;
  - a fixed emission-order rule picks that line 87.7% of the time;
  - away from box closes, 96.6% of memo tokens repeat the previous memo.
- About 73% of M's sampled tokens per step are memo: 47 of 64.5 target tokens on average (`TM.pkl.report.json`).
- Over whole proofs, about 62% of memo tokens are copies of lines in the previous memo: 81% of the memo tokens after
  step 0, and step 0 holds 24% of all memo tokens.

**Reported.**
- Scratchpads (arXiv 2112.00114) and "CoT tokens are program variables" (arXiv 2505.04955) show that useful
  intermediate tokens hold *intermediate results*: carries, variable values, DP cells. In 2505.04955, "preserving
  only tokens that store intermediate results would achieve comparable performance", and editing a stored value
  changes later tokens and the answer accordingly.
- InftyThink (arXiv 2503.06692) carries a *summary of the model's own reasoning segments*, ≤ 4k-token segments
  compressed by Llama-3.3-70B. It reports +4.33 (MATH500) and +10.0 (AIME24) for Qwen2.5-Math-1.5B. It does not
  quantify error accumulation; its appendix notes that "poor summarization can lead to information loss".
- Gist tokens (arXiv 2304.08467) compress prompts up to 26× "with minimal loss in output quality". The Recurrent
  Memory Transformer (arXiv 2207.06881) passes memory tokens between segments. In both, the memory is a
  compression shaped by the downstream loss, not a supervised copy of a reference.

**Interpretation.**
- M's memo holds the *answer* (the remaining proof term), not intermediate results.
- It is not a compression: the memo is about 4× the action per step (60 against ~15 tokens).
- The "program variable" reading predicts that corrupted memo values propagate to later memos and actions (M4).
  That is faithfulness, but of a copy.

**Fix or control.**
- **M-frozen:** MEMO_0 carried unchanged; targets at t ≥ 1 are the action only. If M ≈ M-frozen on ≤ 20-line
  problems, rewriting adds nothing.
- **M-diff** (§6): loss only on new lines.

### M2. M forgets what the state does not show, and keeps what it does: PENCIL's reduction run backwards

**Observed.**
- `state_tokens()` shows every frame's hypotheses but *only the innermost goal*.
- MEMO_t is "the remaining plan of the current frame" (`memo_trace` → `frame_term(rec, idx)`). So on entering a
  box, the parent frame's remaining plan, which the state does not show, is dropped.
- On leaving the box, it is regenerated: 22.9% of decision steps (t ≥ 1) follow a box close, 97.2% of their memo
  tokens are new, and 92.4% of those had been written in an earlier memo.

**Reported.** PENCIL's rule `C [CALL] T [SEP] A [RETURN] ⇒ C A` erases the callee's intermediate work T and *keeps
the caller's context C*. Its traces come from running the actual search algorithm. Delethink (arXiv 2510.06557)
carries the last m = C/2 tokens and learns, by RL, to write "a textual state near the end of each chunk sufficient
for seamless continuation". It degrades where the needed state exceeds m: 14×14 crosswords.

**Interpretation.**
- M erases C (the outer plan) at CALL and recomputes it at RETURN, while carrying the callee's plan, which the state
  already anchors.
- For long, nested proofs (or_reorder, case products), every box exit is a fresh planning problem made from the
  state alone, the same problem T0 faces. The memo's memory is then worth nothing exactly where proofs are long.

**Fix.** An agenda memo: the stack of open goals across all frames, each with at most its planned decomposition.
Pop on close. That is the information the Markov state lacks.

### M3. Cost: about 2× T1's think, mostly copies; it loses matched-token reads before they start

**Observed.** Memo tokens per proof (reference traces):

| | K12 mean | 12 lines | 13–20 lines | 21–30 lines |
|---|---|---|---|---|
| M memo | 479 | 995 | 2,058 | 2,746 |
| T1 v2 think | 238 | 442 | – | – |
| action tokens | 124 | – | – | 497 |

M's prefill also re-reads the previous memo every step. Stage-1 T1 at B 256 used 320 tokens per attempt against
T0's 116, and was 4.7 pp behind T0 at 1k tokens per theorem (bootstrap p 0.06).

**Interpretation.**
- At 21–30 lines, a reference-length M attempt samples about 3.3k tokens: 2,746 memo, 497 action, plus closes.
  That is 6.6× T0's.
- On problems where T0's per-attempt success is p, M breaks even at matched sampled tokens only if its own success
  is ≥ 6.6p. That is not impossible where p ≈ 0, but it is a far higher bar than the per-step format already
  failed below 3k tokens per theorem.
- None of these tokens is search: exploration is 1.1%.

### M4. Exposure bias and drift: the model reads its own memos, the empty memo is absorbing, and cut memos get copied

**Observed.**
- **The empty memo is absorbing.** Non-emptied records never have an empty memo; emptied records have every memo
  empty. So in training an empty previous memo at t ≥ 1 is always followed by an empty memo.
- **Cut memos.** 2.75% of training memos are cut by the 256 cap, and 6.9% of step-0 memos (26% at 12 lines).
  At 21–30 lines, 100% of step-0 memos are cut.

**Reported.** InftyThink's prompt-only ablation (iterating without training) loses information across iterations.
MEM1 (arXiv 2506.15841) and MemAgent (arXiv 2507.02259) do make rewrite-every-turn memories work:
- MEM1-7B: 3.5× the performance of Qwen2.5-14B-Instruct with 3.7× less memory on 16-objective multi-hop QA;
- MemAgent: trained on 32K-token texts with chunked, overwritten memory, extrapolating to 3.5M-token QA with < 5%
  loss.

Both learn the memory end-to-end with RL against outcomes, not by imitating a reference copy.

**Interpretation.**
- **Memo errors compound.** Every memo token is re-sampled at T 0.8. At 2,746 memo tokens per 21–30-line proof, a
  per-token copy-error rate of 10⁻⁴ already corrupts 24% of attempts: 1 − (1 − 10⁻⁴)^2,746. That is arithmetic,
  not a measurement.
- **One empty memo ends the thinking.** A single empty memo (an early `</think>`, or B 0 at one step) turns the rest
  of the attempt into no-think. With the ~15% planless mass at step 0 (R5's arithmetic), about 15% of free-budget
  M attempts never think.
- **B-capping is a plan truncator again.** A memo cut at B is copied cut into the next step.

**Fix or control.**
- **Corruption read:** inject one wrong subgoal line at step k and measure persistence and repair.
- Train repair examples:
  - memos regenerated from the state after a corrupted or empty previous memo;
  - partial-empty transitions;
  - DAgger rounds on model-sampled memos.

### M5. Names

**Observed.** 33.0% of memo binder occurrences are aliases; 44% of proofs have at least one. The base mismatch (§1)
hits MEMO_0 → MEMO_1 in 74% of proofs, those whose MEMO_0 contains `n`-names.

**Interpretation.** It is milder than in R. M rewrites the memo every step after reading the state, so it can
re-sync to the state's names. But it has never seen the conflict.

### M6. GRPO on M

**Interpretation.**
- **Dilution.** The loss is a token-level mean. About 73% of M's sampled tokens are memo, mostly near-deterministic
  copies. They give little gradient but dilute the action tokens' share of each update.
- **Shrinkage.** Dropping memo lines beyond the next emission costs nothing at the current step. So the GRPO
  direction is toward shorter memos, the "name the next line" ritual that T1v1 already is. That is consistent with
  the think shrinkage in the lr-check runs, though not proven by it.
- **Credit.** A wrong decision at step 0 is paid for at step 20, and the proof-level advantage cannot say where.
- **Only two places to learn.** Unlike R, M has no single plan block to receive clean credit. The planning that
  matters, at step 0 and box exits, is a small share of the tokens.

### M7. Missing controls for M

| control | isolates | priority |
|---|---|---|
| M-frozen (MEMO_0 carried, no rewriting; = R-Markov) | rewriting | must |
| oracle memo at every step | memo generation against transcription | must |
| corrupted memo at step k; empty memo at step k | persistence, repair, the absorbing empty state | must |
| agenda memo (open goals across frames) | M2's dropped information | should |
| M against T1 at matched sampled *and* forward tokens | cost | must |
| base 0 against U[0, 32] | the name bug | must |

## 5. Falsifiable predictions

Unless stated otherwise:
- tb72 with 3 seeds per arm;
- "resolved" as in STAGE1_STATS: both the bootstrap and the t-check interval exclude 0;
- MDE at pass@1: about 4.5 pp between arms and about 2.2 pp within a model.

The 13–20-line held-out set with references is not yet built; it is needed for the oracle reads.

**Stage 1**

| # | prediction | falsified if |
|---|---|---|
| P1 | Dependence. R free-B − R B 0 and M free − M B 0 are ≥ +4 pp at pass@1, both resolved. | either is unresolved |
| P2 | Plan-following. Swapping in a foreign ROOT or memo puts pass@1 *below* B 0 by ≥ 2 pp. | swap ≥ B 0 |
| P3 | Oracle. The reference ROOT inserted at test gives ≥ 90% pass@1 on held-out ≤ 12-line K12. The strict validity rate of sampled ROOTs is within 5 pp of R's pass@1: plan generation is the bottleneck. | oracle ROOT < 80% |
| P4 | Against T0. R − T0 and M − T0 at tb72 pass@1 and pass@5 are unresolved, inside ±4.5 pp. | either is resolved positive |
| P5 | Matched tokens. M is below T1 at every matched sampled-token budget ≤ 10k per theorem on tb72. R is within noise of T0 at ≤ 3k sampled tokens, and below it at matched forward tokens. | M ≥ T1 at any budget ≤ 10k |
| P6 | Name base. R and M pass@1 at base 0 exceed their pass@1 at base U[0, 32] by ≥ 3 pp; T1 changes by < 1 pp. If R/M also change by < 1 pp, the model reads names from the state, which is informative too. | R/M change < 1 pp |
| P7 | R distance effect. On 21+-line problems, among attempts that reach step 15, R's per-step rejection rate at steps ≥ 15 is ≥ 1.5× its rate at steps < 15. T1's ratio is < 1.2. | R's ratio < 1.2 |
| P8 | M persistence. A single corrupted subgoal line injected at step k reappears in ≥ 70% of later memos until its frame closes, and the action that consumes it is wrong in ≥ 50%. | persistence < 30% |
| P9 | M-frozen within ±2 pp of M at pass@1 on ≤ 12-line problems. | M − M-frozen ≥ 3 pp, resolved within seeds |
| P10 | Planless mass. Without min-think, 10–20% of free-budget R attempts have an empty ROOT, and 10–20% of M attempts have an empty step-0 memo and every later memo empty. | < 5% |
| P11 | Long proofs. On Charles 21+ and the sealed 21–30 set, neither R nor M beats T0 at pass@16 by a resolved margin, and neither beats T0-guided at matched sampled tokens. | either does |

**RL (GRPO, proof-level reward)**

| # | prediction | falsified if |
|---|---|---|
| G1 | R's empty-ROOT share at free budget moves toward whichever mode has the higher initial pass@1 on the pool. If B 0 ≥ free at init, it exceeds 50% by iteration 20. | the empty share stays within ±5 pp while B 0 ≥ free |
| G2 | M's memo tokens per step fall ≥ 50% within 10 iterations, mainly by dropping lines after the next emission; "action skeleton in memo" stays ≥ 90%. | memo tokens fall < 20% |
| G3 | ROOT exploration (`<no>` lines) falls ≥ 50% within 10 iterations. | it rises |
| G4 | In successful R attempts, ROOT–action skeleton consistency stays ≥ 95%. This is the opposite of Prover-V2's divergence: here actions were trained only as transcription. | < 85% |
| G5 | No GRPO arm of R or M shows a resolved positive slope of pass@1 against B at matched sampled tokens on the sealed 21–30 set. | any does |

## 6. Recommendations

### Go or no-go

- **R: no-go** as the test-time-scaling or long-proof vehicle. **Conditional go** as one cheap diagnostic, because R
  is the cleanest available test of a different question: "is the proof easier to write top-down than as steps?"
- **M: no-go as designed.** Redesign it (below) before spending GPU on it. Its Markov shape is the right one for
  long proofs; its content is not.

### First, CPU only (hours)

1. **Fix the name base for R and M.** Set `env.base` from the model's first emitted name, or read at base 0 and say
   so. Add a test with `NAME_BASE_MAX = 32`.
2. **Handle the planless mass at free budget.** Set min-think 1, or report the planless share separately.
3. **Report the cost properly.** Use KV-cached R sampling, or report forward tokens next to sampled tokens.
4. **Build a 13–20-line held-out set with reference proofs** from the generator families (decontaminated), for
   the oracle and corruption reads. It also serves LIT_CRITIQUE C5's near-OOD band.

### The minimal experiment that decides it

A 2 × 2 that separates plan from history. The existing T0 (3 seeds) is one cell:

| | Markov (state only) | history (full transcript) |
|---|---|---|
| no plan | T0 (have) | R-noplan |
| plan, written once | M-frozen = R-Markov | R |

Plus M (rewritten memo) as a fifth arm.

- **Runs.** 2 seeds each is 8 new Stage-1 runs. That is an estimate of about 1–2 h each on a 3090; R's longer
  sequences will be slower than T1's 41–60 min.
- **Reads.**
  - tb72 and the 13–20-line set, plus Charles's long problems;
  - pass@1 and pass@5;
  - matched sampled and forward tokens;
  - interventions on each planned arm: empty, oracle, swap, corrupt-at-k;
  - per-step error by step index.

**Decision rules.** Each is fixed in advance. All comparisons are at matched sampled tokens, and on the 13–20-line
set unless noted.
- **A plan helps** if both of these exceed the between-arm MDE (about 5 pp):
  - R − R-noplan at pass@1;
  - M-frozen − T0 at pass@1.

  If neither does, stop both formats.
- **Rewriting helps** if M − M-frozen ≥ 3 pp at pass@1 and memo corruption does not persist (P8 falsified).
  Otherwise drop rewriting.
- **History helps** if R − M-frozen > 0 (resolved) and R's per-step error does not rise with distance to ROOT.
  Otherwise use R-Markov.
- **Test-time scaling** is claimed only if a planned arm beats T0-guided at matched tokens on the long set.
  LIT_CRITIQUE C1's falsifier still applies.

If GPU is tight, run R and M-frozen at 1 seed each first. Their within-model reads (B 0, oracle, swap, corrupt)
have about a 2.2 pp MDE and answer "is the plan generation or the transcription the bottleneck" on their own.

### Design changes that would make each worth running

**R → sketch-R.**
1. ROOT keeps the goal reductions *without terms*: the `ctx |- F` subgoal heads, with holes only. That is
   Prover-V2's `have … := by sorry` and DSP's sketch. Each step must then find its own term, so actions stop being
   copies and the plan supplies only lookahead.
2. Add re-plan blocks after rejections and box exits (ReAct), trained with corrupted ROOT lines and repairs.
3. Add per-batch line-prefix truncation (Dualformer Levels 1–3 style) in place of record emptying, so that B < ROOT
   length is in distribution.
4. Make it Markov (ROOT + state), or train on 13–20-line proofs.
5. Write binders by formula.

**M → agenda-M.**
1. The memo is the stack of open goals across frames. That is exactly what the Markov state lacks. Each goal carries
   at most its planned decomposition, and the head is popped on close.
2. Loss on edits only. Carry unchanged lines as observation tokens: the sampler applies "pop head / substitute
   name". This removes the copy cost: about 62% of memo tokens, and 81% of those after step 0.
3. Add repair data: memos regenerated after corrupted or empty predecessors, so that empty is not absorbing.
4. Then RL with the memo learned against outcomes, as in MEM1 and MemAgent, instead of imitation of a reference copy.

**Both.** The think content must vary with *difficulty*, not with proof length, or RL has nothing to scale. That
means the model's own dead ends (SoS, arXiv 2404.03683: +25% from search streams over optimal paths), not
generator-chosen alternatives at 1–9% of think tokens.

## 7. How the numbers were computed

These are ad-hoc scripts in `/tmp/rmcrit/` (not committed, CPU only). All read stored files.

- **Predictability and novelty.**
  - Source: a 1-in-40 sample of `root_traces.jsonl` and `memo_traces.jsonl`.
  - Plan lines are parsed as `ctx |- F := RHS` / `… <no>`.
  - Holes `|- X` and names `h\d+` / `n\d+` are abstracted to one symbol.
  - Action terms are normalised: a box `( fun ( x : A ) => by` becomes `fun ( x : A ) => ·`; `Or.elim … by` is
    matched as a prefix; `Classical.byContradiction ( fun hh => n hh )` becomes `byContradiction n`.
  - Step kinds: `exact`; premise (`:= hK`); otherwise decision.
  - The emission-order rule takes the memo's anchor line (the first line for the state's goal that is not an
    exploration head), descends into the first hole until it reaches a box (`fun`), an `Or.elim` on a name, or a
    line without holes, and compares that line with the action.
- **Names.**
  - Lambda binders are matched against the binders and `have` names the actions introduce.
  - Cited names in decision actions are aligned token by token with the matched ROOT line.
- **Long proofs.**
  - `rootplan.root_trace` and `memo_trace` were run (default `PCfg`) on the `reference_proof` of the five
    `ood_check/sets` files.
  - ND lines are counted from the reference proof.
  - Re-prefill sums the transcript length at each step.
  - K12 step-0 memo truncation comes from `memo_trace` on a 1-in-50 sample of `train_k12.jsonl`.
- **R prefill on K12.** Read from `runs/data/TR.pkl`, a 1-in-10 sample, with the same logic as
  `runs/smoke_rm/prefill_cost.py`.
- **Stage-1 and GRPO figures.** Taken from [STAGE1_STATS.md](STAGE1_STATS.md) and
  `runs/grpo/lrcheck_{5e-5,1.5e-4}/metrics.jsonl`.
- **Literature.** Read from the arXiv abstract and HTML pages on 2026-10-06. Numbers quoted from LIT_CRITIQUE (Snell,
  s1) were not re-read.

**Checks run.** Every number in §2 was recomputed at least twice with different scripts. Two examples:
- R skeleton coverage: the 552 formula-absent and 475 skeleton-miss cases are all reiterations.
- M novelty: split into after-close and not-after-close steps.

This note adds no code and runs no tests. It is a documentation-only change; its links are relative to this folder.
