# Finetuning plan: measuring elicitation against new capability with fine-tunes of the RL models

For Claude Code agents working for Charles. Standing rules (git, Python, experiment folders, when
to stop and ask) are in the root `CLAUDE.md`. Week of Oct 10 to Oct 17, 2026. A third workstream
next to `textbook-problem` (the yardstick) and `pretrain-data` (the Stage-1 lever); neither is
changed here. Items marked **(decide)** are Charles's call; until he answers, use the stated
default and record it in `docs/finetuning/questions.md`.

## 1. Scope

> Take the fixed Stage-1 base and the RL checkpoints that exist (Dan's EI ladder and GRPO runs,
> rounds 1 to 8, 3 seeds) and measure how much of what RL added was already in the base
> (elicitation) and how much RL wrote into the weights (new capability), as a function of the
> amount of RL and of the pass@k curve. Fine-tuning is the measuring instrument here, never a
> way to make a better prover: nothing fine-tuned in this workstream is reported as a model.

**The elicitation problem, as the project states it** (`docs/project_strategy/big_picture/big_picture_notes.md`
question 2; group meeting 2026-09-20): Dmitry expects the first effect of RL to be close to
base-model recovery, and new capability to appear only after some point, a crossover. The
project's write-up wants to show that crossover and say when it occurs. Nothing so far measures
it: `pass@k` at large k (Yue et al. 2025) is the only instrument in use, and Charles ruled on
2026-09-27 that large-k pass@k cannot carry a capability claim on its own. This workstream builds
the instruments and takes the first readings on the checkpoints we already have.

**Two ideas from Charles (2026-10-10) are the spine of the plan.**
1. **Random-subset fine-tunes.** Fine-tune only d randomly chosen parameters of the base on
   correct proofs and ask how much of the base-to-RL gap that closes, for d from 10 to all. The
   motivation is the description-length picture: a gap that a hundred parameters close was a
   setting the base already had; one that needs the whole network was written in. Do it on the
   RL model too, and watch how the measurement changes with the amount of RL, EI or GRPO.
2. **Excess description length against the pass@k curve.** Measure EDL (Donoway, Joren, Roger
   and Leike 2026, arXiv 2601.04728; the literature note is `docs/literature/astra_suggestions/
   learning-beyond-training-data.md` § 7) by prequential coding, and plot it against the area
   under the pass@k curve (k to 64 at least) for every checkpoint along the RL trajectory.

Section 3 adds five further measurements that need no training at all and can be read this
weekend on the laptop, and section 4 turns all of them into experiments with numbers to beat.

**In scope.** Fine-tunes of existing checkpoints under parameter masks, prequential-coding
measurements, log-probability and weight-space analyses, pass@k curves from Dan's k = 256 dumps
and from our own sampling, and a small-model replica on the pinned 3.2M checkpoint. **Out of
scope.** Any new Stage-1 or RL training run (those belong to `pretrain-data` and to Dan); editing
pinned or vendored code; selecting anything on a test split or on the 72; publishing a
fine-tuned checkpoint as a result. If one of these blocks progress, stop and ask.

## 2. What exists today (surveyed 2026-10-10; paths in `notes.md`)

**Subject family (primary): Dan's best-cap12 line.** 9,560,832 parameters (`best_model.py`,
6 × 384 Peri-LN, ALiBi), `lean_staten` proof-state format, Stage-1 on K12 (155,000 proofs,
1,200 s on an A40), Lean as the in-loop judge. All in `hf://buckets/dan-pandori/nd-rl`
(public listing, verified today), three seeds each:

| run | checkpoints | what they are |
| --- | --- | --- |
| `best-state/ckpts/bs/` | `stage1_best12_s{0,1,2}_b1200.pt`; `ladder/la_T1_best12_s*_r{1..8}.pt` | the base and the T1 expert-iteration ladder, every round |
| `trajectory/ckpts/tj/ladder/` | `la_T1_best12_s*_r{1..8}.pt` | the ladder re-run from the same bases (the "EI" arm of grpo-best) |
| `grpo-best/ckpts/gb/` | `gb_{default,unlikely,passk,distinct}_s*_r{1..8}.pt` | GRPO in the state environment, round-equivalent budgets, four advantage variants |
| `rl-from-ckpt/ckpts/rfc/` | ladders from Stage-1 starts at 1,600 / 5,000 / 12,000 / 16,000 / end steps, `r1..r8`, and the replay-only `control/rc_*` | RL as a function of the amount of pretraining; the control trains on K12 replay alone |

Read-outs Dan already published for these (Lean alone; `grpo-best/artifacts/gb/analysis_stdout.txt`,
`rl-from-ckpt/artifacts/rfc/analysis_stdout.txt`): per-problem dumps at **k = 256, T 0.8** on the
72 and on a 250-theorem held-out set (`n_ok`, `n_tried`, proofs, per row), and on the 322
`rl_targets` split into **group A** (base solves), **group B** (base pass@1 0.001, pass@256
0.145) and **group C** (never solved by the base nor by EI r8 at k 256, sample seed 0; 36 / 35 /
28 theorems per seed). Headline facts we build on:

- EI r8 takes group B from pass@1 0.001 to 0.564 and pass@256 0.145 to 0.964, and leaves group
  C at pass@256 0.049. GRPO with the pass@k advantage (Chen et al.) reaches 0.272 on group C
  (+7.7 theorems over EI, outside the MDD) with the same textbook72 count (50.7 vs 50.3).
- The replay-only control (K12 fine-tuning, no RL samples) reaches 31 to 39 textbook72 problems
  against the ladder's 37 to 50. From the early Stage-1 starts (1,600 to 16,000 steps) the
  control gets a third to a half of the ladder's gain; from the end start almost none (34.7 vs
  49.7, base ≈ 33). Part of "RL's" textbook gain on an under-trained base is just more training.
- `rl-from-ckpt` already has one base-support measurement: the start checkpoint's worst-step
  log-probability of the reference proof predicts whether the ladder solves the theorem by r8
  (logistic fit, x50 ≈ −11 nats). We extend it from one step of the reference to whole proofs,
  RL's own proofs, and bits (section 3, M4).
- Our own scoring of `la_T1_best12_s{0,1}_r8` with the pinned verifier
  (`experiments/textbook-problem/eval_dan_best_state/results.md`): 183 of 269 scraped textbook
  problems solved by either RL seed against 44 by the pretrains; RL solves everything the
  pretrains solve (containment holds); all 7 four-premise problems at 0.

**Subject family (secondary, CPU only): the pinned 3.2M model** (`code/ws3/fork_pin/`, `abs`
whole-proof format) with the ws3 harness (`code/ws3/rl/ei.py`, TRL GRPO) and exp-00c's four
arms (`experiments/ws3/exp-00c-lr-ei-cpu/`). Lets the whole measurement be replicated at a
second scale on the laptop, with RL we ran ourselves.

**Correct proofs to fine-tune on.** Every record is verifier-checked:
- RL's own found proofs on `rl_targets`, per round (`found_<r>.jsonl` in each ladder's
  artifacts; GRPO's `found_8.jsonl`), already in the state format;
- the textbook **seed split** (`docs/pretrain-data/plan.md` § 5: candidate_v0 87 + the
  Lean-sourced 47, reference proofs, atom permutations allowed), rendered to `lean_staten` by
  `state_env.canonicalise`; the **test split** (candidate_v1, batch 3, hand-proved 14, derived
  rows of test bases) is scored, never trained on; the 72 are scored in aggregate only;
- generator problems at lengths 13 to 16 with `minlen` proofs (`pretrain-data`'s `k12_long16`
  slice), content the base never saw.

**Tools.** Dan's `state_train.py` (fine-tune), `state_sample.env_generate` and `state_eval.py`
(sampling), `state_env.decompose` (proof to (state, action) pairs, replays byte for byte), all
used unmodified from the worktree at `~/.cache/nd-rl/fork_dan` (branch `dan`, commit
`0c492902`), pinned by a `MANIFEST.json` in `code/finetuning/` the way `textbook_problem/lean_tools`
pins Dan's Lean tools. Our evaluator (`code/ws3/eval/evaluator.py`) has `pass_at_k_curve`
(Chen et al. unbiased estimator) and `k50`. Re-judging of Lean verdicts with the pinned verifier:
`code/textbook_problem/dan_metrics.py`.

**What the literature says to expect.** Yue et al. 2025: RL raises pass@1 and can lower large-k
pass@k; ProRL (Liu et al. 2025): long RL with exploration kept up expands coverage. Donoway et
al. 2026: EDL = prequential codelength − n·L_test(θ*); random labels give EDL 0; elicitation
shows EDL per example falling monotonically with n and of order 0.01 to 0.1 bits per parameter,
teaching above 1 bit per parameter and EDL per example that can rise; score answer tokens only;
hold the training algorithm fixed across comparisons (their Prop. 4.9); population monotonicity
can fail with large learning rates. Li et al. 2018 and Aghajanyan et al. 2020: random
subspaces of a few hundred to a few thousand dimensions recover 90 % of full fine-tuning on
pretrained networks (intrinsic dimension). Mukherjee et al. 2025 and Zhu et al. 2025: RL
updates in LLMs are sparse and sit off the weight matrices' principal directions, SFT updates
do not. None of this has been measured on a model whose entire training corpus is known and
whose task has an exact verifier, which is what this setting adds.

## 3. Measurements

> **Revision 2026-10-10 (evening).** After the literature pass and the survey of every branch
> (`literature_hypotheses.md`), M4 and M6 turned out to be largely done elsewhere (Dan's
> `novelty`, `lit-measures`, `capability-defs`; Leon's `benchmarking` steering null) and Leon's
> EDL on the small model was confounded by starting loss. The ranked list in that file promotes
> M1 and M5 (parameter count and weight-space anatomy, which nobody has measured), adds
> single-example teachability per family and RL-delta extrapolation, keeps M2 only with a
> starting-loss control, reduces M4 to its per-round class histogram, and drops M6. The
> measurement definitions below stand; the order of work follows the ranking.

Each measurement names what it reads, what it costs, and what elicitation and new capability
look like under it. M1 and M2 are Charles's two ideas; M3 is the pass@k area they are plotted
against; M4 to M7 are new and M4, M5 are zero-training.

**Definitions used throughout.** Base = `stage1_best12_s<seed>` (and the matching Stage-1
starts in `rl-from-ckpt`). θ_r = the RL checkpoint after round r. For a problem set S and a
checkpoint θ, `pass@k(θ)` is the mean over S of the unbiased estimator from n = 256 samples
(Dan's dumps) or n = 32 (ours). The **gap** G_r(S) = problems in S with θ_r rate ≥ 0.5 and base
0 of 256 (the `CLAUDE.md` new-capability definition). "Bits" are log₂; every log-prob is
teacher-forced over action tokens only (the state tokens are given by the environment).

**M1. Random-subset fine-tunes (intrinsic dimension of the gap).** Freeze every parameter
except a random set of d scalars (d ∈ {10, 100, 1 000, 10 000, 100 000, all 9.56 M}; three
draws per d), fine-tune the base on a set D of correct proofs with Dan's ladder fine-tune
hyper-parameters (`state_train.py`, lr 3e-4 → 3e-5, warmup 50, 128 proofs per step; steps fixed
in the proposal after a pilot), and score on a **disjoint** evaluation set. Two reads:
(a) *transfer*: D = proofs of half of G_8 by renaming class, score the other half; (b) *in-set*:
score the problems whose proofs were in D (memorisation capacity, the upper line). Report the
fraction of RL's gain in avg solve rate recovered as a function of d, and **d₉₀**, the smallest
d recovering 90 %. Repeat with θ_0 = θ_2, θ_4 (EI and GRPO) to see whether the dimension left to
close falls with the amount of RL. Two variants, cheaper and more expensive than coordinates:
output-head and bias terms only (≈ 40 k parameters: the "format" learner), and a dense random
subspace of dimension d (Li et al.; optional, only if coordinates at d ≤ 10 000 close nothing).
*Elicitation:* d₉₀ ≤ 10³ and the transfer read tracks the in-set read. *New capability:* d₉₀
grows toward the full network, or transfer stays far below in-set at every d.

**M2. Excess description length by prequential coding.** For a stream D of N proofs in a fixed
random order, blocks at n_j = 2^j (j = 0 … J, N ≈ 1 000 to 2 000, 80 / 20 split by renaming
class into stream and held-out), retrain from θ_0 on the first n_j proofs with the same
algorithm A for every θ_0 and D, and code block j + 1 under the model trained on blocks ≤ j:
`EDL(D, θ_0, A) = Σ_j codelength(block j+1 | θ_j) − N · L_test(θ_J)`, action tokens only, in
bits. Report EDL, EDL per proof, EDL per trainable parameter (for the M1 masks), and the
per-block curve (bits per proof against n), which is where the elicitation signature lives
(monotone fall) and the teaching signature (rise, then fall). Streams: **D_RL(r)** = RL's own
found proofs at round r restricted to groups B ∪ C (what RL discovered); **D_seed** = textbook
seed-split reference proofs; **D_gen16** = generator proofs at 13 to 16 lines; **D_random** =
D_RL's prompts with proofs of other theorems (expected EDL ≈ 0, a gate); **D_wrong** = K12 proofs
with one verifier-rejected step (`pretrain-data` E14 pool): format without correctness. Learners
θ_0: the base; θ_2, θ_4, θ_8 of EI and of GRPO default and pass@k; the replay-only control r8;
and a from-scratch initialisation (the reference for "bits the pretraining already paid").
*Elicitation:* EDL(base → D_RL(8)) of a few bits per proof, consistent with the log₂ of the
attempts RL spent per solve (8 rounds × 32 = 256 attempts → ≤ 8 bits), falling with n;
EDL(θ_r → D_RL(8)) falling to ≈ 0 by r = 8 (consistency). *New capability:* tens of bits per
proof on D_RL's group-C part, a rising per-block curve, and a from-scratch learner that is not
much worse than the base on that part.

**M3. Pass@k area and its decomposition.** `A_K(θ) = (1 / log₂K) Σ_{j<log₂K} ½(pass@2^j + pass@2^{j+1})`,
the trapezoid area under mean pass@k in log₂k, K = 256 on Dan's dumps and 32 on our reads,
also at K = 64 as Charles asked, on: the test split, the 72 (aggregate), `rl_targets` groups A,
B, C, and the 250 held-out. For each θ_r: `ΔA = A(θ_r) − A(base)`, split into **sharpening**
(the part on problems the base solves at k = 256) and **expansion** (the part on problems at
0 of 256 under the base), and **k\*** = the base's sample budget whose pass@k equals θ_r's
pass@1 (sharpening in sample units; k\* ≤ 256 means the base at budget 256 matches RL's
greedy-ish behaviour). Cost: CPU minutes from the dumps, plus one pinned-verifier re-judge of
each dump (`dan_metrics.py`). This is the x-axis for Charles's EDL-against-area plot.

**M4. Base support of RL's proofs (new; zero training).** For every proof RL found (D_RL(8),
and per round), the teacher-forced log₂ p_base(proof | theorem) and the per-action minimum, and
the same under θ_r. 1 / p_base is an upper bound on the samples the base needs to produce *that*
proof, so each solved problem gets a budget class: **within budget** (log₂p ≥ −8: the base could
have sampled it in RL's 256 attempts per target), **latent** (−8 > log₂p ≥ −40: reachable by a
large but finite search), **created** (log₂p < −40, or an action the base assigns < 2⁻²⁰).
`log₂ p_θr − log₂ p_base` is the bits RL spent on that problem; summed over the gap it is an
on-policy counterpart of M2 that costs nothing. Also for every textbook reference proof (seed and
test split, the 72 in aggregate) under base and θ_r. Cost: ≈ 6 000 proofs × 2 checkpoints × 3
seeds, laptop CPU, hours. *Elicitation:* most of the gap in the within-budget and latent classes,
and the gap's class histogram unchanged from r = 2 to r = 8. *New capability:* a growing
created class, and it is where GRPO pass@k's extra group-C solves sit.

**M5. Weight-delta anatomy (new; zero training).** For Δ_r = θ_r − base per round, per arm, per
seed: ‖Δ‖ by matrix; the fraction of parameters with |Δ| above 10⁻⁵, 10⁻⁴, 10⁻³ (sparsity);
effective rank of each Δ matrix (participation ratio of its singular values); the share of ‖Δ‖
inside the top-32 principal directions of the base's matrix (Zhu et al.); KL(base ‖ θ_r) and
KL(θ_r ‖ base) over held-out states; and the same for the replay-only control. Cost: minutes.
Prediction (Mukherjee, Zhu): GRPO deltas sparser and more off-principal than EI's at equal
solve gain, the replay control's the least sparse; M1's d₉₀ should be smaller for the arm whose
delta is sparser.

**M6. Activation steering (dropped 2026-10-10 evening: Leon's `benchmarking` found adding the RL − base mean activation difference to the base never beats the base; kept here for the record).** The mean difference of residual-stream
activations at the `<act>` position between θ_8 and the base, per layer, over held-out states;
add α times that vector to the base at inference and read the gap. One direction that recovers
most of the gap is elicitation of a mode; nothing recovered at any α and layer says the change
is distributed. Cost: CPU hours. Only if M1 shows d₉₀ ≤ 10³, where a single direction is plausible.

**M7. Sample complexity of recovery (new).** With full fine-tuning and the fixed algorithm,
the number of correct proofs n_ε the base must see to reach θ_8's avg solve rate on a held-out
half of the gap, per problem family (4-premise, ∨E, depth-3, 13+ lines, hypothetical
syllogism; the `textbook-problem` tags). Read from M2's block curve, so it costs nothing extra
beyond evaluating each θ_j. Families where n_ε is tens are elicitation; families where it is not
reached within the stream are the candidates for the crossover and go to `pretrain-data` as
pool targets.

**The crossover, stated so it can fail.** Let r_c be the first round at which the newly solved
problems (G_r \ G_{r−1}) have a median M4 class of *created* and an M2 per-proof EDL above 20
bits. H1: for EI, r_c does not exist within 8 rounds (EI only ever trains on what it sampled, and
its group-C pass@256 stays at 0.05). H2: for GRPO pass@k, r_c ∈ {4 … 8} and the extra group-C
solves are the created class. H3: across Stage-1 starts (`rl-from-ckpt`), the share of the gap
in the created class falls with pretraining steps (more pretraining, more elicitation). If every
arm's gap is within-budget or latent, the honest conclusion is that nothing in this setting has
shown new capability yet, and the write-up says so.

## 4. Experiments

Everything held fixed: Dan's checkpoints (never retrained; fine-tuned copies live under our run
folders and are deleted or bucketed, never committed); his `state_train.py` fine-tune
hyper-parameters as the algorithm A of every measurement; sampling at T 1.0, k 32, 48 steps,
256 tokens per action through `state_sample.env_generate` with every proof re-judged by the
pinned verifier (the 2026-10-02 protocol), or Dan's k = 256 T 0.8 dumps re-judged, never mixed
in one table; the seed / test split of `pretrain-data` § 5. The only things that vary are the
checkpoint, the mask d, and the data stream.

| exp | measurement | checkpoints | question | seeds |
| --- | --- | --- | --- | --- |
| F0 `support_anatomy` | M3, M4, M5 from the bucket dumps and checkpoints; CPU only | base, EI r1–r8, GRPO default and pass@k r2/r4/r8, control r8; Stage-1 starts × r8 | where does RL's gain sit on the pass@k curve, how far outside the base's support are its proofs, what do the deltas look like | 3 (exist) |
| F1 `subset_ft` | M1 on D = half of G_8's RL proofs (transfer and in-set), six d × three draws | base; then θ_2, θ_4 EI and GRPO | how many parameters close the gap, and does the number fall with RL | 3 |
| F2 `edl_trajectory` | M2 on D_RL(8), D_seed, D_gen16, D_random, D_wrong | base, EI r2/r4/r8, GRPO default and pass@k r2/r4/r8, control r8, from-scratch | EDL against round and against M3's area; the per-block signature per stream | 3 |
| F3 `edl_vs_area` | the plot: every (checkpoint, seed) as a point, x = A_256 on the test split and on groups B / C, y = EDL(θ → D_RL(8)) and EDL(base → found_r); plus ΔA's sharpening / expansion split against M4's class histogram | all of F0–F2 | does EDL fall as area rises (elicitation), or does created mass appear where area grows at large k (new capability) | 3 |
| F4 `family_recovery` | M7 by family from F2's checkpoints, scored on the test split by tag | base → D_seed and D_RL(8) | which families are elicited, which are not reached | 3 |
| F5 `small_replica` | M1, M3, M4, M5 on the pinned 3.2M model with exp-00c's EI and GRPO arms, CPU | ws3 arms (re-run to 3 seeds if their checkpoints are not on the bucket) | does the picture hold at a second scale with RL we ran | 3 |
| F6 `steering` (phase B) | M6 | base, EI r8, GRPO pass@k r8 | is the elicited part one direction | 3 |

**Phase A** = F0 on the laptop (this weekend, no spend), then F1 and F2 on one seed each as
pilots to fix the step counts and confirm the gates. **Phase B** = F1, F2, F4 at 3 seeds, F3,
then F5 and F6 as budget allows. Only phase-B numbers are compared; one-seed phase-A numbers
are signs of life.

**Pre-registered expectations (to be fixed numerically in each proposal before launch).**
F0: ≥ 60 % of EI r8's group-B gap in the within-budget or latent class (median log₂p_base
between −8 and −30); GRPO pass@k's extra group-C solves mostly created (log₂p_base < −40);
sharpening ≥ 70 % of ΔA_256 for EI on every set; GRPO deltas sparser than EI's at τ = 10⁻⁴ by
at least 2×. F1: d₉₀ ≤ 10⁴ for the group-B transfer read with the base as θ_0, and d₉₀ falling
by at least 10× from θ_0 = base to θ_0 = θ_4; the in-set read at d = 100 recovering ≥ 50 % (a
few parameters can memorise a few proofs, which is why the transfer read is the one that counts).
F2: EDL(base → D_RL(8)) between 2 and 20 bits per proof, falling per block; EDL(base → D_gen16)
≥ 3× that per proof with a rising first half; EDL(θ_8 → D_RL(8)) ≤ 1 bit per proof; D_random
within noise of 0; EDL(from-scratch → D_RL(8)) ≥ 10× the base's. F3: a monotone negative
relation between EDL(θ_r → D_RL(8)) and A_256 along each arm; GRPO pass@k's points off that
curve on group C if H2 holds.

## 5. Splits and leak rules

- Fine-tuning data is only ever: RL's found proofs on `rl_targets` (generator theorems), the
  textbook **seed split**, generator pools. The **test split**, the 72, `validation_36`, the
  test files and `heldout` are scored, never trained on, never used to choose d, steps, or a
  checkpoint. A fine-tuned model is never "the best one on the test split"; every d and every
  block is reported.
- Every stream is audited by curry-aware renaming class (`release_1002.BR.keys`,
  `canonical_theorem` second opinion) against every evaluation set before use; twins are removed
  and counted (`rl_targets` has at most 3 twins of any textbook set, `eval_dan_best_state/
  run_best12_r8/leak_audit.md`). The audit file lives in the experiment folder with the stream's hash.
- Dan's groups A / B / C are taken as he defined them (sample seed 0, k 256) and said so; our
  within-budget / latent / created classes are new and defined in M4 before any number is read.
- The 72: aggregate counts only, in the gitignored `docs/textbook-problem/textbook72/`; M4 on
  their reference proofs reports class counts, never a per-problem value.

## 6. Gates

- **G0 (before any fine-tune, CPU):** the code passes its checks: (i) `decompose` replays every
  proof in a stream byte for byte and the teacher-forced log-prob equals the sum over actions;
  (ii) M3 reproduces Dan's published pass@1 and pass@256 for three arms from his dumps to 3
  decimals; (iii) EDL of D_random is within noise of 0 and EDL of K12's own val head under the
  base is ≈ 0; (iv) the d = all mask reproduces an unmasked fine-tune's loss curve; (v) the
  masked fine-tune changes exactly d parameters (checked by diffing state dicts).
- **G1 (after the phase-A pilots, 1 seed):** EDL(from-scratch → D_RL(8)) ≥ 10× EDL(base →
  D_RL(8)); if not, the instrument does not separate pretraining from RL and the step schedule
  is retuned once, then we stop and ask.
- **G2 (phase B):** three seeds per point, intervals over problems, and the independent reviewer
  recount before any number enters a conclusion. A result that lands outside its pre-registered
  band on the good side triggers a leak re-audit of the stream first.
- Population monotonicity (Donoway et al. Def. 4.1): if any block's held-out loss rises during
  retraining, EDL can go negative; such points are flagged and the learning rate halved once.

## 7. Tasks for the week

1. **Fri Oct 10:** this plan, `notes.md`, `questions.md`, the folder skeleton. Commit when asked.
2. **Sat:** T0 orient: fetch and hash the checkpoints and dumps listed in § 2 (≈ 2.5 GB), pin
   Dan's commit in `code/finetuning/MANIFEST.json`; write `passk_area.py`, `support.py`
   (M3, M4) with the G0 (i)–(ii) checks; F0 proposal.
3. **Sun:** run F0 on the laptop (detached, polled); `delta_anatomy.py` (M5); `results.md` with
   the three F0 charts: area by round with the sharpening / expansion split, the class histogram
   of the gap by round and arm, delta sparsity by arm.
4. **Mon:** `subset_ft.py` (parameter masks as gradient hooks around an unmodified
   `state_train.py`), `edl.py` (prequential driver), streams built and audited, G0 (iii)–(v) on
   CPU at toy sizes; F1 and F2 proposals with their step counts.
5. **Tue–Wed:** GPU phase A pilots (one seed), G1; then phase B on one pod.
6. **Thu:** F3 plot, F4 by family, write-up for the Saturday meeting; reviewer recount.
7. **If time:** F5 on the laptop; F6.

## 8. Budget **(decide)**

Phase A (F0) is laptop CPU: $0. The fine-tunes need a GPU: a 600-step `state_train.py`
fine-tune of the 9.56M model took Dan ≈ 2 to 4 minutes on an A40 (`grpo-best/artifacts/gb/
compute_stdout.txt`: 561 updates in 1,401 s at 2,048 rollouts; SFT at 128 proofs per step is
lighter). Estimate: F1 = 6 d × 3 draws × 2 reads × 2 θ_0 × 3 seeds ≈ 216 short fine-tunes ≈ 7
A40-hours with evaluation; F2 = 5 streams × 9 learners × 3 seeds ≈ 135 prequential curves of ≈
8 blocks, retraining from θ_0 each block, ≈ 20 A40-hours; evaluation of each block checkpoint on
the test split at k 32 is the larger cost and is sampled (every other block) if it runs over.
Total ≈ 30 to 40 A40-hours ≈ $20 at Dan's billed rate ($0.49/h) plus pod overhead, the same
envelope as `pretrain-data`. Spend goes to `docs/finetuning/spend.md`. Default: no pod until
Charles confirms the budget; F0 proceeds regardless.

## 9. Metrics and reporting

As in `CLAUDE.md`: every number with its source file, n and seeds; intervals over problems; one
seed is signs of life, three for any comparison; "new capability" is the count of problems at
base 0 of k and trained ≥ ½, never only an average; every selection effect named (Dan's groups
are defined on sample seed 0; the gap halves are split by renaming class; the step schedule is
chosen on a pilot seed and then frozen). Pass@k area and k\* are reported next to, never instead
of, avg solve rate. Dan's Lean-judged numbers and our pinned-verifier numbers are labelled and
never averaged together. Results go to `experiments/finetuning/<exp>/results.md` and to one doc
for the Saturday meeting with three charts: EDL against pass@k area, the gap's base-support
class histogram by round and arm, and d₉₀ against round.
