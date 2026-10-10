# Log (finetuning)

## 2026-10-10 — workstream created

Charles: create a `finetuning` workstream with the same structure as `textbook-problem` and
`pretrain-data`, and propose in its plan how to solve the elicitation problem, starting from two
ideas from the call: (1) fine-tune ~100 randomly chosen parameters on correct proofs and see how
much of the base-to-RL gap that closes, on the RL model too, as a description-length measurement
that should change with the amount of RL (GRPO or EI); (2) measure excess description length and
plot it against the area under the pass@k curve (k to 64).

Survey (no runs): read both existing plans and logs, the literature note (the "Donaway" idea is
Donoway, Joren, Roger and Leike 2026, *Excess Description Length of Learning Generalizable
Predictors*, arXiv 2601.04728; definitions confirmed from the paper's HTML: prequential
codelength minus n × final held-out loss, answer tokens only, algorithm held fixed), the
project's big-picture question 2 (the elicitation-to-emergence crossover), and Dan's bucket.
Found that Dan's bucket already holds every per-round checkpoint needed (EI r1–r8, GRPO
default / unlikely / passk / distinct, replay-only control, ladders from five Stage-1 starts; 3
seeds) with k = 256 per-problem dumps, and that his grpo-best analysis already splits
`rl_targets` into base-solved / base-rare / never-solved groups where GRPO pass@k solves 0.27 of
the never-solved group at k 256 against EI's 0.05. His rl-from-ckpt run has a one-step
base-support measure (worst-step log-prob of the reference predicts ladder solves, x50 ≈ −11
nats) that the plan's M4 generalises.

Written: `docs/finetuning/{plan.md, notes.md, questions.md, spend.md}`, skeletons
`code/finetuning/`, `experiments/finetuning/`. The plan has seven measurements (M1 random-subset
fine-tunes, M2 EDL, M3 pass@k area with a sharpening / expansion split, M4 base support of RL's
proofs, M5 weight-delta anatomy, M6 steering, M7 sample complexity by family), seven
experiments F0–F6, with F0 (M3–M5) entirely CPU and zero-training, gates G0–G2, a ≈ $20 GPU
envelope pending confirmation, and a falsifiable statement of the crossover (H1–H3).

Next: T0 (fetch and hash checkpoints and dumps, pin Dan's commit in a MANIFEST), `passk_area.py`
and `support.py` with the G0 checks, F0 proposal, then F0 on the laptop. Nothing committed yet;
Charles has not asked.

Open questions: `questions.md` 1–7 (primary subject, seed-split fine-tuning, K for the area,
budget, using Dan's artefacts, coordinate vs subspace masks, the fixed algorithm A).

## 2026-10-10 (evening) — literature pass and ranked hypotheses

Charles: "Read some more recent literature on finetuning and elicitation. Propose different
hypotheses to test, with the source, and rank them by how useful they are compared to the work
already done between all the recent branches."

Read (abstracts and HTML pages, 2025–2026): Donoway et al. EDL theory and *Bits That Count* (ICML
2026), Morris et al. *Learning to Reason in 13 Parameters* (RL at 13 parameters matches full
fine-tuning, SFT needs 100–1000× more), *LoRA Without Regret*, Mukherjee et al. sparse RL
subnetworks, Zhu et al. off-principal updates, Shenfeld et al. *RL's Razor*, *Echo Chamber*,
*Spurious Rewards*, one-example RL, *SFT Memorizes RL Generalizes*, *RL Squeezes SFT Expands*,
*Beyond Reasoning*, *Beyond the 80/20 Rule*, Yue et al., *Invisible Leash*, the two-stage
shrink-then-expand view, diversity collapse via overtraining, curriculum RL beyond the base,
Cover@τ, ReLIFT, password-locked elicitation, the Elicitation Game, reference grafting,
unsupervised elicitation, *Base Models Know How to Reason*.

Surveyed every remote branch with two read-only subagents (git show only). Found that base
log-probabilities of RL proofs, sampling support to 1.2 M attempts, compute-matched and
replay-only controls, continued RL to r16, two family knockouts and steering vectors are done on
Dan's side, and that Leon's `benchmarking` branch ran EDL, steering (null), weight
interpolation (α 1.25 doubles pass@128) and pass@4,096 on the 3.2M models, with EDL tracking
starting loss (Spearman 0.94). Nobody has measured weight-space anatomy, parameter-count-limited
fine-tunes or RL, KL-minimality, or single-example teachability beyond excluded middle.

Written `docs/finetuning/literature_hypotheses.md`: a done-already table, the reading list with
links, 14 hypotheses in three tiers (A: masked RL vs SFT parameter counts, intrinsic dimension of
the gap, weight-delta anatomy, single-example teachability per family; B: delta extrapolation,
per-round support-class histogram, overtraining variant, starting-loss-controlled EDL, k 4,096
curve; C: spurious rewards, steering, Cover@τ, boundary curricula, checkpoint averaging), and
the consequences for the plan. `plan.md` § 3 carries a revision note; M6 is dropped.

Next: T0 as before, with M5 and M1 first. Needs Charles: whether the workstream may run the
small masked GRPO arms of H1 (≈ $18) and the overtraining ladder of H7 (≈ $5); recorded in
`questions.md` 8–9. Nothing committed.

## 2026-10-10 (afternoon) — mock timing for the EDL / random-subset measurement (laptop CPU, $0, not pre-registered)

Charles asked whether a mock of Donoway's method can be done quickly and how many fine-tune steps it
needs. Scratch work only (session scratchpad `scripts/mock_edl_timing.py`, `scripts/steps_sweep.py`,
`runs/steps/*.jsonl`); nothing under `experiments/` yet, one seed, signs of life only.

**Setup.** Base `stage1_best12_s0_b1200.pt`; proofs from Dan's seed-0 ladder artefacts
(`best-state/artifacts/bs/la_T1_best12_s0/found_{1,8}.jsonl`, 20 MB / 405 MB). Stream = the 796
`rl_targets` theorems solved by round 8 but not at round 1 (k 32), one proof each, 128 train / 128
held-out (`new_train128.jsonl`, `new_held128.jsonl`). Loss = Dan's `state_train.loss_on` (nats per
action token, `state_env.decompose` pairs), 8 proofs per step, AdamW, no replay, no schedule. Proofs
not yet re-judged by the pinned verifier (Lean-judged by Dan's env); do that before anything is quoted.

**Timings (M1 Pro, 6 torch threads, one process):** load + decompose < 1 s; forward-only coding of
200 proofs 16 s; full step 2 s at 8 proofs, 15 s at 32; masked step costs the same (grad masking
saves nothing). Four processes in parallel run ≈ 4× slower each.

**Starting loss is tiny.** Base on held-out proofs of theorems it never solved: 0.081 nats/token
(found_1 proofs, the base's own samples: 0.038). A proof has ≈ 240 action tokens (13–14 pairs), so
≈ 29 bits per whole never-found proof before training. The elicitation gap per proof is a few bits.

**Steps needed (held-out loss, nats/token; n 128/128, 1 seed):**

| arm | lr | step 0 | 16 | 32 | 64 | 128 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full (9.56M) | 3e-4 | 0.082 | 0.062 | **0.059** | 0.061 | 0.072 | 0.072 |
| 10,000 random | 3e-3 | 0.081 | 0.077 | 0.073 | 0.069 | 0.0625 (still falling) | – |
| 10,000 random | 3e-4 | 0.081 | 0.081 | 0.080 | 0.080 | 0.076 | – |
| 100 random | 3e-2 | 0.081 | 0.081 | 0.080 | 0.079 | 0.0775 | – |
| 100 random | 3e-3 | 0.081 | 0.081 | 0.081 | 0.081 | 0.080 | – |
| 100 random | 3e-4 | 0.081 | 0.081 | 0.081 | 0.081 | 0.081 | 0.082 |

Reading: full fine-tune peaks at ≈ 32 steps (≈ 2 passes over 128 proofs) and overfits back to the
starting loss by step 128, so θ* for EDL needs early stopping, not Dan's 600 steps. 10k random
parameters (0.1 % of the weights) at lr 3e-3 recover most of the full gain by step 128 with no
overfitting yet. 100 parameters move only at lr 3e-2 and slowly. The step count is therefore a
function of d and lr; M1 must tune lr per d (or sweep) and train to a held-out stopping rule.

**Next:** proposal `experiments/finetuning/edl_mock/` (streams from seed 1's ladder so no evaluated
model trained on the exact proofs; prequential blocks of 8 with retrain-from-start to the stopping
rule; base, r2, r4, r8; pinned-verifier re-judge; renaming-class audit against the 72 and test files),
then run detached on the laptop (≈ 1 h). Open: lr per d for M1; whether to measure d₉₀ by held-out
loss or by solve rate.
