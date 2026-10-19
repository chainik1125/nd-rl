---
written_on: 2026-09-27
written_by: agent:claude
status: proposed; experiment 1 authorised by Dan 2026-09-27
---

# Proposal 12: the elicitation curve — does RL against a verifier expand what the base model can do?

## Where the project stands

Numbers are for the 3.2 M-parameter from-scratch GPT in the `lean_seq` Lean format at cap 6 unless a
line says otherwise. Every one is from a reviewed run.

**What holds up.**

1. **Expert iteration (EI) clearly beats the base model at equal attempts.** On the ladder transfer
   pool (2,285 theorems never trained on), EI after 8 rounds solves 856 against the frozen base's
   158 at the same 8 × 32 attempts (`cap-horizon` K6, seed 0; the frozen null range across eight
   independently trained models is 62–265, so this clears the noise floor). The robust length
   frontier `L*` rises by 2–3 lines.
2. **Every manipulation converges on the same frontier.** All six EI variants of ladder-A reach
   transfer `L*` = 10 (token format). Every training cap from 6 to 14 reaches T1 `L*` = 12
   (`cap-horizon`). The depth-3 dial's round-4 endpoint lands at 0.38–0.44, 0.396–0.436 and
   0.387–0.420 in three independent runs, from frozen base rates spanning 0.032–0.322.
3. **The gains look like known proofs written longer or more reliably, not new kinds of proof.**
   Ladder-A's reviewer found its `L*` = 10 was one already-known proof family written longer.
   `cap-horizon` found 114 accepted proofs of ≥ 10 lines, none of which proves a target needing
   more than 8; the extra length is padding.
4. **The one "created from zero" result dissolved.** Depth-3 acquisition at pretraining frequency 0
   was part token-format artifact, part a non-zero base rate. Depth-4 at frequency 0 went 0 / 0
   (round-2 run 2).
5. **Dataset style is noise.** Of twenty standing findings scored against the measured noise floor,
   two survive, and both are the cap-8 arm (`NOISE_FLOOR.md`).

The consistent reading is that **EI makes the model better at what it can already occasionally do,
and generalises somewhat within that, but nothing has crossed a boundary**. That is a pattern. Two
gaps stop it from being a result.

**Gap 1 — the headline metric cannot separate eliciting from creating.** "Frozen base at equal
attempts" asks whether RL is more sample-efficient than the base at the same number of attempts
(256). Pure sharpening passes that test. "New capability" is a claim about *support*: does the RL
model solve theorems the base model solves with essentially zero probability even at very large k?
The project has never measured that. Base-model coverage has only been measured to k = 2,000, on
small pools, and never on the theorems RL newly solves.

**Gap 2 — the wall has not been located.** Required proof length stalls at ≈ 9–10 lines on the
reductio pool (the 10-line stratum is unsolved in all 12 `cap-horizon` cells) and true minimal length
at ≈ 12–13 on the ladder (`L_true` ≥ 13 solved 0 times in 11 `ds-generator` ladder runs). It survives
every change of data, cap, technique and format — the most robust fact in the project — and it is
unexplained: model capacity, search, or data are all live. Data difficulty in particular has never
been varied: `cap-horizon` varied the *written* length of training proofs, and the generator writes
long proofs of easy theorems.

## The question

For each theorem x, let p_base(x) be the base model's per-sample probability of producing an
accepted proof, and p_RL(x) the RL model's. The **elicitation curve** is the probability that RL
acquires a theorem (p_RL(x) above a fixed threshold) as a function of p_base(x), at fixed RL compute,
and how that curve moves with model scale and pretraining coverage.

- **RL as a sampling amplifier** predicts a step: theorems with p_base above roughly 1 / (RL
  samples) are acquired, those far below are not. Base pass@k eventually catches the RL model.
- **RL as a generaliser** predicts mass well below that step: theorems the base essentially cannot
  solve, acquired because training on related successes transfers. Base pass@k never catches up.
- How the curve shifts with model scale and with pretraining coverage is the emergence question.

This is the question the current RL-for-reasoning literature argues about. Yue et al. 2025 ("Does
reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?") find
RL-trained models win at pass@1 but the base overtakes them at large k, so RL narrows. NVIDIA's ProRL
(2025) argues prolonged RL expands the boundary. Both have to extrapolate, because pass@100,000 on an
LLM is unaffordable. **This project's comparative advantage is that it isn't**: a 3.2 M model samples
pass@20,000 over hundreds of theorems for a few dollars, pretraining is fully controlled, and the
verifier is exact. The project has not yet used that advantage.

## Experiments

### 1. Measure support directly (authorised 2026-09-27; brief `elicitation/BRIEF_support-curves.md`)

Per-theorem success probabilities for the base model and its EI model on ≈ 400 ladder transfer
theorems stratified by `L_true` (long ones over-sampled), with an adaptive second stage that spends
up to 40,000 more base attempts on the crux set — theorems the EI model solves and the base has not.
Output: a per-theorem scatter of log p_base against log p_RL, with upper bounds for theorems the base
never solves; pass@k curves for both models, stratified by `L_true`, to k in the tens of thousands;
the crossover (or not). Sampled at T = 0.8 (the RL setting) and, on the crux set, at T = 1.0, because
support at T = 0.8 is not the model's support. This is the first point on the elicitation curve and
it reinterprets every EI result already on file. ≈ $11.5, ceiling $15.

**Prediction to beat.** The pattern above predicts the amplifier: base pass@k on the crux set rises
toward the EI model's with k, and few crux theorems stay at zero through 40,000 base attempts.
**Falsifier:** ≥ 20 crux theorems with 0 base successes in ≥ 40,000 attempts at both temperatures
while the EI model solves each at p_RL ≥ 0.01 — support expansion through generalisation.

### 2. Locate the wall

Three cuts, each cheap and each with a prediction that separates it from the other two. Run after
experiment 1, because its per-theorem p_base tells us which theorems sit behind the wall and whether
the base has any support there at all.

- **Search.** Continue EI to 24 rounds (proposal 7, queued since 2026-09-19; ladder-A's reviewer's
  first-ranked follow-up: "nothing had plateaued") and add verifier-guided search at evaluation on the
  EI model (technique T10). If more search moves `L*`, the wall is exploration.
- **Model.** The same data and schedule at ≈ 8× parameters (8 layers, d 512). If base support behind
  the wall appears and `L*` moves, the wall is capacity.
- **Data difficulty.** Inject theorems verified hard by `minlen.py` (`L_true` 10–13, with minimal
  proofs) into pretraining at small frequencies f — the pattern dial, run on difficulty instead of
  pattern. With experiment 1's measurement this gives the elicitation curve's dependence on coverage.

### 3. The bimodality as a natural experiment

The 52 `noise-floor` models share one data distribution, yet ≈ half land in a high depth-3 mode and
a quarter in a low one (held-out depth-3 slice 0.008–0.918). Run EI on matched low-mode and high-mode
models: does RL lift low-mode models to high-mode performance, and does that track their large-k base
support? This asks whether RL can finish a capability pretraining started but did not form — the
closest thing here to emergence under RL. It pairs with `stage1-dynamics` (when a seed's mode is
decided) and uses existing checkpoints.

## Design standards for this proposal

- **One primary quantity:** per-theorem log p_base against log p_RL, with confidence intervals and
  upper bounds. Pass@k curves, `L*` and solve counts are derived from it.
- **Within-model and paired comparisons** first; cross-run comparisons quote `NOISE_FLOOR.md`.
- **Keep every RL checkpoint** (upload `ckpts/ladder/`); the Lean-format EI models behind every
  headline above were never kept, so experiment 1 has to re-train them.
- **Say the temperature** of every support claim.
- **Lean alone decides** (2026-09-27); both models are judged identically.

## Order and budget

Experiment 1 now (≈ $11.5 of the ≈ $17 above Dan's $100 balance floor). Experiments 2 and 3 are
sized after it: each cut of experiment 2 is ≈ $5–15 at this model size (the ≈ 8× model is the
dearest), experiment 3 ≈ $5.
