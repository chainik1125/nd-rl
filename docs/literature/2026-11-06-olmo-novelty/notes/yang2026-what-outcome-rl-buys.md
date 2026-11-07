---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - yangWhatOutcomeRL2026
---

# What Outcome RL Buys: finishing, reliability, rarely reach (physics and maths, thinking bases, k = 8)

Paper: [@yangWhatOutcomeRL2026] (proposed key; Shan Yang)
Source: https://arxiv.org/abs/2610.34894v1 (v1, 2026-10-26, the only version). Read in full. Ids are in
`ledger/IL.tsv`.

## Setup card

| item | value |
|---|---|
| models, start | 20 released RL models (1.5B–32B) vs their **thinking** bases [IL-85, IL-86]; one zero-RL pair [IL-132]; an 8B probe [IL-142] |
| RL share | not computed; rl-compute-share: ProRL v2 4.4 × 10⁻³ – 2.1 × 10⁻², Skywork-OR1-32B 1.5 × 10⁻³ – 9.0 × 10⁻³ |
| RL steps | 50–4,380 across recipes [IL-194, IL-195]; probe ≤ 60 updates [IL-142] |
| domains | physics (≤ 1 % of RL data, except ProRL v2) and maths [IL-120, IL-122] |
| largest k | 64, on lone wins only; main reads at k = 8 [IL-92, IL-93] |
| grading | strict matcher; Qwen3-32B judge; Claude Opus 5.5 readers; one author [IL-112–IL-116] |
| controls | symmetric budgets **yes** (mirror, both sides resampled) [IL-90, IL-103]; format tolerance **yes** [IL-105, IL-114]; guessing **partly** (chance, not guessing) [IL-104]; contamination **partly** (date strata only) [IL-140, IL-141] |

## Learnings

**Observed.**
- ∆pass@1 = finishing + reliability + net reach; reach needs the other model to finish and fail all 8 samples
  [IL-87–IL-89, IL-91].
- Where the base runs out of tokens, finishing is 82 % of the change at 4k–8k; more tokens close most of it
  [IL-95, IL-96]. Otherwise the gain is reliability [IL-97].
- Net reach is ≤ 2.1 points on all four sets, in-domain maths included [IL-98, IL-122].
- 3.5 % of 144 physics lone wins hold up [IL-99]; 56 more samples let the base solve 87 of 121 [IL-101]; 9 of 10
  upheld "new" RL steps are in base samples [IL-100].
- 32B pair: the gain is reliability; most robust wins are answer form [IL-108–IL-110]. So are three fifths of all
  robust cells read [IL-105].
- ProRL v2 (RL data includes IF [IL-121]): its physics gain is all finishing [IL-125, IL-126]; counting partly
  unfinished never-solved problems as unsolved lifts net reach 0.001 → 0.067 [IL-127].
- Training length does not predict reach (ρ = 0.13) [IL-134, IL-135].
- Probe: held-out gain is finishing [IL-143]; a right question elicits the missing fact [IL-145, IL-146]; two
  auxiliary scorers were gamed [IL-147, IL-148].

**Paper interprets.** RL's gain is "changed behaviour … not new knowledge" [IL-138]. The extra tokens are hesitation,
though finishing and knowledge are not fully separable [IL-129, IL-130]. Scale may matter [IL-136, IL-137].

**Our interpretation.** For physics and maths with thinking bases, it already contains:
- the token-budget account of gains and of part of reach (our C5, C8 Test 1);
- RL lone wins recovered by the base with more samples (our C1 pattern);
- most of C7's methods: mirror, resampling, blind reading, a judge beside the matcher, registration, date-stratified
  contamination.

Absent: IF (no IFEval or IFBench in the text), gaming of evaluation checkers (its gaming is of training-time scorers),
a within-run trajectory, and k > 64. A k = 8 bound is weak evidence of absence next to ours at k = 256.

## Evidence and limitations

- LLM readers; the human check is one author, 50 of 143 pairs, about half an hour [IL-116, IL-202, IL-203].
- The main physics set is filtered on the probe's base [IL-201], but the unfiltered set agrees [IL-123].

## Connections and questions

- **C1, replicates:** the base recovers RL lone wins; new steps are already in the base [IL-100, IL-101]; distilled
  bases [IL-133].
- **C2, replicates:** the 32B distilled pair gains reliability; its reach is answer form or noise [IL-108–IL-111].
- **C3, unrelated:** no IF; the bound covers four physics and maths sets at k = 8 [IL-98].
- **C4, replicates** (direction): checker artifacts inflate reach, but they are answer form, not gaming [IL-105,
  IL-113].
- **C5, replicates:** finishing explains cap-bound gains [IL-95, IL-126, IL-127].
- **C7, replicates** (partly): mirror, resampling, blinding, judge, registration [IL-90, IL-94, IL-114]; no large k,
  trajectory axis or guessing control.
- **C8, replicates** Tests 1, 2, 4 in kind [IL-96, IL-145, IL-134]; Test 3 differs: physics lone wins are mostly
  recovered [IL-101], our IF reach survives 1,024 draws.
