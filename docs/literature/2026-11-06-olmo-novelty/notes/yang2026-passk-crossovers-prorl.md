---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - yangWhenPassCrossovers2026
---

# When pass@k crossovers are real: the ProRL-1.5B re-test (ProRL side-by-side companion note)

Paper: [@yangWhenPassCrossovers2026] (proposed key; Chen Yang, Texas A&M, stat.ML)
Source: arXiv 2609.22547v3 (2026-11-03), PDF text `PR_2609.22547v3.txt`. Read §1–§5 and Tables 1–3 in depth; appendices skimmed. The main note on this paper is [yang2026-passk-crossovers](yang2026-passk-crossovers.md) (IL ledger); this one covers only what bears on ProRL and its correlation.

## Setup card

| | 2609.22547 |
|---|---|
| Models | public checkpoints vs their bases; clean benchmark: ProRL-1.5B, DeepScaleR-1.5B vs R1-Distill-Qwen-1.5B; two 7B RL models vs R1-Distill-Qwen-7B [PR-130]; controlled GRPO / SFT arms [PR-134] |
| RL share of pretraining | not computed (third-party checkpoints; ProRL's own row in rl-compute-share) |
| RL steps | not applicable |
| Domain | competition maths, 1,060 problems [PR-111] |
| Largest k | 128 on the clean benchmark, n = 128 both sides [PR-131]; 4,096 in controlled runs [PR-134] |
| Grading | answer match (Math-Verify) [PR-132] |
| Controls | symmetric budgets yes [PR-131]; format partly (forensic check) [PR-123]; guessing partly (argued away for free-form answers) [PR-143]; contamination yes [PR-133]; truncation yes [PR-115–PR-117] |

## Learnings

**Observed.**
- ProRL-1.5B vs its base at 32k: +0.114 at k = 1, −0.021 at k = 128, first loss κ ∈ [12, 63], dominance p = .0022 [PR-112, PR-113]. The two 7B pairs show no supported loss [PR-110].
- At 16k (n = 256) ProRL's visible crossing at k = 56 is not supported (p = .20) [PR-114]. At 8k the base truncates 30.6 % of generations against ProRL's 2.0 %, worth ≈ 0.7 log-odds of ProRL's pass@1 edge [PR-115, PR-116].
- Per prompt: equal base rates diverge after RL, and the divergence replicates across independent halves [PR-135]. With dispersion modelled, the conditional-mean log-odds slope of RL on base success is 0.84 for ProRL (1.81 under a single-curve fit) [PR-118]. RL lifts hard prompts on average; the large-k loss is a 2–3 % minority of base-solved prompts locked onto a repeated wrong answer [PR-119]. Of 27 problems the base solves once in 128, 44 % fall to zero under ProRL while the rest improve ≈ 7× [PR-120, PR-121]. Replacing dispersion by its mean removes ProRL's crossover [PR-122]. Locked prompts are neither truncated nor misformatted [PR-123].
- Controlled GRPO arms fall behind the base from k ≈ 200 to 4,096; SFT does not [PR-134].

**Paper interprets.** A crossover is "a property of specific checkpoints and training regimes" under a stated protocol [PR-136]; RLVR acts as a kernel that unlocks most hard prompts and locks a few [PR-119].

**Our interpretation.** Among the sources screened here, this is the only one that re-tests ProRL's "largest where the base is weakest" pattern per prompt. On maths the pattern holds on average in log-odds (slope < 1) but hides a bimodal split, and ProRL still loses at large k, as ProRL itself labels math "Diminish" [PR-55]. Its split-half device matches our pre-registered P1 split, and a log-odds slope avoids the bounded-difference ceiling behind ProRL's Fig. 3. It does not touch ProRL's Reasoning Gym zero-base tasks or IF.

## Evidence and limitations

Single author; maths only; the clean benchmark goes to k = 128; κ intervals are wide. Which ProRL was tested is not stated: Table 8 names the repository only [PR-144], whose default load is v2 (3,000 steps), v1 needing `revision="v1"` [PR-145, PR-146]; so probably v2 (UNVERIFIED).

## Connections and questions

- C1: **extends**: no supported large-k loss for the 7B pairs [PR-110], as C1 loses nothing at 7B; our pretrained-base RL-Zero start is a setting it lacks.
- C2: **extends**: same direction beyond 1.5B [PR-110]; our 32B SFT + DPO start is a setting it lacks.
- C7: **extends**: a prompt-bootstrap dominance test with simultaneous bands, and token-limit symmetry, are controls our rule lacks [PR-114, PR-117].
- C5, C8: **replicates**: the token limit changes the verdict; asymmetric truncation inflates RL's apparent edge [PR-116, PR-117], as our token-budget test found on 1 of 3 geometry prompts.
- Q2: add a dispersion-aware log-odds slope beside P1. See [liu2025-prorl-v1-full](liu2025-prorl-v1-full.md).
