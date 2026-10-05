---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - hu2023passuntil
---

# PassUntil: infinite-resolution evaluation and a definition of emergence

Paper: [@hu2023passuntil]
Source: arXiv 2310.03262v3 (HTML rendering read: abstract, Sec. 1, 3-7, App. A, B.1, E.2)

## Learnings

- **Claim: apparent emergence is partly a resolution artefact.** Small models "demonstrate critical and
  consistent task performance improvements that are not captured by conventional evaluation strategies due
  to insufficient measurement resolution" (Abstract). Resolution is defined as "the smallest probability
  difference that the evaluation strategy can detect" (Sec. 1, footnote 1). Small-model performances "are on
  the order of 10^-5 and exhibit steady enhancement as the model scales up" (Sec. 1).
- **PassUntil = sample until r successes.** Fixed-K sampling has a budget problem: "it is hard to define the
  budget K that is both acceptable in computation and has enough resolution for hard samples" (Sec. 4.1).
  Instead "We stop sampling until r (a constant) samples have passed the evaluation and record the sampling
  number K", and PU = r/K (Sec. 4.1, Eq. 2). Theorem 1: "PU is a maximum likelihood estimate for P(s)",
  because "The failure time f=K-r follows the negative binomial distribution" (Sec. 4.1). In practice "we set
  r to as small as 1 or 2" and "the upper bound of K to a large number, such as 10^5" (Sec. 4.1).
- **Why not compute P(s) from token probabilities of the reference answer?** "deriving P(s) theoretically
  from the token probability on the ground truth solution is not feasible", because "there are likely to be
  multiple viable solutions" and multiple tokenisations decode to the same string (Sec. 4.1, "Necessity").
- **The random-baseline condition is stated explicitly.** "our evaluation strategy is designed to be
  applicable when a random baseline achieves P(s)=0"; with multiple choice, guessing (e.g. 0.25) "can
  overshadow the improvements made by smaller models" (Sec. 4.1, Limitations). They also remove test items
  that random guessing can hit by exact match (Sec. 5.3; App. D.5).
- **Task scaling law.** Assuming per-token loss follows a power law in N with zero irreducible loss,
  PU ∼ exp(−c N^(−α)) (Sec. 4.2, Eq. 3-4), i.e. log(−log PU) is linear in log N; verified on HumanEval,
  Emoji Movie, Date Understanding (Sec. 5.3, Fig. 4). Instance-level fits (one curve per problem) improve
  the 2.4B prediction: HumanEval series 1, real 0.05990 vs instance-level fit 0.05987 (Table 1) — "merely
  0.05% deviation before training starts" (Abstract).
- **Loss as an assistant, not a substitute.** For hard instances with too few non-zero PU values, "We
  suggest leveraging test loss on ground truth answers to assist the prediction" (Sec. 5.4; App. E.2), but
  "one cannot deduce actual performance (accuracy) solely from loss values" and "the loss of an individual
  sample does not have a one-to-one correlation with PassUntil results" (App. A.2).
- **Quantitative definition of emergence.** With F(N) = log(−log PU(N)): linear in log N = "scaling law
  growth"; convex = "sub-scaling law growth"; concave = "super-scaling law growth, or 'accelerated
  emergence'" (Sec. 6, Definition 1). Theorem 2: "'multi-step reasoning' leads to sub-scaling law growth"
  (Sec. 6); Theorem 3: a max over several "circuits" each following the law gives a concave F (Sec. 6).
- **Even 10^5 samples leave zeros.** Some tasks "miss 1 or 2 valid estimation points due to their extreme
  difficulty for 0.03B and 0.1B models, since 0 PassUntil is obverseved even with 10^5 sampling time"
  (Sec. 6).
- **Pilot.** On BigBench subsets, "most instances are passable with enough random sampling times" (Sec. 3,
  Fig. 2).

## Evidence and limitations

- Evidence: Fig. 4, 6, 7, 8; Table 1. Scale: models 0.03B-2.4B (App. A.1 "Scale Limitation").
- The fit is across model size N, not across training steps or RL; the authors only fit tasks that show
  emergence (App. A.1 "Scope Limitation") and call their circuit hypothesis test "superficial" (App. A.1).
- Our statistical note (not in the paper): with r = 1, PU = 1/K is the MLE but is strongly biased upward
  for small p: for geometric K, E[1/K] = −p ln p / (1 − p) ≈ p ln(1/p), i.e. about 11× too large at
  p = 10^−5. For r ≥ 2, (r − 1)/(K − 1) is the unbiased estimator. The cap at K = 10^5 censors the
  hardest items; a censored-likelihood treatment is needed.
- The App. E.2 loss-PU relation is rendered without a clear functional form in the text we have; not used.

## Connections and questions

- **Definition offered:** capability on an instance = its single-attempt pass probability P(s), measured
  with unbounded resolution by sampling until r successes (PU = r/K); emergence = the shape of
  log(−log PU) versus log N (linear / convex / concave).
- **New vs better access:** not about RL, but it gives two usable ideas. (1) A "capability present but
  tiny" vs "absent" distinction is made by resolution: an instance has the capability at level P(s) ≈ r/K,
  and a zero is only a censored observation "P(s) < ~1/K_max". (2) Emergence is defined relative to a null
  growth law: growth that follows the extrapolated law is predictable improvement; growth faster than the
  law ("accelerated emergence") is the closest the paper comes to "something new". The analogous RL test:
  does r16's per-theorem log p exceed what pend's trend (over pretraining checkpoints, or the base log-p
  curve) predicts?
- **Null / floor:** explicit — PassUntil is designed for tasks where the random baseline has P(s) = 0,
  and items hittable by guessing are removed. For Lean proofs a random model's P(s) is not 0 but is
  astronomically small, so the right statement is quantitative: report PU for the random-init checkpoint
  as a censored bound (no success in K_max) and compare log PU across checkpoints.
- **Transfer to our setting:** run PassUntil (r = 2, K_max = 10^5 or matched to the RL budget) for pend on
  the theorems that pend never solves at 512 samples but r16 solves; estimate p with (r − 1)/(K − 1) and
  a censored likelihood for those that hit K_max. Compare log p_pend with log p_r16 per theorem. Cost:
  bounded by K_max × (#theorems), e.g. 50 theorems × 10^5 = 5·10^6 samples worst case — feasible on a pod
  for a small model, and early stopping makes easy theorems cheap. Also fit log(−log PU) across the
  available pretraining checkpoints to get a "pretraining trend" null. Failure modes: censoring at K_max;
  r = 1 bias; the method says nothing about theorems that stay at zero.
- Related: schaeffer2023mirage.md (metric choice), schaeffer2025powerlaws.md, kazdan2025passk.md
  (distributional alternative), chen2021evaluating.md.
