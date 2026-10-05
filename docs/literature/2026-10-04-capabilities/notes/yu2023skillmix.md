---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - yu2023skillmix
---

# Skill-Mix: random k-subsets of known skills, and a counting argument that successes are not memorised

Paper: [@yu2023skillmix] (Yu, Kaur, Gupta, Brown-Cohen, Goyal, Arora, 2023; ICLR 2024)
Source: arXiv 2310.17567v1 (HTML rendering read: abstract, Sec. 1, 1.2, 3.2, 4 (grading), 5 opening and 5.1
metrics and saturation, Sec. 6, 7, 8; App. D estimation not read).

## Learnings

- **Claim.** "a key ability of an AI agent is to flexibly combine, as needed, the basic skills it has
  learned"; with N skills and random k-subsets, "Since the number of subsets grows like N^k", the evaluation
  will "require the LLM to produce text significantly different from any text in the training set"
  (Abstract).
- **Protocol.** Pick k of N skills (each with a Wikipedia entry, so known to the model) and one of T topics
  of low corpus frequency; ask for ~3 sentences on the topic exhibiting all k skills; grade with GPT-4 or
  LLaMA-2-70B plus human spot checks (Sec. 3.2, 4). Ratio "of Full Marks: 1 if all k+3 points are earned,
  and 0 otherwise"; "We then take the maximum value of the metrics among the 3 generations" per combination
  (Sec. 5.1) — i.e. a best-of-3 score. The "saturation point" is the k "at which a model's score in
  skill-mix drops off" (Sec. 5): LLaMA-2 7B / 13B / 70B-Chat saturate at k = 2, 3, 3 (Sec. 5.1).
- **Novelty definition.** Going beyond "stochastic parrots" means the "ability to correctly use
  combinations of skills + topic that it had not seen in the training corpus" (Sec. 1.2).
- **Counting argument** (Sec. 6). The training corpus holds at most p_s^k p_t C(N,k) T L texts with a given
  k skills and a topic (skill frequency p_s, topic frequency p_t, L sentences); the model succeeds on at
  least α_k C(N,k) T. If α_k > (3/2) p_s^k p_t L, "more than one-third of the successful generations contain
  combinations not seen in the training corpus". Estimates: p_s ≤ 0.0144, p_t ≤ 0.0022, L ≤ 5×10^10, giving
  p_s^k p_t L ≤ 0.07 (k = 5) and ≤ 0.001 (k = 6); GPT-4 has α_5 ≈ 0.12, α_6 ≈ 0.08 even after removing
  common skills. "We are unable to find similar evidence for any other model". Caveat: "This calculation
  assumes independence among the occurrence of skills" (footnote 10; checked for k = 2 only).
- Frequent skills make the test easier, so 17 high-frequency skills were removed (Sec. 1.2); preliminary
  fine-tuning on synthetic productions "can improve scores on skill-mix to some extent" (Sec. 7).

## Evidence and limitations

- LLM grading with family bias and high human-grader variance (Sec. 7, 8); best-of-3 scoring; the corpus is
  unknown for most models, so frequencies are estimated on RedPajama.
- The counting bound needs independence of skill occurrences; positive correlation between skills would
  increase the number of training texts containing the combination and weaken the conclusion.
- No pre/post-training contrast; skills are taken as given, never measured individually as present.

## Connections and questions

- **Definition offered:** the capability to *combine* skills, measured as the full-marks rate α_k on random
  k-subsets of individually known skills, summarised by the saturation point k*; novelty is certified by
  comparing α_k with the expected training frequency of such combinations.
- **New vs better access:** for *novel outputs*, yes, by a counting rule: successes cannot all be
  retrieval if the success rate exceeds what the corpus could contain. It does not separate RL from
  pretraining (no such contrast), but the rule ports directly: for a schema s, if RL's accepted proofs of
  held-out instances of s are combinations absent from every training source (pretraining corpus, replay,
  RL targets), the capability cannot be retrieval of a stored instance.
- **Null / floor:** the "any k" objection is answered by fixing the attempt budget (best of 3) and
  requiring a success rate α_k above the corpus-frequency bound; a random policy has α_k ≈ 0 at that
  budget.
- **Transfer to our setting:** we know the corpus exactly, so no independence assumption or frequency
  estimate is needed: novelty is a lookup. A Skill-Mix analogue is a generator task: choose k inference
  schemas (¬¬-elimination, ∨-introduction, reductio, →-introduction, ∨-elimination, …) that never
  co-occur in a single pretraining proof, generate theorems whose minimal proofs need exactly those k, and
  measure solve rate at a fixed budget for pend, r8, r16 as a function of k; the saturation point k* per
  model is the summary. RL "created compositional capability" if k* rises after RL at matched budget and
  not after replay-only training. The usual double-negation proof of A ∨ ¬A is a mix of about four
  schemas (classical reductio / ¬¬-elimination, ∨-introduction on both sides, ¬-introduction,
  ¬-elimination) on a premise-free goal (our reading of the proof; check against the seed's actual proof). Cost: generation is free;
  sampling ~100 theorems × k levels × 3 models at k = 256 is a J2-sized job. Failure modes: "skills" must be
  defined by the generator's rules, not post hoc; minimal-proof uniqueness (a theorem may have proofs that
  avoid one of the k schemas) must be checked with the prover.
- Related: `arora2023skills.md` (the theory Skill-Mix tests), `keysers2020cfq.md` (combinatorial novelty as
  compound divergence), `okawa2023multiplicative.md`, `hupkes2020compositionality.md`.
