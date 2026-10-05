---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - lin2023unlocking
---

# The unlocking spell on base LLMs (URIAL): alignment as token distribution shift

Paper: [@lin2023unlocking] (Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula, Choi; ICLR 2024)
Source: arXiv 2312.01552v1 (HTML rendering read: abstract, Sec. 1-2, 5.1, 5.4, 6; Sec. 3-4 (URIAL method
and evaluation) skimmed)

## Learnings

- **Question.** Does alignment tuning (SFT, RLHF) add knowledge or only style? "we believe that it is
  essential to accurately distinguish which knowledge and reasoning capabilities originate from
  pre-training as opposed to those that must be acquired through alignment tuning" (Sec. 1).
- **Method: token distribution shift.** Decode the aligned model greedily ("via greedy decoding") to get
  output o; at each position t feed the same context (query + o_<t) to the base model and record the base
  rank η of o_t: unshifted (η = 1), marginal (1 < η ≤ 3), shifted (η > 3, o_t "is rather unlikely to be
  sampled by" the base) (Sec. 2.1, Fig. 2). Also per-position KL(P_base, P_align) and "base-prob" ("We look
  at the probability of the aligned token" o_t under P_base) (Sec. 2.2, Fig. 4).
- **Findings.** Over 1,000 queries (Llama-2-7b vs -chat), "77.7% of the tokens are at such unshifted
  positions, which increases to 92.2% when including marginal positions" (Sec. 2.2). Across three
  base/aligned pairs (RLHF and SFT) "The shifted token ratios are all very low (5%-7%)" (Sec. 2.2); shifted
  tokens are mostly stylistic or safety tokens ("However", "cannot", "Here", "Thank"). The shift shrinks
  along the output: "the KL-divergence goes down over time and the base-prob keeps increasing over time";
  "the average base-rank of aligned tokens are lower than 5 soon after" t ≥ 5 (Sec. 2.2, Fig. 4). Content
  tokens sit at unshifted positions; "untuned LLMs can fluently generate the answer based solely on the
  context prefix" "Thank you for asking! The" (Sec. 2.2).
- **Conclusion drawn.** "base LLMs and their alignment-tuned versions perform nearly identically in
  decoding on the majority of token positions (i.e., they share the top-ranked tokens)"; "the knowledge
  required for answering user queries predominantly comes from the base LLMs themselves" (Abstract).
  Alignment affects "primarily stylistic elements and safety disclaimers in just 5-8% of cases" (Sec. 6).
- **Constructive test: elicitation without weight change.** URIAL aligns base models by in-context
  learning, "requiring as few as three constant stylistic examples and a system prompt" (Abstract), and
  matches or beats SFT/RLHF versions on their just-eval-instruct benchmark for strong bases (Sec. 4).
- **Stated scope limit.** "model tuning may still be necessary for tasks such as coding (Luo et al., 2023),
  mathematics (Yue et al., 2023), interactive agents" (Sec. 5.4). They also report forgetting from SFT,
  citing Wang et al. 2023 (e.g. BBH "decreasing from 36.9 to 2.8" for SuperNI SFT; Sec. 5.1).

## Evidence and limitations

- Evidence: Fig. 2-4 (shift statistics), Tables 1-2 (URIAL vs tuned models; not checked in detail), App. C
  (more shift examples, not read).
- Rank analysis is on the aligned model's greedy output only, in the aligned model's own context; it does
  not ask whether the base would reach that context by itself (it would not at shifted positions). The η ≤ 3
  threshold is a choice; evaluation of URIAL relies on GPT-4 judging (Sec. 4.2).

## Connections and questions

- **Definition offered:** implicit — a behaviour's *knowledge* is in the base model if the tuned model's
  outputs are made of tokens the base already ranks top-1/top-3 in the same context; tuning that only
  moves a few (stylistic, early) positions changes access, not knowledge. Positive test: a fixed prompt
  (no weight change) reproduces the tuned behaviour.
- **New vs better access:** yes, qualitatively: "superficial" = few shifted positions + content at
  unshifted positions + reproducible by in-context prompting. Turned into a quantitative rule for us: for
  each RL proof y (e.g. r16's excluded-middle proof) decompose the log-likelihood ratio by position,
  Δ_t = log2 π_r16(y_t | ctx) − log2 π_pend(y_t | ctx). **Elicited** if the total "hint cost"
  H(y) = Σ_{t shifted} −log2 π_pend(y_t | ctx) is small (a few forking tokens, few bits) and pend completes
  y once those tokens are forced (prefix-forcing test); **created** if many positions are shifted or pend
  fails to continue even after forcing. This refines Dan's teacher-forced log-likelihood: it says *where*
  the bits are, and whether they sit in a few decisions (access) or throughout (content).
- **Null / floor:** none in the paper. For us, the random-init checkpoint ranks almost every token as
  "shifted" (≈ uniform over the vocabulary), giving the floor; the number of forced tokens a model needs
  before it completes a proof is a budget-free alternative to "any k eventually".
- **Transfer to our setting:** teacher-forced scoring of r16's (and r8's) accepted proofs under pend, r8,
  init: per-token rank, probability, KL; then prefix-forcing reads: force the first j shifted tokens and
  sample pend (k small), checking with Lean. Cost: scoring is seconds; prefix-forcing a few hundred
  samples per proof. Failure modes: our vocabulary is small, so rank ≤ 3 is a weak criterion — use bits;
  shifted positions in proofs are content (tactic choice), not style, so "few shifted tokens" does not
  automatically mean "superficial"; RL proofs are sampled at T 0.8, not greedy.
- Related: `xu2020usable.md` (usable vs Shannon information), `shenfeld2025razor.md` (KL to base on the
  new task), `mukherjee2025subnetworks.md` (parameter-side sparsity), `_screen_L3.md` rows for LIMA, LIMO,
  1-shot RLVR (few-example elicitation), Kadavath et al. 2022 and Eikema & Aziz 2020 (likelihood measures).
