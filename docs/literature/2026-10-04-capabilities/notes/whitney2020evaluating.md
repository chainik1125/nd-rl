---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - whitney2020evaluating
---

# Evaluating representations by the complexity of learning low-loss predictors (SDL, ε sample complexity)

Paper: [@whitney2020evaluating] (Whitney, Song, Bu, Cho, 2020)
Source: arXiv 2009.07368v2 (HTML rendering read: abstract, Sec. 1-4, 5.2, 6-7; appendices A-C not read)

## Learnings

- **Position.** "We propose to measure the quality of a representation by the complexity of learning a
  predictor on top of the representation that achieves low loss on a task of interest" (Abstract); "the best
  representation is the one which allows for the most efficient learning of a predictor to solve the task"
  (Sec. 1), with efficiency in samples or in information.
- **Loss-data curve unifies prior measures.** Plot expected validation loss L(A_φ, n) of a probe trained on
  n examples against n. Validation accuracy = a point at fixed n; MI = the limit n → ∞; MDL (prequential)
  = the area under the curve on [0, n] (Sec. 2.2, Eq. 3, 6, 8). "prior methods measure properties of an
  evaluation dataset of a specified size, whereas our methods measure properties of a predictor with a
  specified loss" (Abstract).
- **Problems with prior measures.** MDL ≥ n·H(Y|φ(X)), so "MDL grows without bound as the size of the
  evaluation dataset" grows (Sec. 4.1, Eq. 10). VA and MDL rankings change with n: in a parity example the
  ranking "favors the noisy label representation when" n ≪ d but the raw data when n ≫ d (Sec. 3.1). "MI is
  insensitive to statistical complexity" and "MI is insensitive to computational complexity" (Sec. 3.2).
  "All three prior methods lack a predefined notion of successfully solving a task" (Sec. 3.3).
- **Surplus description length** (Def. 2, Eq. 12): m_SDL(φ, D, A, ε) = Σ_{i≥1} [L(A_φ, i) − ε]_+, the area
  between the loss-data curve and the tolerance line; "it gives the cost (in terms of information) for
  re-creating an ε-loss predictor when using the representation" (Sec. 4.1).
- **ε sample complexity** (Def. 3, Eq. 13): min{n : L(A_φ, n) ≤ ε}; it "measures the complexity of learning
  an ε-loss predictor by the number of samples it takes to find it" (Sec. 4.2); any monotone objective
  (e.g. accuracy) may replace log-loss (Sec. 4.2).
- **Honest lower bounds.** If ε is unreachable with the data, "an implementation is able to report that the
  given complexity estimate is only a lower bound" (Sec. 4.1); Table 2 reports "> 461" etc.
- **Choosing ε.** It "can be done by training a large model on the raw representation of the full
  evaluation dataset and using its validation loss as" ε (Sec. 4.3); in applications ε is a design
  specification (e.g. 80% per-frame accuracy).
- **Example (ELMo, PoS; Table 2).** At n = 461, MDL = 884.54 / 1009.26 / 1017.72 (layer 0 best); at
  n = 474,838, MDL = 92403.41 / 52648.50 / 65468.54 (layer 1 best). SDL (ε = 0.1) = > 40882.72 / 2765.11 /
  7069.56 and εSC = > 474838 / 237967 / 474838. Cost: "evaluating our measures to high precision took about
  an hour on one GPU" (Sec. 5.2).
- **Caveat stated.** "Each of these measures depends on a choice of algorithm" A, including the probe
  architecture (Sec. 7).

## Evidence and limitations

- Evidence: Table 1 (small tasks, not read in detail), Table 2 and Fig. 4 (PoS with ELMo layers). Monotone
  improvement in n is assumed (Sec. 4.1-4.2). Appendices with the estimation algorithms and the data-
  requirement theorem not read.
- Representations are frozen and probed; nothing about fine-tuning the whole network or RL.

## Connections and questions

- **Definition offered:** quality of a representation for a task = the cost of *re-creating* a predictor
  that meets a stated success criterion ε: in examples (εSC) or in surplus bits (SDL).
- **New vs better access:** not about RL, but this is the most directly usable "elicitation cost" rule we
  found. Fix the capability as a success criterion (e.g. pass@1 ≥ τ, or held-out NLL ≤ ε, on held-out
  instances of the excluded-middle family, with ε = r16's level), fix the learning algorithm (fine-tune of
  the checkpoint on n demonstrations), and measure εSC and SDL starting from init, pretraining checkpoints
  and pend. **Elicited** if εSC(pend) is a handful of demonstrations and SDL(pend) a few bits; **created**
  if εSC(pend) ≈ εSC(init) (the base offers no head start beyond syntax). Because ε is explicit, the method
  can answer "not reached with the data we have" (lower bound) instead of forcing a verdict.
- **Null / floor:** the random-init (or raw-input) representation is the natural reference; the authors
  cite that capacity or data restrictions "are necessary to separate the performance of randomly- and
  linguistically-pretrained representations" (Sec. 6). The "any k" objection does not arise: the measure
  is samples/bits to reach a criterion, and init's εSC is a finite, large number to compare with.
- **Transfer to our setting:** demonstrations = Lean-accepted excluded-middle proofs from `rl_targets`
  (never textbook72), held-out evaluation = other members of the family; n on a log grid 1, 2, 4, ...,
  256; 3 seeds per point; criterion = pass@1 (sampled, Lean-checked) or teacher-forced NLL. Cost: ~9 grid
  points × 3 seeds × ~4 starting checkpoints ≈ 100 short fine-tunes + reads; a few GPU-hours — matches
  pre-registered job J4. Failure modes: the result depends on the fine-tune recipe (lr, epochs, LoRA vs
  full) — sweep at least two; tiny n is dominated by variance; demonstrations that are renamings of the
  evaluation theorems leak; choosing ε from r16 ties the criterion to one seed's behaviour.
- Related: `voita2020mdlprobing.md` and `blier2018description.md` (the MDL quantities criticised here),
  `greenblatt2024passwordlocked.md` (unlocking sample counts), `aghajanyan2020intrinsic.md` (size rather
  than number of samples), `xu2020usable.md` (MI vs bounded families).
