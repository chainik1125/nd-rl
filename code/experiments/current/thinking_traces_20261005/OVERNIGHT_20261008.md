---
written_on: 2026-10-08
written_by: agent:claude
---

# Overnight log, 2026-10-08 (Leon asleep; decisions delegated)

Leon (about 1:15 am PDT): "you have my permission to start experiments and make decisions on your own … when I get
back tell me what you chose in a tldr and a summary of what worked and what didn't."

Times are PDT. Each entry is a decision or a result, with its reason.

## Decisions

- **1:20 am.** Gap analysis done ([gap_analysis/README.md](gap_analysis/README.md)). One unit-test theorem was a
  renaming of a textbook72 problem, so I replaced it. textbook72 must never be trained or tuned on, even in a test
  string.
- **1:25 am.** Started building the proof-first hard 7–12-line generator (`hard712/`) from gap-analysis targets
  T1–T8.
  - The gates: the shortest proof must be 7–12 lines and within 1 line of the constructed proof; the targeted
    structure must be necessary; Lean; exclusion against every set.
  - The sealed test split is an unfiltered random sample, drawn before any model-based filtering.

- **2:10 am. Plan review in** ([PLAN_CRITIQUE.md](PLAN_CRITIQUE.md); Opus plus four research agents, about 70 papers).
  I adopted:
  - **Step validity first.** One invalid step ends 60–92% of failed attempts on textbook 7–12s and on 13–20. So I ran
    the review's first experiment: read the RL'd checkpoints with the `logical` arm, which redraws a step the
    environment rejects, and compare with plain sampling at matched tokens.
    - If logical − plain is at least +5 pp at 2,000 tokens per problem, step validity is the binding constraint, and
      the trace work re-centres on checking the policy's own candidates.
    - If it is at most +1 pp, the search-trace plan goes ahead unchanged.
  - **Ladder:** drop e16 (default 0/64/256). e1024 is gated on evidence of need; none is observed so far.
  - **Search traces get a matched no-think control (T0-S)** trained on the same found proofs, so proof-style changes
    don't masquerade as a thinking gain. They also get CPU go/no-go gates, the key one being the share of genuine
    dead ends.
  - **Model-decided effort is deferred, not dropped (Leon asked for it).**
    - An oracle that picks the best level per problem beats the best fixed level by only −1.4 to +2.2 pp pass@1, but
      with 30–65% fewer tokens.
    - A free verify-and-escalate cascade (e0 → e64 → e256, stop at the first Lean-verified proof) matches the oracle
      above 2,000 tokens per problem.
    - The per-step `min_level` labels are still being built; auto-mode training waits on the step-validity result.
  - **Distractor premises dropped.** Textbook problems have none, and the pool already has 1.2 redundant premises
    per problem.
  - **Noted for RL:**
    - the pool's signal is gone: 62–74% of groups were skipped at it78;
    - add a coverage-preserving objective, because GRPO leaves 3.5× as many hard12 test targets at 0/16 as EI;
    - the composition pipeline's 13–15-line composites are in band 18–37% of the time;
    - a STaR-style arm could use the 105k verified EI attempts on disk.
  - **A new textbook-like test set is needed,** because dev58 is now a design set. Candidate: `transfer_long`'s 282
    textbook-schema theorems, which have 0 renaming overlaps with textbook72; schema overlap still to check.
- **2:15 am. GPU 0:** logical arm on 13–20 for graded GRPO, GRPO no-think seeds 0 and 1, GRPO effort seed 1, and EI
  thinking r2 (about 3 h serial). The textbook72 reads follow on GPUs 1–4 once the EI round 3–4 evals finish.

## Results

- **21–30-line sealed set (2,767 theorems, k 8), final.** pass@1 / pass@8, 95% intervals; RL'd checkpoints at it78
  or r2.

  | model | pass@1 | pass@8 | tokens / attempt |
  |---|---|---|---|
  | Stage-1 T0 | 0.0 | 0.0 | 227 |
  | EI no-think r2 | 0.2 [0.1, 0.3] | 1.0 [0.7, 1.4] | 314 |
  | GRPO no-think s0 / s1 | 0.2 / 0.3 | 0.9 / 0.8 | 300 |
  | **EI thinking r2** | **1.2 [1.0, 1.5]** | **5.2 [4.4, 6.1]** | 1,691 |
  | GRPO effort s0 / s1 | 0.2 / 0.5 | 0.7 / 2.4 | 750 |
  | GRPO token cost | 0.3 | 1.4 | 531 |
  | GRPO graded | 0.8 [0.6, 1.0] | 3.9 [3.3, 4.7] | 1,144 |

  - Paired differences:
    - EI thinking − GRPO graded: +0.4 [+0.2, +0.7] pass@1, +1.3 [+0.4, +2.2] pass@8.
    - Thinking − no-think: EI +1.0 / +4.2; GRPO graded vs no-think +0.6 / +3.1; GRPO effort seed 1 +0.2 / +1.6.
  - **Read:** on the longest proofs EI with thinking is the best model. Thinking is worth more than the choice of RL
    algorithm, and the models that think longest do best.
  - **Interpretation, not tested:** GRPO's shrinking of think blocks may cost it on long horizons.
- **2:00 am.** GPUs 1–4: EI rounds 3 and 4 (thinking at B 256, no-think at B 0) on 13–20 and 21–30, to see whether EI
  keeps climbing out of distribution. These are cheap evals (about 1 h), with no matched GRPO partner beyond it78.
  GPU 0 is kept free for the collapse-audit confirmation.
