# Run brief: olmo-think32b — does Olmo 3.1 Think 32B's long RL run reach problems its DPO start cannot, within 256 samples?

Run id `olmo-think32b`. Repository `~/work/olmo-think32b` (branch `dan_olmo-think32b` from the fork's
`origin/dan_olmo-rlzero-pilot`, made by `newrun`, so the pilot's code is here). Role: executor. `AGENT_POLICY.md`
governs. Authorised by Dan 2026-11-03 (proposal 35), up to $150.

- **Budget:** **$150 / 120 pod-hours.** Dollars are binding.
- **The balance floor binds first.** The RunPod balance was $170.54 at launch, and the $35 floor leaves about $135.
  Check `rpbalance` before every pod.
- **Stage gate:** stage 0 (≤ $8) measures 32B throughput. Then re-cost the whole design, and **pause and ask in
  `QUESTIONS.md`** if the projection exceeds the lesser of $140 and (balance − $45).
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h.

## Why (Dan, 2026-11-03)

The 7B pilot (`olmo-rlzero-pilot`; nd-rl
`experiment-summaries/2026-11-03-olmo-rlzero-pilot-…/README.md` and its `decision_32b.md`) looked like Yue et al.'s
low-RL picture. RL raised AIME pass@1 from 0.09 to 0.42 at 1.5–3.2 × 10⁻³ of pretraining compute. Yet RL's verifier
credited the base on all 17 problems RL newly solved, within 383 samples, and RL lost none. The pre-registered verdict
was "inconclusive".

Two weaknesses limit the pilot:
- AIME answers are integers, and only 10 of the 17 resolving base solutions were sound by hand;
- the confirmation rule was asymmetric.

This run asks the same question at Olmo 3.1 Think 32B. That is one continuous RL run from `Olmo-3-32B-Think-DPO`:
steps 1–750 are the original Olmo 3 Think 32B, steps 751–2,300 the extra 21 days (per the memo, checkpoint identities
checked). Its RL is ≈ 14 % of pretraining GPU-hours, ≈ 6 × 10⁻³ – 2 × 10⁻² in FLOPs (memo). It is a **screening
test**. If several problems stay out of the DPO model's reach at 256 samples, Dan will consider funding a larger
extension afterwards.

## Design (a suggestion; pre-register your version)

1. **Models:**
   - the RL start, `allenai/Olmo-3-32B-Think-DPO` (`main`);
   - `allenai/Olmo-3.1-32B-Think` at `step_0100` (the 7B pilot saw most of the change in the first 100 steps),
     `step_0750` (= Olmo 3 Think 32B's final) and `main` (= `step_2300`).

   Use the same chat template and the same decoding for all four: the Think RL recipe's sampling settings and its
   32,768-token response cap, from open-instruct. Record the commit.
2. **Problems:** AIME 2025 and AIME 2026, about 59, from the pilot's MathArena-checked manifest. Drop AIME 2024, as the
   memo advises.
   - Run a contamination scan against the Think RL data, and against the SFT and DPO data if reachable at reasonable
     cost. Exclude near-duplicates and report them.
3. **The fixed phase:** DPO at k = 16, and each RL checkpoint at k = 8, on every problem. Report unbiased pass@k with
   problem-bootstrap intervals, and the matched-k entries and exits relative to the DPO model.
4. **Extension, symmetric:**
   - **Reach candidates** are problems some RL checkpoint solves and the DPO model has not. Sample the DPO model in
     chunks until its **second** credited success, up to 256 samples.
   - **Reverse candidates** are problems the DPO model solves and `step_2300` has not. Treat them the same way with
     `step_2300`.
   - Apply the same confirmation rule to every model: ≥ 2 successes, including one from fresh samples.
5. **Grades:**
   - **primary:** AI2's verifier as RL ran it (`MathVerifier`, open-instruct);
   - **secondary (pre-registered):** final-stated-answer only;
   - **a hand-check** (blinded sub-agent reading, confirmed by you) of every solution that decides a candidate, on both
     sides. Label each sound, right answer through a flawed step, or guess.

   Report the verdict under all three.
6. **Compute axis:** cumulative RL ÷ pretraining at each checkpoint, in FLOPs (low and high) and in GPU-hours, from the
   memo's inputs and the Think RL script.
7. **Likelihood (secondary, if budget remains):** the DPO model's per-token and whole-solution log-likelihood of RL's
   resolving solutions, against its own.
8. **Engineering, following the memo:**
   - **GPUs:** 1 × H200 141 GB, or 2 × 80 GB with TP 2.
   - **Speculation:** n-gram speculative decoding only after the same equivalence gate the pilot used; an FP8 KV
     cache only if gated the same way.
   - **Weights:** download them on pods only, one model at a time, or size the disk for four 64.5 GB models.
   - **Outputs:** pull them off each pod as each job ends. Store every sample's text.

## Pre-registration

Commit `preregistration/olmo-think32b.md` after stage 0 and before the fixed phase. Write expectations for:
- pass@1 and pass@8 of each model;
- the number of reach and reverse candidates;
- how many candidates the DPO model resolves within 256 samples, under each grade.

**Verdict rule** (adjust and justify before data):
- **"Reach candidates for scale-up":** ≥ 3 problems where `step_2300` is confirmed and the DPO model has no sound
  success within 256.
- **"No reach at 256":** every candidate is confirmed for the DPO model under the final-answer grade, with sound
  solutions by hand.
- **"Inconclusive":** anything else.

## Deliverables

- `run_olmo_think32b.md` (≤ 400 words + figures); `numbers.md` § olmo-think32b; `log.md`; compute rows.
- A plain-language summary of ≤ 600 words.
- `olmo32/`: the code, the problem manifest with contamination flags, `results.csv` (one row per model × problem), the
  hand-check table, and every sample's text in the bucket.
- One figure: entries, exits and reach against cumulative RL ÷ pretraining compute, on the same axis as the 7B pilot
  and the σ figure, with the pilot's points shown for comparison.
- `scaleup_memo.md`: if candidates remain, which problems they are and what extending the DPO model to 1,024 or 2,048
  samples would cost, at this run's measured prices.
- A `STATUS.md` line `OLMO-THINK32B DONE <UTC>`; then upload code and results (not weights) to
  `hf://buckets/dan-pandori/nd-rl/olmo-think32b/`, delete pods, and `touch ~/runs/olmo-think32b/executor.done`.

**For the reviewer:**
- Rebuild the per-cell table from the raw rows.
- Re-grade the stored texts with the verifier.
- Repeat the hand-check on every candidate-deciding solution.
- Check the symmetric confirmation, and that decoding and templates match across models.
- Recompute the compute axis.

**Rules:**
- Do not contact anyone.
- Weights stay on pods, never on the VPS: ≈ 13 GB free.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Do not edit `references.bib` or run `code/tools/literature.py`.
