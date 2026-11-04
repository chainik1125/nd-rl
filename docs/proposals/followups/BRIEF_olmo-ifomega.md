# Run brief: olmo-ifomega — does Olmo 3.1 Think 32B's RL create reach where RL actually moved it: instruction following (IFBench) and OMEGA's held-out and harder maths?

Run id `olmo-ifomega`. Repository `~/work/olmo-ifomega` (branch `dan_olmo-ifomega` from the fork's
`origin/dan_olmo-think32b`, made by `newrun`, so that run's pipeline is here). Role: executor. `AGENT_POLICY.md`
governs. Proposal 36, up to $150.

- **Budget:** **$150 / 120 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $8) measures response lengths and throughput on each task family. Then re-cost the
  design, and **pause and ask in `QUESTIONS.md`** if the projection exceeds the lesser of $140 and (balance − $45).
  The balance was $254.10 when the brief was written.
- **If the projection is over budget, cut in this order:** IFEval first, then OMEGA `test_in`, then step_0100. Keep
  IFBench and OMEGA `test_out` whole.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h.

## Why (Dan, 2026-11-04)

`olmo-think32b` (nd-rl `experiment-summaries/2026-11-03-olmo-think32b-…/README.md`) tested AIME 2025–26:
- it found one reach candidate (aime2025_28) and a verdict of "inconclusive";
- but RL barely moved AIME: pass@1 went from 0.716 to 0.746.

So AIME is a weak probe of what RL installed. Dan asked for problems closer to what was RL'ed on.

`Dolci-Think-RL-32B` (102,026 prompts) holds:
- 29.8k precise instruction-following prompts, the largest source;
- 30.2k maths prompts, 15k of them OMEGA;
- 21.3k code and 20.7k general chat.

AI2's model-card evals (DPO → step 750 → step 2,300) show where RL moved the model:

| benchmark | DPO | step 750 | step 2,300 |
|---|---|---|---|
| IFBench | 34.4 | 47.6 | 68.1 (**+34**) |
| IFEval | 80.6 | 89.0 | 93.8 (+13) |
| OMEGA | 45.2 | 50.8 | 53.4 (+8) |
| AIME 2025 | 70.7 | 72.5 | 78.1 (+7) |

## Data (verify each)

- **IFBench:** `allenai/IFBench_test`, 300 prompts with new, out-of-domain constraints ("Generalizing Verifiable
  Instruction Following").
  - **Grade:** its official constraint checkers. Use **strict prompt-level** success (every constraint satisfied) as
    the primary grade, and also report per-constraint success. Use AI2's code (the IFBench repo or open-instruct's IF
    reward) and record the commit.
- **IFEval (in-distribution control):** `google/IFEval`, a seeded 150–300 prompt subset, with its official strict
  checker. Its constraint types overlap RL's IF training data, which is the point of the control.
- **OMEGA:** `allenai/omega-explorative`, about 41 families with `train`, `test_in` and `test_out` splits.
  - `test_in` is held-out problems at the training difficulty; `test_out` is harder, out-of-distribution difficulty.
  - Take a seeded sample of about 8–12 families × about 10 problems × both splits. Pick families from those in RL's OMEGA
    prompts, and record which.
  - **Grade:** OMEGA's own answer checker, and RL's maths verifier as a second grade. Add `omega-compositional` only
    if budget remains.
- **Contamination:** scan every eval prompt against all 102,026 `Dolci-Think-RL-32B` prompts. Watch especially WildChat
  English and IF Multi-Constraint against IFBench and IFEval, and OMEGA `train` against `test_in`. Scan against the
  Think SFT and DPO prompts too, if that is affordable. Exclude near-duplicates and report them.

## Design (follow `olmo-think32b` unless there is a reason not to; pre-register)

1. **Models:** `allenai/Olmo-3-32B-Think-DPO` `main` (the RL start), and `allenai/Olmo-3.1-32B-Think` at `step_0100`,
   `step_0750` and `main` (= `step_2300`). Use the same chat template and decoding as `olmo-think32b`.
2. **Fixed phase:** DPO at k = 16 and each RL checkpoint at k = 8 on every prompt. Report pass@k with
   problem-bootstrap intervals, and the matched-k entries and exits relative to the DPO model, per task.
3. **Extension, symmetric, as in `olmo-think32b`:**
   - **Reach candidates** (some RL checkpoint succeeds, the DPO model never has): sample the DPO model to its
     **second** success, up to 256.
   - **Reverse candidates:** sample `step_2300` the same way.
   - Confirm with ≥ 2 successes including a fresh one, for every model.
4. **Hand-check every deciding response.** For IF, read for constraint gaming: degenerate text that passes the checker
   but ignores the request. Label it gamed or genuine. For OMEGA, label it sound, a flawed step, or a guess.
5. **Compute axis:** cumulative RL ÷ pretraining (FLOPs low and high, and GPU-hours), as in `olmo-think32b`.
6. **Per-constraint analysis (IF):** for each reach or reverse prompt, say which constraint the DPO model fails, and
   whether that constraint type appears in RL's IF data.

## Pre-registration

Commit `preregistration/olmo-ifomega.md` after stage 0 and before the fixed phase. Per task (IFBench, IFEval, OMEGA
`test_in`, OMEGA `test_out`), write expectations for:
- pass@1 and pass@8 for each model;
- the number of reach and reverse candidates;
- how many the DPO model resolves within 256.

**Verdict rule, per task** (adjust and justify before data):
- **"Reach":** ≥ 3 prompts where `step_2300` is confirmed and the DPO model has no genuine success within 256.
- **"No reach at 256":** every candidate is resolved for the DPO model under the primary grade and by hand.
- **"Inconclusive":** anything else.

## Deliverables

- `run_olmo_ifomega.md` (≤ 400 words + figures); `numbers.md` § olmo-ifomega; `log.md`; compute rows.
- A plain-language summary of ≤ 600 words.
- `olmo_ifomega/`: the code, the prompt manifest with contamination flags, `results.csv` (one row per model × prompt),
  the hand-check table, and every sample's text in the bucket.
- One figure: reach, entries and exits against cumulative RL ÷ pretraining compute, one panel per task, with
  `olmo-think32b`'s AIME points for comparison.
- `scaleup_memo.md`: any surviving candidates, and what a larger extension would cost at this run's measured prices.
- A `STATUS.md` line `OLMO-IFOMEGA DONE <UTC>`; then upload code and results (not weights) to
  `hf://buckets/dan-pandori/nd-rl/olmo-ifomega/`, delete pods, and `touch ~/runs/olmo-ifomega/executor.done`.

**For the reviewer:**
- Rebuild the cell table from raw rows, and re-grade the stored texts with each grader.
- Repeat the hand-check on every deciding response, blind.
- Check the contamination exclusions, and that decoding and templates match across models.
- Recompute the compute axis.

**Rules:**
- Do not contact anyone.
- Weights stay on pods, never on the VPS: ≈ 13–14 GB free.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Do not edit `references.bib` or run `code/tools/literature.py`.
