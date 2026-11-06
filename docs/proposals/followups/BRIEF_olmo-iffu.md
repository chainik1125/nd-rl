# Run brief: olmo-iffu — is olmo-ifomega's "reach" a token budget, a brevity habit, or a sampling budget?

Run id `olmo-iffu`. Repository `~/work/olmo-iffu` (branch `dan_olmo-iffu` from the fork's `origin/dan_olmo-ifomega`,
made by `newrun`, so that run's pipeline, manifests, hand-check rules and stored results are here). Role: executor.
`AGENT_POLICY.md` governs. Proposal 37, up to $60.

- **Budget:** **$60 / 40 pod-hours.** Dollars are binding.
- **Stage gate:** re-cost after the first job's measured throughput, and **pause and ask in `QUESTIONS.md`** if the
  projection exceeds $55.
- **If over budget, cut in this order:** test 4 (optional), then test 3's subset from 10 to 6 prompts. Tests 1 and 2
  are never cut.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## Why (Dan, 2026-11-06)

`olmo-ifomega` (nd-rl `experiment-summaries/2026-11-04-olmo-ifomega-…/README.md`, its review, and `scaleup_memo.md`)
found Olmo 3.1 Think 32B's RL (`step_2300`) confirmed where its DPO start had no success in 256 samples:
- IFBench, 49 prompts strict and 43 by the review's hand check;
- IFEval, 6 prompts;
- OMEGA `test_out`, only on perpendicular-diagonal geometry problems where the DPO model is cut at the 32,768-token cap
  in 99 % of its failures.

Three alternative explanations remain. These are the memo's three cheapest decisive tests, with the review's
specification of the first:
- the DPO model needs more room, not new ability;
- the RL model's IF wins are a brevity habit: its answers are 5–15× shorter on the IFEval reach prompts;
- 256 samples are too few.

Under the project's capability standard (nd-rl `docs/project_strategy/capabilities/REPORT.md`), "no success in 256"
means "not reached at 256", not "created". This run tightens that.

## Tests (pre-register each before its first pod job)

1. **Token budget on the geometry reach prompts.**
   - Sample `dpo` on perp_055 (with its label corrected to 15), perp_068 and perp_069 with the **largest response
     budget the context allows**: `max_position_embeddings` is 65,536 including the prompt, so 65,536 minus the prompt
     length. Take 64 samples each, with the same template and decoding otherwise.
   - Grade with both `omega` and RL's `rl` verifier, and hand-read every credited response, `rl`-only ones included
     (the review's conditions).
   - Report the cut share, the length distribution of finished responses, and successes per prompt.
   - **Outcome:** if `dpo` reaches ≥ 2 sound successes on a prompt, that prompt's "reach" was the token budget.
2. **Brevity control on IF reach.**
   - Sample `dpo` with a length hint appended to the user turn, using one wording fixed in the pre-registration, for
     example "Answer in at most 60 words."
   - Prompts: the three letter-avoidance IFEval reach prompts, and a seeded 10-prompt subset of the IFBench
     hand-reach prompts. Take 64 samples each.
   - Use the strict checker plus the hand gate (`olmo_ifomega/handcheck_rule_if.md`), reading every deciding success
     for gaming.
   - Also run the hinted prompt on `s2300` at 16 samples, to show the hint does not break it.
   - **Outcome:** report how many reach prompts become solvable by the DPO model with the hint (≥ 2 genuine). Label
     those "prompt-elicitable (brevity)". Under the capability standard a prompt change is an elicitation method, so
     say how that changes each verdict.
3. **Deeper sampling on IFBench reach.**
   - Take a seeded 10-prompt subset of the 43 hand-reach prompts (by the review's list), disjoint from test 2's subset
     if possible. Sample `dpo` to **1,024** per prompt (768 more each), in chunks, stopping a prompt at 2 genuine
     successes.
   - Use the strict checker plus the hand gate on every credited success.
   - **Outcome:** report how many of the 10 survive at 1,024, with a binomial upper bound on `dpo`'s per-sample rate for
     the survivors.
4. **Optional, if ≥ $8 remains: `step_0750`'s reach.** Confirm `step_0750` on the IFBench hand-reach set (fresh chunks
   of 16 until 2 genuine), giving the compute axis an intermediate point. The memo estimates $5–7.

## Rules for interpretation

- Keep the pre-registered verdicts of `olmo-ifomega` unchanged. Report these as follow-up measurements, and say for
  each reach prompt which explanation, if any, it now supports: token budget, prompt-elicitable, still out of reach at
  1,024, or untested.
- The hand gate decides. A strict-checker success that reads as gamed does not count.
- Watch rounding-boundary and label errors, as the memo warns. Check every deciding label.

## Deliverables

- `run_olmo_iffu.md` (≤ 400 words + one figure: per prompt, the status before and after each test);
  `numbers.md` § olmo-iffu; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words.
- `olmo_iffu/`: the code, the manifests (which prompts, their seeds), `results.csv`, the hand-check table, and every
  sample's text in the bucket.
- A `STATUS.md` line `OLMO-IFFU DONE <UTC>`; then upload code and results (not weights) to
  `hf://buckets/dan-pandori/nd-rl/olmo-iffu/`, delete pods, and `touch ~/runs/olmo-iffu/executor.done`.

**For the reviewer:**
- Re-grade the stored texts, and repeat the hand gate blind on every deciding response.
- Check the token budget actually used and the hint wording.
- Check that the subsets match their seeds.

**Rules:**
- Do not contact anyone.
- Weights stay on pods, never on the VPS.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long-running local poll rather than many short check-ins; each check-in counts toward the
  driver's invocation cap.
- Do not edit `references.bib` or run `code/tools/literature.py`.
