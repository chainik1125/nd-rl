# Run brief: olmo-transplant — is Olmo 3.1 Think 32B's new instruction-following capability in its thinking?

Run id `olmo-transplant`. Repository `~/work/olmo-transplant` (branch `dan_olmo-transplant` from the fork's
`origin/dan_olmo-iffu`, made by `newrun`; `olmo-ifomega`'s pipeline, hand-gate rules and `olmo-iffu`'s results are on it).
Role: executor. `AGENT_POLICY.md` governs. Proposal 40 (nd-rl `docs/proposals/2026-11-07-olmo-followups.md`).
Authorised by Dan 2026-11-07, up to $60.

- **Budget:** **$60 / 40 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $5) measures prefill-heavy throughput. Re-cost, and pause and ask in `QUESTIONS.md` if the
  projection exceeds $55.
- **If over budget, cut in this order:** the fresh budget-forced generations, then budgets 1k and 16k, then the
  calibration prompts.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

On IFBench, Olmo 3.1 Think 32B (`step_2300`) succeeds on ≈ 20 % of prompts where its DPO start never does in 256
samples (`olmo-ifomega`); 10 of 10 tested survive 1,024 (`olmo-iffu`). RL writes 7k–55k characters of thinking before
answers of 4–90 words.

**Hypothesis (claim 4):** RL learned a *search procedure* in its chain of thought, not a better final-answer skill. If
so, the DPO model given RL's thinking should succeed, and RL's advantage should vanish when its thinking is cut short.

## Design (pre-register your version)

1. **Prompts:**
   - a seeded ≈ 20 IFBench hand-reach prompts (the review's list of 43), stratified by constraint family. Include the
     **6** that survived 1,024 DPO samples and were not hint-elicitable in `olmo-iffu`;
   - plus ≈ 10 IFBench prompts both models solve, as calibration.
2. **Reuse the stored thinking; do not regenerate it.**
   - **Source:** `olmo-ifomega` stored every sample's full text and token ids
     (`hf://buckets/dan-pandori/nd-rl/olmo-ifomega/artifacts/olmo_ifomega/samples/*.jsonl.gz`, fields `text` and
     `ids`; the text contains `</think>`).
   - **Traces:** per prompt, up to 4 of `s2300`'s **hand-labelled genuine** successes (from `handcheck.csv`), and up to 4
     of the DPO model's failures.
   - **Insertion:** both models share the Olmo 3 tokenizer and chat template. Build each input as the templated user
     turn + the trace up to `</think>`, using the stored token ids, not re-tokenised text. Then sample only the answer.
     Verify the template boundary on a few items by hand.
3. **Conditions:** sample 8 answers per prompt × trace × condition, at the Think RL decoding settings.

   | | thinking supplied | answer written by |
   |---|---|---|
   | T1, transplant | RL's genuine thinking | DPO |
   | T2, self | RL's genuine thinking | RL (sanity check: should be high) |
   | T3, reverse | DPO's failed thinking | RL |
   | T0, none | empty thinking | each model |
   | B, budget | RL's thinking cut at N ∈ {0, 1k, 2k, 4k, 8k, 16k tokens, full}, then `</think>` | each model |

   **Fresh budget forcing:** because B uses prefixes of *successful* traces, also run fresh budget-forced `s2300`
   generations (thinking capped at N, then `</think>` forced) at N = 2k and 8k, on 6 prompts × 16.
4. **Grading:** the strict checker, plus the hand gate (`olmo_ifomega/handcheck_rule_if.md`; blinded readers) on every
   success that a condition-level conclusion rests on.
5. **Analysis:**
   - per condition: success rates with prompt-bootstrap intervals;
   - T1 − T2 and T3 − T0, paired by prompt;
   - the B curve, per model, against N;
   - by constraint family.

## Pre-registered readings (state numeric thresholds before data)

- **T1 ≈ T2:** the capability is in the thinking (supports claim 4).
- **T1 ≪ T2 and T3 high:** RL changed answer writing, not search.
- **RL's B curve drops to the DPO level at small N while the DPO curve stays flat:** the win needs long search.
- **Flat curves:** a disposition, not search.

## Deliverables

- `run_olmo_transplant.md` (≤ 400 words + one figure: the condition bars and the B curves); `numbers.md` §
  olmo-transplant; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words.
- `olmo_transplant/`: code, the trace manifest (which stored samples, their hashes), `results.csv`, the hand-check
  table, and every sample's text in the bucket.
- A `STATUS.md` line `OLMO-TRANSPLANT DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-transplant/`,
  delete pods, and `touch ~/runs/olmo-transplant/executor.done`.

**For the reviewer:**
- Check that the inserted traces equal the stored ids.
- Check the template boundary on a sample.
- Re-grade the results, and repeat the hand gate blind on the deciding successes.
- Recompute the paired contrasts.

**Rules:**
- Do not contact anyone.
- Weights stay on pods.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
