# Run brief: style-edl — does EDL rise with proof style when capability is fixed?

Run id `style-edl`. Repository `~/work/style-edl` (branch `dan_style-edl` from the fork's `origin/dan_kalmar-complexity`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-10-22 (proposal 32).

- **Budget:** **$10 / 40 pod-hours.** Dollars are binding. Scoring and fine-tuning a 9.56M model is cheap: use one
  community RTX 3090 (≈ $0.22/h) or the VPS CPU. Pods are mainly for sampling.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question (Dan, 2026-10-22)

Excess description length (EDL) [@donoway2026excessdescriptionlengthlearning] scores specific proof strings. So it
may charge a model for *which* valid proof it is taught (style), even when the model can already prove the theorem
(capability). If that is so, an EDL gap between a pretrained and an RL model can come from style alone. Show directly
whether EDL rises with style distance while capability stays fixed.

Background: nd-rl `docs/project_strategy/2026-10-22-edl-style-vs-capability.md`, which links the team's EDL work and the
evidence so far. The closest evidence is kalmar-pretrain's K12 control, which proves 90–94 % of tf1200 greedily but
agrees with the Kalmár proof's next action on only 23–30 % of states.

## Design (a suggestion; pre-register your version)

1. **Learners:** the K12 bases `stage1_best12_s{0,1,2}_b1200`, the bases kalmar-pretrain used as its control.
2. **Theorems:** for each learner seed, theorems it already proves.
   - Draw from tf1200, kalmar-pretrain's 1,200 held-out theorems, none in a K12 renaming class.
   - Keep those with pass@32 ≥ 0.9 (T 0.8) for that learner.
   - Take 256 of them, split by renaming class into a 128-theorem stream and a 128-theorem test set.
   - Report the atom-count mix of the kept theorems.
3. **Streams:** each stream covers the same theorems, with one Lean-checked proof per theorem.

   | stream | target proofs | role |
   |---|---|---|
   | S_own | the learner's own greedy proofs | gate: expected EDL ≈ 0 |
   | S_own2 | a second, independent T 0.8 sample of the learner's own valid proofs | gate: the noise floor |
   | S_seed | another base seed's greedy proofs | mild style change |
   | S_short | the shortest proofs from `minlen.Search` (bound 14 lines), as in `proof_strategies/search_proofs.py` on the fork branch `dan_proof-strategies` | moderate |
   | S_RL | an EI ladder checkpoint's valid proofs, from round 8 or later | the case in question |
   | S_Kalmár | K12-Kalmár sl proofs (`kalmar/`) | large (secondary; see below) |

   - **S_RL:** find the EI checkpoints from the same K12 base (`la_T1_best12_s*`, from best-state), and say which you
     used. If they are missing, use create-or-elicit's full-arm ladders.
   - **S_Kalmár:** only theorems whose Kalmár proof fits the learner's caps. Report it per action as well as per proof.
4. **Scoring:** teacher-forced NLL over action tokens only (`state_train.loss_on`), in bits. Also report each stream's
   mean proof length in actions and in tokens.
5. **EDL:**
   - **Protocol:** prequential blocks at n ∈ {0, 8, 16, 32, 64, 128}. For each n, retrain from the learner's weights on
     the first n proofs with one fixed algorithm A, then code the next block.
   - **Final loss:** L_test is θ*'s loss on the 128 test proofs *of the same stream*. EDL = Σ block bits − 128 × L_test.
   - **Algorithm A:** Charles's pilot found full fine-tuning peaks at about 2 passes and then overfits. Use early
     stopping, or a fixed pass count chosen on S_own before any other stream is scored.
   - **Learning rate:** pick from 2–3 values by online code length, and add log₂(number of rates) bits for the choice.
   - **Report:** EDL per proof and per action, the starting loss (the n = 0 block) and the per-block curve.
   - **Starting points:** Charles's `edl_mock` (nd-rl `origin/charles/main`: `code/finetuning/edl_mock.py`,
     `experiments/finetuning/edl_mock/proposal.md`), Leon's `code/ndbench/edl.py` (nd-rl `origin/main`) and this
     branch's `kalmar_complexity/b2_preq.py`.
6. **Capability check:** measure pass@32 (T 0.8) on the 128 test theorems before training, and after training θ* on
   each stream. "Capability fixed" means pass@32 stays ≥ 0.9, and within 0.03 of before, on every stream.
7. **A score that ignores style (secondary):**
   - **ℓ_V:** −log Σ_{y ∈ V(x)} p(y | x), where V(x) is every distinct valid proof of x that the run collected, from all
     streams plus any extra valid samples.
   - Recompute each stream's EDL with ℓ_V in place of the single-string loss, keeping the same training.
   - If style is what inflates EDL, the stream-to-stream differences should mostly disappear under ℓ_V.

## Pre-registration

Commit `preregistration/style-edl.md` before the first pod, and before scoring any stream other than the S_own pilot.
Write numeric expectations for:
- EDL per proof on S_own and S_own2;
- the ordering and size of EDL across S_seed, S_short, S_RL and S_Kalmár;
- pass@32 after training on each stream;
- how much of the spread ℓ_V removes.

Also write down what result would count as "EDL is inflated by style at fixed capability", and what would count against
it.

## Deliverables

- `run_style_edl.md` (≤ 400 words + figures); `numbers.md` § style-edl; `log.md`; compute rows.
- A plain-language summary of ≤ 600 words.
- `style_edl/`: stream builders, the EDL runner, `streams_manifest.json` (theorem ids, sources and hashes) and
  `results.csv` (one row per learner seed × stream: EDL per proof and per action, starting loss, L_test, pass@32 before
  and after, mean length, and the ℓ_V version).
- One figure: EDL per proof by stream, each learner seed marked, with pass@32 after training alongside.
- A `STATUS.md` line `STYLE-EDL DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/style-edl/`, delete pods,
  and `touch ~/runs/style-edl/executor.done`.

**For the reviewer:**
- Rebuild one stream per seed independently, and check that every proof is valid under Lean.
- Recompute EDL from the stored per-block losses.
- Check that the test sets share no renaming class with the streams.
- Check that pass@32 was read on the test theorems with the trained θ*.

**Rules:**
- Do not contact anyone.
- textbook72 is scoring only, and no textbook72 theorem is printed.
- At most 4 concurrent processes on the VPS.
- Never run `test_run_once.sh`.
- Keep the RunPod balance above $35.
- Do not edit `references.bib` or run `code/tools/literature.py`.
