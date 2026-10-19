# Run brief: kalmar-pretrain — can pretraining on Kalmár-construction proofs teach a model the algorithm, and does it generalise?

Run id `kalmar-pretrain`. Repository `~/work/kalmar-pretrain` (branch `dan_kalmar-pretrain` from the fork's
`origin/dan_create-or-elicit`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan
2026-10-18 (proposal 29).

- **Budget:** **$50 / 90 pod-hours.**
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 96 h.
- **Start:** queued to begin after `proof-strategies` is recorded.

## The question

Dan: "create a training set of the proofs this prover generates … pretrain the model on those proofs, with the same
approximate number of theorems in the pretrain set … I am curious if we can teach a model to use Kalmár's construction
by pretraining on such proofs, and whether that generalizes well."

The prover is nd-rl `dan:code/tools/tt_prover.py` (checkout `~/nd-rl`). For a sequent over P, Q, R, S it builds a
Kalmár proof:
1. case-split on excluded middle for every atom that occurs (X ∨ ¬X, by `Classical.byContradiction`);
2. in each of the ≤ 16 branches, prove every subformula or its negation by structural recursion;
3. close branches with a false premise by ex falso.

It is a fixed, uniform algorithm. In the proof-state interface, each step is a simple function of the visible state:
- which atoms are already split in this branch;
- the literals in scope;
- the current goal.

So a model that learns it locally could, in principle, prove *every* theorem in our domain. Whether it learns it, and
whether it keeps executing it over long horizons, is the question.

## The main obstacle: length

The prover's proofs grow about 2× per atom: roughly 10 / 28 / 70 / 200 constructor steps for 1 / 2 / 3 / 4 atoms. K12
proofs are 2–12 lines, and 24 % of K12 theorems have 4 atoms and 48 % have 3. A Kalmár K12 is therefore ~10–20× more
training tokens and far longer proofs than the model, its caps (96 steps, 512-token actions) or its pretraining length
budget have seen. Measure all of this first and design around it.

## Design (a suggestion)

1. **Data: K12-Kalmár.**
   - For K12's own 155,000 theorems, emit Kalmár proofs **as ND records** (prompt + ND proof in our token format), so
     the existing pipeline applies unchanged: `lean_seq` rendering and `state_env.decompose`.
   - Use one canonical variant: split atoms in the fixed order P, Q, R, S, as the prover does. You may also build a
     **shared-lemma variant**, deriving X ∨ ¬X once per atom at the top and then nesting ∨-eliminations, if it shortens
     proofs materially. Pick one as primary and say why.
   - **Check every proof with Lean** (the strict `lean_check` judge), and check that every record decomposes and
     replays in the environment.
   - Report length, depth and token statistics against K12's, and the longest (state, action) pair.
2. **Caps and context.** Set `max_steps`, `max_action` and the training sequence and token budget from the data, with
   margin. Report cut-off fractions per policy. ALiBi should extrapolate somewhat; measure, don't assume.
3. **Arms:** best-recipe Stage 1 (`state_train.py --recipe best`, 9.56M ALiBiGPT, `lean_staten`), 3 seeds:
   - **KAL:** Stage 1 on K12-Kalmár.
   - **Control:** the existing K12 bases `stage1_best12_s{0,1,2}_b1200` (`state_env_october` before RL). Reuse their
     stored reads where the protocol matches, and re-read them at the new caps where needed.

   **Training length** is the main design choice; pre-register it with its cost.
   - **Data-matched:** about the same passes over the data as K12 (≈ 3.7). The headline: did it learn the algorithm?
   - **Compute-matched:** K12's 1,200 s.

   Run the first if the budget allows, and the second as a secondary arm or on one seed. Estimate both costs before the
   first pod.
4. **Did it learn the construction?**
   - **Teacher-forced agreement:** on held-out theorems, the share of states where the model's greedy action equals the
     algorithm's action. Break it down by step type (atom split / excluded-middle derivation / leaf evaluation by
     connective / ex falso), by box depth, and by position in the proof.
   - **Free generation:** the share of sampled attempts that follow the algorithm exactly; where they deviate; and
     whether deviations are still valid proofs (shortcuts).
5. **Solve rates,** plain and guided, k 256, pass@1 / 8 / 256, and Lean alone, against the control at the same caps:
   - holdout250;
   - the long pools (lengths 13–16 and ≥ 17);
   - Charles's release (`candidate_v0`, `candidate_v1`, `batch3`, `hand_proved_14`);
   - textbook72 (**scoring only**).

   Break results down by number of atoms and by proof length, to show how per-step error compounds.
6. **Does it generalise?** Kalmár is uniform, so a model that learned it should handle shapes outside K12's
   distribution (still ≤ 4 atoms; the vocabulary has only P, Q, R, S). Test:
   - **more premises** (4–5; Charles's distractor construction or generated);
   - **formulas larger or deeper than K12's** (beyond its size cap of 15 and depth 2–3);
   - **theorems whose K12 proofs would need patterns K12 never shows** (e.g. the classical families from
     `create-or-elicit`).

   Report each against the control.
7. **Optional, if budget remains:** a short EI ladder (≤ 8 rounds) from the KAL base. Does RL keep the algorithm, or find
   shorter proofs? Pre-register before running.

## Pre-registration

Commit `preregistration/kalmar-pretrain.md` before the first pod. Write numeric expectations for:
- teacher-forced agreement;
- full-proof success by atom count;
- solve rates against the control;
- the generalisation tests.

Also pre-register what would count as "learned the construction" and as "generalises".

## Deliverables

- `run_kalmar_pretrain.md` (≤ 400 words + figures); `numbers.md` § kalmar-pretrain; `log.md`; compute rows.
- A plain-language summary of ≤ 800 words, readable by someone outside the subfield.
- The K12-Kalmár builder (with tests) and dataset statistics; `next_steps.md`.
- `STATUS.md` line `KALMAR-PRETRAIN DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/kalmar-pretrain/`,
  delete pods, and `touch ~/runs/kalmar-pretrain/executor.done`.

**For the reviewer:**
- Rebuild a sample of K12-Kalmár independently, and check it is Lean-valid and matches the algorithm.
- Recompute teacher-forced agreement and the solve tables.
- Check that caps did not truncate the comparison.

**Rules:**
- Do not contact anyone.
- textbook72 is scoring only, and no textbook72 theorem is printed.
- At most 4 concurrent processes on the VPS.
- Never run `test_run_once.sh`.
- Keep the RunPod balance above $35.
