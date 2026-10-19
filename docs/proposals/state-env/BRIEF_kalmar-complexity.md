# Run brief: kalmar-complexity — how many bits is Kalmár's construction, and which measure of that predicts how hard it is to learn?

Run id `kalmar-complexity`. Repository `~/work/kalmar-complexity` (branch `dan_kalmar-complexity` from the fork's
`origin/dan_kalmar-pretrain`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan
2026-10-19 (proposal 31).

- **Budget:** **$15 / 40 pod-hours.** Dollars are binding. Most of the work is CPU and Claude time; pods are only for
  the optional small training runs in Q3 (B2).
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h.
- **Start:** queued on agent1 to begin after `kalmar-pretrain` is recorded, so that its builder, training logs and
  reviewed numbers are available.

## The question (Dan, 2026-10-19)

What is the minimum description length of Kalmár's constructive proof algorithm? Dan asks this to get at "how many bits
of information would we realistically have to instill into a model to learn this algorithm". The length of a program
for a Turing machine seems to point at how complicated the algorithm is to learn. Dan holds this loosely and invites
pushback: MDL may be the wrong measure and another one better. Find out which.

**Out of scope:** how many bits pretraining and RL can each supply. That is a separate, later investigation. Report
everything in bits so that it can use these numbers.

## Why one program length is not enough

These are cautions to address, not conclusions.

1. **It depends on the language.** Kolmogorov complexity is defined only up to an additive constant: the size of an
   interpreter from one language to another. For a program of about 1 kB that constant can be as large as the quantity
   itself; the core of `code/tools/tt_prover.py` is 3.5 kB of Python, or 1.3 kB gzipped. So absolute numbers mean
   little, and comparisons within one fixed language mean more.
2. **The relevant quantity is conditional.** What has to be instilled is the description length *given what the learner
   already has*: its architecture, its inductive biases and, for a pretrained model, what it already knows. For our
   models that means a transformer's view of the algorithm, not a Turing machine's.
3. **Knowing an algorithm is not executing it.** The construction is short to describe, but the proofs it writes are
   long: about 109 actions per K12-Kalmár proof, against about 7 lines for a K12 proof. kalmar-pretrain's first,
   unreviewed numbers point the same way: high teacher-forced agreement with the algorithm, much lower exact execution.
   A measure of how much there is to learn may not predict how hard it is to learn it reliably.

## Questions

1. **Upper bounds on the description length.** Write the shortest programs you can for the construction. Cover both
   variants: the K12-Kalmár shared-lemma variant that kalmar-pretrain trained on, and `tt_prover.py`'s per-atom split.
   Use at least three reference languages:
   - minified Python, reported raw, compressed (gzip, xz) and in our tokenizer's tokens;
   - a fixed small universal language with an exact bit count, such as binary lambda calculus or a combinator calculus;
   - a transformer-native language, RASP or RASP-L, for the per-step policy (proof state → next action).

   Also find the smallest exact decision procedure over the proof-state interface's features that reproduces every
   action in K12-Kalmár: a decision list or tree, with its size in bits. Keep the algorithm separate from its rendering
   (the Lean and ND token syntax), and report each part and their sum.
2. **Calibration.** Measure a reference set the same way, choosing items whose learnability is known:
   - truth-table evaluation of a formula;
   - the K12 proof styles, or their most common strategies from `proof-strategies`;
   - simple algorithms with published transformer learning results, such as addition, parity, sorting and copying from
     the RASP-L literature.

   Is Kalmár's construction large or small on this scale, and does the ranking hold across languages?
3. **The bits a model actually absorbs.** Estimate the prequential (online) code length of the Kalmár data under our
   model family. Then estimate the part of it above the asymptotic loss: the model-description part of a two-part code,
   which is the number of bits the network absorbed.
   - **B1 (no pods):** use the first epoch of kalmar-pretrain's KAL-D training curves and of the K12 control's Stage 1.
     In the first epoch each batch is seen once before its update. Check what was logged, and say so if it is not
     enough.
   - **B2 (pods, optional):** a small block-wise prequential experiment in the style of Blier & Ollivier. Train small
     models from scratch on growing prefixes of the Kalmár (state, action) data, at 2–3 model sizes, and do the same for
     1–2 reference tasks from Q2. Pre-register it before the first pod.

   Where possible, break the bits down by step type: atom split, excluded-middle derivation, leaf evaluation by
   connective, and ex falso.
4. **Which measure predicts difficulty?** Compare these measures:
   - program length in each language;
   - the size of the decision procedure;
   - RASP-L length;
   - per-step computational demand: what one forward pass must compute (for example, a subformula's truth value under
     the branch's valuation) and the depth that needs;
   - Levin's Kt (program length plus log running time);
   - prequential bits.

   Test them against kalmar-pretrain's reviewed per-step-type agreement and error analyses (`kalmar/tf_agree.py`,
   `kalmar/error_truth.py`). Which, if any, predicts where the model fails? Recommend how the project should state "how
   much there is to learn", and say how strong the evidence is.
5. **Literature.** Ground Q1–Q4 in what has been written on description length and learnability:
   - Kolmogorov and Levin complexity, and the invariance theorem;
   - MDL and prequential coding for neural networks;
   - RASP, RASP-L and length generalisation;
   - simplicity bias in neural networks;
   - excess description length (from lit-review-2).

   Screen 15–25 sources and read 5–8 in depth.

## Pre-registration

Commit `preregistration/kalmar-complexity.md` before the first pod, and before computing B1's numbers. Write numeric
expectations for:
- the program length in each language;
- Kalmár's rank in the calibration set;
- the absorbed-bits estimate relative to the program length;
- which measure will best predict per-step-type error.

## Deliverables

- `run_kalmar_complexity.md` (≤ 400 words + figures); `numbers.md` § kalmar-complexity; `log.md`; compute rows.
- A plain-language summary of ≤ 800 words, readable by someone outside the subfield.
- `kalmar_complexity/programs/`: each program, its interpreter or checker, and a test that it reproduces the builder's
  output on at least 1,000 K12 theorems.
- `kalmar_complexity/measures.csv`: one row per algorithm × measure, giving units, language and method.
- `next_steps.md`, including what the separate pretraining-vs-RL bits investigation should measure.
- A `STATUS.md` line `KALMAR-COMPLEXITY DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/kalmar-complexity/`,
  delete pods, and `touch ~/runs/kalmar-complexity/executor.done`.

**For the reviewer:**
- Re-run each program against the builder on a fresh sample.
- Recount the bit counts.
- Recompute B1 from the logs.
- Check that every comparison is made within one language.

**Rules:**
- Do not contact anyone.
- textbook72 is scoring only, and no textbook72 theorem is printed.
- At most 4 concurrent processes on the VPS.
- Never run `test_run_once.sh`.
- Keep the RunPod balance above $35.
- Do not edit `references.bib` or run `code/tools/literature.py`.
