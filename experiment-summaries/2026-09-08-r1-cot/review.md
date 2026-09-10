---
written_on: 2026-09-08
---

# Independent review: R1-style natural-deduction sprint

Status: initial source review complete; new implementation review pending. No textbook prompts, reference proofs, generator source, or per-problem held-out results were inspected. No GPU work was performed. Reviewer: Codex review agent.

## High-priority findings

1. **The base verifier is unsound for forward subproof references.** Both the old `nd_sprint/code/nd/verify.py` and current simplex1 `take-home/nd_verify/verify.py` omit the requirement that cited subproof endpoints precede the citing line. The newer `transfer_sprint_20260907/code/strict_verify.py` fixes this by rejecting every reference greater than or equal to its citing line. Every new reward, replay admission, and evaluation must use the strict wrapper, or equivalent independently tested checks. `controlled_rl.py` imports `rl_common_v2.py`, which still imports the base `nd_metrics.safe_verify`; reuse alone does not guarantee strictness.

   CPU verification against the current simplex1 source confirmed that a circular derivation of an invalid arbitrary implication is accepted by the base helper and rejected by the strict wrapper. The fixture uses no dataset records: prompt `THM SEQ ( P > Q ) PRF`; body `N1 ( P > Q ) : IMPI N2 N3 ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ; N4 ( P > Q ) : IMPI N2 N3 ; QED`. Observed base result `(True, 'ok', 4)`; strict result `(False, 'strict: reference N2 is not before line N1', 4)`.

2. **Generated proof length is not theorem difficulty.** The existing `nd_metrics.aggregate_attempts` takes the shortest accepted proof found per theorem, then defines a frontier by at least five solved theorems at an exact found length. This is more robust than raw maximum but still grows under redundant valid derivations and can decrease when a better shorter proof is found. All-lines-cited is insufficient: a chain of reiterations remains live. Do not make this the sole proof-window claim.

3. **Final-proof extraction is a trust boundary for CoT.** Verification must use the immutable target prompt, exactly one final proof, actual termination, and strict citations. A valid proof hidden in scratch tokens must not count when the final answer fails. Later `QED`/extra final tags must not silently select whichever proof works. Keep scratch tokens outside the formal proof length. Store scratch, final body, token budget, extraction status, and verifier status separately.

4. **Dataset identity must extend beyond literal rows.** Old split code deduplicates only `thm`; current structural guards provide canonical theorem hashes, premise-context hashes, and closed-formula skeleton hashes. Fix a textbook split before training, group duplicates/atom-renamings before allocating rows, and assert at most 10% of textbook records enter all supervised/RL/replay streams combined. Use only opaque protected hashes for training exclusions; do not expose held-out prompt-derived diagnostics. Audit the starting checkpoint's historical data exposure, not only this sprint's new files.

5. **End-run evaluation is adaptive development.** The 40% endpoint panel may be inspected only after each RL run ends. Changes informed by those scores are adaptive to that panel. The 50% final-only panel must remain unopened until candidate checkpoints, decoding budgets, and report comparisons are frozen. Running the final-only panel separately for many candidate-selection rounds breaks its interpretation. Saved held-out outputs never enter replay, cold-start data, or failure-repair prompts.

## Metric and graph recommendation

Primary: held-out theorem solve fraction by frozen reference-proof-length bins, with distinct-theorem denominators and binomial intervals, comparing base direct decoding, base scratch decoding, direct RL, and scratch RL under declared budgets. Label these lengths **reference witness lengths**, not minimum necessary lengths. Include cap-six training as a vertical marker, and report the maximum reference witness length among solved held-out targets explicitly. When counts permit, also report the highest prespecified bin with at least five distinct solved targets and its denominator; do not invent a frontier if that criterion is unmet.

Secondary: per-target shortest strict-valid generated proof length, dependency-pruned non-reiteration rule count, and depth. These describe discovered proofs; they cannot establish a shortest-proof lower bound. A stronger difficulty claim requires an independent bounded exhaustive proof-search certificate, not a failed heuristic search.

Inference-time scaling should report both pass@k at fixed per-sample budget and coverage against total generated tokens (scratch plus proof). Compare equal total budgets and include median/p95 actual tokens and truncation frequency. Increased sample count alone is a different intervention from longer sequential reasoning. For tiny panels, include paired solved/lost counts and avoid interpreting a single extreme theorem as broad window growth.

## Required implementation checks

- Circular future-box fixture rejected; valid arbitrary-start indices and normal box discharge accepted; sibling-scope leakage, self-reference, missing QED, malformed formula, and wrong conclusion rejected.
- Adversarial scratch/final marker cases, wrong-target proofs, multiple answers, no answer, partial final proofs, proof-like scratch, trailing output, and actual EOS handling.
- Rewards invariant to scratch verbosity; proof lengths exclude scratch and all post-QED text. Current `nd_metrics.safe_verify` already trims length through first QED: CPU fixture `N1 P : PR ; QED ; ; ;` returned valid length 1. Its permissive trailing text is an acceptance convention, not a padding metric exploit in that helper.
- Correct action mask includes generated scratch and final tokens and excludes prompt/padding; no teacher-forced successful proof is mislabeled as model exploration. Log mixed-reward group fraction and zero-advantage batches.
- Rollout/current-policy logprob consistency, temperature consistency, clipping sign, finite advantages, and no stale logits after update. Match dropout mode during sampling and likelihood reconstruction.
- Truncation and context overflow explicit; no positive reward from a proof reconstructed by the tokenizer that was not literally sampled.
- Freeze/hashes cover data split, tokenizer, verifier, parser, code, base checkpoint, trained checkpoint, seeds, budgets, and evaluation configuration. Training loader and replay loader enforce protected hashes.
- End-run panel paths absent from training code reads; final-only evaluation fails before an explicit frozen final manifest exists. Missing outputs count as failures.

## Sources inspected

Local `nd-rl/AGENTS.md`, README and docs index; requested STATE and experiments index did not exist at initial review. Old proof verifier, split builder, GRPO and eval source were inspected before the newer codebase was located. Current code-only source was read via SSH from `/data/users/dmitry/nd-rl/take-home`: `nd_verify/verify.py`, `nd_metrics.py`, and transfer sprint `strict_verify.py`, `strict_verify_tests.py`, `controlled_rl.py`, `rl_common_v2.py`, `sprint_common.py`, and structural helper definitions. Historical textbook contents and per-problem results were deliberately left sealed. Prior endpoint aggregate summaries have not yet been used as evidence.

## New implementation review

`code/experiments/current/r1_cot/run.py` now uses the strict verifier, requires protected-data guards for normal training, retains actual cached-rollout probabilities in the PPO denominator, defaults to leave-one-out advantages and a fixed configured response-budget divisor, and includes sampled PAD tokens in the objective. Scratch and final boundaries are actual model actions. No blocker was found in the reviewed sampler or objective.

Six independent synthetic-only CPU tests in `code/tests/test_r1_review.py` passed. They check immutable-target final verification, malformed/multiple boundaries, missing/trailing QED, proof-like scratch, scratch-length exclusion, malformed zero-distance references, all three guard levels, PAD action masks, leave-one-out advantages, variable-prompt-length cached sampling, and actual behavior-denominator preservation under deliberate probability perturbation. Across 96 sampled actions, maximum cached-versus-teacher log-probability error was `4.76837158203125e-07` and mean error `7.2022279103597e-08` in FP32 CPU execution. CUDA/BF16 validation is available via `R1_REVIEW_DEVICE=cuda` and was not run by the reviewer during this check.

The evaluator currently writes a per-evaluation checkpoint/configuration freeze. A separate sprint-wide final-release guard was requested because a per-run freeze alone cannot enforce final-only access after candidate selection. Parent orchestration must keep the sealed-final panel closed until the frozen final candidate list exists.

## Repository checks and dispatcher hardening

Ran the repository-prescribed `make -C code install-dev`, `make -C code check`, `make -C code type`, and `make -C code test`. Installation fails before hook setup because the existing package declaration includes absent `code/models`. Existing global failures also include absent `code/src`, missing `bibtexparser` during literature-test collection, absent `code.models` during MLP-test collection, unknown parameter types in existing literature tests, and a formatter change in the existing MNIST script. These unrelated scaffold files were not repaired or staged. The new core tests also initially produced Pyright diagnostics from dynamic vendor imports and untyped/mock inference; those findings were sent to their implementation owner rather than misclassified as preexisting failures.

Added narrow Ruff exclusion for the frozen `r1_cot/vendor` sources, including explicitly passed hook filenames. All eight vendor files still match `source_hashes.json`. Added per-file Python 3.10 target settings for dispatcher/submission/pipeline/data scripts, since those run under simplex1's system Python whereas the training environment supports newer Python. This prevents automated conversion to the Python-3.11-only `datetime.UTC` API.

The dispatcher now stops its owned process group even after its leader exits, bounds driver-query subprocesses, clamps waits to the remaining deadline, and immediately escalates at the deadline. It does not signal unrelated process groups. Submission now hashes pipeline `--config` files and recursively hashes their existing referenced data, checkpoint, guard, and release inputs; future stage outputs remain checked by their consumers. Four new CPU-only regression tests include a real exited-leader/remaining-descendant case and verify an unrelated process group survives cleanup.

Scoped lint and format checks passed for dispatcher, submitter, pipeline, blind builder, review tests, and dispatcher tests. Combined runtime/review/dispatcher/analysis/scratch tests passed: **38 passed, 2 optional CUDA tests skipped**. No GPU tests were run for this repository-check task. No files were staged or committed; parent will choose the task-only commit after assessing the unrelated global failures.
