# Run brief: claim-audit — red-team the project's headline claims from raw artefacts

Run id `claim-audit`. Repository `~/work/claim-audit` (branch `dan_claim-audit` from the fork's `origin/dan`, made by `newrun`).
Role: executor, acting as an adversarial auditor. `AGENT_POLICY.md` governs. Pod budget **$8** / 16 h, for re-sampling
checks only (no training). Claude time is expected to be heavy. Authorised by Dan 2026-10-02 (proposal 21).

## Why

Dan will share results with the team. Each run was reviewed on its own, but nobody has checked the headline claims
**across** runs: whether the same model scored by different runs agrees, whether definitions are consistent, and
whether selection effects, leakage or multiple comparisons weaken anything. Before claims travel, try hard to break
them.

## Claims to audit

Take them from nd-rl `docs/project_strategy/2026-10-01-week-in-review.md` ("What seems important") and the headlines in
`docs/STATE.md` since 2026-09-27. At least:
1. Support expansion: EI solves base-unreached theorems (`support-curves`, `support-followups`).
2. The proof-state base reaches 28 of 29 survivors, and it is the state, not the naming (`support-state`,
   `state-readouts`).
3. The proof state and cap 12 compound (`state-cap12`, `long-pool-2`).
4. Robbie's recipe beats ours after RL at both caps (`best-state`), and the textbook72 numbers (`textbook72`,
   `best-state`, `trajectory`, `trajectory-cap6`).
5. The `trajectory` and `trajectory-cap6` descriptive findings (worst-step climb; RL lifts its own proofs).
6. `rl-from-ckpt`: from step 5k, RL ends level with the end arm; no early start reaches beyond it.
7. GPU nondeterminism is about half the seed spread (`lit-measures` M2).

## For each claim

- **Re-derive** the headline number from raw artefacts with **independent code** (not the run's analysis scripts).
- **Check consistency across runs:** the same checkpoint and pool scored by two runs should agree within the
  sampling spread. List every pair you can find.
- **Check definitions:** groups, `L*`, solved, and the checker are the same where claims compare them.
- **Attack the claim:** selection effects (groups defined and measured on the same samples), leakage by renaming
  class (including premise-order variants), multiple comparisons, forking paths in post-hoc analyses, and seed counts
  against `NOISE_FLOOR.md`.
- **Spot-check in Lean:** a random sample of accepted proofs, with negative controls.
- **Optional (≤ $8):** re-sample a key checkpoint on a key pool to test reproducibility directly.
- **Rate the claim:** *solid* / *holds with caveats* / *weaker than stated* / *not supported*, with the evidence.

## Deliverables

- `audit/CLAIMS.md`: a claims ledger with ratings, evidence links, the strongest objection to each claim, and
  suggested wording fixes. It should be ready for Dan to read before sharing. The librarian copies it to nd-rl
  `docs/project_strategy/2026-10-03-claim-audit.md`.
- `run_claim_audit.md` (≤ 250 words), `numbers.md` § claim-audit, `log.md`, `STATUS.md` line
  `CLAIM-AUDIT DONE <UTC>`, and `touch ~/runs/claim-audit/executor.done`.

Do not contact anyone. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
