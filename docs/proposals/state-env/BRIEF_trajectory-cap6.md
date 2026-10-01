# Run brief: trajectory-cap6 — the same learning-trajectory analysis on the cap-6 model organism

Run id `trajectory-cap6`. Repository `~/work/trajectory-cap6` (branch `dan_trajectory-cap6` from the fork's `origin/dan`,
made by `newrun`; starts after `trajectory` is recorded). Role: executor. `AGENT_POLICY.md` governs, including per-arm
compute rows. Budget **$40**, ceiling **80 pod-hours**. Raised from $25 on 2026-10-01: Dan finds this experiment very
interesting and is happy to spend more than usual, so prefer completeness over cost and run the full scoring plan. Authorised by Dan 2026-10-01: "Let's also do a cap6 version,
which otherwise uses the same configuration." Proposal: nd-rl `docs/proposals/2026-10-01-learning-trajectory.md`.

**Read `docs/proposals/state-env/BRIEF_trajectory.md` in full.** Everything there applies unchanged except:

1. **Recipe and data:** `best-state`'s **best-cap6** cell, which is Robbie's 9.56M recipe in the proof-state format
   (`lean_staten`), Stage-1 1,200 s on an A40, trained on the cap-6 control set
   (`hf://buckets/dan-pandori/nd-rl/state-env/data/p2/train_depth3_f0_a1.jsonl`, 155,000 records). Then the T1 ladder
   with replay from that set. There are 3 fresh seeds, with the same checkpoint schedule and the same scoring as
   `trajectory`.
2. **Sanity check:** against `best-state`'s **best-cap6** per-seed numbers (frozen and T1).
3. **Expected groups:** best-cap6 solves about 19 / 72 textbook problems frozen and about 37 after RL, and about
   132 and about 227 of holdout250. So group B is larger here than at cap 12. Pre-register your own expected group
   sizes.
4. **Compare with cap 12.** `trajectory`'s per-theorem, per-checkpoint records are in
   `hf://buckets/dan-pandori/nd-rl/trajectory/artifacts/`. Add one joint figure for the theorems both runs track:
   log-likelihood of the eventual proof by group, cap 6 against cap 12, on the same axes. Add a short section on
   how the two trajectories differ, especially for theorems that are group B at cap 6 and group A at cap 12. Those
   are the theorems longer training proofs make solvable without RL.
5. **Deliverables** are named `…/trajectory-cap6…`: `preregistration/trajectory-cap6.md`,
   `run_trajectory_cap6.md`, `numbers.md` § trajectory-cap6, `STATUS.md` line `TRAJECTORY-CAP6 DONE <UTC>`, bucket
   `hf://buckets/dan-pandori/nd-rl/trajectory-cap6/{ckpts,artifacts}`, and `touch ~/runs/trajectory-cap6/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the
VPS.
