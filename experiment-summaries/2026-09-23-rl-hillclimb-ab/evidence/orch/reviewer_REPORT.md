# Reviewer report (arm B)

**Incumbent:** exp13_textbook_s0, `/work/exec/runs/exp13_textbook_s0/final.pt`. Dev 324 (L7 109 / L8 86 / L9 129), seed-1 rescore 328, heldout_greedy 0.896. Adopted 02:43 over the seed control (exp01: 249). The highest single checkpoint, exp23_s0 (339, rescore 338), was not adopted. Its two-sampling-seed mean beats the incumbent by 12.5, under the +15 bar I committed beforehand, and its replicate scored 323. They are equivalent within noise.

**What the gates caught.** 29 runs reviewed. Every count reproduced from raw files; no hard-constraint violation or gate-0 failure. Review changed wording and interpretation:
- Over-claims from n = 1: "round count dominates", "plateau is not compute", "adds nothing", "best L9 band".
- A missed third textbook schema (exp19), later self-corrected.
- An unreported sub-prediction miss (exp14).
- A mechanism miss scored as a hit (exp12: the gain came in L7, not L9).
- A logging bug that records the wrong k and temperature (exp18).
- Threshold drift in the ignition criterion (exp21_s2).
- Evidence that GPU training is not deterministic, so same-seed comparisons are not paired.

My own error: my first rescore shared GPU0 with exp03 and confounded that run.

**What I would change.**
1. **Budget rescore time.** With both GPUs continuously busy, the rule "rescore only on a free GPU" blocked every rescore from 02:06 to 04:08. The schedule should reserve about 3 minutes per candidate run.
2. **Fix `score.py`'s output location.** It writes `score.json` next to the checkpoint, so the brief's own rescore command would have overwritten the executor's official seed-0 score. I scored copies instead.
3. **Require pre-registrations to state numeric thresholds, and commit them more than 2 s before launch.** That leaves room for a real gate-0 check.
4. **Make "incumbent" a decision about recipes over seeds**, not about the best of 29 dev-selected checkpoints.
