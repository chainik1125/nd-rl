# Reviewer digest (arm B) — final

_Updated 2026-09-23 04:20 UTC. The executor wrote DONE at 04:14._

**Incumbent:** exp13_textbook_s0, dev **324** (109/86/129), seed-1 rescore **328**, heldout_greedy 0.896,
`/work/exec/runs/exp13_textbook_s0/final.pt`. Adopted 02:43. Replaced exp01 (seed control, 249 / 254).

**Runs reviewed: 29.** Adopted 2 (exp01, exp13_s0), rejected 27, quarantined 0, not-preregistered 0. Every count reproduced
exactly from the raw files (150/150 proofs re-verified per run). No hard-constraint violation in any rl.py. Gate 0 passed everywhere,
usually by 1–2 s.

**What moved the score.** Only the textbook-source exploration stream (sampling the 830 `source == textbook` targets at T ≥ 1.0):
the 16 textbook-family runs score 236–339 (median ~314); the 13 others score 193–269. Dev tracks textbook solves (≥ 130 → 301–339).
Only contraposition and export crack; third schemas appeared 4 times, 1 instance each, never spread.
Best-supported recipe: exp23 (focus_k 16), 339/323 on two seeds. Its s0 checkpoint rescored at 338; +12.5 over the incumbent
on the two-sampling-seed mean, under the pre-committed +15 bar, so REJECT. **exp13_s0 and exp23_s0 are equivalent within noise.**
Three-seed means: exp19 312.7, exp21 313.7.

**Noise measured:** sampling seed (same checkpoint): +5, +4, −4, −5, −1, −4, −3 (exp01, exp13, exp21, exp11, exp23, exp17_s1, exp10). Training seed, same recipe:
spreads 5–95 (driven by whether textbook ignition happens). GPU training is not bit-deterministic (same code and seed diverge at round 2).

**Expectations:** of 29 pre-registrations, 10–13 hit or were not falsified by their own criteria (the count depends on formal vs substantive readings in exp10/15/22). Early misses were over-predictions by 35–80. The one big upside
miss (exp13: 250 → 324) was the only real discovery. All misses were reported as misses. Where the executor scored its own prediction
wrongly, it was stricter than its own rule (exp10, exp15, exp22) except for one schema miscount (exp19), which it self-corrected.

**Open questions, ranked:**
1. Cracking textbook schemas beyond 2 of 19. Temperature, budget and allocation did not do it.
2. Winner's curse: the incumbent is the best of 29 dev-selected runs. The sealed L10+/textbook evaluation is the honest check.
3. Heldout cost: every textbook-family run is at 0.883–0.910 vs the control's 0.920.
