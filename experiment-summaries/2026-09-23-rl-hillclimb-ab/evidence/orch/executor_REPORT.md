# REPORT — executor, arm B (2026-09-23)

**Incumbent (ledger top, 04:15 UTC):** exp13_textbook_s0, `/work/exec/runs/exp13_textbook_s0/final.pt`.
Dev 324 (109/86/129), heldout_greedy 0.896 (reviewer seed-1 rescore 328). Control exp01: 249, 0.920.
Not yet reviewed, and my best recipe: exp23, `runs/exp23_cheapfocus_s0/final.pt`, 339 and 323 on two seeds
(mean 331; heldout_greedy 0.901/0.902).

**What moved the score.** A second sampling stream over the 830 textbook-source targets (pool + extra) at
T 1.0–1.2. Before it, every allocation change scored 193–268, textbook targets were never solved, and dev L8 stayed
at 85–95. Once a few textbook proofs appear, expert iteration spreads them across instances of the same schema
(contraposition, export), and dev L7 goes from ~40 to 100–115. The gain depends on early ignition: exp17 gave
331 vs 236 across seeds. Runs are not bit-reproducible. Starting the stream in round 1 at T 1.2, with a cheap
late-round gen focus (exp23), ignited by round 2 on both seeds. Third schemas appeared only in last rounds, one
instance each.

**Expected → got** (sources in numbers.md)
- Allocation/efficiency (exp02–11): predicted 265–300, got 193–255. All misses. Sampler is compute-bound
  (batch 1024 gave nothing; compaction ×1.15); more than 5 rounds adds nothing.
- exp12 per-target weight: 267 → 268 (hit, one seed; adds nothing with textbook, exp15 303).
- exp13 textbook stream: predicted ~250 and ~5 solves → 324 and 145 (s0); 269 (s1). Belief "out of reach" falsified.
- exp14 gen focus: cum ≥680 → 639 (miss); best L9 131.
- exp16 schema-aware allocation, exp22 double budget: predicted ≥3 spreading schemas → none (misses).
- exp17 round-1 start: 305 → 331 / 236.
- exp18: 315 → 300 (narrow pass).
- exp19 T 1.2: 330 → 328 / 301 / 309 (s2: no ignition by r3, falsified).
- exp21: 335 → 323 / 318 / 300.
- exp23 cheap focus: 325, 5 rounds → 339 / 323 (hits).

**Next.** Review exp23 for adoption. Run 4+ exp23 seeds logging the first-textbook-solve round, to measure ignition
reliability. Up-weight a newly cracked schema's single instance so it spreads before the budget ends: 17 of 19
schemas never spread. Watch heldout_greedy: it fell from 0.92 to 0.90.
