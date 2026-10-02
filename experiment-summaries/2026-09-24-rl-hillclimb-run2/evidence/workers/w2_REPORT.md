# w2 report (ab2, GEAR-Evolve)

**Frontier at close.** BEST is **n078** (mine), two-seed score 1175 (dev 378/388, frontier 205/191 of 1,000). Seed-0 checkpoint: `/scratch/cowifornia/nd-rl/ab2/w2/runs/e12_n071_idx/final.pt` (dev 378: L8 118, L9 102, L10 75, L11 83; frontier 205: L12 162, L13 43, L14 0; held-out 0.924).

**Highest two-seed mean: **n092** (LEAN, 1286/1139, mean 1212.5, 2.5 short of the margin over n078), `/scratch/cowifornia/nd-rl/ab2/w2/runs/e14_x_avg/final.pt` (dev 378, frontier 227: L12 175, L13 52; held-out 0.937).

**What moved the scores**
- **Capped line-index shift (own idea, found by a CPU diagnostic).** The frozen `shift_abs` spreads proof start indices over N1..N59, so long proofs overflowed N64. This killed 12% of L11 samples. Renumbering fine-tuning proofs to start at or below N41 added score on four lines:
  - n023: +31 dev
  - n036: +125 (two seeds)
  - n065: +244
  - n078: +172
- Always starting at N1 is a trap (n030): the N10 -> N11 successor is never trained, and L11 fell to 0.
- **Line-head decision temperature 1.4** (n006; arXiv 2510.08892 and REFT). It was BEST on two seeds and is carried into n060 and n078.
- **Schema-coverage crossover** (n050, 968).
- **Last-3-round weight averaging** (n092, +183).

**What did not help**
- WiSE-FT toward a mid-run checkpoint (n013): validation picked alpha 1.0, so no change.
- Shortest-first proof selection (n018).
- Entropy minimisation on citation tokens (n044, -69).
- Tail-only citation retries (n057): neutral, because rounds got slower.
- Step-DPO (n072, -537): too few pairs, and it memorised them.
- Two ideas screened on CPU only: retry temperature ladders, and head exclusion (158 -> 85 accepted). Rejected lines mostly have the right formula and wrong citations.

**Controller.** One step, v10: a fresh elite forces a crossover only if it beat its parent1 by the improvement margin. It stops no-op crossovers (n005 + n013).

**Sources:** arXiv 2510.08892, 2605.28295, 2504.10478, 2502.20122, 2505.15134, 2502.14356; own ideas. Verdicts: `ideas.md`.

**What to try next**
- **L14 is still at zero (0.5 at most).** Its textbook half is demorgan_nand_to_or, and training never solves an instance of it above L8.
- **Box depth may be the next ceiling.** Found L10-11 proofs reach depth 5, while cap-6 data stops at 3. This is likely a counting gap like the index successor.
- **Citations are the main error source.** They carry most of the invalid-token mass at T 0.8 (0.42 per proof, against 0.05 for formula bodies).
- **Tail retries may pay off at 7,200 s.**
