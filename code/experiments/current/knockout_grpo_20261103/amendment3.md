---
written_on: 2026-11-03
written_by: agent:claude
---

# Amendment 3: ko_strict, a knockout without σ in the proofs or in the narration

Written 2026-11-03 at about 08:55 UTC, after the trace-leak check and the coordinator's approval.
Its sha256 is recorded in `log.md` before the first GPU job it governs.

**Why.** The ko corpus has no σ in its proofs (Dan's classifier) and no DN applied to a NEGI-built
line in its narration, but 949 records plan the classical detour in their narration (`|- G :=
Classical.byContradiction |- ( ~ ( ~ G ) )`, 515 of them with the NEGI half `|- ( ~ ( ~ G ) ) :=
fun ( n : ( ~ G ) ) => |- F`; 637 at the effort level the pickle trained them on). Dan's
plain-proof K12⁻ had no such channel. ko_strict removes it.

- **Corpus** (`assemble_strict.py`, seed 1): ko minus every record whose e64 or e256 narration (the
  two non-empty levels `data.draw_level` can draw) contains `Classical.byContradiction |- ( ~ ( ~`
  (949: 778 kept S1g3 records, 171 refill), each slot refilled in place with a record of the same
  K12 generator length from Dan's eligible pool, not already used, whose search proof is σ⁺- and
  lemma-route-free and whose e256 and e64 narration lack the detour goal.
- **Verified:** 154,630 records; σ⁺ 0 and σ 0 in the proofs; the detour goal in 0 e64 and 0 e256
  traces; no DN cited on a NEGI-built line in the narration (the remaining `Classical.byContradiction`
  mentions are DN of premises and box hypotheses, which Dan's knockout keeps); e64 and e256 actions
  identical on every record. Pickle `S1g3_kos.pkl` (md5 dda2fac8), the S1g3 build command.
- **Deltas against ko:** 949 records swapped (0.61 %); pair tokens per epoch 9.62e7 against 9.86e7
  (−2.4 %; S1g3 1.017e8); pairs 920,446 against 934,896; proofs with a NEGI-built ¬¬G line
  (NEGI without DN) 2,957 (1.91 %) against 3,883 (2.51 %; S1g3 4.89 %).
- **Bases:** `kos_s0–2`, the identical Stage-1 command (train.py at cd089700, 24,000 steps, seeds
  0–2), read at n 256 like every base.
- **Lines:** `kos_s0–2` with our pool and the identical recipe (T_L3 364, bar 26.31, the snapshot
  grid and read path, Amendment 1's hold window), with `--lean_dump`.
- **Reported** like the ko lines; ko_strict replaces ko as the knockout point of the figure, with
  ko shown as the trace-leaking knockout.
