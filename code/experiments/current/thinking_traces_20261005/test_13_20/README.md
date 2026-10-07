---
written_on: 2026-10-07
written_by: agent:claude
---

# test_13_20: graded 13–20-line companion to the sealed 21–30 test set

The 21–30 set ([`../test_21_30/README.md`](../test_21_30/README.md)) is at the floor for the Stage-1
models: T1v1 B256 pass@8 is 0.22%. This companion uses the same recipe at 13–20 lines, so that model
comparisons happen where pass rates are measurable. It is sealed against everything test_21_30 was
sealed against, and also against:

- the hard12 pools and test sets: `rl_hard12.jsonl`, `hard12_test_1k.jsonl`, `pools/hard12_test`,
  `pools/hard12_work`;
- every other `runs/pools/*.jsonl`;
- the 21–30 set itself.

Leon approved it on 2026-10-07.

| file (under `/scratch/cowifornia/nd-rl/think/runs/pools/test_13_20/`) | contents |
|---|---|
| `test_13_20.jsonl` | the set (OOD-set row format, as test_21_30; names `t1320_NNNN`) |
| `test_13_20.keys.jsonl` | renaming-class sidecar (`name`, `key`, `pkey`) |
| `test_13_20.meta.json` | histograms, certification, dedupe, diversity, calibration, power |
| `seal_report.json`, `profile.json`, `calib_report.json`, `power.json` | reports |
| `calib600_13_20.jsonl`, `calib/` | the length-stratified calibration sample and eval outputs |
| `work/` | intermediates (pieces, composites, refinement, certification, logs) |

Scripts:

- **In this folder:**
  - `excl.py`: the exclusion index (120 files);
  - `certify.py`: applies exact-search results;
  - `subsample.py`: the stratified calibration sample;
  - `calib_lock.sh` and `calib_all.sh`: GPU 0 runs under the shared lock;
  - `negf.py`: the 2-line ~F fix;
  - `format_filter.py`: re-applies the format limits;
  - `power13.py`, `meta13.py`;
  - `regenerate.sh`: the full pipeline.
- **Reused from `../test_21_30/`:** `gen_pieces`, `prep_pieces`, `build_composites`, `compose`,
  `ndproof`, `fastlen`, `refine`, `minimality`, `lean_set`, `assemble`, `seal`, `profile_set`,
  `calib_report`, `power`.

## Recipe

1. **Pieces.** These are test_21_30's generator runs: the fork's unchanged generator, with exact
   minimal lengths from `minlen`'s rules.
   - Premises the minimal proof leaves unused are dropped, which keeps the length exact.
   - Pieces whose class is in the 120-file index are dropped.
   - 215,557 pieces remain.
2. **Composites** (`build_composites.py --lo 13 --hi 20`) use the final test_21_30 recipe:
   - cuts on lemmas that end in ANDE or IMPE, or in ORE onto a literal, with no ex falso;
   - premise-overlap renaming and discharge;
   - rejection of inconsistent, redundant or vacuous premises and antecedents;
   - the same format limits (≤ 3 premises, ≤ 4 atoms, depth ≤ 5, box ≤ 3, states ≤ 386 tokens,
     names ≤ 32).

   This built 4,000 composites, 500 per length. Seed 1320; at most 3 uses per piece.
3. **Refinement.** A 15 s vocabulary-restricted exact search on each composite: 3,623 certified
   vocabulary-minimal, 278 shortened, 99 timed out.
4. **Selection and Lean.** Up to 380 per length were selected; Lean accepts all 3,040.
5. **Certification.**
   - **13–17 lines.** An exhaustive search runs to lines − 1 (`minimality.py`, up to bound 16, the
     fork's `minlen` space plus the reference's formulas).
     - If no shorter proof exists, the reference is exactly minimal (`cert = exact`).
     - If one exists, it is minimal (iterative deepening) and replaces the reference
       (`exact_replaced`; the row moves to its new length).
   - **17 lines.** The search to 16 was stopped as too slow (3–5 min, about 1.5 GB per item); 17 then
     went to a lower bound like 18–20.
   - **18–20 lines.** A lower-bound search to 14 lines (to 15 for 20 items).
   - **Lean** re-checks replaced proofs.
6. **Write, seal, sidecar, profile, calibration, power**, as for test_21_30.

7. **Post-fixes**:
   - **`negf.py`.** `minlen` spends three lines on a ( ~ F ) box, but `| F : AS ; ( ~ F ) : NEGI s s`
     is a valid two-line box; the model found one during calibration. The rewrite shortened 26
     references by one line (4 fell to 12 lines and were dropped). The same fix shortened 63
     test_21_30 references; that set now holds 2,767.
   - **`format_filter.py`.** Re-applying the format limits dropped 22 rows: their refined or
     replaced proofs had box depth 4.

The server crashed from 08:50 to 19:00 UTC on 2026-10-07. Work resumed from the saved intermediates.

## Results

**Size.** 3,003 theorems, every reference Lean-accepted and passing `nd_verify`.

| lines | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|
| theorems | 375 | 376 | 374 | 372 | 381 | 374 | 375 | 376 |

**Certification** (in `minlen`'s space plus the reference formulas, with the 2-line ~F box):

- **13–16:** 100% exact. 1,491 references were certified minimal; 6 were replaced by a shorter
  minimal proof found by the search. The 48 first-pass timeouts were re-run with longer limits,
  and all were certified.
- **17:** 175 of 381 exact (46%). The rest have L ≥ 15, apart from 14 timeouts with weaker bounds.
- **18–20:** lower bound only. 1,102 of 1,125 have L ≥ 15 (98%).

**Dedupe.** The check used renaming classes and premise-order-invariant classes against 120 files
(1,666,329 records). These are test_21_30's 106 files, plus `rl_hard12.jsonl`, `hard12_test_1k.jsonl`,
`pools/hard12_test`, `pools/hard12_work`, the other `pools/*.jsonl`, and `test_21_30.jsonl`.

- 0 hits;
- 100% distinct classes within the set;
- pieces whose class appeared in those files were excluded too.

**Diversity against K12** (`profile.json`):

| | test_13_20 | K12 |
|---|---|---|
| premises 1 / 2 / 3 | 1.3 / 20.2 / 78.5% | 25.5 / 48.3 / 20.2% |
| 4 atoms | 75% | 24% |
| box depth 1 / 2 / 3 | 42.2 / 48.1 / 9.7% | 43.1 / 31.9 / 0% |
| case splits per proof (median) | 2 | 0 |
| prompt tokens (median / max) | 62 / 90 | 39 / 156 |
| states (median / max) | 174 / 312 tokens | 98 / 386 tokens |
| state_env actions (median) | 18 | 8 |

The set is less extreme than test_21_30.

**Calibration** (GPU 0, shared lock; 600 items stratified by length, of which 595 are in the final
set; k 8, plain, `--max_steps 96`). Cells are pass@1 / pass@8, % of the bin.

| model | 13–14 | 15–16 | 17–18 | 19–20 | all |
|---|---|---|---|---|---|
| T0 s0, B 0 | 2.67 / 10.7 | 0.77 / 2.7 | 0.25 / 1.4 | 0 / 0 | 0.92 / 3.70 |
| T1v1 s0, B 0 | 2.25 / 7.3 | 0.09 / 0.7 | 0.17 / 1.4 | 0 / 0 | 0.63 / 2.35 |
| T1v1 s0, B 256 | 4.25 / 18.7 | 1.28 / 7.5 | 0.59 / 4.1 | 0.33 / 1.3 | 1.62 / 7.90 |

- **Accepted proofs.** 11–26 were longer than the reference, 6–20 equal, and 1 "shorter". That one
  is a Lean-only step (`¬¬Q` used as `¬Q → False`), not an ND proof.
- **Wall time:** T0 140 s; T1v1 B0 286 s; T1v1 B256 450 s.
- **test_21_30, T0 s0 B0** on 575 stratified items: 0 solved.

**Power** (`power13.py`). Paired variance components from the sample, projected per length bin to
the full set (N 3,003).

| pair | difference | SE at k 8, fixed / population | 80%-power MDE at k 8, fixed / population | MDE at k 16 |
|---|---|---|---|---|
| T0 B0 vs T1v1 B0 | +0.29 pp | 0.07 / 0.09 pp | 0.19 / 0.26 pp | 0.14 / 0.23 pp |
| T0 B0 vs T1v1 B256 | −0.69 pp | 0.09 / 0.11 pp | 0.25 / 0.32 pp | 0.18 / 0.26 pp |
| T1v1 B0 vs T1v1 B256 | −0.99 pp, about 10 SE | 0.09 / 0.10 pp | 0.24 / 0.27 pp | 0.17 / 0.21 pp |

- **The SEs are small because pass@1 is only about 1%.** A 1-point difference would be a 60–100%
  relative change at these rates.
- **For Stage-1 models, the informative band is 13–16 lines.** pass@8 runs 0–21% there.
- **Recommendation:** k 8 on the full set. Scaled from the measured sample walls, one checkpoint
  takes about 12 min (T0 B0), 24 min (T1v1 B0) and 38 min (T1v1 B256) on GPU 0; k 16 doubles
  these.
