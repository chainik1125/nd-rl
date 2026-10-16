---
written_on: 2026-10-13
written_by: agent:claude
---

# REBUILD NOTE: these are L2r / L3r, not serrano's curriculum pools

The files in this folder were built on ynez on 2026-10-13 (09:28–11:55 UTC) after serrano crashed. They are **new
pools**, built with the recorded recipe. **Their test splits (`L2/test_1k.jsonl`, `L3/test_1k.jsonl`) are not
serrano's test splits.** Do not read a model trained on serrano's L2/L3 against them.

Why serrano's files could not be reproduced:

- the gates have time limits;
- the shortest proofs depend on the process hash seed;
- the adaptive recipe weights cascade from both of these;
- serrano's L3 generation was 9 interrupted runs;
- the split is a positional shuffle, so a few changed rows reshuffle `test_1k`.

The investigation is in the coordinator's report of 2026-10-13.

## Settings

- **Code:** branch `leon/curriculum-rebuild-ynez`, commit `e464c05`, on top of serrano's `6bdbd78`, worktree
  `/scratch/cowifornia/nd-rl/think/nd-rl-currb`. The commands are `curriculum/build_ynez.sh` (phases 1–4), run as
  `for p in 1 2 3 4; do bash build_ynez.sh $p; done`.
- **Environment:** `PYTHONHASHSEED=0` for every step, `ND_OFFLINE=1`, at most 20 worker processes, `nice -n 10`,
  CPU only, every step under `timeout`.
- **Time limits:** `PFG_SEARCH_SECS=12` and `PFG_NEC_SECS=14.5` (serrano used 20 and 20). They were chosen with
  `curriculum/calib_timeouts.py` on hard712's recorded checks (7–12 lines); output in `calib/calib_timeouts.json`.
  - **Necessity, case_split:** serrano timed out on 5.78% of checks at 20 s. The ynez estimates are 7.44% at 12 s,
    **5.80% at 14.5 s** and 4.48% at 20 s. Median time ratio ynez / serrano 0.63 (p10 0.57, p90 0.96).
  - **Search:** of serrano's 73 search timeouts, ynez times out on 69 at 12 s and 50 at 20 s. None of the 56 slow
    spare rows times out. Median time ratio 0.60.
- **Measured on L3:**
  - **regen-long** (the same 4,847 candidates as serrano):
    - ynez ok 4,263, not_needed 338, no_structure 238, search_timeout 8;
    - serrano 4,261 / 345 / 241 / 0.
  - **L3 generation, search timeouts:** 803 of 12,611 certified (6.4%). Serrano: 261 of 10,666 (2.45%).
  - **L3 generation, case_split timeouts:** 52.6% of the gated rows' checks. Serrano: 32% of its first 1,407 rows.
  - A re-timing with a 30 s cap shows that ynez would match serrano's search-timeout rate at about 17–18 s for L3
    candidates, not 12 s. Of 98 of our search timeouts:
    - 71 still time out at 14.5 s;
    - 40 at 18 s;
    - 21 at 20 s.

  So the 12 s search limit rejected about 2.6× as many L3 candidates as serrano did; see "Differences".
- **Generation:** the seed ranges are fixed in rounds of 500k from seed 0, which differs slightly from serrano's
  (whose runs were stopped by time).
  - 0–1M: default weights.
  - 1M–8.5M: adaptive, no skip.
  - 8.5M–19M: adaptive with `skip_keys.txt` (47,757 keys; serrano had 47,750 when it added the regen rows).
  - Gap regen, then add its keys: 50,595 (serrano 50,510).
  - 19M–26M: adaptive, gap 3.
  - The adaptive values are `PFG_CLASSICAL_VALUE=0 PFG_EXFALSO_VALUE=0 PFG_SCARCE_BONUS=0`, as serrano's from its
    run 5 on.
- **Exclusion index:** 597,075 classes, 593,090 pkeys, 120 files. Serrano had 977,825 classes from 497 files.
  - It is rebuilt from the files that exist now.
  - Lost: test_13_20 / test_21_30 work files (pieces, composites), hard12_work, gen_runs and the old traces.
  - Added (`excl.py --extra`): `work/sealed_parts.jsonl`, the 7,814 distinct composite parts of the sealed
    test_13_20 / test_21_30 / calib sets.
  - Left out (`--omit`) because they did not exist when serrano built its index: charles_1002 and t2130_n1000.
    Charles is still removed in curried form by `heldout.py`.
  - The hard712 base index has 559,722 classes from 109 files (serrano 564,905 from 364).
  - hard712's `work/gated*.jsonl` were rebuilt class-equivalent from the NFS backup with
    `curriculum/rebuild_gated_ynez.py`: 29,542 rows, Lean 29,542 / 29,542.
- **Caps:**
  - L3: `find_cap` relaxed ex falso to 0.14, then set classical-only to 0.45. Serrano: classical 0.47, ex falso 0.10.
  - L2: L3r's realized shares, per the recipe: classical 0.4477, ex_falso 0.14, reductio 0.3043, nested_neg 0.4417,
    nn_plain 0.3042. Serrano: 0.47 / 0.10 / 0.3155 / 0.439 / 0.2903.
- **Splits:** `run.py split --n_test 1000 --n_dev 0` (`Random(20261008)`), the dev files removed.

## Results against serrano's recorded numbers

| | L2r (ynez) | L2 (serrano) | L3r (ynez) | L3 (serrano) |
|---|---|---|---|---|
| pool / test_1k rows | 5,000 / 1,000 | 5,000 / 1,000 | 5,000 / 1,000 | 5,000 / 1,000 |
| pool lines (%) | 10:28 11:42 12:30 | 10:28 11:42 12:30 | 13:82 14:14 15:4 | 13:78 14:16 15:6 |
| pool mean lines | 11.03 | 11.03 | 13.22 | 13.28 |
| test_1k lines (%) | 10:26 11:41 12:33 | 10:29 11:43 12:28 | 13:82 14:13 15:4 | 13:77 14:16 15:6 |
| test_1k mean lines | 11.07 | 10.99 | 13.22 | 13.29 |
| pool origin | reused 5,000 | reused 5,000 | gen 1,410, regen 3,118, regen_gap 472 | gen 1,592, regen 2,983, regen_gap 425 |
| test origin | reused 1,000 | reused 1,000 | gen 280, regen 625, regen_gap 95 | gen 327, regen 582, regen_gap 91 |
| premises, pool mean | 1.33 | 1.33 | 1.65 | 1.63 |
| premises, pool histogram 0 / 1 / 2 | 344 / 2,658 / 1,998 | – | 24 / 1,698 / 3,278 | – |
| connectives / atoms / box depth | 6.49 / 2.82 / 2.24 | 6.5 / 2.82 / 2.23 | 6.75 / 2.81 / 2.66 | 6.75 / 2.81 / 2.66 |
| state-env steps | 12.03 | 12.03 | 14.22 | 14.28 |
| classical-only | 39.5% | 39.6% | 44.5% | 47.1% |
| **ex falso** | **13.1%** | **10.0%** | **14.1%** | **10.2%** |
| reductio | 27.5% | 27.6% | 30.2% | 31.5% |
| nested negation | 35.7% | 34.4% | 44.0% | 43.8% |
| case split | 26.5% | 30.2% | 19.5% | 22.9% |
| no premises | 6.9% | 7.7% | 0.5% | 0.6% |
| (A>B)>C goal | 12.0% | 11.9% | 21.0% | 22.3% |
| Lean (constructed / reference) | 6,000 / 6,000 each | same | 6,000 / 6,000 each | same |
| index / textbook72 / held-out / sealed hits | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Reservoir and exclusion counts:

| | ynez | serrano |
|---|---|---|
| L2 reuse kept | 14,261 (used / sealed 11,500, out of band 3,781, index 0) | 14,256 (11,500, 3,781, index 5) |
| L2 curried twins of an L3 row | 394 | 403 |
| L2 duplicates in band | 1,179 | 1,171 |
| L2 held-out / sealed removed | 266 | 266 |
| L2 held-out by file | hard712 test_1k 166, dev_500 94, candidate_v0 4, v1 3, batch3 4, derived 4, textbook train 1 | identical |
| L2 reservoir | 12,422; 10: 3,560, 11: 4,874, 12: 3,988 | 12,416; 3,558, 4,871, 3,987 |
| L3 regen | 4,263 → 3,805 | 4,261 → 3,804 |
| L3 regen_gap | 736 of 845 candidates → 590 | 700 of 863 → 572 |
| L3 gen | 2,767 → 1,745 | 3,508 → 2,181 |
| L3 reservoir | 6,140; 13: 5,041, 14: 851, 15: 248 | 6,557; 4,999, 1,141, 417 |
| L3 duplicates (exact or curried) | 1,567 rows | 1,846 |
| L3 held-out / sealed removed (rows by file) | hard712 test_1k 38, dev_500 10, textbook dev 3, batch3 5, candidate_v1 3 | test_1k 30, dev_500 9, textbook dev 2, Charles 7 (distinct classes) |
| L3 fill at default caps | 3,893 | 4,048 |
| first pool row | cL2_71038368 / cL3_84721981 | the same names |
| head -n 8 md5 | L2 `dd7f2d3f…` (= serrano's) / L3 `e6800228…` | L2 `dd7f2d3f…` / L3 `d5b7d909…` |

`grpo_v2/preflight.py` passes (`logs/preflight_grpo_{think,nothink}_vs_s0.log`).

## Differences that matter

1. **Ex falso: 13–14% of rows instead of 10%, in both bands.**
   - L3r's reservoir has fewer rows that are neither classical-only nor ex falso, so `find_cap` had to raise the
     ex-falso cap before the classical-only cap. L2r inherits that cap through the recipe.
   - Root cause: the 12 s search limit. It rejected about 2.6× serrano's share of L3 candidates (6.4% vs 2.45%), and
     case_split necessity timed out more often, so generation gated 2,767 rows instead of 3,508.
2. **L3r is a little shorter: 82 / 14 / 4% at 13 / 14 / 15 lines, against 78 / 16 / 6%.**
3. **Case-split share is lower by 3–4 points.**

A rerun with `PFG_SEARCH_SECS` around 18 would most likely close most of these gaps. It would not reproduce serrano's
pools either.

Files:

- `L2/`, `L3/`: `pool.jsonl`, `test_1k.jsonl`, `keys.jsonl`, `pool.keys.jsonl`, `test_1k.keys.jsonl`;
- `L*/work/`: the reservoirs and select reports;
- `L3/{gen,regen,regen_gap}`: the sources;
- `report.json`, `report.md`;
- `calib/`;
- `logs/`;
- `work/excl_index.pkl`, `work/sealed_parts.jsonl`.

## curriculum_3p (added 2026-10-13, 10:00 UTC)

`runs/pools/curriculum_3p/L2_3p.jsonl` and `L3_3p.jsonl` hold the 3-premise (uncurried) versions of the **L2r/L3r**
pool rows, so they are new files too.

- **Recipe:** `curriculum/uncurry3.py`'s docstring commands (unchanged from `6bdbd78`): build and lean per band with
  `PFG_MIN_L=6 PFG_MAX_L=15 PFG_MAX_GAP=9`, 12 workers, then finish, preflight and report.
- **Script:** `curriculum_3p/build_3p_ynez.sh` (a copy is `curriculum/build_3p_ynez.sh` on the branch).
- **Environment:** `PYTHONHASHSEED=0`, the same limits as L2r/L3r (`PFG_SEARCH_SECS=12`, `PFG_NEC_SECS=14.5`).
  The certifier had 0 search timeouts in both bands, as on serrano.

| | L2_3p (ynez) | L2_3p (serrano) | L3_3p (ynez) | L3_3p (serrano) |
|---|---|---|---|---|
| rows | 1,546 | 1,400 | 1,843 | 1,796 |
| shortest lines | 7–14 | 7–13 | 11–15 | 11–15 |
| mean shortest lines | 9.78 | 9.79 | 12.14 | 12.21 |
| line histogram | 7:6 8:89 9:493 10:625 11:318 12:11 13:3 14:1 | 7:7 8:67 9:452 10:565 11:301 12:6 13:2 | 11:187 12:1263 13:345 14:47 15:1 | 11:164 12:1186 13:354 14:91 15:1 |
| line shares 8 / 9 / 10 / 11 | 6 / 32 / 40 / 21% | 5 / 32 / 40 / 22% | – | – |
| line shares 11 / 12 / 13 / 14 | – | – | 10 / 69 / 19 / 3% | 9 / 66 / 20 / 5% |
| goal cannot reach 3 premises | 3,338 | 3,404 | 2,867 | 2,951 |
| certified | 1,579 | 1,487 | 1,849 | 1,806 |
| exclusion-index removals | 33 | 87 | 6 | 10 |
| antecedents moved 1 / 2 / 3 | 71 / 28 / 1% | 73 / 26 / 1% | 90 / 10% | – |
| md5 | 0c43de02… | d70471f8… | 8d2d6922… | 159700b6… |

Why there are more L2_3p rows than serrano's:
- the exclusion index is smaller (see above): 33 removals instead of 87;
- L2r's pool certifies 92 more rows.

Rows that match a part of a sealed composite are still removed (15 hits on `work/sealed_parts.jsonl`).

**Preflight:** checked as line.sh's `_vs3` arm does.
- The main preflight on L2r/L3r prints PREFLIGHT PASSED.
- The extra-targets preflight on L2_3p/L3_3p (`curriculum_3p/preflight_grpo_nothink_vs3_s0_extra.log`) has 22 of 22
  overlap checks ok.
- Its only BAD lines are the length bands (592 rows outside 10–12, 1,450 outside 13–15). line.sh ignores those by
  design, as on serrano.
