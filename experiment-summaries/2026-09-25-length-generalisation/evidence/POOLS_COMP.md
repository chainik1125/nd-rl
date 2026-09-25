---
written_on: 2026-09-25
---
# ab3 composition-training targets (built 2026-09-25 01:12-02:20 UTC, CPU only, at most 36 cores)

Written by the pool-builder session (agent:claude), at the monitor's request with the PI's approval. The test sets
these targets must stay apart from are described in `/scratch/cowifornia/nd-rl/ab3/data/POOLS.md`.

**Files (agent-visible):**
- `targets_constructed_L6-11.jsonl`: 6,744 constructed RL targets at `L_true` 6-11, with no proofs.
- `targets_L6-11.jsonl`: 22,482 targets, merging run 2's standard `ab2/data/targets_L6-11.jsonl` (15,738, byte-for-byte
  field values, plus `family: null`) with the 6,744 constructed ones. Names are unique; the file is shuffled with seed
  33991.
- `SHA256SUMS`.

Fields are run 2's whitelist plus `family`: `name, thm, key, pkey, prompt, n_lines (= L_true), L_true, source, schema,
n_prem, family`. For constructed records, `source` is `ab3_andi` or `ab3_chain`, `family` is `andi`, `chaink1` or
`chaink2`, and `schema` is null.

The non-visible provenance (parts, part lengths, constructed and minimal proofs, candidate and label files) is in
`/scratch/cowifornia/nd-rl/ab3/data_build/comp/provenance_constructed.jsonl`.

## Counts (merged bin = standard + constructed; constructed share 30%)
| L_true | standard | andi | chaink1 | chaink2 | constructed | merged | share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1,000 | 58 | 84 | 288 | 430 | 1,430 | 30.1% |
| 7 | 2,500 | 536 | 251 | 284 | 1,071 | 3,571 | 30.0% |
| 8 | 2,500 | 536 | 203 | 332 | 1,071 | 3,571 | 30.0% |
| 9 | 2,500 | 536 | 226 | 309 | 1,071 | 3,571 | 30.0% |
| 10 | 2,500 | 536 | 269 | 266 | 1,071 | 3,571 | 30.0% |
| 11 | 4,738 | 1,015 | 496 | 519 | 2,030 | 6,768 | 30.0% |
| total | 15,738 | 3,217 | 1,529 | 1,998 | 6,744 | 22,482 | |

ANDI and chain split evenly (ANDI gets the odd one) except at **L6**. There, ANDI supply is only **58 distinct
classes**. An L6 ANDI theorem needs a 2-line part with at most 2 atoms; the reserve has 38 such parts, and 21 of them
are test parts. So chain fills the remaining 372. Those 58 L6 ANDI targets also reuse one 2-line part in 93% of cases
(29 distinct parts). At L7-11, ANDI parts are diverse: 691-1,786 distinct, with no part in more than 1%.

## Construction and label: the same as the test families
- **Code and settings.** The same `tools/build_cands.py`, with atom renaming, premise interleaving, satisfiable
  (truth-table) parts, at most 5 premises and the same filters:
  - ANDI: parts with at most 2 atoms, and `--max_prompt_tok 60`.
  - chain: k1 at constructed 15-19 and k2 at 16-24, in equal numbers of candidates. The L6-11 chain targets are the
    composites whose exact length landed there, exactly as in the matched test set.
  - The only difference is `--exclude_parts comp/test_parts.json`, plus new seeds.
- **Label.** Exact minlen only, with no certificate labels:
  - ANDI: bound 18, 120 s. Every one of the 6,557 candidates resolved, and each equals L1 + L2 + 1.
  - chain: pass A at bound 12 (rounds 1-2) or bound 11 (rounds 3-4), 15 s. For `L_true` ≤ 11 these give the same
    exact label as bound 12. Driver: `tools/label3.py --intern`.
- **Side effect of part exclusion.** Test parts are removed from the reserve, and the long test sets used relatively
  more long parts (L8-10). Mean constructed length therefore dips slightly: chaink1 15.7 in training against 15.9 in
  the matched L6-11 test records, and chaink2 17.5 against 18.0.

## Disjointness (all checked by `tools/comp_verify.py`)
1. **Class disjointness.** Constructed targets are pkey- and key-disjoint from:
   - `ab3/sealed/matched_L6-14`, `sealed_L15-16` and `sealed_L17-18`;
   - `ab2` `dev_L8-11`, `frontier_L12-14`, `sealed_L12-14` and `sealed_L15-16`;
   - the standard `targets_L6-11`, cap-6 `train_cap6` and `heldout_cap6`, textbook-dev (`thm` only) and
     validation-36.

   All of these are recomputed from the source files. Assembly also excluded everything in `excl.json`, which adds
   the ab2 reserve, the run-1 sealed sets and the ladder pools.
2. **Part disjointness.** The ab3 test theorems use **8,270** distinct reserve parts; the training targets use
   **9,720**. There is **no overlap by reserve name, and none by part pkey** recomputed from `thm`. pkey is invariant
   under injective atom renaming (checked on 2,000 random renamings), so "before and after renaming" is the same
   condition.
3. **Test parts that are standalone standard training targets: 0** (by pkey and by key). This holds by construction:
   parts come from the ab2 reserve, which is pkey-disjoint from `targets_L6-11`. Test parts are also, as whole
   theorems, not in `targets_constructed`, whose classes are composites.

Other checks:
- Visible fields are exactly the whitelist; `key` and `pkey` are recomputed; `n_lines == L_true`; names and pkeys are
  unique.
- The standard records in the merged file equal the ab2 records.
- **13,488 `nd_verify` checks** (the minimal proof and the constructed proof of each constructed target), **0
  failures**.

## Commands (from `ab3/data_build`) and seeds
```
python3 tools/build_cands.py --family andi --lmin 6 --lmax 11 --n 100000 --per_len 1300 --max_prompt_tok 60 --seed 33715 --max_tries 5000000 --exclude_parts comp/test_parts.json --out comp/andi_tr_33715.jsonl
python3 tools/build_cands.py --family chain --k 1 --lmin 15 --lmax 19 --n 6000 --seed S --exclude_parts comp/test_parts.json --out comp/chain_tr_S.jsonl   # S = 33711 33712 33721 33731 33733 33741
python3 tools/build_cands.py --family chain --k 2 --lmin 16 --lmax 24 --n 6000 --seed S --exclude_parts comp/test_parts.json --out comp/chain_tr_S.jsonl   # S = 33713 33714 33722 33732 33734 33742
python3 tools/prep.py --in comp/andi_tr_33715.jsonl --out comp/andi_trp.jsonl --seed 33716
python3 tools/prep.py --in comp/chain_tr_all{,2,3,4}.jsonl --out comp/chain_trp{,2,3,4}.jsonl --seed 33717|33723|33735|33743   # rounds 1-4
python3 tools/label3.py --in comp/andi_trp.jsonl --out labels/comp_andi_b18.jsonl --bound 18 --time 120 --procs 8 --mem_gb 1.2 --maxtasks 50 --intern
python3 tools/label3.py --in comp/chain_trp.jsonl  --out labels/comp_chain_b12.jsonl  --bound 12 --time 15 --procs 24 --mem_gb 0.8 --maxtasks 50 --intern
python3 tools/label3.py --in comp/chain_trp2.jsonl --out labels/comp_chain2_b12.jsonl --bound 12 --time 15 --procs 11 --mem_gb 0.8 --maxtasks 50 --intern
python3 tools/label3.py --in comp/chain_trp3.jsonl --out labels/comp_chain3_b11.jsonl --bound 11 --time 15 --procs 24->35 --mem_gb 0.8 --maxtasks 50 --intern [--resume]
python3 tools/label3.py --in comp/chain_trp4.jsonl --out labels/comp_chain4_b11.jsonl --bound 11 --time 15 --procs 35 --mem_gb 0.8 --maxtasks 50 --intern
python3 tools/comp_assemble.py --seed 33991 && python3 tools/comp_verify.py
```
`comp/test_parts.json` holds the names and pkeys of every part of every theorem in the three ab3 test files.

Supply:
- ANDI: 6,557 candidates, all resolved; 3,217 used.
- Chain: 70,309 candidates in four rounds (about 26 CPU-h of pass A). 32,258 were unresolved: the search was
  exhausted at the bound (`L_true` above 11 or 12) or timed out. Another 6,056 resolved outside 6-11 (at ≤ 5, or at
  12 in the bound-12 rounds). The rest resolved at 6-11, of which 3,527 were used.
- Round 2 onwards was needed only to fill the L6 bin; chain L6 yield is about 0.6% of candidates.
- Wall clock: 01:12-02:20 UTC.

## Caveats
- The constructed targets have **about twice as many premises as run 2's** (chain 3.5-4.7, ANDI 2.9-3.8, against
  1.5-2.2 in run 2). ANDI conclusions are always `&`. An arm trained on the merged pool therefore sees a premise-count
  and shape shift, not only composition.
- L6 ANDI is 58 near-duplicate-part theorems, as described above. L6 constructed is 87% chain.
- Test theorems at 6-14 (matched) and 15-18 (sealed) are built from **unseen parts** and are class-disjoint. But the
  test and training families share the same part *distribution* (reserve generator theorems), so this measures
  composition of new pieces of a familiar kind.
