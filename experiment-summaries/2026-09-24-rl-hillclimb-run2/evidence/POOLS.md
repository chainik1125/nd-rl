---
written_on: 2026-09-24
---
# ab2 theorem pools (built 2026-09-24 00:09–02:20 UTC, CPU only, at most 40 of 48 cores, no GPU)

Tooling: fork clone `/scratch/cowifornia/nd-rl/external/nd-takehome` at `ab629c7` (read-only; scripts run from there),
plus local wrappers in `tools/`: `label.py` (a resumable, deadline-aware driver around the fork's `minlen.minlen`
that uses the same search and the same label; it flushes each record and puts a per-worker memory limit),
`pass2_input.py`, `pass2_loop.sh`, `keys2.py` (`pkey`), `assemble.py`, `verify_pools.py`. Logs are in `logs/`,
raw candidates in `raw/`, and all minlen outputs in `labels/`.

## Label (`L_true`), same definition as `data/ladder/POOLS.md`
`L_true` is the minimal proof length found by `minlen.py`'s restricted iterative-deepening search. The search limits
hypotheses, lemmas and rule partners to the subformula closure of the theorem plus negations and double negations,
and every proof it finds is checked with `nd_verify`. **Caveat:** "no proof of length at most L−1" holds only
*inside the restricted formula space*. The true minimal length can be lower if the only shorter proof uses a formula
outside the closure. Unresolved theorems (timeout, OOM, or none found up to the bound) are dropped and never counted.
Each record keeps its bound as `minlen_bound`: 10 or 16 here, 8 or 14 for the old ladder records, and 6 for the
held-out L6 records.

## Commands (run from the fork directory unless noted)
```
# 1. generation (make_coverage_sets.py; cap_pat raised so pattern caps never cut long proofs; cap_np default 1500)
python make_coverage_sets.py gen --long --min 10 --max 22 --gen_max_depth 5 --gen_max_prem 4 --workers 40 --tries 250000 --seed 24001 --cap_pat 1000000000 --out $D/raw/gen_d5p4
python make_coverage_sets.py gen --long --min 10 --max 22 --gen_max_depth 6 --gen_max_prem 5 --workers 40 --tries 250000 --seed 24101 --cap_pat 1000000000 --out $D/raw/gen_d6p5
python make_coverage_sets.py merge --glob "$D/raw/gen_d*.w*.jsonl" --out $D/raw/pool_gen.jsonl --prefix ab2g
# 2. schema instances
python textbook_pool.py --n 6000 --seed 24002 --out $D/raw/textbook.jsonl --exclude <train_cap6> <heldout_cap6> data/ladder/{rl_targets,transfer,reserve}.jsonl
# 3. labelling (from $D; the gen pass-1 input is pool_gen shuffled with seed 24003, then ordered n_prem>=2, 1, 0 and gen_lines>=12 first)
python tools/label.py --in raw/textbook.jsonl --out labels/tb_b10.jsonl --bound 10 --time 10 --procs 40
python tools/label.py --in raw/tb_u.jsonl     --out labels/tb_b16.jsonl --bound 16 --time 120 --procs 20
python tools/label.py --in raw/pool_gen_ordered.jsonl --out labels/gen_b10.jsonl --bound 10 --time 10 --procs 20|40 --deadline ... [--resume]
tools/pass2_loop.sh 5400; then python tools/pass2_input.py && python tools/label.py --in raw/gen_u.jsonl --out labels/gen_b16.jsonl --bound 16 --time 120 --procs 40 --deadline 2800 --resume
python tools/label.py --in raw/heldout6.jsonl --out labels/heldout6_b6.jsonl --bound 6 --time 10 --procs 20
# 4-5. assembly and checks
python tools/assemble.py --seed 24005 && python tools/verify_pools.py
```
Seeds: generation 24001 (depth 5 / 4 premises) and 24101 (depth 6 / 5 premises); textbook 24002; pass-1 order 24003;
pass-2 order 24004; assembly 24005. The pilot (4,000 random candidates, `raw/pilot.jsonl`, seed 0) was used only to
choose the labelling order. Its labels are not in the pools.

## Stage wall clock
| stage | UTC | wall | notes |
|---|---|---:|---|
| generation d5p4 + d6p5 + merge | 00:09:43–00:11:31 | ~2 min | budget 45 min; stopped because the class target was met many times over |
| pilot labelling | 00:12–00:17 | 5 min | |
| textbook: generate + pass 1 | 00:17:29–00:19:48 | 2.3 min | 40 procs |
| textbook pass 2 | 00:20:29–00:45:00 | 24.5 min | 20 procs, alongside generator pass 1 |
| generator pass 1 | 00:21:43–00:46 and 01:29:40–01:41:45 | 24 + 12 min | 20 procs, then 40 procs |
| generator pass 2 | 00:45:33–01:29 and 01:41:45–02:13:47 | 43 + 32 = **75 min** | 40 procs; queue fully drained, under the 90 min cap |
| assembly + verification | 02:14–02:20 | ~5 min | |

## Generation yield
80 shards held 591,921 verifier-accepted proofs (make_coverage_sets verifies each proof before writing it;
**0 `VERIFIER_REJECT`** in either run, and `tools/verify_pools.py` re-checks every generating proof behind every
output record: 0 failures). The d5p4 run had 10M tries, 7.72M generator failures and 232,605 contradictory-premise
theorems dropped. The d6p5 run had 10M tries, 8.00M failures and 186,151 contradictory. Merge dropped 281
validation-36 classes and 31,695 duplicate classes, leaving **559,945 distinct classes**, of which **386,198 have
gen_lines ≥ 12**.

| gen_lines | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| classes | 101,076 | 72,671 | 69,504 | 58,966 | 51,286 | 43,717 | 37,120 | 31,392 | 25,783 | 22,092 | 18,332 | 15,304 | 12,702 |

**Generation length is a poor proxy for true length.** Of 336k labelled classes with gen_lines ≥ 12, 91% have
`L_true` ≤ 9. Only 0.55% have `L_true` 12 and 0.03% have `L_true` 14.

## Labelling yield
Pass 1 (bound 10, 10 s) costs 0.05 s per candidate with n_prem ≥ 2, 0.5 s with n_prem 1 and 3.2 s with n_prem 0.
In the pilot, n_prem ≥ 1 gave 20–30× more L ≥ 12 labels per CPU-second than n_prem 0. So pass 1 labelled all
385,638 n_prem ≥ 2 candidates and 38,428 of 119,961 n_prem 1 candidates, for 424,066 in total. **135,879 candidates
were never labelled** (all 54,346 with n_prem 0 and 81,533 with n_prem 1); they stay in `raw/pool_gen.jsonl`.
Pass 2 (bound 16, 120 s) ran on every pass-1-unresolved candidate: 7,038 whose search was exhausted at bound 10
(so `L_true` ≥ 11 within the space) came first, then 640 timeouts.

| source / pass | n | ≤6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gen pass 1 (b10) | 424,066 | 257,171 | 86,251 | 40,714 | 22,274 | 9,978 | | | | | | | 7,678 (7,038 exhausted, 640 timeouts) |
| gen pass 2 (b16) | 7,678 | | | | 1 | 207 | 4,577 | 1,838 | 507 | 85 | 9 | 0 | **454** (437 timeouts, 16 OOM, 1 exhausted) |
| textbook pass 1 (b10) | 5,668 | 1,559 | 1,173 | 272 | 402 | 1,195 | | | | | | | 1,067 (1,047 exhausted, 20 timeouts) |
| textbook pass 2 (b16) | 1,067 | | | | | 6 | 201 | 575 | 118 | 138 | 0 | 0 | **29** (timeouts) |
| cap-6 held-out, gen length 6 (b6) | 1,000 | 999 (2–5: 360; 6: 639) | | | | | | | | | | | 1 (none found) |

Textbook: 5,668 instances were produced (not 6,000, because `dn_elim`, `triple_negation`, `excluded_middle` and
`dn_intro` ran out of distinct classes). The schema family appears at every length from 6 to 14 but **not at 15–16**:
no schema instance labelled above 14. It is thin at 13 (only `dist_or_over_and_conv` and `demorgan_nand_to_or`)
and at 14 (only `demorgan_nand_to_or`). Per-schema label histograms can be reproduced from `labels/tb_b*.jsonl`.

## Keys, exclusions, dedup
`key` = `keys.canon_key(thm)`. `pkey` = the lexicographic minimum of `canon_key` over all premise permutations
(`tools/keys2.py`; at most 5 premises). Excluded key and pkey sets: cap-6 `train_cap6` (154,990) and `heldout_cap6`
(5,000); the 58 sealed textbook-dev problems (only `key` read, and `pkey` computed from `thm`; nothing else from that
file is used or copied); and validation-36 (an extra safety measure). That makes 160,066 keys and 158,451 pkeys.

Candidates are the old ladder pools (rl_targets, transfer and reserve; 22,987) first, then new textbook, then new
generator records, all labelled. Removed:

| origin | excluded (by key / only by pkey) | pkey duplicate dropped |
|---|---|---:|
| old ladder gen | 1 (1 / 0), validation-36 | 21 |
| old ladder textbook | 1 (0 / 1), textbook_dev | 0 |
| new textbook | 9 (0 / 9), train_cap6 | 0 |
| new gen | 2,324 (1,874 / 450): train_cap6 2,265, heldout_cap6 111, textbook_dev 1, validation-36 1 (overlapping) | 3,423 |

That leaves 446,459 records. Duplicates keep the earlier record, so old records always win.

## Assembly (`tools/assemble.py --seed 24005`)
Records are drawn in this order: dev, then frontier/sealed, then sealed 15–16, then targets, and the reserve takes
the rest. `draw(L, n)` keeps the textbook share at or below 40% unless the generator runs out, and draws textbook
instances round-robin over schemata. **dev**: 300 each at L 8, 9, 10 and 11. **L 12–14**: every theorem, grouped by
(L, source), ordered by schema and randomly within a schema, then alternately assigned frontier / sealed.
**sealed_L15-16**: everything labelled 15 or 16. **targets**: up to 2,500 at L 7–10, all of what is left at L 11.
The L6 bin is **1,000 records**: the 639 `heldout_cap6` records with gen length 6 whose `L_true` is 6, topped up
with 361 new records at `L_true` 6. The other 360 held-out gen-6 records have `L_true` 2–5 and 1 is unresolved, so
they are not used. The held-out classes are exempt from the held-out exclusion only in this bin.

Agent-visible files (`targets_*`, `dev_*`, `frontier_*`) carry only `name, thm, key, pkey, prompt, n_lines (= L_true),
L_true, source, schema, n_prem`. Sealed files and the reserve keep everything: `orig_name`, `origin`, `minlen_bound`,
`gen_lines`, `rules`, `gen_proof` and `minlen_proof`. `provenance.jsonl` (not agent-visible) maps every output `name`
to its original record.

| file | source | ≤5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| targets_L6-11 | cap6_heldout | | 639 | | | | | | | | | | | 639 |
| targets_L6-11 | gen | | 217 | 1,500 | 2,145 | 1,915 | 1,500 | 4,535 | | | | | | 11,812 |
| targets_L6-11 | textbook | | 144 | 1,000 | 355 | 585 | 1,000 | 203 | | | | | | 3,287 |
| dev_L8-11 | gen | | | | 180 | 180 | 180 | 180 | | | | | | 720 |
| dev_L8-11 | textbook | | | | 120 | 120 | 120 | 120 | | | | | | 480 |
| frontier_L12-14 | gen | | | | | | | | 939 | 258 | 45 | | | 1,242 |
| frontier_L12-14 | textbook | | | | | | | | 475 | 103 | 115 | | | 693 |
| sealed_L12-14 | gen | | | | | | | | 939 | 257 | 44 | | | 1,240 |
| sealed_L12-14 | textbook | | | | | | | | 475 | 102 | 114 | | | 691 |
| sealed_L15-16 | gen | | | | | | | | | | | 9 | 0 | 9 |
| reserve | gen | 123,697 | 128,462 | 96,494 | 43,606 | 21,827 | 8,994 | | | | | | | 423,080 |
| reserve | textbook | 756 | 650 | 932 | | | 867 | | | | | | | 3,205 |

Totals: targets 15,738, dev 1,200, frontier 1,935, sealed_L12-14 1,931, sealed_L15-16 9, reserve 426,285.
Before this build, the existing labelled theorems numbered 12: 415, 13: 95, 14: 95. They are now **12: 2,828,
13: 720, 14: 318, 15: 9**. The old ladder records are included and account for 12: 415, 13: 95, 14: 95.

## Checks (`tools/verify_pools.py`, all pass)
- Visible files contain exactly the whitelisted fields.
- `key` and `pkey` are recomputed and match; `n_lines == L_true`; names are unique.
- All files are pairwise pkey-disjoint and contain no excluded key or pkey, except the 639 intended held-out L6
  targets.
- 862,237 `nd_verify` checks, **0 failures**. They cover every generating proof (lines == gen_lines) and every minlen
  proof (lines == `L_true`) behind each new record, and the generating proof of each old ladder generator record.
  Old ladder textbook records carry no proof, so their `L_true` comes from the ladder labelling.

## What limited the yield
1. **True-length collapse, not generation.** Deeper generator settings make long *proofs* cheaply but mostly not
   long *theorems* (see the gen_lines table).
2. **Pass 2 at bound 16** resolved everything it was given. Only 5.8% of pass-2 inputs stayed unresolved: 437
   timeouts at 120 s and 16 OOMs at 1.2 GB per worker. These are the likeliest candidates for L ≥ 15, and they
   were dropped.
3. **L 15–16 is nearly unreachable** in this budget: 9 generator theorems at L 15, none at 16, and no schema
   instance above 14. L 14 is mostly textbook (`demorgan_nand_to_or`); the generator gives about 0.02% per labelled
   candidate.
4. 135,879 generator candidates (every n_prem 0 and most n_prem 1) were never labelled, because pass 2 was the
   binding budget and these are the most expensive per L ≥ 12 label.
