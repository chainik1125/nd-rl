---
written_on: 2026-09-26
---
# ab3 mid-training data: glued long proofs (built 2026-09-26 04:24-06:05 UTC, CPU only, at most 36 cores)

Written by the pool-builder session (agent:claude), at the monitor's request with the PI's approval. It is for the
"mid-training on long proofs" test. Build files are in `/scratch/cowifornia/nd-rl/ab3/midtrain/build/`: `tools/`,
`raw/`, `labels/`, `logs/` (including `stage_times.txt`) and `assemble_summary.json`. Construction helpers are shared
with the ab3 test sets (`ab3/data_build/tools/comp.py`, `keys2.py`, `label3.py`, `excl.py`).

| file | what | records |
|---|---|---:|
| `sft_long.jsonl` | supervised glued ANDI proofs, written length 16-24 (222-223 per length) | 2,000 |
| `sft_short.jsonl` | supervised glued ANDI proofs, written length 6-11 (333-334 per length); the control arm | 2,000 |
| `test_long.jsonl` | held-out test theorems at 22, 24 and 26 lines (150 each); monitor-only | 450 |

## Construction (`tools/build_mid.py --seed 34001 --split_seed 34000`)
- **Parts.** Parts are ab2-reserve generator theorems with at most 2 atoms, a verified minlen proof and satisfiable
  premises (truth table).
- **Renaming.** k parts are renamed injectively onto **pairwise disjoint atom sets**: 2 + 2 atoms for k = 2, and
  1 + 1 + ≤2 atoms for k = 3 (there are only 4 atoms).
- **Joining.** Premises are merged and interleaved at random. The conclusion is a random binary ANDI tree over the
  part conclusions, such as `(A & B) & C` or `A & (B & C)`, in any order. The proof is the parts' minimal proofs
  concatenated (references renumbered) plus k − 1 ANDI lines, so the written length is ΣL_i + k − 1.
- **ANDI joins only.** Chain (cut) composites are not used, so they stay unseen as a glue type.
- **Filters.** At most 5 premises, prompt ≤ 150 tokens, and **prompt plus proof ≤ 800 tokens** (frozen tokenizer, abs
  mode). Each part is used at most 8 / 15 / 40 times in test_long / sft_long / sft_short, with draws down-weighted by
  1/(1+uses)².
- **k mix.** k = 3 is targeted at 50% wherever both k are possible:
  - k = 2 reaches at most 21 lines, so 22-26 are all k = 3.
  - Short k = 3 joins need 2-3-line parts, which are scarce. sft_short therefore has k = 3 only at 9 (12), 10 (46) and
    11 (166).
- **Record format.** sft files follow `frozen/data/train_cap6.jsonl`: `name, thm, key, prompt, n_lines` (= written
  proof length), `rules, n_prem, proof, text`. They add `pkey, family (andi_k2 | andi_k3), parts, part_L, part_pkeys,
  constructed_lines, n_tokens, exact_checked, exact, exact_L`. `test_long` carries `n_lines = L_true` (the
  constructed length), `label_kind`, `cert14` and `constructed_proof`. It has **no `proof` field**, on purpose.
- **Training note.** `lib/train.load(..., cap)` asserts `n_lines <= cap`, so use cap ≥ 24 (or 0) for sft_long.

## Counts per length
| file | L | n | k=2 | k=3 | premises | prompt tok mean / max | prompt+proof tok median / max | distinct parts | top part share |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| sft_short | 6 | 334 | 334 | 0 | 3.0 | 103 / 147 | 273 / 374 | 329 | 10% |
| sft_short | 7 | 334 | 334 | 0 | 3.9 | 89 / 148 | 235 / 376 | 542 | 1% |
| sft_short | 8 | 333 | 333 | 0 | 4.1 | 86 / 150 | 240 / 401 | 619 | 1% |
| sft_short | 9 | 333 | 321 | 12 | 4.0 | 88 / 149 | 261 / 490 | 641 | 2% |
| sft_short | 10 | 333 | 287 | 46 | 4.2 | 92 / 150 | 280 / 477 | 617 | 3% |
| sft_short | 11 | 333 | 167 | 166 | 4.5 | 106 / 150 | 360 / 510 | 492 | 8% |
| sft_long | 16 | 223 | 111 | 112 | 4.3 | 110 / 150 | 462 / 661 | 509 | 1% |
| sft_long | 17 | 223 | 111 | 112 | 4.2 | 113 / 150 | 505 / 698 | 510 | 2% |
| sft_long | 18 | 222 | 111 | 111 | 4.3 | 113 / 150 | 506 / 704 | 510 | 2% |
| sft_long | 19 | 222 | 111 | 111 | 4.3 | 114 / 150 | 528 / 738 | 500 | 1% |
| sft_long | 20 | 222 | 111 | 111 | 4.3 | 116 / 150 | 559 / 776 | 474 | 2% |
| sft_long | 21 | 222 | 111 | 111 | 4.3 | 116 / 150 | 576 / 765 | 386 | 2% |
| sft_long | 22 | 222 | 0 | 222 | 4.7 | 128 / 150 | 634 / 797 | 524 | 3% |
| sft_long | 23 | 222 | 0 | 222 | 4.6 | 128 / 150 | 650 / 787 | 494 | 3% |
| sft_long | 24 | 222 | 0 | 222 | 4.6 | 127 / 150 | 678 / 798 | 472 | 7% |
| test_long | 22 | 150 | 0 | 150 | 4.6 | 129 / 150 | 638 / 794 | 392 | 2% |
| test_long | 24 | 150 | 0 | 150 | 4.7 | 130 / 150 | 682 / 788 | 363 | 3% |
| test_long | 26 | 150 | 0 | 150 | 4.7 | 131 / 150 | 724 / 800 | 287 | 5% |

**Token lengths.** The maximum prompt plus proof is **800 tokens** (test_long), 798 (sft_long) and 510 (sft_short), out
of 1,024; the cap guarantees at least 224 tokens of room. The longest prompt is 150 tokens. Line references go up to
N26; the tokenizer has N1..N64. `shift_abs` augmentation offsets a 24-line proof by at most 40.

## Labels and exactness
- **sft_short:** exact minlen (bound = written length, 30 s) on all 2,000. **2,000 of 2,000 equal the written
  length** (`exact: true`).
- **sft_long:** exact search is not cheap here, because prompts reach 150 tokens. A 60 s budget on 90 of the L16-17
  records resolved 2, both equal (`exact_checked: true`, `exact: null` for the rest). All other records are
  `exact_checked: false`. The records are supervised proofs, so nothing depends on this.
- **test_long:** `L_true` = constructed length ΣL_i + 2, with `label_kind: constructed_additive+cert14` for **all
  450**.
  - `cert14` means the restricted exhaustive search found **no proof of length ≤ 14** (bound 14; 445 at 300 s, the
    other 5 at 1,200 s). So L_true ≥ 15 is proved, and L_true ≤ the label holds by a verified proof.
  - Equality rests on additivity:
    - **Two-part ANDI:** 2,736 of 2,736 exact in the ab3 build.
    - **Three-part joins, new here:** 420 joins at written length 11-17 were built the same way from SFT-side parts
      (`raw/validate_k3.jsonl`, seed 34101; not in any output). **343 of 343 resolved cases equal the written
      length** (11: 60, 12: 60, 13: 60, 14: 60, 15: 57, 16: 44, 17: 2), with 0 counter-examples. 29 timed out or ran
      out of memory at 400 s.
  - An exact label at 22-26 is not affordable (bound-18 searches already run out of memory at 17-18).

## Disjointness (`tools/verify_mid.py`, all pass)
- **Parts, midtrain against old tests.** Midtrain parts are disjoint from all 17,990 parts used in `ab3/sealed`
  (`sealed_L15-16`, `sealed_L17-18`, `matched_L6-14`) and `ab3/data_comp`, both by reserve name and by part pkey
  recomputed from `thm`. pkey is invariant under atom renaming, so this covers "before and after renaming".
- **Parts, SFT against test.** sft_long and sft_short parts (4,508 distinct) are disjoint from test_long parts (773) by
  name and by pkey. Parts were split by pkey before building (stratified 50/50 for part length ≥ 5; parts of length
  ≤ 4, useless at 22-26, all went to SFT).
- **Test parts that are standard training targets: 0**, by construction (the reserve is pkey-disjoint from
  `targets_L6-11`).
- **Theorems.** Keys and pkeys are unique across the three files. No key or pkey matches any existing file; these are
  recomputed from the sources:
  - cap-6 train / held-out, textbook-dev (`thm` only), validation-36;
  - every run-2 file including reserve and sealed, run-1 sealed, ladder pools;
  - `ab3/sealed/*` and `ab3/data_comp/*`.
- **Proofs.** 6,452 `nd_verify` checks, **0 failures**:
  - every sft proof and every test constructed proof, with lines == `n_lines`;
  - every minlen proof behind an `exact` flag.

  For each record, the last k − 1 lines are the ANDI joins, and the k joined part conclusions have pairwise disjoint
  atoms.

## Commands (from `ab3/midtrain/build`)
```
python3 tools/build_mid.py --seed 34001                      # split seed 34000; caps 8/15/40; prompt <= 150; total <= 800 tokens
L3=/scratch/cowifornia/nd-rl/ab3/data_build/tools/label3.py
python3 $L3 --in raw/test_long_cert.jsonl  --out labels/test_long_cert14.jsonl --bound 14 --time 300 --procs 22 --mem_gb 2 --maxtasks 10 --intern
python3 $L3 --in raw/test_long_cert_retry.jsonl --out labels/test_long_cert14_retry.jsonl --bound 14 --time 1200 --procs 6 --mem_gb 3 --maxtasks 5 --intern
python3 $L3 --in raw/sft_short_exact.jsonl --out labels/sft_short_exact.jsonl --bound 11 --time 30 --procs 8 --mem_gb 1.2 --maxtasks 50 --intern   # per-record bound = n_lines
python3 $L3 --in raw/sft_long_exact.jsonl  --out labels/sft_long_exact.jsonl  --bound 17 --time 60 --procs 9 --mem_gb 2 --maxtasks 10 --intern    # stopped after 90 (not cheap)
python3 tools/validate_k3.py --seed 34101 --per 60
python3 $L3 --in raw/validate_k3.jsonl --out labels/validate_k3.jsonl --bound 17 --time 300|400 --procs 5|12 --mem_gb 2.5 --maxtasks 10 --intern [--resume]   # stopped 06:05
python3 tools/assemble_mid.py && python3 tools/verify_mid.py
```
Wall clock: build 05:03-05:07. Labelling 05:10-06:05: the certificate pass took about 8 CPU-h, validation about 6
CPU-h, and the exact checks about 2 CPU-h. Earlier builder iterations, 04:24-05:03, tuned the part split, reuse caps and
prompt cap; their outputs were overwritten.

## Caveats
- **Prompt size.** Prompts are large (mean 86-131 tokens, against 43-58 in run 2 and at most 60 in the ab3 ANDI test
  family). They also grow with length (sft_short about 95, sft_long 110-128, test_long about 130), because longer
  joins carry more and bigger parts. Part of what "long" teaches is long prompts.
- **k and length are correlated.** 22-26 are 100% k = 3, sft_long 16-21 are 50%, and sft_short is 0-50% (only 9-11
  have k = 3). A short-arm failure on test_long could partly reflect the 3-part join shape.
- **test_long labels are upper bounds** with a proved lower bound of 15. Equality is supported by 2- and 3-part
  additivity (3,079 of 3,079 resolved cases), which is checked only up to 17 lines.
- **Part reuse.** The long 1-atom parts are few (L7: 71, L8: 27, L9: 5, L10: 2 on the test side), so test_long at 26
  reuses parts more (287 distinct parts; the most-used part is in 5% of items).
