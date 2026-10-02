---
written_on: 2026-09-25
---
# ab3 long test sets (built 2026-09-24 20:05 to 2026-09-25 01:10 UTC, CPU only, at most 36 of 48 cores, no GPU)

Written by the pool-builder session (agent:claude). The feasibility study is
`/scratch/cowifornia/nd-rl/ab3/data_build/FEASIBILITY.md`.

**What this is.** These are monitor-only test sets beyond run 2's training range, at `L_true` 15-18. Next to them is a
*matched* set with the same constructions at 6-14. Nothing here goes into any training pool, and run 3 reuses run 2's
agent-visible pools (`ab2/data/targets_L6-11`, `dev_L8-11`, `frontier_L12-14`) unchanged.

| file | lengths | records |
|---|---|---:|
| `/scratch/cowifornia/nd-rl/ab3/sealed/sealed_L15-16.jsonl` | 15, 16 | 1,200 |
| `/scratch/cowifornia/nd-rl/ab3/sealed/sealed_L17-18.jsonl` | 17, 18 | 736 |
| `/scratch/cowifornia/nd-rl/ab3/sealed/matched_L6-14.jsonl` | 6-14 | 3,599 |

Also in `ab3/data/`: `provenance.jsonl` (maps every output name to its candidate, parts, label file and driver),
`pools_summary.json` and `SHA256SUMS`. Working files are in `ab3/data_build/`: `raw/` holds candidates, `labels/`
holds every minlen output, and `logs/` holds the logs, including `stage_times.txt`. Tools are in `ab3/data_build/tools/`.

## Read this first: construction, not run-2 style
The run-2 generator cannot reach these lengths. Deeper settings (`--gen_max_depth 7 --gen_max_prem 5 --min 14`) gave
0 theorems at L15+ from 5,106 classes, and run 2 found 9 at L15 in about 62 CPU-h. So **every long theorem is
constructed** by composing two or three ab2-reserve generator theorems ("parts"). The reserve is never in any run-2
training, dev, frontier or sealed file.

Neither construction occurs in run 2's 6-11 pool:
- Separable conjunctions (the ANDI shape) are 0.3% of `targets_L6-11`.
- Composites of two generator theorems are never generated.

Use `matched_L6-14` to tell a length effect from a construction effect. Every record carries `family`, so each family
has its own length curve from 6 to 18:
- `andi` covers 6-18.
- `chaink1` and `chaink2` cover 6-18, but they are thin at 17 (116) and 18 (20).

## Families (tools/build_cands.py; comp.py holds the construction helpers)
Parts are ab2 reserve records with `source == gen`, a verified `minlen_proof`, and **satisfiable premises**
(truth table over P, Q, R, S; about 15% of reserve theorems have contradictory premises and are dropped). Every
composite has at most 5 premises, satisfiable premises, a conclusion that is not a premise, no duplicate premises, and
a prompt of at most 150 tokens. Its constructed proof, built from the parts' proofs, is `nd_verify`-checked.

- **`andi`, disjoint conjunction.**
  - Two parts, each with at most 2 atoms, are renamed injectively onto disjoint atom pairs (a random split of
    P, Q, R, S).
  - The result is Γ1, Γ2 ⊢ φ1 & φ2, with premises interleaved at random and conjunct order random.
  - The constructed proof has L1 + L2 + 1 lines.
  - Prompts are capped at **60 tokens at every length**. Without the cap, exact minlen at 17-18 runs out of memory;
    with it, the matched set is comparable.
- **`chaink1` and `chaink2`, cut composition.**
  - A consumer is Γ2, π ⊢ ψ, where π is a compound premise that the consumer's minimal proof cites.
  - A producer is Γ1 ⊢ φ, with φ equal to π after an injective renaming. Any extra atoms go to atoms outside π.
  - The result is Γ1, Γ2 ⊢ ψ. `chaink2` replaces a second compound premise with a second producer.
  - The constructed proof (producer body, then consumer body) is **only an upper bound**. Extra premises open
    shortcuts, so most composites collapse, and the label is always exact minlen.
  - Constructed lengths are 15-19 for k1 and 16-24 for k2. The matched 6-14 chain records come from the *same*
    candidate process; they are simply the composites whose exact length landed there.

## Label
- **`label_kind: exact` (5,535 − 578 = 4,957 records):** `L_true` is the minimal proof length found by the fork's
  `minlen` restricted iterative-deepening search, with bound 18. This is the same definition as run 2: subformula
  closure plus negations and double negations, and the proof is checked by `nd_verify`. The caveat also carries over:
  "no shorter proof" holds only inside the restricted formula space.
- **`label_kind: ub_cert14` (578 records, ANDI only: 17: 257, 18: 291, 15: 6, 16: 24):** exact minlen was not
  attempted or not affordable. Instead, the exhaustive restricted search at bound 14 found **no proof of length ≤ 14**
  (so `L_true` ≥ 15), and `L_true` is set to the constructed length L1 + L2 + 1, which is a verified upper bound.
  `minlen_proof` is null, and `constructed_proof` is the verified proof.
  - Validation: across all 2,736 ANDI theorems with an exact label, the exact label **equals** L1 + L2 + 1 in every
    case, with 0 counter-examples. That is 2,700 in the build (6: 222, 7: 213, 8: 251, 9: 243, 10: 232, 11: 229,
    12: 211, 13: 211, 14: 216, 15: 297, 16: 296, **17: 55, 18: 24**) plus 36 in the feasibility probes (17: 7,
    18: 2).
  - Only 88 exact cases exist at 17-18 (79 in the build, 9 in the probes), because exact search there costs
    200-300 s and 3 GB, and about 45% fail.
    So cert labels at 17-18 rest on this additivity evidence, which is strongest at ≤ 16.
- **Bias control:** within each (family, length) bin, records are drawn in random order from all labelled candidates
  (exact or certificate), not exact first. So the ANDI 17-18 bins are not biased toward search-cheap theorems. The
  certificate was also run on every ANDI candidate whose exact search failed, and on the certificate timeouts
  (`labels/andi_cert14_b.jsonl`, 92 of 92 exhausted at 14). Chain records have exact labels only. Unresolved chain candidates are dropped, as in run 2; see
  the caveats.
- **Driver:** labels come from `tools/label3.py`, a robust version of ab2's `label.py`. It adds a per-record bound, an
  emergency memory reserve, worker exceptions recorded instead of crashing the parent (ab2's `label.py` died on a
  worker `MemoryError` at bound 18), and `--maxtasks`. Most runs use `--intern`, which loads `tools/isearch.py`: a
  subclass of `minlen.Search` that only interns the memo-key frozensets. The DP is identical, and peak memory falls
  about 40% (1.44 GB to 0.85 GB on one L18 theorem).
  - Validation (`labels/validate_intern_b18.jsonl`): 120 probe theorems labelled with the plain driver were
    relabelled with `--intern`, and **120 of 120 agree** (L9-18). The fork's `minlen.py --selftest` passes.
  - The chain probe's pass A and first pass B used the plain driver (`label_driver: plain` in provenance); everything
    else used intern.

## Commands (from `ab3/data_build`, `python3` = pyenv 3.13.5 as in ab2; fork at `ab629c7`, read-only)
```
# candidates
python3 tools/build_cands.py --family andi --lmin 6 --lmax 18 --n 100000 --per_len 500 --max_prompt_tok 60 --seed 33401 --max_tries 3000000 --out raw/andi_c60.jsonl
python3 tools/build_cands.py --family andi --lmin 17 --lmax 18 --n 100000 --per_len 800 --max_prompt_tok 60 --seed 33402 --max_tries 3000000 --out raw/andi_c60_b.jsonl
python3 tools/build_cands.py --family chain --k 1 --lmin 15 --lmax 19 --n 2000 --seed 33201 --out raw/chain_probe_k1_unshuffled.jsonl   # probe; k2: --k 2 --lmin 16 --lmax 24 --seed 33202
#   (probe k1+k2 concatenated, shuffled with seed 5 -> raw/chain_probe.jsonl)
python3 tools/build_cands.py --family chain --k 1 --lmin 15 --lmax 19 --n 6000 --seed 33211 --out raw/chain_k1_b.jsonl
python3 tools/build_cands.py --family chain --k 2 --lmin 16 --lmax 24 --n 6000 --seed 33212 --out raw/chain_k2_b.jsonl
python3 tools/excl.py                                  # exclusion key/pkey sets -> excl.json
python3 tools/prep.py --in raw/andi_c60.jsonl --out raw/andi_lo.jsonl --lens 6,7,8,9,10,11,12,13,14 --per 300 --seed 33501
python3 tools/prep.py --in raw/andi_c60.jsonl --out raw/andi_mid.jsonl --lens 15,16 --per 500 --seed 33502
python3 tools/prep.py --in raw/andi_c60.jsonl --out raw/andi_hi.jsonl --lens 17,18 --per 500 --seed 33503
python3 tools/prep.py --in raw/andi_c60_b.jsonl --out raw/andi_hi2.jsonl --seed 33504     # -> raw/andi_hi2_cert.jsonl (bound 14; 18s and 17s interleaved 2:1)
python3 tools/prep.py --in raw/chain_b_unfiltered.jsonl --out raw/chain_b.jsonl --seed 33213   # chain_b_unfiltered = chain_k1_b + chain_k2_b
# ANDI labels
python3 tools/label3.py --in raw/andi_lo.jsonl  --out labels/andi_lo_b18.jsonl  --bound 18 --time 120 --procs 4 --mem_gb 1.2 --intern   # stopped at >= 211 per length
python3 tools/label3.py --in raw/andi_mid.jsonl --out labels/andi_mid_b18.jsonl --bound 18 --time 400 --procs 6 --mem_gb 2 --maxtasks 10 --intern
python3 tools/label3.py --in raw/andi_hi.jsonl  --out labels/andi_hi_b18.jsonl  --bound 18 --time 600 --procs 7 --mem_gb 3 --maxtasks 10 --intern
python3 tools/label3.py --in raw/andi_hi2_cert.jsonl --out labels/andi_cert14.jsonl --bound 14 --time 120 --procs 4 --mem_gb 1.5 --intern
python3 tools/label3.py --in raw/andi_cert14_b.jsonl --out labels/andi_cert14_b.jsonl --bound 14 --time 600 --procs 2 --mem_gb 3 --maxtasks 10 --intern   # every failed exact + cert timeouts
# chain labels: pass A (bound 12, 15 s) on all candidates; pass B (bound 18, 300 s) on everything pass A left unresolved
python3 tools/label3.py --in raw/chain_probe.jsonl --out labels/chain_probe_b12.jsonl --bound 12 --time 15 --procs 18 --mem_gb 1.0
python3 tools/label3.py --in raw/chain_probe_u.jsonl --out labels/chain_probe_b18.jsonl --bound 18 --time 300 --procs 30 --mem_gb 1.5 --maxtasks 1 --deadline 1800
python3 tools/label3.py --in raw/chain_probe_B_rest.jsonl --out labels/chain_probe_b18_rest.jsonl --bound 18 --time 300 --procs 8 --mem_gb 2 --maxtasks 10 --intern --deadline 9000
python3 tools/label3.py --in raw/chain_b.jsonl --out labels/chain_b_b12.jsonl --bound 12 --time 15 --procs 11 --mem_gb 0.8 --maxtasks 50 --intern
python3 tools/label3.py --in raw/chain_b_u.jsonl  --out labels/chain_b_b18.jsonl       --bound 18 --time 300 --procs 11 --mem_gb 2 --maxtasks 10 --intern --deadline 12600
#   parallel slices of the same pass-B queue (pass-B order seed 33214): chain_b_u2 = [2400:], chain_b_u3 = [3300:], chain_b_u4 = [2100:2400]
#   -> labels/chain_b_b18_rest.jsonl (7 procs), chain_b_b18_rest3.jsonl (4 then 12 procs, --resume), chain_b_b18_rest4.jsonl (3 procs), all 300 s / 2 GB / intern
# assembly and checks
python3 tools/assemble3.py --seed 33901 && python3 tools/verify3.py
```
Seeds:
- Candidates: 33401 and 33402 (andi); 33201, 33202 (probe) and 33211, 33212 (chain_b).
- Candidate filters: 33501-33504 and 33213. Probe shuffle: 5. Chain pass-B order: 33214.
- Driver validation sample: 33601. Assembly: 33901.

## Stage wall clock (UTC)
| stage | time | notes |
|---|---|---|
| feasibility probes | 20:05-21:00 | generator d7p5, ANDI and chain probes; FEASIBILITY.md |
| chain probe pass A / pass B | 20:19-20:26 / 20:26-20:56, rest 21:02-23:32 | 1.9 CPU-h / 30.7 CPU-h |
| ANDI 6-14 exact | 21:01-21:33 | 2.1 CPU-h |
| ANDI 15-16 exact | 21:01-23:42 | 15.9 CPU-h |
| ANDI 17-18 exact | 21:01-22:41 | 9.9 CPU-h; stopped, cert covers the bins |
| ANDI 17-18 certificate (bound 14) | 21:33-23:23, retries 23:42-01:04 | 7.3 + 2.7 CPU-h |
| chain_b pass A | 21:03-21:38 | 6.4 CPU-h |
| chain_b pass B (4 slices) | 21:38-01:08 | 73.0 CPU-h; 3,566 of 4,240 queued items labelled when stopped by deadline |
| driver validation | 23:23-23:52 | 120 of 120 agree |
| assembly + verification | 01:09-01:10 | |

## Yields
- **ANDI exact** (constructed length, then outcome):
  - 15: 297 of 303; 16: 296 of 321; 17: 55 of 61; 18: 24 of 54.
  - Failures at 17-18 are OOM at 3 GB, 600 s timeouts and worker errors.
  - 6-14: 2,028 of 2,031 (3 timeouts at 14; the batch was stopped once every length had at least 211).
- **ANDI certificate at 14:** 1,084 of 1,108 exhausted (24 timeouts at 120 s). A 600 s retry exhausted 92 of 92: the
  68 failed exact runs plus those 24 timeouts.
- **Chain**, 15,894 candidates (every pass A and pass B combined):

| stage | n | CPU-h | ≤12 | 13 | 14 | 15 | 16 | 17 | 18 | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pass A (b12) | 15,894 | 8.3 | 10,222 | | | | | | | 5,672 to pass B (5,358 exhausted, 314 timeouts) |
| pass B (b18) | 4,998 of 5,672 queued | 103.7 | 43 | 1,644 | 1,195 | **907** | **426** | **117** | **20** | 646 (569 timeouts, 63 OOM/error, 14 exhausted at 18); 674 never reached before the deadline |

Chain yield per CPU-h, pass A and pass B together: **L15 about 8, L16 about 4, L17 about 1, L18 about 0.2.**

## Exclusions and disjointness
`tools/excl.py` recomputes `key = canon_key(thm)` and `pkey` (the minimum over premise permutations) from `thm` for
every excluded file:
- cap-6 `train_cap6` and `heldout_cap6`;
- the 58 textbook-dev problems (only `thm` read);
- validation-36;
- **every run-2 file**: `targets_L6-11`, `dev_L8-11`, `frontier_L12-14`, `reserve` (426k), and `ab2/sealed`
  `sealed_L12-14` and `sealed_L15-16`;
- the run-1 sealed sets `dev_L7-9`, `test_L10`, `test_L11-14` (`thm` only);
- the fork's ladder pools.

That makes 606,547 keys and 604,910 pkeys. No assembled candidate hit an excluded key or pkey, which is expected
because composites are new classes. The three output files are pairwise pkey-disjoint; 7 duplicate pkeys were dropped
across candidates.

## Files (length × family × label kind)
| file | family | label | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sealed_L15-16 | andi | exact |  |  |  |  |  |  |  |  |  | 294 | 276 |  |  | 570 |
| sealed_L15-16 | andi | ub_cert14 |  |  |  |  |  |  |  |  |  | 6 | 24 |  |  | 30 |
| sealed_L15-16 | chaink1 | exact |  |  |  |  |  |  |  |  |  | 184 | 148 |  |  | 332 |
| sealed_L15-16 | chaink2 | exact |  |  |  |  |  |  |  |  |  | 116 | 152 |  |  | 268 |
| sealed_L17-18 | andi | exact |  |  |  |  |  |  |  |  |  |  |  | 43 | 9 | 52 |
| sealed_L17-18 | andi | ub_cert14 |  |  |  |  |  |  |  |  |  |  |  | 257 | 291 | 548 |
| sealed_L17-18 | chaink1 | exact |  |  |  |  |  |  |  |  |  |  |  | 43 | 7 | 50 |
| sealed_L17-18 | chaink2 | exact |  |  |  |  |  |  |  |  |  |  |  | 73 | 13 | 86 |
| matched_L6-14 | andi | exact | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 |  |  |  |  | 1,800 |
| matched_L6-14 | chaink1 | exact | 57 | 84 | 90 | 90 | 97 | 106 | 109 | 99 | 106 |  |  |  |  | 838 |
| matched_L6-14 | chaink2 | exact | 142 | 116 | 110 | 110 | 103 | 94 | 91 | 101 | 94 |  |  |  |  | 961 |

Per length, the totals are 15: 600, 16: 600, 17: 416, 18: 320. ANDI has 300 at every length from 15 to 18. Chain has
300 at 15, 300 at 16, 116 at 17 and 20 at 18.

Record fields (all files):
- `name, thm, key, pkey, prompt, n_lines (= L_true), L_true`
- `source` (`ab3_andi` / `ab3_chain`), `family`, `label_kind`, `schema` (null), `n_prem`
- `minlen_bound, minlen_proof`
- `constructed_lines, constructed_proof`
- `parts` (ab2 reserve `orig_name`s), `part_L`, `orig_name`, `label_file`, `label_driver`

The scorer reads `name`, `prompt` and `L_true`.

## Shape against run 2 (what else differs besides length)
Run 2's pools have 1.5-2.2 premises, 3.0-3.5 atoms and 43-58 prompt tokens. Their conclusions are 60-94% `>`.

| family | L | premises | atoms | prompt tok | concl > / & / v |
|---|---|---:|---:|---:|---|
| andi | 6-18 | 2.9-3.8 | 3.8-4.0 | 52-56 | 0 / 100 / 0% |
| chaink1 | 6 / 11 / 15 / 17 | 3.0 / 3.8 / 4.0 / 4.1 | 3.4-3.9 | 49 / 74 / 71 / 62 | 84-100% `>` at 6-16, 77% at 17 |
| chaink2 | 6 / 11 / 15 / 17 | 4.5 / 4.5 / 4.7 / 4.7 | 3.7-3.9 | 57 / 79 / 86 / 76 | 13% `>` and 54% `v` at 6 (collapsed composites), 83-91% `>` at 8-15, 74% at 17 |

Minimal-proof ORE use rises with length in every family: 0% at L6, about 15% at L11, about 50% at L14 and 54-92% at
15-18. Run 2's L12-14 are at 21-48%. Every construction therefore has **about twice run 2's premise count**. The
matched set shares this, so a drop that appears at 6-14 in the matched set but not in run 2's pools points to
premises and construction, not length.

## Model-side limits
The tokenizer covers N1..N64, and L18 uses N1..N18. Maxima over the final files:
- prompt: 148 tokens;
- minimal or constructed proof: 416 tokens;
- prompt plus proof: 551 tokens, against GPT `max_len` 1024.

`score_host.py` uses `max_new` 640 for the sealed sets. That covers every reference proof with at least 224 tokens to
spare. A non-minimal model proof can be longer, so **use `max_new` ≥ 768 for 17-18** if proof length should not be the
limit.

## Checks (`tools/verify3.py`, all pass)
- `key` and `pkey` recomputed; `n_lines == L_true`; names unique; provenance covers every record.
- The three files are pairwise pkey-disjoint.
- No key or pkey from any excluded file; `verify3` rebuilds these sets itself from the source files, not from
  `excl.json`.
- **10,492 `nd_verify` checks, 0 failures:**
  - every constructed proof (lines == `constructed_lines`), 5,535;
  - every minlen proof (lines == `L_true`), 4,957.
- 578 certificate records confirmed: bound 14, exhausted, no timeout or OOM, and `L_true == constructed_lines`.
- `L_true ≤ constructed_lines` for all records, and every part is an ab2 reserve generator record.

## Caveats
1. **Construction shift.** The 15-18 sets contain no run-2-style (generator or textbook) theorem. Compare with
   `matched_L6-14` of the same family, not with run 2's pools.
2. **ANDI is easy to decompose.** An ANDI theorem is two independent sub-proofs of L1 and L2 (at 15-18, each part is
   5-10 lines), joined by one ANDI line. Solving ANDI at 17-18 needs long, correctly indexed output (N-references up
   to 18) and composition of in-range pieces, not a single long chain of reasoning. Chain composites are less
   separable, but chain has only 116 at 17 and 20 at 18.
3. **Certificate labels.** 548 of 600 ANDI 17-18 labels are `ub_cert14`: `L_true` ≥ 15 is proved, and `L_true` ≤ L
   holds by a verified proof. Equality rests on additivity (2,736 of 2,736 exact cases, of which 88 are at 17-18).
   Filter `label_kind == 'exact'` for a strict subset (52 records, biased toward search-cheap theorems).
4. **Survivorship in chain.** Chain composites whose pass-B search timed out (569), ran out of memory or errored (63),
   or was exhausted at 18 (14) are dropped, and 674 queued composites were never reached,, as in run 2. These are the search-hardest and possibly longest ones, and
   the chain 17-18 bins are small.
5. **Prompt cap.** ANDI prompts are capped at 60 tokens at all lengths, so ANDI theorems are small-formula theorems.
6. **Restricted space.** As in run 2, "minimal" is minimal within the restricted formula space.
