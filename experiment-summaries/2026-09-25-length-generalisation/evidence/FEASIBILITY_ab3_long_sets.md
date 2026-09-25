---
written_on: 2026-09-24
---
# ab3 long test sets (L_true 15-18): feasibility study

Written 2026-09-24, 20:05-21:10 UTC, by the pool-builder session (agent:claude). CPU only, at most 36 of 48 cores, no
GPU. Working files are in `/scratch/cowifornia/nd-rl/ab3/data_build/` (`explore/`, `logs/`, `tools/`).

**Scope (after the PI's change at ~20:10 UTC).** Training stays at L_true 6-11, and run 3 reuses run 2's agent-visible
pools unchanged. This build makes **test-only, monitor-only** sets: `sealed/sealed_L15-16.jsonl` and
`sealed/sealed_L17-18.jsonl` (target 300 per length, minimum 150), plus `sealed/matched_L6-14.jsonl` (the same
constructions at 6-14, 200 per length per construction). No new theorem goes into any training pool.

**Verdict: FEASIBLE. The build proceeds.** The minimum (150 per length at 15, 16, 17 and 18) is projected within about
3-4 h wall. The limit is memory (about 42 GB for my workers), not CPU. Two caveats come first:

1. **No run-2-style theorem exists at 15-18.** Every long theorem comes from a *new construction*. Neither
   construction occurs in run 2's 6-11 pool (numbers below). That is why the matched 6-14 set is required, not
   optional.
2. The two constructions have different reach. **ANDI** (disjoint conjunction) covers 6-18. **Chain** (cut
   composition, closer to the generator's style) covers 6-17, and is thin at 17 and absent at 18. So the per-length
   construction mix at 15-18 cannot be identical. Every record carries `family`, so each family has its own full
   length curve (matched 6-14 plus sealed 15-18).

## Label (unchanged from run 2)
`L_true` is the minimal length found by the fork's `minlen` restricted iterative-deepening search. The formula space
is the subformula closure plus negations and double negations. The proof is checked by `nd_verify`, and the search
runs at bound 18. Unresolved theorems (timeout, OOM or none found) are dropped. The constructed proof of each theorem
is kept as a second, independently verified upper bound.
- *Memory-lean driver:* `tools/isearch.py` subclasses `minlen.Search` and only interns the citable-set frozensets
  used as memo keys. The DP, the rule order and the pruning are identical. On one L18 theorem, peak RSS fell from
  1.44 GB to 0.85 GB with the same label (18). The build re-labels a probe sample with both drivers and reports
  agreement.
- *Fallback label, only if exact minlen fails on an ANDI 17-18 candidate:* `ub_cert14`. It means the exhaustive
  restricted search found no proof of length <= 14, and a verified constructed proof of length L1 + L2 + 1 exists.
  Such records are flagged `label_kind` and counted separately. The certificate is cheap: 20 of 20 L18 candidates
  were exhausted at bound 14 in 2.5-35 s each.

## Strategies and measured yields

### 1. Run 2's generator with deeper settings: rejected
- **Run 2 (reference):** 424k pass-1 labels plus 7.7k pass-2 labels, about 62 CPU-h, gave **L15: 9 and L16: 0**
  (about 0.15 L15 per CPU-h).
- **ab3 probe:**
  - Generation: `make_coverage_sets.py gen --long --min 14 --max 40 --gen_max_depth 7 --gen_max_prem 5`, 12 workers,
    360k tries, seed 33001, 5,106 classes.
  - Pass 1 (b10, 10 s, 0.30 CPU-h): 158 exhausted (3.1%, against 1.7% in run 2) and 29 timeouts.
  - Pass 2 (b16, 120 s, 1.47 CPU-h) on those 187 gave L10: 10, 11: 101, 12: 26, 13: 17, **14: 8, 15+: 0**, with 25
    unresolved.
  - Deeper settings raise the L14 rate about 8x over run 2. L15+ is still not observed.
  - Projection for 150 at L15: at least 250 CPU-h (at least 7 h on 36 cores). L16-18 is out of reach.

### 2a. Compositional: ANDI (conjunction of two atom-disjoint sub-theorems)
- **Construction:** two ab2-reserve generator theorems, each with at most 2 atoms and satisfiable premises, are
  renamed onto disjoint atom pairs (P,Q / R,S in random assignment). The result is Γ1, Γ2 ⊢ φ1 & φ2, with premises
  interleaved at random and conjunct order random. The constructed proof concatenates the parts' minimal proofs plus
  one ANDI line, is nd_verify-checked, and has length L1 + L2 + 1.
- **Additivity (checked, not assumed):** exact minlen equals L1 + L2 + 1 in **36 of 36** resolved cases (L12: 6,
  13: 6, 14: 6, 15: 6, 16: 3, 17: 7, 18: 2), with no counter-example. Two early candidates had an unsatisfiable
  premise (for example `~((~S) > (~S))`), which would permit explosion shortcuts. Parts are now filtered by truth
  table.
- **Exact-label cost:**
  - L15: 28-84 s at ≤1.2 GB.
  - L16: 81-107 s; 4 of 6 OOM at 1.2 GB, 1 OOM at 5 GB after 528 s.
  - L17-18 with unrestricted prompts: OOM at 5 GB after 600 s or more, so not affordable.
  - With prompts **≤ 60 tokens**: L17 took 30-189 s (8 of 9 resolved at ≤3 GB). L18 took 111-265 s (2 of 5 resolved
    at 3 GB; 3 OOM or error).
  - The build therefore uses the 60-token cap at every length, 6-18, so the matched set shares it.
- **Style:** a separable conjunction. Only **0.3%** of run-2 `targets_L6-11` are separable conjunctions, and only 5.3%
  have an `&` conclusion at all (79% are `>`). This is a new style.

### 2b. Compositional: chain (cut) composition
- **Construction:** take a consumer Γ2, π ⊢ ψ from the reserve, where π is a compound premise that its minimal proof
  cites. Take a producer Γ1 ⊢ φ with φ equal to π after an injective renaming. The result is Γ1, Γ2 ⊢ ψ. With k = 2,
  a second producer replaces another compound premise. The constructed proof has length L1 + L2 − 1 and is **only an
  upper bound**. At constructed 12-15, exact equalled constructed in about 30% of cases. Shortcuts come from
  redundant premises and detours. The label is always exact minlen.
- **Yield probe:**
  - 4,000 candidates: k1 at constructed 15-19 (seed 33201) and k2 at 16-24 (seed 33202).
  - Pass A (b12, 15 s, 1.9 CPU-h, 1.7 s per item) left 1,432 (36%) with L ≥ 13.
  - Pass B (b18, 300 s, 1.5 GB) ran 30 min on 30 procs (15 CPU-h) and finished 956 of 1,432: **13: 355, 14: 237,
    15: 153, 16: 63, 17: 16, 18: 0**. 67 OOM and 64 worker errors are retryable.
  - Per CPU-h, including pass A: **L15 about 9, L16 about 4, L17 about 1, L18 about 0**.
- **Style:** closer to the generator. The premises are random gen formulas and the conclusion comes from a gen
  consumer. But composites of two gen theorems never occur in run 2's pools, so this is also a new construction.
- Filters tried: requiring the producer to end on an elimination (the "detour-free" variant) did not raise
  preservation. Semantic tightness (every premise necessary) raised the L15+ share about 2x, but only 0.8% of
  candidates pass it, so it is not used.

### 3. Other
- The textbook schema family stops at 14 in run 2. Schema proofs treat metavariables opaquely, so bigger instances
  do not lengthen them. Not pursued.
- A curried ANDI variant (discharge one premise: Γ ⊢ p > (φ1 & φ2), L1 + L2 + 2) is implemented in
  `tools/build_cands.py`. It is not labelled here, for lack of time.

## Model-side limits (frozen/lib)
- The tokenizer's line tokens run N1..N64. An L18 proof uses N1..N18.
- Probe maxima: prompt 147 tokens, minimal proof 446 tokens, prompt plus proof ≤ 580 tokens. GPT `max_len` is 1024,
  so this fits.
- `score_host.py` uses `max_new` 640 for the sealed L12-16 sets. A minimal L18 proof leaves about 170-250 tokens of
  slack for longer, non-minimal proofs. **Recommend `max_new` 768 for 17-18.**

## Projection for the build (36 procs, about 42 GB RLIMIT total)
| stage | work | memory | est. wall |
|---|---|---|---|
| ANDI 6-14 exact (matched, 200 per length) | ≈ 2,700 cand., ≤ 30 s each | 1.5 GB | < 20 min |
| ANDI 15-16 exact | ≈ 800 cand., 40-100 s | 1.5-2 GB | ≈ 1 h |
| ANDI 17-18 exact, then `ub_cert14` fallback | ≈ 1,000 cand., 60-250 s | 3 GB | 2.5-3.5 h |
| chain: finish probe pass B + ~12k new candidates (pass A, then B) | as measured | 1-2 GB | 3-4 h, runs alongside |

Expected at the 5 h cap:
- **ANDI:** at least 300 at 15-17 and 150-300 at 18.
- **Chain:** about 300 at 15, about 150-250 at 16, about 40-60 at 17, and about 0 at 18.
- **Minimum (150 per length at 15-18):** met by ANDI alone, which also exists at 6-14 in the matched set.

## Outcome (added 2026-09-25 01:12 UTC)
The build finished within the window. Final counts per length, over all families, are 15: 600, 16: 600, 17: 416,
18: 320. ANDI has 300 at each of 15-18, and chain has 300 / 300 / 116 / 20. Chain 17-18 fell short of the projection;
ANDI 17-18 labels are mostly the `ub_cert14` fallback. Details, checks and caveats are in
`/scratch/cowifornia/nd-rl/ab3/data/POOLS.md`.
