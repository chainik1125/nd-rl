# pretrain-data log

## 2026-10-03 — workstream opened

Charles: "construct a more robust pretrain dataset for the models. Investigate what is currently
done. Generate some ideas … put it in a plan. One example might be making easier versions of the
textbook problem or other dataset problems and seeing if this allows RL models to actually do
better."

**Ran.** No training. Survey of the generator (`gen.py`), the pools (cap-6 Stage-1 set, K12, RL
targets; statistics computed last night for the doc), Dan's data experiments (two Explore agents
over `fork_dan/*.md` and Robbie's fork; summarised in plan § 2.1) and the textbook schemata pool.
Prototype `code/pretrain_data/subproblems.py`: extracts the theorem proved at every line of a
verified proof (open assumptions promoted to premises, cited lines as the proof), verifies each
with the pinned verifier, deduplicates by curry-aware class.

**Learned.**
- Every textbook failure mode found last week maps to a pool gap (plan § 1 table): premises ≤ 3,
  ∧E in 4 % of proofs, depth ≤ 2, length ≤ 12, manufactured splits, conditional goals 8 %.
- Dan's noise floor: at cap 6 / 3.3M, ≈ 99 % of between-arm variance is the training run; only his
  cap-8 findings survive. His pattern dial: a pattern present at f = 10⁻⁴ ignites RL as well as at
  10⁻¹. His stage1-dynamics: steps, not data. None of his data arms was at cap 12, in the state
  format or on the 9.56M recipe.
- Sub-problem yield: 262 seed proofs → 1,359 distinct classes (5.2 per seed), lengths 2–20,
  323 with ≥ 4 premises; 25 twins of the sealed 72 and 26 of other evaluation problems (to be
  removed by the audit). First version mis-shifted box depths for lines inside closed sibling
  boxes (55 verifier rejects); fixed, now 0 rejects.
- Dan's fork forbids search-found proofs as training data and caps supervised data at 6 lines;
  K12 already breaks the cap.
- Robbie's loop never varied the data (order only, and every order curriculum lost). The group
  meeting of 2026-09-20 called the dataset "lever one" and asked about minimal-length proofs vs a
  curated subset. Proposal 10 called schema instances in pretraining "decorative". The fork's
  write-up: depth ≥ 4, →E chains ≥ 4 and nested ∨E are unreachable by the generator.

**Decided (Charles, 2026-10-03).** This is its own workstream, `pretrain-data`. The take-home's data
rules (no prover-found training proofs, cap 6) do not bind it; proposals still name which rule an
arm is outside so results can be set beside Dan's cells.

**Next.** Charles's answers to `questions.md` (split, budget, permutations). Then the four pool builders
and G0 audits, phase-A pilots, phase B at 3 seeds. Nothing committed yet; the plan, questions,
log and prototype are on disk on `charles/main`.

**Open.** Whether Roy's set is a fair transfer target for forall-x-seeded data; whether to run
Dan's ladder ourselves; what the state environment's ladder costs for the 9.56M model.

## 2026-10-03 (later) — ideas pass

Charles: focus for now is designing many kinds of pretrain datasets, not running them. Wrote
`ideas.md`: 30-odd directions in seven families (problem sources, proof style, difficulty
distribution, multiplicity/invariance, scale controls, format-changing ideas kept out of scope,
analysis-first). Each names the type cell it should move, the build cost and the risk. The plan's
§ 3 pools are unchanged until Charles picks from the list.

## 2026-10-06 — why the hard textbook problems are hard

Charles: inspect the hardest problems from the textbook workstream, see why their reference proofs
are so long, and whether the rules that make them hard could be put into smaller pretrain problems.

**Ran.** `code/pretrain_data/hard_problems.py` (analysis only, ~20 s): 155 hard problems from
`per_problem.csv` (never solved by either RL seed, or reference ≥ 14 lines), 136 distinct classes
(hand14 repeats batch3/v1), 77 scraped. Per proof: line attribution (case split / classical /
conditional / ∧∨ shuffle / →E chain), box tree, ↔ count, 2–6-line sub-theorems with premises
pruned (`subproblems.subproblem(prune_prem=True)`), a rule-skeleton abstraction (subformulas whose
structure the sub-proof never uses become atoms), coverage against K12 by class and by skeleton,
and rule bigrams (citing rule ← cited rule) in the hard proofs vs K12. Report:
`docs/pretrain-data/hard_problems.md`; data `experiments/textbook-problem/eval_dan_best_state/analysis/hard_problems.json`.

**Learned.**
- Length comes from ↔ (42 of 77 scraped hard contain (a→b)∧(b→a); Pelletier 12 = 102 lines, 18
  →I boxes), box nesting (only 21 % of lines at depth 0; 34 of 77 need depth ≥ 3, K12 stops at 2),
  reductio (31 of 77) and per-case re-proving under nested ∨E.
- Length is not the only failure: 25 of 67 never-solved scraped problems have references ≤ 13
  lines; `P, Q, P → Q, Q → R ⊢ R` (4 premises) is never solved.
- 85 % of the hard proofs' 2–6-line pieces exist in K12 in skeleton form; 36 of 67 never-solved are
  made only of known pieces. No hard proof's whole skeleton is in K12. The gap is the joins: →E ← ∧E
  ≈ 90× rarer in K12, ¬E ← ¬I 128×, nested ∧E 35×, ∨I ← ¬I 34×, ∧I ← →I 18×, ∨E ← ∨E 12×.
- Added idea A11 (join problems) to `ideas.md`; the structural gaps (≥ 4 premises, depth ≥ 3, ↔
  shape) still need their own pools.

**Next.** Charles to pick from `ideas.md`; then build A11/A1 pools with the audit, and run G1
coverage per never-solved type. Nothing committed.

## 2026-10-06 (later) — eight candidate pools built and compared at the data level

Charles: "Try constructing some new pretrains to compare." Also answered: K12 has 349 rows with a
(a→b)∧(b→a) shape, 63 non-degenerate, none proved by ∧E-then-→E; the RL targets have none. Both
pretrains train on one task only: write the proof (loss on proof / action tokens; the state format
is one (state, action) pair per reference line).

**Ran** (CPU only, `code/pretrain_data/`): `minimal_k12.py` (K12 re-proved minimal, 85 s, 4 workers,
detached), `joins.py` (71 templates × 300 fills, 31 min detached), `subproblems.py --prune_prem --pool`
(seed split v0 + Lean), `mix.py` (eight 155,000-record pools + comparison), `pool_common.py` (record
format, audit that keeps seed-set twins tagged, stats, permutations). Outputs in `data/pretrain_data/`
(gitignored, stats JSONs beside them); report `docs/pretrain-data/pools.md`. Read 20+ raw records
across the pools and before/after pairs of the minimal rewrite.

**Learned.**
- 89 % of K12's long-mode rows (7–12 lines) are padded: 85,244 of 155,000 proofs got shorter under
  the minimal-proof search, 374,537 lines removed, ∨E share 14 % → 3 %. The cap-12 result may be a
  length effect; a minimal pool removes nearly everything above 9 lines.
- Joins: 12,295 audited problems; 23 % with ≥ 4 premises, 7 % depth ≥ 3, 15 % with a real ↔.
  10 % slice raises →E←∧E from 0.55 to 5.3 per 1,000 citations (9.5 on the minimal base). Two joins
  the hard proofs need (¬E←¬I, ∨I←¬I) are still absent: templates bypassed or over bound 14.
- Sub-problems with pruned premises collapse to 383 classes but cover 32 of the 89 piece skeletons
  K12 lacks (joins cover 8: shapes, not the textbooks' exact pieces).
- Audit removed 8 + 14 sealed twins (counted only) and 13 + 9 test-split twins.

**Next.** Charles: which pools to pilot (questions.md 4) and the budget (2). Reductio-shaped join
templates at bound 16; mirrored-conjunct variants; then proposal.md for the phase-A pilot. Nothing
committed.

## 2026-10-06 (evening) — literature-driven directions

Charles: "Brainstorm more ways to take the pretrain. Use inspiration from leading papers and labs."
Added `ideas.md` § H (H1–H20): AlphaGeometry traceback / HER, necessary-steps-only data, mistake-recovery
data, search traces, PACT co-training with reordered tokens, statement curricula and a conjecturer,
AlphaProof-style variants, data-constrained scaling (Robbie's 9.56M pretrain ran 20.7 epochs on the pool),
DoReMi/Skill-it/Rho-1 reweighting, DSIR, rephrasing, ProofWriter iterative depth mixes, SimpleLogic
shortcuts, length-generalisation format tests, lemma libraries, base-model pass@k as the pool screen,
easy-to-hard and reverse curricula, EntiGraph seed graph, FIM/reverse training, easy-example pruning.
Also answered provenance: every Robbie pretrain used the take-home's `data/train.jsonl.gz` (same git blob
at both pinned fork commits), built by `gen.py --n 160000 --per_len 32000 --seed 1` + `make_splits.py`;
its statistics now sit as the first row of `pools.md`.

## 2026-10-06 (night) — size ladder built

Charles: "Can you try some different sizes of pretrain sets." Built `code/pretrain_data/size_ladder.py`
(ideas H8, plan E7): nested pools of K12's content at 15k, 50k, 155k (K12), 500k and 1.5M records.
Below K12: stratified subsets. Above: K12 plus fresh `gen.py` output at the take-home knobs (cap-6
forward mode seed 21, ≈900k raw, stopped when length 2 saturated at 98k distinct classes; long mode
seeds 22–23, 1.4M raw), box depth capped at 2 to match K12 (the raw generator puts depth 3 in ~37 %
of long-mode proofs; 486k such records set aside as `pools/size/gen_depth3.jsonl.gz` for the depth
ladder, C2), deduplicated against K12 by renaming class, audited (10 sealed twins, 26 evaluation
twins removed). Every record verified by the pinned verifier. Outputs `data/pretrain_data/pools/size/`
(gitignored), table appended to `pools.md`.

**Learned.**
- Epochs at Robbie's 1,200 s budget (9.56M model, 3.18M records seen): 212 at 15k, 64 at 50k, 20.5 at
  K12, 6.4 at 500k, 2.1 at 1.5M. The 38M model: 81 / 24 / 7.9 / 2.4 / 0.8.
- The generator's supply is finite at the take-home knobs: 2-line classes run out at ≈105k (31k short
  of a flat 1.5M), and depth-capped long mode is slower to yield 7–8-line classes (deficits 24–25k).
  The 1.5M pool is therefore flat only to within ±20 % per length; the 500k pool is exactly flat.
- Even depth-capped, fresh generator output is not identical in content to K12: ∨E share 0.12–0.13 vs
  0.138, ¬E←→E 1.1–1.2 vs 1.8 per 1,000. K12's halves came from Dan's assembly (`train_depth3_f0_a1`
  and his long-mode run), not from a plain `gen.py` call, so the superset arms carry a small content
  shift as well as the size change. Report both when E7 runs.

**Next.** Charles: pools to pilot and budget (questions 2, 4). Nothing committed.

## 2026-10-06 (late) — recent-literature scan

Charles: "Do another scan of most recent literature for promising ideas. Focus on new papers not currently
sitting in the repo." Four search agents (provers' synthetic data; pretraining data for reasoning; base vs RL;
length/depth generalisation), every paper verified on its arXiv page, deduplicated against § H. Added
`ideas.md` § I: 25 entries (I1–I25) plus a set-aside list and five plan changes: a fraction dial E8 (Dan's 10⁻⁴
vs Zhang 2025's 1 % threshold and Gu 2025's size-dependent phase transition), two CPU screens before any ladder
(textbook-shaped held-out loss, base pass@k), a K12-plus-depth-3-tail pool, three per-ladder measurements
(base log-prob of RL-found proofs, pass@N vs base, proof diversity on unsolved problems), and generator audit
items (dependency-graph balance, breadth per size). Nothing committed.

## 2026-10-06 (late night) — thirteen pools from the literature scan

Charles: "Create new pretrains around the original size (can deviate a bit) based on these ideas." Built
`code/pretrain_data/scan_pools.py` (sub-commands `distract`, `pairs`, `aux`, `subst`, `assemble`); thirteen pools
of exactly 155,000 records under `data/pretrain_data/pools/scan/` (gitignored), compared in `pools.md` "Scan pools";
plan.md rows E8–E13; questions.md 4 updated. Every record verified by the pinned verifier when built and 300 per
pool re-verified at the end (0 failures); every new theorem audited by renaming class against the 412 evaluation
problems and the sealed 72 (counts only).

- **E8 fraction dial** `k12_dial_1e4/1e3/1e2/1e1`: nested slices of 16 / 155 / 1,550 / 15,500 records of
  ↔-elimination problems (the join pool's bicond_elim family, 1,114 sources + verified atom permutations), a
  pattern K12 shows 0 times per 1,000 records; the pools show 0.1 / 1.0 / 9.9 / 99.1.
- **E9 depth tail** `k12_tail10/30`: depth-3+ records from the size ladder's side pool (487,184 available), flat
  over 7–12 lines. The generator's long mode stops at 12 lines, so the tail is deeper, not longer, than K12.
- **E10 distractors** `k12_distract10/30`, `k12_minimal_distract10`: 58,000 minimal-status K12 rows tried, 1–2
  random premises inserted at random positions (satisfiable, not a line of either proof); 8,902 (15 %) rejected
  because `minlen` found a shorter proof of the distracted problem, 3 unknown (timeout), 49,098 kept; premises per
  record now 1–5 (4+: 36 %). Finding: K12 cites every premise, but `k12_minimal` already has an uncited premise in
  29 % of rows (the minimal proof drops the padded ∨E's premise), so E3 and E10 confound; noted in pools.md.
- **E11 substitution** `k12_subst30/100`: 146,618 of 153,976 non-head rows changed (atoms → depth-1 formulas at
  rate 0.7); mean tokens 151 → 222 (max 893 of the model's 1,024), line counts unchanged.
- **E12 long easy auxiliary** `k12_aux10`: 14,547 forward-chain proofs (∧I/∧E/→E, no boxes, all premises cited,
  13–24 lines, mostly 15–19; 3–6 premises, 79 % with 6) in 921 s; the generator rarely yields pruned proofs over
  20 lines, so the 20–30 range is thin (882 records). Filled to 15,500 with 953 permutations. →E←→E rate 15.6 vs
  K12's 2.6 per 1,000.
- **E13 premise pairs** `k12_pairs1`: 9,000 rows tried, 2,130 siblings found (24 %; 4,697 no provable
  replacement within 6 tries, 2,173 rows with no critical premise), 2,112 after audit (2 v1 twins removed, 15
  already in K12); 1,761 need a longer proof than the original, the rest an equally long one with different rules.
- Infrastructure: every pool, including the earlier eight (rebuilt), keeps K12's first 1,024 rows at its head, the
  pinned trainer's held-out val slice, so val_loss is comparable across pools; `mix.py` now writes the cap-6
  reference row itself and preserves the hand-written and script-written sections of pools.md (an earlier rerun
  had dropped them; restored from the session transcript). `size_ladder.py --doc_only` regenerates its section.
- Not buildable as data: the scan's "edge slice last" ordering (I1): the pinned trainer reshuffles every epoch.
- Lost and recovered: the scratchpad copy of K12 was wiped at 20:38; re-fetched `train_k12.jsonl.gz` from Dan's
  bucket (`cap-horizon/data/kh/`, 13,515,773 bytes), prompts and unchanged proofs identical to `k12_minimal`'s.

**Next.** Charles: which pools go to the pilot (questions.md 4 now lists 25 candidates and a default: CPU screens on
all, ladders on the E8 dial + K12 + the best two screened). Nothing trained, nothing committed.

## 2026-10-07 — 1.5M pool removed

Charles: "Remove the super big pool." Deleted `pools/size/k12_1500k.jsonl.gz` (121 MB), dropped 1.5M from `size_ladder.py`'s
sizes and from `pools/size/stats.json`, regenerated the pools.md section (`--doc_only`), E7 now reads 15k / 50k / K12 / 500k.
The raw generator output it was built from (`data/pretrain_data/gen/`) is untouched, so it can be rebuilt if wanted.

## 2026-10-07 — spurious-reward control pools; screening ideas

Charles: "Can you also create a pool of spurious rewards (0 if right 1 if wrong)." Read as a pretraining pool
(questions.md 5 records the other reading, an inverted ladder reward, as a possible E15). `scan_pools.py wrong`:
every non-head K12 row got one local error (formula 82,754; rule label 36,395; citation 34,827), kept only if the
pinned verifier rejects it; 153,976 of 153,976 rows corrupted in 28 s. Pools `k12_wrong10` and `k12_wrong100`
(verifier accepts 90 % and 0.66 % of records, the 0.66 % being the untouched val head); 300 wrong rows per pool
re-checked, all rejected; 300 untouched rows, all accepted. Plan row E14; pools.md "Scan pools" regenerated with
15 pools. The pool is a control for "does Stage-1 teach correctness or surface structure" and is labelled as such
everywhere; a checkpoint trained on it must never be used as a prover.

Charles: "Are there ways to quickly verify whether or not a pool would have good signal?" ideas.md § J: 18 screens
in five cost tiers (data-only, frozen K12 checkpoint, continued pretraining from the K12 checkpoint, round-1 yield,
calibration), ranked; the recommended path screens all 27 pools for about $4 before any ladder. Nothing trained,
nothing committed.

## 2026-10-07 — data-only screens built and run

Charles: "Go and do it" (the data-only screens, ideas J1–J3). `code/pretrain_data/screen_data.py`: validation set of
224 seed-split problems (v0, lean_v2, derived-from-those) with reference proofs, 58 unsolved by every evaluated model;
the sealed 72 and the test split untouched. Three screens per pool in ~10 s: a count-model next-rule predictor
(perplexity, accuracy, median loss, surprise share), nearest (rule, depth)-sequence edit distance, and need-weighted
cell coverage (with and without the premise count). All 27 pools; table and reading in pools.md "Data screens".
Findings: the perplexity-style metrics fail the negative control (`k12_wrong100` beats K12 on all three: corrupted
labels flatten the distribution) and are withdrawn; top-1 accuracy passes and separates (0.45 minimal_joins10 vs
0.40 K12); the distance screen passes and ranks spice1 / sub / pairs1 / joins top, but sub and spice are partly
circular (cut from the seed problems); coverage needs the no-premise variant (distractors otherwise score by premise
count alone) and puts the depth tails first (0.97 vs K12 0.82) with a ≈0.05 noise floor from the control. Seven
pools (dial × 4, subst × 2, aux, distract) are identical to K12 on data screens by construction; their hypotheses need
the checkpoint screens (J6–J8) next. Nothing trained, nothing committed. `mix.py` now also preserves the
"## Data screens" sections of pools.md.

## 2026-10-07 (evening) — seven more pools from unbuilt ideas

Charles: "Try to create some new sample pools experimenting with new ideas." Picked the ideas.md entries that had no
builder yet and fit the fixed format: closure sub-theorems (A2 / H1), lemma-then-use composition (A4 / H15), recovery
data (B3 / H3), second proofs per theorem (B2 / H11), duals (A5), seed-split-shape-matched resampling (C3 / H10) and
rule-balanced resampling (C5 / H13). `code/pretrain_data/new_pools.py` (sub-commands per source + `assemble`); pools under
`data/pretrain_data/pools/new/` (gitignored); plan rows E16–E22; pools.md "New pools" table and reading; data screens
run on all seven (`screen_data.py` now also scans `pools/new/`; `mix.py` preserves the new sections). Every record
verified by the pinned verifier, every new theorem audited against the 412 and the sealed 72 (closure: 7 sealed twins
removed, counted only). 21 raw records read plus before/after pairs of the closure re-proving.

**Learned.**
- Splicing two generator proofs lemma-then-use is almost never minimal: 3,358 of 14,010 splices survive `minlen`; the
  lemma is cheaper to derive in place. Hypothetical-syllogism shapes will not come from composing generator problems.
- 27.5 % of the seed split's (goal, premises, length, depth) mass lies in cells the generator population never produces
  (every 13+-line and every 4+-premise cell); resampling the generator (E21, C3 / C5 / C6) is capped there.
- Sub-theorems cut from the depth-3 side pool inherit the lazy-premise split (98 of 150 ≥ 5-line samples shortenable);
  the builder now re-proves them (17,245 of 32,439 got shorter). Closure yields short pieces (94 % at 3–6 lines) and
  almost no 4-premise problems, because K12's proofs have ≤ 3 premises and depth ≤ 2.
- Second proofs exist for 6.5 % of minimal K12 theorems within +2 lines; the commonest alternative is ∨E moved outside
  an →I box. Rule balancing by rarest rule triples ∧E presence but drags in deep, long, padded generator proofs
  (53 % depth ≥ 3).
- Data screens: `k12_shape` / `k12_balance` lead on coverage and distance (like the depth tails); `k12_compose10`
  fills premise-counted cells only; the other four sit at K12 on data screens, as their hypotheses predict.

**Next.** Charles: questions.md 6 (which of E16–E22, if any, join the pilot list). A minimal-proof variant of
`k12_shape`, and reductio-shaped join templates, remain the open builders. Nothing trained, nothing committed.

## 2026-10-07 — status doc

Charles asked for a summary of the workstream so far; written as a Claude Doc, "Pretrain-data
workstream: status (2026-10-07)", https://claude.ai/code/artifact/62fc567d-c45b-4e6a-b414-2664f91958b1
(private): the question, what was learned before building, the 34 pools by family, the data
screens, literature, open decisions. Nothing else changed.

## 2026-10-07 (late) — pool-to-pool similarity

Charles asked how to tell whether the seven new pools are similar to the earlier 27. The stats table and the data
screens describe pools alone or against the seed split, so a direct comparison was missing. Added
`code/pretrain_data/pool_similarity.py`: for every pair of the 34 pools, Jensen–Shannon distance (base 2) between
(rule, depth)-trigram, rule-bigram and shape-cell ((goal, premises, length bin, depth)) distributions, plus share of
records identical to K12, share of theorem classes present in K12 (curry-normal `canon_key`), class-set Jaccard, and
average-linkage clusters. Whole (rule, depth) sequences were tried first and rejected: two halves of K12 differ at
JS 0.36 on them (the tail is undersampled at any size), trigrams at 0.05. Results in
`data/pretrain_data/screens/pool_similarity.json`, section "## Pool-to-pool similarity" (+ reading) in `pools.md`;
`mix.py` preserves the section. ~5 min, no seeds (deterministic measures).

**Learned.** The 34 pools are three families: a K12 core of 17 (all ≤ 0.06 from K12, floor 0.03), the minimal family
(0.48 from K12), and singletons that move one marginal. Of the new seven, `dual1` and `closure10` are inside the core
(their novelty is the theorem set only: closure 10 % new classes), `multi10` / `compose10` / `recover20` sit at
0.09–0.14, the distance of a 10 % textbook slice, and `shape` / `balance` are the two farthest pools from K12 after
cap-6 (0.41 / 0.52), nearest `tail30` (0.26) but well above the floor. The take-home cap-6 set is closer to
`k12_minimal` (0.36) than to K12 (0.53). The views disagree by design: `distract*`/`subst*` are invisible to the
skeleton views, `wrong*` invisible to the shape view.

**Next.** Unchanged: questions.md 6; a `shape` vs `tail30` training pair is the one comparison data alone cannot
settle. Nothing trained, nothing committed.

## 2026-10-07 (afternoon) — frozen-checkpoint screens

Charles: "Okay go do it" (the frozen-checkpoint screens, ideas J6–J8). The pinned fork clone at 51604b3 had lost
its core files in the same scratchpad wipe as K12; re-cloned from `dan-pandori/nd-takehome` (large repo, ~10 min).
`code/pretrain_data/screen_ckpt.py`: loads Robbie's `scale-d384-p1200_s0` pretrain checkpoint through
`textbook_problem/eval_ckpt.py`, and for each pool's new slice computes mean proof-token loss (3,000 records),
pass@1 / pass@32 at temperature 1.0 (160 prompts × 32, pinned verifier), lean_seq encodability, and the K12 type
cells the slice displaces. 24 pools in 1 h 40 min on the laptop CPU; table and reading in pools.md "Checkpoint
screens". Findings: the loss ratio alone mislabels pools (the dial's short ↔-elimination proofs have lower loss than
K12 yet pass@32 0.26), so regimes are set on pass@32; `k12_aux10` is the only "far" slice (pass@32 0.00, loss ×4.6);
joins / dial / pairs / spice / sub / distract / tail are all "edge" (0.14–0.66); cap6, the 500k generator rows, the
substitution slices and k12_minimal are "redundant" (≥ 0.81). Distractors cut the base's pass@32 on K12 problems
from 0.89 to 0.55. K12's minimal proofs have loss 0.109 vs 0.171 for the padded originals the model was trained on.
11.6 % of wrong-proof rows cannot be lean_seq-encoded (E14 needs a filter step; the 100 % arm is really 88 %).
No slice at the 10 % dose removes a cell with more than 6 K12 records. `screen_ckpt.py --doc_only` regenerates the
section; `mix.py` preserves it. Nothing trained, nothing committed.

## 2026-10-07 (evening) — post-analysis of the screens

Charles: "Do an in-depth post analysis on the results." `docs/pretrain-data/screens_analysis.md`: the two screen tiers
are nearly uncorrelated (pass@32 vs data-screen accuracy ρ −0.38, vs distance −0.01, vs coverage −0.02), so they
form a relevance × novelty placement; joins are the only pools in the target quadrant (relevant, near-far); tails
relevant but mid-edge; distractors novel but invisible to data screens; subst/500k/cap6 redundant; aux far and
irrelevant. Matched-row follow-up: on the same 3,000 K12 rows the base's loss is 0.254/token on the padded proof vs
0.109 on the minimal one (40.4 vs 9.3 nats total; minimal lower on 77 %). Gaps named: the screen's checkpoint is
Robbie's lean_seq model, not the plan's lean_staten ladder base (re-run on Dan's checkpoint first); screens judge the
base on the slice's problems, not textbook targets; loss confounded with length (pattern-local loss proposed); no pool
covers 13+-line relevant proofs (build one); 7 parallel-build pools unscreened; T 1.0 vs the ladder's 0.8. Per-family
join pass@32 follow-up running (`screens/followups.py`). Results also added to the status doc.
Per-family follow-up landed: base pass@32 on the join slice by family: four_prem 0.00, nested_ore 0.00, bicond_intro
0.04, depth3 0.05, negation 0.18, bicond_elim (E8's pattern) 0.23, nested_ande 0.30, impe_chain 0.31, ande_impe 0.45
(100 prompts × 32 each). Recommendation in screens_analysis.md § 6: a second, surfacing dial on four-premise problems.

## 2026-10-07 (night) — Dan-checkpoint screens, paused mid-run

Charles: "Now test these against Dan's" (the top-10 list), then "finish the current pools, then stop them and track
which ones need to be finished." `code/pretrain_data/screen_ckpt_dan.py` screens slices on Dan's `lean_staten`
checkpoints (Stage-1 base `stage1_best12_s0_b1200`, the ladder's actual starting point, and `la_T1_best12_s0_r8`
after 8 RL rounds), state-format loss via his `decompose` + `loss_on`, pass@k via his environment sampler judged by
the PINNED nd_verify. Dan's worktree (commit 0c492902) re-created from the nd-takehome clone; checkpoints re-fetched
from his bucket (r8 sha256 a992485e… matches the earlier eval's record). Mistake corrected: the first launch ran all
24 pools instead of the ten asked for; stopped and relaunched on the ten with a resume flag. Both runs paused at
Charles's request after finishing their current pool. Reference numbers: Stage-1 K12 loss 0.067, pass@1 0.67,
pass@32 0.98; round-8 0.057 / 0.94 / 1.00 (RL sharpened in-distribution without changing pass@32).

**Status of the ten (JSONs under `data/pretrain_data/screens/ckpt_dan_*.json`; partial table in pools.md):**
- Stage-1 done: joins10, minimal_joins10, distract10, minimal, dial_1e1, tail10, wrong100, pairs1. **Remaining: sub10, aux10.**
- Round-8 done: joins10, minimal_joins10, distract10, minimal, dial_1e1. **Remaining: tail10, wrong100, pairs1, sub10, aux10.**
- Extra pools finished before the restriction (kept): Stage-1 cap6, joins1, spice1, sub1; round-8 cap6, joins1.

**To finish** (resumes, skips done pools; ~12 min per pool per run when run alone, ~15 when both run):
```
cd code && ND_REGISTRY=0 nohup uv run python pretrain_data/screen_ckpt_dan.py --ckpt stage1_best12_s0_b1200 --no_doc \
  --pools k12_sub10 k12_aux10 > ../data/pretrain_data/screens_dan_stage1.log 2>&1 &
cd code && ND_REGISTRY=0 nohup uv run python pretrain_data/screen_ckpt_dan.py --ckpt la_T1_best12_s0_r8 --no_doc \
  --pools k12_tail10 k12_wrong100 k12_pairs1 k12_sub10 k12_aux10 > ../data/pretrain_data/screens_dan_r8.log 2>&1 &
cd code && uv run python pretrain_data/screen_ckpt_dan.py --doc_only
```
Nothing trained, nothing committed.

## 2026-10-08 — Dan-checkpoint screens finished, read

Resumed the two paused screen runs (Stage-1: sub10, aux10; round 8: tail10, wrong100, pairs1, sub10, aux10); both
finished by 09:02 UTC-7 (JSONs under `data/pretrain_data/screens/ckpt_dan_*.json`, logs `screens_dan_{stage1,r8}.log`).
The `--doc_only` table step failed because the session scratchpad holding Dan's worktree and checkpoints had been wiped
again; `screen_ckpt_dan.py` now imports Dan's code lazily (`load_dan()`, path overridable with `DAN_SCRATCH`), so the
table step runs from the JSONs alone. Table complete for all ten list pools plus four extras in `pools.md`
"Checkpoint screens on Dan's lean_staten checkpoints"; reading in `screens_analysis.md` § 7.

Learned (one checkpoint per row, 128 prompts × 32, no seeds): Robbie's and Dan's Stage-1 rankings agree (Spearman 0.93 on
pass@32, 0.86 on loss ratio, n = 14; regime agrees on 12 of 14). Two regimes move on the ladder's own base: `k12_minimal`
redundant → edge (the state model solves the problems at 0.98 but gives minimal proofs 1.95× its K12 loss: it learned the
padded shape); `k12_tail10` edge → redundant (depth-3 tails 0.87 at base, 1.00 after RL: E9's gate is met before any new
data, so the tail pool loses its ladder slot). RL's gain per slice is anti-correlated with base pass@32 (ρ −0.89, n = 11)
except distractors (+0.05 from 0.63, loss on their proofs up 27 %): premise selection is what the ladder does not learn
from K12 targets, so `k12_distract10` moves up the list. Join slices gain +0.23 to +0.33 while the loss on the join pool's
reference proofs rises 10–15 %: RL solves them its own way; loss after RL is not solvability, and regime labels on an RL
checkpoint are not comparable with Stage-1 ones. `k12_aux10` 0.00 on both models, confirmed far. Re-based Stage-1 gates for
E10 / E5 / E6 / E1 / E3 written in § 7. Nothing trained, nothing committed.

Next: pre-register the re-based gates in each proposal; build the long relevant pool (§ 6 item 3); screen the seven
new pools on Dan's checkpoint when the worktree is re-created; J9 continued-pretraining pilot once the budget is confirmed.
Open: whether `k12_tail10` is dropped from the ladder list or kept as the depth control (default: dropped, noted in
questions.md).

## 2026-10-08 (evening) — atom-count pools (E23)

Charles asked whether the tokenizer allows more symbols than we use (no: both tokenizers are hard-coded to P Q R S;
19–25 % of released textbook problems use all four; 26 of 91 v1 exclusions were for 5–7 atoms) and then for "pools with
only 1, 2, 3, and 4 symbols respectively". Built `code/pretrain_data/atom_pools.py` → `data/pretrain_data/pools/atoms/atoms{1..4}.jsonl.gz`:
K12's own k-atom rows plus the size ladder's generator population (2.2M rows) with exactly k atoms, one record per
curry-aware class, pinned verifier, depth ≤ 2, audited (removed 7 / 18 / 14 / 0 eval twins and 4 / 6 / 4 / 1 sealed
twins), flat 2–12, K12's 1,024-row val head. `atoms2–4` 153,967 training rows each; `atoms1` 34,296 (population
exhausted, deficit at every length). Verified: every training row has exactly k atoms, no duplicate classes; 20 raw
samples read (1-atom theorems degenerate). Plan row E23 added. Screens (`pools.md` "Atom pools: reading"): data
screens put all four next to K12 (atom count invisible to rule-shape screens); Robbie's base finds all four redundant
(loss 0.69–1.01× K12, pass@32 0.89–0.94). Only training can tell whether a two- or three-letter Stage-1 generalises to
four letters on the textbook sets.

Infrastructure: the scratchpad wipe also took K12 and the fork clone at 51604b3. K12 re-fetched from Dan's bucket to
`data/pretrain_data/dan/` (gitignored, `K12_PATH` override; 13,515,773 bytes, prompts identical to `k12_minimal`'s);
the fork re-cloned to `~/.cache/nd-rl/fork51604b3` (`ND_FORK`, now the `eval_ckpt.py` default); `screen_ckpt_dan.py`'s
`DAN_SCRATCH` default moved to `~/.cache/nd-rl` (Dan's worktree and checkpoints not yet re-created). `mix.py` now
preserves the "## Checkpoint screens" and "## Atom pools" sections of pools.md (it would have dropped them). Nothing
trained, nothing committed.

## 2026-10-09 — § K pools built and screened

Charles: "brainstorm other ways to diversify the pool ... inspiration from the ones that did well (harder to solve)",
then "try running the same checks on these new pools". Brainstorm recorded as ideas.md § K (12 ideas). Built 11 pools
with `code/pretrain_data/k_pools.py` (K3 join chains and K11 case-flip pairs not built: new generation logic needed):
`k12_jdist10`, `k12_jsubst10`, `k12_prem4_10` (+ nested 1e2, 1e3), `k12_long16_10`, `k12_goalimp10`, `k12_nestore10`,
`k12_reductio10`, `k12_reuse10`, `k12_minfloor8`, and `k12_hard10` (Robbie-base mining, still running at the time of
writing: 33,000 candidates × 8 samples, keep ≤ 3 hits). Generator pools use the pinned `gen.py`'s `Gen` class through its
API with `max_prem 5` (the CLI never asks for more than 3 premises); the pinned file is unchanged. Every pool is a 10 %
slice replacing K12 rows outside the val head; sources short of 15,500 filled with atom permutations (jdist 4,573, prem4
3,934, nestore 13,877, reductio 6,014). 21 raw samples read across seven pools. Data screens and Robbie-base checkpoint
screen on all eleven; table and reading in `pools.md` ("K pools", "K pools: reading").

Learned: the four-premise pool is 0.00 at base at every fraction (coverage 0.69), the cleanest hard axis so far and the
surfacing dial § 6 asked for; joins + distractors 0.05 (joins alone 0.19, distractors 0.55): hard axes multiply; the
long pool (13–16 lines) covers more needed cells than any pool (0.77) but its problems are solvable short (pass@32
0.69, loss 3.6×); `nestore` and `reductio` are easy problems with padded proofs (pass@32 0.92, loss 2.7–3.0×) and would
teach padding; `reuse` and `goalimp` are redundant: filtering the generator population by a structural feature
reproduces K12. Default pilot-list change in questions.md 8. Dan-base numbers pending his worktree. Nothing trained,
nothing committed.
- `k12_hard10` finished (2 h 10 min for 33,000 × 8 on the laptop): 4,757 hard of 33,000; enriched in length and ∨E/⊥E,
  not in inconsistent premises; coverage 0.55, loss 1.22×, pass@32 0.64 on the selecting base. Adds little; not on the
  pilot list (`pools.md` "K pools: reading", last paragraph). Twelve § K pools screened in all; the doc's ranking updated.
- Matched distractor check (Charles: "how does the distractor slice compare to the non distracted questions"; "did the
  models get nothing right with more premises than trained on?"): `code/pretrain_data/matched_distract.py`, 200 paired
  problems on Robbie's base. Distracted 0.53 vs 0.92 undistracted; by total premises: ≤ 3 solved 0.74–1.00, 4 and 5 solved
  0 of 71 while their originals are solved 0.83. The wall is the prompt's premise count, not premise use; an earlier
  reading in the doc and in chat (that some four-premise prompts are solved when the extras are unused) was wrong and is
  corrected in pools.md and the doc. Corrected also: the premise-pool zero is on 160 sampled problems × 32, not 15,500.

## 2026-10-09 (later) — release 2026-10-09 published

Charles: "publish those with a readme that describes it for now and the current progress. I will return to this
workstream." Built `code/pretrain_data/release_1009.py` → `data/pretrain_data/release_2026-10-09/`: 17 pools as slice
files (the new records with their row position; K12 itself is not redistributed, it is fetched from Dan's bucket and
hash-checked), `assemble.py` (rebuilds every pool and verifies it against `MANIFEST.json`; all 17 OK), `screens.json`,
`matched_distract.json`, README with the pools, the audit, the screens, the progress and the queued hypotheses.
Published: prem4 (10 %, 1 %, 0.1 %), distract (10 %, 30 %, minimal 10 %), joins (1 %, 10 %, minimal 10 %) with the
12,295 join problems, jdist10, jsubst10, long16_10, the ↔ dial (four fractions), k12_minimal as proofs by position.
30.8 MB. Not released: redundant, far, control, padding-teaching, mined, atom-count and the unscreened 10-07 pools.
Mistake on the way: the first build wiped the hand-written README (the builder rmtree'd the folder); restored, and the
builder now only regenerates `slices/`. The workstream's code and docs are committed with the release (first commit
of the workstream). Nothing trained.

## 2026-10-09 (night) — textbook72_distract2: the wall holds on the sealed set

Charles: copy the 72 with two irrelevant premises each, "just like the distractor set", and re-evaluate Robbie's latest
pretrain and Dan's RL models. Pre-registered in `experiments/pretrain-data/textbook72_distract2/proposal.md` (committed
632a2c9 before the run); sealed-set handling in questions.md 9 (evaluation-only copy, gitignored, aggregates by bin).
Dan's worktree re-created at `~/.cache/nd-rl/fork_dan` and four checkpoints fetched (r8 s0/s1/s2, Stage-1 s0).
Builder `textbook72_distract.py`: minimal length first (minlen, 10 s), then two depth-≤2 distractors not among the
subformulas, satisfiable with the premises, shortcut-checked against minimal + 1; 71 of 72 placed, 25 cross to ≥ 4
premises. Result (k 32, T 1.0, one seed each): solved-at-least-once original → distracted: Robbie d384 12 → 5, d768
11 → 6, Dan Stage-1 25 → 12, Dan r8 s0 49 → 28; on the 25 crossing problems 5 / 3 / 8 / 17 → **0 / 0 / 0 / 0**; on
the 46 staying at ≤ 3 premises 7 → 5, 8 → 6, 17 → 12, 32 → 28. The premise-count wall is a property of the models,
not of generator problems. Charles asked to pause; all four jobs had finished, nothing killed, Dan's r8 seeds 1 and 2
not launched (status PAUSED). Prediction recorded for the premise-pool Stage-1: distracted-72 ≥ 40 of 71 after the
ladder, against 28 now.
