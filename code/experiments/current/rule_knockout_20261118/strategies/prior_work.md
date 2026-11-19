---
written_on: 2026-11-18
written_by: agent:claude
---

# Prior work audit: which existing strategies and sub-types are the best next knockout arms

Scope: what the group has already built (Dan's taxonomy and pattern runs, Dan's σ knockout, Leon's reruns and
direction/habit work, Robbie's long-RL note and today's rule knockouts). Read-only audit, 2026-11-18. Nothing was
fetched, checked out, modified or posted.

Three kinds of statement are kept apart below:

- **[doc]** a number or finding quoted from a document in the repo (source given).
- **[count]** my own quick count with `quick_census.py` (same directory as this file). It is a simplified
  re-implementation of the documents' predicates, **not Dan's `strategies.py`, and not validated** beyond the checks
  in section 6. Treat every [count] as approximate.
- **[interp]** my reading.

## 0. Things that changed my answer while reading

1. **Robbie's cap-6 corpus already is a multi-arm knockout.** [count] Several of Dan's compositions have zero or
   near-zero records in the 154,990-record pretraining file while being in 24 to 91 % of the RL targets' reference
   proofs: disjunctive syllogism 0, IMPE chain ≥ 3 0, nested case split 0, case split on a derived disjunction 71
   (strict: 1), ORE inside an IMPI box 158, case split whose conclusion is ⊥ 33. No new pretraining is needed to
   read their emergence: classify the control arm's `found_stream.jsonl`.
2. **Doing that on tonight's live control runs already gives a spread** [count, files pulled 8:34 pm PT, about 72 min
   of RL]: disjunctive syllogism appears by oracle call 1 to 5, IMPE chain ≥ 3 by call 9 to 10, strict derived-∨E by
   call 23 to 26, a nested case split in the second branch on one of two seeds (call 21), a nested case split in the
   first branch never, a case split aimed at ⊥ never. Table in section 2.
3. **Dan's "cap-6" column is not Robbie's file.** [doc] `TABLES.md` classifies `data/p2/train_depth3_f0_a1.jsonl`
   (155,000 records, 0 proofs at box depth ≥ 3; run 4b summary). Robbie's `make_data.py` reads the fork's
   `data/train.jsonl` (154,990). The radical-scoping summary compares the two: ORE 1.46 vs 1.47 %, DN 8.33 vs
   8.36 %, classical reductio 6.80 vs 6.81 %, NEGI 9.93 vs 10.37 %, BOTE 1.95 vs 1.90 %. [count] Robbie's file has
   5,664 records (3.65 %) at box depth 3, the other file none. So Dan's cap-6 strategy shares are a close proxy
   except for anything depth-related, and sub-type shares need a recount on Robbie's file.
4. **The first-hit-cost hypothesis is already in Dan's September results**, nearly word for word: [doc] ignition
   study, novelty-campaign summary: "Reductio (a rule sequence): ignition is the pre-RL base rate. 4 of 11 draws
   produce the strict shape before RL (8×10⁻⁶ – 2.3×10⁻⁴ per sample); exactly those four ignite, in rate order; the
   seven zero-rate draws end at 0/606 after 155k attempts. The first-proof round matches 1/(r·k·N) within ~2.5×."
   And once one proof exists it spreads: "one training step on a sibling arm's found proofs ignites 11/11
   non-igniters in the next round". The same study has the counterexample: depth-3, where "of five draws with zero
   depth-3 samples in 600k, three ignite (rounds 3–5) and two never — decided during training on depth-≤2 proofs,
   not by sampling luck".

## 1. Sources read

| what | where |
| --- | --- |
| Today's plan and code | `/Users/robbie/repos/nd-rl/code/experiments/current/rule_knockout_20261118/PLAN.md`, `knockout.py`; live pulls in `/Users/robbie/repos/nd-rl/artifacts/rule_knockout_20261118/live/` |
| Robbie's long-RL note | `/Users/robbie/repos/nd-rl/.claude/notes/long-rl-what-models-learn.md`; `/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014/README.md` |
| Dan's taxonomy | `origin/dan:docs/papers/proof-strategies/{TAXONOMY,TABLES,README}.md`; `origin/dan:experiment-summaries/2026-10-18-proof-strategies-…/README.md` |
| Dan's σ knockout | `origin/dan:experiment-summaries/2026-10-12-create-or-elicit-…/README.md`; `origin/dan:docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/README.md` |
| Dan's earlier pattern runs | `origin/dan:experiment-summaries/2026-09-15-novelty-campaign/`, `2026-09-17-round2-run5-required-pools/`, `2026-09-18-round2-run2-six-new-patterns/`, `2026-09-19-round3-run4a-…/`, `2026-09-19-round3-run4b-…/`, `2026-10-02-radical-scoping-…/`, `2026-10-02-organism-analysis-…/` |
| Dan's pattern code | `/Users/robbie/repos/nd-takehome`, `git show dan/dan:patterns.py`, `dan/dan:patterns2.py` |
| Leon: taxonomy rerun | `origin/leon/high-rl-paper:code/experiments/current/proof_strategies_grpo_20261101/README.md`, `common.py`, `classify.py` |
| Leon: knockout with GRPO | `origin/leon/propensity-capability:experiment-summaries/2026-11-03-knockout-grpo-thinking/README.md`, `2026-11-04-grpo-on-dan-knockout-base/README.md` |
| Leon: factor read, directions | `origin/leon/propensity-capability:experiment-summaries/2026-11-06-factor-read-strict-106/README.md`; `code/experiments/current/propensity_capability_20261109/{README,SPEC}.md`, `results/tables.md`, `results/posthoc_anywhere.md` |
| Leon: planted habits | `origin/leon/two-authors:code/experiments/current/author_tags_20261110/{HABITS,GATE_RESULT}.md`, `two_authors_20261111/GATE_RESULT.md` |

**Not found.** The 2026-11-06 meeting notes are not in the repo (`docs/research_calls/` on `origin/dan` stops at
2026-09-27; no branch has the phrase). I relied on the brief's paraphrase and did not open the Google Doc.
`two_authors_20261111/` has no README; I read its `GATE_RESULT.md` only. I did not read `EXAMPLES.md`, `blog.md`,
`coe_sigma.py` or `strategies.py` itself (not available locally, see section 5).

## 2. The natural knockouts in Robbie's corpus, and what tonight's runs already show

[count] `quick_census.py` on `long_rl_20261014/data/pretrain.jsonl.gz` (154,990 records) and on the 13,946 RL
targets in `data/rl.jsonl` that carry a reference proof. Shares are of records.

| composition (existing definition) | pretraining records | RL reference proofs |
| --- | ---: | ---: |
| IMPE chain ≥ 2 (`patterns2.impe_chain`) | 2,594 (1.67 %) | 12,706 (91.1 %) |
| IMPE chain ≥ 3 (#1 `mp_chain3`) | 0 | 8,223 (59.0 %) |
| IMPE chain ≥ 4 (`impe_chain4`) | 0 | 4,075 (29.2 %) |
| any ∨E | 2,277 (1.47 %) | 12,454 (89.3 %) |
| ∨E with a branch longer than its assumption (`patterns.ore_shape`, not both one-line) | 1,908 (1.23 %) | 12,454 (89.3 %) |
| ∨E, both branches one line | 369 (0.24 %) | 0 |
| ∨E on a derived disjunction (#5 `on_derived`) | 71 (0.05 %) | 9,735 (69.8 %) |
| … strict (`derived_ore_strict`: not PR / AS, X ≠ Y) | 1 | 9,735 (69.8 %) |
| nested ∨E (#5 `nested`) | 0 | 8,682 (62.3 %) |
| … nested in the first branch (Robbie's shape) | 0 | 7,213 (51.7 %) |
| … nested in the second branch | 0 | 5,371 (38.5 %) |
| ∨E whose conclusion is ⊥ (Robbie's class 4) | 33 (0.02 %) | 719 (5.2 %) |
| disjunctive syllogism (#6) | 0 | 4,451 (31.9 %) |
| ex falso, ⊥ depends on an open assumption | 651 (0.42 %) | 4,469 (32.0 %) |
| ex falso from the premises alone (#9 `inconsistent_premises`) | 2,300 (1.48 %) | 0 |
| ∨E inside an →I box (`impi_ore`) | 158 (0.10 %) | 3,333 (23.9 %) |
| box depth ≥ 3 (`depth3`) | 5,664 (3.65 %) | 7,436 (53.3 %) |
| box depth ≥ 4 (`depth4`) | 0 | 2,320 (16.6 %) |
| →I inside an →I box (#3 `curried`) | 20,502 (13.2 %) | 1,466 (10.5 %) |
| →I, assumption used / never used | 36,831 (23.8 %) / 39,693 (25.6 %) | 4,215 (30.2 %) / 281 (2.0 %) |
| ¬I that is not only feeding DN (#7) | 5,514 (3.56 %); working 2,061 (1.33 %) | 1,190 (8.5 %), all working |
| modus tollens (#8, route `mt`) | 83 (0.05 %) | 70 (0.5 %) |
| σ, DN on a ¬I line (#10 via ¬I) | 10,554 (6.81 %); routes immediate 6,196, vacuous 4,354, other 4 | 268 (1.9 %); routes cases 175, ∧I 83, mixed 10 |
| σ⁺, DN on any derived line | 10,656 (6.88 %) | 268 (1.9 %) |
| DN on a premise / on an assumption (#11) | 597 / 1,708 | 0 |
| lemma reuse (#13) | 2,993 (1.93 %) | 6,868 (49.2 %) |
| reiteration (rule R) | 4,902 (3.16 %) | 0 |
| `negi_ande_hyp` | 17 | 0 |

The four arms I recommend that need a filter (IMPE chain ≥ 2, ex falso under an assumption, working ∨E, depth ≥ 3)
overlap in only 54 pretraining records; their union is 10,763 records (6.94 %). In the reference proofs they
co-occur heavily (13,926 of 13,946 use at least one).

[count] The same script on tonight's live `found_stream.jsonl` files (distinct Lean-accepted proofs; "call" = oracle
call of first appearance; pulled 8:34 pm PT):

| composition (pretraining records) | full-ei s0 (7,609 proofs, 60 calls) | full-ei s1 (12,505, 68) | dn-ei s0 (7,293, 63) | exfalso-ei s0 (11,074, 61) | mp-ei s0 (3,217, 64) | cp-ei s0 (4,964, 95) |
| --- | --- | --- | --- | --- | --- | --- |
| ∨E on a derived disjunction (71) | 201 proofs, call 3 | 611, call 1 | 365, call 4 | 732, call 1 | 39, call 9 | 1,672, call 5 |
| disjunctive syllogism (0) | 507, call 5 | 515, call 1 | 344, call 1 | 0 | 131, call 5 | 530, call 5 |
| IMPE chain ≥ 3 (0) | 1,036, call 9 | 2,260, call 10 | 1,045, call 6 | 1,976, call 5 | 0 | 3,517, call 9 |
| IMPE chain ≥ 4 (0) | 484, call 14 | 1,145, call 17 | 560, call 13 | 992, call 9 | 0 | 1,996, call 14 |
| strict derived ∨E (1) | 148, call 23 | 529, call 26 | 322, call 14 | 654, call 18 | 5, call 45 | 1,668, call 5 |
| nested ∨E, second branch (0) | 0 | 12, call 21 | 0 | 0 | 0 | 0 |
| box depth ≥ 4 (0) | 0 | 0 | 0 | 343, call 21 | 0 | 0 |
| nested ∨E, first branch (0) | 0 | 0 | 0 | 0 | 0 | 0 |
| ∨E whose conclusion is ⊥ (33) | 0 | 0 | 0 | 0 | 0 | 0 |
| σ (10,554; 0 in the `dn` arm) | 5 | 32 | 0 | 903, call 1 | 19 | 1 |

Readings [interp]:
- The control arm already orders five never-seen compositions by time of first appearance, and two never appear.
  That is the spread of first-hit costs Robbie is asking for, on runs that exist.
- `exfalso-ei_s0` writes 903 σ proofs against 5 and 32 in the two controls: PLAN.md's "ex falso has a substitute"
  looks right. This is one seed and an unvalidated counter.
- The mp-arm seed shown (s0, where IMPE never appeared) has no IMPE chains, as it must.
- These are live, partial runs at about 72 min of RL, classified by an unvalidated script. Rerun with Dan's
  classifier on the finished runs before quoting.

## 3. Every candidate the group has defined

"First-hit cost" is my estimate of how many decisions a model that never saw the composition must get right by
chance, given it knows each rule involved, plus the measured rate where a document has one. "Forced" means the goal
or premises dictate the step; "choice" means a competing accepted proof exists.

### 3a. Robbie's four rule arms (2026-11-18, `PLAN.md`)

| arm | detection | pretraining removed | RL reference proofs | result so far (brief; `live/STATUS.md`) | kind |
| --- | --- | --- | --- | --- | --- |
| `mp` (IMPE) | regex on the rule name | 59,429 of 154,990 | 13,774 of 13,946 | appeared in 1 of 4 runs at 34 min, then spread (mp-ei_s1: 307 proofs on 196 targets at the 8:34 pm pull) | pattern knockout (`nF nA` seen for NEGE) |
| `cp` (IMPI) | same | 57,956 | 4,224 | 0 | pattern knockout (`fun` boxes seen for NEGI and ORE; IMPI's box ends `exact n`, NEGI's `exact ( n : False )`) |
| `exfalso` (BOTE) | same | 2,951 | 4,469 | 0; 23 to 88 solved targets whose reference uses BOTE | token knockout (`.elim` never seen) |
| `dn` (DN) | same | 12,950 | 271 | 0 | token knockout (`Classical.byContradiction`, `hh` never seen) |

### 3b. Dan's 13 strategies and sub-types (`TAXONOMY.md` v1.2; shares from `TABLES.md` § 2 and § 10)

K12 and Charles's textbook shares are [doc]. "Robbie's file" is [count] from section 2. A classifier exists for
every row: `proof_strategies/strategies.py` on the fork branch (not local, section 5).

| # | strategy / sub-type | detection rule (TAXONOMY § 2) | K12 % | Dan's cap-6 % | Charles's textbooks % | Robbie's file / RL refs | knocked out before? | choice or forced | first-hit cost |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | direct: `mp`, `ande` | every IMPE / ANDE line | 51.2 (mp 45.4, ande 8.2) | 44.5 | 84.5 | IMPE 38.3 % / 98.8 % | IMPE: today | forced | 1 |
| 1 | `mp_chain3` | IMPE citing an IMPE citing an IMPE | 0.00 | – | 2.67 | 0 / 59.0 % | natural zero; `impe_chain4` f = 0 at cap 6: 0 / 500 on 2 seeds, no target proved at all (run 2) | forced by the premises | 1 citation choice; appears by call 9 to 10 in tonight's controls [count] |
| 2 | ∧-introduction | ANDI lines | 32.6 | 22.9 | 47.6 | 20.9 % / 30.9 % | no | forced by goal | 1 |
| 3 | conditional proof, `working` | IMPI (s, e) with s = e or s ∈ hyps(e) | 20.8 | 20.6 | 49.2 | 23.8 % / 30.2 % | IMPI: today, 0 | forced by goal (Leon: first-move propensity 76 to 100 % at base) | open + close a box |
| 3 | `vacuous` | IMPI with s ≠ e and s ∉ hyps(e) | 28.2 | 24.4 | 10.7 | 25.6 % / 2.0 % | no | forced by goal | – |
| 3 | `curried` | origin(e) is an IMPI inside the box | 8.21 | – | 8.56 | 13.2 % / 10.5 % | via depth-3 (below) | forced by goal | 1 extra box |
| 3 | `explosive`, `hyp_syll` | origin(e) is a BOTE depending on s / an IMPE reached by ≥ 2 mp steps from s | 0.33, 0.04 | – | 8.02, 3.21 | explosive 0.34 % / 0.03 % | no | mixed | 2 |
| 4 | ∨-introduction (`root`, `case_branch`, `feeds_ore`, `other`) | ORI lines | 44.3 | 39.6 | 31.0 | ORI1 18.9 %, ORI2 18.7 % / 0 [count: no ORI line in the rl.jsonl references] | no | choice of disjunct | 1 |
| 5 | cases, `genuine` | ORE where every branch uses its case | 8.79 | 1.1 | 33.69 | 1.12 % / 89.3 % | whole ∨E proposed 2026-10-03 (`Or.elim`-free Stage-1, radical-scoping pilot A); I found no summary of it being run | forced when the theorem needs ∨E (K12 sample 6.1 %, textbooks 26.6 %, TABLES § 5) | 2 to 3 (open two boxes, close both on the same formula) |
| 5 | `redundant` | a branch proves the goal without its case | 37.64 | 0.4 | 0.00 | 0.35 % / 0.27 % | no | never needed | – |
| 5 | `on_premise` / `on_hypothesis` / `on_derived` | rule of origin(j): PR / AS / other | 29.61 / 0.93 / 14.06 | – | 16.04 / 12.83 / 10.70 | 1.41 / 0.01 / 0.05 % ; 44.4 / 0 / 69.8 % | `derived_ore` dial, below | forced | 1 citation choice |
| 5 | `nested` | the ORE line lies inside a branch of another ORE | 0.00 | – | 6.95 | 0 / 62.3 % | natural zero; never written in the first branch (0 of 72,852, long-RL note) | forced | one token at p ≈ 3e-6 (long-RL note) |
| 5 | `constructive_dilemma`, `trivial_merge`, `or_intro_decoration`, `all_refuted`, `bot_case`, `on_lem` | see TAXONOMY | 0.00, 2.14, 11.41, 0.00, 1.34, – | – | 0.53, 0.53, 0.00, 0.53, 0.00, – | trivial merge 0.26 % | `ori_ore` (decoration): 0 pattern proofs in every arm (run 2) | decoration is never needed | – |
| 6 | disjunctive syllogism | ORE with exactly one refuted branch (ends in a BOTE whose ⊥ depends on the case), other branch uses its case | 0.03 (43 records) | 0.0 | 14.4 | 0 / 31.9 % | natural zero | choice (the refuted case can be closed by a vacuous ¬I or reductio) | 1 (BOTE in a new place); appears by call 1 to 5 [count] |
| 7 | proof of a negation, `working` and routes (`immediate`, `mt`, `or_weaken`, `and_elim`, `and_intro`, `imp_hyp`, `cases`, `self`, `mixed`) | NEGI not cited only by DN; route = how the box reaches ⊥ | 4.74 (working 2.15; immediate 1.50) | 3.1 (1.2) | 38.0 | 3.56 % (1.33 %) / 8.5 % | no; organism-analysis: "¬I boxes are the one hard-step class EI does not move" | forced when the goal is ¬A (Leon: propensity 63 to 71 %) | 2 to 4 by route |
| 8 | modus tollens (`strict`, `neg_side`, `chain`, `classical`, `contraposition`) | a NEGI with route `mt`; or an IMPI from ¬B whose box is such a NEGI | 0.05 (76) | 0.1 | 11.8 | 0.05 % / 0.5 % | no | forced | 3 (box, IMPE, NEGE) |
| 9 | ex falso (`inconsistent_premises`, `refuted_case`, `explosive_cond`, `other`) | BOTE lines, by where the ⊥ comes from | 8.04 (6.23, 0.06, 0.33, 1.42) | 1.9 (1.5 from premises) | 24.1 (2.67, 14.44, 8.02, 0) | from premises 1.48 %, under an assumption 0.42 % / 0, 32.0 % | whole rule: today, 0 | choice: K12 sample requires BOTE on 0 %, textbooks 8.9 % (TABLES § 5); Leon's habits: BOTE at 63.1 % of 22,523 sites, vacuous reductio the rest | whole rule: unseen token. Under an assumption only: 1 |
| 10 | proof by contradiction σ (kinds `vacuous`, `pointless`, `scaffolded`, `neg_goal`, `free`; `template` / `working`) | DN whose cited line is a NEGI (or another derived line with ¬¬G ∉ Sub) | 10.79 (working 0.04 = 59 records; free 6) | 6.8 (4 working) | 20.9 (all working, 20.3 free) | 6.81 %, all but 4 immediate or vacuous / 1.9 % | Dan: 0 / 6 knockout seeds acquire a family in 16 EI rounds vs 6 / 6 full (p = 0.00024); not by 13.2× pretraining compute on s1; 0.1 % dose crosses on 3 / 3. Leon, GRPO on Dan's bases: 0, 1, 0 of 106 on Dan's pool; 13 of 106 on 1 of 3 seeds with a σ-rich pool | forced on classical-only theorems with no ¬¬ supplied; elsewhere a choice | 2 novel decisions (open a box on ¬G; DN on the NEGI). Knockout bases: p̂ ≈ 1e-6 to 3e-4 per member; opening the ¬G box scores −18.0 to −21.6 log p vs −10.5 on the full base (create-or-elicit). Factor read: pick 0 of 64 on 87 to 106 of 106 theorems; given the pick 8.5 to 15.0 % of forced draws finish |
| 11 | double-negation elimination (`on_premise`, `on_hypothesis`, `on_derived`) | DN lines that are not #10 | 3.06 (0.80, 2.10, 0.17) | 1.5 | 3.2 | 0.39 %, 1.10 % / 0 | `on_derived` removed with σ⁺ | forced | 1 |
| 12 | excluded middle (`cases_on_lem`) | a DN line whose formula is X ∨ ¬X | 0.03 (47) | 0.0 | 0 (schemata 5.3) | not counted | no; long-RL: 0 / 19 and 0 / 80 | choice | ≥ 4 |
| 13 | lemma reuse (`across_boxes`, `multi_step`) | a derived line with ≥ 2 distinct citing lines | 11.71 | 2.3 | 16.0 | 1.93 % / 49.2 % | no | choice (re-deriving is always possible) | 1 |

### 3c. Dan's earlier patterns (`patterns.py`, `patterns2.py`, `coe_sigma.py`)

All are predicates on the dependency-pruned proof. These runs used a 3.2M-parameter model at cap 6, 8 EI rounds,
k = 32, judged by `nd_verify` (they predate the Lean-only rule).

| pattern | detection | knockout result [doc] |
| --- | --- | --- |
| `depth3` (P3) | maximum box depth ≥ 3 | f = 0 EI acquisition 0.341 / 0.271, frozen 0 (campaign); 8 arms 0.271 to 0.364 (follow-up); 10 of 16 f = 0 draws emit it before RL; of five zero-rate draws three ignite and two never (ignition study) |
| `reductio` (P2, σ strict) | DN citing a NEGI line that closes a box on ¬G | 0 / 0 at f = 0, but "untested, not a wall": 94 % of targets were intuitionistically provable. On a required pool: 2 of 3 f = 0 draws stay at 0 / 300; the one with a non-zero base rate is elicited to the 7-line stratum (51 targets) and nothing longer (run 5) |
| `derived_ore` (P1), `derived_ore_strict` | ORE whose disjunction line is not PR (strict: not PR / AS, X ≠ Y) | "amplification": the base assigns ≈ 1e-3 to the easiest instance; 16 examples in 155k give 22 %; 97 % of cap-6 derived-OREs are the degenerate `( X v X )` form. Cap-8 strict, required pool: f = 0 acquisition 0.087 / 0.080, base already reaches 27 of 300 targets (run 5) |
| `depth4` | maximum box depth ≥ 4 | cap 6, f = 0: 0 / 500 on both seeds (pre-registered as acquired: wrong). Cap 8: base writes it at 11 to 20 % per sample before RL, EI to 0.96 to 0.97 |
| `impe_chain4` | IMPE chain ≥ 4 | cap 6: 0 / 500 on both seeds, no proof of any chain target in 512k attempts (pre-registered as acquired: wrong) |
| `nested_ore` | ORE inside an ORE branch | 0 / 300 on both seeds (11 to 13-line targets, read as a length limit) |
| `impi_ore` | an IMPI whose box contains an ORE | f = 0: 0 / 500 on both seeds, "uninformative" (no target proved at all); at its natural 130 / 155k: base 12 to 26 theorems → 0.19 to 0.22 |
| `negi_ande_hyp` | a NEGI whose box applies ANDE to its own hypothesis | s0: base 0 → 0; s1: 3 hits in 600k → 0.147 (frozen 0.005). "Met exactly" |
| `ori_ore` | ORE whose disjunction line is an ORI line | decoration control: 0 pattern proofs in EI, frozen, base and transfer |
| σ⁺ (`coe_sigma.dn_derived`) | DN citing a line that is not PR / AS | the removal rule of the σ knockout: 16,961 K12 records (10.94 %) |
| lemma route | DN applied to the ¬¬X assumed by an →I lemma stating double-negation elimination | 0 in K12; 0 to 1 distinct proofs per round in knockout ladders |
| free+ | DN on a derived line, G not a negation, ¬¬G not a subformula of the sequent | knockout ladders: 0, 34, 4, 1, 2, 16 proofs in rounds 1 to 16 (full: 17,507 to 50,222) |

Run 2's verdict [doc]: "the pre-registered syntactic classes predicted nothing that the base rate did not, and the
one genuine zero-to-something event in the run happened in Stage-1 pretraining (cap 8, depth 4), not in RL."

### 3d. Leon

| item | definition | result [doc] | use as a knockout arm |
| --- | --- | --- | --- |
| directions `reductio`, `negation`, `conditional`, `cases`, `direct` (propensity_capability, SPEC.md) | the first proof action after the premise lines: open a box on ¬G; on A when the goal is ¬A; on A when the goal is A → B; start a case split on the first disjunctive premise; open no box | 30 of 54 primary cells read. Conditional: "candidate capability gain" at every dose, propensity flat at 76 to 100 %. Cases: "mix" (propensity 20.1 → 62.0 %, forced success 30.3 → 90.1 % on one 0 % seed). Negation and cases are "never applicable" on the strict 106. Reductio at 0 %: first-move propensity falls by a mean 10^-2.87 on the 106 while forced success rises | two of the four (conditional, negation) are dictated by the goal; they are not choices |
| factor-read stages (setup, pick + ¬¬T, reach ⊥, close box, DN, finish) | forced-prefix sampling along the σ route on the 106 | at 0 % bases two factors are at zero: the pick (0 of 64 on 87 to 106 of 106) and reaching ⊥ (no draw on 63 to 78 of 106). RL raises reaching ⊥ 2 to 4×; the pick stays at 0 of 6,784 at every 0 % RL end | a ready-made measurement of first-hit cost by stage |
| opens with ¬goal (proof_strategies_grpo) | first written step assumes the negated goal and the root is DN of that ¬I | base → RL: 6.2 → 13.0 % of proofs (Charles149), 1.1 → 13.3 % (13 to 20 lines) | – |
| habits (author_tags `HABITS.md`): step order, conjunct order, reiteration, idle reductio, disjunct choice, ∧ detour, different route, ex falso style, redundant case split | checker-neutral rewrites with "room" measured on K12 | room: step order 39.8 %, reiteration 66.5 %, conjunct order 9.1 %, ex falso style 13.0 %, redundant split 37.6 %; "different route (box multiset)" ≥ 8.6 % of 5,000 theorems, half of those with inconsistent premises | style, not capability. Planted habits failed their gates (section 4) |

Leon's rerun of Dan's classifier on his pipeline [doc]: his search-trace corpus has working #10 1.47 %, MT 0.13 %, DS
1.19 %, working ¬I 5.56 %, genuine cases 6.04 % (K12: 0.04 / 0.05 / 0.03 / 2.15 / 8.79). On solved Charles149
problems his base already uses them at textbook rates (17.5 / 9.9 / 19.1 / 32.8 / 33.4 % of proofs): "the problems,
not the corpus mix, set the strategy."

### 3e. Robbie's long-RL shapes (`long-rl-what-models-learn.md`)

| shape | definition | pretraining | result [doc] |
| --- | --- | --- | --- |
| depth-3 boxes | third nested box | 3.8 % | learned: GRPO s0 15 → 97 % of 1,805 ladder targets, s1 never (13 %), EI 99 %. Blocker was closing: "depth after the r-th consecutive close = max(0, 2 − r)" |
| case split nested in the first branch of a case split | – | 0 of 2,277 case splits | never: 0 of 72,852 found proofs; swap test 0 / 30 → 30 / 30; worst token log p ≈ −12.7 |
| box for a formula that is not in the goal | e.g. assume B to derive ¬B as a lemma | not counted | synthetic: 68 / 80 when the box concludes the goal, 0 / 80 when it concludes a lemma |
| classical reductio | – | present as templates | excluded middle 0 / 19, Peirce 0 / 80 in all runs |
| case split aimed at ⊥ | `¬A∨¬B ⊢ ¬(A∧B)` | [count] 33 records | 0 / 18 |

## 4. What argues against candidates

1. **A syntactic class does not predict emergence; the base rate does.** Run 2 pre-registered depth-4 and
   IMPE-chain-4 as "structural, will be acquired" and both were 0 / 500. Consequence: measure each knockout base's
   rate of the removed form before RL (Dan used pass@2,000 on 300 targets). PLAN.md lists this as not measured.
2. **Optional patterns give flat dials.** Campaign: the reductio dial was "untested, not a wall" because 94 % of its
   targets were intuitionistically provable; the cap-8 derived-ORE dial was "flat … base-reachable but never
   required"; `ori_ore` never appeared because it is never needed. Leon: "the problems, not the corpus mix, set the
   strategy". An arm needs targets where the strategy is required or clearly shorter, and the count must be of the
   strategy in accepted proofs, not of solved targets (PLAN.md already does this).
3. **Rule-level removal mostly removes idle templates.** K12: 59 of 16,717 reductio proofs do working reasoning;
   37.64 % of proofs hold a redundant case split; 28.20 % discharge an unused assumption. [count] In Robbie's file
   all but 4 of 10,554 σ records are immediate or vacuous, while the 268 RL references with σ use working routes.
   What a rule arm removes is not what the targets use.
4. **Length is confounded with structure.** Run 2's reviewer: "a pattern whose minimal instance exceeds the cap has
   base rate 0 and is not acquired"; whether that is length or structure "the run does not separate". Disjunctive
   syllogism, nested ∨E and IMPE chain ≥ 3 are absent from Robbie's corpus for this reason. [interp] Tonight's
   control runs disagree with run 2 for IMPE chains (they appear by call 9 to 10), in a different regime: 9.6M
   parameters, short bridge targets, Lean judge.
5. **Definitions the reviewers found wrong.**
   - "No K12 theorem is σ-covered" holds only under a semantic double-negation rule; the pre-registered rule gives
     2,340 (1.51 %). The switch was not logged.
   - "Every proof of a σ-covered theorem needs a working #10" is false: at least 573 accepted RL proofs eliminate a
     self-built double negation through a box hypothesis (an ∨E case or the lemma route) and carry no #10
     (dose1_s0 491; knockout 0). A filter on one spelling leaves other routes open, so the readout has to count the
     function (a self-built ¬¬ eliminated), not the spelling.
   - Disjunctive syllogism and excluded middle were tightened after the blind spot-check (K12 0.24 % → 0.03 % for
     DS); "the final #6 and #12 were never hand-checked blind" (agreement with the earlier labels: DS 10 / 20, LEM
     15 / 20).
6. **Replay.** Dan's EI replays 20,000 records of the arm's own pretraining set per round, so the full arm keeps
   seeing the strategy during RL (review of proof-strategies and of the 10-31 write-up). Robbie's EI does the same
   (`pretrain.py` `RETAIN = 20000`), from the filtered set in a knockout arm.
7. **A knockout lowers general ability.** create-or-elicit: held-out greedy 0.9517 → 0.8829; the gap is on the σ
   theorems (0.96 → 0.45) but pass@256 is also lower on intuitionistic theorems (holdout250 median 0.798 vs 0.748).
   Robbie's arms are not refilled (PLAN.md "Known limits"); Dan refilled per length.
8. **The target pool matters as much as pretraining.** Leon: with a pool where 40 to 45 % of targets are strict
   σ-covered, GRPO restores σ from a knockout base at R 0.037 to 0.065; on Dan's pool (2.7 %) never. He lists the
   pool, the narration channel (σ plans in 0.41 % of thinking traces) and the corpus as open confounds.
9. **σ in Robbie's harness is at the floor even in the control.** Long-RL note: excluded middle 0 / 19 and Peirce
   0 / 80 in runs whose pretraining had 6.8 % σ templates. [count] Tonight's controls wrote 5 and 32 σ proofs. A σ
   arm here would compare a floor with a floor.
10. **Leon's gates.** Author tags: the tagged bases "almost ignore the tag" (the route opening is 1.36, 1.09 and
    1.08 times as likely under tag A as B; the rule asked for 2), so the RL arms were not run; the tag binds only
    from a 10 % tie. Two authors: neither version passes G2 (bundle excess +9.1 against a bar of 10), "stop, no RL
    stage". Habits: "There are few genuinely different routes." Style habits are weak handles.
11. **Direction is mostly not a choice.** Leon's conditional and negation openings sit at 63 to 100 % propensity at
    base because the goal dictates them; 24 of 54 primary cells could not be read (forced attempts refused or
    capped; negation and cases never applicable on the 106).
12. **RL without the strategy can move it further away.** Leon's 0 % lines: reductio first-move propensity on the
    106 falls from 3.3e-05 to 2.1e-08 and from 2.8e-05 to 1.8e-10 (seeds 0 and 1, `tables.md`, % of attempts) while
    success when forced rises. [interp] First-hit cost is not a constant of the base; it drifts during RL, down for
    depth-3 with rewarded neighbours (run 4b's `mix` arm) and up here.
13. **Not a unique decomposition.** TAXONOMY § 1: "Strategies nest: a disjunctive syllogism is also a proof by cases
    and uses ex falso … 'uses' counts are therefore not additive." The primary label is dominated by goal shape:
    conditional proof is primary on 94.5 % of the RL targets' generator proofs, which is why the curried primary was
    added. The brief says the 2026-11-06 notes ask for a domain that breaks down "completely & uniquely"; the
    13-strategy "uses" labels do not, the primary label does but mostly reads the goal.

## 5. Reusing Dan's classifier with Robbie's harness

**Where the code is.**
- `proof_strategies/strategies.py`, `theorem_level.py`, `necessity_ps.py`, 39 unit tests: fork
  `dan-pandori/nd-takehome`, branch `dan_proof-strategies`, DONE `441de472`, review `e6bfe0ea`.
- **Not in the local clone.** `/Users/robbie/repos/nd-takehome` has remotes `origin` (Robbie's fork), `upstream` and
  `dan` with 23 `dan/*` branches, newest `dan/dan` at 2026-09-30; no `dan_proof-strategies`, no
  `dan_create-or-elicit`. It does have `patterns.py` and `patterns2.py` (`git show dan/dan:patterns2.py`).
- nd-rl's `origin/dan_proof-strategies` is the librarian's branch: documents only, no code.
- Leon already wrapped it: `origin/leon/high-rl-paper:code/experiments/current/proof_strategies_grpo_20261101/`
  (`common.dan_strategies()`, `classify.labels(nd, prompt)` → `S.analyse(nd, prompt, written=True)`,
  `theorem_level.label(prompt)`), from a shallow clone at `e6bfe0ea`. He reports 117 / 117 K12 cells reproduced and
  about 4 minutes for 154,630 proofs plus 87 reads on 8 processes.

**What matches.** Robbie's records carry `prompt` (`THM … SEQ … PRF`) and `proof` (`N1 … : RULE N… ; … ; QED`), the
same ND text the classifier parses. The harness converts every sampled Lean text back to ND with the overlay's
`lean_tok.inverse`, and `found_stream.jsonl` stores that ND text in `proof`. `knockout.py` filters with
`rules(r["proof"])` and builds its census from the same text, so a predicate from the classifier drops into both
places.

**What differs.**
1. **Ten atoms.** `nd_verify.verify.ATOM_NAMES` is `{P, Q, R, S}`; the classifier's files use bare imports of
   `patterns` and `nd_verify` for parsing. Robbie's `overlay/lean_tok.py` extends the set on import
   (`ATOM_NAMES.update(ATOMS)`), so import the overlay first, as `harness.py` does.
2. **Distractor premises.** The classifier's `Sub` is "all subformulas of the theorem's premises and conclusion".
   Distractors enlarge it, which changes #10's `scaffolded` kind (¬¬G or ¬G → ⊥ in Sub), `built_other`, #11
   `on_derived`, and the theorem-level `dneg` / σ-covered labels. Use labels do not depend on Sub. Either classify
   against a prompt with the uncited premises removed or accept that the working / template split is off on
   distractor records. I did not measure how many records this touches.
3. **A different file from Dan's "cap-6" column** (section 0, item 3). Recount; do not reuse his column.
4. **Ladder targets have no reference proof** (4,493 of 18,439), so target-side labels exist for 13,946 only.
   Theorem-level labels (`classical-only` by G4ip) need only the prompt. Bounded necessity (`minlen`, 12 lines, 60 s)
   will not reach Robbie's 7 to 80-line targets.
5. **`nd_verify`.** The classifier uses its parser, not its verdict (TAXONOMY: "nothing is judged by `nd_verify`").
   That is within the Lean-only rule, but confirm `analyse` does not call `verify_text`; `patterns.py`'s own
   `--test` does.
6. **Length convention.** Dan's tables bin by pruned lines including premises; Robbie's system of record is body
   lines.
7. **Not checked:** whether `strategies.py` has any cap on line numbers, premises or depth (Robbie's proofs go to
   n160, 10 premises, depth ≥ 4). Leon ran it on 13 to 20-line proofs.

**Stopgap.** `quick_census.py` here runs on both files in seconds. Section 6 says how far to trust it.

## 6. How far to trust `quick_census.py`

- Rule counts on the pretraining file equal PLAN.md's: IMPE 59,429, IMPI 57,956, BOTE 2,951, DN 12,950; ORE 2,277
  equals the long-RL note's "2,277 case splits" and the radical-scoping table.
- On `rl.jsonl` it gives IMPE 13,774, IMPI 4,224, BOTE 4,469 as in PLAN.md, and DN 268 against PLAN.md's 271: 9
  reference proofs have dead lines and the script prunes.
- Against Dan's cap-6 column (a different file): σ 6.81 vs 6.8, DNE 1.49 vs 1.5, ex falso 1.90 vs 1.9, genuine /
  redundant ∨E 1.12 / 0.35 vs 1.1 / 0.4 agree; ¬I 3.56 vs 3.1, →I with the assumption used 23.8 vs 20.6, lemma reuse
  1.93 vs 2.3 and box-free 51.7 vs 53.2 do not.
- Nested-first-branch is 0 in pretraining and 0 in every found file, as the long-RL note says.
- Four flagged proofs were read by eye (one each of DS, strict derived ∨E, IMPE chain 3, modus tollens) and are what
  the label says.
- Not implemented: `bot_case`, `constructive_dilemma`, `hyp_syll`, contraposition, #10 kinds, #12, the lemma route.

## 7. Recommendation

**Step 0, free: classify the runs that exist.** Run Dan's classifier (or the stopgap) over the finished control
runs' found proofs and report, per composition with ≤ 100 pretraining records, the oracle call of first appearance
and the count per call. That is seven natural knockouts with no new pretraining and it already spans "call 1" to
"never".

**Arms that need a filter** (each is a pattern knockout: every token stays in pretraining, so it tests composition
and not an unseen token; removal sizes are [count]):

| # | arm | filter on the pruned ND proof | removed from 154,990 | RL references using it (of 13,946) | prior result | first-hit cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | IMPE chain (`patterns2.impe_chain` ≥ 2) | an IMPE line that cites an IMPE line | 2,594 (1.67 %) | 12,706 (91.1 %) | chain ≥ 3 is absent already and appears by call 9 to 10 in both controls [count]; Dan's `impe_chain4` at cap 6: 0 / 500 ×2, no target proved | lowest: 1 citation choice |
| 2 | ex falso under an assumption (#9 minus `inconsistent_premises`; with ∨E this is #6 disjunctive syllogism) | a BOTE whose cited ⊥ depends on an open assumption | 651 (0.42 %); 2,300 top-level BOTE stay | 4,469 (32.0 %); 4,451 are DS | whole-rule arm today: 0 in > 440,000 accepted attempts, with σ as the substitute (903 σ proofs in exfalso-ei_s0 vs 5 and 32 in the controls [count]); DS, absent already, appears by call 1 to 5 | low: a known token in a new place; a real choice |
| 3 | working case split (`patterns.ore_shape`, not both boxes one-line) | an ORE with a branch longer than its assumption | 1,908 (1.23 %); 369 one-line-box ORE stay | 12,454 (89.3 %) | not run. Whole-∨E knockout was proposed 2026-10-03 and I found no run. Derived-∨E at f = 0 was "amplification" | middle: fill and close two branches on one formula; forced on 6.1 % (K12) to 26.6 % (textbooks) of theorems |
| 4 | depth-3 boxes (`patterns.depth3`) | maximum box depth ≥ 3 | 5,664 (3.65 %) | 7,436 (53.3 %), plus 1,805 ladder targets | Dan f = 0: acquired 0.271 to 0.364 on 8 arms, ignition seed-dependent; Robbie: GRPO s0 15 → 97 %, s1 13 % | middle: one more box, and the close-count rule; forced by the goal |
| 5 | nested case split, first branch | already absent (0 records); the arm is the dose: add disjunct-swapped proofs | 0 | 7,213 (51.7 %) | never: 0 of 72,852 found proofs; swap test 0 / 30 → 30 / 30 | high: one token at p ≈ 3e-6 |
| 6 (optional) | σ as a pattern (Dan's σ⁺), replacing today's token-level `dn` arm | a DN citing a line that is not PR / AS | 10,656 (6.88 %); 2,305 records with DN on a premise or assumption stay | 268 (1.9 %) | Dan: 0 / 6 in 16 rounds, 3 / 3 at a 0.1 % dose; Leon GRPO: 0 / 3 on Dan's pool | highest: two novel decisions; underpowered here (section 4, item 9) |

Why these: 1 to 4 are compositions of rules the base keeps, they remove 0.4 to 3.7 % of pretraining each and overlap
in 54 records, each is in 32 to 91 % of the reference proofs, and with step 0 and arm 5 they cover the range from
"appears in the first oracle calls" to "never in 72,852 proofs". Arm 2 is the only one with a competing accepted
proof, so it separates cost from need. Arms 1 and 4 are graded versions of today's `mp` and `cp` arms.

Not recommended: modus tollens (0.5 % of Robbie's reference proofs; it belongs on a textbook pool), lemma reuse and
reiteration (a model that cites lines can always do them; reiteration is in 0 reference proofs), the conditional and
negation directions (dictated by the goal), Leon's style habits (gates failed), whole-∨E removal (`Or.elim` becomes
an unseen token and 89 % of references need it).
