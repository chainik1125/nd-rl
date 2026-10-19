---
written_on: 2026-10-19
written_by: agent:claude
---

# proof-strategies: a 13-strategy taxonomy of the proofs in K12. K12's proof by contradiction, which contains the knockout's σ, is almost all template: 59 of the 16,717 proofs with a reductio do working reasoning. Textbook problems use a working reductio, modus tollens and disjunctive syllogism in 12–21 % of proofs, against ≤ 0.05 % in K12

Status: completed 2026-10-18 and reviewed 2026-10-19. **No hard-constraint violation, nothing quarantined.** This
was a classification run. It trained and sampled no model. One pod (RTX 3090, GPU idle) ran the CPU searches for
$0.10; everything else ran on the VPS.

The reviewer re-implemented the whole taxonomy blind (all 13 strategies, every sub-type, the primary and curried
primary labels) and wrote their own G4ip prover, Lean harness and ND→Lean renderer.
- **Every count recomputed from the labels reproduces.** The reviewer's classifier agrees with the executor's on every
  record of all 11 classified sets.
- **Lean accepts every proof the run searched or shows**, in the reviewer's harness, whose controls fail when they
  should.
- **Two definitional statements are wrong as stated** (below): "no K12 theorem is σ-covered" and "every proof of a
  σ-covered theorem needs a working reductio (#10)". No reported count changes. Per-proof RL reductio counts would rise
  under the reviewer's proposed fix to #10.
- "Blind spot-check 260 / 260" was against taxonomy v1.1. The final disjunctive-syllogism and excluded-middle rules
  were never hand-checked blind.

Question (Dan, 2026-10-18): "classifying the proofs in the pretraining by proof strategy … I'd like a semi-complete
classification of proof strategies. We should have canonical examples for each strategy as well as a rough percentage
of the pre-training data that uses each strategy." Brief:
[`BRIEF_proof-strategies.md`](../../docs/proposals/claude-heavy/BRIEF_proof-strategies.md) (proposal 28).

**Reading copy in nd-rl:** [`docs/papers/proof-strategies/`](../../docs/papers/proof-strategies/README.md) holds the
taxonomy, tables, canonical examples, blog (≈ 2,290 words after the corrections) and three figures. The review's
corrections are marked *[review]*: 41 marks in all. Eight rule schemas or law formulas in the taxonomy are restated in
words, marked *[librarian]* (see "The librarian's sealed-class check" below). `git diff 0f4b6c4 --
docs/papers/proof-strategies/` shows every change against the executor's files.

## Configuration

- **Repository.** Fork `dan-pandori/nd-takehome`, branch `dan_proof-strategies`, from `origin/dan_create-or-elicit`.
  It ran on the second VPS (agent2), every role at `--effort max`.
  - Pre-registration `preregistration/proof-strategies.md`, committed at `a30bc33b` (18:52:50Z). That was before any pod
    and before any strategy count: the first count, classifier v0, is in `82fa2c47` at 19:04. The pod jobs J1–J6 and
    their expectations were committed at 19:15:40 (`44939fda`), before the pod was created at 19:16:09.
  - DONE [`441de472`](https://github.com/dan-pandori/nd-takehome/commit/441de472) (21:56Z). Deliverables under
    `proof_strategies/`: `TAXONOMY.md`, `TABLES.md`, `EXAMPLES.md`, `blog.md`, `EXPECTATIONS.md`, `figures/`, the
    classifier `strategies.py` with 39 Lean-checked unit tests, and `validation/second_impl.py`. Also
    `run_proof_strategies.md`, `numbers.md` § proof-strategies and `log.md`.
  - Review [`review_proof-strategies.md`](https://github.com/dan-pandori/nd-takehome/blob/e6bfe0ea/review_proof-strategies.md),
    in two commits:
    - [`0c306489`](https://github.com/dan-pandori/nd-takehome/commit/0c306489): §Recount (phase 1, committed before the
      executor's write-ups were read);
    - [`e6bfe0ea`](https://github.com/dan-pandori/nd-takehome/commit/e6bfe0ea): §Compare and §Verdict. It was corrected
      after an adversarial check of the review itself (`review_ps/phase2/adversarial.md`).

    The reviewer kit is `review_ps/`: classifier `cls.py`, G4ip `ipc.py`, Lean harness `lean.py`, RL recount
    `rl_recount.py`, and leak scan `leakscan.py`. Its sub-audits of the blog, tables, taxonomy and examples are in
    `review_ps/phase2/`.
- **Timeline.** Executor Sun 2026-10-18 18:38–21:56 UTC; reviewer 21:58 – Mon 00:26; librarian from 00:28.
- **Data** (classification only; proofs are dependency-pruned):

  | set | file | proofs classified |
  |---|---|---|
  | K12 (main) | `data/kh/train_k12.jsonl`, uncompressed md5 `800b5486` | 155,000 generator proofs |
  | cap-6 | `data/p2/train_depth3_f0_a1.jsonl`, md5 `29276f24` | 155,000 |
  | K12⁻ (knockout) | `create-or-elicit/data/coe/train_k12ko.jsonl.gz`, gz md5 `6326ebdd` | 155,000 |
  | dose set (K12⁻ + 1 % σ) | `train_k12dose1.jsonl.gz`, gz md5 `7433f6ac` | 155,000 |
  | RL targets / transfer / holdout250 | `data/ladder/`, `data/bs/` | 3,735 / 1,525 / 160 generator proofs; plus 1,610 textbook-schema instances with search proofs |
  | Charles's release (private) | nd-rl `origin/charles/main` `297dcbdd`, `release_2026-10-02/` | 187 base problems with his reference proofs (173 found by search, 17 of them minimal per conjunct only; 14 by hand) |
  | textbook72 (sealed, scoring only) | `data/bs/textbook72.jsonl` | search proofs for 58 of 72, kept local; not shown here |

  - **Search proofs.** `minlen.Search` with bound 14 lines and 120 s, for the schema instances, textbook72 and a K12
    sample of 990 (90 per length 2–12, seed 0). Its internal `nd_verify` call is bypassed.
  - **Necessity.** Bans on ∨E, NEGI, BOTE and DN, bound 12 lines and 60 s, on a K12 sample of 594 (54 per length,
    seed 1), on Charles's 187 and on textbook72.
- **Commands.** `proof_strategies/run_all.sh` regenerates every label, summary and theorem-level file on the VPS in
  ≈ 6 min. The pod ran `search_proofs.py` (J1–J3) and `necessity_ps.py` (J4–J6). `rl_found.py` streams the 12
  create-or-elicit `found_16.jsonl` files (≈ 15 min). The spot-check sample is `validation_sample.py`, seed 20261018.
- **Models.** Only the RL section involves models. It classifies proofs written by create-or-elicit's policies:
  best-cap12 recipe, ALiBiGPT, **9,560,832 params**, `lean_staten`, trained from scratch on K12 (full s3–s5), K12⁻
  (knockout s0–s5) or K12⁻ + 1 % σ (dose1 s0–s2). Each ran 16 rounds of T1 expert iteration on A40, judged by Lean
  alone. The proofs come from each ladder's cumulative `found_16.jsonl`. Round 1 is samples of the pretrained model,
  before any RL update.
- **Checker: Lean alone.** Every searched proof was Lean-checked (`nd2lean` + Lean core); examples and blog snippets
  were checked with `lean_check`. `nd_verify` is unmodified (tree = `origin/main`, `9437bb72`) and judges nothing.
  `artifacts/TEST_RUN_DONE` is unchanged. There is no training code and no test-file run.
- **Pods and cost.** One pod, `ps-c0` (RTX 3090, used for CPU only), created 19:16:09Z.
  - Usage: 0.1958 billed hours, **$0.10** (`~/podhours.log`), 705 pod-seconds.
  - Budget: $20 / 40 h, registered at 18:23 and again at 18:52, before the pod.
  - The pre-registration planned $0 (VPS only, with a pod only if the searches would take more than 6 h). The 19:15
    log entry estimated the pod at ≈ 2–3 h and ≈ $1.5.
- **Compute rows.** 40 registry rows (`artifacts/proof-strategies/registry/`; bucket
  `hf://buckets/dan-pandori/nd-rl/registry/proof-strategies/`): 6–7 metric rows for each job J1–J6 and one pod row:
  - gpu_seconds 0, gen_tokens 0, train_steps 0;
  - lean_checks 1,610 / 58 / 990 / 2,880 / 514 / 216 = 6,268.

  Compute was not recorded "per step as VPS CPU-seconds", as the pre-registration said it would be.
- **Bucket.** `hf://buckets/dan-pandori/nd-rl/proof-strategies/` (`artifacts/`, `data/`; 89 files in the reviewer's
  listing). Charles's release and the textbook72 proofs stay local. The per-record label and theorem files carry
  indices and labels only.

## Observations

All numbers are reviewer-checked. Shares are of proofs (records), dependency-pruned. "Uses" means the strategy appears
anywhere in the proof (multi-label). "Primary" is the strategy of the root inference, under a stated precedence rule.

**1. The taxonomy** (`TAXONOMY.md`, v1.2). There are thirteen strategies, ten intuitionistic and three classical, plus
"trivial". Each has a formal predicate over the pruned ND proof, standard names with textbook references, and
sub-types. A *working* instance does logical work. A *template* or *idle* one could be deleted, or only restates a
supplied double negation.
- **σ maps exactly.** σ (DN citing a ¬I line) is exactly #10 via ¬I: 16,703 records. σ⁺ (16,961) = #10 ∪ #11
  `on_derived`, and 0 records differ. #10 has 16,717 records, 14 more than σ: "disguised" reductios whose ¬¬G is built
  another way.
- **Coverage.** Unlabelled proofs: 0 in every set, for both the root and the curried primary label.
- **Agreement.** Three implementations agree on every K12 record: the executor's, an independent second
  implementation, and the reviewer's blind one. The reviewer's also agrees on every record of the other ten sets.

**2. K12's mix** (155,000 proofs, %):

| # | strategy | uses | working | primary | curried primary |
|---|---|---:|---:|---:|---:|
| 1 | direct reasoning (→E, ∧E) | 51.15 | – | 9.71 | 15.29 |
| 2 | conjunction introduction | 32.57 | – | 12.24 | 19.38 |
| 3 | conditional proof (→I) | 42.95 | 20.80 | 37.65 | 0.00 |
| 4 | ∨-introduction | 44.26 | – | 14.08 | 20.50 |
| 5 | proof by cases (∨E) | 44.16 | 8.79 | 20.85 | 33.41 |
| 6 | disjunctive syllogism | 0.03 (43 records; 0.24 % under v0) | – | 0.01 | 0.02 |
| 7 | proof of a negation (¬I) | 4.74 | 2.15 | 1.94 | 2.69 |
| 8 | modus tollens / contraposition | 0.05 (76) | – | 0.04 | 0.04 |
| 9 | ex falso | 8.04 | – | 0.52 | 1.58 |
| 10 | proof by contradiction (classical) | 10.79 (16,717) | 0.04 (59) | 2.91 | 3.50 |
| 11 | double-negation elimination | 3.06 | – | 0.05 | 0.33 |
| 12 | excluded middle | 0.03 (47) | – | 0.00 | 0.00 |
| 13 | lemma reuse | 11.71 | – | – | – |

- **Idle steps are common.**
  - 28.20 % of proofs discharge an assumption they never use.
  - 37.64 % contain a redundant case split, where one case ignores its own assumption.
  - 23.53 % of the theorems have unsatisfiable premises.
  - 24.91 % of proofs open no box.
  - No proof has a dead line.
- **Like for like** (990 K12 theorems re-proved by the shortest-proof search; Lean accepts all 990):
  - the search proof is shorter on 534 theorems and longer on 1;
  - mean pruned lines fall from 7.00 to 4.69;
  - idle splits fall from 34.8 % to 0, and proof by cases from 42.5 % to 6.1 %;
  - proof by contradiction falls from 10.0 % to 2.2 %;
  - unused →I assumptions fall only from 29.1 % to 22.8 %;
  - ex falso from inconsistent premises rises from 5.9 % to 16.3 %.

**3. K12's proof by contradiction is almost all template.**
- 16,658 of the 16,717 reductio proofs are template-only.
- Kinds (records with at least one instance): vacuous 8,209, scaffolded 7,693, pointless 1,082, negated goal 48,
  free 6.
- Of the 8,209 records with a vacuous reductio, 7,300 (88.9 %) have unsatisfiable premises.
- 59 proofs do working reasoning. None is of a classical-only theorem, and 48 of them prove a negation.
- The dose set has 1,550 reductio proofs, 5 of them working (2 free). cap-6 has 6.81 % reductio proofs (4 working);
  K12⁻ has 0.

**4. Theorem level** (the reviewer's own G4ip reproduces every label on every record of all ten sets):
- K12: classical-only 6,238 (4.02 %); a semantic double negation in the sequent 45,630 (29.44 %); inconsistent
  premises 36,472 (23.53 %).
- Of the 16,717 reductio users, 4,279 (25.60 %) are classical-only theorems. Of the 6,238 classical-only theorems,
  4,279 (68.60 %) use #10 and 1,959 use #11 alone.
- **σ-covered (wrong as stated in the run's files).** "No K12 theorem is σ-covered" holds only under the run's
  *semantic* rule, which counts ¬X → ⊥ (and ¬(X → ⊥), (X → ⊥) → ⊥) as a double negation. The pre-registration defines
  σ-covered by a syntactic test: classical-only and no ¬¬ subformula, "create-or-elicit's definition". Under that rule
  K12 has **2,340** σ-covered theorems (1.51 %), all with a (¬X → ⊥) subformula; cap-6 has 2,235 and the dose set 232.
  The switch is an unflagged deviation. Both rules agree on the pools (RL targets 123, transfer 106, holdout250 12),
  on Charles's problems (35) and on textbook72 (16).

**5. Textbooks use what is rare in K12** (Charles's 187 problems, his reference proofs, against K12):

| strategy (share of proofs) | textbook problems | K12 |
|---|---:|---:|
| working proof by contradiction | 20.86 % | 0.04 % |
| modus tollens | 11.76 % | 0.05 % |
| disjunctive syllogism | 14.44 % | 0.03 % (0.24 % under v0) |
| working proof of a negation | 37.97 % | 2.15 % |
| genuine case analysis | 33.69 % | 8.79 % |
| classical-only theorems | 22.46 % | 4.02 % |
| σ-covered theorems | 18.72 % (35) | 0 (semantic) / 1.51 % (pre-registered rule) |

The 1,610 textbook-schema instances in our RL pools point the same way: working #10 20.75 %, ¬I 39.57 %, modus tollens
10.43 %, disjunctive syllogism 5.59 %, excluded middle 5.34 %; 240 are σ-covered. So does textbook72 (58 of 72 with a
search proof): #10 17.2 %, all of it working; ¬I 32.8 %; 16 of 72 σ-covered. Every one of these rows is far above
K12.

**6. What the theorems require** (bounded necessity: a proof of ≤ 12 lines exists, and none exists with the rule
banned; relative to minlen's formula space):
- K12 sample (594): requires ∨E 6.1 %, NEGI 3.7 %, BOTE 0 %, DN 5.4 %. The DN ban agrees with G4ip on 593 / 594.
- Of the sample's proofs that use ∨E, 14.0 % (36 / 258) are of theorems that need it.
- Charles's problems (124 of 187 decided): requires NEGI 46 / 124 = 37.1 %.
- The pre-registered K12 sample was 600, stratified by length and primary; the run used 594, stratified by length only.

**7. RL's own proofs** (create-or-elicit's 12 ladders, models as in Configuration).
- The reviewer streamed every `found_16.jsonl`: 7,855,065 distinct Lean-accepted proofs, 0 unparsable.
- Every per-round cell equals the executor's `rl_found.json`.

| ladder | working #10, round 1 (base) | working #10, rounds 2–16 | free working, round 1 / rounds 2–16 | σ-covered targets with a working #10, round 1 / by round 16 |
|---|---:|---:|---|---|
| full s3 / s4 / s5 | 76 / 32 / 52 | 26,450 / 9,082 / 6,951 | 38 / 27 / 45; 22,013 / 7,886 / 5,915 | 17 / 15 / 20 → 78 / 33 / 36 |
| dose1 s0 / s1 / s2 | 15 / 45 / 7 | 16,970 / 11,057 / 9,969 | 13 / 43 / 7; 15,731 / 9,920 / 9,047 | 2 / 17 / 2 → 69 / 46 / 41 |
| knockout s0–s5 | 1 / 0 / 1 / 0 / 2 / 1 | 358 / 399 / 772 / 169 / 72 / 375 | 0 in round 1; 0 / 32 / 2 / 1 / 0 / 12 | 0 → 0 / 5 / 1 / 0 / 0 / 2 |

- **Knockout ladders.** They wrote working reductios after RL, but almost none free. Of the 2,145 in rounds 2–16, 1,985
  (93 %) are scaffolded on a double negation the theorem supplies; ko_s2 also wrote 113 with a negated goal.
- **"Knockout models proved ≤ 5 σ-covered problems" holds for proofs of any kind** (0 / 5 / 1 / 0 / 0 / 2).
- **Base rates.** Free working reductios are 0.15–0.27 % of the full models' round-1 proofs, and 1.2–3.9 % of the proofs
  first found in round 16, a different population. The full models prove 33–78 of the 123 σ-covered targets by round
  16.
- **Not every proof of a σ-covered target needs #10 (wrong as stated in the run's files).** In rounds 1–16 at least
  **573** Lean-accepted RL proofs of σ-covered targets eliminate a double negation they built themselves through a box
  hypothesis: an ∨E case, or create-or-elicit's →I "lemma route". They carry no #10 label.
  - By ladder: dose1_s0 491 (4.2 % of its σ-target proofs); full s3 / s4 / s5 5 / 11 / 49; dose1 s1 / s2 9 / 8;
    knockout 0.
  - **Target-level results do not change:** every σ-covered target any ladder proved by round 16 also has a
    working-#10 proof.
  - **The per-proof counts with #10, working and free would rise** under the reviewer's fix. Among σ-covered targets
    the rise is up to 573, about 3 % of dose1_s0's free proofs; proofs of other targets were not measured.
  - In the pretraining sets, DN on a self-introduced ¬¬ hypothesis occurs in 37 K12, 11 K12⁻, 12 dose and 1 cap-6
    proofs. Every ∨E-routed instance sits on the case that never occurs, so no pretraining number changes.
- **The knockout reading is not tested.** The blog's reading is that pretraining supplied the *form* and RL put it to
  work, and the blog labels it untested. The review adds a confound the blog should name: every EI round also replays
  20,000 records of the arm's own pretraining set, so the full arm kept seeing K12's σ templates during RL.

**8. Validation.**
- **Lean (reviewer's harness).**
  - Controls: 300 / 300 untouched K12 proofs accepted; 0 / 200 statement swaps; 0 / 100 truncations; 0 / 50 bare
    `sorry`. 5 of 200 single Or.inl↔Or.inr or .1↔.2 flips are accepted, and all 5 are no-ops.
  - It then accepted 9,115 / 9,115 run proofs: 6,268 search and necessity proofs, Charles's 187 reference proofs and
    2,660 sampled generator proofs.
  - It also accepted the 30 canonical examples, whose Lean statements are the ND sequents, the 12 blog snippets and
    2,114 sampled RL proofs (1,694 of them working).
- **Term size.** The per-rule formula equals `lean_check`'s node count on 350 / 350 of the reviewer's renderings, and
  the executor's per-record values on six sets.
- **Spot-check.** Three blind sub-agents labelled 260 proofs (20 per strategy). They agree with the v1.1 labels on
  260 / 260 uses and 260 / 260 primary. Against the final v1.2 rules they agree on 245 / 260 uses and 258 / 260
  primary: disjunctive syllogism 10 / 20 and excluded middle 15 / 20. Those two rules were tightened after the check,
  prompted by the checkers' own notes. Only the second implementation validates the final versions.
- **Canonical examples.** All 30 match their source records, use their strategy, and sit outside every textbook72 key.
  11 of the 12 K12 "shortest clean" examples are minimal. For disjunctive syllogism the textbook72 guard skipped a
  7-line instance; the guard working as intended.

**9. Pre-registered expectations** (45 rows in `EXPECTATIONS.md`; the ledger reports every miss):
- **E1 (K12 uses):**
  - conditional proof 42.95 % (expected 65–90 %);
  - disjunctive syllogism 0.03 % (3–10 %);
  - modus tollens 0.05 % (2–6 %);
  - box-free 24.91 % (3–12 %).
- **E2 (primary):** conditional proof 37.65 % (40–65 %); curried primary #10 3.50 % (5–9 %).
- **E3:** a strategy only in dead lines, 0 % (5–20 %): K12 has no dead lines.
- **E5:** textbook vacuous discharge 10.70 % (< 5 %).
- **J4:** requires-∨E 14.0 % (30–60 %).
- **RL, knockout round 1:** zero working reductios on 2 / 6 seeds (expected ≥ 5 / 6).
- **RL, knockout round 16:** expected ≤ 50 per seed; 84 on one seed.
- **"Round 16 ≥ 10 × round 1"** is a hit on counts (29.8 / 25.2 / 13.2 ×), the pre-registered quantity, and a miss as
  shares (8.4 / 8.9 / 4.6 ×). Both are stated.
- **The RL expectations were not written blind.** They were committed at 19:57, before this run's first RL count.
  But a minute earlier the executor printed create-or-elicit's per-round artefact for full_s3, which showed 17
  σ-covered targets in round 1 and reductio counts rising every round. So they carry less weight as predictions.
- **The first RL count used the wrong data:** the fine-tune mixes, which hold ≤ 4 proofs per problem. The executor's
  own claim check caught it, and the count was redone from the found files.

## Corrections the review requires (applied in the nd-rl reading copy, not on the fork)

No reported count changes, except that the RL per-proof reductio counts would rise under the fix in item 2.
1. **σ-covered is wrong as stated, and an unflagged deviation from the pre-registration.**
   - Wherever the run says "no K12 theorem is σ-covered", "σ-covered 0" or "create-or-elicit's definition", name the
     rule. The pre-registered and create-or-elicit rule gives K12 2,340.
   - Affected (line numbers are the fork's): TAXONOMY §4 and the §5 "(both)" column; TABLES § 4; blog lines 117–119
     and 142; `numbers.md`; the run summary; STATUS.
2. **"Every proof of a σ-covered theorem needs a working reductio (#10)" is wrong as stated** (≥ 573 counterexamples).
   - Say "eliminates a double negation it built itself", or extend #10.
   - Affected (line numbers are the fork's): TAXONOMY §4; TABLES § 11 (twice); blog lines 119 and 155–156.
3. **"Blind spot-check 260 / 260"** is v1.1; the final #6 and #12 were not hand-checked blind. Affected: the run
   summary, STATUS, TAXONOMY §6. The blog's wording is already exact.
4. **TABLES:**
   - the header "No models" is false for § 11;
   - the § 5 caption says the opposite of its share column;
   - the § 5 row labelled "(theorem classical-only) 27" is the bounded requires-DN count (G4ip: 26).
5. **Blog, line 44:** expert iteration also replays 20,000 pretraining records per round. Say so beside the "pretraining
   supplied the form" reading.
6. **EXAMPLES:** three "typical" glosses describe their one record, not K12:
   - cases: only the idle split is typical (89.5 %); inconsistent premises 32.6 %, an exploded disjunction 0.33 %;
   - #10: a ¬G → ⊥ premise in 31.3 %;
   - →I: "two assumptions never used" in 7.5 %.

   "Shortest clean" rests on a readability filter stated only in code, and on the textbook72 guard.
7. **The RL expectations' timing** (above): the ledger should say so.
8. **Smaller wording items:**
   - the run summary's miss list omits four misses that the ledger has;
   - "the knockout removed only their combination": it also removed DNE on a derived ¬¬ (0.17 %);
   - "textbook problems use what K12 lacks": the strategies are rare in K12, not absent;
   - "173 minimal" includes 17 proofs that are minimal per conjunct only;
   - "59 of the 16,717 reductios" counts proofs, not reductios (17,405 reductio steps);
   - the blog does not name the model format and checkpoints;
   - EXAMPLES and TABLES give two different term sizes for examples with a DN step;
   - TAXONOMY lists `bot_case` as never exercised (2,070 K12 records have it);
   - one "shortest" search proof is a line longer than the generator's;
   - the STATUS DONE line says "(= σ)" (16,717 vs 16,703) and "≤ 2 %" (2.15 %), and states the RL reading causally.
- **Hygiene.** Committed unit-test and self-test fixtures quote nine textbook72 renaming classes.
  - One of those files, `second_impl_fixtures_v1.2.jsonl` (3 classes), is also in the public bucket twice, under
    `data/ps/validation/` and `artifacts/ps/validation/`.
  - All nine classes already appear in public pre-run files, and `textbook72.jsonl` is itself tracked in the public
    fork, so nothing new is exposed.
  - The review asks to replace the fixtures with non-sealed variants and re-upload, or to note the exception. Not done.
- **Pre-registration deviations not flagged in `log.md`:**
  - the σ-covered rule;
  - the like-for-like sample (1,000 → 990);
  - the necessity sample (600 by length and primary → 594 by length);
  - the final disjunctive-syllogism tightening (K12 47 → 43 records, `e0511ae6`);
  - compute not recorded per step.
- **`Not.elim` in RL proofs.** The proof-state env decodes Lean's `n.elim` on `n : ¬A` as a BOTE citing a negation, in
  2 of the 2,114 sampled RL proofs. The classifier counts it as ex falso; the effect is negligible.

**The fork's files are unchanged.** That covers `run_proof_strategies.md`, `numbers.md`, `STATUS.md`,
`EXPECTATIONS.md`, `log.md`, the fixtures, and `proof_strategies/{TAXONOMY,TABLES,EXAMPLES,blog}.md`. The corrections
are in the nd-rl reading copy and in this summary.

**Earlier documents that look incomplete** (the librarian's reading of C7; the review did not assess them).
create-or-elicit's texts list the ways a proof of a strict σ-covered member can eliminate its double negation:
- the summary (lines 30–31 and 94–95): "σ, another σ⁺ route, or the lemma route";
- the paper, `docs/papers/create-or-elicit/sections/03_setup.tex` lines 67–68 (the "strict member" row of the concepts
  table): "by a free positive reductio or by the lemma route".

C7's route is DN on a self-built ¬¬ held as a box hypothesis, such as an ∨E case. It is neither σ nor σ⁺ (DN there
cites an assumption), nor the lemma route. create-or-elicit's own per-round classes count 489 such `dn_pr_as` proofs
of σ-covered targets in dose1_s0 alone (review C7; `review_ps/phase2/adversarial.md`, F3). So both lists look incomplete. Target-level results do not change. The
documents are left as they are, and the item is raised in the week-43 digest.

## The librarian's sealed-class check (not part of the review)

The review's leak scan parsed ND prompts, ASCII `|-` sequents and Lean statements. It did not parse formulas written in
Unicode prose. The librarian scanned every formula and sequent in the four copied files, reading their letters as atoms
and using the reviewer's keys (`review_ps/sealkey.py`).
- **Eight places in the fork's `TAXONOMY.md` are in textbook72 renaming classes.** All are rule schemas or law formulas
  inside definitions.
- In the reading copy they are restated in words and marked *[librarian]*, the convention `paper-clean` used for a
  printed schema.
- The fork's file and this directory's first commit `0f4b6c4` still print them.
- The librarian's own additions add none. The blog, tables and examples have none, apart from one scanner artefact: a
  token span that crosses a parenthesis.

## Limitations

- **Labels describe the proofs we have.** They come from the generator's, the search's or Charles's proofs, and another
  proof of the same theorem could differ.
  - The theorem-level labels guard against this: classical-only is exact (G4ip); necessity holds within 12 lines and
    minlen's formula space.
  - The bound shows: one intuitionistically provable theorem in K12 and one in Charles's problems are flagged "requires
    DN".
- **The final #6 and #12 were never hand-checked blind.** "Working" and "template" are syntactic: they say whether a
  step does work in this proof, not whether a cleverer proof exists.
- **The RL section rests on 12 ladders of one 9.56M architecture,** read through the proof-state env's ND decoding.
  The knockout reading ("pretraining supplied the form") is a hypothesis with a named confound (RL-time replay of σ).
- **Proof length is reported in pruned lines and term size.** The examples give `lean_check`'s count on the readable
  rendering, 2 below the per-rule formula for examples with a DN step.

## Next step

The review's next measurements, none of them costed or funded:
- Hand-check the final v1.2 disjunctive-syllogism and excluded-middle predicates blind, 20 each.
- Extend #10 to DN on a self-built ¬¬ box hypothesis and recount the 12 found files (≈ 20 min on the VPS).
- **Test the knockout reading:**
  - compare per-attempt round-1 rates of free reductios between the full and dose bases, from the ladder logs at
    k = 32. Their pretraining holds 6 vs 2 free and 16,658 vs 1,545 template-only reductio proofs;
  - run a full-arm ladder whose replay excludes σ records, to separate "form from pretraining" from "form re-supplied
    during RL".

Queued next on this VPS: `kalmar-pretrain` (proposal 29), which starts once this run is recorded.
